# # # # # # # # # """
# # # # # # # # # scripts/evaluate_models.py
# # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # Script đánh giá tất cả models (LSTM, ST-Trans, FlowMatching) trên bộ test 12 bão.

# # # # # # # # # Cách dùng:
# # # # # # # # #     python scripts/evaluate_models.py \
# # # # # # # # #         --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# # # # # # # # #         --fm_ckpt runs/v59/best_model.pth \
# # # # # # # # #         --lstm_ckpt runs/lstm/best_model.pth \
# # # # # # # # #         --sttrans_ckpt runs/sttrans/best_model.pth \
# # # # # # # # #         --output_dir results/test_eval \
# # # # # # # # #         --gpu_num 0

# # # # # # # # # Output:
# # # # # # # # #     results/test_eval/
# # # # # # # # #         ├── FM_v59_per_storm.csv         # Per-storm, per-timestep predictions
# # # # # # # # #         ├── LSTM_per_storm.csv
# # # # # # # # #         ├── STTrans_per_storm.csv
# # # # # # # # #         ├── summary_all_models.csv       # Bảng so sánh ADE/ATE/CTE tất cả models
# # # # # # # # #         └── per_storm_detail/
# # # # # # # # #             ├── FM_v59_Amphan.csv        # Từng bão riêng
# # # # # # # # #             ├── FM_v59_Mocha.csv
# # # # # # # # #             └── ...

# # # # # # # # # CSV format per storm file:
# # # # # # # # #     storm_id | lead_h | lon_pred | lat_pred | lon_true | lat_true |
# # # # # # # # #     dist_km  | ate_km | cte_km
# # # # # # # # # """
# # # # # # # # # from __future__ import annotations

# # # # # # # # # import sys
# # # # # # # # # import os
# # # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # # # import argparse
# # # # # # # # # import csv
# # # # # # # # # import math
# # # # # # # # # import time
# # # # # # # # # from collections import defaultdict
# # # # # # # # # from datetime import datetime
# # # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # # import numpy as np
# # # # # # # # # import torch
# # # # # # # # # import torch.nn.functional as F
# # # # # # # # # from torch.utils.data import DataLoader

# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # R_EARTH  = 6371.0
# # # # # # # # # DEG2KM   = 111.0
# # # # # # # # # DT_HOURS = 6.0

# # # # # # # # # LEAD_HOURS  = [12, 24, 36, 48, 60, 72]          # 2, 4, 6, 8, 10, 12 steps (6h each)
# # # # # # # # # LEAD_STEPS  = {h: h // 6 - 1 for h in LEAD_HOURS}  # 0-indexed: step idx

# # # # # # # # # # Targets từ papers (Faiaz 2026 / Rahman 2025)
# # # # # # # # # TARGETS = {
# # # # # # # # #     "12h" : 50.0,
# # # # # # # # #     "24h" : 100.0,
# # # # # # # # #     "48h" : 200.0,
# # # # # # # # #     "72h" : 297.0,
# # # # # # # # #     "ADE" : 136.41,
# # # # # # # # #     "ATE" : 79.94,
# # # # # # # # #     "CTE" : 93.58,
# # # # # # # # # }

# # # # # # # # # MODEL_CONFIGS = {
# # # # # # # # #     "LSTM"    : {"color": "🔵", "ref": "Rahman 2025"},
# # # # # # # # #     "STTrans" : {"color": "🟡", "ref": "Faiaz 2026"},
# # # # # # # # #     "FM_v59"  : {"color": "🔴", "ref": "Ours"},
# # # # # # # # # }


# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  Coordinate utilities
# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # # def norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
# # # # # # # # #     out = arr.clone()
# # # # # # # # #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# # # # # # # # #     return out


# # # # # # # # # def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # # # # #     """p1, p2: [..., 2] với [..., 0]=lon, [..., 1]=lat (degrees)"""
# # # # # # # # #     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
# # # # # # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # # # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # # # # # #     a    = (torch.sin(dlat / 2).pow(2) +
# # # # # # # # #             torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
# # # # # # # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# # # # # # # # # def compute_ate_cte(
# # # # # # # # #     pred_deg: torch.Tensor,
# # # # # # # # #     gt_deg:   torch.Tensor,
# # # # # # # # # ) -> Tuple[torch.Tensor, torch.Tensor]:
# # # # # # # # #     """
# # # # # # # # #     Tính ATE và CTE cho từng bước dự báo.

# # # # # # # # #     Args:
# # # # # # # # #         pred_deg: [T, B, 2] predicted positions in degrees
# # # # # # # # #         gt_deg:   [T, B, 2] ground truth positions in degrees

# # # # # # # # #     Returns:
# # # # # # # # #         ate: [T-1, B] along-track error (km), signed
# # # # # # # # #         cte: [T-1, B] cross-track error (km), signed
# # # # # # # # #     """
# # # # # # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # # #     if T < 2:
# # # # # # # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # # # # # #         return z, z

# # # # # # # # #     # Along-track direction từ GT
# # # # # # # # #     lon1 = torch.deg2rad(gt_deg[:T-1, :, 0])
# # # # # # # # #     lat1 = torch.deg2rad(gt_deg[:T-1, :, 1])
# # # # # # # # #     lon2 = torch.deg2rad(gt_deg[1:T,  :, 0])
# # # # # # # # #     lat2 = torch.deg2rad(gt_deg[1:T,  :, 1])

# # # # # # # # #     # Forward azimuth của GT track
# # # # # # # # #     dlon_a = lon2 - lon1
# # # # # # # # #     y_a = torch.sin(dlon_a) * torch.cos(lat2)
# # # # # # # # #     x_a = (torch.cos(lat1) * torch.sin(lat2)
# # # # # # # # #            - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon_a))
# # # # # # # # #     bear_along = torch.atan2(y_a, x_a)  # [T-1, B]

# # # # # # # # #     # Error vector direction
# # # # # # # # #     lon3 = torch.deg2rad(pred_deg[1:T, :, 0])
# # # # # # # # #     lat3 = torch.deg2rad(pred_deg[1:T, :, 1])
# # # # # # # # #     dlon_e = lon3 - lon2
# # # # # # # # #     y_e = torch.sin(dlon_e) * torch.cos(lat3)
# # # # # # # # #     x_e = (torch.cos(lat2) * torch.sin(lat3)
# # # # # # # # #            - torch.sin(lat2) * torch.cos(lat3) * torch.cos(dlon_e))
# # # # # # # # #     bear_err = torch.atan2(y_e, x_e)   # [T-1, B]

# # # # # # # # #     # Tổng lỗi & decompose
# # # # # # # # #     total = haversine_km(pred_deg[1:T], gt_deg[1:T])  # [T-1, B]
# # # # # # # # #     angle = bear_err - bear_along
# # # # # # # # #     ate   = total * torch.cos(angle)   # [T-1, B]
# # # # # # # # #     cte   = total * torch.sin(angle)   # [T-1, B]
# # # # # # # # #     return ate, cte


# # # # # # # # # def move_to_device(batch, device):
# # # # # # # # #     out = list(batch)
# # # # # # # # #     for i, x in enumerate(out):
# # # # # # # # #         if torch.is_tensor(x):
# # # # # # # # #             out[i] = x.to(device)
# # # # # # # # #         elif isinstance(x, dict):
# # # # # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # # # #                       for k, v in x.items()}
# # # # # # # # #     return out


# # # # # # # # # def get_raw_model(model):
# # # # # # # # #     return model._orig_mod if hasattr(model, "_orig_mod") else model


# # # # # # # # # def get_ema_obj(model):
# # # # # # # # #     if hasattr(model, "_ema") and model._ema is not None:
# # # # # # # # #         return model._ema
# # # # # # # # #     if hasattr(model, "_orig_mod"):
# # # # # # # # #         orig = model._orig_mod
# # # # # # # # #         if hasattr(orig, "_ema") and orig._ema is not None:
# # # # # # # # #             return orig._ema
# # # # # # # # #     return None


# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  Per-storm result accumulator
# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # # class StormResult:
# # # # # # # # #     """Lưu kết quả đầy đủ cho 1 cơn bão."""

# # # # # # # # #     def __init__(self, storm_id: str):
# # # # # # # # #         self.storm_id = storm_id
# # # # # # # # #         self.rows: List[dict] = []          # per-timestep rows
# # # # # # # # #         self.dist_by_lead: Dict[int, List[float]] = defaultdict(list)
# # # # # # # # #         self.ate_by_lead:  Dict[int, List[float]] = defaultdict(list)
# # # # # # # # #         self.cte_by_lead:  Dict[int, List[float]] = defaultdict(list)
# # # # # # # # #         self.all_dist: List[float] = []
# # # # # # # # #         self.all_ate:  List[float] = []
# # # # # # # # #         self.all_cte:  List[float] = []
# # # # # # # # #         self.n_seq = 0

# # # # # # # # #     def add_batch(
# # # # # # # # #         self,
# # # # # # # # #         pred_deg: torch.Tensor,   # [T, B, 2]
# # # # # # # # #         gt_deg:   torch.Tensor,   # [T, B, 2]
# # # # # # # # #         seq_indices: Optional[torch.Tensor] = None,
# # # # # # # # #     ):
# # # # # # # # #         T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # # #         B = pred_deg.shape[1]
# # # # # # # # #         if T < 2:
# # # # # # # # #             return

# # # # # # # # #         # Per-step distance
# # # # # # # # #         dist = haversine_km(pred_deg[:T], gt_deg[:T])   # [T, B]
# # # # # # # # #         ate, cte = compute_ate_cte(pred_deg, gt_deg)    # [T-1, B]

# # # # # # # # #         self.n_seq += B

# # # # # # # # #         for b in range(B):
# # # # # # # # #             for t in range(T):
# # # # # # # # #                 lead_h = (t + 1) * 6
# # # # # # # # #                 d_val  = float(dist[t, b])
# # # # # # # # #                 ate_val = float(ate[t, b].abs()) if t < ate.shape[0] else float("nan")
# # # # # # # # #                 cte_val = float(cte[t, b].abs()) if t < cte.shape[0] else float("nan")

# # # # # # # # #                 self.rows.append({
# # # # # # # # #                     "storm_id"  : self.storm_id,
# # # # # # # # #                     "seq_idx"   : int(seq_indices[b]) if seq_indices is not None else b,
# # # # # # # # #                     "lead_h"    : lead_h,
# # # # # # # # #                     "lon_pred"  : float(pred_deg[t, b, 0]),
# # # # # # # # #                     "lat_pred"  : float(pred_deg[t, b, 1]),
# # # # # # # # #                     "lon_true"  : float(gt_deg[t, b, 0]),
# # # # # # # # #                     "lat_true"  : float(gt_deg[t, b, 1]),
# # # # # # # # #                     "dist_km"   : round(d_val,   2),
# # # # # # # # #                     "ate_km"    : round(ate_val, 2) if np.isfinite(ate_val) else "nan",
# # # # # # # # #                     "cte_km"    : round(cte_val, 2) if np.isfinite(cte_val) else "nan",
# # # # # # # # #                 })

# # # # # # # # #                 # Tích lũy theo lead time
# # # # # # # # #                 if lead_h in LEAD_STEPS:
# # # # # # # # #                     self.dist_by_lead[lead_h].append(d_val)
# # # # # # # # #                     if np.isfinite(ate_val): self.ate_by_lead[lead_h].append(ate_val)
# # # # # # # # #                     if np.isfinite(cte_val): self.cte_by_lead[lead_h].append(cte_val)

# # # # # # # # #                 # ADE/ATE/CTE tổng
# # # # # # # # #                 self.all_dist.append(d_val)
# # # # # # # # #                 if t < ate.shape[0]:
# # # # # # # # #                     if np.isfinite(ate_val): self.all_ate.append(ate_val)
# # # # # # # # #                     if np.isfinite(cte_val): self.all_cte.append(cte_val)

# # # # # # # # #     def summary(self) -> dict:
# # # # # # # # #         def _m(lst): return float(np.mean(lst)) if lst else float("nan")

# # # # # # # # #         r = {
# # # # # # # # #             "storm_id" : self.storm_id,
# # # # # # # # #             "n_seq"    : self.n_seq,
# # # # # # # # #             "ADE"      : _m(self.all_dist),
# # # # # # # # #             "ATE"      : _m(self.all_ate),
# # # # # # # # #             "CTE"      : _m(self.all_cte),
# # # # # # # # #         }
# # # # # # # # #         for h in LEAD_HOURS:
# # # # # # # # #             r[f"{h}h_dist"] = _m(self.dist_by_lead.get(h, []))
# # # # # # # # #             r[f"{h}h_ate"]  = _m(self.ate_by_lead.get(h,  []))
# # # # # # # # #             r[f"{h}h_cte"]  = _m(self.cte_by_lead.get(h,  []))
# # # # # # # # #         return r


# # # # # # # # # class MultiStormAccumulator:
# # # # # # # # #     """Tích lũy kết quả cho toàn bộ test set."""

# # # # # # # # #     def __init__(self, model_name: str):
# # # # # # # # #         self.model_name = model_name
# # # # # # # # #         self.storms: Dict[str, StormResult] = {}

# # # # # # # # #     def get_storm(self, storm_id: str) -> StormResult:
# # # # # # # # #         if storm_id not in self.storms:
# # # # # # # # #             self.storms[storm_id] = StormResult(storm_id)
# # # # # # # # #         return self.storms[storm_id]

# # # # # # # # #     def add_batch(
# # # # # # # # #         self,
# # # # # # # # #         pred_norm: torch.Tensor,   # [T, B, 2] normalized
# # # # # # # # #         gt_norm:   torch.Tensor,   # [T, B, 2] normalized
# # # # # # # # #         storm_ids: List[str],
# # # # # # # # #     ):
# # # # # # # # #         # Convert to degrees
# # # # # # # # #         T   = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # # #         B   = pred_norm.shape[1]
# # # # # # # # #         pd  = norm_to_deg(pred_norm[:T])
# # # # # # # # #         gd  = norm_to_deg(gt_norm[:T])

# # # # # # # # #         # Group by storm_id
# # # # # # # # #         sid_map: Dict[str, List[int]] = defaultdict(list)
# # # # # # # # #         for b, sid in enumerate(storm_ids[:B]):
# # # # # # # # #             sid_map[sid].append(b)

# # # # # # # # #         for sid, bidx in sid_map.items():
# # # # # # # # #             bidx_t = torch.tensor(bidx, dtype=torch.long)
# # # # # # # # #             self.get_storm(sid).add_batch(
# # # # # # # # #                 pd[:, bidx_t, :],
# # # # # # # # #                 gd[:, bidx_t, :],
# # # # # # # # #                 seq_indices=bidx_t,
# # # # # # # # #             )

# # # # # # # # #     def all_summaries(self) -> List[dict]:
# # # # # # # # #         rows = []
# # # # # # # # #         for sid in sorted(self.storms):
# # # # # # # # #             rows.append(self.storms[sid].summary())
# # # # # # # # #         return rows

# # # # # # # # #     def overall_summary(self) -> dict:
# # # # # # # # #         """Tổng trung bình tất cả bão."""
# # # # # # # # #         all_d, all_a, all_c = [], [], []
# # # # # # # # #         lead_d: Dict[int, List[float]] = defaultdict(list)
# # # # # # # # #         lead_a: Dict[int, List[float]] = defaultdict(list)
# # # # # # # # #         lead_c: Dict[int, List[float]] = defaultdict(list)

# # # # # # # # #         for sr in self.storms.values():
# # # # # # # # #             all_d.extend(sr.all_dist)
# # # # # # # # #             all_a.extend(sr.all_ate)
# # # # # # # # #             all_c.extend(sr.all_cte)
# # # # # # # # #             for h in LEAD_HOURS:
# # # # # # # # #                 lead_d[h].extend(sr.dist_by_lead.get(h, []))
# # # # # # # # #                 lead_a[h].extend(sr.ate_by_lead.get(h, []))
# # # # # # # # #                 lead_c[h].extend(sr.cte_by_lead.get(h, []))

# # # # # # # # #         def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # # # # # # # #         r = {
# # # # # # # # #             "storm_id" : "📊 OVERALL",
# # # # # # # # #             "n_seq"    : sum(s.n_seq for s in self.storms.values()),
# # # # # # # # #             "ADE"      : _m(all_d),
# # # # # # # # #             "ATE"      : _m(all_a),
# # # # # # # # #             "CTE"      : _m(all_c),
# # # # # # # # #         }
# # # # # # # # #         for h in LEAD_HOURS:
# # # # # # # # #             r[f"{h}h_dist"] = _m(lead_d[h])
# # # # # # # # #             r[f"{h}h_ate"]  = _m(lead_a[h])
# # # # # # # # #             r[f"{h}h_cte"]  = _m(lead_c[h])
# # # # # # # # #         return r


# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  Extract storm IDs từ batch
# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # # def extract_storm_ids(batch, B: int) -> List[str]:
# # # # # # # # #     """Thử các vị trí khác nhau trong batch để lấy storm ID."""
# # # # # # # # #     for idx in [2, 12, 14, 15]:
# # # # # # # # #         if idx < len(batch):
# # # # # # # # #             item = batch[idx]
# # # # # # # # #             if isinstance(item, (list, tuple)) and len(item) >= B:
# # # # # # # # #                 if isinstance(item[0], str):
# # # # # # # # #                     return list(item[:B])
# # # # # # # # #             if torch.is_tensor(item) and item.dtype == torch.long and item.numel() >= B:
# # # # # # # # #                 return [str(x.item()) for x in item[:B]]
# # # # # # # # #     # Fallback: dùng sequential index
# # # # # # # # #     return [f"STORM_{i:03d}" for i in range(B)]


# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  Model loaders
# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # # def load_fm_model(ckpt_path: str, device, args) -> Optional[object]:
# # # # # # # # #     """Load FlowMatching model từ checkpoint."""
# # # # # # # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # # # # # # #         print(f"  ⚠  FM checkpoint not found: {ckpt_path}")
# # # # # # # # #         return None

# # # # # # # # #     try:
# # # # # # # # #         from Model.flow_matching_model import TCFlowMatching
# # # # # # # # #         model = TCFlowMatching(
# # # # # # # # #             pred_len=args.pred_len,
# # # # # # # # #             obs_len=args.obs_len,
# # # # # # # # #             sigma_min=0.02,
# # # # # # # # #             use_ema=True,
# # # # # # # # #             ema_decay=0.995,
# # # # # # # # #         ).to(device)
# # # # # # # # #         model.init_ema()

# # # # # # # # #         ckpt = torch.load(ckpt_path, map_location=device)
# # # # # # # # #         m = get_raw_model(model)
# # # # # # # # #         missing, unexpected = m.load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # # # # # #         if missing:    print(f"    FM missing keys: {len(missing)}")
# # # # # # # # #         if unexpected: print(f"    FM unexpected keys: {len(unexpected)}")

# # # # # # # # #         # Load EMA
# # # # # # # # #         if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
# # # # # # # # #             ema = get_ema_obj(model)
# # # # # # # # #             if ema is not None:
# # # # # # # # #                 for k, v in ckpt["ema_shadow"].items():
# # # # # # # # #                     if k in ema.shadow:
# # # # # # # # #                         ema.shadow[k].copy_(v.to(device))

# # # # # # # # #         ep = ckpt.get("epoch", "?")
# # # # # # # # #         print(f"  ✅ FM model loaded (epoch={ep}) from {ckpt_path}")
# # # # # # # # #         return model

# # # # # # # # #     except Exception as e:
# # # # # # # # #         print(f"  ❌ FM model load failed: {e}")
# # # # # # # # #         import traceback; traceback.print_exc()
# # # # # # # # #         return None


# # # # # # # # # def load_lstm_model(ckpt_path: str, device, args) -> Optional[object]:
# # # # # # # # #     """Load LSTM model từ checkpoint."""
# # # # # # # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # # # # # # #         print(f"  ⚠  LSTM checkpoint not found: {ckpt_path}")
# # # # # # # # #         return None

# # # # # # # # #     try:
# # # # # # # # #         # Thử import từ các vị trí phổ biến
# # # # # # # # #         try:
# # # # # # # # #             from Model.paper_baseline_model import TCLSTMModel as LSTMModel
# # # # # # # # #         except ImportError:
# # # # # # # # #             try:
# # # # # # # # #                 from Model.baselines.lstm import LSTMPredictor as LSTMModel
# # # # # # # # #             except ImportError:
# # # # # # # # #                 from Model.LSTM_model import LSTMModel

# # # # # # # # #         model = LSTMModel(
# # # # # # # # #             pred_len=args.pred_len,
# # # # # # # # #             obs_len=args.obs_len,
# # # # # # # # #         ).to(device)

# # # # # # # # #         ckpt = torch.load(ckpt_path, map_location=device)
# # # # # # # # #         key  = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
# # # # # # # # #         missing, unexpected = model.load_state_dict(ckpt[key], strict=False)
# # # # # # # # #         if missing:    print(f"    LSTM missing keys: {len(missing)}")

# # # # # # # # #         ep = ckpt.get("epoch", "?")
# # # # # # # # #         print(f"  ✅ LSTM model loaded (epoch={ep}) from {ckpt_path}")
# # # # # # # # #         return model

# # # # # # # # #     except Exception as e:
# # # # # # # # #         print(f"  ❌ LSTM model load failed: {e}")
# # # # # # # # #         import traceback; traceback.print_exc()
# # # # # # # # #         return None


# # # # # # # # # def load_sttrans_model(ckpt_path: str, device, args) -> Optional[object]:
# # # # # # # # #     """Load ST-Trans model từ checkpoint."""
# # # # # # # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # # # # # # #         print(f"  ⚠  ST-Trans checkpoint not found: {ckpt_path}")
# # # # # # # # #         return None

# # # # # # # # #     try:
# # # # # # # # #         try:
# # # # # # # # #             from Model.st_trans_model import STTransModel
# # # # # # # # #         except ImportError:
# # # # # # # # #             try:
# # # # # # # # #                 from Model.baselines.st_transformer import STTransformer as STTransModel
# # # # # # # # #             except ImportError:
# # # # # # # # #                 from Model.STTrans_model import STTransModel

# # # # # # # # #         model = STTransModel(
# # # # # # # # #             pred_len=args.pred_len,
# # # # # # # # #             obs_len=args.obs_len,
# # # # # # # # #         ).to(device)

# # # # # # # # #         ckpt = torch.load(ckpt_path, map_location=device)
# # # # # # # # #         key  = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
# # # # # # # # #         missing, unexpected = model.load_state_dict(ckpt[key], strict=False)
# # # # # # # # #         if missing:    print(f"    ST-Trans missing keys: {len(missing)}")

# # # # # # # # #         ep = ckpt.get("epoch", "?")
# # # # # # # # #         print(f"  ✅ ST-Trans model loaded (epoch={ep}) from {ckpt_path}")
# # # # # # # # #         return model

# # # # # # # # #     except Exception as e:
# # # # # # # # #         print(f"  ❌ ST-Trans model load failed: {e}")
# # # # # # # # #         import traceback; traceback.print_exc()
# # # # # # # # #         return None


# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  Inference wrappers
# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # # @torch.no_grad()
# # # # # # # # # def infer_fm(model, batch_list, args, device) -> Optional[torch.Tensor]:
# # # # # # # # #     """Inference với FlowMatching model, trả về [T, B, 2] normalized."""
# # # # # # # # #     try:
# # # # # # # # #         ema = get_ema_obj(model)
# # # # # # # # #         backup = None
# # # # # # # # #         if ema is not None:
# # # # # # # # #             try: backup = ema.apply_to(model)
# # # # # # # # #             except: pass

# # # # # # # # #         model.eval()
# # # # # # # # #         pred, _, _ = model.sample(
# # # # # # # # #             batch_list,
# # # # # # # # #             num_ensemble=args.fm_ensemble,
# # # # # # # # #             ddim_steps=args.fm_ode_steps,
# # # # # # # # #             importance_weight=True,
# # # # # # # # #         )
# # # # # # # # #         # pred: [T, B, 2]

# # # # # # # # #         if backup is not None:
# # # # # # # # #             try: ema.restore(model, backup)
# # # # # # # # #             except: pass

# # # # # # # # #         return pred

# # # # # # # # #     except Exception as e:
# # # # # # # # #         print(f"    FM inference error: {e}")
# # # # # # # # #         return None


# # # # # # # # # @torch.no_grad()
# # # # # # # # # def infer_lstm(model, batch_list, args, device) -> Optional[torch.Tensor]:
# # # # # # # # #     """Inference với LSTM model, trả về [T, B, 2] normalized."""
# # # # # # # # #     try:
# # # # # # # # #         model.eval()
# # # # # # # # #         obs_traj = batch_list[0]  # [T_obs, B, D]

# # # # # # # # #         # Thử các interface khác nhau
# # # # # # # # #         try:
# # # # # # # # #             pred = model(batch_list)
# # # # # # # # #         except TypeError:
# # # # # # # # #             try:
# # # # # # # # #                 pred = model(obs_traj)
# # # # # # # # #             except Exception:
# # # # # # # # #                 pred = model.predict(batch_list)

# # # # # # # # #         # Đảm bảo shape [T, B, 2]
# # # # # # # # #         if pred.dim() == 3:
# # # # # # # # #             if pred.shape[0] == pred.shape[1]:
# # # # # # # # #                 pass  # already [T, B, 2]
# # # # # # # # #             elif pred.shape[1] != obs_traj.shape[1]:
# # # # # # # # #                 pred = pred.permute(1, 0, 2)

# # # # # # # # #         return pred[:, :, :2]

# # # # # # # # #     except Exception as e:
# # # # # # # # #         print(f"    LSTM inference error: {e}")
# # # # # # # # #         return None


# # # # # # # # # @torch.no_grad()
# # # # # # # # # def infer_sttrans(model, batch_list, args, device) -> Optional[torch.Tensor]:
# # # # # # # # #     """Inference với ST-Trans model, trả về [T, B, 2] normalized."""
# # # # # # # # #     try:
# # # # # # # # #         model.eval()
# # # # # # # # #         obs_traj = batch_list[0]

# # # # # # # # #         try:
# # # # # # # # #             pred = model(batch_list)
# # # # # # # # #         except TypeError:
# # # # # # # # #             try:
# # # # # # # # #                 pred = model(obs_traj)
# # # # # # # # #             except Exception:
# # # # # # # # #                 pred = model.predict(batch_list)

# # # # # # # # #         if pred.dim() == 3:
# # # # # # # # #             if pred.shape[1] != obs_traj.shape[1]:
# # # # # # # # #                 pred = pred.permute(1, 0, 2)

# # # # # # # # #         return pred[:, :, :2]

# # # # # # # # #     except Exception as e:
# # # # # # # # #         print(f"    ST-Trans inference error: {e}")
# # # # # # # # #         return None


# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  CSV writers
# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # # _PER_STORM_FIELDS = [
# # # # # # # # #     "model", "storm_id", "seq_idx", "lead_h",
# # # # # # # # #     "lon_pred", "lat_pred", "lon_true", "lat_true",
# # # # # # # # #     "dist_km", "ate_km", "cte_km",
# # # # # # # # # ]

# # # # # # # # # _SUMMARY_FIELDS = [
# # # # # # # # #     "model", "storm_id", "n_seq",
# # # # # # # # #     "ADE", "ATE", "CTE",
# # # # # # # # #     "12h_dist", "24h_dist", "48h_dist", "72h_dist",
# # # # # # # # #     "12h_ate",  "24h_ate",  "48h_ate",  "72h_ate",
# # # # # # # # #     "12h_cte",  "24h_cte",  "48h_cte",  "72h_cte",
# # # # # # # # #     "beat_12h", "beat_24h", "beat_48h", "beat_72h",
# # # # # # # # #     "beat_ADE", "beat_ATE", "beat_CTE",
# # # # # # # # # ]


# # # # # # # # # def write_per_storm_csv(acc: MultiStormAccumulator, out_dir: str):
# # # # # # # # #     """Ghi 1 CSV tổng hợp tất cả rows per timestep."""
# # # # # # # # #     ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # #     out = os.path.join(out_dir, f"{acc.model_name}_per_storm.csv")
# # # # # # # # #     with open(out, "w", newline="") as fh:
# # # # # # # # #         w = csv.DictWriter(fh, fieldnames=_PER_STORM_FIELDS)
# # # # # # # # #         w.writeheader()
# # # # # # # # #         for sr in acc.storms.values():
# # # # # # # # #             for row in sr.rows:
# # # # # # # # #                 w.writerow({
# # # # # # # # #                     "model"    : acc.model_name,
# # # # # # # # #                     "storm_id" : row["storm_id"],
# # # # # # # # #                     "seq_idx"  : row["seq_idx"],
# # # # # # # # #                     "lead_h"   : row["lead_h"],
# # # # # # # # #                     "lon_pred" : f"{row['lon_pred']:.4f}",
# # # # # # # # #                     "lat_pred" : f"{row['lat_pred']:.4f}",
# # # # # # # # #                     "lon_true" : f"{row['lon_true']:.4f}",
# # # # # # # # #                     "lat_true" : f"{row['lat_true']:.4f}",
# # # # # # # # #                     "dist_km"  : row["dist_km"],
# # # # # # # # #                     "ate_km"   : row["ate_km"],
# # # # # # # # #                     "cte_km"   : row["cte_km"],
# # # # # # # # #                 })
# # # # # # # # #     print(f"  💾 {out}")
# # # # # # # # #     return out


# # # # # # # # # def write_per_storm_detail_csvs(acc: MultiStormAccumulator, out_dir: str):
# # # # # # # # #     """Ghi 1 CSV riêng cho từng cơn bão."""
# # # # # # # # #     detail_dir = os.path.join(out_dir, "per_storm_detail")
# # # # # # # # #     os.makedirs(detail_dir, exist_ok=True)
# # # # # # # # #     paths = []
# # # # # # # # #     for sid, sr in sorted(acc.storms.items()):
# # # # # # # # #         safe_id = sid.replace(" ", "_").replace("/", "-")
# # # # # # # # #         fpath   = os.path.join(detail_dir, f"{acc.model_name}_{safe_id}.csv")
# # # # # # # # #         with open(fpath, "w", newline="") as fh:
# # # # # # # # #             w = csv.DictWriter(fh, fieldnames=_PER_STORM_FIELDS)
# # # # # # # # #             w.writeheader()
# # # # # # # # #             for row in sr.rows:
# # # # # # # # #                 w.writerow({
# # # # # # # # #                     "model"    : acc.model_name,
# # # # # # # # #                     "storm_id" : row["storm_id"],
# # # # # # # # #                     "seq_idx"  : row["seq_idx"],
# # # # # # # # #                     "lead_h"   : row["lead_h"],
# # # # # # # # #                     "lon_pred" : f"{row['lon_pred']:.4f}",
# # # # # # # # #                     "lat_pred" : f"{row['lat_pred']:.4f}",
# # # # # # # # #                     "lon_true" : f"{row['lon_true']:.4f}",
# # # # # # # # #                     "lat_true" : f"{row['lat_true']:.4f}",
# # # # # # # # #                     "dist_km"  : row["dist_km"],
# # # # # # # # #                     "ate_km"   : row["ate_km"],
# # # # # # # # #                     "cte_km"   : row["cte_km"],
# # # # # # # # #                 })
# # # # # # # # #         paths.append(fpath)
# # # # # # # # #     print(f"  💾 {len(paths)} per-storm detail CSVs → {detail_dir}/")
# # # # # # # # #     return paths


# # # # # # # # # def _fmt(v, t=None, dec=1):
# # # # # # # # #     if not np.isfinite(v): return "nan"
# # # # # # # # #     s = f"{v:.{dec}f}"
# # # # # # # # #     if t is not None: s += "✅" if v < t else "❌"
# # # # # # # # #     return s


# # # # # # # # # def write_summary_csv(
# # # # # # # # #     all_acc: Dict[str, MultiStormAccumulator],
# # # # # # # # #     out_dir: str,
# # # # # # # # # ):
# # # # # # # # #     """Ghi CSV so sánh tất cả models, per-storm + overall."""
# # # # # # # # #     ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # #     out = os.path.join(out_dir, "summary_all_models.csv")
# # # # # # # # #     with open(out, "w", newline="") as fh:
# # # # # # # # #         w = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
# # # # # # # # #         w.writeheader()

# # # # # # # # #         def _b(v, t): return "YES" if (np.isfinite(v) and v < t) else "NO"
# # # # # # # # #         def _f(v):    return f"{v:.2f}" if np.isfinite(v) else "nan"

# # # # # # # # #         for mname, acc in all_acc.items():
# # # # # # # # #             rows = acc.all_summaries() + [acc.overall_summary()]
# # # # # # # # #             for r in rows:
# # # # # # # # #                 w.writerow({
# # # # # # # # #                     "model"    : mname,
# # # # # # # # #                     "storm_id" : r["storm_id"],
# # # # # # # # #                     "n_seq"    : r["n_seq"],
# # # # # # # # #                     "ADE"      : _f(r["ADE"]), "ATE": _f(r["ATE"]), "CTE": _f(r["CTE"]),
# # # # # # # # #                     "12h_dist" : _f(r["12h_dist"]), "24h_dist": _f(r["24h_dist"]),
# # # # # # # # #                     "48h_dist" : _f(r["48h_dist"]), "72h_dist": _f(r["72h_dist"]),
# # # # # # # # #                     "12h_ate"  : _f(r["12h_ate"]),  "24h_ate":  _f(r["24h_ate"]),
# # # # # # # # #                     "48h_ate"  : _f(r["48h_ate"]),  "72h_ate":  _f(r["72h_ate"]),
# # # # # # # # #                     "12h_cte"  : _f(r["12h_cte"]),  "24h_cte":  _f(r["24h_cte"]),
# # # # # # # # #                     "48h_cte"  : _f(r["48h_cte"]),  "72h_cte":  _f(r["72h_cte"]),
# # # # # # # # #                     "beat_12h" : _b(r["12h_dist"], TARGETS["12h"]),
# # # # # # # # #                     "beat_24h" : _b(r["24h_dist"], TARGETS["24h"]),
# # # # # # # # #                     "beat_48h" : _b(r["48h_dist"], TARGETS["48h"]),
# # # # # # # # #                     "beat_72h" : _b(r["72h_dist"], TARGETS["72h"]),
# # # # # # # # #                     "beat_ADE" : _b(r["ADE"],      TARGETS["ADE"]),
# # # # # # # # #                     "beat_ATE" : _b(r["ATE"],      TARGETS["ATE"]),
# # # # # # # # #                     "beat_CTE" : _b(r["CTE"],      TARGETS["CTE"]),
# # # # # # # # #                 })
# # # # # # # # #     print(f"  💾 Summary: {out}")
# # # # # # # # #     return out


# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  Pretty print tables
# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # # def print_model_results(acc: MultiStormAccumulator, model_color: str = "🔴"):
# # # # # # # # #     """In bảng kết quả từng bão + tổng thể cho 1 model."""
# # # # # # # # #     print(f"\n{'═'*115}")
# # # # # # # # #     cfg  = MODEL_CONFIGS.get(acc.model_name, {})
# # # # # # # # #     ref  = cfg.get("ref", "")
# # # # # # # # #     col  = cfg.get("color", model_color)
# # # # # # # # #     print(f"  {col}  {acc.model_name}  |  {ref}")
# # # # # # # # #     print(f"{'═'*115}")

# # # # # # # # #     # Header
# # # # # # # # #     hdr = (
# # # # # # # # #         f"  {'Storm':<16} {'N':>4}  "
# # # # # # # # #         f"{'ADE':>8}  {'ATE':>8}  {'CTE':>8}  "
# # # # # # # # #         f"║  {'12h':>7}  {'24h':>7}  {'48h':>7}  {'72h':>7}  "
# # # # # # # # #         f"║  {'ATE@72':>8}  {'CTE@72':>8}"
# # # # # # # # #     )
# # # # # # # # #     print(hdr)
# # # # # # # # #     print("  " + "─" * 111)

# # # # # # # # #     rows = acc.all_summaries()
# # # # # # # # #     for r in rows:
# # # # # # # # #         sid  = r["storm_id"]
# # # # # # # # #         n    = r["n_seq"]
# # # # # # # # #         ade  = r["ADE"]
# # # # # # # # #         ate  = r["ATE"]
# # # # # # # # #         cte  = r["CTE"]
# # # # # # # # #         d12  = r["12h_dist"]; d24 = r["24h_dist"]
# # # # # # # # #         d48  = r["48h_dist"]; d72 = r["72h_dist"]
# # # # # # # # #         a72  = r["72h_ate"];  c72 = r["72h_cte"]

# # # # # # # # #         def v(x, t=None, w=8, dec=1):
# # # # # # # # #             if not np.isfinite(x): return f"{'nan':>{w+2}}"
# # # # # # # # #             s = f"{x:>{w}.{dec}f}"
# # # # # # # # #             if t: s += "✅" if x < t else "❌"
# # # # # # # # #             else:  s += "  "
# # # # # # # # #             return s

# # # # # # # # #         print(
# # # # # # # # #             f"  🌀 {sid:<14} {n:>4}  "
# # # # # # # # #             f"{v(ade,TARGETS['ADE'])}  {v(ate,TARGETS['ATE'])}  {v(cte,TARGETS['CTE'])}  "
# # # # # # # # #             f"║  {v(d12,TARGETS['12h'],7)}  {v(d24,TARGETS['24h'],7)}  "
# # # # # # # # #             f"{v(d48,TARGETS['48h'],7)}  {v(d72,TARGETS['72h'],7)}  "
# # # # # # # # #             f"║  {v(a72,None,8)}  {v(c72,None,8)}"
# # # # # # # # #         )

# # # # # # # # #     # Overall
# # # # # # # # #     ov = acc.overall_summary()
# # # # # # # # #     print("  " + "─" * 111)

# # # # # # # # #     def vo(x, t=None, w=8, dec=1):
# # # # # # # # #         if not np.isfinite(x): return f"{'nan':>{w+2}}"
# # # # # # # # #         s = f"{x:>{w}.{dec}f}"
# # # # # # # # #         if t: s += "✅" if x < t else "❌"
# # # # # # # # #         else:  s += "  "
# # # # # # # # #         return s

# # # # # # # # #     print(
# # # # # # # # #         f"  📊 {'OVERALL':<14} {ov['n_seq']:>4}  "
# # # # # # # # #         f"{vo(ov['ADE'],TARGETS['ADE'])}  {vo(ov['ATE'],TARGETS['ATE'])}  "
# # # # # # # # #         f"{vo(ov['CTE'],TARGETS['CTE'])}  "
# # # # # # # # #         f"║  {vo(ov['12h_dist'],TARGETS['12h'],7)}  {vo(ov['24h_dist'],TARGETS['24h'],7)}  "
# # # # # # # # #         f"{vo(ov['48h_dist'],TARGETS['48h'],7)}  {vo(ov['72h_dist'],TARGETS['72h'],7)}  "
# # # # # # # # #         f"║  {vo(ov['72h_ate'],None,8)}  {vo(ov['72h_cte'],None,8)}"
# # # # # # # # #     )
# # # # # # # # #     print(f"{'═'*115}\n")


# # # # # # # # # def print_comparison_table(all_acc: Dict[str, MultiStormAccumulator]):
# # # # # # # # #     """In bảng so sánh tất cả models."""
# # # # # # # # #     print(f"\n{'═'*100}")
# # # # # # # # #     print(f"  📊  MODEL COMPARISON  —  Overall Test Set")
# # # # # # # # #     print(f"{'═'*100}")
# # # # # # # # #     print(
# # # # # # # # #         f"  {'Model':<14}  {'ADE':>9}  {'ATE':>9}  {'CTE':>9}  "
# # # # # # # # #         f"║  {'12h':>8}  {'24h':>8}  {'48h':>8}  {'72h':>8}"
# # # # # # # # #     )
# # # # # # # # #     print("  " + "─" * 96)

# # # # # # # # #     def vv(x, t=None, w=9, dec=1):
# # # # # # # # #         if not np.isfinite(x): return f"{'nan':>{w+2}}"
# # # # # # # # #         s = f"{x:>{w}.{dec}f}"
# # # # # # # # #         if t: s += "✅" if x < t else "❌"
# # # # # # # # #         else:  s += "  "
# # # # # # # # #         return s

# # # # # # # # #     for mname, acc in all_acc.items():
# # # # # # # # #         cfg = MODEL_CONFIGS.get(mname, {})
# # # # # # # # #         col = cfg.get("color", "⚪")
# # # # # # # # #         ov  = acc.overall_summary()
# # # # # # # # #         print(
# # # # # # # # #             f"  {col} {mname:<12}  "
# # # # # # # # #             f"{vv(ov['ADE'],TARGETS['ADE'])}  {vv(ov['ATE'],TARGETS['ATE'])}  "
# # # # # # # # #             f"{vv(ov['CTE'],TARGETS['CTE'])}  "
# # # # # # # # #             f"║  {vv(ov['12h_dist'],TARGETS['12h'],8)}  "
# # # # # # # # #             f"{vv(ov['24h_dist'],TARGETS['24h'],8)}  "
# # # # # # # # #             f"{vv(ov['48h_dist'],TARGETS['48h'],8)}  "
# # # # # # # # #             f"{vv(ov['72h_dist'],TARGETS['72h'],8)}"
# # # # # # # # #         )

# # # # # # # # #     print("  " + "─" * 96)
# # # # # # # # #     print(f"  Targets: "
# # # # # # # # #           f"ADE<{TARGETS['ADE']} | ATE<{TARGETS['ATE']} | CTE<{TARGETS['CTE']} | "
# # # # # # # # #           f"12h<{TARGETS['12h']} | 24h<{TARGETS['24h']} | "
# # # # # # # # #           f"48h<{TARGETS['48h']} | 72h<{TARGETS['72h']} (km)")
# # # # # # # # #     print(f"{'═'*100}\n")


# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  Main evaluation loop
# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # # def run_evaluation(
# # # # # # # # #     model,
# # # # # # # # #     model_name: str,
# # # # # # # # #     test_loader: DataLoader,
# # # # # # # # #     device,
# # # # # # # # #     args,
# # # # # # # # #     infer_fn,
# # # # # # # # # ) -> MultiStormAccumulator:
# # # # # # # # #     """Chạy toàn bộ evaluation loop cho 1 model."""
# # # # # # # # #     acc = MultiStormAccumulator(model_name)
# # # # # # # # #     n_batches = len(test_loader)
# # # # # # # # #     t0 = time.perf_counter()

# # # # # # # # #     print(f"\n  Running {model_name} on {n_batches} batches...")

# # # # # # # # #     for i, batch in enumerate(test_loader):
# # # # # # # # #         bl = move_to_device(list(batch), device)
# # # # # # # # #         B  = bl[0].shape[1]

# # # # # # # # #         # Inference
# # # # # # # # #         pred_norm = infer_fn(model, bl, args, device)
# # # # # # # # #         if pred_norm is None:
# # # # # # # # #             print(f"    batch {i}: inference returned None, skipping")
# # # # # # # # #             continue

# # # # # # # # #         gt_norm   = bl[1]

# # # # # # # # #         # Lấy storm IDs
# # # # # # # # #         storm_ids = extract_storm_ids(batch, B)

# # # # # # # # #         # Accumulate
# # # # # # # # #         T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # # #         acc.add_batch(pred_norm[:T], gt_norm[:T], storm_ids)

# # # # # # # # #         # Progress
# # # # # # # # #         if (i + 1) % max(1, n_batches // 10) == 0 or i == n_batches - 1:
# # # # # # # # #             elapsed = time.perf_counter() - t0
# # # # # # # # #             ov = acc.overall_summary()
# # # # # # # # #             ade_str = f"{ov['ADE']:.1f}km" if np.isfinite(ov["ADE"]) else "nan"
# # # # # # # # #             print(f"    [{i+1:>4}/{n_batches}]  ADE={ade_str}  "
# # # # # # # # #                   f"storms={len(acc.storms)}  elapsed={elapsed:.1f}s")

# # # # # # # # #     elapsed = time.perf_counter() - t0
# # # # # # # # #     ov = acc.overall_summary()
# # # # # # # # #     print(f"  ✅ {model_name} done in {elapsed:.1f}s  "
# # # # # # # # #           f"|  {len(acc.storms)} storms  "
# # # # # # # # #           f"|  ADE={ov['ADE']:.1f}  ATE={ov['ATE']:.1f}  CTE={ov['CTE']:.1f} km")
# # # # # # # # #     return acc


# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  Args
# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # # def get_args():
# # # # # # # # #     p = argparse.ArgumentParser(
# # # # # # # # #         description="Evaluate LSTM / ST-Trans / FlowMatching on private test set (12 storms)",
# # # # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
# # # # # # # # #     )
# # # # # # # # #     # Dataset
# # # # # # # # #     p.add_argument("--dataset_root",   default="TCND_vn", type=str,
# # # # # # # # #                    help="Root thư mục TC-OFM (chứa Data1d, Data3d, Env_Data)")
# # # # # # # # #     p.add_argument("--obs_len",        default=8,  type=int)
# # # # # # # # #     p.add_argument("--pred_len",       default=12, type=int)
# # # # # # # # #     p.add_argument("--batch_size",     default=16, type=int)
# # # # # # # # #     p.add_argument("--num_workers",    default=2,  type=int)
# # # # # # # # #     p.add_argument("--skip",           default=1,  type=int)
# # # # # # # # #     p.add_argument("--min_ped",        default=1,  type=int)
# # # # # # # # #     p.add_argument("--threshold",      default=0.002, type=float)
# # # # # # # # #     p.add_argument("--other_modal",    default="gph")
# # # # # # # # #     p.add_argument("--test_year",      default=None, type=int)
# # # # # # # # #     p.add_argument("--delim",          default=" ")

# # # # # # # # #     # Model checkpoints
# # # # # # # # #     p.add_argument("--fm_ckpt",      default=None, type=str,
# # # # # # # # #                    help="Path tới FM checkpoint (best_model.pth)")
# # # # # # # # #     p.add_argument("--lstm_ckpt",    default=None, type=str,
# # # # # # # # #                    help="Path tới LSTM checkpoint")
# # # # # # # # #     p.add_argument("--sttrans_ckpt", default=None, type=str,
# # # # # # # # #                    help="Path tới ST-Trans checkpoint")

# # # # # # # # #     # FM inference settings
# # # # # # # # #     p.add_argument("--fm_ensemble",  default=30,  type=int,
# # # # # # # # #                    help="Số ensemble members cho FM")
# # # # # # # # #     p.add_argument("--fm_ode_steps", default=20,  type=int,
# # # # # # # # #                    help="Số ODE steps cho FM")

# # # # # # # # #     # Output
# # # # # # # # #     p.add_argument("--output_dir",   default="results/test_eval", type=str)
# # # # # # # # #     p.add_argument("--gpu_num",      default="0", type=str)
# # # # # # # # #     p.add_argument("--no_detail_csv", action="store_true",
# # # # # # # # #                    help="Không ghi CSV chi tiết từng bão")
# # # # # # # # #     p.add_argument("--eval_models",  default="all", type=str,
# # # # # # # # #                    help="Models cần eval: 'all' hoặc 'fm,lstm,sttrans'")

# # # # # # # # #     return p.parse_args()


# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  MAIN
# # # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # # def main(args):
# # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # # # # #     ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# # # # # # # # #     print("=" * 80)
# # # # # # # # #     print(f"  TC Model Evaluator — Private Test Set (12 storms)")
# # # # # # # # #     print(f"  Started: {ts_start}")
# # # # # # # # #     print(f"  Device : {device}")
# # # # # # # # #     print(f"  Output : {args.output_dir}")
# # # # # # # # #     print("=" * 80)

# # # # # # # # #     # Xác định models cần eval
# # # # # # # # #     eval_set = set()
# # # # # # # # #     if args.eval_models == "all":
# # # # # # # # #         eval_set = {"fm", "lstm", "sttrans"}
# # # # # # # # #     else:
# # # # # # # # #         eval_set = set(m.strip().lower() for m in args.eval_models.split(","))

# # # # # # # # #     # ── Load test data ────────────────────────────────────────────────────────
# # # # # # # # #     print("\n  Loading test dataset...")
# # # # # # # # #     try:
# # # # # # # # #         from Model.data.loader_training import data_loader
# # # # # # # # #         test_dataset, test_loader = data_loader(
# # # # # # # # #             args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # # # # # # #         print(f"  ✅ Test set: {len(test_dataset)} sequences, "
# # # # # # # # #               f"{len(test_loader)} batches")
# # # # # # # # #     except Exception as e:
# # # # # # # # #         print(f"  ❌ Failed to load test data: {e}")
# # # # # # # # #         import traceback; traceback.print_exc()
# # # # # # # # #         return

# # # # # # # # #     # ── Load & evaluate models ─────────────────────────────────────────────
# # # # # # # # #     all_acc: Dict[str, MultiStormAccumulator] = {}

# # # # # # # # #     # FlowMatching
# # # # # # # # #     if "fm" in eval_set and args.fm_ckpt:
# # # # # # # # #         print(f"\n{'─'*50}")
# # # # # # # # #         print("  Loading FlowMatching (FM_v59)...")
# # # # # # # # #         fm_model = load_fm_model(args.fm_ckpt, device, args)
# # # # # # # # #         if fm_model is not None:
# # # # # # # # #             fm_acc = run_evaluation(
# # # # # # # # #                 fm_model, "FM_v59", test_loader, device, args, infer_fm)
# # # # # # # # #             all_acc["FM_v59"] = fm_acc
# # # # # # # # #             del fm_model
# # # # # # # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # # # # # # #     # LSTM
# # # # # # # # #     if "lstm" in eval_set and args.lstm_ckpt:
# # # # # # # # #         print(f"\n{'─'*50}")
# # # # # # # # #         print("  Loading LSTM...")
# # # # # # # # #         lstm_model = load_lstm_model(args.lstm_ckpt, device, args)
# # # # # # # # #         if lstm_model is not None:
# # # # # # # # #             lstm_acc = run_evaluation(
# # # # # # # # #                 lstm_model, "LSTM", test_loader, device, args, infer_lstm)
# # # # # # # # #             all_acc["LSTM"] = lstm_acc
# # # # # # # # #             del lstm_model
# # # # # # # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # # # # # # #     # ST-Trans
# # # # # # # # #     if "sttrans" in eval_set and args.sttrans_ckpt:
# # # # # # # # #         print(f"\n{'─'*50}")
# # # # # # # # #         print("  Loading ST-Trans...")
# # # # # # # # #         st_model = load_sttrans_model(args.sttrans_ckpt, device, args)
# # # # # # # # #         if st_model is not None:
# # # # # # # # #             st_acc = run_evaluation(
# # # # # # # # #                 st_model, "STTrans", test_loader, device, args, infer_sttrans)
# # # # # # # # #             all_acc["STTrans"] = st_acc
# # # # # # # # #             del st_model
# # # # # # # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # # # # # # #     if not all_acc:
# # # # # # # # #         print("\n  ❌ No models were evaluated. "
# # # # # # # # #               "Check --fm_ckpt / --lstm_ckpt / --sttrans_ckpt arguments.")
# # # # # # # # #         return

# # # # # # # # #     # ── Print results ──────────────────────────────────────────────────────
# # # # # # # # #     for mname, acc in all_acc.items():
# # # # # # # # #         cfg = MODEL_CONFIGS.get(mname, {})
# # # # # # # # #         print_model_results(acc, cfg.get("color", "⚪"))

# # # # # # # # #     # Comparison table
# # # # # # # # #     if len(all_acc) > 1:
# # # # # # # # #         print_comparison_table(all_acc)

# # # # # # # # #     # ── Write CSVs ─────────────────────────────────────────────────────────
# # # # # # # # #     print("\n  Writing CSVs...")
# # # # # # # # #     csv_paths = []
# # # # # # # # #     for mname, acc in all_acc.items():
# # # # # # # # #         csv_paths.append(write_per_storm_csv(acc, args.output_dir))
# # # # # # # # #         if not args.no_detail_csv:
# # # # # # # # #             write_per_storm_detail_csvs(acc, args.output_dir)

# # # # # # # # #     sum_path = write_summary_csv(all_acc, args.output_dir)
# # # # # # # # #     csv_paths.append(sum_path)

# # # # # # # # #     # ── Final beat summary ─────────────────────────────────────────────────
# # # # # # # # #     print(f"\n{'═'*80}")
# # # # # # # # #     print(f"  FINAL BEAT SUMMARY")
# # # # # # # # #     print(f"{'─'*80}")
# # # # # # # # #     for mname, acc in all_acc.items():
# # # # # # # # #         ov = acc.overall_summary()
# # # # # # # # #         cfg = MODEL_CONFIGS.get(mname, {})
# # # # # # # # #         col = cfg.get("color", "⚪")
# # # # # # # # #         beats = []
# # # # # # # # #         for k, t in [("ADE",TARGETS["ADE"]),("ATE",TARGETS["ATE"]),("CTE",TARGETS["CTE"]),
# # # # # # # # #                      ("12h_dist",TARGETS["12h"]),("24h_dist",TARGETS["24h"]),
# # # # # # # # #                      ("48h_dist",TARGETS["48h"]),("72h_dist",TARGETS["72h"])]:
# # # # # # # # #             label = k.replace("_dist","h") if "_dist" in k else k
# # # # # # # # #             v     = ov.get(k, float("nan"))
# # # # # # # # #             if np.isfinite(v) and v < t:
# # # # # # # # #                 beats.append(f"{label}={v:.1f}✅")
# # # # # # # # #         beat_str = " | ".join(beats) if beats else "no targets beaten"
# # # # # # # # #         print(f"  {col} {mname:<12}: {beat_str}")

# # # # # # # # #     elapsed_total = (datetime.now() - datetime.strptime(ts_start, "%Y-%m-%d %H:%M:%S")).seconds
# # # # # # # # #     print(f"\n  Total time: {elapsed_total}s")
# # # # # # # # #     print(f"  Outputs saved to: {args.output_dir}/")
# # # # # # # # #     for p in csv_paths:
# # # # # # # # #         print(f"    → {os.path.basename(p)}")
# # # # # # # # #     print(f"{'═'*80}\n")


# # # # # # # # # if __name__ == "__main__":
# # # # # # # # #     np.random.seed(42)
# # # # # # # # #     torch.manual_seed(42)
# # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # #         torch.cuda.manual_seed_all(42)

# # # # # # # # #     args = get_args()

# # # # # # # # #     # Kaggle path convenience: tự điền paths nếu chạy trong Kaggle
# # # # # # # # #     _kaggle_base = "/kaggle/working/TC_FM"
# # # # # # # # #     if args.dataset_root == "TCND_vn" and os.path.isdir(_kaggle_base):
# # # # # # # # #         args.dataset_root = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"

# # # # # # # # #     # Auto-detect checkpoints nếu không chỉ định
# # # # # # # # #     _runs = os.path.join(_kaggle_base, "runs") if os.path.isdir(_kaggle_base) else "runs"

# # # # # # # # #     if args.fm_ckpt is None:
# # # # # # # # #         for candidate in [
# # # # # # # # #             os.path.join(_runs, "v59", "best_model.pth"),
# # # # # # # # #             os.path.join(_runs, "v58", "best_model.pth"),
# # # # # # # # #             os.path.join(_runs, "v55", "best_model.pth"),
# # # # # # # # #             "best_model.pth",
# # # # # # # # #         ]:
# # # # # # # # #             if os.path.exists(candidate):
# # # # # # # # #                 args.fm_ckpt = candidate
# # # # # # # # #                 print(f"  Auto-detected FM ckpt: {candidate}")
# # # # # # # # #                 break

# # # # # # # # #     if args.lstm_ckpt is None:
# # # # # # # # #         for candidate in [
# # # # # # # # #             os.path.join(_runs, "lstm", "best_model.pth"),
# # # # # # # # #             os.path.join(_runs, "LSTM", "best_model.pth"),
# # # # # # # # #             os.path.join(_runs, "lstm_baseline", "best_model.pth"),
# # # # # # # # #         ]:
# # # # # # # # #             if os.path.exists(candidate):
# # # # # # # # #                 args.lstm_ckpt = candidate
# # # # # # # # #                 print(f"  Auto-detected LSTM ckpt: {candidate}")
# # # # # # # # #                 break

# # # # # # # # #     if args.sttrans_ckpt is None:
# # # # # # # # #         for candidate in [
# # # # # # # # #             os.path.join(_runs, "sttrans", "best_model.pth"),
# # # # # # # # #             os.path.join(_runs, "STTrans", "best_model.pth"),
# # # # # # # # #             os.path.join(_runs, "st_trans", "best_model.pth"),
# # # # # # # # #         ]:
# # # # # # # # #             if os.path.exists(candidate):
# # # # # # # # #                 args.sttrans_ckpt = candidate
# # # # # # # # #                 print(f"  Auto-detected ST-Trans ckpt: {candidate}")
# # # # # # # # #                 break

# # # # # # # # #     # Nếu không có model nào → báo hướng dẫn
# # # # # # # # #     if not any([args.fm_ckpt, args.lstm_ckpt, args.sttrans_ckpt]):
# # # # # # # # #         print("  ⚠  No checkpoint paths found. Please specify:")
# # # # # # # # #         print("      --fm_ckpt path/to/best_model.pth")
# # # # # # # # #         print("      --lstm_ckpt path/to/lstm_best.pth")
# # # # # # # # #         print("      --sttrans_ckpt path/to/sttrans_best.pth")
# # # # # # # # #         print()
# # # # # # # # #         print("  Or set --eval_models to include only available models:")
# # # # # # # # #         print("      --eval_models fm")
# # # # # # # # #         print()

# # # # # # # # #     main(args)

# # # # # # # # # #     # Đánh giá cả 3 models
# # # # # # # # # # python /kaggle/working/TC_FM/scripts/evaluate_models.py \
# # # # # # # # # #     --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# # # # # # # # # #     --fm_ckpt     /kaggle/working/TC_FM/runs/v59/best_model.pth \
# # # # # # # # # #     --lstm_ckpt   /kaggle/working/TC_FM/runs/lstm/best_model.pth \
# # # # # # # # # #     --sttrans_ckpt /kaggle/working/TC_FM/runs/sttrans/best_model.pth \
# # # # # # # # # #     --output_dir  /kaggle/working/TC_FM/results/test_eval \
# # # # # # # # # #     --fm_ensemble 30 \
# # # # # # # # # #     --fm_ode_steps 20

# # # # # # # # # # # Chỉ đánh giá FM
# # # # # # # # # # python .../evaluate_models.py --eval_models fm --fm_ckpt .../best_model.pth ...

# # # # # # # # """
# # # # # # # # scripts/evaluate_models.py
# # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # # Script đánh giá tất cả models (LSTM/GRU/RNN, ST-Trans, FlowMatching) trên test set.

# # # # # # # # Cách dùng:
# # # # # # # #     python scripts/evaluate_models.py \
# # # # # # # #         --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# # # # # # # #         --fm_ckpt      runs/v61/best_model.pth \
# # # # # # # #         --lstm_ckpt    runs/lstm/best_model.pth \
# # # # # # # #         --sttrans_ckpt runs/sttrans/best_model.pth \
# # # # # # # #         --output_dir   results/test_eval \
# # # # # # # #         --gpu_num 0

# # # # # # # #     # Chọn loại LSTM head (lstm | gru | rnn):
# # # # # # # #         --lstm_type lstm

# # # # # # # #     # Chọn STTrans variant (sttrans | sttrans_ar):
# # # # # # # #         --sttrans_type sttrans

# # # # # # # # Output:
# # # # # # # #     results/test_eval/
# # # # # # # #         ├── FM_v61_per_storm.csv
# # # # # # # #         ├── LSTM_per_storm.csv
# # # # # # # #         ├── STTrans_per_storm.csv
# # # # # # # #         ├── summary_all_models.csv
# # # # # # # #         └── per_storm_detail/
# # # # # # # #             ├── FM_v61_HAIYAN.csv
# # # # # # # #             └── ...
# # # # # # # # """
# # # # # # # # from __future__ import annotations

# # # # # # # # import sys
# # # # # # # # import os
# # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # # import argparse
# # # # # # # # import csv
# # # # # # # # import math
# # # # # # # # import time
# # # # # # # # from collections import defaultdict
# # # # # # # # from datetime import datetime
# # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # import numpy as np
# # # # # # # # import torch
# # # # # # # # import torch.nn.functional as F
# # # # # # # # from torch.utils.data import DataLoader

# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # R_EARTH  = 6371.0
# # # # # # # # DEG2KM   = 111.0
# # # # # # # # DT_HOURS = 6.0

# # # # # # # # LEAD_HOURS = [12, 24, 36, 48, 60, 72]
# # # # # # # # LEAD_STEPS = {h: h // 6 - 1 for h in LEAD_HOURS}   # 0-indexed step

# # # # # # # # TARGETS = {
# # # # # # # #     "12h": 50.0, "24h": 100.0, "48h": 200.0, "72h": 297.0,
# # # # # # # #     "ADE": 136.41, "ATE": 79.94, "CTE": 93.58,
# # # # # # # # }

# # # # # # # # MODEL_CONFIGS = {
# # # # # # # #     "LSTM"    : {"color": "🔵", "ref": "Rahman 2025"},
# # # # # # # #     "GRU"     : {"color": "🔵", "ref": "Rahman 2025"},
# # # # # # # #     "RNN"     : {"color": "🔵", "ref": "Rahman 2025"},
# # # # # # # #     "STTrans" : {"color": "🟡", "ref": "Faiaz 2026"},
# # # # # # # #     "STTransAR": {"color": "🟡", "ref": "Faiaz 2026 AR"},
# # # # # # # #     "FM_v61"  : {"color": "🔴", "ref": "Ours"},
# # # # # # # # }


# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  Coordinate utilities
# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # def norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
# # # # # # # #     out = arr.clone()
# # # # # # # #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# # # # # # # #     return out


# # # # # # # # def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # # # #     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
# # # # # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # # # # #     a    = (torch.sin(dlat / 2).pow(2) +
# # # # # # # #             torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
# # # # # # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# # # # # # # # def compute_ate_cte(pred_deg: torch.Tensor,
# # # # # # # #                     gt_deg:   torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
# # # # # # # #     """ATE, CTE mỗi bước — dùng forward azimuth của GT."""
# # # # # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # #     if T < 2:
# # # # # # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # # # # #         return z, z

# # # # # # # #     lon1 = torch.deg2rad(gt_deg[:T-1, :, 0]); lat1 = torch.deg2rad(gt_deg[:T-1, :, 1])
# # # # # # # #     lon2 = torch.deg2rad(gt_deg[1:T,  :, 0]); lat2 = torch.deg2rad(gt_deg[1:T,  :, 1])
# # # # # # # #     dlon_a = lon2 - lon1
# # # # # # # #     y_a = torch.sin(dlon_a) * torch.cos(lat2)
# # # # # # # #     x_a = torch.cos(lat1)*torch.sin(lat2) - torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon_a)
# # # # # # # #     bear_along = torch.atan2(y_a, x_a)

# # # # # # # #     lon3 = torch.deg2rad(pred_deg[1:T, :, 0]); lat3 = torch.deg2rad(pred_deg[1:T, :, 1])
# # # # # # # #     dlon_e = lon3 - lon2
# # # # # # # #     y_e = torch.sin(dlon_e) * torch.cos(lat3)
# # # # # # # #     x_e = torch.cos(lat2)*torch.sin(lat3) - torch.sin(lat2)*torch.cos(lat3)*torch.cos(dlon_e)
# # # # # # # #     bear_err = torch.atan2(y_e, x_e)

# # # # # # # #     total = haversine_km(pred_deg[1:T], gt_deg[1:T])
# # # # # # # #     angle = bear_err - bear_along
# # # # # # # #     return total * torch.cos(angle), total * torch.sin(angle)


# # # # # # # # def move_to_device(batch, device):
# # # # # # # #     out = list(batch)
# # # # # # # #     for i, x in enumerate(out):
# # # # # # # #         if torch.is_tensor(x):
# # # # # # # #             out[i] = x.to(device)
# # # # # # # #         elif isinstance(x, dict):
# # # # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # # #                       for k, v in x.items()}
# # # # # # # #     return out


# # # # # # # # def get_raw_model(model):
# # # # # # # #     return model._orig_mod if hasattr(model, "_orig_mod") else model


# # # # # # # # def get_ema_obj(model):
# # # # # # # #     """Lấy EMA object từ model hoặc wrapped model."""
# # # # # # # #     for obj in [model, getattr(model, "_orig_mod", None)]:
# # # # # # # #         if obj is not None and hasattr(obj, "_ema") and obj._ema is not None:
# # # # # # # #             return obj._ema
# # # # # # # #     return None


# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  Per-storm result accumulator
# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # class StormResult:
# # # # # # # #     def __init__(self, storm_id: str):
# # # # # # # #         self.storm_id = storm_id
# # # # # # # #         self.rows: List[dict] = []
# # # # # # # #         self.dist_by_lead: Dict[int, List[float]] = defaultdict(list)
# # # # # # # #         self.ate_by_lead:  Dict[int, List[float]] = defaultdict(list)
# # # # # # # #         self.cte_by_lead:  Dict[int, List[float]] = defaultdict(list)
# # # # # # # #         self.all_dist: List[float] = []
# # # # # # # #         self.all_ate:  List[float] = []
# # # # # # # #         self.all_cte:  List[float] = []
# # # # # # # #         self.n_seq = 0

# # # # # # # #     def add_batch(self, pred_deg: torch.Tensor, gt_deg: torch.Tensor,
# # # # # # # #                   seq_indices: Optional[torch.Tensor] = None):
# # # # # # # #         T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # #         B = pred_deg.shape[1]
# # # # # # # #         if T < 2:
# # # # # # # #             return

# # # # # # # #         dist        = haversine_km(pred_deg[:T], gt_deg[:T])   # [T, B]
# # # # # # # #         ate, cte    = compute_ate_cte(pred_deg, gt_deg)         # [T-1, B]
# # # # # # # #         self.n_seq += B

# # # # # # # #         for b in range(B):
# # # # # # # #             for t in range(T):
# # # # # # # #                 lead_h  = (t + 1) * 6
# # # # # # # #                 d_val   = float(dist[t, b])
# # # # # # # #                 ate_val = float(ate[t, b].abs()) if t < ate.shape[0] else float("nan")
# # # # # # # #                 cte_val = float(cte[t, b].abs()) if t < cte.shape[0] else float("nan")

# # # # # # # #                 self.rows.append({
# # # # # # # #                     "storm_id": self.storm_id,
# # # # # # # #                     "seq_idx" : int(seq_indices[b]) if seq_indices is not None else b,
# # # # # # # #                     "lead_h"  : lead_h,
# # # # # # # #                     "lon_pred": float(pred_deg[t, b, 0]),
# # # # # # # #                     "lat_pred": float(pred_deg[t, b, 1]),
# # # # # # # #                     "lon_true": float(gt_deg[t, b, 0]),
# # # # # # # #                     "lat_true": float(gt_deg[t, b, 1]),
# # # # # # # #                     "dist_km" : round(d_val,   2),
# # # # # # # #                     "ate_km"  : round(ate_val, 2) if np.isfinite(ate_val) else "nan",
# # # # # # # #                     "cte_km"  : round(cte_val, 2) if np.isfinite(cte_val) else "nan",
# # # # # # # #                 })

# # # # # # # #                 if lead_h in LEAD_STEPS:
# # # # # # # #                     self.dist_by_lead[lead_h].append(d_val)
# # # # # # # #                     if np.isfinite(ate_val): self.ate_by_lead[lead_h].append(ate_val)
# # # # # # # #                     if np.isfinite(cte_val): self.cte_by_lead[lead_h].append(cte_val)

# # # # # # # #                 self.all_dist.append(d_val)
# # # # # # # #                 if t < ate.shape[0]:
# # # # # # # #                     if np.isfinite(ate_val): self.all_ate.append(ate_val)
# # # # # # # #                     if np.isfinite(cte_val): self.all_cte.append(cte_val)

# # # # # # # #     def summary(self) -> dict:
# # # # # # # #         def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # # # # # # #         r = {"storm_id": self.storm_id, "n_seq": self.n_seq,
# # # # # # # #              "ADE": _m(self.all_dist), "ATE": _m(self.all_ate), "CTE": _m(self.all_cte)}
# # # # # # # #         for h in LEAD_HOURS:
# # # # # # # #             r[f"{h}h_dist"] = _m(self.dist_by_lead.get(h, []))
# # # # # # # #             r[f"{h}h_ate"]  = _m(self.ate_by_lead.get(h,  []))
# # # # # # # #             r[f"{h}h_cte"]  = _m(self.cte_by_lead.get(h,  []))
# # # # # # # #         return r


# # # # # # # # class MultiStormAccumulator:
# # # # # # # #     def __init__(self, model_name: str):
# # # # # # # #         self.model_name = model_name
# # # # # # # #         self.storms: Dict[str, StormResult] = {}

# # # # # # # #     def get_storm(self, storm_id: str) -> StormResult:
# # # # # # # #         if storm_id not in self.storms:
# # # # # # # #             self.storms[storm_id] = StormResult(storm_id)
# # # # # # # #         return self.storms[storm_id]

# # # # # # # #     def add_batch(self, pred_norm: torch.Tensor, gt_norm: torch.Tensor,
# # # # # # # #                   storm_ids: List[str]):
# # # # # # # #         T  = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # #         B  = pred_norm.shape[1]
# # # # # # # #         pd = norm_to_deg(pred_norm[:T])
# # # # # # # #         gd = norm_to_deg(gt_norm[:T])

# # # # # # # #         sid_map: Dict[str, List[int]] = defaultdict(list)
# # # # # # # #         for b, sid in enumerate(storm_ids[:B]):
# # # # # # # #             sid_map[sid].append(b)

# # # # # # # #         for sid, bidx in sid_map.items():
# # # # # # # #             bidx_t = torch.tensor(bidx, dtype=torch.long)
# # # # # # # #             self.get_storm(sid).add_batch(pd[:, bidx_t, :], gd[:, bidx_t, :],
# # # # # # # #                                           seq_indices=bidx_t)

# # # # # # # #     def all_summaries(self) -> List[dict]:
# # # # # # # #         return [self.storms[sid].summary() for sid in sorted(self.storms)]

# # # # # # # #     def overall_summary(self) -> dict:
# # # # # # # #         all_d, all_a, all_c = [], [], []
# # # # # # # #         lead_d: Dict[int, List[float]] = defaultdict(list)
# # # # # # # #         lead_a: Dict[int, List[float]] = defaultdict(list)
# # # # # # # #         lead_c: Dict[int, List[float]] = defaultdict(list)

# # # # # # # #         for sr in self.storms.values():
# # # # # # # #             all_d.extend(sr.all_dist)
# # # # # # # #             all_a.extend(sr.all_ate)
# # # # # # # #             all_c.extend(sr.all_cte)
# # # # # # # #             for h in LEAD_HOURS:
# # # # # # # #                 lead_d[h].extend(sr.dist_by_lead.get(h, []))
# # # # # # # #                 lead_a[h].extend(sr.ate_by_lead.get(h,  []))
# # # # # # # #                 lead_c[h].extend(sr.cte_by_lead.get(h,  []))

# # # # # # # #         def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # # # # # # #         r = {"storm_id": "📊 OVERALL",
# # # # # # # #              "n_seq": sum(s.n_seq for s in self.storms.values()),
# # # # # # # #              "ADE": _m(all_d), "ATE": _m(all_a), "CTE": _m(all_c)}
# # # # # # # #         for h in LEAD_HOURS:
# # # # # # # #             r[f"{h}h_dist"] = _m(lead_d[h])
# # # # # # # #             r[f"{h}h_ate"]  = _m(lead_a[h])
# # # # # # # #             r[f"{h}h_cte"]  = _m(lead_c[h])
# # # # # # # #         return r


# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  Storm ID extraction
# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # def extract_storm_ids(batch, B: int) -> List[str]:
# # # # # # # #     for idx in [2, 12, 14, 15]:
# # # # # # # #         if idx < len(batch):
# # # # # # # #             item = batch[idx]
# # # # # # # #             if isinstance(item, (list, tuple)) and len(item) >= B:
# # # # # # # #                 if isinstance(item[0], str):
# # # # # # # #                     return list(item[:B])
# # # # # # # #             if (torch.is_tensor(item) and item.dtype == torch.long
# # # # # # # #                     and item.numel() >= B):
# # # # # # # #                 return [str(x.item()) for x in item[:B]]
# # # # # # # #     return [f"STORM_{i:03d}" for i in range(B)]


# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  Model loaders
# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # def load_fm_model(ckpt_path: str, device, args):
# # # # # # # #     """Load TCFlowMatching checkpoint."""
# # # # # # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # # # # # #         print(f"  ⚠  FM checkpoint not found: {ckpt_path}"); return None
# # # # # # # #     try:
# # # # # # # #         from Model.flow_matching_model import TCFlowMatching
# # # # # # # #         model = TCFlowMatching(
# # # # # # # #             pred_len=args.pred_len, obs_len=args.obs_len,
# # # # # # # #             sigma_min=0.02, use_ema=True, ema_decay=0.998,
# # # # # # # #         ).to(device)
# # # # # # # #         model.init_ema()

# # # # # # # #         ckpt = torch.load(ckpt_path, map_location=device)
# # # # # # # #         m = get_raw_model(model)
# # # # # # # #         missing, unexpected = m.load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # # # # #         if missing:    print(f"    FM missing keys  : {len(missing)}")
# # # # # # # #         if unexpected: print(f"    FM unexpected keys: {len(unexpected)}")

# # # # # # # #         # Load EMA weights nếu có
# # # # # # # #         if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
# # # # # # # #             ema = get_ema_obj(model)
# # # # # # # #             if ema is not None:
# # # # # # # #                 for k, v in ckpt["ema_shadow"].items():
# # # # # # # #                     if k in ema.shadow:
# # # # # # # #                         ema.shadow[k].copy_(v.to(device))

# # # # # # # #         print(f"  ✅ FM loaded  (epoch={ckpt.get('epoch','?')})  {ckpt_path}")
# # # # # # # #         return model
# # # # # # # #     except Exception as e:
# # # # # # # #         print(f"  ❌ FM load failed: {e}"); import traceback; traceback.print_exc()
# # # # # # # #         return None


# # # # # # # # def load_lstm_model(ckpt_path: str, device, args):
# # # # # # # #     """
# # # # # # # #     Load PaperBaseline (LSTM / GRU / RNN) checkpoint.

# # # # # # # #     PaperBaseline dùng cùng encoder pipeline (FNO3D+Mamba+Env_net) với FM,
# # # # # # # #     chỉ khác prediction head. Import từ Model.paper_baseline_model.
# # # # # # # #     """
# # # # # # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # # # # # #         print(f"  ⚠  LSTM checkpoint not found: {ckpt_path}"); return None
# # # # # # # #     try:
# # # # # # # #         # Import đúng class — PaperBaseline trong paper_baseline_model.py
# # # # # # # #         from Model.paper_baseline_model import PaperBaseline

# # # # # # # #         lstm_type = getattr(args, "lstm_type", "lstm")   # "lstm" | "gru" | "rnn"
# # # # # # # #         model = PaperBaseline(
# # # # # # # #             model_type=lstm_type,
# # # # # # # #             pred_len=args.pred_len,
# # # # # # # #             obs_len=args.obs_len,
# # # # # # # #         ).to(device)

# # # # # # # #         ckpt = torch.load(ckpt_path, map_location=device)

# # # # # # # #         # Thử tất cả key phổ biến — theo thứ tự ưu tiên
# # # # # # # #         # train_paper_baseline.py lưu key "model_state"
# # # # # # # #         if "model_state" in ckpt:
# # # # # # # #             state = ckpt["model_state"]
# # # # # # # #         elif "model_state_dict" in ckpt:
# # # # # # # #             state = ckpt["model_state_dict"]
# # # # # # # #         elif "state_dict" in ckpt:
# # # # # # # #             state = ckpt["state_dict"]
# # # # # # # #         else:
# # # # # # # #             # Fallback cuối: in keys để debug
# # # # # # # #             print(f"    ⚠  Checkpoint keys: {list(ckpt.keys())}")
# # # # # # # #             raise KeyError("Không tìm thấy model weights key. "
# # # # # # # #                            "Thêm 'model_state' / 'model_state_dict' / 'state_dict'.")

# # # # # # # #         missing, unexpected = model.load_state_dict(state, strict=False)
# # # # # # # #         if missing:    print(f"    LSTM missing keys  : {len(missing)}")
# # # # # # # #         if unexpected: print(f"    LSTM unexpected keys: {len(unexpected)}")

# # # # # # # #         print(f"  ✅ LSTM ({lstm_type}) loaded  "
# # # # # # # #               f"(epoch={ckpt.get('epoch','?')})  {ckpt_path}")
# # # # # # # #         return model
# # # # # # # #     except Exception as e:
# # # # # # # #         print(f"  ❌ LSTM load failed: {e}"); import traceback; traceback.print_exc()
# # # # # # # #         return None


# # # # # # # # def load_sttrans_model(ckpt_path: str, device, args):
# # # # # # # #     """
# # # # # # # #     Load STTrans hoặc STTransAR checkpoint.

# # # # # # # #     STTrans dùng cùng PaperEncoder với PaperBaseline.
# # # # # # # #     Import từ Model.st_trans_model.
# # # # # # # #     """
# # # # # # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # # # # # #         print(f"  ⚠  ST-Trans checkpoint not found: {ckpt_path}"); return None
# # # # # # # #     try:
# # # # # # # #         sttrans_type = getattr(args, "sttrans_type", "sttrans")  # "sttrans" | "sttrans_ar"

# # # # # # # #         if sttrans_type == "sttrans_ar":
# # # # # # # #             from Model.st_trans_model import STTransAR as STTransModel
# # # # # # # #         else:
# # # # # # # #             from Model.st_trans_model import STTrans as STTransModel

# # # # # # # #         model = STTransModel(
# # # # # # # #             obs_len=args.obs_len,
# # # # # # # #             pred_len=args.pred_len,
# # # # # # # #         ).to(device)

# # # # # # # #         ckpt = torch.load(ckpt_path, map_location=device)

# # # # # # # #         # Thử tất cả key phổ biến — theo thứ tự ưu tiên
# # # # # # # #         # train_st_trans.py lưu key "model_state"
# # # # # # # #         if "model_state" in ckpt:
# # # # # # # #             state = ckpt["model_state"]
# # # # # # # #         elif "model_state_dict" in ckpt:
# # # # # # # #             state = ckpt["model_state_dict"]
# # # # # # # #         elif "state_dict" in ckpt:
# # # # # # # #             state = ckpt["state_dict"]
# # # # # # # #         else:
# # # # # # # #             print(f"    ⚠  Checkpoint keys: {list(ckpt.keys())}")
# # # # # # # #             raise KeyError("Không tìm thấy model weights key. "
# # # # # # # #                            "Thêm 'model_state' / 'model_state_dict' / 'state_dict'.")

# # # # # # # #         missing, unexpected = model.load_state_dict(state, strict=False)
# # # # # # # #         if missing:    print(f"    STTrans missing keys  : {len(missing)}")
# # # # # # # #         if unexpected: print(f"    STTrans unexpected keys: {len(unexpected)}")

# # # # # # # #         model_tag = "STTransAR" if sttrans_type == "sttrans_ar" else "STTrans"
# # # # # # # #         print(f"  ✅ {model_tag} loaded  "
# # # # # # # #               f"(epoch={ckpt.get('epoch','?')})  {ckpt_path}")
# # # # # # # #         return model
# # # # # # # #     except Exception as e:
# # # # # # # #         print(f"  ❌ ST-Trans load failed: {e}"); import traceback; traceback.print_exc()
# # # # # # # #         return None


# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  Inference wrappers
# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # @torch.no_grad()
# # # # # # # # def infer_fm(model, batch_list, args, device) -> Optional[torch.Tensor]:
# # # # # # # #     """Inference FM — dùng EMA weights. Trả về [T, B, 2] norm."""
# # # # # # # #     try:
# # # # # # # #         ema    = get_ema_obj(model)
# # # # # # # #         backup = None
# # # # # # # #         if ema is not None:
# # # # # # # #             try: backup = ema.apply_to(model)
# # # # # # # #             except: pass

# # # # # # # #         model.eval()
# # # # # # # #         pred, _, _ = model.sample(
# # # # # # # #             batch_list,
# # # # # # # #             num_ensemble=args.fm_ensemble,
# # # # # # # #             ddim_steps=args.fm_ode_steps,
# # # # # # # #             importance_weight=True,
# # # # # # # #         )
# # # # # # # #         # pred: [T, B, 2]
# # # # # # # #         if backup is not None:
# # # # # # # #             try: ema.restore(model, backup)
# # # # # # # #             except: pass
# # # # # # # #         return pred
# # # # # # # #     except Exception as e:
# # # # # # # #         print(f"    FM inference error: {e}"); return None


# # # # # # # # @torch.no_grad()
# # # # # # # # def infer_lstm(model, batch_list, args, device) -> Optional[torch.Tensor]:
# # # # # # # #     """
# # # # # # # #     Inference PaperBaseline (LSTM/GRU/RNN).

# # # # # # # #     PaperBaseline.sample() trả về (pred, me_mean, all_trajs)
# # # # # # # #     trong đó pred shape [T, B, 2] normalised — dùng trực tiếp.

# # # # # # # #     Nếu model không có .sample(), fallback về .forward().
# # # # # # # #     """
# # # # # # # #     try:
# # # # # # # #         model.eval()
# # # # # # # #         if hasattr(model, "sample"):
# # # # # # # #             # Interface chuẩn giống FM: (pred_mean, me_mean, all_trajs)
# # # # # # # #             pred, _, _ = model.sample(batch_list, num_ensemble=1)
# # # # # # # #         else:
# # # # # # # #             pred = model(batch_list)

# # # # # # # #         # Đảm bảo shape [T, B, 2] — chỉ lấy 2 tọa độ đầu
# # # # # # # #         if pred.dim() == 3:
# # # # # # # #             # Nếu shape [B, T, 2] thì permute
# # # # # # # #             if pred.shape[0] != batch_list[0].shape[1]:
# # # # # # # #                 # shape[0] != B → có thể là [T, B, 2] đúng rồi
# # # # # # # #                 pass
# # # # # # # #             if pred.shape[1] == batch_list[0].shape[1]:
# # # # # # # #                 # [T, B, 2] → đúng, không cần permute
# # # # # # # #                 pass
# # # # # # # #             elif pred.shape[0] == batch_list[0].shape[1]:
# # # # # # # #                 # [B, T, 2] → permute
# # # # # # # #                 pred = pred.permute(1, 0, 2)
# # # # # # # #         return pred[..., :2]
# # # # # # # #     except Exception as e:
# # # # # # # #         print(f"    LSTM inference error: {e}"); return None


# # # # # # # # @torch.no_grad()
# # # # # # # # def infer_sttrans(model, batch_list, args, device) -> Optional[torch.Tensor]:
# # # # # # # #     """
# # # # # # # #     Inference STTrans / STTransAR.

# # # # # # # #     STTrans.sample() trả về (pred, me_mean, all_trajs)
# # # # # # # #     trong đó pred shape [T, B, 2] normalised.
# # # # # # # #     STTransAR.sample() tương tự nhưng không dùng teacher forcing.
# # # # # # # #     """
# # # # # # # #     try:
# # # # # # # #         model.eval()
# # # # # # # #         if hasattr(model, "sample"):
# # # # # # # #             pred, _, _ = model.sample(batch_list, num_ensemble=1)
# # # # # # # #         else:
# # # # # # # #             pred = model(batch_list)

# # # # # # # #         # Cùng logic normalize shape như LSTM
# # # # # # # #         if pred.dim() == 3:
# # # # # # # #             if pred.shape[0] == batch_list[0].shape[1]:
# # # # # # # #                 pred = pred.permute(1, 0, 2)   # [B,T,2] → [T,B,2]
# # # # # # # #         return pred[..., :2]
# # # # # # # #     except Exception as e:
# # # # # # # #         print(f"    STTrans inference error: {e}"); return None


# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  CSV writers
# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # _PER_STORM_FIELDS = [
# # # # # # # #     "model", "storm_id", "seq_idx", "lead_h",
# # # # # # # #     "lon_pred", "lat_pred", "lon_true", "lat_true",
# # # # # # # #     "dist_km", "ate_km", "cte_km",
# # # # # # # # ]

# # # # # # # # _SUMMARY_FIELDS = [
# # # # # # # #     "model", "storm_id", "n_seq",
# # # # # # # #     "ADE", "ATE", "CTE",
# # # # # # # #     "12h_dist", "24h_dist", "48h_dist", "72h_dist",
# # # # # # # #     "12h_ate",  "24h_ate",  "48h_ate",  "72h_ate",
# # # # # # # #     "12h_cte",  "24h_cte",  "48h_cte",  "72h_cte",
# # # # # # # #     "beat_12h", "beat_24h", "beat_48h", "beat_72h",
# # # # # # # #     "beat_ADE", "beat_ATE", "beat_CTE",
# # # # # # # # ]


# # # # # # # # def write_per_storm_csv(acc: MultiStormAccumulator, out_dir: str) -> str:
# # # # # # # #     out = os.path.join(out_dir, f"{acc.model_name}_per_storm.csv")
# # # # # # # #     with open(out, "w", newline="") as fh:
# # # # # # # #         w = csv.DictWriter(fh, fieldnames=_PER_STORM_FIELDS)
# # # # # # # #         w.writeheader()
# # # # # # # #         for sr in acc.storms.values():
# # # # # # # #             for row in sr.rows:
# # # # # # # #                 w.writerow({
# # # # # # # #                     "model"    : acc.model_name,
# # # # # # # #                     "storm_id" : row["storm_id"],
# # # # # # # #                     "seq_idx"  : row["seq_idx"],
# # # # # # # #                     "lead_h"   : row["lead_h"],
# # # # # # # #                     "lon_pred" : f"{row['lon_pred']:.4f}",
# # # # # # # #                     "lat_pred" : f"{row['lat_pred']:.4f}",
# # # # # # # #                     "lon_true" : f"{row['lon_true']:.4f}",
# # # # # # # #                     "lat_true" : f"{row['lat_true']:.4f}",
# # # # # # # #                     "dist_km"  : row["dist_km"],
# # # # # # # #                     "ate_km"   : row["ate_km"],
# # # # # # # #                     "cte_km"   : row["cte_km"],
# # # # # # # #                 })
# # # # # # # #     print(f"  💾 {out}")
# # # # # # # #     return out


# # # # # # # # def write_per_storm_detail_csvs(acc: MultiStormAccumulator, out_dir: str) -> List[str]:
# # # # # # # #     detail_dir = os.path.join(out_dir, "per_storm_detail")
# # # # # # # #     os.makedirs(detail_dir, exist_ok=True)
# # # # # # # #     paths = []
# # # # # # # #     for sid, sr in sorted(acc.storms.items()):
# # # # # # # #         safe_id = sid.replace(" ", "_").replace("/", "-")
# # # # # # # #         fpath   = os.path.join(detail_dir, f"{acc.model_name}_{safe_id}.csv")
# # # # # # # #         with open(fpath, "w", newline="") as fh:
# # # # # # # #             w = csv.DictWriter(fh, fieldnames=_PER_STORM_FIELDS)
# # # # # # # #             w.writeheader()
# # # # # # # #             for row in sr.rows:
# # # # # # # #                 w.writerow({
# # # # # # # #                     "model"    : acc.model_name,
# # # # # # # #                     "storm_id" : row["storm_id"],
# # # # # # # #                     "seq_idx"  : row["seq_idx"],
# # # # # # # #                     "lead_h"   : row["lead_h"],
# # # # # # # #                     "lon_pred" : f"{row['lon_pred']:.4f}",
# # # # # # # #                     "lat_pred" : f"{row['lat_pred']:.4f}",
# # # # # # # #                     "lon_true" : f"{row['lon_true']:.4f}",
# # # # # # # #                     "lat_true" : f"{row['lat_true']:.4f}",
# # # # # # # #                     "dist_km"  : row["dist_km"],
# # # # # # # #                     "ate_km"   : row["ate_km"],
# # # # # # # #                     "cte_km"   : row["cte_km"],
# # # # # # # #                 })
# # # # # # # #         paths.append(fpath)
# # # # # # # #     print(f"  💾 {len(paths)} per-storm detail CSVs → {detail_dir}/")
# # # # # # # #     return paths


# # # # # # # # def write_summary_csv(all_acc: Dict[str, MultiStormAccumulator],
# # # # # # # #                       out_dir: str) -> str:
# # # # # # # #     out = os.path.join(out_dir, "summary_all_models.csv")
# # # # # # # #     with open(out, "w", newline="") as fh:
# # # # # # # #         w = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
# # # # # # # #         w.writeheader()

# # # # # # # #         def _b(v, t): return "YES" if (np.isfinite(v) and v < t) else "NO"
# # # # # # # #         def _f(v):    return f"{v:.2f}" if np.isfinite(v) else "nan"

# # # # # # # #         for mname, acc in all_acc.items():
# # # # # # # #             for r in acc.all_summaries() + [acc.overall_summary()]:
# # # # # # # #                 w.writerow({
# # # # # # # #                     "model"    : mname,
# # # # # # # #                     "storm_id" : r["storm_id"],
# # # # # # # #                     "n_seq"    : r["n_seq"],
# # # # # # # #                     "ADE": _f(r["ADE"]), "ATE": _f(r["ATE"]), "CTE": _f(r["CTE"]),
# # # # # # # #                     "12h_dist" : _f(r["12h_dist"]), "24h_dist": _f(r["24h_dist"]),
# # # # # # # #                     "48h_dist" : _f(r["48h_dist"]), "72h_dist": _f(r["72h_dist"]),
# # # # # # # #                     "12h_ate"  : _f(r["12h_ate"]),  "24h_ate":  _f(r["24h_ate"]),
# # # # # # # #                     "48h_ate"  : _f(r["48h_ate"]),  "72h_ate":  _f(r["72h_ate"]),
# # # # # # # #                     "12h_cte"  : _f(r["12h_cte"]),  "24h_cte":  _f(r["24h_cte"]),
# # # # # # # #                     "48h_cte"  : _f(r["48h_cte"]),  "72h_cte":  _f(r["72h_cte"]),
# # # # # # # #                     "beat_12h" : _b(r["12h_dist"], TARGETS["12h"]),
# # # # # # # #                     "beat_24h" : _b(r["24h_dist"], TARGETS["24h"]),
# # # # # # # #                     "beat_48h" : _b(r["48h_dist"], TARGETS["48h"]),
# # # # # # # #                     "beat_72h" : _b(r["72h_dist"], TARGETS["72h"]),
# # # # # # # #                     "beat_ADE" : _b(r["ADE"],      TARGETS["ADE"]),
# # # # # # # #                     "beat_ATE" : _b(r["ATE"],      TARGETS["ATE"]),
# # # # # # # #                     "beat_CTE" : _b(r["CTE"],      TARGETS["CTE"]),
# # # # # # # #                 })
# # # # # # # #     print(f"  💾 Summary: {out}")
# # # # # # # #     return out


# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  Pretty print
# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # def print_model_results(acc: MultiStormAccumulator):
# # # # # # # #     cfg = MODEL_CONFIGS.get(acc.model_name, {})
# # # # # # # #     col = cfg.get("color", "⚪"); ref = cfg.get("ref", "")
# # # # # # # #     print(f"\n{'═'*115}")
# # # # # # # #     print(f"  {col}  {acc.model_name}  |  {ref}")
# # # # # # # #     print(f"{'═'*115}")
# # # # # # # #     print(f"  {'Storm':<16} {'N':>4}  "
# # # # # # # #           f"{'ADE':>8}  {'ATE':>8}  {'CTE':>8}  "
# # # # # # # #           f"║  {'12h':>7}  {'24h':>7}  {'48h':>7}  {'72h':>7}  "
# # # # # # # #           f"║  {'ATE@72':>8}  {'CTE@72':>8}")
# # # # # # # #     print("  " + "─" * 111)

# # # # # # # #     def v(x, t=None, w=8, dec=1):
# # # # # # # #         if not np.isfinite(x): return f"{'nan':>{w+2}}"
# # # # # # # #         s = f"{x:>{w}.{dec}f}"
# # # # # # # #         if t is not None: s += "✅" if x < t else "❌"
# # # # # # # #         else: s += "  "
# # # # # # # #         return s

# # # # # # # #     for r in acc.all_summaries():
# # # # # # # #         print(
# # # # # # # #             f"  🌀 {r['storm_id']:<14} {r['n_seq']:>4}  "
# # # # # # # #             f"{v(r['ADE'],TARGETS['ADE'])}  {v(r['ATE'],TARGETS['ATE'])}  "
# # # # # # # #             f"{v(r['CTE'],TARGETS['CTE'])}  "
# # # # # # # #             f"║  {v(r['12h_dist'],TARGETS['12h'],7)}  {v(r['24h_dist'],TARGETS['24h'],7)}  "
# # # # # # # #             f"{v(r['48h_dist'],TARGETS['48h'],7)}  {v(r['72h_dist'],TARGETS['72h'],7)}  "
# # # # # # # #             f"║  {v(r['72h_ate'],None,8)}  {v(r['72h_cte'],None,8)}"
# # # # # # # #         )

# # # # # # # #     ov = acc.overall_summary()
# # # # # # # #     print("  " + "─" * 111)
# # # # # # # #     print(
# # # # # # # #         f"  📊 {'OVERALL':<14} {ov['n_seq']:>4}  "
# # # # # # # #         f"{v(ov['ADE'],TARGETS['ADE'])}  {v(ov['ATE'],TARGETS['ATE'])}  "
# # # # # # # #         f"{v(ov['CTE'],TARGETS['CTE'])}  "
# # # # # # # #         f"║  {v(ov['12h_dist'],TARGETS['12h'],7)}  {v(ov['24h_dist'],TARGETS['24h'],7)}  "
# # # # # # # #         f"{v(ov['48h_dist'],TARGETS['48h'],7)}  {v(ov['72h_dist'],TARGETS['72h'],7)}  "
# # # # # # # #         f"║  {v(ov['72h_ate'],None,8)}  {v(ov['72h_cte'],None,8)}"
# # # # # # # #     )
# # # # # # # #     print(f"{'═'*115}\n")


# # # # # # # # def print_comparison_table(all_acc: Dict[str, MultiStormAccumulator]):
# # # # # # # #     print(f"\n{'═'*100}")
# # # # # # # #     print(f"  📊  MODEL COMPARISON  —  Overall Test Set")
# # # # # # # #     print(f"{'═'*100}")
# # # # # # # #     print(f"  {'Model':<14}  {'ADE':>9}  {'ATE':>9}  {'CTE':>9}  "
# # # # # # # #           f"║  {'12h':>8}  {'24h':>8}  {'48h':>8}  {'72h':>8}")
# # # # # # # #     print("  " + "─" * 96)

# # # # # # # #     def vv(x, t=None, w=9, dec=1):
# # # # # # # #         if not np.isfinite(x): return f"{'nan':>{w+2}}"
# # # # # # # #         s = f"{x:>{w}.{dec}f}"
# # # # # # # #         if t is not None: s += "✅" if x < t else "❌"
# # # # # # # #         else: s += "  "
# # # # # # # #         return s

# # # # # # # #     for mname, acc in all_acc.items():
# # # # # # # #         cfg = MODEL_CONFIGS.get(mname, {}); col = cfg.get("color", "⚪")
# # # # # # # #         ov  = acc.overall_summary()
# # # # # # # #         print(
# # # # # # # #             f"  {col} {mname:<12}  "
# # # # # # # #             f"{vv(ov['ADE'],TARGETS['ADE'])}  {vv(ov['ATE'],TARGETS['ATE'])}  "
# # # # # # # #             f"{vv(ov['CTE'],TARGETS['CTE'])}  "
# # # # # # # #             f"║  {vv(ov['12h_dist'],TARGETS['12h'],8)}  "
# # # # # # # #             f"{vv(ov['24h_dist'],TARGETS['24h'],8)}  "
# # # # # # # #             f"{vv(ov['48h_dist'],TARGETS['48h'],8)}  "
# # # # # # # #             f"{vv(ov['72h_dist'],TARGETS['72h'],8)}"
# # # # # # # #         )

# # # # # # # #     print("  " + "─" * 96)
# # # # # # # #     print(f"  Targets: ADE<{TARGETS['ADE']} | ATE<{TARGETS['ATE']} | "
# # # # # # # #           f"CTE<{TARGETS['CTE']} | "
# # # # # # # #           f"12h<{TARGETS['12h']} | 24h<{TARGETS['24h']} | "
# # # # # # # #           f"48h<{TARGETS['48h']} | 72h<{TARGETS['72h']} (km)")
# # # # # # # #     print(f"{'═'*100}\n")


# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  Main evaluation loop
# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # def run_evaluation(model, model_name: str, test_loader: DataLoader,
# # # # # # # #                    device, args, infer_fn) -> MultiStormAccumulator:
# # # # # # # #     acc      = MultiStormAccumulator(model_name)
# # # # # # # #     n_batches = len(test_loader)
# # # # # # # #     t0       = time.perf_counter()
# # # # # # # #     print(f"\n  Running {model_name} on {n_batches} batches...")

# # # # # # # #     for i, batch in enumerate(test_loader):
# # # # # # # #         bl = move_to_device(list(batch), device)
# # # # # # # #         B  = bl[0].shape[1]

# # # # # # # #         pred_norm = infer_fn(model, bl, args, device)
# # # # # # # #         if pred_norm is None:
# # # # # # # #             print(f"    batch {i}: inference returned None, skipping")
# # # # # # # #             continue

# # # # # # # #         gt_norm   = bl[1]
# # # # # # # #         storm_ids = extract_storm_ids(batch, B)

# # # # # # # #         T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # #         acc.add_batch(pred_norm[:T], gt_norm[:T], storm_ids)

# # # # # # # #         if (i + 1) % max(1, n_batches // 10) == 0 or i == n_batches - 1:
# # # # # # # #             ov      = acc.overall_summary()
# # # # # # # #             ade_str = f"{ov['ADE']:.1f}km" if np.isfinite(ov["ADE"]) else "nan"
# # # # # # # #             print(f"    [{i+1:>4}/{n_batches}]  ADE={ade_str}  "
# # # # # # # #                   f"storms={len(acc.storms)}  "
# # # # # # # #                   f"elapsed={time.perf_counter()-t0:.1f}s")

# # # # # # # #     ov = acc.overall_summary()
# # # # # # # #     print(f"  ✅ {model_name} done in {time.perf_counter()-t0:.1f}s  "
# # # # # # # #           f"|  {len(acc.storms)} storms  "
# # # # # # # #           f"|  ADE={ov['ADE']:.1f}  ATE={ov['ATE']:.1f}  CTE={ov['CTE']:.1f} km")
# # # # # # # #     return acc


# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  Args
# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # def get_args():
# # # # # # # #     p = argparse.ArgumentParser(
# # # # # # # #         description="Evaluate LSTM/GRU/RNN / ST-Trans / FlowMatching on test set",
# # # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
# # # # # # # #     )
# # # # # # # #     # Dataset
# # # # # # # #     p.add_argument("--dataset_root",  default="TCND_vn", type=str)
# # # # # # # #     p.add_argument("--obs_len",       default=8,  type=int)
# # # # # # # #     p.add_argument("--pred_len",      default=12, type=int)
# # # # # # # #     p.add_argument("--batch_size",    default=16, type=int)
# # # # # # # #     p.add_argument("--num_workers",   default=2,  type=int)
# # # # # # # #     p.add_argument("--skip",          default=1,  type=int)
# # # # # # # #     p.add_argument("--min_ped",       default=1,  type=int)
# # # # # # # #     p.add_argument("--threshold",     default=0.002, type=float)
# # # # # # # #     p.add_argument("--other_modal",   default="gph")
# # # # # # # #     p.add_argument("--test_year",     default=None, type=int)
# # # # # # # #     p.add_argument("--delim",         default=" ")

# # # # # # # #     # Model checkpoints
# # # # # # # #     p.add_argument("--fm_ckpt",      default=None, type=str,
# # # # # # # #                    help="Path tới FM checkpoint")
# # # # # # # #     p.add_argument("--lstm_ckpt",    default=None, type=str,
# # # # # # # #                    help="Path tới PaperBaseline checkpoint")
# # # # # # # #     p.add_argument("--sttrans_ckpt", default=None, type=str,
# # # # # # # #                    help="Path tới STTrans checkpoint")

# # # # # # # #     # Model type selection
# # # # # # # #     p.add_argument("--lstm_type",    default="lstm",
# # # # # # # #                    choices=["lstm", "gru", "rnn"],
# # # # # # # #                    help="Loại head cho PaperBaseline")
# # # # # # # #     p.add_argument("--sttrans_type", default="sttrans",
# # # # # # # #                    choices=["sttrans", "sttrans_ar"],
# # # # # # # #                    help="STTrans non-AR hoặc AR variant")

# # # # # # # #     # FM inference
# # # # # # # #     p.add_argument("--fm_ensemble",  default=30, type=int)
# # # # # # # #     p.add_argument("--fm_ode_steps", default=20, type=int)

# # # # # # # #     # Output / control
# # # # # # # #     p.add_argument("--output_dir",    default="results/test_eval", type=str)
# # # # # # # #     p.add_argument("--gpu_num",       default="0", type=str)
# # # # # # # #     p.add_argument("--no_detail_csv", action="store_true")
# # # # # # # #     p.add_argument("--eval_models",   default="all", type=str,
# # # # # # # #                    help="'all' hoặc csv: 'fm,lstm,sttrans'")

# # # # # # # #     return p.parse_args()


# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  MAIN
# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # def main(args):
# # # # # # # #     if torch.cuda.is_available():
# # # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # # # #     ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# # # # # # # #     print("=" * 80)
# # # # # # # #     print(f"  TC Model Evaluator")
# # # # # # # #     print(f"  Started : {ts_start}")
# # # # # # # #     print(f"  Device  : {device}")
# # # # # # # #     print(f"  Output  : {args.output_dir}")
# # # # # # # #     print("=" * 80)

# # # # # # # #     # Xác định models cần eval
# # # # # # # #     if args.eval_models == "all":
# # # # # # # #         eval_set = {"fm", "lstm", "sttrans"}
# # # # # # # #     else:
# # # # # # # #         eval_set = {m.strip().lower() for m in args.eval_models.split(",")}

# # # # # # # #     # ── Load test data ─────────────────────────────────────────────────────
# # # # # # # #     print("\n  Loading test dataset...")
# # # # # # # #     try:
# # # # # # # #         from Model.data.loader_training import data_loader
# # # # # # # #         test_dataset, test_loader = data_loader(
# # # # # # # #             args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # # # # # #         print(f"  ✅ Test set: {len(test_dataset)} sequences, "
# # # # # # # #               f"{len(test_loader)} batches")
# # # # # # # #     except Exception as e:
# # # # # # # #         print(f"  ❌ Failed to load test data: {e}")
# # # # # # # #         import traceback; traceback.print_exc()
# # # # # # # #         return

# # # # # # # #     # ── Eval loop ──────────────────────────────────────────────────────────
# # # # # # # #     all_acc: Dict[str, MultiStormAccumulator] = {}

# # # # # # # #     # FlowMatching
# # # # # # # #     if "fm" in eval_set and args.fm_ckpt:
# # # # # # # #         print(f"\n{'─'*50}")
# # # # # # # #         print("  Loading FlowMatching...")
# # # # # # # #         model = load_fm_model(args.fm_ckpt, device, args)
# # # # # # # #         if model is not None:
# # # # # # # #             # Model name thể hiện version từ path (runs/v61/... → FM_v61)
# # # # # # # #             fm_tag = "FM_" + _version_from_path(args.fm_ckpt)
# # # # # # # #             all_acc[fm_tag] = run_evaluation(model, fm_tag, test_loader,
# # # # # # # #                                              device, args, infer_fm)
# # # # # # # #             del model
# # # # # # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # # # # # #     # LSTM / GRU / RNN
# # # # # # # #     if "lstm" in eval_set and args.lstm_ckpt:
# # # # # # # #         print(f"\n{'─'*50}")
# # # # # # # #         print(f"  Loading PaperBaseline ({args.lstm_type.upper()})...")
# # # # # # # #         model = load_lstm_model(args.lstm_ckpt, device, args)
# # # # # # # #         if model is not None:
# # # # # # # #             tag = args.lstm_type.upper()
# # # # # # # #             all_acc[tag] = run_evaluation(model, tag, test_loader,
# # # # # # # #                                           device, args, infer_lstm)
# # # # # # # #             del model
# # # # # # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # # # # # #     # ST-Trans
# # # # # # # #     if "sttrans" in eval_set and args.sttrans_ckpt:
# # # # # # # #         print(f"\n{'─'*50}")
# # # # # # # #         print(f"  Loading {args.sttrans_type.upper()}...")
# # # # # # # #         model = load_sttrans_model(args.sttrans_ckpt, device, args)
# # # # # # # #         if model is not None:
# # # # # # # #             tag = "STTransAR" if args.sttrans_type == "sttrans_ar" else "STTrans"
# # # # # # # #             all_acc[tag] = run_evaluation(model, tag, test_loader,
# # # # # # # #                                           device, args, infer_sttrans)
# # # # # # # #             del model
# # # # # # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # # # # # #     if not all_acc:
# # # # # # # #         print("\n  ❌ No models were evaluated.")
# # # # # # # #         print("     Specify: --fm_ckpt / --lstm_ckpt / --sttrans_ckpt")
# # # # # # # #         print("     Or:      --eval_models fm  (chỉ eval FM)")
# # # # # # # #         return

# # # # # # # #     # ── Print ──────────────────────────────────────────────────────────────
# # # # # # # #     for acc in all_acc.values():
# # # # # # # #         print_model_results(acc)
# # # # # # # #     if len(all_acc) > 1:
# # # # # # # #         print_comparison_table(all_acc)

# # # # # # # #     # ── Write CSVs ─────────────────────────────────────────────────────────
# # # # # # # #     print("\n  Writing CSVs...")
# # # # # # # #     csv_paths = []
# # # # # # # #     for acc in all_acc.values():
# # # # # # # #         csv_paths.append(write_per_storm_csv(acc, args.output_dir))
# # # # # # # #         if not args.no_detail_csv:
# # # # # # # #             write_per_storm_detail_csvs(acc, args.output_dir)
# # # # # # # #     csv_paths.append(write_summary_csv(all_acc, args.output_dir))

# # # # # # # #     # ── Beat summary ───────────────────────────────────────────────────────
# # # # # # # #     print(f"\n{'═'*80}")
# # # # # # # #     print(f"  FINAL BEAT SUMMARY")
# # # # # # # #     print(f"{'─'*80}")
# # # # # # # #     for mname, acc in all_acc.items():
# # # # # # # #         ov  = acc.overall_summary()
# # # # # # # #         cfg = MODEL_CONFIGS.get(mname, {}); col = cfg.get("color", "⚪")
# # # # # # # #         beats = [
# # # # # # # #             f"{label}={ov.get(k, float('nan')):.1f}✅"
# # # # # # # #             for k, t, label in [
# # # # # # # #                 ("ADE","ADE","ADE"),("ATE","ATE","ATE"),("CTE","CTE","CTE"),
# # # # # # # #                 ("12h_dist","12h","12h"),("24h_dist","24h","24h"),
# # # # # # # #                 ("48h_dist","48h","48h"),("72h_dist","72h","72h"),
# # # # # # # #             ]
# # # # # # # #             if np.isfinite(ov.get(k, float("nan"))) and ov[k] < TARGETS[t]
# # # # # # # #         ]
# # # # # # # #         print(f"  {col} {mname:<12}: "
# # # # # # # #               + (" | ".join(beats) if beats else "no targets beaten"))

# # # # # # # #     elapsed = (datetime.now()
# # # # # # # #                - datetime.strptime(ts_start, "%Y-%m-%d %H:%M:%S")).seconds
# # # # # # # #     print(f"\n  Total time : {elapsed}s")
# # # # # # # #     print(f"  Outputs    : {args.output_dir}/")
# # # # # # # #     for p in csv_paths:
# # # # # # # #         print(f"    → {os.path.basename(p)}")
# # # # # # # #     print(f"{'═'*80}\n")


# # # # # # # # def _version_from_path(ckpt_path: str) -> str:
# # # # # # # #     """Đọc version từ path: runs/v61/best_model.pth → 'v61'."""
# # # # # # # #     parts = os.path.normpath(ckpt_path).split(os.sep)
# # # # # # # #     for p in reversed(parts[:-1]):
# # # # # # # #         if p.startswith("v") and p[1:].isdigit():
# # # # # # # #             return p
# # # # # # # #     return "ours"


# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  Entry point
# # # # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # # # if __name__ == "__main__":
# # # # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # # # #     if torch.cuda.is_available():
# # # # # # # #         torch.cuda.manual_seed_all(42)

# # # # # # # #     args = get_args()

# # # # # # # #     # Kaggle path convenience
# # # # # # # #     _kaggle_base = "/kaggle/working/TC_FM"
# # # # # # # #     if args.dataset_root == "TCND_vn" and os.path.isdir(_kaggle_base):
# # # # # # # #         args.dataset_root = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"

# # # # # # # #     _runs = os.path.join(_kaggle_base, "runs") if os.path.isdir(_kaggle_base) else "runs"

# # # # # # # #     # Auto-detect FM checkpoint
# # # # # # # #     if args.fm_ckpt is None:
# # # # # # # #         for c in [os.path.join(_runs, f"v{v}", "best_model.pth")
# # # # # # # #                   for v in ["61","60","59","58","55"]] + ["best_model.pth"]:
# # # # # # # #             if os.path.exists(c):
# # # # # # # #                 args.fm_ckpt = c
# # # # # # # #                 print(f"  Auto-detected FM ckpt: {c}")
# # # # # # # #                 break

# # # # # # # #     # Auto-detect LSTM checkpoint
# # # # # # # #     if args.lstm_ckpt is None:
# # # # # # # #         for c in [os.path.join(_runs, d, "best_model.pth")
# # # # # # # #                   for d in ["lstm","LSTM","gru","GRU","paper_baseline"]]:
# # # # # # # #             if os.path.exists(c):
# # # # # # # #                 args.lstm_ckpt = c
# # # # # # # #                 print(f"  Auto-detected LSTM ckpt: {c}")
# # # # # # # #                 break

# # # # # # # #     # Auto-detect STTrans checkpoint
# # # # # # # #     if args.sttrans_ckpt is None:
# # # # # # # #         for c in [os.path.join(_runs, d, "best_model.pth")
# # # # # # # #                   for d in ["sttrans","STTrans","st_trans","strans"]]:
# # # # # # # #             if os.path.exists(c):
# # # # # # # #                 args.sttrans_ckpt = c
# # # # # # # #                 print(f"  Auto-detected STTrans ckpt: {c}")
# # # # # # # #                 break

# # # # # # # #     if not any([args.fm_ckpt, args.lstm_ckpt, args.sttrans_ckpt]):
# # # # # # # #         print("  ⚠  No checkpoints found. Example:")
# # # # # # # #         print("      python evaluate_models.py \\")
# # # # # # # #         print("          --fm_ckpt      runs/v61/best_model.pth \\")
# # # # # # # #         print("          --lstm_ckpt    runs/lstm/best_model.pth \\")
# # # # # # # #         print("          --sttrans_ckpt runs/sttrans/best_model.pth \\")
# # # # # # # #         print("          --lstm_type    lstm \\")
# # # # # # # #         print("          --sttrans_type sttrans")

# # # # # # # #     main(args)


# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # USAGE EXAMPLES:
# # # # # # # # #
# # # # # # # # # # Eval cả 3, LSTM head = GRU
# # # # # # # # # python scripts/evaluate_models.py \
# # # # # # # # #     --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# # # # # # # # #     --fm_ckpt      /kaggle/working/TC_FM/runs/v61/best_model.pth \
# # # # # # # # #     --lstm_ckpt    /kaggle/working/TC_FM/runs/lstm/best_model.pth \
# # # # # # # # #     --sttrans_ckpt /kaggle/working/TC_FM/runs/sttrans/best_model.pth \
# # # # # # # # #     --lstm_type    gru \
# # # # # # # # #     --sttrans_type sttrans \
# # # # # # # # #     --fm_ensemble  30 \
# # # # # # # # #     --fm_ode_steps 20 \
# # # # # # # # #     --output_dir   /kaggle/working/TC_FM/results/test_eval
# # # # # # # # #
# # # # # # # # # # Chỉ eval FM + STTransAR
# # # # # # # # # python scripts/evaluate_models.py \
# # # # # # # # #     --eval_models  fm,sttrans \
# # # # # # # # #     --sttrans_type sttrans_ar \
# # # # # # # # #     --fm_ckpt      runs/v61/best_model.pth \
# # # # # # # # #     --sttrans_ckpt runs/sttrans/best_model.pth
# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # """
# # # # # # # evaluate_models.py — Fixed load_fm_model
# # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # Thay toàn bộ hàm load_fm_model trong evaluate_test_storms.py bằng version này.

# # # # # # # Root cause lỗi:
# # # # # # #   RuntimeError: size mismatch for net.transformer.layers.0.linear1.weight:
# # # # # # #     copying a param with shape torch.Size([2048, 256])
# # # # # # #     the shape in current model is torch.Size([1024, 256])

# # # # # # #   checkpoint v60/v61 được train với: FFN=2048, layers=3, + reg_head, vel_obs_expand
# # # # # # #   evaluate script instantiate TCFlowMatching() với defaults v59: FFN=1024, layers=2
# # # # # # #   strict=False không fix được size mismatch — chỉ fix missing/unexpected keys

# # # # # # # Fix: đọc checkpoint trước → detect architecture → patch model → load weights.
# # # # # # # """
# # # # # # # from __future__ import annotations

# # # # # # # import os
# # # # # # # from typing import Optional
# # # # # # # import torch
# # # # # # # import torch.nn as nn


# # # # # # # def get_ema_obj(model):
# # # # # # #     """Lấy EMA object từ model hoặc wrapped model."""
# # # # # # #     for obj in [model, getattr(model, "_orig_mod", None)]:
# # # # # # #         if obj is not None and hasattr(obj, "_ema") and obj._ema is not None:
# # # # # # #             return obj._ema
# # # # # # #     return None


# # # # # # # def load_fm_model(ckpt_path: str, device, args) -> Optional[object]:
# # # # # # #     """
# # # # # # #     Load TCFlowMatching checkpoint — tự động detect architecture.

# # # # # # #     Hỗ trợ mọi version v59/v60/v61:
# # # # # # #       v59: FFN=1024, layers=2, no reg_head, no vel_obs_expand
# # # # # # #       v60: FFN=2048, layers=3, + reg_head (in_dim=1024), + vel_obs_expand
# # # # # # #       v61: giống v60 + physics scoring (không ảnh hưởng load)
# # # # # # #     """
# # # # # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # # # # #         print(f"  ⚠  FM checkpoint not found: {ckpt_path}")
# # # # # # #         return None

# # # # # # #     try:
# # # # # # #         from Model.flow_matching_model import TCFlowMatching

# # # # # # #         # ── 1. Đọc checkpoint ──────────────────────────────────────────────────
# # # # # # #         ckpt = torch.load(ckpt_path, map_location=device)

# # # # # # #         if "model_state_dict" in ckpt:
# # # # # # #             state = ckpt["model_state_dict"]
# # # # # # #         elif "state_dict" in ckpt:
# # # # # # #             state = ckpt["state_dict"]
# # # # # # #         else:
# # # # # # #             # Raw state dict
# # # # # # #             state = ckpt

# # # # # # #         # ── 2. Detect architecture từ weight shapes ────────────────────────────

# # # # # # #         # FFN dim: net.transformer.layers.0.linear1.weight → [ffn_dim, 256]
# # # # # # #         ffn_key = "net.transformer.layers.0.linear1.weight"
# # # # # # #         ffn_dim = 1024  # v59 default
# # # # # # #         if ffn_key in state:
# # # # # # #             ffn_dim = state[ffn_key].shape[0]

# # # # # # #         # Num transformer layers
# # # # # # #         num_layers = 0
# # # # # # #         for i in range(10):
# # # # # # #             if f"net.transformer.layers.{i}.linear1.weight" in state:
# # # # # # #                 num_layers = i + 1
# # # # # # #             else:
# # # # # # #                 break
# # # # # # #         if num_layers == 0:
# # # # # # #             num_layers = 2  # v59 default

# # # # # # #         # reg_head present?
# # # # # # #         has_reg_head = any(k.startswith("net.reg_head.") for k in state)

# # # # # # #         # vel_obs_expand present?
# # # # # # #         has_vel_expand = "net.vel_obs_expand.weight" in state

# # # # # # #         # reg_head in_dim — detect từ weight shape
# # # # # # #         # net.reg_head.net.0.weight → [512, in_dim]
# # # # # # #         reg_in_dim = 1024  # v60/v61 default (ctx+vel_obs+steer+env_kine)
# # # # # # #         reg_key = "net.reg_head.net.0.weight"
# # # # # # #         if reg_key in state:
# # # # # # #             reg_in_dim = state[reg_key].shape[1]

# # # # # # #         print(f"  Detected: FFN={ffn_dim}, layers={num_layers}, "
# # # # # # #               f"reg_head={has_reg_head}(in={reg_in_dim}), "
# # # # # # #               f"vel_expand={has_vel_expand}")

# # # # # # #         # ── 3. Instantiate model ───────────────────────────────────────────────
# # # # # # #         model = TCFlowMatching(
# # # # # # #             pred_len=args.pred_len,
# # # # # # #             obs_len=args.obs_len,
# # # # # # #             sigma_min=0.02,
# # # # # # #             use_ema=True,
# # # # # # #             ema_decay=0.995,
# # # # # # #         ).to(device)
# # # # # # #         model.init_ema()

# # # # # # #         # ── 4. Patch decoder nếu cần ──────────────────────────────────────────
# # # # # # #         # Default TCFlowMatching có thể được build với v59 (1024, 2L)
# # # # # # #         # Cần patch để match checkpoint
# # # # # # #         current_ffn = (model.net.transformer.layers[0]
# # # # # # #                        .linear1.weight.shape[0])
# # # # # # #         current_layers = len(model.net.transformer.layers)

# # # # # # #         if current_ffn != ffn_dim or current_layers != num_layers:
# # # # # # #             model.net.transformer = nn.TransformerDecoder(
# # # # # # #                 nn.TransformerDecoderLayer(
# # # # # # #                     d_model=256, nhead=8,
# # # # # # #                     dim_feedforward=ffn_dim,
# # # # # # #                     dropout=0.10, activation="gelu",
# # # # # # #                     batch_first=True),
# # # # # # #                 num_layers=num_layers
# # # # # # #             ).to(device)
# # # # # # #             print(f"  Patched decoder: {num_layers}L / {ffn_dim}FFN")

# # # # # # #         # ── 5. Patch vel_obs_expand nếu cần ──────────────────────────────────
# # # # # # #         if has_vel_expand and not hasattr(model.net, "vel_obs_expand"):
# # # # # # #             model.net.vel_obs_expand = nn.Linear(256, 256 * 4).to(device)
# # # # # # #             model.net.vel_obs_ln = nn.LayerNorm(256).to(device)
# # # # # # #             print("  Patched vel_obs_expand + vel_obs_ln")
# # # # # # #         elif has_vel_expand:
# # # # # # #             # Đã có nhưng check shape
# # # # # # #             if model.net.vel_obs_expand.weight.shape != (1024, 256):
# # # # # # #                 model.net.vel_obs_expand = nn.Linear(256, 256 * 4).to(device)
# # # # # # #                 model.net.vel_obs_ln = nn.LayerNorm(256).to(device)
# # # # # # #                 print("  Re-patched vel_obs_expand")

# # # # # # #         # ── 6. Patch reg_head nếu cần ─────────────────────────────────────────
# # # # # # #         if has_reg_head:
# # # # # # #             if not hasattr(model.net, "reg_head"):
# # # # # # #                 # Cần import RegressionHead
# # # # # # #                 try:
# # # # # # #                     from Model.flow_matching_model import RegressionHead
# # # # # # #                     # in_dim = reg_in_dim, pred_len từ args
# # # # # # #                     rh = RegressionHead.__new__(RegressionHead)
# # # # # # #                     nn.Module.__init__(rh)
# # # # # # #                     rh.pred_len = args.pred_len
# # # # # # #                     rh.net = nn.Sequential(
# # # # # # #                         nn.Linear(reg_in_dim, 512),
# # # # # # #                         nn.GELU(), nn.LayerNorm(512), nn.Dropout(0.10),
# # # # # # #                         nn.Linear(512, 512),
# # # # # # #                         nn.GELU(), nn.LayerNorm(512),
# # # # # # #                         nn.Linear(512, 256),
# # # # # # #                         nn.GELU(),
# # # # # # #                         nn.Linear(256, args.pred_len * 2),
# # # # # # #                     ).to(device)
# # # # # # #                     model.net.reg_head = rh.to(device)
# # # # # # #                     print(f"  Patched reg_head (in_dim={reg_in_dim})")
# # # # # # #                 except Exception as e:
# # # # # # #                     print(f"  ⚠  reg_head patch failed: {e} — continuing")
# # # # # # #             else:
# # # # # # #                 # Có reg_head nhưng check in_dim
# # # # # # #                 current_reg_in = model.net.reg_head.net[0].weight.shape[1]
# # # # # # #                 if current_reg_in != reg_in_dim:
# # # # # # #                     # Rebuild với đúng in_dim
# # # # # # #                     model.net.reg_head.net = nn.Sequential(
# # # # # # #                         nn.Linear(reg_in_dim, 512),
# # # # # # #                         nn.GELU(), nn.LayerNorm(512), nn.Dropout(0.10),
# # # # # # #                         nn.Linear(512, 512),
# # # # # # #                         nn.GELU(), nn.LayerNorm(512),
# # # # # # #                         nn.Linear(512, 256),
# # # # # # #                         nn.GELU(),
# # # # # # #                         nn.Linear(256, args.pred_len * 2),
# # # # # # #                     ).to(device)
# # # # # # #                     model.net.reg_head.pred_len = args.pred_len
# # # # # # #                     print(f"  Re-patched reg_head in_dim: "
# # # # # # #                           f"{current_reg_in} → {reg_in_dim}")

# # # # # # #         # ── 7. Load weights ────────────────────────────────────────────────────
# # # # # # #         result = model.load_state_dict(state, strict=False)
# # # # # # #         missing    = result.missing_keys
# # # # # # #         unexpected = result.unexpected_keys

# # # # # # #         # Log chỉ các key thực sự bị miss (không phải vì chưa patch)
# # # # # # #         if missing:
# # # # # # #             print(f"  Missing keys  : {len(missing)}")
# # # # # # #             for k in missing[:5]:
# # # # # # #                 print(f"    - {k}")
# # # # # # #             if len(missing) > 5:
# # # # # # #                 print(f"    ... and {len(missing)-5} more")
# # # # # # #         if unexpected:
# # # # # # #             print(f"  Unexpected keys: {len(unexpected)}")

# # # # # # #         # ── 8. Load EMA weights ────────────────────────────────────────────────
# # # # # # #         ema_keys_loaded = 0
# # # # # # #         ema_key = None
# # # # # # #         for k in ["ema_shadow", "ema_state_dict", "ema"]:
# # # # # # #             if k in ckpt and ckpt[k] is not None:
# # # # # # #                 ema_key = k
# # # # # # #                 break

# # # # # # #         if ema_key is not None:
# # # # # # #             ema = get_ema_obj(model)
# # # # # # #             if ema is not None:
# # # # # # #                 for k, v in ckpt[ema_key].items():
# # # # # # #                     if k in ema.shadow:
# # # # # # #                         ema.shadow[k].copy_(v.to(device))
# # # # # # #                         ema_keys_loaded += 1
# # # # # # #                 print(f"  EMA loaded: {ema_keys_loaded} keys")
# # # # # # #             else:
# # # # # # #                 print("  ⚠  EMA object not found in model")
# # # # # # #         else:
# # # # # # #             print("  ⚠  No EMA weights in checkpoint "
# # # # # # #                   "(keys available: " + str(list(ckpt.keys())
# # # # # # #                   if isinstance(ckpt, dict) else []) + ")")

# # # # # # #         ep = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
# # # # # # #         print(f"  ✅ FM loaded (epoch={ep})  {ckpt_path}")
# # # # # # #         return model

# # # # # # #     except Exception as e:
# # # # # # #         print(f"  ❌ FM load failed: {e}")
# # # # # # #         import traceback
# # # # # # #         traceback.print_exc()
# # # # # # #         return None

# # # # # # """
# # # # # # scripts/evaluate_test_storms.py
# # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # Script đánh giá tất cả models (LSTM/GRU/RNN, ST-Trans, FlowMatching) trên test set.

# # # # # # Cách dùng:
# # # # # #     python scripts/evaluate_test_storms.py \
# # # # # #         --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# # # # # #         --fm_ckpt      runs/v61/best_model.pth \
# # # # # #         --lstm_ckpt    runs/lstm/best_model.pth \
# # # # # #         --sttrans_ckpt runs/sttrans/best_model.pth \
# # # # # #         --output_dir   results/test_eval \
# # # # # #         --gpu_num 0

# # # # # #     # Chọn loại LSTM head (lstm | gru | rnn):
# # # # # #         --lstm_type lstm

# # # # # #     # Chọn STTrans variant (sttrans | sttrans_ar):
# # # # # #         --sttrans_type sttrans

# # # # # # Output:
# # # # # #     results/test_eval/
# # # # # #         ├── FM_v61_per_storm.csv
# # # # # #         ├── LSTM_per_storm.csv
# # # # # #         ├── STTrans_per_storm.csv
# # # # # #         ├── summary_all_models.csv
# # # # # #         └── per_storm_detail/
# # # # # #             ├── FM_v61_HAIYAN.csv
# # # # # #             └── ...

# # # # # # FIX: load_fm_model tự động detect architecture từ checkpoint
# # # # # #      → hỗ trợ v59 (1024FFN/2L) và v60/v61 (2048FFN/3L + reg_head + vel_expand)
# # # # # # """
# # # # # # from __future__ import annotations

# # # # # # import sys
# # # # # # import os
# # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # import argparse
# # # # # # import csv
# # # # # # import math
# # # # # # import time
# # # # # # from collections import defaultdict
# # # # # # from datetime import datetime
# # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # import numpy as np
# # # # # # import torch
# # # # # # import torch.nn as nn
# # # # # # import torch.nn.functional as F
# # # # # # from torch.utils.data import DataLoader

# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # R_EARTH  = 6371.0
# # # # # # DEG2KM   = 111.0
# # # # # # DT_HOURS = 6.0

# # # # # # LEAD_HOURS = [12, 24, 36, 48, 60, 72]
# # # # # # LEAD_STEPS = {h: h // 6 - 1 for h in LEAD_HOURS}

# # # # # # TARGETS = {
# # # # # #     "12h": 50.0, "24h": 100.0, "48h": 200.0, "72h": 297.0,
# # # # # #     "ADE": 136.41, "ATE": 79.94, "CTE": 93.58,
# # # # # # }

# # # # # # MODEL_CONFIGS = {
# # # # # #     "LSTM"     : {"color": "🔵", "ref": "Rahman 2025"},
# # # # # #     "GRU"      : {"color": "🔵", "ref": "Rahman 2025"},
# # # # # #     "RNN"      : {"color": "🔵", "ref": "Rahman 2025"},
# # # # # #     "STTrans"  : {"color": "🟡", "ref": "Faiaz 2026"},
# # # # # #     "STTransAR": {"color": "🟡", "ref": "Faiaz 2026 AR"},
# # # # # #     "FM_v59"   : {"color": "🔴", "ref": "Ours v59"},
# # # # # #     "FM_v60"   : {"color": "🔴", "ref": "Ours v60"},
# # # # # #     "FM_v61"   : {"color": "🔴", "ref": "Ours v61"},
# # # # # #     "FM_ours"  : {"color": "🔴", "ref": "Ours"},
# # # # # # }


# # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # #  Coordinate utilities
# # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # def norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
# # # # # #     out = arr.clone()
# # # # # #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# # # # # #     return out


# # # # # # def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # #     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
# # # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # # #     a    = (torch.sin(dlat / 2).pow(2) +
# # # # # #             torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
# # # # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# # # # # # def compute_ate_cte(
# # # # # #     pred_deg: torch.Tensor,
# # # # # #     gt_deg:   torch.Tensor,
# # # # # # ) -> Tuple[torch.Tensor, torch.Tensor]:
# # # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # #     if T < 2:
# # # # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # # #         return z, z

# # # # # #     lon1 = torch.deg2rad(gt_deg[:T-1, :, 0]); lat1 = torch.deg2rad(gt_deg[:T-1, :, 1])
# # # # # #     lon2 = torch.deg2rad(gt_deg[1:T,  :, 0]); lat2 = torch.deg2rad(gt_deg[1:T,  :, 1])
# # # # # #     dlon_a = lon2 - lon1
# # # # # #     y_a = torch.sin(dlon_a) * torch.cos(lat2)
# # # # # #     x_a = (torch.cos(lat1) * torch.sin(lat2)
# # # # # #            - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon_a))
# # # # # #     bear_along = torch.atan2(y_a, x_a)

# # # # # #     lon3 = torch.deg2rad(pred_deg[1:T, :, 0]); lat3 = torch.deg2rad(pred_deg[1:T, :, 1])
# # # # # #     dlon_e = lon3 - lon2
# # # # # #     y_e = torch.sin(dlon_e) * torch.cos(lat3)
# # # # # #     x_e = (torch.cos(lat2) * torch.sin(lat3)
# # # # # #            - torch.sin(lat2) * torch.cos(lat3) * torch.cos(dlon_e))
# # # # # #     bear_err = torch.atan2(y_e, x_e)

# # # # # #     total = haversine_km(pred_deg[1:T], gt_deg[1:T])
# # # # # #     angle = bear_err - bear_along
# # # # # #     return total * torch.cos(angle), total * torch.sin(angle)


# # # # # # def move_to_device(batch, device):
# # # # # #     out = list(batch)
# # # # # #     for i, x in enumerate(out):
# # # # # #         if torch.is_tensor(x):
# # # # # #             out[i] = x.to(device)
# # # # # #         elif isinstance(x, dict):
# # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # #                       for k, v in x.items()}
# # # # # #     return out


# # # # # # def get_raw_model(model):
# # # # # #     return model._orig_mod if hasattr(model, "_orig_mod") else model


# # # # # # def get_ema_obj(model):
# # # # # #     for obj in [model, getattr(model, "_orig_mod", None)]:
# # # # # #         if obj is not None and hasattr(obj, "_ema") and obj._ema is not None:
# # # # # #             return obj._ema
# # # # # #     return None


# # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # #  Per-storm result accumulator
# # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # class StormResult:
# # # # # #     def __init__(self, storm_id: str):
# # # # # #         self.storm_id = storm_id
# # # # # #         self.rows: List[dict] = []
# # # # # #         self.dist_by_lead: Dict[int, List[float]] = defaultdict(list)
# # # # # #         self.ate_by_lead:  Dict[int, List[float]] = defaultdict(list)
# # # # # #         self.cte_by_lead:  Dict[int, List[float]] = defaultdict(list)
# # # # # #         self.all_dist: List[float] = []
# # # # # #         self.all_ate:  List[float] = []
# # # # # #         self.all_cte:  List[float] = []
# # # # # #         self.n_seq = 0

# # # # # #     def add_batch(self, pred_deg: torch.Tensor, gt_deg: torch.Tensor,
# # # # # #                   seq_indices: Optional[torch.Tensor] = None):
# # # # # #         T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # #         B = pred_deg.shape[1]
# # # # # #         if T < 2:
# # # # # #             return

# # # # # #         dist        = haversine_km(pred_deg[:T], gt_deg[:T])
# # # # # #         ate, cte    = compute_ate_cte(pred_deg, gt_deg)
# # # # # #         self.n_seq += B

# # # # # #         for b in range(B):
# # # # # #             for t in range(T):
# # # # # #                 lead_h  = (t + 1) * 6
# # # # # #                 d_val   = float(dist[t, b])
# # # # # #                 ate_val = float(ate[t, b].abs()) if t < ate.shape[0] else float("nan")
# # # # # #                 cte_val = float(cte[t, b].abs()) if t < cte.shape[0] else float("nan")

# # # # # #                 self.rows.append({
# # # # # #                     "storm_id": self.storm_id,
# # # # # #                     "seq_idx" : int(seq_indices[b]) if seq_indices is not None else b,
# # # # # #                     "lead_h"  : lead_h,
# # # # # #                     "lon_pred": float(pred_deg[t, b, 0]),
# # # # # #                     "lat_pred": float(pred_deg[t, b, 1]),
# # # # # #                     "lon_true": float(gt_deg[t, b, 0]),
# # # # # #                     "lat_true": float(gt_deg[t, b, 1]),
# # # # # #                     "dist_km" : round(d_val,   2),
# # # # # #                     "ate_km"  : round(ate_val, 2) if np.isfinite(ate_val) else "nan",
# # # # # #                     "cte_km"  : round(cte_val, 2) if np.isfinite(cte_val) else "nan",
# # # # # #                 })

# # # # # #                 if lead_h in LEAD_STEPS:
# # # # # #                     self.dist_by_lead[lead_h].append(d_val)
# # # # # #                     if np.isfinite(ate_val): self.ate_by_lead[lead_h].append(ate_val)
# # # # # #                     if np.isfinite(cte_val): self.cte_by_lead[lead_h].append(cte_val)

# # # # # #                 self.all_dist.append(d_val)
# # # # # #                 if t < ate.shape[0]:
# # # # # #                     if np.isfinite(ate_val): self.all_ate.append(ate_val)
# # # # # #                     if np.isfinite(cte_val): self.all_cte.append(cte_val)

# # # # # #     def summary(self) -> dict:
# # # # # #         def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # # # # #         r = {"storm_id": self.storm_id, "n_seq": self.n_seq,
# # # # # #              "ADE": _m(self.all_dist), "ATE": _m(self.all_ate), "CTE": _m(self.all_cte)}
# # # # # #         for h in LEAD_HOURS:
# # # # # #             r[f"{h}h_dist"] = _m(self.dist_by_lead.get(h, []))
# # # # # #             r[f"{h}h_ate"]  = _m(self.ate_by_lead.get(h,  []))
# # # # # #             r[f"{h}h_cte"]  = _m(self.cte_by_lead.get(h,  []))
# # # # # #         return r


# # # # # # class MultiStormAccumulator:
# # # # # #     def __init__(self, model_name: str):
# # # # # #         self.model_name = model_name
# # # # # #         self.storms: Dict[str, StormResult] = {}

# # # # # #     def get_storm(self, storm_id: str) -> StormResult:
# # # # # #         if storm_id not in self.storms:
# # # # # #             self.storms[storm_id] = StormResult(storm_id)
# # # # # #         return self.storms[storm_id]

# # # # # #     def add_batch(self, pred_norm: torch.Tensor, gt_norm: torch.Tensor,
# # # # # #                   storm_ids: List[str]):
# # # # # #         T  = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # #         B  = pred_norm.shape[1]
# # # # # #         pd = norm_to_deg(pred_norm[:T])
# # # # # #         gd = norm_to_deg(gt_norm[:T])

# # # # # #         sid_map: Dict[str, List[int]] = defaultdict(list)
# # # # # #         for b, sid in enumerate(storm_ids[:B]):
# # # # # #             sid_map[sid].append(b)

# # # # # #         for sid, bidx in sid_map.items():
# # # # # #             bidx_t = torch.tensor(bidx, dtype=torch.long)
# # # # # #             self.get_storm(sid).add_batch(
# # # # # #                 pd[:, bidx_t, :], gd[:, bidx_t, :], seq_indices=bidx_t)

# # # # # #     def all_summaries(self) -> List[dict]:
# # # # # #         return [self.storms[sid].summary() for sid in sorted(self.storms)]

# # # # # #     def overall_summary(self) -> dict:
# # # # # #         all_d, all_a, all_c = [], [], []
# # # # # #         lead_d: Dict[int, List[float]] = defaultdict(list)
# # # # # #         lead_a: Dict[int, List[float]] = defaultdict(list)
# # # # # #         lead_c: Dict[int, List[float]] = defaultdict(list)

# # # # # #         for sr in self.storms.values():
# # # # # #             all_d.extend(sr.all_dist)
# # # # # #             all_a.extend(sr.all_ate)
# # # # # #             all_c.extend(sr.all_cte)
# # # # # #             for h in LEAD_HOURS:
# # # # # #                 lead_d[h].extend(sr.dist_by_lead.get(h, []))
# # # # # #                 lead_a[h].extend(sr.ate_by_lead.get(h,  []))
# # # # # #                 lead_c[h].extend(sr.cte_by_lead.get(h,  []))

# # # # # #         def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # # # # #         r = {"storm_id": "📊 OVERALL",
# # # # # #              "n_seq": sum(s.n_seq for s in self.storms.values()),
# # # # # #              "ADE": _m(all_d), "ATE": _m(all_a), "CTE": _m(all_c)}
# # # # # #         for h in LEAD_HOURS:
# # # # # #             r[f"{h}h_dist"] = _m(lead_d[h])
# # # # # #             r[f"{h}h_ate"]  = _m(lead_a[h])
# # # # # #             r[f"{h}h_cte"]  = _m(lead_c[h])
# # # # # #         return r


# # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # #  Storm ID extraction
# # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # def extract_storm_ids(batch, B: int) -> List[str]:
# # # # # #     for idx in [2, 12, 14, 15]:
# # # # # #         if idx < len(batch):
# # # # # #             item = batch[idx]
# # # # # #             if isinstance(item, (list, tuple)) and len(item) >= B:
# # # # # #                 if isinstance(item[0], str):
# # # # # #                     return list(item[:B])
# # # # # #             if (torch.is_tensor(item) and item.dtype == torch.long
# # # # # #                     and item.numel() >= B):
# # # # # #                 return [str(x.item()) for x in item[:B]]
# # # # # #     return [f"STORM_{i:03d}" for i in range(B)]


# # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # #  Model loaders
# # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # def load_fm_model(ckpt_path: str, device, args) -> Optional[object]:
# # # # # #     """
# # # # # #     Load TCFlowMatching — tự động detect architecture từ checkpoint.

# # # # # #     Hỗ trợ mọi version:
# # # # # #       v59: FFN=1024, layers=2, no reg_head, no vel_obs_expand
# # # # # #       v60: FFN=2048, layers=3, reg_head(in=1024), vel_obs_expand
# # # # # #       v61: giống v60 + physics scoring (không ảnh hưởng load)

# # # # # #     FIX root cause lỗi size mismatch:
# # # # # #       strict=False KHÔNG fix size mismatch, chỉ fix missing/unexpected keys.
# # # # # #       Phải patch model architecture TRƯỚC khi load_state_dict.
# # # # # #     """
# # # # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # # # #         print(f"  ⚠  FM checkpoint not found: {ckpt_path}")
# # # # # #         return None
# # # # # #     try:
# # # # # #         from Model.flow_matching_model import TCFlowMatching

# # # # # #         # ── 1. Đọc checkpoint và lấy state dict ───────────────────────────────
# # # # # #         ckpt = torch.load(ckpt_path, map_location=device)
# # # # # #         if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
# # # # # #             state = ckpt["model_state_dict"]
# # # # # #         elif isinstance(ckpt, dict) and "state_dict" in ckpt:
# # # # # #             state = ckpt["state_dict"]
# # # # # #         elif isinstance(ckpt, dict) and "model_state" in ckpt:
# # # # # #             state = ckpt["model_state"]
# # # # # #         else:
# # # # # #             state = ckpt  # raw state dict

# # # # # #         # ── 2. Detect architecture từ weight shapes ────────────────────────────
# # # # # #         # FFN dim từ transformer layer 0
# # # # # #         ffn_key = "net.transformer.layers.0.linear1.weight"
# # # # # #         ffn_dim = 1024  # v59 default
# # # # # #         if ffn_key in state:
# # # # # #             ffn_dim = int(state[ffn_key].shape[0])

# # # # # #         # Số transformer layers
# # # # # #         num_layers = 0
# # # # # #         for i in range(10):
# # # # # #             if f"net.transformer.layers.{i}.linear1.weight" in state:
# # # # # #                 num_layers = i + 1
# # # # # #             else:
# # # # # #                 break
# # # # # #         if num_layers == 0:
# # # # # #             num_layers = 2  # v59 default

# # # # # #         # reg_head
# # # # # #         has_reg_head = any(k.startswith("net.reg_head.") for k in state)
# # # # # #         reg_in_dim   = 1024  # v60/v61 default
# # # # # #         reg_key      = "net.reg_head.net.0.weight"
# # # # # #         if reg_key in state:
# # # # # #             reg_in_dim = int(state[reg_key].shape[1])

# # # # # #         # vel_obs_expand
# # # # # #         has_vel_expand = "net.vel_obs_expand.weight" in state

# # # # # #         print(f"  Detected: FFN={ffn_dim}, layers={num_layers}, "
# # # # # #               f"reg_head={has_reg_head}(in={reg_in_dim}), "
# # # # # #               f"vel_expand={has_vel_expand}")

# # # # # #         # ── 3. Instantiate model với defaults ─────────────────────────────────
# # # # # #         model = TCFlowMatching(
# # # # # #             pred_len=args.pred_len,
# # # # # #             obs_len=args.obs_len,
# # # # # #             sigma_min=0.02,
# # # # # #             use_ema=True,
# # # # # #             ema_decay=0.995,
# # # # # #         ).to(device)
# # # # # #         model.init_ema()

# # # # # #         # ── 4. Patch TransformerDecoder nếu khác default ──────────────────────
# # # # # #         try:
# # # # # #             cur_ffn    = model.net.transformer.layers[0].linear1.weight.shape[0]
# # # # # #             cur_layers = len(model.net.transformer.layers)
# # # # # #         except Exception:
# # # # # #             cur_ffn, cur_layers = 1024, 2

# # # # # #         if cur_ffn != ffn_dim or cur_layers != num_layers:
# # # # # #             model.net.transformer = nn.TransformerDecoder(
# # # # # #                 nn.TransformerDecoderLayer(
# # # # # #                     d_model=256, nhead=8,
# # # # # #                     dim_feedforward=ffn_dim,
# # # # # #                     dropout=0.10, activation="gelu",
# # # # # #                     batch_first=True),
# # # # # #                 num_layers=num_layers
# # # # # #             ).to(device)
# # # # # #             print(f"  Patched decoder → {num_layers}L / {ffn_dim}FFN")

# # # # # #         # ── 5. Patch vel_obs_expand ────────────────────────────────────────────
# # # # # #         if has_vel_expand:
# # # # # #             needs_patch = (not hasattr(model.net, "vel_obs_expand") or
# # # # # #                            model.net.vel_obs_expand.weight.shape[0] != 1024)
# # # # # #             if needs_patch:
# # # # # #                 model.net.vel_obs_expand = nn.Linear(256, 1024).to(device)
# # # # # #                 model.net.vel_obs_ln     = nn.LayerNorm(256).to(device)
# # # # # #                 print("  Patched vel_obs_expand (256→1024)")

# # # # # #         # ── 6. Patch reg_head ──────────────────────────────────────────────────
# # # # # #         if has_reg_head:
# # # # # #             cur_in = -1
# # # # # #             if hasattr(model.net, "reg_head"):
# # # # # #                 try:
# # # # # #                     cur_in = model.net.reg_head.net[0].weight.shape[1]
# # # # # #                 except Exception:
# # # # # #                     cur_in = -1

# # # # # #             if cur_in != reg_in_dim:
# # # # # #                 pred_len = args.pred_len

# # # # # #                 class _RH(nn.Module):
# # # # # #                     def __init__(self, in_dim: int, pl: int):
# # # # # #                         super().__init__()
# # # # # #                         self.pred_len = pl
# # # # # #                         self.net = nn.Sequential(
# # # # # #                             nn.Linear(in_dim, 512),
# # # # # #                             nn.GELU(), nn.LayerNorm(512),
# # # # # #                             nn.Dropout(0.10),
# # # # # #                             nn.Linear(512, 512),
# # # # # #                             nn.GELU(), nn.LayerNorm(512),
# # # # # #                             nn.Linear(512, 256),
# # # # # #                             nn.GELU(),
# # # # # #                             nn.Linear(256, pl * 2),
# # # # # #                         )

# # # # # #                     def forward(self, ctx, vel_obs, steer, env_kine):
# # # # # #                         h = torch.cat([ctx, vel_obs, steer, env_kine], dim=-1)
# # # # # #                         return self.net(h).view(-1, self.pred_len, 2)

# # # # # #                 model.net.reg_head = _RH(reg_in_dim, pred_len).to(device)
# # # # # #                 print(f"  Patched reg_head (in_dim={reg_in_dim}, pred_len={pred_len})")

# # # # # #         # ── 7. Load weights ────────────────────────────────────────────────────
# # # # # #         result = model.load_state_dict(state, strict=False)
# # # # # #         missing    = result.missing_keys
# # # # # #         unexpected = result.unexpected_keys

# # # # # #         if missing:
# # # # # #             print(f"  Missing keys  : {len(missing)}")
# # # # # #             for k in missing[:5]:
# # # # # #                 print(f"    - {k}")
# # # # # #             if len(missing) > 5:
# # # # # #                 print(f"    ... và {len(missing) - 5} more")
# # # # # #         if unexpected:
# # # # # #             print(f"  Unexpected keys: {len(unexpected)}")

# # # # # #         # ── 8. Load EMA weights ────────────────────────────────────────────────
# # # # # #         ema_loaded = 0
# # # # # #         if isinstance(ckpt, dict):
# # # # # #             for ema_key in ["ema_shadow", "ema_state_dict", "ema"]:
# # # # # #                 if ema_key in ckpt and ckpt[ema_key] is not None:
# # # # # #                     ema = get_ema_obj(model)
# # # # # #                     if ema is not None:
# # # # # #                         for k, v in ckpt[ema_key].items():
# # # # # #                             if k in ema.shadow:
# # # # # #                                 ema.shadow[k].copy_(v.to(device))
# # # # # #                                 ema_loaded += 1
# # # # # #                         print(f"  EMA loaded: {ema_loaded} keys")
# # # # # #                     break

# # # # # #         if ema_loaded == 0:
# # # # # #             print("  ⚠  No EMA weights found — using raw model weights")

# # # # # #         ep = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
# # # # # #         print(f"  ✅ FM loaded (epoch={ep})  {ckpt_path}")
# # # # # #         return model

# # # # # #     except Exception as e:
# # # # # #         print(f"  ❌ FM load failed: {e}")
# # # # # #         import traceback
# # # # # #         traceback.print_exc()
# # # # # #         return None


# # # # # # def load_lstm_model(ckpt_path: str, device, args) -> Optional[object]:
# # # # # #     """Load PaperBaseline (LSTM / GRU / RNN) checkpoint."""
# # # # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # # # #         print(f"  ⚠  LSTM checkpoint not found: {ckpt_path}")
# # # # # #         return None
# # # # # #     try:
# # # # # #         from Model.paper_baseline_model import PaperBaseline

# # # # # #         lstm_type = getattr(args, "lstm_type", "lstm")
# # # # # #         model = PaperBaseline(
# # # # # #             model_type=lstm_type,
# # # # # #             pred_len=args.pred_len,
# # # # # #             obs_len=args.obs_len,
# # # # # #         ).to(device)

# # # # # #         ckpt = torch.load(ckpt_path, map_location=device)
# # # # # #         for key in ["model_state", "model_state_dict", "state_dict"]:
# # # # # #             if isinstance(ckpt, dict) and key in ckpt:
# # # # # #                 state = ckpt[key]
# # # # # #                 break
# # # # # #         else:
# # # # # #             state = ckpt

# # # # # #         missing, unexpected = model.load_state_dict(state, strict=False)
# # # # # #         if missing:    print(f"  Missing   : {len(missing)}")
# # # # # #         if unexpected: print(f"  Unexpected: {len(unexpected)}")

# # # # # #         ep = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
# # # # # #         print(f"  ✅ LSTM ({lstm_type}) loaded (epoch={ep})  {ckpt_path}")
# # # # # #         return model

# # # # # #     except Exception as e:
# # # # # #         print(f"  ❌ LSTM load failed: {e}")
# # # # # #         import traceback; traceback.print_exc()
# # # # # #         return None


# # # # # # def load_sttrans_model(ckpt_path: str, device, args) -> Optional[object]:
# # # # # #     """Load STTrans hoặc STTransAR checkpoint."""
# # # # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # # # #         print(f"  ⚠  ST-Trans checkpoint not found: {ckpt_path}")
# # # # # #         return None
# # # # # #     try:
# # # # # #         sttrans_type = getattr(args, "sttrans_type", "sttrans")
# # # # # #         if sttrans_type == "sttrans_ar":
# # # # # #             from Model.st_trans_model import STTransAR as STTransModel
# # # # # #         else:
# # # # # #             from Model.st_trans_model import STTrans as STTransModel

# # # # # #         model = STTransModel(
# # # # # #             obs_len=args.obs_len,
# # # # # #             pred_len=args.pred_len,
# # # # # #         ).to(device)

# # # # # #         ckpt = torch.load(ckpt_path, map_location=device)
# # # # # #         for key in ["model_state", "model_state_dict", "state_dict"]:
# # # # # #             if isinstance(ckpt, dict) and key in ckpt:
# # # # # #                 state = ckpt[key]
# # # # # #                 break
# # # # # #         else:
# # # # # #             state = ckpt

# # # # # #         missing, unexpected = model.load_state_dict(state, strict=False)
# # # # # #         if missing:    print(f"  Missing   : {len(missing)}")
# # # # # #         if unexpected: print(f"  Unexpected: {len(unexpected)}")

# # # # # #         tag = "STTransAR" if sttrans_type == "sttrans_ar" else "STTrans"
# # # # # #         ep  = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
# # # # # #         print(f"  ✅ {tag} loaded (epoch={ep})  {ckpt_path}")
# # # # # #         return model

# # # # # #     except Exception as e:
# # # # # #         print(f"  ❌ ST-Trans load failed: {e}")
# # # # # #         import traceback; traceback.print_exc()
# # # # # #         return None


# # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # #  Inference wrappers
# # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # @torch.no_grad()
# # # # # # def infer_fm(model, batch_list, args, device) -> Optional[torch.Tensor]:
# # # # # #     """Inference FM với EMA weights. Trả về [T, B, 2] norm."""
# # # # # #     try:
# # # # # #         ema    = get_ema_obj(model)
# # # # # #         backup = None
# # # # # #         if ema is not None:
# # # # # #             try: backup = ema.apply_to(model)
# # # # # #             except Exception: pass

# # # # # #         model.eval()
# # # # # #         pred, _, _ = model.sample(
# # # # # #             batch_list,
# # # # # #             num_ensemble=args.fm_ensemble,
# # # # # #             ddim_steps=args.fm_ode_steps,
# # # # # #             importance_weight=True,
# # # # # #         )

# # # # # #         if backup is not None:
# # # # # #             try: ema.restore(model, backup)
# # # # # #             except Exception: pass
# # # # # #         return pred

# # # # # #     except Exception as e:
# # # # # #         print(f"    FM inference error: {e}")
# # # # # #         return None


# # # # # # @torch.no_grad()
# # # # # # def infer_lstm(model, batch_list, args, device) -> Optional[torch.Tensor]:
# # # # # #     """Inference PaperBaseline. Trả về [T, B, 2] norm."""
# # # # # #     try:
# # # # # #         model.eval()
# # # # # #         if hasattr(model, "sample"):
# # # # # #             pred, _, _ = model.sample(batch_list, num_ensemble=1)
# # # # # #         else:
# # # # # #             pred = model(batch_list)

# # # # # #         if pred.dim() == 3:
# # # # # #             B = batch_list[0].shape[1]
# # # # # #             if pred.shape[0] == B:
# # # # # #                 pred = pred.permute(1, 0, 2)
# # # # # #         return pred[..., :2]

# # # # # #     except Exception as e:
# # # # # #         print(f"    LSTM inference error: {e}")
# # # # # #         return None


# # # # # # @torch.no_grad()
# # # # # # def infer_sttrans(model, batch_list, args, device) -> Optional[torch.Tensor]:
# # # # # #     """Inference STTrans / STTransAR. Trả về [T, B, 2] norm."""
# # # # # #     try:
# # # # # #         model.eval()
# # # # # #         if hasattr(model, "sample"):
# # # # # #             pred, _, _ = model.sample(batch_list, num_ensemble=1)
# # # # # #         else:
# # # # # #             pred = model(batch_list)

# # # # # #         if pred.dim() == 3:
# # # # # #             B = batch_list[0].shape[1]
# # # # # #             if pred.shape[0] == B:
# # # # # #                 pred = pred.permute(1, 0, 2)
# # # # # #         return pred[..., :2]

# # # # # #     except Exception as e:
# # # # # #         print(f"    STTrans inference error: {e}")
# # # # # #         return None


# # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # #  CSV writers
# # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # _PER_STORM_FIELDS = [
# # # # # #     "model", "storm_id", "seq_idx", "lead_h",
# # # # # #     "lon_pred", "lat_pred", "lon_true", "lat_true",
# # # # # #     "dist_km", "ate_km", "cte_km",
# # # # # # ]

# # # # # # _SUMMARY_FIELDS = [
# # # # # #     "model", "storm_id", "n_seq",
# # # # # #     "ADE", "ATE", "CTE",
# # # # # #     "12h_dist", "24h_dist", "48h_dist", "72h_dist",
# # # # # #     "12h_ate",  "24h_ate",  "48h_ate",  "72h_ate",
# # # # # #     "12h_cte",  "24h_cte",  "48h_cte",  "72h_cte",
# # # # # #     "beat_12h", "beat_24h", "beat_48h", "beat_72h",
# # # # # #     "beat_ADE", "beat_ATE", "beat_CTE",
# # # # # # ]


# # # # # # def write_per_storm_csv(acc: MultiStormAccumulator, out_dir: str) -> str:
# # # # # #     out = os.path.join(out_dir, f"{acc.model_name}_per_storm.csv")
# # # # # #     with open(out, "w", newline="") as fh:
# # # # # #         w = csv.DictWriter(fh, fieldnames=_PER_STORM_FIELDS)
# # # # # #         w.writeheader()
# # # # # #         for sr in acc.storms.values():
# # # # # #             for row in sr.rows:
# # # # # #                 w.writerow({
# # # # # #                     "model"    : acc.model_name,
# # # # # #                     "storm_id" : row["storm_id"],
# # # # # #                     "seq_idx"  : row["seq_idx"],
# # # # # #                     "lead_h"   : row["lead_h"],
# # # # # #                     "lon_pred" : f"{row['lon_pred']:.4f}",
# # # # # #                     "lat_pred" : f"{row['lat_pred']:.4f}",
# # # # # #                     "lon_true" : f"{row['lon_true']:.4f}",
# # # # # #                     "lat_true" : f"{row['lat_true']:.4f}",
# # # # # #                     "dist_km"  : row["dist_km"],
# # # # # #                     "ate_km"   : row["ate_km"],
# # # # # #                     "cte_km"   : row["cte_km"],
# # # # # #                 })
# # # # # #     print(f"  💾 {out}")
# # # # # #     return out


# # # # # # def write_per_storm_detail_csvs(acc: MultiStormAccumulator,
# # # # # #                                  out_dir: str) -> List[str]:
# # # # # #     detail_dir = os.path.join(out_dir, "per_storm_detail")
# # # # # #     os.makedirs(detail_dir, exist_ok=True)
# # # # # #     paths = []
# # # # # #     for sid, sr in sorted(acc.storms.items()):
# # # # # #         safe_id = sid.replace(" ", "_").replace("/", "-")
# # # # # #         fpath   = os.path.join(detail_dir, f"{acc.model_name}_{safe_id}.csv")
# # # # # #         with open(fpath, "w", newline="") as fh:
# # # # # #             w = csv.DictWriter(fh, fieldnames=_PER_STORM_FIELDS)
# # # # # #             w.writeheader()
# # # # # #             for row in sr.rows:
# # # # # #                 w.writerow({
# # # # # #                     "model"    : acc.model_name,
# # # # # #                     "storm_id" : row["storm_id"],
# # # # # #                     "seq_idx"  : row["seq_idx"],
# # # # # #                     "lead_h"   : row["lead_h"],
# # # # # #                     "lon_pred" : f"{row['lon_pred']:.4f}",
# # # # # #                     "lat_pred" : f"{row['lat_pred']:.4f}",
# # # # # #                     "lon_true" : f"{row['lon_true']:.4f}",
# # # # # #                     "lat_true" : f"{row['lat_true']:.4f}",
# # # # # #                     "dist_km"  : row["dist_km"],
# # # # # #                     "ate_km"   : row["ate_km"],
# # # # # #                     "cte_km"   : row["cte_km"],
# # # # # #                 })
# # # # # #         paths.append(fpath)
# # # # # #     print(f"  💾 {len(paths)} per-storm CSVs → {detail_dir}/")
# # # # # #     return paths


# # # # # # def write_summary_csv(all_acc: Dict[str, MultiStormAccumulator],
# # # # # #                       out_dir: str) -> str:
# # # # # #     out = os.path.join(out_dir, "summary_all_models.csv")
# # # # # #     with open(out, "w", newline="") as fh:
# # # # # #         w = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
# # # # # #         w.writeheader()

# # # # # #         def _b(v, t): return "YES" if (np.isfinite(v) and v < t) else "NO"
# # # # # #         def _f(v):    return f"{v:.2f}" if np.isfinite(v) else "nan"

# # # # # #         for mname, acc in all_acc.items():
# # # # # #             for r in acc.all_summaries() + [acc.overall_summary()]:
# # # # # #                 w.writerow({
# # # # # #                     "model"    : mname, "storm_id": r["storm_id"],
# # # # # #                     "n_seq"    : r["n_seq"],
# # # # # #                     "ADE": _f(r["ADE"]), "ATE": _f(r["ATE"]), "CTE": _f(r["CTE"]),
# # # # # #                     "12h_dist" : _f(r["12h_dist"]), "24h_dist": _f(r["24h_dist"]),
# # # # # #                     "48h_dist" : _f(r["48h_dist"]), "72h_dist": _f(r["72h_dist"]),
# # # # # #                     "12h_ate"  : _f(r["12h_ate"]),  "24h_ate":  _f(r["24h_ate"]),
# # # # # #                     "48h_ate"  : _f(r["48h_ate"]),  "72h_ate":  _f(r["72h_ate"]),
# # # # # #                     "12h_cte"  : _f(r["12h_cte"]),  "24h_cte":  _f(r["24h_cte"]),
# # # # # #                     "48h_cte"  : _f(r["48h_cte"]),  "72h_cte":  _f(r["72h_cte"]),
# # # # # #                     "beat_12h" : _b(r["12h_dist"], TARGETS["12h"]),
# # # # # #                     "beat_24h" : _b(r["24h_dist"], TARGETS["24h"]),
# # # # # #                     "beat_48h" : _b(r["48h_dist"], TARGETS["48h"]),
# # # # # #                     "beat_72h" : _b(r["72h_dist"], TARGETS["72h"]),
# # # # # #                     "beat_ADE" : _b(r["ADE"],      TARGETS["ADE"]),
# # # # # #                     "beat_ATE" : _b(r["ATE"],      TARGETS["ATE"]),
# # # # # #                     "beat_CTE" : _b(r["CTE"],      TARGETS["CTE"]),
# # # # # #                 })
# # # # # #     print(f"  💾 Summary: {out}")
# # # # # #     return out


# # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # #  Pretty print
# # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # def print_model_results(acc: MultiStormAccumulator):
# # # # # #     cfg = MODEL_CONFIGS.get(acc.model_name, {})
# # # # # #     col = cfg.get("color", "⚪"); ref = cfg.get("ref", "")
# # # # # #     print(f"\n{'═'*115}")
# # # # # #     print(f"  {col}  {acc.model_name}  |  {ref}")
# # # # # #     print(f"{'═'*115}")
# # # # # #     print(f"  {'Storm':<16} {'N':>4}  "
# # # # # #           f"{'ADE':>8}  {'ATE':>8}  {'CTE':>8}  "
# # # # # #           f"║  {'12h':>7}  {'24h':>7}  {'48h':>7}  {'72h':>7}  "
# # # # # #           f"║  {'ATE@72':>8}  {'CTE@72':>8}")
# # # # # #     print("  " + "─" * 111)

# # # # # #     def v(x, t=None, w=8, dec=1):
# # # # # #         if not np.isfinite(x): return f"{'nan':>{w+2}}"
# # # # # #         s = f"{x:>{w}.{dec}f}"
# # # # # #         s += ("✅" if x < t else "❌") if t is not None else "  "
# # # # # #         return s

# # # # # #     for r in acc.all_summaries():
# # # # # #         print(
# # # # # #             f"  🌀 {r['storm_id']:<14} {r['n_seq']:>4}  "
# # # # # #             f"{v(r['ADE'],TARGETS['ADE'])}  {v(r['ATE'],TARGETS['ATE'])}  "
# # # # # #             f"{v(r['CTE'],TARGETS['CTE'])}  "
# # # # # #             f"║  {v(r['12h_dist'],TARGETS['12h'],7)}  "
# # # # # #             f"{v(r['24h_dist'],TARGETS['24h'],7)}  "
# # # # # #             f"{v(r['48h_dist'],TARGETS['48h'],7)}  "
# # # # # #             f"{v(r['72h_dist'],TARGETS['72h'],7)}  "
# # # # # #             f"║  {v(r['72h_ate'],None,8)}  {v(r['72h_cte'],None,8)}"
# # # # # #         )

# # # # # #     ov = acc.overall_summary()
# # # # # #     print("  " + "─" * 111)
# # # # # #     print(
# # # # # #         f"  📊 {'OVERALL':<14} {ov['n_seq']:>4}  "
# # # # # #         f"{v(ov['ADE'],TARGETS['ADE'])}  {v(ov['ATE'],TARGETS['ATE'])}  "
# # # # # #         f"{v(ov['CTE'],TARGETS['CTE'])}  "
# # # # # #         f"║  {v(ov['12h_dist'],TARGETS['12h'],7)}  "
# # # # # #         f"{v(ov['24h_dist'],TARGETS['24h'],7)}  "
# # # # # #         f"{v(ov['48h_dist'],TARGETS['48h'],7)}  "
# # # # # #         f"{v(ov['72h_dist'],TARGETS['72h'],7)}  "
# # # # # #         f"║  {v(ov['72h_ate'],None,8)}  {v(ov['72h_cte'],None,8)}"
# # # # # #     )
# # # # # #     print(f"{'═'*115}\n")


# # # # # # def print_comparison_table(all_acc: Dict[str, MultiStormAccumulator]):
# # # # # #     print(f"\n{'═'*100}")
# # # # # #     print(f"  📊  MODEL COMPARISON  —  Overall Test Set")
# # # # # #     print(f"{'═'*100}")
# # # # # #     print(f"  {'Model':<14}  {'ADE':>9}  {'ATE':>9}  {'CTE':>9}  "
# # # # # #           f"║  {'12h':>8}  {'24h':>8}  {'48h':>8}  {'72h':>8}")
# # # # # #     print("  " + "─" * 96)

# # # # # #     def vv(x, t=None, w=9, dec=1):
# # # # # #         if not np.isfinite(x): return f"{'nan':>{w+2}}"
# # # # # #         s = f"{x:>{w}.{dec}f}"
# # # # # #         s += ("✅" if x < t else "❌") if t is not None else "  "
# # # # # #         return s

# # # # # #     for mname, acc in all_acc.items():
# # # # # #         cfg = MODEL_CONFIGS.get(mname, {}); col = cfg.get("color", "⚪")
# # # # # #         ov  = acc.overall_summary()
# # # # # #         print(
# # # # # #             f"  {col} {mname:<12}  "
# # # # # #             f"{vv(ov['ADE'],TARGETS['ADE'])}  {vv(ov['ATE'],TARGETS['ATE'])}  "
# # # # # #             f"{vv(ov['CTE'],TARGETS['CTE'])}  "
# # # # # #             f"║  {vv(ov['12h_dist'],TARGETS['12h'],8)}  "
# # # # # #             f"{vv(ov['24h_dist'],TARGETS['24h'],8)}  "
# # # # # #             f"{vv(ov['48h_dist'],TARGETS['48h'],8)}  "
# # # # # #             f"{vv(ov['72h_dist'],TARGETS['72h'],8)}"
# # # # # #         )

# # # # # #     print("  " + "─" * 96)
# # # # # #     print(f"  Targets: ADE<{TARGETS['ADE']} | ATE<{TARGETS['ATE']} | "
# # # # # #           f"CTE<{TARGETS['CTE']} | "
# # # # # #           f"12h<{TARGETS['12h']} | 24h<{TARGETS['24h']} | "
# # # # # #           f"48h<{TARGETS['48h']} | 72h<{TARGETS['72h']} (km)")
# # # # # #     print(f"{'═'*100}\n")


# # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # #  Main evaluation loop
# # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # def run_evaluation(model, model_name: str, test_loader: DataLoader,
# # # # # #                    device, args, infer_fn) -> MultiStormAccumulator:
# # # # # #     acc       = MultiStormAccumulator(model_name)
# # # # # #     n_batches = len(test_loader)
# # # # # #     t0        = time.perf_counter()
# # # # # #     print(f"\n  Running {model_name} on {n_batches} batches...")

# # # # # #     for i, batch in enumerate(test_loader):
# # # # # #         bl = move_to_device(list(batch), device)
# # # # # #         B  = bl[0].shape[1]

# # # # # #         pred_norm = infer_fn(model, bl, args, device)
# # # # # #         if pred_norm is None:
# # # # # #             print(f"    batch {i}: inference returned None, skipping")
# # # # # #             continue

# # # # # #         gt_norm   = bl[1]
# # # # # #         storm_ids = extract_storm_ids(batch, B)
# # # # # #         T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # #         acc.add_batch(pred_norm[:T], gt_norm[:T], storm_ids)

# # # # # #         if (i + 1) % max(1, n_batches // 10) == 0 or i == n_batches - 1:
# # # # # #             ov      = acc.overall_summary()
# # # # # #             ade_str = f"{ov['ADE']:.1f}km" if np.isfinite(ov["ADE"]) else "nan"
# # # # # #             print(f"    [{i+1:>4}/{n_batches}]  ADE={ade_str}  "
# # # # # #                   f"storms={len(acc.storms)}  "
# # # # # #                   f"elapsed={time.perf_counter()-t0:.1f}s")

# # # # # #     ov = acc.overall_summary()
# # # # # #     print(f"  ✅ {model_name} done  "
# # # # # #           f"ADE={ov['ADE']:.1f}  ATE={ov['ATE']:.1f}  CTE={ov['CTE']:.1f} km")
# # # # # #     return acc


# # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # #  Args
# # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # def get_args():
# # # # # #     p = argparse.ArgumentParser(
# # # # # #         description="Evaluate LSTM/GRU/RNN / ST-Trans / FlowMatching on test set",
# # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
# # # # # #     )
# # # # # #     p.add_argument("--dataset_root",  default="TCND_vn")
# # # # # #     p.add_argument("--obs_len",       default=8,  type=int)
# # # # # #     p.add_argument("--pred_len",      default=12, type=int)
# # # # # #     p.add_argument("--batch_size",    default=16, type=int)
# # # # # #     p.add_argument("--num_workers",   default=2,  type=int)
# # # # # #     p.add_argument("--skip",          default=1,  type=int)
# # # # # #     p.add_argument("--min_ped",       default=1,  type=int)
# # # # # #     p.add_argument("--threshold",     default=0.002, type=float)
# # # # # #     p.add_argument("--other_modal",   default="gph")
# # # # # #     p.add_argument("--test_year",     default=None, type=int)
# # # # # #     p.add_argument("--delim",         default=" ")

# # # # # #     p.add_argument("--fm_ckpt",       default=None)
# # # # # #     p.add_argument("--lstm_ckpt",     default=None)
# # # # # #     p.add_argument("--sttrans_ckpt",  default=None)

# # # # # #     p.add_argument("--lstm_type",     default="lstm",
# # # # # #                    choices=["lstm", "gru", "rnn"])
# # # # # #     p.add_argument("--sttrans_type",  default="sttrans",
# # # # # #                    choices=["sttrans", "sttrans_ar"])

# # # # # #     p.add_argument("--fm_ensemble",   default=30, type=int)
# # # # # #     p.add_argument("--fm_ode_steps",  default=20, type=int)

# # # # # #     p.add_argument("--output_dir",    default="results/test_eval")
# # # # # #     p.add_argument("--gpu_num",       default="0")
# # # # # #     p.add_argument("--no_detail_csv", action="store_true")
# # # # # #     p.add_argument("--eval_models",   default="all",
# # # # # #                    help="'all' hoặc csv: 'fm,lstm,sttrans'")
# # # # # #     return p.parse_args()


# # # # # # def _version_from_path(ckpt_path: str) -> str:
# # # # # #     parts = os.path.normpath(ckpt_path).split(os.sep)
# # # # # #     for p in reversed(parts[:-1]):
# # # # # #         if p.startswith("v") and p[1:].isdigit():
# # # # # #             return p
# # # # # #     return "ours"


# # # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # # #  MAIN
# # # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # # def main(args):
# # # # # #     if torch.cuda.is_available():
# # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # #     ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# # # # # #     print("=" * 80)
# # # # # #     print(f"  TC Model Evaluator")
# # # # # #     print(f"  Started : {ts_start}")
# # # # # #     print(f"  Device  : {device}")
# # # # # #     print(f"  Output  : {args.output_dir}")
# # # # # #     print("=" * 80)

# # # # # #     eval_set = ({"fm", "lstm", "sttrans"} if args.eval_models == "all"
# # # # # #                 else {m.strip().lower() for m in args.eval_models.split(",")})

# # # # # #     # ── Load test data ─────────────────────────────────────────────────────
# # # # # #     print("\n  Loading test dataset...")
# # # # # #     try:
# # # # # #         from Model.data.loader_training import data_loader
# # # # # #         test_dataset, test_loader = data_loader(
# # # # # #             args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # # # #         print(f"  ✅ Test set: {len(test_dataset)} sequences, "
# # # # # #               f"{len(test_loader)} batches")
# # # # # #     except Exception as e:
# # # # # #         print(f"  ❌ Failed to load test data: {e}")
# # # # # #         import traceback; traceback.print_exc()
# # # # # #         return

# # # # # #     # ── Eval ──────────────────────────────────────────────────────────────
# # # # # #     all_acc: Dict[str, MultiStormAccumulator] = {}

# # # # # #     if "fm" in eval_set and args.fm_ckpt:
# # # # # #         print(f"\n{'─'*50}")
# # # # # #         print("  Loading FlowMatching...")
# # # # # #         model = load_fm_model(args.fm_ckpt, device, args)
# # # # # #         if model is not None:
# # # # # #             tag = "FM_" + _version_from_path(args.fm_ckpt)
# # # # # #             all_acc[tag] = run_evaluation(model, tag, test_loader,
# # # # # #                                           device, args, infer_fm)
# # # # # #             del model
# # # # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # # # #     if "lstm" in eval_set and args.lstm_ckpt:
# # # # # #         print(f"\n{'─'*50}")
# # # # # #         print(f"  Loading {args.lstm_type.upper()}...")
# # # # # #         model = load_lstm_model(args.lstm_ckpt, device, args)
# # # # # #         if model is not None:
# # # # # #             tag = args.lstm_type.upper()
# # # # # #             all_acc[tag] = run_evaluation(model, tag, test_loader,
# # # # # #                                           device, args, infer_lstm)
# # # # # #             del model
# # # # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # # # #     if "sttrans" in eval_set and args.sttrans_ckpt:
# # # # # #         print(f"\n{'─'*50}")
# # # # # #         print(f"  Loading {args.sttrans_type.upper()}...")
# # # # # #         model = load_sttrans_model(args.sttrans_ckpt, device, args)
# # # # # #         if model is not None:
# # # # # #             tag = "STTransAR" if args.sttrans_type == "sttrans_ar" else "STTrans"
# # # # # #             all_acc[tag] = run_evaluation(model, tag, test_loader,
# # # # # #                                           device, args, infer_sttrans)
# # # # # #             del model
# # # # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # # # #     if not all_acc:
# # # # # #         print("\n  ❌ No models were evaluated.")
# # # # # #         print("     Specify: --fm_ckpt / --lstm_ckpt / --sttrans_ckpt")
# # # # # #         print("     Or:      --eval_models fm")
# # # # # #         return

# # # # # #     # ── Print ──────────────────────────────────────────────────────────────
# # # # # #     for acc in all_acc.values():
# # # # # #         print_model_results(acc)
# # # # # #     if len(all_acc) > 1:
# # # # # #         print_comparison_table(all_acc)

# # # # # #     # ── Write CSVs ─────────────────────────────────────────────────────────
# # # # # #     print("\n  Writing CSVs...")
# # # # # #     csv_paths = []
# # # # # #     for acc in all_acc.values():
# # # # # #         csv_paths.append(write_per_storm_csv(acc, args.output_dir))
# # # # # #         if not args.no_detail_csv:
# # # # # #             write_per_storm_detail_csvs(acc, args.output_dir)
# # # # # #     csv_paths.append(write_summary_csv(all_acc, args.output_dir))

# # # # # #     # ── Beat summary ───────────────────────────────────────────────────────
# # # # # #     print(f"\n{'═'*80}")
# # # # # #     print(f"  FINAL BEAT SUMMARY")
# # # # # #     print(f"{'─'*80}")
# # # # # #     metric_map = [
# # # # # #         ("ADE","ADE"),("ATE","ATE"),("CTE","CTE"),
# # # # # #         ("12h_dist","12h"),("24h_dist","24h"),
# # # # # #         ("48h_dist","48h"),("72h_dist","72h"),
# # # # # #     ]
# # # # # #     for mname, acc in all_acc.items():
# # # # # #         ov  = acc.overall_summary()
# # # # # #         cfg = MODEL_CONFIGS.get(mname, {}); col = cfg.get("color", "⚪")
# # # # # #         beats = [
# # # # # #             f"{label}={ov.get(k, float('nan')):.1f}✅"
# # # # # #             for k, label in metric_map
# # # # # #             if np.isfinite(ov.get(k, float("nan")))
# # # # # #             and ov[k] < TARGETS.get(label, float("inf"))
# # # # # #         ]
# # # # # #         print(f"  {col} {mname:<12}: "
# # # # # #               + (" | ".join(beats) if beats else "no targets beaten"))

# # # # # #     elapsed = (datetime.now()
# # # # # #                - datetime.strptime(ts_start, "%Y-%m-%d %H:%M:%S")).seconds
# # # # # #     print(f"\n  Total: {elapsed}s  |  Output: {args.output_dir}/")
# # # # # #     for p in csv_paths:
# # # # # #         print(f"    → {os.path.basename(p)}")
# # # # # #     print(f"{'═'*80}\n")


# # # # # # if __name__ == "__main__":
# # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # #     if torch.cuda.is_available():
# # # # # #         torch.cuda.manual_seed_all(42)

# # # # # #     args = get_args()

# # # # # #     _kaggle_base = "/kaggle/working/TC_FM"
# # # # # #     if args.dataset_root == "TCND_vn" and os.path.isdir(_kaggle_base):
# # # # # #         args.dataset_root = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"

# # # # # #     _runs = (os.path.join(_kaggle_base, "runs")
# # # # # #              if os.path.isdir(_kaggle_base) else "runs")

# # # # # #     if args.fm_ckpt is None:
# # # # # #         for c in [os.path.join(_runs, f"v{v}", "best_model.pth")
# # # # # #                   for v in ["61","60","59","58"]] + ["best_model.pth"]:
# # # # # #             if os.path.exists(c):
# # # # # #                 args.fm_ckpt = c
# # # # # #                 print(f"  Auto FM ckpt: {c}")
# # # # # #                 break

# # # # # #     if args.lstm_ckpt is None:
# # # # # #         for c in [os.path.join(_runs, d, "best_model.pth")
# # # # # #                   for d in ["lstm","LSTM","gru","GRU","paper_baseline"]]:
# # # # # #             if os.path.exists(c):
# # # # # #                 args.lstm_ckpt = c
# # # # # #                 print(f"  Auto LSTM ckpt: {c}")
# # # # # #                 break

# # # # # #     if args.sttrans_ckpt is None:
# # # # # #         for c in [os.path.join(_runs, d, "best_model.pth")
# # # # # #                   for d in ["sttrans","STTrans","st_trans"]]:
# # # # # #             if os.path.exists(c):
# # # # # #                 args.sttrans_ckpt = c
# # # # # #                 print(f"  Auto STTrans ckpt: {c}")
# # # # # #                 break

# # # # # #     main(args)


# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # USAGE:
# # # # # # #
# # # # # # # python scripts/evaluate_test_storms.py \
# # # # # # #     --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# # # # # # #     --fm_ckpt      /kaggle/working/TC_FM/runs/v61/best_model.pth \
# # # # # # #     --lstm_ckpt    /kaggle/working/TC_FM/runs/lstm/best_model.pth \
# # # # # # #     --sttrans_ckpt /kaggle/working/TC_FM/runs/sttrans/best_model.pth \
# # # # # # #     --lstm_type    gru \
# # # # # # #     --sttrans_type sttrans \
# # # # # # #     --fm_ensemble  30 \
# # # # # # #     --fm_ode_steps 20 \
# # # # # # #     --output_dir   /kaggle/working/TC_FM/results/test_eval
# # # # # # #
# # # # # # # Chỉ FM:
# # # # # # # python scripts/evaluate_test_storms.py \
# # # # # # #     --eval_models fm \
# # # # # # #     --fm_ckpt runs/v61/best_model.pth
# # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # """
# # # # # scripts/evaluate_test_storms.py
# # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # Script đánh giá tất cả models (LSTM/GRU/RNN, ST-Trans, FlowMatching) trên test set.

# # # # # FIX v2:
# # # # #   ✅ Robust FM import — auto-detect class name trong module
# # # # #   ✅ Auto-detect checkpoint type: TCFlowMatching vs FMResidualCorrector
# # # # #   ✅ det_head support — set _det_epoch=999 khi checkpoint có det_head weights
# # # # #   ✅ Architecture detection mở rộng cho v67 (det_head, obs_enc, horizon_queries)
# # # # #   ✅ FMResidualCorrector support trong load và infer
# # # # #   ✅ Graceful fallback khi import fail

# # # # # Cách dùng:
# # # # #     python scripts/evaluate_test_storms.py \
# # # # #         --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# # # # #         --fm_ckpt      runs/v67/best_ema.pth \
# # # # #         --lstm_ckpt    runs/lstm/best_model.pth \
# # # # #         --sttrans_ckpt runs/sttrans/best_model.pth \
# # # # #         --output_dir   results/test_eval \
# # # # #         --gpu_num 0
# # # # # """
# # # # # from __future__ import annotations

# # # # # import sys
# # # # # import os
# # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # import argparse
# # # # # import csv
# # # # # import math
# # # # # import time
# # # # # import importlib
# # # # # import inspect
# # # # # from collections import defaultdict
# # # # # from datetime import datetime
# # # # # from typing import Dict, List, Optional, Tuple

# # # # # import numpy as np
# # # # # import torch
# # # # # import torch.nn as nn
# # # # # import torch.nn.functional as F
# # # # # from torch.utils.data import DataLoader

# # # # # R_EARTH  = 6371.0
# # # # # DEG2KM   = 111.0
# # # # # DT_HOURS = 6.0

# # # # # LEAD_HOURS = [12, 24, 36, 48, 60, 72]
# # # # # LEAD_STEPS = {h: h // 6 - 1 for h in LEAD_HOURS}

# # # # # TARGETS = {
# # # # #     "12h": 50.0, "24h": 100.0, "48h": 200.0, "72h": 297.0,
# # # # #     "ADE": 136.41, "ATE": 79.94, "CTE": 93.58,
# # # # # }

# # # # # MODEL_CONFIGS = {
# # # # #     "LSTM"     : {"color": "🔵", "ref": "Rahman 2025"},
# # # # #     "GRU"      : {"color": "🔵", "ref": "Rahman 2025"},
# # # # #     "RNN"      : {"color": "🔵", "ref": "Rahman 2025"},
# # # # #     "STTrans"  : {"color": "🟡", "ref": "Faiaz 2026"},
# # # # #     "STTransAR": {"color": "🟡", "ref": "Faiaz 2026 AR"},
# # # # #     "FM"       : {"color": "🔴", "ref": "Ours FM"},
# # # # #     "FM_v59"   : {"color": "🔴", "ref": "Ours v59"},
# # # # #     "FM_v67"   : {"color": "🔴", "ref": "Ours v67"},
# # # # #     "FM_ours"  : {"color": "🔴", "ref": "Ours"},
# # # # #     "FMResidual": {"color": "🟠", "ref": "Ours Residual"},
# # # # # }


# # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # #  Coordinate utilities
# # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # def norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
# # # # #     out = arr.clone()
# # # # #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# # # # #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# # # # #     return out


# # # # # def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # #     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
# # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # #     a    = (torch.sin(dlat / 2).pow(2) +
# # # # #             torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
# # # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# # # # # def compute_ate_cte(pred_deg, gt_deg):
# # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # #     if T < 2:
# # # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # #         return z, z
# # # # #     lon1 = torch.deg2rad(gt_deg[:T-1,:,0]); lat1 = torch.deg2rad(gt_deg[:T-1,:,1])
# # # # #     lon2 = torch.deg2rad(gt_deg[1:T, :,0]); lat2 = torch.deg2rad(gt_deg[1:T, :,1])
# # # # #     dlon_a = lon2 - lon1
# # # # #     y_a = torch.sin(dlon_a)*torch.cos(lat2)
# # # # #     x_a = torch.cos(lat1)*torch.sin(lat2) - torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon_a)
# # # # #     bear_along = torch.atan2(y_a, x_a)
# # # # #     lon3 = torch.deg2rad(pred_deg[1:T,:,0]); lat3 = torch.deg2rad(pred_deg[1:T,:,1])
# # # # #     dlon_e = lon3 - lon2
# # # # #     y_e = torch.sin(dlon_e)*torch.cos(lat3)
# # # # #     x_e = torch.cos(lat2)*torch.sin(lat3) - torch.sin(lat2)*torch.cos(lat3)*torch.cos(dlon_e)
# # # # #     bear_err  = torch.atan2(y_e, x_e)
# # # # #     total     = haversine_km(pred_deg[1:T], gt_deg[1:T])
# # # # #     angle     = bear_err - bear_along
# # # # #     return total*torch.cos(angle), total*torch.sin(angle)


# # # # # def move_to_device(batch, device):
# # # # #     out = list(batch)
# # # # #     for i, x in enumerate(out):
# # # # #         if torch.is_tensor(x):
# # # # #             out[i] = x.to(device)
# # # # #         elif isinstance(x, dict):
# # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # #                       for k, v in x.items()}
# # # # #     return out


# # # # # def get_raw_model(model):
# # # # #     return model._orig_mod if hasattr(model, "_orig_mod") else model


# # # # # def get_ema_obj(model):
# # # # #     raw = get_raw_model(model)
# # # # #     if hasattr(raw, "_ema") and raw._ema is not None:
# # # # #         return raw._ema
# # # # #     return None


# # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # #  Per-storm accumulator (unchanged)
# # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # class StormResult:
# # # # #     def __init__(self, storm_id):
# # # # #         self.storm_id = storm_id
# # # # #         self.rows = []
# # # # #         self.dist_by_lead = defaultdict(list)
# # # # #         self.ate_by_lead  = defaultdict(list)
# # # # #         self.cte_by_lead  = defaultdict(list)
# # # # #         self.all_dist = []; self.all_ate = []; self.all_cte = []
# # # # #         self.n_seq = 0

# # # # #     def add_batch(self, pred_deg, gt_deg, seq_indices=None):
# # # # #         T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # #         B = pred_deg.shape[1]
# # # # #         if T < 2: return
# # # # #         dist     = haversine_km(pred_deg[:T], gt_deg[:T])
# # # # #         ate, cte = compute_ate_cte(pred_deg, gt_deg)
# # # # #         self.n_seq += B
# # # # #         for b in range(B):
# # # # #             for t in range(T):
# # # # #                 lead_h  = (t+1)*6
# # # # #                 d_val   = float(dist[t,b])
# # # # #                 ate_val = float(ate[t,b].abs()) if t < ate.shape[0] else float("nan")
# # # # #                 cte_val = float(cte[t,b].abs()) if t < cte.shape[0] else float("nan")
# # # # #                 self.rows.append({
# # # # #                     "storm_id": self.storm_id,
# # # # #                     "seq_idx" : int(seq_indices[b]) if seq_indices is not None else b,
# # # # #                     "lead_h"  : lead_h,
# # # # #                     "lon_pred": float(pred_deg[t,b,0]),
# # # # #                     "lat_pred": float(pred_deg[t,b,1]),
# # # # #                     "lon_true": float(gt_deg[t,b,0]),
# # # # #                     "lat_true": float(gt_deg[t,b,1]),
# # # # #                     "dist_km" : round(d_val, 2),
# # # # #                     "ate_km"  : round(ate_val,2) if np.isfinite(ate_val) else "nan",
# # # # #                     "cte_km"  : round(cte_val,2) if np.isfinite(cte_val) else "nan",
# # # # #                 })
# # # # #                 if lead_h in LEAD_STEPS:
# # # # #                     self.dist_by_lead[lead_h].append(d_val)
# # # # #                     if np.isfinite(ate_val): self.ate_by_lead[lead_h].append(ate_val)
# # # # #                     if np.isfinite(cte_val): self.cte_by_lead[lead_h].append(cte_val)
# # # # #                 self.all_dist.append(d_val)
# # # # #                 if t < ate.shape[0]:
# # # # #                     if np.isfinite(ate_val): self.all_ate.append(ate_val)
# # # # #                     if np.isfinite(cte_val): self.all_cte.append(cte_val)

# # # # #     def summary(self):
# # # # #         def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # # # #         r = {"storm_id": self.storm_id, "n_seq": self.n_seq,
# # # # #              "ADE": _m(self.all_dist), "ATE": _m(self.all_ate), "CTE": _m(self.all_cte)}
# # # # #         for h in LEAD_HOURS:
# # # # #             r[f"{h}h_dist"] = _m(self.dist_by_lead.get(h,[]))
# # # # #             r[f"{h}h_ate"]  = _m(self.ate_by_lead.get(h,[]))
# # # # #             r[f"{h}h_cte"]  = _m(self.cte_by_lead.get(h,[]))
# # # # #         return r


# # # # # class MultiStormAccumulator:
# # # # #     def __init__(self, model_name):
# # # # #         self.model_name = model_name
# # # # #         self.storms: Dict[str, StormResult] = {}

# # # # #     def get_storm(self, storm_id):
# # # # #         if storm_id not in self.storms:
# # # # #             self.storms[storm_id] = StormResult(storm_id)
# # # # #         return self.storms[storm_id]

# # # # #     def add_batch(self, pred_norm, gt_norm, storm_ids):
# # # # #         T  = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # #         B  = pred_norm.shape[1]
# # # # #         pd = norm_to_deg(pred_norm[:T])
# # # # #         gd = norm_to_deg(gt_norm[:T])
# # # # #         sid_map = defaultdict(list)
# # # # #         for b, sid in enumerate(storm_ids[:B]):
# # # # #             sid_map[sid].append(b)
# # # # #         for sid, bidx in sid_map.items():
# # # # #             bidx_t = torch.tensor(bidx, dtype=torch.long)
# # # # #             self.get_storm(sid).add_batch(pd[:,bidx_t,:], gd[:,bidx_t,:],
# # # # #                                            seq_indices=bidx_t)

# # # # #     def all_summaries(self):
# # # # #         return [self.storms[sid].summary() for sid in sorted(self.storms)]

# # # # #     def overall_summary(self):
# # # # #         all_d, all_a, all_c = [], [], []
# # # # #         lead_d = defaultdict(list); lead_a = defaultdict(list); lead_c = defaultdict(list)
# # # # #         for sr in self.storms.values():
# # # # #             all_d.extend(sr.all_dist); all_a.extend(sr.all_ate); all_c.extend(sr.all_cte)
# # # # #             for h in LEAD_HOURS:
# # # # #                 lead_d[h].extend(sr.dist_by_lead.get(h,[]))
# # # # #                 lead_a[h].extend(sr.ate_by_lead.get(h,[]))
# # # # #                 lead_c[h].extend(sr.cte_by_lead.get(h,[]))
# # # # #         def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # # # #         r = {"storm_id": "📊 OVERALL",
# # # # #              "n_seq": sum(s.n_seq for s in self.storms.values()),
# # # # #              "ADE": _m(all_d), "ATE": _m(all_a), "CTE": _m(all_c)}
# # # # #         for h in LEAD_HOURS:
# # # # #             r[f"{h}h_dist"] = _m(lead_d[h])
# # # # #             r[f"{h}h_ate"]  = _m(lead_a[h])
# # # # #             r[f"{h}h_cte"]  = _m(lead_c[h])
# # # # #         return r


# # # # # def extract_storm_ids(batch, B):
# # # # #     for idx in [2, 12, 14, 15]:
# # # # #         if idx < len(batch):
# # # # #             item = batch[idx]
# # # # #             if isinstance(item, (list, tuple)) and len(item) >= B:
# # # # #                 if isinstance(item[0], str):
# # # # #                     return list(item[:B])
# # # # #             if (torch.is_tensor(item) and item.dtype == torch.long
# # # # #                     and item.numel() >= B):
# # # # #                 return [str(x.item()) for x in item[:B]]
# # # # #     return [f"STORM_{i:03d}" for i in range(B)]


# # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # #  [FIX] Robust FM import
# # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # def _import_fm_class():
# # # # #     """
# # # # #     Robust import của TCFlowMatching.
    
# # # # #     Thử theo thứ tự:
# # # # #     1. Import trực tiếp TCFlowMatching (v59+)
# # # # #     2. Import TCDiffusion (alias trong v59)
# # # # #     3. Scan module để tìm class có sample() + get_loss_breakdown()
# # # # #     4. Return None nếu không tìm được
# # # # #     """
# # # # #     try:
# # # # #         from Model.flow_matching_model import TCFlowMatching
# # # # #         print("  [import] TCFlowMatching ✓")
# # # # #         return TCFlowMatching
# # # # #     except ImportError:
# # # # #         pass

# # # # #     # Thử alias cũ
# # # # #     try:
# # # # #         from Model.flow_matching_model import TCDiffusion
# # # # #         print("  [import] TCDiffusion (alias) ✓")
# # # # #         return TCDiffusion
# # # # #     except ImportError:
# # # # #         pass

# # # # #     # Scan module tìm class phù hợp
# # # # #     try:
# # # # #         mod = importlib.import_module("Model.flow_matching_model")
# # # # #         candidates = []
# # # # #         for name, cls in inspect.getmembers(mod, inspect.isclass):
# # # # #             if (hasattr(cls, "sample") and hasattr(cls, "get_loss_breakdown")
# # # # #                     and issubclass(cls, nn.Module)
# # # # #                     and cls.__module__ == mod.__name__):
# # # # #                 candidates.append((name, cls))
# # # # #         if candidates:
# # # # #             # Ưu tiên class có "Flow" hoặc "TC" trong tên
# # # # #             for name, cls in candidates:
# # # # #                 if "Flow" in name or "TC" in name:
# # # # #                     print(f"  [import] Found {name} by scan ✓")
# # # # #                     return cls
# # # # #             name, cls = candidates[0]
# # # # #             print(f"  [import] Fallback to {name} by scan ✓")
# # # # #             return cls
# # # # #     except Exception as e:
# # # # #         print(f"  [import] Module scan failed: {e}")

# # # # #     print("  [import] ❌ Cannot find FM class in Model.flow_matching_model")
# # # # #     return None


# # # # # def _import_residual_class():
# # # # #     """Import FMResidualCorrector nếu có."""
# # # # #     try:
# # # # #         from Model.fm_residual_corrector import FMResidualCorrector
# # # # #         return FMResidualCorrector
# # # # #     except ImportError:
# # # # #         pass
# # # # #     try:
# # # # #         mod = importlib.import_module("Model.flow_matching_model")
# # # # #         if hasattr(mod, "FMResidualCorrector"):
# # # # #             return mod.FMResidualCorrector
# # # # #     except Exception:
# # # # #         pass
# # # # #     return None


# # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # #  [FIX] Detect checkpoint type
# # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # def _detect_ckpt_type(state: dict) -> str:
# # # # #     """
# # # # #     Detect loại model từ state dict keys.

# # # # #     FMResidualCorrector có ĐỒNG THỜI:
# # # # #       ctx_enc.* (ResidualContextEncoder) VÀ corr_net.* (CorrectionNet)
# # # # #     TCFlowMatching có: net.* (VelocityField)

# # # # #     KHÔNG dùng 'sttrans.*' để detect vì TCFlowMatching cũ cũng có thể
# # # # #     chứa STTrans reference bên trong (bị detect nhầm).

# # # # #     Returns: 'residual' | 'tcfm'
# # # # #     """
# # # # #     keys = set(state.keys())
# # # # #     has_ctx_enc  = any(k.startswith("ctx_enc.")  for k in keys)
# # # # #     has_corr_net = any(k.startswith("corr_net.") for k in keys)
# # # # #     has_gate     = any(k.startswith("gate.")     for k in keys)

# # # # #     # FMResidualCorrector: phải có ít nhất 2 trong 3 unique keys
# # # # #     if sum([has_ctx_enc, has_corr_net, has_gate]) >= 2:
# # # # #         return "residual"

# # # # #     # Mặc định: TCFlowMatching (safe default)
# # # # #     return "tcfm"


# # # # # def _detect_has_det_head(state: dict) -> bool:
# # # # #     """Checkpoint có det_head weights không (v67+)?"""
# # # # #     return any(k.startswith("net.det_head.") or
# # # # #                k.startswith("net.det_head") for k in state.keys())


# # # # # def _load_state_dict(ckpt) -> dict:
# # # # #     """Extract state_dict từ checkpoint (nhiều format)."""
# # # # #     if not isinstance(ckpt, dict):
# # # # #         return ckpt
# # # # #     for key in ["model_state_dict", "model_state", "state_dict", "model"]:
# # # # #         if key in ckpt and isinstance(ckpt[key], dict):
# # # # #             sd = ckpt[key]
# # # # #             if any(torch.is_tensor(v) for v in sd.values()):
# # # # #                 return sd
# # # # #     # Có thể là raw state dict
# # # # #     if any(torch.is_tensor(v) for v in ckpt.values()
# # # # #            if isinstance(v, torch.Tensor)):
# # # # #         return ckpt
# # # # #     return ckpt


# # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # #  [FIX] load_fm_model — robust, hỗ trợ v47→v67
# # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # def load_fm_model(ckpt_path: str, device, args):
# # # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # # #         print(f"  ⚠  FM checkpoint not found: {ckpt_path}")
# # # # #         return None

# # # # #     try:
# # # # #         print(f"  Loading: {ckpt_path}")
# # # # #         ckpt  = torch.load(ckpt_path, map_location=device)
# # # # #         state = _load_state_dict(ckpt)
# # # # #         ep    = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"

# # # # #         # Debug: print key prefixes to help diagnose
# # # # #         from collections import Counter
# # # # #         prefix_counts = Counter(k.split(".")[0] for k in state.keys())
# # # # #         print(f"  Key prefixes: " +
# # # # #               ", ".join(f"{p}({n})" for p,n in
# # # # #                         sorted(prefix_counts.items(), key=lambda x:-x[1])[:8]))

# # # # #         # [FIX] --fm_type flag overrides auto-detection
# # # # #         forced = getattr(args, "fm_type", "auto")
# # # # #         if forced == "tcfm":
# # # # #             ckpt_type = "tcfm"
# # # # #             print(f"  Checkpoint type: tcfm (forced by --fm_type)")
# # # # #         elif forced == "residual":
# # # # #             ckpt_type = "residual"
# # # # #             print(f"  Checkpoint type: residual (forced by --fm_type)")
# # # # #         else:
# # # # #             ckpt_type = _detect_ckpt_type(state)
# # # # #             print(f"  Checkpoint type: {ckpt_type}  (epoch={ep})")

# # # # #         # ── FMResidualCorrector ──────────────────────────────────────────
# # # # #         if ckpt_type == "residual":
# # # # #             return _load_residual_model(ckpt, state, device, args, ep)

# # # # #         # ── TCFlowMatching ───────────────────────────────────────────────
# # # # #         TCFlowMatching = _import_fm_class()
# # # # #         if TCFlowMatching is None:
# # # # #             print("  ❌ Cannot import FM class — check flow_matching_model.py")
# # # # #             return None

# # # # #         # Detect architecture từ checkpoint
# # # # #         ffn_key = "net.transformer.layers.0.linear1.weight"
# # # # #         ffn_dim = int(state[ffn_key].shape[0]) if ffn_key in state else 1024

# # # # #         num_layers = sum(
# # # # #             1 for i in range(10)
# # # # #             if f"net.transformer.layers.{i}.linear1.weight" in state
# # # # #         ) or 2

# # # # #         has_det_head    = _detect_has_det_head(state)
# # # # #         has_reg_head    = any(k.startswith("net.reg_head.") for k in state)
# # # # #         has_vel_expand  = "net.vel_obs_expand.weight" in state
# # # # #         has_speed_head  = any(k.startswith("net.speed_head.") for k in state)

# # # # #         print(f"  Architecture: FFN={ffn_dim}, layers={num_layers}, "
# # # # #               f"det_head={has_det_head}, reg_head={has_reg_head}, "
# # # # #               f"speed_head={has_speed_head}")

# # # # #         # Instantiate
# # # # #         model = TCFlowMatching(
# # # # #             pred_len=args.pred_len, obs_len=args.obs_len,
# # # # #             sigma_min=0.02, use_ema=True, ema_decay=0.995,
# # # # #         ).to(device)

# # # # #         if hasattr(model, "init_ema"):
# # # # #             model.init_ema()

# # # # #         raw = get_raw_model(model)

# # # # #         # Patch transformer nếu cần
# # # # #         try:
# # # # #             cur_ffn    = raw.net.transformer.layers[0].linear1.weight.shape[0]
# # # # #             cur_layers = len(raw.net.transformer.layers)
# # # # #         except Exception:
# # # # #             cur_ffn, cur_layers = 1024, 2

# # # # #         if cur_ffn != ffn_dim or cur_layers != num_layers:
# # # # #             try:
# # # # #                 raw.net.transformer = nn.TransformerDecoder(
# # # # #                     nn.TransformerDecoderLayer(
# # # # #                         d_model=256, nhead=8, dim_feedforward=ffn_dim,
# # # # #                         dropout=0.10, activation="gelu", batch_first=True),
# # # # #                     num_layers=num_layers).to(device)
# # # # #                 print(f"  Patched transformer → {num_layers}L / {ffn_dim}FFN")
# # # # #             except Exception as e:
# # # # #                 print(f"  ⚠ Transformer patch failed: {e}")

# # # # #         # Patch vel_obs_expand nếu checkpoint có nhưng model không có
# # # # #         if has_vel_expand and not hasattr(raw.net, "vel_obs_expand"):
# # # # #             raw.net.vel_obs_expand = nn.Linear(256, 1024).to(device)
# # # # #             raw.net.vel_obs_ln     = nn.LayerNorm(256).to(device)
# # # # #             print("  Patched vel_obs_expand")

# # # # #         # Load weights
# # # # #         result     = model.load_state_dict(state, strict=False)
# # # # #         missing    = [k for k in result.missing_keys
# # # # #                       if not k.startswith("net.det_head")  # expected
# # # # #                       and "_ema" not in k]
# # # # #         unexpected = result.unexpected_keys

# # # # #         if missing:
# # # # #             print(f"  Missing keys (non-det_head): {len(missing)}")
# # # # #             for k in missing[:3]: print(f"    - {k}")
# # # # #         if unexpected:
# # # # #             print(f"  Unexpected keys: {len(unexpected)}")
# # # # #             for k in list(unexpected)[:3]: print(f"    + {k}")

# # # # #         det_missing = [k for k in result.missing_keys
# # # # #                        if k.startswith("net.det_head")]
# # # # #         if det_missing:
# # # # #             print(f"  det_head: {len(det_missing)} keys missing "
# # # # #                   f"(old checkpoint, det blend disabled)")

# # # # #         # Load EMA
# # # # #         ema_loaded = 0
# # # # #         if isinstance(ckpt, dict):
# # # # #             for ema_key in ["ema_shadow", "ema_state_dict", "ema"]:
# # # # #                 if ema_key in ckpt and ckpt[ema_key]:
# # # # #                     ema = get_ema_obj(model)
# # # # #                     if ema is not None and hasattr(ema, "shadow"):
# # # # #                         for k, v in ckpt[ema_key].items():
# # # # #                             if k in ema.shadow:
# # # # #                                 ema.shadow[k].copy_(v.to(device))
# # # # #                                 ema_loaded += 1
# # # # #                         if ema_loaded:
# # # # #                             print(f"  EMA: {ema_loaded} keys loaded")
# # # # #                     break

# # # # #         if not ema_loaded:
# # # # #             print("  ⚠ No EMA weights — using raw model weights")

# # # # #         # [KEY FIX] Set _det_epoch để control DET_BLEND
# # # # #         # Checkpoint có det_head weights → full blend (0.70)
# # # # #         # Checkpoint không có det_head → blend=0 (pure FM)
# # # # #         raw_model = get_raw_model(model)
# # # # #         if has_det_head:
# # # # #             raw_model._det_epoch = 999   # → DET_BLEND = 0.70
# # # # #             print("  det_head: weights loaded → DET_BLEND=0.70 ✓")
# # # # #         else:
# # # # #             raw_model._det_epoch = 0     # → DET_BLEND = 0.00
# # # # #             print("  det_head: not in checkpoint → DET_BLEND=0.00 (pure FM)")

# # # # #         print(f"  ✅ FM loaded (epoch={ep})")
# # # # #         return model

# # # # #     except Exception as e:
# # # # #         print(f"  ❌ FM load failed: {e}")
# # # # #         import traceback; traceback.print_exc()
# # # # #         return None


# # # # # def _load_residual_model(ckpt, state, device, args, ep):
# # # # #     """
# # # # #     Load FMResidualCorrector từ checkpoint.

# # # # #     KEY INSIGHT: BEST_V47.pth chứa sttrans.* weights (242 keys) bên trong
# # # # #     → KHÔNG cần --sttrans_ckpt riêng để load.
# # # # #     → Tạo model với dummy STTrans, rồi load_state_dict(strict=False)
# # # # #       sẽ load cả sttrans.* weights từ checkpoint.

# # # # #     Nếu FMResidualCorrector không import được, fallback:
# # # # #     → Load chỉ phần sttrans.* vào STTrans model và evaluate như STTrans.
# # # # #     """
# # # # #     FMResidualCorrector = _import_residual_class()

# # # # #     if FMResidualCorrector is not None:
# # # # #         # Approach A: load full FMResidualCorrector
# # # # #         # Tạo model với sttrans_checkpoint dummy (sẽ bị overwrite bởi state_dict)
# # # # #         # Trick: save sttrans weights tạm thời, init model, load state
# # # # #         try:
# # # # #             # Tạo temp file với sttrans weights để init
# # # # #             import tempfile

# # # # #             # Extract sttrans state dict từ checkpoint
# # # # #             sttrans_state = {k[len("sttrans."):]: v
# # # # #                              for k, v in state.items()
# # # # #                              if k.startswith("sttrans.")}

# # # # #             # Save temp checkpoint cho STTrans
# # # # #             with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tf:
# # # # #                 tmp_path = tf.name
# # # # #             torch.save({"model_state_dict": sttrans_state}, tmp_path)

# # # # #             model = FMResidualCorrector(
# # # # #                 sttrans_checkpoint=tmp_path,
# # # # #                 obs_len=args.obs_len, pred_len=args.pred_len,
# # # # #                 device=str(device),
# # # # #             ).to(device)

# # # # #             os.unlink(tmp_path)  # cleanup

# # # # #             if hasattr(model, "init_ema"):
# # # # #                 model.init_ema()

# # # # #             result = model.load_state_dict(state, strict=False)
# # # # #             n_miss = len(result.missing_keys)
# # # # #             n_unex = len(result.unexpected_keys)
# # # # #             if n_miss:  print(f"  Missing   : {n_miss}")
# # # # #             if n_unex:  print(f"  Unexpected: {n_unex}")

# # # # #             print(f"  ✅ FMResidualCorrector loaded (epoch={ep})")
# # # # #             return model

# # # # #         except Exception as e:
# # # # #             print(f"  ⚠ FMResidualCorrector full load failed: {e}")
# # # # #             print("  → Falling back to STTrans-only evaluation")

# # # # #     # Approach B: fallback — load chỉ STTrans part và evaluate như STTrans
# # # # #     print("  Loading sttrans.* weights as standalone STTrans...")
# # # # #     try:
# # # # #         from Model.st_trans_model import STTrans
# # # # #         model = STTrans(obs_len=args.obs_len, pred_len=args.pred_len).to(device)

# # # # #         # Extract sttrans.* prefix
# # # # #         sttrans_state = {k[len("sttrans."):]: v
# # # # #                          for k, v in state.items()
# # # # #                          if k.startswith("sttrans.")}

# # # # #         result = model.load_state_dict(sttrans_state, strict=False)
# # # # #         n_miss = len(result.missing_keys)
# # # # #         if n_miss: print(f"  STTrans missing: {n_miss}")

# # # # #         print(f"  ✅ STTrans base loaded from FMResidualCorrector ckpt (epoch={ep})")
# # # # #         print("  ⚠  Note: evaluating STTrans base only (not the corrector)")
# # # # #         return model

# # # # #     except Exception as e2:
# # # # #         print(f"  ❌ Fallback STTrans load also failed: {e2}")
# # # # #         return None


# # # # # def load_lstm_model(ckpt_path: str, device, args):
# # # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # # #         print(f"  ⚠  LSTM checkpoint not found: {ckpt_path}")
# # # # #         return None
# # # # #     try:
# # # # #         from Model.paper_baseline_model import PaperBaseline
# # # # #         lstm_type = getattr(args, "lstm_type", "lstm")
# # # # #         model = PaperBaseline(model_type=lstm_type,
# # # # #                               pred_len=args.pred_len, obs_len=args.obs_len).to(device)
# # # # #         ckpt  = torch.load(ckpt_path, map_location=device)
# # # # #         state = _load_state_dict(ckpt)
# # # # #         res   = model.load_state_dict(state, strict=False)
# # # # #         if res.missing_keys:    print(f"  Missing   : {len(res.missing_keys)}")
# # # # #         if res.unexpected_keys: print(f"  Unexpected: {len(res.unexpected_keys)}")
# # # # #         ep = ckpt.get("epoch","?") if isinstance(ckpt,dict) else "?"
# # # # #         print(f"  ✅ {lstm_type.upper()} loaded (epoch={ep})")
# # # # #         return model
# # # # #     except Exception as e:
# # # # #         print(f"  ❌ LSTM load failed: {e}")
# # # # #         import traceback; traceback.print_exc()
# # # # #         return None


# # # # # def load_sttrans_model(ckpt_path: str, device, args):
# # # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # # #         print(f"  ⚠  ST-Trans checkpoint not found: {ckpt_path}")
# # # # #         return None
# # # # #     try:
# # # # #         sttrans_type = getattr(args, "sttrans_type", "sttrans")
# # # # #         if sttrans_type == "sttrans_ar":
# # # # #             from Model.st_trans_model import STTransAR as STTransModel
# # # # #         else:
# # # # #             from Model.st_trans_model import STTrans as STTransModel
# # # # #         model = STTransModel(obs_len=args.obs_len, pred_len=args.pred_len).to(device)
# # # # #         ckpt  = torch.load(ckpt_path, map_location=device)
# # # # #         state = _load_state_dict(ckpt)
# # # # #         res   = model.load_state_dict(state, strict=False)
# # # # #         if res.missing_keys:    print(f"  Missing   : {len(res.missing_keys)}")
# # # # #         if res.unexpected_keys: print(f"  Unexpected: {len(res.unexpected_keys)}")
# # # # #         ep  = ckpt.get("epoch","?") if isinstance(ckpt,dict) else "?"
# # # # #         tag = "STTransAR" if sttrans_type=="sttrans_ar" else "STTrans"
# # # # #         print(f"  ✅ {tag} loaded (epoch={ep})")
# # # # #         return model
# # # # #     except Exception as e:
# # # # #         print(f"  ❌ ST-Trans load failed: {e}")
# # # # #         import traceback; traceback.print_exc()
# # # # #         return None


# # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # #  Inference wrappers
# # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # @torch.no_grad()
# # # # # def infer_fm(model, batch_list, args, device):
# # # # #     try:
# # # # #         model.eval()

# # # # #         # FMResidualCorrector và TCFlowMatching đều có sample()
# # # # #         # STTrans fallback cũng có sample()
# # # # #         # → Dùng chung interface
# # # # #         if hasattr(model, "sample"):
# # # # #             # TCFlowMatching: cần EMA apply
# # # # #             ema = get_ema_obj(model); backup = None
# # # # #             if ema is not None:
# # # # #                 try: backup = ema.apply_to(model)
# # # # #                 except Exception: pass

# # # # #             # TCFlowMatching cần importance_weight, FMResidualCorrector ignore extra kwargs
# # # # #             try:
# # # # #                 pred, _, _ = model.sample(
# # # # #                     batch_list,
# # # # #                     num_ensemble=args.fm_ensemble,
# # # # #                     ddim_steps=args.fm_ode_steps,
# # # # #                     importance_weight=True)
# # # # #             except TypeError:
# # # # #                 # FMResidualCorrector/STTrans không có importance_weight
# # # # #                 pred, _, _ = model.sample(batch_list,
# # # # #                                           num_ensemble=args.fm_ensemble)

# # # # #             if backup is not None:
# # # # #                 try: ema.restore(model, backup)
# # # # #                 except Exception: pass

# # # # #         else:
# # # # #             # Direct forward (old STTrans without sample())
# # # # #             pred = model(batch_list)
# # # # #             if pred.dim() == 3 and pred.shape[0] == batch_list[0].shape[1]:
# # # # #                 pred = pred.permute(1, 0, 2)

# # # # #         return pred[..., :2] if pred.shape[-1] > 2 else pred

# # # # #     except Exception as e:
# # # # #         print(f"    FM inference error: {e}")
# # # # #         import traceback; traceback.print_exc()
# # # # #         return None


# # # # # @torch.no_grad()
# # # # # def infer_lstm(model, batch_list, args, device):
# # # # #     try:
# # # # #         model.eval()
# # # # #         if hasattr(model, "sample"):
# # # # #             pred, _, _ = model.sample(batch_list, num_ensemble=1)
# # # # #         else:
# # # # #             pred = model(batch_list)
# # # # #         if pred.dim() == 3 and pred.shape[0] == batch_list[0].shape[1]:
# # # # #             pred = pred.permute(1, 0, 2)
# # # # #         return pred[..., :2]
# # # # #     except Exception as e:
# # # # #         print(f"    LSTM inference error: {e}")
# # # # #         return None


# # # # # @torch.no_grad()
# # # # # def infer_sttrans(model, batch_list, args, device):
# # # # #     try:
# # # # #         model.eval()
# # # # #         if hasattr(model, "sample"):
# # # # #             pred, _, _ = model.sample(batch_list, num_ensemble=1)
# # # # #         else:
# # # # #             pred = model(batch_list)
# # # # #         if pred.dim() == 3 and pred.shape[0] == batch_list[0].shape[1]:
# # # # #             pred = pred.permute(1, 0, 2)
# # # # #         return pred[..., :2]
# # # # #     except Exception as e:
# # # # #         print(f"    STTrans inference error: {e}")
# # # # #         return None


# # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # #  CSV + pretty print (unchanged from original)
# # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # _PER_STORM_FIELDS = ["model","storm_id","seq_idx","lead_h",
# # # # #                      "lon_pred","lat_pred","lon_true","lat_true",
# # # # #                      "dist_km","ate_km","cte_km"]

# # # # # _SUMMARY_FIELDS = ["model","storm_id","n_seq","ADE","ATE","CTE",
# # # # #                    "12h_dist","24h_dist","48h_dist","72h_dist",
# # # # #                    "12h_ate","24h_ate","48h_ate","72h_ate",
# # # # #                    "12h_cte","24h_cte","48h_cte","72h_cte",
# # # # #                    "beat_12h","beat_24h","beat_48h","beat_72h",
# # # # #                    "beat_ADE","beat_ATE","beat_CTE"]


# # # # # def write_per_storm_csv(acc, out_dir):
# # # # #     out = os.path.join(out_dir, f"{acc.model_name}_per_storm.csv")
# # # # #     with open(out, "w", newline="") as fh:
# # # # #         w = csv.DictWriter(fh, fieldnames=_PER_STORM_FIELDS)
# # # # #         w.writeheader()
# # # # #         for sr in acc.storms.values():
# # # # #             for row in sr.rows:
# # # # #                 w.writerow({"model": acc.model_name, **row,
# # # # #                             "lon_pred": f"{row['lon_pred']:.4f}",
# # # # #                             "lat_pred": f"{row['lat_pred']:.4f}",
# # # # #                             "lon_true": f"{row['lon_true']:.4f}",
# # # # #                             "lat_true": f"{row['lat_true']:.4f}"})
# # # # #     print(f"  💾 {out}")
# # # # #     return out


# # # # # def write_summary_csv(all_acc, out_dir):
# # # # #     out = os.path.join(out_dir, "summary_all_models.csv")
# # # # #     with open(out, "w", newline="") as fh:
# # # # #         w = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
# # # # #         w.writeheader()
# # # # #         def _b(v,t): return "YES" if (np.isfinite(v) and v<t) else "NO"
# # # # #         def _f(v):   return f"{v:.2f}" if np.isfinite(v) else "nan"
# # # # #         for mname, acc in all_acc.items():
# # # # #             for r in acc.all_summaries() + [acc.overall_summary()]:
# # # # #                 w.writerow({
# # # # #                     "model": mname, "storm_id": r["storm_id"],
# # # # #                     "n_seq": r["n_seq"],
# # # # #                     "ADE": _f(r["ADE"]), "ATE": _f(r["ATE"]), "CTE": _f(r["CTE"]),
# # # # #                     "12h_dist": _f(r["12h_dist"]), "24h_dist": _f(r["24h_dist"]),
# # # # #                     "48h_dist": _f(r["48h_dist"]), "72h_dist": _f(r["72h_dist"]),
# # # # #                     "12h_ate": _f(r["12h_ate"]),   "24h_ate": _f(r["24h_ate"]),
# # # # #                     "48h_ate": _f(r["48h_ate"]),   "72h_ate": _f(r["72h_ate"]),
# # # # #                     "12h_cte": _f(r["12h_cte"]),   "24h_cte": _f(r["24h_cte"]),
# # # # #                     "48h_cte": _f(r["48h_cte"]),   "72h_cte": _f(r["72h_cte"]),
# # # # #                     "beat_12h": _b(r["12h_dist"],TARGETS["12h"]),
# # # # #                     "beat_24h": _b(r["24h_dist"],TARGETS["24h"]),
# # # # #                     "beat_48h": _b(r["48h_dist"],TARGETS["48h"]),
# # # # #                     "beat_72h": _b(r["72h_dist"],TARGETS["72h"]),
# # # # #                     "beat_ADE": _b(r["ADE"],TARGETS["ADE"]),
# # # # #                     "beat_ATE": _b(r["ATE"],TARGETS["ATE"]),
# # # # #                     "beat_CTE": _b(r["CTE"],TARGETS["CTE"]),
# # # # #                 })
# # # # #     print(f"  💾 Summary: {out}")
# # # # #     return out


# # # # # def print_model_results(acc):
# # # # #     cfg = MODEL_CONFIGS.get(acc.model_name, {}); col = cfg.get("color","⚪")
# # # # #     ref = cfg.get("ref","")
# # # # #     print(f"\n{'═'*110}")
# # # # #     print(f"  {col}  {acc.model_name}  |  {ref}")
# # # # #     print(f"{'═'*110}")
# # # # #     print(f"  {'Storm':<16} {'N':>4}  {'ADE':>8}  {'ATE':>8}  {'CTE':>8}  "
# # # # #           f"║  {'12h':>7}  {'24h':>7}  {'48h':>7}  {'72h':>7}")
# # # # #     print("  " + "─"*106)

# # # # #     def v(x, t=None, w=8, dec=1):
# # # # #         if not np.isfinite(x): return f"{'nan':>{w+2}}"
# # # # #         s = f"{x:>{w}.{dec}f}"
# # # # #         s += ("✅" if x<t else "❌") if t is not None else "  "
# # # # #         return s

# # # # #     for r in acc.all_summaries():
# # # # #         print(f"  🌀 {r['storm_id']:<14} {r['n_seq']:>4}  "
# # # # #               f"{v(r['ADE'],TARGETS['ADE'])}  {v(r['ATE'],TARGETS['ATE'])}  "
# # # # #               f"{v(r['CTE'],TARGETS['CTE'])}  "
# # # # #               f"║  {v(r['12h_dist'],TARGETS['12h'],7)}  "
# # # # #               f"{v(r['24h_dist'],TARGETS['24h'],7)}  "
# # # # #               f"{v(r['48h_dist'],TARGETS['48h'],7)}  "
# # # # #               f"{v(r['72h_dist'],TARGETS['72h'],7)}")

# # # # #     ov = acc.overall_summary()
# # # # #     print("  " + "─"*106)
# # # # #     print(f"  📊 {'OVERALL':<14} {ov['n_seq']:>4}  "
# # # # #           f"{v(ov['ADE'],TARGETS['ADE'])}  {v(ov['ATE'],TARGETS['ATE'])}  "
# # # # #           f"{v(ov['CTE'],TARGETS['CTE'])}  "
# # # # #           f"║  {v(ov['12h_dist'],TARGETS['12h'],7)}  "
# # # # #           f"{v(ov['24h_dist'],TARGETS['24h'],7)}  "
# # # # #           f"{v(ov['48h_dist'],TARGETS['48h'],7)}  "
# # # # #           f"{v(ov['72h_dist'],TARGETS['72h'],7)}")
# # # # #     print(f"{'═'*110}\n")


# # # # # def print_comparison_table(all_acc):
# # # # #     print(f"\n{'═'*95}")
# # # # #     print(f"  📊  MODEL COMPARISON  —  Overall Test Set")
# # # # #     print(f"{'═'*95}")
# # # # #     print(f"  {'Model':<14}  {'ADE':>9}  {'ATE':>9}  {'CTE':>9}  "
# # # # #           f"║  {'12h':>8}  {'24h':>8}  {'48h':>8}  {'72h':>8}")
# # # # #     print("  " + "─"*91)

# # # # #     def vv(x, t=None, w=9, dec=1):
# # # # #         if not np.isfinite(x): return f"{'nan':>{w+2}}"
# # # # #         s = f"{x:>{w}.{dec}f}"
# # # # #         s += ("✅" if x<t else "❌") if t is not None else "  "
# # # # #         return s

# # # # #     for mname, acc in all_acc.items():
# # # # #         col = MODEL_CONFIGS.get(mname, {}).get("color","⚪")
# # # # #         ov  = acc.overall_summary()
# # # # #         print(f"  {col} {mname:<12}  "
# # # # #               f"{vv(ov['ADE'],TARGETS['ADE'])}  {vv(ov['ATE'],TARGETS['ATE'])}  "
# # # # #               f"{vv(ov['CTE'],TARGETS['CTE'])}  "
# # # # #               f"║  {vv(ov['12h_dist'],TARGETS['12h'],8)}  "
# # # # #               f"{vv(ov['24h_dist'],TARGETS['24h'],8)}  "
# # # # #               f"{vv(ov['48h_dist'],TARGETS['48h'],8)}  "
# # # # #               f"{vv(ov['72h_dist'],TARGETS['72h'],8)}")

# # # # #     print("  " + "─"*91)
# # # # #     print(f"  Targets: ADE<{TARGETS['ADE']} | ATE<{TARGETS['ATE']} | "
# # # # #           f"CTE<{TARGETS['CTE']} | 12h<{TARGETS['12h']} | 24h<{TARGETS['24h']} | "
# # # # #           f"48h<{TARGETS['48h']} | 72h<{TARGETS['72h']} (km)")
# # # # #     print(f"{'═'*95}\n")


# # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # #  Main evaluation loop
# # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # def run_evaluation(model, model_name, test_loader, device, args, infer_fn):
# # # # #     acc = MultiStormAccumulator(model_name)
# # # # #     n   = len(test_loader); t0 = time.perf_counter()
# # # # #     print(f"\n  Running {model_name} on {n} batches...")

# # # # #     for i, batch in enumerate(test_loader):
# # # # #         bl        = move_to_device(list(batch), device)
# # # # #         B         = bl[0].shape[1]
# # # # #         pred_norm = infer_fn(model, bl, args, device)
# # # # #         if pred_norm is None: continue

# # # # #         gt_norm   = bl[1]
# # # # #         storm_ids = extract_storm_ids(batch, B)
# # # # #         T         = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # #         acc.add_batch(pred_norm[:T], gt_norm[:T], storm_ids)

# # # # #         if (i+1) % max(1, n//5) == 0 or i == n-1:
# # # # #             ov = acc.overall_summary()
# # # # #             print(f"    [{i+1:>4}/{n}]  ADE={ov['ADE']:.1f}km  "
# # # # #                   f"storms={len(acc.storms)}  "
# # # # #                   f"elapsed={time.perf_counter()-t0:.1f}s")

# # # # #     ov = acc.overall_summary()
# # # # #     print(f"  ✅ {model_name}: ADE={ov['ADE']:.1f}  "
# # # # #           f"ATE={ov['ATE']:.1f}  CTE={ov['CTE']:.1f} km")
# # # # #     return acc


# # # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # # #  Args + Main
# # # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # # def get_args():
# # # # #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # #     p.add_argument("--dataset_root",  default="TCND_vn")
# # # # #     p.add_argument("--obs_len",       default=8,   type=int)
# # # # #     p.add_argument("--pred_len",      default=12,  type=int)
# # # # #     p.add_argument("--batch_size",    default=16,  type=int)
# # # # #     p.add_argument("--num_workers",   default=2,   type=int)
# # # # #     p.add_argument("--skip",          default=1,   type=int)
# # # # #     p.add_argument("--min_ped",       default=1,   type=int)
# # # # #     p.add_argument("--threshold",     default=0.002, type=float)
# # # # #     p.add_argument("--other_modal",   default="gph")
# # # # #     p.add_argument("--test_year",     default=None, type=int)
# # # # #     p.add_argument("--delim",         default=" ")
# # # # #     p.add_argument("--fm_ckpt",       default=None)
# # # # #     p.add_argument("--lstm_ckpt",     default=None)
# # # # #     p.add_argument("--sttrans_ckpt",  default=None)
# # # # #     p.add_argument("--lstm_type",     default="lstm", choices=["lstm","gru","rnn"])
# # # # #     p.add_argument("--sttrans_type",  default="sttrans", choices=["sttrans","sttrans_ar"])
# # # # #     p.add_argument("--fm_ensemble",   default=30, type=int)
# # # # #     p.add_argument("--fm_ode_steps",  default=20, type=int)
# # # # #     p.add_argument("--fm_type",       default="auto",
# # # # #                    choices=["auto","tcfm","residual"],
# # # # #                    help="Force FM model type: auto|tcfm|residual")
# # # # #     p.add_argument("--output_dir",    default="results/test_eval")
# # # # #     p.add_argument("--gpu_num",       default="0")
# # # # #     p.add_argument("--eval_models",   default="all",
# # # # #                    help="'all' hoặc csv: 'fm,lstm,sttrans'")
# # # # #     p.add_argument("--no_detail_csv", action="store_true")
# # # # #     return p.parse_args()


# # # # # def _version_from_path(p):
# # # # #     parts = os.path.normpath(p).split(os.sep)
# # # # #     for part in reversed(parts[:-1]):
# # # # #         if part.startswith("v") and part[1:].isdigit():
# # # # #             return part
# # # # #     return "ours"


# # # # # def main(args):
# # # # #     if torch.cuda.is_available():
# # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # #     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# # # # #     print("="*80)
# # # # #     print(f"  TC Model Evaluator  (fix v2 — robust import)")
# # # # #     print(f"  Started : {ts}")
# # # # #     print(f"  Device  : {device}")
# # # # #     print(f"  Output  : {args.output_dir}")
# # # # #     print("="*80)

# # # # #     eval_set = ({"fm","lstm","sttrans"} if args.eval_models=="all"
# # # # #                 else {m.strip().lower() for m in args.eval_models.split(",")})

# # # # #     print("\n  Loading test dataset...")
# # # # #     try:
# # # # #         from Model.data.loader_training import data_loader
# # # # #         _, test_loader = data_loader(
# # # # #             args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # # #         print(f"  ✅ {len(test_loader)} batches")
# # # # #     except Exception as e:
# # # # #         print(f"  ❌ Test data load failed: {e}")
# # # # #         import traceback; traceback.print_exc()
# # # # #         return

# # # # #     all_acc = {}

# # # # #     if "fm" in eval_set and args.fm_ckpt:
# # # # #         print(f"\n{'─'*50}")
# # # # #         print("  Loading FlowMatching...")
# # # # #         model = load_fm_model(args.fm_ckpt, device, args)
# # # # #         if model is not None:
# # # # #             tag = "FM_" + _version_from_path(args.fm_ckpt)
# # # # #             all_acc[tag] = run_evaluation(model, tag, test_loader,
# # # # #                                           device, args, infer_fm)
# # # # #             del model
# # # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # # #     if "lstm" in eval_set and args.lstm_ckpt:
# # # # #         print(f"\n{'─'*50}")
# # # # #         model = load_lstm_model(args.lstm_ckpt, device, args)
# # # # #         if model is not None:
# # # # #             tag = args.lstm_type.upper()
# # # # #             all_acc[tag] = run_evaluation(model, tag, test_loader,
# # # # #                                           device, args, infer_lstm)
# # # # #             del model
# # # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # # #     if "sttrans" in eval_set and args.sttrans_ckpt:
# # # # #         print(f"\n{'─'*50}")
# # # # #         model = load_sttrans_model(args.sttrans_ckpt, device, args)
# # # # #         if model is not None:
# # # # #             tag = "STTransAR" if args.sttrans_type=="sttrans_ar" else "STTrans"
# # # # #             all_acc[tag] = run_evaluation(model, tag, test_loader,
# # # # #                                           device, args, infer_sttrans)
# # # # #             del model
# # # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # # #     if not all_acc:
# # # # #         print("\n  ❌ No models evaluated.")
# # # # #         print("     Specify: --fm_ckpt / --lstm_ckpt / --sttrans_ckpt")
# # # # #         return

# # # # #     for acc in all_acc.values():
# # # # #         print_model_results(acc)
# # # # #     if len(all_acc) > 1:
# # # # #         print_comparison_table(all_acc)

# # # # #     print("\n  Writing CSVs...")
# # # # #     csv_paths = []
# # # # #     for acc in all_acc.values():
# # # # #         csv_paths.append(write_per_storm_csv(acc, args.output_dir))
# # # # #     csv_paths.append(write_summary_csv(all_acc, args.output_dir))

# # # # #     metric_map = [("ADE","ADE"),("ATE","ATE"),("CTE","CTE"),
# # # # #                   ("12h_dist","12h"),("24h_dist","24h"),
# # # # #                   ("48h_dist","48h"),("72h_dist","72h")]
# # # # #     print(f"\n{'═'*80}")
# # # # #     print("  FINAL BEAT SUMMARY")
# # # # #     print(f"{'─'*80}")
# # # # #     for mname, acc in all_acc.items():
# # # # #         ov  = acc.overall_summary()
# # # # #         col = MODEL_CONFIGS.get(mname, {}).get("color","⚪")
# # # # #         beats = [
# # # # #             f"{label}={ov.get(k,float('nan')):.1f}✅"
# # # # #             for k, label in metric_map
# # # # #             if np.isfinite(ov.get(k,float("nan")))
# # # # #             and ov[k] < TARGETS.get(label, float("inf"))
# # # # #         ]
# # # # #         print(f"  {col} {mname:<12}: " +
# # # # #               (" | ".join(beats) if beats else "no targets beaten"))
# # # # #     print(f"{'═'*80}\n")


# # # # # if __name__ == "__main__":
# # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# # # # #     args = get_args()

# # # # #     _kaggle_base = "/kaggle/working/TC_FM"
# # # # #     if args.dataset_root == "TCND_vn" and os.path.isdir(_kaggle_base):
# # # # #         args.dataset_root = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"

# # # # #     _runs = os.path.join(_kaggle_base,"runs") if os.path.isdir(_kaggle_base) else "runs"

# # # # #     if args.fm_ckpt is None:
# # # # #         for c in ([os.path.join(_runs,f"v{v}","best_ema.pth") for v in ["67","66","65","61","60","59","58"]] +
# # # # #                   [os.path.join(_runs,f"v{v}","best_model.pth") for v in ["67","61","60","59"]]):
# # # # #             if os.path.exists(c):
# # # # #                 args.fm_ckpt = c
# # # # #                 print(f"  Auto FM ckpt: {c}")
# # # # #                 break

# # # # #     if args.lstm_ckpt is None:
# # # # #         for c in [os.path.join(_runs,d,"best_model.pth")
# # # # #                   for d in ["lstm","LSTM","gru","GRU","paper_baseline"]]:
# # # # #             if os.path.exists(c):
# # # # #                 args.lstm_ckpt = c; print(f"  Auto LSTM ckpt: {c}"); break

# # # # #     if args.sttrans_ckpt is None:
# # # # #         for c in [os.path.join(_runs,d,"best_model.pth")
# # # # #                   for d in ["sttrans","STTrans","st_trans"]]:
# # # # #             if os.path.exists(c):
# # # # #                 args.sttrans_ckpt = c; print(f"  Auto STTrans ckpt: {c}"); break

# # # # #     main(args)

# # # # """
# # # # scripts/evaluate_test_storms.py
# # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # Script đánh giá tất cả models (LSTM/GRU/RNN, ST-Trans, FlowMatching) trên test set.

# # # # FIX v3:
# # # #   ✅ Hỗ trợ TCFlowMatching v2 (encoder.* + velocity.*) lẫn v1 (net.*)
# # # #   ✅ Robust FM import — auto-detect class name trong module
# # # #   ✅ Auto-detect checkpoint type: TCFlowMatching vs FMResidualCorrector
# # # #   ✅ det_head support — set _det_epoch=999 khi checkpoint có det_head weights
# # # #   ✅ Architecture detection mở rộng cho v67 (det_head, obs_enc, horizon_queries)
# # # #   ✅ FMResidualCorrector support trong load và infer
# # # #   ✅ Graceful fallback khi import fail

# # # # Cách dùng:
# # # #     python scripts/evaluate_test_storms.py \
# # # #         --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# # # #         --fm_ckpt      runs/v67/best_ema.pth \
# # # #         --lstm_ckpt    runs/lstm/best_model.pth \
# # # #         --sttrans_ckpt runs/sttrans/best_model.pth \
# # # #         --output_dir   results/test_eval \
# # # #         --gpu_num 0
# # # # """
# # # # from __future__ import annotations

# # # # import sys
# # # # import os
# # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # import argparse
# # # # import csv
# # # # import math
# # # # import time
# # # # import importlib
# # # # import inspect
# # # # from collections import defaultdict
# # # # from datetime import datetime
# # # # from typing import Dict, List, Optional, Tuple

# # # # import numpy as np
# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.nn.functional as F
# # # # from torch.utils.data import DataLoader

# # # # R_EARTH  = 6371.0
# # # # DEG2KM   = 111.0
# # # # DT_HOURS = 6.0

# # # # LEAD_HOURS = [12, 24, 36, 48, 60, 72]
# # # # LEAD_STEPS = {h: h // 6 - 1 for h in LEAD_HOURS}

# # # # TARGETS = {
# # # #     "12h": 50.0, "24h": 100.0, "48h": 200.0, "72h": 297.0,
# # # #     "ADE": 136.41, "ATE": 79.94, "CTE": 93.58,
# # # # }

# # # # MODEL_CONFIGS = {
# # # #     "LSTM"     : {"color": "🔵", "ref": "Rahman 2025"},
# # # #     "GRU"      : {"color": "🔵", "ref": "Rahman 2025"},
# # # #     "RNN"      : {"color": "🔵", "ref": "Rahman 2025"},
# # # #     "STTrans"  : {"color": "🟡", "ref": "Faiaz 2026"},
# # # #     "STTransAR": {"color": "🟡", "ref": "Faiaz 2026 AR"},
# # # #     "FM"       : {"color": "🔴", "ref": "Ours FM"},
# # # #     "FM_v59"   : {"color": "🔴", "ref": "Ours v59"},
# # # #     "FM_v67"   : {"color": "🔴", "ref": "Ours v67"},
# # # #     "FM_ours"  : {"color": "🔴", "ref": "Ours"},
# # # #     "FMResidual": {"color": "🟠", "ref": "Ours Residual"},
# # # # }


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  Coordinate utilities
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # def norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
# # # #     out = arr.clone()
# # # #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# # # #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# # # #     return out


# # # # def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # #     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
# # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # #     a    = (torch.sin(dlat / 2).pow(2) +
# # # #             torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
# # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# # # # def compute_ate_cte(pred_deg, gt_deg):
# # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # #     if T < 2:
# # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # #         return z, z
# # # #     lon1 = torch.deg2rad(gt_deg[:T-1,:,0]); lat1 = torch.deg2rad(gt_deg[:T-1,:,1])
# # # #     lon2 = torch.deg2rad(gt_deg[1:T, :,0]); lat2 = torch.deg2rad(gt_deg[1:T, :,1])
# # # #     dlon_a = lon2 - lon1
# # # #     y_a = torch.sin(dlon_a)*torch.cos(lat2)
# # # #     x_a = torch.cos(lat1)*torch.sin(lat2) - torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon_a)
# # # #     bear_along = torch.atan2(y_a, x_a)
# # # #     lon3 = torch.deg2rad(pred_deg[1:T,:,0]); lat3 = torch.deg2rad(pred_deg[1:T,:,1])
# # # #     dlon_e = lon3 - lon2
# # # #     y_e = torch.sin(dlon_e)*torch.cos(lat3)
# # # #     x_e = torch.cos(lat2)*torch.sin(lat3) - torch.sin(lat2)*torch.cos(lat3)*torch.cos(dlon_e)
# # # #     bear_err  = torch.atan2(y_e, x_e)
# # # #     total     = haversine_km(pred_deg[1:T], gt_deg[1:T])
# # # #     angle     = bear_err - bear_along
# # # #     return total*torch.cos(angle), total*torch.sin(angle)


# # # # def move_to_device(batch, device):
# # # #     out = list(batch)
# # # #     for i, x in enumerate(out):
# # # #         if torch.is_tensor(x):
# # # #             out[i] = x.to(device)
# # # #         elif isinstance(x, dict):
# # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # #                       for k, v in x.items()}
# # # #     return out


# # # # def get_raw_model(model):
# # # #     return model._orig_mod if hasattr(model, "_orig_mod") else model


# # # # def get_ema_obj(model):
# # # #     raw = get_raw_model(model)
# # # #     if hasattr(raw, "_ema") and raw._ema is not None:
# # # #         return raw._ema
# # # #     return None


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  Per-storm accumulator
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # class StormResult:
# # # #     def __init__(self, storm_id):
# # # #         self.storm_id = storm_id
# # # #         self.rows = []
# # # #         self.dist_by_lead = defaultdict(list)
# # # #         self.ate_by_lead  = defaultdict(list)
# # # #         self.cte_by_lead  = defaultdict(list)
# # # #         self.all_dist = []; self.all_ate = []; self.all_cte = []
# # # #         self.n_seq = 0

# # # #     def add_batch(self, pred_deg, gt_deg, seq_indices=None):
# # # #         T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # #         B = pred_deg.shape[1]
# # # #         if T < 2: return
# # # #         dist     = haversine_km(pred_deg[:T], gt_deg[:T])
# # # #         ate, cte = compute_ate_cte(pred_deg, gt_deg)
# # # #         self.n_seq += B
# # # #         for b in range(B):
# # # #             for t in range(T):
# # # #                 lead_h  = (t+1)*6
# # # #                 d_val   = float(dist[t,b])
# # # #                 ate_val = float(ate[t,b].abs()) if t < ate.shape[0] else float("nan")
# # # #                 cte_val = float(cte[t,b].abs()) if t < cte.shape[0] else float("nan")
# # # #                 self.rows.append({
# # # #                     "storm_id": self.storm_id,
# # # #                     "seq_idx" : int(seq_indices[b]) if seq_indices is not None else b,
# # # #                     "lead_h"  : lead_h,
# # # #                     "lon_pred": float(pred_deg[t,b,0]),
# # # #                     "lat_pred": float(pred_deg[t,b,1]),
# # # #                     "lon_true": float(gt_deg[t,b,0]),
# # # #                     "lat_true": float(gt_deg[t,b,1]),
# # # #                     "dist_km" : round(d_val, 2),
# # # #                     "ate_km"  : round(ate_val,2) if np.isfinite(ate_val) else "nan",
# # # #                     "cte_km"  : round(cte_val,2) if np.isfinite(cte_val) else "nan",
# # # #                 })
# # # #                 if lead_h in LEAD_STEPS:
# # # #                     self.dist_by_lead[lead_h].append(d_val)
# # # #                     if np.isfinite(ate_val): self.ate_by_lead[lead_h].append(ate_val)
# # # #                     if np.isfinite(cte_val): self.cte_by_lead[lead_h].append(cte_val)
# # # #                 self.all_dist.append(d_val)
# # # #                 if t < ate.shape[0]:
# # # #                     if np.isfinite(ate_val): self.all_ate.append(ate_val)
# # # #                     if np.isfinite(cte_val): self.all_cte.append(cte_val)

# # # #     def summary(self):
# # # #         def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # # #         r = {"storm_id": self.storm_id, "n_seq": self.n_seq,
# # # #              "ADE": _m(self.all_dist), "ATE": _m(self.all_ate), "CTE": _m(self.all_cte)}
# # # #         for h in LEAD_HOURS:
# # # #             r[f"{h}h_dist"] = _m(self.dist_by_lead.get(h,[]))
# # # #             r[f"{h}h_ate"]  = _m(self.ate_by_lead.get(h,[]))
# # # #             r[f"{h}h_cte"]  = _m(self.cte_by_lead.get(h,[]))
# # # #         return r


# # # # class MultiStormAccumulator:
# # # #     def __init__(self, model_name):
# # # #         self.model_name = model_name
# # # #         self.storms: Dict[str, StormResult] = {}

# # # #     def get_storm(self, storm_id):
# # # #         if storm_id not in self.storms:
# # # #             self.storms[storm_id] = StormResult(storm_id)
# # # #         return self.storms[storm_id]

# # # #     def add_batch(self, pred_norm, gt_norm, storm_ids):
# # # #         T  = min(pred_norm.shape[0], gt_norm.shape[0])
# # # #         B  = pred_norm.shape[1]
# # # #         pd = norm_to_deg(pred_norm[:T])
# # # #         gd = norm_to_deg(gt_norm[:T])
# # # #         sid_map = defaultdict(list)
# # # #         for b, sid in enumerate(storm_ids[:B]):
# # # #             sid_map[sid].append(b)
# # # #         for sid, bidx in sid_map.items():
# # # #             bidx_t = torch.tensor(bidx, dtype=torch.long)
# # # #             self.get_storm(sid).add_batch(pd[:,bidx_t,:], gd[:,bidx_t,:],
# # # #                                            seq_indices=bidx_t)

# # # #     def all_summaries(self):
# # # #         return [self.storms[sid].summary() for sid in sorted(self.storms)]

# # # #     def overall_summary(self):
# # # #         all_d, all_a, all_c = [], [], []
# # # #         lead_d = defaultdict(list); lead_a = defaultdict(list); lead_c = defaultdict(list)
# # # #         for sr in self.storms.values():
# # # #             all_d.extend(sr.all_dist); all_a.extend(sr.all_ate); all_c.extend(sr.all_cte)
# # # #             for h in LEAD_HOURS:
# # # #                 lead_d[h].extend(sr.dist_by_lead.get(h,[]))
# # # #                 lead_a[h].extend(sr.ate_by_lead.get(h,[]))
# # # #                 lead_c[h].extend(sr.cte_by_lead.get(h,[]))
# # # #         def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # # #         r = {"storm_id": "📊 OVERALL",
# # # #              "n_seq": sum(s.n_seq for s in self.storms.values()),
# # # #              "ADE": _m(all_d), "ATE": _m(all_a), "CTE": _m(all_c)}
# # # #         for h in LEAD_HOURS:
# # # #             r[f"{h}h_dist"] = _m(lead_d[h])
# # # #             r[f"{h}h_ate"]  = _m(lead_a[h])
# # # #             r[f"{h}h_cte"]  = _m(lead_c[h])
# # # #         return r


# # # # def extract_storm_ids(batch, B):
# # # #     for idx in [2, 12, 14, 15]:
# # # #         if idx < len(batch):
# # # #             item = batch[idx]
# # # #             if isinstance(item, (list, tuple)) and len(item) >= B:
# # # #                 if isinstance(item[0], str):
# # # #                     return list(item[:B])
# # # #             if (torch.is_tensor(item) and item.dtype == torch.long
# # # #                     and item.numel() >= B):
# # # #                 return [str(x.item()) for x in item[:B]]
# # # #     return [f"STORM_{i:03d}" for i in range(B)]


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  Robust FM import
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # def _import_fm_class():
# # # #     """
# # # #     Robust import của TCFlowMatching.
# # # #     Thử theo thứ tự:
# # # #     1. Import trực tiếp TCFlowMatching (v2+)
# # # #     2. Import TCDiffusion (alias)
# # # #     3. Scan module tìm class có sample() + get_loss_breakdown()
# # # #     """
# # # #     try:
# # # #         from Model.flow_matching_model import TCFlowMatching
# # # #         print("  [import] TCFlowMatching ✓")
# # # #         return TCFlowMatching
# # # #     except ImportError:
# # # #         pass

# # # #     try:
# # # #         from Model.flow_matching_model import TCDiffusion
# # # #         print("  [import] TCDiffusion (alias) ✓")
# # # #         return TCDiffusion
# # # #     except ImportError:
# # # #         pass

# # # #     try:
# # # #         mod = importlib.import_module("Model.flow_matching_model")
# # # #         candidates = []
# # # #         for name, cls in inspect.getmembers(mod, inspect.isclass):
# # # #             if (hasattr(cls, "sample") and hasattr(cls, "get_loss_breakdown")
# # # #                     and issubclass(cls, nn.Module)
# # # #                     and cls.__module__ == mod.__name__):
# # # #                 candidates.append((name, cls))
# # # #         if candidates:
# # # #             for name, cls in candidates:
# # # #                 if "Flow" in name or "TC" in name:
# # # #                     print(f"  [import] Found {name} by scan ✓")
# # # #                     return cls
# # # #             name, cls = candidates[0]
# # # #             print(f"  [import] Fallback to {name} by scan ✓")
# # # #             return cls
# # # #     except Exception as e:
# # # #         print(f"  [import] Module scan failed: {e}")

# # # #     print("  [import] ❌ Cannot find FM class in Model.flow_matching_model")
# # # #     return None


# # # # def _import_residual_class():
# # # #     """Import FMResidualCorrector nếu có."""
# # # #     try:
# # # #         from Model.fm_residual_corrector import FMResidualCorrector
# # # #         return FMResidualCorrector
# # # #     except ImportError:
# # # #         pass
# # # #     try:
# # # #         mod = importlib.import_module("Model.flow_matching_model")
# # # #         if hasattr(mod, "FMResidualCorrector"):
# # # #             return mod.FMResidualCorrector
# # # #     except Exception:
# # # #         pass
# # # #     return None


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  Detect checkpoint type
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # def _detect_ckpt_type(state: dict) -> str:
# # # #     """
# # # #     Detect loại model từ state dict keys.
# # # #     FMResidualCorrector: có ctx_enc.* VÀ corr_net.*
# # # #     TCFlowMatching:      có net.* (v1) hoặc encoder.* + velocity.* (v2)
# # # #     Returns: 'residual' | 'tcfm'
# # # #     """
# # # #     keys = set(state.keys())
# # # #     has_ctx_enc  = any(k.startswith("ctx_enc.")  for k in keys)
# # # #     has_corr_net = any(k.startswith("corr_net.") for k in keys)
# # # #     has_gate     = any(k.startswith("gate.")     for k in keys)

# # # #     if sum([has_ctx_enc, has_corr_net, has_gate]) >= 2:
# # # #         return "residual"

# # # #     return "tcfm"


# # # # def _detect_fm_version(state: dict) -> str:
# # # #     """
# # # #     Detect version của TCFlowMatching từ state dict.
# # # #     v2: có encoder.* và velocity.*  (flow_matching_model.py v2)
# # # #     v1: có net.*                    (flow_matching_model.py v1/v59/v67)
# # # #     Returns: 'v2' | 'v1'
# # # #     """
# # # #     has_encoder  = any(k.startswith("encoder.")  for k in state)
# # # #     has_velocity = any(k.startswith("velocity.") for k in state)
# # # #     has_net      = any(k.startswith("net.")       for k in state)

# # # #     if has_encoder and has_velocity and not has_net:
# # # #         return "v2"
# # # #     return "v1"


# # # # def _detect_has_det_head(state: dict) -> bool:
# # # #     """Checkpoint có det_head weights không (v67+)?"""
# # # #     return any(k.startswith("net.det_head.") or
# # # #                k.startswith("net.det_head") for k in state.keys())


# # # # def _load_state_dict(ckpt) -> dict:
# # # #     """Extract state_dict từ checkpoint (nhiều format)."""
# # # #     if not isinstance(ckpt, dict):
# # # #         return ckpt
# # # #     # train_fm.py v2 lưu key "model" (không phải "model_state_dict")
# # # #     for key in ["model_state_dict", "model_state", "state_dict", "model"]:
# # # #         if key in ckpt and isinstance(ckpt[key], dict):
# # # #             sd = ckpt[key]
# # # #             if any(torch.is_tensor(v) for v in sd.values()):
# # # #                 return sd
# # # #     if any(torch.is_tensor(v) for v in ckpt.values()
# # # #            if isinstance(v, torch.Tensor)):
# # # #         return ckpt
# # # #     return ckpt


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  load_fm_model — hỗ trợ v2 (encoder/velocity) lẫn v1 (net)
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # def load_fm_model(ckpt_path: str, device, args):
# # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # #         print(f"  ⚠  FM checkpoint not found: {ckpt_path}")
# # # #         return None

# # # #     try:
# # # #         print(f"  Loading: {ckpt_path}")
# # # #         ckpt  = torch.load(ckpt_path, map_location=device)
# # # #         state = _load_state_dict(ckpt)
# # # #         ep    = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"

# # # #         from collections import Counter
# # # #         prefix_counts = Counter(k.split(".")[0] for k in state.keys())
# # # #         print(f"  Key prefixes: " +
# # # #               ", ".join(f"{p}({n})" for p, n in
# # # #                         sorted(prefix_counts.items(), key=lambda x: -x[1])[:8]))

# # # #         # --fm_type flag overrides auto-detection
# # # #         forced = getattr(args, "fm_type", "auto")
# # # #         if forced == "tcfm":
# # # #             ckpt_type = "tcfm"
# # # #             print(f"  Checkpoint type: tcfm (forced by --fm_type)")
# # # #         elif forced == "residual":
# # # #             ckpt_type = "residual"
# # # #             print(f"  Checkpoint type: residual (forced by --fm_type)")
# # # #         else:
# # # #             ckpt_type = _detect_ckpt_type(state)
# # # #             print(f"  Checkpoint type: {ckpt_type}  (epoch={ep})")

# # # #         # ── FMResidualCorrector ──────────────────────────────────────────
# # # #         if ckpt_type == "residual":
# # # #             return _load_residual_model(ckpt, state, device, args, ep)

# # # #         # ── TCFlowMatching ───────────────────────────────────────────────
# # # #         TCFlowMatching = _import_fm_class()
# # # #         if TCFlowMatching is None:
# # # #             print("  ❌ Cannot import FM class — check flow_matching_model.py")
# # # #             return None

# # # #         # Detect v2 vs v1
# # # #         fm_version = _detect_fm_version(state)
# # # #         print(f"  FM version: {fm_version}")

# # # #         if fm_version == "v2":
# # # #             model = _load_fm_v2(ckpt, state, device, args, ep, TCFlowMatching)
# # # #         else:
# # # #             model = _load_fm_v1(ckpt, state, device, args, ep, TCFlowMatching)

# # # #         return model

# # # #     except Exception as e:
# # # #         print(f"  ❌ FM load failed: {e}")
# # # #         import traceback; traceback.print_exc()
# # # #         return None


# # # # def _load_fm_v2(ckpt, state, device, args, ep, TCFlowMatching):
# # # #     """
# # # #     Load TCFlowMatching v2 (encoder.* + velocity.*).
# # # #     train_fm.py v2 lưu: model.encoder, model.velocity, model._ema
# # # #     """
# # # #     print(f"  Architecture: v2 (ContextEncoder + VelocityTransformer)")

# # # #     # Detect hyperparams từ state dict
# # # #     # VelocityTransformer: velocity.decoder.layers.N.*
# # # #     num_dec_layers = sum(
# # # #         1 for i in range(10)
# # # #         if f"velocity.decoder.layers.{i}.self_attn.in_proj_weight" in state
# # # #            or f"velocity.decoder.layers.{i}.norm1.weight" in state
# # # #     ) or 4

# # # #     # d_model từ velocity.traj_embed.weight shape
# # # #     d_model_key = "velocity.traj_embed.weight"
# # # #     d_model = int(state[d_model_key].shape[0]) if d_model_key in state else 256

# # # #     # d_cond từ velocity.cond_proj.0.weight
# # # #     d_cond_key = "velocity.cond_proj.0.weight"
# # # #     d_cond = int(state[d_cond_key].shape[0]) if d_cond_key in state else 256

# # # #     # dim_ff từ velocity.decoder.layers.0.linear1.weight
# # # #     dim_ff_key = "velocity.decoder.layers.0.linear1.weight"
# # # #     dim_ff = int(state[dim_ff_key].shape[0]) if dim_ff_key in state else 512

# # # #     print(f"  v2 arch: d_model={d_model}, d_cond={d_cond}, "
# # # #           f"num_dec_layers={num_dec_layers}, dim_ff={dim_ff}")

# # # #     model = TCFlowMatching(
# # # #         pred_len=args.pred_len,
# # # #         obs_len=args.obs_len,
# # # #         d_model=d_model,
# # # #         d_cond=d_cond,
# # # #         num_dec_layers=num_dec_layers,
# # # #         dim_ff=dim_ff,
# # # #         sigma_min=0.04,
# # # #         sigma_max=0.08,
# # # #         use_ema=True,
# # # #         ema_decay=0.995,
# # # #     ).to(device)

# # # #     if hasattr(model, "init_ema"):
# # # #         model.init_ema()

# # # #     result     = model.load_state_dict(state, strict=False)
# # # #     missing    = [k for k in result.missing_keys if "_ema" not in k]
# # # #     unexpected = result.unexpected_keys

# # # #     if missing:
# # # #         print(f"  Missing keys: {len(missing)}")
# # # #         for k in missing[:5]: print(f"    - {k}")
# # # #     if unexpected:
# # # #         print(f"  Unexpected keys: {len(unexpected)}")
# # # #         for k in list(unexpected)[:3]: print(f"    + {k}")

# # # #     # Load EMA — train_fm.py v2 lưu key "ema"
# # # #     ema_loaded = 0
# # # #     if isinstance(ckpt, dict):
# # # #         for ema_key in ["ema", "ema_shadow", "ema_state_dict"]:
# # # #             if ema_key in ckpt and ckpt[ema_key]:
# # # #                 ema = get_ema_obj(model)
# # # #                 if ema is not None and hasattr(ema, "shadow"):
# # # #                     for k, v in ckpt[ema_key].items():
# # # #                         if k in ema.shadow:
# # # #                             ema.shadow[k].copy_(v.to(device))
# # # #                             ema_loaded += 1
# # # #                     if ema_loaded:
# # # #                         print(f"  EMA: {ema_loaded} keys loaded ✓")
# # # #                 break

# # # #     if not ema_loaded:
# # # #         print("  ⚠ No EMA weights — using raw model weights")

# # # #     print(f"  ✅ FM v2 loaded (epoch={ep})")
# # # #     return model


# # # # def _load_fm_v1(ckpt, state, device, args, ep, TCFlowMatching):
# # # #     """
# # # #     Load TCFlowMatching v1 (net.*) — giữ nguyên logic cũ.
# # # #     """
# # # #     ffn_key = "net.transformer.layers.0.linear1.weight"
# # # #     ffn_dim = int(state[ffn_key].shape[0]) if ffn_key in state else 1024

# # # #     num_layers = sum(
# # # #         1 for i in range(10)
# # # #         if f"net.transformer.layers.{i}.linear1.weight" in state
# # # #     ) or 2

# # # #     has_det_head    = _detect_has_det_head(state)
# # # #     has_reg_head    = any(k.startswith("net.reg_head.") for k in state)
# # # #     has_vel_expand  = "net.vel_obs_expand.weight" in state
# # # #     has_speed_head  = any(k.startswith("net.speed_head.") for k in state)

# # # #     print(f"  v1 arch: FFN={ffn_dim}, layers={num_layers}, "
# # # #           f"det_head={has_det_head}, reg_head={has_reg_head}, "
# # # #           f"speed_head={has_speed_head}")

# # # #     model = TCFlowMatching(
# # # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # # #         sigma_min=0.02, use_ema=True, ema_decay=0.995,
# # # #     ).to(device)

# # # #     if hasattr(model, "init_ema"):
# # # #         model.init_ema()

# # # #     raw = get_raw_model(model)

# # # #     # Patch transformer nếu cần
# # # #     try:
# # # #         cur_ffn    = raw.net.transformer.layers[0].linear1.weight.shape[0]
# # # #         cur_layers = len(raw.net.transformer.layers)
# # # #     except Exception:
# # # #         cur_ffn, cur_layers = 1024, 2

# # # #     if cur_ffn != ffn_dim or cur_layers != num_layers:
# # # #         try:
# # # #             raw.net.transformer = nn.TransformerDecoder(
# # # #                 nn.TransformerDecoderLayer(
# # # #                     d_model=256, nhead=8, dim_feedforward=ffn_dim,
# # # #                     dropout=0.10, activation="gelu", batch_first=True),
# # # #                 num_layers=num_layers).to(device)
# # # #             print(f"  Patched transformer → {num_layers}L / {ffn_dim}FFN")
# # # #         except Exception as e:
# # # #             print(f"  ⚠ Transformer patch failed: {e}")

# # # #     if has_vel_expand and not hasattr(raw.net, "vel_obs_expand"):
# # # #         raw.net.vel_obs_expand = nn.Linear(256, 1024).to(device)
# # # #         raw.net.vel_obs_ln     = nn.LayerNorm(256).to(device)
# # # #         print("  Patched vel_obs_expand")

# # # #     result     = model.load_state_dict(state, strict=False)
# # # #     missing    = [k for k in result.missing_keys
# # # #                   if not k.startswith("net.det_head") and "_ema" not in k]
# # # #     unexpected = result.unexpected_keys

# # # #     if missing:
# # # #         print(f"  Missing keys (non-det_head): {len(missing)}")
# # # #         for k in missing[:3]: print(f"    - {k}")
# # # #     if unexpected:
# # # #         print(f"  Unexpected keys: {len(unexpected)}")
# # # #         for k in list(unexpected)[:3]: print(f"    + {k}")

# # # #     det_missing = [k for k in result.missing_keys if k.startswith("net.det_head")]
# # # #     if det_missing:
# # # #         print(f"  det_head: {len(det_missing)} keys missing "
# # # #               f"(old checkpoint, det blend disabled)")

# # # #     # Load EMA
# # # #     ema_loaded = 0
# # # #     if isinstance(ckpt, dict):
# # # #         for ema_key in ["ema_shadow", "ema_state_dict", "ema"]:
# # # #             if ema_key in ckpt and ckpt[ema_key]:
# # # #                 ema = get_ema_obj(model)
# # # #                 if ema is not None and hasattr(ema, "shadow"):
# # # #                     for k, v in ckpt[ema_key].items():
# # # #                         if k in ema.shadow:
# # # #                             ema.shadow[k].copy_(v.to(device))
# # # #                             ema_loaded += 1
# # # #                     if ema_loaded:
# # # #                         print(f"  EMA: {ema_loaded} keys loaded")
# # # #                 break

# # # #     if not ema_loaded:
# # # #         print("  ⚠ No EMA weights — using raw model weights")

# # # #     # Set _det_epoch
# # # #     raw_model = get_raw_model(model)
# # # #     if has_det_head:
# # # #         raw_model._det_epoch = 999
# # # #         print("  det_head: weights loaded → DET_BLEND=0.70 ✓")
# # # #     else:
# # # #         raw_model._det_epoch = 0
# # # #         print("  det_head: not in checkpoint → DET_BLEND=0.00 (pure FM)")

# # # #     print(f"  ✅ FM v1 loaded (epoch={ep})")
# # # #     return model


# # # # def _load_residual_model(ckpt, state, device, args, ep):
# # # #     """
# # # #     Load FMResidualCorrector từ checkpoint.
# # # #     """
# # # #     FMResidualCorrector = _import_residual_class()

# # # #     if FMResidualCorrector is not None:
# # # #         try:
# # # #             import tempfile
# # # #             sttrans_state = {k[len("sttrans."):]: v
# # # #                              for k, v in state.items()
# # # #                              if k.startswith("sttrans.")}

# # # #             with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tf:
# # # #                 tmp_path = tf.name
# # # #             torch.save({"model_state_dict": sttrans_state}, tmp_path)

# # # #             model = FMResidualCorrector(
# # # #                 sttrans_checkpoint=tmp_path,
# # # #                 obs_len=args.obs_len, pred_len=args.pred_len,
# # # #                 device=str(device),
# # # #             ).to(device)

# # # #             os.unlink(tmp_path)

# # # #             if hasattr(model, "init_ema"):
# # # #                 model.init_ema()

# # # #             result = model.load_state_dict(state, strict=False)
# # # #             n_miss = len(result.missing_keys)
# # # #             n_unex = len(result.unexpected_keys)
# # # #             if n_miss:  print(f"  Missing   : {n_miss}")
# # # #             if n_unex:  print(f"  Unexpected: {n_unex}")

# # # #             print(f"  ✅ FMResidualCorrector loaded (epoch={ep})")
# # # #             return model

# # # #         except Exception as e:
# # # #             print(f"  ⚠ FMResidualCorrector full load failed: {e}")
# # # #             print("  → Falling back to STTrans-only evaluation")

# # # #     # Fallback: load chỉ STTrans part
# # # #     print("  Loading sttrans.* weights as standalone STTrans...")
# # # #     try:
# # # #         from Model.st_trans_model import STTrans
# # # #         model = STTrans(obs_len=args.obs_len, pred_len=args.pred_len).to(device)

# # # #         sttrans_state = {k[len("sttrans."):]: v
# # # #                          for k, v in state.items()
# # # #                          if k.startswith("sttrans.")}

# # # #         result = model.load_state_dict(sttrans_state, strict=False)
# # # #         n_miss = len(result.missing_keys)
# # # #         if n_miss: print(f"  STTrans missing: {n_miss}")

# # # #         print(f"  ✅ STTrans base loaded from FMResidualCorrector ckpt (epoch={ep})")
# # # #         print("  ⚠  Note: evaluating STTrans base only (not the corrector)")
# # # #         return model

# # # #     except Exception as e2:
# # # #         print(f"  ❌ Fallback STTrans load also failed: {e2}")
# # # #         return None


# # # # def load_lstm_model(ckpt_path: str, device, args):
# # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # #         print(f"  ⚠  LSTM checkpoint not found: {ckpt_path}")
# # # #         return None
# # # #     try:
# # # #         from Model.paper_baseline_model import PaperBaseline
# # # #         lstm_type = getattr(args, "lstm_type", "lstm")
# # # #         model = PaperBaseline(model_type=lstm_type,
# # # #                               pred_len=args.pred_len, obs_len=args.obs_len).to(device)
# # # #         ckpt  = torch.load(ckpt_path, map_location=device)
# # # #         state = _load_state_dict(ckpt)
# # # #         res   = model.load_state_dict(state, strict=False)
# # # #         if res.missing_keys:    print(f"  Missing   : {len(res.missing_keys)}")
# # # #         if res.unexpected_keys: print(f"  Unexpected: {len(res.unexpected_keys)}")
# # # #         ep = ckpt.get("epoch","?") if isinstance(ckpt,dict) else "?"
# # # #         print(f"  ✅ {lstm_type.upper()} loaded (epoch={ep})")
# # # #         return model
# # # #     except Exception as e:
# # # #         print(f"  ❌ LSTM load failed: {e}")
# # # #         import traceback; traceback.print_exc()
# # # #         return None

# # # # def load_fm_v2_compat(ckpt, state, device, args, ep, TCFlowMatching):
# # # #     """
# # # #     Load TCFlowMatching v2 với backward compat:
# # # #     - vel_obs_enc [256, 48] (6 features) → expand thành [256, 56] (7 features, FIX-1)
# # # #     """
# # # #     print(f"  Architecture: v2 (ContextEncoder + VelocityTransformer)")
 
# # # #     # Detect hyperparams
# # # #     num_dec_layers = sum(
# # # #         1 for i in range(10)
# # # #         if f"velocity.decoder.layers.{i}.norm1.weight" in state
# # # #     ) or 4
# # # #     d_model  = int(state["velocity.traj_embed.weight"].shape[0]) \
# # # #                if "velocity.traj_embed.weight" in state else 256
# # # #     d_cond   = int(state["velocity.cond_proj.0.weight"].shape[0]) \
# # # #                if "velocity.cond_proj.0.weight" in state else 256
# # # #     dim_ff   = int(state["velocity.decoder.layers.0.linear1.weight"].shape[0]) \
# # # #                if "velocity.decoder.layers.0.linear1.weight" in state else 512
 
# # # #     print(f"  v2 arch: d_model={d_model}, d_cond={d_cond}, "
# # # #           f"num_dec_layers={num_dec_layers}, dim_ff={dim_ff}")
 
# # # #     model = TCFlowMatching(
# # # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # # #         d_model=d_model, d_cond=d_cond,
# # # #         num_dec_layers=num_dec_layers, dim_ff=dim_ff,
# # # #         sigma_min=0.04, sigma_max=0.08,
# # # #         use_ema=True, ema_decay=0.995,
# # # #     ).to(device)
 
# # # #     if hasattr(model, "init_ema"):
# # # #         model.init_ema()
 
# # # #     # ── VEL_OBS_ENC COMPAT: expand 48→56 nếu cần ──────────────────────
# # # #     key = "encoder.vel_obs_enc.0.weight"
# # # #     if key in state:
# # # #         w_old = state[key]          # [256, 48] hoặc [256, 56]
# # # #         obs_len = args.obs_len      # =8
# # # #         expected = obs_len * 7      # =56
 
# # # #         if w_old.shape[1] == obs_len * 6:
# # # #             # Expand: thêm obs_len=8 columns cho log_speed feature
# # # #             extra = torch.randn(w_old.shape[0], obs_len,
# # # #                                 device=w_old.device) * 0.01
# # # #             state[key] = torch.cat([w_old, extra], dim=1)
# # # #             print(f"  ✅ Expanded vel_obs_enc: {w_old.shape[1]} → {expected} cols")
# # # #         elif w_old.shape[1] == expected:
# # # #             print(f"  ✅ vel_obs_enc already {expected} cols")
# # # #         else:
# # # #             print(f"  ⚠ vel_obs_enc unexpected shape {w_old.shape}")
 
# # # #     result     = model.load_state_dict(state, strict=False)
# # # #     missing    = [k for k in result.missing_keys if "_ema" not in k]
# # # #     unexpected = result.unexpected_keys
# # # #     if missing:    print(f"  Missing   : {len(missing)}")
# # # #     if unexpected: print(f"  Unexpected: {len(unexpected)}")
 
# # # #     # Load EMA
# # # #     ema_loaded = 0
# # # #     if isinstance(ckpt, dict):
# # # #         for ema_key in ["ema", "ema_shadow", "ema_state_dict"]:
# # # #             if ema_key in ckpt and ckpt[ema_key]:
# # # #                 raw = model._orig_mod if hasattr(model, "_orig_mod") else model
# # # #                 ema = getattr(raw, "_ema", None)
# # # #                 if ema is not None and hasattr(ema, "shadow"):
# # # #                     for k, v in ckpt[ema_key].items():
# # # #                         if k in ema.shadow:
# # # #                             ema.shadow[k].copy_(v.to(device))
# # # #                             ema_loaded += 1
# # # #                     if ema_loaded:
# # # #                         print(f"  EMA: {ema_loaded} keys loaded ✓")
# # # #                 break
 
# # # #     if not ema_loaded:
# # # #         print("  ⚠ No EMA weights — using raw model weights")
 
# # # #     print(f"  ✅ FM v2 loaded (epoch={ep})")
# # # #     return model
 
# # # # def load_sttrans_model(ckpt_path: str, device, args):
# # # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # # #         print(f"  ⚠  ST-Trans checkpoint not found: {ckpt_path}")
# # # #         return None
# # # #     try:
# # # #         ckpt  = torch.load(ckpt_path, map_location=device)
# # # #         state = _load_state_dict(ckpt)
# # # #         ep    = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"

# # # #         # Auto-detect STTransV2 vs STTrans gốc
# # # #         is_v2 = any(k.startswith("steering_gate.") or
# # # #                     k.startswith("recurv_head.") or
# # # #                     k.startswith("uw.")
# # # #                     for k in state.keys())

# # # #         if is_v2 or getattr(args, "sttrans_type", "") == "sttrans_v2":
# # # #             from Model.st_trans_model import STTransV2

# # # #             # ── Auto-detect hyperparams từ checkpoint ──
# # # #             # d_model từ ctx_proj.0.weight shape[0]
# # # #             d_model = int(state["ctx_proj.0.weight"].shape[0]) \
# # # #                       if "ctx_proj.0.weight" in state else 64

# # # #             # nhead: d_model / head_dim, head_dim thường = 32 hoặc 64
# # # #             # in_proj_weight shape = [3*d_model, d_model] → nhead từ attention
# # # #             attn_key = "obs_enc.enc.layers.0.self_attn.in_proj_weight"
# # # #             if attn_key in state:
# # # #                 # nhead * head_dim = d_model, head_dim thường 32
# # # #                 nhead = max(1, d_model // 32)
# # # #             else:
# # # #                 nhead = 4

# # # #             # num_dec_layers: đếm transformer_dec.layers
# # # #             num_dec_layers = sum(
# # # #                 1 for i in range(10)
# # # #                 if f"transformer_dec.layers.{i}.norm1.weight" in state
# # # #             ) or 3

# # # #             # dim_ff từ transformer_dec.layers.0.linear1.weight shape[0]
# # # #             dim_ff_key = "transformer_dec.layers.0.linear1.weight"
# # # #             dim_ff = int(state[dim_ff_key].shape[0]) \
# # # #                      if dim_ff_key in state else 512

# # # #             # obs_enc feat_dim: obs_enc.proj.0.weight shape[1]
# # # #             obs_feat_key = "obs_enc.proj.0.weight"
# # # #             obs_feat_dim = int(state[obs_feat_key].shape[1]) \
# # # #                            if obs_feat_key in state else 8

# # # #             # gate_hidden từ steering_gate.gate_net.0.weight shape[0]
# # # #             gate_key = "steering_gate.gate_net.0.weight"
# # # #             gate_hidden = int(state[gate_key].shape[0]) \
# # # #                           if gate_key in state else 32

# # # #             # recurv_hidden từ recurv_head.net.0.weight shape[0]
# # # #             recurv_key = "recurv_head.net.0.weight"
# # # #             recurv_hidden = int(state[recurv_key].shape[0]) \
# # # #                             if recurv_key in state else 64

# # # #             # threshold từ checkpoint nếu có
# # # #             threshold_curv = ckpt.get("threshold_curv", 15.0) \
# # # #                              if isinstance(ckpt, dict) else 15.0
# # # #             threshold_spd  = ckpt.get("threshold_spd",  0.5) \
# # # #                              if isinstance(ckpt, dict) else 0.5

# # # #             print(f"  STTransV2 arch detected: d_model={d_model}, nhead={nhead}, "
# # # #                   f"num_dec_layers={num_dec_layers}, dim_ff={dim_ff}, "
# # # #                   f"obs_feat={obs_feat_dim}, gate_hidden={gate_hidden}, "
# # # #                   f"recurv_hidden={recurv_hidden}")
# # # #             print(f"  Thresholds: curv={threshold_curv:.3f}, spd={threshold_spd:.4f}")

# # # #             model = STTransV2(
# # # #                 obs_len        = args.obs_len,
# # # #                 pred_len       = args.pred_len,
# # # #                 d_model        = d_model,
# # # #                 nhead          = nhead,
# # # #                 num_dec_layers = num_dec_layers,
# # # #                 dim_ff         = dim_ff,
# # # #                 gate_hidden    = gate_hidden,
# # # #                 recurv_hidden  = recurv_hidden,
# # # #                 threshold_curv = threshold_curv,
# # # #                 threshold_spd  = threshold_spd,
# # # #             ).to(device)

# # # #             res = model.load_state_dict(state, strict=False)
# # # #             missing    = [k for k in res.missing_keys]
# # # #             unexpected = res.unexpected_keys
# # # #             if missing:    print(f"  Missing   : {len(missing)}")
# # # #             if unexpected: print(f"  Unexpected: {len(unexpected)}")
# # # #             print(f"  ✅ STTransV2 loaded (epoch={ep})")
# # # #             return model

# # # #         else:
# # # #             sttrans_type = getattr(args, "sttrans_type", "sttrans")
# # # #             if sttrans_type == "sttrans_ar":
# # # #                 from Model.st_trans_model import STTransAR as STTransModel
# # # #             else:
# # # #                 from Model.st_trans_model import STTrans as STTransModel
# # # #             model = STTransModel(obs_len=args.obs_len,
# # # #                                  pred_len=args.pred_len).to(device)
# # # #             res = model.load_state_dict(state, strict=False)
# # # #             if res.missing_keys:    print(f"  Missing   : {len(res.missing_keys)}")
# # # #             if res.unexpected_keys: print(f"  Unexpected: {len(res.unexpected_keys)}")
# # # #             tag = "STTransAR" if sttrans_type == "sttrans_ar" else "STTrans"
# # # #             print(f"  ✅ {tag} loaded (epoch={ep})")
# # # #             return model

# # # #     except Exception as e:
# # # #         print(f"  ❌ ST-Trans load failed: {e}")
# # # #         import traceback; traceback.print_exc()
# # # #         return None


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  Inference wrappers
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # @torch.no_grad()
# # # # def infer_fm(model, batch_list, args, device):
# # # #     try:
# # # #         model.eval()

# # # #         if hasattr(model, "sample"):
# # # #             ema = get_ema_obj(model); backup = None
# # # #             if ema is not None:
# # # #                 try: backup = ema.apply_to(model)
# # # #                 except Exception: pass

# # # #             try:
# # # #                 pred, _, _ = model.sample(
# # # #                     batch_list,
# # # #                     num_ensemble=args.fm_ensemble,
# # # #                     ddim_steps=args.fm_ode_steps,
# # # #                     importance_weight=True)
# # # #             except TypeError:
# # # #                 # v2 không có importance_weight
# # # #                 pred, _, _ = model.sample(
# # # #                     batch_list,
# # # #                     num_ensemble=args.fm_ensemble,
# # # #                     ddim_steps=args.fm_ode_steps)

# # # #             if backup is not None:
# # # #                 try: ema.restore(model, backup)
# # # #                 except Exception: pass

# # # #         else:
# # # #             pred = model(batch_list)
# # # #             if pred.dim() == 3 and pred.shape[0] == batch_list[0].shape[1]:
# # # #                 pred = pred.permute(1, 0, 2)

# # # #         return pred[..., :2] if pred.shape[-1] > 2 else pred

# # # #     except Exception as e:
# # # #         print(f"    FM inference error: {e}")
# # # #         import traceback; traceback.print_exc()
# # # #         return None


# # # # @torch.no_grad()
# # # # def infer_lstm(model, batch_list, args, device):
# # # #     try:
# # # #         model.eval()
# # # #         if hasattr(model, "sample"):
# # # #             pred, _, _ = model.sample(batch_list, num_ensemble=1)
# # # #         else:
# # # #             pred = model(batch_list)
# # # #         if pred.dim() == 3 and pred.shape[0] == batch_list[0].shape[1]:
# # # #             pred = pred.permute(1, 0, 2)
# # # #         return pred[..., :2]
# # # #     except Exception as e:
# # # #         print(f"    LSTM inference error: {e}")
# # # #         return None


# # # # @torch.no_grad()
# # # # def infer_sttrans(model, batch_list, args, device):
# # # #     try:
# # # #         model.eval()
# # # #         if hasattr(model, "sample"):
# # # #             pred, _, _ = model.sample(batch_list, num_ensemble=1)
# # # #         else:
# # # #             pred = model(batch_list)
# # # #         if pred.dim() == 3 and pred.shape[0] == batch_list[0].shape[1]:
# # # #             pred = pred.permute(1, 0, 2)
# # # #         return pred[..., :2]
# # # #     except Exception as e:
# # # #         print(f"    STTrans inference error: {e}")
# # # #         return None


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  CSV + pretty print
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # _PER_STORM_FIELDS = ["model","storm_id","seq_idx","lead_h",
# # # #                      "lon_pred","lat_pred","lon_true","lat_true",
# # # #                      "dist_km","ate_km","cte_km"]

# # # # _SUMMARY_FIELDS = ["model","storm_id","n_seq","ADE","ATE","CTE",
# # # #                    "12h_dist","24h_dist","48h_dist","72h_dist",
# # # #                    "12h_ate","24h_ate","48h_ate","72h_ate",
# # # #                    "12h_cte","24h_cte","48h_cte","72h_cte",
# # # #                    "beat_12h","beat_24h","beat_48h","beat_72h",
# # # #                    "beat_ADE","beat_ATE","beat_CTE"]


# # # # def write_per_storm_csv(acc, out_dir):
# # # #     out = os.path.join(out_dir, f"{acc.model_name}_per_storm.csv")
# # # #     with open(out, "w", newline="") as fh:
# # # #         w = csv.DictWriter(fh, fieldnames=_PER_STORM_FIELDS)
# # # #         w.writeheader()
# # # #         for sr in acc.storms.values():
# # # #             for row in sr.rows:
# # # #                 w.writerow({"model": acc.model_name, **row,
# # # #                             "lon_pred": f"{row['lon_pred']:.4f}",
# # # #                             "lat_pred": f"{row['lat_pred']:.4f}",
# # # #                             "lon_true": f"{row['lon_true']:.4f}",
# # # #                             "lat_true": f"{row['lat_true']:.4f}"})
# # # #     print(f"  💾 {out}")
# # # #     return out


# # # # def write_summary_csv(all_acc, out_dir):
# # # #     out = os.path.join(out_dir, "summary_all_models.csv")
# # # #     with open(out, "w", newline="") as fh:
# # # #         w = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
# # # #         w.writeheader()
# # # #         def _b(v,t): return "YES" if (np.isfinite(v) and v<t) else "NO"
# # # #         def _f(v):   return f"{v:.2f}" if np.isfinite(v) else "nan"
# # # #         for mname, acc in all_acc.items():
# # # #             for r in acc.all_summaries() + [acc.overall_summary()]:
# # # #                 w.writerow({
# # # #                     "model": mname, "storm_id": r["storm_id"],
# # # #                     "n_seq": r["n_seq"],
# # # #                     "ADE": _f(r["ADE"]), "ATE": _f(r["ATE"]), "CTE": _f(r["CTE"]),
# # # #                     "12h_dist": _f(r["12h_dist"]), "24h_dist": _f(r["24h_dist"]),
# # # #                     "48h_dist": _f(r["48h_dist"]), "72h_dist": _f(r["72h_dist"]),
# # # #                     "12h_ate": _f(r["12h_ate"]),   "24h_ate": _f(r["24h_ate"]),
# # # #                     "48h_ate": _f(r["48h_ate"]),   "72h_ate": _f(r["72h_ate"]),
# # # #                     "12h_cte": _f(r["12h_cte"]),   "24h_cte": _f(r["24h_cte"]),
# # # #                     "48h_cte": _f(r["48h_cte"]),   "72h_cte": _f(r["72h_cte"]),
# # # #                     "beat_12h": _b(r["12h_dist"],TARGETS["12h"]),
# # # #                     "beat_24h": _b(r["24h_dist"],TARGETS["24h"]),
# # # #                     "beat_48h": _b(r["48h_dist"],TARGETS["48h"]),
# # # #                     "beat_72h": _b(r["72h_dist"],TARGETS["72h"]),
# # # #                     "beat_ADE": _b(r["ADE"],TARGETS["ADE"]),
# # # #                     "beat_ATE": _b(r["ATE"],TARGETS["ATE"]),
# # # #                     "beat_CTE": _b(r["CTE"],TARGETS["CTE"]),
# # # #                 })
# # # #     print(f"  💾 Summary: {out}")
# # # #     return out


# # # # def print_model_results(acc):
# # # #     cfg = MODEL_CONFIGS.get(acc.model_name, {}); col = cfg.get("color","⚪")
# # # #     ref = cfg.get("ref","")
# # # #     print(f"\n{'═'*110}")
# # # #     print(f"  {col}  {acc.model_name}  |  {ref}")
# # # #     print(f"{'═'*110}")
# # # #     print(f"  {'Storm':<16} {'N':>4}  {'ADE':>8}  {'ATE':>8}  {'CTE':>8}  "
# # # #           f"║  {'12h':>7}  {'24h':>7}  {'48h':>7}  {'72h':>7}")
# # # #     print("  " + "─"*106)

# # # #     def v(x, t=None, w=8, dec=1):
# # # #         if not np.isfinite(x): return f"{'nan':>{w+2}}"
# # # #         s = f"{x:>{w}.{dec}f}"
# # # #         s += ("✅" if x<t else "❌") if t is not None else "  "
# # # #         return s

# # # #     for r in acc.all_summaries():
# # # #         print(f"  🌀 {r['storm_id']:<14} {r['n_seq']:>4}  "
# # # #               f"{v(r['ADE'],TARGETS['ADE'])}  {v(r['ATE'],TARGETS['ATE'])}  "
# # # #               f"{v(r['CTE'],TARGETS['CTE'])}  "
# # # #               f"║  {v(r['12h_dist'],TARGETS['12h'],7)}  "
# # # #               f"{v(r['24h_dist'],TARGETS['24h'],7)}  "
# # # #               f"{v(r['48h_dist'],TARGETS['48h'],7)}  "
# # # #               f"{v(r['72h_dist'],TARGETS['72h'],7)}")

# # # #     ov = acc.overall_summary()
# # # #     print("  " + "─"*106)
# # # #     print(f"  📊 {'OVERALL':<14} {ov['n_seq']:>4}  "
# # # #           f"{v(ov['ADE'],TARGETS['ADE'])}  {v(ov['ATE'],TARGETS['ATE'])}  "
# # # #           f"{v(ov['CTE'],TARGETS['CTE'])}  "
# # # #           f"║  {v(ov['12h_dist'],TARGETS['12h'],7)}  "
# # # #           f"{v(ov['24h_dist'],TARGETS['24h'],7)}  "
# # # #           f"{v(ov['48h_dist'],TARGETS['48h'],7)}  "
# # # #           f"{v(ov['72h_dist'],TARGETS['72h'],7)}")
# # # #     print(f"{'═'*110}\n")


# # # # def print_comparison_table(all_acc):
# # # #     print(f"\n{'═'*95}")
# # # #     print(f"  📊  MODEL COMPARISON  —  Overall Test Set")
# # # #     print(f"{'═'*95}")
# # # #     print(f"  {'Model':<14}  {'ADE':>9}  {'ATE':>9}  {'CTE':>9}  "
# # # #           f"║  {'12h':>8}  {'24h':>8}  {'48h':>8}  {'72h':>8}")
# # # #     print("  " + "─"*91)

# # # #     def vv(x, t=None, w=9, dec=1):
# # # #         if not np.isfinite(x): return f"{'nan':>{w+2}}"
# # # #         s = f"{x:>{w}.{dec}f}"
# # # #         s += ("✅" if x<t else "❌") if t is not None else "  "
# # # #         return s

# # # #     for mname, acc in all_acc.items():
# # # #         col = MODEL_CONFIGS.get(mname, {}).get("color","⚪")
# # # #         ov  = acc.overall_summary()
# # # #         print(f"  {col} {mname:<12}  "
# # # #               f"{vv(ov['ADE'],TARGETS['ADE'])}  {vv(ov['ATE'],TARGETS['ATE'])}  "
# # # #               f"{vv(ov['CTE'],TARGETS['CTE'])}  "
# # # #               f"║  {vv(ov['12h_dist'],TARGETS['12h'],8)}  "
# # # #               f"{vv(ov['24h_dist'],TARGETS['24h'],8)}  "
# # # #               f"{vv(ov['48h_dist'],TARGETS['48h'],8)}  "
# # # #               f"{vv(ov['72h_dist'],TARGETS['72h'],8)}")

# # # #     print("  " + "─"*91)
# # # #     print(f"  Targets: ADE<{TARGETS['ADE']} | ATE<{TARGETS['ATE']} | "
# # # #           f"CTE<{TARGETS['CTE']} | 12h<{TARGETS['12h']} | 24h<{TARGETS['24h']} | "
# # # #           f"48h<{TARGETS['48h']} | 72h<{TARGETS['72h']} (km)")
# # # #     print(f"{'═'*95}\n")


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  Main evaluation loop
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # def run_evaluation(model, model_name, test_loader, device, args, infer_fn):
# # # #     acc = MultiStormAccumulator(model_name)
# # # #     n   = len(test_loader); t0 = time.perf_counter()
# # # #     print(f"\n  Running {model_name} on {n} batches...")

# # # #     for i, batch in enumerate(test_loader):
# # # #         bl        = move_to_device(list(batch), device)
# # # #         B         = bl[0].shape[1]
# # # #         pred_norm = infer_fn(model, bl, args, device)
# # # #         if pred_norm is None: continue

# # # #         gt_norm   = bl[1]
# # # #         storm_ids = extract_storm_ids(batch, B)
# # # #         T         = min(pred_norm.shape[0], gt_norm.shape[0])
# # # #         acc.add_batch(pred_norm[:T], gt_norm[:T], storm_ids)

# # # #         if (i+1) % max(1, n//5) == 0 or i == n-1:
# # # #             ov = acc.overall_summary()
# # # #             print(f"    [{i+1:>4}/{n}]  ADE={ov['ADE']:.1f}km  "
# # # #                   f"storms={len(acc.storms)}  "
# # # #                   f"elapsed={time.perf_counter()-t0:.1f}s")

# # # #     ov = acc.overall_summary()
# # # #     print(f"  ✅ {model_name}: ADE={ov['ADE']:.1f}  "
# # # #           f"ATE={ov['ATE']:.1f}  CTE={ov['CTE']:.1f} km")
# # # #     return acc


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  Args + Main
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # def get_args():
# # # #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # #     p.add_argument("--dataset_root",  default="TCND_vn")
# # # #     p.add_argument("--obs_len",       default=8,   type=int)
# # # #     p.add_argument("--pred_len",      default=12,  type=int)
# # # #     p.add_argument("--batch_size",    default=16,  type=int)
# # # #     p.add_argument("--num_workers",   default=2,   type=int)
# # # #     p.add_argument("--skip",          default=1,   type=int)
# # # #     p.add_argument("--min_ped",       default=1,   type=int)
# # # #     p.add_argument("--threshold",     default=0.002, type=float)
# # # #     p.add_argument("--other_modal",   default="gph")
# # # #     p.add_argument("--test_year",     default=None, type=int)
# # # #     p.add_argument("--delim",         default=" ")
# # # #     p.add_argument("--fm_ckpt",       default=None)
# # # #     p.add_argument("--lstm_ckpt",     default=None)
# # # #     p.add_argument("--sttrans_ckpt",  default=None)
# # # #     p.add_argument("--lstm_type",     default="lstm", choices=["lstm","gru","rnn"])
# # # #     p.add_argument("--sttrans_type",  default="sttrans", choices=["sttrans","sttrans_ar"])
# # # #     p.add_argument("--fm_ensemble",   default=30, type=int)
# # # #     p.add_argument("--fm_ode_steps",  default=20, type=int)
# # # #     p.add_argument("--fm_type",       default="auto",
# # # #                    choices=["auto","tcfm","residual"],
# # # #                    help="Force FM model type: auto|tcfm|residual")
# # # #     p.add_argument("--output_dir",    default="results/test_eval")
# # # #     p.add_argument("--gpu_num",       default="0")
# # # #     p.add_argument("--eval_models",   default="all",
# # # #                    help="'all' hoặc csv: 'fm,lstm,sttrans'")
# # # #     p.add_argument("--no_detail_csv", action="store_true")
# # # #     return p.parse_args()


# # # # def _version_from_path(p):
# # # #     parts = os.path.normpath(p).split(os.sep)
# # # #     for part in reversed(parts[:-1]):
# # # #         if part.startswith("v") and part[1:].isdigit():
# # # #             return part
# # # #     return "ours"


# # # # def main(args):
# # # #     if torch.cuda.is_available():
# # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # #     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# # # #     print("="*80)
# # # #     print(f"  TC Model Evaluator  (fix v3 — v2 model support)")
# # # #     print(f"  Started : {ts}")
# # # #     print(f"  Device  : {device}")
# # # #     print(f"  Output  : {args.output_dir}")
# # # #     print("="*80)

# # # #     eval_set = ({"fm","lstm","sttrans"} if args.eval_models=="all"
# # # #                 else {m.strip().lower() for m in args.eval_models.split(",")})

# # # #     print("\n  Loading test dataset...")
# # # #     try:
# # # #         from Model.data.loader_training import data_loader
# # # #         _, test_loader = data_loader(
# # # #             args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # #         print(f"  ✅ {len(test_loader)} batches")
# # # #     except Exception as e:
# # # #         print(f"  ❌ Test data load failed: {e}")
# # # #         import traceback; traceback.print_exc()
# # # #         return

# # # #     all_acc = {}

# # # #     if "fm" in eval_set and args.fm_ckpt:
# # # #         print(f"\n{'─'*50}")
# # # #         print("  Loading FlowMatching...")
# # # #         model = load_fm_model(args.fm_ckpt, device, args)
# # # #         if model is not None:
# # # #             tag = "FM_" + _version_from_path(args.fm_ckpt)
# # # #             all_acc[tag] = run_evaluation(model, tag, test_loader,
# # # #                                           device, args, infer_fm)
# # # #             del model
# # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # #     if "lstm" in eval_set and args.lstm_ckpt:
# # # #         print(f"\n{'─'*50}")
# # # #         model = load_lstm_model(args.lstm_ckpt, device, args)
# # # #         if model is not None:
# # # #             tag = args.lstm_type.upper()
# # # #             all_acc[tag] = run_evaluation(model, tag, test_loader,
# # # #                                           device, args, infer_lstm)
# # # #             del model
# # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # #     if "sttrans" in eval_set and args.sttrans_ckpt:
# # # #         print(f"\n{'─'*50}")
# # # #         model = load_sttrans_model(args.sttrans_ckpt, device, args)
# # # #         if model is not None:
# # # #             tag = "STTransAR" if args.sttrans_type=="sttrans_ar" else "STTrans"
# # # #             all_acc[tag] = run_evaluation(model, tag, test_loader,
# # # #                                           device, args, infer_sttrans)
# # # #             del model
# # # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # # #     if not all_acc:
# # # #         print("\n  ❌ No models evaluated.")
# # # #         print("     Specify: --fm_ckpt / --lstm_ckpt / --sttrans_ckpt")
# # # #         return

# # # #     for acc in all_acc.values():
# # # #         print_model_results(acc)
# # # #     if len(all_acc) > 1:
# # # #         print_comparison_table(all_acc)

# # # #     print("\n  Writing CSVs...")
# # # #     csv_paths = []
# # # #     for acc in all_acc.values():
# # # #         csv_paths.append(write_per_storm_csv(acc, args.output_dir))
# # # #     csv_paths.append(write_summary_csv(all_acc, args.output_dir))

# # # #     metric_map = [("ADE","ADE"),("ATE","ATE"),("CTE","CTE"),
# # # #                   ("12h_dist","12h"),("24h_dist","24h"),
# # # #                   ("48h_dist","48h"),("72h_dist","72h")]
# # # #     print(f"\n{'═'*80}")
# # # #     print("  FINAL BEAT SUMMARY")
# # # #     print(f"{'─'*80}")
# # # #     for mname, acc in all_acc.items():
# # # #         ov  = acc.overall_summary()
# # # #         col = MODEL_CONFIGS.get(mname, {}).get("color","⚪")
# # # #         beats = [
# # # #             f"{label}={ov.get(k,float('nan')):.1f}✅"
# # # #             for k, label in metric_map
# # # #             if np.isfinite(ov.get(k,float("nan")))
# # # #             and ov[k] < TARGETS.get(label, float("inf"))
# # # #         ]
# # # #         print(f"  {col} {mname:<12}: " +
# # # #               (" | ".join(beats) if beats else "no targets beaten"))
# # # #     print(f"{'═'*80}\n")


# # # # if __name__ == "__main__":
# # # #     np.random.seed(42); torch.manual_seed(42)
# # # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# # # #     args = get_args()

# # # #     _kaggle_base = "/kaggle/working/TC_FM"
# # # #     if args.dataset_root == "TCND_vn" and os.path.isdir(_kaggle_base):
# # # #         args.dataset_root = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"

# # # #     _runs = os.path.join(_kaggle_base,"runs") if os.path.isdir(_kaggle_base) else "runs"

# # # #     if args.fm_ckpt is None:
# # # #         for c in ([os.path.join(_runs,f"v{v}","best_ema.pth") for v in ["67","66","65","61","60","59","58"]] +
# # # #                   [os.path.join(_runs,f"v{v}","best_model.pth") for v in ["67","61","60","59"]]):
# # # #             if os.path.exists(c):
# # # #                 args.fm_ckpt = c
# # # #                 print(f"  Auto FM ckpt: {c}")
# # # #                 break

# # # #     if args.lstm_ckpt is None:
# # # #         for c in [os.path.join(_runs,d,"best_model.pth")
# # # #                   for d in ["lstm","LSTM","gru","GRU","paper_baseline"]]:
# # # #             if os.path.exists(c):
# # # #                 args.lstm_ckpt = c; print(f"  Auto LSTM ckpt: {c}"); break

# # # #     if args.sttrans_ckpt is None:
# # # #         for c in [os.path.join(_runs,d,"best_model.pth")
# # # #                   for d in ["sttrans","STTrans","st_trans"]]:
# # # #             if os.path.exists(c):
# # # #                 args.sttrans_ckpt = c; print(f"  Auto STTrans ckpt: {c}"); break

# # # #     main(args)

# # # """
# # # scripts/evaluate_test_storms.py  — TC Model Evaluator v2.4-compat
# # # ═══════════════════════════════════════════════════════════════════

# # # BUGS ĐÃ FIX (bản này):
# # #   BUG-1+5: load_fm_v2_compat là dead code → fix: gọi đúng trong load_fm_model
# # #   BUG-2:   infer_fm dùng importance_weight=True → TypeError → fix: bỏ kwarg đó
# # #   BUG-4:   _version_from_path dùng r'\d+' không escape → SyntaxWarning → fix raw string
# # #   BUG-EMA: EMA shadow còn vel_obs_enc shape 48 → crash khi copy vào model 56
# # #            → fix: expand EMA shadow cùng logic với state_dict trước khi copy_()
# # #   BUG-F:   infer_fm unpack cứng 3 giá trị → crash khi sample() trả 4 (xai path)
# # #            → fix: lấy result[0]
# # #   BUG-G:   infer_fm permute sai chiều v2.4 — sample() luôn trả [T,B,2]
# # #            → fix: bỏ permute cho FM path
# # #   BUG-H:   gt_norm có thể có extra features → fix: slice [...,:2]

# # # Cách dùng:
# # #     python scripts/evaluate_test_storms.py \
# # #         --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# # #         --fm_ckpt      /kaggle/.../best_model_v24.pth \
# # #         --output_dir   results/test_eval \
# # #         --eval_models  fm \
# # #         --fm_ensemble  20 \
# # #         --fm_ode_steps 1 \
# # #         --gpu_num 0
# # # """
# # # from __future__ import annotations

# # # import sys
# # # import os
# # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # import argparse
# # # import csv
# # # import re
# # # import time
# # # import importlib
# # # import inspect
# # # from collections import defaultdict, Counter
# # # from datetime import datetime
# # # from typing import Dict, List, Optional, Tuple

# # # import numpy as np
# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # # from torch.utils.data import DataLoader

# # # R_EARTH  = 6371.0
# # # DT_HOURS = 6.0

# # # LEAD_HOURS = [12, 24, 36, 48, 60, 72]
# # # LEAD_STEPS = {h: h // 6 - 1 for h in LEAD_HOURS}

# # # TARGETS = {
# # #     "12h": 50.0, "24h": 100.0, "48h": 200.0, "72h": 297.0,
# # #     "ADE": 136.41, "ATE": 79.94, "CTE": 93.58,
# # # }

# # # MODEL_CONFIGS = {
# # #     "LSTM"     : {"color": "🔵", "ref": "Rahman 2025"},
# # #     "GRU"      : {"color": "🔵", "ref": "Rahman 2025"},
# # #     "RNN"      : {"color": "🔵", "ref": "Rahman 2025"},
# # #     "STTrans"  : {"color": "🟡", "ref": "Faiaz 2026"},
# # #     "STTransAR": {"color": "🟡", "ref": "Faiaz 2026 AR"},
# # #     "FM_ours"  : {"color": "🔴", "ref": "Ours v2.4"},
# # # }


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Coordinate utilities
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def norm_to_deg(arr):
# # #     out = arr.clone()
# # #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# # #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# # #     return out


# # # def haversine_km(p1, p2):
# # #     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
# # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # #     a = (torch.sin(dlat/2).pow(2) +
# # #          torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2))
# # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())


# # # def compute_ate_cte(pred_deg, gt_deg):
# # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # #     if T < 2:
# # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # #         return z, z
# # #     lon1 = torch.deg2rad(gt_deg[:T-1,:,0]); lat1 = torch.deg2rad(gt_deg[:T-1,:,1])
# # #     lon2 = torch.deg2rad(gt_deg[1:T, :,0]); lat2 = torch.deg2rad(gt_deg[1:T, :,1])
# # #     dlon_a = lon2 - lon1
# # #     y_a = torch.sin(dlon_a)*torch.cos(lat2)
# # #     x_a = torch.cos(lat1)*torch.sin(lat2) - torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon_a)
# # #     bear_along = torch.atan2(y_a, x_a)
# # #     lon3 = torch.deg2rad(pred_deg[1:T,:,0]); lat3 = torch.deg2rad(pred_deg[1:T,:,1])
# # #     dlon_e = lon3 - lon2
# # #     y_e = torch.sin(dlon_e)*torch.cos(lat3)
# # #     x_e = torch.cos(lat2)*torch.sin(lat3) - torch.sin(lat2)*torch.cos(lat3)*torch.cos(dlon_e)
# # #     bear_err = torch.atan2(y_e, x_e)
# # #     total = haversine_km(pred_deg[1:T], gt_deg[1:T])
# # #     angle = bear_err - bear_along
# # #     return total*torch.cos(angle), total*torch.sin(angle)


# # # def move_to_device(batch, device):
# # #     out = list(batch)
# # #     for i, x in enumerate(out):
# # #         if torch.is_tensor(x):
# # #             out[i] = x.to(device)
# # #         elif isinstance(x, dict):
# # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k,v in x.items()}
# # #     return out


# # # def get_raw_model(model):
# # #     return model._orig_mod if hasattr(model, "_orig_mod") else model


# # # def get_ema_obj(model):
# # #     raw = get_raw_model(model)
# # #     if hasattr(raw, "_ema") and raw._ema is not None:
# # #         return raw._ema
# # #     return None


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Per-storm accumulator
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # class StormResult:
# # #     def __init__(self, storm_id):
# # #         self.storm_id = storm_id
# # #         self.rows = []
# # #         self.dist_by_lead = defaultdict(list)
# # #         self.ate_by_lead  = defaultdict(list)
# # #         self.cte_by_lead  = defaultdict(list)
# # #         self.all_dist = []; self.all_ate = []; self.all_cte = []
# # #         self.n_seq = 0

# # #     def add_batch(self, pred_deg, gt_deg, seq_indices=None):
# # #         T = min(pred_deg.shape[0], gt_deg.shape[0])
# # #         B = pred_deg.shape[1]
# # #         if T < 2: return
# # #         dist     = haversine_km(pred_deg[:T], gt_deg[:T])
# # #         ate, cte = compute_ate_cte(pred_deg, gt_deg)
# # #         self.n_seq += B
# # #         for b in range(B):
# # #             for t in range(T):
# # #                 lead_h  = (t+1)*6
# # #                 d_val   = float(dist[t,b])
# # #                 ate_val = float(ate[t,b].abs()) if t < ate.shape[0] else float("nan")
# # #                 cte_val = float(cte[t,b].abs()) if t < cte.shape[0] else float("nan")
# # #                 self.rows.append({
# # #                     "storm_id": self.storm_id,
# # #                     "seq_idx" : int(seq_indices[b]) if seq_indices is not None else b,
# # #                     "lead_h"  : lead_h,
# # #                     "lon_pred": float(pred_deg[t,b,0]),
# # #                     "lat_pred": float(pred_deg[t,b,1]),
# # #                     "lon_true": float(gt_deg[t,b,0]),
# # #                     "lat_true": float(gt_deg[t,b,1]),
# # #                     "dist_km" : round(d_val, 2),
# # #                     "ate_km"  : round(ate_val,2) if np.isfinite(ate_val) else "nan",
# # #                     "cte_km"  : round(cte_val,2) if np.isfinite(cte_val) else "nan",
# # #                 })
# # #                 if lead_h in LEAD_STEPS:
# # #                     self.dist_by_lead[lead_h].append(d_val)
# # #                     if np.isfinite(ate_val): self.ate_by_lead[lead_h].append(ate_val)
# # #                     if np.isfinite(cte_val): self.cte_by_lead[lead_h].append(cte_val)
# # #                 self.all_dist.append(d_val)
# # #                 if t < ate.shape[0]:
# # #                     if np.isfinite(ate_val): self.all_ate.append(ate_val)
# # #                     if np.isfinite(cte_val): self.all_cte.append(cte_val)

# # #     def summary(self):
# # #         def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # #         r = {"storm_id": self.storm_id, "n_seq": self.n_seq,
# # #              "ADE": _m(self.all_dist), "ATE": _m(self.all_ate), "CTE": _m(self.all_cte)}
# # #         for h in LEAD_HOURS:
# # #             r[f"{h}h_dist"] = _m(self.dist_by_lead.get(h,[]))
# # #             r[f"{h}h_ate"]  = _m(self.ate_by_lead.get(h,[]))
# # #             r[f"{h}h_cte"]  = _m(self.cte_by_lead.get(h,[]))
# # #         return r


# # # class MultiStormAccumulator:
# # #     def __init__(self, model_name):
# # #         self.model_name = model_name
# # #         self.storms = {}

# # #     def get_storm(self, storm_id):
# # #         if storm_id not in self.storms:
# # #             self.storms[storm_id] = StormResult(storm_id)
# # #         return self.storms[storm_id]

# # #     def add_batch(self, pred_norm, gt_norm, storm_ids):
# # #         T  = min(pred_norm.shape[0], gt_norm.shape[0])
# # #         B  = pred_norm.shape[1]
# # #         # BUG-H FIX: chỉ lấy 2 features lon/lat
# # #         pd = norm_to_deg(pred_norm[:T, :, :2])
# # #         gd = norm_to_deg(gt_norm[:T,  :, :2])
# # #         sid_map = defaultdict(list)
# # #         for b, sid in enumerate(storm_ids[:B]):
# # #             sid_map[sid].append(b)
# # #         for sid, bidx in sid_map.items():
# # #             bidx_t = torch.tensor(bidx, dtype=torch.long)
# # #             self.get_storm(sid).add_batch(pd[:,bidx_t,:], gd[:,bidx_t,:],
# # #                                           seq_indices=bidx_t)

# # #     def all_summaries(self):
# # #         return [self.storms[sid].summary() for sid in sorted(self.storms)]

# # #     def overall_summary(self):
# # #         all_d, all_a, all_c = [], [], []
# # #         lead_d = defaultdict(list); lead_a = defaultdict(list); lead_c = defaultdict(list)
# # #         for sr in self.storms.values():
# # #             all_d.extend(sr.all_dist); all_a.extend(sr.all_ate); all_c.extend(sr.all_cte)
# # #             for h in LEAD_HOURS:
# # #                 lead_d[h].extend(sr.dist_by_lead.get(h,[]))
# # #                 lead_a[h].extend(sr.ate_by_lead.get(h,[]))
# # #                 lead_c[h].extend(sr.cte_by_lead.get(h,[]))
# # #         def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # #         r = {"storm_id": "OVERALL",
# # #              "n_seq": sum(s.n_seq for s in self.storms.values()),
# # #              "ADE": _m(all_d), "ATE": _m(all_a), "CTE": _m(all_c)}
# # #         for h in LEAD_HOURS:
# # #             r[f"{h}h_dist"] = _m(lead_d[h])
# # #             r[f"{h}h_ate"]  = _m(lead_a[h])
# # #             r[f"{h}h_cte"]  = _m(lead_c[h])
# # #         return r


# # # def extract_storm_ids(batch, B):
# # #     for idx in [2, 12, 14, 15]:
# # #         if idx < len(batch):
# # #             item = batch[idx]
# # #             if isinstance(item, (list, tuple)) and len(item) >= B:
# # #                 if isinstance(item[0], str):
# # #                     return list(item[:B])
# # #             if (torch.is_tensor(item) and item.dtype == torch.long
# # #                     and item.numel() >= B):
# # #                 return [str(x.item()) for x in item[:B]]
# # #     return [f"STORM_{i:03d}" for i in range(B)]


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Checkpoint utilities
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def _load_state_dict(ckpt):
# # #     """train_fm.py v2.4 lưu keys: 'epoch', 'model', 'ema'."""
# # #     if not isinstance(ckpt, dict):
# # #         return ckpt
# # #     for key in ["model", "model_state_dict", "model_state", "state_dict"]:
# # #         if key in ckpt and isinstance(ckpt[key], dict):
# # #             sd = ckpt[key]
# # #             if any(torch.is_tensor(v) for v in sd.values()):
# # #                 return sd
# # #     if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
# # #         return ckpt
# # #     return ckpt


# # # def _detect_ckpt_type(state):
# # #     keys = set(state.keys())
# # #     has_ctx  = any(k.startswith("ctx_enc.")  for k in keys)
# # #     has_corr = any(k.startswith("corr_net.") for k in keys)
# # #     has_gate = any(k.startswith("gate.")     for k in keys)
# # #     if sum([has_ctx, has_corr, has_gate]) >= 2:
# # #         return "residual"
# # #     return "tcfm"


# # # def _detect_fm_version(state):
# # #     """'v2' nếu encoder.* + velocity.*, 'v1' nếu net.*"""
# # #     has_enc = any(k.startswith("encoder.")  for k in state)
# # #     has_vel = any(k.startswith("velocity.") for k in state)
# # #     has_net = any(k.startswith("net.")      for k in state)
# # #     if has_enc and has_vel and not has_net:
# # #         return "v2"
# # #     return "v1"


# # # def _version_from_path(ckpt_path):
# # #     """
# # #     BUG-4 FIX: dùng raw string r'v(\d+)' để tránh SyntaxWarning.
# # #     'runs/fm_v24/best.pth'    -> 'v24'
# # #     'best_model_v24.pth'      -> 'v24'
# # #     'runs/v61/best_model.pth' -> 'v61'
# # #     """
# # #     parts = os.path.normpath(ckpt_path).split(os.sep)
# # #     for part in reversed(parts[:-1]):
# # #         m = re.search(r'v(\d+)', part)
# # #         if m: return f"v{m.group(1)}"
# # #     fname = os.path.splitext(os.path.basename(ckpt_path))[0]
# # #     m = re.search(r'v(\d+)', fname)
# # #     if m: return f"v{m.group(1)}"
# # #     return "ours"


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  EMA compat expand — dùng chung cho state_dict và EMA shadow
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # _VEL_OBS_KEY = "encoder.vel_obs_enc.0.weight"


# # # def _expand_vel_obs_enc(sd: dict, obs_len: int, label: str = "state") -> dict:
# # #     """
# # #     BUG-EMA FIX: Expand vel_obs_enc từ obs_len*6 → obs_len*7 trong bất kỳ dict nào
# # #     (state_dict hoặc EMA shadow). Phải gọi hàm này cho CẢ HAI trước khi load.

# # #     Lý do crash:
# # #       - state_dict đã được expand 48→56 trước load_state_dict()
# # #       - Nhưng EMA shadow vẫn còn key cũ shape [256, 48]
# # #       - Khi ema.shadow[k].copy_(v.to(device)) → size mismatch RuntimeError
# # #     """
# # #     if _VEL_OBS_KEY not in sd:
# # #         return sd
# # #     w_old    = sd[_VEL_OBS_KEY]
# # #     expected = obs_len * 7
# # #     if w_old.shape[1] == obs_len * 6:
# # #         extra = torch.randn(w_old.shape[0], obs_len,
# # #                             device=w_old.device, dtype=w_old.dtype) * 0.01
# # #         sd[_VEL_OBS_KEY] = torch.cat([w_old, extra], dim=1)
# # #         print(f"  COMPAT [{label}]: vel_obs_enc {w_old.shape[1]} → {expected}")
# # #     elif w_old.shape[1] == expected:
# # #         print(f"  COMPAT [{label}]: vel_obs_enc already {expected} cols")
# # #     else:
# # #         print(f"  COMPAT [{label}]: vel_obs_enc unexpected shape {w_old.shape}, skip")
# # #     return sd


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  FM import
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def _import_fm_class():
# # #     try:
# # #         from Model.flow_matching_model import TCFlowMatching
# # #         print("  [import] TCFlowMatching ok")
# # #         return TCFlowMatching
# # #     except ImportError: pass
# # #     try:
# # #         from Model.flow_matching_model import TCDiffusion
# # #         print("  [import] TCDiffusion (alias) ok")
# # #         return TCDiffusion
# # #     except ImportError: pass
# # #     try:
# # #         mod = importlib.import_module("Model.flow_matching_model")
# # #         for name, cls in inspect.getmembers(mod, inspect.isclass):
# # #             if (hasattr(cls, "sample") and hasattr(cls, "get_loss_breakdown")
# # #                     and issubclass(cls, nn.Module)
# # #                     and cls.__module__ == mod.__name__
# # #                     and ("Flow" in name or "TC" in name)):
# # #                 print(f"  [import] {name} (scan) ok")
# # #                 return cls
# # #     except Exception as e:
# # #         print(f"  [import] scan failed: {e}")
# # #     print("  [import] ERROR: cannot find FM class")
# # #     return None


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  FM v2 loader
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def _load_fm_v2_compat(ckpt, state, device, args, ep, TCFlowMatching):
# # #     """
# # #     Load TCFlowMatching v2.4 với backward compat vel_obs_enc 6→7 features.

# # #     BUG-EMA FIX: Expand EMA shadow dict TRƯỚC KHI copy_ vào model.
# # #     EMA shadow có cùng key set với model state_dict, nên cần cùng expand logic.
# # #     """
# # #     print("  Architecture: v2 (ContextEncoder + VelocityTransformer)")

# # #     num_dec_layers = sum(
# # #         1 for i in range(10)
# # #         if f"velocity.decoder.layers.{i}.norm1.weight" in state
# # #     ) or 4
# # #     d_model = int(state["velocity.traj_embed.weight"].shape[0]) \
# # #               if "velocity.traj_embed.weight" in state else 256
# # #     d_cond  = int(state["velocity.cond_proj.0.weight"].shape[0]) \
# # #               if "velocity.cond_proj.0.weight" in state else 256
# # #     dim_ff  = int(state["velocity.decoder.layers.0.linear1.weight"].shape[0]) \
# # #               if "velocity.decoder.layers.0.linear1.weight" in state else 512

# # #     print(f"  v2 arch: d_model={d_model}, d_cond={d_cond}, "
# # #           f"num_dec_layers={num_dec_layers}, dim_ff={dim_ff}")

# # #     model = TCFlowMatching(
# # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # #         d_model=d_model, d_cond=d_cond,
# # #         num_dec_layers=num_dec_layers, dim_ff=dim_ff,
# # #         sigma_min=0.04, sigma_max=0.08,
# # #         use_ema=True, ema_decay=0.995,
# # #     ).to(device)

# # #     if hasattr(model, "init_ema"):
# # #         model.init_ema()

# # #     # ── Expand model state_dict ──────────────────────────────────────────
# # #     state = _expand_vel_obs_enc(state, args.obs_len, label="model")

# # #     # ── Load model weights ───────────────────────────────────────────────
# # #     result = model.load_state_dict(state, strict=False)
# # #     missing    = [k for k in result.missing_keys if "_ema" not in k]
# # #     unexpected = result.unexpected_keys
# # #     if missing:
# # #         print(f"  Missing keys   : {len(missing)}")
# # #         for k in missing[:5]: print(f"    - {k}")
# # #     if unexpected:
# # #         print(f"  Unexpected keys: {len(unexpected)}")

# # #     # ── Load EMA shadow (key='ema' theo train_fm.py v2.4) ───────────────
# # #     # BUG-EMA FIX: expand EMA shadow dict TRƯỚC copy_()
# # #     ema_loaded = 0
# # #     if isinstance(ckpt, dict):
# # #         for ema_key in ["ema", "ema_shadow", "ema_state_dict"]:
# # #             if ema_key in ckpt and ckpt[ema_key]:
# # #                 ema_sd = dict(ckpt[ema_key])  # mutable copy
# # #                 # Expand EMA shadow cùng logic với model state_dict
# # #                 ema_sd = _expand_vel_obs_enc(ema_sd, args.obs_len, label="EMA")
# # #                 ema = get_ema_obj(model)
# # #                 if ema is not None and hasattr(ema, "shadow"):
# # #                     for k, v in ema_sd.items():
# # #                         if k in ema.shadow:
# # #                             try:
# # #                                 ema.shadow[k].copy_(v.to(device))
# # #                                 ema_loaded += 1
# # #                             except Exception as ce:
# # #                                 print(f"  EMA copy failed [{k}]: {ce}")
# # #                     if ema_loaded:
# # #                         print(f"  EMA: {ema_loaded} keys loaded (key='{ema_key}')")
# # #                 break

# # #     if not ema_loaded:
# # #         print("  EMA: không tìm thấy — dùng raw weights")

# # #     print(f"  FM v2 loaded ok (epoch={ep})")
# # #     return model


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  FM v1 loader (net.* — v59/v61/v67)
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def _load_fm_v1(ckpt, state, device, args, ep, TCFlowMatching):
# # #     ffn_key    = "net.transformer.layers.0.linear1.weight"
# # #     ffn_dim    = int(state[ffn_key].shape[0]) if ffn_key in state else 1024
# # #     num_layers = sum(1 for i in range(10)
# # #                      if f"net.transformer.layers.{i}.linear1.weight" in state) or 2
# # #     has_vel_expand = "net.vel_obs_expand.weight" in state
# # #     has_det_head   = any(k.startswith("net.det_head") for k in state)

# # #     print(f"  v1 arch: FFN={ffn_dim}, layers={num_layers}, "
# # #           f"det_head={has_det_head}")

# # #     model = TCFlowMatching(
# # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # #         sigma_min=0.02, use_ema=True, ema_decay=0.995,
# # #     ).to(device)
# # #     if hasattr(model, "init_ema"):
# # #         model.init_ema()

# # #     raw = get_raw_model(model)
# # #     try:
# # #         cur_ffn    = raw.net.transformer.layers[0].linear1.weight.shape[0]
# # #         cur_layers = len(raw.net.transformer.layers)
# # #     except Exception:
# # #         cur_ffn, cur_layers = 1024, 2

# # #     if cur_ffn != ffn_dim or cur_layers != num_layers:
# # #         try:
# # #             raw.net.transformer = nn.TransformerDecoder(
# # #                 nn.TransformerDecoderLayer(
# # #                     d_model=256, nhead=8, dim_feedforward=ffn_dim,
# # #                     dropout=0.10, activation="gelu", batch_first=True),
# # #                 num_layers=num_layers).to(device)
# # #             print(f"  Patched transformer -> {num_layers}L / {ffn_dim}FFN")
# # #         except Exception as e:
# # #             print(f"  Patch failed: {e}")

# # #     if has_vel_expand and not hasattr(raw.net, "vel_obs_expand"):
# # #         raw.net.vel_obs_expand = nn.Linear(256, 1024).to(device)
# # #         raw.net.vel_obs_ln     = nn.LayerNorm(256).to(device)

# # #     result  = model.load_state_dict(state, strict=False)
# # #     missing = [k for k in result.missing_keys
# # #                if not k.startswith("net.det_head") and "_ema" not in k]
# # #     if missing:
# # #         print(f"  Missing: {len(missing)}")
# # #         for k in missing[:3]: print(f"    - {k}")

# # #     ema_loaded = 0
# # #     if isinstance(ckpt, dict):
# # #         for ema_key in ["ema", "ema_shadow", "ema_state_dict"]:
# # #             if ema_key in ckpt and ckpt[ema_key]:
# # #                 ema = get_ema_obj(model)
# # #                 if ema is not None and hasattr(ema, "shadow"):
# # #                     for k, v in ckpt[ema_key].items():
# # #                         if k in ema.shadow:
# # #                             try:
# # #                                 ema.shadow[k].copy_(v.to(device))
# # #                                 ema_loaded += 1
# # #                             except Exception: pass
# # #                     if ema_loaded: print(f"  EMA: {ema_loaded} keys")
# # #                 break
# # #     if not ema_loaded:
# # #         print("  EMA: not found")

# # #     raw_model = get_raw_model(model)
# # #     raw_model._det_epoch = 999 if has_det_head else 0
# # #     print(f"  FM v1 loaded ok (epoch={ep})")
# # #     return model


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  load_fm_model — entry point
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def load_fm_model(ckpt_path, device, args):
# # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # #         print(f"  FM checkpoint not found: {ckpt_path}")
# # #         return None
# # #     try:
# # #         print(f"  Loading: {ckpt_path}")
# # #         ckpt  = torch.load(ckpt_path, map_location=device)
# # #         state = _load_state_dict(ckpt)
# # #         ep    = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"

# # #         prefix_counts = Counter(k.split(".")[0] for k in state.keys())
# # #         print("  Key prefixes: " +
# # #               ", ".join(f"{p}({n})" for p, n in
# # #                         sorted(prefix_counts.items(), key=lambda x: -x[1])[:8]))

# # #         forced = getattr(args, "fm_type", "auto")
# # #         ckpt_type = (forced if forced in ("tcfm","residual")
# # #                      else _detect_ckpt_type(state))
# # #         print(f"  Checkpoint type: {ckpt_type}  (epoch={ep})")

# # #         if ckpt_type == "residual":
# # #             print("  Residual checkpoint — STTrans fallback")
# # #             try:
# # #                 from Model.st_trans_model import STTrans
# # #                 sttrans_state = {k[len("sttrans."):]: v for k, v in state.items()
# # #                                  if k.startswith("sttrans.")}
# # #                 model = STTrans(obs_len=args.obs_len, pred_len=args.pred_len).to(device)
# # #                 model.load_state_dict(sttrans_state, strict=False)
# # #                 print("  STTrans (fallback) loaded ok")
# # #                 return model
# # #             except Exception as e:
# # #                 print(f"  Fallback failed: {e}"); return None

# # #         TCFlowMatching = _import_fm_class()
# # #         if TCFlowMatching is None:
# # #             return None

# # #         fm_version = _detect_fm_version(state)
# # #         print(f"  FM version: {fm_version}")

# # #         if fm_version == "v2":
# # #             return _load_fm_v2_compat(ckpt, state, device, args, ep, TCFlowMatching)
# # #         else:
# # #             return _load_fm_v1(ckpt, state, device, args, ep, TCFlowMatching)

# # #     except Exception as e:
# # #         print(f"  FM load failed: {e}")
# # #         import traceback; traceback.print_exc()
# # #         return None


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  LSTM / ST-Trans loaders
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def load_lstm_model(ckpt_path, device, args):
# # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # #         print(f"  LSTM checkpoint not found: {ckpt_path}"); return None
# # #     try:
# # #         from Model.paper_baseline_model import PaperBaseline
# # #         lstm_type = getattr(args, "lstm_type", "lstm")
# # #         model = PaperBaseline(model_type=lstm_type,
# # #                               pred_len=args.pred_len, obs_len=args.obs_len).to(device)
# # #         ckpt  = torch.load(ckpt_path, map_location=device)
# # #         state = _load_state_dict(ckpt)
# # #         res   = model.load_state_dict(state, strict=False)
# # #         if res.missing_keys:    print(f"  Missing: {len(res.missing_keys)}")
# # #         if res.unexpected_keys: print(f"  Unexpected: {len(res.unexpected_keys)}")
# # #         ep = ckpt.get("epoch","?") if isinstance(ckpt,dict) else "?"
# # #         print(f"  {lstm_type.upper()} loaded ok (epoch={ep})")
# # #         return model
# # #     except Exception as e:
# # #         print(f"  LSTM load failed: {e}")
# # #         import traceback; traceback.print_exc()
# # #         return None


# # # def load_sttrans_model(ckpt_path, device, args):
# # #     if not ckpt_path or not os.path.exists(ckpt_path):
# # #         print(f"  STTrans checkpoint not found: {ckpt_path}"); return None
# # #     try:
# # #         ckpt  = torch.load(ckpt_path, map_location=device)
# # #         state = _load_state_dict(ckpt)
# # #         ep    = ckpt.get("epoch","?") if isinstance(ckpt,dict) else "?"

# # #         is_v2 = any(k.startswith("steering_gate.") or k.startswith("recurv_head.")
# # #                     for k in state.keys())

# # #         if is_v2 or getattr(args,"sttrans_type","") == "sttrans_v2":
# # #             from Model.st_trans_model import STTransV2
# # #             d_model       = int(state["ctx_proj.0.weight"].shape[0]) \
# # #                             if "ctx_proj.0.weight" in state else 64
# # #             nhead         = max(1, d_model//32)
# # #             num_dec_layers= sum(1 for i in range(10)
# # #                                 if f"transformer_dec.layers.{i}.norm1.weight" in state) or 3
# # #             dim_ff        = int(state["transformer_dec.layers.0.linear1.weight"].shape[0]) \
# # #                             if "transformer_dec.layers.0.linear1.weight" in state else 512
# # #             gate_hidden   = int(state["steering_gate.gate_net.0.weight"].shape[0]) \
# # #                             if "steering_gate.gate_net.0.weight" in state else 32
# # #             recurv_hidden = int(state["recurv_head.net.0.weight"].shape[0]) \
# # #                             if "recurv_head.net.0.weight" in state else 64
# # #             threshold_curv= ckpt.get("threshold_curv",15.0) if isinstance(ckpt,dict) else 15.0
# # #             threshold_spd = ckpt.get("threshold_spd", 0.5)  if isinstance(ckpt,dict) else 0.5
# # #             model = STTransV2(
# # #                 obs_len=args.obs_len, pred_len=args.pred_len,
# # #                 d_model=d_model, nhead=nhead, num_dec_layers=num_dec_layers,
# # #                 dim_ff=dim_ff, gate_hidden=gate_hidden, recurv_hidden=recurv_hidden,
# # #                 threshold_curv=threshold_curv, threshold_spd=threshold_spd,
# # #             ).to(device)
# # #             res = model.load_state_dict(state, strict=False)
# # #             if res.missing_keys:    print(f"  Missing: {len(res.missing_keys)}")
# # #             if res.unexpected_keys: print(f"  Unexpected: {len(res.unexpected_keys)}")
# # #             print(f"  STTransV2 loaded ok (epoch={ep})"); return model
# # #         else:
# # #             sttrans_type = getattr(args,"sttrans_type","sttrans")
# # #             if sttrans_type == "sttrans_ar":
# # #                 from Model.st_trans_model import STTransAR as M
# # #             else:
# # #                 from Model.st_trans_model import STTrans as M
# # #             model = M(obs_len=args.obs_len, pred_len=args.pred_len).to(device)
# # #             res   = model.load_state_dict(state, strict=False)
# # #             if res.missing_keys:    print(f"  Missing: {len(res.missing_keys)}")
# # #             if res.unexpected_keys: print(f"  Unexpected: {len(res.unexpected_keys)}")
# # #             tag = "STTransAR" if sttrans_type=="sttrans_ar" else "STTrans"
# # #             print(f"  {tag} loaded ok (epoch={ep})"); return model
# # #     except Exception as e:
# # #         print(f"  STTrans load failed: {e}")
# # #         import traceback; traceback.print_exc()
# # #         return None


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Inference wrappers
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def infer_fm(model, batch_list, args, device):
# # #     """
# # #     BUG-F FIX: sample() v2.4 trả tuple độ dài 3 hoặc 4 → lấy result[0].
# # #     BUG-G FIX: sample() luôn trả pred shape [T, B, 2] → không cần permute.
# # #     """
# # #     try:
# # #         model.eval()
# # #         ema = get_ema_obj(model); backup = None
# # #         if ema is not None:
# # #             try:
# # #                 backup = ema.apply_to(model)
# # #             except Exception as e:
# # #                 print(f"    EMA apply failed ({e}), using raw weights")
# # #                 backup = None

# # #         if hasattr(model, "sample"):
# # #             # BUG-F FIX: unpack an toàn
# # #             result = model.sample(
# # #                 batch_list,
# # #                 num_ensemble=args.fm_ensemble,
# # #                 ddim_steps=args.fm_ode_steps,
# # #             )
# # #             pred = result[0]  # [T, B, 2] — luôn đúng chiều với v2.4
# # #         else:
# # #             pred = model(batch_list)
# # #             # Fallback cho model không có sample(): permute nếu [B, T, 2]
# # #             if pred.dim() == 3 and pred.shape[0] != batch_list[1].shape[0]:
# # #                 pred = pred.permute(1, 0, 2)

# # #         if backup is not None:
# # #             try: ema.restore(model, backup)
# # #             except Exception: pass

# # #         # BUG-G FIX: không permute — sample() đã trả [T, B, 2]
# # #         # Chỉ slice last dim nếu extra features
# # #         if pred.dim() == 3 and pred.shape[-1] > 2:
# # #             pred = pred[..., :2]
# # #         return pred

# # #     except Exception as e:
# # #         print(f"    FM inference error: {e}")
# # #         import traceback; traceback.print_exc()
# # #         return None


# # # @torch.no_grad()
# # # def infer_lstm(model, batch_list, args, device):
# # #     try:
# # #         model.eval()
# # #         if hasattr(model, "sample"):
# # #             result = model.sample(batch_list, num_ensemble=1)
# # #             pred   = result[0]
# # #         else:
# # #             pred = model(batch_list)
# # #         if pred.dim() == 3 and pred.shape[0] == batch_list[0].shape[1]:
# # #             pred = pred.permute(1, 0, 2)
# # #         return pred[..., :2] if pred.shape[-1] > 2 else pred
# # #     except Exception as e:
# # #         print(f"    LSTM inference error: {e}"); return None


# # # @torch.no_grad()
# # # def infer_sttrans(model, batch_list, args, device):
# # #     try:
# # #         model.eval()
# # #         if hasattr(model, "sample"):
# # #             result = model.sample(batch_list, num_ensemble=1)
# # #             pred   = result[0]
# # #         else:
# # #             pred = model(batch_list)
# # #         if pred.dim() == 3 and pred.shape[0] == batch_list[0].shape[1]:
# # #             pred = pred.permute(1, 0, 2)
# # #         return pred[..., :2] if pred.shape[-1] > 2 else pred
# # #     except Exception as e:
# # #         print(f"    STTrans inference error: {e}"); return None


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  CSV writers
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # _PER_STORM_FIELDS = ["model","storm_id","seq_idx","lead_h",
# # #                      "lon_pred","lat_pred","lon_true","lat_true",
# # #                      "dist_km","ate_km","cte_km"]

# # # _SUMMARY_FIELDS = ["model","storm_id","n_seq","ADE","ATE","CTE",
# # #                    "12h_dist","24h_dist","48h_dist","72h_dist",
# # #                    "12h_ate","24h_ate","48h_ate","72h_ate",
# # #                    "12h_cte","24h_cte","48h_cte","72h_cte",
# # #                    "beat_12h","beat_24h","beat_48h","beat_72h",
# # #                    "beat_ADE","beat_ATE","beat_CTE"]


# # # def write_per_storm_csv(acc, out_dir):
# # #     out = os.path.join(out_dir, f"{acc.model_name}_per_storm.csv")
# # #     with open(out, "w", newline="") as fh:
# # #         w = csv.DictWriter(fh, fieldnames=_PER_STORM_FIELDS)
# # #         w.writeheader()
# # #         for sr in acc.storms.values():
# # #             for row in sr.rows:
# # #                 w.writerow({"model": acc.model_name, **row,
# # #                             "lon_pred": f"{row['lon_pred']:.4f}",
# # #                             "lat_pred": f"{row['lat_pred']:.4f}",
# # #                             "lon_true": f"{row['lon_true']:.4f}",
# # #                             "lat_true": f"{row['lat_true']:.4f}"})
# # #     print(f"  saved: {out}"); return out


# # # def write_summary_csv(all_acc, out_dir):
# # #     out = os.path.join(out_dir, "summary_all_models.csv")
# # #     with open(out, "w", newline="") as fh:
# # #         w = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
# # #         w.writeheader()
# # #         def _b(v,t): return "YES" if (np.isfinite(v) and v<t) else "NO"
# # #         def _f(v):   return f"{v:.2f}" if np.isfinite(v) else "nan"
# # #         for mname, acc in all_acc.items():
# # #             for r in acc.all_summaries() + [acc.overall_summary()]:
# # #                 w.writerow({
# # #                     "model": mname, "storm_id": r["storm_id"], "n_seq": r["n_seq"],
# # #                     "ADE": _f(r["ADE"]), "ATE": _f(r["ATE"]), "CTE": _f(r["CTE"]),
# # #                     "12h_dist":_f(r["12h_dist"]),"24h_dist":_f(r["24h_dist"]),
# # #                     "48h_dist":_f(r["48h_dist"]),"72h_dist":_f(r["72h_dist"]),
# # #                     "12h_ate":_f(r["12h_ate"]),"24h_ate":_f(r["24h_ate"]),
# # #                     "48h_ate":_f(r["48h_ate"]),"72h_ate":_f(r["72h_ate"]),
# # #                     "12h_cte":_f(r["12h_cte"]),"24h_cte":_f(r["24h_cte"]),
# # #                     "48h_cte":_f(r["48h_cte"]),"72h_cte":_f(r["72h_cte"]),
# # #                     "beat_12h":_b(r["12h_dist"],TARGETS["12h"]),
# # #                     "beat_24h":_b(r["24h_dist"],TARGETS["24h"]),
# # #                     "beat_48h":_b(r["48h_dist"],TARGETS["48h"]),
# # #                     "beat_72h":_b(r["72h_dist"],TARGETS["72h"]),
# # #                     "beat_ADE":_b(r["ADE"],TARGETS["ADE"]),
# # #                     "beat_ATE":_b(r["ATE"],TARGETS["ATE"]),
# # #                     "beat_CTE":_b(r["CTE"],TARGETS["CTE"]),
# # #                 })
# # #     print(f"  summary: {out}"); return out


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Pretty print
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def print_model_results(acc):
# # #     cfg = MODEL_CONFIGS.get(acc.model_name, {}); col = cfg.get("color",""); ref = cfg.get("ref","")
# # #     print(f"\n{'='*115}")
# # #     print(f"  {col}  {acc.model_name}  |  {ref}")
# # #     print(f"{'='*115}")
# # #     print(f"  {'Storm':<16} {'N':>4}  {'ADE':>8}  {'ATE':>8}  {'CTE':>8}  "
# # #           f"|  {'12h':>7}  {'24h':>7}  {'48h':>7}  {'72h':>7}  "
# # #           f"|  {'ATE@72':>8}  {'CTE@72':>8}")
# # #     print("  " + "-"*111)

# # #     def v(x, t=None, w=8, dec=1):
# # #         if not np.isfinite(x): return f"{'nan':>{w+2}}"
# # #         s = f"{x:>{w}.{dec}f}"
# # #         s += (" ok" if x<t else " --") if t is not None else "   "
# # #         return s

# # #     for r in acc.all_summaries():
# # #         print(f"  * {r['storm_id']:<14} {r['n_seq']:>4}  "
# # #               f"{v(r['ADE'],TARGETS['ADE'])}  {v(r['ATE'],TARGETS['ATE'])}  "
# # #               f"{v(r['CTE'],TARGETS['CTE'])}  "
# # #               f"|  {v(r['12h_dist'],TARGETS['12h'],7)}  "
# # #               f"{v(r['24h_dist'],TARGETS['24h'],7)}  "
# # #               f"{v(r['48h_dist'],TARGETS['48h'],7)}  "
# # #               f"{v(r['72h_dist'],TARGETS['72h'],7)}  "
# # #               f"|  {v(r['72h_ate'],None,8)}  {v(r['72h_cte'],None,8)}")

# # #     ov = acc.overall_summary()
# # #     print("  " + "-"*111)
# # #     print(f"  OVERALL            {ov['n_seq']:>4}  "
# # #           f"{v(ov['ADE'],TARGETS['ADE'])}  {v(ov['ATE'],TARGETS['ATE'])}  "
# # #           f"{v(ov['CTE'],TARGETS['CTE'])}  "
# # #           f"|  {v(ov['12h_dist'],TARGETS['12h'],7)}  "
# # #           f"{v(ov['24h_dist'],TARGETS['24h'],7)}  "
# # #           f"{v(ov['48h_dist'],TARGETS['48h'],7)}  "
# # #           f"{v(ov['72h_dist'],TARGETS['72h'],7)}  "
# # #           f"|  {v(ov['72h_ate'],None,8)}  {v(ov['72h_cte'],None,8)}")
# # #     print(f"{'='*115}\n")
# # #     print(f"  ST-Trans VAL  ref: ADE=172.7  ATE=142.2  CTE=42.0")
# # #     print(f"  ST-Trans TEST ref: ADE=224.4  ATE=213.7  CTE=59.4")


# # # def print_comparison_table(all_acc):
# # #     print(f"\n{'='*95}")
# # #     print(f"  MODEL COMPARISON -- Overall Test Set")
# # #     print(f"{'='*95}")
# # #     print(f"  {'Model':<14}  {'ADE':>9}  {'ATE':>9}  {'CTE':>9}  "
# # #           f"|  {'12h':>8}  {'24h':>8}  {'48h':>8}  {'72h':>8}")
# # #     print("  " + "-"*91)

# # #     def vv(x, t=None, w=9, dec=1):
# # #         if not np.isfinite(x): return f"{'nan':>{w+3}}"
# # #         s = f"{x:>{w}.{dec}f}"
# # #         s += (" ok" if x<t else " --") if t is not None else "   "
# # #         return s

# # #     for mname, acc in all_acc.items():
# # #         ov  = acc.overall_summary()
# # #         print(f"  {mname:<14}  "
# # #               f"{vv(ov['ADE'],TARGETS['ADE'])}  {vv(ov['ATE'],TARGETS['ATE'])}  "
# # #               f"{vv(ov['CTE'],TARGETS['CTE'])}  "
# # #               f"|  {vv(ov['12h_dist'],TARGETS['12h'],8)}  "
# # #               f"{vv(ov['24h_dist'],TARGETS['24h'],8)}  "
# # #               f"{vv(ov['48h_dist'],TARGETS['48h'],8)}  "
# # #               f"{vv(ov['72h_dist'],TARGETS['72h'],8)}")
# # #     print(f"{'='*95}\n")


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Main loop
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def run_evaluation(model, model_name, test_loader, device, args, infer_fn):
# # #     acc = MultiStormAccumulator(model_name)
# # #     n   = len(test_loader); t0 = time.perf_counter()
# # #     print(f"\n  Running {model_name} on {n} batches...")

# # #     for i, batch in enumerate(test_loader):
# # #         bl        = move_to_device(list(batch), device)
# # #         B         = bl[0].shape[1]
# # #         pred_norm = infer_fn(model, bl, args, device)
# # #         if pred_norm is None: continue

# # #         # BUG-H FIX: gt có thể có extra features → slice :2
# # #         gt_norm = bl[1]
# # #         if gt_norm.shape[-1] > 2:
# # #             gt_norm = gt_norm[..., :2]

# # #         storm_ids = extract_storm_ids(batch, B)
# # #         T = min(pred_norm.shape[0], gt_norm.shape[0])
# # #         acc.add_batch(pred_norm[:T], gt_norm[:T], storm_ids)

# # #         if (i+1) % max(1, n//5) == 0 or i == n-1:
# # #             ov = acc.overall_summary()
# # #             print(f"    [{i+1:>4}/{n}]  ADE={ov['ADE']:.1f}  "
# # #                   f"ATE={ov['ATE']:.1f}  CTE={ov['CTE']:.1f} km  "
# # #                   f"elapsed={time.perf_counter()-t0:.1f}s")

# # #     ov = acc.overall_summary()
# # #     print(f"  DONE {model_name}: ADE={ov['ADE']:.1f}  "
# # #           f"ATE={ov['ATE']:.1f}  CTE={ov['CTE']:.1f} km")
# # #     return acc


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Args + main
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def get_args():
# # #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # #     p.add_argument("--dataset_root",  default="TCND_vn")
# # #     p.add_argument("--obs_len",       default=8,   type=int)
# # #     p.add_argument("--pred_len",      default=12,  type=int)
# # #     p.add_argument("--batch_size",    default=16,  type=int)
# # #     p.add_argument("--num_workers",   default=2,   type=int)
# # #     p.add_argument("--skip",          default=1,   type=int)
# # #     p.add_argument("--min_ped",       default=1,   type=int)
# # #     p.add_argument("--threshold",     default=0.002, type=float)
# # #     p.add_argument("--other_modal",   default="gph")
# # #     p.add_argument("--test_year",     default=None, type=int)
# # #     p.add_argument("--delim",         default=" ")
# # #     p.add_argument("--fm_ckpt",       default=None)
# # #     p.add_argument("--lstm_ckpt",     default=None)
# # #     p.add_argument("--sttrans_ckpt",  default=None)
# # #     p.add_argument("--lstm_type",     default="lstm", choices=["lstm","gru","rnn"])
# # #     p.add_argument("--sttrans_type",  default="sttrans",
# # #                    choices=["sttrans","sttrans_ar","sttrans_v2"])
# # #     p.add_argument("--fm_ensemble",   default=20,  type=int)
# # #     p.add_argument("--fm_ode_steps",  default=1,   type=int)
# # #     p.add_argument("--fm_type",       default="auto",
# # #                    choices=["auto","tcfm","residual"])
# # #     p.add_argument("--output_dir",    default="results/test_eval")
# # #     p.add_argument("--gpu_num",       default="0")
# # #     p.add_argument("--eval_models",   default="all",
# # #                    help="'all' hoac csv: 'fm,lstm,sttrans'")
# # #     p.add_argument("--no_detail_csv", action="store_true")
# # #     return p.parse_args()


# # # def main(args):
# # #     if torch.cuda.is_available():
# # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #     os.makedirs(args.output_dir, exist_ok=True)

# # #     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# # #     print("="*80)
# # #     print(f"  TC Model Evaluator v2.4-compat")
# # #     print(f"  Started: {ts}  |  Device: {device}")
# # #     print(f"  Output : {args.output_dir}")
# # #     print("="*80)

# # #     eval_set = ({"fm","lstm","sttrans"} if args.eval_models=="all"
# # #                 else {m.strip().lower() for m in args.eval_models.split(",")})

# # #     print("\n  Loading test dataset...")
# # #     try:
# # #         from Model.data.loader_training import data_loader
# # #         _, test_loader = data_loader(
# # #             args, {"root": args.dataset_root, "type": "test"}, test=True)
# # #         print(f"  {len(test_loader)} batches loaded")
# # #     except Exception as e:
# # #         print(f"  Test data load failed: {e}")
# # #         import traceback; traceback.print_exc(); return

# # #     all_acc = {}

# # #     if "fm" in eval_set and args.fm_ckpt:
# # #         print(f"\n{'─'*50}")
# # #         model = load_fm_model(args.fm_ckpt, device, args)
# # #         if model is not None:
# # #             tag = "FM_" + _version_from_path(args.fm_ckpt)
# # #             all_acc[tag] = run_evaluation(model, tag, test_loader, device, args, infer_fm)
# # #             del model
# # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # #     if "lstm" in eval_set and args.lstm_ckpt:
# # #         print(f"\n{'─'*50}")
# # #         model = load_lstm_model(args.lstm_ckpt, device, args)
# # #         if model is not None:
# # #             tag = args.lstm_type.upper()
# # #             all_acc[tag] = run_evaluation(model, tag, test_loader, device, args, infer_lstm)
# # #             del model
# # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # #     if "sttrans" in eval_set and args.sttrans_ckpt:
# # #         print(f"\n{'─'*50}")
# # #         model = load_sttrans_model(args.sttrans_ckpt, device, args)
# # #         if model is not None:
# # #             tag = ("STTransAR" if args.sttrans_type=="sttrans_ar"
# # #                    else "STTransV2" if args.sttrans_type=="sttrans_v2"
# # #                    else "STTrans")
# # #             all_acc[tag] = run_evaluation(model, tag, test_loader, device, args, infer_sttrans)
# # #             del model
# # #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# # #     if not all_acc:
# # #         print("\n  No models evaluated.")
# # #         print("  Specify: --fm_ckpt / --lstm_ckpt / --sttrans_ckpt")
# # #         return

# # #     for acc in all_acc.values():
# # #         print_model_results(acc)
# # #     if len(all_acc) > 1:
# # #         print_comparison_table(all_acc)

# # #     print("\n  Writing CSVs...")
# # #     csv_paths = []
# # #     for acc in all_acc.values():
# # #         csv_paths.append(write_per_storm_csv(acc, args.output_dir))
# # #     csv_paths.append(write_summary_csv(all_acc, args.output_dir))

# # #     metric_map = [("ADE","ADE"),("ATE","ATE"),("CTE","CTE"),
# # #                   ("12h_dist","12h"),("24h_dist","24h"),
# # #                   ("48h_dist","48h"),("72h_dist","72h")]
# # #     print(f"\n{'='*80}")
# # #     print("  FINAL BEAT SUMMARY")
# # #     print(f"{'─'*80}")
# # #     for mname, acc in all_acc.items():
# # #         ov  = acc.overall_summary()
# # #         beats = [f"{label}={ov.get(k,float('nan')):.1f} ok"
# # #                  for k, label in metric_map
# # #                  if np.isfinite(ov.get(k,float("nan")))
# # #                  and ov[k] < TARGETS.get(label, float("inf"))]
# # #         print(f"  {mname:<14}: " + (" | ".join(beats) if beats else "no targets beaten"))
# # #     print(f"{'='*80}\n")


# # # if __name__ == "__main__":
# # #     np.random.seed(42); torch.manual_seed(42)
# # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# # #     args = get_args()
# # #     _kaggle_base = "/kaggle/working/TC_OFM"
# # #     if args.dataset_root == "TCND_vn" and os.path.isdir(_kaggle_base):
# # #         args.dataset_root = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
# # #     main(args)
# # """
# # scripts/evaluate_test_storms.py  — TC Model Evaluator v2.4-compat
# # ═══════════════════════════════════════════════════════════════════

# # BUGS ĐÃ FIX (bản này):
# #   BUG-1+5: load_fm_v2_compat là dead code → fix: gọi đúng trong load_fm_model
# #   BUG-2:   infer_fm dùng importance_weight=True → TypeError → fix: bỏ kwarg đó
# #   BUG-4:   _version_from_path dùng r'v(d+)' không escape → SyntaxWarning → fix raw string
# #   BUG-EMA: EMA shadow còn vel_obs_enc shape 48 → crash khi copy vào model 56
# #            → fix: expand EMA shadow cùng logic với state_dict trước khi copy_()
# #   BUG-F:   infer_fm unpack cứng 3 giá trị → crash khi sample() trả 4 (xai path)
# #            → fix: lấy result[0]
# #   BUG-G:   infer_fm permute sai chiều v2.4 — sample() luôn trả [T,B,2]
# #            → fix: bỏ permute cho FM path
# #   BUG-H:   gt_norm có thể có extra features → fix: slice [...,:2]

# #   ── Fixes bổ sung (review) ──────────────────────────────────────────────────
# #   FIX-R1:  infer_fm EMA restore không được gọi nếu model.sample() raise exception
# #            → fix: dùng try/finally để đảm bảo restore luôn chạy
# #   FIX-R2:  _load_state_dict không kiểm tra ckpt là raw state_dict (không có key
# #            'model', 'epoch', ...) nhưng giá trị là Tensor → trả về ckpt đúng
# #            Tuy nhiên nếu ckpt chứa mixed keys (vừa có Tensor vừa có dict con)
# #            thì loop "for key in [...]" trả về sai → fix: guard thêm trường hợp
# #            ckpt đã là flat state_dict ngay từ đầu
# #   FIX-R3:  _detect_fm_version nhận state có thể là outer ckpt dict (chứa 'epoch',
# #            'optimizer', ...) nếu _load_state_dict trả về sai → fix: thêm assert
# #            guard + fallback rõ ràng
# #   FIX-R4:  infer_lstm / infer_sttrans dùng bl[0].shape[1] làm B nhưng bl[0] dim-0
# #            là T_obs → đúng, nhưng thêm comment giải thích để tránh nhầm

# # Cách dùng:
# #     python scripts/evaluate_test_storms.py \
# #         --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# #         --fm_ckpt      /kaggle/.../best_model_v24.pth \
# #         --output_dir   results/test_eval \
# #         --eval_models  fm \
# #         --fm_ensemble  20 \
# #         --fm_ode_steps 1 \
# #         --gpu_num 0
# # """
# # from __future__ import annotations

# # import sys
# # import os
# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # import argparse
# # import csv
# # import re
# # import time
# # import importlib
# # import inspect
# # from collections import defaultdict, Counter
# # from datetime import datetime
# # from typing import Dict, List, Optional, Tuple

# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import DataLoader

# # R_EARTH  = 6371.0
# # DT_HOURS = 6.0

# # LEAD_HOURS = [12, 24, 36, 48, 60, 72]
# # LEAD_STEPS = {h: h // 6 - 1 for h in LEAD_HOURS}

# # TARGETS = {
# #     "12h": 50.0, "24h": 100.0, "48h": 200.0, "72h": 297.0,
# #     "ADE": 136.41, "ATE": 79.94, "CTE": 93.58,
# # }

# # MODEL_CONFIGS = {
# #     "LSTM"     : {"color": "🔵", "ref": "Rahman 2025"},
# #     "GRU"      : {"color": "🔵", "ref": "Rahman 2025"},
# #     "RNN"      : {"color": "🔵", "ref": "Rahman 2025"},
# #     "STTrans"  : {"color": "🟡", "ref": "Faiaz 2026"},
# #     "STTransAR": {"color": "🟡", "ref": "Faiaz 2026 AR"},
# #     "FM_ours"  : {"color": "🔴", "ref": "Ours v2.4"},
# # }


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Coordinate utilities
# # # ─────────────────────────────────────────────────────────────────────────────

# # def norm_to_deg(arr):
# #     out = arr.clone()
# #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# #     return out


# # def haversine_km(p1, p2):
# #     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
# #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# #     a = (torch.sin(dlat/2).pow(2) +
# #          torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2))
# #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())


# # def compute_ate_cte(pred_deg, gt_deg):
# #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# #     if T < 2:
# #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# #         return z, z
# #     lon1 = torch.deg2rad(gt_deg[:T-1,:,0]); lat1 = torch.deg2rad(gt_deg[:T-1,:,1])
# #     lon2 = torch.deg2rad(gt_deg[1:T, :,0]); lat2 = torch.deg2rad(gt_deg[1:T, :,1])
# #     dlon_a = lon2 - lon1
# #     y_a = torch.sin(dlon_a)*torch.cos(lat2)
# #     x_a = torch.cos(lat1)*torch.sin(lat2) - torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon_a)
# #     bear_along = torch.atan2(y_a, x_a)
# #     lon3 = torch.deg2rad(pred_deg[1:T,:,0]); lat3 = torch.deg2rad(pred_deg[1:T,:,1])
# #     dlon_e = lon3 - lon2
# #     y_e = torch.sin(dlon_e)*torch.cos(lat3)
# #     x_e = torch.cos(lat2)*torch.sin(lat3) - torch.sin(lat2)*torch.cos(lat3)*torch.cos(dlon_e)
# #     bear_err = torch.atan2(y_e, x_e)
# #     total = haversine_km(pred_deg[1:T], gt_deg[1:T])
# #     angle = bear_err - bear_along
# #     return total*torch.cos(angle), total*torch.sin(angle)


# # def move_to_device(batch, device):
# #     out = list(batch)
# #     for i, x in enumerate(out):
# #         if torch.is_tensor(x):
# #             out[i] = x.to(device)
# #         elif isinstance(x, dict):
# #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k,v in x.items()}
# #     return out


# # def get_raw_model(model):
# #     return model._orig_mod if hasattr(model, "_orig_mod") else model


# # def get_ema_obj(model):
# #     raw = get_raw_model(model)
# #     if hasattr(raw, "_ema") and raw._ema is not None:
# #         return raw._ema
# #     return None


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Per-storm accumulator
# # # ─────────────────────────────────────────────────────────────────────────────

# # class StormResult:
# #     def __init__(self, storm_id):
# #         self.storm_id = storm_id
# #         self.rows = []
# #         self.dist_by_lead = defaultdict(list)
# #         self.ate_by_lead  = defaultdict(list)
# #         self.cte_by_lead  = defaultdict(list)
# #         self.all_dist = []; self.all_ate = []; self.all_cte = []
# #         self.n_seq = 0

# #     def add_batch(self, pred_deg, gt_deg, seq_indices=None):
# #         T = min(pred_deg.shape[0], gt_deg.shape[0])
# #         B = pred_deg.shape[1]
# #         if T < 2: return
# #         dist     = haversine_km(pred_deg[:T], gt_deg[:T])
# #         ate, cte = compute_ate_cte(pred_deg, gt_deg)
# #         self.n_seq += B
# #         for b in range(B):
# #             for t in range(T):
# #                 lead_h  = (t+1)*6
# #                 d_val   = float(dist[t,b])
# #                 ate_val = float(ate[t,b].abs()) if t < ate.shape[0] else float("nan")
# #                 cte_val = float(cte[t,b].abs()) if t < cte.shape[0] else float("nan")
# #                 self.rows.append({
# #                     "storm_id": self.storm_id,
# #                     "seq_idx" : int(seq_indices[b]) if seq_indices is not None else b,
# #                     "lead_h"  : lead_h,
# #                     "lon_pred": float(pred_deg[t,b,0]),
# #                     "lat_pred": float(pred_deg[t,b,1]),
# #                     "lon_true": float(gt_deg[t,b,0]),
# #                     "lat_true": float(gt_deg[t,b,1]),
# #                     "dist_km" : round(d_val, 2),
# #                     "ate_km"  : round(ate_val,2) if np.isfinite(ate_val) else "nan",
# #                     "cte_km"  : round(cte_val,2) if np.isfinite(cte_val) else "nan",
# #                 })
# #                 if lead_h in LEAD_STEPS:
# #                     self.dist_by_lead[lead_h].append(d_val)
# #                     if np.isfinite(ate_val): self.ate_by_lead[lead_h].append(ate_val)
# #                     if np.isfinite(cte_val): self.cte_by_lead[lead_h].append(cte_val)
# #                 self.all_dist.append(d_val)
# #                 if t < ate.shape[0]:
# #                     if np.isfinite(ate_val): self.all_ate.append(ate_val)
# #                     if np.isfinite(cte_val): self.all_cte.append(cte_val)

# #     def summary(self):
# #         def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# #         r = {"storm_id": self.storm_id, "n_seq": self.n_seq,
# #              "ADE": _m(self.all_dist), "ATE": _m(self.all_ate), "CTE": _m(self.all_cte)}
# #         for h in LEAD_HOURS:
# #             r[f"{h}h_dist"] = _m(self.dist_by_lead.get(h,[]))
# #             r[f"{h}h_ate"]  = _m(self.ate_by_lead.get(h,[]))
# #             r[f"{h}h_cte"]  = _m(self.cte_by_lead.get(h,[]))
# #         return r


# # class MultiStormAccumulator:
# #     def __init__(self, model_name):
# #         self.model_name = model_name
# #         self.storms = {}

# #     def get_storm(self, storm_id):
# #         if storm_id not in self.storms:
# #             self.storms[storm_id] = StormResult(storm_id)
# #         return self.storms[storm_id]

# #     def add_batch(self, pred_norm, gt_norm, storm_ids):
# #         T  = min(pred_norm.shape[0], gt_norm.shape[0])
# #         B  = pred_norm.shape[1]
# #         # BUG-H FIX: chỉ lấy 2 features lon/lat
# #         pd = norm_to_deg(pred_norm[:T, :, :2])
# #         gd = norm_to_deg(gt_norm[:T,  :, :2])
# #         sid_map = defaultdict(list)
# #         for b, sid in enumerate(storm_ids[:B]):
# #             sid_map[sid].append(b)
# #         for sid, bidx in sid_map.items():
# #             bidx_t = torch.tensor(bidx, dtype=torch.long)
# #             self.get_storm(sid).add_batch(pd[:,bidx_t,:], gd[:,bidx_t,:],
# #                                           seq_indices=bidx_t)

# #     def all_summaries(self):
# #         return [self.storms[sid].summary() for sid in sorted(self.storms)]

# #     def overall_summary(self):
# #         all_d, all_a, all_c = [], [], []
# #         lead_d = defaultdict(list); lead_a = defaultdict(list); lead_c = defaultdict(list)
# #         for sr in self.storms.values():
# #             all_d.extend(sr.all_dist); all_a.extend(sr.all_ate); all_c.extend(sr.all_cte)
# #             for h in LEAD_HOURS:
# #                 lead_d[h].extend(sr.dist_by_lead.get(h,[]))
# #                 lead_a[h].extend(sr.ate_by_lead.get(h,[]))
# #                 lead_c[h].extend(sr.cte_by_lead.get(h,[]))
# #         def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# #         r = {"storm_id": "OVERALL",
# #              "n_seq": sum(s.n_seq for s in self.storms.values()),
# #              "ADE": _m(all_d), "ATE": _m(all_a), "CTE": _m(all_c)}
# #         for h in LEAD_HOURS:
# #             r[f"{h}h_dist"] = _m(lead_d[h])
# #             r[f"{h}h_ate"]  = _m(lead_a[h])
# #             r[f"{h}h_cte"]  = _m(lead_c[h])
# #         return r


# # def extract_storm_ids(batch, B):
# #     for idx in [2, 12, 14, 15]:
# #         if idx < len(batch):
# #             item = batch[idx]
# #             if isinstance(item, (list, tuple)) and len(item) >= B:
# #                 if isinstance(item[0], str):
# #                     return list(item[:B])
# #             if (torch.is_tensor(item) and item.dtype == torch.long
# #                     and item.numel() >= B):
# #                 return [str(x.item()) for x in item[:B]]
# #     return [f"STORM_{i:03d}" for i in range(B)]


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Checkpoint utilities
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _load_state_dict(ckpt):
# #     """
# #     train_fm.py v2.4 lưu keys: 'epoch', 'model', 'ema'.

# #     FIX-R2: Xử lý đúng 3 trường hợp:
# #       (a) ckpt là flat state_dict (tất cả values đều là Tensor) → trả về ngay
# #       (b) ckpt là outer dict có key 'model'/'model_state_dict'/... → unwrap
# #       (c) ckpt không phải dict (raw tensor, legacy format) → trả về nguyên
# #     """
# #     if not isinstance(ckpt, dict):
# #         return ckpt

# #     # Trường hợp (a): flat state_dict — tất cả values là Tensor
# #     # Kiểm tra này phải đứng TRƯỚC loop unwrap để tránh unwrap nhầm
# #     all_tensor = all(torch.is_tensor(v) for v in ckpt.values())
# #     if all_tensor:
# #         return ckpt

# #     # Trường hợp (b): outer checkpoint dict — tìm key chứa model state
# #     for key in ["model", "model_state_dict", "model_state", "state_dict"]:
# #         if key in ckpt and isinstance(ckpt[key], dict):
# #             sd = ckpt[key]
# #             # Chỉ chấp nhận nếu có ít nhất 1 Tensor (tránh unwrap dict rỗng hoặc dict meta)
# #             if any(torch.is_tensor(v) for v in sd.values()):
# #                 return sd

# #     # Trường hợp (c): mixed dict — có thể là state_dict có vài non-Tensor keys
# #     # (ví dụ: có key '_metadata' dạng dict trong một số PyTorch version)
# #     # Nếu majority là Tensor thì vẫn coi là state_dict
# #     tensor_count = sum(1 for v in ckpt.values() if torch.is_tensor(v))
# #     if tensor_count > 0 and tensor_count >= len(ckpt) // 2:
# #         return ckpt

# #     return ckpt


# # def _detect_ckpt_type(state):
# #     keys = set(state.keys())
# #     has_ctx  = any(k.startswith("ctx_enc.")  for k in keys)
# #     has_corr = any(k.startswith("corr_net.") for k in keys)
# #     has_gate = any(k.startswith("gate.")     for k in keys)
# #     if sum([has_ctx, has_corr, has_gate]) >= 2:
# #         return "residual"
# #     return "tcfm"


# # def _detect_fm_version(state):
# #     """
# #     'v2' nếu encoder.* + velocity.*, 'v1' nếu net.*

# #     FIX-R3: Guard thêm kiểm tra state không phải outer checkpoint dict.
# #     Nếu _load_state_dict trả về sai (ckpt vẫn còn key 'epoch', 'optimizer'...),
# #     hàm này sẽ không tìm thấy 'encoder.' hay 'velocity.' → fallback 'v1' nhầm.
# #     Thêm warning rõ ràng để debug dễ hơn.
# #     """
# #     # Kiểm tra state có vẻ là outer checkpoint dict không
# #     outer_keys = {"epoch", "optimizer", "scheduler_ep", "best_score", "scaler"}
# #     if outer_keys.intersection(state.keys()):
# #         print("  WARNING [_detect_fm_version]: state có vẻ là outer checkpoint dict "
# #               "(còn keys 'epoch'/'optimizer'...). _load_state_dict có thể đã unwrap sai.")

# #     has_enc = any(k.startswith("encoder.")  for k in state)
# #     has_vel = any(k.startswith("velocity.") for k in state)
# #     has_net = any(k.startswith("net.")      for k in state)
# #     if has_enc and has_vel and not has_net:
# #         return "v2"
# #     return "v1"


# # def _version_from_path(ckpt_path):
# #     """
# #     BUG-4 FIX: dùng raw string r'v(d+)' để tránh SyntaxWarning.
# #     'runs/fm_v24/best.pth'    -> 'v24'
# #     'best_model_v24.pth'      -> 'v24'
# #     'runs/v61/best_model.pth' -> 'v61'
# #     """
# #     parts = os.path.normpath(ckpt_path).split(os.sep)
# #     for part in reversed(parts[:-1]):
# #         m = re.search(r'v(\d+)', part)
# #         if m: return f"v{m.group(1)}"
# #     fname = os.path.splitext(os.path.basename(ckpt_path))[0]
# #     m = re.search(r'v(\d+)', fname)
# #     if m: return f"v{m.group(1)}"
# #     return "ours"


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  EMA compat expand — dùng chung cho state_dict và EMA shadow
# # # ─────────────────────────────────────────────────────────────────────────────

# # _VEL_OBS_KEY = "encoder.vel_obs_enc.0.weight"


# # def _expand_vel_obs_enc(sd: dict, obs_len: int, label: str = "state") -> dict:
# #     """
# #     BUG-EMA FIX: Expand vel_obs_enc từ obs_len*6 → obs_len*7 trong bất kỳ dict nào
# #     (state_dict hoặc EMA shadow). Phải gọi hàm này cho CẢ HAI trước khi load.

# #     Lý do crash:
# #       - state_dict đã được expand 48→56 trước load_state_dict()
# #       - Nhưng EMA shadow vẫn còn key cũ shape [256, 48]
# #       - Khi ema.shadow[k].copy_(v.to(device)) → size mismatch RuntimeError
# #     """
# #     if _VEL_OBS_KEY not in sd:
# #         return sd
# #     w_old    = sd[_VEL_OBS_KEY]
# #     expected = obs_len * 7
# #     if w_old.shape[1] == obs_len * 6:
# #         extra = torch.randn(w_old.shape[0], obs_len,
# #                             device=w_old.device, dtype=w_old.dtype) * 0.01
# #         sd[_VEL_OBS_KEY] = torch.cat([w_old, extra], dim=1)
# #         print(f"  COMPAT [{label}]: vel_obs_enc {w_old.shape[1]} → {expected}")
# #     elif w_old.shape[1] == expected:
# #         print(f"  COMPAT [{label}]: vel_obs_enc already {expected} cols")
# #     else:
# #         print(f"  COMPAT [{label}]: vel_obs_enc unexpected shape {w_old.shape}, skip")
# #     return sd


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  FM import
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _import_fm_class():
# #     try:
# #         from Model.flow_matching_model import TCFlowMatching
# #         print("  [import] TCFlowMatching ok")
# #         return TCFlowMatching
# #     except ImportError: pass
# #     try:
# #         from Model.flow_matching_model import TCDiffusion
# #         print("  [import] TCDiffusion (alias) ok")
# #         return TCDiffusion
# #     except ImportError: pass
# #     try:
# #         mod = importlib.import_module("Model.flow_matching_model")
# #         for name, cls in inspect.getmembers(mod, inspect.isclass):
# #             if (hasattr(cls, "sample") and hasattr(cls, "get_loss_breakdown")
# #                     and issubclass(cls, nn.Module)
# #                     and cls.__module__ == mod.__name__
# #                     and ("Flow" in name or "TC" in name)):
# #                 print(f"  [import] {name} (scan) ok")
# #                 return cls
# #     except Exception as e:
# #         print(f"  [import] scan failed: {e}")
# #     print("  [import] ERROR: cannot find FM class")
# #     return None


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  FM v2 loader
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _load_fm_v2_compat(ckpt, state, device, args, ep, TCFlowMatching):
# #     """
# #     Load TCFlowMatching v2.4 với backward compat vel_obs_enc 6→7 features.

# #     BUG-EMA FIX: Expand EMA shadow dict TRƯỚC KHI copy_ vào model.
# #     EMA shadow có cùng key set với model state_dict, nên cần cùng expand logic.
# #     """
# #     print("  Architecture: v2 (ContextEncoder + VelocityTransformer)")

# #     num_dec_layers = sum(
# #         1 for i in range(10)
# #         if f"velocity.decoder.layers.{i}.norm1.weight" in state
# #     ) or 4
# #     d_model = int(state["velocity.traj_embed.weight"].shape[0]) \
# #               if "velocity.traj_embed.weight" in state else 256
# #     d_cond  = int(state["velocity.cond_proj.0.weight"].shape[0]) \
# #               if "velocity.cond_proj.0.weight" in state else 256
# #     dim_ff  = int(state["velocity.decoder.layers.0.linear1.weight"].shape[0]) \
# #               if "velocity.decoder.layers.0.linear1.weight" in state else 512

# #     print(f"  v2 arch: d_model={d_model}, d_cond={d_cond}, "
# #           f"num_dec_layers={num_dec_layers}, dim_ff={dim_ff}")

# #     model = TCFlowMatching(
# #         pred_len=args.pred_len, obs_len=args.obs_len,
# #         d_model=d_model, d_cond=d_cond,
# #         num_dec_layers=num_dec_layers, dim_ff=dim_ff,
# #         sigma_min=0.04, sigma_max=0.08,
# #         use_ema=True, ema_decay=0.995,
# #     ).to(device)

# #     if hasattr(model, "init_ema"):
# #         model.init_ema()

# #     # ── Expand model state_dict ──────────────────────────────────────────
# #     state = _expand_vel_obs_enc(state, args.obs_len, label="model")

# #     # ── Load model weights ───────────────────────────────────────────────
# #     result = model.load_state_dict(state, strict=False)
# #     missing    = [k for k in result.missing_keys if "_ema" not in k]
# #     unexpected = result.unexpected_keys
# #     if missing:
# #         print(f"  Missing keys   : {len(missing)}")
# #         for k in missing[:5]: print(f"    - {k}")
# #     if unexpected:
# #         print(f"  Unexpected keys: {len(unexpected)}")

# #     # ── Load EMA shadow (key='ema' theo train_fm.py v2.4) ───────────────
# #     # BUG-EMA FIX: expand EMA shadow dict TRƯỚC copy_()
# #     ema_loaded = 0
# #     if isinstance(ckpt, dict):
# #         for ema_key in ["ema", "ema_shadow", "ema_state_dict"]:
# #             if ema_key in ckpt and ckpt[ema_key]:
# #                 ema_sd = dict(ckpt[ema_key])  # mutable copy
# #                 # Expand EMA shadow cùng logic với model state_dict
# #                 ema_sd = _expand_vel_obs_enc(ema_sd, args.obs_len, label="EMA")
# #                 ema = get_ema_obj(model)
# #                 if ema is not None and hasattr(ema, "shadow"):
# #                     for k, v in ema_sd.items():
# #                         if k in ema.shadow:
# #                             try:
# #                                 ema.shadow[k].copy_(v.to(device))
# #                                 ema_loaded += 1
# #                             except Exception as ce:
# #                                 print(f"  EMA copy failed [{k}]: {ce}")
# #                     if ema_loaded:
# #                         print(f"  EMA: {ema_loaded} keys loaded (key='{ema_key}')")
# #                 break

# #     if not ema_loaded:
# #         print("  EMA: không tìm thấy — dùng raw weights")

# #     print(f"  FM v2 loaded ok (epoch={ep})")
# #     return model


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  FM v1 loader (net.* — v59/v61/v67)
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _load_fm_v1(ckpt, state, device, args, ep, TCFlowMatching):
# #     ffn_key    = "net.transformer.layers.0.linear1.weight"
# #     ffn_dim    = int(state[ffn_key].shape[0]) if ffn_key in state else 1024
# #     num_layers = sum(1 for i in range(10)
# #                      if f"net.transformer.layers.{i}.linear1.weight" in state) or 2
# #     has_vel_expand = "net.vel_obs_expand.weight" in state
# #     has_det_head   = any(k.startswith("net.det_head") for k in state)

# #     print(f"  v1 arch: FFN={ffn_dim}, layers={num_layers}, "
# #           f"det_head={has_det_head}")

# #     model = TCFlowMatching(
# #         pred_len=args.pred_len, obs_len=args.obs_len,
# #         sigma_min=0.02, use_ema=True, ema_decay=0.995,
# #     ).to(device)
# #     if hasattr(model, "init_ema"):
# #         model.init_ema()

# #     raw = get_raw_model(model)
# #     try:
# #         cur_ffn    = raw.net.transformer.layers[0].linear1.weight.shape[0]
# #         cur_layers = len(raw.net.transformer.layers)
# #     except Exception:
# #         cur_ffn, cur_layers = 1024, 2

# #     if cur_ffn != ffn_dim or cur_layers != num_layers:
# #         try:
# #             raw.net.transformer = nn.TransformerDecoder(
# #                 nn.TransformerDecoderLayer(
# #                     d_model=256, nhead=8, dim_feedforward=ffn_dim,
# #                     dropout=0.10, activation="gelu", batch_first=True),
# #                 num_layers=num_layers).to(device)
# #             print(f"  Patched transformer -> {num_layers}L / {ffn_dim}FFN")
# #         except Exception as e:
# #             print(f"  Patch failed: {e}")

# #     if has_vel_expand and not hasattr(raw.net, "vel_obs_expand"):
# #         raw.net.vel_obs_expand = nn.Linear(256, 1024).to(device)
# #         raw.net.vel_obs_ln     = nn.LayerNorm(256).to(device)

# #     result  = model.load_state_dict(state, strict=False)
# #     missing = [k for k in result.missing_keys
# #                if not k.startswith("net.det_head") and "_ema" not in k]
# #     if missing:
# #         print(f"  Missing: {len(missing)}")
# #         for k in missing[:3]: print(f"    - {k}")

# #     ema_loaded = 0
# #     if isinstance(ckpt, dict):
# #         for ema_key in ["ema", "ema_shadow", "ema_state_dict"]:
# #             if ema_key in ckpt and ckpt[ema_key]:
# #                 ema = get_ema_obj(model)
# #                 if ema is not None and hasattr(ema, "shadow"):
# #                     for k, v in ckpt[ema_key].items():
# #                         if k in ema.shadow:
# #                             try:
# #                                 ema.shadow[k].copy_(v.to(device))
# #                                 ema_loaded += 1
# #                             except Exception: pass
# #                     if ema_loaded: print(f"  EMA: {ema_loaded} keys")
# #                 break
# #     if not ema_loaded:
# #         print("  EMA: not found")

# #     raw_model = get_raw_model(model)
# #     raw_model._det_epoch = 999 if has_det_head else 0
# #     print(f"  FM v1 loaded ok (epoch={ep})")
# #     return model


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  load_fm_model — entry point
# # # ─────────────────────────────────────────────────────────────────────────────

# # def load_fm_model(ckpt_path, device, args):
# #     if not ckpt_path or not os.path.exists(ckpt_path):
# #         print(f"  FM checkpoint not found: {ckpt_path}")
# #         return None
# #     try:
# #         print(f"  Loading: {ckpt_path}")
# #         ckpt  = torch.load(ckpt_path, map_location=device)
# #         state = _load_state_dict(ckpt)
# #         ep    = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"

# #         prefix_counts = Counter(k.split(".")[0] for k in state.keys()
# #                                 if torch.is_tensor(state[k]))
# #         print("  Key prefixes: " +
# #               ", ".join(f"{p}({n})" for p, n in
# #                         sorted(prefix_counts.items(), key=lambda x: -x[1])[:8]))

# #         forced = getattr(args, "fm_type", "auto")
# #         ckpt_type = (forced if forced in ("tcfm","residual")
# #                      else _detect_ckpt_type(state))
# #         print(f"  Checkpoint type: {ckpt_type}  (epoch={ep})")

# #         if ckpt_type == "residual":
# #             print("  Residual checkpoint — STTrans fallback")
# #             try:
# #                 from Model.st_trans_model import STTrans
# #                 sttrans_state = {k[len("sttrans."):]: v for k, v in state.items()
# #                                  if k.startswith("sttrans.")}
# #                 model = STTrans(obs_len=args.obs_len, pred_len=args.pred_len).to(device)
# #                 model.load_state_dict(sttrans_state, strict=False)
# #                 print("  STTrans (fallback) loaded ok")
# #                 return model
# #             except Exception as e:
# #                 print(f"  Fallback failed: {e}"); return None

# #         TCFlowMatching = _import_fm_class()
# #         if TCFlowMatching is None:
# #             return None

# #         fm_version = _detect_fm_version(state)
# #         print(f"  FM version: {fm_version}")

# #         if fm_version == "v2":
# #             return _load_fm_v2_compat(ckpt, state, device, args, ep, TCFlowMatching)
# #         else:
# #             return _load_fm_v1(ckpt, state, device, args, ep, TCFlowMatching)

# #     except Exception as e:
# #         print(f"  FM load failed: {e}")
# #         import traceback; traceback.print_exc()
# #         return None


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  LSTM / ST-Trans loaders
# # # ─────────────────────────────────────────────────────────────────────────────

# # def load_lstm_model(ckpt_path, device, args):
# #     if not ckpt_path or not os.path.exists(ckpt_path):
# #         print(f"  LSTM checkpoint not found: {ckpt_path}"); return None
# #     try:
# #         from Model.paper_baseline_model import PaperBaseline
# #         lstm_type = getattr(args, "lstm_type", "lstm")
# #         model = PaperBaseline(model_type=lstm_type,
# #                               pred_len=args.pred_len, obs_len=args.obs_len).to(device)
# #         ckpt  = torch.load(ckpt_path, map_location=device)
# #         state = _load_state_dict(ckpt)
# #         res   = model.load_state_dict(state, strict=False)
# #         if res.missing_keys:    print(f"  Missing: {len(res.missing_keys)}")
# #         if res.unexpected_keys: print(f"  Unexpected: {len(res.unexpected_keys)}")
# #         ep = ckpt.get("epoch","?") if isinstance(ckpt,dict) else "?"
# #         print(f"  {lstm_type.upper()} loaded ok (epoch={ep})")
# #         return model
# #     except Exception as e:
# #         print(f"  LSTM load failed: {e}")
# #         import traceback; traceback.print_exc()
# #         return None


# # def load_sttrans_model(ckpt_path, device, args):
# #     if not ckpt_path or not os.path.exists(ckpt_path):
# #         print(f"  STTrans checkpoint not found: {ckpt_path}"); return None
# #     try:
# #         ckpt  = torch.load(ckpt_path, map_location=device)
# #         state = _load_state_dict(ckpt)
# #         ep    = ckpt.get("epoch","?") if isinstance(ckpt,dict) else "?"

# #         is_v2 = any(k.startswith("steering_gate.") or k.startswith("recurv_head.")
# #                     for k in state.keys())

# #         if is_v2 or getattr(args,"sttrans_type","") == "sttrans_v2":
# #             from Model.st_trans_model import STTransV2
# #             d_model       = int(state["ctx_proj.0.weight"].shape[0]) \
# #                             if "ctx_proj.0.weight" in state else 64
# #             nhead         = max(1, d_model//32)
# #             num_dec_layers= sum(1 for i in range(10)
# #                                 if f"transformer_dec.layers.{i}.norm1.weight" in state) or 3
# #             dim_ff        = int(state["transformer_dec.layers.0.linear1.weight"].shape[0]) \
# #                             if "transformer_dec.layers.0.linear1.weight" in state else 512
# #             gate_hidden   = int(state["steering_gate.gate_net.0.weight"].shape[0]) \
# #                             if "steering_gate.gate_net.0.weight" in state else 32
# #             recurv_hidden = int(state["recurv_head.net.0.weight"].shape[0]) \
# #                             if "recurv_head.net.0.weight" in state else 64
# #             threshold_curv= ckpt.get("threshold_curv",15.0) if isinstance(ckpt,dict) else 15.0
# #             threshold_spd = ckpt.get("threshold_spd", 0.5)  if isinstance(ckpt,dict) else 0.5
# #             model = STTransV2(
# #                 obs_len=args.obs_len, pred_len=args.pred_len,
# #                 d_model=d_model, nhead=nhead, num_dec_layers=num_dec_layers,
# #                 dim_ff=dim_ff, gate_hidden=gate_hidden, recurv_hidden=recurv_hidden,
# #                 threshold_curv=threshold_curv, threshold_spd=threshold_spd,
# #             ).to(device)
# #             res = model.load_state_dict(state, strict=False)
# #             if res.missing_keys:    print(f"  Missing: {len(res.missing_keys)}")
# #             if res.unexpected_keys: print(f"  Unexpected: {len(res.unexpected_keys)}")
# #             print(f"  STTransV2 loaded ok (epoch={ep})"); return model
# #         else:
# #             sttrans_type = getattr(args,"sttrans_type","sttrans")
# #             if sttrans_type == "sttrans_ar":
# #                 from Model.st_trans_model import STTransAR as M
# #             else:
# #                 from Model.st_trans_model import STTrans as M
# #             model = M(obs_len=args.obs_len, pred_len=args.pred_len).to(device)
# #             res   = model.load_state_dict(state, strict=False)
# #             if res.missing_keys:    print(f"  Missing: {len(res.missing_keys)}")
# #             if res.unexpected_keys: print(f"  Unexpected: {len(res.unexpected_keys)}")
# #             tag = "STTransAR" if sttrans_type=="sttrans_ar" else "STTrans"
# #             print(f"  {tag} loaded ok (epoch={ep})"); return model
# #     except Exception as e:
# #         print(f"  STTrans load failed: {e}")
# #         import traceback; traceback.print_exc()
# #         return None


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Inference wrappers
# # # ─────────────────────────────────────────────────────────────────────────────

# # @torch.no_grad()
# # def infer_fm(model, batch_list, args, device):
# #     """
# #     BUG-F FIX: sample() v2.4 trả tuple độ dài 3 hoặc 4 → lấy result[0].
# #     BUG-G FIX: sample() luôn trả pred shape [T, B, 2] → không cần permute.

# #     FIX-R1: EMA restore được đảm bảo chạy qua try/finally ngay cả khi
# #     model.sample() raise exception. Trước đây nếu sample() crash thì
# #     backup không được restore, model bị dirty EMA weights cho các batch sau.

# #     Lưu ý về batch dimension:
# #       batch_list[0] shape = [T_obs, B, feat]  (dim-0 là time, dim-1 là batch)
# #       sample() trả pred_mean shape = [T_pred, B, 2]  (absolute normalized coords)
# #     """
# #     try:
# #         model.eval()
# #         ema = get_ema_obj(model)
# #         backup = None

# #         # FIX-R1: tách apply và sample ra để finally luôn restore được
# #         if ema is not None:
# #             try:
# #                 backup = ema.apply_to(model)
# #             except Exception as e:
# #                 print(f"    EMA apply failed ({e}), using raw weights")
# #                 backup = None

# #         try:
# #             if hasattr(model, "sample"):
# #                 # BUG-F FIX: unpack an toàn — result[0] dù tuple 3 hay 4 phần tử
# #                 result = model.sample(
# #                     batch_list,
# #                     num_ensemble=args.fm_ensemble,
# #                     ddim_steps=args.fm_ode_steps,
# #                 )
# #                 pred = result[0]  # [T, B, 2] — luôn đúng chiều với v2.4
# #             else:
# #                 pred = model(batch_list)
# #                 # Fallback cho model không có sample(): permute nếu [B, T, 2]
# #                 # batch_list[0].shape[1] là B (dim-1)
# #                 if pred.dim() == 3 and pred.shape[0] == batch_list[0].shape[1]:
# #                     pred = pred.permute(1, 0, 2)

# #         finally:
# #             # FIX-R1: restore luôn chạy dù sample() có crash hay không
# #             if backup is not None:
# #                 try:
# #                     ema.restore(model, backup)
# #                 except Exception as re_:
# #                     print(f"    EMA restore failed: {re_}")

# #         # BUG-G FIX: không permute — sample() đã trả [T, B, 2]
# #         # Chỉ slice last dim nếu extra features
# #         if pred.dim() == 3 and pred.shape[-1] > 2:
# #             pred = pred[..., :2]
# #         return pred

# #     except Exception as e:
# #         print(f"    FM inference error: {e}")
# #         import traceback; traceback.print_exc()
# #         return None


# # @torch.no_grad()
# # def infer_lstm(model, batch_list, args, device):
# #     """
# #     Lưu ý về batch dimension:
# #       batch_list[0] shape = [T_obs, B, feat]  (dim-0 là time, dim-1 là batch)
# #       B = batch_list[0].shape[1]
# #     """
# #     try:
# #         model.eval()
# #         if hasattr(model, "sample"):
# #             result = model.sample(batch_list, num_ensemble=1)
# #             pred   = result[0]
# #         else:
# #             pred = model(batch_list)
# #         # Nếu model trả [B, T, 2] thay vì [T, B, 2]: B = batch_list[0].shape[1]
# #         if pred.dim() == 3 and pred.shape[0] == batch_list[0].shape[1]:
# #             pred = pred.permute(1, 0, 2)
# #         return pred[..., :2] if pred.shape[-1] > 2 else pred
# #     except Exception as e:
# #         print(f"    LSTM inference error: {e}"); return None


# # @torch.no_grad()
# # def infer_sttrans(model, batch_list, args, device):
# #     """
# #     Lưu ý về batch dimension:
# #       batch_list[0] shape = [T_obs, B, feat]  (dim-0 là time, dim-1 là batch)
# #       B = batch_list[0].shape[1]
# #     """
# #     try:
# #         model.eval()
# #         if hasattr(model, "sample"):
# #             result = model.sample(batch_list, num_ensemble=1)
# #             pred   = result[0]
# #         else:
# #             pred = model(batch_list)
# #         # Nếu model trả [B, T, 2] thay vì [T, B, 2]: B = batch_list[0].shape[1]
# #         if pred.dim() == 3 and pred.shape[0] == batch_list[0].shape[1]:
# #             pred = pred.permute(1, 0, 2)
# #         return pred[..., :2] if pred.shape[-1] > 2 else pred
# #     except Exception as e:
# #         print(f"    STTrans inference error: {e}"); return None


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  CSV writers
# # # ─────────────────────────────────────────────────────────────────────────────

# # _PER_STORM_FIELDS = ["model","storm_id","seq_idx","lead_h",
# #                      "lon_pred","lat_pred","lon_true","lat_true",
# #                      "dist_km","ate_km","cte_km"]

# # _SUMMARY_FIELDS = ["model","storm_id","n_seq","ADE","ATE","CTE",
# #                    "12h_dist","24h_dist","48h_dist","72h_dist",
# #                    "12h_ate","24h_ate","48h_ate","72h_ate",
# #                    "12h_cte","24h_cte","48h_cte","72h_cte",
# #                    "beat_12h","beat_24h","beat_48h","beat_72h",
# #                    "beat_ADE","beat_ATE","beat_CTE"]


# # def write_per_storm_csv(acc, out_dir):
# #     out = os.path.join(out_dir, f"{acc.model_name}_per_storm.csv")
# #     with open(out, "w", newline="") as fh:
# #         w = csv.DictWriter(fh, fieldnames=_PER_STORM_FIELDS)
# #         w.writeheader()
# #         for sr in acc.storms.values():
# #             for row in sr.rows:
# #                 w.writerow({"model": acc.model_name, **row,
# #                             "lon_pred": f"{row['lon_pred']:.4f}",
# #                             "lat_pred": f"{row['lat_pred']:.4f}",
# #                             "lon_true": f"{row['lon_true']:.4f}",
# #                             "lat_true": f"{row['lat_true']:.4f}"})
# #     print(f"  saved: {out}"); return out


# # def write_summary_csv(all_acc, out_dir):
# #     out = os.path.join(out_dir, "summary_all_models.csv")
# #     with open(out, "w", newline="") as fh:
# #         w = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
# #         w.writeheader()
# #         def _b(v,t): return "YES" if (np.isfinite(v) and v<t) else "NO"
# #         def _f(v):   return f"{v:.2f}" if np.isfinite(v) else "nan"
# #         for mname, acc in all_acc.items():
# #             for r in acc.all_summaries() + [acc.overall_summary()]:
# #                 w.writerow({
# #                     "model": mname, "storm_id": r["storm_id"], "n_seq": r["n_seq"],
# #                     "ADE": _f(r["ADE"]), "ATE": _f(r["ATE"]), "CTE": _f(r["CTE"]),
# #                     "12h_dist":_f(r["12h_dist"]),"24h_dist":_f(r["24h_dist"]),
# #                     "48h_dist":_f(r["48h_dist"]),"72h_dist":_f(r["72h_dist"]),
# #                     "12h_ate":_f(r["12h_ate"]),"24h_ate":_f(r["24h_ate"]),
# #                     "48h_ate":_f(r["48h_ate"]),"72h_ate":_f(r["72h_ate"]),
# #                     "12h_cte":_f(r["12h_cte"]),"24h_cte":_f(r["24h_cte"]),
# #                     "48h_cte":_f(r["48h_cte"]),"72h_cte":_f(r["72h_cte"]),
# #                     "beat_12h":_b(r["12h_dist"],TARGETS["12h"]),
# #                     "beat_24h":_b(r["24h_dist"],TARGETS["24h"]),
# #                     "beat_48h":_b(r["48h_dist"],TARGETS["48h"]),
# #                     "beat_72h":_b(r["72h_dist"],TARGETS["72h"]),
# #                     "beat_ADE":_b(r["ADE"],TARGETS["ADE"]),
# #                     "beat_ATE":_b(r["ATE"],TARGETS["ATE"]),
# #                     "beat_CTE":_b(r["CTE"],TARGETS["CTE"]),
# #                 })
# #     print(f"  summary: {out}"); return out


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Pretty print
# # # ─────────────────────────────────────────────────────────────────────────────

# # def print_model_results(acc):
# #     cfg = MODEL_CONFIGS.get(acc.model_name, {}); col = cfg.get("color",""); ref = cfg.get("ref","")
# #     print(f"\n{'='*115}")
# #     print(f"  {col}  {acc.model_name}  |  {ref}")
# #     print(f"{'='*115}")
# #     print(f"  {'Storm':<16} {'N':>4}  {'ADE':>8}  {'ATE':>8}  {'CTE':>8}  "
# #           f"|  {'12h':>7}  {'24h':>7}  {'48h':>7}  {'72h':>7}  "
# #           f"|  {'ATE@72':>8}  {'CTE@72':>8}")
# #     print("  " + "-"*111)

# #     def v(x, t=None, w=8, dec=1):
# #         if not np.isfinite(x): return f"{'nan':>{w+2}}"
# #         s = f"{x:>{w}.{dec}f}"
# #         s += (" ok" if x<t else " --") if t is not None else "   "
# #         return s

# #     for r in acc.all_summaries():
# #         print(f"  * {r['storm_id']:<14} {r['n_seq']:>4}  "
# #               f"{v(r['ADE'],TARGETS['ADE'])}  {v(r['ATE'],TARGETS['ATE'])}  "
# #               f"{v(r['CTE'],TARGETS['CTE'])}  "
# #               f"|  {v(r['12h_dist'],TARGETS['12h'],7)}  "
# #               f"{v(r['24h_dist'],TARGETS['24h'],7)}  "
# #               f"{v(r['48h_dist'],TARGETS['48h'],7)}  "
# #               f"{v(r['72h_dist'],TARGETS['72h'],7)}  "
# #               f"|  {v(r['72h_ate'],None,8)}  {v(r['72h_cte'],None,8)}")

# #     ov = acc.overall_summary()
# #     print("  " + "-"*111)
# #     print(f"  OVERALL            {ov['n_seq']:>4}  "
# #           f"{v(ov['ADE'],TARGETS['ADE'])}  {v(ov['ATE'],TARGETS['ATE'])}  "
# #           f"{v(ov['CTE'],TARGETS['CTE'])}  "
# #           f"|  {v(ov['12h_dist'],TARGETS['12h'],7)}  "
# #           f"{v(ov['24h_dist'],TARGETS['24h'],7)}  "
# #           f"{v(ov['48h_dist'],TARGETS['48h'],7)}  "
# #           f"{v(ov['72h_dist'],TARGETS['72h'],7)}  "
# #           f"|  {v(ov['72h_ate'],None,8)}  {v(ov['72h_cte'],None,8)}")
# #     print(f"{'='*115}\n")
# #     print(f"  ST-Trans VAL  ref: ADE=172.7  ATE=142.2  CTE=42.0")
# #     print(f"  ST-Trans TEST ref: ADE=224.4  ATE=213.7  CTE=59.4")


# # def print_comparison_table(all_acc):
# #     print(f"\n{'='*95}")
# #     print(f"  MODEL COMPARISON -- Overall Test Set")
# #     print(f"{'='*95}")
# #     print(f"  {'Model':<14}  {'ADE':>9}  {'ATE':>9}  {'CTE':>9}  "
# #           f"|  {'12h':>8}  {'24h':>8}  {'48h':>8}  {'72h':>8}")
# #     print("  " + "-"*91)

# #     def vv(x, t=None, w=9, dec=1):
# #         if not np.isfinite(x): return f"{'nan':>{w+3}}"
# #         s = f"{x:>{w}.{dec}f}"
# #         s += (" ok" if x<t else " --") if t is not None else "   "
# #         return s

# #     for mname, acc in all_acc.items():
# #         ov  = acc.overall_summary()
# #         print(f"  {mname:<14}  "
# #               f"{vv(ov['ADE'],TARGETS['ADE'])}  {vv(ov['ATE'],TARGETS['ATE'])}  "
# #               f"{vv(ov['CTE'],TARGETS['CTE'])}  "
# #               f"|  {vv(ov['12h_dist'],TARGETS['12h'],8)}  "
# #               f"{vv(ov['24h_dist'],TARGETS['24h'],8)}  "
# #               f"{vv(ov['48h_dist'],TARGETS['48h'],8)}  "
# #               f"{vv(ov['72h_dist'],TARGETS['72h'],8)}")
# #     print(f"{'='*95}\n")


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Main loop
# # # ─────────────────────────────────────────────────────────────────────────────

# # def run_evaluation(model, model_name, test_loader, device, args, infer_fn):
# #     acc = MultiStormAccumulator(model_name)
# #     n   = len(test_loader); t0 = time.perf_counter()
# #     print(f"\n  Running {model_name} on {n} batches...")

# #     for i, batch in enumerate(test_loader):
# #         bl        = move_to_device(list(batch), device)
# #         # batch[0] shape = [T_obs, B, feat] — dim-1 là batch size
# #         B         = bl[0].shape[1]
# #         pred_norm = infer_fn(model, bl, args, device)
# #         if pred_norm is None: continue

# #         # BUG-H FIX: gt có thể có extra features → slice :2
# #         gt_norm = bl[1]
# #         if gt_norm.shape[-1] > 2:
# #             gt_norm = gt_norm[..., :2]

# #         storm_ids = extract_storm_ids(batch, B)
# #         T = min(pred_norm.shape[0], gt_norm.shape[0])
# #         acc.add_batch(pred_norm[:T], gt_norm[:T], storm_ids)

# #         if (i+1) % max(1, n//5) == 0 or i == n-1:
# #             ov = acc.overall_summary()
# #             print(f"    [{i+1:>4}/{n}]  ADE={ov['ADE']:.1f}  "
# #                   f"ATE={ov['ATE']:.1f}  CTE={ov['CTE']:.1f} km  "
# #                   f"elapsed={time.perf_counter()-t0:.1f}s")

# #     ov = acc.overall_summary()
# #     print(f"  DONE {model_name}: ADE={ov['ADE']:.1f}  "
# #           f"ATE={ov['ATE']:.1f}  CTE={ov['CTE']:.1f} km")
# #     return acc


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Args + main
# # # ─────────────────────────────────────────────────────────────────────────────

# # def get_args():
# #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# #     p.add_argument("--dataset_root",  default="TCND_vn")
# #     p.add_argument("--obs_len",       default=8,   type=int)
# #     p.add_argument("--pred_len",      default=12,  type=int)
# #     p.add_argument("--batch_size",    default=16,  type=int)
# #     p.add_argument("--num_workers",   default=2,   type=int)
# #     p.add_argument("--skip",          default=1,   type=int)
# #     p.add_argument("--min_ped",       default=1,   type=int)
# #     p.add_argument("--threshold",     default=0.002, type=float)
# #     p.add_argument("--other_modal",   default="gph")
# #     p.add_argument("--test_year",     default=None, type=int)
# #     p.add_argument("--delim",         default=" ")
# #     p.add_argument("--fm_ckpt",       default=None)
# #     p.add_argument("--lstm_ckpt",     default=None)
# #     p.add_argument("--sttrans_ckpt",  default=None)
# #     p.add_argument("--lstm_type",     default="lstm", choices=["lstm","gru","rnn"])
# #     p.add_argument("--sttrans_type",  default="sttrans",
# #                    choices=["sttrans","sttrans_ar","sttrans_v2"])
# #     p.add_argument("--fm_ensemble",   default=20,  type=int)
# #     p.add_argument("--fm_ode_steps",  default=1,   type=int)
# #     p.add_argument("--fm_type",       default="auto",
# #                    choices=["auto","tcfm","residual"])
# #     p.add_argument("--output_dir",    default="results/test_eval")
# #     p.add_argument("--gpu_num",       default="0")
# #     p.add_argument("--eval_models",   default="all",
# #                    help="'all' hoac csv: 'fm,lstm,sttrans'")
# #     p.add_argument("--no_detail_csv", action="store_true")
# #     return p.parse_args()


# # def main(args):
# #     if torch.cuda.is_available():
# #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     os.makedirs(args.output_dir, exist_ok=True)

# #     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# #     print("="*80)
# #     print(f"  TC Model Evaluator v2.4-compat")
# #     print(f"  Started: {ts}  |  Device: {device}")
# #     print(f"  Output : {args.output_dir}")
# #     print("="*80)

# #     eval_set = ({"fm","lstm","sttrans"} if args.eval_models=="all"
# #                 else {m.strip().lower() for m in args.eval_models.split(",")})

# #     print("\n  Loading test dataset...")
# #     try:
# #         from Model.data.loader_training import data_loader
# #         _, test_loader = data_loader(
# #             args, {"root": args.dataset_root, "type": "test"}, test=True)
# #         print(f"  {len(test_loader)} batches loaded")
# #     except Exception as e:
# #         print(f"  Test data load failed: {e}")
# #         import traceback; traceback.print_exc(); return

# #     all_acc = {}

# #     if "fm" in eval_set and args.fm_ckpt:
# #         print(f"\n{'─'*50}")
# #         model = load_fm_model(args.fm_ckpt, device, args)
# #         if model is not None:
# #             tag = "FM_" + _version_from_path(args.fm_ckpt)
# #             all_acc[tag] = run_evaluation(model, tag, test_loader, device, args, infer_fm)
# #             del model
# #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# #     if "lstm" in eval_set and args.lstm_ckpt:
# #         print(f"\n{'─'*50}")
# #         model = load_lstm_model(args.lstm_ckpt, device, args)
# #         if model is not None:
# #             tag = args.lstm_type.upper()
# #             all_acc[tag] = run_evaluation(model, tag, test_loader, device, args, infer_lstm)
# #             del model
# #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# #     if "sttrans" in eval_set and args.sttrans_ckpt:
# #         print(f"\n{'─'*50}")
# #         model = load_sttrans_model(args.sttrans_ckpt, device, args)
# #         if model is not None:
# #             tag = ("STTransAR" if args.sttrans_type=="sttrans_ar"
# #                    else "STTransV2" if args.sttrans_type=="sttrans_v2"
# #                    else "STTrans")
# #             all_acc[tag] = run_evaluation(model, tag, test_loader, device, args, infer_sttrans)
# #             del model
# #             if torch.cuda.is_available(): torch.cuda.empty_cache()

# #     if not all_acc:
# #         print("\n  No models evaluated.")
# #         print("  Specify: --fm_ckpt / --lstm_ckpt / --sttrans_ckpt")
# #         return

# #     for acc in all_acc.values():
# #         print_model_results(acc)
# #     if len(all_acc) > 1:
# #         print_comparison_table(all_acc)

# #     print("\n  Writing CSVs...")
# #     csv_paths = []
# #     for acc in all_acc.values():
# #         csv_paths.append(write_per_storm_csv(acc, args.output_dir))
# #     csv_paths.append(write_summary_csv(all_acc, args.output_dir))

# #     metric_map = [("ADE","ADE"),("ATE","ATE"),("CTE","CTE"),
# #                   ("12h_dist","12h"),("24h_dist","24h"),
# #                   ("48h_dist","48h"),("72h_dist","72h")]
# #     print(f"\n{'='*80}")
# #     print("  FINAL BEAT SUMMARY")
# #     print(f"{'─'*80}")
# #     for mname, acc in all_acc.items():
# #         ov  = acc.overall_summary()
# #         beats = [f"{label}={ov.get(k,float('nan')):.1f} ok"
# #                  for k, label in metric_map
# #                  if np.isfinite(ov.get(k,float("nan")))
# #                  and ov[k] < TARGETS.get(label, float("inf"))]
# #         print(f"  {mname:<14}: " + (" | ".join(beats) if beats else "no targets beaten"))
# #     print(f"{'='*80}\n")


# # if __name__ == "__main__":
# #     np.random.seed(42); torch.manual_seed(42)
# #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# #     args = get_args()
# #     _kaggle_base = "/kaggle/working/TC_OFM"
# #     if args.dataset_root == "TCND_vn" and os.path.isdir(_kaggle_base):
# #         args.dataset_root = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
# #     main(args)

# """
# scripts/evaluate.py  ──  TC-FlowMatching v2.1-XAI Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

# Standalone evaluation script. Chạy sau khi train xong để:
#   1. Đánh giá performance (ADE/ATE/CTE) trên val và test
#   2. Thu thập XAI-1 đến XAI-9 đầy đủ trên toàn bộ dataset
#   3. Phân tích physical consistency
#   4. Uncertainty calibration (CRPS)
#   5. Failure mode analysis per storm category
#   6. Export kết quả ra CSV/JSON cho paper

# Dùng:
#   python scripts/evaluate.py \
#     --checkpoint runs/fm_v21xai/best_model.pth \
#     --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
#     --output_dir runs/fm_v21xai/xai_results \
#     --split test

# ST-Trans targets:
#   Val:  ADE=172.68  ATE=142.21  CTE=42.04
#   Test: ADE=224.4   ATE=213.7   CTE=59.4
# """
# from __future__ import annotations
# import sys, os, json, time
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse
# import math
# from collections import defaultdict
# from typing import Dict, List, Optional

# import numpy as np
# import torch
# import torch.nn.functional as F

# from Model.data.loader_training import data_loader
# from Model.flow_matching_model import (
#     TCFlowMatching, _norm_to_deg, _haversine_deg,
#     augment_batch, hard_score_from_obs,
#     compute_ensemble_uncertainty,
#     compute_heading_deviation, compute_cte_contribution,
#     compute_obs_attribution,
# )

# # ─────────────────────────────────────────────────────────────────────────────
# #  Constants
# # ─────────────────────────────────────────────────────────────────────────────

# HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}
# ST_TRANS_VAL  = {"ADE": 172.68, "ATE": 142.21, "CTE": 42.04,
#                  "12h": 65.42,  "24h": 104.67, "48h": 205.10, "72h": 321.39}
# ST_TRANS_TEST = {"ADE": 224.4,  "ATE": 213.7,  "CTE": 59.4,
#                  "12h": 77.5,   "24h": 130.5,  "48h": 269.9,  "72h": 423.3}

# R_EARTH  = 6371.0
# DT_HOURS = 6.0


# # ─────────────────────────────────────────────────────────────────────────────
# #  Utilities
# # ─────────────────────────────────────────────────────────────────────────────

# def _unwrap(m):
#     return m._orig_mod if hasattr(m, "_orig_mod") else m

# def move(batch, device):
#     out = list(batch)
#     for i, x in enumerate(out):
#         if torch.is_tensor(x):
#             out[i] = x.to(device)
#         elif isinstance(x, dict):
#             out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
#     return out

# def _ate_cte(pred_deg, gt_deg):
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         z = pred_deg.new_zeros(1, pred_deg.shape[1])
#         return z, z
#     lo1 = torch.deg2rad(gt_deg[:T-1,:,0]);  la1 = torch.deg2rad(gt_deg[:T-1,:,1])
#     lo2 = torch.deg2rad(gt_deg[1:T, :,0]);  la2 = torch.deg2rad(gt_deg[1:T, :,1])
#     lo3 = torch.deg2rad(pred_deg[1:T,:,0]); la3 = torch.deg2rad(pred_deg[1:T,:,1])
#     ya  = torch.sin(lo2-lo1)*torch.cos(la2)
#     xa  = torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
#     be  = torch.atan2(ya, xa)
#     ye  = torch.sin(lo3-lo2)*torch.cos(la3)
#     xe  = torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
#     bee = torch.atan2(ye, xe)
#     tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
#     ang = bee - be
#     return tot*torch.cos(ang), tot*torch.sin(ang)

# def _step_speeds_kmh(traj_deg):
#     if traj_deg.shape[0] < 2:
#         return traj_deg.new_zeros(1, traj_deg.shape[1])
#     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# # ─────────────────────────────────────────────────────────────────────────────
# #  Core evaluation — with full per-storm XAI collection
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def evaluate_full(
#     model, loader, device,
#     tag: str = "EVAL",
#     n_ensemble: int = 20,
#     ref_targets: Optional[dict] = None,
#     collect_xai: bool = False,
#     collect_per_storm: bool = False,
# ) -> Dict:
#     """
#     Đánh giá toàn diện với XAI collection.

#     collect_xai=True: thu thập XAI trên mỗi batch (chậm hơn)
#     collect_per_storm=True: lưu từng storm riêng (cho failure analysis)
#     """
#     model.eval()
#     ref = ref_targets or ST_TRANS_VAL

#     # Accumulators
#     all_ade, all_ate, all_cte = [], [], []
#     step_dist = defaultdict(list)
#     step_ate  = defaultdict(list)
#     step_cte  = defaultdict(list)

#     # XAI accumulators
#     all_obs_speed    = []   # [N] mean obs speed per storm
#     all_pred_speed   = []   # [N] mean pred speed per storm
#     all_hard_score   = []   # [N] hard score per storm
#     all_uncertainty  = []   # [N] 72h ensemble std per storm
#     all_heading_dev  = []   # [N] 72h heading deviation per storm
#     all_lat          = []   # [N] mean obs latitude per storm

#     # Per-storm records (for CSV export)
#     per_storm = [] if collect_per_storm else None

#     t0 = time.perf_counter()
#     n_batches = 0

#     for batch in loader:
#         bl = move(list(batch), device)
#         gt = bl[1]; B = bl[0].shape[1]
#         n_batches += 1

#         # ── Inference ───────────────────────────────────────────────────
#         try:
#             if collect_xai:
#                 pred, _, all_traj, xai_b = _unwrap(model).sample(
#                     bl, num_ensemble=n_ensemble, return_xai=True)
#             else:
#                 pred, _, all_traj = model.sample(bl, num_ensemble=n_ensemble)
#                 xai_b = None
#         except Exception as e:
#             print(f"  sample error: {e}"); continue

#         T   = min(pred.shape[0], gt.shape[0])
#         pd  = _norm_to_deg(pred[:T]);  gd  = _norm_to_deg(gt[:T])
#         dist = _haversine_deg(pd, gd)  # [T, B]
#         ate, cte = _ate_cte(pd, gd)    # [T-1, B]

#         all_ade.extend(dist.mean(0).tolist())
#         if ate.shape[0] > 0:
#             all_ate.extend(ate.abs().mean(0).tolist())
#             all_cte.extend(cte.abs().mean(0).tolist())

#         for h, s in HORIZON_STEPS.items():
#             if s < T:
#                 step_dist[h].extend(dist[s].tolist())
#             if s-1 < ate.shape[0]:
#                 step_ate[h].extend(ate.abs()[s-1].tolist())
#                 step_cte[h].extend(cte.abs()[s-1].tolist())

#         # ── XAI collection per batch ─────────────────────────────────────
#         obs_deg = _norm_to_deg(bl[0][:, :, :2])   # [T_obs, B, 2]

#         # Obs speed per storm
#         obs_spd_b = _step_speeds_kmh(obs_deg).mean(0)   # [B] km/h
#         all_obs_speed.extend(obs_spd_b.tolist())

#         # Pred speed per storm
#         last_deg = obs_deg[-1:]
#         pts_pred = torch.cat([last_deg, pd], 0)
#         pred_spd_b = _step_speeds_kmh(pts_pred).mean(0) # [B]
#         all_pred_speed.extend(pred_spd_b.tolist())

#         # Hard score
#         h_score = hard_score_from_obs(bl[0][:, :, :2])  # [B]
#         all_hard_score.extend(h_score.tolist())

#         # Ensemble uncertainty (72h std)
#         unc = compute_ensemble_uncertainty(all_traj)
#         std_72h = unc["std_per_step"][min(11, unc["std_per_step"].shape[0]-1)]  # [B]
#         all_uncertainty.extend(std_72h.tolist())

#         # Heading deviation (72h)
#         if pd.shape[0] >= 2 and obs_deg.shape[0] >= 2:
#             hdev = compute_heading_deviation(pd, obs_deg)  # [T, B]
#             hdev_72h = hdev[min(10, hdev.shape[0]-1)]      # [B]
#             all_heading_dev.extend(hdev_72h.tolist())
#         else:
#             all_heading_dev.extend([float("nan")] * B)

#         # Mean latitude
#         lat_mean = obs_deg[:, :, 1].mean(0)    # [B]
#         all_lat.extend(lat_mean.tolist())

#         # Per-storm records
#         if per_storm is not None:
#             for b in range(B):
#                 per_storm.append({
#                     "ade":          float(dist.mean(0)[b]),
#                     "ate":          float(ate.abs().mean(0)[b]) if ate.shape[0] > 0 else float("nan"),
#                     "cte":          float(cte.abs().mean(0)[b]) if cte.shape[0] > 0 else float("nan"),
#                     "ade_12h":      float(dist[HORIZON_STEPS[12]][b]) if HORIZON_STEPS[12] < T else float("nan"),
#                     "ade_24h":      float(dist[HORIZON_STEPS[24]][b]) if HORIZON_STEPS[24] < T else float("nan"),
#                     "ade_48h":      float(dist[HORIZON_STEPS[48]][b]) if HORIZON_STEPS[48] < T else float("nan"),
#                     "ade_72h":      float(dist[HORIZON_STEPS[72]][b]) if HORIZON_STEPS[72] < T else float("nan"),
#                     "obs_speed":    float(obs_spd_b[b]),
#                     "pred_speed":   float(pred_spd_b[b]),
#                     "hard_score":   float(h_score[b]),
#                     "uncertainty_72h": float(std_72h[b]),
#                     "heading_dev_72h": float(all_heading_dev[-B+b]),
#                     "lat_mean":     float(lat_mean[b]),
#                     "speed_ratio":  float(pred_spd_b[b] / max(obs_spd_b[b], 1.)),
#                     "storm_type": (
#                         "slow"   if obs_spd_b[b] < 8  else
#                         "fast"   if obs_spd_b[b] > 15 else "medium"
#                     ),
#                 })

#     elapsed = time.perf_counter() - t0

#     def _m(lst): return float(np.nanmean(lst)) if lst else float("nan")
#     def _p(lst, q): return float(np.nanpercentile(lst, q)) if lst else float("nan")

#     result = {
#         # Core metrics
#         "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
#         "n": len(all_ade), "n_batches": n_batches, "elapsed_s": elapsed,

#         # Horizon metrics
#         **{f"{h}h_ADE": _m(step_dist[h]) for h in HORIZON_STEPS},
#         **{f"{h}h_ATE": _m(step_ate[h])  for h in HORIZON_STEPS},
#         **{f"{h}h_CTE": _m(step_cte[h])  for h in HORIZON_STEPS},

#         # XAI summary
#         "obs_speed_mean":   _m(all_obs_speed),
#         "pred_speed_mean":  _m(all_pred_speed),
#         "speed_ratio_mean": _m([p/max(o,1.) for p,o in zip(all_pred_speed, all_obs_speed)]),
#         "hard_score_mean":  _m(all_hard_score),
#         "uncertainty_72h_mean": _m(all_uncertainty),
#         "heading_dev_72h_mean": _m(all_heading_dev),
#         "lat_mean":         _m(all_lat),

#         # Storm category breakdown
#         "n_slow":   sum(1 for s in all_obs_speed if s < 8.0),
#         "n_medium": sum(1 for s in all_obs_speed if 8.0 <= s < 15.0),
#         "n_fast":   sum(1 for s in all_obs_speed if s >= 15.0),

#         # Per-storm
#         "per_storm": per_storm,
#     }

#     # Combined score
#     ade, ate_, cte_ = result["ADE"], result["ATE"], result["CTE"]
#     result["combined"] = (
#         0.6*ade + 0.2*ate_ + 0.2*cte_
#         if all(np.isfinite(x) for x in [ade, ate_, cte_]) else ade)

#     # ── Print ────────────────────────────────────────────────────────────
#     def _ok(k):
#         v = result.get(k, float("nan"))
#         return "✓" if np.isfinite(v) and v < ref.get(k, 1e9) else "✗"

#     print(f"\n  {'='*72}")
#     print(f"  [{tag}]  n={result['n']}  ({elapsed:.1f}s)")
#     print(f"  ADE={ade:7.1f}km {_ok('ADE')}  "
#           f"ATE={ate_:7.1f}km {_ok('ATE')}  "
#           f"CTE={cte_:7.1f}km {_ok('CTE')}")
#     print(f"  Combined = {result['combined']:.1f}")
#     print(f"  12h={result['12h_ADE']:6.1f}  24h={result['24h_ADE']:6.1f}  "
#           f"48h={result['48h_ADE']:6.1f}  72h={result['72h_ADE']:6.1f} km")

#     beat = [f"{k}={result.get(k, result.get(k+'_ADE', float('nan'))):.0f}<{ref.get(k, 1e9):.0f}"
#             for k in ["ADE","ATE","CTE","12h_ADE","24h_ADE","48h_ADE","72h_ADE"]
#             if np.isfinite(result.get(k, float("nan")))
#             and result.get(k, 1e9) < ref.get(k.replace("_ADE","h"), 1e9)]
#     print(f"  BEAT: {' | '.join(beat) if beat else 'none'}")

#     print(f"\n  ── XAI Summary ──")
#     print(f"  Obs speed: {result['obs_speed_mean']:.1f}km/h  "
#           f"Pred speed: {result['pred_speed_mean']:.1f}km/h  "
#           f"Ratio: {result['speed_ratio_mean']:.3f}")
#     print(f"  Hard score: {result['hard_score_mean']:.3f}  "
#           f"Uncertainty 72h: {result['uncertainty_72h_mean']:.1f}km  "
#           f"Heading dev 72h: {result['heading_dev_72h_mean']:.1f}°")
#     print(f"  Storms: slow={result['n_slow']} "
#           f"medium={result['n_medium']} fast={result['n_fast']}")
#     print(f"  {'='*72}\n")

#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# #  Physical consistency analysis
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def analyze_physical_consistency(model, loader, device, n_ensemble: int = 20) -> Dict:
#     """
#     4 physical consistency tests:
#     1. Beta drift: lat cao → more NW track tendency?
#     2. Speed-lat correlation: bão ở lat thấp slower?
#     3. Heading consistency: pred heading follows obs heading trend?
#     4. Uncertainty-difficulty correlation: hard storms more uncertain?
#     """
#     model.eval()

#     records = []   # per-storm: lat, pred_heading_west, obs_speed, hard_score, uncertainty

#     for batch in loader:
#         bl = move(list(batch), device)
#         B  = bl[0].shape[1]

#         try:
#             pred, _, all_traj = model.sample(bl, num_ensemble=n_ensemble)
#         except Exception:
#             continue

#         obs_deg  = _norm_to_deg(bl[0][:, :, :2])   # [T_obs, B, 2]
#         pred_deg = _norm_to_deg(pred)               # [T, B, 2]

#         # Mean obs latitude
#         lat_mean = obs_deg[:, :, 1].mean(0)   # [B] degrees

#         # Predicted heading (first step)
#         if pred_deg.shape[0] >= 1 and obs_deg.shape[0] >= 1:
#             last_obs = obs_deg[-1]          # [B, 2]
#             first_pred = pred_deg[0]        # [B, 2]
#             d_lon = first_pred[:, 0] - last_obs[:, 0]  # negative = westward
#             d_lat = first_pred[:, 1] - last_obs[:, 1]  # positive = northward
#         else:
#             d_lon = torch.zeros(B, device=device)
#             d_lat = torch.zeros(B, device=device)

#         # Obs speed
#         obs_spd = _step_speeds_kmh(obs_deg).mean(0)   # [B]

#         # Hard score
#         h_score = hard_score_from_obs(bl[0][:, :, :2])

#         # Uncertainty
#         unc_data = compute_ensemble_uncertainty(all_traj)
#         unc_72h = unc_data["std_per_step"][min(11, unc_data["std_per_step"].shape[0]-1)]

#         for b in range(B):
#             records.append({
#                 "lat":         float(lat_mean[b]),
#                 "pred_d_lon":  float(d_lon[b]),   # negative = westward (good for TC)
#                 "pred_d_lat":  float(d_lat[b]),
#                 "obs_speed":   float(obs_spd[b]),
#                 "hard_score":  float(h_score[b]),
#                 "uncertainty": float(unc_72h[b]),
#             })

#     if not records:
#         return {}

#     lats      = np.array([r["lat"]         for r in records])
#     d_lons    = np.array([r["pred_d_lon"]  for r in records])
#     d_lats    = np.array([r["pred_d_lat"]  for r in records])
#     obs_spds  = np.array([r["obs_speed"]   for r in records])
#     hard_scrs = np.array([r["hard_score"]  for r in records])
#     uncs      = np.array([r["uncertainty"] for r in records])

#     # Test 1: Beta drift — lat cao → pred_d_lon more negative (westward)?
#     # Expected positive correlation: low lat → more westward (strongly negative d_lon)
#     # Actually: equatorial → W track, higher lat → NW → less westward
#     # So: corr(lat, d_lon) should be POSITIVE (lat up → d_lon less negative)
#     corr_beta = float(np.corrcoef(lats, d_lons)[0, 1]) if len(lats) > 2 else float("nan")

#     # Test 2: Lat-speed (lat higher → storms moving faster in extratropics)
#     corr_lat_speed = float(np.corrcoef(lats, obs_spds)[0, 1]) if len(lats) > 2 else float("nan")

#     # Test 3: Hard score → uncertainty (hard storms should be more uncertain)
#     corr_hard_unc = float(np.corrcoef(hard_scrs, uncs)[0, 1]) if len(hard_scrs) > 2 else float("nan")

#     # Test 4: Speed calibration ratio by storm type
#     slow_mask = obs_spds < 8.0; fast_mask = obs_spds >= 15.0; med_mask = ~slow_mask & ~fast_mask

#     result = {
#         "n_storms":          len(records),
#         # Beta drift test
#         "corr_lat_dlon":     corr_beta,
#         "beta_drift_ok":     corr_beta > 0,  # Expected positive
#         # Speed-latitude
#         "corr_lat_speed":    corr_lat_speed,
#         # Hard-uncertainty
#         "corr_hard_unc":     corr_hard_unc,
#         "calib_ok":          corr_hard_unc > 0.1,  # Hard → more uncertain
#         # Category counts
#         "n_slow": int(slow_mask.sum()), "n_medium": int(med_mask.sum()), "n_fast": int(fast_mask.sum()),
#         # Category stats
#         "lat_q25": float(np.percentile(lats, 25)),
#         "lat_q75": float(np.percentile(lats, 75)),
#         "speed_low_lat":  float(obs_spds[lats < np.median(lats)].mean()) if (lats < np.median(lats)).any() else float("nan"),
#         "speed_high_lat": float(obs_spds[lats >= np.median(lats)].mean()) if (lats >= np.median(lats)).any() else float("nan"),
#     }

#     print(f"\n  ── Physical Consistency Analysis ──")
#     print(f"  n_storms = {result['n_storms']}")
#     print(f"  [BETA]  corr(lat, pred_d_lon) = {corr_beta:+.3f}  "
#           f"{'✅ consistent' if result['beta_drift_ok'] else '⚠ unexpected'}")
#     print(f"  [SPEED] corr(lat, obs_speed)  = {corr_lat_speed:+.3f}  "
#           f"(expect +: high lat storms faster)")
#     print(f"  [CALIB] corr(hard_score, uncertainty) = {corr_hard_unc:+.3f}  "
#           f"{'✅ calibrated' if result['calib_ok'] else '⚠ uncalibrated'}")
#     print(f"  [SPD]   low-lat={result['speed_low_lat']:.1f}km/h  "
#           f"high-lat={result['speed_high_lat']:.1f}km/h")

#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# #  CRPS (uncertainty calibration)
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def compute_crps(model, loader, device, n_ensemble: int = 20) -> Dict:
#     """
#     CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
#     X = ensemble members, y = ground truth
#     Lower CRPS = better calibrated uncertainty.
#     """
#     model.eval()
#     crps_vals = []

#     for batch in loader:
#         bl = move(list(batch), device)
#         gt = bl[1]

#         try:
#             # Sample multiple times for CRPS
#             preds = []
#             for _ in range(n_ensemble):
#                 p, _, _ = model.sample(bl, num_ensemble=1)
#                 preds.append(p)
#             preds = torch.stack(preds, 0)   # [K, T, B, 2]
#         except Exception:
#             continue

#         T = min(preds.shape[1], gt.shape[0])
#         K = preds.shape[0]
#         B = preds.shape[2]

#         pred_deg = _norm_to_deg(preds[:, :T])    # [K, T, B, 2]
#         gt_deg   = _norm_to_deg(gt[:T])          # [T, B, 2]

#         # E[|X - y|]: mean distance from each ensemble member to GT
#         gt_exp   = gt_deg.unsqueeze(0).expand(K, -1, -1, -1)  # [K, T, B, 2]
#         dist_xy  = _haversine_deg(
#             pred_deg.reshape(K*T*B, 2),
#             gt_exp.reshape(K*T*B, 2)
#         ).reshape(K, T, B).mean(0)   # [T, B]
#         e_dist_xy = dist_xy.mean(0)  # [B]

#         # E[|X - X'|]: mean pairwise distance between ensemble members
#         pairwise = []
#         for k1 in range(0, K, 2):
#             for k2 in range(k1+1, min(k1+4, K)):
#                 d = _haversine_deg(
#                     pred_deg[k1].reshape(T*B, 2),
#                     pred_deg[k2].reshape(T*B, 2)
#                 ).reshape(T, B).mean(0)
#                 pairwise.append(d)
#         e_dist_xx = torch.stack(pairwise).mean(0) if pairwise else torch.zeros(B, device=device)

#         crps_b = e_dist_xy - 0.5 * e_dist_xx   # [B]
#         crps_vals.extend(crps_b.tolist())

#     crps_mean = float(np.mean(crps_vals)) if crps_vals else float("nan")
#     print(f"\n  CRPS = {crps_mean:.2f}km  (lower = better calibrated, n={len(crps_vals)})")
#     return {"crps_mean": crps_mean, "crps_vals": crps_vals, "n": len(crps_vals)}


# # ─────────────────────────────────────────────────────────────────────────────
# #  Failure mode analysis
# # ─────────────────────────────────────────────────────────────────────────────

# def analyze_failure_modes(per_storm: List[Dict]) -> Dict:
#     """
#     Phân tích failure modes từ per_storm records.
#     Trả về summary cho paper.
#     """
#     if not per_storm:
#         return {}

#     # Sort by ADE (worst first)
#     sorted_storms = sorted(per_storm, key=lambda x: x.get("ade", 0), reverse=True)
#     n = len(sorted_storms)

#     # Top/bottom 10%
#     n10 = max(1, n // 10)
#     worst  = sorted_storms[:n10]
#     best   = sorted_storms[-n10:]

#     def _avg(lst, key): return float(np.nanmean([s.get(key, float("nan")) for s in lst]))

#     # Category breakdown
#     by_type = {"slow": [], "medium": [], "fast": []}
#     for s in per_storm:
#         t = s.get("storm_type", "medium")
#         by_type.get(t, by_type["medium"]).append(s)

#     result = {
#         "n_total": n,
#         "n_worst": n10,
#         # Overall
#         "ade_mean":  _avg(per_storm, "ade"),
#         "ate_mean":  _avg(per_storm, "ate"),
#         "cte_mean":  _avg(per_storm, "cte"),
#         # Worst cases
#         "worst_ade_mean":        _avg(worst, "ade"),
#         "worst_ate_mean":        _avg(worst, "ate"),
#         "worst_cte_mean":        _avg(worst, "cte"),
#         "worst_hard_score_mean": _avg(worst, "hard_score"),
#         "worst_uncertainty_mean":_avg(worst, "uncertainty_72h"),
#         "worst_obs_speed_mean":  _avg(worst, "obs_speed"),
#         # Best cases
#         "best_ade_mean": _avg(best, "ade"),
#         # By storm category
#         "slow_ade":   _avg(by_type["slow"],   "ade") if by_type["slow"]   else float("nan"),
#         "medium_ade": _avg(by_type["medium"], "ade") if by_type["medium"] else float("nan"),
#         "fast_ade":   _avg(by_type["fast"],   "ade") if by_type["fast"]   else float("nan"),
#         "slow_cte":   _avg(by_type["slow"],   "cte") if by_type["slow"]   else float("nan"),
#         "medium_cte": _avg(by_type["medium"], "cte") if by_type["medium"] else float("nan"),
#         "fast_cte":   _avg(by_type["fast"],   "cte") if by_type["fast"]   else float("nan"),
#         "n_slow":   len(by_type["slow"]),
#         "n_medium": len(by_type["medium"]),
#         "n_fast":   len(by_type["fast"]),
#     }

#     print(f"\n  ── Failure Mode Analysis ──")
#     print(f"  Total storms: {n}")
#     print(f"  Worst {n10} storms: ADE={result['worst_ade_mean']:.1f}km  "
#           f"hard_score={result['worst_hard_score_mean']:.3f}  "
#           f"uncertainty={result['worst_uncertainty_mean']:.1f}km  "
#           f"speed={result['worst_obs_speed_mean']:.1f}km/h")
#     print(f"  By category:")
#     print(f"    slow  (n={result['n_slow']:3d}): ADE={result['slow_ade']:6.1f}km  CTE={result['slow_cte']:6.1f}km")
#     print(f"    medium(n={result['n_medium']:3d}): ADE={result['medium_ade']:6.1f}km  CTE={result['medium_cte']:6.1f}km")
#     print(f"    fast  (n={result['n_fast']:3d}): ADE={result['fast_ade']:6.1f}km  CTE={result['fast_cte']:6.1f}km")

#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# #  Args
# # ─────────────────────────────────────────────────────────────────────────────

# def get_args():
#     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     p.add_argument("--checkpoint",       required=True,  help="Path to model checkpoint")
#     p.add_argument("--dataset_root",     default="/kaggle/input/datasets/kaggle1234uitvn/tc-ofm")
#     p.add_argument("--split",            default="test", choices=["val","test","both"])
#     p.add_argument("--output_dir",       default="runs/xai_results")
#     p.add_argument("--obs_len",          default=8,   type=int)
#     p.add_argument("--pred_len",         default=12,  type=int)
#     p.add_argument("--batch_size",        default=60,  type=int,
#                    help="Batch size for eval DataLoader")
#     p.add_argument("--num_workers",      default=2,   type=int)
#     p.add_argument("--other_modal",      default="gph")
#     p.add_argument("--delim",            default=" ")
#     p.add_argument("--skip",             default=1,   type=int)
#     p.add_argument("--min_ped",          default=1,   type=int)
#     p.add_argument("--threshold",        default=0.002, type=float)
#     p.add_argument("--d_cond",           default=256, type=int)
#     p.add_argument("--d_model",          default=256, type=int)
#     p.add_argument("--nhead",            default=8,   type=int)
#     p.add_argument("--num_dec_layers",   default=4,   type=int)
#     p.add_argument("--dim_ff",           default=512, type=int)
#     p.add_argument("--dropout",          default=0.1, type=float)
#     p.add_argument("--unet_in_ch",       default=13,  type=int)
#     p.add_argument("--sigma_min",        default=0.04, type=float)
#     p.add_argument("--sigma_max",        default=0.08, type=float)
#     p.add_argument("--lambda_reg",       default=0.2, type=float)
#     p.add_argument("--lambda_heading",   default=0.10, type=float)
#     p.add_argument("--use_ot",           default=True, action="store_true")
#     p.add_argument("--ot_epsilon",       default=0.05, type=float)
#     p.add_argument("--n_ensemble",       default=30,  type=int,
#                    help="K ensemble samples (default 30 for v2.1-XAI)")
#     p.add_argument("--sigma_inference",  default=0.055, type=float,
#                    help="Inference sigma (0.055 for diversity, was 0.04)")
#     p.add_argument("--clip_min",         default=0.72, type=float,
#                    help="Speed calibration clip_min (0.72 for wider range)")
#     p.add_argument("--clip_max",         default=1.35, type=float,
#                    help="Speed calibration clip_max (1.35 for wider range)")
#     p.add_argument("--n_inference_steps",default=1,   type=int,
#                    help="NFE for FM inference (1=1-shot)")
#     p.add_argument("--gpu_num",          default="0")
#     p.add_argument("--run_crps",         action="store_true", default=False,
#                    help="Compute CRPS (slow, requires n_ensemble² samples)")
#     p.add_argument("--run_physics",      action="store_true", default=True)
#     p.add_argument("--run_failure",      action="store_true", default=True)
#     p.add_argument("--export_csv",       action="store_true", default=True)
#     return p.parse_args()


# # ─────────────────────────────────────────────────────────────────────────────
# #  Main
# # ─────────────────────────────────────────────────────────────────────────────

# def main(args):
#     if torch.cuda.is_available():
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output_dir, exist_ok=True)

#     print("=" * 72)
#     print("  TC-FlowMatching v2.1-XAI — Evaluation & XAI Analysis")
#     print(f"  Checkpoint: {args.checkpoint}")
#     print(f"  Split: {args.split}  |  Device: {device}")
#     print("=" * 72)

#     # ── Load model ────────────────────────────────────────────────────────
#     model = TCFlowMatching(
#         pred_len=args.pred_len,      obs_len=args.obs_len,
#         unet_in_ch=args.unet_in_ch,  d_cond=args.d_cond,
#         d_model=args.d_model,        nhead=args.nhead,
#         num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
#         dropout=args.dropout,
#         sigma_min=args.sigma_min,    sigma_max=args.sigma_max,
#         lambda_reg=args.lambda_reg,  lambda_heading=args.lambda_heading,
#         lambda_momentum=0.0,
#         use_ot=args.use_ot,          ot_epsilon=args.ot_epsilon,
#         n_ensemble=args.n_ensemble,
#         n_inference_steps=args.n_inference_steps,
#         sigma_inference=args.sigma_inference,
#     ).to(device)

#     ck = torch.load(args.checkpoint, map_location=device)
#     _unwrap(model).load_state_dict(ck["model"], strict=False)
#     print(f"\n  Loaded checkpoint ep{ck.get('epoch','?')}  "
#           f"(is_swa={ck.get('is_swa', False)})")
#     print(f"  Saved val ADE={ck.get('val_ade', '?')}  "
#           f"ATE={ck.get('val_ate', '?')}  "
#           f"CTE={ck.get('val_cte', '?')}")
#     # Override inference params (quick-win, no retrain needed)
#     # Evidence: XAI-8 24h ratio=1.61 → clip_min=0.72 needed (old 0.85 truncated)
#     # Evidence: XAI-4 72h_std=3.8km → sigma=0.055 for diversity
#     raw = _unwrap(model)
#     raw.sigma_inference = args.sigma_inference
#     raw.n_ensemble      = args.n_ensemble
#     print(f"  Inference override: sigma={args.sigma_inference}  "
#           f"K={args.n_ensemble}  clip=[{args.clip_min},{args.clip_max}]")

#     # Monkey-patch speed calibration clip range (inference-only quick-win)
#     _orig_calib = raw.speed_calibrate_pred.__func__ if hasattr(raw.speed_calibrate_pred, '__func__') else None
#     _clip_min, _clip_max = args.clip_min, args.clip_max
#     import types
#     def _patched_calib(self, pred_mean, last_obs, obs_norm,
#                        clip_min=_clip_min, clip_max=_clip_max):
#         from Model.flow_matching_model import _norm_to_deg, _step_speeds_kmh
#         import torch
#         obs_deg  = _norm_to_deg(obs_norm)
#         last_deg = _norm_to_deg(last_obs)
#         pred_deg = _norm_to_deg(pred_mean)
#         obs_spd  = _step_speeds_kmh(obs_deg)
#         T_sc = obs_spd.shape[0]
#         w    = torch.exp(torch.arange(T_sc, dtype=obs_spd.dtype, device=obs_spd.device) * 0.5)
#         obs_spd_ref = (obs_spd * (w / w.sum()).unsqueeze(1)).sum(0).clamp(min=3.0)
#         pts  = torch.cat([last_deg.unsqueeze(0), pred_deg], 0)
#         from Model.flow_matching_model import _step_speeds_kmh as _spd
#         pred_spd = _spd(pts)
#         pred_spd_ref = pred_spd[:min(4, pred_spd.shape[0])].mean(0).clamp(min=3.0)
#         scale = (obs_spd_ref / pred_spd_ref).clamp(0.0, 10.0).clamp(clip_min, clip_max)
#         return last_obs.unsqueeze(0) + (pred_mean - last_obs.unsqueeze(0)) * scale.view(1, -1, 1)
#     raw.speed_calibrate_pred = types.MethodType(_patched_calib, raw)

#     # ── Load data ─────────────────────────────────────────────────────────
#     loaders = {}
#     if args.split in ("val", "both"):
#         _, loaders["val"] = data_loader(
#             args, {"root": args.dataset_root, "type": "val"}, test=True)
#         print(f"  val: {len(loaders['val'])} batches")
#     if args.split in ("test", "both"):
#         try:
#             _, loaders["test"] = data_loader(
#                 args, {"root": args.dataset_root, "type": "test"}, test=True)
#             print(f"  test: {len(loaders['test'])} batches")
#         except Exception as e:
#             print(f"  test loader error: {e} — using val")
#             loaders["test"] = loaders.get("val")

#     all_results = {}

#     # ── Evaluation loop per split ─────────────────────────────────────────
#     for split, loader in loaders.items():
#         if loader is None: continue
#         ref = ST_TRANS_TEST if split == "test" else ST_TRANS_VAL

#         print(f"\n{'═'*72}")
#         print(f"  EVALUATING: {split.upper()}")
#         print(f"{'═'*72}")

#         r = evaluate_full(
#             model, loader, device,
#             tag=split.upper(),
#             n_ensemble=args.n_ensemble,
#             ref_targets=ref,
#             collect_xai=True,
#             collect_per_storm=args.run_failure,
#         )
#         all_results[split] = r

#         # Physics consistency
#         if args.run_physics:
#             phys = analyze_physical_consistency(
#                 model, loader, device, n_ensemble=args.n_ensemble)
#             all_results[f"{split}_physics"] = phys

#         # CRPS (optional, slow)
#         if args.run_crps:
#             crps = compute_crps(model, loader, device, n_ensemble=args.n_ensemble)
#             all_results[f"{split}_crps"] = crps

#         # Failure mode analysis
#         if args.run_failure and r.get("per_storm"):
#             fail = analyze_failure_modes(r["per_storm"])
#             all_results[f"{split}_failure"] = fail

#     # ── ST-Trans comparison ───────────────────────────────────────────────
#     print(f"\n{'='*72}")
#     print(f"  COMPARISON vs ST-Trans")
#     print(f"{'='*72}")
#     print(f"  {'Metric':<12} {'ST-Trans':>12} {'v2.1-XAI':>12} {'Δ':>8} {'Status'}")
#     print(f"  {'─'*55}")
#     for split in ["val", "test"]:
#         if split not in all_results: continue
#         r = all_results[split]
#         ref = ST_TRANS_TEST if split == "test" else ST_TRANS_VAL
#         print(f"\n  [{split.upper()}]")
#         for k in ["ADE", "ATE", "CTE"]:
#             v = r.get(k, float("nan"))
#             st = ref.get(k, float("nan"))
#             delta = v - st
#             flag = "✅ BEAT" if delta < 0 else f"  +{delta:.1f}km"
#             print(f"  {k:<12} {st:>12.1f} {v:>12.1f} {delta:>+8.1f}  {flag}")

#     # ── Export ────────────────────────────────────────────────────────────
#     out_path = os.path.join(args.output_dir, "results.json")
#     export = {}
#     for k, v in all_results.items():
#         if isinstance(v, dict):
#             export[k] = {
#                 kk: vv if not isinstance(vv, (list, torch.Tensor)) else
#                     (vv[:5] if isinstance(vv, list) else str(vv.shape))
#                 for kk, vv in v.items()
#                 if kk != "per_storm"   # per_storm exported separately
#             }
#     with open(out_path, "w") as f:
#         json.dump(export, f, indent=2, default=str)
#     print(f"\n  Results saved: {out_path}")

#     # CSV per-storm export
#     if args.export_csv:
#         for split in loaders:
#             r = all_results.get(split, {})
#             ps = r.get("per_storm")
#             if ps:
#                 import csv
#                 csv_path = os.path.join(args.output_dir, f"per_storm_{split}.csv")
#                 with open(csv_path, "w", newline="") as f:
#                     writer = csv.DictWriter(f, fieldnames=ps[0].keys())
#                     writer.writeheader(); writer.writerows(ps)
#                 print(f"  Per-storm CSV: {csv_path}  ({len(ps)} storms)")


# if __name__ == "__main__":
#     args = get_args()
#     if args.dataset_root == "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm":
#         if not os.path.isdir(args.dataset_root):
#             args.dataset_root = "TCND_vn"
#     main(args)

"""
evaluate.py — TC-FlowMatching evaluation script
════════════════════════════════════════════════════════════════════════════════

THIẾT KẾ: Script này KHÔNG override bất kỳ inference parameter nào
(sigma, clip, K...) vì từ v2.1-learn các tham số đó đã là LEARNABLE
(speed_correction_logits, log_sigma_reg/heading/calib...) và được lưu
trong checkpoint. Override thủ công sẽ BỎ QUA những gì model đã học.

Chỉ cần: --checkpoint path/to/best_model.pth --dataset_root ...
"""
from __future__ import annotations

import sys, os, argparse, time, json
import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import defaultdict
from typing import Dict

import numpy as np
import torch

from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, EMAModel,
    _norm_to_deg, _haversine_deg, _forward_azimuth,
    _unwrap,
)

# ─────────────────────────────────────────────────────────────────────────────
ST_TRANS = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
            "12h": 65.0, "24h": 130.0, "48h": 205.0, "72h": 321.0}

HORIZON_STEPS = {"12h": 1, "24h": 3, "48h": 7, "72h": 11}

# ─────────────────────────────────────────────────────────────────────────────

def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


def _ate_cte(pred_deg, gt_deg):
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    bear_ref = _forward_azimuth(gt_deg[:T-1], gt_deg[1:T])
    bear_err = _forward_azimuth(gt_deg[1:T], pred_deg[1:T])
    dist_err = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang = bear_err - bear_ref
    return dist_err * torch.cos(ang), dist_err * torch.sin(ang)


@torch.no_grad()
def run_evaluation(model, loader, device, tag="TEST",
                    n_ensemble=20, ema=None, use_tta=False) -> Dict:
    """
    Chạy evaluation trên loader.
    - Không augmentation
    - Dùng model.sample() với các learnable params đã có từ checkpoint
    - KHÔNG override sigma/K/clip
    """
    bk = None
    if ema is not None:
        try: bk = ema.apply_to(model)
        except Exception as e: print(f"  ⚠ EMA apply: {e}")

    model.eval()
    all_ade, all_ate, all_cte = [], [], []
    step_dist = defaultdict(list)
    n_batches = len(loader)

    t0 = time.time()
    for i, batch in enumerate(loader):
        bl = move(list(batch), device)
        gt = bl[1]; B = bl[0].shape[1]

        if use_tta:
            # TTA: scale obs speed x5 levels, weighted average
            obs = bl[0]; anchor = obs[-1:, :, :2].detach()
            scales = [0.875, 0.9375, 1.0, 1.0625, 1.125]
            preds, weights = [], []
            for sc in scales:
                obs_s = obs.clone()
                obs_s[..., :2] = anchor + (obs[..., :2] - anchor) * sc
                bl_s = list(bl); bl_s[0] = obs_s
                try:
                    p, _, _ = model.sample(bl_s, num_ensemble=n_ensemble)
                    preds.append(p)
                    weights.append(2.0 if abs(sc - 1.0) < 1e-6 else 1.0)
                except Exception: continue
            if not preds: continue
            tw = sum(weights)
            pred = sum(w / tw * p for w, p in zip(weights, preds))
        else:
            try:
                pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
            except Exception as e:
                print(f"  [batch {i}] sample error: {e}"); continue

        T = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
        dist = _haversine_deg(pd, gd)
        ate, cte = _ate_cte(pd, gd)

        all_ade.extend(dist.mean(0).tolist())
        if ate.shape[0] > 0:
            all_ate.extend(ate.abs().mean(0).tolist())
            all_cte.extend(cte.abs().mean(0).tolist())
        for h, s in HORIZON_STEPS.items():
            if s < T: step_dist[h].extend(dist[s].tolist())

    if bk is not None:
        try: ema.restore(model, bk)
        except Exception: pass

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")
    elapsed = time.time() - t0

    result = {
        "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
        "n":   len(all_ade), "time_s": elapsed,
    }
    for h in HORIZON_STEPS: result[h] = _m(step_dist[h])

    # Print
    def _ok(k):
        v = result.get(k, float("nan"))
        return "✓" if np.isfinite(v) and v < ST_TRANS.get(k, 1e9) else "✗"

    tta_tag = " [TTA]" if use_tta else ""
    print(f"\n  {'='*72}")
    print(f"  [{tag}]{tta_tag}  n={result['n']}  ({elapsed:.1f}s)")
    print(f"  ADE={result['ADE']:7.1f}km {_ok('ADE')}  "
          f"ATE={result['ATE']:7.1f}km {_ok('ATE')}  "
          f"CTE={result['CTE']:7.1f}km {_ok('CTE')}")
    print(f"  12h={result['12h']:6.1f}  24h={result['24h']:6.1f}  "
          f"48h={result['48h']:6.1f}  72h={result['72h']:6.1f} km")
    beat = [f"{k}={result.get(k,999):.0f}<{ST_TRANS.get(k,999):.0f}"
            for k in ["ADE","ATE","CTE","12h","24h","48h","72h"]
            if np.isfinite(result.get(k, float("nan")))
            and result.get(k, 999) < ST_TRANS.get(k, 1e9)]
    print(f"  BEAT: {' | '.join(beat) if beat else 'none yet'}")

    # So sánh với ST-Trans
    print(f"\n  COMPARISON vs ST-Trans:")
    print(f"  {'Metric':<8} {'ST-Trans':>10} {'Model':>10} {'Δ':>8} Status")
    print(f"  {'─'*50}")
    for k in ["ADE","ATE","CTE"]:
        v = result.get(k, float("nan"))
        ref = ST_TRANS[k]
        delta = v - ref
        status = "✓ BEAT" if v < ref else "✗"
        print(f"  {k:<8} {ref:>10.1f} {v:>10.1f} {delta:>+8.1f}km  {status}")
    print(f"  {'='*72}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="TC-FlowMatching evaluation")
    parser.add_argument("--checkpoint",    required=True, help="Path to checkpoint .pth")
    parser.add_argument("--dataset_root",  required=True, help="Dataset root directory")
    parser.add_argument("--split",         default="test", choices=["test","val","train"])
    parser.add_argument("--n_ensemble",    type=int,   default=20,
                        help="K ensemble samples (default: 20, same as training)")
    parser.add_argument("--use_ema",       action="store_true", default=True,
                        help="Use EMA weights if available in checkpoint (default: True)")
    parser.add_argument("--no_ema",        action="store_true",
                        help="Force disable EMA even if available")
    parser.add_argument("--tta",           action="store_true",
                        help="Test-time augmentation (speed scale TTA)")
    parser.add_argument("--save_json",     type=str, default="",
                        help="Save results to JSON file")
    parser.add_argument("--gpu",           type=int, default=0)
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n  Checkpoint: {args.checkpoint}")
    print(f"  Split:      {args.split}  |  Device: {device}")
    print(f"  n_ensemble: {args.n_ensemble} (model defaults used, no override)")
    print("="*72)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ck = torch.load(args.checkpoint, map_location="cpu")
    is_swa = ck.get("is_swa", False)

    # Reconstruct model từ checkpoint config
    model_cfg = ck.get("model_cfg", {})
    model = TCFlowMatching(**model_cfg).to(device)

    # Load weights
    state = ck.get("model", ck)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"  ⚠ Missing keys: {missing[:5]}...")
    if unexpected: print(f"  ⚠ Unexpected keys: {unexpected[:5]}...")

    ep = ck.get("epoch", "?")
    print(f"  Loaded ep{ep} (is_swa={is_swa})")

    # In các giá trị đã học để kiểm tra
    raw = _unwrap(model)
    import torch.nn.functional as F
    if hasattr(raw, "speed_correction_logits"):
        corr = (torch.sigmoid(raw.speed_correction_logits) * 2.0).tolist()
        print(f"  [LEARN] speed_correction (12h-72h): {[f'{v:.3f}' for v in corr[:4]]}...")
    if hasattr(raw, "log_sigma_reg"):
        print(f"  [LEARN] log_sigma: reg={raw.log_sigma_reg.item():.3f}  "
              f"heading={raw.log_sigma_heading.item():.3f}  "
              f"calib={raw.log_sigma_calib.item():.3f}")
        prec_r = torch.exp(-2.0*raw.log_sigma_reg.clamp(min=-3.0))
        prec_h = torch.exp(-2.0*raw.log_sigma_heading.clamp(min=-3.0))
        prec_c = torch.exp(-2.0*raw.log_sigma_calib.clamp(min=-3.0))
        print(f"  [LEARN] eff_lambda: reg={0.5*prec_r.item():.3f}  "
              f"heading={0.5*prec_h.item():.3f}  "
              f"calib={0.5*prec_c.item():.3f}")

    # ── EMA ──────────────────────────────────────────────────────────────────
    ema = None
    use_ema = args.use_ema and not args.no_ema and not is_swa
    if use_ema and ck.get("ema"):
        try:
            ema = EMAModel(model)
            for k, v in ck["ema"].items():
                if k in ema.shadow:
                    ema.shadow[k].copy_(v.to(device))
            print(f"  EMA loaded ({len(ema.shadow)} params)")
        except Exception as e:
            print(f"  ⚠ EMA load failed: {e}"); ema = None

    # ── Data ──────────────────────────────────────────────────────────────────
    try:
        _, loader = data_loader(
            argparse.Namespace(dataset_root=args.dataset_root),
            {"root": args.dataset_root, "type": args.split},
            test=(args.split != "train"))
        print(f"  {args.split}: {sum(1 for _ in loader)*loader.batch_size} sequences, "
              f"{len(loader)} batches")
    except Exception as e:
        print(f"  Data loader error: {e}"); return

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = {}

    # Standard
    r = run_evaluation(model, loader, device,
                       tag=f"{args.split.upper()} (ep{ep})",
                       n_ensemble=args.n_ensemble, ema=ema, use_tta=False)
    results["standard"] = {k: v for k, v in r.items()
                            if isinstance(v, (int, float))}

    # TTA nếu yêu cầu
    if args.tta:
        r_tta = run_evaluation(model, loader, device,
                                tag=f"{args.split.upper()} TTA",
                                n_ensemble=args.n_ensemble, ema=ema, use_tta=True)
        results["tta"] = {k: v for k, v in r_tta.items()
                           if isinstance(v, (int, float))}
        print(f"  TTA Δ: ADE {r['ADE']:.1f}→{r_tta['ADE']:.1f}km  "
              f"ATE {r['ATE']:.1f}→{r_tta['ATE']:.1f}km  "
              f"CTE {r['CTE']:.1f}→{r_tta['CTE']:.1f}km")

    # Save JSON
    if args.save_json:
        results["checkpoint"] = args.checkpoint
        results["split"] = args.split
        results["epoch"] = ep
        results["n_ensemble"] = args.n_ensemble
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {args.save_json}")


if __name__ == "__main__":
    main()
# """
# evaluate_multi_model.py
# =========================
# Runs test-set evaluation for RNN / GRU / LSTM / ST-Trans / FM checkpoints,
# using ONE SHARED ATE/CTE formula for all five models.

# WHY A SEPARATE SCRIPT, NOT REUSING EACH MODEL'S OWN METRIC CODE
# -----------------------------------------------------------------
# paper_baseline_model.py (used by RNN/GRU/LSTM/ST-Trans) has its own
# _ate_cte_tensors() with a DIFFERENT convention than flow_matching_model.py's
# _ate_cte_full() (used by FM):
#   - flat-earth approx (111*cos(lat)) vs haversine — small effect (~0.2%)
#   - reference heading computed OUTGOING from each point (gt[t]->gt[t+1])
#     vs INCOMING to each point (gt[t-1]->gt[t]) — this is NOT a small
#     effect: verified numerically on a synthetic turning trajectory, the
#     two conventions can even disagree on the SIGN of CTE at a given step.
# Comparing FM's ATE/CTE (computed one way) against RNN/GRU/LSTM/ST-Trans's
# ATE/CTE (computed the other way) would not be a fair like-for-like
# comparison — any difference could be an artifact of the formula, not the
# model. This script computes ADE/ATE/CTE for ALL FIVE models using the
# SAME function (_ate_cte_full, _haversine_deg, _forward_azimuth, imported
# from flow_matching_model.py — the version with the off-by-one fix already
# verified in evaluate_full.py), so Table-10-style comparisons are sound.

# USAGE
# -----
# python evaluate_multi_model.py \
#     --dataset_root <root> \
#     --fm_checkpoint runs/fm_seed42/best_model.pth \
#     --st_trans_checkpoint runs/st_trans/best_model.pth \
#     --lstm_checkpoint runs/lstm/best_model.pth \
#     --gru_checkpoint runs/gru/best_model.pth \
#     --rnn_checkpoint runs/rnn/best_model.pth \
#     --output_dir eval_multi/

# Any checkpoint arg can be omitted to skip that model. Produces one
# combined JSON of per-window records (one row per (model, storm, window))
# that generate_comparison_table.py consumes directly for the Table-10-style
# statistical significance tables.
# """
# from __future__ import annotations
# import sys, os, argparse, json
# from typing import Dict, List, Optional

# import numpy as np
# import torch

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from Model.data.loader_training import data_loader
# from Model.flow_matching_model import (
#     TCFlowMatching, _norm_to_deg, _haversine_deg, _forward_azimuth, _unwrap,
# )
# from Model.paper_baseline_model import PaperBaseline, MODEL_TYPES
# from Model.st_trans_model import STTrans


# def move(batch, device):
#     return [x.to(device) if torch.is_tensor(x) else x for x in batch]


# def ate_cte_full(pred_deg: torch.Tensor, gt_deg: torch.Tensor):
#     """
#     SAME formula as evaluate_full.py's _ate_cte_full (off-by-one already
#     fixed there — see that file's own comments for the fix history).
#     Returns [T-1, B] each; index k holds the error at ORIGINAL step k+1.
#     """
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         z = pred_deg.new_zeros(1, pred_deg.shape[1])
#         return z, z
#     bear_ref = _forward_azimuth(gt_deg[:T - 1], gt_deg[1:T])
#     bear_err = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
#     dist_err = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
#     ang = bear_err - bear_ref
#     return dist_err * torch.cos(ang), dist_err * torch.sin(ang)


# def load_fm(checkpoint: str, device):
#     ck = torch.load(checkpoint, map_location="cpu")
#     model_cfg = ck.get("model_cfg") or {}
#     if not model_cfg:
#         print(f"  ⚠ FM checkpoint has no model_cfg — using constructor "
#               f"defaults (correct only if trained with default architecture).")
#     model = TCFlowMatching(**model_cfg).to(device)
#     state = ck.get("model", ck.get("model_state"))
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     if missing or unexpected:
#         print(f"  ⚠ FM load_state_dict: {len(missing)} missing, "
#               f"{len(unexpected)} unexpected keys")
#     model.eval()
#     return model


# def load_paper_baseline(checkpoint: str, model_type: str, device,
#                          hidden_dim: int = 256, n_layers: int = 3,
#                          obs_len: int = 8, pred_len: int = 12,
#                          unet_in_ch: int = 13, dropout: float = 0.20):
#     """
#     [UPDATED] train_paper_baseline.py now saves a full model_cfg dict in
#     newer checkpoints (added alongside --seed support). If present, it is
#     used directly and the CLI-default args above are ignored — this is
#     the reliable path. If absent (checkpoint trained before that fix),
#     falls back to the CLI-default-matching values passed in here, with a
#     warning, same caveat as FM's missing-model_cfg case.
#     """
#     assert model_type in MODEL_TYPES, f"model_type must be one of {MODEL_TYPES}"
#     ck = torch.load(checkpoint, map_location="cpu")
#     saved_type = ck.get("model_type", model_type)
#     if saved_type != model_type:
#         print(f"  ⚠ Checkpoint's saved model_type='{saved_type}' differs "
#               f"from requested '{model_type}' — using requested value. "
#               f"Verify this checkpoint is really the {model_type.upper()} one.")

#     model_cfg = ck.get("model_cfg")
#     if model_cfg:
#         model = PaperBaseline(**model_cfg).to(device)
#     else:
#         print(f"  ⚠ {model_type.upper()} checkpoint has no model_cfg — "
#               f"using CLI-default-matching args (hidden_dim={hidden_dim}, "
#               f"n_layers={n_layers}, dropout={dropout}). Only correct if "
#               f"trained with train_paper_baseline.py's own defaults.")
#         model = PaperBaseline(model_type=model_type, pred_len=pred_len,
#                                obs_len=obs_len, hidden_dim=hidden_dim,
#                                n_layers=n_layers, unet_in_ch=unet_in_ch,
#                                dropout=dropout).to(device)
#     state = ck.get("model_state", ck.get("model"))
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     if missing or unexpected:
#         print(f"  ⚠ {model_type.upper()} load_state_dict: {len(missing)} "
#               f"missing, {len(unexpected)} unexpected keys")
#     model.eval()
#     return model


# def load_st_trans(checkpoint: str, device,
#                    obs_len: int = 8, pred_len: int = 12, unet_in_ch: int = 13,
#                    d_model: int = 64, nhead: int = 4, num_enc_layers: int = 1,
#                    num_dec_layers: int = 3, dim_ff: int = 512, dropout: float = 0.1):
#     """
#     [UPDATED] Same model_cfg pattern as load_paper_baseline. Note STTrans
#     (non_ar) and STTransAR have DIFFERENT constructor signatures (AR has
#     no num_dec_layers) — the saved model_cfg already accounts for this
#     (see train_st_trans.py's checkpoint-save branch), so **kwargs here
#     naturally works for either. This loader only instantiates STTrans
#     (non-AR) — STTransAR support would need its own loader if you also
#     want to evaluate that variant.
#     """
#     ck = torch.load(checkpoint, map_location="cpu")
#     model_cfg = ck.get("model_cfg")
#     if model_cfg:
#         model = STTrans(**model_cfg).to(device)
#     else:
#         print(f"  ⚠ ST-Trans checkpoint has no model_cfg — using "
#               f"CLI-default-matching args (d_model={d_model}, nhead={nhead}, "
#               f"num_dec_layers={num_dec_layers}). Only correct if trained "
#               f"with train_st_trans.py's own defaults.")
#         model = STTrans(obs_len=obs_len, pred_len=pred_len, unet_in_ch=unet_in_ch,
#                          d_model=d_model, nhead=nhead, num_enc_layers=num_enc_layers,
#                          num_dec_layers=num_dec_layers, dim_ff=dim_ff,
#                          dropout=dropout).to(device)
#     state = ck.get("model_state", ck.get("model"))
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     if missing or unexpected:
#         print(f"  ⚠ ST-Trans load_state_dict: {len(missing)} missing, "
#               f"{len(unexpected)} unexpected keys")
#     model.eval()
#     return model


# @torch.no_grad()
# def evaluate_one_model(model, loader, device, model_name: str,
#                         n_ensemble: int = 20) -> List[Dict]:
#     """
#     Returns a list of PER-LEAD-TIME records:
#       {"model": name, "storm": storm_key, "window": idx, "lead_time": t,
#        "ade": .., "ate": .., "cte": .., "obs_speed": ..}
#     One record per (storm, window, lead_time) triple — matches the
#     paper's Table 10 pairing granularity (140 windows x 16 lead-times =
#     2240 matched pairs, i.e. paired PER FORECAST STEP, not averaged over
#     the whole trajectory first). An earlier version of this function
#     emitted one record per (storm, window) with ADE/ATE/CTE already
#     averaged over all lead-times — that is a coarser, methodologically
#     weaker pairing than the paper's own design (fewer, less granular
#     pairs => less statistical power, and not directly comparable to the
#     paper's n=2240 convention).

#     ade/ate/cte alignment: ate_cte_full() returns [T-1, B] arrays where
#     index i holds the error at ORIGINAL step index i+1 (see its own
#     docstring / evaluate_full.py's established convention — no ATE/CTE
#     is defined at the very first predicted step, since there's no prior
#     heading reference). To keep all three metrics referring to the EXACT
#     SAME lead-time in every emitted record, ADE is also read at the
#     matching original step index (d[t+1], not d[t]) rather than emitting
#     all T ADE values independently of ATE/CTE's T-1 range.
#     """
#     records = []
#     is_fm = isinstance(model, TCFlowMatching) or hasattr(model, "sigma_inference")

#     for bi, batch in enumerate(loader):
#         bl = move(list(batch), device)
#         gt = bl[1]
#         obs = bl[0]
#         try:
#             tyid_list = bl[15]
#         except IndexError:
#             tyid_list = None

#         try:
#             if is_fm:
#                 pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
#             else:
#                 pred, _, _ = model.sample(bl, num_ensemble=1)
#         except Exception as e:
#             print(f"  [{model_name}] batch {bi}: sample error: {e}")
#             continue

#         T = min(pred.shape[0], gt.shape[0])
#         pd = _norm_to_deg(pred[:T])
#         gd = _norm_to_deg(gt[:T, :, :2])
#         d  = _haversine_deg(pd, gd)                  # [T, B]
#         ate, cte = ate_cte_full(pd, gd)               # [T-1, B]
#         T_valid = ate.shape[0]                        # number of lead-times with ATE/CTE defined

#         obs_deg = _norm_to_deg(obs[:, :, :2])
#         if obs_deg.shape[0] >= 2:
#             step_km = _haversine_deg(obs_deg[:-1], obs_deg[1:])
#             obs_speed = step_km.mean(0) / 6.0
#         else:
#             obs_speed = torch.zeros(obs.shape[1], device=device)

#         B = obs.shape[1]
#         for b in range(B):
#             if tyid_list is not None and b < len(tyid_list) and \
#                isinstance(tyid_list[b], dict) and "old" in tyid_list[b]:
#                 info = tyid_list[b]
#                 storm_key = f"{info['old'][1]}_{info['old'][0]}"
#             else:
#                 storm_key = f"UNKNOWN_batch{bi}"
#             for i in range(T_valid):
#                 lead_time = i + 1  # original step index (see docstring)
#                 records.append({
#                     "model":     model_name,
#                     "storm":     storm_key,
#                     "window":    b,
#                     "lead_time": lead_time,
#                     "ade":       float(d[lead_time, b]),
#                     "ate":       float(ate[i, b].abs()),
#                     "cte":       float(cte[i, b].abs()),
#                     "obs_speed": float(obs_speed[b]),
#                 })
#     return records


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset_root", required=True)
#     p.add_argument("--split", default="test", choices=["test", "val", "train"])
#     p.add_argument("--output_dir", default="eval_multi")
#     p.add_argument("--gpu", type=int, default=0)
#     p.add_argument("--n_ensemble", type=int, default=20)
#     p.add_argument("--test_year", type=int, default=None)

#     p.add_argument("--fm_checkpoint",       default=None)
#     p.add_argument("--st_trans_checkpoint", default=None)
#     p.add_argument("--lstm_checkpoint",     default=None)
#     p.add_argument("--gru_checkpoint",      default=None)
#     p.add_argument("--rnn_checkpoint",      default=None)

#     p.add_argument("--paper_hidden_dim", type=int, default=256)
#     p.add_argument("--paper_n_layers",   type=int, default=3)
#     p.add_argument("--paper_dropout",    type=float, default=0.20)
#     p.add_argument("--st_d_model",        type=int, default=64)
#     p.add_argument("--st_nhead",          type=int, default=4)
#     p.add_argument("--st_num_enc_layers", type=int, default=1)
#     p.add_argument("--st_num_dec_layers", type=int, default=3)
#     p.add_argument("--st_dim_ff",         type=int, default=512)
#     p.add_argument("--st_dropout",        type=float, default=0.1)

#     args = p.parse_args()
#     os.makedirs(args.output_dir, exist_ok=True)
#     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

#     print(f"  Loading test data...")
#     import argparse as _ap
#     _loader_args = _ap.Namespace(
#         dataset_root = args.dataset_root,
#         obs_len      = 8,
#         pred_len     = 12,
#         batch_size   = 64,
#         num_workers  = 2,
#         test_year    = args.test_year,
#         skip         = 1,
#         min_ped      = 1,
#         threshold    = 0.002,
#     )
#     _, loader = data_loader(_loader_args,
#                              {"root": args.dataset_root, "type": args.split},
#                              test=(args.split != "train"))
#     print(f"  Data: {len(loader)} batches")

#     jobs = []
#     if args.fm_checkpoint:
#         jobs.append(("FM", "fm", args.fm_checkpoint))
#     if args.st_trans_checkpoint:
#         jobs.append(("ST-Trans", "st_trans", args.st_trans_checkpoint))
#     if args.lstm_checkpoint:
#         jobs.append(("LSTM", "lstm", args.lstm_checkpoint))
#     if args.gru_checkpoint:
#         jobs.append(("GRU", "gru", args.gru_checkpoint))
#     if args.rnn_checkpoint:
#         jobs.append(("RNN", "rnn", args.rnn_checkpoint))

#     if not jobs:
#         print("  No checkpoints given — nothing to do.")
#         return

#     all_records = []
#     for display_name, kind, ckpt_path in jobs:
#         print(f"\n  {'='*70}\n  Loading {display_name}: {ckpt_path}\n  {'='*70}")
#         if kind == "fm":
#             model = load_fm(ckpt_path, device)
#         elif kind == "st_trans":
#             model = load_st_trans(ckpt_path, device,
#                                    d_model=args.st_d_model, nhead=args.st_nhead,
#                                    num_enc_layers=args.st_num_enc_layers,
#                                    num_dec_layers=args.st_num_dec_layers,
#                                    dim_ff=args.st_dim_ff, dropout=args.st_dropout)
#         else:
#             model = load_paper_baseline(ckpt_path, kind, device,
#                                          hidden_dim=args.paper_hidden_dim,
#                                          n_layers=args.paper_n_layers,
#                                          dropout=args.paper_dropout)

#         n_params = sum(pm.numel() for pm in model.parameters())
#         print(f"  {display_name}: {n_params:,} params")

#         recs = evaluate_one_model(model, loader, device, display_name,
#                                    n_ensemble=args.n_ensemble)
#         all_records.extend(recs)

#         ade = np.mean([r["ade"] for r in recs])
#         ate = np.mean([r["ate"] for r in recs])
#         cte = np.mean([r["cte"] for r in recs])
#         print(f"  {display_name}: n={len(recs)}  ADE={ade:.2f}  "
#               f"ATE={ate:.2f}  CTE={cte:.2f}")

#         del model
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     out_path = os.path.join(args.output_dir, f"multi_model_{args.split}.json")
#     with open(out_path, "w") as f:
#         json.dump(all_records, f, indent=2)
#     print(f"\n  Saved {len(all_records)} records → {out_path}")
#     print(f"  Run generate_comparison_table.py --records {out_path} "
#           f"to produce the Table-10-style significance table.")


# if __name__ == "__main__":
#     main()



"""
evaluate_multi_model.py
=========================
Runs test-set evaluation for RNN / GRU / LSTM / ST-Trans / FM checkpoints,
using ONE SHARED ATE/CTE formula for all five models.

WHY A SEPARATE SCRIPT, NOT REUSING EACH MODEL'S OWN METRIC CODE
-----------------------------------------------------------------
paper_baseline_model.py (used by RNN/GRU/LSTM/ST-Trans) has its own
_ate_cte_tensors() with a DIFFERENT convention than flow_matching_model.py's
_ate_cte_full() (used by FM):
  - flat-earth approx (111*cos(lat)) vs haversine — small effect (~0.2%)
  - reference heading computed OUTGOING from each point (gt[t]->gt[t+1])
    vs INCOMING to each point (gt[t-1]->gt[t]) — this is NOT a small
    effect: verified numerically on a synthetic turning trajectory, the
    two conventions can even disagree on the SIGN of CTE at a given step.
Comparing FM's ATE/CTE (computed one way) against RNN/GRU/LSTM/ST-Trans's
ATE/CTE (computed the other way) would not be a fair like-for-like
comparison — any difference could be an artifact of the formula, not the
model. This script computes ADE/ATE/CTE for ALL FIVE models using the
SAME function (_ate_cte_full, _haversine_deg, _forward_azimuth, imported
from flow_matching_model.py — the version with the off-by-one fix already
verified in evaluate_full.py), so Table-10-style comparisons are sound.

USAGE
-----
python evaluate_multi_model.py \
    --dataset_root <root> \
    --fm_checkpoint runs/fm_seed42/best_model.pth \
    --st_trans_checkpoint runs/st_trans/best_model.pth \
    --lstm_checkpoint runs/lstm/best_model.pth \
    --gru_checkpoint runs/gru/best_model.pth \
    --rnn_checkpoint runs/rnn/best_model.pth \
    --output_dir eval_multi/

Any checkpoint arg can be omitted to skip that model. Produces one
combined JSON of per-window records (one row per (model, storm, window))
that generate_comparison_table.py consumes directly for the Table-10-style
statistical significance tables.
"""
from __future__ import annotations
import sys, os, argparse, json
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _haversine_deg, _forward_azimuth, _unwrap,
)
from Model.paper_baseline_model import PaperBaseline, MODEL_TYPES
from Model.st_trans_model import STTrans


def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


def ate_cte_full(pred_deg: torch.Tensor, gt_deg: torch.Tensor):
    """
    SAME formula as evaluate_full.py's _ate_cte_full (off-by-one already
    fixed there — see that file's own comments for the fix history).
    Returns [T-1, B] each; index k holds the error at ORIGINAL step k+1.
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    bear_ref = _forward_azimuth(gt_deg[:T - 1], gt_deg[1:T])
    bear_err = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
    dist_err = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang = bear_err - bear_ref
    return dist_err * torch.cos(ang), dist_err * torch.sin(ang)


def load_fm(checkpoint: str, device):
    ck = torch.load(checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg") or {}
    if not model_cfg:
        print(f"  ⚠ FM checkpoint has no model_cfg — using constructor "
              f"defaults (correct only if trained with default architecture).")
    model = TCFlowMatching(**model_cfg).to(device)
    state = ck.get("model", ck.get("model_state"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ FM load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
    model.eval()
    return model


def load_paper_baseline(checkpoint: str, model_type: str, device,
                         hidden_dim: int = 256, n_layers: int = 3,
                         obs_len: int = 8, pred_len: int = 12,
                         unet_in_ch: int = 13, dropout: float = 0.20):
    """
    [UPDATED] train_paper_baseline.py now saves a full model_cfg dict in
    newer checkpoints (added alongside --seed support). If present, it is
    used directly and the CLI-default args above are ignored — this is
    the reliable path. If absent (checkpoint trained before that fix),
    falls back to the CLI-default-matching values passed in here, with a
    warning, same caveat as FM's missing-model_cfg case.
    """
    assert model_type in MODEL_TYPES, f"model_type must be one of {MODEL_TYPES}"
    ck = torch.load(checkpoint, map_location="cpu")
    saved_type = ck.get("model_type", model_type)
    if saved_type != model_type:
        print(f"  ⚠ Checkpoint's saved model_type='{saved_type}' differs "
              f"from requested '{model_type}' — using requested value. "
              f"Verify this checkpoint is really the {model_type.upper()} one.")

    model_cfg = ck.get("model_cfg")
    if model_cfg:
        model = PaperBaseline(**model_cfg).to(device)
    else:
        print(f"  ⚠ {model_type.upper()} checkpoint has no model_cfg — "
              f"using CLI-default-matching args (hidden_dim={hidden_dim}, "
              f"n_layers={n_layers}, dropout={dropout}). Only correct if "
              f"trained with train_paper_baseline.py's own defaults.")
        model = PaperBaseline(model_type=model_type, pred_len=pred_len,
                               obs_len=obs_len, hidden_dim=hidden_dim,
                               n_layers=n_layers, unet_in_ch=unet_in_ch,
                               dropout=dropout).to(device)
    state = ck.get("model_state", ck.get("model"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ {model_type.upper()} load_state_dict: {len(missing)} "
              f"missing, {len(unexpected)} unexpected keys")
    model.eval()
    return model


def load_st_trans(checkpoint: str, device,
                   obs_len: int = 8, pred_len: int = 12, unet_in_ch: int = 13,
                   d_model: int = 64, nhead: int = 4, num_enc_layers: int = 1,
                   num_dec_layers: int = 3, dim_ff: int = 512, dropout: float = 0.1):
    """
    [UPDATED] Same model_cfg pattern as load_paper_baseline. Note STTrans
    (non_ar) and STTransAR have DIFFERENT constructor signatures (AR has
    no num_dec_layers) — the saved model_cfg already accounts for this
    (see train_st_trans.py's checkpoint-save branch), so **kwargs here
    naturally works for either. This loader only instantiates STTrans
    (non-AR) — STTransAR support would need its own loader if you also
    want to evaluate that variant.
    """
    ck = torch.load(checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg")
    if model_cfg:
        model = STTrans(**model_cfg).to(device)
    else:
        print(f"  ⚠ ST-Trans checkpoint has no model_cfg — using "
              f"CLI-default-matching args (d_model={d_model}, nhead={nhead}, "
              f"num_dec_layers={num_dec_layers}). Only correct if trained "
              f"with train_st_trans.py's own defaults.")
        model = STTrans(obs_len=obs_len, pred_len=pred_len, unet_in_ch=unet_in_ch,
                         d_model=d_model, nhead=nhead, num_enc_layers=num_enc_layers,
                         num_dec_layers=num_dec_layers, dim_ff=dim_ff,
                         dropout=dropout).to(device)
    state = ck.get("model_state", ck.get("model"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ ST-Trans load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
    model.eval()
    return model


@torch.no_grad()
def evaluate_one_model(model, loader, device, model_name: str,
                        n_ensemble: int = 20) -> List[Dict]:
    """
    Returns a list of per-window records:
      {"model": name, "storm": storm_key, "window": idx, "ade": ..,
       "ate": .., "cte": .., "obs_speed": ..}
    One record per (storm, window) — this is the raw material for paired
    statistical tests (Wilcoxon/t-test on matched (storm, window) pairs,
    same design as the paper's Table 10).
    """
    records = []
    is_fm = isinstance(model, TCFlowMatching) or hasattr(model, "sigma_inference")

    for bi, batch in enumerate(loader):
        bl = move(list(batch), device)
        gt = bl[1]
        obs = bl[0]
        try:
            tyid_list = bl[15]
        except IndexError:
            tyid_list = None

        try:
            if is_fm:
                pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
            else:
                pred, _, _ = model.sample(bl, num_ensemble=1)
        except Exception as e:
            print(f"  [{model_name}] batch {bi}: sample error: {e}")
            continue

        T = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T])
        gd = _norm_to_deg(gt[:T, :, :2])
        d  = _haversine_deg(pd, gd)                  # [T, B]
        ate, cte = ate_cte_full(pd, gd)               # [T-1, B]

        obs_deg = _norm_to_deg(obs[:, :, :2])
        if obs_deg.shape[0] >= 2:
            step_km = _haversine_deg(obs_deg[:-1], obs_deg[1:])
            obs_speed = step_km.mean(0) / 6.0
        else:
            obs_speed = torch.zeros(obs.shape[1], device=device)

        B = obs.shape[1]
        for b in range(B):
            if tyid_list is not None and b < len(tyid_list) and \
               isinstance(tyid_list[b], dict) and "old" in tyid_list[b]:
                info = tyid_list[b]
                storm_key = f"{info['old'][1]}_{info['old'][0]}"
            else:
                storm_key = f"UNKNOWN_batch{bi}"
            records.append({
                "model":     model_name,
                "storm":     storm_key,
                "window":    b,
                "ade":       float(d[:, b].mean()),
                "ate":       float(ate[:, b].abs().mean()) if ate.shape[0] > 0 else 0.0,
                "cte":       float(cte[:, b].abs().mean()) if cte.shape[0] > 0 else 0.0,
                "obs_speed": float(obs_speed[b]),
            })
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--output_dir", default="eval_multi")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_ensemble", type=int, default=20)
    p.add_argument("--test_year", type=int, default=None)

    p.add_argument("--fm_checkpoint",       default=None)
    p.add_argument("--st_trans_checkpoint", default=None)
    p.add_argument("--lstm_checkpoint",     default=None)
    p.add_argument("--gru_checkpoint",      default=None)
    p.add_argument("--rnn_checkpoint",      default=None)

    p.add_argument("--paper_hidden_dim", type=int, default=256)
    p.add_argument("--paper_n_layers",   type=int, default=3)
    p.add_argument("--paper_dropout",    type=float, default=0.20)
    p.add_argument("--st_d_model",        type=int, default=64)
    p.add_argument("--st_nhead",          type=int, default=4)
    p.add_argument("--st_num_enc_layers", type=int, default=1)
    p.add_argument("--st_num_dec_layers", type=int, default=3)
    p.add_argument("--st_dim_ff",         type=int, default=512)
    p.add_argument("--st_dropout",        type=float, default=0.1)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"  Loading test data...")
    import argparse as _ap
    _loader_args = _ap.Namespace(
        dataset_root = args.dataset_root,
        obs_len      = 8,
        pred_len     = 12,
        batch_size   = 64,
        num_workers  = 2,
        test_year    = args.test_year,
        skip         = 1,
        min_ped      = 1,
        threshold    = 0.002,
    )
    _, loader = data_loader(_loader_args,
                             {"root": args.dataset_root, "type": args.split},
                             test=(args.split != "train"))
    print(f"  Data: {len(loader)} batches")

    jobs = []
    if args.fm_checkpoint:
        jobs.append(("FM", "fm", args.fm_checkpoint))
    if args.st_trans_checkpoint:
        jobs.append(("ST-Trans", "st_trans", args.st_trans_checkpoint))
    if args.lstm_checkpoint:
        jobs.append(("LSTM", "lstm", args.lstm_checkpoint))
    if args.gru_checkpoint:
        jobs.append(("GRU", "gru", args.gru_checkpoint))
    if args.rnn_checkpoint:
        jobs.append(("RNN", "rnn", args.rnn_checkpoint))

    if not jobs:
        print("  No checkpoints given — nothing to do.")
        return

    all_records = []
    for display_name, kind, ckpt_path in jobs:
        print(f"\n  {'='*70}\n  Loading {display_name}: {ckpt_path}\n  {'='*70}")
        if kind == "fm":
            model = load_fm(ckpt_path, device)
        elif kind == "st_trans":
            model = load_st_trans(ckpt_path, device,
                                   d_model=args.st_d_model, nhead=args.st_nhead,
                                   num_enc_layers=args.st_num_enc_layers,
                                   num_dec_layers=args.st_num_dec_layers,
                                   dim_ff=args.st_dim_ff, dropout=args.st_dropout)
        else:
            model = load_paper_baseline(ckpt_path, kind, device,
                                         hidden_dim=args.paper_hidden_dim,
                                         n_layers=args.paper_n_layers,
                                         dropout=args.paper_dropout)

        n_params = sum(pm.numel() for pm in model.parameters())
        print(f"  {display_name}: {n_params:,} params")

        recs = evaluate_one_model(model, loader, device, display_name,
                                   n_ensemble=args.n_ensemble)
        all_records.extend(recs)

        ade = np.mean([r["ade"] for r in recs])
        ate = np.mean([r["ate"] for r in recs])
        cte = np.mean([r["cte"] for r in recs])
        print(f"  {display_name}: n={len(recs)}  ADE={ade:.2f}  "
              f"ATE={ate:.2f}  CTE={cte:.2f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = os.path.join(args.output_dir, f"multi_model_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n  Saved {len(all_records)} records → {out_path}")
    print(f"  Run generate_comparison_table.py --records {out_path} "
          f"to produce the Table-10-style significance table.")


if __name__ == "__main__":
    main()
    
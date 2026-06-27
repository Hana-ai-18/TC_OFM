# # # # # """
# # # # # scripts/train_fm.py  ——  TC-FlowMatching v2.4
# # # # #   Core v2.1 + augmentation mạnh (rotation/mixup) + exp step weights
# # # # #   XAI: attribution, hard_score components, physics score, uncertainty
# # # # # """
# # # # # from __future__ import annotations

# # # # # import sys, os
# # # # # _SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
# # # # # _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
# # # # # if _PROJECT_ROOT not in sys.path:
# # # # #     sys.path.insert(0, _PROJECT_ROOT)

# # # # # import argparse, math, time
# # # # # from collections import defaultdict
# # # # # from typing import Dict, List, Optional

# # # # # import numpy as np
# # # # # import torch
# # # # # import torch.optim as optim
# # # # # from torch.amp import autocast, GradScaler

# # # # # from Model.data.loader_training import data_loader
# # # # # from Model.flow_matching_model import (
# # # # #     TCFlowMatching, _norm_to_deg, _haversine_deg,
# # # # #     EMAModel, augment_batch, hard_score_from_obs,
# # # # #     compute_obs_attribution, compute_ensemble_uncertainty,
# # # # #     load_checkpoint_compat,
# # # # # )

# # # # # HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}
# # # # # ST_TRANS_VAL  = {"ADE": 172.68, "ATE": 142.21, "CTE": 42.04,
# # # # #                  "12h": 65.42, "24h": 104.67, "48h": 205.10, "72h": 321.39}
# # # # # ST_TRANS_TEST = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
# # # # #                  "12h": 77.5, "24h": 130.5, "48h": 269.9, "72h": 423.3}


# # # # # def _unwrap(m):
# # # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # # # # def move(batch, device):
# # # # #     out = list(batch)
# # # # #     for i, x in enumerate(out):
# # # # #         if torch.is_tensor(x): out[i] = x.to(device)
# # # # #         elif isinstance(x, dict):
# # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
# # # # #     return out

# # # # # def _ate_cte(pred_deg, gt_deg):
# # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # #     if T < 2: z = pred_deg.new_zeros(1, pred_deg.shape[1]); return z, z
# # # # #     lo1=torch.deg2rad(gt_deg[:T-1,:,0]); la1=torch.deg2rad(gt_deg[:T-1,:,1])
# # # # #     lo2=torch.deg2rad(gt_deg[1:T,:,0]);  la2=torch.deg2rad(gt_deg[1:T,:,1])
# # # # #     lo3=torch.deg2rad(pred_deg[1:T,:,0]);la3=torch.deg2rad(pred_deg[1:T,:,1])
# # # # #     ya=torch.sin(lo2-lo1)*torch.cos(la2)
# # # # #     xa=torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # # # #     be=torch.atan2(ya,xa)
# # # # #     ye=torch.sin(lo3-lo2)*torch.cos(la3)
# # # # #     xe=torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # # # #     bee=torch.atan2(ye,xe)
# # # # #     tot=_haversine_deg(pred_deg[1:T],gt_deg[1:T]); ang=bee-be
# # # # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  2-group optimizer + scheduler
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def build_optimizer(model, lr_vel, lr_enc, weight_decay):
# # # # #     raw = _unwrap(model)
# # # # #     enc_params = list(raw.encoder.parameters())
# # # # #     vel_params = list(raw.velocity.parameters())
# # # # #     return optim.AdamW([
# # # # #         {"params": enc_params,  "lr": lr_enc, "name": "encoder"},
# # # # #         {"params": vel_params,  "lr": lr_vel, "name": "velocity"},
# # # # #     ], weight_decay=weight_decay)


# # # # # class TwoGroupScheduler:
# # # # #     def __init__(self, opt, warmup_ep, total_ep, lr_vel, lr_enc,
# # # # #                  lr_min, freeze_enc_ep, enc_warmup_ep):
# # # # #         self.opt = opt; self.warmup = warmup_ep; self.total = total_ep
# # # # #         self.lr_vel = lr_vel; self.lr_enc = lr_enc; self.lr_min = lr_min
# # # # #         self.freeze_enc = freeze_enc_ep; self.enc_warmup = enc_warmup_ep
# # # # #         self.epoch = 0

# # # # #     def _cos(self, ep_s, ep_e, lr_s, lr_e, ep):
# # # # #         if ep <= ep_s: return lr_s
# # # # #         if ep >= ep_e: return lr_e
# # # # #         t = (ep - ep_s) / (ep_e - ep_s)
# # # # #         return lr_e + 0.5*(lr_s - lr_e)*(1 + math.cos(math.pi * t))

# # # # #     def step(self):
# # # # #         ep = self.epoch
# # # # #         lr_vel = (self.lr_vel * max(0.1, ep/max(self.warmup-1,1)) if ep < self.warmup
# # # # #                   else self._cos(self.warmup, self.total, self.lr_vel, self.lr_min, ep))
# # # # #         enc_start = self.freeze_enc + self.enc_warmup
# # # # #         if ep < self.freeze_enc:
# # # # #             lr_enc = 0.0
# # # # #         elif ep < enc_start:
# # # # #             lr_enc = self.lr_enc * (ep - self.freeze_enc) / max(self.enc_warmup, 1)
# # # # #         else:
# # # # #             lr_enc = self._cos(enc_start, self.total, self.lr_enc, self.lr_min, ep)
# # # # #         for pg in self.opt.param_groups:
# # # # #             pg["lr"] = lr_enc if pg.get("name") == "encoder" else lr_vel
# # # # #         self.epoch += 1
# # # # #         return lr_vel, lr_enc

# # # # #     def get_lrs(self):
# # # # #         d = {pg.get("name","?"): pg["lr"] for pg in self.opt.param_groups}
# # # # #         return d.get("velocity", 0.0), d.get("encoder", 0.0)


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Evaluate
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # @torch.no_grad()
# # # # # def evaluate(model, loader, device, tag="", ema=None, ref_targets=None,
# # # # #              use_tta=False, n_tta=8, epoch_for_loss=9999,
# # # # #              run_xai=False, xai_batch=None) -> Dict:
# # # # #     bk = None
# # # # #     if ema is not None:
# # # # #         try: bk = ema.apply_to(model)
# # # # #         except Exception as e: print(f"  ⚠ EMA: {e}")

# # # # #     model.eval()
# # # # #     all_ade, all_ate, all_cte = [], [], []
# # # # #     step_dist = defaultdict(list)
# # # # #     sum_loss = 0.0; sum_n = 0

# # # # #     for batch in loader:
# # # # #         bl = move(list(batch), device)
# # # # #         gt = bl[1]; B = bl[0].shape[1]

# # # # #         try:
# # # # #             bd = model.get_loss_breakdown(bl, epoch=epoch_for_loss)
# # # # #             if torch.isfinite(bd["total"]):
# # # # #                 sum_loss += bd["total"].item() * B; sum_n += B
# # # # #         except Exception:
# # # # #             pass

# # # # #         if use_tta:
# # # # #             obs  = bl[0]
# # # # #             anchor = obs[-1:, :, :2]   # [1, B, 2] — BUG-3 FIXED
# # # # #             preds_tta = []
# # # # #             for tta_i in range(n_tta):
# # # # #                 if tta_i == 0:
# # # # #                     bl_tta = bl
# # # # #                 else:
# # # # #                     angle = (tta_i / (n_tta-1) - 0.5) * (math.pi/18)
# # # # #                     cos_a, sin_a = math.cos(angle), math.sin(angle)
# # # # #                     rot = torch.tensor([[cos_a,-sin_a],[sin_a,cos_a]],
# # # # #                                        dtype=obs.dtype, device=device)
# # # # #                     T_o = obs.shape[0]
# # # # #                     rel_obs = (obs[..., :2] - anchor).reshape(T_o*B, 2)
# # # # #                     obs_rot = obs.clone()
# # # # #                     obs_rot[..., :2] = (rot @ rel_obs.T).T.reshape(T_o, B, 2) + anchor
# # # # #                     bl_tta = list(bl); bl_tta[0] = obs_rot

# # # # #                 try:
# # # # #                     pred, _, _ = model.sample(bl_tta)
# # # # #                 except Exception as e:
# # # # #                     print(f"  TTA error: {e}"); continue

# # # # #                 if tta_i > 0:
# # # # #                     rot_inv = rot.T
# # # # #                     T_p = pred.shape[0]
# # # # #                     rel_p = (pred - anchor).reshape(T_p*B, 2)
# # # # #                     pred  = (rot_inv @ rel_p.T).T.reshape(T_p, B, 2) + anchor
# # # # #                 preds_tta.append(pred)

# # # # #             if not preds_tta: continue
# # # # #             weights = [2.0 if i == 0 else 1.0 for i in range(len(preds_tta))]
# # # # #             total_w = sum(weights)
# # # # #             pred = sum(w/total_w * p for w, p in zip(weights, preds_tta))
# # # # #         else:
# # # # #             try:
# # # # #                 pred, _, _ = model.sample(bl)
# # # # #             except Exception as e:
# # # # #                 print(f"  sample error: {e}"); continue

# # # # #         T        = min(pred.shape[0], gt.shape[0])
# # # # #         pred_deg = _norm_to_deg(pred[:T]); gt_deg = _norm_to_deg(gt[:T])
# # # # #         dist     = _haversine_deg(pred_deg, gt_deg)
# # # # #         ate, cte = _ate_cte(pred_deg, gt_deg)

# # # # #         all_ade.extend(dist.mean(0).tolist())
# # # # #         if ate.shape[0] > 0:
# # # # #             all_ate.extend(ate.abs().mean(0).tolist())
# # # # #             all_cte.extend(cte.abs().mean(0).tolist())
# # # # #         for h, s in HORIZON_STEPS.items():
# # # # #             if s < T: step_dist[h].extend(dist[s].tolist())

# # # # #     if bk is not None:
# # # # #         try: ema.restore(model, bk)
# # # # #         except Exception: pass

# # # # #     def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # # # #     val_loss = sum_loss / max(sum_n, 1)
# # # # #     result = {"ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
# # # # #               "n": len(all_ade), "val_loss": val_loss}
# # # # #     for h in HORIZON_STEPS: result[f"{h}h"] = _m(step_dist[h])

# # # # #     ade, ate_, cte_ = result["ADE"], result["ATE"], result["CTE"]
# # # # #     result["combined"] = (0.5*ade + 0.3*ate_ + 0.2*cte_
# # # # #                           if all(np.isfinite(x) for x in [ade,ate_,cte_]) else ade)

# # # # #     ref = ref_targets or ST_TRANS_VAL
# # # # #     def _v(k): return result.get(k, float("nan"))
# # # # #     def _ok(k): return "✓" if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9) else "✗"

# # # # #     tta_str = " [TTA]" if use_tta else ""
# # # # #     print(f"\n  {'='*72}")
# # # # #     print(f"  [{tag}]{tta_str}  n={result['n']}")
# # # # #     print(f"  Val Loss : {val_loss:.6f}")
# # # # #     print(f"  ADE={_v('ADE'):7.1f}km {_ok('ADE')}  ATE={_v('ATE'):7.1f}km {_ok('ATE')}"
# # # # #           f"  CTE={_v('CTE'):7.1f}km {_ok('CTE')}")
# # # # #     print(f"  Combined = {_v('combined'):.1f}")
# # # # #     print(f"  12h={_v('12h'):6.1f}  24h={_v('24h'):6.1f}"
# # # # #           f"  48h={_v('48h'):6.1f}  72h={_v('72h'):6.1f} km")
# # # # #     beat = [f"{k}={_v(k):.0f}<{ref.get(k,999):.0f}"
# # # # #             for k in ["ADE","ATE","CTE","12h","24h","48h","72h"]
# # # # #             if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9)]
# # # # #     print(f"  BEAT: {' | '.join(beat) if beat else 'none'}")

# # # # #     if run_xai and xai_batch is not None:
# # # # #         try:
# # # # #             model.train()  # BUG-C fix: không dùng eval() cho XAI
# # # # #             with torch.no_grad():
# # # # #                 _, _, all_t, xai = _unwrap(model).sample(xai_batch, return_xai=True)
# # # # #             print(f"  ── XAI Summary {'─'*48}")
# # # # #             print(f"  [XAI-4] Uncertainty: 12h={xai['mean_12h_std']:.1f}km  "
# # # # #                   f"72h={xai['mean_72h_std']:.1f}km  "
# # # # #                   f"ratio={float(xai['uncertainty_ratio'].mean()):.2f}x")
# # # # #             pc = xai.get("physics_components", {})
# # # # #             if pc:
# # # # #                 print(f"  [XAI-3] Physics(best): speed={float(pc['speed'].mean()):.3f}  "
# # # # #                       f"smooth={float(pc['smooth'].mean()):.3f}  "
# # # # #                       f"heading={float(pc['heading'].mean()):.3f}")
# # # # #             hc = xai.get("hard_components", {})
# # # # #             if hc:
# # # # #                 print(f"  [XAI-2] HardScore: curvature={float(hc['curvature'].mean()):.3f}  "
# # # # #                       f"speed_var={float(hc['speed_var'].mean()):.3f}  "
# # # # #                       f"dir_change={float(hc['dir_change'].mean()):.3f}")
# # # # #             sc = xai.get("speed_comparison", {})
# # # # #             if sc:
# # # # #                 ratio = sc.get("speed_ratio", 1.0)
# # # # #                 flag  = "⚠ OVER" if ratio > 1.15 else ("⚠ UNDER" if ratio < 0.85 else "✓ OK")
# # # # #                 print(f"  [XAI-5] Speed: obs={sc['obs_speed_mean']:.1f}km/h  "
# # # # #                       f"pred={sc['pred_speed_mean']:.1f}km/h  "
# # # # #                       f"ratio={ratio:.2f} {flag}")
# # # # #             print(f"  {'─'*60}")
# # # # #             result["xai"] = xai
# # # # #         except Exception as e:
# # # # #             print(f"  XAI error: {e}")

# # # # #     print(f"  {'='*72}\n")
# # # # #     return result


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Checkpoint
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def _save(path, epoch, model, opt, sched, best_score,
# # # # #           ema=None, scaler=None, patience_cnt=0, extra=None):
# # # # #     m   = _unwrap(model)
# # # # #     esd = None
# # # # #     if ema is not None:
# # # # #         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # # # #         except Exception: pass
# # # # #     payload = {"epoch": epoch, "model": m.state_dict(), "ema": esd,
# # # # #                "optimizer": opt.state_dict(), "scheduler_ep": sched.epoch,
# # # # #                "scaler": scaler.state_dict() if scaler else None,
# # # # #                "best_score": best_score, "patience_cnt": patience_cnt}
# # # # #     if extra: payload.update(extra)
# # # # #     torch.save(payload, path)


# # # # # def load_checkpoint(path, model, opt, sched, ema, scaler, device):
# # # # #     # Dùng compat loader để handle vel_obs_enc 6→7 feature expansion
# # # # #     ck = load_checkpoint_compat(path, _unwrap(model), device)
# # # # #     print(f"  Loaded: {path}")
# # # # #     if "optimizer" in ck:
# # # # #         try: opt.load_state_dict(ck["optimizer"])
# # # # #         except Exception as e: print(f"  ⚠ Optimizer: {e}")
# # # # #     sched.epoch = ck.get("scheduler_ep", ck.get("scheduler", 0))
# # # # #     if scaler and ck.get("scaler"):
# # # # #         try: scaler.load_state_dict(ck["scaler"])
# # # # #         except Exception: pass
# # # # #     if ema and ck.get("ema"):
# # # # #         for k, v in ck["ema"].items():
# # # # #             if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
# # # # #     return (ck.get("epoch", 0) + 1,
# # # # #             ck.get("best_score", ck.get("best_ade", float("inf"))),
# # # # #             ck.get("patience_cnt", 0))


# # # # # def ensemble_checkpoints(ckpt_paths, model, device):
# # # # #     """[BUG-4 FIXED] Chỉ ensemble periodic ckpts."""
# # # # #     if not ckpt_paths: return
# # # # #     sds = []
# # # # #     for p in ckpt_paths:
# # # # #         if not os.path.exists(p): continue
# # # # #         try:
# # # # #             ck = torch.load(p, map_location=device)
# # # # #             sds.append(ck.get("model", ck))
# # # # #             print(f"  Ensemble: {os.path.basename(p)}")
# # # # #         except Exception as e:
# # # # #             print(f"  ⚠ {p}: {e}")
# # # # #     if len(sds) < 2: return
# # # # #     avg_sd = {}
# # # # #     for key in sds[0]:
# # # # #         try: avg_sd[key] = sum(sd[key].float() for sd in sds) / len(sds)
# # # # #         except Exception: avg_sd[key] = sds[0][key]
# # # # #     _unwrap(model).load_state_dict(avg_sd, strict=False)
# # # # #     print(f"  Ensemble: averaged {len(sds)} ckpts")


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Args
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def get_args():
# # # # #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # #     p.add_argument("--dataset_root",      default="TCND_vn")
# # # # #     p.add_argument("--obs_len",           default=8,     type=int)
# # # # #     p.add_argument("--pred_len",          default=12,    type=int)
# # # # #     p.add_argument("--num_workers",       default=2,     type=int)
# # # # #     p.add_argument("--other_modal",       default="gph")
# # # # #     p.add_argument("--delim",             default=" ")
# # # # #     p.add_argument("--skip",              default=1,     type=int)
# # # # #     p.add_argument("--min_ped",           default=1,     type=int)
# # # # #     p.add_argument("--threshold",         default=0.002, type=float)
# # # # #     # Model
# # # # #     p.add_argument("--d_cond",            default=256,   type=int)
# # # # #     p.add_argument("--d_model",           default=256,   type=int)
# # # # #     p.add_argument("--nhead",             default=8,     type=int)
# # # # #     p.add_argument("--num_dec_layers",    default=4,     type=int)
# # # # #     p.add_argument("--dim_ff",            default=512,   type=int)
# # # # #     p.add_argument("--dropout",           default=0.1,   type=float)
# # # # #     p.add_argument("--unet_in_ch",        default=13,    type=int)
# # # # #     p.add_argument("--sigma_min",         default=0.04,  type=float)
# # # # #     p.add_argument("--sigma_max",         default=0.08,  type=float)
# # # # #     p.add_argument("--lambda_reg",        default=0.2,   type=float)
# # # # #     p.add_argument("--use_ot",            default=True,  action="store_true")
# # # # #     p.add_argument("--no_ot",             dest="use_ot", action="store_false")
# # # # #     p.add_argument("--ot_epsilon",        default=0.05,  type=float)
# # # # #     p.add_argument("--n_ensemble",        default=20,    type=int)
# # # # #     p.add_argument("--sigma_inference",   default=0.04,  type=float)
# # # # #     p.add_argument("--n_inference_steps", default=1,     type=int)
# # # # #     # Training
# # # # #     p.add_argument("--num_epochs",        default=150,   type=int)
# # # # #     p.add_argument("--batch_size",        default=64,    type=int)
# # # # #     p.add_argument("--lr",                default=1e-4,  type=float)
# # # # #     p.add_argument("--lr_enc",            default=2e-5,  type=float)
# # # # #     p.add_argument("--lr_min",            default=1e-6,  type=float)
# # # # #     p.add_argument("--warmup_epochs",     default=5,     type=int)
# # # # #     p.add_argument("--freeze_enc_epochs", default=5,     type=int)
# # # # #     p.add_argument("--enc_warmup_epochs", default=3,     type=int)
# # # # #     p.add_argument("--weight_decay",      default=1e-4,  type=float)
# # # # #     p.add_argument("--grad_clip",         default=1.0,   type=float)
# # # # #     p.add_argument("--use_amp",           default=False, action="store_true")
# # # # #     p.add_argument("--use_ema",           default=True,  action="store_true")
# # # # #     p.add_argument("--no_ema",            dest="use_ema",action="store_false")
# # # # #     # Eval
# # # # #     p.add_argument("--val_freq",          default=5,     type=int)     # giảm từ 10 → 5
# # # # #     p.add_argument("--patience",          default=30,    type=int)
# # # # #     p.add_argument("--min_ep",            default=10,    type=int)
# # # # #     # IO
# # # # #     p.add_argument("--output_dir",        default="runs/fm_v24")
# # # # #     p.add_argument("--gpu_num",           default="0")
# # # # #     p.add_argument("--resume",            default=None)
# # # # #     p.add_argument("--test_at_end",       default=True,  action="store_true")
# # # # #     p.add_argument("--no_test",           dest="test_at_end", action="store_false")
# # # # #     p.add_argument("--tta_test",          default=True,  action="store_true")
# # # # #     p.add_argument("--n_tta",             default=8,     type=int)
# # # # #     return p.parse_args()


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Main
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def main(args):
# # # # #     if torch.cuda.is_available():
# # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # #     os.makedirs(args.output_dir, exist_ok=True)
# # # # #     best_ckpt = os.path.join(args.output_dir, "best_model.pth")
# # # # #     last_ckpt = os.path.join(args.output_dir, "last_model.pth")

# # # # #     print("=" * 72)
# # # # #     print("  TC-FlowMatching v2.4 — v2.1 core + aug mạnh hơn + exp step weights")
# # # # #     print(f"  sigma_max={args.sigma_max}  sigma_min={args.sigma_min}")
# # # # #     print(f"  lambda_reg={args.lambda_reg}  L_reg ramp: ep10→ep30")
# # # # #     print(f"  Encoder freeze {args.freeze_enc_epochs}ep + warmup {args.enc_warmup_epochs}ep")
# # # # #     print(f"  n_inference_steps={args.n_inference_steps} (1-step Euler)")
# # # # #     print(f"  XAI: hard_score components, physics score, uncertainty")
# # # # #     print("=" * 72)

# # # # #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # #     vd, val_loader = data_loader(args, {"root": args.dataset_root, "type": "val"}, test=True)
# # # # #     print(f"  train: {len(trd)} ({len(trl)} batches/ep)")
# # # # #     # Lấy 1 batch val để dùng cho XAI logging
# # # # #     _xai_batch_raw = next(iter(val_loader))
# # # # #     xai_batch = move(list(_xai_batch_raw), device) if _xai_batch_raw else None
# # # # #     print(f"  val  : {len(vd)} ({len(val_loader)} batches)")

# # # # #     model = TCFlowMatching(
# # # # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # # # #         unet_in_ch=args.unet_in_ch, d_cond=args.d_cond,
# # # # #         d_model=args.d_model, nhead=args.nhead,
# # # # #         num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
# # # # #         dropout=args.dropout, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
# # # # #         lambda_reg=args.lambda_reg, use_ot=args.use_ot, ot_epsilon=args.ot_epsilon,
# # # # #         use_ema=args.use_ema, n_ensemble=args.n_ensemble,
# # # # #         sigma_inference=args.sigma_inference,
# # # # #         n_inference_steps=args.n_inference_steps,
# # # # #     ).to(device)

# # # # #     model.init_ema()
# # # # #     ema = getattr(_unwrap(model), "_ema", None)
# # # # #     raw = _unwrap(model)
# # # # #     n_enc = sum(p.numel() for p in raw.encoder.parameters())
# # # # #     n_vel = sum(p.numel() for p in raw.velocity.parameters())
# # # # #     print(f"  Encoder: {n_enc:,}  Velocity: {n_vel:,}  Total: {n_enc+n_vel:,}")

# # # # #     opt    = build_optimizer(model, args.lr, args.lr_enc, args.weight_decay)
# # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # # # #     sched  = TwoGroupScheduler(
# # # # #         opt, args.warmup_epochs, args.num_epochs,
# # # # #         args.lr, args.lr_enc, args.lr_min,
# # # # #         args.freeze_enc_epochs, args.enc_warmup_epochs)

# # # # #     start_ep = 0; best_score = float("inf"); patience_cnt = 0
# # # # #     if args.resume and os.path.exists(args.resume):
# # # # #         start_ep, best_score, patience_cnt = load_checkpoint(
# # # # #             args.resume, model, opt, sched, ema, scaler, device)
# # # # #         print(f"  ↩ Resume ep{start_ep}  best={best_score:.1f}")

# # # # #     # [BUG-H FIXED] torch.compile TRƯỚC freeze logic.
# # # # #     # Freeze thực sự được handle bằng zero_grad + skip update (không phải
# # # # #     # requires_grad=False sau compile vì compile có thể cache graph).
# # # # #     # Thay vào đó: vẫn set requires_grad nhưng THÊM zero_grad cho frozen params
# # # # #     # sau backward để đảm bảo optimizer không update chúng.
# # # # #     try:
# # # # #         model = torch.compile(model, mode="reduce-overhead"); print("  torch.compile: ok")
# # # # #     except Exception: pass

# # # # #     nstep = len(trl)
# # # # #     val_ade_history = []
# # # # #     periodic_ckpts  = []   # [BUG-4 FIXED]

# # # # #     print(f"\n  TRAINING ({nstep} steps/ep × {args.num_epochs} ep)")

# # # # #     for ep in range(start_ep, start_ep + args.num_epochs):
# # # # #         model.train()
# # # # #         sum_loss = sum_cfm = sum_reg = sum_ade1 = 0.0
# # # # #         t0 = time.perf_counter()

# # # # #         rel_ep = ep - start_ep

# # # # #         # Freeze logic — [BUG-H FIXED]
# # # # #         # Sau torch.compile, KHÔNG dùng requires_grad=False/True để freeze/unfreeze
# # # # #         # vì compiled graph có thể không re-trace khi requires_grad thay đổi.
# # # # #         # Thay vào đó: set requires_grad NHƯNG sau backward, zero gradient của
# # # # #         # frozen params để optimizer không update chúng.
# # # # #         # requires_grad=False vẫn set để tránh tính gradient không cần thiết,
# # # # #         # nhưng không dựa vào nó hoàn toàn.
# # # # #         # v2.4: không có ar_enc/ar_gate nữa — chỉ freeze/unfreeze encoder backbone
# # # # #         freeze_enc = rel_ep < args.freeze_enc_epochs
# # # # #         raw_enc = _unwrap(model).encoder
# # # # #         for p in raw_enc.parameters():
# # # # #             p.requires_grad_(not freeze_enc)

# # # # #         if rel_ep == 0 and freeze_enc:
# # # # #             print(f"  *** Ep{ep}: encoder frozen ({args.freeze_enc_epochs}ep) ***")
# # # # #         if rel_ep == args.freeze_enc_epochs:
# # # # #             print(f"\n  *** Ep{ep}: encoder unfrozen ***")

# # # # #         for i, batch in enumerate(trl):
# # # # #             bl = move(list(batch), device)
# # # # #             bl_aug = augment_batch(bl)

# # # # #             opt.zero_grad()
# # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # #                 bd = model.get_loss_breakdown(bl_aug, epoch=ep)

# # # # #             scaler.scale(bd["total"]).backward()
# # # # #             scaler.unscale_(opt)
# # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

# # # # #             # Zero grad cho encoder frozen params sau backward
# # # # #             if freeze_enc:
# # # # #                 for p in _unwrap(model).encoder.parameters():
# # # # #                     if p.grad is not None:
# # # # #                         p.grad.zero_()

# # # # #             scaler.step(opt); scaler.update()
# # # # #             model.ema_update()

# # # # #             sum_loss += bd["total"].item(); sum_cfm += bd["l_cfm"]
# # # # #             sum_reg  += bd["l_reg"];       sum_ade1 += bd["ade_1step"]

# # # # #             if i % 30 == 0:
# # # # #                 lr_vel, lr_enc = sched.get_lrs()
# # # # #                 enc_status = "frozen" if freeze_enc else "active"
# # # # #                 print(f"  [{ep:>3}][{i:>3}/{nstep}]"
# # # # #                       f"  total={bd['total'].item():.4f}"
# # # # #                       f"  cfm={bd['l_cfm']:.4f}"
# # # # #                       f"  reg={bd['l_reg']:.4f}"
# # # # #                       f"  lam={bd['lam_reg']:.3f}"
# # # # #                       f"  ade1={bd['ade_1step']:.0f}km"
# # # # #                       f"  σ={bd['sigma']:.3f}"
# # # # #                       f"  enc={enc_status}"
# # # # #                       f"  lr_vel={lr_vel:.2e}")

# # # # #         train_loss = sum_loss / nstep
# # # # #         lr_vel, lr_enc = sched.get_lrs()
# # # # #         sched.step()

# # # # #         print(f"\n  ── Ep{ep:>3}  train={train_loss:.6f}"
# # # # #               f"  cfm={sum_cfm/nstep:.4f}  reg={sum_reg/nstep:.4f}"
# # # # #               f"  ade1={sum_ade1/nstep:.0f}km"
# # # # #               f"  lr_vel={lr_vel:.2e}  lr_enc={lr_enc:.2e}"
# # # # #               f"  t={time.perf_counter()-t0:.0f}s")

# # # # #         _save(last_ckpt, ep, model, opt, sched, best_score, ema, scaler, patience_cnt)

# # # # #         # Periodic checkpoint — [BUG-4 FIXED] track riêng
# # # # #         if rel_ep % 5 == 0:
# # # # #             ckpt_path = os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth")
# # # # #             _save(ckpt_path, ep, model, opt, sched, best_score, ema, scaler, patience_cnt)
# # # # #             periodic_ckpts.append(ckpt_path)
# # # # #             if len(periodic_ckpts) > 5: periodic_ckpts.pop(0)
# # # # #             print(f"  💾 ckpt_ep{ep:03d}.pth")

# # # # #         # Val
# # # # #         if rel_ep % args.val_freq == 0:
# # # # #             r = evaluate(model, val_loader, device,
# # # # #                          tag=f"VAL ep{ep}", ema=ema,
# # # # #                          ref_targets=ST_TRANS_VAL,
# # # # #                          use_tta=False, epoch_for_loss=ep,
# # # # #                          run_xai=(rel_ep % 10 == 0), xai_batch=xai_batch)

# # # # #             val_ade   = r["ADE"]
# # # # #             score     = r["combined"]
# # # # #             val_ade_history.append(val_ade)

# # # # #             if len(val_ade_history) >= 4:
# # # # #                 trend = float(np.mean(val_ade_history[-2:])) - float(np.mean(val_ade_history[-4:-2]))
# # # # #                 trend_str = (f"↑{trend:+.1f}km ⚠" if trend > 5
# # # # #                              else f"↓{trend:+.1f}km ✓" if trend < -5
# # # # #                              else f"→{trend:+.1f}km")
# # # # #             else:
# # # # #                 trend_str = "—"

# # # # #             print(f"  train={train_loss:.6f}  val_ADE={val_ade:.1f}km"
# # # # #                   f"  combined={score:.1f}  trend={trend_str}")

# # # # #             if score < best_score:
# # # # #                 best_score = score; patience_cnt = 0
# # # # #                 _save(best_ckpt, ep, model, opt, sched, best_score, ema, scaler, 0,
# # # # #                       extra={"val_ade": r["ADE"], "val_ate": r["ATE"],
# # # # #                              "val_cte": r["CTE"], "val_loss": r["val_loss"]})
# # # # #                 print(f"  ✅ Best! score={best_score:.2f}"
# # # # #                       f"  ADE={r['ADE']:.1f} ATE={r['ATE']:.1f} CTE={r['CTE']:.1f}")
# # # # #             else:
# # # # #                 if rel_ep >= args.min_ep: patience_cnt += args.val_freq
# # # # #                 print(f"  No improve {patience_cnt}/{args.patience} (best={best_score:.1f})")
# # # # #                 if rel_ep >= args.min_ep and patience_cnt >= args.patience:
# # # # #                     print(f"  ⛔ Early stop @ ep{ep}"); break

# # # # #     print(f"\n  Done! best_score={best_score:.2f}")

# # # # #     # Test
# # # # #     if args.test_at_end and os.path.exists(best_ckpt):
# # # # #         print("\n  Loading best checkpoint...")
# # # # #         ck = torch.load(best_ckpt, map_location=device)
# # # # #         _unwrap(model).load_state_dict(ck["model"], strict=False)
# # # # #         if ema and ck.get("ema"):
# # # # #             for k, v in ck["ema"].items():
# # # # #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(device))

# # # # #         # [BUG-4 FIXED] ensemble từ periodic ckpts
# # # # #         if len(periodic_ckpts) >= 3:
# # # # #             ensemble_checkpoints(periodic_ckpts[-3:], model, device)

# # # # #         try:
# # # # #             _, test_loader = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # # #         except Exception:
# # # # #             print("  No test set → val"); test_loader = val_loader

# # # # #         r_test = evaluate(model, test_loader, device,
# # # # #                           tag="TEST (best + ensemble)",
# # # # #                           ema=ema, ref_targets=ST_TRANS_TEST,
# # # # #                           use_tta=args.tta_test, n_tta=args.n_tta)

# # # # #         val_ade = ck.get("val_ade", float("nan"))
# # # # #         print("\n  VAL vs TEST gap:")
# # # # #         for m_name, ref in [("ADE", ST_TRANS_TEST["ADE"]),
# # # # #                              ("72h", ST_TRANS_TEST["72h"])]:
# # # # #             tv  = r_test.get(m_name, float("nan"))
# # # # #             vv  = val_ade if m_name == "ADE" else float("nan")
# # # # #             gap = tv - vv if (np.isfinite(tv) and np.isfinite(vv)) else float("nan")
# # # # #             beat = "✅" if np.isfinite(tv) and tv < ref else "✗"
# # # # #             print(f"  {m_name}: val={vv:.1f}  test={tv:.1f}  gap={gap:+.1f}"
# # # # #                   f"  ST-Trans={ref:.1f}  {beat}")


# # # # # if __name__ == "__main__":
# # # # #     args = get_args()
# # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# # # # #     if args.dataset_root == "TCND_vn":
# # # # #         _auto = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
# # # # #         if os.path.isdir(_auto): args.dataset_root = _auto
# # # # #     main(args)

# # # # """
# # # # scripts/train_fm.py  ──  TC-FlowMatching v2.5 Training
# # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # Cải tiến so v2.1:
# # # #   [ATE-1] L_heading — heading continuation loss (ramp ep5→ep20)
# # # #   [ATE-2] L_momentum — speed momentum loss (ramp ep5→ep20)
# # # #   [AUG-1] Recurvature augmentation (20% probability)
# # # #   [SWA]   Stochastic Weight Averaging sau khi plateau (ep ~55–60)
# # # #   [HVAL]  Hard-val checkpoint selection
# # # #   [XAI]   Full XAI logging mỗi 10 epoch

# # # # Giữ nguyên từ v2.1 (những gì đã proven):
# # # #   ✅ sigma_inference=0.04 cố định
# # # #   ✅ L_reg linear step weights
# # # #   ✅ 2-group optimizer (encoder freeze)
# # # #   ✅ val loop KHÔNG augment
# # # #   ✅ 1-shot inference
# # # # """
# # # # from __future__ import annotations
# # # # import sys, os
# # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # import argparse, math, time
# # # # from collections import defaultdict
# # # # from typing import Dict, List, Optional

# # # # import numpy as np
# # # # import torch
# # # # import torch.optim as optim
# # # # from torch.amp import autocast, GradScaler

# # # # from Model.data.loader_training import data_loader
# # # # from Model.flow_matching_model import (
# # # #     TCFlowMatching, _norm_to_deg, _haversine_deg,
# # # #     EMAModel, augment_batch, hard_score_from_obs,
# # # #     compute_ensemble_uncertainty,
# # # #     compute_heading_deviation, compute_cte_contribution,
# # # #     compute_obs_attribution,
# # # # )

# # # # HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}
# # # # ST_TRANS_VAL  = {"ADE": 172.68, "ATE": 142.21, "CTE": 42.04,
# # # #                  "12h": 65.42, "24h": 104.67, "48h": 205.10, "72h": 321.39}
# # # # ST_TRANS_TEST = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
# # # #                  "12h": 77.5, "24h": 130.5, "48h": 269.9, "72h": 423.3}


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Utilities
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def _unwrap(m):
# # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # # # def move(batch, device):
# # # #     out = list(batch)
# # # #     for i, x in enumerate(out):
# # # #         if torch.is_tensor(x):   out[i] = x.to(device)
# # # #         elif isinstance(x, dict): out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # #                                              for k, v in x.items()}
# # # #     return out

# # # # def _ate_cte(pred_deg, gt_deg):
# # # #     """Compute ATE and CTE using bearing decomposition."""
# # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # #     if T < 2:
# # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # #         return z, z
# # # #     lo1 = torch.deg2rad(gt_deg[:T-1,:,0]);  la1 = torch.deg2rad(gt_deg[:T-1,:,1])
# # # #     lo2 = torch.deg2rad(gt_deg[1:T, :,0]);  la2 = torch.deg2rad(gt_deg[1:T, :,1])
# # # #     lo3 = torch.deg2rad(pred_deg[1:T,:,0]); la3 = torch.deg2rad(pred_deg[1:T,:,1])
# # # #     ya  = torch.sin(lo2-lo1)*torch.cos(la2)
# # # #     xa  = torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # # #     be  = torch.atan2(ya, xa)
# # # #     ye  = torch.sin(lo3-lo2)*torch.cos(la3)
# # # #     xe  = torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # # #     bee = torch.atan2(ye, xe)
# # # #     tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
# # # #     ang = bee - be
# # # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  2-group optimizer + scheduler (giữ nguyên từ v2.1)
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def build_optimizer(model, lr_velocity, lr_encoder, weight_decay):
# # # #     raw = _unwrap(model)
# # # #     enc_params = list(raw.encoder.parameters())
# # # #     vel_params = list(raw.velocity.parameters())
# # # #     return optim.AdamW([
# # # #         {"params": enc_params, "lr": lr_encoder,  "name": "encoder"},
# # # #         {"params": vel_params, "lr": lr_velocity, "name": "velocity"},
# # # #     ], weight_decay=weight_decay)


# # # # def get_lrs(opt):
# # # #     lr_enc = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "encoder")
# # # #     lr_vel = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "velocity")
# # # #     return lr_enc, lr_vel


# # # # class TwoGroupScheduler:
# # # #     def __init__(self, opt, warmup_epochs, total_epochs,
# # # #                  lr_vel, lr_vel_min, freeze_end_ep,
# # # #                  lr_enc_peak, encoder_warmup_epochs=5):
# # # #         self.opt         = opt
# # # #         self.warmup      = warmup_epochs
# # # #         self.total       = total_epochs
# # # #         self.lr_vel      = lr_vel
# # # #         self.lr_vel_min  = lr_vel_min
# # # #         self.freeze_end  = freeze_end_ep
# # # #         self.lr_enc_peak = lr_enc_peak
# # # #         self.enc_warmup  = encoder_warmup_epochs
# # # #         self.epoch       = 0

# # # #     def _cosine(self, ep_from, ep_to, lr_s, lr_e, ep):
# # # #         t = max(0., min(1., (ep - ep_from) / max(ep_to - ep_from, 1)))
# # # #         return lr_e + 0.5 * (lr_s - lr_e) * (1 + math.cos(math.pi * t))

# # # #     def step(self):
# # # #         ep = self.epoch
# # # #         # Velocity: warmup → cosine
# # # #         if ep < self.warmup:
# # # #             lr_vel = self.lr_vel * (0.1 + 0.9 * ep / max(self.warmup - 1, 1))
# # # #         else:
# # # #             lr_vel = self._cosine(self.warmup, self.total,
# # # #                                   self.lr_vel, self.lr_vel_min, ep)
# # # #         # Encoder: freeze → warmup → cosine
# # # #         if ep < self.freeze_end:
# # # #             lr_enc = 0.0
# # # #         elif ep < self.freeze_end + self.enc_warmup:
# # # #             lr_enc = self.lr_enc_peak * (ep - self.freeze_end) / self.enc_warmup
# # # #         else:
# # # #             lr_enc = self._cosine(
# # # #                 self.freeze_end + self.enc_warmup, self.total,
# # # #                 self.lr_enc_peak, self.lr_vel_min, ep)

# # # #         for pg in self.opt.param_groups:
# # # #             pg["lr"] = lr_enc if pg.get("name") == "encoder" else lr_vel
# # # #         self.epoch += 1
# # # #         return lr_vel, lr_enc


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  SWA handler
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # class SWAHandler:
# # # #     """
# # # #     Stochastic Weight Averaging sau khi val ADE plateau.

# # # #     Kích hoạt khi improvement < threshold km trong window lần val liên tiếp.
# # # #     Sau khi active, average model weights mỗi epoch.

# # # #     SWA cho weights "flatter" hơn trong loss landscape → generalize tốt hơn.
# # # #     Theo [Izmailov et al. 2018]: SWA giảm generalization gap 2–5%.
# # # #     """
# # # #     def __init__(self, swa_lr: float = 2e-6):
# # # #         self.swa_lr      = swa_lr
# # # #         self.active      = False
# # # #         self.start_ep    = None
# # # #         self.n_updates   = 0
# # # #         self.avg_state   = {}    # averaged state dict

# # # #     def should_activate(self, ade_history: List[float],
# # # #                          window: int = 3, threshold: float = 1.5) -> bool:
# # # #         if len(ade_history) < window:
# # # #             return False
# # # #         recent = ade_history[-window:]
# # # #         # Total improvement in last `window` val checks < threshold
# # # #         return (recent[0] - recent[-1]) < threshold

# # # #     def activate(self, model, opt, ep: int):
# # # #         """Switch optimizer to flat SWA LR and initialize avg_state."""
# # # #         self.active   = True
# # # #         self.start_ep = ep
# # # #         # Set all LR to SWA LR
# # # #         for pg in opt.param_groups:
# # # #             pg["lr"] = self.swa_lr
# # # #         # Initialize avg_state from current model
# # # #         m = _unwrap(model)
# # # #         self.avg_state = {k: v.detach().clone().float()
# # # #                           for k, v in m.state_dict().items()
# # # #                           if v.dtype.is_floating_point}
# # # #         self.n_updates = 1
# # # #         print(f"  *** SWA ACTIVATED @ ep{ep} (lr → {self.swa_lr:.1e}) ***")

# # # #     def update(self, model):
# # # #         """Running average: new_avg = (n*avg + w) / (n+1)"""
# # # #         if not self.active:
# # # #             return
# # # #         m  = _unwrap(model)
# # # #         sd = m.state_dict()
# # # #         n  = self.n_updates
# # # #         for k in self.avg_state:
# # # #             if k in sd:
# # # #                 self.avg_state[k] = (
# # # #                     (n * self.avg_state[k] + sd[k].detach().float()) / (n + 1)
# # # #                 )
# # # #         self.n_updates += 1

# # # #     def apply_to_model(self, model):
# # # #         """Copy averaged weights into model for evaluation."""
# # # #         if not self.active or not self.avg_state:
# # # #             return
# # # #         m  = _unwrap(model)
# # # #         sd = m.state_dict()
# # # #         for k in self.avg_state:
# # # #             if k in sd:
# # # #                 sd[k].copy_(self.avg_state[k].to(sd[k].device))

# # # #     def restore_from_backup(self, model, backup):
# # # #         """Restore model weights from backup after SWA evaluation."""
# # # #         m  = _unwrap(model)
# # # #         sd = m.state_dict()
# # # #         for k, v in backup.items():
# # # #             if k in sd:
# # # #                 sd[k].copy_(v)

# # # #     def save_avg_state(self, path: str, epoch: int, best_score: float,
# # # #                        extra: Optional[dict] = None):
# # # #         payload = {"epoch": epoch, "model": self.avg_state,
# # # #                    "best_score": best_score, "is_swa": True,
# # # #                    "swa_updates": self.n_updates}
# # # #         if extra:
# # # #             payload.update(extra)
# # # #         torch.save(payload, path)


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Hard-val evaluator
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # @torch.no_grad()
# # # # def evaluate_hard_val(model, val_loader, device, hard_threshold: float = 0.35,
# # # #                        n_ensemble: int = 20, ema=None, epoch_for_loss: int = 9999):
# # # #     """
# # # #     Evaluate ONLY on hard storms (hard_score > hard_threshold).

# # # #     Hard storms gần test distribution hơn vì test có nhiều recurvature.
# # # #     Dùng để chọn checkpoint tốt nhất cho test generalization.

# # # #     Returns dict với ADE, ATE, CTE của hard storms only.
# # # #     """
# # # #     bk = None
# # # #     if ema is not None:
# # # #         try: bk = ema.apply_to(model)
# # # #         except Exception: pass

# # # #     model.eval()
# # # #     all_ade, all_ate, all_cte = [], [], []
# # # #     n_hard = 0

# # # #     for batch in val_loader:
# # # #         bl = move(list(batch), device)
# # # #         B  = bl[0].shape[1]

# # # #         h_score   = hard_score_from_obs(bl[0][:, :, :2])  # [B]
# # # #         hard_mask = h_score > hard_threshold               # [B] bool

# # # #         if hard_mask.sum() == 0:
# # # #             continue

# # # #         hard_idx = hard_mask.nonzero(as_tuple=True)[0]  # [n_hard]

# # # #         # Build sub-batch of hard storms only
# # # #         bl_hard = list(bl)
# # # #         for i, item in enumerate(bl_hard):
# # # #             if torch.is_tensor(item):
# # # #                 # dim-1 is batch for [T_obs, B, feat] format
# # # #                 if item.dim() >= 2 and item.shape[1] == B:
# # # #                     bl_hard[i] = item[:, hard_idx, ...]
# # # #                 elif item.dim() >= 1 and item.shape[0] == B:
# # # #                     bl_hard[i] = item[hard_idx, ...]

# # # #         try:
# # # #             pred, _, _ = model.sample(bl_hard, num_ensemble=n_ensemble)
# # # #         except Exception as e:
# # # #             print(f"  hard val sample error: {e}")
# # # #             continue

# # # #         gt = bl_hard[1]
# # # #         T  = min(pred.shape[0], gt.shape[0])
# # # #         pred_deg = _norm_to_deg(pred[:T])
# # # #         gt_deg   = _norm_to_deg(gt[:T])
# # # #         dist     = _haversine_deg(pred_deg, gt_deg)
# # # #         ate, cte = _ate_cte(pred_deg, gt_deg)

# # # #         all_ade.extend(dist.mean(0).tolist())
# # # #         if ate.shape[0] > 0:
# # # #             all_ate.extend(ate.abs().mean(0).tolist())
# # # #             all_cte.extend(cte.abs().mean(0).tolist())
# # # #         n_hard += len(hard_idx)

# # # #     if bk is not None:
# # # #         try: ema.restore(model, bk)
# # # #         except Exception: pass

# # # #     def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # # #     result = {
# # # #         "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
# # # #         "n_hard": n_hard,
# # # #         "combined_score": 0.6*_m(all_ade) + 0.2*_m(all_ate) + 0.2*_m(all_cte),
# # # #     }
# # # #     return result


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Main evaluation
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # @torch.no_grad()
# # # # def evaluate(model, loader, device, tag: str = "",
# # # #              n_ensemble: int = 20, ema=None,
# # # #              ref_targets=None, use_tta: bool = False,
# # # #              n_tta: int = 5, epoch_for_loss: int = 9999,
# # # #              run_xai: bool = False, xai_batch=None) -> Dict:
# # # #     """
# # # #     Full evaluation với optional XAI logging và speed-scale TTA.

# # # #     KHÔNG augment — val distribution nhất quán với test distribution.
# # # #     """
# # # #     bk = None
# # # #     if ema is not None:
# # # #         try: bk = ema.apply_to(model)
# # # #         except Exception as e: print(f"  ⚠ EMA apply: {e}")

# # # #     model.eval()
# # # #     all_ade, all_ate, all_cte = [], [], []
# # # #     step_dist = defaultdict(list)
# # # #     sum_loss_w = sum_cfm_w = sum_head_w = sum_mom_w = 0.0
# # # #     sum_n = 0

# # # #     for batch in loader:
# # # #         bl = move(list(batch), device)
# # # #         gt = bl[1]; B = bl[0].shape[1]

# # # #         # Loss breakdown (no augment)
# # # #         try:
# # # #             bd = model.get_loss_breakdown(bl, epoch=epoch_for_loss)
# # # #             if torch.isfinite(bd["total"]):
# # # #                 sum_loss_w += bd["total"].item() * B
# # # #                 sum_cfm_w  += bd["l_cfm"] * B
# # # #                 sum_head_w += bd["l_heading"] * B
# # # #                 sum_mom_w  += bd["l_momentum"] * B
# # # #                 sum_n      += B
# # # #         except Exception:
# # # #             pass

# # # #         # Inference: standard or TTA
# # # #         if use_tta:
# # # #             obs    = bl[0]
# # # #             anchor = obs[-1:, :, :2].detach()
# # # #             speed_scales = [0.875, 0.9375, 1.0, 1.0625, 1.125]
# # # #             # n_tta controls how many scales to use
# # # #             scales_to_use = speed_scales[:n_tta] if n_tta < len(speed_scales) else speed_scales
# # # #             preds_tta = []
# # # #             weights_tta = []
# # # #             for scale in scales_to_use:
# # # #                 obs_s = obs.clone()
# # # #                 obs_s[..., :2] = anchor + (obs[..., :2] - anchor) * scale
# # # #                 bl_s = list(bl); bl_s[0] = obs_s
# # # #                 try:
# # # #                     pred_s, _, _ = model.sample(bl_s, num_ensemble=n_ensemble)
# # # #                     preds_tta.append(pred_s)
# # # #                     weights_tta.append(2.0 if abs(scale - 1.0) < 1e-6 else 1.0)
# # # #                 except Exception:
# # # #                     continue
# # # #             if not preds_tta:
# # # #                 continue
# # # #             total_w = sum(weights_tta)
# # # #             pred = sum(w / total_w * p for w, p in zip(weights_tta, preds_tta))
# # # #         else:
# # # #             try:
# # # #                 pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
# # # #             except Exception as e:
# # # #                 print(f"  sample error: {e}")
# # # #                 continue

# # # #         T        = min(pred.shape[0], gt.shape[0])
# # # #         pred_deg = _norm_to_deg(pred[:T])
# # # #         gt_deg   = _norm_to_deg(gt[:T])
# # # #         dist     = _haversine_deg(pred_deg, gt_deg)
# # # #         ate, cte = _ate_cte(pred_deg, gt_deg)

# # # #         all_ade.extend(dist.mean(0).tolist())
# # # #         if ate.shape[0] > 0:
# # # #             all_ate.extend(ate.abs().mean(0).tolist())
# # # #             all_cte.extend(cte.abs().mean(0).tolist())
# # # #         for h, s in HORIZON_STEPS.items():
# # # #             if s < T:
# # # #                 step_dist[h].extend(dist[s].tolist())

# # # #     if bk is not None:
# # # #         try: ema.restore(model, bk)
# # # #         except Exception: pass

# # # #     def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # # #     val_loss = sum_loss_w / max(sum_n, 1)

# # # #     result = {
# # # #         "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
# # # #         "n":   len(all_ade),
# # # #         "val_loss":     val_loss,
# # # #         "val_cfm_loss": sum_cfm_w  / max(sum_n, 1),
# # # #         "val_head_loss": sum_head_w / max(sum_n, 1),
# # # #         "val_mom_loss": sum_mom_w  / max(sum_n, 1),
# # # #     }
# # # #     for h in HORIZON_STEPS:
# # # #         result[f"{h}h"] = _m(step_dist[h])

# # # #     ade, ate_, cte_ = result["ADE"], result["ATE"], result["CTE"]
# # # #     result["combined_score"] = (
# # # #         0.6 * ade + 0.2 * ate_ + 0.2 * cte_
# # # #         if all(np.isfinite(x) for x in [ade, ate_, cte_]) else ade)

# # # #     ref = ref_targets or ST_TRANS_VAL
# # # #     def _v(k): return result.get(k, float("nan"))
# # # #     def _ok(k): return "✓" if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9) else "✗"

# # # #     tta_str = " [TTA]" if use_tta else ""
# # # #     print(f"\n  {'='*72}")
# # # #     print(f"  [{tag}]{tta_str}  n={result['n']}")
# # # #     print(f"  Val Loss : {val_loss:.6f}  "
# # # #           f"cfm={result['val_cfm_loss']:.6f}  "
# # # #           f"head={result['val_head_loss']:.6f}  "
# # # #           f"mom={result['val_mom_loss']:.6f}")
# # # #     print(f"  ADE={_v('ADE'):7.1f}km {_ok('ADE')}  "
# # # #           f"ATE={_v('ATE'):7.1f}km {_ok('ATE')}  "
# # # #           f"CTE={_v('CTE'):7.1f}km {_ok('CTE')}")
# # # #     print(f"  Combined = {_v('combined_score'):.1f}")
# # # #     print(f"  12h={_v('12h'):6.1f}  24h={_v('24h'):6.1f}  "
# # # #           f"48h={_v('48h'):6.1f}  72h={_v('72h'):6.1f} km")
# # # #     beat = [f"{k}={_v(k):.0f}<{ref.get(k,999):.0f}"
# # # #             for k in ["ADE","ATE","CTE","12h","24h","48h","72h"]
# # # #             if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9)]
# # # #     print(f"  BEAT: {' | '.join(beat) if beat else 'none yet'}")

# # # #     # ── XAI logging ─────────────────────────────────────────────────────────
# # # #     if run_xai and xai_batch is not None:
# # # #         try:
# # # #             model.train()  # keep dropout active for attribution
# # # #             with torch.no_grad():
# # # #                 xai_result = _unwrap(model).sample(xai_batch, return_xai=True)
# # # #             _, _, _, xai = xai_result

# # # #             print(f"  {'─'*60}")
# # # #             print(f"  XAI Summary")
# # # #             print(f"  {'─'*60}")

# # # #             # XAI-4: Uncertainty
# # # #             print(f"  [XAI-4] Uncertainty:"
# # # #                   f" 12h={xai['mean_12h_std']:.1f}km"
# # # #                   f"  72h={xai['mean_72h_std']:.1f}km"
# # # #                   f"  ratio={float(xai['uncertainty_ratio'].mean()):.2f}×")
# # # #             n_high = xai["high_uncertainty"].sum().item()
# # # #             if n_high > 0:
# # # #                 print(f"           ⚠ {n_high} storms with high 72h uncertainty (>80km)")

# # # #             # XAI-2: Hard score components
# # # #             hc = xai.get("hard_components", {})
# # # #             if hc:
# # # #                 print(f"  [XAI-2] HardScore:"
# # # #                       f" curv={float(hc['curvature'].mean()):.3f}"
# # # #                       f"  spd_var={float(hc['speed_var'].mean()):.3f}"
# # # #                       f"  dir_chg={float(hc['dir_change'].mean()):.3f}")

# # # #             # XAI-3: Physics components
# # # #             pc = xai.get("physics_components", {})
# # # #             if pc:
# # # #                 print(f"  [XAI-3] Physics(best):"
# # # #                       f" speed={float(pc['speed'].mean()):.3f}"
# # # #                       f"  smooth={float(pc['smooth'].mean()):.3f}"
# # # #                       f"  heading={float(pc['heading'].mean()):.3f}")

# # # #             # XAI-5: Speed comparison
# # # #             sc = xai.get("speed_comparison", {})
# # # #             if sc:
# # # #                 ratio = sc.get("speed_ratio", 1.0)
# # # #                 flag  = ("⚠ OVER" if ratio > 1.15
# # # #                          else ("⚠ UNDER" if ratio < 0.85 else "✓ OK"))
# # # #                 n_over  = sc.get("over_predict",  torch.zeros(1)).sum().item()
# # # #                 n_under = sc.get("under_predict", torch.zeros(1)).sum().item()
# # # #                 print(f"  [XAI-5] Speed:"
# # # #                       f" obs={sc['obs_speed_mean']:.1f}km/h"
# # # #                       f"  pred={sc['pred_speed_mean']:.1f}km/h"
# # # #                       f"  ratio={ratio:.2f} {flag}"
# # # #                       f"  (over:{int(n_over)} under:{int(n_under)})")

# # # #             # XAI-6: Heading deviation (ROOT CAUSE of ATE)
# # # #             hd = xai.get("heading_deviation_deg")
# # # #             if hd is not None:
# # # #                 hd_mean = hd.mean(1)  # [T-1] mean across batch
# # # #                 print(f"  [XAI-6] Heading deviation (deg):"
# # # #                       f" 12h={hd_mean[0].item():.1f}°"
# # # #                       f"  24h={hd_mean[min(2,len(hd_mean)-1)].item():.1f}°"
# # # #                       f"  72h={hd_mean[min(10,len(hd_mean)-1)].item():.1f}°")

# # # #             # XAI-7: ATE/CTE decomposition
# # # #             ac = xai.get("ate_cte_decomp", {})
# # # #             if ac:
# # # #                 print(f"  [XAI-7] Error decomp:"
# # # #                       f" ATE={ac['ate_abs_mean']:.1f}km"
# # # #                       f"  CTE={ac['cte_abs_mean']:.1f}km"
# # # #                       f"  (ATE/CTE ratio={ac['ate_abs_mean']/(ac['cte_abs_mean']+1e-3):.2f})")

# # # #             print(f"  {'─'*60}")
# # # #             result["xai"] = xai
# # # #         except Exception as e:
# # # #             print(f"  XAI error: {e}")
# # # #             import traceback; traceback.print_exc()

# # # #     print(f"  {'='*72}\n")
# # # #     return result


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Checkpoint
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def _save(path, epoch, model, opt, sched, best_score,
# # # #           ema=None, scaler=None, extra=None):
# # # #     m   = _unwrap(model)
# # # #     esd = None
# # # #     if ema is not None:
# # # #         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # # #         except Exception: pass
# # # #     payload = {
# # # #         "epoch": epoch, "model": m.state_dict(),
# # # #         "optimizer": opt.state_dict(),
# # # #         "scheduler": sched.epoch,
# # # #         "scaler": scaler.state_dict() if scaler is not None else None,
# # # #         "best_score": best_score,
# # # #         "best_ade":   best_score,   # compat
# # # #         "ema": esd,
# # # #     }
# # # #     if extra:
# # # #         payload.update(extra)
# # # #     torch.save(payload, path)


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Args
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def get_args():
# # # #     p = argparse.ArgumentParser(
# # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # #     p.add_argument("--dataset_root",           default="TCND_vn")
# # # #     p.add_argument("--obs_len",                default=8,      type=int)
# # # #     p.add_argument("--pred_len",               default=12,     type=int)
# # # #     p.add_argument("--num_workers",            default=2,      type=int)
# # # #     p.add_argument("--other_modal",            default="gph")
# # # #     p.add_argument("--delim",                  default=" ")
# # # #     p.add_argument("--skip",                   default=1,      type=int)
# # # #     p.add_argument("--min_ped",                default=1,      type=int)
# # # #     p.add_argument("--threshold",              default=0.002,  type=float)
# # # #     # Model (same defaults as v2.1)
# # # #     p.add_argument("--d_cond",                 default=256,    type=int)
# # # #     p.add_argument("--d_model",                default=256,    type=int)
# # # #     p.add_argument("--nhead",                  default=8,      type=int)
# # # #     p.add_argument("--num_dec_layers",         default=4,      type=int)
# # # #     p.add_argument("--dim_ff",                 default=512,    type=int)
# # # #     p.add_argument("--dropout",                default=0.1,    type=float)
# # # #     p.add_argument("--unet_in_ch",             default=13,     type=int)
# # # #     p.add_argument("--sigma_min",              default=0.04,   type=float)
# # # #     p.add_argument("--sigma_max",              default=0.08,   type=float)
# # # #     p.add_argument("--lambda_reg",             default=0.2,    type=float)
# # # #     # v2.5 new losses
# # # #     p.add_argument("--lambda_heading",         default=0.05,   type=float,
# # # #                    help="[ATE-1] heading continuation loss weight")
# # # #     p.add_argument("--lambda_momentum",        default=0.05,   type=float,
# # # #                    help="[ATE-2] speed momentum loss weight")
# # # #     p.add_argument("--use_ot",                 default=True,   action="store_true")
# # # #     p.add_argument("--no_ot",                  dest="use_ot",  action="store_false")
# # # #     p.add_argument("--ot_epsilon",             default=0.05,   type=float)
# # # #     p.add_argument("--n_ensemble",             default=20,     type=int)
# # # #     p.add_argument("--sigma_inference",        default=0.04,   type=float,
# # # #                    help="FIXED — do not set speed-conditioned (v2.4 failed)")
# # # #     p.add_argument("--n_inference_steps",      default=1,      type=int)
# # # #     # Training
# # # #     p.add_argument("--num_epochs",             default=150,    type=int)
# # # #     p.add_argument("--batch_size",             default=64,     type=int)
# # # #     p.add_argument("--lr",                     default=2e-4,   type=float)
# # # #     p.add_argument("--lr_min",                 default=1e-6,   type=float)
# # # #     p.add_argument("--warmup_epochs",          default=5,      type=int)
# # # #     p.add_argument("--weight_decay",           default=1e-4,   type=float)
# # # #     p.add_argument("--grad_clip",              default=1.0,    type=float)
# # # #     p.add_argument("--use_amp",                action="store_true", default=False)
# # # #     p.add_argument("--use_ema",                default=True,   action="store_true")
# # # #     p.add_argument("--no_ema",                 dest="use_ema", action="store_false")
# # # #     # Encoder freeze (from v2.1)
# # # #     p.add_argument("--freeze_encoder_epochs",  default=10,     type=int)
# # # #     p.add_argument("--encoder_warmup_epochs",  default=5,      type=int)
# # # #     p.add_argument("--lr_enc_peak",            default=5e-5,   type=float)
# # # #     # Eval
# # # #     p.add_argument("--val_freq",               default=5,      type=int)
# # # #     p.add_argument("--patience",               default=20,     type=int,
# # # #                    help="Reduced from 30 — plateau detected earlier with SWA")
# # # #     p.add_argument("--min_ep",                 default=20,     type=int)
# # # #     p.add_argument("--hard_val_threshold",     default=0.35,   type=float,
# # # #                    help="hard_score threshold for hard-val checkpoint")
# # # #     p.add_argument("--hard_val_freq",          default=10,     type=int,
# # # #                    help="Run hard-val evaluation every N epochs")
# # # #     # SWA
# # # #     p.add_argument("--swa_lr",                 default=2e-6,   type=float)
# # # #     p.add_argument("--swa_window",             default=3,      type=int,
# # # #                    help="Number of val checks to detect plateau for SWA")
# # # #     p.add_argument("--swa_threshold",          default=1.5,    type=float,
# # # #                    help="Min improvement (km) to NOT activate SWA")
# # # #     p.add_argument("--swa_min_ep",             default=50,     type=int,
# # # #                    help="Do not activate SWA before this epoch")
# # # #     # TTA at test time
# # # #     p.add_argument("--tta_test",               default=True,   action="store_true")
# # # #     p.add_argument("--n_tta",                  default=5,      type=int)
# # # #     p.add_argument("--multiscale_test",        default=True,   action="store_true",
# # # #                    help="Use multi-scale sigma ensemble at test time")
# # # #     # IO
# # # #     p.add_argument("--output_dir",             default="runs/fm_v25")
# # # #     p.add_argument("--gpu_num",                default="0")
# # # #     p.add_argument("--resume",                 default=None)
# # # #     p.add_argument("--test_at_end",            action="store_true", default=True)
# # # #     p.add_argument("--no_test",                dest="test_at_end", action="store_false")
# # # #     return p.parse_args()


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Main
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def main(args):
# # # #     if torch.cuda.is_available():
# # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # #     os.makedirs(args.output_dir, exist_ok=True)
# # # #     best_ckpt      = os.path.join(args.output_dir, "best_model.pth")
# # # #     hard_best_ckpt = os.path.join(args.output_dir, "hard_best_model.pth")
# # # #     swa_ckpt       = os.path.join(args.output_dir, "swa_model.pth")
# # # #     last_ckpt      = os.path.join(args.output_dir, "last_model.pth")

# # # #     print("=" * 72)
# # # #     print("  TC-FlowMatching v2.5")
# # # #     print(f"  [ATE-1] L_heading  weight={args.lambda_heading} (ramp ep5→ep20)")
# # # #     print(f"  [ATE-2] L_momentum weight={args.lambda_momentum} (ramp ep5→ep20)")
# # # #     print(f"  [AUG-1] Recurvature augmentation 20%")
# # # #     print(f"  [SWA]   Activation after plateau (threshold={args.swa_threshold}km)")
# # # #     print(f"  [HVAL]  Hard-val checkpoint (threshold={args.hard_val_threshold})")
# # # #     print(f"  KEEP:   sigma_inf={args.sigma_inference} FIXED, L_reg linear weights")
# # # #     print("=" * 72)

# # # #     # ── Data ──────────────────────────────────────────────────────────────
# # # #     print("\n  Loading data...")
# # # #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # #     vd, val_loader = data_loader(args, {"root": args.dataset_root, "type": "val"}, test=True)
# # # #     print(f"  train: {len(trd)} ({len(trl)} batches/ep)")
# # # #     print(f"  val:   {len(vd)} ({len(val_loader)} batches) ← FULL VAL")

# # # #     # One fixed val batch for XAI logging (same batch every epoch for comparability)
# # # #     _xai_batch_raw = next(iter(val_loader))
# # # #     xai_batch = move(list(_xai_batch_raw), device) if _xai_batch_raw is not None else None

# # # #     # ── Model ─────────────────────────────────────────────────────────────
# # # #     model = TCFlowMatching(
# # # #         pred_len=args.pred_len,        obs_len=args.obs_len,
# # # #         unet_in_ch=args.unet_in_ch,   d_cond=args.d_cond,
# # # #         d_model=args.d_model,          nhead=args.nhead,
# # # #         num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
# # # #         dropout=args.dropout,
# # # #         sigma_min=args.sigma_min,      sigma_max=args.sigma_max,
# # # #         lambda_reg=args.lambda_reg,
# # # #         lambda_heading=args.lambda_heading,
# # # #         lambda_momentum=args.lambda_momentum,
# # # #         use_ot=args.use_ot,            ot_epsilon=args.ot_epsilon,
# # # #         use_ema=args.use_ema,
# # # #         n_ensemble=args.n_ensemble,
# # # #         n_inference_steps=args.n_inference_steps,
# # # #         sigma_inference=args.sigma_inference,
# # # #     ).to(device)

# # # #     model.init_ema()
# # # #     ema = getattr(_unwrap(model), "_ema", None)

# # # #     raw   = _unwrap(model)
# # # #     n_enc = sum(p.numel() for p in raw.encoder.parameters())
# # # #     n_vel = sum(p.numel() for p in raw.velocity.parameters())
# # # #     print(f"\n  Encoder:  {n_enc:,}  VelocityTrans: {n_vel:,}  Total: {n_enc+n_vel:,}")

# # # #     # ── Optimizer + Scheduler ─────────────────────────────────────────────
# # # #     opt    = build_optimizer(model, lr_velocity=args.lr,
# # # #                              lr_encoder=0.0, weight_decay=args.weight_decay)
# # # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # # #     sched  = TwoGroupScheduler(
# # # #         opt=opt, warmup_epochs=args.warmup_epochs, total_epochs=args.num_epochs,
# # # #         lr_vel=args.lr, lr_vel_min=args.lr_min,
# # # #         freeze_end_ep=args.freeze_encoder_epochs, lr_enc_peak=args.lr_enc_peak,
# # # #         encoder_warmup_epochs=args.encoder_warmup_epochs)

# # # #     print(f"\n  LR velocity: {args.lr:.0e} → {args.lr_min:.0e}")
# # # #     print(f"  LR encoder:  0 ({args.freeze_encoder_epochs}ep)"
# # # #           f" → {args.lr_enc_peak:.0e} ({args.encoder_warmup_epochs}ep) → cosine")

# # # #     # ── SWA handler ───────────────────────────────────────────────────────
# # # #     swa = SWAHandler(swa_lr=args.swa_lr)

# # # #     # ── Resume ────────────────────────────────────────────────────────────
# # # #     start_ep     = 0
# # # #     best_score   = float("inf")
# # # #     best_hard    = float("inf")
# # # #     patience_cnt = 0
# # # #     val_ade_history = []

# # # #     if args.resume and os.path.exists(args.resume):
# # # #         ck = torch.load(args.resume, map_location=device)
# # # #         _unwrap(model).load_state_dict(ck["model"], strict=False)
# # # #         try: opt.load_state_dict(ck["optimizer"])
# # # #         except Exception as e: print(f"  ⚠ Opt: {e}")
# # # #         sched.epoch  = ck.get("scheduler", 0)
# # # #         start_ep     = ck.get("epoch", 0) + 1
# # # #         best_score   = ck.get("best_score", ck.get("best_ade", float("inf")))
# # # #         patience_cnt = ck.get("patience_cnt", 0)
# # # #         if scaler is not None and ck.get("scaler"):
# # # #             try: scaler.load_state_dict(ck["scaler"])
# # # #             except Exception: pass
# # # #         if ema and ck.get("ema"):
# # # #             for k, v in ck["ema"].items():
# # # #                 if k in ema.shadow:
# # # #                     ema.shadow[k].copy_(v.to(device))
# # # #         print(f"  ↩ Resume ep{start_ep}  best={best_score:.1f}  patience={patience_cnt}")

# # # #     try:
# # # #         model = torch.compile(model, mode="reduce-overhead")
# # # #         print("  torch.compile: ok")
# # # #     except Exception:
# # # #         pass

# # # #     # ── Training loop ─────────────────────────────────────────────────────
# # # #     nstep = len(trl)
# # # #     print(f"\n  TRAINING ({nstep} steps/ep × {args.num_epochs} ep)")
# # # #     print(f"  Train: augment=v2.5(shift+speed+recurv)  Val: NO augment")
# # # #     print(f"  Inference: 1-shot, sigma={args.sigma_inference} FIXED")
# # # #     print()

# # # #     for ep in range(start_ep, start_ep + args.num_epochs):
# # # #         # Freeze/unfreeze logic
# # # #         rel_ep = ep - start_ep
# # # #         freeze = rel_ep < args.freeze_encoder_epochs
# # # #         raw_enc = _unwrap(model).encoder
# # # #         for p in raw_enc.parameters():
# # # #             p.requires_grad_(not freeze)

# # # #         if rel_ep == 0 and freeze:
# # # #             print(f"  *** Ep{ep}: encoder frozen ({args.freeze_encoder_epochs}ep) ***")
# # # #         if rel_ep == args.freeze_encoder_epochs:
# # # #             print(f"\n  *** Ep{ep}: encoder unfrozen ***")

# # # #         model.train()
# # # #         sum_loss = sum_cfm = sum_reg = sum_head = sum_mom = sum_ade1 = 0.0
# # # #         t0_ep    = time.perf_counter()

# # # #         for i, batch in enumerate(trl):
# # # #             bl = move(list(batch), device)

# # # #             # [AUG-1] Augment only in training loop
# # # #             bl_aug = augment_batch(bl)

# # # #             opt.zero_grad()
# # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # #                 bd = model.get_loss_breakdown(bl_aug, epoch=ep)

# # # #             scaler.scale(bd["total"]).backward()
# # # #             scaler.unscale_(opt)
# # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

# # # #             # Zero grad for frozen encoder params
# # # #             if freeze:
# # # #                 for p in _unwrap(model).encoder.parameters():
# # # #                     if p.grad is not None:
# # # #                         p.grad.zero_()

# # # #             scaler.step(opt)
# # # #             scaler.update()
# # # #             model.ema_update()

# # # #             # SWA update (only after activation)
# # # #             swa.update(model)

# # # #             sum_loss += bd["total"].item(); sum_cfm  += bd["l_cfm"]
# # # #             sum_reg  += bd["l_reg"];        sum_head += bd["l_heading"]
# # # #             sum_mom  += bd["l_momentum"];   sum_ade1 += bd["ade_1step"]

# # # #             if i % 30 == 0:
# # # #                 lr_enc, lr_vel = get_lrs(opt)
# # # #                 enc_s = "frozen" if freeze else "active"
# # # #                 swa_s = " [SWA]" if swa.active else ""
# # # #                 print(f"  [{ep:>3}][{i:>3}/{nstep}]"
# # # #                       f"  tot={bd['total'].item():.4f}"
# # # #                       f"  cfm={bd['l_cfm']:.4f}"
# # # #                       f"  reg={bd['l_reg']:.4f}"
# # # #                       f"  head={bd['l_heading']:.4f}"
# # # #                       f"  mom={bd['l_momentum']:.4f}"
# # # #                       f"  lam_d={bd['lam_dir']:.2f}"
# # # #                       f"  ade1={bd['ade_1step']:.0f}km"
# # # #                       f"  enc={enc_s}{swa_s}"
# # # #                       f"  lr_vel={lr_vel:.2e}")

# # # #         train_loss = sum_loss / nstep
# # # #         lr_enc_used, lr_vel_used = get_lrs(opt)
# # # #         lr_vel_next, lr_enc_next = sched.step()

# # # #         print(f"\n  ── Ep{ep:>3}"
# # # #               f"  train={train_loss:.6f}"
# # # #               f"  cfm={sum_cfm/nstep:.4f}"
# # # #               f"  reg={sum_reg/nstep:.4f}"
# # # #               f"  head={sum_head/nstep:.4f}"
# # # #               f"  mom={sum_mom/nstep:.4f}"
# # # #               f"  ade1={sum_ade1/nstep:.0f}km"
# # # #               f"  lr_vel={lr_vel_used:.2e}"
# # # #               f"  t={time.perf_counter()-t0_ep:.0f}s")

# # # #         _save(last_ckpt, ep, model, opt, sched, best_score, ema, scaler)

# # # #         if ep % 5 == 0:
# # # #             ckpt_ep = os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth")
# # # #             _save(ckpt_ep, ep, model, opt, sched, best_score, ema, scaler)
# # # #             print(f"  💾 {ckpt_ep}")

# # # #         # ── Val evaluation ───────────────────────────────────────────────
# # # #         if rel_ep % args.val_freq == 0:
# # # #             run_xai_this = (rel_ep % 10 == 0)  # XAI every 10 epochs
# # # #             r = evaluate(model, val_loader, device,
# # # #                          tag=f"VAL ep{ep}",
# # # #                          n_ensemble=args.n_ensemble, ema=ema,
# # # #                          ref_targets=ST_TRANS_VAL,
# # # #                          epoch_for_loss=ep,
# # # #                          run_xai=run_xai_this, xai_batch=xai_batch)

# # # #             val_ade  = r["ADE"]
# # # #             score    = r["combined_score"]
# # # #             val_ade_history.append(val_ade)

# # # #             # Trend detection
# # # #             if len(val_ade_history) >= 4:
# # # #                 trend = (float(np.mean(val_ade_history[-2:]))
# # # #                          - float(np.mean(val_ade_history[-4:-2])))
# # # #                 trend_str = (f"↑{trend:+.1f}km ⚠" if trend >  5
# # # #                              else f"↓{trend:+.1f}km ✓" if trend < -5
# # # #                              else f"→{trend:+.1f}km flat")
# # # #             else:
# # # #                 trend_str = "—"

# # # #             print(f"  train={train_loss:.6f}  val_ADE={val_ade:.1f}km"
# # # #                   f"  combined={score:.1f}  trend={trend_str}")

# # # #             # ── SWA activation check ─────────────────────────────────────
# # # #             if (not swa.active
# # # #                     and ep >= args.swa_min_ep
# # # #                     and swa.should_activate(val_ade_history,
# # # #                                              window=args.swa_window,
# # # #                                              threshold=args.swa_threshold)):
# # # #                 swa.activate(model, opt, ep)

# # # #             # ── Best checkpoint (overall val) ────────────────────────────
# # # #             if score < best_score:
# # # #                 best_score   = score
# # # #                 patience_cnt = 0
# # # #                 _save(best_ckpt, ep, model, opt, sched, best_score, ema, scaler,
# # # #                       extra={"val_ade":   r["ADE"], "val_ate":   r["ATE"],
# # # #                              "val_cte":   r["CTE"], "val_loss":  r["val_loss"],
# # # #                              "patience_cnt": 0})
# # # #                 print(f"  ✅ Best! score={best_score:.2f}"
# # # #                       f"  ADE={r['ADE']:.1f} ATE={r['ATE']:.1f} CTE={r['CTE']:.1f}")
# # # #             else:
# # # #                 if rel_ep >= args.min_ep:
# # # #                     patience_cnt += args.val_freq
# # # #                 print(f"  No improve {patience_cnt}/{args.patience} (best={best_score:.1f})")
# # # #                 if rel_ep >= args.min_ep and patience_cnt >= args.patience:
# # # #                     print(f"  ⛔ Early stop @ ep{ep}")
# # # #                     break

# # # #         # ── Hard-val evaluation ──────────────────────────────────────────
# # # #         if rel_ep % args.hard_val_freq == 0 and rel_ep >= args.min_ep:
# # # #             r_hard = evaluate_hard_val(
# # # #                 model, val_loader, device,
# # # #                 hard_threshold=args.hard_val_threshold,
# # # #                 n_ensemble=args.n_ensemble, ema=ema, epoch_for_loss=ep)

# # # #             print(f"  [Hard-Val] n={r_hard['n_hard']}"
# # # #                   f"  ADE={r_hard['ADE']:.1f}"
# # # #                   f"  ATE={r_hard['ATE']:.1f}"
# # # #                   f"  CTE={r_hard['CTE']:.1f}"
# # # #                   f"  combined={r_hard['combined_score']:.1f}")

# # # #             if r_hard["combined_score"] < best_hard and r_hard["n_hard"] >= 10:
# # # #                 best_hard = r_hard["combined_score"]
# # # #                 _save(hard_best_ckpt, ep, model, opt, sched, best_hard, ema, scaler,
# # # #                       extra={"hard_val_ade": r_hard["ADE"],
# # # #                              "hard_val_ate": r_hard["ATE"],
# # # #                              "hard_val_cte": r_hard["CTE"],
# # # #                              "selection_criterion": "hard_val"})
# # # #                 print(f"  💎 Hard-best! score={best_hard:.2f}"
# # # #                       f"  ADE={r_hard['ADE']:.1f}")

# # # #         # ── SWA checkpoint ───────────────────────────────────────────────
# # # #         if swa.active and rel_ep % args.val_freq == 0 and swa.n_updates >= 10:
# # # #             # Apply SWA weights, evaluate, restore
# # # #             backup = {k: v.detach().clone()
# # # #                       for k, v in _unwrap(model).state_dict().items()
# # # #                       if v.dtype.is_floating_point}
# # # #             swa.apply_to_model(model)

# # # #             r_swa = evaluate(model, val_loader, device,
# # # #                              tag=f"SWA ep{ep}",
# # # #                              n_ensemble=args.n_ensemble, ema=None,
# # # #                              ref_targets=ST_TRANS_VAL, epoch_for_loss=ep)

# # # #             swa.restore_from_backup(model, backup)

# # # #             swa_score = r_swa["combined_score"]
# # # #             print(f"  [SWA] score={swa_score:.2f} ({swa.n_updates} updates)"
# # # #                   f"  vs best={best_score:.2f}")

# # # #             if swa_score < best_score:
# # # #                 best_score = swa_score
# # # #                 swa.save_avg_state(swa_ckpt, ep, best_score,
# # # #                                    extra={"val_ade": r_swa["ADE"],
# # # #                                           "val_ate": r_swa["ATE"],
# # # #                                           "val_cte": r_swa["CTE"]})
# # # #                 # Also save as best_ckpt for downstream compatibility
# # # #                 import shutil
# # # #                 shutil.copy(swa_ckpt, best_ckpt)
# # # #                 print(f"  ✅ SWA best! score={best_score:.2f}"
# # # #                       f"  ADE={r_swa['ADE']:.1f}")
# # # #                 patience_cnt = 0

# # # #     # ── End of training ───────────────────────────────────────────────────
# # # #     print(f"\n  Done! best_score={best_score:.2f}")

# # # #     # ── Test evaluation ───────────────────────────────────────────────────
# # # #     if not args.test_at_end:
# # # #         return

# # # #     print("\n  Loading best checkpoint for TEST...")
# # # #     if not os.path.exists(best_ckpt):
# # # #         print("  No best checkpoint found.")
# # # #         return

# # # #     ck = torch.load(best_ckpt, map_location=device)
# # # #     is_swa = ck.get("is_swa", False)
# # # #     _unwrap(model).load_state_dict(ck["model"], strict=False)
# # # #     if not is_swa and ema and ck.get("ema"):
# # # #         for k, v in ck["ema"].items():
# # # #             if k in ema.shadow:
# # # #                 ema.shadow[k].copy_(v.to(device))
# # # #     print(f"  Loaded ep{ck.get('epoch','?')} (is_swa={is_swa})")

# # # #     try:
# # # #         _, test_loader = data_loader(
# # # #             args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # #         print(f"  Test: {len(test_loader)} batches")
# # # #     except Exception:
# # # #         print("  No test set → using val")
# # # #         test_loader = val_loader

# # # #     # Standard test
# # # #     r_test = evaluate(model, test_loader, device,
# # # #                       tag="TEST (best ckpt)",
# # # #                       n_ensemble=args.n_ensemble,
# # # #                       ema=None if is_swa else ema,
# # # #                       ref_targets=ST_TRANS_TEST,
# # # #                       run_xai=True, xai_batch=xai_batch)

# # # #     # TTA test (speed-scale)
# # # #     if args.tta_test:
# # # #         r_tta = evaluate(model, test_loader, device,
# # # #                          tag="TEST+TTA (speed-scale)",
# # # #                          n_ensemble=args.n_ensemble,
# # # #                          ema=None if is_swa else ema,
# # # #                          ref_targets=ST_TRANS_TEST,
# # # #                          use_tta=True, n_tta=args.n_tta)
# # # #         print(f"\n  TTA improvement: ADE {r_test['ADE']:.1f} → {r_tta['ADE']:.1f}"
# # # #               f"  ATE {r_test['ATE']:.1f} → {r_tta['ATE']:.1f}"
# # # #               f"  CTE {r_test['CTE']:.1f} → {r_tta['CTE']:.1f}")

# # # #     # Multi-scale sigma test
# # # #     if args.multiscale_test:
# # # #         raw_m = _unwrap(model)
# # # #         if hasattr(raw_m, "sample_multiscale"):
# # # #             print("\n  Running multi-scale sigma test...")
# # # #             ms_ades, ms_ates, ms_ctes = [], [], []
# # # #             ms_steps = defaultdict(list)
# # # #             raw_m.eval()
# # # #             bk_ms = ema.apply_to(model) if (ema and not is_swa) else None
# # # #             with torch.no_grad():
# # # #                 for batch in test_loader:
# # # #                     bl = move(list(batch), device)
# # # #                     try:
# # # #                         pred, _, _ = raw_m.sample_multiscale(bl)
# # # #                     except Exception:
# # # #                         continue
# # # #                     gt   = bl[1]
# # # #                     T    = min(pred.shape[0], gt.shape[0])
# # # #                     pd   = _norm_to_deg(pred[:T])
# # # #                     gd   = _norm_to_deg(gt[:T])
# # # #                     dist = _haversine_deg(pd, gd)
# # # #                     ate, cte = _ate_cte(pd, gd)
# # # #                     ms_ades.extend(dist.mean(0).tolist())
# # # #                     if ate.shape[0] > 0:
# # # #                         ms_ates.extend(ate.abs().mean(0).tolist())
# # # #                         ms_ctes.extend(cte.abs().mean(0).tolist())
# # # #                     for h, s in HORIZON_STEPS.items():
# # # #                         if s < T: ms_steps[h].extend(dist[s].tolist())
# # # #             if bk_ms: ema.restore(model, bk_ms)
# # # #             def _mm(lst): return float(np.mean(lst)) if lst else float("nan")
# # # #             print(f"\n  Multi-scale sigma test:")
# # # #             print(f"  ADE={_mm(ms_ades):.1f}  ATE={_mm(ms_ates):.1f}  CTE={_mm(ms_ctes):.1f}")
# # # #             for h in HORIZON_STEPS:
# # # #                 print(f"    {h}h={_mm(ms_steps[h]):.1f}", end="")
# # # #             print()

# # # #     # ── Final comparison table ────────────────────────────────────────────
# # # #     val_ade_best = ck.get("val_ade", float("nan"))
# # # #     val_ate_best = ck.get("val_ate", float("nan"))
# # # #     val_cte_best = ck.get("val_cte", float("nan"))

# # # #     print("\n" + "=" * 72)
# # # #     print("  v2.5 FINAL RESULTS vs v2.1 baseline")
# # # #     print("=" * 72)
# # # #     print(f"  {'Metric':<12}  {'Val':>10}  {'Test':>10}  {'Gap':>10}  "
# # # #           f"{'v2.1 Test':>10}  Status")
# # # #     print("  " + "─" * 65)
# # # #     v21_test = {"ADE": 229.8, "ATE": 214.4, "CTE": 71.6}
# # # #     for m_name, val_v, test_v, v21_v in [
# # # #         ("ADE",  val_ade_best,   r_test["ADE"],  v21_test["ADE"]),
# # # #         ("ATE",  val_ate_best,   r_test["ATE"],  v21_test["ATE"]),
# # # #         ("CTE",  val_cte_best,   r_test["CTE"],  v21_test["CTE"]),
# # # #     ]:
# # # #         gap  = test_v - val_v if (np.isfinite(test_v) and np.isfinite(val_v)) else float("nan")
# # # #         impr = v21_v - test_v if np.isfinite(test_v) else float("nan")
# # # #         flag = (f"↓{impr:+.1f}" if np.isfinite(impr) and impr > 0 else
# # # #                 f"↑{-impr:+.1f}" if np.isfinite(impr) else "?")
# # # #         print(f"  {m_name:<12}  {val_v:>10.2f}  {test_v:>10.2f}  "
# # # #               f"{gap:>+10.2f}  {v21_v:>10.1f}  {flag}")
# # # #     print("=" * 72)


# # # # if __name__ == "__main__":
# # # #     args = get_args()
# # # #     np.random.seed(42)
# # # #     torch.manual_seed(42)
# # # #     if torch.cuda.is_available():
# # # #         torch.cuda.manual_seed_all(42)

# # # #     if args.dataset_root == "TCND_vn":
# # # #         _auto = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
# # # #         if os.path.isdir(_auto):
# # # #             args.dataset_root = _auto
# # # #             print(f"  Auto dataset: {_auto}")
# # # #     main(args)

# # # """
# # # scripts/train_fm.py  ──  TC-FlowMatching v2.6 Training
# # # ═══════════════════════════════════════════════════════════════════════════════

# # # Thay đổi từ v2.5 (dựa trên phân tích 140 epoch log đầy đủ):

# # #   [REMOVE]  L_momentum — làm ATE test tệ hơn +7.9km (anchor to val speed)
# # #   [UPGRADE] L_heading multi-step (4 steps, decay=0.5), weight 0.05→0.10
# # #   [NEW-INF] Speed calibration tại inference (per-storm, clip ±15%)
# # #   [UPGRADE] Physics score + displacement_score
# # #   [AUG-D]   Obs-speed scaling ×0.7–1.4 (15% prob)

# # # Giữ nguyên (proven):
# # #   ✅ sigma_inference=0.04 FIXED
# # #   ✅ L_reg softmax-linspace weights
# # #   ✅ 2-group optimizer, encoder freeze 10ep
# # #   ✅ val loop NO augmentation
# # #   ✅ 1-shot inference
# # #   ✅ SWA sau plateau
# # #   ✅ Hard-val checkpoint (HVAL)
# # #   ✅ XAI logging mỗi 10 epoch
# # # """
# # # from __future__ import annotations
# # # import sys, os
# # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # import argparse, math, time
# # # from collections import defaultdict
# # # from typing import Dict, List, Optional

# # # import numpy as np
# # # import torch
# # # import torch.optim as optim
# # # from torch.amp import autocast, GradScaler

# # # from Model.data.loader_training import data_loader
# # # from Model.flow_matching_model import (
# # #     TCFlowMatching, _norm_to_deg, _haversine_deg,
# # #     EMAModel, augment_batch, hard_score_from_obs,
# # #     compute_ensemble_uncertainty,
# # #     compute_heading_deviation, compute_cte_contribution,
# # #     compute_obs_attribution,
# # # )

# # # HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}
# # # ST_TRANS_VAL  = {"ADE": 172.68, "ATE": 142.21, "CTE": 42.04,
# # #                  "12h": 65.42, "24h": 104.67, "48h": 205.10, "72h": 321.39}
# # # ST_TRANS_TEST = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
# # #                  "12h": 77.5, "24h": 130.5, "48h": 269.9, "72h": 423.3}

# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Utilities
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def _unwrap(m):
# # #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # # def move(batch, device):
# # #     out = list(batch)
# # #     for i, x in enumerate(out):
# # #         if torch.is_tensor(x):
# # #             out[i] = x.to(device)
# # #         elif isinstance(x, dict):
# # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
# # #     return out

# # # def _ate_cte(pred_deg, gt_deg):
# # #     """Decompose error into along-track and cross-track components."""
# # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # #     if T < 2:
# # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # #         return z, z
# # #     lo1 = torch.deg2rad(gt_deg[:T-1,:,0]); la1 = torch.deg2rad(gt_deg[:T-1,:,1])
# # #     lo2 = torch.deg2rad(gt_deg[1:T, :,0]); la2 = torch.deg2rad(gt_deg[1:T, :,1])
# # #     lo3 = torch.deg2rad(pred_deg[1:T,:,0]); la3 = torch.deg2rad(pred_deg[1:T,:,1])
# # #     ya  = torch.sin(lo2-lo1)*torch.cos(la2)
# # #     xa  = torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # #     be  = torch.atan2(ya, xa)
# # #     ye  = torch.sin(lo3-lo2)*torch.cos(la3)
# # #     xe  = torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # #     bee = torch.atan2(ye, xe)
# # #     tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
# # #     ang = bee - be
# # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  2-group optimizer (encoder freeze pattern from v2.1)
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def build_optimizer(model, lr_velocity, lr_encoder, weight_decay):
# # #     raw = _unwrap(model)
# # #     return optim.AdamW([
# # #         {"params": list(raw.encoder.parameters()),  "lr": lr_encoder,  "name": "encoder"},
# # #         {"params": list(raw.velocity.parameters()), "lr": lr_velocity, "name": "velocity"},
# # #     ], weight_decay=weight_decay)

# # # def get_lrs(opt):
# # #     lr_enc = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "encoder")
# # #     lr_vel = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "velocity")
# # #     return lr_enc, lr_vel

# # # class TwoGroupScheduler:
# # #     def __init__(self, opt, warmup_epochs, total_epochs,
# # #                  lr_vel, lr_vel_min, freeze_end_ep,
# # #                  lr_enc_peak, encoder_warmup_epochs=5):
# # #         self.opt         = opt
# # #         self.warmup      = warmup_epochs
# # #         self.total       = total_epochs
# # #         self.lr_vel      = lr_vel
# # #         self.lr_vel_min  = lr_vel_min
# # #         self.freeze_end  = freeze_end_ep
# # #         self.lr_enc_peak = lr_enc_peak
# # #         self.enc_warmup  = encoder_warmup_epochs
# # #         self.epoch       = 0

# # #     def _cosine(self, ep_from, ep_to, lr_s, lr_e, ep):
# # #         t = max(0., min(1., (ep - ep_from) / max(ep_to - ep_from, 1)))
# # #         return lr_e + 0.5 * (lr_s - lr_e) * (1 + math.cos(math.pi * t))

# # #     def step(self):
# # #         ep = self.epoch
# # #         if ep < self.warmup:
# # #             lr_vel = self.lr_vel * (0.1 + 0.9 * ep / max(self.warmup - 1, 1))
# # #         else:
# # #             lr_vel = self._cosine(self.warmup, self.total, self.lr_vel, self.lr_vel_min, ep)
# # #         if ep < self.freeze_end:
# # #             lr_enc = 0.0
# # #         elif ep < self.freeze_end + self.enc_warmup:
# # #             lr_enc = self.lr_enc_peak * (ep - self.freeze_end) / self.enc_warmup
# # #         else:
# # #             lr_enc = self._cosine(self.freeze_end + self.enc_warmup, self.total,
# # #                                   self.lr_enc_peak, self.lr_vel_min, ep)
# # #         for pg in self.opt.param_groups:
# # #             pg["lr"] = lr_enc if pg.get("name") == "encoder" else lr_vel
# # #         self.epoch += 1
# # #         return lr_vel, lr_enc


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  SWA handler (same as v2.5 — proven)
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # class SWAHandler:
# # #     """
# # #     Stochastic Weight Averaging after val ADE plateau.
# # #     Activation: val ADE improvement < threshold km in last `window` checks.
# # #     After activation: average model weights each epoch → flatter loss landscape
# # #     → better generalization.
# # #     """
# # #     def __init__(self, swa_lr: float = 2e-6):
# # #         self.swa_lr    = swa_lr
# # #         self.active    = False
# # #         self.start_ep  = None
# # #         self.n_updates = 0
# # #         self.avg_state = {}

# # #     def should_activate(self, ade_history: List[float],
# # #                          window: int = 3, threshold: float = 1.5) -> bool:
# # #         if len(ade_history) < window: return False
# # #         return (ade_history[-window] - ade_history[-1]) < threshold

# # #     def activate(self, model, opt, ep: int):
# # #         self.active = True; self.start_ep = ep
# # #         for pg in opt.param_groups: pg["lr"] = self.swa_lr
# # #         m = _unwrap(model)
# # #         self.avg_state = {k: v.detach().clone().float()
# # #                           for k, v in m.state_dict().items()
# # #                           if v.dtype.is_floating_point}
# # #         self.n_updates = 1
# # #         print(f"  *** SWA ACTIVATED @ ep{ep} (lr → {self.swa_lr:.1e}) ***")

# # #     def update(self, model):
# # #         if not self.active: return
# # #         m = _unwrap(model); sd = m.state_dict(); n = self.n_updates
# # #         for k in self.avg_state:
# # #             if k in sd:
# # #                 self.avg_state[k] = (n * self.avg_state[k] + sd[k].detach().float()) / (n + 1)
# # #         self.n_updates += 1

# # #     def apply_to_model(self, model):
# # #         if not self.active or not self.avg_state: return
# # #         m = _unwrap(model); sd = m.state_dict()
# # #         for k in self.avg_state:
# # #             if k in sd: sd[k].copy_(self.avg_state[k].to(sd[k].device))

# # #     def restore_from_backup(self, model, backup):
# # #         m = _unwrap(model); sd = m.state_dict()
# # #         for k, v in backup.items():
# # #             if k in sd: sd[k].copy_(v)

# # #     def save_avg_state(self, path: str, epoch: int, best_score: float,
# # #                        extra: Optional[dict] = None):
# # #         payload = {"epoch": epoch, "model": self.avg_state,
# # #                    "best_score": best_score, "is_swa": True,
# # #                    "swa_updates": self.n_updates}
# # #         if extra: payload.update(extra)
# # #         torch.save(payload, path)


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Hard-val evaluator (HVAL)
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def evaluate_hard_val(model, val_loader, device, hard_threshold: float = 0.35,
# # #                        n_ensemble: int = 20, ema=None, epoch_for_loss: int = 9999):
# # #     """
# # #     Evaluate on hard storms only (hard_score > threshold).
# # #     Hard storms are closer to test distribution (more recurvature).
# # #     Best hard-val checkpoint often generalizes better than overall best.
# # #     """
# # #     bk = None
# # #     if ema is not None:
# # #         try: bk = ema.apply_to(model)
# # #         except Exception: pass

# # #     model.eval()
# # #     all_ade, all_ate, all_cte = [], [], []
# # #     n_hard = 0

# # #     for batch in val_loader:
# # #         bl = move(list(batch), device)
# # #         B  = bl[0].shape[1]
# # #         h_score   = hard_score_from_obs(bl[0][:, :, :2])
# # #         hard_mask = h_score > hard_threshold
# # #         if hard_mask.sum() == 0: continue
# # #         hard_idx  = hard_mask.nonzero(as_tuple=True)[0]

# # #         # Build sub-batch for hard storms
# # #         bl_h = list(bl)
# # #         for i, item in enumerate(bl_h):
# # #             if torch.is_tensor(item):
# # #                 if item.dim() >= 2 and item.shape[1] == B:
# # #                     bl_h[i] = item[:, hard_idx, ...]
# # #                 elif item.dim() >= 1 and item.shape[0] == B:
# # #                     bl_h[i] = item[hard_idx, ...]

# # #         try:
# # #             pred, _, _ = model.sample(bl_h, num_ensemble=n_ensemble)
# # #         except Exception as e:
# # #             print(f"  hard val error: {e}"); continue

# # #         gt = bl_h[1]
# # #         T  = min(pred.shape[0], gt.shape[0])
# # #         pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
# # #         dist = _haversine_deg(pd, gd)
# # #         ate, cte = _ate_cte(pd, gd)
# # #         all_ade.extend(dist.mean(0).tolist())
# # #         if ate.shape[0] > 0:
# # #             all_ate.extend(ate.abs().mean(0).tolist())
# # #             all_cte.extend(cte.abs().mean(0).tolist())
# # #         n_hard += len(hard_idx)

# # #     if bk is not None:
# # #         try: ema.restore(model, bk)
# # #         except Exception: pass

# # #     def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # #     return {
# # #         "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
# # #         "n_hard": n_hard,
# # #         "combined_score": 0.6*_m(all_ade) + 0.2*_m(all_ate) + 0.2*_m(all_cte),
# # #     }


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Main evaluation
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def evaluate(model, loader, device, tag: str = "",
# # #              n_ensemble: int = 20, ema=None,
# # #              ref_targets=None, use_tta: bool = False,
# # #              n_tta: int = 5, epoch_for_loss: int = 9999,
# # #              run_xai: bool = False, xai_batch=None) -> Dict:
# # #     """
# # #     Full evaluation. NO augmentation (val_loss is reliable generalization signal).
# # #     XAI logging every 10 epochs shows training dynamics.
# # #     """
# # #     bk = None
# # #     if ema is not None:
# # #         try: bk = ema.apply_to(model)
# # #         except Exception as e: print(f"  ⚠ EMA: {e}")

# # #     model.eval()
# # #     all_ade, all_ate, all_cte = [], [], []
# # #     step_dist = defaultdict(list)
# # #     sum_loss = sum_cfm = sum_head = 0.0
# # #     sum_n = 0

# # #     for batch in loader:
# # #         bl = move(list(batch), device)
# # #         gt = bl[1]; B = bl[0].shape[1]

# # #         # Val loss (no augmentation)
# # #         try:
# # #             bd = model.get_loss_breakdown(bl, epoch=epoch_for_loss)
# # #             if torch.isfinite(bd["total"]):
# # #                 sum_loss += bd["total"].item() * B
# # #                 sum_cfm  += bd["l_cfm"] * B
# # #                 sum_head += bd["l_heading"] * B
# # #                 sum_n    += B
# # #         except Exception: pass

# # #         # Inference (standard or speed-scale TTA)
# # #         if use_tta:
# # #             obs = bl[0]; anchor = obs[-1:, :, :2].detach()
# # #             scales = [0.875, 0.9375, 1.0, 1.0625, 1.125][:n_tta]
# # #             preds_t, weights_t = [], []
# # #             for sc in scales:
# # #                 obs_s = obs.clone(); obs_s[..., :2] = anchor + (obs[..., :2] - anchor) * sc
# # #                 bl_s = list(bl); bl_s[0] = obs_s
# # #                 try:
# # #                     p, _, _ = model.sample(bl_s, num_ensemble=n_ensemble)
# # #                     preds_t.append(p)
# # #                     weights_t.append(2.0 if abs(sc - 1.0) < 1e-6 else 1.0)
# # #                 except Exception: continue
# # #             if not preds_t: continue
# # #             tw   = sum(weights_t)
# # #             pred = sum(w / tw * p for w, p in zip(weights_t, preds_t))
# # #         else:
# # #             try:
# # #                 pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
# # #             except Exception as e:
# # #                 print(f"  sample error: {e}"); continue

# # #         T = min(pred.shape[0], gt.shape[0])
# # #         pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
# # #         dist = _haversine_deg(pd, gd)
# # #         ate, cte = _ate_cte(pd, gd)
# # #         all_ade.extend(dist.mean(0).tolist())
# # #         if ate.shape[0] > 0:
# # #             all_ate.extend(ate.abs().mean(0).tolist())
# # #             all_cte.extend(cte.abs().mean(0).tolist())
# # #         for h, s in HORIZON_STEPS.items():
# # #             if s < T: step_dist[h].extend(dist[s].tolist())

# # #     if bk is not None:
# # #         try: ema.restore(model, bk)
# # #         except Exception: pass

# # #     def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# # #     val_loss = sum_loss / max(sum_n, 1)

# # #     result = {
# # #         "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
# # #         "n": len(all_ade),
# # #         "val_loss": val_loss,
# # #         "val_cfm_loss":  sum_cfm / max(sum_n, 1),
# # #         "val_head_loss": sum_head / max(sum_n, 1),
# # #         "val_mom_loss":  0.0,   # always 0
# # #     }
# # #     for h in HORIZON_STEPS: result[f"{h}h"] = _m(step_dist[h])
# # #     ade, ate_, cte_ = result["ADE"], result["ATE"], result["CTE"]
# # #     result["combined_score"] = (
# # #         0.6 * ade + 0.2 * ate_ + 0.2 * cte_
# # #         if all(np.isfinite(x) for x in [ade, ate_, cte_]) else ade)

# # #     ref = ref_targets or ST_TRANS_VAL
# # #     def _v(k): return result.get(k, float("nan"))
# # #     def _ok(k): return "✓" if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9) else "✗"

# # #     tta_str = " [TTA]" if use_tta else ""
# # #     print(f"\n  {'='*72}")
# # #     print(f"  [{tag}]{tta_str}  n={result['n']}")
# # #     print(f"  Val Loss : {val_loss:.6f}  cfm={result['val_cfm_loss']:.6f}  "
# # #           f"head4s={result['val_head_loss']:.6f}  [mom=DISABLED]")
# # #     print(f"  ADE={_v('ADE'):7.1f}km {_ok('ADE')}  "
# # #           f"ATE={_v('ATE'):7.1f}km {_ok('ATE')}  "
# # #           f"CTE={_v('CTE'):7.1f}km {_ok('CTE')}")
# # #     print(f"  Combined = {_v('combined_score'):.1f}")
# # #     print(f"  12h={_v('12h'):6.1f}  24h={_v('24h'):6.1f}  "
# # #           f"48h={_v('48h'):6.1f}  72h={_v('72h'):6.1f} km")
# # #     beat = [f"{k}={_v(k):.0f}<{ref.get(k,999):.0f}"
# # #             for k in ["ADE","ATE","CTE","12h","24h","48h","72h"]
# # #             if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9)]
# # #     print(f"  BEAT: {' | '.join(beat) if beat else 'none yet'}")

# # #     # XAI logging
# # #     if run_xai and xai_batch is not None:
# # #         try:
# # #             _, _, _, xai = _unwrap(model).sample(xai_batch, return_xai=True)

# # #             print(f"  {'─'*60}")
# # #             print(f"  XAI Summary (fixed val batch)")
# # #             print(f"  {'─'*60}")

# # #             # XAI-4: Uncertainty
# # #             print(f"  [XAI-4] Uncertainty:"
# # #                   f" 12h={xai['mean_12h_std']:.1f}km"
# # #                   f"  72h={xai['mean_72h_std']:.1f}km"
# # #                   f"  ratio={float(xai['uncertainty_ratio'].mean()):.2f}×"
# # #                   f"  high_uncert={xai['high_uncertainty'].sum().item()}")

# # #             # XAI-2: Hard score components
# # #             hc = xai.get("hard_components", {})
# # #             if hc:
# # #                 print(f"  [XAI-2] HardScore:"
# # #                       f" curv={float(hc['curvature'].mean()):.3f}"
# # #                       f"  spd_var={float(hc['speed_var'].mean()):.3f}"
# # #                       f"  dir_chg={float(hc['dir_change'].mean()):.3f}")

# # #             # XAI-3: Physics score components
# # #             pc = xai.get("physics_components", {})
# # #             if pc:
# # #                 print(f"  [XAI-3] Physics:"
# # #                       f" speed={float(pc['speed'].mean()):.3f}"
# # #                       f"  smooth={float(pc['smooth'].mean()):.3f}"
# # #                       f"  heading={float(pc['heading'].mean()):.3f}")

# # #             # XAI-5: Speed comparison (shows calibration effect)
# # #             sc = xai.get("speed_comparison", {})
# # #             if sc:
# # #                 ratio = sc.get("speed_ratio", 1.0)
# # #                 flag  = ("⚠ OVER" if ratio > 1.15 else
# # #                          "⚠ UNDER" if ratio < 0.85 else "✓ OK")
# # #                 n_over  = sc.get("over_predict",  torch.zeros(1)).sum().item()
# # #                 n_under = sc.get("under_predict", torch.zeros(1)).sum().item()
# # #                 print(f"  [XAI-5] Speed (post-calibration):"
# # #                       f" obs={sc['obs_speed_mean']:.1f}km/h"
# # #                       f"  pred={sc['pred_speed_mean']:.1f}km/h"
# # #                       f"  ratio={ratio:.2f} {flag}"
# # #                       f"  (over:{int(n_over)} under:{int(n_under)})")

# # #             # XAI-6: Heading deviation — key metric for heading loss efficacy
# # #             hd = xai.get("heading_deviation_deg")
# # #             if hd is not None and hd.shape[0] >= 1:
# # #                 hd_mean = hd.mean(1)
# # #                 print(f"  [XAI-6] Heading deviation:"
# # #                       f" 12h={hd_mean[0].item():.1f}°"
# # #                       f"  24h={hd_mean[min(2,len(hd_mean)-1)].item():.1f}°"
# # #                       f"  72h={hd_mean[min(10,len(hd_mean)-1)].item():.1f}°"
# # #                       f"  (target: 12h<20°, 72h<100°)")

# # #             # XAI-7: ATE/CTE decomposition
# # #             ac = xai.get("ate_cte_decomp", {})
# # #             if ac:
# # #                 print(f"  [XAI-7] Error:"
# # #                       f" ATE={ac['ate_abs_mean']:.1f}km"
# # #                       f"  CTE={ac['cte_abs_mean']:.1f}km"
# # #                       f"  ratio={ac['ate_abs_mean']/(ac['cte_abs_mean']+1e-3):.2f}")
# # #             print(f"  {'─'*60}")
# # #             result["xai"] = xai
# # #         except Exception as e:
# # #             print(f"  XAI error: {e}")
# # #             import traceback; traceback.print_exc()

# # #     print(f"  {'='*72}\n")
# # #     return result


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Checkpoint
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def _save(path, epoch, model, opt, sched, best_score,
# # #           ema=None, scaler=None, extra=None):
# # #     m   = _unwrap(model)
# # #     esd = None
# # #     if ema is not None:
# # #         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # #         except Exception: pass
# # #     payload = {
# # #         "epoch": epoch, "model": m.state_dict(),
# # #         "optimizer": opt.state_dict(), "scheduler": sched.epoch,
# # #         "scaler": scaler.state_dict() if scaler is not None else None,
# # #         "best_score": best_score, "best_ade": best_score, "ema": esd,
# # #     }
# # #     if extra: payload.update(extra)
# # #     torch.save(payload, path)


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Args
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def get_args():
# # #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # #     # Data
# # #     p.add_argument("--dataset_root",           default="TCND_vn")
# # #     p.add_argument("--obs_len",                default=8,      type=int)
# # #     p.add_argument("--pred_len",               default=12,     type=int)
# # #     p.add_argument("--num_workers",            default=2,      type=int)
# # #     p.add_argument("--other_modal",            default="gph")
# # #     p.add_argument("--delim",                  default=" ")
# # #     p.add_argument("--skip",                   default=1,      type=int)
# # #     p.add_argument("--min_ped",                default=1,      type=int)
# # #     p.add_argument("--threshold",              default=0.002,  type=float)
# # #     # Model
# # #     p.add_argument("--d_cond",                 default=256,    type=int)
# # #     p.add_argument("--d_model",                default=256,    type=int)
# # #     p.add_argument("--nhead",                  default=8,      type=int)
# # #     p.add_argument("--num_dec_layers",         default=4,      type=int)
# # #     p.add_argument("--dim_ff",                 default=512,    type=int)
# # #     p.add_argument("--dropout",                default=0.1,    type=float)
# # #     p.add_argument("--unet_in_ch",             default=13,     type=int)
# # #     p.add_argument("--sigma_min",              default=0.04,   type=float)
# # #     p.add_argument("--sigma_max",              default=0.08,   type=float)
# # #     p.add_argument("--lambda_reg",             default=0.2,    type=float)
# # #     p.add_argument("--lambda_heading",         default=0.10,   type=float,
# # #                    help="[v2.6] multi-step heading, 4 steps, decay=0.5 (was 0.05 v2.5)")
# # #     p.add_argument("--lambda_momentum",        default=0.0,    type=float,
# # #                    help="[v2.6] DISABLED — hurt test ATE by +7.9km")
# # #     p.add_argument("--use_ot",                 default=True,   action="store_true")
# # #     p.add_argument("--no_ot",                  dest="use_ot",  action="store_false")
# # #     p.add_argument("--ot_epsilon",             default=0.05,   type=float)
# # #     p.add_argument("--n_ensemble",             default=20,     type=int)
# # #     p.add_argument("--sigma_inference",        default=0.04,   type=float)
# # #     p.add_argument("--n_inference_steps",      default=1,      type=int)
# # #     # Training
# # #     p.add_argument("--num_epochs",             default=150,    type=int)
# # #     p.add_argument("--batch_size",             default=64,     type=int)
# # #     p.add_argument("--lr",                     default=2e-4,   type=float)
# # #     p.add_argument("--lr_min",                 default=1e-6,   type=float)
# # #     p.add_argument("--warmup_epochs",          default=5,      type=int)
# # #     p.add_argument("--weight_decay",           default=1e-4,   type=float)
# # #     p.add_argument("--grad_clip",              default=1.0,    type=float)
# # #     p.add_argument("--use_amp",                action="store_true", default=False)
# # #     p.add_argument("--use_ema",                default=True,   action="store_true")
# # #     p.add_argument("--no_ema",                 dest="use_ema", action="store_false")
# # #     # Encoder freeze
# # #     p.add_argument("--freeze_encoder_epochs",  default=10,     type=int)
# # #     p.add_argument("--encoder_warmup_epochs",  default=5,      type=int)
# # #     p.add_argument("--lr_enc_peak",            default=5e-5,   type=float)
# # #     # Eval
# # #     p.add_argument("--val_freq",               default=5,      type=int)
# # #     p.add_argument("--patience",               default=20,     type=int)
# # #     p.add_argument("--min_ep",                 default=20,     type=int)
# # #     p.add_argument("--hard_val_threshold",     default=0.35,   type=float)
# # #     p.add_argument("--hard_val_freq",          default=10,     type=int)
# # #     # SWA
# # #     p.add_argument("--swa_lr",                 default=2e-6,   type=float)
# # #     p.add_argument("--swa_window",             default=3,      type=int)
# # #     p.add_argument("--swa_threshold",          default=1.5,    type=float)
# # #     p.add_argument("--swa_min_ep",             default=50,     type=int)
# # #     # Test
# # #     p.add_argument("--tta_test",               default=True,   action="store_true")
# # #     p.add_argument("--n_tta",                  default=5,      type=int)
# # #     p.add_argument("--multiscale_test",        default=True,   action="store_true")
# # #     # IO
# # #     p.add_argument("--output_dir",             default="runs/fm_v26")
# # #     p.add_argument("--gpu_num",                default="0")
# # #     p.add_argument("--resume",                 default=None)
# # #     p.add_argument("--test_at_end",            action="store_true", default=True)
# # #     p.add_argument("--no_test",                dest="test_at_end", action="store_false")
# # #     return p.parse_args()


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Main
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def main(args):
# # #     if torch.cuda.is_available():
# # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #     os.makedirs(args.output_dir, exist_ok=True)
# # #     best_ckpt      = os.path.join(args.output_dir, "best_model.pth")
# # #     hard_best_ckpt = os.path.join(args.output_dir, "hard_best_model.pth")
# # #     swa_ckpt       = os.path.join(args.output_dir, "swa_model.pth")
# # #     last_ckpt      = os.path.join(args.output_dir, "last_model.pth")

# # #     print("=" * 72)
# # #     print("  TC-FlowMatching v2.6")
# # #     print(f"  [REMOVE]  L_momentum (was hurting test ATE by +7.9km)")
# # #     print(f"  [UPGRADE] L_heading multi-step 4 steps decay=0.5 weight={args.lambda_heading}")
# # #     print(f"  [NEW-INF] Speed calibration at inference (clip 0.85–1.15)")
# # #     print(f"  [NEW-AUG] Obs-speed aug ×0.7–1.4 (15%)")
# # #     print(f"  [UPGRADE] Physics score + displacement_score")
# # #     print(f"  KEEP:     sigma_inf={args.sigma_inference} FIXED, L_reg linear weights")
# # #     print("=" * 72)

# # #     # ── Data ──────────────────────────────────────────────────────────────
# # #     print("\n  Loading data...")
# # #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
# # #     vd, val_loader = data_loader(args, {"root": args.dataset_root, "type": "val"}, test=True)
# # #     print(f"  train: {len(trd)} ({len(trl)} batches/ep)")
# # #     print(f"  val:   {len(vd)} ({len(val_loader)} batches)")

# # #     # Fixed XAI batch (same every epoch for comparability)
# # #     try:
# # #         xai_batch = move(list(next(iter(val_loader))), device)
# # #     except Exception:
# # #         xai_batch = None

# # #     # ── Model ─────────────────────────────────────────────────────────────
# # #     model = TCFlowMatching(
# # #         pred_len=args.pred_len,        obs_len=args.obs_len,
# # #         unet_in_ch=args.unet_in_ch,   d_cond=args.d_cond,
# # #         d_model=args.d_model,          nhead=args.nhead,
# # #         num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
# # #         dropout=args.dropout,
# # #         sigma_min=args.sigma_min,      sigma_max=args.sigma_max,
# # #         lambda_reg=args.lambda_reg,    lambda_heading=args.lambda_heading,
# # #         lambda_momentum=0.0,
# # #         use_ot=args.use_ot,            ot_epsilon=args.ot_epsilon,
# # #         use_ema=args.use_ema,          n_ensemble=args.n_ensemble,
# # #         n_inference_steps=args.n_inference_steps,
# # #         sigma_inference=args.sigma_inference,
# # #     ).to(device)

# # #     model.init_ema()
# # #     ema = getattr(_unwrap(model), "_ema", None)
# # #     raw = _unwrap(model)
# # #     n_enc = sum(p.numel() for p in raw.encoder.parameters())
# # #     n_vel = sum(p.numel() for p in raw.velocity.parameters())
# # #     print(f"\n  Encoder: {n_enc:,}  VelocityTrans: {n_vel:,}  Total: {n_enc+n_vel:,}")

# # #     # ── Optimizer + Scheduler ─────────────────────────────────────────────
# # #     opt    = build_optimizer(model, lr_velocity=args.lr, lr_encoder=0.0,
# # #                              weight_decay=args.weight_decay)
# # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # #     sched  = TwoGroupScheduler(
# # #         opt=opt, warmup_epochs=args.warmup_epochs, total_epochs=args.num_epochs,
# # #         lr_vel=args.lr, lr_vel_min=args.lr_min,
# # #         freeze_end_ep=args.freeze_encoder_epochs, lr_enc_peak=args.lr_enc_peak,
# # #         encoder_warmup_epochs=args.encoder_warmup_epochs)
# # #     print(f"\n  LR vel: {args.lr:.0e} → {args.lr_min:.0e}  "
# # #           f"LR enc: 0 ({args.freeze_encoder_epochs}ep) → {args.lr_enc_peak:.0e}")

# # #     swa = SWAHandler(swa_lr=args.swa_lr)

# # #     # ── Resume ────────────────────────────────────────────────────────────
# # #     start_ep = 0; best_score = float("inf"); best_hard = float("inf")
# # #     patience_cnt = 0; val_ade_history = []

# # #     if args.resume and os.path.exists(args.resume):
# # #         ck = torch.load(args.resume, map_location=device)
# # #         _unwrap(model).load_state_dict(ck["model"], strict=False)
# # #         try: opt.load_state_dict(ck["optimizer"])
# # #         except Exception as e: print(f"  ⚠ Opt: {e}")
# # #         sched.epoch  = ck.get("scheduler", 0)
# # #         start_ep     = ck.get("epoch", 0) + 1
# # #         best_score   = ck.get("best_score", ck.get("best_ade", float("inf")))
# # #         patience_cnt = ck.get("patience_cnt", 0)
# # #         if scaler and ck.get("scaler"):
# # #             try: scaler.load_state_dict(ck["scaler"])
# # #             except Exception: pass
# # #         if ema and ck.get("ema"):
# # #             for k, v in ck["ema"].items():
# # #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
# # #         print(f"  ↩ Resume ep{start_ep}  best={best_score:.1f}  patience={patience_cnt}")

# # #     try:
# # #         model = torch.compile(model, mode="reduce-overhead")
# # #         print("  torch.compile: ok")
# # #     except Exception: pass

# # #     # ── Training loop ─────────────────────────────────────────────────────
# # #     nstep = len(trl)
# # #     print(f"\n  TRAINING ({nstep} steps/ep × {args.num_epochs} ep)")
# # #     print(f"  Aug: shift±5km + speed×[0.85,1.15] + recurv±20° + obs-speed×[0.7,1.4] + noise")
# # #     print(f"  Loss: L_CFM + L_reg(linear) + L_heading_ms(4steps,decay=0.5)")
# # #     print(f"  Inf:  1-shot sigma=0.04 + speed_calibrate(±15%) + top3 physics")
# # #     print()

# # #     for ep in range(start_ep, start_ep + args.num_epochs):
# # #         rel_ep = ep - start_ep
# # #         freeze = rel_ep < args.freeze_encoder_epochs
# # #         for p in _unwrap(model).encoder.parameters():
# # #             p.requires_grad_(not freeze)

# # #         if rel_ep == 0 and freeze:
# # #             print(f"  *** Ep{ep}: encoder frozen ***")
# # #         if rel_ep == args.freeze_encoder_epochs:
# # #             print(f"\n  *** Ep{ep}: encoder unfrozen ***")

# # #         model.train()
# # #         sum_loss = sum_cfm = sum_reg = sum_head = sum_ade1 = 0.0
# # #         t0_ep = time.perf_counter()

# # #         for i, batch in enumerate(trl):
# # #             bl = move(list(batch), device)
# # #             bl_aug = augment_batch(bl)   # training augmentation

# # #             opt.zero_grad()
# # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # #                 bd = model.get_loss_breakdown(bl_aug, epoch=ep)

# # #             scaler.scale(bd["total"]).backward()
# # #             scaler.unscale_(opt)
# # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

# # #             if freeze:
# # #                 for p in _unwrap(model).encoder.parameters():
# # #                     if p.grad is not None: p.grad.zero_()

# # #             scaler.step(opt); scaler.update()
# # #             model.ema_update()
# # #             swa.update(model)

# # #             sum_loss += bd["total"].item(); sum_cfm  += bd["l_cfm"]
# # #             sum_reg  += bd["l_reg"];        sum_head += bd["l_heading"]
# # #             sum_ade1 += bd["ade_1step"]

# # #             if i % 30 == 0:
# # #                 _, lr_vel = get_lrs(opt)
# # #                 enc_s = "frozen" if freeze else "active"
# # #                 swa_s = " [SWA]" if swa.active else ""
# # #                 print(f"  [{ep:>3}][{i:>3}/{nstep}]"
# # #                       f"  tot={bd['total'].item():.4f}"
# # #                       f"  cfm={bd['l_cfm']:.4f}"
# # #                       f"  reg={bd['l_reg']:.4f}"
# # #                       f"  h4s={bd['l_heading']:.4f}"
# # #                       f"  lam_d={bd['lam_dir']:.2f}"
# # #                       f"  ade1={bd['ade_1step']:.0f}km"
# # #                       f"  enc={enc_s}{swa_s}"
# # #                       f"  lr={lr_vel:.2e}")

# # #         train_loss = sum_loss / nstep
# # #         _, lr_vel_used = get_lrs(opt)
# # #         sched.step()

# # #         print(f"\n  ── Ep{ep:>3}"
# # #               f"  train={train_loss:.6f}"
# # #               f"  cfm={sum_cfm/nstep:.4f}"
# # #               f"  reg={sum_reg/nstep:.4f}"
# # #               f"  h4s={sum_head/nstep:.4f}"
# # #               f"  ade1={sum_ade1/nstep:.0f}km"
# # #               f"  lr={lr_vel_used:.2e}"
# # #               f"  t={time.perf_counter()-t0_ep:.0f}s")

# # #         _save(last_ckpt, ep, model, opt, sched, best_score, ema, scaler)
# # #         if ep % 5 == 0:
# # #             ep_ckpt = os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth")
# # #             _save(ep_ckpt, ep, model, opt, sched, best_score, ema, scaler)
# # #             print(f"  💾 {ep_ckpt}")

# # #         # ── Val evaluation ───────────────────────────────────────────────
# # #         if rel_ep % args.val_freq == 0:
# # #             run_xai_this = (rel_ep % 10 == 0)
# # #             r = evaluate(model, val_loader, device, tag=f"VAL ep{ep}",
# # #                          n_ensemble=args.n_ensemble, ema=ema,
# # #                          ref_targets=ST_TRANS_VAL, epoch_for_loss=ep,
# # #                          run_xai=run_xai_this, xai_batch=xai_batch)

# # #             val_ade = r["ADE"]; score = r["combined_score"]
# # #             val_ade_history.append(val_ade)

# # #             # Trend
# # #             if len(val_ade_history) >= 4:
# # #                 trend = float(np.mean(val_ade_history[-2:])) - float(np.mean(val_ade_history[-4:-2]))
# # #                 trend_s = f"↑{trend:+.1f}km⚠" if trend > 5 else f"↓{trend:+.1f}km✓" if trend < -5 else f"→{trend:+.1f}flat"
# # #             else:
# # #                 trend_s = "—"
# # #             print(f"  train={train_loss:.6f}  val_ADE={val_ade:.1f}  combined={score:.1f}  trend={trend_s}")

# # #             # SWA activation check
# # #             if (not swa.active and ep >= args.swa_min_ep
# # #                     and swa.should_activate(val_ade_history, args.swa_window, args.swa_threshold)):
# # #                 swa.activate(model, opt, ep)

# # #             # Best checkpoint
# # #             if score < best_score:
# # #                 best_score = score; patience_cnt = 0
# # #                 _save(best_ckpt, ep, model, opt, sched, best_score, ema, scaler,
# # #                       extra={"val_ade": r["ADE"], "val_ate": r["ATE"],
# # #                              "val_cte": r["CTE"], "patience_cnt": 0})
# # #                 print(f"  ✅ Best! score={best_score:.2f}"
# # #                       f"  ADE={r['ADE']:.1f} ATE={r['ATE']:.1f} CTE={r['CTE']:.1f}")
# # #             else:
# # #                 if rel_ep >= args.min_ep: patience_cnt += args.val_freq
# # #                 print(f"  No improve {patience_cnt}/{args.patience} (best={best_score:.1f})")
# # #                 if rel_ep >= args.min_ep and patience_cnt >= args.patience:
# # #                     print(f"  ⛔ Early stop @ ep{ep}")
# # #                     break

# # #         # ── Hard-val evaluation ──────────────────────────────────────────
# # #         if rel_ep % args.hard_val_freq == 0 and rel_ep >= args.min_ep:
# # #             r_h = evaluate_hard_val(model, val_loader, device,
# # #                                      hard_threshold=args.hard_val_threshold,
# # #                                      n_ensemble=args.n_ensemble, ema=ema,
# # #                                      epoch_for_loss=ep)
# # #             print(f"  [HVAL] n={r_h['n_hard']}"
# # #                   f"  ADE={r_h['ADE']:.1f} ATE={r_h['ATE']:.1f} CTE={r_h['CTE']:.1f}"
# # #                   f"  combined={r_h['combined_score']:.1f}")
# # #             if r_h["combined_score"] < best_hard and r_h["n_hard"] >= 10:
# # #                 best_hard = r_h["combined_score"]
# # #                 _save(hard_best_ckpt, ep, model, opt, sched, best_hard, ema, scaler,
# # #                       extra={"hard_val_ade": r_h["ADE"], "selection_criterion": "hard_val"})
# # #                 print(f"  💎 Hard-best! score={best_hard:.2f} ADE={r_h['ADE']:.1f}")

# # #         # ── SWA eval ─────────────────────────────────────────────────────
# # #         if swa.active and rel_ep % args.val_freq == 0 and swa.n_updates >= 10:
# # #             backup = {k: v.detach().clone()
# # #                       for k, v in _unwrap(model).state_dict().items()
# # #                       if v.dtype.is_floating_point}
# # #             swa.apply_to_model(model)
# # #             r_swa = evaluate(model, val_loader, device, tag=f"SWA ep{ep}",
# # #                               n_ensemble=args.n_ensemble, ema=None,
# # #                               ref_targets=ST_TRANS_VAL, epoch_for_loss=ep)
# # #             swa.restore_from_backup(model, backup)

# # #             swa_score = r_swa["combined_score"]
# # #             print(f"  [SWA] score={swa_score:.2f} ({swa.n_updates} updates) vs best={best_score:.2f}")
# # #             if swa_score < best_score:
# # #                 best_score = swa_score; patience_cnt = 0
# # #                 swa.save_avg_state(swa_ckpt, ep, best_score,
# # #                                    extra={"val_ade": r_swa["ADE"],
# # #                                           "val_ate": r_swa["ATE"],
# # #                                           "val_cte": r_swa["CTE"]})
# # #                 import shutil; shutil.copy(swa_ckpt, best_ckpt)
# # #                 print(f"  ✅ SWA best! score={best_score:.2f} ADE={r_swa['ADE']:.1f}")

# # #     # ── Test evaluation ───────────────────────────────────────────────────
# # #     print(f"\n  Done! best_score={best_score:.2f}")
# # #     if not args.test_at_end: return

# # #     print("\n  Loading best checkpoint for TEST...")
# # #     if not os.path.exists(best_ckpt):
# # #         print("  No checkpoint found."); return

# # #     ck = torch.load(best_ckpt, map_location=device)
# # #     is_swa = ck.get("is_swa", False)
# # #     _unwrap(model).load_state_dict(ck["model"], strict=False)
# # #     if not is_swa and ema and ck.get("ema"):
# # #         for k, v in ck["ema"].items():
# # #             if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
# # #     print(f"  Loaded ep{ck.get('epoch','?')} (is_swa={is_swa})")

# # #     try:
# # #         _, test_loader = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
# # #         print(f"  Test: {len(test_loader)} batches")
# # #     except Exception:
# # #         print("  No test set → using val"); test_loader = val_loader

# # #     # Standard test
# # #     r_test = evaluate(model, test_loader, device, tag="TEST (best ckpt)",
# # #                       n_ensemble=args.n_ensemble, ema=None if is_swa else ema,
# # #                       ref_targets=ST_TRANS_TEST, run_xai=True, xai_batch=xai_batch)

# # #     # TTA test
# # #     if args.tta_test:
# # #         r_tta = evaluate(model, test_loader, device, tag="TEST+TTA",
# # #                          n_ensemble=args.n_ensemble, ema=None if is_swa else ema,
# # #                          ref_targets=ST_TRANS_TEST, use_tta=True, n_tta=args.n_tta)
# # #         print(f"\n  TTA: ADE {r_test['ADE']:.1f}→{r_tta['ADE']:.1f}  "
# # #               f"ATE {r_test['ATE']:.1f}→{r_tta['ATE']:.1f}  "
# # #               f"CTE {r_test['CTE']:.1f}→{r_tta['CTE']:.1f}")

# # #     # Multi-scale test
# # #     if args.multiscale_test:
# # #         raw_m = _unwrap(model)
# # #         if hasattr(raw_m, "sample_multiscale"):
# # #             print("\n  Multi-scale sigma test...")
# # #             ms_ades, ms_ates, ms_ctes = [], [], []
# # #             ms_steps = defaultdict(list)
# # #             raw_m.eval()
# # #             bk_ms = (ema.apply_to(model) if (ema and not is_swa) else None)
# # #             with torch.no_grad():
# # #                 for batch in test_loader:
# # #                     bl = move(list(batch), device)
# # #                     try:
# # #                         pred, _, _ = raw_m.sample_multiscale(bl)
# # #                     except Exception: continue
# # #                     gt = bl[1]; T = min(pred.shape[0], gt.shape[0])
# # #                     pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
# # #                     dist = _haversine_deg(pd, gd)
# # #                     ate, cte = _ate_cte(pd, gd)
# # #                     ms_ades.extend(dist.mean(0).tolist())
# # #                     if ate.shape[0] > 0:
# # #                         ms_ates.extend(ate.abs().mean(0).tolist())
# # #                         ms_ctes.extend(cte.abs().mean(0).tolist())
# # #                     for h, s in HORIZON_STEPS.items():
# # #                         if s < T: ms_steps[h].extend(dist[s].tolist())
# # #             if bk_ms: ema.restore(model, bk_ms)
# # #             def _mm(lst): return float(np.mean(lst)) if lst else float("nan")
# # #             print(f"  Multi-scale: ADE={_mm(ms_ades):.1f} ATE={_mm(ms_ates):.1f} CTE={_mm(ms_ctes):.1f}")

# # #     # Final comparison
# # #     val_ade_b = ck.get("val_ade", float("nan"))
# # #     val_ate_b = ck.get("val_ate", float("nan"))
# # #     val_cte_b = ck.get("val_cte", float("nan"))
# # #     v21 = {"ADE": 229.8, "ATE": 214.4, "CTE": 71.6}
# # #     print("\n" + "=" * 72)
# # #     print("  v2.6 FINAL RESULTS vs v2.1 baseline")
# # #     print("=" * 72)
# # #     print(f"  {'Metric':<10} {'Val':>10} {'Test':>10} {'Gap':>10} {'v2.1 Test':>12} Status")
# # #     print("  " + "─"*60)
# # #     for m_n, val_v, test_v, v21_v in [
# # #         ("ADE", val_ade_b, r_test["ADE"], v21["ADE"]),
# # #         ("ATE", val_ate_b, r_test["ATE"], v21["ATE"]),
# # #         ("CTE", val_cte_b, r_test["CTE"], v21["CTE"]),
# # #     ]:
# # #         gap  = test_v - val_v if (np.isfinite(test_v) and np.isfinite(val_v)) else float("nan")
# # #         impr = v21_v - test_v
# # #         flag = (f"↓{impr:+.1f}" if np.isfinite(impr) and impr > 0 else
# # #                 f"↑{-impr:+.1f}" if np.isfinite(impr) else "?")
# # #         print(f"  {m_n:<10} {val_v:>10.2f} {test_v:>10.2f} {gap:>+10.2f} {v21_v:>12.1f}  {flag}")
# # #     print("=" * 72)


# # # if __name__ == "__main__":
# # #     args = get_args()
# # #     np.random.seed(42); torch.manual_seed(42)
# # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# # #     if args.dataset_root == "TCND_vn":
# # #         _auto = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
# # #         if os.path.isdir(_auto):
# # #             args.dataset_root = _auto
# # #     main(args)

# # """
# # scripts/train_fm.py  ──  TC-FlowMatching v2.6 Training
# # ═══════════════════════════════════════════════════════════════════════════════

# # Thay đổi từ v2.5 (dựa trên phân tích 140 epoch log đầy đủ):

# #   [REMOVE]  L_momentum — làm ATE test tệ hơn +7.9km (anchor to val speed)
# #   [UPGRADE] L_heading multi-step (4 steps, decay=0.5), weight 0.05→0.10
# #   [NEW-INF] Speed calibration tại inference (per-storm, clip ±15%)
# #   [UPGRADE] Physics score + displacement_score
# #   [AUG-D]   Obs-speed scaling ×0.7–1.4 (15% prob)

# # Giữ nguyên (proven):
# #   ✅ sigma_inference=0.04 FIXED
# #   ✅ L_reg softmax-linspace weights
# #   ✅ 2-group optimizer, encoder freeze 10ep
# #   ✅ val loop NO augmentation
# #   ✅ 1-shot inference
# #   ✅ SWA sau plateau
# #   ✅ Hard-val checkpoint (HVAL)
# #   ✅ XAI logging mỗi 10 epoch
# # """
# # from __future__ import annotations
# # import sys, os
# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # import argparse, math, time
# # from collections import defaultdict
# # from typing import Dict, List, Optional

# # import numpy as np
# # import torch
# # import torch.optim as optim
# # from torch.amp import autocast, GradScaler

# # from Model.data.loader_training import data_loader
# # from Model.flow_matching_model import (
# #     TCFlowMatching, _norm_to_deg, _haversine_deg,
# #     EMAModel, augment_batch, hard_score_from_obs,
# #     compute_ensemble_uncertainty,
# #     compute_heading_deviation, compute_cte_contribution,
# #     compute_obs_attribution,
# # )

# # HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}
# # ST_TRANS_VAL  = {"ADE": 172.68, "ATE": 142.21, "CTE": 42.04,
# #                  "12h": 65.42, "24h": 104.67, "48h": 205.10, "72h": 321.39}
# # ST_TRANS_TEST = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
# #                  "12h": 77.5, "24h": 130.5, "48h": 269.9, "72h": 423.3}

# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Utilities
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _unwrap(m):
# #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # def move(batch, device):
# #     out = list(batch)
# #     for i, x in enumerate(out):
# #         if torch.is_tensor(x):
# #             out[i] = x.to(device)
# #         elif isinstance(x, dict):
# #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
# #     return out

# # def _ate_cte(pred_deg, gt_deg):
# #     """Decompose error into along-track and cross-track components."""
# #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# #     if T < 2:
# #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# #         return z, z
# #     lo1 = torch.deg2rad(gt_deg[:T-1,:,0]); la1 = torch.deg2rad(gt_deg[:T-1,:,1])
# #     lo2 = torch.deg2rad(gt_deg[1:T, :,0]); la2 = torch.deg2rad(gt_deg[1:T, :,1])
# #     lo3 = torch.deg2rad(pred_deg[1:T,:,0]); la3 = torch.deg2rad(pred_deg[1:T,:,1])
# #     ya  = torch.sin(lo2-lo1)*torch.cos(la2)
# #     xa  = torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# #     be  = torch.atan2(ya, xa)
# #     ye  = torch.sin(lo3-lo2)*torch.cos(la3)
# #     xe  = torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# #     bee = torch.atan2(ye, xe)
# #     tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
# #     ang = bee - be
# #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  2-group optimizer (encoder freeze pattern from v2.1)
# # # ─────────────────────────────────────────────────────────────────────────────

# # def build_optimizer(model, lr_velocity, lr_encoder, weight_decay):
# #     raw = _unwrap(model)
# #     return optim.AdamW([
# #         {"params": list(raw.encoder.parameters()),  "lr": lr_encoder,  "name": "encoder"},
# #         {"params": list(raw.velocity.parameters()), "lr": lr_velocity, "name": "velocity"},
# #     ], weight_decay=weight_decay)

# # def get_lrs(opt):
# #     lr_enc = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "encoder")
# #     lr_vel = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "velocity")
# #     return lr_enc, lr_vel

# # class TwoGroupScheduler:
# #     def __init__(self, opt, warmup_epochs, total_epochs,
# #                  lr_vel, lr_vel_min, freeze_end_ep,
# #                  lr_enc_peak, encoder_warmup_epochs=5):
# #         self.opt         = opt
# #         self.warmup      = warmup_epochs
# #         self.total       = total_epochs
# #         self.lr_vel      = lr_vel
# #         self.lr_vel_min  = lr_vel_min
# #         self.freeze_end  = freeze_end_ep
# #         self.lr_enc_peak = lr_enc_peak
# #         self.enc_warmup  = encoder_warmup_epochs
# #         self.epoch       = 0

# #     def _cosine(self, ep_from, ep_to, lr_s, lr_e, ep):
# #         t = max(0., min(1., (ep - ep_from) / max(ep_to - ep_from, 1)))
# #         return lr_e + 0.5 * (lr_s - lr_e) * (1 + math.cos(math.pi * t))

# #     def step(self):
# #         ep = self.epoch
# #         if ep < self.warmup:
# #             lr_vel = self.lr_vel * (0.1 + 0.9 * ep / max(self.warmup - 1, 1))
# #         else:
# #             lr_vel = self._cosine(self.warmup, self.total, self.lr_vel, self.lr_vel_min, ep)
# #         if ep < self.freeze_end:
# #             lr_enc = 0.0
# #         elif ep < self.freeze_end + self.enc_warmup:
# #             lr_enc = self.lr_enc_peak * (ep - self.freeze_end) / self.enc_warmup
# #         else:
# #             lr_enc = self._cosine(self.freeze_end + self.enc_warmup, self.total,
# #                                   self.lr_enc_peak, self.lr_vel_min, ep)
# #         for pg in self.opt.param_groups:
# #             pg["lr"] = lr_enc if pg.get("name") == "encoder" else lr_vel
# #         self.epoch += 1
# #         return lr_vel, lr_enc


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  SWA handler (same as v2.5 — proven)
# # # ─────────────────────────────────────────────────────────────────────────────

# # class SWAHandler:
# #     """
# #     Stochastic Weight Averaging after val ADE plateau.
# #     Activation: val ADE improvement < threshold km in last `window` checks.
# #     After activation: average model weights each epoch → flatter loss landscape
# #     → better generalization.
# #     """
# #     def __init__(self, swa_lr: float = 2e-6):
# #         self.swa_lr    = swa_lr
# #         self.active    = False
# #         self.start_ep  = None
# #         self.n_updates = 0
# #         self.avg_state = {}

# #     def should_activate(self, ade_history: List[float],
# #                          window: int = 3, threshold: float = 1.5) -> bool:
# #         if len(ade_history) < window: return False
# #         return (ade_history[-window] - ade_history[-1]) < threshold

# #     def activate(self, model, opt, ep: int):
# #         self.active = True; self.start_ep = ep
# #         for pg in opt.param_groups: pg["lr"] = self.swa_lr
# #         m = _unwrap(model)
# #         self.avg_state = {k: v.detach().clone().float()
# #                           for k, v in m.state_dict().items()
# #                           if v.dtype.is_floating_point}
# #         self.n_updates = 1
# #         print(f"  *** SWA ACTIVATED @ ep{ep} (lr → {self.swa_lr:.1e}) ***")

# #     def update(self, model):
# #         if not self.active: return
# #         m = _unwrap(model); sd = m.state_dict(); n = self.n_updates
# #         for k in self.avg_state:
# #             if k in sd:
# #                 self.avg_state[k] = (n * self.avg_state[k] + sd[k].detach().float()) / (n + 1)
# #         self.n_updates += 1

# #     def apply_to_model(self, model):
# #         if not self.active or not self.avg_state: return
# #         m = _unwrap(model); sd = m.state_dict()
# #         for k in self.avg_state:
# #             if k in sd: sd[k].copy_(self.avg_state[k].to(sd[k].device))

# #     def restore_from_backup(self, model, backup):
# #         m = _unwrap(model); sd = m.state_dict()
# #         for k, v in backup.items():
# #             if k in sd: sd[k].copy_(v)

# #     def save_avg_state(self, path: str, epoch: int, best_score: float,
# #                        extra: Optional[dict] = None):
# #         payload = {"epoch": epoch, "model": self.avg_state,
# #                    "best_score": best_score, "is_swa": True,
# #                    "swa_updates": self.n_updates}
# #         if extra: payload.update(extra)
# #         torch.save(payload, path)


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Hard-val evaluator (HVAL)
# # # ─────────────────────────────────────────────────────────────────────────────

# # @torch.no_grad()
# # def evaluate_hard_val(model, val_loader, device, hard_threshold: float = 0.35,
# #                        n_ensemble: int = 20, ema=None, epoch_for_loss: int = 9999):
# #     """
# #     Evaluate on hard storms only (hard_score > threshold).
# #     Hard storms are closer to test distribution (more recurvature).
# #     Best hard-val checkpoint often generalizes better than overall best.
# #     """
# #     bk = None
# #     if ema is not None:
# #         try: bk = ema.apply_to(model)
# #         except Exception: pass

# #     model.eval()
# #     all_ade, all_ate, all_cte = [], [], []
# #     n_hard = 0

# #     for batch in val_loader:
# #         bl = move(list(batch), device)
# #         B  = bl[0].shape[1]
# #         h_score   = hard_score_from_obs(bl[0][:, :, :2])
# #         hard_mask = h_score > hard_threshold
# #         if hard_mask.sum() == 0: continue
# #         hard_idx  = hard_mask.nonzero(as_tuple=True)[0]

# #         # Build sub-batch for hard storms
# #         bl_h = list(bl)
# #         for i, item in enumerate(bl_h):
# #             if torch.is_tensor(item):
# #                 if item.dim() >= 2 and item.shape[1] == B:
# #                     bl_h[i] = item[:, hard_idx, ...]
# #                 elif item.dim() >= 1 and item.shape[0] == B:
# #                     bl_h[i] = item[hard_idx, ...]

# #         try:
# #             pred, _, _ = model.sample(bl_h, num_ensemble=n_ensemble)
# #         except Exception as e:
# #             print(f"  hard val error: {e}"); continue

# #         gt = bl_h[1]
# #         T  = min(pred.shape[0], gt.shape[0])
# #         pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
# #         dist = _haversine_deg(pd, gd)
# #         ate, cte = _ate_cte(pd, gd)
# #         all_ade.extend(dist.mean(0).tolist())
# #         if ate.shape[0] > 0:
# #             all_ate.extend(ate.abs().mean(0).tolist())
# #             all_cte.extend(cte.abs().mean(0).tolist())
# #         n_hard += len(hard_idx)

# #     if bk is not None:
# #         try: ema.restore(model, bk)
# #         except Exception: pass

# #     def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# #     return {
# #         "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
# #         "n_hard": n_hard,
# #         "combined_score": 0.6*_m(all_ade) + 0.2*_m(all_ate) + 0.2*_m(all_cte),
# #     }


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Main evaluation
# # # ─────────────────────────────────────────────────────────────────────────────

# # @torch.no_grad()
# # def evaluate(model, loader, device, tag: str = "",
# #              n_ensemble: int = 20, ema=None,
# #              ref_targets=None, use_tta: bool = False,
# #              n_tta: int = 5, epoch_for_loss: int = 9999,
# #              run_xai: bool = False, xai_batch=None) -> Dict:
# #     """
# #     Full evaluation. NO augmentation (val_loss is reliable generalization signal).
# #     XAI logging every 10 epochs shows training dynamics.
# #     """
# #     bk = None
# #     if ema is not None:
# #         try: bk = ema.apply_to(model)
# #         except Exception as e: print(f"  ⚠ EMA: {e}")

# #     model.eval()
# #     all_ade, all_ate, all_cte = [], [], []
# #     step_dist = defaultdict(list)
# #     sum_loss = sum_cfm = sum_head = 0.0
# #     sum_n = 0

# #     for batch in loader:
# #         bl = move(list(batch), device)
# #         gt = bl[1]; B = bl[0].shape[1]

# #         # Val loss (no augmentation)
# #         try:
# #             bd = model.get_loss_breakdown(bl, epoch=epoch_for_loss)
# #             if torch.isfinite(bd["total"]):
# #                 sum_loss += bd["total"].item() * B
# #                 sum_cfm  += bd["l_cfm"] * B
# #                 sum_head += bd["l_heading"] * B
# #                 sum_n    += B
# #         except Exception: pass

# #         # Inference (standard or speed-scale TTA)
# #         if use_tta:
# #             obs = bl[0]; anchor = obs[-1:, :, :2].detach()
# #             scales = [0.875, 0.9375, 1.0, 1.0625, 1.125][:n_tta]
# #             preds_t, weights_t = [], []
# #             for sc in scales:
# #                 obs_s = obs.clone(); obs_s[..., :2] = anchor + (obs[..., :2] - anchor) * sc
# #                 bl_s = list(bl); bl_s[0] = obs_s
# #                 try:
# #                     p, _, _ = model.sample(bl_s, num_ensemble=n_ensemble)
# #                     preds_t.append(p)
# #                     weights_t.append(2.0 if abs(sc - 1.0) < 1e-6 else 1.0)
# #                 except Exception: continue
# #             if not preds_t: continue
# #             tw   = sum(weights_t)
# #             pred = sum(w / tw * p for w, p in zip(weights_t, preds_t))
# #         else:
# #             try:
# #                 pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
# #             except Exception as e:
# #                 print(f"  sample error: {e}"); continue

# #         T = min(pred.shape[0], gt.shape[0])
# #         pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
# #         dist = _haversine_deg(pd, gd)
# #         ate, cte = _ate_cte(pd, gd)
# #         all_ade.extend(dist.mean(0).tolist())
# #         if ate.shape[0] > 0:
# #             all_ate.extend(ate.abs().mean(0).tolist())
# #             all_cte.extend(cte.abs().mean(0).tolist())
# #         for h, s in HORIZON_STEPS.items():
# #             if s < T: step_dist[h].extend(dist[s].tolist())

# #     if bk is not None:
# #         try: ema.restore(model, bk)
# #         except Exception: pass

# #     def _m(lst): return float(np.mean(lst)) if lst else float("nan")
# #     val_loss = sum_loss / max(sum_n, 1)

# #     result = {
# #         "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
# #         "n": len(all_ade),
# #         "val_loss": val_loss,
# #         "val_cfm_loss":  sum_cfm / max(sum_n, 1),
# #         "val_head_loss": sum_head / max(sum_n, 1),
# #         "val_mom_loss":  0.0,   # always 0
# #     }
# #     for h in HORIZON_STEPS: result[f"{h}h"] = _m(step_dist[h])
# #     ade, ate_, cte_ = result["ADE"], result["ATE"], result["CTE"]
# #     result["combined_score"] = (
# #         0.6 * ade + 0.2 * ate_ + 0.2 * cte_
# #         if all(np.isfinite(x) for x in [ade, ate_, cte_]) else ade)

# #     ref = ref_targets or ST_TRANS_VAL
# #     def _v(k): return result.get(k, float("nan"))
# #     def _ok(k): return "✓" if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9) else "✗"

# #     tta_str = " [TTA]" if use_tta else ""
# #     print(f"\n  {'='*72}")
# #     print(f"  [{tag}]{tta_str}  n={result['n']}")
# #     print(f"  Val Loss : {val_loss:.6f}  cfm={result['val_cfm_loss']:.6f}  "
# #           f"head4s={result['val_head_loss']:.6f}  [mom=DISABLED]")
# #     print(f"  ADE={_v('ADE'):7.1f}km {_ok('ADE')}  "
# #           f"ATE={_v('ATE'):7.1f}km {_ok('ATE')}  "
# #           f"CTE={_v('CTE'):7.1f}km {_ok('CTE')}")
# #     print(f"  Combined = {_v('combined_score'):.1f}")
# #     print(f"  12h={_v('12h'):6.1f}  24h={_v('24h'):6.1f}  "
# #           f"48h={_v('48h'):6.1f}  72h={_v('72h'):6.1f} km")
# #     beat = [f"{k}={_v(k):.0f}<{ref.get(k,999):.0f}"
# #             for k in ["ADE","ATE","CTE","12h","24h","48h","72h"]
# #             if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9)]
# #     print(f"  BEAT: {' | '.join(beat) if beat else 'none yet'}")

# #     # XAI logging
# #     if run_xai and xai_batch is not None:
# #         try:
# #             _, _, _, xai = _unwrap(model).sample(xai_batch, return_xai=True)

# #             print(f"  {'─'*60}")
# #             print(f"  XAI Summary (fixed val batch)")
# #             print(f"  {'─'*60}")

# #             # XAI-4: Uncertainty
# #             print(f"  [XAI-4] Uncertainty:"
# #                   f" 12h={xai['mean_12h_std']:.1f}km"
# #                   f"  72h={xai['mean_72h_std']:.1f}km"
# #                   f"  ratio={float(xai['uncertainty_ratio'].mean()):.2f}×"
# #                   f"  high_uncert={xai['high_uncertainty'].sum().item()}")

# #             # XAI-2: Hard score components (v2.7: thêm speed_cv)
# #             hc = xai.get("hard_components", {})
# #             if hc:
# #                 spd_cv_str = f"  spd_cv={float(hc['speed_cv'].mean()):.3f}" if 'speed_cv' in hc else ""
# #                 print(f"  [XAI-2] HardScore:"
# #                       f" curv={float(hc['curvature'].mean()):.3f}"
# #                       f"  spd_var={float(hc['speed_var'].mean()):.3f}"
# #                       f"  dir_chg={float(hc['dir_change'].mean()):.3f}"
# #                       f"{spd_cv_str}")

# #             # XAI-3: Physics score components
# #             pc = xai.get("physics_components", {})
# #             if pc:
# #                 print(f"  [XAI-3] Physics:"
# #                       f" speed={float(pc['speed'].mean()):.3f}"
# #                       f"  smooth={float(pc['smooth'].mean()):.3f}"
# #                       f"  heading={float(pc['heading'].mean()):.3f}")

# #             # XAI-5: Speed comparison (shows calibration effect)
# #             sc = xai.get("speed_comparison", {})
# #             if sc:
# #                 ratio = sc.get("speed_ratio", 1.0)
# #                 flag  = ("⚠ OVER" if ratio > 1.15 else
# #                          "⚠ UNDER" if ratio < 0.85 else "✓ OK")
# #                 n_over  = sc.get("over_predict",  torch.zeros(1)).sum().item()
# #                 n_under = sc.get("under_predict", torch.zeros(1)).sum().item()
# #                 print(f"  [XAI-5] Speed (post-calibration):"
# #                       f" obs={sc['obs_speed_mean']:.1f}km/h"
# #                       f"  pred={sc['pred_speed_mean']:.1f}km/h"
# #                       f"  ratio={ratio:.2f} {flag}"
# #                       f"  (over:{int(n_over)} under:{int(n_under)})")

# #             # XAI-6: Heading deviation — key metric for heading loss efficacy
# #             hd = xai.get("heading_deviation_deg")
# #             if hd is not None and hd.shape[0] >= 1:
# #                 hd_mean = hd.mean(1)
# #                 print(f"  [XAI-6] Heading deviation:"
# #                       f" 12h={hd_mean[0].item():.1f}°"
# #                       f"  24h={hd_mean[min(2,len(hd_mean)-1)].item():.1f}°"
# #                       f"  72h={hd_mean[min(10,len(hd_mean)-1)].item():.1f}°"
# #                       f"  (target: 12h<20°, 72h<100°)")

# #             # XAI-7: ATE/CTE decomposition
# #             ac = xai.get("ate_cte_decomp", {})
# #             if ac:
# #                 print(f"  [XAI-7] Error:"
# #                       f" ATE={ac['ate_abs_mean']:.1f}km"
# #                       f"  CTE={ac['cte_abs_mean']:.1f}km"
# #                       f"  ratio={ac['ate_abs_mean']/(ac['cte_abs_mean']+1e-3):.2f}")

# #             # XAI-8: Per-horizon speed analysis [v2.7 NEW]
# #             sph = xai.get("speed_per_horizon", {})
# #             if sph and "pred_speeds_kmh" in sph and "gt_speeds_kmh" in sph:
# #                 pred_s = sph["pred_speeds_kmh"]
# #                 gt_s   = sph["gt_speeds_kmh"]
# #                 ratio_s = sph.get("speed_ratio_per_step", [])
# #                 # Report at 12h(0), 24h(3), 48h(7), 72h(11)
# #                 horizons = [(0,"12h"),(3,"24h"),(7,"48h"),(11,"72h")]
# #                 parts = []
# #                 for idx, label in horizons:
# #                     if idx < len(pred_s) and idx < len(gt_s):
# #                         r = ratio_s[idx] if idx < len(ratio_s) else pred_s[idx]/max(gt_s[idx],1.)
# #                         flag = "⚠" if r > 1.3 or r < 0.7 else "✓"
# #                         parts.append(f"{label}: pred={pred_s[idx]:.1f} gt={gt_s[idx]:.1f} r={r:.2f}{flag}")
# #                 print(f"  [XAI-8] Speed/horizon: {' | '.join(parts)}")

# #             # XAI-9: Storm category breakdown [v2.7 NEW]
# #             sc9 = xai.get("storm_categories", {})
# #             if sc9:
# #                 print(f"  [XAI-9] Storms:"
# #                       f" slow(<8km/h)={sc9.get('n_slow',0)}"
# #                       f"  medium(8-15)={sc9.get('n_medium',0)}"
# #                       f"  fast(>15km/h)={sc9.get('n_fast',0)}"
# #                       f"  obs_spd_mean={sc9.get('obs_speed_mean',0):.1f}±{sc9.get('obs_speed_std',0):.1f}km/h"
# #                       f"  max={sc9.get('obs_speed_max',0):.1f}km/h")
# #             print(f"  {'─'*60}")
# #             result["xai"] = xai
# #         except Exception as e:
# #             print(f"  XAI error: {e}")
# #             import traceback; traceback.print_exc()

# #     print(f"  {'='*72}\n")
# #     return result


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Checkpoint
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _save(path, epoch, model, opt, sched, best_score,
# #           ema=None, scaler=None, extra=None):
# #     m   = _unwrap(model)
# #     esd = None
# #     if ema is not None:
# #         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# #         except Exception: pass
# #     payload = {
# #         "epoch": epoch, "model": m.state_dict(),
# #         "optimizer": opt.state_dict(), "scheduler": sched.epoch,
# #         "scaler": scaler.state_dict() if scaler is not None else None,
# #         "best_score": best_score, "best_ade": best_score, "ema": esd,
# #     }
# #     if extra: payload.update(extra)
# #     torch.save(payload, path)


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Args
# # # ─────────────────────────────────────────────────────────────────────────────

# # def get_args():
# #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# #     # Data
# #     p.add_argument("--dataset_root",           default="TCND_vn")
# #     p.add_argument("--obs_len",                default=8,      type=int)
# #     p.add_argument("--pred_len",               default=12,     type=int)
# #     p.add_argument("--num_workers",            default=2,      type=int)
# #     p.add_argument("--other_modal",            default="gph")
# #     p.add_argument("--delim",                  default=" ")
# #     p.add_argument("--skip",                   default=1,      type=int)
# #     p.add_argument("--min_ped",                default=1,      type=int)
# #     p.add_argument("--threshold",              default=0.002,  type=float)
# #     # Model
# #     p.add_argument("--d_cond",                 default=256,    type=int)
# #     p.add_argument("--d_model",                default=256,    type=int)
# #     p.add_argument("--nhead",                  default=8,      type=int)
# #     p.add_argument("--num_dec_layers",         default=4,      type=int)
# #     p.add_argument("--dim_ff",                 default=512,    type=int)
# #     p.add_argument("--dropout",                default=0.1,    type=float)
# #     p.add_argument("--unet_in_ch",             default=13,     type=int)
# #     p.add_argument("--sigma_min",              default=0.04,   type=float)
# #     p.add_argument("--sigma_max",              default=0.08,   type=float)
# #     p.add_argument("--lambda_reg",             default=0.2,    type=float)
# #     p.add_argument("--lambda_heading",         default=0.10,   type=float,
# #                    help="[v2.6] multi-step heading, 4 steps, decay=0.5 (was 0.05 v2.5)")
# #     p.add_argument("--lambda_momentum",        default=0.0,    type=float,
# #                    help="[v2.6] DISABLED — hurt test ATE by +7.9km")
# #     p.add_argument("--use_ot",                 default=True,   action="store_true")
# #     p.add_argument("--no_ot",                  dest="use_ot",  action="store_false")
# #     p.add_argument("--ot_epsilon",             default=0.05,   type=float)
# #     p.add_argument("--n_ensemble",             default=20,     type=int)
# #     p.add_argument("--sigma_inference",        default=0.04,   type=float)
# #     p.add_argument("--n_inference_steps",      default=1,      type=int)
# #     # Training
# #     p.add_argument("--num_epochs",             default=150,    type=int)
# #     p.add_argument("--batch_size",             default=64,     type=int)
# #     p.add_argument("--lr",                     default=2e-4,   type=float)
# #     p.add_argument("--lr_min",                 default=1e-6,   type=float)
# #     p.add_argument("--warmup_epochs",          default=5,      type=int)
# #     p.add_argument("--weight_decay",           default=1e-4,   type=float)
# #     p.add_argument("--grad_clip",              default=1.0,    type=float)
# #     p.add_argument("--use_amp",                action="store_true", default=False)
# #     p.add_argument("--use_ema",                default=True,   action="store_true")
# #     p.add_argument("--no_ema",                 dest="use_ema", action="store_false")
# #     # Encoder freeze
# #     p.add_argument("--freeze_encoder_epochs",  default=10,     type=int)
# #     p.add_argument("--encoder_warmup_epochs",  default=5,      type=int)
# #     p.add_argument("--lr_enc_peak",            default=5e-5,   type=float)
# #     # Eval
# #     p.add_argument("--val_freq",               default=5,      type=int)
# #     p.add_argument("--patience",               default=20,     type=int)
# #     p.add_argument("--min_ep",                 default=20,     type=int)
# #     p.add_argument("--hard_val_threshold",     default=0.35,   type=float)
# #     p.add_argument("--hard_val_freq",          default=10,     type=int)
# #     # SWA
# #     p.add_argument("--swa_lr",                 default=2e-6,   type=float)
# #     p.add_argument("--swa_window",             default=3,      type=int)
# #     p.add_argument("--swa_threshold",          default=1.5,    type=float)
# #     p.add_argument("--swa_min_ep",             default=50,     type=int)
# #     # Test
# #     p.add_argument("--tta_test",               default=True,   action="store_true")
# #     p.add_argument("--n_tta",                  default=5,      type=int)
# #     p.add_argument("--multiscale_test",        default=True,   action="store_true")
# #     # IO
# #     p.add_argument("--output_dir",             default="runs/fm_v26")
# #     p.add_argument("--gpu_num",                default="0")
# #     p.add_argument("--resume",                 default=None)
# #     p.add_argument("--test_at_end",            action="store_true", default=True)
# #     p.add_argument("--no_test",                dest="test_at_end", action="store_false")
# #     return p.parse_args()


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Main
# # # ─────────────────────────────────────────────────────────────────────────────

# # def main(args):
# #     if torch.cuda.is_available():
# #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     os.makedirs(args.output_dir, exist_ok=True)
# #     best_ckpt      = os.path.join(args.output_dir, "best_model.pth")
# #     hard_best_ckpt = os.path.join(args.output_dir, "hard_best_model.pth")
# #     swa_ckpt       = os.path.join(args.output_dir, "swa_model.pth")
# #     last_ckpt      = os.path.join(args.output_dir, "last_model.pth")

# #     print("=" * 72)
# #     print("  TC-FlowMatching v2.6")
# #     print(f"  [REMOVE]  L_momentum (was hurting test ATE by +7.9km)")
# #     print(f"  [UPGRADE] L_heading multi-step 4 steps decay=0.5 weight={args.lambda_heading}")
# #     print(f"  [NEW-INF] Speed calibration at inference (clip 0.85–1.15)")
# #     print(f"  [NEW-AUG] Obs-speed aug ×0.7–1.4 (15%)")
# #     print(f"  [UPGRADE] Physics score + displacement_score")
# #     print(f"  KEEP:     sigma_inf={args.sigma_inference} FIXED, L_reg linear weights")
# #     print("=" * 72)

# #     # ── Data ──────────────────────────────────────────────────────────────
# #     print("\n  Loading data...")
# #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
# #     vd, val_loader = data_loader(args, {"root": args.dataset_root, "type": "val"}, test=True)
# #     print(f"  train: {len(trd)} ({len(trl)} batches/ep)")
# #     print(f"  val:   {len(vd)} ({len(val_loader)} batches)")

# #     # Fixed XAI batch (same every epoch for comparability)
# #     try:
# #         xai_batch = move(list(next(iter(val_loader))), device)
# #     except Exception:
# #         xai_batch = None

# #     # ── Model ─────────────────────────────────────────────────────────────
# #     model = TCFlowMatching(
# #         pred_len=args.pred_len,        obs_len=args.obs_len,
# #         unet_in_ch=args.unet_in_ch,   d_cond=args.d_cond,
# #         d_model=args.d_model,          nhead=args.nhead,
# #         num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
# #         dropout=args.dropout,
# #         sigma_min=args.sigma_min,      sigma_max=args.sigma_max,
# #         lambda_reg=args.lambda_reg,    lambda_heading=args.lambda_heading,
# #         lambda_momentum=0.0,
# #         use_ot=args.use_ot,            ot_epsilon=args.ot_epsilon,
# #         use_ema=args.use_ema,          n_ensemble=args.n_ensemble,
# #         n_inference_steps=args.n_inference_steps,
# #         sigma_inference=args.sigma_inference,
# #     ).to(device)

# #     model.init_ema()
# #     ema = getattr(_unwrap(model), "_ema", None)
# #     raw = _unwrap(model)
# #     n_enc = sum(p.numel() for p in raw.encoder.parameters())
# #     n_vel = sum(p.numel() for p in raw.velocity.parameters())
# #     print(f"\n  Encoder: {n_enc:,}  VelocityTrans: {n_vel:,}  Total: {n_enc+n_vel:,}")

# #     # ── Optimizer + Scheduler ─────────────────────────────────────────────
# #     opt    = build_optimizer(model, lr_velocity=args.lr, lr_encoder=0.0,
# #                              weight_decay=args.weight_decay)
# #     scaler = GradScaler("cuda", enabled=args.use_amp)
# #     sched  = TwoGroupScheduler(
# #         opt=opt, warmup_epochs=args.warmup_epochs, total_epochs=args.num_epochs,
# #         lr_vel=args.lr, lr_vel_min=args.lr_min,
# #         freeze_end_ep=args.freeze_encoder_epochs, lr_enc_peak=args.lr_enc_peak,
# #         encoder_warmup_epochs=args.encoder_warmup_epochs)
# #     print(f"\n  LR vel: {args.lr:.0e} → {args.lr_min:.0e}  "
# #           f"LR enc: 0 ({args.freeze_encoder_epochs}ep) → {args.lr_enc_peak:.0e}")

# #     swa = SWAHandler(swa_lr=args.swa_lr)

# #     # ── Resume ────────────────────────────────────────────────────────────
# #     start_ep = 0; best_score = float("inf"); best_hard = float("inf")
# #     patience_cnt = 0; val_ade_history = []

# #     if args.resume and os.path.exists(args.resume):
# #         ck = torch.load(args.resume, map_location=device)
# #         _unwrap(model).load_state_dict(ck["model"], strict=False)
# #         try: opt.load_state_dict(ck["optimizer"])
# #         except Exception as e: print(f"  ⚠ Opt: {e}")
# #         sched.epoch  = ck.get("scheduler", 0)
# #         start_ep     = ck.get("epoch", 0) + 1
# #         best_score   = ck.get("best_score", ck.get("best_ade", float("inf")))
# #         patience_cnt = ck.get("patience_cnt", 0)
# #         if scaler and ck.get("scaler"):
# #             try: scaler.load_state_dict(ck["scaler"])
# #             except Exception: pass
# #         if ema and ck.get("ema"):
# #             for k, v in ck["ema"].items():
# #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
# #         print(f"  ↩ Resume ep{start_ep}  best={best_score:.1f}  patience={patience_cnt}")

# #     try:
# #         model = torch.compile(model, mode="reduce-overhead")
# #         print("  torch.compile: ok")
# #     except Exception: pass

# #     # ── Training loop ─────────────────────────────────────────────────────
# #     nstep = len(trl)
# #     print(f"\n  TRAINING ({nstep} steps/ep × {args.num_epochs} ep)")
# #     print(f"  Aug: shift±5km + speed×[0.85,1.15] + recurv±20° + obs-speed×[0.7,1.4] + noise")
# #     print(f"  Loss: L_CFM + L_reg(linear) + L_heading_ms(4steps,decay=0.5)")
# #     print(f"  Inf:  1-shot sigma=0.04 + speed_calibrate(±15%) + top3 physics")
# #     print()

# #     for ep in range(start_ep, start_ep + args.num_epochs):
# #         rel_ep = ep - start_ep
# #         freeze = rel_ep < args.freeze_encoder_epochs
# #         for p in _unwrap(model).encoder.parameters():
# #             p.requires_grad_(not freeze)

# #         if rel_ep == 0 and freeze:
# #             print(f"  *** Ep{ep}: encoder frozen ***")
# #         if rel_ep == args.freeze_encoder_epochs:
# #             print(f"\n  *** Ep{ep}: encoder unfrozen ***")

# #         model.train()
# #         sum_loss = sum_cfm = sum_reg = sum_head = sum_ade1 = 0.0
# #         t0_ep = time.perf_counter()

# #         for i, batch in enumerate(trl):
# #             bl = move(list(batch), device)
# #             bl_aug = augment_batch(bl)   # training augmentation

# #             opt.zero_grad()
# #             with autocast(device_type="cuda", enabled=args.use_amp):
# #                 bd = model.get_loss_breakdown(bl_aug, epoch=ep)

# #             scaler.scale(bd["total"]).backward()
# #             scaler.unscale_(opt)
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

# #             if freeze:
# #                 for p in _unwrap(model).encoder.parameters():
# #                     if p.grad is not None: p.grad.zero_()

# #             scaler.step(opt); scaler.update()
# #             model.ema_update()
# #             swa.update(model)

# #             sum_loss += bd["total"].item(); sum_cfm  += bd["l_cfm"]
# #             sum_reg  += bd["l_reg"];        sum_head += bd["l_heading"]
# #             sum_ade1 += bd["ade_1step"]

# #             if i % 30 == 0:
# #                 _, lr_vel = get_lrs(opt)
# #                 enc_s = "frozen" if freeze else "active"
# #                 swa_s = " [SWA]" if swa.active else ""
# #                 print(f"  [{ep:>3}][{i:>3}/{nstep}]"
# #                       f"  tot={bd['total'].item():.4f}"
# #                       f"  cfm={bd['l_cfm']:.4f}"
# #                       f"  reg={bd['l_reg']:.4f}"
# #                       f"  h4s={bd['l_heading']:.4f}"
# #                       f"  lam_d={bd['lam_dir']:.2f}"
# #                       f"  ade1={bd['ade_1step']:.0f}km"
# #                       f"  enc={enc_s}{swa_s}"
# #                       f"  lr={lr_vel:.2e}")

# #         train_loss = sum_loss / nstep
# #         _, lr_vel_used = get_lrs(opt)
# #         sched.step()

# #         print(f"\n  ── Ep{ep:>3}"
# #               f"  train={train_loss:.6f}"
# #               f"  cfm={sum_cfm/nstep:.4f}"
# #               f"  reg={sum_reg/nstep:.4f}"
# #               f"  h4s={sum_head/nstep:.4f}"
# #               f"  ade1={sum_ade1/nstep:.0f}km"
# #               f"  lr={lr_vel_used:.2e}"
# #               f"  t={time.perf_counter()-t0_ep:.0f}s")

# #         _save(last_ckpt, ep, model, opt, sched, best_score, ema, scaler)
# #         if ep % 5 == 0:
# #             ep_ckpt = os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth")
# #             _save(ep_ckpt, ep, model, opt, sched, best_score, ema, scaler)
# #             print(f"  💾 {ep_ckpt}")

# #         # ── Val evaluation ───────────────────────────────────────────────
# #         if rel_ep % args.val_freq == 0:
# #             run_xai_this = (rel_ep % 10 == 0)
# #             r = evaluate(model, val_loader, device, tag=f"VAL ep{ep}",
# #                          n_ensemble=args.n_ensemble, ema=ema,
# #                          ref_targets=ST_TRANS_VAL, epoch_for_loss=ep,
# #                          run_xai=run_xai_this, xai_batch=xai_batch)

# #             val_ade = r["ADE"]; score = r["combined_score"]
# #             val_ade_history.append(val_ade)

# #             # Trend
# #             if len(val_ade_history) >= 4:
# #                 trend = float(np.mean(val_ade_history[-2:])) - float(np.mean(val_ade_history[-4:-2]))
# #                 trend_s = f"↑{trend:+.1f}km⚠" if trend > 5 else f"↓{trend:+.1f}km✓" if trend < -5 else f"→{trend:+.1f}flat"
# #             else:
# #                 trend_s = "—"
# #             print(f"  train={train_loss:.6f}  val_ADE={val_ade:.1f}  combined={score:.1f}  trend={trend_s}")

# #             # SWA activation check
# #             if (not swa.active and ep >= args.swa_min_ep
# #                     and swa.should_activate(val_ade_history, args.swa_window, args.swa_threshold)):
# #                 swa.activate(model, opt, ep)

# #             # Best checkpoint
# #             if score < best_score:
# #                 best_score = score; patience_cnt = 0
# #                 _save(best_ckpt, ep, model, opt, sched, best_score, ema, scaler,
# #                       extra={"val_ade": r["ADE"], "val_ate": r["ATE"],
# #                              "val_cte": r["CTE"], "patience_cnt": 0})
# #                 print(f"  ✅ Best! score={best_score:.2f}"
# #                       f"  ADE={r['ADE']:.1f} ATE={r['ATE']:.1f} CTE={r['CTE']:.1f}")
# #             else:
# #                 if rel_ep >= args.min_ep: patience_cnt += args.val_freq
# #                 print(f"  No improve {patience_cnt}/{args.patience} (best={best_score:.1f})")
# #                 if rel_ep >= args.min_ep and patience_cnt >= args.patience:
# #                     print(f"  ⛔ Early stop @ ep{ep}")
# #                     break

# #         # ── Hard-val evaluation ──────────────────────────────────────────
# #         if rel_ep % args.hard_val_freq == 0 and rel_ep >= args.min_ep:
# #             r_h = evaluate_hard_val(model, val_loader, device,
# #                                      hard_threshold=args.hard_val_threshold,
# #                                      n_ensemble=args.n_ensemble, ema=ema,
# #                                      epoch_for_loss=ep)
# #             print(f"  [HVAL] n={r_h['n_hard']}"
# #                   f"  ADE={r_h['ADE']:.1f} ATE={r_h['ATE']:.1f} CTE={r_h['CTE']:.1f}"
# #                   f"  combined={r_h['combined_score']:.1f}")
# #             if r_h["combined_score"] < best_hard and r_h["n_hard"] >= 10:
# #                 best_hard = r_h["combined_score"]
# #                 _save(hard_best_ckpt, ep, model, opt, sched, best_hard, ema, scaler,
# #                       extra={"hard_val_ade": r_h["ADE"], "selection_criterion": "hard_val"})
# #                 print(f"  💎 Hard-best! score={best_hard:.2f} ADE={r_h['ADE']:.1f}")

# #         # ── SWA eval ─────────────────────────────────────────────────────
# #         if swa.active and rel_ep % args.val_freq == 0 and swa.n_updates >= 10:
# #             backup = {k: v.detach().clone()
# #                       for k, v in _unwrap(model).state_dict().items()
# #                       if v.dtype.is_floating_point}
# #             swa.apply_to_model(model)
# #             r_swa = evaluate(model, val_loader, device, tag=f"SWA ep{ep}",
# #                               n_ensemble=args.n_ensemble, ema=None,
# #                               ref_targets=ST_TRANS_VAL, epoch_for_loss=ep)
# #             swa.restore_from_backup(model, backup)

# #             swa_score = r_swa["combined_score"]
# #             print(f"  [SWA] score={swa_score:.2f} ({swa.n_updates} updates) vs best={best_score:.2f}")
# #             if swa_score < best_score:
# #                 best_score = swa_score; patience_cnt = 0
# #                 swa.save_avg_state(swa_ckpt, ep, best_score,
# #                                    extra={"val_ade": r_swa["ADE"],
# #                                           "val_ate": r_swa["ATE"],
# #                                           "val_cte": r_swa["CTE"]})
# #                 import shutil; shutil.copy(swa_ckpt, best_ckpt)
# #                 print(f"  ✅ SWA best! score={best_score:.2f} ADE={r_swa['ADE']:.1f}")

# #     # ── Test evaluation ───────────────────────────────────────────────────
# #     print(f"\n  Done! best_score={best_score:.2f}")
# #     if not args.test_at_end: return

# #     print("\n  Loading best checkpoint for TEST...")
# #     if not os.path.exists(best_ckpt):
# #         print("  No checkpoint found."); return

# #     ck = torch.load(best_ckpt, map_location=device)
# #     is_swa = ck.get("is_swa", False)
# #     _unwrap(model).load_state_dict(ck["model"], strict=False)
# #     if not is_swa and ema and ck.get("ema"):
# #         for k, v in ck["ema"].items():
# #             if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
# #     print(f"  Loaded ep{ck.get('epoch','?')} (is_swa={is_swa})")

# #     try:
# #         _, test_loader = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
# #         print(f"  Test: {len(test_loader)} batches")
# #     except Exception:
# #         print("  No test set → using val"); test_loader = val_loader

# #     # Standard test
# #     r_test = evaluate(model, test_loader, device, tag="TEST (best ckpt)",
# #                       n_ensemble=args.n_ensemble, ema=None if is_swa else ema,
# #                       ref_targets=ST_TRANS_TEST, run_xai=True, xai_batch=xai_batch)

# #     # TTA test
# #     if args.tta_test:
# #         r_tta = evaluate(model, test_loader, device, tag="TEST+TTA",
# #                          n_ensemble=args.n_ensemble, ema=None if is_swa else ema,
# #                          ref_targets=ST_TRANS_TEST, use_tta=True, n_tta=args.n_tta)
# #         print(f"\n  TTA: ADE {r_test['ADE']:.1f}→{r_tta['ADE']:.1f}  "
# #               f"ATE {r_test['ATE']:.1f}→{r_tta['ATE']:.1f}  "
# #               f"CTE {r_test['CTE']:.1f}→{r_tta['CTE']:.1f}")

# #     # Multi-scale test
# #     if args.multiscale_test:
# #         raw_m = _unwrap(model)
# #         if hasattr(raw_m, "sample_multiscale"):
# #             print("\n  Multi-scale sigma test...")
# #             ms_ades, ms_ates, ms_ctes = [], [], []
# #             ms_steps = defaultdict(list)
# #             raw_m.eval()
# #             bk_ms = (ema.apply_to(model) if (ema and not is_swa) else None)
# #             with torch.no_grad():
# #                 for batch in test_loader:
# #                     bl = move(list(batch), device)
# #                     try:
# #                         pred, _, _ = raw_m.sample_multiscale(bl)
# #                     except Exception: continue
# #                     gt = bl[1]; T = min(pred.shape[0], gt.shape[0])
# #                     pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
# #                     dist = _haversine_deg(pd, gd)
# #                     ate, cte = _ate_cte(pd, gd)
# #                     ms_ades.extend(dist.mean(0).tolist())
# #                     if ate.shape[0] > 0:
# #                         ms_ates.extend(ate.abs().mean(0).tolist())
# #                         ms_ctes.extend(cte.abs().mean(0).tolist())
# #                     for h, s in HORIZON_STEPS.items():
# #                         if s < T: ms_steps[h].extend(dist[s].tolist())
# #             if bk_ms: ema.restore(model, bk_ms)
# #             def _mm(lst): return float(np.mean(lst)) if lst else float("nan")
# #             print(f"  Multi-scale: ADE={_mm(ms_ades):.1f} ATE={_mm(ms_ates):.1f} CTE={_mm(ms_ctes):.1f}")

# #     # Final comparison
# #     val_ade_b = ck.get("val_ade", float("nan"))
# #     val_ate_b = ck.get("val_ate", float("nan"))
# #     val_cte_b = ck.get("val_cte", float("nan"))
# #     v21 = {"ADE": 229.8, "ATE": 214.4, "CTE": 71.6}
# #     print("\n" + "=" * 72)
# #     print("  v2.6 FINAL RESULTS vs v2.1 baseline")
# #     print("=" * 72)
# #     print(f"  {'Metric':<10} {'Val':>10} {'Test':>10} {'Gap':>10} {'v2.1 Test':>12} Status")
# #     print("  " + "─"*60)
# #     for m_n, val_v, test_v, v21_v in [
# #         ("ADE", val_ade_b, r_test["ADE"], v21["ADE"]),
# #         ("ATE", val_ate_b, r_test["ATE"], v21["ATE"]),
# #         ("CTE", val_cte_b, r_test["CTE"], v21["CTE"]),
# #     ]:
# #         gap  = test_v - val_v if (np.isfinite(test_v) and np.isfinite(val_v)) else float("nan")
# #         impr = v21_v - test_v
# #         flag = (f"↓{impr:+.1f}" if np.isfinite(impr) and impr > 0 else
# #                 f"↑{-impr:+.1f}" if np.isfinite(impr) else "?")
# #         print(f"  {m_n:<10} {val_v:>10.2f} {test_v:>10.2f} {gap:>+10.2f} {v21_v:>12.1f}  {flag}")
# #     print("=" * 72)


# # if __name__ == "__main__":
# #     args = get_args()
# #     np.random.seed(42); torch.manual_seed(42)
# #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# #     if args.dataset_root == "TCND_vn":
# #         _auto = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
# #         if os.path.isdir(_auto):
# #             args.dataset_root = _auto
# #     main(args)

# """
# scripts/train_fm.py  ──  TC-FlowMatching v2.6 Training
# ═══════════════════════════════════════════════════════════════════════════════

# Thay đổi từ v2.5 (dựa trên phân tích 140 epoch log đầy đủ):

#   [REMOVE]  L_momentum — làm ATE test tệ hơn +7.9km (anchor to val speed)
#   [UPGRADE] L_heading multi-step (4 steps, decay=0.5), weight 0.05→0.10
#   [NEW-INF] Speed calibration tại inference (per-storm, clip ±15%)
#   [UPGRADE] Physics score + displacement_score
#   [AUG-D]   Obs-speed scaling ×0.7–1.4 (15% prob)

# Giữ nguyên (proven):
#   ✅ sigma_inference=0.04 FIXED
#   ✅ L_reg softmax-linspace weights
#   ✅ 2-group optimizer, encoder freeze 10ep
#   ✅ val loop NO augmentation
#   ✅ 1-shot inference
#   ✅ SWA sau plateau
#   ✅ Hard-val checkpoint (HVAL)
#   ✅ XAI logging mỗi 10 epoch
# """
# from __future__ import annotations
# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse, math, time
# from collections import defaultdict
# from typing import Dict, List, Optional

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.amp import autocast, GradScaler

# from Model.data.loader_training import data_loader
# from Model.flow_matching_model import (
#     TCFlowMatching, _norm_to_deg, _haversine_deg,
#     EMAModel, augment_batch, hard_score_from_obs,
#     compute_ensemble_uncertainty,
#     compute_heading_deviation, compute_cte_contribution,
#     compute_obs_attribution,
# )

# HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}
# ST_TRANS_VAL  = {"ADE": 172.68, "ATE": 142.21, "CTE": 42.04,
#                  "12h": 65.42, "24h": 104.67, "48h": 205.10, "72h": 321.39}
# ST_TRANS_TEST = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
#                  "12h": 77.5, "24h": 130.5, "48h": 269.9, "72h": 423.3}

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
#     """Decompose error into along-track and cross-track components."""
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         z = pred_deg.new_zeros(1, pred_deg.shape[1])
#         return z, z
#     lo1 = torch.deg2rad(gt_deg[:T-1,:,0]); la1 = torch.deg2rad(gt_deg[:T-1,:,1])
#     lo2 = torch.deg2rad(gt_deg[1:T, :,0]); la2 = torch.deg2rad(gt_deg[1:T, :,1])
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


# # ─────────────────────────────────────────────────────────────────────────────
# #  2-group optimizer (encoder freeze pattern from v2.1)
# # ─────────────────────────────────────────────────────────────────────────────

# def build_optimizer(model, lr_velocity, lr_encoder, weight_decay):
#     raw = _unwrap(model)
#     return optim.AdamW([
#         {"params": list(raw.encoder.parameters()),  "lr": lr_encoder,  "name": "encoder"},
#         {"params": list(raw.velocity.parameters()), "lr": lr_velocity, "name": "velocity"},
#     ], weight_decay=weight_decay)

# def get_lrs(opt):
#     lr_enc = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "encoder")
#     lr_vel = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "velocity")
#     return lr_enc, lr_vel

# class TwoGroupScheduler:
#     def __init__(self, opt, warmup_epochs, total_epochs,
#                  lr_vel, lr_vel_min, freeze_end_ep,
#                  lr_enc_peak, encoder_warmup_epochs=5):
#         self.opt         = opt
#         self.warmup      = warmup_epochs
#         self.total       = total_epochs
#         self.lr_vel      = lr_vel
#         self.lr_vel_min  = lr_vel_min
#         self.freeze_end  = freeze_end_ep
#         self.lr_enc_peak = lr_enc_peak
#         self.enc_warmup  = encoder_warmup_epochs
#         self.epoch       = 0

#     def _cosine(self, ep_from, ep_to, lr_s, lr_e, ep):
#         t = max(0., min(1., (ep - ep_from) / max(ep_to - ep_from, 1)))
#         return lr_e + 0.5 * (lr_s - lr_e) * (1 + math.cos(math.pi * t))

#     def step(self):
#         ep = self.epoch
#         if ep < self.warmup:
#             lr_vel = self.lr_vel * (0.1 + 0.9 * ep / max(self.warmup - 1, 1))
#         else:
#             lr_vel = self._cosine(self.warmup, self.total, self.lr_vel, self.lr_vel_min, ep)
#         if ep < self.freeze_end:
#             lr_enc = 0.0
#         elif ep < self.freeze_end + self.enc_warmup:
#             lr_enc = self.lr_enc_peak * (ep - self.freeze_end) / self.enc_warmup
#         else:
#             lr_enc = self._cosine(self.freeze_end + self.enc_warmup, self.total,
#                                   self.lr_enc_peak, self.lr_vel_min, ep)
#         for pg in self.opt.param_groups:
#             pg["lr"] = lr_enc if pg.get("name") == "encoder" else lr_vel
#         self.epoch += 1
#         return lr_vel, lr_enc


# # ─────────────────────────────────────────────────────────────────────────────
# #  SWA handler (same as v2.5 — proven)
# # ─────────────────────────────────────────────────────────────────────────────

# class SWAHandler:
#     """
#     Stochastic Weight Averaging after val ADE plateau.
#     Activation: val ADE improvement < threshold km in last `window` checks.
#     After activation: average model weights each epoch → flatter loss landscape
#     → better generalization.
#     """
#     def __init__(self, swa_lr: float = 2e-6):
#         self.swa_lr    = swa_lr
#         self.active    = False
#         self.start_ep  = None
#         self.n_updates = 0
#         self.avg_state = {}

#     def should_activate(self, ade_history: List[float],
#                          window: int = 3, threshold: float = 1.5) -> bool:
#         if len(ade_history) < window: return False
#         return (ade_history[-window] - ade_history[-1]) < threshold

#     def activate(self, model, opt, ep: int):
#         self.active = True; self.start_ep = ep
#         for pg in opt.param_groups: pg["lr"] = self.swa_lr
#         m = _unwrap(model)
#         self.avg_state = {k: v.detach().clone().float()
#                           for k, v in m.state_dict().items()
#                           if v.dtype.is_floating_point}
#         self.n_updates = 1
#         print(f"  *** SWA ACTIVATED @ ep{ep} (lr → {self.swa_lr:.1e}) ***")

#     def update(self, model):
#         if not self.active: return
#         m = _unwrap(model); sd = m.state_dict(); n = self.n_updates
#         for k in self.avg_state:
#             if k in sd:
#                 self.avg_state[k] = (n * self.avg_state[k] + sd[k].detach().float()) / (n + 1)
#         self.n_updates += 1

#     def apply_to_model(self, model):
#         if not self.active or not self.avg_state: return
#         m = _unwrap(model); sd = m.state_dict()
#         for k in self.avg_state:
#             if k in sd: sd[k].copy_(self.avg_state[k].to(sd[k].device))

#     def restore_from_backup(self, model, backup):
#         m = _unwrap(model); sd = m.state_dict()
#         for k, v in backup.items():
#             if k in sd: sd[k].copy_(v)

#     def save_avg_state(self, path: str, epoch: int, best_score: float,
#                        extra: Optional[dict] = None):
#         payload = {"epoch": epoch, "model": self.avg_state,
#                    "best_score": best_score, "is_swa": True,
#                    "swa_updates": self.n_updates}
#         if extra: payload.update(extra)
#         torch.save(payload, path)


# # ─────────────────────────────────────────────────────────────────────────────
# #  Hard-val evaluator (HVAL)
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def evaluate_hard_val(model, val_loader, device, hard_threshold: float = 0.35,
#                        n_ensemble: int = 20, ema=None, epoch_for_loss: int = 9999):
#     """
#     Evaluate on hard storms only (hard_score > threshold).
#     Hard storms are closer to test distribution (more recurvature).
#     Best hard-val checkpoint often generalizes better than overall best.
#     """
#     bk = None
#     if ema is not None:
#         try: bk = ema.apply_to(model)
#         except Exception: pass

#     model.eval()
#     all_ade, all_ate, all_cte = [], [], []
#     n_hard = 0

#     for batch in val_loader:
#         bl = move(list(batch), device)
#         B  = bl[0].shape[1]
#         h_score   = hard_score_from_obs(bl[0][:, :, :2])
#         hard_mask = h_score > hard_threshold
#         if hard_mask.sum() == 0: continue
#         hard_idx  = hard_mask.nonzero(as_tuple=True)[0]

#         # Build sub-batch for hard storms
#         bl_h = list(bl)
#         for i, item in enumerate(bl_h):
#             if torch.is_tensor(item):
#                 if item.dim() >= 2 and item.shape[1] == B:
#                     bl_h[i] = item[:, hard_idx, ...]
#                 elif item.dim() >= 1 and item.shape[0] == B:
#                     bl_h[i] = item[hard_idx, ...]

#         try:
#             pred, _, _ = model.sample(bl_h, num_ensemble=n_ensemble)
#         except Exception as e:
#             print(f"  hard val error: {e}"); continue

#         gt = bl_h[1]
#         T  = min(pred.shape[0], gt.shape[0])
#         pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
#         dist = _haversine_deg(pd, gd)
#         ate, cte = _ate_cte(pd, gd)
#         all_ade.extend(dist.mean(0).tolist())
#         if ate.shape[0] > 0:
#             all_ate.extend(ate.abs().mean(0).tolist())
#             all_cte.extend(cte.abs().mean(0).tolist())
#         n_hard += len(hard_idx)

#     if bk is not None:
#         try: ema.restore(model, bk)
#         except Exception: pass

#     def _m(lst): return float(np.mean(lst)) if lst else float("nan")
#     return {
#         "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
#         "n_hard": n_hard,
#         "combined_score": 0.6*_m(all_ade) + 0.2*_m(all_ate) + 0.2*_m(all_cte),
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# #  Main evaluation
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def evaluate(model, loader, device, tag: str = "",
#              n_ensemble: int = 20, ema=None,
#              ref_targets=None, use_tta: bool = False,
#              n_tta: int = 5, epoch_for_loss: int = 9999,
#              run_xai: bool = False, xai_batch=None) -> Dict:
#     """
#     Full evaluation. NO augmentation (val_loss is reliable generalization signal).
#     XAI logging every 10 epochs shows training dynamics.
#     """
#     bk = None
#     if ema is not None:
#         try: bk = ema.apply_to(model)
#         except Exception as e: print(f"  ⚠ EMA: {e}")

#     model.eval()
#     all_ade, all_ate, all_cte = [], [], []
#     step_dist = defaultdict(list)
#     sum_loss = sum_cfm = sum_head = 0.0
#     sum_n = 0

#     for batch in loader:
#         bl = move(list(batch), device)
#         gt = bl[1]; B = bl[0].shape[1]

#         # Val loss (no augmentation)
#         try:
#             bd = model.get_loss_breakdown(bl, epoch=epoch_for_loss)
#             if torch.isfinite(bd["total"]):
#                 sum_loss += bd["total"].item() * B
#                 sum_cfm  += bd["l_cfm"] * B
#                 sum_head += bd["l_heading"]
#                 sum_jerk  += bd.get("l_jerk", 0.0) * B
#                 sum_n    += B
#         except Exception: pass

#         # Inference (standard or speed-scale TTA)
#         if use_tta:
#             obs = bl[0]; anchor = obs[-1:, :, :2].detach()
#             scales = [0.875, 0.9375, 1.0, 1.0625, 1.125][:n_tta]
#             preds_t, weights_t = [], []
#             for sc in scales:
#                 obs_s = obs.clone(); obs_s[..., :2] = anchor + (obs[..., :2] - anchor) * sc
#                 bl_s = list(bl); bl_s[0] = obs_s
#                 try:
#                     p, _, _ = model.sample(bl_s, num_ensemble=n_ensemble)
#                     preds_t.append(p)
#                     weights_t.append(2.0 if abs(sc - 1.0) < 1e-6 else 1.0)
#                 except Exception: continue
#             if not preds_t: continue
#             tw   = sum(weights_t)
#             pred = sum(w / tw * p for w, p in zip(weights_t, preds_t))
#         else:
#             try:
#                 pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
#             except Exception as e:
#                 print(f"  sample error: {e}"); continue

#         T = min(pred.shape[0], gt.shape[0])
#         pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
#         dist = _haversine_deg(pd, gd)
#         ate, cte = _ate_cte(pd, gd)
#         all_ade.extend(dist.mean(0).tolist())
#         if ate.shape[0] > 0:
#             all_ate.extend(ate.abs().mean(0).tolist())
#             all_cte.extend(cte.abs().mean(0).tolist())
#         for h, s in HORIZON_STEPS.items():
#             if s < T: step_dist[h].extend(dist[s].tolist())

#     if bk is not None:
#         try: ema.restore(model, bk)
#         except Exception: pass

#     def _m(lst): return float(np.mean(lst)) if lst else float("nan")
#     val_loss = sum_loss / max(sum_n, 1)

#     result = {
#         "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
#         "n": len(all_ade),
#         "val_loss": val_loss,
#         "val_cfm_loss":  sum_cfm / max(sum_n, 1),
#         "val_head_loss": sum_head / max(sum_n, 1),
#         "val_mom_loss":  0.0,   # always 0
#     }
#     for h in HORIZON_STEPS: result[f"{h}h"] = _m(step_dist[h])
#     ade, ate_, cte_ = result["ADE"], result["ATE"], result["CTE"]
#     result["combined_score"] = (
#         0.6 * ade + 0.2 * ate_ + 0.2 * cte_
#         if all(np.isfinite(x) for x in [ade, ate_, cte_]) else ade)

#     ref = ref_targets or ST_TRANS_VAL
#     def _v(k): return result.get(k, float("nan"))
#     def _ok(k): return "✓" if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9) else "✗"

#     tta_str = " [TTA]" if use_tta else ""
#     print(f"\n  {'='*72}")
#     print(f"  [{tag}]{tta_str}  n={result['n']}")
#     print(f"  Val Loss : {val_loss:.6f}  cfm={result['val_cfm_loss']:.6f}  "
#           f"head4s={result['val_head_loss']:.6f}  [mom=DISABLED]")
#     print(f"  ADE={_v('ADE'):7.1f}km {_ok('ADE')}  "
#           f"ATE={_v('ATE'):7.1f}km {_ok('ATE')}  "
#           f"CTE={_v('CTE'):7.1f}km {_ok('CTE')}")
#     print(f"  Combined = {_v('combined_score'):.1f}")
#     print(f"  12h={_v('12h'):6.1f}  24h={_v('24h'):6.1f}  "
#           f"48h={_v('48h'):6.1f}  72h={_v('72h'):6.1f} km")
#     beat = [f"{k}={_v(k):.0f}<{ref.get(k,999):.0f}"
#             for k in ["ADE","ATE","CTE","12h","24h","48h","72h"]
#             if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9)]
#     print(f"  BEAT: {' | '.join(beat) if beat else 'none yet'}")

#     # XAI logging
#     if run_xai and xai_batch is not None:
#         try:
#             _, _, _, xai = _unwrap(model).sample(xai_batch, return_xai=True)

#             print(f"  {'─'*60}")
#             print(f"  XAI Summary (fixed val batch)")
#             print(f"  {'─'*60}")

#             # XAI-4: Uncertainty
#             print(f"  [XAI-4] Uncertainty:"
#                   f" 12h={xai['mean_12h_std']:.1f}km"
#                   f"  72h={xai['mean_72h_std']:.1f}km"
#                   f"  ratio={float(xai['uncertainty_ratio'].mean()):.2f}×"
#                   f"  high_uncert={xai['high_uncertainty'].sum().item()}")

#             # XAI-2: Hard score components (v2.7: thêm speed_cv)
#             hc = xai.get("hard_components", {})
#             if hc:
#                 spd_cv_str = f"  spd_cv={float(hc['speed_cv'].mean()):.3f}" if 'speed_cv' in hc else ""
#                 print(f"  [XAI-2] HardScore:"
#                       f" curv={float(hc['curvature'].mean()):.3f}"
#                       f"  spd_var={float(hc['speed_var'].mean()):.3f}"
#                       f"  dir_chg={float(hc['dir_change'].mean()):.3f}"
#                       f"{spd_cv_str}")

#             # XAI-3: Physics score components
#             pc = xai.get("physics_components", {})
#             if pc:
#                 print(f"  [XAI-3] Physics:"
#                       f" speed={float(pc['speed'].mean()):.3f}"
#                       f"  smooth={float(pc['smooth'].mean()):.3f}"
#                       f"  heading={float(pc['heading'].mean()):.3f}")

#             # XAI-5: Speed comparison (shows calibration effect)
#             sc = xai.get("speed_comparison", {})
#             if sc:
#                 ratio = sc.get("speed_ratio", 1.0)
#                 flag  = ("⚠ OVER" if ratio > 1.15 else
#                          "⚠ UNDER" if ratio < 0.85 else "✓ OK")
#                 n_over  = sc.get("over_predict",  torch.zeros(1)).sum().item()
#                 n_under = sc.get("under_predict", torch.zeros(1)).sum().item()
#                 print(f"  [XAI-5] Speed (post-calibration):"
#                       f" obs={sc['obs_speed_mean']:.1f}km/h"
#                       f"  pred={sc['pred_speed_mean']:.1f}km/h"
#                       f"  ratio={ratio:.2f} {flag}"
#                       f"  (over:{int(n_over)} under:{int(n_under)})")

#             # Learned step weights (mỗi 10 epoch → xem model học gì)
#             sw_learned = xai.get("learned_step_weights", [])
#             if sw_learned:
#                 sw_t = torch.tensor(sw_learned)
#                 early = float(sw_t[:4].sum())
#                 mid   = float(sw_t[4:8].sum())
#                 late  = float(sw_t[8:].sum())
#                 print(f"  [XAI-W] Step weights:"
#                       f" early(6-24h)={early:.1%}"
#                       f"  mid(30-48h)={mid:.1%}"
#                       f"  late(54-72h)={late:.1%}"
#                       f"  (want: late > 40%)")
#                 ratio = sw_t.max().item() / sw_t.min().item()
#                 print(f"           max/min={ratio:.2f}×  "
#                       f"w[0]={sw_t[0]:.3f} w[5]={sw_t[5]:.3f} w[11]={sw_t[11]:.3f}")

#             # XAI-6: Heading deviation — key metric for heading loss efficacy
#             hd = xai.get("heading_deviation_deg")
#             if hd is not None and hd.shape[0] >= 1:
#                 hd_mean = hd.mean(1)
#                 print(f"  [XAI-6] Heading deviation:"
#                       f" 12h={hd_mean[0].item():.1f}°"
#                       f"  24h={hd_mean[min(2,len(hd_mean)-1)].item():.1f}°"
#                       f"  72h={hd_mean[min(10,len(hd_mean)-1)].item():.1f}°"
#                       f"  (target: 12h<20°, 72h<100°)")

#             # XAI-7: ATE/CTE decomposition
#             ac = xai.get("ate_cte_decomp", {})
#             if ac:
#                 print(f"  [XAI-7] Error:"
#                       f" ATE={ac['ate_abs_mean']:.1f}km"
#                       f"  CTE={ac['cte_abs_mean']:.1f}km"
#                       f"  ratio={ac['ate_abs_mean']/(ac['cte_abs_mean']+1e-3):.2f}")

#             # XAI-8: Per-horizon speed analysis [v2.7 NEW]
#             sph = xai.get("speed_per_horizon", {})
#             if sph and "pred_speeds_kmh" in sph and "gt_speeds_kmh" in sph:
#                 pred_s = sph["pred_speeds_kmh"]
#                 gt_s   = sph["gt_speeds_kmh"]
#                 ratio_s = sph.get("speed_ratio_per_step", [])
#                 # Report at 12h(0), 24h(3), 48h(7), 72h(11)
#                 horizons = [(0,"12h"),(3,"24h"),(7,"48h"),(11,"72h")]
#                 parts = []
#                 for idx, label in horizons:
#                     if idx < len(pred_s) and idx < len(gt_s):
#                         r = ratio_s[idx] if idx < len(ratio_s) else pred_s[idx]/max(gt_s[idx],1.)
#                         flag = "⚠" if r > 1.3 or r < 0.7 else "✓"
#                         parts.append(f"{label}: pred={pred_s[idx]:.1f} gt={gt_s[idx]:.1f} r={r:.2f}{flag}")
#                 print(f"  [XAI-8] Speed/horizon: {' | '.join(parts)}")

#             # XAI-9: Storm category breakdown [v2.7 NEW]
#             sc9 = xai.get("storm_categories", {})
#             if sc9:
#                 print(f"  [XAI-9] Storms:"
#                       f" slow(<8km/h)={sc9.get('n_slow',0)}"
#                       f"  medium(8-15)={sc9.get('n_medium',0)}"
#                       f"  fast(>15km/h)={sc9.get('n_fast',0)}"
#                       f"  obs_spd_mean={sc9.get('obs_speed_mean',0):.1f}±{sc9.get('obs_speed_std',0):.1f}km/h"
#                       f"  max={sc9.get('obs_speed_max',0):.1f}km/h")
#             print(f"  {'─'*60}")
#             result["xai"] = xai
#         except Exception as e:
#             print(f"  XAI error: {e}")
#             import traceback; traceback.print_exc()

#     print(f"  {'='*72}\n")
#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# #  Checkpoint
# # ─────────────────────────────────────────────────────────────────────────────

# def _save(path, epoch, model, opt, sched, best_score,
#           ema=None, scaler=None, extra=None):
#     m   = _unwrap(model)
#     esd = None
#     if ema is not None:
#         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
#         except Exception: pass
#     payload = {
#         "epoch": epoch, "model": m.state_dict(),
#         "optimizer": opt.state_dict(), "scheduler": sched.epoch,
#         "scaler": scaler.state_dict() if scaler is not None else None,
#         "best_score": best_score, "best_ade": best_score, "ema": esd,
#     }
#     if extra: payload.update(extra)
#     torch.save(payload, path)


# # ─────────────────────────────────────────────────────────────────────────────
# #  Args
# # ─────────────────────────────────────────────────────────────────────────────

# def get_args():
#     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     # Data
#     p.add_argument("--dataset_root",           default="TCND_vn")
#     p.add_argument("--obs_len",                default=8,      type=int)
#     p.add_argument("--pred_len",               default=12,     type=int)
#     p.add_argument("--num_workers",            default=2,      type=int)
#     p.add_argument("--other_modal",            default="gph")
#     p.add_argument("--delim",                  default=" ")
#     p.add_argument("--skip",                   default=1,      type=int)
#     p.add_argument("--min_ped",                default=1,      type=int)
#     p.add_argument("--threshold",              default=0.002,  type=float)
#     # Model
#     p.add_argument("--d_cond",                 default=256,    type=int)
#     p.add_argument("--d_model",                default=256,    type=int)
#     p.add_argument("--nhead",                  default=8,      type=int)
#     p.add_argument("--num_dec_layers",         default=4,      type=int)
#     p.add_argument("--dim_ff",                 default=512,    type=int)
#     p.add_argument("--dropout",                default=0.1,    type=float)
#     p.add_argument("--unet_in_ch",             default=13,     type=int)
#     p.add_argument("--sigma_min",              default=0.04,   type=float)
#     p.add_argument("--sigma_max",              default=0.08,   type=float)
#     p.add_argument("--lambda_reg",             default=0.2,    type=float)
#     p.add_argument("--lambda_jerk",            default=0.05,   type=float,
#                    help="[NEW v2.8] trajectory jerk loss weight (physics smoothness)")
#     p.add_argument("--lambda_heading",         default=0.10,   type=float,
#                    help="[v2.6] multi-step heading, 4 steps, decay=0.5 (was 0.05 v2.5)")
#     p.add_argument("--lambda_momentum",        default=0.0,    type=float,
#                    help="[v2.6] DISABLED — hurt test ATE by +7.9km")
#     p.add_argument("--use_ot",                 default=True,   action="store_true")
#     p.add_argument("--no_ot",                  dest="use_ot",  action="store_false")
#     p.add_argument("--ot_epsilon",             default=0.05,   type=float)
#     p.add_argument("--n_ensemble",             default=20,     type=int)
#     p.add_argument("--sigma_inference",        default=0.04,   type=float)
#     p.add_argument("--n_inference_steps",      default=1,      type=int)
#     # Training
#     p.add_argument("--num_epochs",             default=150,    type=int)
#     p.add_argument("--batch_size",             default=64,     type=int)
#     p.add_argument("--lr",                     default=2e-4,   type=float)
#     p.add_argument("--lr_min",                 default=1e-6,   type=float)
#     p.add_argument("--warmup_epochs",          default=5,      type=int)
#     p.add_argument("--weight_decay",           default=1e-4,   type=float)
#     p.add_argument("--grad_clip",              default=1.0,    type=float)
#     p.add_argument("--use_amp",                action="store_true", default=False)
#     p.add_argument("--use_ema",                default=True,   action="store_true")
#     p.add_argument("--no_ema",                 dest="use_ema", action="store_false")
#     # Encoder freeze
#     p.add_argument("--freeze_encoder_epochs",  default=10,     type=int)
#     p.add_argument("--encoder_warmup_epochs",  default=5,      type=int)
#     p.add_argument("--lr_enc_peak",            default=5e-5,   type=float)
#     # Eval
#     p.add_argument("--val_freq",               default=5,      type=int)
#     p.add_argument("--patience",               default=20,     type=int)
#     p.add_argument("--min_ep",                 default=20,     type=int)
#     p.add_argument("--hard_val_threshold",     default=0.35,   type=float)
#     p.add_argument("--hard_val_freq",          default=10,     type=int)
#     # SWA
#     p.add_argument("--swa_lr",                 default=2e-6,   type=float)
#     p.add_argument("--swa_window",             default=3,      type=int)
#     p.add_argument("--swa_threshold",          default=1.5,    type=float)
#     p.add_argument("--swa_min_ep",             default=50,     type=int)
#     # Test
#     p.add_argument("--tta_test",               default=True,   action="store_true")
#     p.add_argument("--n_tta",                  default=5,      type=int)
#     p.add_argument("--speed_tta_test",          default=True,   action="store_true",
#                    help="[NEW v2.8] Speed TTA at test: 8 scales [0.70..1.50]")
#     p.add_argument("--multiscale_test",        default=True,   action="store_true")
#     # IO
#     p.add_argument("--output_dir",             default="runs/fm_v26")
#     p.add_argument("--gpu_num",                default="0")
#     p.add_argument("--resume",                 default=None)
#     p.add_argument("--test_at_end",            action="store_true", default=True)
#     p.add_argument("--no_test",                dest="test_at_end", action="store_false")
#     return p.parse_args()


# # ─────────────────────────────────────────────────────────────────────────────
# #  Main
# # ─────────────────────────────────────────────────────────────────────────────

# def main(args):
#     if torch.cuda.is_available():
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output_dir, exist_ok=True)
#     best_ckpt      = os.path.join(args.output_dir, "best_model.pth")
#     hard_best_ckpt = os.path.join(args.output_dir, "hard_best_model.pth")
#     swa_ckpt       = os.path.join(args.output_dir, "swa_model.pth")
#     last_ckpt      = os.path.join(args.output_dir, "last_model.pth")

#     print("=" * 72)
#     print("  TC-FlowMatching v2.6")
#     print(f"  [REMOVE]  L_momentum (was hurting test ATE by +7.9km)")
#     print(f"  [UPGRADE] L_heading multi-step 4 steps decay=0.5 weight={args.lambda_heading}")
#     print(f"  [NEW-INF] Speed calibration at inference (clip 0.85–1.15)")
#     print(f"  [NEW-AUG] Obs-speed aug ×0.7–1.4 (15%)")
#     print(f"  [UPGRADE] Physics score + displacement_score")
#     print(f"  KEEP:     sigma_inf={args.sigma_inference} FIXED, L_reg linear weights")
#     print("=" * 72)

#     # ── Data ──────────────────────────────────────────────────────────────
#     print("\n  Loading data...")
#     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
#     vd, val_loader = data_loader(args, {"root": args.dataset_root, "type": "val"}, test=True)
#     print(f"  train: {len(trd)} ({len(trl)} batches/ep)")
#     print(f"  val:   {len(vd)} ({len(val_loader)} batches)")

#     # Fixed XAI batch (same every epoch for comparability)
#     try:
#         xai_batch = move(list(next(iter(val_loader))), device)
#     except Exception:
#         xai_batch = None

#     # ── Model ─────────────────────────────────────────────────────────────
#     model = TCFlowMatching(
#         pred_len=args.pred_len,        obs_len=args.obs_len,
#         unet_in_ch=args.unet_in_ch,   d_cond=args.d_cond,
#         d_model=args.d_model,          nhead=args.nhead,
#         num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
#         dropout=args.dropout,
#         sigma_min=args.sigma_min,      sigma_max=args.sigma_max,
#         lambda_reg=args.lambda_reg,    lambda_heading=args.lambda_heading,
#         lambda_jerk=args.lambda_jerk,
#         lambda_momentum=0.0,
#         use_ot=args.use_ot,            ot_epsilon=args.ot_epsilon,
#         use_ema=args.use_ema,          n_ensemble=args.n_ensemble,
#         n_inference_steps=args.n_inference_steps,
#         sigma_inference=args.sigma_inference,
#     ).to(device)

#     model.init_ema()
#     ema = getattr(_unwrap(model), "_ema", None)
#     raw = _unwrap(model)
#     n_enc = sum(p.numel() for p in raw.encoder.parameters())
#     n_vel = sum(p.numel() for p in raw.velocity.parameters())
#     print(f"\n  Encoder: {n_enc:,}  VelocityTrans: {n_vel:,}  Total: {n_enc+n_vel:,}")

#     # ── Optimizer + Scheduler ─────────────────────────────────────────────
#     opt    = build_optimizer(model, lr_velocity=args.lr, lr_encoder=0.0,
#                              weight_decay=args.weight_decay)
#     scaler = GradScaler("cuda", enabled=args.use_amp)
#     sched  = TwoGroupScheduler(
#         opt=opt, warmup_epochs=args.warmup_epochs, total_epochs=args.num_epochs,
#         lr_vel=args.lr, lr_vel_min=args.lr_min,
#         freeze_end_ep=args.freeze_encoder_epochs, lr_enc_peak=args.lr_enc_peak,
#         encoder_warmup_epochs=args.encoder_warmup_epochs)
#     print(f"\n  LR vel: {args.lr:.0e} → {args.lr_min:.0e}  "
#           f"LR enc: 0 ({args.freeze_encoder_epochs}ep) → {args.lr_enc_peak:.0e}")

#     swa = SWAHandler(swa_lr=args.swa_lr)

#     # ── Resume ────────────────────────────────────────────────────────────
#     start_ep = 0; best_score = float("inf"); best_hard = float("inf")
#     patience_cnt = 0; val_ade_history = []

#     if args.resume and os.path.exists(args.resume):
#         ck = torch.load(args.resume, map_location=device)
#         _unwrap(model).load_state_dict(ck["model"], strict=False)
#         try: opt.load_state_dict(ck["optimizer"])
#         except Exception as e: print(f"  ⚠ Opt: {e}")
#         sched.epoch  = ck.get("scheduler", 0)
#         start_ep     = ck.get("epoch", 0) + 1
#         best_score   = ck.get("best_score", ck.get("best_ade", float("inf")))
#         patience_cnt = ck.get("patience_cnt", 0)
#         if scaler and ck.get("scaler"):
#             try: scaler.load_state_dict(ck["scaler"])
#             except Exception: pass
#         if ema and ck.get("ema"):
#             for k, v in ck["ema"].items():
#                 if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
#         print(f"  ↩ Resume ep{start_ep}  best={best_score:.1f}  patience={patience_cnt}")

#     try:
#         model = torch.compile(model, mode="reduce-overhead")
#         print("  torch.compile: ok")
#     except Exception: pass

#     # ── Training loop ─────────────────────────────────────────────────────
#     nstep = len(trl)
#     print(f"\n  TRAINING ({nstep} steps/ep × {args.num_epochs} ep)")
#     print(f"  Aug: shift±5km + speed×[0.85,1.15] + recurv±20° + obs-speed×[0.7,1.4] + noise")
#     print(f"  Loss: L_CFM + L_reg(linear) + L_heading_ms(4steps,decay=0.5)")
#     print(f"  Inf:  1-shot sigma=0.04 + speed_calibrate(±15%) + top3 physics")
#     print()

#     for ep in range(start_ep, start_ep + args.num_epochs):
#         rel_ep = ep - start_ep
#         freeze = rel_ep < args.freeze_encoder_epochs
#         for p in _unwrap(model).encoder.parameters():
#             p.requires_grad_(not freeze)

#         if rel_ep == 0 and freeze:
#             print(f"  *** Ep{ep}: encoder frozen ***")
#         if rel_ep == args.freeze_encoder_epochs:
#             print(f"\n  *** Ep{ep}: encoder unfrozen ***")

#         model.train()
#         sum_loss = sum_cfm = sum_reg = sum_head = sum_jerk = sum_ade1 = 0.0
#         t0_ep = time.perf_counter()

#         for i, batch in enumerate(trl):
#             bl = move(list(batch), device)
#             bl_aug = augment_batch(bl)   # training augmentation

#             opt.zero_grad()
#             with autocast(device_type="cuda", enabled=args.use_amp):
#                 bd = model.get_loss_breakdown(bl_aug, epoch=ep)

#             scaler.scale(bd["total"]).backward()
#             scaler.unscale_(opt)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

#             if freeze:
#                 for p in _unwrap(model).encoder.parameters():
#                     if p.grad is not None: p.grad.zero_()

#             scaler.step(opt); scaler.update()
#             model.ema_update()
#             swa.update(model)

#             sum_loss += bd["total"].item(); sum_cfm  += bd["l_cfm"]
#             sum_reg  += bd["l_reg"];        sum_head += bd["l_heading"]
#             sum_ade1 += bd["ade_1step"]

#             if i % 30 == 0:
#                 _, lr_vel = get_lrs(opt)
#                 enc_s = "frozen" if freeze else "active"
#                 swa_s = " [SWA]" if swa.active else ""
#                 print(f"  [{ep:>3}][{i:>3}/{nstep}]"
#                       f"  tot={bd['total'].item():.4f}"
#                       f"  cfm={bd['l_cfm']:.4f}"
#                       f"  reg={bd['l_reg']:.4f}"
#                       f"  h4s={bd['l_heading']:.4f}  jrk={bd.get('l_jerk',0):.4f}"
#                       f"  lam_d={bd['lam_dir']:.2f}"
#                       f"  ade1={bd['ade_1step']:.0f}km"
#                       f"  enc={enc_s}{swa_s}"
#                       f"  lr={lr_vel:.2e}")

#         train_loss = sum_loss / nstep
#         _, lr_vel_used = get_lrs(opt)
#         sched.step()

#         print(f"\n  ── Ep{ep:>3}"
#               f"  train={train_loss:.6f}"
#               f"  cfm={sum_cfm/nstep:.4f}"
#               f"  reg={sum_reg/nstep:.4f}"
#               f"  h4s={sum_head/nstep:.4f}"
#               f"  ade1={sum_ade1/nstep:.0f}km"
#               f"  lr={lr_vel_used:.2e}"
#               f"  t={time.perf_counter()-t0_ep:.0f}s")

#         _save(last_ckpt, ep, model, opt, sched, best_score, ema, scaler)
#         if ep % 5 == 0:
#             ep_ckpt = os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth")
#             _save(ep_ckpt, ep, model, opt, sched, best_score, ema, scaler)
#             print(f"  💾 {ep_ckpt}")

#         # ── Val evaluation ───────────────────────────────────────────────
#         if rel_ep % args.val_freq == 0:
#             run_xai_this = (rel_ep % 10 == 0)
#             r = evaluate(model, val_loader, device, tag=f"VAL ep{ep}",
#                          n_ensemble=args.n_ensemble, ema=ema,
#                          ref_targets=ST_TRANS_VAL, epoch_for_loss=ep,
#                          run_xai=run_xai_this, xai_batch=xai_batch)

#             val_ade = r["ADE"]; score = r["combined_score"]
#             val_ade_history.append(val_ade)

#             # Trend
#             if len(val_ade_history) >= 4:
#                 trend = float(np.mean(val_ade_history[-2:])) - float(np.mean(val_ade_history[-4:-2]))
#                 trend_s = f"↑{trend:+.1f}km⚠" if trend > 5 else f"↓{trend:+.1f}km✓" if trend < -5 else f"→{trend:+.1f}flat"
#             else:
#                 trend_s = "—"
#             print(f"  train={train_loss:.6f}  val_ADE={val_ade:.1f}  combined={score:.1f}  trend={trend_s}")

#             # SWA activation check
#             if (not swa.active and ep >= args.swa_min_ep
#                     and swa.should_activate(val_ade_history, args.swa_window, args.swa_threshold)):
#                 swa.activate(model, opt, ep)

#             # Best checkpoint
#             if score < best_score:
#                 best_score = score; patience_cnt = 0
#                 _save(best_ckpt, ep, model, opt, sched, best_score, ema, scaler,
#                       extra={"val_ade": r["ADE"], "val_ate": r["ATE"],
#                              "val_cte": r["CTE"], "patience_cnt": 0})
#                 print(f"  ✅ Best! score={best_score:.2f}"
#                       f"  ADE={r['ADE']:.1f} ATE={r['ATE']:.1f} CTE={r['CTE']:.1f}")
#             else:
#                 if rel_ep >= args.min_ep: patience_cnt += args.val_freq
#                 print(f"  No improve {patience_cnt}/{args.patience} (best={best_score:.1f})")
#                 if rel_ep >= args.min_ep and patience_cnt >= args.patience:
#                     print(f"  ⛔ Early stop @ ep{ep}")
#                     break

#         # ── Hard-val evaluation ──────────────────────────────────────────
#         if rel_ep % args.hard_val_freq == 0 and rel_ep >= args.min_ep:
#             r_h = evaluate_hard_val(model, val_loader, device,
#                                      hard_threshold=args.hard_val_threshold,
#                                      n_ensemble=args.n_ensemble, ema=ema,
#                                      epoch_for_loss=ep)
#             print(f"  [HVAL] n={r_h['n_hard']}"
#                   f"  ADE={r_h['ADE']:.1f} ATE={r_h['ATE']:.1f} CTE={r_h['CTE']:.1f}"
#                   f"  combined={r_h['combined_score']:.1f}")
#             if r_h["combined_score"] < best_hard and r_h["n_hard"] >= 10:
#                 best_hard = r_h["combined_score"]
#                 _save(hard_best_ckpt, ep, model, opt, sched, best_hard, ema, scaler,
#                       extra={"hard_val_ade": r_h["ADE"], "selection_criterion": "hard_val"})
#                 print(f"  💎 Hard-best! score={best_hard:.2f} ADE={r_h['ADE']:.1f}")

#         # ── SWA eval ─────────────────────────────────────────────────────
#         if swa.active and rel_ep % args.val_freq == 0 and swa.n_updates >= 10:
#             backup = {k: v.detach().clone()
#                       for k, v in _unwrap(model).state_dict().items()
#                       if v.dtype.is_floating_point}
#             swa.apply_to_model(model)
#             r_swa = evaluate(model, val_loader, device, tag=f"SWA ep{ep}",
#                               n_ensemble=args.n_ensemble, ema=None,
#                               ref_targets=ST_TRANS_VAL, epoch_for_loss=ep)
#             swa.restore_from_backup(model, backup)

#             swa_score = r_swa["combined_score"]
#             print(f"  [SWA] score={swa_score:.2f} ({swa.n_updates} updates) vs best={best_score:.2f}")
#             if swa_score < best_score:
#                 best_score = swa_score; patience_cnt = 0
#                 swa.save_avg_state(swa_ckpt, ep, best_score,
#                                    extra={"val_ade": r_swa["ADE"],
#                                           "val_ate": r_swa["ATE"],
#                                           "val_cte": r_swa["CTE"]})
#                 import shutil; shutil.copy(swa_ckpt, best_ckpt)
#                 print(f"  ✅ SWA best! score={best_score:.2f} ADE={r_swa['ADE']:.1f}")

#     # ── Test evaluation ───────────────────────────────────────────────────
#     print(f"\n  Done! best_score={best_score:.2f}")
#     if not args.test_at_end: return

#     print("\n  Loading best checkpoint for TEST...")
#     if not os.path.exists(best_ckpt):
#         print("  No checkpoint found."); return

#     ck = torch.load(best_ckpt, map_location=device)
#     is_swa = ck.get("is_swa", False)
#     _unwrap(model).load_state_dict(ck["model"], strict=False)
#     if not is_swa and ema and ck.get("ema"):
#         for k, v in ck["ema"].items():
#             if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
#     print(f"  Loaded ep{ck.get('epoch','?')} (is_swa={is_swa})")

#     try:
#         _, test_loader = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
#         print(f"  Test: {len(test_loader)} batches")
#     except Exception:
#         print("  No test set → using val"); test_loader = val_loader

#     # Standard test
#     r_test = evaluate(model, test_loader, device, tag="TEST (best ckpt)",
#                       n_ensemble=args.n_ensemble, ema=None if is_swa else ema,
#                       ref_targets=ST_TRANS_TEST, run_xai=True, xai_batch=xai_batch)

#     # TTA test
#     if args.tta_test:
#         r_tta = evaluate(model, test_loader, device, tag="TEST+TTA",
#                          n_ensemble=args.n_ensemble, ema=None if is_swa else ema,
#                          ref_targets=ST_TRANS_TEST, use_tta=True, n_tta=args.n_tta)
#         print(f"\n  TTA: ADE {r_test['ADE']:.1f}→{r_tta['ADE']:.1f}  "
#               f"ATE {r_test['ATE']:.1f}→{r_tta['ATE']:.1f}  "
#               f"CTE {r_test['CTE']:.1f}→{r_tta['CTE']:.1f}")

#     # Multi-scale test
#     if args.multiscale_test:
#         raw_m = _unwrap(model)
#         if hasattr(raw_m, "sample_multiscale"):
#             print("\n  Multi-scale sigma test...")
#             ms_ades, ms_ates, ms_ctes = [], [], []
#             ms_steps = defaultdict(list)
#             raw_m.eval()
#             bk_ms = (ema.apply_to(model) if (ema and not is_swa) else None)
#             with torch.no_grad():
#                 for batch in test_loader:
#                     bl = move(list(batch), device)
#                     try:
#                         pred, _, _ = raw_m.sample_multiscale(bl)
#                     except Exception: continue
#                     gt = bl[1]; T = min(pred.shape[0], gt.shape[0])
#                     pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
#                     dist = _haversine_deg(pd, gd)
#                     ate, cte = _ate_cte(pd, gd)
#                     ms_ades.extend(dist.mean(0).tolist())
#                     if ate.shape[0] > 0:
#                         ms_ates.extend(ate.abs().mean(0).tolist())
#                         ms_ctes.extend(cte.abs().mean(0).tolist())
#                     for h, s in HORIZON_STEPS.items():
#                         if s < T: ms_steps[h].extend(dist[s].tolist())
#             if bk_ms: ema.restore(model, bk_ms)
#             def _mm(lst): return float(np.mean(lst)) if lst else float("nan")
#             print(f"  Multi-scale: ADE={_mm(ms_ades):.1f} ATE={_mm(ms_ates):.1f} CTE={_mm(ms_ctes):.1f}")

#     # Speed TTA test
#     if args.speed_tta_test and hasattr(_unwrap(model), "sample_speed_tta"):
#         print("\n  Speed TTA test (8 scales [0.70..1.50])...")
#         stta_ades, stta_ates, stta_ctes = [], [], []
#         raw_m.eval()
#         bk_stta = (ema.apply_to(model) if (ema and not is_swa) else None)
#         with torch.no_grad():
#             for batch in test_loader:
#                 bl = move(list(batch), device)
#                 try:
#                     pred, _, _ = raw_m.sample_speed_tta(bl)
#                 except Exception as e:
#                     print(f"  speed_tta error: {e}"); continue
#                 gt = bl[1]; T = min(pred.shape[0], gt.shape[0])
#                 pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
#                 dist = _haversine_deg(pd, gd)
#                 ate, cte = _ate_cte(pd, gd)
#                 stta_ades.extend(dist.mean(0).tolist())
#                 if ate.shape[0] > 0:
#                     stta_ates.extend(ate.abs().mean(0).tolist())
#                     stta_ctes.extend(cte.abs().mean(0).tolist())
#         if bk_stta: ema.restore(model, bk_stta)
#         def _smm(lst): return float(np.mean(lst)) if lst else float("nan")
#         print(f"  Speed TTA: ADE={_smm(stta_ades):.1f} ATE={_smm(stta_ates):.1f} CTE={_smm(stta_ctes):.1f}")
#         print(f"  vs Standard: ADE={r_test['ADE']:.1f} ATE={r_test['ATE']:.1f} CTE={r_test['CTE']:.1f}")

#     # Final comparison
#     val_ade_b = ck.get("val_ade", float("nan"))
#     val_ate_b = ck.get("val_ate", float("nan"))
#     val_cte_b = ck.get("val_cte", float("nan"))
#     v21 = {"ADE": 229.8, "ATE": 214.4, "CTE": 71.6}
#     print("\n" + "=" * 72)
#     print("  v2.6 FINAL RESULTS vs v2.1 baseline")
#     print("=" * 72)
#     print(f"  {'Metric':<10} {'Val':>10} {'Test':>10} {'Gap':>10} {'v2.1 Test':>12} Status")
#     print("  " + "─"*60)
#     for m_n, val_v, test_v, v21_v in [
#         ("ADE", val_ade_b, r_test["ADE"], v21["ADE"]),
#         ("ATE", val_ate_b, r_test["ATE"], v21["ATE"]),
#         ("CTE", val_cte_b, r_test["CTE"], v21["CTE"]),
#     ]:
#         gap  = test_v - val_v if (np.isfinite(test_v) and np.isfinite(val_v)) else float("nan")
#         impr = v21_v - test_v
#         flag = (f"↓{impr:+.1f}" if np.isfinite(impr) and impr > 0 else
#                 f"↑{-impr:+.1f}" if np.isfinite(impr) else "?")
#         print(f"  {m_n:<10} {val_v:>10.2f} {test_v:>10.2f} {gap:>+10.2f} {v21_v:>12.1f}  {flag}")
#     print("=" * 72)


# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(42); torch.manual_seed(42)
#     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
#     if args.dataset_root == "TCND_vn":
#         _auto = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
#         if os.path.isdir(_auto):
#             args.dataset_root = _auto
#     main(args)
"""
scripts/train_fm.py  ──  TC-FlowMatching v2.6 Training
═══════════════════════════════════════════════════════════════════════════════

Thay đổi từ v2.5 (dựa trên phân tích 140 epoch log đầy đủ):

  [REMOVE]  L_momentum — làm ATE test tệ hơn +7.9km (anchor to val speed)
  [UPGRADE] L_heading multi-step (4 steps, decay=0.5), weight 0.05→0.10
  [NEW-INF] Speed calibration tại inference (per-storm, clip ±15%)
  [UPGRADE] Physics score + displacement_score
  [AUG-D]   Obs-speed scaling ×0.7–1.4 (15% prob)

Giữ nguyên (proven):
  ✅ sigma_inference=0.04 FIXED
  ✅ L_reg softmax-linspace weights
  ✅ 2-group optimizer, encoder freeze 10ep
  ✅ val loop NO augmentation
  ✅ 1-shot inference
  ✅ SWA sau plateau
  ✅ Hard-val checkpoint (HVAL)
  ✅ XAI logging mỗi 10 epoch
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, math, time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler

from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _haversine_deg,
    EMAModel, augment_batch, hard_score_from_obs,
    compute_ensemble_uncertainty,
    compute_heading_deviation, compute_cte_contribution,
    compute_obs_attribution,
)

HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}
ST_TRANS_VAL  = {"ADE": 172.68, "ATE": 142.21, "CTE": 42.04,
                 "12h": 65.42, "24h": 104.67, "48h": 205.10, "72h": 321.39}
ST_TRANS_TEST = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
                 "12h": 77.5, "24h": 130.5, "48h": 269.9, "72h": 423.3}

# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m

def move(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
    return out

def _ate_cte(pred_deg, gt_deg):
    """Decompose error into along-track and cross-track components."""
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    lo1 = torch.deg2rad(gt_deg[:T-1,:,0]); la1 = torch.deg2rad(gt_deg[:T-1,:,1])
    lo2 = torch.deg2rad(gt_deg[1:T, :,0]); la2 = torch.deg2rad(gt_deg[1:T, :,1])
    lo3 = torch.deg2rad(pred_deg[1:T,:,0]); la3 = torch.deg2rad(pred_deg[1:T,:,1])
    ya  = torch.sin(lo2-lo1)*torch.cos(la2)
    xa  = torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
    be  = torch.atan2(ya, xa)
    ye  = torch.sin(lo3-lo2)*torch.cos(la3)
    xe  = torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
    bee = torch.atan2(ye, xe)
    tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang = bee - be
    return tot*torch.cos(ang), tot*torch.sin(ang)


# ─────────────────────────────────────────────────────────────────────────────
#  2-group optimizer (encoder freeze pattern from v2.1)
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model, lr_velocity, lr_encoder, weight_decay):
    raw = _unwrap(model)
    # [FIX BUG #1] raw_step_w must be in a dedicated param group
    # Not in encoder/velocity → would be missing from optimizer entirely
    # lr=5e-3: faster than model params (step weights are scalars, converge quickly)
    step_w_params = [raw.raw_step_w] if hasattr(raw, 'raw_step_w') else []
    groups = [
        {"params": list(raw.encoder.parameters()),  "lr": lr_encoder,  "name": "encoder"},
        {"params": list(raw.velocity.parameters()), "lr": lr_velocity, "name": "velocity"},
    ]
    if step_w_params:
        groups.append({"params": step_w_params, "lr": 5e-3, "name": "step_weights"})
    return optim.AdamW(groups, weight_decay=weight_decay)

def get_lrs(opt):
    lr_enc = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "encoder")
    lr_vel = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "velocity")
    return lr_enc, lr_vel

class TwoGroupScheduler:
    def __init__(self, opt, warmup_epochs, total_epochs,
                 lr_vel, lr_vel_min, freeze_end_ep,
                 lr_enc_peak, encoder_warmup_epochs=5):
        self.opt         = opt
        self.warmup      = warmup_epochs
        self.total       = total_epochs
        self.lr_vel      = lr_vel
        self.lr_vel_min  = lr_vel_min
        self.freeze_end  = freeze_end_ep
        self.lr_enc_peak = lr_enc_peak
        self.enc_warmup  = encoder_warmup_epochs
        self.epoch       = 0

    def _cosine(self, ep_from, ep_to, lr_s, lr_e, ep):
        t = max(0., min(1., (ep - ep_from) / max(ep_to - ep_from, 1)))
        return lr_e + 0.5 * (lr_s - lr_e) * (1 + math.cos(math.pi * t))

    def step(self):
        ep = self.epoch
        if ep < self.warmup:
            lr_vel = self.lr_vel * (0.1 + 0.9 * ep / max(self.warmup - 1, 1))
        else:
            lr_vel = self._cosine(self.warmup, self.total, self.lr_vel, self.lr_vel_min, ep)
        if ep < self.freeze_end:
            lr_enc = 0.0
        elif ep < self.freeze_end + self.enc_warmup:
            lr_enc = self.lr_enc_peak * (ep - self.freeze_end) / self.enc_warmup
        else:
            lr_enc = self._cosine(self.freeze_end + self.enc_warmup, self.total,
                                  self.lr_enc_peak, self.lr_vel_min, ep)
        for pg in self.opt.param_groups:
            if pg.get("name") == "encoder":
                pg["lr"] = lr_enc
            elif pg.get("name") == "step_weights":
                pg["lr"] = 1e-3   # fixed lr for learned step weights (no decay needed)
            else:
                pg["lr"] = lr_vel
        self.epoch += 1
        return lr_vel, lr_enc


# ─────────────────────────────────────────────────────────────────────────────
#  SWA handler (same as v2.5 — proven)
# ─────────────────────────────────────────────────────────────────────────────

class SWAHandler:
    """
    Stochastic Weight Averaging after val ADE plateau.
    Activation: val ADE improvement < threshold km in last `window` checks.
    After activation: average model weights each epoch → flatter loss landscape
    → better generalization.
    """
    def __init__(self, swa_lr: float = 2e-6):
        self.swa_lr    = swa_lr
        self.active    = False
        self.start_ep  = None
        self.n_updates = 0
        self.avg_state = {}

    def should_activate(self, ade_history: List[float],
                         window: int = 3, threshold: float = 1.5) -> bool:
        if len(ade_history) < window: return False
        return (ade_history[-window] - ade_history[-1]) < threshold

    def activate(self, model, opt, ep: int):
        self.active = True; self.start_ep = ep
        for pg in opt.param_groups: pg["lr"] = self.swa_lr
        m = _unwrap(model)
        self.avg_state = {k: v.detach().clone().float()
                          for k, v in m.state_dict().items()
                          if v.dtype.is_floating_point}
        self.n_updates = 1
        print(f"  *** SWA ACTIVATED @ ep{ep} (lr → {self.swa_lr:.1e}) ***")

    def update(self, model):
        if not self.active: return
        m = _unwrap(model); sd = m.state_dict(); n = self.n_updates
        for k in self.avg_state:
            if k in sd:
                self.avg_state[k] = (n * self.avg_state[k] + sd[k].detach().float()) / (n + 1)
        self.n_updates += 1

    def apply_to_model(self, model):
        if not self.active or not self.avg_state: return
        m = _unwrap(model); sd = m.state_dict()
        for k in self.avg_state:
            if k in sd: sd[k].copy_(self.avg_state[k].to(sd[k].device))

    def restore_from_backup(self, model, backup):
        m = _unwrap(model); sd = m.state_dict()
        for k, v in backup.items():
            if k in sd: sd[k].copy_(v)

    def save_avg_state(self, path: str, epoch: int, best_score: float,
                       extra: Optional[dict] = None):
        payload = {"epoch": epoch, "model": self.avg_state,
                   "best_score": best_score, "is_swa": True,
                   "swa_updates": self.n_updates}
        if extra: payload.update(extra)
        torch.save(payload, path)


# ─────────────────────────────────────────────────────────────────────────────
#  Hard-val evaluator (HVAL)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_hard_val(model, val_loader, device, hard_threshold: float = 0.35,
                       n_ensemble: int = 20, ema=None, epoch_for_loss: int = 9999):
    """
    Evaluate on hard storms only (hard_score > threshold).
    Hard storms are closer to test distribution (more recurvature).
    Best hard-val checkpoint often generalizes better than overall best.
    """
    bk = None
    if ema is not None:
        try: bk = ema.apply_to(model)
        except Exception: pass

    model.eval()
    all_ade, all_ate, all_cte = [], [], []
    n_hard = 0

    for batch in val_loader:
        bl = move(list(batch), device)
        B  = bl[0].shape[1]
        h_score   = hard_score_from_obs(bl[0][:, :, :2])
        hard_mask = h_score > hard_threshold
        if hard_mask.sum() == 0: continue
        hard_idx  = hard_mask.nonzero(as_tuple=True)[0]

        # Build sub-batch for hard storms
        bl_h = list(bl)
        for i, item in enumerate(bl_h):
            if torch.is_tensor(item):
                if item.dim() >= 2 and item.shape[1] == B:
                    bl_h[i] = item[:, hard_idx, ...]
                elif item.dim() >= 1 and item.shape[0] == B:
                    bl_h[i] = item[hard_idx, ...]

        try:
            pred, _, _ = model.sample(bl_h, num_ensemble=n_ensemble)
        except Exception as e:
            print(f"  hard val error: {e}"); continue

        gt = bl_h[1]
        T  = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
        dist = _haversine_deg(pd, gd)
        ate, cte = _ate_cte(pd, gd)
        all_ade.extend(dist.mean(0).tolist())
        if ate.shape[0] > 0:
            all_ate.extend(ate.abs().mean(0).tolist())
            all_cte.extend(cte.abs().mean(0).tolist())
        n_hard += len(hard_idx)

    if bk is not None:
        try: ema.restore(model, bk)
        except Exception: pass

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")
    return {
        "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
        "n_hard": n_hard,
        "combined_score": 0.6*_m(all_ade) + 0.2*_m(all_ate) + 0.2*_m(all_cte),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, tag: str = "",
             n_ensemble: int = 20, ema=None,
             ref_targets=None, use_tta: bool = False,
             n_tta: int = 5, epoch_for_loss: int = 9999,
             run_xai: bool = False, xai_batch=None) -> Dict:
    """
    Full evaluation. NO augmentation (val_loss is reliable generalization signal).
    XAI logging every 10 epochs shows training dynamics.
    """
    bk = None
    if ema is not None:
        try: bk = ema.apply_to(model)
        except Exception as e: print(f"  ⚠ EMA: {e}")

    model.eval()
    all_ade, all_ate, all_cte = [], [], []
    step_dist = defaultdict(list)
    sum_loss = sum_cfm = sum_head = 0.0
    sum_n = 0

    for batch in loader:
        bl = move(list(batch), device)
        gt = bl[1]; B = bl[0].shape[1]

        # Val loss (no augmentation)
        try:
            bd = model.get_loss_breakdown(bl, epoch=epoch_for_loss)
            if torch.isfinite(bd["total"]):
                sum_loss += bd["total"].item() * B
                sum_cfm  += bd["l_cfm"] * B
                sum_head += bd["l_heading"]
                sum_jerk  += bd.get("l_jerk", 0.0) * B
                sum_n    += B
        except Exception: pass

        # Inference (standard or speed-scale TTA)
        if use_tta:
            obs = bl[0]; anchor = obs[-1:, :, :2].detach()
            scales = [0.875, 0.9375, 1.0, 1.0625, 1.125][:n_tta]
            preds_t, weights_t = [], []
            for sc in scales:
                obs_s = obs.clone(); obs_s[..., :2] = anchor + (obs[..., :2] - anchor) * sc
                bl_s = list(bl); bl_s[0] = obs_s
                try:
                    p, _, _ = model.sample(bl_s, num_ensemble=n_ensemble)
                    preds_t.append(p)
                    weights_t.append(2.0 if abs(sc - 1.0) < 1e-6 else 1.0)
                except Exception: continue
            if not preds_t: continue
            tw   = sum(weights_t)
            pred = sum(w / tw * p for w, p in zip(weights_t, preds_t))
        else:
            try:
                pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
            except Exception as e:
                print(f"  sample error: {e}"); continue

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
    val_loss = sum_loss / max(sum_n, 1)

    result = {
        "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
        "n": len(all_ade),
        "val_loss": val_loss,
        "val_cfm_loss":  sum_cfm / max(sum_n, 1),
        "val_head_loss": sum_head / max(sum_n, 1),
        "val_mom_loss":  0.0,   # always 0
    }
    for h in HORIZON_STEPS: result[f"{h}h"] = _m(step_dist[h])
    ade, ate_, cte_ = result["ADE"], result["ATE"], result["CTE"]
    result["combined_score"] = (
        0.6 * ade + 0.2 * ate_ + 0.2 * cte_
        if all(np.isfinite(x) for x in [ade, ate_, cte_]) else ade)

    ref = ref_targets or ST_TRANS_VAL
    def _v(k): return result.get(k, float("nan"))
    def _ok(k): return "✓" if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9) else "✗"

    tta_str = " [TTA]" if use_tta else ""
    print(f"\n  {'='*72}")
    print(f"  [{tag}]{tta_str}  n={result['n']}")
    print(f"  Val Loss : {val_loss:.6f}  cfm={result['val_cfm_loss']:.6f}  "
          f"head4s={result['val_head_loss']:.6f}  [mom=DISABLED]")
    print(f"  ADE={_v('ADE'):7.1f}km {_ok('ADE')}  "
          f"ATE={_v('ATE'):7.1f}km {_ok('ATE')}  "
          f"CTE={_v('CTE'):7.1f}km {_ok('CTE')}")
    print(f"  Combined = {_v('combined_score'):.1f}")
    print(f"  12h={_v('12h'):6.1f}  24h={_v('24h'):6.1f}  "
          f"48h={_v('48h'):6.1f}  72h={_v('72h'):6.1f} km")
    beat = [f"{k}={_v(k):.0f}<{ref.get(k,999):.0f}"
            for k in ["ADE","ATE","CTE","12h","24h","48h","72h"]
            if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9)]
    print(f"  BEAT: {' | '.join(beat) if beat else 'none yet'}")

    # XAI logging
    if run_xai and xai_batch is not None:
        try:
            _, _, _, xai = _unwrap(model).sample(xai_batch, return_xai=True)

            print(f"  {'─'*60}")
            print(f"  XAI Summary (fixed val batch)")
            print(f"  {'─'*60}")

            # XAI-4: Uncertainty
            print(f"  [XAI-4] Uncertainty:"
                  f" 12h={xai['mean_12h_std']:.1f}km"
                  f"  72h={xai['mean_72h_std']:.1f}km"
                  f"  ratio={float(xai['uncertainty_ratio'].mean()):.2f}×"
                  f"  high_uncert={xai['high_uncertainty'].sum().item()}")

            # XAI-2: Hard score components (v2.7: thêm speed_cv)
            hc = xai.get("hard_components", {})
            if hc:
                spd_cv_str = f"  spd_cv={float(hc['speed_cv'].mean()):.3f}" if 'speed_cv' in hc else ""
                print(f"  [XAI-2] HardScore:"
                      f" curv={float(hc['curvature'].mean()):.3f}"
                      f"  spd_var={float(hc['speed_var'].mean()):.3f}"
                      f"  dir_chg={float(hc['dir_change'].mean()):.3f}"
                      f"{spd_cv_str}")

            # XAI-3: Physics score components
            pc = xai.get("physics_components", {})
            if pc:
                print(f"  [XAI-3] Physics:"
                      f" speed={float(pc['speed'].mean()):.3f}"
                      f"  smooth={float(pc['smooth'].mean()):.3f}"
                      f"  heading={float(pc['heading'].mean()):.3f}")

            # XAI-5: Speed comparison (shows calibration effect)
            sc = xai.get("speed_comparison", {})
            if sc:
                ratio = sc.get("speed_ratio", 1.0)
                flag  = ("⚠ OVER" if ratio > 1.15 else
                         "⚠ UNDER" if ratio < 0.85 else "✓ OK")
                n_over  = sc.get("over_predict",  torch.zeros(1)).sum().item()
                n_under = sc.get("under_predict", torch.zeros(1)).sum().item()
                print(f"  [XAI-5] Speed (post-calibration):"
                      f" obs={sc['obs_speed_mean']:.1f}km/h"
                      f"  pred={sc['pred_speed_mean']:.1f}km/h"
                      f"  ratio={ratio:.2f} {flag}"
                      f"  (over:{int(n_over)} under:{int(n_under)})")

            # Learned step weights (mỗi 10 epoch → xem model học gì)
            sw_learned = xai.get("learned_step_weights", [])
            if sw_learned:
                sw_t = torch.tensor(sw_learned)
                early = float(sw_t[:4].sum())
                mid   = float(sw_t[4:8].sum())
                late  = float(sw_t[8:].sum())
                print(f"  [XAI-W] Step weights:"
                      f" early(6-24h)={early:.1%}"
                      f"  mid(30-48h)={mid:.1%}"
                      f"  late(54-72h)={late:.1%}"
                      f"  (want: late > 40%)")
                ratio = sw_t.max().item() / sw_t.min().item()
                print(f"           max/min={ratio:.2f}×  "
                      f"w[0]={sw_t[0]:.3f} w[5]={sw_t[5]:.3f} w[11]={sw_t[11]:.3f}")

            # XAI-6: Heading deviation — key metric for heading loss efficacy
            hd = xai.get("heading_deviation_deg")
            if hd is not None and hd.shape[0] >= 1:
                hd_mean = hd.mean(1)
                print(f"  [XAI-6] Heading deviation:"
                      f" 12h={hd_mean[0].item():.1f}°"
                      f"  24h={hd_mean[min(2,len(hd_mean)-1)].item():.1f}°"
                      f"  72h={hd_mean[min(10,len(hd_mean)-1)].item():.1f}°"
                      f"  (target: 12h<20°, 72h<100°)")

            # XAI-7: ATE/CTE decomposition
            ac = xai.get("ate_cte_decomp", {})
            if ac:
                print(f"  [XAI-7] Error:"
                      f" ATE={ac['ate_abs_mean']:.1f}km"
                      f"  CTE={ac['cte_abs_mean']:.1f}km"
                      f"  ratio={ac['ate_abs_mean']/(ac['cte_abs_mean']+1e-3):.2f}")

            # XAI-8: Per-horizon speed analysis [v2.7 NEW]
            sph = xai.get("speed_per_horizon", {})
            if sph and "pred_speeds_kmh" in sph and "gt_speeds_kmh" in sph:
                pred_s = sph["pred_speeds_kmh"]
                gt_s   = sph["gt_speeds_kmh"]
                ratio_s = sph.get("speed_ratio_per_step", [])
                # Report at 12h(0), 24h(3), 48h(7), 72h(11)
                horizons = [(0,"12h"),(3,"24h"),(7,"48h"),(11,"72h")]
                parts = []
                for idx, label in horizons:
                    if idx < len(pred_s) and idx < len(gt_s):
                        r = ratio_s[idx] if idx < len(ratio_s) else pred_s[idx]/max(gt_s[idx],1.)
                        flag = "⚠" if r > 1.3 or r < 0.7 else "✓"
                        parts.append(f"{label}: pred={pred_s[idx]:.1f} gt={gt_s[idx]:.1f} r={r:.2f}{flag}")
                print(f"  [XAI-8] Speed/horizon: {' | '.join(parts)}")

            # XAI-9: Storm category breakdown [v2.7 NEW]
            sc9 = xai.get("storm_categories", {})
            if sc9:
                print(f"  [XAI-9] Storms:"
                      f" slow(<8km/h)={sc9.get('n_slow',0)}"
                      f"  medium(8-15)={sc9.get('n_medium',0)}"
                      f"  fast(>15km/h)={sc9.get('n_fast',0)}"
                      f"  obs_spd_mean={sc9.get('obs_speed_mean',0):.1f}±{sc9.get('obs_speed_std',0):.1f}km/h"
                      f"  max={sc9.get('obs_speed_max',0):.1f}km/h")
            print(f"  {'─'*60}")
            result["xai"] = xai
        except Exception as e:
            print(f"  XAI error: {e}")
            import traceback; traceback.print_exc()

    print(f"  {'='*72}\n")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _save(path, epoch, model, opt, sched, best_score,
          ema=None, scaler=None, extra=None):
    m   = _unwrap(model)
    esd = None
    if ema is not None:
        try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
        except Exception: pass
    payload = {
        "epoch": epoch, "model": m.state_dict(),
        "optimizer": opt.state_dict(), "scheduler": sched.epoch,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_score": best_score, "best_ade": best_score, "ema": esd,
    }
    if extra: payload.update(extra)
    torch.save(payload, path)


# ─────────────────────────────────────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    p.add_argument("--dataset_root",           default="TCND_vn")
    p.add_argument("--obs_len",                default=8,      type=int)
    p.add_argument("--pred_len",               default=12,     type=int)
    p.add_argument("--num_workers",            default=2,      type=int)
    p.add_argument("--other_modal",            default="gph")
    p.add_argument("--delim",                  default=" ")
    p.add_argument("--skip",                   default=1,      type=int)
    p.add_argument("--min_ped",                default=1,      type=int)
    p.add_argument("--threshold",              default=0.002,  type=float)
    # Model
    p.add_argument("--d_cond",                 default=256,    type=int)
    p.add_argument("--d_model",                default=256,    type=int)
    p.add_argument("--nhead",                  default=8,      type=int)
    p.add_argument("--num_dec_layers",         default=4,      type=int)
    p.add_argument("--dim_ff",                 default=512,    type=int)
    p.add_argument("--dropout",                default=0.1,    type=float)
    p.add_argument("--unet_in_ch",             default=13,     type=int)
    p.add_argument("--sigma_min",              default=0.04,   type=float)
    p.add_argument("--sigma_max",              default=0.08,   type=float)
    p.add_argument("--lambda_reg",             default=0.2,    type=float)
    p.add_argument("--lambda_jerk",            default=0.05,   type=float,
                   help="[NEW v2.8] trajectory jerk loss weight (physics smoothness)")
    p.add_argument("--lambda_heading",         default=0.10,   type=float,
                   help="[v2.6] multi-step heading, 4 steps, decay=0.5 (was 0.05 v2.5)")
    p.add_argument("--lambda_momentum",        default=0.0,    type=float,
                   help="[v2.6] DISABLED — hurt test ATE by +7.9km")
    p.add_argument("--use_ot",                 default=True,   action="store_true")
    p.add_argument("--no_ot",                  dest="use_ot",  action="store_false")
    p.add_argument("--ot_epsilon",             default=0.05,   type=float)
    p.add_argument("--n_ensemble",             default=20,     type=int)
    p.add_argument("--sigma_inference",        default=0.04,   type=float)
    p.add_argument("--n_inference_steps",      default=1,      type=int)
    # Training
    p.add_argument("--num_epochs",             default=150,    type=int)
    p.add_argument("--batch_size",             default=64,     type=int)
    p.add_argument("--lr",                     default=2e-4,   type=float)
    p.add_argument("--lr_min",                 default=1e-6,   type=float)
    p.add_argument("--warmup_epochs",          default=5,      type=int)
    p.add_argument("--weight_decay",           default=1e-4,   type=float)
    p.add_argument("--grad_clip",              default=1.0,    type=float)
    p.add_argument("--use_amp",                action="store_true", default=False)
    p.add_argument("--use_ema",                default=True,   action="store_true")
    p.add_argument("--no_ema",                 dest="use_ema", action="store_false")
    # Encoder freeze
    p.add_argument("--freeze_encoder_epochs",  default=10,     type=int)
    p.add_argument("--encoder_warmup_epochs",  default=5,      type=int)
    p.add_argument("--lr_enc_peak",            default=5e-5,   type=float)
    # Eval
    p.add_argument("--val_freq",               default=5,      type=int)
    p.add_argument("--patience",               default=20,     type=int)
    p.add_argument("--min_ep",                 default=20,     type=int)
    p.add_argument("--hard_val_threshold",     default=0.35,   type=float)
    p.add_argument("--hard_val_freq",          default=10,     type=int)
    # SWA
    p.add_argument("--swa_lr",                 default=2e-6,   type=float)
    p.add_argument("--swa_window",             default=3,      type=int)
    p.add_argument("--swa_threshold",          default=1.5,    type=float)
    p.add_argument("--swa_min_ep",             default=50,     type=int)
    # Test
    p.add_argument("--tta_test",               default=True,   action="store_true")
    p.add_argument("--n_tta",                  default=5,      type=int)
    p.add_argument("--speed_tta_test",          default=True,   action="store_true",
                   help="[NEW v2.8] Speed TTA at test: 8 scales [0.70..1.50]")
    p.add_argument("--multiscale_test",        default=True,   action="store_true")
    # IO
    p.add_argument("--output_dir",             default="runs/fm_v26")
    p.add_argument("--gpu_num",                default="0")
    p.add_argument("--resume",                 default=None)
    p.add_argument("--test_at_end",            action="store_true", default=True)
    p.add_argument("--no_test",                dest="test_at_end", action="store_false")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt      = os.path.join(args.output_dir, "best_model.pth")
    hard_best_ckpt = os.path.join(args.output_dir, "hard_best_model.pth")
    swa_ckpt       = os.path.join(args.output_dir, "swa_model.pth")
    last_ckpt      = os.path.join(args.output_dir, "last_model.pth")

    print("=" * 72)
    print("  TC-FlowMatching v2.6")
    print(f"  [REMOVE]  L_momentum (was hurting test ATE by +7.9km)")
    print(f"  [UPGRADE] L_heading multi-step 4 steps decay=0.5 weight={args.lambda_heading}")
    print(f"  [NEW-INF] Speed calibration at inference (clip 0.85–1.15)")
    print(f"  [NEW-AUG] Obs-speed aug ×0.7–1.4 (15%)")
    print(f"  [UPGRADE] Physics score + displacement_score")
    print(f"  KEEP:     sigma_inf={args.sigma_inference} FIXED, L_reg linear weights")
    print("=" * 72)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n  Loading data...")
    trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
    vd, val_loader = data_loader(args, {"root": args.dataset_root, "type": "val"}, test=True)
    print(f"  train: {len(trd)} ({len(trl)} batches/ep)")
    print(f"  val:   {len(vd)} ({len(val_loader)} batches)")

    # Fixed XAI batch (same every epoch for comparability)
    try:
        xai_batch = move(list(next(iter(val_loader))), device)
    except Exception:
        xai_batch = None

    # ── Model ─────────────────────────────────────────────────────────────
    model = TCFlowMatching(
        pred_len=args.pred_len,        obs_len=args.obs_len,
        unet_in_ch=args.unet_in_ch,   d_cond=args.d_cond,
        d_model=args.d_model,          nhead=args.nhead,
        num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
        dropout=args.dropout,
        sigma_min=args.sigma_min,      sigma_max=args.sigma_max,
        lambda_reg=args.lambda_reg,    lambda_heading=args.lambda_heading,
        lambda_jerk=args.lambda_jerk,
        lambda_momentum=0.0,
        use_ot=args.use_ot,            ot_epsilon=args.ot_epsilon,
        use_ema=args.use_ema,          n_ensemble=args.n_ensemble,
        n_inference_steps=args.n_inference_steps,
        sigma_inference=args.sigma_inference,
    ).to(device)

    model.init_ema()
    ema = getattr(_unwrap(model), "_ema", None)
    raw = _unwrap(model)
    n_enc = sum(p.numel() for p in raw.encoder.parameters())
    n_vel = sum(p.numel() for p in raw.velocity.parameters())
    print(f"\n  Encoder: {n_enc:,}  VelocityTrans: {n_vel:,}  Total: {n_enc+n_vel:,}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    opt    = build_optimizer(model, lr_velocity=args.lr, lr_encoder=0.0,
                             weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=args.use_amp)
    sched  = TwoGroupScheduler(
        opt=opt, warmup_epochs=args.warmup_epochs, total_epochs=args.num_epochs,
        lr_vel=args.lr, lr_vel_min=args.lr_min,
        freeze_end_ep=args.freeze_encoder_epochs, lr_enc_peak=args.lr_enc_peak,
        encoder_warmup_epochs=args.encoder_warmup_epochs)
    print(f"\n  LR vel: {args.lr:.0e} → {args.lr_min:.0e}  "
          f"LR enc: 0 ({args.freeze_encoder_epochs}ep) → {args.lr_enc_peak:.0e}")

    swa = SWAHandler(swa_lr=args.swa_lr)

    # ── Resume ────────────────────────────────────────────────────────────
    start_ep = 0; best_score = float("inf"); best_hard = float("inf")
    patience_cnt = 0; val_ade_history = []

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        _unwrap(model).load_state_dict(ck["model"], strict=False)
        try: opt.load_state_dict(ck["optimizer"])
        except Exception as e: print(f"  ⚠ Opt: {e}")
        sched.epoch  = ck.get("scheduler", 0)
        start_ep     = ck.get("epoch", 0) + 1
        best_score   = ck.get("best_score", ck.get("best_ade", float("inf")))
        patience_cnt = ck.get("patience_cnt", 0)
        if scaler and ck.get("scaler"):
            try: scaler.load_state_dict(ck["scaler"])
            except Exception: pass
        if ema and ck.get("ema"):
            for k, v in ck["ema"].items():
                if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
        print(f"  ↩ Resume ep{start_ep}  best={best_score:.1f}  patience={patience_cnt}")

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: ok")
    except Exception: pass

    # ── Training loop ─────────────────────────────────────────────────────
    nstep = len(trl)
    print(f"\n  TRAINING ({nstep} steps/ep × {args.num_epochs} ep)")
    print(f"  Aug: shift±5km + speed×[0.85,1.15] + recurv±20° + obs-speed×[0.7,1.4] + noise")
    print(f"  Loss: L_CFM + L_reg(linear) + L_heading_ms(4steps,decay=0.5)")
    print(f"  Inf:  1-shot sigma=0.04 + speed_calibrate(±15%) + top3 physics")
    print()

    for ep in range(start_ep, start_ep + args.num_epochs):
        rel_ep = ep - start_ep
        freeze = rel_ep < args.freeze_encoder_epochs
        for p in _unwrap(model).encoder.parameters():
            p.requires_grad_(not freeze)

        if rel_ep == 0 and freeze:
            print(f"  *** Ep{ep}: encoder frozen ***")
        if rel_ep == args.freeze_encoder_epochs:
            print(f"\n  *** Ep{ep}: encoder unfrozen ***")

        # step_weights only train when L_reg is active (ep>=10)
        # Already handled by lam_reg ramp — no explicit freeze needed

        model.train()
        sum_loss = sum_cfm = sum_reg = sum_head = sum_jerk = sum_ade1 = 0.0
        t0_ep = time.perf_counter()

        for i, batch in enumerate(trl):
            bl = move(list(batch), device)
            bl_aug = augment_batch(bl)   # training augmentation

            opt.zero_grad()
            with autocast(device_type="cuda", enabled=args.use_amp):
                bd = model.get_loss_breakdown(bl_aug, epoch=ep)

            scaler.scale(bd["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if freeze:
                for p in _unwrap(model).encoder.parameters():
                    if p.grad is not None: p.grad.zero_()

            scaler.step(opt); scaler.update()
            model.ema_update()
            swa.update(model)

            sum_loss += bd["total"].item(); sum_cfm  += bd["l_cfm"]
            sum_reg  += bd["l_reg"];        sum_head += bd["l_heading"]
            sum_ade1 += bd["ade_1step"]

            if i % 30 == 0:
                _, lr_vel = get_lrs(opt)
                enc_s = "frozen" if freeze else "active"
                swa_s = " [SWA]" if swa.active else ""
                print(f"  [{ep:>3}][{i:>3}/{nstep}]"
                      f"  tot={bd['total'].item():.4f}"
                      f"  cfm={bd['l_cfm']:.4f}"
                      f"  reg={bd['l_reg']:.4f}"
                      f"  h4s={bd['l_heading']:.4f}  jrk={bd.get('l_jerk',0):.4f}"
                      f"  lam_d={bd['lam_dir']:.2f}"
                      f"  ade1={bd['ade_1step']:.0f}km"
                      f"  enc={enc_s}{swa_s}"
                      f"  lr={lr_vel:.2e}")

        train_loss = sum_loss / nstep
        _, lr_vel_used = get_lrs(opt)
        sched.step()

        print(f"\n  ── Ep{ep:>3}"
              f"  train={train_loss:.6f}"
              f"  cfm={sum_cfm/nstep:.4f}"
              f"  reg={sum_reg/nstep:.4f}"
              f"  h4s={sum_head/nstep:.4f}"
              f"  ade1={sum_ade1/nstep:.0f}km"
              f"  lr={lr_vel_used:.2e}"
              f"  t={time.perf_counter()-t0_ep:.0f}s")

        _save(last_ckpt, ep, model, opt, sched, best_score, ema, scaler)
        if ep % 5 == 0:
            ep_ckpt = os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth")
            _save(ep_ckpt, ep, model, opt, sched, best_score, ema, scaler)
            print(f"  💾 {ep_ckpt}")

        # ── Val evaluation ───────────────────────────────────────────────
        if rel_ep % args.val_freq == 0:
            run_xai_this = (rel_ep % 10 == 0)
            r = evaluate(model, val_loader, device, tag=f"VAL ep{ep}",
                         n_ensemble=args.n_ensemble, ema=ema,
                         ref_targets=ST_TRANS_VAL, epoch_for_loss=ep,
                         run_xai=run_xai_this, xai_batch=xai_batch)

            val_ade = r["ADE"]; score = r["combined_score"]
            val_ade_history.append(val_ade)

            # Trend
            if len(val_ade_history) >= 4:
                trend = float(np.mean(val_ade_history[-2:])) - float(np.mean(val_ade_history[-4:-2]))
                trend_s = f"↑{trend:+.1f}km⚠" if trend > 5 else f"↓{trend:+.1f}km✓" if trend < -5 else f"→{trend:+.1f}flat"
            else:
                trend_s = "—"
            print(f"  train={train_loss:.6f}  val_ADE={val_ade:.1f}  combined={score:.1f}  trend={trend_s}")

            # SWA activation check
            if (not swa.active and ep >= args.swa_min_ep
                    and swa.should_activate(val_ade_history, args.swa_window, args.swa_threshold)):
                swa.activate(model, opt, ep)

            # Best checkpoint
            if score < best_score:
                best_score = score; patience_cnt = 0
                _save(best_ckpt, ep, model, opt, sched, best_score, ema, scaler,
                      extra={"val_ade": r["ADE"], "val_ate": r["ATE"],
                             "val_cte": r["CTE"], "patience_cnt": 0})
                print(f"  ✅ Best! score={best_score:.2f}"
                      f"  ADE={r['ADE']:.1f} ATE={r['ATE']:.1f} CTE={r['CTE']:.1f}")
            else:
                if rel_ep >= args.min_ep: patience_cnt += args.val_freq
                print(f"  No improve {patience_cnt}/{args.patience} (best={best_score:.1f})")
                if rel_ep >= args.min_ep and patience_cnt >= args.patience:
                    print(f"  ⛔ Early stop @ ep{ep}")
                    break

        # ── Hard-val evaluation ──────────────────────────────────────────
        if rel_ep % args.hard_val_freq == 0 and rel_ep >= args.min_ep:
            r_h = evaluate_hard_val(model, val_loader, device,
                                     hard_threshold=args.hard_val_threshold,
                                     n_ensemble=args.n_ensemble, ema=ema,
                                     epoch_for_loss=ep)
            print(f"  [HVAL] n={r_h['n_hard']}"
                  f"  ADE={r_h['ADE']:.1f} ATE={r_h['ATE']:.1f} CTE={r_h['CTE']:.1f}"
                  f"  combined={r_h['combined_score']:.1f}")
            if r_h["combined_score"] < best_hard and r_h["n_hard"] >= 10:
                best_hard = r_h["combined_score"]
                _save(hard_best_ckpt, ep, model, opt, sched, best_hard, ema, scaler,
                      extra={"hard_val_ade": r_h["ADE"], "selection_criterion": "hard_val"})
                print(f"  💎 Hard-best! score={best_hard:.2f} ADE={r_h['ADE']:.1f}")

        # ── SWA eval ─────────────────────────────────────────────────────
        if swa.active and rel_ep % args.val_freq == 0 and swa.n_updates >= 10:
            backup = {k: v.detach().clone()
                      for k, v in _unwrap(model).state_dict().items()
                      if v.dtype.is_floating_point}
            swa.apply_to_model(model)
            r_swa = evaluate(model, val_loader, device, tag=f"SWA ep{ep}",
                              n_ensemble=args.n_ensemble, ema=None,
                              ref_targets=ST_TRANS_VAL, epoch_for_loss=ep)
            swa.restore_from_backup(model, backup)

            swa_score = r_swa["combined_score"]
            print(f"  [SWA] score={swa_score:.2f} ({swa.n_updates} updates) vs best={best_score:.2f}")
            if swa_score < best_score:
                best_score = swa_score; patience_cnt = 0
                swa.save_avg_state(swa_ckpt, ep, best_score,
                                   extra={"val_ade": r_swa["ADE"],
                                          "val_ate": r_swa["ATE"],
                                          "val_cte": r_swa["CTE"]})
                import shutil; shutil.copy(swa_ckpt, best_ckpt)
                print(f"  ✅ SWA best! score={best_score:.2f} ADE={r_swa['ADE']:.1f}")

    # ── Test evaluation ───────────────────────────────────────────────────
    print(f"\n  Done! best_score={best_score:.2f}")
    if not args.test_at_end: return

    print("\n  Loading best checkpoint for TEST...")
    if not os.path.exists(best_ckpt):
        print("  No checkpoint found."); return

    ck = torch.load(best_ckpt, map_location=device)
    is_swa = ck.get("is_swa", False)
    _unwrap(model).load_state_dict(ck["model"], strict=False)
    if not is_swa and ema and ck.get("ema"):
        for k, v in ck["ema"].items():
            if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
    print(f"  Loaded ep{ck.get('epoch','?')} (is_swa={is_swa})")

    try:
        _, test_loader = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
        print(f"  Test: {len(test_loader)} batches")
    except Exception:
        print("  No test set → using val"); test_loader = val_loader

    # Standard test
    r_test = evaluate(model, test_loader, device, tag="TEST (best ckpt)",
                      n_ensemble=args.n_ensemble, ema=None if is_swa else ema,
                      ref_targets=ST_TRANS_TEST, run_xai=True, xai_batch=xai_batch)

    # TTA test
    if args.tta_test:
        r_tta = evaluate(model, test_loader, device, tag="TEST+TTA",
                         n_ensemble=args.n_ensemble, ema=None if is_swa else ema,
                         ref_targets=ST_TRANS_TEST, use_tta=True, n_tta=args.n_tta)
        print(f"\n  TTA: ADE {r_test['ADE']:.1f}→{r_tta['ADE']:.1f}  "
              f"ATE {r_test['ATE']:.1f}→{r_tta['ATE']:.1f}  "
              f"CTE {r_test['CTE']:.1f}→{r_tta['CTE']:.1f}")

    # Multi-scale test
    if args.multiscale_test:
        raw_m = _unwrap(model)
        if hasattr(raw_m, "sample_multiscale"):
            print("\n  Multi-scale sigma test...")
            ms_ades, ms_ates, ms_ctes = [], [], []
            ms_steps = defaultdict(list)
            raw_m.eval()
            bk_ms = (ema.apply_to(model) if (ema and not is_swa) else None)
            with torch.no_grad():
                for batch in test_loader:
                    bl = move(list(batch), device)
                    try:
                        pred, _, _ = raw_m.sample_multiscale(bl)
                    except Exception: continue
                    gt = bl[1]; T = min(pred.shape[0], gt.shape[0])
                    pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
                    dist = _haversine_deg(pd, gd)
                    ate, cte = _ate_cte(pd, gd)
                    ms_ades.extend(dist.mean(0).tolist())
                    if ate.shape[0] > 0:
                        ms_ates.extend(ate.abs().mean(0).tolist())
                        ms_ctes.extend(cte.abs().mean(0).tolist())
                    for h, s in HORIZON_STEPS.items():
                        if s < T: ms_steps[h].extend(dist[s].tolist())
            if bk_ms: ema.restore(model, bk_ms)
            def _mm(lst): return float(np.mean(lst)) if lst else float("nan")
            print(f"  Multi-scale: ADE={_mm(ms_ades):.1f} ATE={_mm(ms_ates):.1f} CTE={_mm(ms_ctes):.1f}")

    # Speed TTA test
    if args.speed_tta_test and hasattr(_unwrap(model), "sample_speed_tta"):
        print("\n  Speed TTA test (8 scales [0.70..1.50])...")
        stta_ades, stta_ates, stta_ctes = [], [], []
        raw_m.eval()
        bk_stta = (ema.apply_to(model) if (ema and not is_swa) else None)
        with torch.no_grad():
            for batch in test_loader:
                bl = move(list(batch), device)
                try:
                    pred, _, _ = raw_m.sample_speed_tta(bl)
                except Exception as e:
                    print(f"  speed_tta error: {e}"); continue
                gt = bl[1]; T = min(pred.shape[0], gt.shape[0])
                pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
                dist = _haversine_deg(pd, gd)
                ate, cte = _ate_cte(pd, gd)
                stta_ades.extend(dist.mean(0).tolist())
                if ate.shape[0] > 0:
                    stta_ates.extend(ate.abs().mean(0).tolist())
                    stta_ctes.extend(cte.abs().mean(0).tolist())
        if bk_stta: ema.restore(model, bk_stta)
        def _smm(lst): return float(np.mean(lst)) if lst else float("nan")
        print(f"  Speed TTA: ADE={_smm(stta_ades):.1f} ATE={_smm(stta_ates):.1f} CTE={_smm(stta_ctes):.1f}")
        print(f"  vs Standard: ADE={r_test['ADE']:.1f} ATE={r_test['ATE']:.1f} CTE={r_test['CTE']:.1f}")

    # Final comparison
    val_ade_b = ck.get("val_ade", float("nan"))
    val_ate_b = ck.get("val_ate", float("nan"))
    val_cte_b = ck.get("val_cte", float("nan"))
    v21 = {"ADE": 229.8, "ATE": 214.4, "CTE": 71.6}
    print("\n" + "=" * 72)
    print("  v2.6 FINAL RESULTS vs v2.1 baseline")
    print("=" * 72)
    print(f"  {'Metric':<10} {'Val':>10} {'Test':>10} {'Gap':>10} {'v2.1 Test':>12} Status")
    print("  " + "─"*60)
    for m_n, val_v, test_v, v21_v in [
        ("ADE", val_ade_b, r_test["ADE"], v21["ADE"]),
        ("ATE", val_ate_b, r_test["ATE"], v21["ATE"]),
        ("CTE", val_cte_b, r_test["CTE"], v21["CTE"]),
    ]:
        gap  = test_v - val_v if (np.isfinite(test_v) and np.isfinite(val_v)) else float("nan")
        impr = v21_v - test_v
        flag = (f"↓{impr:+.1f}" if np.isfinite(impr) and impr > 0 else
                f"↑{-impr:+.1f}" if np.isfinite(impr) else "?")
        print(f"  {m_n:<10} {val_v:>10.2f} {test_v:>10.2f} {gap:>+10.2f} {v21_v:>12.1f}  {flag}")
    print("=" * 72)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    if args.dataset_root == "TCND_vn":
        _auto = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
        if os.path.isdir(_auto):
            args.dataset_root = _auto
    main(args)
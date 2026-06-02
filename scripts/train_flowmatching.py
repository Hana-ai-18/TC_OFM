# # # # # # # # # # # """
# # # # # # # # # # # scripts/train_flowmatching.py  ── FM v60 Training Entry Point
# # # # # # # # # # # ==============================================================
# # # # # # # # # # # THAY THẾ scripts/train_flowmatching.py cũ.
# # # # # # # # # # # Cùng interface, chạy được trên Kaggle với lệnh:

# # # # # # # # # # #   !python scripts/train_flowmatching.py \
# # # # # # # # # # #       --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# # # # # # # # # # #       --output_dir   /kaggle/working/runs/fm_v60 \
# # # # # # # # # # #       --batch_size 32 --num_epochs 70 --learning_rate 2e-4 --use_amp
# # # # # # # # # # # """
# # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # import argparse
# # # # # # # # # # # import json
# # # # # # # # # # # import os
# # # # # # # # # # # import random
# # # # # # # # # # # import sys
# # # # # # # # # # # import time
# # # # # # # # # # # from collections import defaultdict
# # # # # # # # # # # from typing import Dict, Optional

# # # # # # # # # # # import numpy as np
# # # # # # # # # # # import torch
# # # # # # # # # # # import torch.nn as nn
# # # # # # # # # # # from torch.optim import AdamW
# # # # # # # # # # # from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# # # # # # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # # # # # # ── Path setup ────────────────────────────────────────────────
# # # # # # # # # # # # __file__ = project_root/scripts/train_flowmatching.py
# # # # # # # # # # # # _root    = project_root/          ← chứa Model/ và scripts/
# # # # # # # # # # # _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # # # # # # # # # # if _root not in sys.path:
# # # # # # # # # # #     sys.path.insert(0, _root)

# # # # # # # # # # # # ── Imports ───────────────────────────────────────────────────
# # # # # # # # # # # from Model.flow_matching_model import (
# # # # # # # # # # #     TCFlowMatching, compute_speed_stats_from_norm,
# # # # # # # # # # #     _norm_to_deg, haversine_km
# # # # # # # # # # # )

# # # # # # # # # # # try:
# # # # # # # # # # #     from Model.data.loader_training import data_loader as tc_data_loader
# # # # # # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # # # # # #     _LOADER_OK = True
# # # # # # # # # # # except ImportError as e:
# # # # # # # # # # #     print(f'[WARN] Data loader import failed: {e}')
# # # # # # # # # # #     _LOADER_OK = False

# # # # # # # # # # # ST_TRANS = {
# # # # # # # # # # #     'ADE': 224.4, 'ATE': 213.7, 'CTE': 59.4,
# # # # # # # # # # #     '12h': 77.5,  '24h': 130.5, '48h': 269.9, '72h': 423.3,
# # # # # # # # # # # }
# # # # # # # # # # # FM_V59 = {'ADE': 236.1, 'ATE': 223.6, 'CTE': 74.9, '72h': 471.6}


# # # # # # # # # # # # ══════════════════════════════════════════════════════════════
# # # # # # # # # # # #  Utilities
# # # # # # # # # # # # ══════════════════════════════════════════════════════════════

# # # # # # # # # # # def set_seed(s):
# # # # # # # # # # #     random.seed(s); np.random.seed(s); torch.manual_seed(s)
# # # # # # # # # # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


# # # # # # # # # # # def count_params(m):
# # # # # # # # # # #     return sum(p.numel() for p in m.parameters() if p.requires_grad)


# # # # # # # # # # # def _move(batch_list, device):
# # # # # # # # # # #     out = []
# # # # # # # # # # #     for item in batch_list:
# # # # # # # # # # #         if isinstance(item, torch.Tensor):
# # # # # # # # # # #             out.append(item.to(device))
# # # # # # # # # # #         elif isinstance(item, dict):
# # # # # # # # # # #             out.append({k: v.to(device) if isinstance(v, torch.Tensor) else v
# # # # # # # # # # #                         for k, v in item.items()})
# # # # # # # # # # #         else:
# # # # # # # # # # #             out.append(item)
# # # # # # # # # # #     return out


# # # # # # # # # # # # ══════════════════════════════════════════════════════════════
# # # # # # # # # # # #  Metrics
# # # # # # # # # # # # ══════════════════════════════════════════════════════════════

# # # # # # # # # # # def compute_metrics(pred_deg, gt_deg):
# # # # # # # # # # #     """[N,T,2] lon°,lat° → dict"""
# # # # # # # # # # #     N, T, _ = pred_deg.shape
# # # # # # # # # # #     dist = haversine_km(
# # # # # # # # # # #         pred_deg.reshape(N*T, 2), gt_deg.reshape(N*T, 2)
# # # # # # # # # # #     ).reshape(N, T)

# # # # # # # # # # #     out = {'ADE': float(dist.mean()), 'FDE': float(dist[:,-1].mean())}
# # # # # # # # # # #     for step, name in {2:'12h', 4:'24h', 8:'48h', 12:'72h'}.items():
# # # # # # # # # # #         if step-1 < T: out[name] = float(dist[:, step-1].mean())

# # # # # # # # # # #     # ATE / CTE
# # # # # # # # # # #     cos_lat = torch.cos(torch.deg2rad(gt_deg[...,1])).clamp(1e-3)
# # # # # # # # # # #     elon = (pred_deg[...,0]-gt_deg[...,0])*cos_lat*111.
# # # # # # # # # # #     elat = (pred_deg[...,1]-gt_deg[...,1])*111.
# # # # # # # # # # #     if T >= 2:
# # # # # # # # # # #         dlon = (gt_deg[:,1:,0]-gt_deg[:,:-1,0])*cos_lat[:,1:]*111.
# # # # # # # # # # #         dlat = (gt_deg[:,1:,1]-gt_deg[:,:-1,1])*111.
# # # # # # # # # # #         mag  = (dlon**2+dlat**2).sqrt().clamp(1e-3)
# # # # # # # # # # #         ux = torch.cat([dlon[:,:1]/mag[:,:1], dlon/mag], 1)[:,:T]
# # # # # # # # # # #         uy = torch.cat([dlat[:,:1]/mag[:,:1], dlat/mag], 1)[:,:T]
# # # # # # # # # # #         out['ATE'] = float((elon*ux+elat*uy).abs().mean())
# # # # # # # # # # #         out['CTE'] = float((elon*uy-elat*ux).abs().mean())
# # # # # # # # # # #     else:
# # # # # # # # # # #         out['ATE'] = out['CTE'] = float(dist.mean())

# # # # # # # # # # #     # Speed bias
# # # # # # # # # # #     if T >= 2:
# # # # # # # # # # #         def _s(t):
# # # # # # # # # # #             dl=t[:,1:,1]-t[:,:-1,1]; dlo=t[:,1:,0]-t[:,:-1,0]
# # # # # # # # # # #             cl=torch.cos(torch.deg2rad((t[:,1:,1]+t[:,:-1,1])/2)).clamp(1e-3)
# # # # # # # # # # #             return float((dl*111)**2+(dlo*111*cl)**2).sqrt().mean() if False \
# # # # # # # # # # #                    else float(torch.sqrt((dl*111)**2+(dlo*111*cl)**2).mean())
# # # # # # # # # # #         out['speed_bias'] = _s(pred_deg) - _s(gt_deg)
# # # # # # # # # # #         out['mean_pred_speed'] = _s(pred_deg)
# # # # # # # # # # #         out['mean_gt_speed']   = _s(gt_deg)
# # # # # # # # # # #     return out


# # # # # # # # # # # def print_metrics(metrics, tag='', elapsed=0.):
# # # # # # # # # # #     print(f'\n{"="*70}')
# # # # # # # # # # #     if tag: print(f'  [{tag}  {elapsed:.0f}s]')
# # # # # # # # # # #     for k in ['ADE','12h','24h','48h','72h']:
# # # # # # # # # # #         v = metrics.get(k, float('nan'))
# # # # # # # # # # #         ref = ST_TRANS.get(k, float('nan'))
# # # # # # # # # # #         ok  = '✅ BEAT' if v < ref else '❌'
# # # # # # # # # # #         print(f'  {k:>4} = {v:>7.1f} km  {ok} ST-Trans={ref:.1f}  ({v-ref:+.1f})')
# # # # # # # # # # #     if 'ATE' in metrics:
# # # # # # # # # # #         print(f'  ATE  = {metrics["ATE"]:>7.1f}  '
# # # # # # # # # # #               f'{"✅ BEAT" if metrics["ATE"]<ST_TRANS["ATE"] else "❌"}')
# # # # # # # # # # #         print(f'  CTE  = {metrics["CTE"]:>7.1f}  '
# # # # # # # # # # #               f'{"✅ BEAT" if metrics["CTE"]<ST_TRANS["CTE"] else "❌"}')
# # # # # # # # # # #     if 'speed_bias' in metrics:
# # # # # # # # # # #         sb = metrics['speed_bias']
# # # # # # # # # # #         print(f'  speed_bias={sb:+.2f} km/6h  '
# # # # # # # # # # #               f'{"✅" if abs(sb)<10 else "❌"}  '
# # # # # # # # # # #               f'pred={metrics["mean_pred_speed"]:.1f}  '
# # # # # # # # # # #               f'gt={metrics["mean_gt_speed"]:.1f}')
# # # # # # # # # # #     print(f'{"="*70}\n')


# # # # # # # # # # # # ══════════════════════════════════════════════════════════════
# # # # # # # # # # # #  Evaluate
# # # # # # # # # # # # ══════════════════════════════════════════════════════════════

# # # # # # # # # # # @torch.no_grad()
# # # # # # # # # # # def evaluate(model, loader, device, tag='', num_ens=20, steps=10):
# # # # # # # # # # #     model.eval(); t0 = time.perf_counter()
# # # # # # # # # # #     all_p, all_g = [], []
# # # # # # # # # # #     for bl in loader:
# # # # # # # # # # #         bl = _move(bl, device)
# # # # # # # # # # #         try:
# # # # # # # # # # #             pred_n, _ = model.sample(bl, num_ensemble=num_ens, ddim_steps=steps)
# # # # # # # # # # #         except Exception as e:
# # # # # # # # # # #             print(f'  [WARN] sample failed: {e}'); continue
# # # # # # # # # # #         gt_n = bl[1]
# # # # # # # # # # #         T = min(pred_n.shape[0], gt_n.shape[0])
# # # # # # # # # # #         all_p.append(_norm_to_deg(pred_n[:T].permute(1,0,2)).cpu())
# # # # # # # # # # #         all_g.append(_norm_to_deg(gt_n[:T].permute(1,0,2)).cpu())
# # # # # # # # # # #     if not all_p: return {}
# # # # # # # # # # #     m = compute_metrics(torch.cat(all_p,0), torch.cat(all_g,0))
# # # # # # # # # # #     print_metrics(m, tag, time.perf_counter()-t0)
# # # # # # # # # # #     return m


# # # # # # # # # # # # ══════════════════════════════════════════════════════════════
# # # # # # # # # # # #  Checkpoint
# # # # # # # # # # # # ══════════════════════════════════════════════════════════════

# # # # # # # # # # # def save_ckpt(path, epoch, model, opt, sched, metrics):
# # # # # # # # # # #     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
# # # # # # # # # # #     torch.save({'epoch': epoch, 'model_state': model.state_dict(),
# # # # # # # # # # #                 'optim_state': opt.state_dict(),
# # # # # # # # # # #                 'sched_state': sched.state_dict() if sched else None,
# # # # # # # # # # #                 'metrics': metrics}, path)


# # # # # # # # # # # def load_ckpt(path, model, opt=None, sched=None):
# # # # # # # # # # #     ck = torch.load(path, map_location='cpu')
# # # # # # # # # # #     model.load_state_dict(ck['model_state'], strict=False)
# # # # # # # # # # #     if opt and 'optim_state' in ck:
# # # # # # # # # # #         try: opt.load_state_dict(ck['optim_state'])
# # # # # # # # # # #         except Exception as e: print(f'  [warn] opt state: {e}')
# # # # # # # # # # #     if sched and ck.get('sched_state'):
# # # # # # # # # # #         try: sched.load_state_dict(ck['sched_state'])
# # # # # # # # # # #         except: pass
# # # # # # # # # # #     return ck.get('epoch', 0), ck.get('metrics', {})


# # # # # # # # # # # class BestSaver:
# # # # # # # # # # #     def __init__(self, d, patience=15, min_ep=20):
# # # # # # # # # # #         self.d=d; self.patience=patience; self.min_ep=min_ep
# # # # # # # # # # #         self.best={k:float('inf') for k in ['ADE','72h','ATE','CTE']}
# # # # # # # # # # #         self.no_imp=0; self.stop=False; os.makedirs(d, exist_ok=True)

# # # # # # # # # # #     def update(self, m, ep, model, opt, sched):
# # # # # # # # # # #         imp = False
# # # # # # # # # # #         for k,fn in [('ADE','best_ade.pth'),('72h','best_72h.pth'),
# # # # # # # # # # #                      ('ATE','best_ate.pth'),('CTE','best_cte.pth')]:
# # # # # # # # # # #             v = m.get(k, float('inf'))
# # # # # # # # # # #             if v < self.best[k]:
# # # # # # # # # # #                 self.best[k]=v
# # # # # # # # # # #                 save_ckpt(os.path.join(self.d,fn), ep, model, opt, sched, m)
# # # # # # # # # # #                 print(f'  [BEST {k}] ep={ep}  '
# # # # # # # # # # #                       f'ADE={m.get("ADE",0):.1f}  '
# # # # # # # # # # #                       f'CTE={m.get("CTE",0):.1f}  '
# # # # # # # # # # #                       f'72h={m.get("72h",0):.0f}')
# # # # # # # # # # #                 imp=True
# # # # # # # # # # #         if imp: self.no_imp=0
# # # # # # # # # # #         else:
# # # # # # # # # # #             self.no_imp+=1
# # # # # # # # # # #             if ep>=self.min_ep and self.no_imp>=self.patience:
# # # # # # # # # # #                 self.stop=True; print(f'  [EARLY STOP] ep={ep}')
# # # # # # # # # # #         return imp


# # # # # # # # # # # # ══════════════════════════════════════════════════════════════
# # # # # # # # # # # #  Data loaders
# # # # # # # # # # # # ══════════════════════════════════════════════════════════════

# # # # # # # # # # # class _Args:
# # # # # # # # # # #     def __init__(self, **kw):
# # # # # # # # # # #         for k,v in kw.items(): setattr(self, k, v)


# # # # # # # # # # # def build_loaders(data_root, batch_size, num_workers):
# # # # # # # # # # #     if not _LOADER_OK:
# # # # # # # # # # #         raise RuntimeError('Data loader not available. Check Model/data/.')
# # # # # # # # # # #     dummy = _Args(obs_len=8, pred_len=12, skip=1, threshold=0.002,
# # # # # # # # # # #                   min_ped=1, delim=' ', other_modal='gph',
# # # # # # # # # # #                   batch_size=batch_size, num_workers=num_workers)
# # # # # # # # # # #     tr_ds, tr_ld = tc_data_loader(dummy, {'root':data_root,'type':'train'},
# # # # # # # # # # #                                    test=False, batch_size=batch_size)
# # # # # # # # # # #     vl_ds, vl_ld = tc_data_loader(dummy, {'root':data_root,'type':'val'},
# # # # # # # # # # #                                    test=True, batch_size=batch_size)
# # # # # # # # # # #     return tr_ds, tr_ld, vl_ds, vl_ld


# # # # # # # # # # # # ══════════════════════════════════════════════════════════════
# # # # # # # # # # # #  Train epoch
# # # # # # # # # # # # ══════════════════════════════════════════════════════════════

# # # # # # # # # # # def train_epoch(model, loader, optimizer, epoch, device, scaler=None):
# # # # # # # # # # #     model.train(); tot=0.; terms=defaultdict(float); n=0
# # # # # # # # # # #     t0=time.perf_counter()

# # # # # # # # # # #     for i, bl in enumerate(loader):
# # # # # # # # # # #         bl = _move(bl, device)
# # # # # # # # # # #         optimizer.zero_grad()

# # # # # # # # # # #         if scaler:
# # # # # # # # # # #             with torch.amp.autocast('cuda'):
# # # # # # # # # # #                 bd = model.get_loss_breakdown(bl, epoch=epoch)
# # # # # # # # # # #             loss = bd['total']
# # # # # # # # # # #             scaler.scale(loss).backward()
# # # # # # # # # # #             scaler.unscale_(optimizer)
# # # # # # # # # # #             nn.utils.clip_grad_norm_(model.parameters(), 5.)
# # # # # # # # # # #             scaler.step(optimizer); scaler.update()
# # # # # # # # # # #         else:
# # # # # # # # # # #             bd = model.get_loss_breakdown(bl, epoch=epoch)
# # # # # # # # # # #             loss = bd['total']
# # # # # # # # # # #             if torch.is_tensor(loss) and loss.requires_grad:
# # # # # # # # # # #                 loss.backward()
# # # # # # # # # # #                 nn.utils.clip_grad_norm_(model.parameters(), 5.)
# # # # # # # # # # #                 optimizer.step()

# # # # # # # # # # #         if not (torch.is_tensor(loss) and torch.isfinite(loss)):
# # # # # # # # # # #             if i < 5: print(f'  [WARN] non-finite loss at batch {i}')
# # # # # # # # # # #             continue

# # # # # # # # # # #         tot += loss.item(); n += 1
# # # # # # # # # # #         for k,v in bd.items():
# # # # # # # # # # #             if k != 'total' and not torch.is_tensor(v): terms[k] += float(v)

# # # # # # # # # # #         if i % 20 == 0:
# # # # # # # # # # #             lr = optimizer.param_groups[0]['lr']
# # # # # # # # # # #             print(f'  [{epoch:>3}][{i:>4}/{len(loader)}]'
# # # # # # # # # # #                   f' loss={loss.item():.3f}'
# # # # # # # # # # #                   f' l_kin={bd.get("l_kin",0):.3f}'
# # # # # # # # # # #                   f' l_logspd={bd.get("l_logspd",0):.3f}'
# # # # # # # # # # #                   f' l_fm={bd.get("l_fm",0):.3f}'
# # # # # # # # # # #                   f' l_scr={bd.get("l_scorer",0):.3f}'
# # # # # # # # # # #                   f' sw={bd.get("sw_ratio",0):.1f}'
# # # # # # # # # # #                   f' λk={bd.get("lam_kin",0):.3f}'
# # # # # # # # # # #                   f' λs={bd.get("lam_logspd",0):.3f}'
# # # # # # # # # # #                   f' dw={bd.get("diff_w_mean",1):.2f}'
# # # # # # # # # # #                   f' lr={lr:.1e}')

# # # # # # # # # # #     elapsed = time.perf_counter()-t0
# # # # # # # # # # #     avg = {k: v/max(n,1) for k,v in terms.items()}
# # # # # # # # # # #     avg['loss'] = tot/max(n,1); avg['elapsed'] = elapsed
# # # # # # # # # # #     return avg


# # # # # # # # # # # # ══════════════════════════════════════════════════════════════
# # # # # # # # # # # #  Main
# # # # # # # # # # # # ══════════════════════════════════════════════════════════════

# # # # # # # # # # # def train(data_root, save_dir, max_epochs=70, batch_size=32, lr=2e-4,
# # # # # # # # # # #           num_workers=0, seed=42, use_amp=False, resume=None, patience=15):
# # # # # # # # # # #     set_seed(seed)
# # # # # # # # # # #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # # # # # # # # # #     os.makedirs(save_dir, exist_ok=True)

# # # # # # # # # # #     print(f'\n{"="*72}')
# # # # # # # # # # #     print(f'  FM v60 — Physics-Grounded TC Track Forecasting')
# # # # # # # # # # #     print(f'  Device: {device}  AMP: {use_amp}')
# # # # # # # # # # #     print(f'  Baseline → FM_v59={FM_V59["ADE"]:.1f}  ST-Trans={ST_TRANS["ADE"]:.1f}')
# # # # # # # # # # #     print(f'  Targets  → ADE<170  CTE<52  72h<360')
# # # # # # # # # # #     print(f'{"="*72}\n')

# # # # # # # # # # #     tr_ds, tr_ld, vl_ds, vl_ld = build_loaders(data_root, batch_size, num_workers)
# # # # # # # # # # #     print(f'  train: {len(tr_ds)}  val: {len(vl_ds)}')

# # # # # # # # # # #     model = TCFlowMatching(
# # # # # # # # # # #         pred_len=12, obs_len=8, sigma_min=0.02, unet_in_ch=13,
# # # # # # # # # # #         ctx_noise_scale=0.01, use_ema=True, ema_decay=0.995,
# # # # # # # # # # #         use_ate_ot=True, ot_epsilon=0.05, cfg_uncond_prob=0.1,
# # # # # # # # # # #     ).to(device)
# # # # # # # # # # #     model.init_ema()
# # # # # # # # # # #     print(f'  Params: {count_params(model):,}')

# # # # # # # # # # #     # 3 param groups với LR khác nhau
# # # # # # # # # # #     model_p   = [p for n,p in model.named_parameters()
# # # # # # # # # # #                  if not n.startswith('criterion.')]
# # # # # # # # # # #     weight_p  = (list(model.criterion.step_weights.parameters())
# # # # # # # # # # #                + list(model.criterion.loss_weights.parameters())
# # # # # # # # # # #                + list(model.criterion.diff_weighter.parameters()))
# # # # # # # # # # #     scorer_p  = list(model.criterion.scorer.parameters())

# # # # # # # # # # #     optimizer = AdamW([
# # # # # # # # # # #         {'params': model_p,  'lr': lr,      'weight_decay': 1e-4},
# # # # # # # # # # #         {'params': weight_p, 'lr': lr*0.5,  'weight_decay': 0.},
# # # # # # # # # # #         {'params': scorer_p, 'lr': lr,      'weight_decay': 1e-5},
# # # # # # # # # # #     ])
# # # # # # # # # # #     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=35, T_mult=1, eta_min=1e-5)
# # # # # # # # # # #     scaler    = torch.amp.GradScaler('cuda') if torch.cuda.is_available() and use_amp else None

# # # # # # # # # # #     start_ep = 1
# # # # # # # # # # #     if resume and os.path.exists(resume):
# # # # # # # # # # #         start_ep, _ = load_ckpt(resume, model, optimizer, scheduler)
# # # # # # # # # # #         start_ep += 1
# # # # # # # # # # #         print(f'  Resumed from {resume} (ep {start_ep})')

# # # # # # # # # # #     saver   = BestSaver(save_dir, patience=patience, min_ep=20)
# # # # # # # # # # #     history = []

# # # # # # # # # # #     for epoch in range(start_ep, max_epochs+1):
# # # # # # # # # # #         phase = ('1-foundation' if epoch<=10 else
# # # # # # # # # # #                  '2-integration' if epoch<=35 else '3-refinement')
# # # # # # # # # # #         print(f'\n  ── Epoch {epoch}/{max_epochs}  Phase={phase} ──────────')

# # # # # # # # # # #         if epoch > 1: model.ema_update()

# # # # # # # # # # #         ts = train_epoch(model, tr_ld, optimizer, epoch, device, scaler)
# # # # # # # # # # #         scheduler.step()

# # # # # # # # # # #         sw = model.criterion.step_weights.stats()
# # # # # # # # # # #         lw = model.criterion.loss_weights.stats()
# # # # # # # # # # #         print(f'  train={ts["loss"]:.4f}'
# # # # # # # # # # #               f'  sw_ratio={sw["sw_ratio"]:.2f}'
# # # # # # # # # # #               f'  mono={sw["sw_monotonic"]}'
# # # # # # # # # # #               f'  λ_kin={lw["lam_kin"]:.3f}'
# # # # # # # # # # #               f'  λ_spd={lw["lam_logspd"]:.3f}'
# # # # # # # # # # #               f'  λ_curv={lw["lam_curv"]:.3f}')

# # # # # # # # # # #         # Fast val mỗi epoch
# # # # # # # # # # #         fast_n  = min(200, len(vl_ds))
# # # # # # # # # # #         fast_idx = random.sample(range(len(vl_ds)), fast_n)
# # # # # # # # # # #         fast_ld  = DataLoader(
# # # # # # # # # # #             Subset(vl_ds, fast_idx),
# # # # # # # # # # #             batch_size=batch_size, shuffle=False,
# # # # # # # # # # #             num_workers=0, drop_last=False,
# # # # # # # # # # #             collate_fn=seq_collate if _LOADER_OK else None)
# # # # # # # # # # #         fast_m = evaluate(model, fast_ld, device,
# # # # # # # # # # #                            tag=f'FAST ep{epoch} n={fast_n}',
# # # # # # # # # # #                            num_ens=10, steps=8)

# # # # # # # # # # #         # Full val mỗi 3 epochs
# # # # # # # # # # #         if epoch % 3 == 0 or epoch == max_epochs:
# # # # # # # # # # #             full_m = evaluate(model, vl_ld, device,
# # # # # # # # # # #                                tag=f'FULL ep{epoch}',
# # # # # # # # # # #                                num_ens=20, steps=10)
# # # # # # # # # # #             saver.update(full_m, epoch, model, optimizer, scheduler)
# # # # # # # # # # #             history.append({'epoch': epoch, 'train_loss': ts['loss'], **full_m})
# # # # # # # # # # #         else:
# # # # # # # # # # #             saver.update(fast_m, epoch, model, optimizer, scheduler)

# # # # # # # # # # #         if epoch % 10 == 0:
# # # # # # # # # # #             save_ckpt(os.path.join(save_dir, f'ckpt_ep{epoch:03d}.pth'),
# # # # # # # # # # #                        epoch, model, optimizer, scheduler, fast_m)

# # # # # # # # # # #         if saver.stop:
# # # # # # # # # # #             print(f'  Early stop ep={epoch}'); break

# # # # # # # # # # #     print(f'\n{"="*72}')
# # # # # # # # # # #     print(f'  DONE  Best ADE={saver.best["ADE"]:.1f}'
# # # # # # # # # # #           f'  CTE={saver.best["CTE"]:.1f}  72h={saver.best["72h"]:.1f}')
# # # # # # # # # # #     with open(os.path.join(save_dir, 'history.json'), 'w') as f:
# # # # # # # # # # #         json.dump(history, f, indent=2)
# # # # # # # # # # #     print(f'{"="*72}\n')
# # # # # # # # # # #     return saver.best['ADE']


# # # # # # # # # # # # ══════════════════════════════════════════════════════════════
# # # # # # # # # # # #  CLI
# # # # # # # # # # # # ══════════════════════════════════════════════════════════════

# # # # # # # # # # # def parse_args():
# # # # # # # # # # #     p = argparse.ArgumentParser('FM v60')
# # # # # # # # # # #     p.add_argument('--data_root',    default='TCND_vn')
# # # # # # # # # # #     p.add_argument('--save_dir',     default='checkpoints/fm_v60/')
# # # # # # # # # # #     p.add_argument('--max_epochs',   type=int,   default=70)
# # # # # # # # # # #     p.add_argument('--batch_size',   type=int,   default=32)
# # # # # # # # # # #     p.add_argument('--lr',           type=float, default=2e-4)
# # # # # # # # # # #     p.add_argument('--num_workers',  type=int,   default=0)
# # # # # # # # # # #     p.add_argument('--use_amp',      action='store_true')
# # # # # # # # # # #     p.add_argument('--resume',       default=None)
# # # # # # # # # # #     p.add_argument('--seed',         type=int,   default=42)
# # # # # # # # # # #     p.add_argument('--patience',     type=int,   default=15)
# # # # # # # # # # #     # Aliases
# # # # # # # # # # #     p.add_argument('--dataset_root', default=None)
# # # # # # # # # # #     p.add_argument('--output_dir',   default=None)
# # # # # # # # # # #     p.add_argument('--learning_rate',type=float, default=None)
# # # # # # # # # # #     p.add_argument('--num_epochs',   type=int,   default=None)
# # # # # # # # # # #     return p.parse_args()


# # # # # # # # # # # def _apply_aliases(args):
# # # # # # # # # # #     if getattr(args,'dataset_root',None) and (not getattr(args,'data_root',None) or args.data_root=='TCND_vn'):
# # # # # # # # # # #         args.data_root = args.dataset_root
# # # # # # # # # # #     if getattr(args,'output_dir',None) and (not getattr(args,'save_dir',None) or args.save_dir=='checkpoints/fm_v60/'):
# # # # # # # # # # #         args.save_dir = args.output_dir
# # # # # # # # # # #     if getattr(args,'learning_rate',None): args.lr = args.learning_rate
# # # # # # # # # # #     if getattr(args,'num_epochs',None):    args.max_epochs = args.num_epochs
# # # # # # # # # # #     return args


# # # # # # # # # # # if __name__ == '__main__':
# # # # # # # # # # #     args = parse_args(); args = _apply_aliases(args)
# # # # # # # # # # #     train(data_root=args.data_root, save_dir=args.save_dir,
# # # # # # # # # # #           max_epochs=args.max_epochs, batch_size=args.batch_size,
# # # # # # # # # # #           lr=args.lr, num_workers=args.num_workers, seed=args.seed,
# # # # # # # # # # #           use_amp=args.use_amp, resume=args.resume, patience=args.patience)

# # # # # # # # # # """
# # # # # # # # # # train_v74.py — TC-FlowMatching v74
# # # # # # # # # # ════════════════════════════════════════════════════════════════════════
# # # # # # # # # # TRAINING STRATEGY: Single phase, như v59 — không freeze, không transition

# # # # # # # # # # KEY DIFFERENCES vs failed versions:
# # # # # # # # # #   - V68/V72 fail: phase separation, best ep9 plateau
# # # # # # # # # #   - V69 fail: loss scale 700 (disp_smooth in km)
# # # # # # # # # #   - V74: single phase như v59, loss scale 1-4

# # # # # # # # # # MONITORING (xem để đảm bảo stable):
# # # # # # # # # #   sw_ratio >= 3.0          → step weights đang đúng hướng
# # # # # # # # # #   lw_dpe_fm ∈ [1.5, 3.0]  → DPE/FM balance ổn
# # # # # # # # # #   L_hard > L_easy          → hard storms được penalize nhiều hơn (correct)
# # # # # # # # # #   easy_frac ≈ 0.20-0.40    → delta score distribution hợp lý
# # # # # # # # # #   tot ∈ [1.0, 5.0]         → loss scale tốt (nếu tot > 10: có bug)

# # # # # # # # # # RUN:
# # # # # # # # # #   python scripts/train_v74.py \\
# # # # # # # # # #       --dataset_root /path/to/tc-ofm \\
# # # # # # # # # #       --output_dir   runs/v74 \\
# # # # # # # # # #       --batch_size   32 \\
# # # # # # # # # #       --use_amp
# # # # # # # # # # """
# # # # # # # # # # from __future__ import annotations

# # # # # # # # # # import sys, os
# # # # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # # # # import argparse, time, random
# # # # # # # # # # from collections import defaultdict

# # # # # # # # # # import numpy as np
# # # # # # # # # # import torch
# # # # # # # # # # import torch.optim as optim
# # # # # # # # # # from torch.amp import autocast, GradScaler
# # # # # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # # # # from Model.data.loader_training import data_loader
# # # # # # # # # # from Model.flow_matching_model import TCFlowMatching

# # # # # # # # # # try:
# # # # # # # # # #     from Model.utils import get_cosine_schedule_with_warmup
# # # # # # # # # # except ImportError:
# # # # # # # # # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # # # # # # # # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # # # # # # # # #         return CosineAnnealingLR(opt, T_max=max(total_steps,1), eta_min=min_lr)

# # # # # # # # # # TARGETS = {"ADE":172.68,"72h":321.39,"ATE":142.21,"CTE":42.04,
# # # # # # # # # #            "12h":65.42,"24h":104.67,"48h":205.10}
# # # # # # # # # # R_EARTH = 6371.0


# # # # # # # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # # # # # # def _unwrap(m): return m._orig_mod if hasattr(m,'_orig_mod') else m

# # # # # # # # # # def move(b, dev):
# # # # # # # # # #     out=list(b)
# # # # # # # # # #     for i,x in enumerate(out):
# # # # # # # # # #         if torch.is_tensor(x): out[i]=x.to(dev)
# # # # # # # # # #         elif isinstance(x,dict):
# # # # # # # # # #             out[i]={k:v.to(dev) if torch.is_tensor(v) else v for k,v in x.items()}
# # # # # # # # # #     return out


# # # # # # # # # # # ── Metric utils ──────────────────────────────────────────────────────────────

# # # # # # # # # # def _ntd(a):
# # # # # # # # # #     o=a.clone(); o[...,0]=(a[...,0]*50.+1800.)/10.; o[...,1]=(a[...,1]*50.)/10.; return o

# # # # # # # # # # def _hav(p1,p2):
# # # # # # # # # #     la1=torch.deg2rad(p1[...,1]); la2=torch.deg2rad(p2[...,1])
# # # # # # # # # #     dlat=torch.deg2rad(p2[...,1]-p1[...,1]); dlon=torch.deg2rad(p2[...,0]-p1[...,0])
# # # # # # # # # #     a=torch.sin(dlat/2).pow(2)+torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2)
# # # # # # # # # #     return 2.*R_EARTH*torch.asin(a.clamp(1e-12,1-1e-12).sqrt())

# # # # # # # # # # def _atecte(pd,gd):
# # # # # # # # # #     T=min(pd.shape[0],gd.shape[0])
# # # # # # # # # #     if T<2: z=pd.new_zeros(1,pd.shape[1]); return z,z
# # # # # # # # # #     lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
# # # # # # # # # #     lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
# # # # # # # # # #     lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
# # # # # # # # # #     ya=torch.sin(lo2-lo1)*torch.cos(la2)
# # # # # # # # # #     xa=torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # # # # # # # # #     be=torch.atan2(ya,xa)
# # # # # # # # # #     ye=torch.sin(lo3-lo2)*torch.cos(la3)
# # # # # # # # # #     xe=torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # # # # # # # # #     bee=torch.atan2(ye,xe); tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
# # # # # # # # # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # # # # # # # class Acc:
# # # # # # # # # #     def __init__(self):
# # # # # # # # # #         self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list)
# # # # # # # # # #         self._h={12:1,24:3,48:7,72:11}
# # # # # # # # # #     def update(self,dist,ate=None,cte=None):
# # # # # # # # # #         self.d.extend(dist.mean(0).tolist())
# # # # # # # # # #         for h,s in self._h.items():
# # # # # # # # # #             if s<dist.shape[0]: self.sd[h].extend(dist[s].tolist())
# # # # # # # # # #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# # # # # # # # # #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())
# # # # # # # # # #     def compute(self):
# # # # # # # # # #         r={"ADE":float(np.mean(self.d)) if self.d else float("nan"),
# # # # # # # # # #            "ATE_mean":float(np.mean(self.a)) if self.a else float("nan"),
# # # # # # # # # #            "CTE_mean":float(np.mean(self.c)) if self.c else float("nan"),
# # # # # # # # # #            "n":len(self.d)}
# # # # # # # # # #         for h in self._h:
# # # # # # # # # #             v=self.sd.get(h,[]); r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
# # # # # # # # # #         return r


# # # # # # # # # # def _score(r):
# # # # # # # # # #     ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
# # # # # # # # # #     ate=r.get("ATE_mean",1e9); cte=r.get("CTE_mean",1e9)
# # # # # # # # # #     if not np.isfinite(ate): ate=ade*.46
# # # # # # # # # #     if not np.isfinite(cte): cte=ade*.53
# # # # # # # # # #     return 100.*(0.05*(ade/136.)+0.10*(r.get("12h",ade)/50.)+
# # # # # # # # # #                  0.15*(r.get("24h",ade)/100.)+0.20*(r.get("48h",ade)/200.)+
# # # # # # # # # #                  0.25*(h72/300.)+0.13*(ate/80.)+0.12*(cte/94.))

# # # # # # # # # # def _beat(r):
# # # # # # # # # #     p=[]
# # # # # # # # # #     for k,t in [("ADE",172.68),("ATE_mean",142.21),("CTE_mean",42.04),
# # # # # # # # # #                   ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
# # # # # # # # # #         v=r.get(k,1e9)
# # # # # # # # # #         if np.isfinite(v) and v<t: p.append(f"{k.replace('_mean','')}:{v:.1f}")
# # # # # # # # # #     return "*** BEAT ST-TRANS: "+" ".join(p)+" ***" if p else ""

# # # # # # # # # # def _gap(r):
# # # # # # # # # #     out=[]
# # # # # # # # # #     for k,ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
# # # # # # # # # #         v=r.get(k,float("nan"))
# # # # # # # # # #         if np.isfinite(v):
# # # # # # # # # #             out.append(f"{k.replace('_mean','')}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
# # # # # # # # # #     return " | ".join(out)


# # # # # # # # # # # ── Evaluation ────────────────────────────────────────────────────────────────

# # # # # # # # # # @torch.no_grad()
# # # # # # # # # # def evaluate(model, loader, dev, tag="", ema=None, steps=20):
# # # # # # # # # #     bk=None
# # # # # # # # # #     if ema:
# # # # # # # # # #         try: bk=ema.apply_to(model)
# # # # # # # # # #         except: pass
# # # # # # # # # #     model.eval(); acc=Acc(); t0=time.perf_counter()
# # # # # # # # # #     for b in loader:
# # # # # # # # # #         bl=move(list(b),dev)
# # # # # # # # # #         result=model.sample(bl,ddim_steps=steps)
# # # # # # # # # #         p=result[0] if isinstance(result,(tuple,list)) else result
# # # # # # # # # #         g=bl[1]; T=min(p.shape[0],g.shape[0])
# # # # # # # # # #         pd=_ntd(p[:T]); gd=_ntd(g[:T])
# # # # # # # # # #         dist=_hav(pd,gd); at,ct=_atecte(pd,gd)
# # # # # # # # # #         acc.update(dist,at,ct)
# # # # # # # # # #     if bk:
# # # # # # # # # #         try: ema.restore(model,bk)
# # # # # # # # # #         except: pass
# # # # # # # # # #     r=acc.compute()
# # # # # # # # # #     def _v(k): return r.get(k,float("nan"))
# # # # # # # # # #     def _m(v,t): return "ok" if np.isfinite(v) and v<t else "no"
# # # # # # # # # #     el=time.perf_counter()-t0
# # # # # # # # # #     print(f"\n{'='*70}")
# # # # # # # # # #     print(f"  [{tag}  {el:.0f}s]")
# # # # # # # # # #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]  "
# # # # # # # # # #           f"12h={_v('12h'):.0f}  24h={_v('24h'):.0f}  "
# # # # # # # # # #           f"48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
# # # # # # # # # #     if np.isfinite(_v("ATE_mean")):
# # # # # # # # # #         print(f"  ATE={_v('ATE_mean'):.1f}[{_m(_v('ATE_mean'),142.21)}]  "
# # # # # # # # # #               f"CTE={_v('CTE_mean'):.1f}[{_m(_v('CTE_mean'),42.04)}]")
# # # # # # # # # #     print(f"  vs ST-Trans: {_gap(r)}")
# # # # # # # # # #     bt=_beat(r)
# # # # # # # # # #     if bt: print(f"  {bt}")
# # # # # # # # # #     print(f"  Score={_score(r):.2f}")
# # # # # # # # # #     print(f"{'='*70}\n")
# # # # # # # # # #     return r


# # # # # # # # # # # ── Saver ─────────────────────────────────────────────────────────────────────

# # # # # # # # # # def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl, extra=None):
# # # # # # # # # #     m=_unwrap(model); ema=getattr(m,"_ema",None); esd=None
# # # # # # # # # #     if ema and hasattr(ema,"shadow"):
# # # # # # # # # #         try: esd={k:v.cpu().clone() for k,v in ema.shadow.items()}
# # # # # # # # # #         except: pass
# # # # # # # # # #     d={"epoch":ep,"model_state_dict":m.state_dict(),
# # # # # # # # # #        "optimizer_state":opt.state_dict(),"scheduler_state":sched.state_dict(),
# # # # # # # # # #        "ema_shadow":esd,"best_score":saver.bs,"best_ade":saver.ba,
# # # # # # # # # #        "best_72h":saver.b7,"best_ate":saver.bat,"best_cte":saver.bc,
# # # # # # # # # #        "train_loss":tl,"val_loss":vl}
# # # # # # # # # #     if extra: d.update(extra)
# # # # # # # # # #     torch.save(d,path)


# # # # # # # # # # class Saver:
# # # # # # # # # #     """
# # # # # # # # # #     Single-phase saver với patience=40, min_ep=35.
# # # # # # # # # #     Longer patience than v59 (35) để cho model đủ thời gian.
# # # # # # # # # #     """
# # # # # # # # # #     def __init__(self, patience=40, min_ep=35):
# # # # # # # # # #         self.patience=patience; self.min_ep=min_ep; self.cnt=0; self.stop=False
# # # # # # # # # #         self.bs=self.ba=self.b7=self.bat=self.bc=float("inf")

# # # # # # # # # #     def update(self, r, model, out, ep, opt, sched, tl, vl, tag=""):
# # # # # # # # # #         sc=_score(r); ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
# # # # # # # # # #         ate=r.get("ATE_mean",1e9); cte=r.get("CTE_mean",1e9)
# # # # # # # # # #         for v,a,fn in [(ade,"ba","best_ade.pth"),(h72,"b7","best_72h.pth"),
# # # # # # # # # #                         (ate,"bat","best_ate.pth"),(cte,"bc","best_cte.pth")]:
# # # # # # # # # #             if v<getattr(self,a):
# # # # # # # # # #                 setattr(self,a,v)
# # # # # # # # # #                 _save_ckpt(os.path.join(out,fn),ep,model,opt,sched,self,tl,vl)
# # # # # # # # # #         if sc<self.bs:
# # # # # # # # # #             self.bs=sc; self.cnt=0
# # # # # # # # # #             _save_ckpt(os.path.join(out,f"best_{tag or 'composite'}.pth"),
# # # # # # # # # #                        ep,model,opt,sched,self,tl,vl,{"score":sc,"ade":ade,"h72":h72})
# # # # # # # # # #             print(f"  [BEST] {tag} ep={ep} score={sc:.2f} "
# # # # # # # # # #                   f"ADE={ade:.1f} 72h={h72:.0f} ATE={ate:.1f} CTE={cte:.1f}")
# # # # # # # # # #         else:
# # # # # # # # # #             self.cnt+=1
# # # # # # # # # #             print(f"  No improve {self.cnt}/{self.patience}  best={self.bs:.2f}  cur={sc:.2f}")
# # # # # # # # # #         if ep>=self.min_ep and self.cnt>=self.patience:
# # # # # # # # # #             self.stop=True; print(f"  [STOP] ep={ep}")


# # # # # # # # # # def mksub(ds,n,bs,cf):
# # # # # # # # # #     idx=random.Random(42).sample(range(len(ds)),min(n,len(ds)))
# # # # # # # # # #     return DataLoader(Subset(ds,idx),batch_size=bs,shuffle=False,
# # # # # # # # # #                        collate_fn=cf,num_workers=0,drop_last=False)


# # # # # # # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # # # # # # def get_args():
# # # # # # # # # #     p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # # # # # #     p.add_argument("--dataset_root",       default="TCND_vn")
# # # # # # # # # #     p.add_argument("--obs_len",            default=8,    type=int)
# # # # # # # # # #     p.add_argument("--pred_len",           default=12,   type=int)
# # # # # # # # # #     p.add_argument("--batch_size",         default=32,   type=int)
# # # # # # # # # #     p.add_argument("--num_epochs",         default=120,  type=int)
# # # # # # # # # #     p.add_argument("--learning_rate",      default=1e-4, type=float)
# # # # # # # # # #     p.add_argument("--weight_decay",       default=1e-3, type=float)
# # # # # # # # # #     p.add_argument("--warmup_epochs",      default=3,    type=int)
# # # # # # # # # #     p.add_argument("--grad_clip",          default=1.0,  type=float)
# # # # # # # # # #     p.add_argument("--patience",           default=40,   type=int)
# # # # # # # # # #     p.add_argument("--min_ep",             default=35,   type=int)
# # # # # # # # # #     p.add_argument("--use_amp",            action="store_true")
# # # # # # # # # #     p.add_argument("--num_workers",        default=2,    type=int)
# # # # # # # # # #     p.add_argument("--sigma_min",          default=0.02, type=float)
# # # # # # # # # #     p.add_argument("--use_ot",             default=True, action="store_true")
# # # # # # # # # #     p.add_argument("--no_ot",              dest="use_ot",action="store_false")
# # # # # # # # # #     p.add_argument("--cfg_guidance_scale", default=1.5,  type=float)
# # # # # # # # # #     p.add_argument("--easy_thresh",        default=0.25, type=float,
# # # # # # # # # #                     help="delta < easy_thresh → easy storm (default 0.25)")
# # # # # # # # # #     p.add_argument("--n_ensemble",         default=50,   type=int)
# # # # # # # # # #     p.add_argument("--val_freq",           default=3,    type=int)
# # # # # # # # # #     p.add_argument("--val_subset_size",    default=500,  type=int)
# # # # # # # # # #     p.add_argument("--fast_ddim",          default=10,   type=int)
# # # # # # # # # #     p.add_argument("--full_ddim",          default=20,   type=int)
# # # # # # # # # #     p.add_argument("--use_ema",            default=True, action="store_true")
# # # # # # # # # #     p.add_argument("--no_ema",             dest="use_ema",action="store_false")
# # # # # # # # # #     p.add_argument("--ema_decay",          default=0.995,type=float)
# # # # # # # # # #     p.add_argument("--output_dir",         default="runs/v74")
# # # # # # # # # #     p.add_argument("--gpu_num",            default="0")
# # # # # # # # # #     p.add_argument("--delim",              default=" ")
# # # # # # # # # #     p.add_argument("--skip",               default=1,    type=int)
# # # # # # # # # #     p.add_argument("--min_ped",            default=1,    type=int)
# # # # # # # # # #     p.add_argument("--threshold",          default=0.002,type=float)
# # # # # # # # # #     p.add_argument("--other_modal",        default="gph")
# # # # # # # # # #     p.add_argument("--test_year",          default=None, type=int)
# # # # # # # # # #     p.add_argument("--resume",             default=None)
# # # # # # # # # #     p.add_argument("--resume_epoch",       default=None, type=int)
# # # # # # # # # #     p.add_argument("--eval_test_after_train",default=True,action="store_true")
# # # # # # # # # #     return p.parse_args()


# # # # # # # # # # # ── Main ──────────────────────────────────────────────────────────────────────

# # # # # # # # # # def main(args):
# # # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_num)
# # # # # # # # # #     dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # # # #     os.makedirs(args.output_dir,exist_ok=True)

# # # # # # # # # #     print("="*72)
# # # # # # # # # #     print(f"  TC-FlowMatching v74")
# # # # # # # # # #     print(f"  Loss: L_fm + L_dpe(step_w,d=300) + L_disp(norm) + L_ep(hard)")
# # # # # # # # # #     print(f"  Split: easy(delta<{args.easy_thresh}) / hard(delta>={args.easy_thresh})")
# # # # # # # # # #     print(f"  Training: single phase, patience={args.patience}, min_ep={args.min_ep}")
# # # # # # # # # #     print(f"  Inference: K=3 mode clustering at 72h endpoint")
# # # # # # # # # #     print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']} ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
# # # # # # # # # #     print("="*72)

# # # # # # # # # #     trd,trl=data_loader(args,{"root":args.dataset_root,"type":"train"},test=False)
# # # # # # # # # #     vd,vl  =data_loader(args,{"root":args.dataset_root,"type":"val"},  test=True)
# # # # # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # # # # #     vsub=mksub(vd,args.val_subset_size,args.batch_size,seq_collate)
# # # # # # # # # #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# # # # # # # # # #     model=TCFlowMatching(
# # # # # # # # # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # # # # # # # # #         sigma_min=args.sigma_min, use_ema=args.use_ema, ema_decay=args.ema_decay,
# # # # # # # # # #         use_ot=args.use_ot, cfg_guidance_scale=args.cfg_guidance_scale,
# # # # # # # # # #         easy_thresh=args.easy_thresh, n_ensemble=args.n_ensemble,
# # # # # # # # # #     ).to(dev)
# # # # # # # # # #     model.init_ema()
# # # # # # # # # #     n_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # # # # #     sw_params=sum(p.numel() for p in model.step_weights.parameters())
# # # # # # # # # #     lw_params=sum(p.numel() for p in model.loss_weights.parameters())
# # # # # # # # # #     print(f"  params total: {n_params:,} | step_w: {sw_params} | loss_w: {lw_params}")

# # # # # # # # # #     # Single optimizer, tất cả params — như v59
# # # # # # # # # #     opt=optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
# # # # # # # # # #     saver=Saver(patience=args.patience,min_ep=args.min_ep)
# # # # # # # # # #     scaler=GradScaler("cuda",enabled=args.use_amp)
# # # # # # # # # #     nstep=len(trl); total=nstep*args.num_epochs; wstp=nstep*args.warmup_epochs
# # # # # # # # # #     sched=get_cosine_schedule_with_warmup(opt,wstp,total,min_lr=1e-6)

# # # # # # # # # #     # Resume
# # # # # # # # # #     start=0
# # # # # # # # # #     if args.resume and os.path.exists(args.resume):
# # # # # # # # # #         print(f"  Loading: {args.resume}")
# # # # # # # # # #         ck=torch.load(args.resume,map_location=dev)
# # # # # # # # # #         m=_unwrap(model); ms,_=m.load_state_dict(ck["model_state_dict"],strict=False)
# # # # # # # # # #         if ms: print(f"  Missing: {len(ms)}")
# # # # # # # # # #         ema=getattr(m,"_ema",None)
# # # # # # # # # #         if ema and ck.get("ema_shadow"):
# # # # # # # # # #             for k,v in ck["ema_shadow"].items():
# # # # # # # # # #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
# # # # # # # # # #         try: opt.load_state_dict(ck["optimizer_state"])
# # # # # # # # # #         except Exception as e: print(f"  Opt not loaded: {e}")
# # # # # # # # # #         try: sched.load_state_dict(ck["scheduler_state"])
# # # # # # # # # #         except:
# # # # # # # # # #             for _ in range(ck.get("epoch",0)*nstep): sched.step()
# # # # # # # # # #         for a,attr in [("best_score","bs"),("best_ade","ba"),("best_72h","b7"),
# # # # # # # # # #                          ("best_ate","bat"),("best_cte","bc")]:
# # # # # # # # # #             if a in ck: setattr(saver,attr,ck[a])
# # # # # # # # # #         start=args.resume_epoch or ck.get("epoch",0)+1
# # # # # # # # # #         print(f"  Resume ep={start}")

# # # # # # # # # #     try:
# # # # # # # # # #         model=torch.compile(model,mode="reduce-overhead"); print("  torch.compile: ok")
# # # # # # # # # #     except: pass

# # # # # # # # # #     ts=time.perf_counter()
# # # # # # # # # #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# # # # # # # # # #     print("="*72)

# # # # # # # # # #     for ep in range(start, args.num_epochs):
# # # # # # # # # #         model.train(); sl=0.; t0=time.perf_counter()

# # # # # # # # # #         for i, batch in enumerate(trl):
# # # # # # # # # #             bl=move(list(batch),dev)
# # # # # # # # # #             with autocast(device_type="cuda",enabled=args.use_amp):
# # # # # # # # # #                 bd=model.get_loss_breakdown(bl,epoch=ep)

# # # # # # # # # #             opt.zero_grad()
# # # # # # # # # #             scaler.scale(bd["total"]).backward()
# # # # # # # # # #             scaler.unscale_(opt)
# # # # # # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
# # # # # # # # # #             sb=scaler.get_scale(); scaler.step(opt); scaler.update()
# # # # # # # # # #             if scaler.get_scale()>=sb: sched.step()
# # # # # # # # # #             model.ema_update()
# # # # # # # # # #             sl+=bd["total"].item()

# # # # # # # # # #             if i % 20 == 0:
# # # # # # # # # #                 lr=opt.param_groups[0]["lr"]
# # # # # # # # # #                 sw_r=bd.get("sw_ratio",0.)
# # # # # # # # # #                 # Stability warnings
# # # # # # # # # #                 sw_warn=" [!sw<3]" if sw_r<3.0 else ""
# # # # # # # # # #                 tot_warn=" [!tot>10]" if bd["total"].item()>10. else ""
# # # # # # # # # #                 print(
# # # # # # # # # #                     f"  [{ep:>3}][{i:>3}/{nstep}]"
# # # # # # # # # #                     f"  tot={bd['total'].item():.3f}{tot_warn}"
# # # # # # # # # #                     f"  fm={bd.get('L_fm',0):.4f}"
# # # # # # # # # #                     f"  dpe={bd.get('L_dpe',0):.4f}"
# # # # # # # # # #                     f"  disp={bd.get('L_disp',0):.4f}"
# # # # # # # # # #                     f"  ep_l={bd.get('L_ep',0):.4f}"
# # # # # # # # # #                     f"  E={bd.get('L_easy',0):.3f}"
# # # # # # # # # #                     f"  H={bd.get('L_hard',0):.3f}"
# # # # # # # # # #                     f"  sw24={bd.get('sw_sw_24h',bd.get('sw_24h',0)):.2f}"
# # # # # # # # # #                     f"  sw72={bd.get('sw_sw_72h',bd.get('sw_72h',0)):.2f}"
# # # # # # # # # #                     f"  swr={sw_r:.1f}{sw_warn}"
# # # # # # # # # #                     f"  dpe/fm={bd.get('lw_lw_dpe_fm',bd.get('lw_dpe_fm',0)):.2f}"
# # # # # # # # # #                     f"  efrac={bd.get('easy_frac',0):.2f}"
# # # # # # # # # #                     f"  d={bd.get('delta_mean',0):.2f}"
# # # # # # # # # #                     f"  lr={lr:.2e}"
# # # # # # # # # #                 )

# # # # # # # # # #         avt=sl/nstep

# # # # # # # # # #         # Val loss
# # # # # # # # # #         model.eval(); vls=0.
# # # # # # # # # #         with torch.no_grad():
# # # # # # # # # #             for batch in vl:
# # # # # # # # # #                 bv=move(list(batch),dev)
# # # # # # # # # #                 with autocast(device_type="cuda",enabled=args.use_amp):
# # # # # # # # # #                     vls+=model.get_loss(bv,epoch=ep).item()
# # # # # # # # # #         avv=vls/len(vl); eps=time.perf_counter()-t0
# # # # # # # # # #         lr_cur=opt.param_groups[0]["lr"]
# # # # # # # # # #         print(f"  Epoch {ep:>3} | train={avt:.4f} val={avv:.4f} | lr={lr_cur:.2e} | {eps:.0f}s")

# # # # # # # # # #         # Fast eval (subset)
# # # # # # # # # #         rf=evaluate(model,vsub,dev,tag=f"FAST ep{ep}",steps=args.fast_ddim)
# # # # # # # # # #         saver.update(rf,model,args.output_dir,ep,opt,sched,avt,avv,tag="fast")

# # # # # # # # # #         # Full val every val_freq
# # # # # # # # # #         if ep%args.val_freq==0:
# # # # # # # # # #             em=getattr(_unwrap(model),"_ema",None)
# # # # # # # # # #             rr=evaluate(model,vl,dev,tag=f"RAW ep{ep}",steps=args.full_ddim)
# # # # # # # # # #             saver.update(rr,model,args.output_dir,ep,opt,sched,avt,avv,tag="raw")
# # # # # # # # # #             if em and ep>=3:
# # # # # # # # # #                 re=evaluate(model,vl,dev,tag=f"EMA ep{ep}",ema=em,steps=args.full_ddim)
# # # # # # # # # #                 saver.update(re,model,args.output_dir,ep,opt,sched,avt,avv,tag="ema")

# # # # # # # # # #         # Periodic checkpoint
# # # # # # # # # #         if ep%10==0 or ep==args.num_epochs-1:
# # # # # # # # # #             _save_ckpt(os.path.join(args.output_dir,f"ckpt_ep{ep:03d}.pth"),
# # # # # # # # # #                        ep,model,opt,sched,saver,avt,avv)

# # # # # # # # # #         if saver.stop: print(f"  Early stop ep={ep}"); break

# # # # # # # # # #     th=(time.perf_counter()-ts)/3600.
# # # # # # # # # #     print(f"\n  Best: ADE={saver.ba:.1f} 72h={saver.b7:.0f} "
# # # # # # # # # #           f"ATE={saver.bat:.1f} CTE={saver.bc:.1f} ({th:.2f}h)")

# # # # # # # # # #     # Post-training test
# # # # # # # # # #     if args.eval_test_after_train:
# # # # # # # # # #         print("\n"+"="*72+"\n  POST-TRAINING TEST\n"+"="*72)
# # # # # # # # # #         try: _,tl2=data_loader(args,{"root":args.dataset_root,"type":"test"},test=True)
# # # # # # # # # #         except: print("  No test set → using val"); tl2=vl
# # # # # # # # # #         for fn,lb in [("best_composite.pth","COMPOSITE"),("best_72h.pth","72H"),
# # # # # # # # # #                          ("best_ema.pth","EMA")]:
# # # # # # # # # #             pp=os.path.join(args.output_dir,fn)
# # # # # # # # # #             if not os.path.exists(pp): continue
# # # # # # # # # #             ck=torch.load(pp,map_location=dev)
# # # # # # # # # #             _unwrap(model).load_state_dict(ck["model_state_dict"],strict=False)
# # # # # # # # # #             em=getattr(_unwrap(model),"_ema",None)
# # # # # # # # # #             if em and ck.get("ema_shadow"):
# # # # # # # # # #                 for k,v in ck["ema_shadow"].items():
# # # # # # # # # #                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
# # # # # # # # # #             r=evaluate(model,tl2,dev,tag=f"TEST/{lb}",steps=args.full_ddim)
# # # # # # # # # #             print(f"\n  --- {lb} ---")
# # # # # # # # # #             for key,ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
# # # # # # # # # #                 v=r.get(key,float("nan"))
# # # # # # # # # #                 mk="BEAT!" if np.isfinite(v) and v<ref else f"need {ref:.0f}"
# # # # # # # # # #                 print(f"    {key:<10}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
# # # # # # # # # #     print("="*72)


# # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # #     args=get_args()
# # # # # # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # # # # # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# # # # # # # # # #     main(args)

# # # # # # # # # """
# # # # # # # # # train_v74.py — TC-FlowMatching v74
# # # # # # # # # ════════════════════════════════════════════════════════════════════════
# # # # # # # # # TRAINING STRATEGY: Single phase, như v59 — không freeze, không transition

# # # # # # # # # KEY DIFFERENCES vs failed versions:
# # # # # # # # #   - V68/V72 fail: phase separation, best ep9 plateau
# # # # # # # # #   - V69 fail: loss scale 700 (disp_smooth in km)
# # # # # # # # #   - V74: single phase như v59, loss scale 1-4

# # # # # # # # # MONITORING (xem để đảm bảo stable):
# # # # # # # # #   sw_ratio >= 3.0          → step weights đang đúng hướng
# # # # # # # # #   lw_dpe_fm ∈ [1.5, 3.0]  → DPE/FM balance ổn
# # # # # # # # #   L_hard > L_easy          → hard storms được penalize nhiều hơn (correct)
# # # # # # # # #   easy_frac ≈ 0.20-0.40    → delta score distribution hợp lý
# # # # # # # # #   tot ∈ [1.0, 5.0]         → loss scale tốt (nếu tot > 10: có bug)

# # # # # # # # # RUN:
# # # # # # # # #   python scripts/train_v74.py \\
# # # # # # # # #       --dataset_root /path/to/tc-ofm \\
# # # # # # # # #       --output_dir   runs/v74 \\
# # # # # # # # #       --batch_size   32 \\
# # # # # # # # #       --use_amp
# # # # # # # # # """
# # # # # # # # # from __future__ import annotations

# # # # # # # # # import sys, os
# # # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # # # import argparse, time, random
# # # # # # # # # from collections import defaultdict

# # # # # # # # # import numpy as np
# # # # # # # # # import torch
# # # # # # # # # import torch.optim as optim
# # # # # # # # # from torch.amp import autocast, GradScaler
# # # # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # # # from Model.data.loader_training import data_loader
# # # # # # # # # from Model.flow_matching_model import TCFlowMatching

# # # # # # # # # try:
# # # # # # # # #     from Model.utils import get_cosine_schedule_with_warmup
# # # # # # # # # except ImportError:
# # # # # # # # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # # # # # # # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # # # # # # # #         return CosineAnnealingLR(opt, T_max=max(total_steps,1), eta_min=min_lr)

# # # # # # # # # TARGETS = {"ADE":172.68,"72h":321.39,"ATE":142.21,"CTE":42.04,
# # # # # # # # #            "12h":65.42,"24h":104.67,"48h":205.10}
# # # # # # # # # R_EARTH = 6371.0


# # # # # # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # # # # # def _unwrap(m): return m._orig_mod if hasattr(m,'_orig_mod') else m

# # # # # # # # # def move(b, dev):
# # # # # # # # #     out=list(b)
# # # # # # # # #     for i,x in enumerate(out):
# # # # # # # # #         if torch.is_tensor(x): out[i]=x.to(dev)
# # # # # # # # #         elif isinstance(x,dict):
# # # # # # # # #             out[i]={k:v.to(dev) if torch.is_tensor(v) else v for k,v in x.items()}
# # # # # # # # #     return out


# # # # # # # # # # ── Metric utils ──────────────────────────────────────────────────────────────

# # # # # # # # # def _ntd(a):
# # # # # # # # #     o=a.clone(); o[...,0]=(a[...,0]*50.+1800.)/10.; o[...,1]=(a[...,1]*50.)/10.; return o

# # # # # # # # # def _hav(p1,p2):
# # # # # # # # #     la1=torch.deg2rad(p1[...,1]); la2=torch.deg2rad(p2[...,1])
# # # # # # # # #     dlat=torch.deg2rad(p2[...,1]-p1[...,1]); dlon=torch.deg2rad(p2[...,0]-p1[...,0])
# # # # # # # # #     a=torch.sin(dlat/2).pow(2)+torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2)
# # # # # # # # #     return 2.*R_EARTH*torch.asin(a.clamp(1e-12,1-1e-12).sqrt())

# # # # # # # # # def _atecte(pd,gd):
# # # # # # # # #     T=min(pd.shape[0],gd.shape[0])
# # # # # # # # #     if T<2: z=pd.new_zeros(1,pd.shape[1]); return z,z
# # # # # # # # #     lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
# # # # # # # # #     lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
# # # # # # # # #     lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
# # # # # # # # #     ya=torch.sin(lo2-lo1)*torch.cos(la2)
# # # # # # # # #     xa=torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # # # # # # # #     be=torch.atan2(ya,xa)
# # # # # # # # #     ye=torch.sin(lo3-lo2)*torch.cos(la3)
# # # # # # # # #     xe=torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # # # # # # # #     bee=torch.atan2(ye,xe); tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
# # # # # # # # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # # # # # # class Acc:
# # # # # # # # #     def __init__(self):
# # # # # # # # #         self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list)
# # # # # # # # #         self._h={12:1,24:3,48:7,72:11}
# # # # # # # # #     def update(self,dist,ate=None,cte=None):
# # # # # # # # #         self.d.extend(dist.mean(0).tolist())
# # # # # # # # #         for h,s in self._h.items():
# # # # # # # # #             if s<dist.shape[0]: self.sd[h].extend(dist[s].tolist())
# # # # # # # # #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# # # # # # # # #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())
# # # # # # # # #     def compute(self):
# # # # # # # # #         r={"ADE":float(np.mean(self.d)) if self.d else float("nan"),
# # # # # # # # #            "ATE_mean":float(np.mean(self.a)) if self.a else float("nan"),
# # # # # # # # #            "CTE_mean":float(np.mean(self.c)) if self.c else float("nan"),
# # # # # # # # #            "n":len(self.d)}
# # # # # # # # #         for h in self._h:
# # # # # # # # #             v=self.sd.get(h,[]); r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
# # # # # # # # #         return r


# # # # # # # # # def _score(r):
# # # # # # # # #     ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
# # # # # # # # #     ate=r.get("ATE_mean",1e9); cte=r.get("CTE_mean",1e9)
# # # # # # # # #     if not np.isfinite(ate): ate=ade*.46
# # # # # # # # #     if not np.isfinite(cte): cte=ade*.53
# # # # # # # # #     return 100.*(0.05*(ade/136.)+0.10*(r.get("12h",ade)/50.)+
# # # # # # # # #                  0.15*(r.get("24h",ade)/100.)+0.20*(r.get("48h",ade)/200.)+
# # # # # # # # #                  0.25*(h72/300.)+0.13*(ate/80.)+0.12*(cte/94.))

# # # # # # # # # def _beat(r):
# # # # # # # # #     p=[]
# # # # # # # # #     for k,t in [("ADE",172.68),("ATE_mean",142.21),("CTE_mean",42.04),
# # # # # # # # #                   ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
# # # # # # # # #         v=r.get(k,1e9)
# # # # # # # # #         if np.isfinite(v) and v<t: p.append(f"{k.replace('_mean','')}:{v:.1f}")
# # # # # # # # #     return "*** BEAT ST-TRANS: "+" ".join(p)+" ***" if p else ""

# # # # # # # # # def _gap(r):
# # # # # # # # #     out=[]
# # # # # # # # #     for k,ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
# # # # # # # # #         v=r.get(k,float("nan"))
# # # # # # # # #         if np.isfinite(v):
# # # # # # # # #             out.append(f"{k.replace('_mean','')}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
# # # # # # # # #     return " | ".join(out)


# # # # # # # # # # ── Evaluation ────────────────────────────────────────────────────────────────

# # # # # # # # # @torch.no_grad()
# # # # # # # # # def evaluate(model, loader, dev, tag="", ema=None, steps=20):
# # # # # # # # #     bk=None
# # # # # # # # #     if ema:
# # # # # # # # #         try: bk=ema.apply_to(model)
# # # # # # # # #         except: pass
# # # # # # # # #     model.eval(); acc=Acc(); t0=time.perf_counter()
# # # # # # # # #     for b in loader:
# # # # # # # # #         bl=move(list(b),dev)
# # # # # # # # #         result=model.sample(bl,ddim_steps=steps)
# # # # # # # # #         p=result[0] if isinstance(result,(tuple,list)) else result
# # # # # # # # #         g=bl[1]; T=min(p.shape[0],g.shape[0])
# # # # # # # # #         pd=_ntd(p[:T]); gd=_ntd(g[:T])
# # # # # # # # #         dist=_hav(pd,gd); at,ct=_atecte(pd,gd)
# # # # # # # # #         acc.update(dist,at,ct)
# # # # # # # # #     if bk:
# # # # # # # # #         try: ema.restore(model,bk)
# # # # # # # # #         except: pass
# # # # # # # # #     r=acc.compute()
# # # # # # # # #     def _v(k): return r.get(k,float("nan"))
# # # # # # # # #     def _m(v,t): return "ok" if np.isfinite(v) and v<t else "no"
# # # # # # # # #     el=time.perf_counter()-t0
# # # # # # # # #     print(f"\n{'='*70}")
# # # # # # # # #     print(f"  [{tag}  {el:.0f}s]")
# # # # # # # # #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]  "
# # # # # # # # #           f"12h={_v('12h'):.0f}  24h={_v('24h'):.0f}  "
# # # # # # # # #           f"48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
# # # # # # # # #     if np.isfinite(_v("ATE_mean")):
# # # # # # # # #         print(f"  ATE={_v('ATE_mean'):.1f}[{_m(_v('ATE_mean'),142.21)}]  "
# # # # # # # # #               f"CTE={_v('CTE_mean'):.1f}[{_m(_v('CTE_mean'),42.04)}]")
# # # # # # # # #     print(f"  vs ST-Trans: {_gap(r)}")
# # # # # # # # #     bt=_beat(r)
# # # # # # # # #     if bt: print(f"  {bt}")
# # # # # # # # #     print(f"  Score={_score(r):.2f}")
# # # # # # # # #     print(f"{'='*70}\n")
# # # # # # # # #     return r


# # # # # # # # # # ── Saver ─────────────────────────────────────────────────────────────────────

# # # # # # # # # def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl, extra=None):
# # # # # # # # #     m=_unwrap(model); ema=getattr(m,"_ema",None); esd=None
# # # # # # # # #     if ema and hasattr(ema,"shadow"):
# # # # # # # # #         try: esd={k:v.cpu().clone() for k,v in ema.shadow.items()}
# # # # # # # # #         except: pass
# # # # # # # # #     d={"epoch":ep,"model_state_dict":m.state_dict(),
# # # # # # # # #        "optimizer_state":opt.state_dict(),"scheduler_state":sched.state_dict(),
# # # # # # # # #        "ema_shadow":esd,"best_score":saver.bs,"best_ade":saver.ba,
# # # # # # # # #        "best_72h":saver.b7,"best_ate":saver.bat,"best_cte":saver.bc,
# # # # # # # # #        "train_loss":tl,"val_loss":vl}
# # # # # # # # #     if extra: d.update(extra)
# # # # # # # # #     torch.save(d,path)


# # # # # # # # # class Saver:
# # # # # # # # #     """
# # # # # # # # #     Single-phase saver với patience=40, min_ep=35.
# # # # # # # # #     Longer patience than v59 (35) để cho model đủ thời gian.
# # # # # # # # #     """
# # # # # # # # #     def __init__(self, patience=40, min_ep=35):
# # # # # # # # #         self.patience=patience; self.min_ep=min_ep; self.cnt=0; self.stop=False
# # # # # # # # #         self.bs=self.ba=self.b7=self.bat=self.bc=float("inf")

# # # # # # # # #     def update(self, r, model, out, ep, opt, sched, tl, vl, tag=""):
# # # # # # # # #         sc=_score(r); ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
# # # # # # # # #         ate=r.get("ATE_mean",1e9); cte=r.get("CTE_mean",1e9)
# # # # # # # # #         for v,a,fn in [(ade,"ba","best_ade.pth"),(h72,"b7","best_72h.pth"),
# # # # # # # # #                         (ate,"bat","best_ate.pth"),(cte,"bc","best_cte.pth")]:
# # # # # # # # #             if v<getattr(self,a):
# # # # # # # # #                 setattr(self,a,v)
# # # # # # # # #                 _save_ckpt(os.path.join(out,fn),ep,model,opt,sched,self,tl,vl)
# # # # # # # # #         if sc<self.bs:
# # # # # # # # #             self.bs=sc; self.cnt=0
# # # # # # # # #             _save_ckpt(os.path.join(out,f"best_{tag or 'composite'}.pth"),
# # # # # # # # #                        ep,model,opt,sched,self,tl,vl,{"score":sc,"ade":ade,"h72":h72})
# # # # # # # # #             print(f"  [BEST] {tag} ep={ep} score={sc:.2f} "
# # # # # # # # #                   f"ADE={ade:.1f} 72h={h72:.0f} ATE={ate:.1f} CTE={cte:.1f}")
# # # # # # # # #         else:
# # # # # # # # #             self.cnt+=1
# # # # # # # # #             print(f"  No improve {self.cnt}/{self.patience}  best={self.bs:.2f}  cur={sc:.2f}")
# # # # # # # # #         if ep>=self.min_ep and self.cnt>=self.patience:
# # # # # # # # #             self.stop=True; print(f"  [STOP] ep={ep}")


# # # # # # # # # def mksub(ds,n,bs,cf):
# # # # # # # # #     idx=random.Random(42).sample(range(len(ds)),min(n,len(ds)))
# # # # # # # # #     return DataLoader(Subset(ds,idx),batch_size=bs,shuffle=False,
# # # # # # # # #                        collate_fn=cf,num_workers=0,drop_last=False)


# # # # # # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # # # # # def get_args():
# # # # # # # # #     p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # # # # #     p.add_argument("--dataset_root",       default="TCND_vn")
# # # # # # # # #     p.add_argument("--obs_len",            default=8,    type=int)
# # # # # # # # #     p.add_argument("--pred_len",           default=12,   type=int)
# # # # # # # # #     p.add_argument("--batch_size",         default=32,   type=int)
# # # # # # # # #     p.add_argument("--num_epochs",         default=120,  type=int)
# # # # # # # # #     p.add_argument("--learning_rate",      default=1e-4, type=float)
# # # # # # # # #     p.add_argument("--weight_decay",       default=1e-3, type=float)
# # # # # # # # #     p.add_argument("--warmup_epochs",      default=3,    type=int)
# # # # # # # # #     p.add_argument("--grad_clip",          default=1.0,  type=float)
# # # # # # # # #     p.add_argument("--patience",           default=40,   type=int)
# # # # # # # # #     p.add_argument("--min_ep",             default=35,   type=int)
# # # # # # # # #     p.add_argument("--use_amp",            action="store_true")
# # # # # # # # #     p.add_argument("--num_workers",        default=2,    type=int)
# # # # # # # # #     p.add_argument("--sigma_min",          default=0.02, type=float)
# # # # # # # # #     p.add_argument("--use_ot",             default=True, action="store_true")
# # # # # # # # #     p.add_argument("--no_ot",              dest="use_ot",action="store_false")
# # # # # # # # #     p.add_argument("--cfg_guidance_scale", default=1.5,  type=float)
# # # # # # # # #     p.add_argument("--easy_thresh",        default=0.25, type=float,
# # # # # # # # #                     help="delta < easy_thresh → easy storm (default 0.25)")
# # # # # # # # #     p.add_argument("--n_ensemble",         default=50,   type=int)
# # # # # # # # #     p.add_argument("--val_freq",           default=3,    type=int)
# # # # # # # # #     p.add_argument("--val_subset_size",    default=500,  type=int)
# # # # # # # # #     p.add_argument("--fast_ddim",          default=10,   type=int)
# # # # # # # # #     p.add_argument("--full_ddim",          default=20,   type=int)
# # # # # # # # #     p.add_argument("--use_ema",            default=True, action="store_true")
# # # # # # # # #     p.add_argument("--no_ema",             dest="use_ema",action="store_false")
# # # # # # # # #     p.add_argument("--ema_decay",          default=0.995,type=float)
# # # # # # # # #     p.add_argument("--output_dir",         default="runs/v74")
# # # # # # # # #     p.add_argument("--gpu_num",            default="0")
# # # # # # # # #     p.add_argument("--delim",              default=" ")
# # # # # # # # #     p.add_argument("--skip",               default=1,    type=int)
# # # # # # # # #     p.add_argument("--min_ped",            default=1,    type=int)
# # # # # # # # #     p.add_argument("--threshold",          default=0.002,type=float)
# # # # # # # # #     p.add_argument("--other_modal",        default="gph")
# # # # # # # # #     p.add_argument("--test_year",          default=None, type=int)
# # # # # # # # #     p.add_argument("--resume",             default=None)
# # # # # # # # #     p.add_argument("--resume_epoch",       default=None, type=int)
# # # # # # # # #     p.add_argument("--eval_test_after_train",default=True,action="store_true")
# # # # # # # # #     return p.parse_args()


# # # # # # # # # # ── Main ──────────────────────────────────────────────────────────────────────

# # # # # # # # # def main(args):
# # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_num)
# # # # # # # # #     dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # # #     os.makedirs(args.output_dir,exist_ok=True)

# # # # # # # # #     print("="*72)
# # # # # # # # #     print(f"  TC-FlowMatching v74")
# # # # # # # # #     print(f"  Loss: L_fm + L_dpe(step_w,d=300) + L_disp(norm) + L_ep(hard)")
# # # # # # # # #     print(f"  Split: easy(delta<{args.easy_thresh}) / hard(delta>={args.easy_thresh})")
# # # # # # # # #     print(f"  Training: single phase, patience={args.patience}, min_ep={args.min_ep}")
# # # # # # # # #     print(f"  Inference: K=3 mode clustering at 72h endpoint")
# # # # # # # # #     print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']} ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
# # # # # # # # #     print("="*72)

# # # # # # # # #     trd,trl=data_loader(args,{"root":args.dataset_root,"type":"train"},test=False)
# # # # # # # # #     vd,vl  =data_loader(args,{"root":args.dataset_root,"type":"val"},  test=True)
# # # # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # # # #     vsub=mksub(vd,args.val_subset_size,args.batch_size,seq_collate)
# # # # # # # # #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# # # # # # # # #     model=TCFlowMatching(
# # # # # # # # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # # # # # # # #         sigma_min=args.sigma_min, use_ema=args.use_ema, ema_decay=args.ema_decay,
# # # # # # # # #         use_ot=args.use_ot, cfg_guidance_scale=args.cfg_guidance_scale,
# # # # # # # # #         easy_thresh=args.easy_thresh, n_ensemble=args.n_ensemble,
# # # # # # # # #     ).to(dev)
# # # # # # # # #     model.init_ema()
# # # # # # # # #     n_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # # # #     sw_params=sum(p.numel() for p in model.criterion.step_weights.parameters())
# # # # # # # # #     lw_params=sum(p.numel() for p in model.criterion.loss_weights.parameters())
# # # # # # # # #     print(f"  params total: {n_params:,} | step_w: {sw_params} | loss_w: {lw_params}")

# # # # # # # # #     # Single optimizer, tất cả params — như v59
# # # # # # # # #     opt=optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
# # # # # # # # #     saver=Saver(patience=args.patience,min_ep=args.min_ep)
# # # # # # # # #     scaler=GradScaler("cuda",enabled=args.use_amp)
# # # # # # # # #     nstep=len(trl); total=nstep*args.num_epochs; wstp=nstep*args.warmup_epochs
# # # # # # # # #     sched=get_cosine_schedule_with_warmup(opt,wstp,total,min_lr=1e-6)

# # # # # # # # #     # Resume
# # # # # # # # #     start=0
# # # # # # # # #     if args.resume and os.path.exists(args.resume):
# # # # # # # # #         print(f"  Loading: {args.resume}")
# # # # # # # # #         ck=torch.load(args.resume,map_location=dev)
# # # # # # # # #         m=_unwrap(model); ms,_=m.load_state_dict(ck["model_state_dict"],strict=False)
# # # # # # # # #         if ms: print(f"  Missing: {len(ms)}")
# # # # # # # # #         ema=getattr(m,"_ema",None)
# # # # # # # # #         if ema and ck.get("ema_shadow"):
# # # # # # # # #             for k,v in ck["ema_shadow"].items():
# # # # # # # # #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
# # # # # # # # #         try: opt.load_state_dict(ck["optimizer_state"])
# # # # # # # # #         except Exception as e: print(f"  Opt not loaded: {e}")
# # # # # # # # #         try: sched.load_state_dict(ck["scheduler_state"])
# # # # # # # # #         except:
# # # # # # # # #             for _ in range(ck.get("epoch",0)*nstep): sched.step()
# # # # # # # # #         for a,attr in [("best_score","bs"),("best_ade","ba"),("best_72h","b7"),
# # # # # # # # #                          ("best_ate","bat"),("best_cte","bc")]:
# # # # # # # # #             if a in ck: setattr(saver,attr,ck[a])
# # # # # # # # #         start=args.resume_epoch or ck.get("epoch",0)+1
# # # # # # # # #         print(f"  Resume ep={start}")

# # # # # # # # #     try:
# # # # # # # # #         model=torch.compile(model,mode="reduce-overhead"); print("  torch.compile: ok")
# # # # # # # # #     except: pass

# # # # # # # # #     ts=time.perf_counter()
# # # # # # # # #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# # # # # # # # #     print("="*72)

# # # # # # # # #     for ep in range(start, args.num_epochs):
# # # # # # # # #         model.train(); sl=0.; t0=time.perf_counter()

# # # # # # # # #         for i, batch in enumerate(trl):
# # # # # # # # #             bl=move(list(batch),dev)
# # # # # # # # #             with autocast(device_type="cuda",enabled=args.use_amp):
# # # # # # # # #                 bd=model.get_loss_breakdown(bl,epoch=ep)

# # # # # # # # #             opt.zero_grad()
# # # # # # # # #             scaler.scale(bd["total"]).backward()
# # # # # # # # #             scaler.unscale_(opt)
# # # # # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
# # # # # # # # #             sb=scaler.get_scale(); scaler.step(opt); scaler.update()
# # # # # # # # #             if scaler.get_scale()>=sb: sched.step()
# # # # # # # # #             model.ema_update()
# # # # # # # # #             sl+=bd["total"].item()

# # # # # # # # #             if i % 20 == 0:
# # # # # # # # #                 lr=opt.param_groups[0]["lr"]
# # # # # # # # #                 sw_r=bd.get("sw_sw_ratio", bd.get("sw_ratio", 0.))
# # # # # # # # #                 sw_warn=" [!sw<3]" if sw_r<3.0 else ""
# # # # # # # # #                 tot_warn=" [!tot>10]" if bd["total"].item()>10. else ""
# # # # # # # # #                 print(
# # # # # # # # #                     f"  [{ep:>3}][{i:>3}/{nstep}]"
# # # # # # # # #                     f"  tot={bd['total'].item():.3f}{tot_warn}"
# # # # # # # # #                     f"  fm={bd.get('l_fm',0):.4f}"
# # # # # # # # #                     f"  pos={bd.get('l_pos',0):.4f}"
# # # # # # # # #                     f"  disp={bd.get('l_disp',0):.4f}"
# # # # # # # # #                     f"  sw72={bd.get('sw_sw_72h', bd.get('sw_72h',0)):.2f}"
# # # # # # # # #                     f"  swr={sw_r:.1f}{sw_warn}"
# # # # # # # # #                     f"  pos/fm={bd.get('lw_lw_pos_fm', bd.get('lw_pos_fm',0)):.2f}"
# # # # # # # # #                     f"  dw={bd.get('diff_w_mean',1):.2f}"
# # # # # # # # #                     f"  lr={lr:.2e}"
# # # # # # # # #                 )

# # # # # # # # #         avt=sl/nstep

# # # # # # # # #         # Val loss
# # # # # # # # #         model.eval(); vls=0.
# # # # # # # # #         with torch.no_grad():
# # # # # # # # #             for batch in vl:
# # # # # # # # #                 bv=move(list(batch),dev)
# # # # # # # # #                 with autocast(device_type="cuda",enabled=args.use_amp):
# # # # # # # # #                     vls+=model.get_loss(bv,epoch=ep).item()
# # # # # # # # #         avv=vls/len(vl); eps=time.perf_counter()-t0
# # # # # # # # #         lr_cur=opt.param_groups[0]["lr"]
# # # # # # # # #         print(f"  Epoch {ep:>3} | train={avt:.4f} val={avv:.4f} | lr={lr_cur:.2e} | {eps:.0f}s")

# # # # # # # # #         # Fast eval (subset)
# # # # # # # # #         rf=evaluate(model,vsub,dev,tag=f"FAST ep{ep}",steps=args.fast_ddim)
# # # # # # # # #         saver.update(rf,model,args.output_dir,ep,opt,sched,avt,avv,tag="fast")

# # # # # # # # #         # Full val every val_freq
# # # # # # # # #         if ep%args.val_freq==0:
# # # # # # # # #             em=getattr(_unwrap(model),"_ema",None)
# # # # # # # # #             rr=evaluate(model,vl,dev,tag=f"RAW ep{ep}",steps=args.full_ddim)
# # # # # # # # #             saver.update(rr,model,args.output_dir,ep,opt,sched,avt,avv,tag="raw")
# # # # # # # # #             if em and ep>=3:
# # # # # # # # #                 re=evaluate(model,vl,dev,tag=f"EMA ep{ep}",ema=em,steps=args.full_ddim)
# # # # # # # # #                 saver.update(re,model,args.output_dir,ep,opt,sched,avt,avv,tag="ema")

# # # # # # # # #         # Periodic checkpoint
# # # # # # # # #         if ep%10==0 or ep==args.num_epochs-1:
# # # # # # # # #             _save_ckpt(os.path.join(args.output_dir,f"ckpt_ep{ep:03d}.pth"),
# # # # # # # # #                        ep,model,opt,sched,saver,avt,avv)

# # # # # # # # #         if saver.stop: print(f"  Early stop ep={ep}"); break

# # # # # # # # #     th=(time.perf_counter()-ts)/3600.
# # # # # # # # #     print(f"\n  Best: ADE={saver.ba:.1f} 72h={saver.b7:.0f} "
# # # # # # # # #           f"ATE={saver.bat:.1f} CTE={saver.bc:.1f} ({th:.2f}h)")

# # # # # # # # #     # Post-training test
# # # # # # # # #     if args.eval_test_after_train:
# # # # # # # # #         print("\n"+"="*72+"\n  POST-TRAINING TEST\n"+"="*72)
# # # # # # # # #         try: _,tl2=data_loader(args,{"root":args.dataset_root,"type":"test"},test=True)
# # # # # # # # #         except: print("  No test set → using val"); tl2=vl
# # # # # # # # #         for fn,lb in [("best_composite.pth","COMPOSITE"),("best_72h.pth","72H"),
# # # # # # # # #                          ("best_ema.pth","EMA")]:
# # # # # # # # #             pp=os.path.join(args.output_dir,fn)
# # # # # # # # #             if not os.path.exists(pp): continue
# # # # # # # # #             ck=torch.load(pp,map_location=dev)
# # # # # # # # #             _unwrap(model).load_state_dict(ck["model_state_dict"],strict=False)
# # # # # # # # #             em=getattr(_unwrap(model),"_ema",None)
# # # # # # # # #             if em and ck.get("ema_shadow"):
# # # # # # # # #                 for k,v in ck["ema_shadow"].items():
# # # # # # # # #                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
# # # # # # # # #             r=evaluate(model,tl2,dev,tag=f"TEST/{lb}",steps=args.full_ddim)
# # # # # # # # #             print(f"\n  --- {lb} ---")
# # # # # # # # #             for key,ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
# # # # # # # # #                 v=r.get(key,float("nan"))
# # # # # # # # #                 mk="BEAT!" if np.isfinite(v) and v<ref else f"need {ref:.0f}"
# # # # # # # # #                 print(f"    {key:<10}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
# # # # # # # # #     print("="*72)


# # # # # # # # # if __name__ == "__main__":
# # # # # # # # #     args=get_args()
# # # # # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # # # # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# # # # # # # # #     main(args)

# # # # # # # # """
# # # # # # # # train_v75.py  TC-FlowMatching v75
# # # # # # # # ============================================================
# # # # # # # # TARGET: ADE<172.68 | 72h<321.39 | ATE<142.21 | CTE<42.04

# # # # # # # # BUGS FIXED vs previous version:
# # # # # # # #   BUG-A: sched.step() dùng conditional (chỉ step khi scaler không skip)
# # # # # # # #           → OneCycleLR bị lệch khi có NaN batches
# # # # # # # #           FIX: step() unconditionally mỗi batch

# # # # # # # #   BUG-B: grad_clip clips criterion.parameters() (step_weights, loss_weights)
# # # # # # # #           → step_weights học chậm vì gradient bị clip trước khi update
# # # # # # # #           FIX: clip backbone và criterion RIÊNG, criterion không bị clip

# # # # # # # #   BUG-C: BestModelSaver.cnt logic sai
# # # # # # # #           → cnt tăng dù ade/72h/ate/cte đã improve (chỉ score không improve)
# # # # # # # #           → early stop quá sớm
# # # # # # # #           FIX: cnt chỉ tăng khi KHÔNG CÓ GÌ improve

# # # # # # # #   BUG-D: val_loss gọi get_loss() với epoch>=0 → augmentation bị apply
# # # # # # # #           → val_loss bị noisy → scheduler nhận wrong signal
# # # # # # # #           FIX: gọi get_loss(batch, epoch=-1) → skip augmentation

# # # # # # # # RUN:
# # # # # # # #   python scripts/train_v75.py \\
# # # # # # # #       --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \\
# # # # # # # #       --output_dir   /kaggle/working/runs/v75 \\
# # # # # # # #       --batch_size   32 --use_amp
# # # # # # # # """
# # # # # # # # from __future__ import annotations
# # # # # # # # import argparse, json, math, os, random, sys, time
# # # # # # # # from collections import defaultdict
# # # # # # # # from typing import Dict, Optional

# # # # # # # # import numpy as np
# # # # # # # # import torch
# # # # # # # # import torch.nn as nn
# # # # # # # # import torch.optim as optim
# # # # # # # # from torch.amp import autocast, GradScaler
# # # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# # # # # # # # from Model.data.loader_training import data_loader
# # # # # # # # from Model.flow_matching_model import TCFlowMatching

# # # # # # # # try:
# # # # # # # #     from Model.utils import get_cosine_schedule_with_warmup
# # # # # # # # except ImportError:
# # # # # # # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # # # # # # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # # # # # # #         return CosineAnnealingLR(opt, T_max=max(total_steps,1), eta_min=min_lr)

# # # # # # # # try:
# # # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # # # except ImportError:
# # # # # # # #     seq_collate = None

# # # # # # # # TARGETS = {"ADE":172.68,"72h":321.39,"ATE":142.21,"CTE":42.04,
# # # # # # # #            "12h":65.42,"24h":104.67,"48h":205.10}
# # # # # # # # R_EARTH = 6371.0


# # # # # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # # # # def _unwrap(m):
# # # # # # # #     return m._orig_mod if hasattr(m,'_orig_mod') else m

# # # # # # # # def move(batch, device):
# # # # # # # #     out=list(batch)
# # # # # # # #     for i,x in enumerate(out):
# # # # # # # #         if torch.is_tensor(x): out[i]=x.to(device)
# # # # # # # #         elif isinstance(x,dict): out[i]={k:v.to(device) if torch.is_tensor(v) else v for k,v in x.items()}
# # # # # # # #     return out

# # # # # # # # def set_seed(seed=42):
# # # # # # # #     random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
# # # # # # # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# # # # # # # # # ── Metrics ───────────────────────────────────────────────────────────────────

# # # # # # # # def _ntd(a):
# # # # # # # #     o=a.clone(); o[...,0]=(a[...,0]*50.+1800.)/10.; o[...,1]=(a[...,1]*50.)/10.
# # # # # # # #     return o

# # # # # # # # def _hav(p1,p2):
# # # # # # # #     la1=torch.deg2rad(p1[...,1]); la2=torch.deg2rad(p2[...,1])
# # # # # # # #     dlat=torch.deg2rad(p2[...,1]-p1[...,1]); dlon=torch.deg2rad(p2[...,0]-p1[...,0])
# # # # # # # #     a=torch.sin(dlat/2).pow(2)+torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2)
# # # # # # # #     return 2.*R_EARTH*torch.asin(a.clamp(1e-12,1-1e-12).sqrt())

# # # # # # # # def _atecte(pd,gd):
# # # # # # # #     T=min(pd.shape[0],gd.shape[0])
# # # # # # # #     if T<2: z=pd.new_zeros(1,pd.shape[1]); return z,z
# # # # # # # #     lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
# # # # # # # #     lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
# # # # # # # #     lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
# # # # # # # #     ya=torch.sin(lo2-lo1)*torch.cos(la2)
# # # # # # # #     xa=torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # # # # # # #     be=torch.atan2(ya,xa)
# # # # # # # #     ye=torch.sin(lo3-lo2)*torch.cos(la3)
# # # # # # # #     xe=torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # # # # # # #     bee=torch.atan2(ye,xe); tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
# # # # # # # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # # # # # class Acc:
# # # # # # # #     def __init__(self):
# # # # # # # #         self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list)
# # # # # # # #         self._h={12:1,24:3,48:7,72:11}
# # # # # # # #     def update(self,dist,ate=None,cte=None):
# # # # # # # #         self.d.extend(dist.mean(0).tolist())
# # # # # # # #         for h,s in self._h.items():
# # # # # # # #             if s<dist.shape[0]: self.sd[h].extend(dist[s].tolist())
# # # # # # # #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# # # # # # # #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())
# # # # # # # #     def compute(self):
# # # # # # # #         r={"ADE":float(np.mean(self.d)) if self.d else float("nan"),
# # # # # # # #            "ATE_mean":float(np.mean(self.a)) if self.a else float("nan"),
# # # # # # # #            "CTE_mean":float(np.mean(self.c)) if self.c else float("nan"),
# # # # # # # #            "n":len(self.d)}
# # # # # # # #         for h in self._h:
# # # # # # # #             v=self.sd.get(h,[]); r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
# # # # # # # #         return r


# # # # # # # # def _score(r):
# # # # # # # #     ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
# # # # # # # #     ate=r.get("ATE_mean",1e9); cte=r.get("CTE_mean",1e9)
# # # # # # # #     if not np.isfinite(ate): ate=ade*.46
# # # # # # # #     if not np.isfinite(cte): cte=ade*.53
# # # # # # # #     return 100.*(0.05*(ade/136.)+0.10*(r.get("12h",ade)/50.)+
# # # # # # # #                  0.15*(r.get("24h",ade)/100.)+0.20*(r.get("48h",ade)/200.)+
# # # # # # # #                  0.25*(h72/300.)+0.13*(ate/80.)+0.12*(cte/94.))

# # # # # # # # def _beat(r):
# # # # # # # #     p=[]
# # # # # # # #     for k,t in [("ADE",172.68),("ATE_mean",142.21),("CTE_mean",42.04),
# # # # # # # #                   ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
# # # # # # # #         v=r.get(k,1e9)
# # # # # # # #         if np.isfinite(v) and v<t: p.append(f"{k.replace('_mean','')}:{v:.1f}")
# # # # # # # #     return "*** BEAT ST-TRANS: "+" ".join(p)+" ***" if p else ""

# # # # # # # # def _gap(r):
# # # # # # # #     out=[]
# # # # # # # #     for k,ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
# # # # # # # #         v=r.get(k,float("nan"))
# # # # # # # #         if np.isfinite(v):
# # # # # # # #             out.append(f"{k.replace('_mean','')}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
# # # # # # # #     return " | ".join(out)


# # # # # # # # # ── Evaluate ──────────────────────────────────────────────────────────────────

# # # # # # # # @torch.no_grad()
# # # # # # # # def evaluate(model, loader, device, tag="", ema=None, steps=20):
# # # # # # # #     """
# # # # # # # #     FIX-D: apply EMA SAU khi set model.eval().
# # # # # # # #     EMA apply() là copy weights → không ảnh hưởng eval mode.
# # # # # # # #     """
# # # # # # # #     # Set eval mode TRƯỚC
# # # # # # # #     model.eval()

# # # # # # # #     # Apply EMA weights nếu có (SAU eval mode)
# # # # # # # #     bk=None
# # # # # # # #     if ema is not None:
# # # # # # # #         try: bk=ema.apply_to(model)
# # # # # # # #         except Exception: bk=None

# # # # # # # #     acc=Acc(); t0=time.perf_counter()
# # # # # # # #     for b in loader:
# # # # # # # #         bl=move(list(b),device)
# # # # # # # #         # use_ema_for_inference=False vì đã apply EMA manually ở trên
# # # # # # # #         result=model.sample(bl,ddim_steps=steps,use_ema_for_inference=False)
# # # # # # # #         p=result[0] if isinstance(result,(tuple,list)) else result
# # # # # # # #         g=bl[1]; T=min(p.shape[0],g.shape[0])
# # # # # # # #         pd=_ntd(p[:T]); gd=_ntd(g[:T])
# # # # # # # #         dist=_hav(pd,gd); at,ct=_atecte(pd,gd)
# # # # # # # #         acc.update(dist,at,ct)

# # # # # # # #     if bk is not None:
# # # # # # # #         try: ema.restore(model,bk)
# # # # # # # #         except Exception: pass

# # # # # # # #     r=acc.compute()
# # # # # # # #     def _v(k): return r.get(k,float("nan"))
# # # # # # # #     def _m(v,t): return "ok" if np.isfinite(v) and v<t else "no"
# # # # # # # #     el=time.perf_counter()-t0
# # # # # # # #     print(f"\n{'='*70}")
# # # # # # # #     print(f"  [{tag}  {el:.0f}s]")
# # # # # # # #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]  "
# # # # # # # #           f"12h={_v('12h'):.0f}  24h={_v('24h'):.0f}  "
# # # # # # # #           f"48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
# # # # # # # #     if np.isfinite(_v("ATE_mean")):
# # # # # # # #         print(f"  ATE={_v('ATE_mean'):.1f}[{_m(_v('ATE_mean'),142.21)}]  "
# # # # # # # #               f"CTE={_v('CTE_mean'):.1f}[{_m(_v('CTE_mean'),42.04)}]")
# # # # # # # #     print(f"  vs ST-Trans: {_gap(r)}")
# # # # # # # #     bt=_beat(r)
# # # # # # # #     if bt: print(f"  {bt}")
# # # # # # # #     print(f"  Score={_score(r):.2f}")
# # # # # # # #     print(f"{'='*70}\n")
# # # # # # # #     return r


# # # # # # # # # ── Checkpoint ────────────────────────────────────────────────────────────────

# # # # # # # # def _save(path,ep,model,opt,sched,saver,tl,vl,extra=None):
# # # # # # # #     m=_unwrap(model); ema=getattr(m,'_ema',None); esd=None
# # # # # # # #     if ema and hasattr(ema,'shadow'):
# # # # # # # #         try: esd={k:v.cpu().clone() for k,v in ema.shadow.items()}
# # # # # # # #         except: pass
# # # # # # # #     d={"epoch":ep,"model_state_dict":m.state_dict(),
# # # # # # # #        "optimizer_state":opt.state_dict(),
# # # # # # # #        "scheduler_state":sched.state_dict() if sched else {},
# # # # # # # #        "ema_shadow":esd,"best_score":saver.bs,"best_ade":saver.ba,
# # # # # # # #        "best_72h":saver.b7,"best_ate":saver.bat,"best_cte":saver.bc,
# # # # # # # #        "train_loss":tl,"val_loss":vl}
# # # # # # # #     if extra: d.update(extra)
# # # # # # # #     os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
# # # # # # # #     torch.save(d,path)


# # # # # # # # # ── BestModelSaver ────────────────────────────────────────────────────────────

# # # # # # # # class BestModelSaver:
# # # # # # # #     """
# # # # # # # #     FIX-C: cnt chỉ tăng khi KHÔNG CÓ GÌ improve (kể cả individual metrics).
# # # # # # # #     v74 bug: cnt tăng dù ade/72h/ate/cte đã improve → early stop quá sớm.
# # # # # # # #     """
# # # # # # # #     def __init__(self, patience=40, min_ep=30):
# # # # # # # #         self.patience=patience; self.min_ep=min_ep
# # # # # # # #         self.cnt=0; self.stop=False
# # # # # # # #         self.bs=self.ba=self.b7=self.bat=self.bc=float("inf")

# # # # # # # #     def update(self,r,model,out,ep,opt,sched,tl,vl,tag=""):
# # # # # # # #         sc=_score(r)
# # # # # # # #         ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
# # # # # # # #         ate=r.get("ATE_mean",1e9); cte=r.get("CTE_mean",1e9)

# # # # # # # #         any_improved=False
# # # # # # # #         # Save per-metric bests
# # # # # # # #         for v,attr,fname in [(ade,"ba","best_ade.pth"),(h72,"b7","best_72h.pth"),
# # # # # # # #                               (ate,"bat","best_ate.pth"),(cte,"bc","best_cte.pth")]:
# # # # # # # #             if v<getattr(self,attr):
# # # # # # # #                 setattr(self,attr,v)
# # # # # # # #                 _save(os.path.join(out,fname),ep,model,opt,sched,self,tl,vl)
# # # # # # # #                 any_improved=True

# # # # # # # #         # Save composite best
# # # # # # # #         if sc<self.bs:
# # # # # # # #             self.bs=sc; any_improved=True
# # # # # # # #             _save(os.path.join(out,f"best_{tag or 'composite'}.pth"),
# # # # # # # #                   ep,model,opt,sched,self,tl,vl,{"score":sc,"ade":ade,"h72":h72})
# # # # # # # #             print(f"  [BEST] {tag} ep={ep} score={sc:.2f} "
# # # # # # # #                   f"ADE={ade:.1f} 72h={h72:.0f} ATE={ate:.1f} CTE={cte:.1f}")

# # # # # # # #         # FIX-C: cnt chỉ tăng khi KHÔNG CÓ GÌ improve
# # # # # # # #         if any_improved:
# # # # # # # #             self.cnt=0
# # # # # # # #         else:
# # # # # # # #             self.cnt+=1
# # # # # # # #             print(f"  No improve {self.cnt}/{self.patience}  best={self.bs:.2f}  cur={sc:.2f}")
# # # # # # # #             if ep>=self.min_ep and self.cnt>=self.patience:
# # # # # # # #                 self.stop=True; print(f"  [EARLY STOP] ep={ep}")

# # # # # # # #         return any_improved


# # # # # # # # # ── Dataset helpers ───────────────────────────────────────────────────────────

# # # # # # # # def make_subset_loader(ds,n,bs,cf):
# # # # # # # #     idx=random.Random(42).sample(range(len(ds)),min(n,len(ds)))
# # # # # # # #     return DataLoader(Subset(ds,idx),batch_size=bs,shuffle=False,collate_fn=cf,num_workers=0,drop_last=False)


# # # # # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # # # # def get_args():
# # # # # # # #     p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # # # #     p.add_argument("--dataset_root",  default="TCND_vn")
# # # # # # # #     p.add_argument("--obs_len",       default=8,    type=int)
# # # # # # # #     p.add_argument("--pred_len",      default=12,   type=int)
# # # # # # # #     p.add_argument("--batch_size",    default=32,   type=int)
# # # # # # # #     p.add_argument("--num_epochs",    default=120,  type=int)
# # # # # # # #     p.add_argument("--learning_rate", default=1e-4, type=float)
# # # # # # # #     p.add_argument("--min_lr",        default=1e-6, type=float)
# # # # # # # #     p.add_argument("--warmup_epochs", default=4,    type=int)
# # # # # # # #     p.add_argument("--weight_decay",  default=1e-3, type=float)
# # # # # # # #     p.add_argument("--grad_clip",     default=1.0,  type=float)
# # # # # # # #     p.add_argument("--patience",      default=40,   type=int)
# # # # # # # #     p.add_argument("--min_ep",        default=30,   type=int)
# # # # # # # #     p.add_argument("--sigma_min",     default=0.02, type=float)
# # # # # # # #     p.add_argument("--use_ot",        default=True, action="store_true")
# # # # # # # #     p.add_argument("--no_ot",         dest="use_ot",action="store_false")
# # # # # # # #     p.add_argument("--use_ema",       default=True, action="store_true")
# # # # # # # #     p.add_argument("--no_ema",        dest="use_ema",action="store_false")
# # # # # # # #     p.add_argument("--ema_decay",     default=0.995,type=float)
# # # # # # # #     p.add_argument("--n_ensemble",    default=50,   type=int)
# # # # # # # #     p.add_argument("--fast_ddim",     default=10,   type=int)
# # # # # # # #     p.add_argument("--full_ddim",     default=20,   type=int)
# # # # # # # #     p.add_argument("--val_freq",      default=3,    type=int)
# # # # # # # #     p.add_argument("--val_subset",    default=500,  type=int)
# # # # # # # #     p.add_argument("--use_amp",       action="store_true")
# # # # # # # #     p.add_argument("--num_workers",   default=2,    type=int)
# # # # # # # #     p.add_argument("--output_dir",    default="runs/v75")
# # # # # # # #     p.add_argument("--gpu_num",       default="0")
# # # # # # # #     p.add_argument("--resume",        default=None)
# # # # # # # #     p.add_argument("--resume_epoch",  default=None, type=int)
# # # # # # # #     p.add_argument("--eval_test",     default=True, action="store_true")
# # # # # # # #     p.add_argument("--delim",         default=" ")
# # # # # # # #     p.add_argument("--skip",          default=1,    type=int)
# # # # # # # #     p.add_argument("--min_ped",       default=1,    type=int)
# # # # # # # #     p.add_argument("--threshold",     default=0.002,type=float)
# # # # # # # #     p.add_argument("--other_modal",   default="gph")
# # # # # # # #     p.add_argument("--test_year",     default=None, type=int)
# # # # # # # #     return p.parse_args()


# # # # # # # # # ── Main ──────────────────────────────────────────────────────────────────────

# # # # # # # # def main(args):
# # # # # # # #     if torch.cuda.is_available(): os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_num)
# # # # # # # #     dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # #     os.makedirs(args.output_dir,exist_ok=True); set_seed(42)

# # # # # # # #     print("="*72)
# # # # # # # #     print(f"  TC-FlowMatching v75")
# # # # # # # #     print(f"  Loss: ALL weights self-learned")
# # # # # # # #     print(f"  sw_pos ratio>=5x | sw_disp ratio>=3x | w_pos_l init=3x > w_pos_e=1.5x")
# # # # # # # #     print(f"  Fixes: sched unconditional | grad_clip separate | cnt logic | val aug off")
# # # # # # # #     print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']} ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
# # # # # # # #     print("="*72)

# # # # # # # #     # Datasets
# # # # # # # #     trd,trl=data_loader(args,{"root":args.dataset_root,"type":"train"},test=False)
# # # # # # # #     vd, vl =data_loader(args,{"root":args.dataset_root,"type":"val"},  test=True)
# # # # # # # #     cf=seq_collate
# # # # # # # #     vsub=make_subset_loader(vd,args.val_subset,args.batch_size,cf)
# # # # # # # #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# # # # # # # #     # Model
# # # # # # # #     model=TCFlowMatching(pred_len=args.pred_len,obs_len=args.obs_len,sigma_min=args.sigma_min,
# # # # # # # #                           use_ema=args.use_ema,ema_decay=args.ema_decay,use_ot=args.use_ot,
# # # # # # # #                           cfg_uncond_prob=0.1).to(dev)
# # # # # # # #     model.init_ema()
# # # # # # # #     n_p=sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # # #     n_sw=sum(p.numel() for p in model.criterion.step_weights.parameters())
# # # # # # # #     n_lw=sum(p.numel() for p in model.criterion.loss_weights.parameters())
# # # # # # # #     print(f"  params: {n_p:,} | step_w:{n_sw} | loss_w:{n_lw}")

# # # # # # # #     # 4 param groups
# # # # # # # #     m=_unwrap(model)
# # # # # # # #     speed_ids ={id(p) for p in m.net.speed_head.parameters()}
# # # # # # # #     step_ids  ={id(p) for p in m.criterion.step_weights.parameters()}
# # # # # # # #     loss_w_ids={id(p) for p in m.criterion.loss_weights.parameters()}
# # # # # # # #     # diff_weighter and scorer go in backbone group (standard LR)

# # # # # # # #     backbone_p=[]; speed_p=[]; step_w_p=[]; loss_w_p=[]
# # # # # # # #     for p in model.parameters():
# # # # # # # #         if not p.requires_grad: continue
# # # # # # # #         pid=id(p)
# # # # # # # #         if pid in step_ids:   step_w_p.append(p)
# # # # # # # #         elif pid in loss_w_ids: loss_w_p.append(p)
# # # # # # # #         elif pid in speed_ids:  speed_p.append(p)
# # # # # # # #         else:                   backbone_p.append(p)

# # # # # # # #     opt=optim.AdamW([
# # # # # # # #         {"params":backbone_p,  "lr":args.learning_rate,       "weight_decay":args.weight_decay},
# # # # # # # #         {"params":speed_p,     "lr":args.learning_rate*5.,    "weight_decay":0.0},
# # # # # # # #         {"params":step_w_p,    "lr":args.learning_rate*10.,   "weight_decay":0.0},
# # # # # # # #         {"params":loss_w_p,    "lr":args.learning_rate*5.,    "weight_decay":0.0},
# # # # # # # #     ])

# # # # # # # #     nstep=len(trl); total_steps=nstep*args.num_epochs
# # # # # # # #     sched=optim.lr_scheduler.OneCycleLR(
# # # # # # # #         opt,
# # # # # # # #         max_lr=[args.learning_rate, args.learning_rate*5.,
# # # # # # # #                 args.learning_rate*10., args.learning_rate*5.],
# # # # # # # #         total_steps=total_steps,
# # # # # # # #         pct_start=args.warmup_epochs/args.num_epochs,
# # # # # # # #         anneal_strategy='cos', div_factor=25., final_div_factor=1e4,
# # # # # # # #         three_phase=False)

# # # # # # # #     scaler=GradScaler("cuda",enabled=args.use_amp)
# # # # # # # #     saver=BestModelSaver(patience=args.patience,min_ep=args.min_ep)

# # # # # # # #     # Resume
# # # # # # # #     start=0
# # # # # # # #     if args.resume and os.path.exists(args.resume):
# # # # # # # #         print(f"  Loading: {args.resume}")
# # # # # # # #         ck=torch.load(args.resume,map_location=dev); mu=_unwrap(model)
# # # # # # # #         missing,_=mu.load_state_dict(ck["model_state_dict"],strict=False)
# # # # # # # #         if missing: print(f"  Missing: {len(missing)}")
# # # # # # # #         ema_obj=getattr(mu,'_ema',None)
# # # # # # # #         if ema_obj and ck.get("ema_shadow"):
# # # # # # # #             for k,v in ck["ema_shadow"].items():
# # # # # # # #                 if k in ema_obj.shadow: ema_obj.shadow[k].copy_(v.to(dev))
# # # # # # # #         try: opt.load_state_dict(ck["optimizer_state"])
# # # # # # # #         except Exception as e: print(f"  Opt skip: {e}")
# # # # # # # #         try: sched.load_state_dict(ck["scheduler_state"])
# # # # # # # #         except Exception:
# # # # # # # #             ep0=args.resume_epoch or ck.get("epoch",0)
# # # # # # # #             for _ in range(ep0*nstep):
# # # # # # # #                 try: sched.step()
# # # # # # # #                 except: break
# # # # # # # #         for a,attr in [("best_score","bs"),("best_ade","ba"),("best_72h","b7"),
# # # # # # # #                          ("best_ate","bat"),("best_cte","bc")]:
# # # # # # # #             if a in ck: setattr(saver,attr,ck[a])
# # # # # # # #         start=args.resume_epoch or ck.get("epoch",0)+1
# # # # # # # #         print(f"  Resumed ep={start}")

# # # # # # # #     try:
# # # # # # # #         model=torch.compile(model,mode="reduce-overhead"); print("  torch.compile: ok")
# # # # # # # #     except: pass

# # # # # # # #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# # # # # # # #     print("="*72)

# # # # # # # #     ts=time.perf_counter()

# # # # # # # #     for ep in range(start,args.num_epochs):
# # # # # # # #         model.train(); sl=0.; n=0; t0=time.perf_counter()

# # # # # # # #         for i,batch in enumerate(trl):
# # # # # # # #             bl=move(list(batch),dev)
# # # # # # # #             with autocast(device_type="cuda",enabled=args.use_amp):
# # # # # # # #                 bd=model.get_loss_breakdown(bl,epoch=ep)

# # # # # # # #             opt.zero_grad()
# # # # # # # #             scaler.scale(bd["total"]).backward()
# # # # # # # #             scaler.unscale_(opt)

# # # # # # # #             # FIX-B: clip backbone và criterion RIÊNG
# # # # # # # #             # step_weights cần gradient không bị clip → học nhanh ratio>=5x
# # # # # # # #             nn.utils.clip_grad_norm_(backbone_p+speed_p, args.grad_clip)
# # # # # # # #             # criterion params (step_w, loss_w) KHÔNG clip → tự học tự do
# # # # # # # #             # nhưng clamp bounds trong forward() đảm bảo không explode

# # # # # # # #             scaler.step(opt); scaler.update()

# # # # # # # #             # FIX-A: step() unconditionally sau mỗi batch
# # # # # # # #             # OneCycleLR phải được step mỗi batch, không phải mỗi epoch
# # # # # # # #             try: sched.step()
# # # # # # # #             except Exception: pass

# # # # # # # #             model.ema_update()

# # # # # # # #             if torch.isfinite(bd["total"]): sl+=bd["total"].item(); n+=1

# # # # # # # #             if i%20==0:
# # # # # # # #                 lr=opt.param_groups[0]["lr"]
# # # # # # # #                 sw_r=bd.get("sw_sw_ratio",bd.get("sw_pos_ratio",0.))
# # # # # # # #                 sw_72=bd.get("sw_sw_72h",bd.get("sw_pos_72h",0.))
# # # # # # # #                 lw_pl=bd.get("lw_lw_pos_l",bd.get("lw_pos_l",0.))
# # # # # # # #                 lw_fm=bd.get("lw_lw_fm",bd.get("lw_fm",1.))
# # # # # # # #                 ratio_pl_fm=lw_pl/max(lw_fm,1e-6)
# # # # # # # #                 w1=" [!swr<4]" if sw_r<4. else ""
# # # # # # # #                 w2=" [!pl/fm<2]" if ratio_pl_fm<2. else ""
# # # # # # # #                 w3=" [!tot>10]" if bd["total"].item()>10. else ""
# # # # # # # #                 print(
# # # # # # # #                     f"  [{ep:>3}][{i:>3}/{nstep}]"
# # # # # # # #                     f"  tot={bd['total'].item():.3f}{w3}"
# # # # # # # #                     f"  fm={bd.get('l_fm',0.):.4f}"
# # # # # # # #                     f"  pos_e={bd.get('l_pos_e',0.):.4f}"
# # # # # # # #                     f"  pos_l={bd.get('l_pos_l',0.):.4f}"
# # # # # # # #                     f"  disp={bd.get('l_disp',0.):.4f}"
# # # # # # # #                     f"  ep72={bd.get('l_ep',0.):.4f}"
# # # # # # # #                     f"  sw72={sw_72:.2f} swr={sw_r:.2f}{w1}"
# # # # # # # #                     f"  pl/fm={ratio_pl_fm:.2f}{w2}"
# # # # # # # #                     f"  dw={bd.get('diff_w_mean',1.):.2f}"
# # # # # # # #                     f"  lr={lr:.2e}"
# # # # # # # #                 )

# # # # # # # #         avt=sl/max(n,1); eps=time.perf_counter()-t0

# # # # # # # #         # FIX-D: val_loss với epoch=-1 → skip augmentation → clean val signal
# # # # # # # #         model.eval(); vls=0.
# # # # # # # #         with torch.no_grad():
# # # # # # # #             for batch in vl:
# # # # # # # #                 bv=move(list(batch),dev)
# # # # # # # #                 with autocast(device_type="cuda",enabled=args.use_amp):
# # # # # # # #                     vls+=model.get_loss(bv,epoch=-1).item()  # epoch=-1: no aug
# # # # # # # #         avv=vls/max(len(vl),1)

# # # # # # # #         lr_cur=opt.param_groups[0]["lr"]
# # # # # # # #         print(f"  Epoch {ep:>3} | train={avt:.4f} val={avv:.4f} | lr={lr_cur:.2e} | {eps:.0f}s")

# # # # # # # #         # Fast eval (subset, raw weights) — monitoring only
# # # # # # # #         rf=evaluate(model,vsub,dev,tag=f"FAST ep{ep}",steps=args.fast_ddim)

# # # # # # # #         # Full eval mỗi val_freq epochs
# # # # # # # #         if ep%args.val_freq==0:
# # # # # # # #             ema_obj=getattr(_unwrap(model),'_ema',None)

# # # # # # # #             # EMA FIRST — primary signal (tốt hơn raw 5-10đ theo v74 log)
# # # # # # # #             if ema_obj is not None and ep>=2:
# # # # # # # #                 re=evaluate(model,vl,dev,tag=f"EMA ep{ep}",ema=ema_obj,steps=args.full_ddim)
# # # # # # # #                 saver.update(re,model,args.output_dir,ep,opt,sched,avt,avv,tag="ema")

# # # # # # # #             # RAW second — secondary
# # # # # # # #             rr=evaluate(model,vl,dev,tag=f"RAW ep{ep}",steps=args.full_ddim)
# # # # # # # #             saver.update(rr,model,args.output_dir,ep,opt,sched,avt,avv,tag="raw")
# # # # # # # #         else:
# # # # # # # #             # Non-full-val epochs: fast metrics
# # # # # # # #             saver.update(rf,model,args.output_dir,ep,opt,sched,avt,avv,tag="fast")

# # # # # # # #         # Periodic checkpoint
# # # # # # # #         if ep%10==0 or ep==args.num_epochs-1:
# # # # # # # #             _save(os.path.join(args.output_dir,f"ckpt_ep{ep:03d}.pth"),
# # # # # # # #                   ep,model,opt,sched,saver,avt,avv)

# # # # # # # #         # Weight stats every 5 epochs
# # # # # # # #         if ep%5==0:
# # # # # # # #             crit=_unwrap(model).criterion
# # # # # # # #             sw_s=crit.step_weights.stats(); lw_s=crit.loss_weights.stats()
# # # # # # # #             print(f"  [W ep{ep}]"
# # # # # # # #                   f" swr={sw_s['sw_pos_ratio']:.2f}"
# # # # # # # #                   f" sw72={sw_s['sw_pos_72h']:.3f}"
# # # # # # # #                   f" wfm={lw_s['lw_fm']:.3f}"
# # # # # # # #                   f" wpe={lw_s['lw_pos_e']:.3f}"
# # # # # # # #                   f" wpl={lw_s['lw_pos_l']:.3f}"
# # # # # # # #                   f" wdi={lw_s['lw_disp']:.3f}"
# # # # # # # #                   f" wep={lw_s['lw_ep']:.3f}"
# # # # # # # #                   f" pl/pe={lw_s['lw_pos_l_e_ratio']:.2f}"
# # # # # # # #                   f" pl/fm={lw_s['lw_pos_l']/max(lw_s['lw_fm'],1e-6):.2f}")

# # # # # # # #         if saver.stop: print(f"  [EARLY STOP] ep={ep}"); break

# # # # # # # #     th=(time.perf_counter()-ts)/3600.
# # # # # # # #     print(f"\n  Best: ADE={saver.ba:.1f} 72h={saver.b7:.0f} ATE={saver.bat:.1f} CTE={saver.bc:.1f} ({th:.2f}h)")

# # # # # # # #     # Post-training test
# # # # # # # #     if args.eval_test:
# # # # # # # #         print("\n"+"="*72+"\n  POST-TRAINING TEST\n"+"="*72)
# # # # # # # #         try: _,tl2=data_loader(args,{"root":args.dataset_root,"type":"test"},test=True)
# # # # # # # #         except: print("  No test set → using val"); tl2=vl
# # # # # # # #         for fname,label in [("best_ema.pth","EMA"),("best_composite.pth","COMPOSITE"),
# # # # # # # #                               ("best_72h.pth","72H"),("best_ade.pth","ADE")]:
# # # # # # # #             pp=os.path.join(args.output_dir,fname)
# # # # # # # #             if not os.path.exists(pp): continue
# # # # # # # #             ck=torch.load(pp,map_location=dev); mu=_unwrap(model)
# # # # # # # #             mu.load_state_dict(ck["model_state_dict"],strict=False)
# # # # # # # #             ema_obj=getattr(mu,'_ema',None)
# # # # # # # #             if ema_obj and ck.get("ema_shadow"):
# # # # # # # #                 for k,v in ck["ema_shadow"].items():
# # # # # # # #                     if k in ema_obj.shadow: ema_obj.shadow[k].copy_(v.to(dev))
# # # # # # # #             r=evaluate(model,tl2,dev,tag=f"TEST/{label}",steps=args.full_ddim)
# # # # # # # #             print(f"\n  --- {label} ---")
# # # # # # # #             for key,ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
# # # # # # # #                 v=r.get(key,float("nan"))
# # # # # # # #                 if np.isfinite(v):
# # # # # # # #                     beat="BEAT" if v<ref else "miss"
# # # # # # # #                     print(f"    {key:<10}: {v:>8.1f}  [{beat}  gap:{v-ref:+.1f}]")
# # # # # # # #     print("="*72)


# # # # # # # # if __name__=="__main__":
# # # # # # # #     args=get_args()
# # # # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # # # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# # # # # # # #     main(args)

# # # # # # # """
# # # # # # # train_v74.py — TC-FlowMatching v74  (FIXED)
# # # # # # # ════════════════════════════════════════════
# # # # # # # FIXES vs original:
# # # # # # #   BUG-1: Saver.update() gọi 3 lần/epoch → patience giảm 2x thực tế
# # # # # # #          FIX: Saver chỉ track best score 1 lần/epoch (best_of_epoch pattern)
# # # # # # #   BUG-2: Score formula weight ATE/CTE quá nhỏ dù gap lớn nhất
# # # # # # #          FIX: Reweight theo gap size relative to target
# # # # # # #   BUG-3: Resume không override LR được
# # # # # # #          FIX: Thêm --finetune_lr để set LR mới sau resume

# # # # # # # RUN (fresh):
# # # # # # #   python scripts/train_v74.py \
# # # # # # #       --dataset_root /path/to/tc-ofm \
# # # # # # #       --output_dir   runs/v74 \
# # # # # # #       --batch_size   32 \
# # # # # # #       --use_amp

# # # # # # # RESUME từ best checkpoint:
# # # # # # #   python scripts/train_v74.py \
# # # # # # #       --dataset_root /path/to/tc-ofm \
# # # # # # #       --output_dir   runs/v74_resume \
# # # # # # #       --resume       runs/v74/best_composite.pth \
# # # # # # #       --finetune_lr  5e-5 \
# # # # # # #       --batch_size   32 \
# # # # # # #       --use_amp

# # # # # # # TARGETS (ST-Trans thực tế trên val):
# # # # # # #   ADE<172.68  72h<321.39  ATE<142.21  CTE<42.04
# # # # # # #   12h<65.42   24h<104.67  48h<205.10
# # # # # # # """
# # # # # # # from __future__ import annotations

# # # # # # # import sys, os
# # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # import argparse, time, random
# # # # # # # from collections import defaultdict

# # # # # # # import numpy as np
# # # # # # # import torch
# # # # # # # import torch.optim as optim
# # # # # # # from torch.amp import autocast, GradScaler
# # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # from Model.data.loader_training import data_loader
# # # # # # # from Model.flow_matching_model import TCFlowMatching

# # # # # # # try:
# # # # # # #     from Model.utils import get_cosine_schedule_with_warmup
# # # # # # # except ImportError:
# # # # # # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # # # # # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # # # # # #         return CosineAnnealingLR(opt, T_max=max(total_steps,1), eta_min=min_lr)

# # # # # # # # ST-Trans thực tế trên val (không phải paper)
# # # # # # # TARGETS = {
# # # # # # #     "ADE":   172.68,
# # # # # # #     "72h":   321.39,
# # # # # # #     "ATE":   142.21,
# # # # # # #     "CTE":    42.04,
# # # # # # #     "12h":    65.42,
# # # # # # #     "24h":   104.67,
# # # # # # #     "48h":   205.10,
# # # # # # # }
# # # # # # # R_EARTH = 6371.0


# # # # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # # # def _unwrap(m):
# # # # # # #     return m._orig_mod if hasattr(m, '_orig_mod') else m

# # # # # # # def move(b, dev):
# # # # # # #     out = list(b)
# # # # # # #     for i, x in enumerate(out):
# # # # # # #         if torch.is_tensor(x):
# # # # # # #             out[i] = x.to(dev)
# # # # # # #         elif isinstance(x, dict):
# # # # # # #             out[i] = {k: v.to(dev) if torch.is_tensor(v) else v
# # # # # # #                       for k, v in x.items()}
# # # # # # #     return out


# # # # # # # # ── Metric utils ──────────────────────────────────────────────────────────────

# # # # # # # def _ntd(a):
# # # # # # #     o = a.clone()
# # # # # # #     o[..., 0] = (a[..., 0] * 50. + 1800.) / 10.
# # # # # # #     o[..., 1] = (a[..., 1] * 50.) / 10.
# # # # # # #     return o

# # # # # # # def _hav(p1, p2):
# # # # # # #     la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
# # # # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # # # #     a = (torch.sin(dlat/2).pow(2)
# # # # # # #          + torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2))
# # # # # # #     return 2.*R_EARTH*torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

# # # # # # # def _atecte(pd, gd):
# # # # # # #     T = min(pd.shape[0], gd.shape[0])
# # # # # # #     if T < 2:
# # # # # # #         z = pd.new_zeros(1, pd.shape[1])
# # # # # # #         return z, z
# # # # # # #     lo1 = torch.deg2rad(gd[:T-1, :, 0]); la1 = torch.deg2rad(gd[:T-1, :, 1])
# # # # # # #     lo2 = torch.deg2rad(gd[1:T,  :, 0]); la2 = torch.deg2rad(gd[1:T,  :, 1])
# # # # # # #     lo3 = torch.deg2rad(pd[1:T,  :, 0]); la3 = torch.deg2rad(pd[1:T,  :, 1])
# # # # # # #     ya  = torch.sin(lo2-lo1)*torch.cos(la2)
# # # # # # #     xa  = torch.cos(la1)*torch.sin(la2) - torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # # # # # #     be  = torch.atan2(ya, xa)
# # # # # # #     ye  = torch.sin(lo3-lo2)*torch.cos(la3)
# # # # # # #     xe  = torch.cos(la2)*torch.sin(la3) - torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # # # # # #     bee = torch.atan2(ye, xe)
# # # # # # #     tot = _hav(pd[1:T], gd[1:T])
# # # # # # #     ang = bee - be
# # # # # # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # # # # class Acc:
# # # # # # #     def __init__(self):
# # # # # # #         self.d = []; self.a = []; self.c = []
# # # # # # #         self.sd = defaultdict(list)
# # # # # # #         self._h = {12: 1, 24: 3, 48: 7, 72: 11}

# # # # # # #     def update(self, dist, ate=None, cte=None):
# # # # # # #         self.d.extend(dist.mean(0).tolist())
# # # # # # #         for h, s in self._h.items():
# # # # # # #             if s < dist.shape[0]:
# # # # # # #                 self.sd[h].extend(dist[s].tolist())
# # # # # # #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# # # # # # #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())

# # # # # # #     def compute(self):
# # # # # # #         r = {
# # # # # # #             "ADE": float(np.mean(self.d))  if self.d else float("nan"),
# # # # # # #             "ATE": float(np.mean(self.a))  if self.a else float("nan"),
# # # # # # #             "CTE": float(np.mean(self.c))  if self.c else float("nan"),
# # # # # # #             "n":   len(self.d),
# # # # # # #         }
# # # # # # #         for h in self._h:
# # # # # # #             v = self.sd.get(h, [])
# # # # # # #             r[f"{h}h"] = float(np.mean(v)) if v else float("nan")
# # # # # # #         return r


# # # # # # # def _score(r):
# # # # # # #     """
# # # # # # #     FIX BUG-2: Reweight score theo gap size từ targets thực tế.
# # # # # # #     ATE gap=24km (lớn nhất) → weight cao hơn
# # # # # # #     CTE gap=13km → weight trung bình
# # # # # # #     Normalized: score thấp hơn = tốt hơn, =100 khi bằng ST-Trans
# # # # # # #     """
# # # # # # #     ade  = r.get("ADE",   1e9)
# # # # # # #     h72  = r.get("72h",   1e9)
# # # # # # #     h48  = r.get("48h",   1e9)
# # # # # # #     h24  = r.get("24h",   1e9)
# # # # # # #     h12  = r.get("12h",   1e9)
# # # # # # #     ate  = r.get("ATE",   1e9)
# # # # # # #     cte  = r.get("CTE",   1e9)

# # # # # # #     if not np.isfinite(ate): ate = ade * 0.82   # fallback
# # # # # # #     if not np.isfinite(cte): cte = ade * 0.24   # fallback

# # # # # # #     # Weights phản ánh gap size và difficulty
# # # # # # #     # ATE: gap 24km, weight 0.20 (tăng từ 0.13)
# # # # # # #     # CTE: gap 13km, weight 0.18 (tăng từ 0.12)
# # # # # # #     # 72h: gap 10km, weight 0.22
# # # # # # #     # ADE: gap 9km,  weight 0.18
# # # # # # #     return 100. * (
# # # # # # #         0.05 * (h12  / TARGETS["12h"])  +
# # # # # # #         0.08 * (h24  / TARGETS["24h"])  +
# # # # # # #         0.09 * (h48  / TARGETS["48h"])  +
# # # # # # #         0.22 * (h72  / TARGETS["72h"])  +
# # # # # # #         0.18 * (ade  / TARGETS["ADE"])  +
# # # # # # #         0.20 * (ate  / TARGETS["ATE"])  +  # tăng weight ATE
# # # # # # #         0.18 * (cte  / TARGETS["CTE"])     # tăng weight CTE
# # # # # # #     )


# # # # # # # def _beat(r):
# # # # # # #     p = []
# # # # # # #     for k, t in [("ADE", TARGETS["ADE"]), ("ATE", TARGETS["ATE"]),
# # # # # # #                   ("CTE", TARGETS["CTE"]), ("72h", TARGETS["72h"]),
# # # # # # #                   ("12h", TARGETS["12h"]), ("24h", TARGETS["24h"]),
# # # # # # #                   ("48h", TARGETS["48h"])]:
# # # # # # #         v = r.get(k, 1e9)
# # # # # # #         if np.isfinite(v) and v < t:
# # # # # # #             p.append(f"{k}:{v:.1f}")
# # # # # # #     return "*** BEAT ST-TRANS: " + " ".join(p) + " ***" if p else ""


# # # # # # # def _gap(r):
# # # # # # #     out = []
# # # # # # #     for k, ref in [("ADE", TARGETS["ADE"]), ("72h", TARGETS["72h"]),
# # # # # # #                     ("ATE", TARGETS["ATE"]),  ("CTE", TARGETS["CTE"])]:
# # # # # # #         v = r.get(k, float("nan"))
# # # # # # #         if np.isfinite(v):
# # # # # # #             out.append(f"{k}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
# # # # # # #     return " | ".join(out)


# # # # # # # # ── Evaluation ────────────────────────────────────────────────────────────────

# # # # # # # @torch.no_grad()
# # # # # # # def evaluate(model, loader, dev, tag="", ema=None, steps=20):
# # # # # # #     bk = None
# # # # # # #     if ema:
# # # # # # #         try:
# # # # # # #             bk = ema.apply_to(model)
# # # # # # #         except:
# # # # # # #             pass
# # # # # # #     model.eval()
# # # # # # #     acc = Acc()
# # # # # # #     t0  = time.perf_counter()
# # # # # # #     for b in loader:
# # # # # # #         bl = move(list(b), dev)
# # # # # # #         result = model.sample(bl, ddim_steps=steps)
# # # # # # #         p = result[0] if isinstance(result, (tuple, list)) else result
# # # # # # #         g = bl[1]
# # # # # # #         T = min(p.shape[0], g.shape[0])
# # # # # # #         pd = _ntd(p[:T]); gd = _ntd(g[:T])
# # # # # # #         dist = _hav(pd, gd)
# # # # # # #         at, ct = _atecte(pd, gd)
# # # # # # #         acc.update(dist, at, ct)
# # # # # # #     if bk:
# # # # # # #         try:
# # # # # # #             ema.restore(model, bk)
# # # # # # #         except:
# # # # # # #             pass
# # # # # # #     r  = acc.compute()
# # # # # # #     el = time.perf_counter() - t0

# # # # # # #     def _v(k):  return r.get(k, float("nan"))
# # # # # # #     def _m(v,t): return "ok" if np.isfinite(v) and v < t else "no"

# # # # # # #     print(f"\n{'='*70}")
# # # # # # #     print(f"  [{tag}  {el:.0f}s]")
# # # # # # #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'), TARGETS['ADE'])}]"
# # # # # # #           f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
# # # # # # #           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'), TARGETS['72h'])}]")
# # # # # # #     print(f"  ATE={_v('ATE'):.1f}[{_m(_v('ATE'), TARGETS['ATE'])}]"
# # # # # # #           f"  CTE={_v('CTE'):.1f}[{_m(_v('CTE'), TARGETS['CTE'])}]")
# # # # # # #     print(f"  vs ST-Trans: {_gap(r)}")
# # # # # # #     bt = _beat(r)
# # # # # # #     if bt: print(f"  {bt}")
# # # # # # #     print(f"  Score={_score(r):.2f}")
# # # # # # #     print(f"{'='*70}\n")
# # # # # # #     return r


# # # # # # # # ── Saver ─────────────────────────────────────────────────────────────────────

# # # # # # # def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl, extra=None):
# # # # # # #     m   = _unwrap(model)
# # # # # # #     ema = getattr(m, "_ema", None)
# # # # # # #     esd = None
# # # # # # #     if ema and hasattr(ema, "shadow"):
# # # # # # #         try:
# # # # # # #             esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # # # # # #         except:
# # # # # # #             pass
# # # # # # #     d = {
# # # # # # #         "epoch":              ep,
# # # # # # #         "model_state_dict":   m.state_dict(),
# # # # # # #         "optimizer_state":    opt.state_dict(),
# # # # # # #         "scheduler_state":    sched.state_dict(),
# # # # # # #         "ema_shadow":         esd,
# # # # # # #         "best_score":         saver.bs,
# # # # # # #         "best_ade":           saver.ba,
# # # # # # #         "best_72h":           saver.b7,
# # # # # # #         "best_ate":           saver.bat,
# # # # # # #         "best_cte":           saver.bc,
# # # # # # #         "train_loss":         tl,
# # # # # # #         "val_loss":           vl,
# # # # # # #     }
# # # # # # #     if extra:
# # # # # # #         d.update(extra)
# # # # # # #     torch.save(d, path)


# # # # # # # class Saver:
# # # # # # #     """
# # # # # # #     FIX BUG-1: collect_epoch() + commit_epoch() pattern.
# # # # # # #     Tất cả eval trong 1 epoch được gom lại, chỉ commit 1 lần.
# # # # # # #     → patience đếm đúng theo epoch, không theo số lần eval.
# # # # # # #     """
# # # # # # #     def __init__(self, patience=40, min_ep=35):
# # # # # # #         self.patience = patience
# # # # # # #         self.min_ep   = min_ep
# # # # # # #         self.cnt  = 0
# # # # # # #         self.stop = False
# # # # # # #         self.bs = self.ba = self.b7 = self.bat = self.bc = float("inf")
# # # # # # #         # Per-epoch buffer
# # # # # # #         self._epoch_best_score  = float("inf")
# # # # # # #         self._epoch_best_result = None
# # # # # # #         self._epoch_best_tag    = ""

# # # # # # #     def collect(self, r, model, out, ep, opt, sched, tl, vl, tag=""):
# # # # # # #         """Ghi nhận kết quả eval, lưu checkpoint nếu improve từng metric."""
# # # # # # #         sc  = _score(r)
# # # # # # #         ade = r.get("ADE", 1e9); h72 = r.get("72h", 1e9)
# # # # # # #         ate = r.get("ATE",  1e9); cte = r.get("CTE", 1e9)

# # # # # # #         # Lưu best per-metric (luôn lưu nếu tốt hơn)
# # # # # # #         for v, a, fn in [(ade, "ba", "best_ade.pth"),
# # # # # # #                           (h72, "b7", "best_72h.pth"),
# # # # # # #                           (ate, "bat","best_ate.pth"),
# # # # # # #                           (cte, "bc", "best_cte.pth")]:
# # # # # # #             if v < getattr(self, a):
# # # # # # #                 setattr(self, a, v)
# # # # # # #                 _save_ckpt(os.path.join(out, fn),
# # # # # # #                            ep, model, opt, sched, self, tl, vl)

# # # # # # #         # Track best score của epoch này
# # # # # # #         if sc < self._epoch_best_score:
# # # # # # #             self._epoch_best_score  = sc
# # # # # # #             self._epoch_best_result = r
# # # # # # #             self._epoch_best_tag    = tag
# # # # # # #             # Lưu checkpoint composite
# # # # # # #             if sc < self.bs:
# # # # # # #                 _save_ckpt(os.path.join(out, f"best_composite.pth"),
# # # # # # #                            ep, model, opt, sched, self, tl, vl,
# # # # # # #                            {"score": sc, "ade": ade, "h72": h72})
# # # # # # #                 print(f"  [BEST] {tag} ep={ep} score={sc:.2f}"
# # # # # # #                       f"  ADE={ade:.1f}  72h={h72:.0f}"
# # # # # # #                       f"  ATE={ate:.1f}  CTE={cte:.1f}")

# # # # # # #     def commit_epoch(self, ep):
# # # # # # #         """Gọi 1 lần cuối mỗi epoch để update patience."""
# # # # # # #         sc = self._epoch_best_score
# # # # # # #         if sc < self.bs:
# # # # # # #             self.bs  = sc
# # # # # # #             self.cnt = 0
# # # # # # #         else:
# # # # # # #             self.cnt += 1
# # # # # # #             print(f"  No improve {self.cnt}/{self.patience}"
# # # # # # #                   f"  best={self.bs:.2f}  cur={sc:.2f}")

# # # # # # #         # Reset buffer cho epoch mới
# # # # # # #         self._epoch_best_score  = float("inf")
# # # # # # #         self._epoch_best_result = None
# # # # # # #         self._epoch_best_tag    = ""

# # # # # # #         if ep >= self.min_ep and self.cnt >= self.patience:
# # # # # # #             self.stop = True
# # # # # # #             print(f"  [STOP] ep={ep}")


# # # # # # # def mksub(ds, n, bs, cf):
# # # # # # #     idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
# # # # # # #     return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
# # # # # # #                       collate_fn=cf, num_workers=0, drop_last=False)


# # # # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # # # def get_args():
# # # # # # #     p = argparse.ArgumentParser(
# # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # # #     p.add_argument("--dataset_root",    default="TCND_vn")
# # # # # # #     p.add_argument("--obs_len",         default=8,    type=int)
# # # # # # #     p.add_argument("--pred_len",        default=12,   type=int)
# # # # # # #     p.add_argument("--batch_size",      default=32,   type=int)
# # # # # # #     p.add_argument("--num_epochs",      default=120,  type=int)
# # # # # # #     p.add_argument("--learning_rate",   default=1e-4, type=float)
# # # # # # #     p.add_argument("--weight_decay",    default=1e-3, type=float)
# # # # # # #     p.add_argument("--warmup_epochs",   default=3,    type=int)
# # # # # # #     p.add_argument("--grad_clip",       default=1.0,  type=float)
# # # # # # #     p.add_argument("--patience",        default=40,   type=int)
# # # # # # #     p.add_argument("--min_ep",          default=35,   type=int)
# # # # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # # # #     p.add_argument("--num_workers",     default=2,    type=int)
# # # # # # #     p.add_argument("--sigma_min",       default=0.02, type=float)
# # # # # # #     p.add_argument("--use_ot",          default=True, action="store_true")
# # # # # # #     p.add_argument("--no_ot",           dest="use_ot", action="store_false")
# # # # # # #     p.add_argument("--cfg_guidance_scale", default=1.5, type=float)
# # # # # # #     p.add_argument("--easy_thresh",     default=0.25, type=float)
# # # # # # #     p.add_argument("--n_ensemble",      default=50,   type=int)
# # # # # # #     p.add_argument("--val_freq",        default=3,    type=int)
# # # # # # #     p.add_argument("--val_subset_size", default=500,  type=int)
# # # # # # #     p.add_argument("--fast_ddim",       default=10,   type=int)
# # # # # # #     p.add_argument("--full_ddim",       default=20,   type=int)
# # # # # # #     p.add_argument("--use_ema",         default=True, action="store_true")
# # # # # # #     p.add_argument("--no_ema",          dest="use_ema", action="store_false")
# # # # # # #     p.add_argument("--ema_decay",       default=0.995, type=float)
# # # # # # #     p.add_argument("--output_dir",      default="runs/v74")
# # # # # # #     p.add_argument("--gpu_num",         default="0")
# # # # # # #     p.add_argument("--delim",           default=" ")
# # # # # # #     p.add_argument("--skip",            default=1,    type=int)
# # # # # # #     p.add_argument("--min_ped",         default=1,    type=int)
# # # # # # #     p.add_argument("--threshold",       default=0.002, type=float)
# # # # # # #     p.add_argument("--other_modal",     default="gph")
# # # # # # #     p.add_argument("--test_year",       default=None, type=int)
# # # # # # #     p.add_argument("--resume",          default=None,
# # # # # # #                    help="Path to checkpoint để resume")
# # # # # # #     p.add_argument("--resume_epoch",    default=None, type=int,
# # # # # # #                    help="Override start epoch (nếu None: dùng từ checkpoint)")
# # # # # # #     # FIX BUG-3: thêm finetune_lr
# # # # # # #     p.add_argument("--finetune_lr",     default=None, type=float,
# # # # # # #                    help="Override LR sau resume (fine-tune mode). "
# # # # # # #                         "Ví dụ: --finetune_lr 5e-5")
# # # # # # #     p.add_argument("--eval_test_after_train", default=True, action="store_true")
# # # # # # #     return p.parse_args()


# # # # # # # # ── Main ──────────────────────────────────────────────────────────────────────

# # # # # # # def main(args):
# # # # # # #     if torch.cuda.is_available():
# # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # # #     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # # #     print("=" * 72)
# # # # # # #     print(f"  TC-FlowMatching v74  (FIXED)")
# # # # # # #     print(f"  Loss: L_fm + L_dpe(step_w,d=300) + L_disp(norm) + L_ep(hard)")
# # # # # # #     print(f"  Split: easy(delta<{args.easy_thresh}) / hard(delta>={args.easy_thresh})")
# # # # # # #     print(f"  Patience: {args.patience} epoch  min_ep: {args.min_ep}")
# # # # # # #     print(f"  Target: ADE<{TARGETS['ADE']}  72h<{TARGETS['72h']}"
# # # # # # #           f"  ATE<{TARGETS['ATE']}  CTE<{TARGETS['CTE']}")
# # # # # # #     if args.resume:
# # # # # # #         print(f"  RESUME: {args.resume}")
# # # # # # #     if args.finetune_lr:
# # # # # # #         print(f"  FINETUNE LR: {args.finetune_lr}")
# # # # # # #     print("=" * 72)

# # # # # # #     # ── Data ──────────────────────────────────────────────────
# # # # # # #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"},
# # # # # # #                             test=False)
# # # # # # #     vd, vl   = data_loader(args, {"root": args.dataset_root, "type": "val"},
# # # # # # #                             test=True)
# # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # #     vsub = mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
# # # # # # #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# # # # # # #     # ── Model ─────────────────────────────────────────────────
# # # # # # #     model = TCFlowMatching(
# # # # # # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # # # # # #         sigma_min=args.sigma_min, use_ema=args.use_ema,
# # # # # # #         ema_decay=args.ema_decay, use_ot=args.use_ot,
# # # # # # #         cfg_guidance_scale=args.cfg_guidance_scale,
# # # # # # #         easy_thresh=args.easy_thresh, n_ensemble=args.n_ensemble,
# # # # # # #     ).to(dev)
# # # # # # #     model.init_ema()
# # # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # #     sw_p = sum(p.numel() for p in model.criterion.step_weights.parameters())
# # # # # # #     lw_p = sum(p.numel() for p in model.criterion.loss_weights.parameters())
# # # # # # #     print(f"  params total: {n_params:,} | step_w: {sw_p} | loss_w: {lw_p}")

# # # # # # #     # ── Optimizer & Scheduler ──────────────────────────────────
# # # # # # #     # Nếu finetune_lr thì dùng LR thấp hơn
# # # # # # #     init_lr = args.finetune_lr if args.finetune_lr else args.learning_rate
# # # # # # #     opt     = optim.AdamW(model.parameters(), lr=init_lr,
# # # # # # #                            weight_decay=args.weight_decay)
# # # # # # #     saver   = Saver(patience=args.patience, min_ep=args.min_ep)
# # # # # # #     scaler  = GradScaler("cuda", enabled=args.use_amp)
# # # # # # #     nstep   = len(trl)
# # # # # # #     total   = nstep * args.num_epochs
# # # # # # #     wstp    = nstep * args.warmup_epochs
# # # # # # #     sched   = get_cosine_schedule_with_warmup(opt, wstp, total, min_lr=1e-6)

# # # # # # #     # ── Resume ────────────────────────────────────────────────
# # # # # # #     start = 0
# # # # # # #     if args.resume and os.path.exists(args.resume):
# # # # # # #         print(f"  Loading checkpoint: {args.resume}")
# # # # # # #         ck = torch.load(args.resume, map_location=dev)
# # # # # # #         m  = _unwrap(model)
# # # # # # #         missing, unexpected = m.load_state_dict(ck["model_state_dict"], strict=False)
# # # # # # #         if missing:
# # # # # # #             print(f"  Missing keys: {len(missing)}")
# # # # # # #         if unexpected:
# # # # # # #             print(f"  Unexpected keys: {len(unexpected)}")

# # # # # # #         # Load EMA shadow
# # # # # # #         ema = getattr(m, "_ema", None)
# # # # # # #         if ema and ck.get("ema_shadow"):
# # # # # # #             loaded = 0
# # # # # # #             for k, v in ck["ema_shadow"].items():
# # # # # # #                 if k in ema.shadow:
# # # # # # #                     ema.shadow[k].copy_(v.to(dev))
# # # # # # #                     loaded += 1
# # # # # # #             print(f"  EMA shadow loaded: {loaded} tensors")

# # # # # # #         # FIX BUG-3: Override LR nếu finetune_lr được set
# # # # # # #         if args.finetune_lr:
# # # # # # #             # Dùng LR mới, không load scheduler cũ
# # # # # # #             print(f"  Fine-tune mode: LR={args.finetune_lr} (không load scheduler cũ)")
# # # # # # #             # Rebuild scheduler với LR mới từ epoch 0 (warm start)
# # # # # # #             sched = get_cosine_schedule_with_warmup(
# # # # # # #                 opt, wstp//4, total, min_lr=args.finetune_lr/20.)
# # # # # # #         else:
# # # # # # #             # Load optimizer và scheduler state
# # # # # # #             try:
# # # # # # #                 opt.load_state_dict(ck["optimizer_state"])
# # # # # # #                 print(f"  Optimizer loaded")
# # # # # # #             except Exception as e:
# # # # # # #                 print(f"  Optimizer not loaded: {e}")
# # # # # # #             try:
# # # # # # #                 sched.load_state_dict(ck["scheduler_state"])
# # # # # # #                 print(f"  Scheduler loaded")
# # # # # # #             except Exception as e:
# # # # # # #                 print(f"  Scheduler not loaded, manual forward: {e}")
# # # # # # #                 ep_ck = ck.get("epoch", 0)
# # # # # # #                 for _ in range(ep_ck * nstep):
# # # # # # #                     sched.step()

# # # # # # #         # Restore best scores
# # # # # # #         for a, attr in [("best_score", "bs"), ("best_ade",  "ba"),
# # # # # # #                           ("best_72h",  "b7"), ("best_ate",  "bat"),
# # # # # # #                           ("best_cte",  "bc")]:
# # # # # # #             if a in ck:
# # # # # # #                 setattr(saver, attr, ck[a])
# # # # # # #         start = args.resume_epoch if args.resume_epoch is not None \
# # # # # # #                 else ck.get("epoch", 0) + 1
# # # # # # #         print(f"  Resume from epoch {start}"
# # # # # # #               f"  best_score={saver.bs:.2f}  best_ade={saver.ba:.1f}")

# # # # # # #     # ── Compile ───────────────────────────────────────────────
# # # # # # #     try:
# # # # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # # # #         print("  torch.compile: ok")
# # # # # # #     except Exception as e:
# # # # # # #         print(f"  torch.compile skipped: {e}")

# # # # # # #     ts = time.perf_counter()
# # # # # # #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# # # # # # #     print("=" * 72)

# # # # # # #     # ══════════════════════════════════════════════════════════
# # # # # # #     for ep in range(start, args.num_epochs):
# # # # # # #         model.train()
# # # # # # #         sl = 0.
# # # # # # #         t0 = time.perf_counter()

# # # # # # #         for i, batch in enumerate(trl):
# # # # # # #             bl = move(list(batch), dev)
# # # # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # #                 bd = model.get_loss_breakdown(bl, epoch=ep)

# # # # # # #             opt.zero_grad()
# # # # # # #             scaler.scale(bd["total"]).backward()
# # # # # # #             scaler.unscale_(opt)
# # # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # # # #             sb = scaler.get_scale()
# # # # # # #             scaler.step(opt)
# # # # # # #             scaler.update()
# # # # # # #             if scaler.get_scale() >= sb:
# # # # # # #                 sched.step()
# # # # # # #             model.ema_update()
# # # # # # #             sl += bd["total"].item()

# # # # # # #             if i % 20 == 0:
# # # # # # #                 lr     = opt.param_groups[0]["lr"]
# # # # # # #                 sw_r   = bd.get("sw_ratio", 0.)
# # # # # # #                 tot    = bd["total"].item()
# # # # # # #                 sw_w   = " [!sw<3]" if sw_r < 3.0 else ""
# # # # # # #                 tot_w  = " [!tot>10]" if tot > 10. else ""
# # # # # # #                 print(
# # # # # # #                     f"  [{ep:>3}][{i:>3}/{nstep}]"
# # # # # # #                     f"  tot={tot:.3f}{tot_w}"
# # # # # # #                     f"  fm={bd.get('l_fm',0):.4f}"
# # # # # # #                     f"  pos={bd.get('l_pos',0):.4f}"
# # # # # # #                     f"  disp={bd.get('l_disp',0):.4f}"
# # # # # # #                     f"  sw72={bd.get('sw_72h',0):.2f}"
# # # # # # #                     f"  swr={sw_r:.1f}{sw_w}"
# # # # # # #                     f"  pos/fm={bd.get('lw_pos_fm',0):.2f}"
# # # # # # #                     f"  dw={bd.get('diff_w_mean',1):.2f}"
# # # # # # #                     f"  lr={lr:.2e}"
# # # # # # #                 )

# # # # # # #         avt = sl / nstep

# # # # # # #         # Val loss
# # # # # # #         model.eval()
# # # # # # #         vls = 0.
# # # # # # #         with torch.no_grad():
# # # # # # #             for batch in vl:
# # # # # # #                 bv = move(list(batch), dev)
# # # # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # #                     vls += model.get_loss(bv, epoch=ep).item()
# # # # # # #         avv = vls / len(vl)
# # # # # # #         eps = time.perf_counter() - t0
# # # # # # #         lr_cur = opt.param_groups[0]["lr"]
# # # # # # #         print(f"  Epoch {ep:>3} | train={avt:.4f} val={avv:.4f}"
# # # # # # #               f" | lr={lr_cur:.2e} | {eps:.0f}s")

# # # # # # #         # ── Eval: FIX BUG-1: dùng collect + commit ───────────
# # # # # # #         # Fast eval (subset) — mỗi epoch
# # # # # # #         rf = evaluate(model, vsub, dev, tag=f"FAST ep{ep}",
# # # # # # #                       steps=args.fast_ddim)
# # # # # # #         saver.collect(rf, model, args.output_dir, ep, opt, sched, avt, avv,
# # # # # # #                       tag="fast")

# # # # # # #         # Full val — mỗi val_freq epoch
# # # # # # #         if ep % args.val_freq == 0:
# # # # # # #             em = getattr(_unwrap(model), "_ema", None)

# # # # # # #             rr = evaluate(model, vl, dev, tag=f"RAW ep{ep}",
# # # # # # #                           steps=args.full_ddim)
# # # # # # #             saver.collect(rr, model, args.output_dir, ep, opt, sched, avt, avv,
# # # # # # #                           tag="raw")

# # # # # # #             if em and ep >= 3:
# # # # # # #                 re = evaluate(model, vl, dev, tag=f"EMA ep{ep}",
# # # # # # #                               ema=em, steps=args.full_ddim)
# # # # # # #                 saver.collect(re, model, args.output_dir, ep, opt, sched,
# # # # # # #                               avt, avv, tag="ema")

# # # # # # #         # FIX BUG-1: commit 1 lần sau tất cả evals
# # # # # # #         saver.commit_epoch(ep)

# # # # # # #         # Periodic checkpoint
# # # # # # #         if ep % 10 == 0 or ep == args.num_epochs - 1:
# # # # # # #             _save_ckpt(
# # # # # # #                 os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# # # # # # #                 ep, model, opt, sched, saver, avt, avv)

# # # # # # #         if saver.stop:
# # # # # # #             print(f"  Early stop ep={ep}")
# # # # # # #             break

# # # # # # #     th = (time.perf_counter() - ts) / 3600.
# # # # # # #     print(f"\n  Best: ADE={saver.ba:.1f}  72h={saver.b7:.0f}"
# # # # # # #           f"  ATE={saver.bat:.1f}  CTE={saver.bc:.1f}  ({th:.2f}h)")

# # # # # # #     # ── Post-training test ─────────────────────────────────────
# # # # # # #     if args.eval_test_after_train:
# # # # # # #         print("\n" + "="*72 + "\n  POST-TRAINING TEST\n" + "="*72)
# # # # # # #         try:
# # # # # # #             _, tl2 = data_loader(args,
# # # # # # #                                   {"root": args.dataset_root, "type": "test"},
# # # # # # #                                   test=True)
# # # # # # #         except:
# # # # # # #             print("  No test set → using val")
# # # # # # #             tl2 = vl

# # # # # # #         for fn, lb in [("best_composite.pth", "COMPOSITE"),
# # # # # # #                         ("best_72h.pth", "72H"),
# # # # # # #                         ("best_ema.pth", "EMA")]:
# # # # # # #             pp = os.path.join(args.output_dir, fn)
# # # # # # #             if not os.path.exists(pp):
# # # # # # #                 continue
# # # # # # #             ck = torch.load(pp, map_location=dev)
# # # # # # #             _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
# # # # # # #             em = getattr(_unwrap(model), "_ema", None)
# # # # # # #             if em and ck.get("ema_shadow"):
# # # # # # #                 for k, v in ck["ema_shadow"].items():
# # # # # # #                     if k in em.shadow:
# # # # # # #                         em.shadow[k].copy_(v.to(dev))
# # # # # # #             r = evaluate(model, tl2, dev, tag=f"TEST/{lb}",
# # # # # # #                          steps=args.full_ddim)
# # # # # # #             print(f"\n  --- {lb} ---")
# # # # # # #             for key, ref in [("ADE", TARGETS["ADE"]), ("72h",  TARGETS["72h"]),
# # # # # # #                                ("ATE", TARGETS["ATE"]), ("CTE",  TARGETS["CTE"])]:
# # # # # # #                 v  = r.get(key, float("nan"))
# # # # # # #                 mk = "BEAT!" if np.isfinite(v) and v < ref else f"need {ref:.0f}"
# # # # # # #                 print(f"    {key:<6}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
# # # # # # #     print("=" * 72)


# # # # # # # if __name__ == "__main__":
# # # # # # #     args = get_args()
# # # # # # #     np.random.seed(42)
# # # # # # #     torch.manual_seed(42)
# # # # # # #     if torch.cuda.is_available():
# # # # # # #         torch.cuda.manual_seed_all(42)
# # # # # # #     main(args)


# # # # # # """
# # # # # # train_v75.py — TC-FlowMatching v75  (Physics-Grounded Loss)
# # # # # # ════════════════════════════════════════════════════════════
# # # # # # CHANGES vs train_v74.py:

# # # # # #   MODEL:
# # # # # #     Import flow_matching_model_v75 (FMv75Loss)
# # # # # #     L_disp → L_kinematic + L_logspeed + L_smooth (3 physics terms)

# # # # # #   DEFAULTS CHANGED:
# # # # # #     --n_ensemble:         50  → 100   (reduce sampling variance ±5→±2.5 km)
# # # # # #     --fast_ddim:          10  → 15    (better fast eval estimate)
# # # # # #     --full_ddim:          20  → 30    (better full eval accuracy)
# # # # # #     --ema_decay:          0.995 → 0.999 (EMA was lagging in v74)
# # # # # #     --cfg_guidance_scale: 1.5 → 2.0   (reduce cross-track spread → CTE)

# # # # # #   PRINT LOOP: Added l_kin, l_lspd, l_sm monitoring
# # # # # #     l_kin > 0.5 after ep10 → increase norm_scale to 150
# # # # # #     l_sm  > 0.2 stuck      → increase w_smooth to 0.30
# # # # # #     l_lspd > 1.0 after ep15 → check _to_deg conversion

# # # # # #   KEEPS FROM v74 (all BUG fixes):
# # # # # #     BUG-1 FIX: Saver collect+commit_epoch pattern (patience counts once/epoch)
# # # # # #     BUG-2 FIX: Score formula reweighted by gap size
# # # # # #     BUG-3 FIX: --finetune_lr override after resume

# # # # # # RUN (fresh v75):
# # # # # #   python scripts/train_v75.py \\
# # # # # #       --dataset_root /path/to/tc-ofm \\
# # # # # #       --output_dir   runs/v75 \\
# # # # # #       --batch_size   32 \\
# # # # # #       --use_amp

# # # # # # RESUME from v74 best:
# # # # # #   python scripts/train_v75.py \\
# # # # # #       --dataset_root /path/to/tc-ofm \\
# # # # # #       --output_dir   runs/v75_from_v74 \\
# # # # # #       --resume       runs/v74/best_composite.pth \\
# # # # # #       --finetune_lr  5e-5 \\
# # # # # #       --batch_size   32 \\
# # # # # #       --use_amp

# # # # # # TARGETS (ST-Trans actual val):
# # # # # #   ADE<172.68  72h<321.39  ATE<142.21  CTE<42.04
# # # # # #   12h<65.42   24h<104.67  48h<205.10

# # # # # # MONITORING:
# # # # # #   Watch l_kin (should decrease ep0 ~1.0 → ep10 ~0.3 → ep30 ~0.1)
# # # # # #   Watch l_sm  (should decrease ep0 ~0.15 → ep10 ~0.06)
# # # # # #   Watch l_lspd (should decrease ep0 ~0.8 → ep5 ~0.3 → stabilize)
# # # # # #   ATE/ADE ratio: target < 0.88 (v74 was ~0.92)
# # # # # # """
# # # # # # from __future__ import annotations

# # # # # # import sys, os
# # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # import argparse, time, random
# # # # # # from collections import defaultdict

# # # # # # import numpy as np
# # # # # # import torch
# # # # # # import torch.optim as optim
# # # # # # from torch.amp import autocast, GradScaler
# # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # from Model.data.loader_training import data_loader
# # # # # # # v75: import updated model
# # # # # # from Model.flow_matching_model import TCFlowMatching

# # # # # # try:
# # # # # #     from Model.utils import get_cosine_schedule_with_warmup
# # # # # # except ImportError:
# # # # # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # # # # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # # # # #         return CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=min_lr)

# # # # # # # ST-Trans actual val targets
# # # # # # TARGETS = {
# # # # # #     "ADE":   172.68,
# # # # # #     "72h":   321.39,
# # # # # #     "ATE":   142.21,
# # # # # #     "CTE":    42.04,
# # # # # #     "12h":    65.42,
# # # # # #     "24h":   104.67,
# # # # # #     "48h":   205.10,
# # # # # # }
# # # # # # R_EARTH = 6371.0


# # # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # # def _unwrap(m):
# # # # # #     return m._orig_mod if hasattr(m, '_orig_mod') else m

# # # # # # def move(b, dev):
# # # # # #     out = list(b)
# # # # # #     for i, x in enumerate(out):
# # # # # #         if torch.is_tensor(x):
# # # # # #             out[i] = x.to(dev)
# # # # # #         elif isinstance(x, dict):
# # # # # #             out[i] = {k: v.to(dev) if torch.is_tensor(v) else v
# # # # # #                       for k, v in x.items()}
# # # # # #     return out


# # # # # # # ── Metric utils ──────────────────────────────────────────────────────────────

# # # # # # def _ntd(a):
# # # # # #     o = a.clone()
# # # # # #     o[..., 0] = (a[..., 0] * 50. + 1800.) / 10.
# # # # # #     o[..., 1] = (a[..., 1] * 50.) / 10.
# # # # # #     return o

# # # # # # def _hav(p1, p2):
# # # # # #     la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
# # # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # # #     a = (torch.sin(dlat/2).pow(2)
# # # # # #          + torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2))
# # # # # #     return 2.*R_EARTH*torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

# # # # # # def _atecte(pd, gd):
# # # # # #     T = min(pd.shape[0], gd.shape[0])
# # # # # #     if T < 2:
# # # # # #         z = pd.new_zeros(1, pd.shape[1])
# # # # # #         return z, z
# # # # # #     lo1 = torch.deg2rad(gd[:T-1, :, 0]); la1 = torch.deg2rad(gd[:T-1, :, 1])
# # # # # #     lo2 = torch.deg2rad(gd[1:T,  :, 0]); la2 = torch.deg2rad(gd[1:T,  :, 1])
# # # # # #     lo3 = torch.deg2rad(pd[1:T,  :, 0]); la3 = torch.deg2rad(pd[1:T,  :, 1])
# # # # # #     ya  = torch.sin(lo2-lo1)*torch.cos(la2)
# # # # # #     xa  = torch.cos(la1)*torch.sin(la2) - torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # # # # #     be  = torch.atan2(ya, xa)
# # # # # #     ye  = torch.sin(lo3-lo2)*torch.cos(la3)
# # # # # #     xe  = torch.cos(la2)*torch.sin(la3) - torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # # # # #     bee = torch.atan2(ye, xe)
# # # # # #     tot = _hav(pd[1:T], gd[1:T])
# # # # # #     ang = bee - be
# # # # # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # # # class Acc:
# # # # # #     def __init__(self):
# # # # # #         self.d = []; self.a = []; self.c = []
# # # # # #         self.sd = defaultdict(list)
# # # # # #         self._h = {12: 1, 24: 3, 48: 7, 72: 11}

# # # # # #     def update(self, dist, ate=None, cte=None):
# # # # # #         self.d.extend(dist.mean(0).tolist())
# # # # # #         for h, s in self._h.items():
# # # # # #             if s < dist.shape[0]:
# # # # # #                 self.sd[h].extend(dist[s].tolist())
# # # # # #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# # # # # #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())

# # # # # #     def compute(self):
# # # # # #         r = {
# # # # # #             "ADE": float(np.mean(self.d))  if self.d else float("nan"),
# # # # # #             "ATE": float(np.mean(self.a))  if self.a else float("nan"),
# # # # # #             "CTE": float(np.mean(self.c))  if self.c else float("nan"),
# # # # # #             "n":   len(self.d),
# # # # # #         }
# # # # # #         for h in self._h:
# # # # # #             v = self.sd.get(h, [])
# # # # # #             r[f"{h}h"] = float(np.mean(v)) if v else float("nan")
# # # # # #         return r


# # # # # # def _score(r):
# # # # # #     """
# # # # # #     Score formula — weights reflect gap size relative to ST-Trans targets.
# # # # # #     ATE gap=24km → weight 0.20 (highest)
# # # # # #     CTE gap=13km → weight 0.18
# # # # # #     72h gap=10km → weight 0.22
# # # # # #     ADE gap=9km  → weight 0.18
# # # # # #     Lower score = better (100 = ST-Trans level)
# # # # # #     """
# # # # # #     ade  = r.get("ADE",   1e9)
# # # # # #     h72  = r.get("72h",   1e9)
# # # # # #     h48  = r.get("48h",   1e9)
# # # # # #     h24  = r.get("24h",   1e9)
# # # # # #     h12  = r.get("12h",   1e9)
# # # # # #     ate  = r.get("ATE",   1e9)
# # # # # #     cte  = r.get("CTE",   1e9)

# # # # # #     if not np.isfinite(ate): ate = ade * 0.82
# # # # # #     if not np.isfinite(cte): cte = ade * 0.24

# # # # # #     return 100. * (
# # # # # #         0.05 * (h12  / TARGETS["12h"])  +
# # # # # #         0.08 * (h24  / TARGETS["24h"])  +
# # # # # #         0.09 * (h48  / TARGETS["48h"])  +
# # # # # #         0.22 * (h72  / TARGETS["72h"])  +
# # # # # #         0.18 * (ade  / TARGETS["ADE"])  +
# # # # # #         0.20 * (ate  / TARGETS["ATE"])  +
# # # # # #         0.18 * (cte  / TARGETS["CTE"])
# # # # # #     )


# # # # # # def _beat(r):
# # # # # #     p = []
# # # # # #     for k, t in [("ADE", TARGETS["ADE"]), ("ATE", TARGETS["ATE"]),
# # # # # #                   ("CTE", TARGETS["CTE"]), ("72h", TARGETS["72h"]),
# # # # # #                   ("12h", TARGETS["12h"]), ("24h", TARGETS["24h"]),
# # # # # #                   ("48h", TARGETS["48h"])]:
# # # # # #         v = r.get(k, 1e9)
# # # # # #         if np.isfinite(v) and v < t:
# # # # # #             p.append(f"{k}:{v:.1f}")
# # # # # #     return "*** BEAT ST-TRANS: " + " ".join(p) + " ***" if p else ""


# # # # # # def _gap(r):
# # # # # #     out = []
# # # # # #     for k, ref in [("ADE", TARGETS["ADE"]), ("72h", TARGETS["72h"]),
# # # # # #                     ("ATE", TARGETS["ATE"]),  ("CTE", TARGETS["CTE"])]:
# # # # # #         v = r.get(k, float("nan"))
# # # # # #         if np.isfinite(v):
# # # # # #             out.append(f"{k}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
# # # # # #     return " | ".join(out)


# # # # # # # ── Evaluation ────────────────────────────────────────────────────────────────

# # # # # # @torch.no_grad()
# # # # # # def evaluate(model, loader, dev, tag="", ema=None, steps=20):
# # # # # #     bk = None
# # # # # #     if ema:
# # # # # #         try:
# # # # # #             bk = ema.apply_to(model)
# # # # # #         except:
# # # # # #             pass
# # # # # #     model.eval()
# # # # # #     acc = Acc()
# # # # # #     t0  = time.perf_counter()
# # # # # #     for b in loader:
# # # # # #         bl = move(list(b), dev)
# # # # # #         result = model.sample(bl, ddim_steps=steps)
# # # # # #         p = result[0] if isinstance(result, (tuple, list)) else result
# # # # # #         g = bl[1]
# # # # # #         T = min(p.shape[0], g.shape[0])
# # # # # #         pd = _ntd(p[:T]); gd = _ntd(g[:T])
# # # # # #         dist = _hav(pd, gd)
# # # # # #         at, ct = _atecte(pd, gd)
# # # # # #         acc.update(dist, at, ct)
# # # # # #     if bk:
# # # # # #         try:
# # # # # #             ema.restore(model, bk)
# # # # # #         except:
# # # # # #             pass
# # # # # #     r  = acc.compute()
# # # # # #     el = time.perf_counter() - t0

# # # # # #     def _v(k):   return r.get(k, float("nan"))
# # # # # #     def _m(v, t): return "ok" if np.isfinite(v) and v < t else "no"

# # # # # #     print(f"\n{'='*70}")
# # # # # #     print(f"  [{tag}  {el:.0f}s]")
# # # # # #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'), TARGETS['ADE'])}]"
# # # # # #           f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
# # # # # #           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'), TARGETS['72h'])}]")
# # # # # #     print(f"  ATE={_v('ATE'):.1f}[{_m(_v('ATE'), TARGETS['ATE'])}]"
# # # # # #           f"  CTE={_v('CTE'):.1f}[{_m(_v('CTE'), TARGETS['CTE'])}]")
# # # # # #     print(f"  vs ST-Trans: {_gap(r)}")
# # # # # #     bt = _beat(r)
# # # # # #     if bt: print(f"  {bt}")
# # # # # #     print(f"  Score={_score(r):.2f}")
# # # # # #     print(f"{'='*70}\n")
# # # # # #     return r


# # # # # # # ── Saver (BUG-1 FIX: collect+commit_epoch) ──────────────────────────────────

# # # # # # def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl, extra=None):
# # # # # #     m   = _unwrap(model)
# # # # # #     ema = getattr(m, "_ema", None)
# # # # # #     esd = None
# # # # # #     if ema and hasattr(ema, "shadow"):
# # # # # #         try:
# # # # # #             esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # # # # #         except:
# # # # # #             pass
# # # # # #     d = {
# # # # # #         "epoch":            ep,
# # # # # #         "model_state_dict": m.state_dict(),
# # # # # #         "optimizer_state":  opt.state_dict(),
# # # # # #         "scheduler_state":  sched.state_dict(),
# # # # # #         "ema_shadow":       esd,
# # # # # #         "best_score":       saver.bs,
# # # # # #         "best_ade":         saver.ba,
# # # # # #         "best_72h":         saver.b7,
# # # # # #         "best_ate":         saver.bat,
# # # # # #         "best_cte":         saver.bc,
# # # # # #         "train_loss":       tl,
# # # # # #         "val_loss":         vl,
# # # # # #         "version":          "v75",
# # # # # #     }
# # # # # #     if extra:
# # # # # #         d.update(extra)
# # # # # #     torch.save(d, path)


# # # # # # class Saver:
# # # # # #     """
# # # # # #     BUG-1 FIX from v74: collect_epoch() + commit_epoch() pattern.
# # # # # #     All evals within one epoch are gathered, patience updated only once.
# # # # # #     → patience counts correctly per epoch, not per eval call.
# # # # # #     """
# # # # # #     def __init__(self, patience=40, min_ep=35):
# # # # # #         self.patience = patience
# # # # # #         self.min_ep   = min_ep
# # # # # #         self.cnt  = 0
# # # # # #         self.stop = False
# # # # # #         self.bs = self.ba = self.b7 = self.bat = self.bc = float("inf")
# # # # # #         self._epoch_best_score  = float("inf")
# # # # # #         self._epoch_best_result = None
# # # # # #         self._epoch_best_tag    = ""

# # # # # #     def collect(self, r, model, out, ep, opt, sched, tl, vl, tag=""):
# # # # # #         sc  = _score(r)
# # # # # #         ade = r.get("ADE", 1e9); h72 = r.get("72h", 1e9)
# # # # # #         ate = r.get("ATE",  1e9); cte = r.get("CTE", 1e9)

# # # # # #         for v, a, fn in [(ade, "ba", "best_ade.pth"),
# # # # # #                           (h72, "b7", "best_72h.pth"),
# # # # # #                           (ate, "bat","best_ate.pth"),
# # # # # #                           (cte, "bc", "best_cte.pth")]:
# # # # # #             if v < getattr(self, a):
# # # # # #                 setattr(self, a, v)
# # # # # #                 _save_ckpt(os.path.join(out, fn),
# # # # # #                            ep, model, opt, sched, self, tl, vl)

# # # # # #         if sc < self._epoch_best_score:
# # # # # #             self._epoch_best_score  = sc
# # # # # #             self._epoch_best_result = r
# # # # # #             self._epoch_best_tag    = tag
# # # # # #             if sc < self.bs:
# # # # # #                 _save_ckpt(os.path.join(out, "best_composite.pth"),
# # # # # #                            ep, model, opt, sched, self, tl, vl,
# # # # # #                            {"score": sc, "ade": ade, "h72": h72})
# # # # # #                 print(f"  [BEST] {tag} ep={ep} score={sc:.2f}"
# # # # # #                       f"  ADE={ade:.1f}  72h={h72:.0f}"
# # # # # #                       f"  ATE={ate:.1f}  CTE={cte:.1f}")

# # # # # #         # Mode collapse detector:
# # # # # #         # Collapse signature: CTE stops improving while ADE increases
# # # # # #         # Compare to best known values
# # # # # #         if ep >= 10 and np.isfinite(ade) and np.isfinite(cte):
# # # # # #             if ade > self.ba * 1.05 and cte > self.bc * 1.10:
# # # # # #                 print(f"  [COLLAPSE?] {tag} ADE={ade:.1f}(best={self.ba:.1f}) "
# # # # # #                       f"CTE={cte:.1f}(best={self.bc:.1f})"
# # # # # #                       f" → if persistent, reduce --cfg_guidance_scale to 1.5")

# # # # # #     def commit_epoch(self, ep):
# # # # # #         sc = self._epoch_best_score
# # # # # #         if sc < self.bs:
# # # # # #             self.bs  = sc
# # # # # #             self.cnt = 0
# # # # # #         else:
# # # # # #             self.cnt += 1
# # # # # #             print(f"  No improve {self.cnt}/{self.patience}"
# # # # # #                   f"  best={self.bs:.2f}  cur={sc:.2f}")

# # # # # #         self._epoch_best_score  = float("inf")
# # # # # #         self._epoch_best_result = None
# # # # # #         self._epoch_best_tag    = ""

# # # # # #         if ep >= self.min_ep and self.cnt >= self.patience:
# # # # # #             self.stop = True
# # # # # #             print(f"  [STOP] ep={ep}")


# # # # # # def mksub(ds, n, bs, cf):
# # # # # #     idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
# # # # # #     return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
# # # # # #                       collate_fn=cf, num_workers=0, drop_last=False)


# # # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # # def get_args():
# # # # # #     p = argparse.ArgumentParser(
# # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # #     p.add_argument("--dataset_root",    default="TCND_vn")
# # # # # #     p.add_argument("--obs_len",         default=8,    type=int)
# # # # # #     p.add_argument("--pred_len",        default=12,   type=int)
# # # # # #     p.add_argument("--batch_size",      default=32,   type=int)
# # # # # #     p.add_argument("--num_epochs",      default=120,  type=int)
# # # # # #     p.add_argument("--learning_rate",   default=1e-4, type=float)
# # # # # #     p.add_argument("--weight_decay",    default=1e-3, type=float)
# # # # # #     p.add_argument("--warmup_epochs",   default=3,    type=int)
# # # # # #     p.add_argument("--grad_clip",       default=1.0,  type=float)
# # # # # #     p.add_argument("--patience",        default=40,   type=int)
# # # # # #     p.add_argument("--min_ep",          default=35,   type=int)
# # # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # # #     p.add_argument("--num_workers",     default=2,    type=int)
# # # # # #     p.add_argument("--sigma_min",       default=0.02, type=float)
# # # # # #     p.add_argument("--use_ot",          default=True, action="store_true")
# # # # # #     p.add_argument("--no_ot",           dest="use_ot", action="store_false")
# # # # # #     p.add_argument("--cfg_guidance_scale", default=2.0, type=float)   # v75: 1.5→2.0
# # # # # #     p.add_argument("--easy_thresh",     default=0.25, type=float)
# # # # # #     p.add_argument("--n_ensemble",      default=100,  type=int)        # v75: 50→100
# # # # # #     p.add_argument("--val_freq",        default=3,    type=int)
# # # # # #     p.add_argument("--val_subset_size", default=500,  type=int)
# # # # # #     p.add_argument("--fast_ddim",       default=15,   type=int)        # v75: 10→15
# # # # # #     p.add_argument("--full_ddim",       default=30,   type=int)        # v75: 20→30
# # # # # #     p.add_argument("--use_ema",         default=True, action="store_true")
# # # # # #     p.add_argument("--no_ema",          dest="use_ema", action="store_false")
# # # # # #     p.add_argument("--ema_decay",       default=0.999, type=float)     # v75: 0.995→0.999
# # # # # #     p.add_argument("--output_dir",      default="runs/v75")
# # # # # #     p.add_argument("--gpu_num",         default="0")
# # # # # #     p.add_argument("--delim",           default=" ")
# # # # # #     p.add_argument("--skip",            default=1,    type=int)
# # # # # #     p.add_argument("--min_ped",         default=1,    type=int)
# # # # # #     p.add_argument("--threshold",       default=0.002, type=float)
# # # # # #     p.add_argument("--other_modal",     default="gph")
# # # # # #     p.add_argument("--test_year",       default=None, type=int)
# # # # # #     p.add_argument("--resume",          default=None,
# # # # # #                    help="Path to checkpoint for resume")
# # # # # #     p.add_argument("--resume_epoch",    default=None, type=int)
# # # # # #     p.add_argument("--finetune_lr",     default=None, type=float,
# # # # # #                    help="Override LR after resume (fine-tune mode). E.g. --finetune_lr 5e-5")
# # # # # #     p.add_argument("--eval_test_after_train", default=True, action="store_true")
# # # # # #     # v75 physics loss monitoring args (read-only — model uses fixed weights)
# # # # # #     p.add_argument("--w_kin",   default=1.0,  type=float,
# # # # # #                    help="[INFO] Initial w_kin for ConstrainedLossWeights (model learns from this init)")
# # # # # #     p.add_argument("--w_lspd",  default=0.15, type=float,
# # # # # #                    help="Fixed weight for L_logspeed auxiliary")
# # # # # #     p.add_argument("--w_smooth", default=0.20, type=float,
# # # # # #                    help="Fixed weight for L_smooth auxiliary")
# # # # # #     return p.parse_args()


# # # # # # # ── Main ──────────────────────────────────────────────────────────────────────

# # # # # # def main(args):
# # # # # #     if torch.cuda.is_available():
# # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # #     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # #     print("=" * 72)
# # # # # #     print(f"  TC-FlowMatching v75  (Physics-Grounded Loss)")
# # # # # #     print(f"  Loss: L_fm + L_pos(step_w) + L_kinematic + 0.15*L_logspeed + 0.20*L_smooth")
# # # # # #     print(f"  L_kinematic = velocity_MSE = ATE² + CTE²  (identity decomposition)")
# # # # # #     print(f"  Split: DifficultyWeighter (continuous soft, NOT binary easy/hard)")
# # # # # #     print(f"  Training: single phase, patience={args.patience}, min_ep={args.min_ep}")
# # # # # #     print(f"  Inference: K=3 mode clustering at 72h endpoint, n_ensemble={args.n_ensemble}")
# # # # # #     print(f"  EMA decay={args.ema_decay}  cfg_guidance={args.cfg_guidance_scale}")
# # # # # #     print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']}"
# # # # # #           f" ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
# # # # # #     if args.resume:
# # # # # #         print(f"  RESUME: {args.resume}")
# # # # # #     if args.finetune_lr:
# # # # # #         print(f"  FINETUNE LR: {args.finetune_lr}")
# # # # # #     print("=" * 72)

# # # # # #     # ── Data ──────────────────────────────────────────────────
# # # # # #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"},
# # # # # #                             test=False)
# # # # # #     vd, vl   = data_loader(args, {"root": args.dataset_root, "type": "val"},
# # # # # #                             test=True)
# # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # #     vsub = mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
# # # # # #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# # # # # #     # ── Model ─────────────────────────────────────────────────
# # # # # #     model = TCFlowMatching(
# # # # # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # # # # #         sigma_min=args.sigma_min, use_ema=args.use_ema,
# # # # # #         ema_decay=args.ema_decay, use_ot=args.use_ot,
# # # # # #         cfg_guidance_scale=args.cfg_guidance_scale,
# # # # # #         easy_thresh=args.easy_thresh, n_ensemble=args.n_ensemble,
# # # # # #     ).to(dev)
# # # # # #     model.init_ema()
# # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # #     sw_p = sum(p.numel() for p in model.criterion.step_weights.parameters())
# # # # # #     lw_p = sum(p.numel() for p in model.criterion.loss_weights.parameters())
# # # # # #     print(f"  params total: {n_params:,} | step_w: {sw_p} | loss_w: {lw_p}")

# # # # # #     # ── Optimizer & Scheduler ──────────────────────────────────
# # # # # #     init_lr = args.finetune_lr if args.finetune_lr else args.learning_rate
# # # # # #     opt     = optim.AdamW(model.parameters(), lr=init_lr,
# # # # # #                            weight_decay=args.weight_decay)
# # # # # #     saver   = Saver(patience=args.patience, min_ep=args.min_ep)
# # # # # #     scaler  = GradScaler("cuda", enabled=args.use_amp)
# # # # # #     nstep   = len(trl)
# # # # # #     total   = nstep * args.num_epochs
# # # # # #     wstp    = nstep * args.warmup_epochs
# # # # # #     sched   = get_cosine_schedule_with_warmup(opt, wstp, total, min_lr=1e-6)

# # # # # #     # ── Resume ────────────────────────────────────────────────
# # # # # #     start = 0
# # # # # #     if args.resume and os.path.exists(args.resume):
# # # # # #         print(f"  Loading checkpoint: {args.resume}")
# # # # # #         ck = torch.load(args.resume, map_location=dev)
# # # # # #         m  = _unwrap(model)
# # # # # #         missing, unexpected = m.load_state_dict(ck["model_state_dict"], strict=False)
# # # # # #         if missing:    print(f"  Missing keys: {len(missing)}")
# # # # # #         if unexpected: print(f"  Unexpected keys: {len(unexpected)}")

# # # # # #         ema = getattr(m, "_ema", None)
# # # # # #         if ema and ck.get("ema_shadow"):
# # # # # #             loaded = 0
# # # # # #             for k, v in ck["ema_shadow"].items():
# # # # # #                 if k in ema.shadow:
# # # # # #                     ema.shadow[k].copy_(v.to(dev)); loaded += 1
# # # # # #             print(f"  EMA shadow loaded: {loaded} tensors")

# # # # # #         if args.finetune_lr:
# # # # # #             print(f"  Fine-tune mode: LR={args.finetune_lr}")
# # # # # #             sched = get_cosine_schedule_with_warmup(
# # # # # #                 opt, wstp//4, total, min_lr=args.finetune_lr/20.)
# # # # # #         else:
# # # # # #             try:
# # # # # #                 opt.load_state_dict(ck["optimizer_state"])
# # # # # #                 print(f"  Optimizer loaded")
# # # # # #             except Exception as e:
# # # # # #                 print(f"  Optimizer not loaded: {e}")
# # # # # #             try:
# # # # # #                 sched.load_state_dict(ck["scheduler_state"])
# # # # # #                 print(f"  Scheduler loaded")
# # # # # #             except Exception as e:
# # # # # #                 print(f"  Scheduler not loaded, manual forward: {e}")
# # # # # #                 ep_ck = ck.get("epoch", 0)
# # # # # #                 for _ in range(ep_ck * nstep):
# # # # # #                     sched.step()

# # # # # #         for a, attr in [("best_score", "bs"), ("best_ade", "ba"),
# # # # # #                           ("best_72h", "b7"), ("best_ate", "bat"),
# # # # # #                           ("best_cte", "bc")]:
# # # # # #             if a in ck: setattr(saver, attr, ck[a])
# # # # # #         start = args.resume_epoch if args.resume_epoch is not None \
# # # # # #                 else ck.get("epoch", 0) + 1
# # # # # #         print(f"  Resume from epoch {start}"
# # # # # #               f"  best_score={saver.bs:.2f}  best_ade={saver.ba:.1f}")

# # # # # #     # ── Compile ───────────────────────────────────────────────
# # # # # #     try:
# # # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # # #         print("  torch.compile: ok")
# # # # # #     except Exception as e:
# # # # # #         print(f"  torch.compile skipped: {e}")

# # # # # #     ts = time.perf_counter()
# # # # # #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# # # # # #     print("=" * 72)

# # # # # #     # ══════════════════════════════════════════════════════════
# # # # # #     for ep in range(start, args.num_epochs):
# # # # # #         model.train()
# # # # # #         sl = 0.
# # # # # #         t0 = time.perf_counter()

# # # # # #         for i, batch in enumerate(trl):
# # # # # #             bl = move(list(batch), dev)
# # # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # #                 bd = model.get_loss_breakdown(bl, epoch=ep)

# # # # # #             opt.zero_grad()
# # # # # #             scaler.scale(bd["total"]).backward()
# # # # # #             scaler.unscale_(opt)
# # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # # #             sb = scaler.get_scale()
# # # # # #             scaler.step(opt)
# # # # # #             scaler.update()
# # # # # #             if scaler.get_scale() >= sb:
# # # # # #                 sched.step()
# # # # # #             model.ema_update()
# # # # # #             sl += bd["total"].item()

# # # # # #             if i % 20 == 0:
# # # # # #                 lr     = opt.param_groups[0]["lr"]
# # # # # #                 sw_r   = bd.get("sw_ratio", 0.)
# # # # # #                 tot    = bd["total"].item()
# # # # # #                 sw_w   = " [!sw<3]"   if sw_r < 3.0 else ""
# # # # # #                 tot_w  = " [!tot>10]" if tot  > 10. else ""
# # # # # #                 l_kin  = bd.get("l_kin",  0.)
# # # # # #                 l_lspd = bd.get("l_lspd", 0.)
# # # # # #                 l_sm   = bd.get("l_sm",   0.)
# # # # # #                 # Accurate thresholds (from numerical analysis):
# # # # # #                 # l_kin: auto-calibrated by gt_speed, expect ~0.04-0.15 ep0 → ~0.02 ep30
# # # # # #                 #        warn if > 0.5 ONLY after ep20 (sign of divergence, not scale issue)
# # # # # #                 # l_sm:  max possible = 0.5 by construction. ep0 ~0.15-0.25, ep10+ ~0.04-0.08
# # # # # #                 #        warn if > 0.15 after ep15 (should have converged by then)
# # # # # #                 # l_lspd: ep0 ~0.4, ep10 ~0.2, warn if > 0.5 after ep20
# # # # # #                 kin_w  = " [!kin_div]"  if l_kin  > 0.5  and ep >= 20 else ""
# # # # # #                 sm_w   = " [!sm_high]"  if l_sm   > 0.15 and ep >= 15 else ""
# # # # # #                 lspd_w = " [!lspd_div]" if l_lspd > 0.5  and ep >= 20 else ""
# # # # # #                 print(
# # # # # #                     f"  [{ep:>3}][{i:>3}/{nstep}]"
# # # # # #                     f"  tot={tot:.3f}{tot_w}"
# # # # # #                     f"  fm={bd.get('l_fm',0):.4f}"
# # # # # #                     f"  pos={bd.get('l_pos',0):.4f}"
# # # # # #                     f"  kin={l_kin:.4f}{kin_w}"
# # # # # #                     f"  lspd={l_lspd:.4f}{lspd_w}"
# # # # # #                     f"  sm={l_sm:.4f}{sm_w}"
# # # # # #                     f"  sw72={bd.get('sw_72h',0):.2f}"
# # # # # #                     f"  swr={sw_r:.1f}{sw_w}"
# # # # # #                     f"  pos/fm={bd.get('lw_pos_fm',0):.2f}"
# # # # # #                     f"  dw={bd.get('diff_w_mean',1):.2f}"
# # # # # #                     f"  lr={lr:.2e}"
# # # # # #                 )

# # # # # #         avt = sl / nstep

# # # # # #         # Val loss
# # # # # #         model.eval()
# # # # # #         vls = 0.
# # # # # #         with torch.no_grad():
# # # # # #             for batch in vl:
# # # # # #                 bv = move(list(batch), dev)
# # # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # #                     vls += model.get_loss(bv, epoch=ep).item()
# # # # # #         avv = vls / len(vl)
# # # # # #         eps = time.perf_counter() - t0
# # # # # #         lr_cur = opt.param_groups[0]["lr"]
# # # # # #         print(f"  Epoch {ep:>3} | train={avt:.4f} val={avv:.4f}"
# # # # # #               f" | lr={lr_cur:.2e} | {eps:.0f}s")

# # # # # #         # ── Eval: BUG-1 FIX: collect + commit_epoch ───────────
# # # # # #         rf = evaluate(model, vsub, dev, tag=f"FAST ep{ep}",
# # # # # #                       steps=args.fast_ddim)
# # # # # #         saver.collect(rf, model, args.output_dir, ep, opt, sched, avt, avv,
# # # # # #                       tag="fast")

# # # # # #         if ep % args.val_freq == 0:
# # # # # #             em = getattr(_unwrap(model), "_ema", None)

# # # # # #             rr = evaluate(model, vl, dev, tag=f"RAW ep{ep}",
# # # # # #                           steps=args.full_ddim)
# # # # # #             saver.collect(rr, model, args.output_dir, ep, opt, sched, avt, avv,
# # # # # #                           tag="raw")

# # # # # #             if em and ep >= 3:
# # # # # #                 re = evaluate(model, vl, dev, tag=f"EMA ep{ep}",
# # # # # #                               ema=em, steps=args.full_ddim)
# # # # # #                 saver.collect(re, model, args.output_dir, ep, opt, sched,
# # # # # #                               avt, avv, tag="ema")

# # # # # #         # BUG-1 FIX: commit once after all evals
# # # # # #         saver.commit_epoch(ep)

# # # # # #         if ep % 10 == 0 or ep == args.num_epochs - 1:
# # # # # #             _save_ckpt(
# # # # # #                 os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# # # # # #                 ep, model, opt, sched, saver, avt, avv)

# # # # # #         if saver.stop:
# # # # # #             print(f"  Early stop ep={ep}")
# # # # # #             break

# # # # # #     th = (time.perf_counter() - ts) / 3600.
# # # # # #     print(f"\n  Best: ADE={saver.ba:.1f}  72h={saver.b7:.0f}"
# # # # # #           f"  ATE={saver.bat:.1f}  CTE={saver.bc:.1f}  ({th:.2f}h)")

# # # # # #     # ── Post-training test ─────────────────────────────────────
# # # # # #     if args.eval_test_after_train:
# # # # # #         print("\n" + "="*72 + "\n  POST-TRAINING TEST\n" + "="*72)
# # # # # #         try:
# # # # # #             _, tl2 = data_loader(args,
# # # # # #                                   {"root": args.dataset_root, "type": "test"},
# # # # # #                                   test=True)
# # # # # #         except:
# # # # # #             print("  No test set → using val")
# # # # # #             tl2 = vl

# # # # # #         for fn, lb in [("best_composite.pth", "COMPOSITE"),
# # # # # #                         ("best_72h.pth",       "72H"),
# # # # # #                         ("best_ate.pth",       "ATE"),
# # # # # #                         ("best_cte.pth",       "CTE"),
# # # # # #                         ("best_ema.pth",       "EMA")]:
# # # # # #             pp = os.path.join(args.output_dir, fn)
# # # # # #             if not os.path.exists(pp): continue
# # # # # #             ck = torch.load(pp, map_location=dev)
# # # # # #             _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
# # # # # #             em = getattr(_unwrap(model), "_ema", None)
# # # # # #             if em and ck.get("ema_shadow"):
# # # # # #                 for k, v in ck["ema_shadow"].items():
# # # # # #                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
# # # # # #             r = evaluate(model, tl2, dev, tag=f"TEST/{lb}",
# # # # # #                          steps=args.full_ddim)
# # # # # #             print(f"\n  --- {lb} ---")
# # # # # #             for key, ref in [("ADE", TARGETS["ADE"]), ("72h",  TARGETS["72h"]),
# # # # # #                                ("ATE", TARGETS["ATE"]), ("CTE",  TARGETS["CTE"])]:
# # # # # #                 v  = r.get(key, float("nan"))
# # # # # #                 mk = "BEAT!" if np.isfinite(v) and v < ref else f"need {ref:.0f}"
# # # # # #                 print(f"    {key:<6}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
# # # # # #     print("=" * 72)


# # # # # # if __name__ == "__main__":
# # # # # #     args = get_args()
# # # # # #     np.random.seed(42)
# # # # # #     torch.manual_seed(42)
# # # # # #     if torch.cuda.is_available():
# # # # # #         torch.cuda.manual_seed_all(42)
# # # # # #     main(args)

# # # # # """
# # # # # train_v75.py — TC-FlowMatching v75  (Physics-Grounded Loss)
# # # # # ════════════════════════════════════════════════════════════
# # # # # CHANGES vs train_v74.py:

# # # # #   MODEL:
# # # # #     Import flow_matching_model_v75 (FMv75Loss)
# # # # #     L_disp → L_kinematic + L_logspeed + L_smooth (3 physics terms)

# # # # #   DEFAULTS CHANGED:
# # # # #     --n_ensemble:         50  → 100   (reduce sampling variance ±5→±2.5 km)
# # # # #     --fast_ddim:          10  → 15    (better fast eval estimate)
# # # # #     --full_ddim:          20  → 30    (better full eval accuracy)
# # # # #     --ema_decay:          0.995 → 0.999 (EMA was lagging in v74)
# # # # #     --cfg_guidance_scale: 1.5 → 2.0   (reduce cross-track spread → CTE)

# # # # #   PRINT LOOP: Added l_kin, l_lspd, l_sm monitoring
# # # # #     l_kin > 0.5 after ep10 → increase norm_scale to 150
# # # # #     l_sm  > 0.2 stuck      → increase w_smooth to 0.30
# # # # #     l_lspd > 1.0 after ep15 → check _to_deg conversion

# # # # #   KEEPS FROM v74 (all BUG fixes):
# # # # #     BUG-1 FIX: Saver collect+commit_epoch pattern (patience counts once/epoch)
# # # # #     BUG-2 FIX: Score formula reweighted by gap size
# # # # #     BUG-3 FIX: --finetune_lr override after resume

# # # # # RUN (fresh v75):
# # # # #   python scripts/train_v75.py \\
# # # # #       --dataset_root /path/to/tc-ofm \\
# # # # #       --output_dir   runs/v75 \\
# # # # #       --batch_size   32 \\
# # # # #       --use_amp

# # # # # RESUME from v74 best:
# # # # #   python scripts/train_v75.py \\
# # # # #       --dataset_root /path/to/tc-ofm \\
# # # # #       --output_dir   runs/v75_from_v74 \\
# # # # #       --resume       runs/v74/best_composite.pth \\
# # # # #       --finetune_lr  5e-5 \\
# # # # #       --batch_size   32 \\
# # # # #       --use_amp

# # # # # TARGETS (ST-Trans actual val):
# # # # #   ADE<172.68  72h<321.39  ATE<142.21  CTE<42.04
# # # # #   12h<65.42   24h<104.67  48h<205.10

# # # # # MONITORING:
# # # # #   Watch l_kin (should decrease ep0 ~1.0 → ep10 ~0.3 → ep30 ~0.1)
# # # # #   Watch l_sm  (should decrease ep0 ~0.15 → ep10 ~0.06)
# # # # #   Watch l_lspd (should decrease ep0 ~0.8 → ep5 ~0.3 → stabilize)
# # # # #   ATE/ADE ratio: target < 0.88 (v74 was ~0.92)
# # # # # """
# # # # # from __future__ import annotations

# # # # # import sys, os
# # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # import argparse, time, random
# # # # # from collections import defaultdict

# # # # # import numpy as np
# # # # # import torch
# # # # # import torch.optim as optim
# # # # # from torch.amp import autocast, GradScaler
# # # # # from torch.utils.data import DataLoader, Subset

# # # # # from Model.data.loader_training import data_loader
# # # # # # v75: import updated model
# # # # # from Model.flow_matching_model import TCFlowMatching

# # # # # try:
# # # # #     from Model.utils import get_cosine_schedule_with_warmup
# # # # # except ImportError:
# # # # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # # # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # # # #         return CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=min_lr)

# # # # # # ST-Trans actual val targets
# # # # # TARGETS = {
# # # # #     "ADE":   172.68,
# # # # #     "72h":   321.39,
# # # # #     "ATE":   142.21,
# # # # #     "CTE":    42.04,
# # # # #     "12h":    65.42,
# # # # #     "24h":   104.67,
# # # # #     "48h":   205.10,
# # # # # }
# # # # # R_EARTH = 6371.0


# # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # def _unwrap(m):
# # # # #     return m._orig_mod if hasattr(m, '_orig_mod') else m

# # # # # def move(b, dev):
# # # # #     out = list(b)
# # # # #     for i, x in enumerate(out):
# # # # #         if torch.is_tensor(x):
# # # # #             out[i] = x.to(dev)
# # # # #         elif isinstance(x, dict):
# # # # #             out[i] = {k: v.to(dev) if torch.is_tensor(v) else v
# # # # #                       for k, v in x.items()}
# # # # #     return out


# # # # # # ── Metric utils ──────────────────────────────────────────────────────────────

# # # # # def _ntd(a):
# # # # #     o = a.clone()
# # # # #     o[..., 0] = (a[..., 0] * 50. + 1800.) / 10.
# # # # #     o[..., 1] = (a[..., 1] * 50.) / 10.
# # # # #     return o

# # # # # def _hav(p1, p2):
# # # # #     la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
# # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # #     a = (torch.sin(dlat/2).pow(2)
# # # # #          + torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2))
# # # # #     return 2.*R_EARTH*torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

# # # # # def _atecte(pd, gd):
# # # # #     T = min(pd.shape[0], gd.shape[0])
# # # # #     if T < 2:
# # # # #         z = pd.new_zeros(1, pd.shape[1])
# # # # #         return z, z
# # # # #     lo1 = torch.deg2rad(gd[:T-1, :, 0]); la1 = torch.deg2rad(gd[:T-1, :, 1])
# # # # #     lo2 = torch.deg2rad(gd[1:T,  :, 0]); la2 = torch.deg2rad(gd[1:T,  :, 1])
# # # # #     lo3 = torch.deg2rad(pd[1:T,  :, 0]); la3 = torch.deg2rad(pd[1:T,  :, 1])
# # # # #     ya  = torch.sin(lo2-lo1)*torch.cos(la2)
# # # # #     xa  = torch.cos(la1)*torch.sin(la2) - torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # # # #     be  = torch.atan2(ya, xa)
# # # # #     ye  = torch.sin(lo3-lo2)*torch.cos(la3)
# # # # #     xe  = torch.cos(la2)*torch.sin(la3) - torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # # # #     bee = torch.atan2(ye, xe)
# # # # #     tot = _hav(pd[1:T], gd[1:T])
# # # # #     ang = bee - be
# # # # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # # class Acc:
# # # # #     def __init__(self):
# # # # #         self.d = []; self.a = []; self.c = []
# # # # #         self.sd = defaultdict(list)
# # # # #         self._h = {12: 1, 24: 3, 48: 7, 72: 11}

# # # # #     def update(self, dist, ate=None, cte=None):
# # # # #         self.d.extend(dist.mean(0).tolist())
# # # # #         for h, s in self._h.items():
# # # # #             if s < dist.shape[0]:
# # # # #                 self.sd[h].extend(dist[s].tolist())
# # # # #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# # # # #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())

# # # # #     def compute(self):
# # # # #         r = {
# # # # #             "ADE": float(np.mean(self.d))  if self.d else float("nan"),
# # # # #             "ATE": float(np.mean(self.a))  if self.a else float("nan"),
# # # # #             "CTE": float(np.mean(self.c))  if self.c else float("nan"),
# # # # #             "n":   len(self.d),
# # # # #         }
# # # # #         for h in self._h:
# # # # #             v = self.sd.get(h, [])
# # # # #             r[f"{h}h"] = float(np.mean(v)) if v else float("nan")
# # # # #         return r


# # # # # def _score(r):
# # # # #     """
# # # # #     Score formula — weights reflect gap size relative to ST-Trans targets.
# # # # #     ATE gap=24km → weight 0.20 (highest)
# # # # #     CTE gap=13km → weight 0.18
# # # # #     72h gap=10km → weight 0.22
# # # # #     ADE gap=9km  → weight 0.18
# # # # #     Lower score = better (100 = ST-Trans level)
# # # # #     """
# # # # #     ade  = r.get("ADE",   1e9)
# # # # #     h72  = r.get("72h",   1e9)
# # # # #     h48  = r.get("48h",   1e9)
# # # # #     h24  = r.get("24h",   1e9)
# # # # #     h12  = r.get("12h",   1e9)
# # # # #     ate  = r.get("ATE",   1e9)
# # # # #     cte  = r.get("CTE",   1e9)

# # # # #     if not np.isfinite(ate): ate = ade * 0.82
# # # # #     if not np.isfinite(cte): cte = ade * 0.24

# # # # #     return 100. * (
# # # # #         0.05 * (h12  / TARGETS["12h"])  +
# # # # #         0.08 * (h24  / TARGETS["24h"])  +
# # # # #         0.09 * (h48  / TARGETS["48h"])  +
# # # # #         0.22 * (h72  / TARGETS["72h"])  +
# # # # #         0.18 * (ade  / TARGETS["ADE"])  +
# # # # #         0.20 * (ate  / TARGETS["ATE"])  +
# # # # #         0.18 * (cte  / TARGETS["CTE"])
# # # # #     )


# # # # # def _beat(r):
# # # # #     p = []
# # # # #     for k, t in [("ADE", TARGETS["ADE"]), ("ATE", TARGETS["ATE"]),
# # # # #                   ("CTE", TARGETS["CTE"]), ("72h", TARGETS["72h"]),
# # # # #                   ("12h", TARGETS["12h"]), ("24h", TARGETS["24h"]),
# # # # #                   ("48h", TARGETS["48h"])]:
# # # # #         v = r.get(k, 1e9)
# # # # #         if np.isfinite(v) and v < t:
# # # # #             p.append(f"{k}:{v:.1f}")
# # # # #     return "*** BEAT ST-TRANS: " + " ".join(p) + " ***" if p else ""


# # # # # def _gap(r):
# # # # #     out = []
# # # # #     for k, ref in [("ADE", TARGETS["ADE"]), ("72h", TARGETS["72h"]),
# # # # #                     ("ATE", TARGETS["ATE"]),  ("CTE", TARGETS["CTE"])]:
# # # # #         v = r.get(k, float("nan"))
# # # # #         if np.isfinite(v):
# # # # #             out.append(f"{k}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
# # # # #     return " | ".join(out)


# # # # # # ── Evaluation ────────────────────────────────────────────────────────────────

# # # # # @torch.no_grad()
# # # # # def evaluate(model, loader, dev, tag="", ema=None, steps=20):
# # # # #     bk = None
# # # # #     if ema:
# # # # #         try:
# # # # #             bk = ema.apply_to(model)
# # # # #         except:
# # # # #             pass
# # # # #     model.eval()
# # # # #     acc = Acc()
# # # # #     t0  = time.perf_counter()
# # # # #     for b in loader:
# # # # #         bl = move(list(b), dev)
# # # # #         result = model.sample(bl, ddim_steps=steps)
# # # # #         p = result[0] if isinstance(result, (tuple, list)) else result
# # # # #         g = bl[1]
# # # # #         T = min(p.shape[0], g.shape[0])
# # # # #         pd = _ntd(p[:T]); gd = _ntd(g[:T])
# # # # #         dist = _hav(pd, gd)
# # # # #         at, ct = _atecte(pd, gd)
# # # # #         acc.update(dist, at, ct)
# # # # #     if bk:
# # # # #         try:
# # # # #             ema.restore(model, bk)
# # # # #         except:
# # # # #             pass
# # # # #     r  = acc.compute()
# # # # #     el = time.perf_counter() - t0

# # # # #     def _v(k):   return r.get(k, float("nan"))
# # # # #     def _m(v, t): return "ok" if np.isfinite(v) and v < t else "no"

# # # # #     print(f"\n{'='*70}")
# # # # #     print(f"  [{tag}  {el:.0f}s]")
# # # # #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'), TARGETS['ADE'])}]"
# # # # #           f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
# # # # #           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'), TARGETS['72h'])}]")
# # # # #     print(f"  ATE={_v('ATE'):.1f}[{_m(_v('ATE'), TARGETS['ATE'])}]"
# # # # #           f"  CTE={_v('CTE'):.1f}[{_m(_v('CTE'), TARGETS['CTE'])}]")
# # # # #     print(f"  vs ST-Trans: {_gap(r)}")
# # # # #     bt = _beat(r)
# # # # #     if bt: print(f"  {bt}")
# # # # #     print(f"  Score={_score(r):.2f}")
# # # # #     print(f"{'='*70}\n")
# # # # #     return r


# # # # # # ── Saver (BUG-1 FIX: collect+commit_epoch) ──────────────────────────────────

# # # # # def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl, extra=None):
# # # # #     m   = _unwrap(model)
# # # # #     ema = getattr(m, "_ema", None)
# # # # #     esd = None
# # # # #     if ema and hasattr(ema, "shadow"):
# # # # #         try:
# # # # #             esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # # # #         except:
# # # # #             pass
# # # # #     d = {
# # # # #         "epoch":            ep,
# # # # #         "model_state_dict": m.state_dict(),
# # # # #         "optimizer_state":  opt.state_dict(),
# # # # #         "scheduler_state":  sched.state_dict(),
# # # # #         "ema_shadow":       esd,
# # # # #         "best_score":       saver.bs,
# # # # #         "best_ade":         saver.ba,
# # # # #         "best_72h":         saver.b7,
# # # # #         "best_ate":         saver.bat,
# # # # #         "best_cte":         saver.bc,
# # # # #         "train_loss":       tl,
# # # # #         "val_loss":         vl,
# # # # #         "version":          "v75",
# # # # #     }
# # # # #     if extra:
# # # # #         d.update(extra)
# # # # #     torch.save(d, path)


# # # # # class Saver:
# # # # #     """
# # # # #     BUG-1 FIX from v74: collect_epoch() + commit_epoch() pattern.
# # # # #     All evals within one epoch are gathered, patience updated only once.
# # # # #     → patience counts correctly per epoch, not per eval call.
# # # # #     """
# # # # #     def __init__(self, patience=40, min_ep=35):
# # # # #         self.patience = patience
# # # # #         self.min_ep   = min_ep
# # # # #         self.cnt  = 0
# # # # #         self.stop = False
# # # # #         self.bs = self.ba = self.b7 = self.bat = self.bc = float("inf")
# # # # #         self._epoch_best_score  = float("inf")
# # # # #         self._epoch_best_result = None
# # # # #         self._epoch_best_tag    = ""

# # # # #     def collect(self, r, model, out, ep, opt, sched, tl, vl, tag=""):
# # # # #         sc  = _score(r)
# # # # #         ade = r.get("ADE", 1e9); h72 = r.get("72h", 1e9)
# # # # #         ate = r.get("ATE",  1e9); cte = r.get("CTE", 1e9)

# # # # #         for v, a, fn in [(ade, "ba", "best_ade.pth"),
# # # # #                           (h72, "b7", "best_72h.pth"),
# # # # #                           (ate, "bat","best_ate.pth"),
# # # # #                           (cte, "bc", "best_cte.pth")]:
# # # # #             if v < getattr(self, a):
# # # # #                 setattr(self, a, v)
# # # # #                 _save_ckpt(os.path.join(out, fn),
# # # # #                            ep, model, opt, sched, self, tl, vl)

# # # # #         if sc < self._epoch_best_score:
# # # # #             self._epoch_best_score  = sc
# # # # #             self._epoch_best_result = r
# # # # #             self._epoch_best_tag    = tag
# # # # #             if sc < self.bs:
# # # # #                 _save_ckpt(os.path.join(out, "best_composite.pth"),
# # # # #                            ep, model, opt, sched, self, tl, vl,
# # # # #                            {"score": sc, "ade": ade, "h72": h72})
# # # # #                 print(f"  [BEST] {tag} ep={ep} score={sc:.2f}"
# # # # #                       f"  ADE={ade:.1f}  72h={h72:.0f}"
# # # # #                       f"  ATE={ate:.1f}  CTE={cte:.1f}")

# # # # #         # Mode collapse detector:
# # # # #         # Collapse signature: CTE stops improving while ADE increases
# # # # #         # Compare to best known values
# # # # #         if ep >= 10 and np.isfinite(ade) and np.isfinite(cte):
# # # # #             if ade > self.ba * 1.05 and cte > self.bc * 1.10:
# # # # #                 print(f"  [COLLAPSE?] {tag} ADE={ade:.1f}(best={self.ba:.1f}) "
# # # # #                       f"CTE={cte:.1f}(best={self.bc:.1f})"
# # # # #                       f" → if persistent, reduce --cfg_guidance_scale to 1.5")

# # # # #     def commit_epoch(self, ep):
# # # # #         sc = self._epoch_best_score
# # # # #         if sc < self.bs:
# # # # #             self.bs  = sc
# # # # #             self.cnt = 0
# # # # #         else:
# # # # #             self.cnt += 1
# # # # #             print(f"  No improve {self.cnt}/{self.patience}"
# # # # #                   f"  best={self.bs:.2f}  cur={sc:.2f}")

# # # # #         self._epoch_best_score  = float("inf")
# # # # #         self._epoch_best_result = None
# # # # #         self._epoch_best_tag    = ""

# # # # #         if ep >= self.min_ep and self.cnt >= self.patience:
# # # # #             self.stop = True
# # # # #             print(f"  [STOP] ep={ep}")


# # # # # def mksub(ds, n, bs, cf):
# # # # #     idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
# # # # #     return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
# # # # #                       collate_fn=cf, num_workers=0, drop_last=False)


# # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # def get_args():
# # # # #     p = argparse.ArgumentParser(
# # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # #     p.add_argument("--dataset_root",    default="TCND_vn")
# # # # #     p.add_argument("--obs_len",         default=8,    type=int)
# # # # #     p.add_argument("--pred_len",        default=12,   type=int)
# # # # #     p.add_argument("--batch_size",      default=32,   type=int)
# # # # #     p.add_argument("--num_epochs",      default=120,  type=int)
# # # # #     p.add_argument("--learning_rate",   default=1e-4, type=float)
# # # # #     p.add_argument("--weight_decay",    default=1e-3, type=float)
# # # # #     p.add_argument("--warmup_epochs",   default=3,    type=int)
# # # # #     p.add_argument("--grad_clip",       default=1.0,  type=float)
# # # # #     p.add_argument("--patience",        default=40,   type=int)
# # # # #     p.add_argument("--min_ep",          default=35,   type=int)
# # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # #     p.add_argument("--num_workers",     default=2,    type=int)
# # # # #     p.add_argument("--sigma_min",       default=0.02, type=float)
# # # # #     p.add_argument("--use_ot",          default=True, action="store_true")
# # # # #     p.add_argument("--no_ot",           dest="use_ot", action="store_false")
# # # # #     p.add_argument("--cfg_guidance_scale", default=2.0, type=float)
# # # # #     p.add_argument("--easy_thresh",     default=0.25, type=float)
# # # # #     p.add_argument("--n_ensemble",      default=100,  type=int)
# # # # #     p.add_argument("--val_freq",        default=3,    type=int)
# # # # #     p.add_argument("--val_subset_size", default=500,  type=int)
# # # # #     p.add_argument("--fast_ddim",       default=15,   type=int)
# # # # #     p.add_argument("--full_ddim",       default=30,   type=int)
# # # # #     p.add_argument("--use_ema",         default=True, action="store_true")
# # # # #     p.add_argument("--no_ema",          dest="use_ema", action="store_false")
# # # # #     p.add_argument("--ema_decay",       default=0.999, type=float)
# # # # #     p.add_argument("--output_dir",      default="runs/v75")
# # # # #     p.add_argument("--gpu_num",         default="0")
# # # # #     p.add_argument("--delim",           default=" ")
# # # # #     p.add_argument("--skip",            default=1,    type=int)
# # # # #     p.add_argument("--min_ped",         default=1,    type=int)
# # # # #     p.add_argument("--threshold",       default=0.002, type=float)
# # # # #     p.add_argument("--other_modal",     default="gph")
# # # # #     p.add_argument("--test_year",       default=None, type=int)
# # # # #     p.add_argument("--resume",          default=None)
# # # # #     p.add_argument("--resume_epoch",    default=None, type=int)
# # # # #     p.add_argument("--finetune_lr",     default=None, type=float)
# # # # #     p.add_argument("--eval_test_after_train", default=True, action="store_true")
# # # # #     p.add_argument("--w_kin",    default=1.0,  type=float)
# # # # #     p.add_argument("--w_lspd",   default=0.15, type=float)
# # # # #     p.add_argument("--w_smooth", default=0.20, type=float)
# # # # #     return p.parse_args()


# # # # # # ── Main ──────────────────────────────────────────────────────────────────────

# # # # # def main(args):
# # # # #     if torch.cuda.is_available():
# # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # #     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # #     print("=" * 72)
# # # # #     print(f"  TC-FlowMatching v75  (Physics-Grounded Loss)")
# # # # #     print(f"  Loss: L_fm + L_pos(step_w) + L_kinematic + 0.15*L_logspeed + 0.20*L_smooth")
# # # # #     print(f"  L_kinematic = velocity_MSE = ATE² + CTE²  (identity decomposition)")
# # # # #     print(f"  Split: DifficultyWeighter (continuous soft, NOT binary easy/hard)")
# # # # #     print(f"  Training: single phase, patience={args.patience}, min_ep={args.min_ep}")
# # # # #     print(f"  Inference: K=3 mode clustering at 72h endpoint, n_ensemble={args.n_ensemble}")
# # # # #     print(f"  EMA decay={args.ema_decay}  cfg_guidance={args.cfg_guidance_scale}")
# # # # #     print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']}"
# # # # #           f" ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
# # # # #     if args.resume:
# # # # #         print(f"  RESUME: {args.resume}")
# # # # #     if args.finetune_lr:
# # # # #         print(f"  FINETUNE LR: {args.finetune_lr}")
# # # # #     print("=" * 72)

# # # # #     # ── Data ──────────────────────────────────────────────────
# # # # #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"},
# # # # #                             test=False)
# # # # #     vd, vl   = data_loader(args, {"root": args.dataset_root, "type": "val"},
# # # # #                             test=True)
# # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # #     vsub = mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
# # # # #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# # # # #     # ── Model ─────────────────────────────────────────────────
# # # # #     model = TCFlowMatching(
# # # # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # # # #         sigma_min=args.sigma_min, use_ema=args.use_ema,
# # # # #         ema_decay=args.ema_decay, use_ot=args.use_ot,
# # # # #         cfg_guidance_scale=args.cfg_guidance_scale,
# # # # #         easy_thresh=args.easy_thresh, n_ensemble=args.n_ensemble,
# # # # #     ).to(dev)
# # # # #     model.init_ema()
# # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # #     sw_p = sum(p.numel() for p in model.criterion.step_weights.parameters())
# # # # #     lw_p = sum(p.numel() for p in model.criterion.loss_weights.parameters())
# # # # #     print(f"  params total: {n_params:,} | step_w: {sw_p} | loss_w: {lw_p}")

# # # # #     # ── Optimizer & Scheduler ──────────────────────────────────
# # # # #     init_lr = args.finetune_lr if args.finetune_lr else args.learning_rate
# # # # #     opt     = optim.AdamW(model.parameters(), lr=init_lr,
# # # # #                            weight_decay=args.weight_decay)
# # # # #     saver   = Saver(patience=args.patience, min_ep=args.min_ep)
# # # # #     scaler  = GradScaler("cuda", enabled=args.use_amp)
# # # # #     nstep   = len(trl)
# # # # #     total   = nstep * args.num_epochs
# # # # #     wstp    = nstep * args.warmup_epochs
# # # # #     sched   = get_cosine_schedule_with_warmup(opt, wstp, total, min_lr=1e-6)

# # # # #     # ── Resume ────────────────────────────────────────────────
# # # # #     start = 0
# # # # #     if args.resume and os.path.exists(args.resume):
# # # # #         print(f"  Loading checkpoint: {args.resume}")
# # # # #         ck = torch.load(args.resume, map_location=dev)
# # # # #         m  = _unwrap(model)
# # # # #         missing, unexpected = m.load_state_dict(ck["model_state_dict"], strict=False)
# # # # #         if missing:    print(f"  Missing keys: {len(missing)}")
# # # # #         if unexpected: print(f"  Unexpected keys: {len(unexpected)}")

# # # # #         ema = getattr(m, "_ema", None)
# # # # #         if ema and ck.get("ema_shadow"):
# # # # #             loaded = 0
# # # # #             for k, v in ck["ema_shadow"].items():
# # # # #                 if k in ema.shadow:
# # # # #                     ema.shadow[k].copy_(v.to(dev)); loaded += 1
# # # # #             print(f"  EMA shadow loaded: {loaded} tensors")

# # # # #         if args.finetune_lr:
# # # # #             print(f"  Fine-tune mode: LR={args.finetune_lr}")
# # # # #             sched = get_cosine_schedule_with_warmup(
# # # # #                 opt, wstp//4, total, min_lr=args.finetune_lr/20.)
# # # # #         else:
# # # # #             try:
# # # # #                 opt.load_state_dict(ck["optimizer_state"])
# # # # #                 print(f"  Optimizer loaded")
# # # # #             except Exception as e:
# # # # #                 print(f"  Optimizer not loaded: {e}")
# # # # #             try:
# # # # #                 sched.load_state_dict(ck["scheduler_state"])
# # # # #                 print(f"  Scheduler loaded")
# # # # #             except Exception as e:
# # # # #                 print(f"  Scheduler not loaded, manual forward: {e}")
# # # # #                 ep_ck = ck.get("epoch", 0)
# # # # #                 for _ in range(ep_ck * nstep):
# # # # #                     sched.step()

# # # # #         for a, attr in [("best_score", "bs"), ("best_ade", "ba"),
# # # # #                           ("best_72h", "b7"), ("best_ate", "bat"),
# # # # #                           ("best_cte", "bc")]:
# # # # #             if a in ck: setattr(saver, attr, ck[a])
# # # # #         start = args.resume_epoch if args.resume_epoch is not None \
# # # # #                 else ck.get("epoch", 0) + 1
# # # # #         print(f"  Resume from epoch {start}"
# # # # #               f"  best_score={saver.bs:.2f}  best_ade={saver.ba:.1f}")

# # # # #     # ── Compile ───────────────────────────────────────────────
# # # # #     try:
# # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # #         print("  torch.compile: ok")
# # # # #     except Exception as e:
# # # # #         print(f"  torch.compile skipped: {e}")

# # # # #     ts = time.perf_counter()
# # # # #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# # # # #     print("=" * 72)

# # # # #     # ══════════════════════════════════════════════════════════
# # # # #     for ep in range(start, args.num_epochs):
# # # # #         model.train()
# # # # #         sl = 0.
# # # # #         t0 = time.perf_counter()

# # # # #         for i, batch in enumerate(trl):
# # # # #             bl = move(list(batch), dev)
# # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # #                 bd = model.get_loss_breakdown(bl, epoch=ep)

# # # # #             opt.zero_grad()
# # # # #             scaler.scale(bd["total"]).backward()
# # # # #             scaler.unscale_(opt)
# # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # #             sb = scaler.get_scale()
# # # # #             scaler.step(opt)
# # # # #             scaler.update()
# # # # #             if scaler.get_scale() >= sb:
# # # # #                 sched.step()
# # # # #             model.ema_update()
# # # # #             sl += bd["total"].item()

# # # # #             if i % 20 == 0:
# # # # #                 lr     = opt.param_groups[0]["lr"]
# # # # #                 sw_r   = bd.get("sw_ratio", 0.)
# # # # #                 tot    = bd["total"].item()
# # # # #                 sw_w   = " [!sw<3]"   if sw_r < 3.0 else ""
# # # # #                 tot_w  = " [!tot>10]" if tot  > 10. else ""
# # # # #                 l_kin  = bd.get("l_kin",  0.)
# # # # #                 l_lspd = bd.get("l_lspd", 0.)
# # # # #                 l_sm   = bd.get("l_sm",   0.)
# # # # #                 # Accurate thresholds (from numerical analysis):
# # # # #                 # l_kin: auto-calibrated by gt_speed, expect ~0.04-0.15 ep0 → ~0.02 ep30
# # # # #                 #        warn if > 0.5 ONLY after ep20 (sign of divergence, not scale issue)
# # # # #                 # l_sm:  max possible = 0.5 by construction. ep0 ~0.15-0.25, ep10+ ~0.04-0.08
# # # # #                 #        warn if > 0.15 after ep15 (should have converged by then)
# # # # #                 # l_lspd: ep0 ~0.4, ep10 ~0.2, warn if > 0.5 after ep20
# # # # #                 kin_w  = " [!kin_div]"  if l_kin  > 0.5  and ep >= 20 else ""
# # # # #                 sm_w   = " [!sm_high]"  if l_sm   > 0.15 and ep >= 15 else ""
# # # # #                 lspd_w = " [!lspd_div]" if l_lspd > 0.5  and ep >= 20 else ""
# # # # #                 print(
# # # # #                     f"  [{ep:>3}][{i:>3}/{nstep}]"
# # # # #                     f"  tot={tot:.3f}{tot_w}"
# # # # #                     f"  fm={bd.get('l_fm',0):.4f}"
# # # # #                     f"  pos={bd.get('l_pos',0):.4f}"
# # # # #                     f"  kin={l_kin:.4f}{kin_w}"
# # # # #                     f"  lspd={l_lspd:.4f}{lspd_w}"
# # # # #                     f"  sm={l_sm:.4f}{sm_w}"
# # # # #                     f"  sw72={bd.get('sw_72h',0):.2f}"
# # # # #                     f"  swr={sw_r:.1f}{sw_w}"
# # # # #                     f"  pos/fm={bd.get('lw_pos_fm',0):.2f}"
# # # # #                     f"  dw={bd.get('diff_w_mean',1):.2f}"
# # # # #                     f"  lr={lr:.2e}"
# # # # #                 )

# # # # #         avt = sl / nstep

# # # # #         # Val loss
# # # # #         model.eval()
# # # # #         vls = 0.
# # # # #         with torch.no_grad():
# # # # #             for batch in vl:
# # # # #                 bv = move(list(batch), dev)
# # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # #                     vls += model.get_loss(bv, epoch=ep).item()
# # # # #         avv = vls / len(vl)
# # # # #         eps = time.perf_counter() - t0
# # # # #         lr_cur = opt.param_groups[0]["lr"]
# # # # #         print(f"  Epoch {ep:>3} | train={avt:.4f} val={avv:.4f}"
# # # # #               f" | lr={lr_cur:.2e} | {eps:.0f}s")

# # # # #         # ── Eval: collect + commit_epoch ──────────────────────
# # # # #         rf = evaluate(model, vsub, dev, tag=f"FAST ep{ep}",
# # # # #                       steps=args.fast_ddim)
# # # # #         saver.collect(rf, model, args.output_dir, ep, opt, sched, avt, avv,
# # # # #                       tag="fast")

# # # # #         if ep % args.val_freq == 0:
# # # # #             em = getattr(_unwrap(model), "_ema", None)

# # # # #             rr = evaluate(model, vl, dev, tag=f"RAW ep{ep}",
# # # # #                           steps=args.full_ddim)
# # # # #             saver.collect(rr, model, args.output_dir, ep, opt, sched, avt, avv,
# # # # #                           tag="raw")

# # # # #             if em and ep >= 3:
# # # # #                 re = evaluate(model, vl, dev, tag=f"EMA ep{ep}",
# # # # #                               ema=em, steps=args.full_ddim)
# # # # #                 saver.collect(re, model, args.output_dir, ep, opt, sched,
# # # # #                               avt, avv, tag="ema")

# # # # #         # BUG-1 FIX: commit once after all evals
# # # # #         saver.commit_epoch(ep)

# # # # #         if ep % 10 == 0 or ep == args.num_epochs - 1:
# # # # #             _save_ckpt(
# # # # #                 os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# # # # #                 ep, model, opt, sched, saver, avt, avv)

# # # # #         if saver.stop:
# # # # #             print(f"  Early stop ep={ep}")
# # # # #             break

# # # # #     th = (time.perf_counter() - ts) / 3600.
# # # # #     print(f"\n  Best: ADE={saver.ba:.1f}  72h={saver.b7:.0f}"
# # # # #           f"  ATE={saver.bat:.1f}  CTE={saver.bc:.1f}  ({th:.2f}h)")

# # # # #     # ── Post-training test ─────────────────────────────────────
# # # # #     if args.eval_test_after_train:
# # # # #         print("\n" + "="*72 + "\n  POST-TRAINING TEST\n" + "="*72)
# # # # #         try:
# # # # #             _, tl2 = data_loader(args,
# # # # #                                   {"root": args.dataset_root, "type": "test"},
# # # # #                                   test=True)
# # # # #         except:
# # # # #             print("  No test set → using val")
# # # # #             tl2 = vl

# # # # #         for fn, lb in [("best_composite.pth", "COMPOSITE"),
# # # # #                         ("best_72h.pth",       "72H"),
# # # # #                         ("best_ate.pth",       "ATE"),
# # # # #                         ("best_cte.pth",       "CTE"),
# # # # #                         ("best_ema.pth",       "EMA")]:
# # # # #             pp = os.path.join(args.output_dir, fn)
# # # # #             if not os.path.exists(pp): continue
# # # # #             ck = torch.load(pp, map_location=dev)
# # # # #             _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
# # # # #             em = getattr(_unwrap(model), "_ema", None)
# # # # #             if em and ck.get("ema_shadow"):
# # # # #                 for k, v in ck["ema_shadow"].items():
# # # # #                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
# # # # #             r = evaluate(model, tl2, dev, tag=f"TEST/{lb}",
# # # # #                          steps=args.full_ddim)
# # # # #             print(f"\n  --- {lb} ---")
# # # # #             for key, ref in [("ADE", TARGETS["ADE"]), ("72h",  TARGETS["72h"]),
# # # # #                                ("ATE", TARGETS["ATE"]), ("CTE",  TARGETS["CTE"])]:
# # # # #                 v  = r.get(key, float("nan"))
# # # # #                 mk = "BEAT!" if np.isfinite(v) and v < ref else f"need {ref:.0f}"
# # # # #                 print(f"    {key:<6}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
# # # # #     print("=" * 72)


# # # # # if __name__ == "__main__":
# # # # #     args = get_args()
# # # # #     np.random.seed(42)
# # # # #     torch.manual_seed(42)
# # # # #     if torch.cuda.is_available():
# # # # #         torch.cuda.manual_seed_all(42)
# # # # #     main(args)

# # # # """
# # # # train_v76.py — TC-FlowMatching v76
# # # # ════════════════════════════════════════════════════════════════════════════════
# # # # Training loop với 4 bugs từ train_v74.py đã được fix:

# # # #   BUG-5 FIX: Val loss dùng epoch=-1 → skip augmentation → clean scheduler signal
# # # #   BUG-6 FIX: Saver reset cnt khi ANY metric improve, không chỉ composite score
# # # #   BUG-7 FIX: sched.step() unconditional (trong try/except)
# # # #   BUG-8 FIX: grad_clip chỉ áp dụng backbone, không clip learned weights

# # # # Model changes ở flow_matching_model_v76.py (BUG-1..4).

# # # # RUN (fresh):
# # # #   python scripts/train_v76.py \
# # # #       --dataset_root /path/to/tc-ofm \
# # # #       --output_dir   runs/v76 \
# # # #       --batch_size   32 \
# # # #       --use_amp

# # # # RESUME từ v74/v75 checkpoint:
# # # #   python scripts/train_v76.py \
# # # #       --dataset_root /path/to/tc-ofm \
# # # #       --output_dir   runs/v76_resume \
# # # #       --resume       runs/v74/best_composite.pth \
# # # #       --batch_size   32 \
# # # #       --use_amp

# # # # MONITORING:
# # # #   sw_ratio ≥ 8.0     → step weights đang maintain long-range focus (BUG-3 fix)
# # # #   lw_pos_fm ≥ 2.0    → L_pos_late dominant so với L_fm (đúng hướng)
# # # #   tot ∈ [1.5, 6.0]   → loss scale healthy
# # # #   dw ∈ [1.05, 1.40]  → DifficultyWeighter đang hoạt động (BUG-1 fix)
# # # #   val_loss stable     → clean (no aug) signal cho scheduler (BUG-5 fix)

# # # # TARGETS (ST-Trans val):
# # # #   ADE < 172.68 | 72h < 321.39 | ATE < 142.21 | CTE < 42.04
# # # #   12h < 65.42  | 24h < 104.67 | 48h < 205.10
# # # # """
# # # # from __future__ import annotations

# # # # import sys, os
# # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # import argparse, time, random
# # # # from collections import defaultdict

# # # # import numpy as np
# # # # import torch
# # # # import torch.optim as optim
# # # # from torch.amp import autocast, GradScaler
# # # # from torch.utils.data import DataLoader, Subset

# # # # from Model.data.loader_training import data_loader
# # # # from Model.flow_matching_model import TCFlowMatchingV76

# # # # try:
# # # #     from Model.utils import get_cosine_schedule_with_warmup
# # # # except ImportError:
# # # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # # #         return CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=min_lr)

# # # # TARGETS = {
# # # #     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
# # # #     "12h":  65.42, "24h": 104.67, "48h": 205.10,
# # # # }
# # # # R_EARTH = 6371.0


# # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # def _unwrap(m): return m._orig_mod if hasattr(m, '_orig_mod') else m

# # # # def move(b, dev):
# # # #     out = list(b)
# # # #     for i, x in enumerate(out):
# # # #         if torch.is_tensor(x): out[i] = x.to(dev)
# # # #         elif isinstance(x, dict):
# # # #             out[i] = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in x.items()}
# # # #     return out


# # # # # ── Metric utils  (identical to v74) ─────────────────────────────────────────

# # # # def _ntd(a):
# # # #     o = a.clone()
# # # #     o[...,0] = (a[...,0]*50.+1800.)/10.
# # # #     o[...,1] = (a[...,1]*50.)/10.
# # # #     return o

# # # # def _hav(p1, p2):
# # # #     la1=torch.deg2rad(p1[...,1]); la2=torch.deg2rad(p2[...,1])
# # # #     dlat=torch.deg2rad(p2[...,1]-p1[...,1]); dlon=torch.deg2rad(p2[...,0]-p1[...,0])
# # # #     a=torch.sin(dlat/2).pow(2)+torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2)
# # # #     return 2.*R_EARTH*torch.asin(a.clamp(1e-12,1-1e-12).sqrt())

# # # # def _atecte(pd, gd):
# # # #     T=min(pd.shape[0],gd.shape[0])
# # # #     if T<2: z=pd.new_zeros(1,pd.shape[1]); return z,z
# # # #     lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
# # # #     lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
# # # #     lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
# # # #     ya=torch.sin(lo2-lo1)*torch.cos(la2)
# # # #     xa=torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # # #     be=torch.atan2(ya,xa)
# # # #     ye=torch.sin(lo3-lo2)*torch.cos(la3)
# # # #     xe=torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # # #     bee=torch.atan2(ye,xe); tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
# # # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # class Acc:
# # # #     def __init__(self):
# # # #         self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list)
# # # #         self._h={12:1, 24:3, 48:7, 72:11}
# # # #     def update(self, dist, ate=None, cte=None):
# # # #         self.d.extend(dist.mean(0).tolist())
# # # #         for h,s in self._h.items():
# # # #             if s<dist.shape[0]: self.sd[h].extend(dist[s].tolist())
# # # #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# # # #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())
# # # #     def compute(self):
# # # #         r = {"ADE":     float(np.mean(self.d))     if self.d else float("nan"),
# # # #              "ATE_mean":float(np.mean(self.a))     if self.a else float("nan"),
# # # #              "CTE_mean":float(np.mean(self.c))     if self.c else float("nan"),
# # # #              "n": len(self.d)}
# # # #         for h in self._h:
# # # #             v=self.sd.get(h,[]); r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
# # # #         return r


# # # # def _score(r):
# # # #     ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
# # # #     ate=r.get("ATE_mean",1e9); cte=r.get("CTE_mean",1e9)
# # # #     if not np.isfinite(ate): ate=ade*.46
# # # #     if not np.isfinite(cte): cte=ade*.53
# # # #     return 100.*(0.05*(ade/136.) + 0.10*(r.get("12h",ade)/50.)
# # # #                  + 0.15*(r.get("24h",ade)/100.) + 0.20*(r.get("48h",ade)/200.)
# # # #                  + 0.25*(h72/300.) + 0.13*(ate/80.) + 0.12*(cte/94.))

# # # # def _beat(r):
# # # #     p=[]
# # # #     for k,t in [("ADE",172.68),("ATE_mean",142.21),("CTE_mean",42.04),
# # # #                   ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
# # # #         v=r.get(k,1e9)
# # # #         if np.isfinite(v) and v<t: p.append(f"{k.replace('_mean','')}:{v:.1f}")
# # # #     return "*** BEAT ST-TRANS: "+" ".join(p)+" ***" if p else ""

# # # # def _gap(r):
# # # #     out=[]
# # # #     for k,ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
# # # #         v=r.get(k,float("nan"))
# # # #         if np.isfinite(v):
# # # #             out.append(f"{k.replace('_mean','')}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
# # # #     return " | ".join(out)


# # # # # ── Evaluation  (identical to v74) ───────────────────────────────────────────

# # # # @torch.no_grad()
# # # # def evaluate(model, loader, dev, tag="", ema=None, steps=20):
# # # #     bk=None
# # # #     if ema:
# # # #         try: bk=ema.apply_to(model)
# # # #         except: pass
# # # #     model.eval(); acc=Acc(); t0=time.perf_counter()
# # # #     for b in loader:
# # # #         bl=move(list(b), dev)
# # # #         result=model.sample(bl, ddim_steps=steps)
# # # #         p=result[0] if isinstance(result,(tuple,list)) else result
# # # #         g=bl[1]; T=min(p.shape[0],g.shape[0])
# # # #         pd=_ntd(p[:T]); gd=_ntd(g[:T])
# # # #         dist=_hav(pd,gd); at,ct=_atecte(pd,gd)
# # # #         acc.update(dist,at,ct)
# # # #     if bk:
# # # #         try: ema.restore(model,bk)
# # # #         except: pass
# # # #     r=acc.compute()
# # # #     el=time.perf_counter()-t0
# # # #     def _v(k): return r.get(k, float("nan"))
# # # #     def _m(v,t): return "ok" if np.isfinite(v) and v<t else "no"
# # # #     print(f"\n{'='*70}")
# # # #     print(f"  [{tag}  {el:.0f}s]")
# # # #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]"
# # # #           f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
# # # #           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
# # # #     if np.isfinite(_v("ATE_mean")):
# # # #         print(f"  ATE={_v('ATE_mean'):.1f}[{_m(_v('ATE_mean'),142.21)}]"
# # # #               f"  CTE={_v('CTE_mean'):.1f}[{_m(_v('CTE_mean'),42.04)}]")
# # # #     print(f"  vs ST-Trans: {_gap(r)}")
# # # #     bt=_beat(r)
# # # #     if bt: print(f"  {bt}")
# # # #     print(f"  Score={_score(r):.2f}")
# # # #     print(f"{'='*70}\n")
# # # #     return r


# # # # # ── Checkpoint ────────────────────────────────────────────────────────────────

# # # # def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl, extra=None):
# # # #     m=_unwrap(model); ema=getattr(m,"_ema",None); esd=None
# # # #     if ema and hasattr(ema,"shadow"):
# # # #         try: esd={k:v.cpu().clone() for k,v in ema.shadow.items()}
# # # #         except: pass
# # # #     d={"epoch":ep, "model_state_dict":m.state_dict(),
# # # #        "optimizer_state":opt.state_dict(), "scheduler_state":sched.state_dict(),
# # # #        "ema_shadow":esd,
# # # #        "best_score":saver.bs, "best_ade":saver.ba, "best_72h":saver.b7,
# # # #        "best_ate":saver.bat, "best_cte":saver.bc,
# # # #        "train_loss":tl, "val_loss":vl, "version":"v76"}
# # # #     if extra: d.update(extra)
# # # #     torch.save(d, path)


# # # # # ── Saver  [BUG-6 FIX] ───────────────────────────────────────────────────────

# # # # class Saver:
# # # #     """
# # # #     BUG-6 FIX: cnt reset khi ANY metric improve.

# # # #     v74 bug scenario:
# # # #       ep N:   ADE 180→178 (↓ improve)  CTE 50→51 (↑ worse)
# # # #               composite score không improve → cnt += 1
# # # #       ep N+1: ADE 178→177 (↓)  72h improve
# # # #               composite might not improve → cnt += 1 again
# # # #       → Sau 40 epochs: early stop dù model đang rõ ràng học được!

# # # #     Fix: any_improved = True nếu BẤT KỲ metric nào trong {ADE, 72h, ATE, CTE}
# # # #     đạt best mới. cnt chỉ tăng khi KHÔNG CÓ gì improve.

# # # #     Tại sao đúng hơn:
# # # #       Multi-objective optimization thường improve từng metric một, không đồng thời.
# # # #       Early stop nên trigger chỉ khi tất cả metrics đều stuck.
# # # #     """
# # # #     def __init__(self, patience=40, min_ep=35):
# # # #         self.patience = patience; self.min_ep = min_ep
# # # #         self.cnt  = 0; self.stop = False
# # # #         self.bs = self.ba = self.b7 = self.bat = self.bc = float("inf")

# # # #     def update(self, r, model, out, ep, opt, sched, tl, vl, tag=""):
# # # #         sc  = _score(r)
# # # #         ade = r.get("ADE", 1e9);       h72 = r.get("72h", 1e9)
# # # #         ate = r.get("ATE_mean", 1e9);  cte = r.get("CTE_mean", 1e9)

# # # #         # BUG-6 FIX: track any_improved across all metrics
# # # #         any_improved = False

# # # #         # Save best per-metric checkpoints
# # # #         for v, attr, fn in [(ade,"ba","best_ade.pth"),
# # # #                              (h72,"b7","best_72h.pth"),
# # # #                              (ate,"bat","best_ate.pth"),
# # # #                              (cte,"bc","best_cte.pth")]:
# # # #             if v < getattr(self, attr):
# # # #                 setattr(self, attr, v)
# # # #                 _save_ckpt(os.path.join(out, fn), ep, model, opt, sched, self, tl, vl)
# # # #                 any_improved = True   # ← BUG-6 FIX: any metric improve counts

# # # #         # Save best composite
# # # #         if sc < self.bs:
# # # #             self.bs = sc
# # # #             any_improved = True
# # # #             _save_ckpt(os.path.join(out, f"best_{tag or 'composite'}.pth"),
# # # #                        ep, model, opt, sched, self, tl, vl,
# # # #                        {"score":sc, "ade":ade, "h72":h72})
# # # #             print(f"  [BEST] {tag} ep={ep} score={sc:.2f}"
# # # #                   f"  ADE={ade:.1f}  72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}")

# # # #         # BUG-6 FIX: cnt resets when ANY metric improves
# # # #         if any_improved:
# # # #             self.cnt = 0
# # # #         else:
# # # #             self.cnt += 1
# # # #             print(f"  No improve {self.cnt}/{self.patience}"
# # # #                   f"  best={self.bs:.2f}  cur={sc:.2f}")

# # # #         if ep >= self.min_ep and self.cnt >= self.patience:
# # # #             self.stop = True; print(f"  [STOP] ep={ep}")


# # # # def mksub(ds, n, bs, cf):
# # # #     idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
# # # #     return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
# # # #                        collate_fn=cf, num_workers=0, drop_last=False)


# # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # def get_args():
# # # #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # #     p.add_argument("--dataset_root",       default="TCND_vn")
# # # #     p.add_argument("--obs_len",            default=8,     type=int)
# # # #     p.add_argument("--pred_len",           default=12,    type=int)
# # # #     p.add_argument("--batch_size",         default=32,    type=int)
# # # #     p.add_argument("--num_epochs",         default=120,   type=int)
# # # #     p.add_argument("--learning_rate",      default=1e-4,  type=float)
# # # #     p.add_argument("--weight_decay",       default=1e-3,  type=float)
# # # #     p.add_argument("--warmup_epochs",      default=3,     type=int)
# # # #     p.add_argument("--grad_clip",          default=1.0,   type=float)
# # # #     p.add_argument("--patience",           default=40,    type=int)
# # # #     p.add_argument("--min_ep",             default=35,    type=int)
# # # #     p.add_argument("--use_amp",            action="store_true")
# # # #     p.add_argument("--num_workers",        default=2,     type=int)
# # # #     p.add_argument("--sigma_min",          default=0.02,  type=float)
# # # #     p.add_argument("--use_ot",             default=True,  action="store_true")
# # # #     p.add_argument("--no_ot",              dest="use_ot", action="store_false")
# # # #     p.add_argument("--cfg_guidance_scale", default=1.5,   type=float)
# # # #     p.add_argument("--n_ensemble",         default=100,   type=int)
# # # #     p.add_argument("--val_freq",           default=3,     type=int)
# # # #     p.add_argument("--val_subset_size",    default=500,   type=int)
# # # #     p.add_argument("--fast_ddim",          default=10,    type=int)
# # # #     p.add_argument("--full_ddim",          default=20,    type=int)
# # # #     p.add_argument("--use_ema",            default=True,  action="store_true")
# # # #     p.add_argument("--no_ema",             dest="use_ema",action="store_false")
# # # #     p.add_argument("--ema_decay",          default=0.999, type=float)
# # # #     p.add_argument("--output_dir",         default="runs/v76")
# # # #     p.add_argument("--gpu_num",            default="0")
# # # #     p.add_argument("--delim",              default=" ")
# # # #     p.add_argument("--skip",               default=1,     type=int)
# # # #     p.add_argument("--min_ped",            default=1,     type=int)
# # # #     p.add_argument("--threshold",          default=0.002, type=float)
# # # #     p.add_argument("--other_modal",        default="gph")
# # # #     p.add_argument("--test_year",          default=None,  type=int)
# # # #     p.add_argument("--resume",             default=None)
# # # #     p.add_argument("--resume_epoch",       default=None,  type=int)
# # # #     p.add_argument("--eval_test_after_train", default=True, action="store_true")
# # # #     return p.parse_args()


# # # # # ── Main ──────────────────────────────────────────────────────────────────────

# # # # def main(args):
# # # #     if torch.cuda.is_available():
# # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # #     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # #     print("="*72)
# # # #     print(f"  TC-FlowMatching v76")
# # # #     print(f"  Loss: L_fm + L_pos_early + L_pos_late + L_disp  [ALL weights learned]")
# # # #     print(f"  Bugs fixed: BUG-1(sample_w) BUG-2(ratio_clamp) BUG-3(sw_penalty)")
# # # #     print(f"              BUG-4(pos_split) BUG-5(val_noaug) BUG-6(any_improve)")
# # # #     print(f"              BUG-7(sched_uncond) BUG-8(clip_backbone)")
# # # #     print(f"  n_ensemble={args.n_ensemble}  ema_decay={args.ema_decay}  bs={args.batch_size}")
# # # #     print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']}"
# # # #           f" ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
# # # #     print("="*72)

# # # #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # #     vd,  vl  = data_loader(args, {"root": args.dataset_root, "type": "val"},   test=True)
# # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # #     vsub = mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
# # # #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# # # #     model = TCFlowMatchingV76(
# # # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # # #         sigma_min=args.sigma_min, use_ema=args.use_ema, ema_decay=args.ema_decay,
# # # #         use_ot=args.use_ot, cfg_guidance_scale=args.cfg_guidance_scale,
# # # #         n_ensemble=args.n_ensemble,
# # # #     ).to(dev)
# # # #     model.init_ema()

# # # #     n_total  = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # #     sw_n     = sum(p.numel() for p in model.criterion.step_weights.parameters())
# # # #     lw_n     = sum(p.numel() for p in model.criterion.loss_weights.parameters())
# # # #     dw_n     = sum(p.numel() for p in model.criterion.diff_weighter.parameters())
# # # #     print(f"  params total: {n_total:,} | step_w:{sw_n} | loss_w:{lw_n} | diff_w:{dw_n}")

# # # #     # ── BUG-8 FIX: separate backbone from learned weight params ──────────────
# # # #     # step_weights, loss_weights, diff_weighter:
# # # #     #   - Need gradient to flow freely (no clip) to adapt quickly
# # # #     #   - Especially step_weights: needs large gradient to maintain sw_ratio ≥ 8
# # # #     #   - loss_weights: anchor_loss already provides regularization, no need for clip
# # # #     # backbone params: standard clip_grad_norm with grad_clip=1.0
# # # #     learned_ids = set(
# # # #         id(p)
# # # #         for grp in [model.criterion.step_weights.parameters(),
# # # #                     model.criterion.loss_weights.parameters(),
# # # #                     model.criterion.diff_weighter.parameters()]
# # # #         for p in grp
# # # #     )
# # # #     backbone_params = [p for p in model.parameters()
# # # #                        if p.requires_grad and id(p) not in learned_ids]
# # # #     learned_params  = [p for p in model.parameters()
# # # #                        if p.requires_grad and id(p) in learned_ids]
# # # #     print(f"  backbone: {sum(p.numel() for p in backbone_params):,} (clipped at {args.grad_clip})")
# # # #     print(f"  learned_w: {sum(p.numel() for p in learned_params):,} (NOT clipped)")

# # # #     # Single optimizer — same LR for all params
# # # #     opt    = optim.AdamW(model.parameters(), lr=args.learning_rate,
# # # #                           weight_decay=args.weight_decay)
# # # #     saver  = Saver(patience=args.patience, min_ep=args.min_ep)
# # # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # # #     nstep  = len(trl)
# # # #     total  = nstep * args.num_epochs
# # # #     wstp   = nstep * args.warmup_epochs
# # # #     sched  = get_cosine_schedule_with_warmup(opt, wstp, total, min_lr=1e-6)

# # # #     # ── Resume ────────────────────────────────────────────────────────────────
# # # #     start = 0
# # # #     if args.resume and os.path.exists(args.resume):
# # # #         print(f"  Loading: {args.resume}")
# # # #         ck = torch.load(args.resume, map_location=dev)
# # # #         m  = _unwrap(model)
# # # #         ms, _ = m.load_state_dict(ck["model_state_dict"], strict=False)
# # # #         if ms: print(f"  Missing keys: {len(ms)}")
# # # #         ema = getattr(m, "_ema", None)
# # # #         if ema and ck.get("ema_shadow"):
# # # #             for k, v in ck["ema_shadow"].items():
# # # #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
# # # #         try:
# # # #             opt.load_state_dict(ck["optimizer_state"])
# # # #         except Exception as e:
# # # #             print(f"  Opt not loaded: {e}")
# # # #         try:
# # # #             sched.load_state_dict(ck["scheduler_state"])
# # # #         except Exception:
# # # #             for _ in range(ck.get("epoch", 0) * nstep): sched.step()
# # # #         for a, attr in [("best_score","bs"),("best_ade","ba"),("best_72h","b7"),
# # # #                           ("best_ate","bat"),("best_cte","bc")]:
# # # #             if a in ck: setattr(saver, attr, ck[a])
# # # #         start = args.resume_epoch or ck.get("epoch", 0) + 1
# # # #         print(f"  Resume ep={start}  best_score={saver.bs:.2f}  best_ade={saver.ba:.1f}")

# # # #     try:
# # # #         model = torch.compile(model, mode="reduce-overhead")
# # # #         print("  torch.compile: ok")
# # # #     except Exception: pass

# # # #     ts = time.perf_counter()
# # # #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# # # #     print("="*72)

# # # #     for ep in range(start, args.num_epochs):
# # # #         model.train(); sl = 0.; t0 = time.perf_counter()

# # # #         for i, batch in enumerate(trl):
# # # #             bl = move(list(batch), dev)

# # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # #                 # epoch>=0 → augmentation applied (train mode)
# # # #                 bd = model.get_loss_breakdown(bl, epoch=ep)

# # # #             opt.zero_grad()
# # # #             scaler.scale(bd["total"]).backward()
# # # #             scaler.unscale_(opt)

# # # #             # BUG-8 FIX: clip ONLY backbone params, not learned weights
# # # #             torch.nn.utils.clip_grad_norm_(backbone_params, args.grad_clip)

# # # #             scaler.step(opt); scaler.update()

# # # #             # BUG-7 FIX: sched.step() unconditional
# # # #             # v74 bug: conditional on scaler.get_scale()>=sb → ~5% batches skip
# # # #             # → LR curve lệch khỏi schedule, đặc biệt trong late training
# # # #             try:
# # # #                 sched.step()
# # # #             except Exception:
# # # #                 pass

# # # #             model.ema_update()
# # # #             sl += bd["total"].item()

# # # #             if i % 20 == 0:
# # # #                 lr   = opt.param_groups[0]["lr"]
# # # #                 swr  = bd.get("sw_ratio", 0.)
# # # #                 tot  = bd["total"].item()
# # # #                 sw_w = " [!sw<8]" if swr < 8.0  else ""
# # # #                 to_w = " [!>10]"  if tot > 10.  else ""
# # # #                 print(
# # # #                     f"  [{ep:>3}][{i:>3}/{nstep}]"
# # # #                     f"  tot={tot:.3f}{to_w}"
# # # #                     f"  fm={bd.get('l_fm',0):.4f}"
# # # #                     f"  pos_e={bd.get('l_pos_e',0):.4f}"
# # # #                     f"  pos_l={bd.get('l_pos_l',0):.4f}"
# # # #                     f"  disp={bd.get('l_disp',0):.4f}"
# # # #                     f"  sw72={bd.get('sw_72h',0):.2f}"
# # # #                     f"  swr={swr:.1f}{sw_w}"
# # # #                     f"  pl/fm={bd.get('lw_pos_fm',0):.2f}"
# # # #                     f"  dw={bd.get('diff_w_mean',1):.2f}"
# # # #                     f"  lr={lr:.2e}"
# # # #                 )

# # # #         avt = sl / nstep

# # # #         # BUG-5 FIX: val loss với epoch=-1 → skip augmentation
# # # #         # → val_loss là clean signal cho scheduler, không bị noise từ aug
# # # #         model.eval(); vls = 0.
# # # #         with torch.no_grad():
# # # #             for batch in vl:
# # # #                 bv = move(list(batch), dev)
# # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # #                     vls += model.get_loss(bv, epoch=-1).item()   # epoch=-1: no aug
# # # #         avv    = vls / len(vl)
# # # #         eps_t  = time.perf_counter() - t0
# # # #         lr_cur = opt.param_groups[0]["lr"]
# # # #         print(f"  Epoch {ep:>3} | train={avt:.4f} val={avv:.4f}"
# # # #               f" | lr={lr_cur:.2e} | {eps_t:.0f}s")

# # # #         # Fast eval (subset)
# # # #         rf = evaluate(model, vsub, dev, tag=f"FAST ep{ep}", steps=args.fast_ddim)
# # # #         saver.update(rf, model, args.output_dir, ep, opt, sched, avt, avv, tag="fast")

# # # #         # Full val every val_freq + EMA
# # # #         if ep % args.val_freq == 0:
# # # #             em = getattr(_unwrap(model), "_ema", None)
# # # #             rr = evaluate(model, vl, dev, tag=f"RAW ep{ep}", steps=args.full_ddim)
# # # #             saver.update(rr, model, args.output_dir, ep, opt, sched, avt, avv, tag="raw")
# # # #             if em and ep >= 3:
# # # #                 re = evaluate(model, vl, dev, tag=f"EMA ep{ep}",
# # # #                                ema=em, steps=args.full_ddim)
# # # #                 saver.update(re, model, args.output_dir, ep, opt, sched,
# # # #                               avt, avv, tag="ema")

# # # #         if ep % 10 == 0 or ep == args.num_epochs - 1:
# # # #             _save_ckpt(os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# # # #                         ep, model, opt, sched, saver, avt, avv)

# # # #         if saver.stop: print(f"  Early stop ep={ep}"); break

# # # #     th = (time.perf_counter() - ts) / 3600.
# # # #     print(f"\n  Best: ADE={saver.ba:.1f}  72h={saver.b7:.0f}"
# # # #           f"  ATE={saver.bat:.1f}  CTE={saver.bc:.1f}  ({th:.2f}h)")

# # # #     # Post-training test
# # # #     if args.eval_test_after_train:
# # # #         print("\n"+"="*72+"\n  POST-TRAINING TEST\n"+"="*72)
# # # #         try: _, tl2 = data_loader(args, {"root":args.dataset_root,"type":"test"}, test=True)
# # # #         except: print("  No test set → using val"); tl2 = vl
# # # #         for fn, lb in [("best_composite.pth","COMPOSITE"),("best_72h.pth","72H"),
# # # #                         ("best_ema.pth","EMA"),("best_ate.pth","ATE"),("best_cte.pth","CTE")]:
# # # #             pp = os.path.join(args.output_dir, fn)
# # # #             if not os.path.exists(pp): continue
# # # #             ck = torch.load(pp, map_location=dev)
# # # #             _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
# # # #             em = getattr(_unwrap(model), "_ema", None)
# # # #             if em and ck.get("ema_shadow"):
# # # #                 for k, v in ck["ema_shadow"].items():
# # # #                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
# # # #             r = evaluate(model, tl2, dev, tag=f"TEST/{lb}", steps=args.full_ddim)
# # # #             print(f"\n  --- {lb} ---")
# # # #             for key, ref in [("ADE",172.68),("72h",321.39),
# # # #                                ("ATE_mean",142.21),("CTE_mean",42.04)]:
# # # #                 v  = r.get(key, float("nan"))
# # # #                 mk = "BEAT!" if np.isfinite(v) and v < ref else f"need {ref:.0f}"
# # # #                 print(f"    {key:<12}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
# # # #     print("="*72)


# # # # if __name__ == "__main__":
# # # #     args = get_args()
# # # #     np.random.seed(42); torch.manual_seed(42)
# # # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# # # #     main(args)

# # # """
# # # train_v83.py  —  TC-FlowMatching v83
# # # ══════════════════════════════════════════════════════════════════════════════
# # # Thay đổi so với train_v76.py:

# # #   [T1] Early stop CHỈ tính cho full val (RAW và EMA).
# # #        Fast val (subset) CHỈ để monitor + logging, không affect early stop.
# # #        Không so sánh fast vs full.

# # #   [T2] Saver tách biệt cho fast và full:
# # #        fast_saver: lưu checkpoint tốt nhất trên fast val (save best fast)
# # #        full_saver: lưu checkpoint tốt nhất trên full val + điều khiển early stop
# # #        EMA saver: separate tracking

# # #   [T3] BUG-5 fix: val_loss dùng epoch=-1 → skip augmentation

# # #   [T4] BUG-7 fix: sched.step() unconditional

# # #   [T5] BUG-8 fix: grad_clip chỉ backbone

# # #   [T6] BUG-6 fix: any_improve = any metric improve (không chỉ composite score)

# # #   [T7] V83 curriculum params exposed qua args

# # #   [T8] Log thêm lambda_hard, q_hard_mean để monitor curriculum

# # #   [T9] Full val freq = mỗi 3 epoch (giống v76)
# # #        Early stop patience = 40 epochs full val (không fast val)
# # #        Min epoch để early stop check = 35

# # # MONITORING:
# # #   lambda_hard > 0 bắt đầu từ ep 20 → hard curriculum active
# # #   q_hard_mean ∈ [0.3, 0.6] → difficulty gate đang phân biệt được
# # #   dpe giảm đều → DPE loss working
# # #   signed_cte giảm → CTE improving
# # #   val loss stable → không oscillate (BUG-5 fix)

# # # TARGETS: ADE ≈ 173 (gần ST-Trans 172.68) | 72h < 321 | ATE < 142 | CTE < 42
# # # """
# # # from __future__ import annotations

# # # import sys, os
# # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # import argparse, time, random
# # # from collections import defaultdict

# # # import numpy as np
# # # import torch
# # # import torch.optim as optim
# # # from torch.amp import autocast, GradScaler
# # # from torch.utils.data import DataLoader, Subset

# # # from Model.data.loader_training import data_loader
# # # from Model.flow_matching_model import TCFlowMatchingV83

# # # try:
# # #     from Model.utils import get_cosine_schedule_with_warmup
# # # except ImportError:
# # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # #         return CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=min_lr)

# # # TARGETS = {
# # #     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
# # #     "12h":  65.42, "24h": 104.67, "48h": 205.10,
# # # }
# # # R_EARTH = 6371.0


# # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # def _unwrap(m):
# # #     return m._orig_mod if hasattr(m, "_orig_mod") else m


# # # def move(b, dev):
# # #     out = list(b)
# # #     for i, x in enumerate(out):
# # #         if torch.is_tensor(x):
# # #             out[i] = x.to(dev)
# # #         elif isinstance(x, dict):
# # #             out[i] = {k: v.to(dev) if torch.is_tensor(v) else v
# # #                       for k, v in x.items()}
# # #     return out


# # # # ── Metric utils ──────────────────────────────────────────────────────────────

# # # def _ntd(a: torch.Tensor) -> torch.Tensor:
# # #     o = a.clone()
# # #     o[...,0] = (a[...,0]*50.+1800.)/10.
# # #     o[...,1] = (a[...,1]*50.)/10.
# # #     return o


# # # def _hav(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # #     la1=torch.deg2rad(p1[...,1]); la2=torch.deg2rad(p2[...,1])
# # #     dlat=torch.deg2rad(p2[...,1]-p1[...,1])
# # #     dlon=torch.deg2rad(p2[...,0]-p1[...,0])
# # #     a=torch.sin(dlat/2).pow(2)+torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2)
# # #     return 2.*R_EARTH*torch.asin(a.clamp(1e-12,1-1e-12).sqrt())


# # # def _atecte(pd: torch.Tensor, gd: torch.Tensor):
# # #     T=min(pd.shape[0],gd.shape[0])
# # #     if T<2:
# # #         z=pd.new_zeros(1,pd.shape[1]); return z,z
# # #     lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
# # #     lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
# # #     lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
# # #     ya=torch.sin(lo2-lo1)*torch.cos(la2)
# # #     xa=torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # #     be=torch.atan2(ya,xa)
# # #     ye=torch.sin(lo3-lo2)*torch.cos(la3)
# # #     xe=torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # #     bee=torch.atan2(ye,xe)
# # #     tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
# # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # class Acc:
# # #     def __init__(self):
# # #         self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list)
# # #         self._h={12:1, 24:3, 48:7, 72:11}

# # #     def update(self, dist, ate=None, cte=None):
# # #         self.d.extend(dist.mean(0).tolist())
# # #         for h,s in self._h.items():
# # #             if s<dist.shape[0]: self.sd[h].extend(dist[s].tolist())
# # #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# # #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())

# # #     def compute(self) -> dict:
# # #         r = {
# # #             "ADE":      float(np.mean(self.d))     if self.d else float("nan"),
# # #             "ATE_mean": float(np.mean(self.a))     if self.a else float("nan"),
# # #             "CTE_mean": float(np.mean(self.c))     if self.c else float("nan"),
# # #             "n": len(self.d),
# # #         }
# # #         for h in self._h:
# # #             v=self.sd.get(h,[])
# # #             r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
# # #         return r


# # # def _score(r: dict) -> float:
# # #     ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
# # #     ate=r.get("ATE_mean",1e9); cte=r.get("CTE_mean",1e9)
# # #     if not np.isfinite(ate): ate=ade*.46
# # #     if not np.isfinite(cte): cte=ade*.53
# # #     return 100.*(
# # #         0.05*(ade/136.)
# # #         + 0.10*(r.get("12h",ade)/50.)
# # #         + 0.15*(r.get("24h",ade)/100.)
# # #         + 0.20*(r.get("48h",ade)/200.)
# # #         + 0.25*(h72/300.)
# # #         + 0.13*(ate/80.)
# # #         + 0.12*(cte/94.)
# # #     )


# # # def _beat(r: dict) -> str:
# # #     p=[]
# # #     for k,t in [("ADE",172.68),("ATE_mean",142.21),("CTE_mean",42.04),
# # #                   ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
# # #         v=r.get(k,1e9)
# # #         if np.isfinite(v) and v<t:
# # #             p.append(f"{k.replace('_mean','')}:{v:.1f}")
# # #     return "*** BEAT ST-TRANS: "+" ".join(p)+" ***" if p else ""


# # # def _gap(r: dict) -> str:
# # #     out=[]
# # #     for k,ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
# # #         v=r.get(k,float("nan"))
# # #         if np.isfinite(v):
# # #             out.append(f"{k.replace('_mean','')}:{v:.0f}"
# # #                        f"({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
# # #     return " | ".join(out)


# # # # ── Evaluation ────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def evaluate(model, loader, dev, tag: str = "", ema=None, steps: int = 20) -> dict:
# # #     bk = None
# # #     if ema:
# # #         try: bk = ema.apply_to(model)
# # #         except: pass
# # #     model.eval(); acc = Acc(); t0 = time.perf_counter()
# # #     for b in loader:
# # #         bl = move(list(b), dev)
# # #         result = model.sample(bl, ddim_steps=steps)
# # #         p = result[0] if isinstance(result, (tuple, list)) else result
# # #         g = bl[1]; T = min(p.shape[0], g.shape[0])
# # #         pd = _ntd(p[:T]); gd = _ntd(g[:T])
# # #         dist = _hav(pd, gd); at, ct = _atecte(pd, gd)
# # #         acc.update(dist, at, ct)
# # #     if bk:
# # #         try: ema.restore(model, bk)
# # #         except: pass
# # #     r = acc.compute()
# # #     el = time.perf_counter() - t0

# # #     def _v(k): return r.get(k, float("nan"))
# # #     def _m(v, t): return "ok" if np.isfinite(v) and v < t else "no"

# # #     print(f"\n{'='*70}")
# # #     print(f"  [{tag}  {el:.0f}s]")
# # #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]"
# # #           f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
# # #           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
# # #     if np.isfinite(_v("ATE_mean")):
# # #         print(f"  ATE={_v('ATE_mean'):.1f}[{_m(_v('ATE_mean'),142.21)}]"
# # #               f"  CTE={_v('CTE_mean'):.1f}[{_m(_v('CTE_mean'),42.04)}]")
# # #     print(f"  vs ST-Trans: {_gap(r)}")
# # #     bt = _beat(r)
# # #     if bt: print(f"  {bt}")
# # #     print(f"  Score={_score(r):.2f}")
# # #     print(f"{'='*70}\n")
# # #     return r


# # # # ── Checkpoint ────────────────────────────────────────────────────────────────

# # # def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl, extra=None):
# # #     m = _unwrap(model); ema = getattr(m, "_ema", None); esd = None
# # #     if ema and hasattr(ema, "shadow"):
# # #         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # #         except: pass
# # #     d = {
# # #         "epoch": ep,
# # #         "model_state_dict": m.state_dict(),
# # #         "optimizer_state": opt.state_dict(),
# # #         "scheduler_state": sched.state_dict(),
# # #         "ema_shadow": esd,
# # #         "best_score": saver.bs, "best_ade": saver.ba,
# # #         "best_72h": saver.b7,  "best_ate": saver.bat,
# # #         "best_cte": saver.bc,
# # #         "train_loss": tl, "val_loss": vl, "version": "v83",
# # #     }
# # #     if extra: d.update(extra)
# # #     torch.save(d, path)


# # # # ── Saver — tách biệt fast và full ────────────────────────────────────────────

# # # class Saver:
# # #     """
# # #     [T1] Early stop CHỈ khi mode='full'.
# # #          mode='fast': chỉ save checkpoint, không affect stop.
# # #          mode='ema': track riêng, can stop.

# # #     [T2] any_improve = True nếu BẤT KỲ metric nào improve (BUG-6 fix).
# # #     """
# # #     def __init__(self, patience: int = 40, min_ep: int = 35,
# # #                  enable_stop: bool = True):
# # #         self.patience    = patience
# # #         self.min_ep      = min_ep
# # #         self.enable_stop = enable_stop   # False for fast_saver → không stop
# # #         self.cnt  = 0
# # #         self.stop = False
# # #         self.bs = self.ba = self.b7 = self.bat = self.bc = float("inf")

# # #     def update(self, r: dict, model, out_dir: str, ep: int,
# # #                opt, sched, tl: float, vl: float, tag: str = ""):
# # #         sc  = _score(r)
# # #         ade = r.get("ADE",     1e9)
# # #         h72 = r.get("72h",     1e9)
# # #         ate = r.get("ATE_mean",1e9)
# # #         cte = r.get("CTE_mean",1e9)

# # #         any_improved = False

# # #         # Save best per-metric checkpoints
# # #         for v, attr, fn in [
# # #             (ade, "ba", f"best_ade_{tag}.pth"  if tag else "best_ade.pth"),
# # #             (h72, "b7", f"best_72h_{tag}.pth"  if tag else "best_72h.pth"),
# # #             (ate, "bat",f"best_ate_{tag}.pth"  if tag else "best_ate.pth"),
# # #             (cte, "bc", f"best_cte_{tag}.pth"  if tag else "best_cte.pth"),
# # #         ]:
# # #             if v < getattr(self, attr):
# # #                 setattr(self, attr, v)
# # #                 _save_ckpt(os.path.join(out_dir, fn),
# # #                            ep, model, opt, sched, self, tl, vl)
# # #                 any_improved = True

# # #         # Save best composite
# # #         if sc < self.bs:
# # #             self.bs = sc
# # #             any_improved = True
# # #             fn = f"best_{tag or 'composite'}.pth"
# # #             _save_ckpt(os.path.join(out_dir, fn),
# # #                        ep, model, opt, sched, self, tl, vl,
# # #                        {"score": sc, "ade": ade, "h72": h72})
# # #             print(f"  [BEST] {tag} ep={ep} score={sc:.2f}"
# # #                   f"  ADE={ade:.1f}  72h={h72:.0f}"
# # #                   f"  ATE={ate:.1f}  CTE={cte:.1f}")

# # #         # Early stop logic — chỉ khi enable_stop=True
# # #         if self.enable_stop:
# # #             if any_improved:
# # #                 self.cnt = 0
# # #             else:
# # #                 self.cnt += 1
# # #                 print(f"  No improve {self.cnt}/{self.patience}"
# # #                       f"  best={self.bs:.2f}  cur={sc:.2f}")
# # #             if ep >= self.min_ep and self.cnt >= self.patience:
# # #                 self.stop = True
# # #                 print(f"  [STOP] ep={ep} patience={self.patience}")
# # #         else:
# # #             # Fast saver: just log, no stop
# # #             if any_improved:
# # #                 pass  # already printed BEST above
# # #             else:
# # #                 pass  # silent for fast val no-improve


# # # def mksub(ds, n: int, bs: int, cf):
# # #     idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
# # #     return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
# # #                        collate_fn=cf, num_workers=0, drop_last=False)


# # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # def get_args():
# # #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # #     # Dataset
# # #     p.add_argument("--dataset_root",          default="TCND_vn")
# # #     p.add_argument("--obs_len",               default=8,     type=int)
# # #     p.add_argument("--pred_len",              default=12,    type=int)
# # #     # Training
# # #     p.add_argument("--batch_size",            default=32,    type=int)
# # #     p.add_argument("--num_epochs",            default=120,   type=int)
# # #     p.add_argument("--learning_rate",         default=1e-4,  type=float)
# # #     p.add_argument("--weight_decay",          default=1e-3,  type=float)
# # #     p.add_argument("--warmup_epochs",         default=3,     type=int)
# # #     p.add_argument("--grad_clip",             default=1.0,   type=float)
# # #     p.add_argument("--use_amp",               action="store_true")
# # #     p.add_argument("--num_workers",           default=2,     type=int)
# # #     # Model
# # #     p.add_argument("--sigma_min",             default=0.02,  type=float)
# # #     p.add_argument("--use_ot",                default=True,  action="store_true")
# # #     p.add_argument("--no_ot",                 dest="use_ot", action="store_false")
# # #     p.add_argument("--cfg_guidance_scale",    default=1.5,   type=float)
# # #     p.add_argument("--n_ensemble",            default=50,    type=int)
# # #     # EMA
# # #     p.add_argument("--use_ema",               default=True,  action="store_true")
# # #     p.add_argument("--no_ema",                dest="use_ema",action="store_false")
# # #     p.add_argument("--ema_decay",             default=0.999, type=float)
# # #     # [V83] Curriculum params
# # #     p.add_argument("--hard_start_epoch",      default=20,    type=int,
# # #                    help="Epoch bắt đầu hard curriculum ramp")
# # #     p.add_argument("--hard_ramp_epochs",      default=10,    type=int,
# # #                    help="Số epoch ramp lambda_hard từ 0→1")
# # #     p.add_argument("--diff_threshold",        default=0.4,   type=float,
# # #                    help="Ngưỡng difficulty để phân biệt easy/hard")
# # #     p.add_argument("--diff_temp",             default=0.15,  type=float,
# # #                    help="Temperature của soft gate sigmoid")
# # #     # [T1] Early stop — chỉ cho full val
# # #     p.add_argument("--patience",              default=40,    type=int,
# # #                    help="Early stop patience (chỉ tính trên full val)")
# # #     p.add_argument("--min_ep",                default=35,    type=int)
# # #     # Eval schedule
# # #     p.add_argument("--val_freq",              default=3,     type=int,
# # #                    help="Full val mỗi N epochs")
# # #     p.add_argument("--val_subset_size",       default=500,   type=int)
# # #     p.add_argument("--fast_ddim",             default=10,    type=int)
# # #     p.add_argument("--full_ddim",             default=20,    type=int)
# # #     # Output
# # #     p.add_argument("--output_dir",            default="runs/v83")
# # #     p.add_argument("--gpu_num",               default="0")
# # #     # Data args
# # #     p.add_argument("--delim",                 default=" ")
# # #     p.add_argument("--skip",                  default=1,     type=int)
# # #     p.add_argument("--min_ped",               default=1,     type=int)
# # #     p.add_argument("--threshold",             default=0.002, type=float)
# # #     p.add_argument("--other_modal",           default="gph")
# # #     p.add_argument("--test_year",             default=None,  type=int)
# # #     # Resume
# # #     p.add_argument("--resume",                default=None)
# # #     p.add_argument("--resume_epoch",          default=None,  type=int)
# # #     p.add_argument("--eval_test_after_train", default=True,  action="store_true")
# # #     return p.parse_args()


# # # # ── Main ──────────────────────────────────────────────────────────────────────

# # # def main(args):
# # #     if torch.cuda.is_available():
# # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # #     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #     os.makedirs(args.output_dir, exist_ok=True)

# # #     print("="*72)
# # #     print(f"  TC-FlowMatching v83")
# # #     print(f"  Base: v59 backbone (FNO3D+Mamba+Transformer)")
# # #     print(f"  + Physics anchor x0  + Soft difficulty curriculum")
# # #     print(f"  + v76 bug fixes (BUG-5,6,7,8)")
# # #     print(f"  + Full-traj medoid inference (không blend mode)")
# # #     print(f"  EMA decay={args.ema_decay}  n_ensemble={args.n_ensemble}")
# # #     print(f"  Hard curriculum: start_ep={args.hard_start_epoch}"
# # #           f"  ramp={args.hard_ramp_epochs}")
# # #     print(f"  Early stop: patience={args.patience} min_ep={args.min_ep}"
# # #           f"  [FULL VAL ONLY]")
# # #     print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']}"
# # #           f" ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
# # #     print("="*72)

# # #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"},
# # #                             test=False)
# # #     vd, vl   = data_loader(args, {"root": args.dataset_root, "type": "val"},
# # #                              test=True)
# # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # #     vsub = mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
# # #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# # #     model = TCFlowMatchingV83(
# # #         pred_len            = args.pred_len,
# # #         obs_len             = args.obs_len,
# # #         sigma_min           = args.sigma_min,
# # #         use_ema             = args.use_ema,
# # #         ema_decay           = args.ema_decay,
# # #         use_ate_ot          = args.use_ot,
# # #         cfg_guidance_scale  = args.cfg_guidance_scale,
# # #         hard_start_epoch    = args.hard_start_epoch,
# # #         hard_ramp_epochs    = args.hard_ramp_epochs,
# # #         diff_threshold      = args.diff_threshold,
# # #         diff_temp           = args.diff_temp,
# # #     ).to(dev)
# # #     model.init_ema()

# # #     n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # #     print(f"  params: {n_total:,}")

# # #     # ── BUG-8 FIX: backbone vs learned weight params ───────────────────────
# # #     # v83 không có explicit learned weight modules như v76,
# # #     # nhưng step_scale, physics_scale, steering_scale vẫn cần gradient tự do
# # #     # → clip chỉ main backbone (tất cả trừ những params rất nhỏ)
# # #     # Thực tế: clip_grad_norm_ áp dụng toàn bộ là an toàn hơn nếu không có
# # #     # explicit learned_w module. Vẫn giữ full backbone clip như v59 nhưng
# # #     # chỉ clip params có numel > 100 (bỏ qua bias và scale vectors nhỏ).
# # #     small_params   = [p for p in model.parameters()
# # #                       if p.requires_grad and p.numel() <= 100]
# # #     backbone_params = [p for p in model.parameters()
# # #                        if p.requires_grad and p.numel() > 100]
# # #     print(f"  backbone params (clipped): {sum(p.numel() for p in backbone_params):,}")
# # #     print(f"  small params (not clipped): {sum(p.numel() for p in small_params):,}")

# # #     opt    = optim.AdamW(model.parameters(), lr=args.learning_rate,
# # #                           weight_decay=args.weight_decay)
# # #     scaler = GradScaler("cuda", enabled=args.use_amp)

# # #     nstep  = len(trl)
# # #     total  = nstep * args.num_epochs
# # #     wstp   = nstep * args.warmup_epochs
# # #     sched  = get_cosine_schedule_with_warmup(opt, wstp, total, min_lr=1e-6)

# # #     # [T1] Tách biệt fast_saver và full_saver
# # #     # fast_saver: không có early stop (enable_stop=False)
# # #     # full_saver: có early stop (enable_stop=True) — đây là "chính"
# # #     fast_saver = Saver(patience=args.patience, min_ep=args.min_ep,
# # #                        enable_stop=False)  # monitor only
# # #     full_saver = Saver(patience=args.patience, min_ep=args.min_ep,
# # #                        enable_stop=True)   # controls early stop

# # #     # ── Resume ────────────────────────────────────────────────────────────────
# # #     start = 0
# # #     if args.resume and os.path.exists(args.resume):
# # #         print(f"  Loading: {args.resume}")
# # #         ck = torch.load(args.resume, map_location=dev)
# # #         m  = _unwrap(model)
# # #         ms, _ = m.load_state_dict(ck["model_state_dict"], strict=False)
# # #         if ms: print(f"  Missing keys: {len(ms)}")
# # #         ema = getattr(m, "_ema", None)
# # #         if ema and ck.get("ema_shadow"):
# # #             for k, v in ck["ema_shadow"].items():
# # #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
# # #         try:
# # #             opt.load_state_dict(ck["optimizer_state"])
# # #         except Exception as e:
# # #             print(f"  Opt not loaded: {e}")
# # #         try:
# # #             sched.load_state_dict(ck["scheduler_state"])
# # #         except Exception:
# # #             for _ in range(ck.get("epoch", 0) * nstep): sched.step()
# # #         # Restore saver state
# # #         for a, attr in [("best_score","bs"),("best_ade","ba"),("best_72h","b7"),
# # #                           ("best_ate","bat"),("best_cte","bc")]:
# # #             if a in ck:
# # #                 setattr(full_saver, attr, ck[a])
# # #                 setattr(fast_saver, attr, ck[a])
# # #         start = args.resume_epoch or ck.get("epoch", 0) + 1
# # #         print(f"  Resume ep={start}  best_score={full_saver.bs:.2f}"
# # #               f"  best_ade={full_saver.ba:.1f}")

# # #     try:
# # #         model = torch.compile(model, mode="reduce-overhead")
# # #         print("  torch.compile: ok")
# # #     except Exception:
# # #         pass

# # #     ts = time.perf_counter()
# # #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# # #     print("="*72)

# # #     for ep in range(start, args.num_epochs):
# # #         model.train(); sl = 0.; t0 = time.perf_counter()

# # #         for i, batch in enumerate(trl):
# # #             bl = move(list(batch), dev)

# # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # #                 bd = model.get_loss_breakdown(bl, epoch=ep)

# # #             opt.zero_grad()
# # #             scaler.scale(bd["total"]).backward()
# # #             scaler.unscale_(opt)

# # #             # BUG-8 FIX: clip chỉ backbone (params lớn)
# # #             torch.nn.utils.clip_grad_norm_(backbone_params, args.grad_clip)

# # #             scaler.step(opt); scaler.update()

# # #             # BUG-7 FIX: sched.step() unconditional
# # #             try:
# # #                 sched.step()
# # #             except Exception:
# # #                 pass

# # #             model.ema_update()
# # #             sl += bd["total"].item()

# # #             if i % 20 == 0:
# # #                 lr  = opt.param_groups[0]["lr"]
# # #                 tot = bd["total"].item()
# # #                 lh  = bd.get("lambda_hard", 0.0)
# # #                 qh  = bd.get("q_hard_mean", 0.0)
# # #                 to_w = " [!>10]" if tot > 10.0 else ""
# # #                 lh_s = f"  lh={lh:.2f}" if lh > 0.0 else ""
# # #                 print(
# # #                     f"  [{ep:>3}][{i:>3}/{nstep}]"
# # #                     f"  tot={tot:.3f}{to_w}"
# # #                     f"  fm={bd.get('l_fm',0):.4f}"
# # #                     f"  pos={bd.get('l_pos',0):.4f}"
# # #                     f"  anc={bd.get('l_anchor',0):.4f}"
# # #                     f"  disp={bd.get('l_disp',0):.4f}"
# # #                     f"  head={bd.get('l_head',0):.4f}"
# # #                     f"  smo={bd.get('l_smooth',0):.4f}"
# # #                     f"  hard={bd.get('l_hard',0):.4f}"
# # #                     f"{lh_s}"
# # #                     f"  qh={qh:.2f}"
# # #                     f"  lr={lr:.2e}"
# # #                 )

# # #         avt = sl / nstep

# # #         # BUG-5 FIX: val với epoch=-1 → skip augmentation
# # #         model.eval(); vls = 0.
# # #         with torch.no_grad():
# # #             for batch in vl:
# # #                 bv = move(list(batch), dev)
# # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # #                     vls += model.get_loss(bv, epoch=-1).item()
# # #         avv    = vls / len(vl)
# # #         eps_t  = time.perf_counter() - t0
# # #         lr_cur = opt.param_groups[0]["lr"]
# # #         print(f"  Epoch {ep:>3} | train={avt:.4f} val={avv:.4f}"
# # #               f" | lr={lr_cur:.2e} | {eps_t:.0f}s")

# # #         # ── Fast eval (subset) — monitor only, không stop ────────────────────
# # #         rf = evaluate(model, vsub, dev, tag=f"FAST ep{ep}", steps=args.fast_ddim)
# # #         fast_saver.update(rf, model, args.output_dir, ep, opt, sched,
# # #                           avt, avv, tag="fast")
# # #         # fast_saver.stop luôn False, không cần check

# # #         # ── Full val + EMA — điều khiển early stop ───────────────────────────
# # #         if ep % args.val_freq == 0:
# # #             em = getattr(_unwrap(model), "_ema", None)

# # #             rr = evaluate(model, vl, dev, tag=f"RAW ep{ep}", steps=args.full_ddim)
# # #             full_saver.update(rr, model, args.output_dir, ep, opt, sched,
# # #                               avt, avv, tag="raw")

# # #             if em and ep >= 3:
# # #                 re = evaluate(model, vl, dev, tag=f"EMA ep{ep}",
# # #                                ema=em, steps=args.full_ddim)
# # #                 # EMA cũng check full_saver (cùng early stop counter)
# # #                 full_saver.update(re, model, args.output_dir, ep, opt, sched,
# # #                                   avt, avv, tag="ema")

# # #         # Periodic checkpoint
# # #         if ep % 10 == 0 or ep == args.num_epochs - 1:
# # #             _save_ckpt(os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# # #                         ep, model, opt, sched, full_saver, avt, avv)

# # #         # [T1] Early stop chỉ dựa vào full_saver
# # #         if full_saver.stop:
# # #             print(f"  Early stop triggered at ep={ep} (full val patience)")
# # #             break

# # #     th = (time.perf_counter() - ts) / 3600.
# # #     print(f"\n  Best (full val): ADE={full_saver.ba:.1f}  72h={full_saver.b7:.0f}"
# # #           f"  ATE={full_saver.bat:.1f}  CTE={full_saver.bc:.1f}"
# # #           f"  ({th:.2f}h)")

# # #     # ── Post-training test ────────────────────────────────────────────────────
# # #     if args.eval_test_after_train:
# # #         print("\n" + "="*72 + "\n  POST-TRAINING TEST\n" + "="*72)
# # #         try:
# # #             _, tl2 = data_loader(args, {"root": args.dataset_root,
# # #                                          "type": "test"}, test=True)
# # #         except Exception:
# # #             print("  No test set → using val"); tl2 = vl

# # #         from Model.data.trajectoriesWithMe_unet_training import seq_collate as _sc
# # #         print(f"  test seqs: {len(tl2.dataset) if hasattr(tl2,'dataset') else '?'}")

# # #         test_ckpts = [
# # #             ("best_raw.pth",  "RAW"),
# # #             ("best_ema.pth",  "EMA"),
# # #             ("best_ade_raw.pth", "BEST_ADE_RAW"),
# # #             ("best_cte_raw.pth", "BEST_CTE_RAW"),
# # #             ("best_ade_ema.pth", "BEST_ADE_EMA"),
# # #             ("best_cte_ema.pth", "BEST_CTE_EMA"),
# # #         ]

# # #         for fn, lb in test_ckpts:
# # #             pp = os.path.join(args.output_dir, fn)
# # #             if not os.path.exists(pp): continue
# # #             ck = torch.load(pp, map_location=dev)
# # #             _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
# # #             em = getattr(_unwrap(model), "_ema", None)
# # #             if em and ck.get("ema_shadow"):
# # #                 for k, v in ck["ema_shadow"].items():
# # #                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
# # #             r = evaluate(model, tl2, dev, tag=f"TEST/{lb}", steps=args.full_ddim)
# # #             print(f"\n  --- {lb} (ep={ck.get('epoch','?')}) ---")
# # #             for key, ref in [("ADE",172.68),("72h",321.39),
# # #                                ("ATE_mean",142.21),("CTE_mean",42.04)]:
# # #                 v  = r.get(key, float("nan"))
# # #                 mk = "BEAT!" if np.isfinite(v) and v < ref else f"need {ref:.0f}"
# # #                 gap = v - ref if np.isfinite(v) else float("nan")
# # #                 print(f"    {key:<12}: {v:>8.1f} km  [{mk}  gap:{gap:+.1f}]")

# # #     print("="*72)


# # # if __name__ == "__main__":
# # #     args = get_args()
# # #     np.random.seed(42); torch.manual_seed(42)
# # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# # #     main(args)

# # """
# # train_v83_fixed.py  —  TC-FlowMatching v83-Fixed Training Script
# # ══════════════════════════════════════════════════════════════════════════════
# # Thay đổi so với train_v83.py (cũ):

# #   [T1] Log keys đồng bộ với v83_fixed:
# #        - Cũ: fm, dpe, cte, ate, scte, vel → luôn 0.0000 (key mismatch)
# #        - Mới: fm, pos, anchor, disp, head, smooth, hard, qh, lh, anc_ade

# #   [T2] Early stop CHỈ tính full val (RAW + EMA), fast val chỉ monitor.
# #        fast_saver: enable_stop=False
# #        full_saver: enable_stop=True, patience=40, min_ep=35

# #   [T3] BUG-5 fix: val_loss dùng epoch=-1 → skip augmentation

# #   [T4] BUG-7 fix: sched.step() unconditional

# #   [T5] BUG-8 fix: clip chỉ backbone params (numel > 100), không clip learned weights

# #   [T6] BUG-6 fix: any_improve = ANY metric improve → reset early stop counter

# #   [T7] Full val freq = mỗi 3 epoch, oracle@20 logged mỗi 9 epoch

# #   [T8] Saver lưu best theo từng metric riêng (ADE, 72h, ATE, CTE, composite)

# # TARGETS: ADE<172.68 72h<321.39 ATE<142.21 CTE<42.04
# # """
# # from __future__ import annotations

# # import sys
# # import os

# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # import argparse
# # import random
# # import time
# # from collections import defaultdict

# # import numpy as np
# # import torch
# # import torch.optim as optim
# # from torch.amp import autocast, GradScaler
# # from torch.utils.data import DataLoader, Subset

# # from Model.data.loader_training import data_loader
# # from Model.flow_matching_model import TCFlowMatchingV83Fixed

# # try:
# #     from Model.utils import get_cosine_schedule_with_warmup
# # except ImportError:
# #     from torch.optim.lr_scheduler import CosineAnnealingLR

# #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# #         return CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=min_lr)

# # TARGETS = {
# #     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
# #     "12h": 65.42,  "24h": 104.67, "48h": 205.10,
# # }
# # R_EARTH = 6371.0


# # # ── Helpers ───────────────────────────────────────────────────────────────────

# # def _unwrap(m):
# #     return m._orig_mod if hasattr(m, "_orig_mod") else m


# # def move(b, dev):
# #     out = list(b)
# #     for i, x in enumerate(out):
# #         if torch.is_tensor(x):
# #             out[i] = x.to(dev)
# #         elif isinstance(x, dict):
# #             out[i] = {k: v.to(dev) if torch.is_tensor(v) else v
# #                       for k, v in x.items()}
# #     return out


# # # ── Metric utils ──────────────────────────────────────────────────────────────

# # def _ntd(a: torch.Tensor) -> torch.Tensor:
# #     o = a.clone()
# #     o[..., 0] = (a[..., 0] * 50.0 + 1800.0) / 10.0
# #     o[..., 1] = (a[..., 1] * 50.0) / 10.0
# #     return o


# # def _hav(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# #     la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
# #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# #     a = (torch.sin(dlat / 2).pow(2)
# #          + torch.cos(la1) * torch.cos(la2) * torch.sin(dlon / 2).pow(2))
# #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# # def _atecte(pd: torch.Tensor, gd: torch.Tensor):
# #     T = min(pd.shape[0], gd.shape[0])
# #     if T < 2:
# #         z = pd.new_zeros(1, pd.shape[1])
# #         return z, z
# #     lo1 = torch.deg2rad(gd[:T-1, :, 0]); la1 = torch.deg2rad(gd[:T-1, :, 1])
# #     lo2 = torch.deg2rad(gd[1:T,  :, 0]); la2 = torch.deg2rad(gd[1:T,  :, 1])
# #     lo3 = torch.deg2rad(pd[1:T,  :, 0]); la3 = torch.deg2rad(pd[1:T,  :, 1])
# #     ya = torch.sin(lo2 - lo1) * torch.cos(la2)
# #     xa = (torch.cos(la1) * torch.sin(la2)
# #           - torch.sin(la1) * torch.cos(la2) * torch.cos(lo2 - lo1))
# #     be  = torch.atan2(ya, xa)
# #     ye  = torch.sin(lo3 - lo2) * torch.cos(la3)
# #     xe  = (torch.cos(la2) * torch.sin(la3)
# #            - torch.sin(la2) * torch.cos(la3) * torch.cos(lo3 - lo2))
# #     bee = torch.atan2(ye, xe)
# #     tot = _hav(pd[1:T], gd[1:T]); ang = bee - be
# #     return tot * torch.cos(ang), tot * torch.sin(ang)


# # class Acc:
# #     def __init__(self):
# #         self.d = []; self.a = []; self.c = []
# #         self.sd = defaultdict(list)
# #         self._h = {12: 1, 24: 3, 48: 7, 72: 11}

# #     def update(self, dist, ate=None, cte=None):
# #         self.d.extend(dist.mean(0).tolist())
# #         for h, s in self._h.items():
# #             if s < dist.shape[0]:
# #                 self.sd[h].extend(dist[s].tolist())
# #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())

# #     def compute(self) -> dict:
# #         r = {
# #             "ADE":      float(np.mean(self.d))  if self.d else float("nan"),
# #             "ATE_mean": float(np.mean(self.a))  if self.a else float("nan"),
# #             "CTE_mean": float(np.mean(self.c))  if self.c else float("nan"),
# #             "n": len(self.d),
# #         }
# #         for h in self._h:
# #             v = self.sd.get(h, [])
# #             r[f"{h}h"] = float(np.mean(v)) if v else float("nan")
# #         return r


# # def _score(r: dict) -> float:
# #     ade = r.get("ADE",     1e9)
# #     h72 = r.get("72h",     1e9)
# #     ate = r.get("ATE_mean",1e9)
# #     cte = r.get("CTE_mean",1e9)
# #     if not np.isfinite(ate): ate = ade * 0.46
# #     if not np.isfinite(cte): cte = ade * 0.53
# #     return 100.0 * (
# #         0.05 * (ade / 136.0)
# #         + 0.10 * (r.get("12h", ade) / 50.0)
# #         + 0.15 * (r.get("24h", ade) / 100.0)
# #         + 0.20 * (r.get("48h", ade) / 200.0)
# #         + 0.25 * (h72 / 300.0)
# #         + 0.13 * (ate / 80.0)
# #         + 0.12 * (cte / 94.0)
# #     )


# # def _beat(r: dict) -> str:
# #     p = []
# #     for k, t in [("ADE", 172.68), ("ATE_mean", 142.21), ("CTE_mean", 42.04),
# #                    ("72h", 321.39), ("12h", 65.42), ("24h", 104.67), ("48h", 205.10)]:
# #         v = r.get(k, 1e9)
# #         if np.isfinite(v) and v < t:
# #             p.append(f"{k.replace('_mean','').replace('72','72h').replace('12','12h')}:{v:.1f}")
# #     return "*** BEAT ST-TRANS: " + " ".join(p) + " ***" if p else ""


# # def _gap(r: dict) -> str:
# #     out = []
# #     for k, ref in [("ADE", 172.68), ("72h", 321.39),
# #                     ("ATE_mean", 142.21), ("CTE_mean", 42.04)]:
# #         v = r.get(k, float("nan"))
# #         if np.isfinite(v):
# #             out.append(f"{k.replace('_mean','')}"
# #                         f":{v:.0f}({'dn' if v < ref else 'up'}{abs(v-ref):.0f})")
# #     return " | ".join(out)


# # # ── Evaluation ────────────────────────────────────────────────────────────────

# # @torch.no_grad()
# # def evaluate(model, loader, dev, tag: str = "", ema=None, steps: int = 20) -> dict:
# #     bk = None
# #     if ema:
# #         try:   bk = ema.apply_to(model)
# #         except: pass
# #     model.eval(); acc = Acc(); t0 = time.perf_counter()
# #     for b in loader:
# #         bl = move(list(b), dev)
# #         result = model.sample(bl, ddim_steps=steps)
# #         p = result[0] if isinstance(result, (tuple, list)) else result
# #         g = bl[1]; T = min(p.shape[0], g.shape[0])
# #         pd = _ntd(p[:T]); gd = _ntd(g[:T])
# #         dist = _hav(pd, gd)
# #         at, ct = _atecte(pd, gd)
# #         acc.update(dist, at, ct)
# #     if bk:
# #         try: ema.restore(model, bk)
# #         except: pass
# #     r = acc.compute()
# #     el = time.perf_counter() - t0

# #     def _v(k): return r.get(k, float("nan"))
# #     def _m(v, t): return "ok" if np.isfinite(v) and v < t else "no"

# #     print(f"\n{'='*70}")
# #     print(f"  [{tag}  {el:.0f}s]")
# #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]"
# #           f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
# #           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
# #     if np.isfinite(_v("ATE_mean")):
# #         print(f"  ATE={_v('ATE_mean'):.1f}[{_m(_v('ATE_mean'),142.21)}]"
# #               f"  CTE={_v('CTE_mean'):.1f}[{_m(_v('CTE_mean'),42.04)}]")
# #     print(f"  vs ST-Trans: {_gap(r)}")
# #     bt = _beat(r)
# #     if bt: print(f"  {bt}")
# #     print(f"  Score={_score(r):.2f}")
# #     print(f"{'='*70}\n")
# #     return r


# # # ── Checkpoint ────────────────────────────────────────────────────────────────

# # def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl, extra=None):
# #     m = _unwrap(model)
# #     ema = getattr(m, "_ema", None)
# #     esd = None
# #     if ema and hasattr(ema, "shadow"):
# #         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# #         except: pass
# #     d = {
# #         "epoch": ep,
# #         "model_state_dict": m.state_dict(),
# #         "optimizer_state":  opt.state_dict(),
# #         "scheduler_state":  sched.state_dict(),
# #         "ema_shadow":       esd,
# #         "best_score": saver.bs, "best_ade": saver.ba,
# #         "best_72h":   saver.b7, "best_ate": saver.bat,
# #         "best_cte":   saver.bc,
# #         "train_loss": tl, "val_loss": vl, "version": "v83fixed",
# #     }
# #     if extra: d.update(extra)
# #     torch.save(d, path)


# # # ── Saver ─────────────────────────────────────────────────────────────────────

# # class Saver:
# #     """
# #     [T2] Tách biệt fast_saver (no stop) và full_saver (controls early stop).
# #     [T6] BUG-6: any_improve = ANY metric improve → reset cnt.
# #     """
# #     def __init__(self, patience: int = 40, min_ep: int = 35,
# #                  enable_stop: bool = True):
# #         self.patience    = patience
# #         self.min_ep      = min_ep
# #         self.enable_stop = enable_stop
# #         self.cnt  = 0
# #         self.stop = False
# #         self.bs = self.ba = self.b7 = self.bat = self.bc = float("inf")

# #     def update(self, r: dict, model, out_dir: str, ep: int,
# #                opt, sched, tl: float, vl: float, tag: str = ""):
# #         sc  = _score(r)
# #         ade = r.get("ADE",      1e9)
# #         h72 = r.get("72h",      1e9)
# #         ate = r.get("ATE_mean", 1e9)
# #         cte = r.get("CTE_mean", 1e9)

# #         any_improved = False

# #         # Save best per-metric checkpoints (BUG-6: any_improve)
# #         for v, attr, fn in [
# #             (ade, "ba",  f"best_ade_{tag}.pth"  if tag else "best_ade.pth"),
# #             (h72, "b7",  f"best_72h_{tag}.pth"  if tag else "best_72h.pth"),
# #             (ate, "bat", f"best_ate_{tag}.pth"  if tag else "best_ate.pth"),
# #             (cte, "bc",  f"best_cte_{tag}.pth"  if tag else "best_cte.pth"),
# #         ]:
# #             if v < getattr(self, attr):
# #                 setattr(self, attr, v)
# #                 _save_ckpt(os.path.join(out_dir, fn),
# #                            ep, model, opt, sched, self, tl, vl)
# #                 any_improved = True

# #         # Save best composite
# #         if sc < self.bs:
# #             self.bs = sc
# #             any_improved = True
# #             fn = f"best_{tag or 'composite'}.pth"
# #             _save_ckpt(os.path.join(out_dir, fn),
# #                         ep, model, opt, sched, self, tl, vl,
# #                         {"score": sc, "ade": ade, "h72": h72})
# #             print(f"  [BEST] {tag} ep={ep} score={sc:.2f}"
# #                   f"  ADE={ade:.1f}  72h={h72:.0f}"
# #                   f"  ATE={ate:.1f}  CTE={cte:.1f}")

# #         # Early stop logic — chỉ khi enable_stop=True
# #         if self.enable_stop:
# #             if any_improved:
# #                 self.cnt = 0
# #             else:
# #                 self.cnt += 1
# #                 print(f"  No improve {self.cnt}/{self.patience}"
# #                       f"  best={self.bs:.2f}  cur={sc:.2f}")
# #             if ep >= self.min_ep and self.cnt >= self.patience:
# #                 self.stop = True
# #                 print(f"  [STOP] ep={ep} patience={self.patience}")


# # def _mksub(ds, n: int, bs: int, cf):
# #     idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
# #     return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
# #                        collate_fn=cf, num_workers=0, drop_last=False)


# # # ── Args ──────────────────────────────────────────────────────────────────────

# # def get_args():
# #     p = argparse.ArgumentParser(
# #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# #     # Dataset
# #     p.add_argument("--dataset_root",       default="TCND_vn")
# #     p.add_argument("--obs_len",            default=8,     type=int)
# #     p.add_argument("--pred_len",           default=12,    type=int)
# #     # Training
# #     p.add_argument("--batch_size",         default=32,    type=int)
# #     p.add_argument("--num_epochs",         default=120,   type=int)
# #     p.add_argument("--learning_rate",      default=1e-4,  type=float)
# #     p.add_argument("--weight_decay",       default=1e-3,  type=float)
# #     p.add_argument("--warmup_epochs",      default=3,     type=int)
# #     p.add_argument("--grad_clip",          default=1.0,   type=float)
# #     p.add_argument("--use_amp",            action="store_true")
# #     p.add_argument("--num_workers",        default=2,     type=int)
# #     # Model
# #     p.add_argument("--sigma_min",          default=0.02,  type=float)
# #     p.add_argument("--use_ot",             default=True,  action="store_true")
# #     p.add_argument("--no_ot",              dest="use_ot", action="store_false")
# #     p.add_argument("--cfg_guidance_scale", default=1.5,   type=float)
# #     p.add_argument("--n_ensemble",         default=50,    type=int)
# #     # EMA
# #     p.add_argument("--use_ema",            default=True,  action="store_true")
# #     p.add_argument("--no_ema",             dest="use_ema", action="store_false")
# #     p.add_argument("--ema_decay",          default=0.999, type=float)
# #     # Curriculum params
# #     p.add_argument("--hard_start_epoch",   default=20,    type=int)
# #     p.add_argument("--hard_ramp_epochs",   default=10,    type=int)
# #     p.add_argument("--diff_threshold",     default=0.4,   type=float)
# #     p.add_argument("--diff_temp",          default=0.15,  type=float)
# #     p.add_argument("--lw_freeze_epochs",   default=5,     type=int)
# #     # Early stop
# #     p.add_argument("--patience",           default=40,    type=int)
# #     p.add_argument("--min_ep",             default=35,    type=int)
# #     # Eval schedule
# #     p.add_argument("--val_freq",           default=3,     type=int,
# #                    help="Full val mỗi N epochs")
# #     p.add_argument("--oracle_freq",        default=9,     type=int,
# #                    help="Log oracle@20 mỗi N epochs (0=off)")
# #     p.add_argument("--val_subset_size",    default=500,   type=int)
# #     p.add_argument("--fast_ddim",          default=10,    type=int)
# #     p.add_argument("--full_ddim",          default=20,    type=int)
# #     # Output
# #     p.add_argument("--output_dir",         default="runs/v83fixed")
# #     p.add_argument("--gpu_num",            default="0")
# #     # Data args
# #     p.add_argument("--delim",              default=" ")
# #     p.add_argument("--skip",               default=1,     type=int)
# #     p.add_argument("--min_ped",            default=1,     type=int)
# #     p.add_argument("--threshold",          default=0.002, type=float)
# #     p.add_argument("--other_modal",        default="gph")
# #     p.add_argument("--test_year",          default=None,  type=int)
# #     # Resume
# #     p.add_argument("--resume",             default=None)
# #     p.add_argument("--resume_epoch",       default=None,  type=int)
# #     p.add_argument("--eval_test_after_train", default=True, action="store_true")
# #     return p.parse_args()


# # # ── Main ──────────────────────────────────────────────────────────────────────

# # def main(args):
# #     if torch.cuda.is_available():
# #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# #     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     os.makedirs(args.output_dir, exist_ok=True)

# #     print("=" * 72)
# #     print(f"  TC-FlowMatching v83-Fixed")
# #     print(f"  Loss: L_anchor(pred→anchor) + L_pos + L_fm + L_disp + L_head + L_smooth + L_hard")
# #     print(f"  Fixes: anchor(FIX-1) focal(FIX-2) context(FIX-3) OT(FIX-4)")
# #     print(f"         heading(FIX-5) stepW(FIX-6) whard(FIX-7)")
# #     print(f"  BUG fixes: BUG-5(val_noaug) BUG-6(any_improve) BUG-7(sched_uncond) BUG-8(clip_backbone)")
# #     print(f"  EMA decay={args.ema_decay}  n_ensemble={args.n_ensemble}")
# #     print(f"  Hard curriculum: start_ep={args.hard_start_epoch} ramp={args.hard_ramp_epochs}")
# #     print(f"  Early stop: patience={args.patience} min_ep={args.min_ep} [FULL VAL ONLY]")
# #     print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']} ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
# #     print("=" * 72)

# #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
# #     vd,  vl  = data_loader(args, {"root": args.dataset_root, "type": "val"},   test=True)
# #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# #     vsub = _mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
# #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# #     model = TCFlowMatchingV83Fixed(
# #         pred_len           = args.pred_len,
# #         obs_len            = args.obs_len,
# #         sigma_min          = args.sigma_min,
# #         use_ema            = args.use_ema,
# #         ema_decay          = args.ema_decay,
# #         use_ot             = args.use_ot,
# #         cfg_guidance_scale = args.cfg_guidance_scale,
# #         hard_start_epoch   = args.hard_start_epoch,
# #         hard_ramp_epochs   = args.hard_ramp_epochs,
# #         diff_threshold     = args.diff_threshold,
# #         diff_temp          = args.diff_temp,
# #         lw_freeze_epochs   = args.lw_freeze_epochs,
# #     ).to(dev)
# #     model.init_ema()

# #     n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #     print(f"  params total: {n_total:,}")

# #     # BUG-8: tách backbone vs learned weight params
# #     backbone_params = [p for p in model.parameters()
# #                        if p.requires_grad and p.numel() > 100]
# #     small_params    = [p for p in model.parameters()
# #                        if p.requires_grad and p.numel() <= 100]
# #     print(f"  backbone (clipped at {args.grad_clip}): {sum(p.numel() for p in backbone_params):,}")
# #     print(f"  small/learned_w (NOT clipped): {sum(p.numel() for p in small_params):,}")

# #     opt    = optim.AdamW(model.parameters(), lr=args.learning_rate,
# #                           weight_decay=args.weight_decay)
# #     scaler = GradScaler("cuda", enabled=args.use_amp)

# #     nstep  = len(trl)
# #     total  = nstep * args.num_epochs
# #     wstp   = nstep * args.warmup_epochs
# #     sched  = get_cosine_schedule_with_warmup(opt, wstp, total, min_lr=1e-6)

# #     # [T2] Tách fast_saver (no stop) và full_saver (controls early stop)
# #     fast_saver = Saver(patience=args.patience, min_ep=args.min_ep, enable_stop=False)
# #     full_saver = Saver(patience=args.patience, min_ep=args.min_ep, enable_stop=True)

# #     # ── Resume ────────────────────────────────────────────────────────────────
# #     start = 0
# #     if args.resume and os.path.exists(args.resume):
# #         print(f"  Loading: {args.resume}")
# #         ck = torch.load(args.resume, map_location=dev)
# #         m  = _unwrap(model)
# #         ms, _ = m.load_state_dict(ck["model_state_dict"], strict=False)
# #         if ms: print(f"  Missing keys: {len(ms)}")
# #         ema = getattr(m, "_ema", None)
# #         if ema and ck.get("ema_shadow"):
# #             for k, v in ck["ema_shadow"].items():
# #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
# #         try:   opt.load_state_dict(ck["optimizer_state"])
# #         except Exception as e: print(f"  Opt not loaded: {e}")
# #         try:   sched.load_state_dict(ck["scheduler_state"])
# #         except Exception:
# #             for _ in range(ck.get("epoch", 0) * nstep): sched.step()
# #         for a, attr in [("best_score","bs"), ("best_ade","ba"), ("best_72h","b7"),
# #                           ("best_ate","bat"), ("best_cte","bc")]:
# #             if a in ck:
# #                 setattr(full_saver, attr, ck[a])
# #                 setattr(fast_saver, attr, ck[a])
# #         start = args.resume_epoch or ck.get("epoch", 0) + 1
# #         print(f"  Resume ep={start}  best_score={full_saver.bs:.2f}")

# #     try:
# #         model = torch.compile(model, mode="reduce-overhead")
# #         print("  torch.compile: ok")
# #     except Exception:
# #         pass

# #     ts = time.perf_counter()
# #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# #     print("=" * 72)

# #     for ep in range(start, args.num_epochs):
# #         model.train()
# #         sl = 0.0
# #         t0 = time.perf_counter()

# #         for i, batch in enumerate(trl):
# #             bl = move(list(batch), dev)

# #             with autocast(device_type="cuda", enabled=args.use_amp):
# #                 bd = model.get_loss_breakdown(bl, epoch=ep)

# #             opt.zero_grad()
# #             scaler.scale(bd["total"]).backward()
# #             scaler.unscale_(opt)

# #             # BUG-8: clip chỉ backbone
# #             torch.nn.utils.clip_grad_norm_(backbone_params, args.grad_clip)

# #             scaler.step(opt)
# #             scaler.update()

# #             # BUG-7: sched.step() unconditional
# #             try:
# #                 sched.step()
# #             except Exception:
# #                 pass

# #             model.ema_update()
# #             sl += bd["total"].item()

# #             if i % 20 == 0:
# #                 lr   = opt.param_groups[0]["lr"]
# #                 tot  = bd["total"].item()
# #                 lh   = bd.get("lambda_hard", 0.0)
# #                 qh   = bd.get("q_hard_mean", 0.0)
# #                 anc  = bd.get("anchor_ade", 0.0)

# #                 # [T1] Log keys đồng bộ với v83_fixed output
# #                 warn = " [!>10]" if tot > 10.0 else ""
# #                 lh_s = f"  lh={lh:.2f}" if lh > 0.0 else ""
# #                 print(
# #                     f"  [{ep:>3}][{i:>3}/{nstep}]"
# #                     f"  tot={tot:.3f}{warn}"
# #                     f"  fm={bd.get('l_fm',0):.4f}"
# #                     f"  pos={bd.get('l_pos',0):.4f}"
# #                     f"  anc={bd.get('l_anchor',0):.4f}"
# #                     f"  disp={bd.get('l_disp',0):.4f}"
# #                     f"  head={bd.get('l_head',0):.4f}"
# #                     f"  smo={bd.get('l_smooth',0):.4f}"
# #                     f"  hard={bd.get('l_hard',0):.4f}"
# #                     f"{lh_s}"
# #                     f"  qh={qh:.2f}"
# #                     f"  anc_ade={anc:.0f}"
# #                     f"  lr={lr:.2e}"
# #                 )

# #         avt    = sl / nstep
# #         t_ep   = time.perf_counter() - t0

# #         # BUG-5: val với epoch=-1 → skip augmentation
# #         model.eval()
# #         vls = 0.0
# #         with torch.no_grad():
# #             for batch in vl:
# #                 bv = move(list(batch), dev)
# #                 with autocast(device_type="cuda", enabled=args.use_amp):
# #                     vls += model.get_loss(bv, epoch=-1).item()
# #         avv    = vls / len(vl)
# #         lr_cur = opt.param_groups[0]["lr"]
# #         print(f"  Epoch {ep:>3} | train={avt:.4f} val={avv:.4f}"
# #               f" | lr={lr_cur:.2e} | {t_ep:.0f}s")

# #         # ── Fast eval (subset) — monitor only, không điều khiển early stop ──
# #         rf = evaluate(model, vsub, dev, tag=f"FAST ep{ep}", steps=args.fast_ddim)
# #         fast_saver.update(rf, model, args.output_dir, ep, opt, sched, avt, avv, tag="fast")
# #         # fast_saver.stop luôn False

# #         # ── Oracle@20 logging (optional, mỗi oracle_freq epochs) ─────────────
# #         if args.oracle_freq > 0 and ep % args.oracle_freq == 0 and ep >= 2:
# #             try:
# #                 _oracle_log(model, vsub, dev, ep)
# #             except Exception as e:
# #                 print(f"  [oracle] skip: {e}")

# #         # ── Full val + EMA — điều khiển early stop ────────────────────────────
# #         if ep % args.val_freq == 0:
# #             em = getattr(_unwrap(model), "_ema", None)

# #             rr = evaluate(model, vl, dev, tag=f"RAW ep{ep}", steps=args.full_ddim)
# #             full_saver.update(rr, model, args.output_dir, ep, opt, sched, avt, avv, tag="raw")

# #             if em and ep >= 3:
# #                 re = evaluate(model, vl, dev, tag=f"EMA ep{ep}",
# #                                ema=em, steps=args.full_ddim)
# #                 full_saver.update(re, model, args.output_dir, ep, opt, sched, avt, avv, tag="ema")

# #         # Periodic checkpoint
# #         if ep % 10 == 0 or ep == args.num_epochs - 1:
# #             _save_ckpt(os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# #                         ep, model, opt, sched, full_saver, avt, avv)

# #         # [T2] Early stop chỉ từ full_saver
# #         if full_saver.stop:
# #             print(f"  Early stop triggered at ep={ep} (full val patience)")
# #             break

# #     th = (time.perf_counter() - ts) / 3600.0
# #     print(f"\n  Best full val: ADE={full_saver.ba:.1f}  72h={full_saver.b7:.0f}"
# #           f"  ATE={full_saver.bat:.1f}  CTE={full_saver.bc:.1f}  ({th:.2f}h)")

# #     # ── Post-training test ────────────────────────────────────────────────────
# #     if args.eval_test_after_train:
# #         print("\n" + "=" * 72 + "\n  POST-TRAINING TEST\n" + "=" * 72)
# #         try:
# #             _, tl2 = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
# #         except Exception:
# #             print("  No test set → using val"); tl2 = vl

# #         print(f"  test seqs: {len(tl2.dataset) if hasattr(tl2,'dataset') else '?'}")

# #         test_ckpts = [
# #             ("best_raw.pth",     "RAW"),
# #             ("best_ema.pth",     "EMA"),
# #             ("best_ade_raw.pth", "BEST_ADE_RAW"),
# #             ("best_cte_raw.pth", "BEST_CTE_RAW"),
# #             ("best_ade_ema.pth", "BEST_ADE_EMA"),
# #             ("best_cte_ema.pth", "BEST_CTE_EMA"),
# #         ]
# #         for fn, lb in test_ckpts:
# #             pp = os.path.join(args.output_dir, fn)
# #             if not os.path.exists(pp): continue
# #             ck = torch.load(pp, map_location=dev)
# #             _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
# #             em = getattr(_unwrap(model), "_ema", None)
# #             if em and ck.get("ema_shadow"):
# #                 for k, v in ck["ema_shadow"].items():
# #                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
# #             r = evaluate(model, tl2, dev, tag=f"TEST/{lb}", steps=args.full_ddim)
# #             print(f"\n  --- {lb} (ep={ck.get('epoch','?')}) ---")
# #             for key, ref in [("ADE", 172.68), ("72h", 321.39),
# #                                ("ATE_mean", 142.21), ("CTE_mean", 42.04)]:
# #                 v   = r.get(key, float("nan"))
# #                 mk  = "BEAT!" if np.isfinite(v) and v < ref else f"need {ref:.0f}"
# #                 gap = v - ref if np.isfinite(v) else float("nan")
# #                 print(f"    {key:<12}: {v:>8.1f} km  [{mk}  gap:{gap:+.1f}]")

# #     print("=" * 72)


# # # ── Oracle logging helper ─────────────────────────────────────────────────────

# # @torch.no_grad()
# # def _oracle_log(model, loader, dev, ep: int, K: int = 20):
# #     """
# #     Log oracle@K: min ADE nếu chọn candidate tốt nhất.
# #     Nếu oracle cao → generator cần cải thiện, không phải selector.
# #     """
# #     m = _unwrap(model)
# #     m.eval()
# #     oracle_vals = []
# #     count = 0
# #     for b in loader:
# #         if count >= 5:  # Chỉ sample 5 batch để nhanh
# #             break
# #         bl = move(list(b), dev)
# #         try:
# #             _, all_t = m.sample(bl, num_ensemble=K, ddim_steps=10, return_all=True)
# #             gt = bl[1]
# #             gd = _ntd(gt)
# #             T_gt = gd.shape[0]
# #             ades = []
# #             for traj in all_t:
# #                 td = _ntd(traj)
# #                 T  = min(td.shape[0], T_gt)
# #                 ades.append(_hav(td[:T], gd[:T]).mean(0))
# #             if ades:
# #                 ades_t = torch.stack(ades, 0)  # (K, B)
# #                 oracle_vals.append(ades_t.min(0).values.mean().item())
# #         except Exception:
# #             pass
# #         count += 1
# #     if oracle_vals:
# #         print(f"  [oracle@{K} ep{ep}] mean={np.mean(oracle_vals):.1f} km"
# #               f"  (if >175: generator needs work)")


# # # ─────────────────────────────────────────────────────────────────────────────

# # if __name__ == "__main__":
# #     args = get_args()
# #     np.random.seed(42)
# #     torch.manual_seed(42)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(42)
# #     main(args)

# """
# train_v59_tweaked.py  —  TC-FlowMatching v59-Tweaked Training Script
# ══════════════════════════════════════════════════════════════════════════════
# Adapted từ train_v83_fixed.py cho model v59-tweaked.

# Model thay đổi:
#   [TWEAK-1] Bỏ ATE/CTE/endpoint losses → clean 3-component: dpe+vel_reg+heading
#   [TWEAK-2] LearnedStepWeights thay STEP_WEIGHTS fixed array
#   [TWEAK-3] K=3 mode clustering với multi-checkpoint (24h+48h+72h)

# Bug fixes giữ nguyên:
#   BUG-5: val_loss dùng epoch=-1 → skip augmentation
#   BUG-6: any_improve = ANY metric improve → reset early stop
#   BUG-7: sched.step() unconditional
#   BUG-8: clip chỉ backbone params, không clip learned weights (step_weights)

# Log format: fm, pos(=dpe), head, smooth(=accel), disp(=vel_reg), sw72, swr
# """
# from __future__ import annotations
# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse, random, time
# from collections import defaultdict

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.amp import autocast, GradScaler
# from torch.utils.data import DataLoader, Subset

# from Model.data.loader_training import data_loader
# from Model.flow_matching_model import TCFlowMatching   # v59-tweaked class

# try:
#     from Model.utils import get_cosine_schedule_with_warmup
# except ImportError:
#     from torch.optim.lr_scheduler import CosineAnnealingLR
#     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
#         return CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=min_lr)

# TARGETS = {
#     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
#     "12h": 65.42,  "24h": 104.67, "48h": 205.10,
# }
# R_EARTH = 6371.0


# def _unwrap(m):
#     return m._orig_mod if hasattr(m, "_orig_mod") else m

# def move(b, dev):
#     out = list(b)
#     for i, x in enumerate(out):
#         if torch.is_tensor(x): out[i] = x.to(dev)
#         elif isinstance(x, dict):
#             out[i] = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in x.items()}
#     return out


# # ── Metrics ───────────────────────────────────────────────────────────────────

# def _ntd(a):
#     o = a.clone()
#     o[..., 0] = (a[..., 0] * 50.0 + 1800.0) / 10.0
#     o[..., 1] = (a[..., 1] * 50.0) / 10.0
#     return o

# def _hav(p1, p2):
#     la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
#     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
#     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
#     a = (torch.sin(dlat/2).pow(2)
#          + torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2))
#     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

# def _atecte(pd, gd):
#     T = min(pd.shape[0], gd.shape[0])
#     if T < 2:
#         z = pd.new_zeros(1, pd.shape[1]); return z, z
#     lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
#     lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
#     lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
#     ya=torch.sin(lo2-lo1)*torch.cos(la2)
#     xa=(torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1))
#     be=torch.atan2(ya,xa)
#     ye=torch.sin(lo3-lo2)*torch.cos(la3)
#     xe=(torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2))
#     bee=torch.atan2(ye,xe)
#     tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
#     return tot*torch.cos(ang), tot*torch.sin(ang)

# class Acc:
#     def __init__(self):
#         self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list)
#         self._h={12:1, 24:3, 48:7, 72:11}
#     def update(self, dist, ate=None, cte=None):
#         self.d.extend(dist.mean(0).tolist())
#         for h,s in self._h.items():
#             if s < dist.shape[0]: self.sd[h].extend(dist[s].tolist())
#         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
#         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())
#     def compute(self):
#         r = {"ADE": float(np.mean(self.d)) if self.d else float("nan"),
#              "ATE": float(np.mean(self.a)) if self.a else float("nan"),
#              "CTE": float(np.mean(self.c)) if self.c else float("nan"),
#              "n": len(self.d)}
#         for h in self._h:
#             v=self.sd.get(h,[]); r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
#         return r

# def _score(r):
#     ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
#     ate=r.get("ATE",1e9); cte=r.get("CTE",1e9)
#     if not np.isfinite(ate): ate=ade*0.46
#     if not np.isfinite(cte): cte=ade*0.53
#     return 100.0*(0.05*(ade/136.)+0.10*(r.get("12h",ade)/50.)
#                   +0.15*(r.get("24h",ade)/100.)+0.20*(r.get("48h",ade)/200.)
#                   +0.25*(h72/300.)+0.13*(ate/80.)+0.12*(cte/94.))

# def _beat(r):
#     p=[]
#     for k,t in [("ADE",172.68),("ATE",142.21),("CTE",42.04),
#                 ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
#         v=r.get(k,1e9)
#         if np.isfinite(v) and v < t: p.append(f"{k}:{v:.1f}")
#     return "*** BEAT ST-TRANS: "+" ".join(p)+" ***" if p else ""

# def _gap(r):
#     out=[]
#     for k,ref in [("ADE",172.68),("72h",321.39),("ATE",142.21),("CTE",42.04)]:
#         v=r.get(k,float("nan"))
#         if np.isfinite(v):
#             out.append(f"{k}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
#     return " | ".join(out)


# # ── Evaluation ────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def evaluate(model, loader, dev, tag="", ema=None, steps=20):
#     bk=None
#     if ema:
#         try: bk=ema.apply_to(model)
#         except: pass
#     model.eval(); acc=Acc(); t0=time.perf_counter()
#     for b in loader:
#         bl=move(list(b),dev)
#         result=model.sample(bl, ddim_steps=steps)
#         p=result[0] if isinstance(result,(tuple,list)) else result
#         g=bl[1]; T=min(p.shape[0],g.shape[0])
#         pd=_ntd(p[:T]); gd=_ntd(g[:T])
#         dist=_hav(pd,gd); at,ct=_atecte(pd,gd)
#         acc.update(dist,at,ct)
#     if bk:
#         try: ema.restore(model,bk)
#         except: pass
#     r=acc.compute(); el=time.perf_counter()-t0
#     def _v(k): return r.get(k,float("nan"))
#     def _m(v,t): return "ok" if np.isfinite(v) and v<t else "no"
#     print(f"\n{'='*70}")
#     print(f"  [{tag}  {el:.0f}s]")
#     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]"
#           f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
#           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
#     print(f"  ATE={_v('ATE'):.1f}[{_m(_v('ATE'),142.21)}]"
#           f"  CTE={_v('CTE'):.1f}[{_m(_v('CTE'),42.04)}]")
#     print(f"  vs ST-Trans: {_gap(r)}")
#     bt=_beat(r)
#     if bt: print(f"  {bt}")
#     print(f"  Score={_score(r):.2f}")
#     print(f"{'='*70}\n")
#     return r


# # ── Checkpoint ────────────────────────────────────────────────────────────────

# def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl):
#     m=_unwrap(model); ema=getattr(m,"_ema",None); esd=None
#     if ema and hasattr(ema,"shadow"):
#         try: esd={k:v.cpu().clone() for k,v in ema.shadow.items()}
#         except: pass
#     torch.save({"epoch":ep,"model_state_dict":m.state_dict(),
#                 "optimizer_state":opt.state_dict(),"scheduler_state":sched.state_dict(),
#                 "ema_shadow":esd,"best_score":saver.bs,"best_ade":saver.ba,
#                 "best_72h":saver.b7,"best_ate":saver.bat,"best_cte":saver.bc,
#                 "train_loss":tl,"val_loss":vl,"version":"v59tweaked"}, path)


# # ── Saver ─────────────────────────────────────────────────────────────────────

# class Saver:
#     def __init__(self, patience=40, min_ep=35, enable_stop=True):
#         self.patience=patience; self.min_ep=min_ep; self.enable_stop=enable_stop
#         self.cnt=0; self.stop=False
#         self.bs=self.ba=self.b7=self.bat=self.bc=float("inf")

#     def update(self, r, model, out_dir, ep, opt, sched, tl, vl, tag=""):
#         sc=_score(r); ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
#         ate=r.get("ATE",1e9); cte=r.get("CTE",1e9)
#         any_improved=False
#         for v,attr,fn in [(ade,"ba",f"best_ade_{tag}.pth"),
#                           (h72,"b7",f"best_72h_{tag}.pth"),
#                           (ate,"bat",f"best_ate_{tag}.pth"),
#                           (cte,"bc",f"best_cte_{tag}.pth")]:
#             if v < getattr(self,attr):
#                 setattr(self,attr,v)
#                 _save_ckpt(os.path.join(out_dir,fn),ep,model,opt,sched,self,tl,vl)
#                 any_improved=True
#         if sc < self.bs:
#             self.bs=sc; any_improved=True
#             _save_ckpt(os.path.join(out_dir,f"best_{tag or 'composite'}.pth"),
#                        ep,model,opt,sched,self,tl,vl)
#             print(f"  [BEST] {tag} ep={ep} score={sc:.2f}"
#                   f"  ADE={ade:.1f}  72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}")
#         if self.enable_stop:
#             if any_improved: self.cnt=0
#             else:
#                 self.cnt+=1
#                 print(f"  No improve {self.cnt}/{self.patience}"
#                       f"  best={self.bs:.2f}  cur={sc:.2f}")
#             if ep>=self.min_ep and self.cnt>=self.patience:
#                 self.stop=True


# def _mksub(ds, n, bs, cf):
#     idx=random.Random(42).sample(range(len(ds)), min(n,len(ds)))
#     return DataLoader(Subset(ds,idx),batch_size=bs,shuffle=False,
#                       collate_fn=cf,num_workers=0,drop_last=False)


# # ── Args ──────────────────────────────────────────────────────────────────────

# def get_args():
#     p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     p.add_argument("--dataset_root",       default="TCND_vn")
#     p.add_argument("--obs_len",            default=8,    type=int)
#     p.add_argument("--pred_len",           default=12,   type=int)
#     p.add_argument("--batch_size",         default=32,   type=int)
#     p.add_argument("--num_epochs",         default=120,  type=int)
#     p.add_argument("--learning_rate",      default=1e-4, type=float)
#     p.add_argument("--weight_decay",       default=1e-3, type=float)
#     p.add_argument("--warmup_epochs",      default=3,    type=int)
#     p.add_argument("--grad_clip",          default=1.0,  type=float)
#     p.add_argument("--use_amp",            action="store_true")
#     p.add_argument("--num_workers",        default=2,    type=int)
#     p.add_argument("--sigma_min",          default=0.02, type=float)
#     p.add_argument("--use_ot",             default=True, action="store_true")
#     p.add_argument("--no_ot",              dest="use_ot", action="store_false")
#     p.add_argument("--cfg_guidance_scale", default=1.5,  type=float)
#     p.add_argument("--n_ensemble",         default=50,   type=int)
#     p.add_argument("--use_ema",            default=True, action="store_true")
#     p.add_argument("--no_ema",             dest="use_ema", action="store_false")
#     p.add_argument("--ema_decay",          default=0.995, type=float)
#     p.add_argument("--patience",           default=40,   type=int)
#     p.add_argument("--min_ep",             default=35,   type=int)
#     p.add_argument("--val_freq",           default=3,    type=int)
#     p.add_argument("--val_subset_size",    default=500,  type=int)
#     p.add_argument("--fast_ddim",          default=10,   type=int)
#     p.add_argument("--full_ddim",          default=20,   type=int)
#     p.add_argument("--output_dir",         default="runs/v59tweaked")
#     p.add_argument("--gpu_num",            default="0")
#     p.add_argument("--delim",              default=" ")
#     p.add_argument("--skip",               default=1,    type=int)
#     p.add_argument("--min_ped",            default=1,    type=int)
#     p.add_argument("--threshold",          default=0.002,type=float)
#     p.add_argument("--other_modal",        default="gph")
#     p.add_argument("--test_year",          default=None, type=int)
#     p.add_argument("--resume",             default=None)
#     p.add_argument("--resume_epoch",       default=None, type=int)
#     p.add_argument("--eval_test_after_train", default=True, action="store_true")
#     return p.parse_args()


# # ── Main ──────────────────────────────────────────────────────────────────────

# def main(args):
#     if torch.cuda.is_available():
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
#     dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output_dir, exist_ok=True)

#     print("=" * 72)
#     print("  TC-FlowMatching v59-Tweaked")
#     print("  Loss: L_fm + dpe(1.2x) + vel_reg(1.4x) + heading(0.4x) + sw_penalty")
#     print("  TWEAK-1: bỏ ATE/CTE/endpoint (không tổng quát)")
#     print("  TWEAK-2: LearnedStepWeights — tự học từ data, init ratio=12x")
#     print("  TWEAK-3: K=3 multi-checkpoint clustering (24h+48h+72h)")
#     print("  BUG fixes: BUG-5(val_noaug) BUG-6(any_improve) BUG-7(sched_uncond) BUG-8(clip_backbone)")
#     print(f"  EMA decay={args.ema_decay}  n_ensemble={args.n_ensemble}")
#     print(f"  patience={args.patience}  min_ep={args.min_ep}")
#     print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']} ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
#     print("=" * 72)

#     trd,trl=data_loader(args,{"root":args.dataset_root,"type":"train"},test=False)
#     vd, vl =data_loader(args,{"root":args.dataset_root,"type":"val"},  test=True)
#     from Model.data.trajectoriesWithMe_unet_training import seq_collate
#     vsub=_mksub(vd,args.val_subset_size,args.batch_size,seq_collate)
#     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

#     model=TCFlowMatching(
#         pred_len=args.pred_len, obs_len=args.obs_len,
#         sigma_min=args.sigma_min, use_ema=args.use_ema,
#         ema_decay=args.ema_decay, use_ate_ot=args.use_ot,
#         cfg_guidance_scale=args.cfg_guidance_scale,
#         n_ensemble=args.n_ensemble,
#     ).to(dev)
#     model.init_ema()

#     n_total=sum(p.numel() for p in model.parameters() if p.requires_grad)
#     # BUG-8: backbone (numel>100) vs learned weights (step_weights, numel<=12)
#     backbone_params=[p for p in model.parameters() if p.requires_grad and p.numel()>100]
#     small_params   =[p for p in model.parameters() if p.requires_grad and p.numel()<=100]
#     print(f"  params total: {n_total:,}")
#     print(f"  backbone (clipped at {args.grad_clip}): {sum(p.numel() for p in backbone_params):,}")
#     print(f"  learned_w step_weights (NOT clipped): {sum(p.numel() for p in small_params):,}")

#     opt=optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
#     scaler=GradScaler("cuda", enabled=args.use_amp)
#     nstep=len(trl); total=nstep*args.num_epochs; wstp=nstep*args.warmup_epochs
#     sched=get_cosine_schedule_with_warmup(opt,wstp,total,min_lr=1e-6)

#     fast_saver=Saver(patience=args.patience,min_ep=args.min_ep,enable_stop=False)
#     full_saver=Saver(patience=args.patience,min_ep=args.min_ep,enable_stop=True)

#     start=0
#     if args.resume and os.path.exists(args.resume):
#         print(f"  Loading: {args.resume}")
#         ck=torch.load(args.resume,map_location=dev)
#         m=_unwrap(model)
#         ms,_=m.load_state_dict(ck["model_state_dict"],strict=False)
#         if ms: print(f"  Missing: {len(ms)}")
#         ema=getattr(m,"_ema",None)
#         if ema and ck.get("ema_shadow"):
#             for k,v in ck["ema_shadow"].items():
#                 if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
#         try: opt.load_state_dict(ck["optimizer_state"])
#         except Exception as e: print(f"  Opt: {e}")
#         try: sched.load_state_dict(ck["scheduler_state"])
#         except: 
#             for _ in range(ck.get("epoch",0)*nstep): sched.step()
#         for a,attr in [("best_score","bs"),("best_ade","ba"),("best_72h","b7"),
#                        ("best_ate","bat"),("best_cte","bc")]:
#             if a in ck:
#                 setattr(full_saver,attr,ck[a]); setattr(fast_saver,attr,ck[a])
#         start=args.resume_epoch or ck.get("epoch",0)+1
#         print(f"  Resume ep={start}  best={full_saver.bs:.2f}")

#     try:
#         model=torch.compile(model,mode="reduce-overhead"); print("  torch.compile: ok")
#     except: pass

#     print(f"  Training: {nstep} steps/ep, start ep={start}")
#     print("=" * 72)
#     ts=time.perf_counter()

#     for ep in range(start, args.num_epochs):
#         model.train(); sl=0.0; t0=time.perf_counter()

#         for i, batch in enumerate(trl):
#             bl=move(list(batch),dev)
#             with autocast(device_type="cuda", enabled=args.use_amp):
#                 bd=model.get_loss_breakdown(bl, epoch=ep)

#             opt.zero_grad()
#             scaler.scale(bd["total"]).backward()
#             scaler.unscale_(opt)
#             # BUG-8: clip only backbone
#             torch.nn.utils.clip_grad_norm_(backbone_params, args.grad_clip)
#             scaler.step(opt); scaler.update()
#             # BUG-7: unconditional
#             try: sched.step()
#             except: pass
#             model.ema_update()
#             sl+=bd["total"].item()

#             if i % 20 == 0:
#                 lr=opt.param_groups[0]["lr"]
#                 tot=bd["total"].item()
#                 warn=" [!>10]" if tot>10.0 else ""
#                 swr=bd.get("sw_ratio",0.0); sw72=bd.get("sw_72h",0.0)
#                 print(
#                     f"  [{ep:>3}][{i:>3}/{nstep}]"
#                     f"  tot={tot:.3f}{warn}"
#                     f"  fm={bd.get('l_fm',0):.4f}"
#                     f"  pos={bd.get('l_pos',0):.4f}"
#                     f"  head={bd.get('l_head',0):.4f}"
#                     f"  disp={bd.get('l_disp',0):.4f}"
#                     f"  smo={bd.get('l_smooth',0):.4f}"
#                     f"  sw72={sw72:.2f}  swr={swr:.1f}"
#                     f"  lr={lr:.2e}"
#                 )

#         avt=sl/nstep; t_ep=time.perf_counter()-t0

#         # BUG-5: val với epoch=-1 → skip aug
#         model.eval(); vls=0.0
#         with torch.no_grad():
#             for batch in vl:
#                 bv=move(list(batch),dev)
#                 with autocast(device_type="cuda", enabled=args.use_amp):
#                     vls+=model.get_loss(bv, epoch=-1).item()
#         avv=vls/len(vl)
#         lr_cur=opt.param_groups[0]["lr"]
#         print(f"  Epoch {ep:>3} | train={avt:.4f} val={avv:.4f}"
#               f" | lr={lr_cur:.2e} | {t_ep:.0f}s")

#         # Fast eval — monitor only, no early stop
#         rf=evaluate(model, vsub, dev, tag=f"FAST ep{ep}", steps=args.fast_ddim)
#         fast_saver.update(rf, model, args.output_dir, ep, opt, sched, avt, avv, tag="fast")

#         # Full val every val_freq epochs — controls early stop
#         if ep % args.val_freq == 0:
#             em=getattr(_unwrap(model),"_ema",None)
#             rr=evaluate(model, vl, dev, tag=f"RAW ep{ep}", steps=args.full_ddim)
#             full_saver.update(rr, model, args.output_dir, ep, opt, sched, avt, avv, tag="raw")
#             if em and ep >= 3:
#                 re=evaluate(model, vl, dev, tag=f"EMA ep{ep}", ema=em, steps=args.full_ddim)
#                 full_saver.update(re, model, args.output_dir, ep, opt, sched, avt, avv, tag="ema")

#         if ep%10==0 or ep==args.num_epochs-1:
#             _save_ckpt(os.path.join(args.output_dir,f"ckpt_ep{ep:03d}.pth"),
#                        ep, model, opt, sched, full_saver, avt, avv)

#         if full_saver.stop:
#             print(f"  Early stop ep={ep}"); break

#     th=(time.perf_counter()-ts)/3600.0
#     print(f"\n  Best: ADE={full_saver.ba:.1f}  72h={full_saver.b7:.0f}"
#           f"  ATE={full_saver.bat:.1f}  CTE={full_saver.bc:.1f}  ({th:.2f}h)")

#     # Post-training test
#     if args.eval_test_after_train:
#         print("\n"+"="*72+"\n  POST-TRAINING TEST\n"+"="*72)
#         try: _,tl2=data_loader(args,{"root":args.dataset_root,"type":"test"},test=True)
#         except: print("  No test → using val"); tl2=vl
#         for fn,lb in [("best_raw.pth","RAW"),("best_ema.pth","EMA"),
#                       ("best_ade_raw.pth","BEST_ADE"),("best_cte_raw.pth","BEST_CTE")]:
#             pp=os.path.join(args.output_dir,fn)
#             if not os.path.exists(pp): continue
#             ck=torch.load(pp,map_location=dev)
#             _unwrap(model).load_state_dict(ck["model_state_dict"],strict=False)
#             em=getattr(_unwrap(model),"_ema",None)
#             if em and ck.get("ema_shadow"):
#                 for k,v in ck["ema_shadow"].items():
#                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
#             r=evaluate(model,tl2,dev,tag=f"TEST/{lb}",steps=args.full_ddim)
#             print(f"\n  --- {lb} (ep={ck.get('epoch','?')}) ---")
#             for key,ref in [("ADE",172.68),("72h",321.39),("ATE",142.21),("CTE",42.04)]:
#                 v=r.get(key,float("nan"))
#                 mk="BEAT!" if np.isfinite(v) and v<ref else f"need {ref:.0f}"
#                 print(f"    {key:<8}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
#     print("="*72)


# if __name__ == "__main__":
#     args=get_args()
#     np.random.seed(42); torch.manual_seed(42)
#     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
#     main(args)

"""
train_v59_tweaked.py  —  TC-FlowMatching v59-Tweaked Training Script
══════════════════════════════════════════════════════════════════════════════
Adapted từ train_v83_fixed.py cho model v59-tweaked.

Model thay đổi:
  [TWEAK-1] Bỏ ATE/CTE/endpoint losses → clean 3-component: dpe+vel_reg+heading
  [TWEAK-2] LearnedStepWeights thay STEP_WEIGHTS fixed array
  [TWEAK-3] K=3 mode clustering với multi-checkpoint (24h+48h+72h)

Bug fixes giữ nguyên:
  BUG-5: val_loss dùng epoch=-1 → skip augmentation
  BUG-6: any_improve = ANY metric improve → reset early stop
  BUG-7: sched.step() unconditional
  BUG-8: clip chỉ backbone params, không clip learned weights (step_weights)

Log format: fm, pos(=dpe), head, smooth(=accel), disp(=vel_reg), sw72, swr
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, random, time
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from Model.data.loader_training import data_loader
from Model.flow_matching_model import TCFlowMatching   # v59-tweaked class

try:
    from Model.utils import get_cosine_schedule_with_warmup
except ImportError:
    from torch.optim.lr_scheduler import CosineAnnealingLR
    def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
        return CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=min_lr)

TARGETS = {
    "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
    "12h": 65.42,  "24h": 104.67, "48h": 205.10,
}
R_EARTH = 6371.0


def _unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m

def move(b, dev):
    out = list(b)
    for i, x in enumerate(out):
        if torch.is_tensor(x): out[i] = x.to(dev)
        elif isinstance(x, dict):
            out[i] = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in x.items()}
    return out


# ── Metrics ───────────────────────────────────────────────────────────────────

def _ntd(a):
    o = a.clone()
    o[..., 0] = (a[..., 0] * 50.0 + 1800.0) / 10.0
    o[..., 1] = (a[..., 1] * 50.0) / 10.0
    return o

def _hav(p1, p2):
    la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
    dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
    dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
    a = (torch.sin(dlat/2).pow(2)
         + torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2))
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

def _atecte(pd, gd):
    T = min(pd.shape[0], gd.shape[0])
    if T < 2:
        z = pd.new_zeros(1, pd.shape[1]); return z, z
    lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
    lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
    lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
    ya=torch.sin(lo2-lo1)*torch.cos(la2)
    xa=(torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1))
    be=torch.atan2(ya,xa)
    ye=torch.sin(lo3-lo2)*torch.cos(la3)
    xe=(torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2))
    bee=torch.atan2(ye,xe)
    tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
    return tot*torch.cos(ang), tot*torch.sin(ang)

class Acc:
    def __init__(self):
        self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list)
        self._h={12:1, 24:3, 48:7, 72:11}
    def update(self, dist, ate=None, cte=None):
        self.d.extend(dist.mean(0).tolist())
        for h,s in self._h.items():
            if s < dist.shape[0]: self.sd[h].extend(dist[s].tolist())
        if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
        if cte is not None: self.c.extend(cte.abs().mean(0).tolist())
    def compute(self):
        r = {"ADE": float(np.mean(self.d)) if self.d else float("nan"),
             "ATE": float(np.mean(self.a)) if self.a else float("nan"),
             "CTE": float(np.mean(self.c)) if self.c else float("nan"),
             "n": len(self.d)}
        for h in self._h:
            v=self.sd.get(h,[]); r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
        return r

def _score(r):
    ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
    ate=r.get("ATE",1e9); cte=r.get("CTE",1e9)
    if not np.isfinite(ate): ate=ade*0.46
    if not np.isfinite(cte): cte=ade*0.53
    return 100.0*(0.05*(ade/136.)+0.10*(r.get("12h",ade)/50.)
                  +0.15*(r.get("24h",ade)/100.)+0.20*(r.get("48h",ade)/200.)
                  +0.25*(h72/300.)+0.13*(ate/80.)+0.12*(cte/94.))

def _beat(r):
    p=[]
    for k,t in [("ADE",172.68),("ATE",142.21),("CTE",42.04),
                ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
        v=r.get(k,1e9)
        if np.isfinite(v) and v < t: p.append(f"{k}:{v:.1f}")
    return "*** BEAT ST-TRANS: "+" ".join(p)+" ***" if p else ""

def _gap(r):
    out=[]
    for k,ref in [("ADE",172.68),("72h",321.39),("ATE",142.21),("CTE",42.04)]:
        v=r.get(k,float("nan"))
        if np.isfinite(v):
            out.append(f"{k}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
    return " | ".join(out)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, dev, tag="", ema=None, steps=20):
    bk=None
    if ema:
        try: bk=ema.apply_to(model)
        except: pass
    model.eval(); acc=Acc(); t0=time.perf_counter()
    for b in loader:
        bl=move(list(b),dev)
        result=model.sample(bl, ddim_steps=steps)
        p=result[0] if isinstance(result,(tuple,list)) else result
        g=bl[1]; T=min(p.shape[0],g.shape[0])
        pd=_ntd(p[:T]); gd=_ntd(g[:T])
        dist=_hav(pd,gd); at,ct=_atecte(pd,gd)
        acc.update(dist,at,ct)
    if bk:
        try: ema.restore(model,bk)
        except: pass
    r=acc.compute(); el=time.perf_counter()-t0
    def _v(k): return r.get(k,float("nan"))
    def _m(v,t): return "ok" if np.isfinite(v) and v<t else "no"
    print(f"\n{'='*70}")
    print(f"  [{tag}  {el:.0f}s]")
    print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]"
          f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
          f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
    print(f"  ATE={_v('ATE'):.1f}[{_m(_v('ATE'),142.21)}]"
          f"  CTE={_v('CTE'):.1f}[{_m(_v('CTE'),42.04)}]")
    print(f"  vs ST-Trans: {_gap(r)}")
    bt=_beat(r)
    if bt: print(f"  {bt}")
    print(f"  Score={_score(r):.2f}")
    print(f"{'='*70}\n")
    return r


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl):
    m=_unwrap(model); ema=getattr(m,"_ema",None); esd=None
    if ema and hasattr(ema,"shadow"):
        try: esd={k:v.cpu().clone() for k,v in ema.shadow.items()}
        except: pass
    torch.save({"epoch":ep,"model_state_dict":m.state_dict(),
                "optimizer_state":opt.state_dict(),"scheduler_state":sched.state_dict(),
                "ema_shadow":esd,"best_score":saver.bs,"best_ade":saver.ba,
                "best_72h":saver.b7,"best_ate":saver.bat,"best_cte":saver.bc,
                "train_loss":tl,"val_loss":vl,"version":"v59tweaked"}, path)


# ── Saver ─────────────────────────────────────────────────────────────────────

class Saver:
    def __init__(self, patience=40, min_ep=35, enable_stop=True):
        self.patience=patience; self.min_ep=min_ep; self.enable_stop=enable_stop
        self.cnt=0; self.stop=False
        self.bs=self.ba=self.b7=self.bat=self.bc=float("inf")

    def update(self, r, model, out_dir, ep, opt, sched, tl, vl, tag=""):
        sc=_score(r); ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
        ate=r.get("ATE",1e9); cte=r.get("CTE",1e9)
        any_improved=False
        for v,attr,fn in [(ade,"ba",f"best_ade_{tag}.pth"),
                          (h72,"b7",f"best_72h_{tag}.pth"),
                          (ate,"bat",f"best_ate_{tag}.pth"),
                          (cte,"bc",f"best_cte_{tag}.pth")]:
            if v < getattr(self,attr):
                setattr(self,attr,v)
                _save_ckpt(os.path.join(out_dir,fn),ep,model,opt,sched,self,tl,vl)
                any_improved=True
        if sc < self.bs:
            self.bs=sc; any_improved=True
            _save_ckpt(os.path.join(out_dir,f"best_{tag or 'composite'}.pth"),
                       ep,model,opt,sched,self,tl,vl)
            print(f"  [BEST] {tag} ep={ep} score={sc:.2f}"
                  f"  ADE={ade:.1f}  72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}")
        if self.enable_stop:
            if any_improved: self.cnt=0
            else:
                self.cnt+=1
                print(f"  No improve {self.cnt}/{self.patience}"
                      f"  best={self.bs:.2f}  cur={sc:.2f}")
            if ep>=self.min_ep and self.cnt>=self.patience:
                self.stop=True


def _mksub(ds, n, bs, cf):
    idx=random.Random(42).sample(range(len(ds)), min(n,len(ds)))
    return DataLoader(Subset(ds,idx),batch_size=bs,shuffle=False,
                      collate_fn=cf,num_workers=0,drop_last=False)


# ── Args ──────────────────────────────────────────────────────────────────────

def get_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_root",       default="TCND_vn")
    p.add_argument("--obs_len",            default=8,    type=int)
    p.add_argument("--pred_len",           default=12,   type=int)
    p.add_argument("--batch_size",         default=32,   type=int)
    p.add_argument("--num_epochs",         default=120,  type=int)
    p.add_argument("--learning_rate",      default=1e-4, type=float)
    p.add_argument("--weight_decay",       default=1e-3, type=float)
    p.add_argument("--warmup_epochs",      default=3,    type=int)
    p.add_argument("--grad_clip",          default=1.0,  type=float)
    p.add_argument("--use_amp",            action="store_true")
    p.add_argument("--num_workers",        default=2,    type=int)
    p.add_argument("--sigma_min",          default=0.02, type=float)
    p.add_argument("--use_ot",             default=True, action="store_true")
    p.add_argument("--no_ot",              dest="use_ot", action="store_false")
    p.add_argument("--cfg_guidance_scale", default=1.5,  type=float)
    p.add_argument("--n_ensemble",         default=50,   type=int)
    p.add_argument("--use_ema",            default=True, action="store_true")
    p.add_argument("--no_ema",             dest="use_ema", action="store_false")
    p.add_argument("--ema_decay",          default=0.995, type=float)
    p.add_argument("--patience",           default=40,   type=int)
    p.add_argument("--min_ep",             default=35,   type=int)
    p.add_argument("--val_freq",           default=3,    type=int)
    p.add_argument("--val_subset_size",    default=500,  type=int)
    p.add_argument("--fast_ddim",          default=10,   type=int)
    p.add_argument("--full_ddim",          default=20,   type=int)
    p.add_argument("--output_dir",         default="runs/v59tweaked")
    p.add_argument("--gpu_num",            default="0")
    p.add_argument("--delim",              default=" ")
    p.add_argument("--skip",               default=1,    type=int)
    p.add_argument("--min_ped",            default=1,    type=int)
    p.add_argument("--threshold",          default=0.002,type=float)
    p.add_argument("--other_modal",        default="gph")
    p.add_argument("--test_year",          default=None, type=int)
    p.add_argument("--resume",             default=None)
    p.add_argument("--resume_epoch",       default=None, type=int)
    p.add_argument("--eval_test_after_train", default=True, action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("  TC-FlowMatching v59-Tweaked")
    print("  Loss: L_fm + dpe(1.2x) + vel_reg(1.4x) + heading(0.4x) + sw_penalty")
    print("  TWEAK-1: bỏ ATE/CTE/endpoint (không tổng quát)")
    print("  TWEAK-2: LearnedStepWeights — tự học từ data, init ratio=12x")
    print("  TWEAK-3: K=3 multi-checkpoint clustering (24h+48h+72h)")
    print("  BUG fixes: BUG-5(val_noaug) BUG-6(any_improve) BUG-7(sched_uncond) BUG-8(clip_backbone)")
    print(f"  EMA decay={args.ema_decay}  n_ensemble={args.n_ensemble}")
    print(f"  patience={args.patience}  min_ep={args.min_ep}")
    print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']} ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
    print("=" * 72)

    trd,trl=data_loader(args,{"root":args.dataset_root,"type":"train"},test=False)
    vd, vl =data_loader(args,{"root":args.dataset_root,"type":"val"},  test=True)
    from Model.data.trajectoriesWithMe_unet_training import seq_collate
    vsub=_mksub(vd,args.val_subset_size,args.batch_size,seq_collate)
    print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

    model=TCFlowMatching(
        pred_len=args.pred_len, obs_len=args.obs_len,
        sigma_min=args.sigma_min, use_ema=args.use_ema,
        ema_decay=args.ema_decay, use_ate_ot=args.use_ot,
        cfg_guidance_scale=args.cfg_guidance_scale,
        n_ensemble=args.n_ensemble,
    ).to(dev)
    model.init_ema()

    n_total=sum(p.numel() for p in model.parameters() if p.requires_grad)
    # BUG-8: backbone (numel>100) vs learned weights (step_weights, numel<=12)
    backbone_params=[p for p in model.parameters() if p.requires_grad and p.numel()>100]
    small_params   =[p for p in model.parameters() if p.requires_grad and p.numel()<=100]
    print(f"  params total: {n_total:,}")
    print(f"  backbone (clipped at {args.grad_clip}): {sum(p.numel() for p in backbone_params):,}")
    print(f"  learned_w step_weights (NOT clipped): {sum(p.numel() for p in small_params):,}")

    opt=optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler=GradScaler("cuda", enabled=args.use_amp)
    nstep=len(trl); total=nstep*args.num_epochs; wstp=nstep*args.warmup_epochs
    sched=get_cosine_schedule_with_warmup(opt,wstp,total,min_lr=1e-6)

    fast_saver=Saver(patience=args.patience,min_ep=args.min_ep,enable_stop=False)
    full_saver=Saver(patience=args.patience,min_ep=args.min_ep,enable_stop=True)

    start=0
    if args.resume and os.path.exists(args.resume):
        print(f"  Loading: {args.resume}")
        ck=torch.load(args.resume,map_location=dev)
        m=_unwrap(model)
        ms,_=m.load_state_dict(ck["model_state_dict"],strict=False)
        if ms: print(f"  Missing: {len(ms)}")
        ema=getattr(m,"_ema",None)
        if ema and ck.get("ema_shadow"):
            for k,v in ck["ema_shadow"].items():
                if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
        try: opt.load_state_dict(ck["optimizer_state"])
        except Exception as e: print(f"  Opt: {e}")
        try: sched.load_state_dict(ck["scheduler_state"])
        except: 
            for _ in range(ck.get("epoch",0)*nstep): sched.step()
        for a,attr in [("best_score","bs"),("best_ade","ba"),("best_72h","b7"),
                       ("best_ate","bat"),("best_cte","bc")]:
            if a in ck:
                setattr(full_saver,attr,ck[a]); setattr(fast_saver,attr,ck[a])
        start=args.resume_epoch or ck.get("epoch",0)+1
        print(f"  Resume ep={start}  best={full_saver.bs:.2f}")

    try:
        model=torch.compile(model,mode="reduce-overhead"); print("  torch.compile: ok")
    except: pass

    print(f"  Training: {nstep} steps/ep, start ep={start}")
    print("=" * 72)
    ts=time.perf_counter()

    for ep in range(start, args.num_epochs):
        model.train(); sl=0.0; t0=time.perf_counter()

        for i, batch in enumerate(trl):
            bl=move(list(batch),dev)
            with autocast(device_type="cuda", enabled=args.use_amp):
                bd=model.get_loss_breakdown(bl, epoch=ep)

            opt.zero_grad()
            scaler.scale(bd["total"]).backward()
            scaler.unscale_(opt)
            # BUG-8: clip only backbone
            torch.nn.utils.clip_grad_norm_(backbone_params, args.grad_clip)
            scaler.step(opt); scaler.update()
            # BUG-7: unconditional
            try: sched.step()
            except: pass
            model.ema_update()
            sl+=bd["total"].item()

            if i % 20 == 0:
                lr=opt.param_groups[0]["lr"]
                tot=bd["total"].item()
                warn=" [!>10]" if tot>10.0 else ""
                swr=bd.get("sw_ratio",0.0); sw72=bd.get("sw_72h",0.0)
                lw_fm=bd.get("lw_fm",2.0); lw_pos=bd.get("lw_pos",0.6)
                lw_fmpos=bd.get("lw_fm_pos",3.3)
                print(
                    f"  [{ep:>3}][{i:>3}/{nstep}]"
                    f"  tot={tot:.3f}{warn}"
                    f"  fm={bd.get('l_fm',0):.4f}"
                    f"  pos={bd.get('l_pos',0):.4f}"
                    f"  head={bd.get('l_head',0):.4f}"
                    f"  disp={bd.get('l_disp',0):.4f}"
                    f"  smo={bd.get('l_smooth',0):.4f}"
                    f"  sw72={sw72:.2f}  swr={swr:.1f}"
                    f"  wfm={lw_fm:.2f}  wpos={lw_pos:.2f}  r={lw_fmpos:.1f}"
                    f"  lr={lr:.2e}"
                )

        avt=sl/nstep; t_ep=time.perf_counter()-t0

        # BUG-5: val với epoch=-1 → skip aug
        model.eval(); vls=0.0
        with torch.no_grad():
            for batch in vl:
                bv=move(list(batch),dev)
                with autocast(device_type="cuda", enabled=args.use_amp):
                    vls+=model.get_loss(bv, epoch=-1).item()
        avv=vls/len(vl)
        lr_cur=opt.param_groups[0]["lr"]
        print(f"  Epoch {ep:>3} | train={avt:.4f} val={avv:.4f}"
              f" | lr={lr_cur:.2e} | {t_ep:.0f}s")

        # Fast eval — monitor only, no early stop
        rf=evaluate(model, vsub, dev, tag=f"FAST ep{ep}", steps=args.fast_ddim)
        fast_saver.update(rf, model, args.output_dir, ep, opt, sched, avt, avv, tag="fast")

        # Full val every val_freq epochs — controls early stop
        if ep % args.val_freq == 0:
            em=getattr(_unwrap(model),"_ema",None)
            rr=evaluate(model, vl, dev, tag=f"RAW ep{ep}", steps=args.full_ddim)
            full_saver.update(rr, model, args.output_dir, ep, opt, sched, avt, avv, tag="raw")
            if em and ep >= 3:
                re=evaluate(model, vl, dev, tag=f"EMA ep{ep}", ema=em, steps=args.full_ddim)
                full_saver.update(re, model, args.output_dir, ep, opt, sched, avt, avv, tag="ema")

        if ep%10==0 or ep==args.num_epochs-1:
            _save_ckpt(os.path.join(args.output_dir,f"ckpt_ep{ep:03d}.pth"),
                       ep, model, opt, sched, full_saver, avt, avv)

        if full_saver.stop:
            print(f"  Early stop ep={ep}"); break

    th=(time.perf_counter()-ts)/3600.0
    print(f"\n  Best: ADE={full_saver.ba:.1f}  72h={full_saver.b7:.0f}"
          f"  ATE={full_saver.bat:.1f}  CTE={full_saver.bc:.1f}  ({th:.2f}h)")

    # Post-training test
    if args.eval_test_after_train:
        print("\n"+"="*72+"\n  POST-TRAINING TEST\n"+"="*72)
        try: _,tl2=data_loader(args,{"root":args.dataset_root,"type":"test"},test=True)
        except: print("  No test → using val"); tl2=vl
        for fn,lb in [("best_raw.pth","RAW"),("best_ema.pth","EMA"),
                      ("best_ade_raw.pth","BEST_ADE"),("best_cte_raw.pth","BEST_CTE")]:
            pp=os.path.join(args.output_dir,fn)
            if not os.path.exists(pp): continue
            ck=torch.load(pp,map_location=dev)
            _unwrap(model).load_state_dict(ck["model_state_dict"],strict=False)
            em=getattr(_unwrap(model),"_ema",None)
            if em and ck.get("ema_shadow"):
                for k,v in ck["ema_shadow"].items():
                    if k in em.shadow: em.shadow[k].copy_(v.to(dev))
            r=evaluate(model,tl2,dev,tag=f"TEST/{lb}",steps=args.full_ddim)
            print(f"\n  --- {lb} (ep={ck.get('epoch','?')}) ---")
            for key,ref in [("ADE",172.68),("72h",321.39),("ATE",142.21),("CTE",42.04)]:
                v=r.get(key,float("nan"))
                mk="BEAT!" if np.isfinite(v) and v<ref else f"need {ref:.0f}"
                print(f"    {key:<8}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
    print("="*72)


if __name__ == "__main__":
    args=get_args()
    np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    main(args)
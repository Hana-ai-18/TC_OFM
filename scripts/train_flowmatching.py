# """
# scripts/train_flowmatching.py  ── FM v60 Training Entry Point
# ==============================================================
# THAY THẾ scripts/train_flowmatching.py cũ.
# Cùng interface, chạy được trên Kaggle với lệnh:

#   !python scripts/train_flowmatching.py \
#       --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
#       --output_dir   /kaggle/working/runs/fm_v60 \
#       --batch_size 32 --num_epochs 70 --learning_rate 2e-4 --use_amp
# """
# from __future__ import annotations

# import argparse
# import json
# import os
# import random
# import sys
# import time
# from collections import defaultdict
# from typing import Dict, Optional

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# from torch.utils.data import DataLoader, Subset

# # ── Path setup ────────────────────────────────────────────────
# # __file__ = project_root/scripts/train_flowmatching.py
# # _root    = project_root/          ← chứa Model/ và scripts/
# _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if _root not in sys.path:
#     sys.path.insert(0, _root)

# # ── Imports ───────────────────────────────────────────────────
# from Model.flow_matching_model import (
#     TCFlowMatching, compute_speed_stats_from_norm,
#     _norm_to_deg, haversine_km
# )

# try:
#     from Model.data.loader_training import data_loader as tc_data_loader
#     from Model.data.trajectoriesWithMe_unet_training import seq_collate
#     _LOADER_OK = True
# except ImportError as e:
#     print(f'[WARN] Data loader import failed: {e}')
#     _LOADER_OK = False

# ST_TRANS = {
#     'ADE': 224.4, 'ATE': 213.7, 'CTE': 59.4,
#     '12h': 77.5,  '24h': 130.5, '48h': 269.9, '72h': 423.3,
# }
# FM_V59 = {'ADE': 236.1, 'ATE': 223.6, 'CTE': 74.9, '72h': 471.6}


# # ══════════════════════════════════════════════════════════════
# #  Utilities
# # ══════════════════════════════════════════════════════════════

# def set_seed(s):
#     random.seed(s); np.random.seed(s); torch.manual_seed(s)
#     if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


# def count_params(m):
#     return sum(p.numel() for p in m.parameters() if p.requires_grad)


# def _move(batch_list, device):
#     out = []
#     for item in batch_list:
#         if isinstance(item, torch.Tensor):
#             out.append(item.to(device))
#         elif isinstance(item, dict):
#             out.append({k: v.to(device) if isinstance(v, torch.Tensor) else v
#                         for k, v in item.items()})
#         else:
#             out.append(item)
#     return out


# # ══════════════════════════════════════════════════════════════
# #  Metrics
# # ══════════════════════════════════════════════════════════════

# def compute_metrics(pred_deg, gt_deg):
#     """[N,T,2] lon°,lat° → dict"""
#     N, T, _ = pred_deg.shape
#     dist = haversine_km(
#         pred_deg.reshape(N*T, 2), gt_deg.reshape(N*T, 2)
#     ).reshape(N, T)

#     out = {'ADE': float(dist.mean()), 'FDE': float(dist[:,-1].mean())}
#     for step, name in {2:'12h', 4:'24h', 8:'48h', 12:'72h'}.items():
#         if step-1 < T: out[name] = float(dist[:, step-1].mean())

#     # ATE / CTE
#     cos_lat = torch.cos(torch.deg2rad(gt_deg[...,1])).clamp(1e-3)
#     elon = (pred_deg[...,0]-gt_deg[...,0])*cos_lat*111.
#     elat = (pred_deg[...,1]-gt_deg[...,1])*111.
#     if T >= 2:
#         dlon = (gt_deg[:,1:,0]-gt_deg[:,:-1,0])*cos_lat[:,1:]*111.
#         dlat = (gt_deg[:,1:,1]-gt_deg[:,:-1,1])*111.
#         mag  = (dlon**2+dlat**2).sqrt().clamp(1e-3)
#         ux = torch.cat([dlon[:,:1]/mag[:,:1], dlon/mag], 1)[:,:T]
#         uy = torch.cat([dlat[:,:1]/mag[:,:1], dlat/mag], 1)[:,:T]
#         out['ATE'] = float((elon*ux+elat*uy).abs().mean())
#         out['CTE'] = float((elon*uy-elat*ux).abs().mean())
#     else:
#         out['ATE'] = out['CTE'] = float(dist.mean())

#     # Speed bias
#     if T >= 2:
#         def _s(t):
#             dl=t[:,1:,1]-t[:,:-1,1]; dlo=t[:,1:,0]-t[:,:-1,0]
#             cl=torch.cos(torch.deg2rad((t[:,1:,1]+t[:,:-1,1])/2)).clamp(1e-3)
#             return float((dl*111)**2+(dlo*111*cl)**2).sqrt().mean() if False \
#                    else float(torch.sqrt((dl*111)**2+(dlo*111*cl)**2).mean())
#         out['speed_bias'] = _s(pred_deg) - _s(gt_deg)
#         out['mean_pred_speed'] = _s(pred_deg)
#         out['mean_gt_speed']   = _s(gt_deg)
#     return out


# def print_metrics(metrics, tag='', elapsed=0.):
#     print(f'\n{"="*70}')
#     if tag: print(f'  [{tag}  {elapsed:.0f}s]')
#     for k in ['ADE','12h','24h','48h','72h']:
#         v = metrics.get(k, float('nan'))
#         ref = ST_TRANS.get(k, float('nan'))
#         ok  = '✅ BEAT' if v < ref else '❌'
#         print(f'  {k:>4} = {v:>7.1f} km  {ok} ST-Trans={ref:.1f}  ({v-ref:+.1f})')
#     if 'ATE' in metrics:
#         print(f'  ATE  = {metrics["ATE"]:>7.1f}  '
#               f'{"✅ BEAT" if metrics["ATE"]<ST_TRANS["ATE"] else "❌"}')
#         print(f'  CTE  = {metrics["CTE"]:>7.1f}  '
#               f'{"✅ BEAT" if metrics["CTE"]<ST_TRANS["CTE"] else "❌"}')
#     if 'speed_bias' in metrics:
#         sb = metrics['speed_bias']
#         print(f'  speed_bias={sb:+.2f} km/6h  '
#               f'{"✅" if abs(sb)<10 else "❌"}  '
#               f'pred={metrics["mean_pred_speed"]:.1f}  '
#               f'gt={metrics["mean_gt_speed"]:.1f}')
#     print(f'{"="*70}\n')


# # ══════════════════════════════════════════════════════════════
# #  Evaluate
# # ══════════════════════════════════════════════════════════════

# @torch.no_grad()
# def evaluate(model, loader, device, tag='', num_ens=20, steps=10):
#     model.eval(); t0 = time.perf_counter()
#     all_p, all_g = [], []
#     for bl in loader:
#         bl = _move(bl, device)
#         try:
#             pred_n, _ = model.sample(bl, num_ensemble=num_ens, ddim_steps=steps)
#         except Exception as e:
#             print(f'  [WARN] sample failed: {e}'); continue
#         gt_n = bl[1]
#         T = min(pred_n.shape[0], gt_n.shape[0])
#         all_p.append(_norm_to_deg(pred_n[:T].permute(1,0,2)).cpu())
#         all_g.append(_norm_to_deg(gt_n[:T].permute(1,0,2)).cpu())
#     if not all_p: return {}
#     m = compute_metrics(torch.cat(all_p,0), torch.cat(all_g,0))
#     print_metrics(m, tag, time.perf_counter()-t0)
#     return m


# # ══════════════════════════════════════════════════════════════
# #  Checkpoint
# # ══════════════════════════════════════════════════════════════

# def save_ckpt(path, epoch, model, opt, sched, metrics):
#     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
#     torch.save({'epoch': epoch, 'model_state': model.state_dict(),
#                 'optim_state': opt.state_dict(),
#                 'sched_state': sched.state_dict() if sched else None,
#                 'metrics': metrics}, path)


# def load_ckpt(path, model, opt=None, sched=None):
#     ck = torch.load(path, map_location='cpu')
#     model.load_state_dict(ck['model_state'], strict=False)
#     if opt and 'optim_state' in ck:
#         try: opt.load_state_dict(ck['optim_state'])
#         except Exception as e: print(f'  [warn] opt state: {e}')
#     if sched and ck.get('sched_state'):
#         try: sched.load_state_dict(ck['sched_state'])
#         except: pass
#     return ck.get('epoch', 0), ck.get('metrics', {})


# class BestSaver:
#     def __init__(self, d, patience=15, min_ep=20):
#         self.d=d; self.patience=patience; self.min_ep=min_ep
#         self.best={k:float('inf') for k in ['ADE','72h','ATE','CTE']}
#         self.no_imp=0; self.stop=False; os.makedirs(d, exist_ok=True)

#     def update(self, m, ep, model, opt, sched):
#         imp = False
#         for k,fn in [('ADE','best_ade.pth'),('72h','best_72h.pth'),
#                      ('ATE','best_ate.pth'),('CTE','best_cte.pth')]:
#             v = m.get(k, float('inf'))
#             if v < self.best[k]:
#                 self.best[k]=v
#                 save_ckpt(os.path.join(self.d,fn), ep, model, opt, sched, m)
#                 print(f'  [BEST {k}] ep={ep}  '
#                       f'ADE={m.get("ADE",0):.1f}  '
#                       f'CTE={m.get("CTE",0):.1f}  '
#                       f'72h={m.get("72h",0):.0f}')
#                 imp=True
#         if imp: self.no_imp=0
#         else:
#             self.no_imp+=1
#             if ep>=self.min_ep and self.no_imp>=self.patience:
#                 self.stop=True; print(f'  [EARLY STOP] ep={ep}')
#         return imp


# # ══════════════════════════════════════════════════════════════
# #  Data loaders
# # ══════════════════════════════════════════════════════════════

# class _Args:
#     def __init__(self, **kw):
#         for k,v in kw.items(): setattr(self, k, v)


# def build_loaders(data_root, batch_size, num_workers):
#     if not _LOADER_OK:
#         raise RuntimeError('Data loader not available. Check Model/data/.')
#     dummy = _Args(obs_len=8, pred_len=12, skip=1, threshold=0.002,
#                   min_ped=1, delim=' ', other_modal='gph',
#                   batch_size=batch_size, num_workers=num_workers)
#     tr_ds, tr_ld = tc_data_loader(dummy, {'root':data_root,'type':'train'},
#                                    test=False, batch_size=batch_size)
#     vl_ds, vl_ld = tc_data_loader(dummy, {'root':data_root,'type':'val'},
#                                    test=True, batch_size=batch_size)
#     return tr_ds, tr_ld, vl_ds, vl_ld


# # ══════════════════════════════════════════════════════════════
# #  Train epoch
# # ══════════════════════════════════════════════════════════════

# def train_epoch(model, loader, optimizer, epoch, device, scaler=None):
#     model.train(); tot=0.; terms=defaultdict(float); n=0
#     t0=time.perf_counter()

#     for i, bl in enumerate(loader):
#         bl = _move(bl, device)
#         optimizer.zero_grad()

#         if scaler:
#             with torch.amp.autocast('cuda'):
#                 bd = model.get_loss_breakdown(bl, epoch=epoch)
#             loss = bd['total']
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             nn.utils.clip_grad_norm_(model.parameters(), 5.)
#             scaler.step(optimizer); scaler.update()
#         else:
#             bd = model.get_loss_breakdown(bl, epoch=epoch)
#             loss = bd['total']
#             if torch.is_tensor(loss) and loss.requires_grad:
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(model.parameters(), 5.)
#                 optimizer.step()

#         if not (torch.is_tensor(loss) and torch.isfinite(loss)):
#             if i < 5: print(f'  [WARN] non-finite loss at batch {i}')
#             continue

#         tot += loss.item(); n += 1
#         for k,v in bd.items():
#             if k != 'total' and not torch.is_tensor(v): terms[k] += float(v)

#         if i % 20 == 0:
#             lr = optimizer.param_groups[0]['lr']
#             print(f'  [{epoch:>3}][{i:>4}/{len(loader)}]'
#                   f' loss={loss.item():.3f}'
#                   f' l_kin={bd.get("l_kin",0):.3f}'
#                   f' l_logspd={bd.get("l_logspd",0):.3f}'
#                   f' l_fm={bd.get("l_fm",0):.3f}'
#                   f' l_scr={bd.get("l_scorer",0):.3f}'
#                   f' sw={bd.get("sw_ratio",0):.1f}'
#                   f' λk={bd.get("lam_kin",0):.3f}'
#                   f' λs={bd.get("lam_logspd",0):.3f}'
#                   f' dw={bd.get("diff_w_mean",1):.2f}'
#                   f' lr={lr:.1e}')

#     elapsed = time.perf_counter()-t0
#     avg = {k: v/max(n,1) for k,v in terms.items()}
#     avg['loss'] = tot/max(n,1); avg['elapsed'] = elapsed
#     return avg


# # ══════════════════════════════════════════════════════════════
# #  Main
# # ══════════════════════════════════════════════════════════════

# def train(data_root, save_dir, max_epochs=70, batch_size=32, lr=2e-4,
#           num_workers=0, seed=42, use_amp=False, resume=None, patience=15):
#     set_seed(seed)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     os.makedirs(save_dir, exist_ok=True)

#     print(f'\n{"="*72}')
#     print(f'  FM v60 — Physics-Grounded TC Track Forecasting')
#     print(f'  Device: {device}  AMP: {use_amp}')
#     print(f'  Baseline → FM_v59={FM_V59["ADE"]:.1f}  ST-Trans={ST_TRANS["ADE"]:.1f}')
#     print(f'  Targets  → ADE<170  CTE<52  72h<360')
#     print(f'{"="*72}\n')

#     tr_ds, tr_ld, vl_ds, vl_ld = build_loaders(data_root, batch_size, num_workers)
#     print(f'  train: {len(tr_ds)}  val: {len(vl_ds)}')

#     model = TCFlowMatching(
#         pred_len=12, obs_len=8, sigma_min=0.02, unet_in_ch=13,
#         ctx_noise_scale=0.01, use_ema=True, ema_decay=0.995,
#         use_ate_ot=True, ot_epsilon=0.05, cfg_uncond_prob=0.1,
#     ).to(device)
#     model.init_ema()
#     print(f'  Params: {count_params(model):,}')

#     # 3 param groups với LR khác nhau
#     model_p   = [p for n,p in model.named_parameters()
#                  if not n.startswith('criterion.')]
#     weight_p  = (list(model.criterion.step_weights.parameters())
#                + list(model.criterion.loss_weights.parameters())
#                + list(model.criterion.diff_weighter.parameters()))
#     scorer_p  = list(model.criterion.scorer.parameters())

#     optimizer = AdamW([
#         {'params': model_p,  'lr': lr,      'weight_decay': 1e-4},
#         {'params': weight_p, 'lr': lr*0.5,  'weight_decay': 0.},
#         {'params': scorer_p, 'lr': lr,      'weight_decay': 1e-5},
#     ])
#     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=35, T_mult=1, eta_min=1e-5)
#     scaler    = torch.amp.GradScaler('cuda') if torch.cuda.is_available() and use_amp else None

#     start_ep = 1
#     if resume and os.path.exists(resume):
#         start_ep, _ = load_ckpt(resume, model, optimizer, scheduler)
#         start_ep += 1
#         print(f'  Resumed from {resume} (ep {start_ep})')

#     saver   = BestSaver(save_dir, patience=patience, min_ep=20)
#     history = []

#     for epoch in range(start_ep, max_epochs+1):
#         phase = ('1-foundation' if epoch<=10 else
#                  '2-integration' if epoch<=35 else '3-refinement')
#         print(f'\n  ── Epoch {epoch}/{max_epochs}  Phase={phase} ──────────')

#         if epoch > 1: model.ema_update()

#         ts = train_epoch(model, tr_ld, optimizer, epoch, device, scaler)
#         scheduler.step()

#         sw = model.criterion.step_weights.stats()
#         lw = model.criterion.loss_weights.stats()
#         print(f'  train={ts["loss"]:.4f}'
#               f'  sw_ratio={sw["sw_ratio"]:.2f}'
#               f'  mono={sw["sw_monotonic"]}'
#               f'  λ_kin={lw["lam_kin"]:.3f}'
#               f'  λ_spd={lw["lam_logspd"]:.3f}'
#               f'  λ_curv={lw["lam_curv"]:.3f}')

#         # Fast val mỗi epoch
#         fast_n  = min(200, len(vl_ds))
#         fast_idx = random.sample(range(len(vl_ds)), fast_n)
#         fast_ld  = DataLoader(
#             Subset(vl_ds, fast_idx),
#             batch_size=batch_size, shuffle=False,
#             num_workers=0, drop_last=False,
#             collate_fn=seq_collate if _LOADER_OK else None)
#         fast_m = evaluate(model, fast_ld, device,
#                            tag=f'FAST ep{epoch} n={fast_n}',
#                            num_ens=10, steps=8)

#         # Full val mỗi 3 epochs
#         if epoch % 3 == 0 or epoch == max_epochs:
#             full_m = evaluate(model, vl_ld, device,
#                                tag=f'FULL ep{epoch}',
#                                num_ens=20, steps=10)
#             saver.update(full_m, epoch, model, optimizer, scheduler)
#             history.append({'epoch': epoch, 'train_loss': ts['loss'], **full_m})
#         else:
#             saver.update(fast_m, epoch, model, optimizer, scheduler)

#         if epoch % 10 == 0:
#             save_ckpt(os.path.join(save_dir, f'ckpt_ep{epoch:03d}.pth'),
#                        epoch, model, optimizer, scheduler, fast_m)

#         if saver.stop:
#             print(f'  Early stop ep={epoch}'); break

#     print(f'\n{"="*72}')
#     print(f'  DONE  Best ADE={saver.best["ADE"]:.1f}'
#           f'  CTE={saver.best["CTE"]:.1f}  72h={saver.best["72h"]:.1f}')
#     with open(os.path.join(save_dir, 'history.json'), 'w') as f:
#         json.dump(history, f, indent=2)
#     print(f'{"="*72}\n')
#     return saver.best['ADE']


# # ══════════════════════════════════════════════════════════════
# #  CLI
# # ══════════════════════════════════════════════════════════════

# def parse_args():
#     p = argparse.ArgumentParser('FM v60')
#     p.add_argument('--data_root',    default='TCND_vn')
#     p.add_argument('--save_dir',     default='checkpoints/fm_v60/')
#     p.add_argument('--max_epochs',   type=int,   default=70)
#     p.add_argument('--batch_size',   type=int,   default=32)
#     p.add_argument('--lr',           type=float, default=2e-4)
#     p.add_argument('--num_workers',  type=int,   default=0)
#     p.add_argument('--use_amp',      action='store_true')
#     p.add_argument('--resume',       default=None)
#     p.add_argument('--seed',         type=int,   default=42)
#     p.add_argument('--patience',     type=int,   default=15)
#     # Aliases
#     p.add_argument('--dataset_root', default=None)
#     p.add_argument('--output_dir',   default=None)
#     p.add_argument('--learning_rate',type=float, default=None)
#     p.add_argument('--num_epochs',   type=int,   default=None)
#     return p.parse_args()


# def _apply_aliases(args):
#     if getattr(args,'dataset_root',None) and (not getattr(args,'data_root',None) or args.data_root=='TCND_vn'):
#         args.data_root = args.dataset_root
#     if getattr(args,'output_dir',None) and (not getattr(args,'save_dir',None) or args.save_dir=='checkpoints/fm_v60/'):
#         args.save_dir = args.output_dir
#     if getattr(args,'learning_rate',None): args.lr = args.learning_rate
#     if getattr(args,'num_epochs',None):    args.max_epochs = args.num_epochs
#     return args


# if __name__ == '__main__':
#     args = parse_args(); args = _apply_aliases(args)
#     train(data_root=args.data_root, save_dir=args.save_dir,
#           max_epochs=args.max_epochs, batch_size=args.batch_size,
#           lr=args.lr, num_workers=args.num_workers, seed=args.seed,
#           use_amp=args.use_amp, resume=args.resume, patience=args.patience)

"""
train_v74.py — TC-FlowMatching v74
════════════════════════════════════════════════════════════════════════
TRAINING STRATEGY: Single phase, như v59 — không freeze, không transition

KEY DIFFERENCES vs failed versions:
  - V68/V72 fail: phase separation, best ep9 plateau
  - V69 fail: loss scale 700 (disp_smooth in km)
  - V74: single phase như v59, loss scale 1-4

MONITORING (xem để đảm bảo stable):
  sw_ratio >= 3.0          → step weights đang đúng hướng
  lw_dpe_fm ∈ [1.5, 3.0]  → DPE/FM balance ổn
  L_hard > L_easy          → hard storms được penalize nhiều hơn (correct)
  easy_frac ≈ 0.20-0.40    → delta score distribution hợp lý
  tot ∈ [1.0, 5.0]         → loss scale tốt (nếu tot > 10: có bug)

RUN:
  python scripts/train_v74.py \\
      --dataset_root /path/to/tc-ofm \\
      --output_dir   runs/v74 \\
      --batch_size   32 \\
      --use_amp
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, time, random
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from Model.data.loader_training import data_loader
from Model.flow_matching_model import TCFlowMatching

try:
    from Model.utils import get_cosine_schedule_with_warmup
except ImportError:
    from torch.optim.lr_scheduler import CosineAnnealingLR
    def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
        return CosineAnnealingLR(opt, T_max=max(total_steps,1), eta_min=min_lr)

TARGETS = {"ADE":172.68,"72h":321.39,"ATE":142.21,"CTE":42.04,
           "12h":65.42,"24h":104.67,"48h":205.10}
R_EARTH = 6371.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unwrap(m): return m._orig_mod if hasattr(m,'_orig_mod') else m

def move(b, dev):
    out=list(b)
    for i,x in enumerate(out):
        if torch.is_tensor(x): out[i]=x.to(dev)
        elif isinstance(x,dict):
            out[i]={k:v.to(dev) if torch.is_tensor(v) else v for k,v in x.items()}
    return out


# ── Metric utils ──────────────────────────────────────────────────────────────

def _ntd(a):
    o=a.clone(); o[...,0]=(a[...,0]*50.+1800.)/10.; o[...,1]=(a[...,1]*50.)/10.; return o

def _hav(p1,p2):
    la1=torch.deg2rad(p1[...,1]); la2=torch.deg2rad(p2[...,1])
    dlat=torch.deg2rad(p2[...,1]-p1[...,1]); dlon=torch.deg2rad(p2[...,0]-p1[...,0])
    a=torch.sin(dlat/2).pow(2)+torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2)
    return 2.*R_EARTH*torch.asin(a.clamp(1e-12,1-1e-12).sqrt())

def _atecte(pd,gd):
    T=min(pd.shape[0],gd.shape[0])
    if T<2: z=pd.new_zeros(1,pd.shape[1]); return z,z
    lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
    lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
    lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
    ya=torch.sin(lo2-lo1)*torch.cos(la2)
    xa=torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
    be=torch.atan2(ya,xa)
    ye=torch.sin(lo3-lo2)*torch.cos(la3)
    xe=torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
    bee=torch.atan2(ye,xe); tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
    return tot*torch.cos(ang), tot*torch.sin(ang)


class Acc:
    def __init__(self):
        self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list)
        self._h={12:1,24:3,48:7,72:11}
    def update(self,dist,ate=None,cte=None):
        self.d.extend(dist.mean(0).tolist())
        for h,s in self._h.items():
            if s<dist.shape[0]: self.sd[h].extend(dist[s].tolist())
        if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
        if cte is not None: self.c.extend(cte.abs().mean(0).tolist())
    def compute(self):
        r={"ADE":float(np.mean(self.d)) if self.d else float("nan"),
           "ATE_mean":float(np.mean(self.a)) if self.a else float("nan"),
           "CTE_mean":float(np.mean(self.c)) if self.c else float("nan"),
           "n":len(self.d)}
        for h in self._h:
            v=self.sd.get(h,[]); r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
        return r


def _score(r):
    ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
    ate=r.get("ATE_mean",1e9); cte=r.get("CTE_mean",1e9)
    if not np.isfinite(ate): ate=ade*.46
    if not np.isfinite(cte): cte=ade*.53
    return 100.*(0.05*(ade/136.)+0.10*(r.get("12h",ade)/50.)+
                 0.15*(r.get("24h",ade)/100.)+0.20*(r.get("48h",ade)/200.)+
                 0.25*(h72/300.)+0.13*(ate/80.)+0.12*(cte/94.))

def _beat(r):
    p=[]
    for k,t in [("ADE",172.68),("ATE_mean",142.21),("CTE_mean",42.04),
                  ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
        v=r.get(k,1e9)
        if np.isfinite(v) and v<t: p.append(f"{k.replace('_mean','')}:{v:.1f}")
    return "*** BEAT ST-TRANS: "+" ".join(p)+" ***" if p else ""

def _gap(r):
    out=[]
    for k,ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
        v=r.get(k,float("nan"))
        if np.isfinite(v):
            out.append(f"{k.replace('_mean','')}:{v:.0f}({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
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
        result=model.sample(bl,ddim_steps=steps)
        p=result[0] if isinstance(result,(tuple,list)) else result
        g=bl[1]; T=min(p.shape[0],g.shape[0])
        pd=_ntd(p[:T]); gd=_ntd(g[:T])
        dist=_hav(pd,gd); at,ct=_atecte(pd,gd)
        acc.update(dist,at,ct)
    if bk:
        try: ema.restore(model,bk)
        except: pass
    r=acc.compute()
    def _v(k): return r.get(k,float("nan"))
    def _m(v,t): return "ok" if np.isfinite(v) and v<t else "no"
    el=time.perf_counter()-t0
    print(f"\n{'='*70}")
    print(f"  [{tag}  {el:.0f}s]")
    print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]  "
          f"12h={_v('12h'):.0f}  24h={_v('24h'):.0f}  "
          f"48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
    if np.isfinite(_v("ATE_mean")):
        print(f"  ATE={_v('ATE_mean'):.1f}[{_m(_v('ATE_mean'),142.21)}]  "
              f"CTE={_v('CTE_mean'):.1f}[{_m(_v('CTE_mean'),42.04)}]")
    print(f"  vs ST-Trans: {_gap(r)}")
    bt=_beat(r)
    if bt: print(f"  {bt}")
    print(f"  Score={_score(r):.2f}")
    print(f"{'='*70}\n")
    return r


# ── Saver ─────────────────────────────────────────────────────────────────────

def _save_ckpt(path, ep, model, opt, sched, saver, tl, vl, extra=None):
    m=_unwrap(model); ema=getattr(m,"_ema",None); esd=None
    if ema and hasattr(ema,"shadow"):
        try: esd={k:v.cpu().clone() for k,v in ema.shadow.items()}
        except: pass
    d={"epoch":ep,"model_state_dict":m.state_dict(),
       "optimizer_state":opt.state_dict(),"scheduler_state":sched.state_dict(),
       "ema_shadow":esd,"best_score":saver.bs,"best_ade":saver.ba,
       "best_72h":saver.b7,"best_ate":saver.bat,"best_cte":saver.bc,
       "train_loss":tl,"val_loss":vl}
    if extra: d.update(extra)
    torch.save(d,path)


class Saver:
    """
    Single-phase saver với patience=40, min_ep=35.
    Longer patience than v59 (35) để cho model đủ thời gian.
    """
    def __init__(self, patience=40, min_ep=35):
        self.patience=patience; self.min_ep=min_ep; self.cnt=0; self.stop=False
        self.bs=self.ba=self.b7=self.bat=self.bc=float("inf")

    def update(self, r, model, out, ep, opt, sched, tl, vl, tag=""):
        sc=_score(r); ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
        ate=r.get("ATE_mean",1e9); cte=r.get("CTE_mean",1e9)
        for v,a,fn in [(ade,"ba","best_ade.pth"),(h72,"b7","best_72h.pth"),
                        (ate,"bat","best_ate.pth"),(cte,"bc","best_cte.pth")]:
            if v<getattr(self,a):
                setattr(self,a,v)
                _save_ckpt(os.path.join(out,fn),ep,model,opt,sched,self,tl,vl)
        if sc<self.bs:
            self.bs=sc; self.cnt=0
            _save_ckpt(os.path.join(out,f"best_{tag or 'composite'}.pth"),
                       ep,model,opt,sched,self,tl,vl,{"score":sc,"ade":ade,"h72":h72})
            print(f"  [BEST] {tag} ep={ep} score={sc:.2f} "
                  f"ADE={ade:.1f} 72h={h72:.0f} ATE={ate:.1f} CTE={cte:.1f}")
        else:
            self.cnt+=1
            print(f"  No improve {self.cnt}/{self.patience}  best={self.bs:.2f}  cur={sc:.2f}")
        if ep>=self.min_ep and self.cnt>=self.patience:
            self.stop=True; print(f"  [STOP] ep={ep}")


def mksub(ds,n,bs,cf):
    idx=random.Random(42).sample(range(len(ds)),min(n,len(ds)))
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
    p.add_argument("--patience",           default=40,   type=int)
    p.add_argument("--min_ep",             default=35,   type=int)
    p.add_argument("--use_amp",            action="store_true")
    p.add_argument("--num_workers",        default=2,    type=int)
    p.add_argument("--sigma_min",          default=0.02, type=float)
    p.add_argument("--use_ot",             default=True, action="store_true")
    p.add_argument("--no_ot",              dest="use_ot",action="store_false")
    p.add_argument("--cfg_guidance_scale", default=1.5,  type=float)
    p.add_argument("--easy_thresh",        default=0.25, type=float,
                    help="delta < easy_thresh → easy storm (default 0.25)")
    p.add_argument("--n_ensemble",         default=50,   type=int)
    p.add_argument("--val_freq",           default=3,    type=int)
    p.add_argument("--val_subset_size",    default=500,  type=int)
    p.add_argument("--fast_ddim",          default=10,   type=int)
    p.add_argument("--full_ddim",          default=20,   type=int)
    p.add_argument("--use_ema",            default=True, action="store_true")
    p.add_argument("--no_ema",             dest="use_ema",action="store_false")
    p.add_argument("--ema_decay",          default=0.995,type=float)
    p.add_argument("--output_dir",         default="runs/v74")
    p.add_argument("--gpu_num",            default="0")
    p.add_argument("--delim",              default=" ")
    p.add_argument("--skip",               default=1,    type=int)
    p.add_argument("--min_ped",            default=1,    type=int)
    p.add_argument("--threshold",          default=0.002,type=float)
    p.add_argument("--other_modal",        default="gph")
    p.add_argument("--test_year",          default=None, type=int)
    p.add_argument("--resume",             default=None)
    p.add_argument("--resume_epoch",       default=None, type=int)
    p.add_argument("--eval_test_after_train",default=True,action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_num)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir,exist_ok=True)

    print("="*72)
    print(f"  TC-FlowMatching v74")
    print(f"  Loss: L_fm + L_dpe(step_w,d=300) + L_disp(norm) + L_ep(hard)")
    print(f"  Split: easy(delta<{args.easy_thresh}) / hard(delta>={args.easy_thresh})")
    print(f"  Training: single phase, patience={args.patience}, min_ep={args.min_ep}")
    print(f"  Inference: K=3 mode clustering at 72h endpoint")
    print(f"  Target: ADE<{TARGETS['ADE']} 72h<{TARGETS['72h']} ATE<{TARGETS['ATE']} CTE<{TARGETS['CTE']}")
    print("="*72)

    trd,trl=data_loader(args,{"root":args.dataset_root,"type":"train"},test=False)
    vd,vl  =data_loader(args,{"root":args.dataset_root,"type":"val"},  test=True)
    from Model.data.trajectoriesWithMe_unet_training import seq_collate
    vsub=mksub(vd,args.val_subset_size,args.batch_size,seq_collate)
    print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

    model=TCFlowMatching(
        pred_len=args.pred_len, obs_len=args.obs_len,
        sigma_min=args.sigma_min, use_ema=args.use_ema, ema_decay=args.ema_decay,
        use_ot=args.use_ot, cfg_guidance_scale=args.cfg_guidance_scale,
        easy_thresh=args.easy_thresh, n_ensemble=args.n_ensemble,
    ).to(dev)
    model.init_ema()
    n_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    sw_params=sum(p.numel() for p in model.step_weights.parameters())
    lw_params=sum(p.numel() for p in model.loss_weights.parameters())
    print(f"  params total: {n_params:,} | step_w: {sw_params} | loss_w: {lw_params}")

    # Single optimizer, tất cả params — như v59
    opt=optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    saver=Saver(patience=args.patience,min_ep=args.min_ep)
    scaler=GradScaler("cuda",enabled=args.use_amp)
    nstep=len(trl); total=nstep*args.num_epochs; wstp=nstep*args.warmup_epochs
    sched=get_cosine_schedule_with_warmup(opt,wstp,total,min_lr=1e-6)

    # Resume
    start=0
    if args.resume and os.path.exists(args.resume):
        print(f"  Loading: {args.resume}")
        ck=torch.load(args.resume,map_location=dev)
        m=_unwrap(model); ms,_=m.load_state_dict(ck["model_state_dict"],strict=False)
        if ms: print(f"  Missing: {len(ms)}")
        ema=getattr(m,"_ema",None)
        if ema and ck.get("ema_shadow"):
            for k,v in ck["ema_shadow"].items():
                if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
        try: opt.load_state_dict(ck["optimizer_state"])
        except Exception as e: print(f"  Opt not loaded: {e}")
        try: sched.load_state_dict(ck["scheduler_state"])
        except:
            for _ in range(ck.get("epoch",0)*nstep): sched.step()
        for a,attr in [("best_score","bs"),("best_ade","ba"),("best_72h","b7"),
                         ("best_ate","bat"),("best_cte","bc")]:
            if a in ck: setattr(saver,attr,ck[a])
        start=args.resume_epoch or ck.get("epoch",0)+1
        print(f"  Resume ep={start}")

    try:
        model=torch.compile(model,mode="reduce-overhead"); print("  torch.compile: ok")
    except: pass

    ts=time.perf_counter()
    print(f"  Training: {nstep} steps/ep, start ep={start}")
    print("="*72)

    for ep in range(start, args.num_epochs):
        model.train(); sl=0.; t0=time.perf_counter()

        for i, batch in enumerate(trl):
            bl=move(list(batch),dev)
            with autocast(device_type="cuda",enabled=args.use_amp):
                bd=model.get_loss_breakdown(bl,epoch=ep)

            opt.zero_grad()
            scaler.scale(bd["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
            sb=scaler.get_scale(); scaler.step(opt); scaler.update()
            if scaler.get_scale()>=sb: sched.step()
            model.ema_update()
            sl+=bd["total"].item()

            if i % 20 == 0:
                lr=opt.param_groups[0]["lr"]
                sw_r=bd.get("sw_ratio",0.)
                # Stability warnings
                sw_warn=" [!sw<3]" if sw_r<3.0 else ""
                tot_warn=" [!tot>10]" if bd["total"].item()>10. else ""
                print(
                    f"  [{ep:>3}][{i:>3}/{nstep}]"
                    f"  tot={bd['total'].item():.3f}{tot_warn}"
                    f"  fm={bd.get('L_fm',0):.4f}"
                    f"  dpe={bd.get('L_dpe',0):.4f}"
                    f"  disp={bd.get('L_disp',0):.4f}"
                    f"  ep_l={bd.get('L_ep',0):.4f}"
                    f"  E={bd.get('L_easy',0):.3f}"
                    f"  H={bd.get('L_hard',0):.3f}"
                    f"  sw24={bd.get('sw_sw_24h',bd.get('sw_24h',0)):.2f}"
                    f"  sw72={bd.get('sw_sw_72h',bd.get('sw_72h',0)):.2f}"
                    f"  swr={sw_r:.1f}{sw_warn}"
                    f"  dpe/fm={bd.get('lw_lw_dpe_fm',bd.get('lw_dpe_fm',0)):.2f}"
                    f"  efrac={bd.get('easy_frac',0):.2f}"
                    f"  d={bd.get('delta_mean',0):.2f}"
                    f"  lr={lr:.2e}"
                )

        avt=sl/nstep

        # Val loss
        model.eval(); vls=0.
        with torch.no_grad():
            for batch in vl:
                bv=move(list(batch),dev)
                with autocast(device_type="cuda",enabled=args.use_amp):
                    vls+=model.get_loss(bv,epoch=ep).item()
        avv=vls/len(vl); eps=time.perf_counter()-t0
        lr_cur=opt.param_groups[0]["lr"]
        print(f"  Epoch {ep:>3} | train={avt:.4f} val={avv:.4f} | lr={lr_cur:.2e} | {eps:.0f}s")

        # Fast eval (subset)
        rf=evaluate(model,vsub,dev,tag=f"FAST ep{ep}",steps=args.fast_ddim)
        saver.update(rf,model,args.output_dir,ep,opt,sched,avt,avv,tag="fast")

        # Full val every val_freq
        if ep%args.val_freq==0:
            em=getattr(_unwrap(model),"_ema",None)
            rr=evaluate(model,vl,dev,tag=f"RAW ep{ep}",steps=args.full_ddim)
            saver.update(rr,model,args.output_dir,ep,opt,sched,avt,avv,tag="raw")
            if em and ep>=3:
                re=evaluate(model,vl,dev,tag=f"EMA ep{ep}",ema=em,steps=args.full_ddim)
                saver.update(re,model,args.output_dir,ep,opt,sched,avt,avv,tag="ema")

        # Periodic checkpoint
        if ep%10==0 or ep==args.num_epochs-1:
            _save_ckpt(os.path.join(args.output_dir,f"ckpt_ep{ep:03d}.pth"),
                       ep,model,opt,sched,saver,avt,avv)

        if saver.stop: print(f"  Early stop ep={ep}"); break

    th=(time.perf_counter()-ts)/3600.
    print(f"\n  Best: ADE={saver.ba:.1f} 72h={saver.b7:.0f} "
          f"ATE={saver.bat:.1f} CTE={saver.bc:.1f} ({th:.2f}h)")

    # Post-training test
    if args.eval_test_after_train:
        print("\n"+"="*72+"\n  POST-TRAINING TEST\n"+"="*72)
        try: _,tl2=data_loader(args,{"root":args.dataset_root,"type":"test"},test=True)
        except: print("  No test set → using val"); tl2=vl
        for fn,lb in [("best_composite.pth","COMPOSITE"),("best_72h.pth","72H"),
                         ("best_ema.pth","EMA")]:
            pp=os.path.join(args.output_dir,fn)
            if not os.path.exists(pp): continue
            ck=torch.load(pp,map_location=dev)
            _unwrap(model).load_state_dict(ck["model_state_dict"],strict=False)
            em=getattr(_unwrap(model),"_ema",None)
            if em and ck.get("ema_shadow"):
                for k,v in ck["ema_shadow"].items():
                    if k in em.shadow: em.shadow[k].copy_(v.to(dev))
            r=evaluate(model,tl2,dev,tag=f"TEST/{lb}",steps=args.full_ddim)
            print(f"\n  --- {lb} ---")
            for key,ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
                v=r.get(key,float("nan"))
                mk="BEAT!" if np.isfinite(v) and v<ref else f"need {ref:.0f}"
                print(f"    {key:<10}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")
    print("="*72)


if __name__ == "__main__":
    args=get_args()
    np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    main(args)
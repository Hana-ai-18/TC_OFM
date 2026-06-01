"""
scripts/train_flowmatching.py  ── FM v60 Training Entry Point
==============================================================
THAY THẾ scripts/train_flowmatching.py cũ.
Cùng interface, chạy được trên Kaggle với lệnh:

  !python scripts/train_flowmatching.py \
      --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
      --output_dir   /kaggle/working/runs/fm_v60 \
      --batch_size 32 --num_epochs 70 --learning_rate 2e-4 --use_amp
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Subset

# ── Path setup ────────────────────────────────────────────────
# __file__ = project_root/scripts/train_flowmatching.py
# _root    = project_root/          ← chứa Model/ và scripts/
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# ── Imports ───────────────────────────────────────────────────
from Model.flow_matching_model import (
    TCFlowMatching, compute_speed_stats_from_norm,
    _norm_to_deg, _haversine_km
)

try:
    from Model.data.loader_training import data_loader as tc_data_loader
    from Model.data.trajectoriesWithMe_unet_training import seq_collate
    _LOADER_OK = True
except ImportError as e:
    print(f'[WARN] Data loader import failed: {e}')
    _LOADER_OK = False

ST_TRANS = {
    'ADE': 224.4, 'ATE': 213.7, 'CTE': 59.4,
    '12h': 77.5,  '24h': 130.5, '48h': 269.9, '72h': 423.3,
}
FM_V59 = {'ADE': 236.1, 'ATE': 223.6, 'CTE': 74.9, '72h': 471.6}


# ══════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _move(batch_list, device):
    out = []
    for item in batch_list:
        if isinstance(item, torch.Tensor):
            out.append(item.to(device))
        elif isinstance(item, dict):
            out.append({k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in item.items()})
        else:
            out.append(item)
    return out


# ══════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════

def compute_metrics(pred_deg, gt_deg):
    """[N,T,2] lon°,lat° → dict"""
    N, T, _ = pred_deg.shape
    dist = _haversine_km(
        pred_deg.reshape(N*T, 2), gt_deg.reshape(N*T, 2)
    ).reshape(N, T)

    out = {'ADE': float(dist.mean()), 'FDE': float(dist[:,-1].mean())}
    for step, name in {2:'12h', 4:'24h', 8:'48h', 12:'72h'}.items():
        if step-1 < T: out[name] = float(dist[:, step-1].mean())

    # ATE / CTE
    cos_lat = torch.cos(torch.deg2rad(gt_deg[...,1])).clamp(1e-3)
    elon = (pred_deg[...,0]-gt_deg[...,0])*cos_lat*111.
    elat = (pred_deg[...,1]-gt_deg[...,1])*111.
    if T >= 2:
        dlon = (gt_deg[:,1:,0]-gt_deg[:,:-1,0])*cos_lat[:,1:]*111.
        dlat = (gt_deg[:,1:,1]-gt_deg[:,:-1,1])*111.
        mag  = (dlon**2+dlat**2).sqrt().clamp(1e-3)
        ux = torch.cat([dlon[:,:1]/mag[:,:1], dlon/mag], 1)[:,:T]
        uy = torch.cat([dlat[:,:1]/mag[:,:1], dlat/mag], 1)[:,:T]
        out['ATE'] = float((elon*ux+elat*uy).abs().mean())
        out['CTE'] = float((elon*uy-elat*ux).abs().mean())
    else:
        out['ATE'] = out['CTE'] = float(dist.mean())

    # Speed bias
    if T >= 2:
        def _s(t):
            dl=t[:,1:,1]-t[:,:-1,1]; dlo=t[:,1:,0]-t[:,:-1,0]
            cl=torch.cos(torch.deg2rad((t[:,1:,1]+t[:,:-1,1])/2)).clamp(1e-3)
            return float((dl*111)**2+(dlo*111*cl)**2).sqrt().mean() if False \
                   else float(torch.sqrt((dl*111)**2+(dlo*111*cl)**2).mean())
        out['speed_bias'] = _s(pred_deg) - _s(gt_deg)
        out['mean_pred_speed'] = _s(pred_deg)
        out['mean_gt_speed']   = _s(gt_deg)
    return out


def print_metrics(metrics, tag='', elapsed=0.):
    print(f'\n{"="*70}')
    if tag: print(f'  [{tag}  {elapsed:.0f}s]')
    for k in ['ADE','12h','24h','48h','72h']:
        v = metrics.get(k, float('nan'))
        ref = ST_TRANS.get(k, float('nan'))
        ok  = '✅ BEAT' if v < ref else '❌'
        print(f'  {k:>4} = {v:>7.1f} km  {ok} ST-Trans={ref:.1f}  ({v-ref:+.1f})')
    if 'ATE' in metrics:
        print(f'  ATE  = {metrics["ATE"]:>7.1f}  '
              f'{"✅ BEAT" if metrics["ATE"]<ST_TRANS["ATE"] else "❌"}')
        print(f'  CTE  = {metrics["CTE"]:>7.1f}  '
              f'{"✅ BEAT" if metrics["CTE"]<ST_TRANS["CTE"] else "❌"}')
    if 'speed_bias' in metrics:
        sb = metrics['speed_bias']
        print(f'  speed_bias={sb:+.2f} km/6h  '
              f'{"✅" if abs(sb)<10 else "❌"}  '
              f'pred={metrics["mean_pred_speed"]:.1f}  '
              f'gt={metrics["mean_gt_speed"]:.1f}')
    print(f'{"="*70}\n')


# ══════════════════════════════════════════════════════════════
#  Evaluate
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, tag='', num_ens=20, steps=10):
    model.eval(); t0 = time.perf_counter()
    all_p, all_g = [], []
    for bl in loader:
        bl = _move(bl, device)
        try:
            pred_n, _ = model.sample(bl, num_ensemble=num_ens, ddim_steps=steps)
        except Exception as e:
            print(f'  [WARN] sample failed: {e}'); continue
        gt_n = bl[1]
        T = min(pred_n.shape[0], gt_n.shape[0])
        all_p.append(_norm_to_deg(pred_n[:T].permute(1,0,2)).cpu())
        all_g.append(_norm_to_deg(gt_n[:T].permute(1,0,2)).cpu())
    if not all_p: return {}
    m = compute_metrics(torch.cat(all_p,0), torch.cat(all_g,0))
    print_metrics(m, tag, time.perf_counter()-t0)
    return m


# ══════════════════════════════════════════════════════════════
#  Checkpoint
# ══════════════════════════════════════════════════════════════

def save_ckpt(path, epoch, model, opt, sched, metrics):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                'optim_state': opt.state_dict(),
                'sched_state': sched.state_dict() if sched else None,
                'metrics': metrics}, path)


def load_ckpt(path, model, opt=None, sched=None):
    ck = torch.load(path, map_location='cpu')
    model.load_state_dict(ck['model_state'], strict=False)
    if opt and 'optim_state' in ck:
        try: opt.load_state_dict(ck['optim_state'])
        except Exception as e: print(f'  [warn] opt state: {e}')
    if sched and ck.get('sched_state'):
        try: sched.load_state_dict(ck['sched_state'])
        except: pass
    return ck.get('epoch', 0), ck.get('metrics', {})


class BestSaver:
    def __init__(self, d, patience=15, min_ep=20):
        self.d=d; self.patience=patience; self.min_ep=min_ep
        self.best={k:float('inf') for k in ['ADE','72h','ATE','CTE']}
        self.no_imp=0; self.stop=False; os.makedirs(d, exist_ok=True)

    def update(self, m, ep, model, opt, sched):
        imp = False
        for k,fn in [('ADE','best_ade.pth'),('72h','best_72h.pth'),
                     ('ATE','best_ate.pth'),('CTE','best_cte.pth')]:
            v = m.get(k, float('inf'))
            if v < self.best[k]:
                self.best[k]=v
                save_ckpt(os.path.join(self.d,fn), ep, model, opt, sched, m)
                print(f'  [BEST {k}] ep={ep}  '
                      f'ADE={m.get("ADE",0):.1f}  '
                      f'CTE={m.get("CTE",0):.1f}  '
                      f'72h={m.get("72h",0):.0f}')
                imp=True
        if imp: self.no_imp=0
        else:
            self.no_imp+=1
            if ep>=self.min_ep and self.no_imp>=self.patience:
                self.stop=True; print(f'  [EARLY STOP] ep={ep}')
        return imp


# ══════════════════════════════════════════════════════════════
#  Data loaders
# ══════════════════════════════════════════════════════════════

class _Args:
    def __init__(self, **kw):
        for k,v in kw.items(): setattr(self, k, v)


def build_loaders(data_root, batch_size, num_workers):
    if not _LOADER_OK:
        raise RuntimeError('Data loader not available. Check Model/data/.')
    dummy = _Args(obs_len=8, pred_len=12, skip=1, threshold=0.002,
                  min_ped=1, delim=' ', other_modal='gph',
                  batch_size=batch_size, num_workers=num_workers)
    tr_ds, tr_ld = tc_data_loader(dummy, {'root':data_root,'type':'train'},
                                   test=False, batch_size=batch_size)
    vl_ds, vl_ld = tc_data_loader(dummy, {'root':data_root,'type':'val'},
                                   test=True, batch_size=batch_size)
    return tr_ds, tr_ld, vl_ds, vl_ld


# ══════════════════════════════════════════════════════════════
#  Train epoch
# ══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, epoch, device, scaler=None):
    model.train(); tot=0.; terms=defaultdict(float); n=0
    t0=time.perf_counter()

    for i, bl in enumerate(loader):
        bl = _move(bl, device)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast('cuda'):
                bd = model.get_loss_breakdown(bl, epoch=epoch)
            loss = bd['total']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer); scaler.update()
        else:
            bd = model.get_loss_breakdown(bl, epoch=epoch)
            loss = bd['total']
            if torch.is_tensor(loss) and loss.requires_grad:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()

        if not (torch.is_tensor(loss) and torch.isfinite(loss)):
            if i < 5: print(f'  [WARN] non-finite loss at batch {i}')
            continue

        tot += loss.item(); n += 1
        for k,v in bd.items():
            if k != 'total' and not torch.is_tensor(v): terms[k] += float(v)

        if i % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'  [{epoch:>3}][{i:>4}/{len(loader)}]'
                  f' loss={loss.item():.3f}'
                  f' l_kin={bd.get("l_kin",0):.3f}'
                  f' l_logspd={bd.get("l_logspd",0):.3f}'
                  f' l_fm={bd.get("l_fm",0):.3f}'
                  f' l_scr={bd.get("l_scorer",0):.3f}'
                  f' sw={bd.get("sw_ratio",0):.1f}'
                  f' λk={bd.get("lam_kin",0):.3f}'
                  f' λs={bd.get("lam_logspd",0):.3f}'
                  f' dw={bd.get("diff_w_mean",1):.2f}'
                  f' lr={lr:.1e}')

    elapsed = time.perf_counter()-t0
    avg = {k: v/max(n,1) for k,v in terms.items()}
    avg['loss'] = tot/max(n,1); avg['elapsed'] = elapsed
    return avg


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def train(data_root, save_dir, max_epochs=70, batch_size=32, lr=2e-4,
          num_workers=0, seed=42, use_amp=False, resume=None, patience=15):
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n{"="*72}')
    print(f'  FM v60 — Physics-Grounded TC Track Forecasting')
    print(f'  Device: {device}  AMP: {use_amp}')
    print(f'  Baseline → FM_v59={FM_V59["ADE"]:.1f}  ST-Trans={ST_TRANS["ADE"]:.1f}')
    print(f'  Targets  → ADE<170  CTE<52  72h<360')
    print(f'{"="*72}\n')

    tr_ds, tr_ld, vl_ds, vl_ld = build_loaders(data_root, batch_size, num_workers)
    print(f'  train: {len(tr_ds)}  val: {len(vl_ds)}')

    model = TCFlowMatching(
        pred_len=12, obs_len=8, sigma_min=0.02, unet_in_ch=13,
        ctx_noise_scale=0.01, use_ema=True, ema_decay=0.995,
        use_ate_ot=True, ot_epsilon=0.05, cfg_uncond_prob=0.1,
    ).to(device)
    model.init_ema()
    print(f'  Params: {count_params(model):,}')

    # 3 param groups với LR khác nhau
    model_p   = [p for n,p in model.named_parameters()
                 if not n.startswith('criterion.')]
    weight_p  = (list(model.criterion.step_weights.parameters())
               + list(model.criterion.loss_weights.parameters())
               + list(model.criterion.diff_weighter.parameters()))
    scorer_p  = list(model.criterion.scorer.parameters())

    optimizer = AdamW([
        {'params': model_p,  'lr': lr,      'weight_decay': 1e-4},
        {'params': weight_p, 'lr': lr*0.5,  'weight_decay': 0.},
        {'params': scorer_p, 'lr': lr,      'weight_decay': 1e-5},
    ])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=35, T_mult=1, eta_min=1e-5)
    scaler    = torch.amp.GradScaler('cuda') if torch.cuda.is_available() and use_amp else None

    start_ep = 1
    if resume and os.path.exists(resume):
        start_ep, _ = load_ckpt(resume, model, optimizer, scheduler)
        start_ep += 1
        print(f'  Resumed from {resume} (ep {start_ep})')

    saver   = BestSaver(save_dir, patience=patience, min_ep=20)
    history = []

    for epoch in range(start_ep, max_epochs+1):
        phase = ('1-foundation' if epoch<=10 else
                 '2-integration' if epoch<=35 else '3-refinement')
        print(f'\n  ── Epoch {epoch}/{max_epochs}  Phase={phase} ──────────')

        if epoch > 1: model.ema_update()

        ts = train_epoch(model, tr_ld, optimizer, epoch, device, scaler)
        scheduler.step()

        sw = model.criterion.step_weights.stats()
        lw = model.criterion.loss_weights.stats()
        print(f'  train={ts["loss"]:.4f}'
              f'  sw_ratio={sw["sw_ratio"]:.2f}'
              f'  mono={sw["sw_monotonic"]}'
              f'  λ_kin={lw["lam_kin"]:.3f}'
              f'  λ_spd={lw["lam_logspd"]:.3f}'
              f'  λ_curv={lw["lam_curv"]:.3f}')

        # Fast val mỗi epoch
        fast_n  = min(200, len(vl_ds))
        fast_idx = random.sample(range(len(vl_ds)), fast_n)
        fast_ld  = DataLoader(
            Subset(vl_ds, fast_idx),
            batch_size=batch_size, shuffle=False,
            num_workers=0, drop_last=False,
            collate_fn=seq_collate if _LOADER_OK else None)
        fast_m = evaluate(model, fast_ld, device,
                           tag=f'FAST ep{epoch} n={fast_n}',
                           num_ens=10, steps=8)

        # Full val mỗi 3 epochs
        if epoch % 3 == 0 or epoch == max_epochs:
            full_m = evaluate(model, vl_ld, device,
                               tag=f'FULL ep{epoch}',
                               num_ens=20, steps=10)
            saver.update(full_m, epoch, model, optimizer, scheduler)
            history.append({'epoch': epoch, 'train_loss': ts['loss'], **full_m})
        else:
            saver.update(fast_m, epoch, model, optimizer, scheduler)

        if epoch % 10 == 0:
            save_ckpt(os.path.join(save_dir, f'ckpt_ep{epoch:03d}.pth'),
                       epoch, model, optimizer, scheduler, fast_m)

        if saver.stop:
            print(f'  Early stop ep={epoch}'); break

    print(f'\n{"="*72}')
    print(f'  DONE  Best ADE={saver.best["ADE"]:.1f}'
          f'  CTE={saver.best["CTE"]:.1f}  72h={saver.best["72h"]:.1f}')
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f'{"="*72}\n')
    return saver.best['ADE']


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser('FM v60')
    p.add_argument('--data_root',    default='TCND_vn')
    p.add_argument('--save_dir',     default='checkpoints/fm_v60/')
    p.add_argument('--max_epochs',   type=int,   default=70)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=2e-4)
    p.add_argument('--num_workers',  type=int,   default=0)
    p.add_argument('--use_amp',      action='store_true')
    p.add_argument('--resume',       default=None)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--patience',     type=int,   default=15)
    # Aliases
    p.add_argument('--dataset_root', default=None)
    p.add_argument('--output_dir',   default=None)
    p.add_argument('--learning_rate',type=float, default=None)
    p.add_argument('--num_epochs',   type=int,   default=None)
    return p.parse_args()


def _apply_aliases(args):
    if getattr(args,'dataset_root',None) and (not getattr(args,'data_root',None) or args.data_root=='TCND_vn'):
        args.data_root = args.dataset_root
    if getattr(args,'output_dir',None) and (not getattr(args,'save_dir',None) or args.save_dir=='checkpoints/fm_v60/'):
        args.save_dir = args.output_dir
    if getattr(args,'learning_rate',None): args.lr = args.learning_rate
    if getattr(args,'num_epochs',None):    args.max_epochs = args.num_epochs
    return args


if __name__ == '__main__':
    args = parse_args(); args = _apply_aliases(args)
    train(data_root=args.data_root, save_dir=args.save_dir,
          max_epochs=args.max_epochs, batch_size=args.batch_size,
          lr=args.lr, num_workers=args.num_workers, seed=args.seed,
          use_amp=args.use_amp, resume=args.resume, patience=args.patience)

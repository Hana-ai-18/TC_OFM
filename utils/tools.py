"""utils/tools.py - Additional utility functions"""
import os
import time
import torch
import numpy as np
from contextlib import contextmanager


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    raise ValueError('Only 0 or 1 accepted for bool flag')


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def dic2cuda(env_data, device=None):
    """Move environment dict to device safely"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for key in env_data:
        if torch.is_tensor(env_data[key]):
            env_data[key] = env_data[key].to(device)
    
    return env_data


class EarlyStopping:
    """Early stopping - NumPy 2.0 compatible"""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf  # FIX: np.inf
        self.delta = delta
    
    def __call__(self, val_loss, model_state, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model_state, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model_state, path)
            self.counter = 0
    
    def _save_checkpoint(self, val_loss, model_state, path):
        if self.verbose:
            print(f'Loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving checkpoint...')
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(model_state, os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust learning rate by epoch"""
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.g_learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    else:
        lr_adjust = {}
    
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updated learning rate to: {lr}')


class StandardScaler:
    """Normalize and denormalize TC data"""
    def __init__(self):
        # Mean and std from Nature paper
        # LONG, LAT, PRES, WND
        self.mean = torch.tensor([1316.42, 218.44, 979.47, 28.18])
        self.std = torch.tensor([145.29, 88.04, 23.42, 13.26])
    
    def transform(self, data):
        device = data.device
        return (data - self.mean.to(device)) / self.std.to(device)
    
    def inverse_transform(self, data):
        """Denormalize to physical units (km, hPa, m/s)"""
        device = data.device
        return (data * self.std.to(device)) + self.mean.to(device)


def relative_to_abs(rel_traj, start_pos):
    """Convert displacement to absolute coordinates"""
    # rel_traj: [Time, Batch, 2], start_pos: [Batch, 2]
    displacement = torch.cumsum(rel_traj, dim=0)
    abs_traj = displacement + start_pos.unsqueeze(0)
    return abs_traj


@contextmanager
def timeit(msg, should_time=True):
    """Timing context manager"""
    if should_time and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    yield
    if should_time:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f'{msg}: {(time.time() - t0) * 1000.0:.2f} ms')


class dotdict(dict):
    """Dict with dot access"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
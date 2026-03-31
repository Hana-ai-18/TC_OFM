"""
scripts/count_params.py
=======================
Chạy: python scripts/count_params.py
In ra số params của từng sub-module để debug tại sao model phình to.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from Model.flow_matching_model import TCFlowMatching

model = TCFlowMatching(pred_len=12, obs_len=8)
# Không cần .cuda() để đếm params
net = model.net

def count_params(m):
    return sum(p.numel() for p in m.parameters())

total = count_params(model)

rows = [
    ("spatial_enc (FNO3D)",   net.spatial_enc),
    ("enc_1d (Mamba)",        net.enc_1d),
    ("env_enc (Env_net)",     net.env_enc),
    ("transformer decoder",   net.transformer),
    ("bottleneck_pool",       net.bottleneck_pool),
    ("bottleneck_proj",       net.bottleneck_proj),
    ("decoder_proj",          net.decoder_proj),
    ("ctx_fc1",               net.ctx_fc1),
    ("ctx_ln",                net.ctx_ln),
    ("ctx_fc2",               net.ctx_fc2),
    ("time_fc1",              net.time_fc1),
    ("time_fc2",              net.time_fc2),
    ("traj_embed",            net.traj_embed),
    ("out_fc1",               net.out_fc1),
    ("out_fc2",               net.out_fc2),
]

print(f"\n{'Module':<35} {'Params':>12}  {'%':>6}")
print("=" * 58)
for name, m in rows:
    n = count_params(m)
    bar = "█" * int(40 * n / total)
    print(f"{name:<35} {n:>12,}  {100*n/total:>5.1f}%  {bar}")
print("=" * 58)
print(f"{'TOTAL':<35} {total:>12,}  100.0%")
print()

# Đào sâu vào module lớn nhất
print("── Chi tiết enc_1d (Mamba) ──")
for name, m in net.enc_1d.named_children():
    n = count_params(m)
    print(f"  enc_1d.{name:<28} {n:>10,}  {100*n/total:>5.1f}%")

print()
print("── Chi tiết env_enc (Env_net) ──")
for name, m in net.env_enc.named_children():
    n = count_params(m)
    print(f"  env_enc.{name:<27} {n:>10,}  {100*n/total:>5.1f}%")

print()
print("── Chi tiết spatial_enc (FNO3D) ──")
for name, m in net.spatial_enc.named_children():
    n = count_params(m)
    print(f"  spatial_enc.{name:<23} {n:>10,}  {100*n/total:>5.1f}%")

print()
print("── Chi tiết transformer decoder ──")
for name, m in net.transformer.named_children():
    n = count_params(m)
    print(f"  transformer.{name:<23} {n:>10,}  {100*n/total:>5.1f}%")
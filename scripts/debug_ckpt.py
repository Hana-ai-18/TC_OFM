"""
Script debug: in ra tất cả keys của checkpoint để xác định loại model.
Chạy trước evaluate để biết checkpoint structure.

Usage:
    python debug_ckpt.py --ckpt /kaggle/input/datasets/nguyennhaquan/tc-model-v47/BEST_V47.pth
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()

    print(f"Loading: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # Top-level keys
    if isinstance(ckpt, dict):
        print(f"\nTop-level keys ({len(ckpt)}):")
        for k in sorted(ckpt.keys()):
            v = ckpt[k]
            if torch.is_tensor(v):
                print(f"  {k}: tensor {tuple(v.shape)}")
            elif isinstance(v, dict):
                print(f"  {k}: dict ({len(v)} keys)")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")

        # Find state dict
        state = None
        for key in ["model_state_dict", "model_state", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                print(f"\nState dict found under key: '{key}' ({len(state)} keys)")
                break
        if state is None and any(torch.is_tensor(v) for v in ckpt.values()
                                 if isinstance(v, torch.Tensor)):
            state = ckpt
            print(f"\nState dict is the top-level dict ({len(state)} keys)")
    else:
        state = ckpt
        print(f"Checkpoint is raw state dict ({len(state)} keys)")

    if state:
        # Print first 30 keys grouped by prefix
        from collections import defaultdict
        prefixes = defaultdict(int)
        for k in state.keys():
            prefix = k.split(".")[0]
            prefixes[prefix] += 1

        print(f"\nKey prefixes:")
        for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
            print(f"  {prefix}.*  ({count} keys)")

        print(f"\nFirst 20 keys:")
        for k in list(state.keys())[:20]:
            v = state[k]
            shape = tuple(v.shape) if torch.is_tensor(v) else type(v).__name__
            print(f"  {k}: {shape}")

        # Detection
        keys = set(state.keys())
        has_ctx_enc  = any(k.startswith("ctx_enc.")  for k in keys)
        has_corr_net = any(k.startswith("corr_net.") for k in keys)
        has_gate     = any(k.startswith("gate.")     for k in keys)
        has_net      = any(k.startswith("net.")      for k in keys)
        has_sttrans  = any(k.startswith("sttrans.")  for k in keys)

        print(f"\nDetection signals:")
        print(f"  ctx_enc.*  : {has_ctx_enc}")
        print(f"  corr_net.* : {has_corr_net}")
        print(f"  gate.*     : {has_gate}")
        print(f"  net.*      : {has_net}")
        print(f"  sttrans.*  : {has_sttrans}")

        if sum([has_ctx_enc, has_corr_net, has_gate]) >= 2:
            print("\n  → Detected: FMResidualCorrector")
        elif has_net:
            print("\n  → Detected: TCFlowMatching")
        else:
            print("\n  → Detected: UNKNOWN — need manual inspection")

if __name__ == "__main__":
    main()
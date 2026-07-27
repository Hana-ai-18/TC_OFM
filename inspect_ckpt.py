import torch, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/input/datasets/gmnguynhng/new-checkpoint/best_model_st-trans_seed0.pth"

ck = torch.load(path, map_location="cpu")

print("=" * 70)
print(f"Checkpoint: {path}")
print("=" * 70)

print("\n[Top-level keys]")
print(list(ck.keys()))

for k in ["model_type", "paper", "seed", "epoch", "best_ade"]:
    if k in ck:
        print(f"\n[{k}] = {ck[k]}")

print("\n[model_cfg]")
print(ck.get("model_cfg"))

sd_key = "model_state" if "model_state" in ck else ("model" if "model" in ck else None)
print(f"\n[state_dict key used] = {sd_key!r}")
if sd_key:
    sd = ck[sd_key]
    print(f"[state_dict] {len(sd)} tensors")
    print("\nFirst 20 param names (shape):")
    for i, (name, tensor) in enumerate(sd.items()):
        if i >= 20:
            print("  ...")
            break
        print(f"  {name}: {tuple(tensor.shape)}")

    print("\nLast 10 param names (shape):")
    items = list(sd.items())
    for name, tensor in items[-10:]:
        print(f"  {name}: {tuple(tensor.shape)}")

print("\n" + "=" * 70)
print("Done. Paste this ENTIRE output back to Claude.")
print("=" * 70)
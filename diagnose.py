"""
diagnose.py — Chạy script này TRƯỚC để xác định root cause.
Kiểm tra: img_obs zeros, u500 zeros, env_data keys, coord filter rate.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np

def run_diagnosis(train_loader, val_loader, device, n_batches=5):
    print("\n" + "="*70)
    print("  DIAGNOSIS RUN")
    print("="*70)

    img_nonzero_ratios = []
    u500_nonzero_ratios = []
    v500_nonzero_ratios = []
    gph500_vals = []
    lon_ranges = []
    lat_ranges = []

    for bi, batch in enumerate(train_loader):
        if bi >= n_batches:
            break
        bl = [x.to(device) if torch.is_tensor(x) else x for x in batch]

        # --- img_obs check ---
        img_obs = bl[11]  # [B, C, T, H, W] or [B, T, H, W, C]
        nonzero = (img_obs.abs() > 1e-6).float().mean().item()
        img_nonzero_ratios.append(nonzero)

        # --- env_data check ---
        env_data = bl[13]
        if isinstance(env_data, dict):
            for key in ["u500_mean_n", "v500_mean_n", "u500_center", "v500_center"]:
                if key in env_data:
                    v = env_data[key]
                    nz = (v.abs() > 1e-6).float().mean().item()
                    if key == "u500_mean_n":
                        u500_nonzero_ratios.append(nz)
                    elif key == "v500_mean_n":
                        v500_nonzero_ratios.append(nz)
            if "gph500_mean" in env_data:
                gph500_vals.append(env_data["gph500_mean"].mean().item())

        # --- trajectory range check ---
        obs_traj = bl[0]  # [T, B, 2]
        lon_norm = obs_traj[..., 0]
        lat_norm = obs_traj[..., 1]
        lon_deg = (lon_norm * 50.0 + 1800.0) / 10.0
        lat_deg = (lat_norm * 50.0) / 10.0
        lon_ranges.append((lon_deg.min().item(), lon_deg.max().item()))
        lat_ranges.append((lat_deg.min().item(), lat_deg.max().item()))

    print(f"\n[img_obs] nonzero ratio: {np.mean(img_nonzero_ratios):.4f}")
    if np.mean(img_nonzero_ratios) < 0.05:
        print("  ⛔ CRITICAL: img_obs mostly ZERO → Data3d không load được!")
        print("     FNO3D (61% params) đang encode toàn zeros → wasted")
    else:
        print("  ✅ img_obs OK")

    print(f"\n[u500_mean] nonzero ratio: {np.mean(u500_nonzero_ratios) if u500_nonzero_ratios else 'NOT FOUND':.4f}" 
          if u500_nonzero_ratios else "\n[u500_mean] NOT FOUND IN ENV_DATA")
    if not u500_nonzero_ratios or np.mean(u500_nonzero_ratios) < 0.1:
        print("  ⛔ CRITICAL: u500 steering flow = 0 → model mù với wind direction!")
    else:
        print("  ✅ u500 OK")

    print(f"\n[v500_mean] nonzero ratio: {np.mean(v500_nonzero_ratios) if v500_nonzero_ratios else 'NOT FOUND'}")

    if gph500_vals:
        print(f"\n[gph500] mean={np.mean(gph500_vals):.4f} "
              f"(expected: -5 to 5 if pre-normed, or 25-95 if raw dam)")

    print(f"\n[trajectory lon] range: "
          f"{min(r[0] for r in lon_ranges):.1f}° - {max(r[1] for r in lon_ranges):.1f}°E")
    print(f"[trajectory lat] range: "
          f"{min(r[0] for r in lat_ranges):.1f}° - {max(r[1] for r in lat_ranges):.1f}°N")

    print("\n" + "="*70)
    print("  ROOT CAUSE SUMMARY")
    print("="*70)
    causes = []
    if np.mean(img_nonzero_ratios) < 0.05:
        causes.append("1. Data3d (img_obs) = zeros → FNO3D không học được gì")
    if not u500_nonzero_ratios or np.mean(u500_nonzero_ratios) < 0.1:
        causes.append("2. u500/v500 steering flow = zeros → không biết gió thổi hướng nào")
    if not causes:
        print("  Data loading OK. Vấn đề nằm ở loss/architecture.")
    else:
        for c in causes:
            print(f"  ⛔ {c}")
    print()
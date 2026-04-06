"""
scripts/check_env_npy.py
========================
Kiểm tra nội dung file .npy trong Env_Data để xác định:
  1. Keys có trong mỗi file (có u500/v500 raw không?)
  2. Value ranges của từng key
  3. Tại sao u500_mean = 0 trong training

Usage:
  python scripts/check_env_npy.py --env_path /path/to/Env_Data
  python scripts/check_env_npy.py --env_path /kaggle/input/.../TCND_vn/Env_data

  # Hoặc tự tìm:
  python scripts/check_env_npy.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np


# ── Tìm Env_Data directory ────────────────────────────────────────────────────

SEARCH_ROOTS = [
    "/kaggle/input",
    "/kaggle/working",
    "/content",
    ".",
    "..",
    "../..",
]

def find_env_data(hint: str | None = None) -> str | None:
    if hint and os.path.isdir(hint):
        return hint
    for root in SEARCH_ROOTS:
        for dirpath, dirnames, _ in os.walk(root):
            for d in dirnames:
                if d.lower() in ("env_data", "env_dat", "envdata"):
                    full = os.path.join(dirpath, d)
                    # Phải chứa ít nhất 1 subdirectory (year)
                    try:
                        if any(os.path.isdir(os.path.join(full, x))
                               for x in os.listdir(full)):
                            return full
                    except Exception:
                        pass
    return None


def collect_npy_samples(env_path: str, max_files: int = 30) -> list[str]:
    """Thu thập tối đa max_files file .npy từ Env_Data."""
    samples = []
    for root, dirs, files in os.walk(env_path):
        for f in sorted(files):
            if f.endswith(".npy"):
                samples.append(os.path.join(root, f))
                if len(samples) >= max_files:
                    return samples
    return samples


def inspect_npy(path: str) -> dict:
    """Load 1 file .npy và trả về thông tin keys + values."""
    try:
        obj = np.load(path, allow_pickle=True)
        # Thử .item() nếu là 0-d array chứa dict
        if hasattr(obj, "item"):
            obj = obj.item()
        if not isinstance(obj, dict):
            return {"error": f"Not a dict, type={type(obj)}", "path": path}

        info = {"path": path, "keys": list(obj.keys()), "values": {}}
        for k, v in obj.items():
            try:
                if isinstance(v, (int, float, bool)):
                    info["values"][k] = f"scalar={v}"
                elif isinstance(v, np.ndarray):
                    info["values"][k] = (f"ndarray shape={v.shape} "
                                         f"dtype={v.dtype} "
                                         f"min={v.min():.4f} max={v.max():.4f}")
                elif isinstance(v, (list, tuple)):
                    arr = np.array(v)
                    info["values"][k] = (f"list len={len(v)} "
                                         f"min={arr.min():.4f} max={arr.max():.4f}")
                else:
                    info["values"][k] = str(v)[:80]
            except Exception as e:
                info["values"][k] = f"<error: {e}>"
        return info
    except Exception as e:
        return {"error": str(e), "path": path}


def aggregate_key_stats(samples: list[str]) -> dict:
    """
    Chạy qua tất cả samples, tổng hợp:
      - Tần suất mỗi key xuất hiện
      - Min/max value trung bình
    """
    from collections import defaultdict
    key_count   = defaultdict(int)
    key_vals    = defaultdict(list)
    total_files = 0
    errors      = 0

    for path in samples:
        info = inspect_npy(path)
        if "error" in info:
            errors += 1
            continue
        total_files += 1
        for k in info["keys"]:
            key_count[k] += 1
        # Thu thập raw values để tính stats
        try:
            obj = np.load(path, allow_pickle=True).item()
            for k, v in obj.items():
                try:
                    if isinstance(v, (int, float)):
                        key_vals[k].append(float(v))
                    elif isinstance(v, (list, np.ndarray)):
                        arr = np.array(v, dtype=float).flatten()
                        key_vals[k].extend(arr.tolist())
                except Exception:
                    pass
        except Exception:
            pass

    return {
        "total_files": total_files,
        "errors": errors,
        "key_freq": dict(sorted(key_count.items(), key=lambda x: -x[1])),
        "key_stats": {
            k: {
                "count": len(v),
                "mean":  float(np.mean(v)) if v else 0,
                "std":   float(np.std(v))  if v else 0,
                "min":   float(np.min(v))  if v else 0,
                "max":   float(np.max(v))  if v else 0,
            }
            for k, v in key_vals.items()
        }
    }


# ── U500 key finder ───────────────────────────────────────────────────────────

U500_CANDIDATE_KEYS = [
    # Raw wind/geopotential
    "u500", "v500", "u_500", "v_500",
    "u500_mean", "v500_mean", "u500_center", "v500_center",
    "u500_mean_n", "v500_mean_n",
    "u500_mean_raw", "v500_mean_raw",
    "u500_raw_mean", "v500_raw_mean",
    "u500_raw", "v500_raw",
    # GPH500
    "gph500_mean", "gph500_center",
    "gph500_mean_n", "gph500_center_n",
    "gph500_mean_raw", "gph500_center_raw",
    # Move velocity
    "move_velocity", "move_vel",
    # History
    "history_direction12", "history_direction24",
    "history_inte_change24",
]

def check_u500_keys(stats: dict) -> None:
    """Print diagnosis về u500/v500."""
    print("\n" + "=" * 60)
    print("  DIAGNOSIS: u500 / v500 keys")
    print("=" * 60)

    key_freq = stats["key_freq"]
    key_st   = stats["key_stats"]
    total    = stats["total_files"]

    found_u500 = False
    for k in U500_CANDIDATE_KEYS:
        if k in key_freq:
            freq = key_freq[k]
            pct  = freq / total * 100 if total > 0 else 0
            st   = key_st.get(k, {})
            mn   = st.get("mean", 0)
            mx   = st.get("max", 0)
            print(f"  {'[FOUND]':8} {k:30} freq={freq}/{total} ({pct:.0f}%)  "
                  f"mean={mn:.4f}  max={mx:.4f}")
            if "u500" in k or "v500" in k:
                found_u500 = True

    if not found_u500:
        print("\n  ❌ KHÔNG TÌM THẤY u500/v500 key nào!")
        print("  Các keys thực sự có trong file:")
        for k, freq in list(key_freq.items())[:20]:
            st = key_st.get(k, {})
            print(f"    {k:35} freq={freq}/{total}  "
                  f"mean={st.get('mean',0):.4f}  max={st.get('max',0):.4f}")
    else:
        print("\n  ✅ Tìm thấy u500/v500. Kiểm tra xem code có đọc đúng key không.")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_path", default=None, type=str,
                   help="Path tới Env_Data directory")
    p.add_argument("--max_files", default=50, type=int,
                   help="Số file .npy tối đa để inspect")
    p.add_argument("--show_sample", default=3, type=int,
                   help="Số file mẫu in chi tiết")
    args = p.parse_args()

    # Tìm Env_Data
    env_path = find_env_data(args.env_path)
    if env_path is None:
        print("❌ Không tìm thấy Env_Data directory.")
        print("   Chạy lại với: --env_path /path/to/Env_Data")
        sys.exit(1)

    print(f"✅ Env_Data found: {env_path}")

    # List cấu trúc thư mục
    print("\n=== Directory structure (top 3 levels) ===")
    level = 0
    for root, dirs, files in os.walk(env_path):
        depth = root.replace(env_path, "").count(os.sep)
        if depth > 2:
            continue
        indent = "  " * depth
        npy_count = sum(1 for f in files if f.endswith(".npy"))
        print(f"{indent}{os.path.basename(root)}/  ({npy_count} .npy files)")
        if depth == 2 and dirs:
            print(f"{indent}  ...")
            break

    # Collect samples
    samples = collect_npy_samples(env_path, max_files=args.max_files)
    print(f"\n=== Collected {len(samples)} .npy files for inspection ===")

    if not samples:
        print("❌ Không tìm thấy file .npy nào!")
        sys.exit(1)

    # In chi tiết vài file
    print(f"\n=== Chi tiết {min(args.show_sample, len(samples))} file mẫu ===")
    for path in samples[:args.show_sample]:
        info = inspect_npy(path)
        rel  = os.path.relpath(path, env_path)
        print(f"\n--- {rel} ---")
        if "error" in info:
            print(f"  ERROR: {info['error']}")
            continue
        print(f"  Keys ({len(info['keys'])}): {info['keys']}")
        for k, v in info["values"].items():
            print(f"    {k:30}: {v}")

    # Aggregate stats
    print(f"\n=== Aggregate stats trên {len(samples)} files ===")
    stats = aggregate_key_stats(samples)
    print(f"  Total valid files: {stats['total_files']}")
    print(f"  Errors:            {stats['errors']}")

    print(f"\n  All keys found (sorted by frequency):")
    for k, freq in stats["key_freq"].items():
        pct = freq / stats["total_files"] * 100 if stats["total_files"] > 0 else 0
        st  = stats["key_stats"].get(k, {})
        print(f"    {k:35} {freq:4}/{stats['total_files']} ({pct:5.1f}%)  "
              f"mean={st.get('mean',0):9.4f}  "
              f"min={st.get('min',0):9.4f}  "
              f"max={st.get('max',0):9.4f}")

    # Specific u500 diagnosis
    check_u500_keys(stats)

    # Kết luận và hướng fix
    print("=" * 60)
    print("  KẾT LUẬN & HƯỚNG FIX")
    print("=" * 60)
    key_freq = stats["key_freq"]

    has_u500_n   = "u500_mean_n"   in key_freq or "u500_center_n" in key_freq
    has_u500_raw = "u500_mean_raw" in key_freq or "u500_raw_mean" in key_freq
    has_u500_plain = "u500_mean"   in key_freq

    if has_u500_n:
        u_key = "u500_mean_n"
        print(f"  ✅ File có key '{u_key}' (pre-normalized)")
        st = stats["key_stats"].get(u_key, {})
        mn, mx = st.get("min",0), st.get("max",0)
        print(f"     Range: {mn:.4f} → {mx:.4f}")
        if -5 < mn < mx < 5:
            print("     → Giá trị đã normalized. Dùng trực tiếp, clip [-5,5].")
            print("     → Trong _NPY_U500_KEY_REMAP, key này map sang 'u500_mean'.")
            print("     → Trong build_env_features: đọc 'u500_raw_mean' nhưng")
            print("       file không có key này → trả 0. CẦN ĐỌC 'u500_mean' TRỰC TIẾP.")
        else:
            print("     → Giá trị lớn, cần normalize thêm.")
    elif has_u500_plain:
        u_key = "u500_mean"
        st = stats["key_stats"].get(u_key, {})
        mn, mx = st.get("min",0), st.get("max",0)
        print(f"  ⚠️  File có key '{u_key}': range {mn:.4f} → {mx:.4f}")
        if 0 <= mn <= mx <= 1.01:
            print("     → Đây là BOOLEAN FLAG (availability). Không phải steering flow!")
            print("     → Cần tìm key khác chứa actual wind value.")
        else:
            print(f"     → Có thể là actual value. Mean={st.get('mean',0):.2f}")
    elif has_u500_raw:
        print("  ✅ File có key 'u500_mean_raw' / 'u500_raw_mean'")
        print("     → Code đang đọc đúng key, nhưng có thể path bị sai.")
    else:
        print("  ❌ Không có key u500 nào! Data không chứa steering flow.")
        print("     → Cần dùng d3d_u500_mean_raw từ CSV thay vì .npy")

    print()
    print("  Để fix trong trajectoriesWithMe_unet_training.py:")
    print("  Xem key thực tế trên → chỉnh _NPY_U500_KEY_REMAP và")
    print("  build_env_features_one_step() cho đúng key tên.")
    print("=" * 60)


if __name__ == "__main__":
    main()
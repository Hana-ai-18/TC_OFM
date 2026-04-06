"""
scripts/generate_synthetic_tracks.py  ── v1
=============================================
Generate synthetic TC tracks từ all_storms_final.csv bằng 4 strategies:

  STRATEGY-1  Position perturbation:
              Shift toàn bộ track ±Δlat, ±Δlon (Gaussian, σ=0.3°).
              Giữ nguyên velocity pattern, chỉ shift không gian.
              → Giúp model học generalize về spatial location.

  STRATEGY-2  Velocity perturbation:
              Thêm noise Gaussian vào từng step displacement (σ=0.1°/step).
              Cumsum để tạo track mới có hướng hơi khác.
              → Tăng diversity về trajectory shape.

  STRATEGY-3  Intensity perturbation:
              Pertub pres ±Δp (σ=5 hPa) và wnd ±Δw (σ=5 kt).
              Clip về physical range. Re-normalize.
              → Tăng intensity diversity, model học cả TC yếu và mạnh.

  STRATEGY-4  Time-mirror augmentation:
              Flip lon (x → -x quanh trung tâm basin) để tạo mirror track.
              Tương đương với lon_flip_aug trong flow_matching_model nhưng
              ở cấp data để env features cũng được flip.
              Chỉ apply cho storms trong WP basin (lon 110-180°E).

Output:
  - all_storms_synthetic.csv: file CSV cùng format với all_storms_final.csv
  - Chỉ augment TRAIN set (year < 2015) để tránh data leakage
  - N_AUGMENT = 5 per storm → ~292 × 5 × avg_seqs ≈ 40K+ sequences thêm

Usage:
  python scripts/generate_synthetic_tracks.py \
      --input all_storms_final.csv \
      --output all_storms_synthetic.csv \
      --n_augment 5 \
      --seed 42

  # Sau đó trong data_loader: load cả hai file, concat vào train set.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

# ── Physical bounds (WP basin) ───────────────────────────────────────────────
LON_MIN, LON_MAX = 100.0, 190.0
LAT_MIN, LAT_MAX = 0.0, 50.0
PRES_MIN, PRES_MAX = 870.0, 1015.0
WND_MIN, WND_MAX = 10.0, 185.0    # kt

# Normalization (phải khớp với model)
LON_NORM = lambda lon: (lon * 10.0 - 1800.0) / 50.0
LAT_NORM = lambda lat: (lat * 10.0) / 50.0
PRES_NORM = lambda p: (p - 960.0) / 50.0
WND_NORM = lambda w: (w - 40.0) / 25.0

# ── Sentinel / filter thresholds ─────────────────────────────────────────────
# Chỉ augment storms trong WP basin core; bỏ outliers
LON_VALID_MIN = 100.0
LON_VALID_MAX = 185.0
LAT_VALID_MAX = 50.0
MIN_STEPS_FOR_SEQ = 20  # seq_len = obs_len(8) + pred_len(12)


def _clip_coords(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lon"] = df["lon"].clip(LON_MIN, LON_MAX)
    df["lat"] = df["lat"].clip(LAT_MIN, LAT_MAX)
    df["pres"] = df["pres"].clip(PRES_MIN, PRES_MAX)
    df["wnd"] = df["wnd"].clip(WND_MIN, WND_MAX)
    df["lon_norm"] = LON_NORM(df["lon"])
    df["lat_norm"] = LAT_NORM(df["lat"])
    df["pres_norm"] = PRES_NORM(df["pres"])
    df["wnd_norm"] = WND_NORM(df["wnd"])
    return df


def _recompute_env_spatial(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute spatial env features (location onehot, bearing, dist)
    after coordinate shift. move_velocity và history features giữ nguyên
    vì chúng depend on relative displacement, không absolute position.
    """
    df = df.copy()

    # loc_lon: 10 bins [100,125]°E
    for i in range(10):
        lo = 100.0 + i * 2.5
        hi = lo + 2.5
        df[f"env_loc_lon_{i}"] = ((df["lon"] >= lo) & (df["lon"] < hi)).astype(float)

    # loc_lat: 8 bins [5,25]°N
    for i in range(8):
        lo = 5.0 + i * 2.5
        hi = lo + 2.5
        df[f"env_loc_lat_{i}"] = ((df["lat"] >= lo) & (df["lat"] < hi)).astype(float)

    # bearing to SCS center (112.5°E, 12.5°N): 16 bins of 22.5°
    c_lon, c_lat = 112.5, 12.5
    mid_lat_rad = np.deg2rad((df["lat"] + c_lat) / 2.0)
    dx = (c_lon - df["lon"]) * np.cos(mid_lat_rad)
    dy = c_lat - df["lat"]
    bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
    b_idx = ((bearing + 11.25) / 22.5).astype(int) % 16
    for i in range(16):
        df[f"env_bearing_{i}"] = (b_idx == i).astype(float)

    # dist to SCS boundary (5 bins)
    in_scs = (
        (df["lon"] >= 100) & (df["lon"] <= 125) &
        (df["lat"] >= 5) & (df["lat"] <= 20)
    )
    d_min = pd.Series(np.inf, index=df.index)
    cos_lat = np.cos(np.deg2rad(df["lat"]))
    d_min = np.minimum(d_min, (df["lon"] - 100).clip(lower=0) * cos_lat * 111.0)
    d_min = np.minimum(d_min, (125 - df["lon"]).clip(lower=0) * cos_lat * 111.0)
    d_min = np.minimum(d_min, (df["lat"] - 5).clip(lower=0) * 111.0)
    d_min = np.minimum(d_min, (20 - df["lat"]).clip(lower=0) * 111.0)
    r = d_min / 3100.0  # SCS diagonal ~3100 km
    df["env_dist_0"] = (~in_scs).astype(float)
    df["env_dist_1"] = (in_scs & (r >= 0.30)).astype(float)
    df["env_dist_2"] = (in_scs & (r >= 0.15) & (r < 0.30)).astype(float)
    df["env_dist_3"] = (in_scs & (r >= 0.05) & (r < 0.15)).astype(float)
    df["env_dist_4"] = (in_scs & (r < 0.05)).astype(float)

    return df


# ── Strategy 1: Position shift ────────────────────────────────────────────────

def augment_position_shift(
    storm_df: pd.DataFrame,
    rng: np.random.Generator,
    sigma_lon: float = 0.3,
    sigma_lat: float = 0.3,
    aug_idx: int = 0,
) -> pd.DataFrame:
    aug = storm_df.copy()
    delta_lon = rng.normal(0, sigma_lon)
    delta_lat = rng.normal(0, sigma_lat)
    # Đảm bảo sau shift vẫn trong basin
    lon_min_storm = aug["lon"].min()
    lon_max_storm = aug["lon"].max()
    lat_min_storm = aug["lat"].min()
    lat_max_storm = aug["lat"].max()
    delta_lon = np.clip(delta_lon, LON_VALID_MIN - lon_min_storm, LON_VALID_MAX - lon_max_storm)
    delta_lat = np.clip(delta_lat, LAT_MIN - lat_min_storm, LAT_VALID_MAX - lat_max_storm)

    aug["lon"] = aug["lon"] + delta_lon
    aug["lat"] = aug["lat"] + delta_lat
    aug = _clip_coords(aug)
    aug = _recompute_env_spatial(aug)
    aug["storm_name"] = aug["storm_name"].astype(str) + f"_s1a{aug_idx}"
    return aug


# ── Strategy 2: Velocity perturbation ─────────────────────────────────────────

def augment_velocity_perturb(
    storm_df: pd.DataFrame,
    rng: np.random.Generator,
    sigma_dlon: float = 0.08,
    sigma_dlat: float = 0.05,
    aug_idx: int = 0,
) -> pd.DataFrame:
    aug = storm_df.copy().reset_index(drop=True)
    n = len(aug)
    if n < 3:
        return aug

    # Tính displacement gốc
    orig_lon = aug["lon"].values.copy()
    orig_lat = aug["lat"].values.copy()

    # Noise trên displacement (cumulative)
    noise_lon = rng.normal(0, sigma_dlon, n)
    noise_lat = rng.normal(0, sigma_dlat, n)
    # Noise nhỏ dần ở cuối để tránh diverge quá
    decay = np.linspace(1.0, 0.3, n)
    noise_lon *= decay
    noise_lat *= decay

    # Áp noise vào displacement rồi cumsum
    dlon = np.diff(orig_lon, prepend=orig_lon[0])
    dlat = np.diff(orig_lat, prepend=orig_lat[0])
    new_dlon = dlon + noise_lon
    new_dlat = dlat + noise_lat
    new_lon = orig_lon[0] + np.cumsum(new_dlon)
    new_lon[0] = orig_lon[0]
    new_lat = orig_lat[0] + np.cumsum(new_dlat)
    new_lat[0] = orig_lat[0]

    aug["lon"] = new_lon
    aug["lat"] = new_lat
    aug = _clip_coords(aug)
    aug = _recompute_env_spatial(aug)
    aug["storm_name"] = aug["storm_name"].astype(str) + f"_s2a{aug_idx}"
    return aug


# ── Strategy 3: Intensity perturbation ────────────────────────────────────────

def augment_intensity_perturb(
    storm_df: pd.DataFrame,
    rng: np.random.Generator,
    sigma_pres: float = 5.0,
    sigma_wnd: float = 5.0,
    aug_idx: int = 0,
) -> pd.DataFrame:
    aug = storm_df.copy()
    delta_pres = rng.normal(0, sigma_pres)
    delta_wnd = rng.normal(0, sigma_wnd)
    aug["pres"] = (aug["pres"] + delta_pres).clip(PRES_MIN, PRES_MAX)
    aug["wnd"] = (aug["wnd"] + delta_wnd).clip(WND_MIN, WND_MAX)
    # env_wind cũng update
    wind_ms = aug["wnd"] * 0.5144
    aug["env_wind"] = wind_ms / 150.0

    # intensity_class onehot
    thresholds_ms = [17.2, 32.7, 41.5, 51.5, 65.0]
    for i in range(6):
        aug[f"env_intensity_class_{i}"] = 0.0
    for idx_row in aug.index:
        wms = wind_ms[idx_row]
        cls_idx = sum(wms >= t for t in thresholds_ms)
        aug.loc[idx_row, f"env_intensity_class_{min(cls_idx,5)}"] = 1.0

    aug = _clip_coords(aug)
    aug["storm_name"] = aug["storm_name"].astype(str) + f"_s3a{aug_idx}"
    return aug


# ── Strategy 4: Lon-flip (hemispheric mirror) ─────────────────────────────────

def augment_lon_flip(
    storm_df: pd.DataFrame,
    aug_idx: int = 0,
) -> pd.DataFrame:
    """Mirror track quanh lon_center = (lon_min + lon_max) / 2."""
    aug = storm_df.copy()
    lon_center = (aug["lon"].min() + aug["lon"].max()) / 2.0
    aug["lon"] = 2 * lon_center - aug["lon"]
    aug = _clip_coords(aug)
    aug = _recompute_env_spatial(aug)
    # Flip bearing (16-bin circular): bearing_i ↔ bearing_(16-i)%16
    bear_cols = [f"env_bearing_{i}" for i in range(16)]
    orig_bear = aug[bear_cols].values.copy()
    for i in range(16):
        aug[f"env_bearing_{i}"] = orig_bear[:, (16 - i) % 16]
    # Flip dir12/dir24 (8 directional bins): bin i ↔ bin (8-i)%8
    for prefix in ["env_dir12_", "env_dir24_"]:
        cols = [f"{prefix}{i}" for i in range(8)]
        orig = aug[cols].values.copy()
        for i in range(8):
            aug[f"{prefix}{i}"] = orig[:, (8 - i) % 8]
    aug["storm_name"] = aug["storm_name"].astype(str) + f"_s4a{aug_idx}"
    return aug


# ── Main generation loop ──────────────────────────────────────────────────────

def generate_synthetic(
    df: pd.DataFrame,
    n_augment: int = 5,
    seed: int = 42,
    train_year_max: int = 2015,
) -> pd.DataFrame:
    """
    Generate synthetic storms.

    Args:
        df: all_storms_final.csv đã load
        n_augment: số augmented versions per storm
        seed: random seed
        train_year_max: chỉ augment storms có year < train_year_max

    Returns:
        DataFrame với chỉ synthetic rows (không gồm originals)
    """
    rng = np.random.default_rng(seed)

    # Lọc train storms và có đủ steps
    train_df = df[df["year"] < train_year_max].copy()
    storm_groups = train_df.groupby(["year", "storm_name"])

    # Chỉ augment storms đủ dài
    eligible = [
        (yr, nm, grp)
        for (yr, nm), grp in storm_groups
        if len(grp) >= MIN_STEPS_FOR_SEQ
        and grp["lon"].min() >= LON_VALID_MIN - 5
        and grp["lat"].max() <= LAT_VALID_MAX
    ]
    print(f"Eligible storms for augmentation: {len(eligible)}")

    all_synthetic: List[pd.DataFrame] = []

    # Strategy distribution per augmentation:
    # [pos_shift, vel_perturb, intensity, flip, combined(pos+vel)]
    strategies = ["pos", "vel", "inten", "flip", "pos+vel"]

    for storm_idx, (yr, nm, grp) in enumerate(eligible):
        grp = grp.sort_values("step_i").reset_index(drop=True)

        for aug_i in range(n_augment):
            strategy = strategies[aug_i % len(strategies)]
            aug_rng = np.random.default_rng(seed + storm_idx * 1000 + aug_i)

            try:
                if strategy == "pos":
                    syn = augment_position_shift(grp, aug_rng, aug_idx=aug_i)
                elif strategy == "vel":
                    syn = augment_velocity_perturb(grp, aug_rng, aug_idx=aug_i)
                elif strategy == "inten":
                    syn = augment_intensity_perturb(grp, aug_rng, aug_idx=aug_i)
                elif strategy == "flip":
                    # Chỉ flip nếu storm ở WP core (không quá gần boundary)
                    if grp["lon"].min() > 110 and grp["lon"].max() < 180:
                        syn = augment_lon_flip(grp, aug_idx=aug_i)
                    else:
                        syn = augment_position_shift(grp, aug_rng, aug_idx=aug_i)
                elif strategy == "pos+vel":
                    tmp = augment_position_shift(grp, aug_rng, sigma_lon=0.2, sigma_lat=0.2, aug_idx=aug_i)
                    syn = augment_velocity_perturb(tmp, aug_rng, sigma_dlon=0.05, sigma_dlat=0.03, aug_idx=aug_i)
                else:
                    syn = grp.copy()

                all_synthetic.append(syn)
            except Exception as e:
                print(f"  Warning: failed aug {strategy} for {yr}/{nm}: {e}")
                continue

        if (storm_idx + 1) % 50 == 0:
            print(f"  Processed {storm_idx + 1}/{len(eligible)} storms, "
                  f"{len(all_synthetic)} synthetic tracks generated")

    result = pd.concat(all_synthetic, ignore_index=True)
    print(f"\nTotal synthetic rows: {len(result)}")
    synth_storms = result["storm_name"].nunique()
    print(f"Synthetic unique storm IDs: {synth_storms}")
    # Ước tính số sequences
    synth_seqs = result.groupby(["year", "storm_name"]).size()
    n_seqs = (synth_seqs - MIN_STEPS_FOR_SEQ + 1).clip(lower=0).sum()
    print(f"Estimated synthetic sequences (seq_len=20): {n_seqs}")
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input",  default="all_storms_final.csv", type=str)
    p.add_argument("--output", default="all_storms_synthetic.csv", type=str)
    p.add_argument("--n_augment", default=5, type=int,
                   help="Số augmented versions per storm (5 → ~5× data)")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--train_year_max", default=2015, type=int,
                   help="Chỉ augment storms có year < này (tránh val/test leakage)")
    return p.parse_args()


def main():
    args = get_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input, dtype={"storm_name": str, "dt": str})
    print(f"Loaded {len(df)} rows, {df['storm_name'].nunique()} storms")

    print(f"\nGenerating synthetic tracks (n_augment={args.n_augment})...")
    synthetic_df = generate_synthetic(
        df,
        n_augment=args.n_augment,
        seed=args.seed,
        train_year_max=args.train_year_max,
    )

    synthetic_df.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")
    print(f"  Rows: {len(synthetic_df)}")
    print(f"  Unique storms: {synthetic_df['storm_name'].nunique()}")

    # Quick sanity check
    print("\n=== Sanity checks ===")
    print(f"lon range: {synthetic_df['lon'].min():.1f} - {synthetic_df['lon'].max():.1f}")
    print(f"lat range: {synthetic_df['lat'].min():.1f} - {synthetic_df['lat'].max():.1f}")
    print(f"lon_norm range: {synthetic_df['lon_norm'].min():.3f} - {synthetic_df['lon_norm'].max():.3f}")
    print(f"lat_norm range: {synthetic_df['lat_norm'].min():.3f} - {synthetic_df['lat_norm'].max():.3f}")
    print(f"pres range: {synthetic_df['pres'].min():.1f} - {synthetic_df['pres'].max():.1f}")
    print(f"wnd range: {synthetic_df['wnd'].min():.1f} - {synthetic_df['wnd'].max():.1f}")
    seqs = (synthetic_df.groupby(["year","storm_name"]).size() - MIN_STEPS_FOR_SEQ + 1).clip(lower=0)
    print(f"Sequences estimable: {seqs.sum()}")


if __name__ == "__main__":
    main()
"""
visualize_test_tracks.py
=========================
Ve duong di THAT cua tung con bao trong Data1d/test (moi bao 1 hinh
rieng), tren nen ban do don gian Bien Dong / Viet Nam (khong can
internet, khong can cartopy/geopandas - chi dung matplotlib thuan).

Du lieu duong di lay TRUC TIEP tu file .txt trong Data1d/test (toan bo
quy dao, khong chi gioi han obs_len+pred_len), dung dung cong thuc quy
doi toa do nhu prepare_dataset.py / trajectoriesWithMe_unet_training.py:
    lon_deg = (lon_norm * 50 + 1800) / 10
    lat_deg = (lat_norm * 50) / 10

CACH DUNG:
    python visualize_test_tracks.py --root /path/to/TCND_vn
    python visualize_test_tracks.py --root /path/to/TCND_vn --out_dir /path/to/output

Se tao 1 file .png cho MOI bao trong Data1d/test/*.txt, luu vao
<root>/_test_track_plots/ (hoac --out_dir neu chi dinh), ten file
"<year>_<name>_track.png".
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Duong bo bien Bien Dong / Viet Nam - toa do XAP XI (don gian hoa, ve tay
# tu hinh dang thuc te tren ban do), chi de ĐỊNH HUONG khong phai ban do
# chinh xac dia ly tuyet doi. Khong can internet / khong can cartopy.
# ─────────────────────────────────────────────────────────────────────────────
_VN_COASTLINE = [
    (106.60, 23.35), (106.75, 22.85), (107.20, 22.20), (107.75, 21.75),
    (108.05, 21.45), (107.95, 21.05), (107.50, 20.75), (107.05, 20.55),
    (106.60, 20.35), (106.30, 20.00), (106.05, 19.55), (105.95, 19.05),
    (105.95, 18.55), (106.10, 18.10), (106.40, 17.65), (106.75, 17.15),
    (107.05, 16.65), (107.35, 16.30), (107.85, 16.10), (108.20, 16.05),
    (108.25, 15.75), (108.90, 15.35), (109.15, 14.90), (109.30, 14.35),
    (109.35, 13.85), (109.30, 13.35), (109.20, 12.85), (109.20, 12.35),
    (109.15, 11.85), (109.05, 11.40), (108.90, 11.00), (108.60, 10.60),
    (108.30, 10.45), (107.85, 10.40), (107.35, 10.45), (106.85, 10.35),
    (106.70, 9.95), (106.60, 9.50), (106.20, 9.20), (105.75, 8.95),
    (105.15, 9.20), (104.80, 9.65), (104.75, 10.10), (104.95, 10.40),
    (105.35, 10.65), (105.80, 10.75), (106.20, 10.75), (106.50, 10.85),
    (106.60, 23.35),
]

# Cac nuoc lang gieng - hinh dang rat tho, chi de lap day khong gian dat
# lien lan can (Trung Quoc phia bac, Campuchia/Thai Lan phia tay, Philippines
# phia dong) giup nguoi xem dinh huong, KHONG chinh xac ve dia ly.
_CHINA_ROUGH = [
    (106.60, 23.35), (108.50, 24.50), (111.00, 25.20), (114.00, 24.80),
    (116.50, 24.00), (117.80, 23.20), (117.50, 22.30), (116.00, 21.50),
    (114.30, 21.00), (112.50, 21.20), (110.80, 21.60), (108.90, 21.55),
    (108.05, 21.45), (107.75, 21.75), (107.20, 22.20), (106.75, 22.85),
    (106.60, 23.35),
]
_HAINAN_ROUGH = [
    (108.60, 20.20), (109.30, 20.05), (110.20, 19.90), (110.75, 19.40),
    (110.90, 18.70), (110.40, 18.30), (109.60, 18.25), (108.80, 18.75),
    (108.55, 19.50), (108.60, 20.20),
]
_LUZON_ROUGH = [
    (120.30, 18.60), (121.60, 18.40), (122.10, 17.20), (122.20, 15.80),
    (121.60, 14.30), (120.80, 13.20), (120.10, 13.60), (119.85, 15.00),
    (119.95, 16.50), (120.30, 18.60),
]
_MINDORO_PANAY_ROUGH = [
    (120.90, 13.40), (121.20, 12.60), (120.60, 11.90), (119.90, 12.20),
    (120.00, 13.00), (120.90, 13.40),
]
_CAMBODIA_THAI_ROUGH = [
    (102.10, 13.60), (103.50, 14.30), (104.80, 14.40), (105.80, 12.60),
    (105.20, 11.00), (104.30, 10.50), (102.90, 10.90), (102.30, 12.20),
    (102.10, 13.60),
]

_LAND_POLYGONS = [_VN_COASTLINE, _CHINA_ROUGH, _HAINAN_ROUGH,
                  _LUZON_ROUGH, _MINDORO_PANAY_ROUGH, _CAMBODIA_THAI_ROUGH]

# Bounding box mac dinh cho vung Bien Dong / Viet Nam
_MAP_LON_MIN, _MAP_LON_MAX = 99.0, 125.0
_MAP_LAT_MIN, _MAP_LAT_MAX = 3.0, 25.0


def lonlat_from_norm(lon_norm: np.ndarray, lat_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dung cong thuc y het prepare_dataset.py / code training goc."""
    lon_deg = (lon_norm * 50.0 + 1800.0) / 10.0
    lat_deg = (lat_norm * 50.0) / 10.0
    return lon_deg, lat_deg


def parse_storm_file(path: str):
    """Doc toan bo quy dao tu 1 file .txt Data1d, tra ve (dates, lon_deg,
    lat_deg, pres_norm, wnd_norm). Parse dung format y het _read_file()
    trong trajectoriesWithMe_unet_training.py."""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "//", "-", "=")):
                continue
            toks = line.split()
            if len(toks) < 7:
                continue
            try:
                int(toks[0])
            except ValueError:
                continue
            try:
                lon_norm = float(toks[1])
                lat_norm = float(toks[2])
                pres_norm = float(toks[3])
                wnd_norm = float(toks[4])
                date = toks[5]
            except (ValueError, IndexError):
                continue
            rows.append((date, lon_norm, lat_norm, pres_norm, wnd_norm))

    if not rows:
        return None

    dates = [r[0] for r in rows]
    lon_norm_arr = np.array([r[1] for r in rows], dtype=np.float64)
    lat_norm_arr = np.array([r[2] for r in rows], dtype=np.float64)
    pres_arr = np.array([r[3] for r in rows], dtype=np.float64)
    wnd_arr = np.array([r[4] for r in rows], dtype=np.float64)
    lon_deg, lat_deg = lonlat_from_norm(lon_norm_arr, lat_norm_arr)
    return dates, lon_deg, lat_deg, pres_arr, wnd_arr


def draw_base_map(ax, lon_min, lon_max, lat_min, lat_max):
    """Ve nen ban do don gian: bien mau xanh nhat, dat lien mau xam nhat,
    luoi kinh vi tuyen, khong can internet/cartopy."""
    ax.set_facecolor("#dbeeff")  # mau bien

    for poly in _LAND_POLYGONS:
        patch = Polygon(poly, closed=True, facecolor="#e8e4d8",
                         edgecolor="#8a8a78", linewidth=0.8, zorder=1)
        ax.add_patch(patch)

    # Luoi kinh vi tuyen moi 2 do
    for lon in np.arange(np.floor(lon_min / 2) * 2, np.ceil(lon_max / 2) * 2 + 1, 2):
        ax.axvline(lon, color="white", linewidth=0.5, alpha=0.6, zorder=0.5)
    for lat in np.arange(np.floor(lat_min / 2) * 2, np.ceil(lat_max / 2) * 2 + 1, 2):
        ax.axhline(lat, color="white", linewidth=0.5, alpha=0.6, zorder=0.5)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Kinh do (°E)")
    ax.set_ylabel("Vi do (°N)")
    ax.text(0.01, 0.01, "Duong bo bien: don gian hoa, chi mang tinh dinh huong",
            transform=ax.transAxes, fontsize=7, color="gray", style="italic")


def plot_one_storm(year: str, name: str, dates, lon_deg, lat_deg,
                    wnd_arr, out_path: str, pad_deg: float = 4.0) -> None:
    lon_min = max(_MAP_LON_MIN, lon_deg.min() - pad_deg)
    lon_max = min(_MAP_LON_MAX, lon_deg.max() + pad_deg)
    lat_min = max(_MAP_LAT_MIN, lat_deg.min() - pad_deg)
    lat_max = min(_MAP_LAT_MAX, lat_deg.max() + pad_deg)
    # Dam bao khung nhin toi thieu, khong bi qua hep neu bao it di chuyen
    if lon_max - lon_min < 8:
        c = (lon_max + lon_min) / 2
        lon_min, lon_max = c - 4, c + 4
    if lat_max - lat_min < 8:
        c = (lat_max + lat_min) / 2
        lat_min, lat_max = c - 4, c + 4

    fig, ax = plt.subplots(figsize=(9, 8), dpi=130)
    draw_base_map(ax, lon_min, lon_max, lat_min, lat_max)

    # To mau duong di theo cuong do gio (wnd_norm) - normalize don gian
    # bang min-max cua chinh bao nay de nhin ro bien thien cuong do.
    points = np.array([lon_deg, lat_deg]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    if wnd_arr.max() > wnd_arr.min():
        norm_wnd = (wnd_arr - wnd_arr.min()) / (wnd_arr.max() - wnd_arr.min())
    else:
        norm_wnd = np.zeros_like(wnd_arr)
    lc = LineCollection(segments, cmap="YlOrRd", linewidth=3, zorder=3)
    lc.set_array(norm_wnd[:-1])
    ax.add_collection(lc)

    ax.scatter(lon_deg, lat_deg, c=norm_wnd, cmap="YlOrRd",
               s=35, edgecolors="black", linewidths=0.5, zorder=4)
    ax.scatter(lon_deg[0], lat_deg[0], marker="^", s=180, c="lime",
               edgecolors="black", linewidths=1.2, zorder=5, label="Bat dau")
    ax.scatter(lon_deg[-1], lat_deg[-1], marker="s", s=180, c="red",
               edgecolors="black", linewidths=1.2, zorder=5, label="Ket thuc")

    cbar = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Cuong do gio (wnd_norm, chuan hoa theo bao nay)")

    n = len(dates)
    ax.set_title(f"Duong di that: bao {name} (nam {year})\n"
                 f"{n} diem quan trac  |  {dates[0]} -> {dates[-1]}",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Thu muc goc chua Data1d/test")
    ap.add_argument("--out_dir", default=None,
                     help="Thu muc luu anh (mac dinh: <root>/_test_track_plots)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    test_dir = os.path.join(root, "Data1d", "test")
    if not os.path.isdir(test_dir):
        raise SystemExit(f"LOI: khong tim thay {test_dir}. Hay chay prepare_dataset.py "
                          f"voi --apply truoc de tao Data1d/test.")

    out_dir = args.out_dir or os.path.join(root, "_test_track_plots")
    os.makedirs(out_dir, exist_ok=True)

    txt_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".txt"))
    if not txt_files:
        raise SystemExit(f"LOI: khong co file .txt nao trong {test_dir}")

    print(f"Tim thay {len(txt_files)} bao trong tap test. Dang ve...")
    n_ok = 0
    for fname in txt_files:
        base = os.path.splitext(fname)[0]
        parts = base.split("_")
        year = parts[0] if parts else "unknown"
        name = "_".join(parts[1:]) if len(parts) > 1 else base

        result = parse_storm_file(os.path.join(test_dir, fname))
        if result is None:
            print(f"  [BO QUA] {fname}: khong parse duoc dong nao")
            continue
        dates, lon_deg, lat_deg, pres_arr, wnd_arr = result

        out_path = os.path.join(out_dir, f"{year}_{name}_track.png")
        plot_one_storm(year, name, dates, lon_deg, lat_deg, wnd_arr, out_path)
        print(f"  [OK] {year}_{name}: {len(dates)} diem -> {out_path}")
        n_ok += 1

    print(f"\nDa ve xong {n_ok}/{len(txt_files)} bao. Anh luu tai: {out_dir}")


if __name__ == "__main__":
    main()
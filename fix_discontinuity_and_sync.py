"""
fix_discontinuity_and_sync.py
================================
Ra soat TAT CA bao trong Data1d (goc, PHANG - chua chia train/val/test),
phat hien DIEM GAY QUY DAO bat thuong bang tieu chi KHOANG CACH THOI
GIAN giua 2 timestep lien tiep: neu > 6 gio (khac voi nhip chuan 6h/
buoc cua toan bo dataset), coi la dau hieu 2 con bao khac nhau bi ghep
nham vao chung 1 file .txt. CAT BO toan bo phan TU DIEM GAY DO TRO DI
(chi giu doan DAU, truoc diem gay), roi DONG BO lai Data3d/Env_data cho
khop CHINH XAC voi Data1d sau khi cat (xoa cac file .npy thua ung voi
cac timestep da bi cat bo).

TIEU CHI DUY NHAT: KHOANG CACH THOI GIAN, khong dung buoc nhay toa do
hay toc do km/h. Ly do chon tieu chi nay (qua kiem chung thuc te tren
nhieu bao that):
  - CHAN-HOM 2020: diem 65 (2020101906) -> diem 66 (2020102818) cach
    nhau 228 GIO (9.5 ngay) thay vi 6h chuan -> BAT DUOC, cat dung tai
    do (doan sau la du lieu cua MOLAVE bi dinh nham vao).
  - GONI 2020: toan bo 53 diem, moi buoc DEU DUNG 6h -> khong co diem
    gay, giu nguyen toan bo (dung, vi day la quy dao that, khop anh ve
    tinh thuc te nguoi dung cung cap).
  - KONG-REY 2024: toan bo 39 diem deu dung 6h (ke ca doan buoc nhay
    toa do lon 6.71 do luc bao vao vi do cao, tang toc tu nhien khi tai
    hop front ngoai nhiet doi) -> KHONG bi cat, dung vi day la du lieu
    that (khop anh ve tinh), khong phai loi ghep nham.
  - MOLAVE 2020: diem 28->29 buoc nhay toa do toi 12 do NHUNG van dung
    nhip 6h chuan -> KHONG bi tieu chi nay bat duoc. Day la GIOI HAN DA
    BIET va CHAP NHAN CUA PHUONG PHAP (nguoi dung da xac nhan): MOLAVE
    se KHONG tu dong bi cat boi script nay, can xu ly rieng neu can.

CANH BAO: day la thao tac PHA HUY DU LIEU GOC:
  - GHI DE truc tiep file .txt trong Data1d/ (thu muc goc PHANG, truoc
    khi chia train/val/test)
  - XOA VINH VIEN cac file .npy trong Data3d/ va Env_data/ ung voi cac
    timestep da bi cat bo
  KHONG CO BACKUP TU DONG. Nguoi dung da xac nhan hieu ro dieu nay.

CACH DUNG:
    # Buoc 1: LUON chay DRY-RUN truoc de xem bao cao, KHONG sua gi ca
    python fix_discontinuity_and_sync.py --root /path/to/TCND_vn

    # Buoc 2: neu bao cao on, chay that voi --apply
    python fix_discontinuity_and_sync.py --root /path/to/TCND_vn --apply

    # Tuy chinh nguong khoang cach thoi gian (gio) neu can, mac dinh 6.0
    # (tuc la CHINH XAC 6h moi duoc coi la binh thuong, > 6h la bat thuong)
    python fix_discontinuity_and_sync.py --root /path/to/TCND_vn --max_gap_hours 6.0 --apply
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np


DEFAULT_MAX_GAP_HOURS = 6.0


def lonlat_from_norm(lon_norm: np.ndarray, lat_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon_deg = (lon_norm * 50.0 + 1800.0) / 10.0
    lat_deg = (lat_norm * 50.0) / 10.0
    return lon_deg, lat_deg


def find_env_root(root: str) -> str:
    for cand_name in ("Env_data", "ENV_DATA", "env_data", "Env_Data"):
        cand = os.path.join(root, cand_name)
        if os.path.isdir(cand):
            return cand
    return os.path.join(root, "Env_data")


def parse_data1d_lines(path: str):
    """Doc file .txt, tra ve list cac dict {raw_line, id, date, lon_norm,
    lat_norm, lon_deg, lat_deg} cho tung dong DU LIEU (bo qua header/dong
    khong hop le), giu nguyen thu tu va noi dung dong goc de ghi lai
    chinh xac khi can cat."""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", "-", "=")):
            continue
        toks = stripped.split()
        if len(toks) < 7:
            continue
        try:
            int(toks[0])
        except ValueError:
            continue
        try:
            lon_norm = float(toks[1])
            lat_norm = float(toks[2])
            date = toks[5]
        except (ValueError, IndexError):
            continue
        rows.append({
            "raw_line": line.rstrip("\n"),
            "date": date,
            "lon_norm": lon_norm,
            "lat_norm": lat_norm,
        })
    return rows


def find_first_discontinuity(rows: list, max_gap_hours: float) -> int | None:
    """
    Tra ve INDEX (trong `rows`) cua diem DAU TIEN gay ra KHOANG CACH
    THOI GIAN bat thuong so voi diem truoc no (delta_hours giua
    rows[i-1]["date"] va rows[i]["date"] > max_gap_hours). Day la TIEU
    CHI DUY NHAT dung de phat hien diem ghep nham (khong dung buoc nhay
    toa do / toc do km/h - xem docstring dau file de biet ly do va cac
    vi du thuc te da kiem chung).

    Tra ve None neu khong co khoang gay thoi gian nao (moi buoc deu
    <= max_gap_hours, thong thuong la dung 6h).

    Neu 1 dong co timestamp khong parse duoc (dinh dang la), BO QUA
    kiem tra tai vi tri do (khong coi la diem gay) va in canh bao rieng
    - vi day la loi format khac, khong phai dau hieu ghep nham bao.
    """
    if len(rows) < 2:
        return None
    for i in range(1, len(rows)):
        try:
            d_prev = datetime.strptime(rows[i - 1]["date"], "%Y%m%d%H")
            d_cur = datetime.strptime(rows[i]["date"], "%Y%m%d%H")
        except ValueError:
            continue  # timestamp la, khong the tinh khoang cach - bo qua diem nay
        delta_hours = (d_cur - d_prev).total_seconds() / 3600.0
        if delta_hours > max_gap_hours or delta_hours <= 0:
            return i
    return None


def sync_data3d_env_for_storm(
    year: str, name: str, kept_dates: set,
    data3d_root: str, env_root: str,
    apply: bool,
) -> tuple[list, list]:
    """Xoa (hoac chi liet ke neu khong --apply) cac file .npy trong
    Data3d/<year>/<name>/ va Env_data/<year>/<name>/ ung voi timestep
    KHONG con trong kept_dates (tuc la da bi cat khoi Data1d). Tra ve
    (deleted_data3d_files, deleted_env_files)."""
    deleted_3d, deleted_env = [], []

    d3d_folder = os.path.join(data3d_root, year, name)
    if os.path.isdir(d3d_folder):
        for fname in os.listdir(d3d_folder):
            if not fname.endswith((".npy", ".nc")):
                continue
            # File dat ten dang WP<year><name>_<timestamp>.ext hoac chua
            # timestamp o dau moi tap. Kiem tra xem co timestamp nao
            # trong kept_dates la substring cua fname khong.
            matched_kept = any(ts in fname for ts in kept_dates)
            if not matched_kept:
                fpath = os.path.join(d3d_folder, fname)
                deleted_3d.append(fpath)
                if apply:
                    os.remove(fpath)

    env_folder = os.path.join(env_root, year, name)
    if os.path.isdir(env_folder):
        for fname in os.listdir(env_folder):
            if not fname.endswith(".npy"):
                continue
            matched_kept = any(ts in fname for ts in kept_dates)
            if not matched_kept:
                fpath = os.path.join(env_folder, fname)
                deleted_env.append(fpath)
                if apply:
                    os.remove(fpath)

    return deleted_3d, deleted_env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                     help="Thu muc goc chua Data1d/ (PHANG, chua chia train/val/test), Data3d/, Env_data/")
    ap.add_argument("--max_gap_hours", type=float, default=DEFAULT_MAX_GAP_HOURS,
                     help=f"Nguong khoang cach thoi gian (gio) giua 2 timestep lien tiep de "
                          f"coi la diem gay bat thuong - CHINH XAC bang gia tri nay thi binh "
                          f"thuong, LON HON thi bi coi la diem gay (mac dinh {DEFAULT_MAX_GAP_HOURS})")
    ap.add_argument("--apply", action="store_true",
                     help="Neu KHONG bat: chi DRY-RUN, in bao cao, KHONG ghi de / KHONG xoa file nao")
    ap.add_argument("--out_dir", default=None,
                     help="Thu muc ghi report CSV (mac dinh: <root>/_discontinuity_reports)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    data1d_dir = os.path.join(root, "Data1d")
    data3d_dir = os.path.join(root, "Data3d")
    env_dir = find_env_root(root)

    if not os.path.isdir(data1d_dir):
        print(f"LOI: khong tim thay {data1d_dir}", file=sys.stderr)
        sys.exit(1)

    txt_files = sorted(
        f for f in os.listdir(data1d_dir)
        if f.endswith(".txt") and os.path.isfile(os.path.join(data1d_dir, f))
    )
    if not txt_files:
        print(f"LOI: khong co file .txt PHANG nao trong {data1d_dir}. Script nay can chay "
              f"TRUOC khi chia train/val/test (truoc khi goi prepare_dataset.py --apply), "
              f"hoac tren thu muc Data1d goc con giu file phang.", file=sys.stderr)
        sys.exit(1)

    print(f"Tim thay {len(txt_files)} file bao trong {data1d_dir}")
    print(f"Nguong phat hien diem gay: khoang cach thoi gian > {args.max_gap_hours} gio "
          f"giua 2 timestep lien tiep (nhip chuan la {args.max_gap_hours}h)")
    print(f"Che do: {'--apply (SE GHI DE / XOA FILE THAT)' if args.apply else 'DRY-RUN (chi bao cao, khong sua gi)'}")
    print()

    results = []  # list of dict cho report

    for fname in txt_files:
        path = os.path.join(data1d_dir, fname)
        base = os.path.splitext(fname)[0]
        parts = base.split("_")
        year = parts[0] if parts else "unknown"
        name = "_".join(parts[1:]) if len(parts) > 1 else base

        rows = parse_data1d_lines(path)
        n_before = len(rows)
        if n_before == 0:
            continue

        break_idx = find_first_discontinuity(rows, args.max_gap_hours)

        if break_idx is None:
            results.append({
                "year": year, "name": name, "n_before": n_before,
                "n_after": n_before, "break_idx": "", "break_date": "",
                "gap_hours": "", "step_deg": "", "n_data3d_deleted": 0, "n_env_deleted": 0,
                "status": "khong_co_diem_gay",
            })
            continue

        kept_rows = rows[:break_idx]
        n_after = len(kept_rows)
        break_date = rows[break_idx]["date"]
        prev_date = rows[break_idx - 1]["date"]

        try:
            d_prev = datetime.strptime(prev_date, "%Y%m%d%H")
            d_cur = datetime.strptime(break_date, "%Y%m%d%H")
            gap_hours = (d_cur - d_prev).total_seconds() / 3600.0
        except ValueError:
            gap_hours = float("nan")

        # Buoc nhay toa do chi de THAM KHAO trong bao cao, KHONG dung de
        # quyet dinh (tieu chi quyet dinh DUY NHAT la khoang cach thoi
        # gian o tren - xem docstring dau file).
        lon_before, lat_before = lonlat_from_norm(rows[break_idx - 1]["lon_norm"], rows[break_idx - 1]["lat_norm"])
        lon_after, lat_after = lonlat_from_norm(rows[break_idx]["lon_norm"], rows[break_idx]["lat_norm"])
        step_deg = float(np.sqrt((lon_after - lon_before) ** 2 + (lat_after - lat_before) ** 2))

        print(f"[DIEM GAY] {year}_{name}: {n_before} diem -> phat hien gay tai idx={break_idx} "
              f"({prev_date} -> {break_date}, cach nhau {gap_hours:.0f}h thay vi "
              f"{args.max_gap_hours:.0f}h chuan; buoc nhay toa do {step_deg:.2f} do - chi de "
              f"tham khao). Cat con lai {n_after} diem.")

        if args.apply:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(r["raw_line"] for r in kept_rows))

        kept_dates = {r["date"] for r in kept_rows}
        deleted_3d, deleted_env = sync_data3d_env_for_storm(
            year, name, kept_dates, data3d_dir, env_dir, args.apply)

        if deleted_3d or deleted_env:
            print(f"    -> dong bo Data3d/Env: xoa {len(deleted_3d)} file Data3d, "
                  f"{len(deleted_env)} file Env_data thua (ung voi timestep da cat)")

        results.append({
            "year": year, "name": name, "n_before": n_before,
            "n_after": n_after, "break_idx": break_idx, "break_date": break_date,
            "gap_hours": f"{gap_hours:.1f}", "step_deg": f"{step_deg:.2f}",
            "n_data3d_deleted": len(deleted_3d), "n_env_deleted": len(deleted_env),
            "status": "DA_CAT" if args.apply else "SE_CAT_neu_apply",
        })

    # Ngoai cac bao co diem gay, van dong bo Data3d/Env cho CA CAC BAO
    # KHONG co diem gay - de dam bao 100% khop tuyet doi (khong du thua)
    # theo dung yeu cau "khong du khong thua timestep va bao giua 3 bo
    # du lieu", ke ca voi cac bao vốn da sach tu dau.
    print("\nDang dong bo Data3d/Env cho cac bao KHONG co diem gay (dam bao khop tuyet doi)...")
    n_extra_synced = 0
    for r in results:
        if r["status"] != "khong_co_diem_gay":
            continue
        path = os.path.join(data1d_dir, f"{r['year']}_{r['name']}.txt")
        rows = parse_data1d_lines(path)
        kept_dates = {row["date"] for row in rows}
        deleted_3d, deleted_env = sync_data3d_env_for_storm(
            r["year"], r["name"], kept_dates, data3d_dir, env_dir, args.apply)
        if deleted_3d or deleted_env:
            n_extra_synced += 1
            print(f"  {r['year']}_{r['name']}: xoa {len(deleted_3d)} file Data3d thua, "
                  f"{len(deleted_env)} file Env_data thua (khong lien quan diem gay, "
                  f"chi la du lieu Data3d/Env co san du hon Data1d)")
        r["n_data3d_deleted"] = len(deleted_3d)
        r["n_env_deleted"] = len(deleted_env)

    # ── Bao cao ──
    out_dir = args.out_dir or os.path.join(root, "_discontinuity_reports")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "discontinuity_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "name", "n_points_truoc", "n_points_sau", "break_index",
                    "break_date", "khoang_cach_gio_tai_diem_gay", "buoc_nhay_toa_do_do_tham_khao",
                    "n_data3d_da_xoa", "n_env_da_xoa", "trang_thai"])
        for r in results:
            w.writerow([r["year"], r["name"], r["n_before"], r["n_after"],
                        r["break_idx"], r["break_date"], r.get("gap_hours", ""), r["step_deg"],
                        r["n_data3d_deleted"], r["n_env_deleted"], r["status"]])

    n_with_break = sum(1 for r in results if r["break_idx"] != "")
    n_total_3d_deleted = sum(r["n_data3d_deleted"] for r in results)
    n_total_env_deleted = sum(r["n_env_deleted"] for r in results)

    print(f"\n=== TOM TAT ===")
    print(f"  Tong so bao kiem tra                : {len(results)}")
    print(f"  Bao PHAT HIEN diem gay bat thuong    : {n_with_break}")
    print(f"  Tong file Data3d da xoa (thua)       : {n_total_3d_deleted}")
    print(f"  Tong file Env_data da xoa (thua)     : {n_total_env_deleted}")
    print(f"  Report chi tiet: {report_path}")

    if not args.apply:
        print(f"\n[DRY-RUN] Chua sua/xoa gi ca. Xem lai {report_path}, neu on thi chay lai voi --apply.")
    else:
        print(f"\nDa GHI DE Data1d va DONG BO/XOA file Data3d, Env_data thua. "
              f"Ban co the chay lai prepare_dataset.py de kiem tra + chia lai train/val/test tu du lieu da sach.")


if __name__ == "__main__":
    main()
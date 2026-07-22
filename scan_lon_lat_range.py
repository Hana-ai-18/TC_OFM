"""
scan_lonlat_range.py
=====================
Quét TOÀN BỘ file .txt trong Data1d/{train,val,test} để tính đúng
phạm vi kinh độ/vĩ độ THẬT của mọi storm — dùng kết quả này để đặt
đúng _LON_NORM_MIN/_LON_NORM_MAX/_LAT_NORM_MIN/_LAT_NORM_MAX trong
trajectoriesWithMe_unet_training.py, THAY VÌ đoán/ước lượng.

Đây là bước bắt buộc trước khi sửa bug clip (FIX-DATA-27) — nếu chỉ
ước lượng theo địa lý chung (như Claude từng thử), rất dễ vô tình cắt
cụt storm đã hoạt động đúng trước đó (ví dụ RITA đi tới tận 45°N,
150°E — vượt xa mọi ước lượng "vùng Biển Đông" thông thường).

Công thức chuẩn hóa (đã xác nhận đúng với dữ liệu thật):
  LONG_norm = (LONG × 10 − 1800) / 50
  LAT_norm  = (LAT  × 10) / 50
=> Đảo ngược để đọc lại độ thật từ file .txt đã normalize sẵn:
  LONG = (LONG_norm × 50 + 1800) / 10
  LAT  = (LAT_norm  × 50) / 10

USAGE (chạy trên Kaggle, trỏ đúng root chứa Data1d/):
  python scan_lonlat_range.py --root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm
"""
import os
import re
import argparse


def norm_to_deg_lon(v: float) -> float:
    return (v * 50.0 + 1800.0) / 10.0


def norm_to_deg_lat(v: float) -> float:
    return (v * 50.0) / 10.0


def scan_file(filepath):
    """
    Đọc 1 file .txt đã normalize (theo đúng format bạn cho xem trước:
    cột ID/LONG/LAT/PRES/WND/YYYYMMDDHH/Name), trả về list các
    (lon_deg, lat_deg) đã denorm.
    """
    points = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        up = line.split("#")[0].strip().upper()
        if "LAT" in up and "LONG" in up:
            header_idx = i
            break
    if header_idx is None:
        return points

    for line in lines[header_idx + 1:]:
        s = line.strip()
        if not s or re.match(r'^[-=\s]+$', s):
            continue
        parts = re.split(r'\s+', s)
        if len(parts) < 3:
            continue
        try:
            lon_norm = float(parts[1])
            lat_norm = float(parts[2])
        except (ValueError, IndexError):
            continue
        lon_deg = norm_to_deg_lon(lon_norm)
        lat_deg = norm_to_deg_lat(lat_norm)
        points.append((lon_deg, lat_deg))

    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Thư mục chứa Data1d/ (vd: /kaggle/input/datasets/kaggle1234uitvn/tc-ofm)")
    ap.add_argument("--scs_lon_min", type=float, default=99.0,
                    help="[MỚI] Biên Tây của hộp 'Biển Đông' dùng để lọc storm -- "
                         "mặc định 99°E (khoảng Vịnh Thái Lan/phía Nam Biển Đông).")
    ap.add_argument("--scs_lon_max", type=float, default=121.0,
                    help="[MỚI] Biên Đông của hộp 'Biển Đông' -- mặc định 121°E "
                         "(khoảng Luzon Strait, ranh giới với Tây TBD).")
    ap.add_argument("--scs_lat_min", type=float, default=0.0,
                    help="[MỚI] Biên Nam của hộp 'Biển Đông' -- mặc định 0°N.")
    ap.add_argument("--scs_lat_max", type=float, default=23.0,
                    help="[MỚI] Biên Bắc của hộp 'Biển Đông' -- mặc định 23°N "
                         "(khoảng biên giới Việt Nam - Trung Quốc / đảo Hải Nam).")
    ap.add_argument("--min_pct_in_scs", type=float, default=15.0,
                    help="[MỚI, FIX] Trước đây chỉ cần >=1 điểm trong hộp là GIỮ "
                         "toàn bộ storm -- điều này giữ NHẦM các storm như "
                         "1989_GAY (chủ yếu ở Vịnh Bengal, 74-92°E, chỉ chạm "
                         "rìa Tây Biển Đông ở vài điểm CUỐI track, 99-105°E) vì "
                         "về mặt TOÁN HỌC nó có >=1 điểm trong hộp. Giờ yêu cầu "
                         "ÍT NHẤT bao nhiêu %% số điểm của track phải nằm trong "
                         "hộp mới được GIỮ -- mặc định 15%% (đủ thấp để giữ storm "
                         "hình thành xa rồi mới vào Biển Đông ở nửa sau track như "
                         "MOLAVE, đủ cao để loại storm chỉ chạm rìa thoáng qua).")
    args = ap.parse_args()

    data1d = os.path.join(args.root, "Data1d")
    if not os.path.isdir(data1d):
        print(f"❌ Không tìm thấy {data1d}")
        return

    splits = ["train", "val", "test"]
    global_lon_min, global_lon_max = float("inf"), float("-inf")
    global_lat_min, global_lat_max = float("inf"), float("-inf")
    kept, dropped = [], []

    print(f"  Hộp 'Biển Đông' dùng để lọc: LON [{args.scs_lon_min}, {args.scs_lon_max}]°E, "
          f"LAT [{args.scs_lat_min}, {args.scs_lat_max}]°N")
    print(f"  Tiêu chí: GIỮ storm nếu ÍT NHẤT {args.min_pct_in_scs}% số điểm "
          f"trong track rơi vào hộp trên (không phải chỉ >=1 điểm — tránh giữ")
    print(f"  nhầm storm chỉ chạm rìa thoáng qua như 1989_GAY, chủ yếu ở Vịnh Bengal).\n")

    print(f"{'Split':<8} {'File':<28} {'#pts':>6} {'lon_min':>9} {'lon_max':>9} {'lat_min':>9} {'lat_max':>9} {'%inSCS':>7}  {'Note':<10}")
    print("-" * 108)

    for split in splits:
        split_dir = os.path.join(data1d, split)
        if not os.path.isdir(split_dir):
            print(f"  (không có thư mục {split}/, bỏ qua)")
            continue

        for fname in sorted(os.listdir(split_dir)):
            if not fname.lower().endswith(".txt"):
                continue
            fpath = os.path.join(split_dir, fname)
            pts = scan_file(fpath)
            if not pts:
                print(f"{split:<8} {fname:<28} {'(không đọc được / rỗng)'}")
                continue

            lons = [p[0] for p in pts]
            lats = [p[1] for p in pts]
            lo_min, lo_max = min(lons), max(lons)
            la_min, la_max = min(lats), max(lats)

            # [MỚI, FIX] Đếm % điểm trong hộp, không chỉ check >=1 điểm.
            # Bug đã tìm ra ở lần chạy trước: any() cho phép storm như
            # 1989_GAY (74-105°E, chủ yếu ở Vịnh Bengal, chỉ vài điểm
            # CUỐI chạm rìa Tây Biển Đông 99-105°E) vẫn được GIỮ vì về
            # mặt toán học nó có >=1 điểm trong hộp -- kéo _LON_NORM_MIN
            # xuống tận 73°E một cách sai lệch. Giờ yêu cầu tỷ lệ % tối
            # thiểu số điểm phải nằm trong hộp.
            n_in_scs = sum(
                1 for lon, lat in pts
                if args.scs_lon_min <= lon <= args.scs_lon_max and
                   args.scs_lat_min <= lat <= args.scs_lat_max
            )
            pct_in_scs = 100.0 * n_in_scs / len(pts)
            crosses_scs = pct_in_scs >= args.min_pct_in_scs
            note = "" if crosses_scs else "KHONG_QUA_SCS"

            if crosses_scs:
                kept.append((split, fname, lo_min, lo_max, la_min, la_max))
                global_lon_min = min(global_lon_min, lo_min)
                global_lon_max = max(global_lon_max, lo_max)
                global_lat_min = min(global_lat_min, la_min)
                global_lat_max = max(global_lat_max, la_max)
            else:
                dropped.append((split, fname, lo_min, lo_max, la_min, la_max, pct_in_scs))

            print(f"{split:<8} {fname:<28} {len(pts):>6} "
                  f"{lo_min:>9.2f} {lo_max:>9.2f} {la_min:>9.2f} {la_max:>9.2f} {pct_in_scs:>6.1f}%  {note:<10}")

    print("-" * 108)

    print(f"\n  Tổng: {len(kept)} storm GIỮ (có đi qua Biển Đông), "
          f"{len(dropped)} storm LOẠI (không hề đi qua Biển Đông).")

    if dropped:
        print(f"\n  ⚠ {len(dropped)} STORM BỊ LOẠI (dưới {args.min_pct_in_scs}% điểm trong hộp Biển Đông):")
        for split, fname, lo_min, lo_max, la_min, la_max, pct in dropped:
            print(f"    {split}/{fname}: lon=[{lo_min:.1f},{lo_max:.1f}] lat=[{la_min:.1f},{la_max:.1f}]  ({pct:.1f}% trong hộp)")
        print(f"\n  ⚠ KIỂM TRA LẠI danh sách trên — nếu có storm nào bạn biết CHẮC CHẮN")
        print(f"  có đi qua Biển Đông/gần Việt Nam mà vẫn bị loại, giảm --min_pct_in_scs")
        print(f"  hoặc nới rộng hộp bằng --scs_lon_min/--scs_lon_max/--scs_lat_min/--scs_lat_max.")

    if global_lon_min == float("inf"):
        print("\n  ❌ Không có storm nào khớp tiêu chí — kiểm tra lại hộp Biển Đông.")
        return

    print(f"\n  PHẠM VI THẬT (chỉ tính {len(kept)} storm có đi qua Biển Đông):")
    print(f"    LON: [{global_lon_min:.2f}, {global_lon_max:.2f}]°E")
    print(f"    LAT: [{global_lat_min:.2f}, {global_lat_max:.2f}]°N")

    # Thêm buffer nhỏ (1°) để tránh clip đúng ngay biên do sai số làm tròn
    buf = 1.0
    lon_min_buf = global_lon_min - buf
    lon_max_buf = global_lon_max + buf
    lat_min_buf = max(0.0, global_lat_min - buf)  # LAT không nên âm cho TC Bắc bán cầu
    lat_max_buf = global_lat_max + buf

    lon_norm_min = (lon_min_buf * 10 - 1800) / 50
    lon_norm_max = (lon_max_buf * 10 - 1800) / 50
    lat_norm_min = (lat_min_buf * 10) / 50
    lat_norm_max = (lat_max_buf * 10) / 50

    print(f"\n  ĐỀ XUẤT (đã cộng buffer ±{buf}°):")
    print(f"    _LON_VALID_MIN  = {lon_min_buf:.2f}   # (độ thật)")
    print(f"    _LON_VALID_MAX  = {lon_max_buf:.2f}")
    print(f"    _LAT_VALID_MAX  = {lat_max_buf:.2f}")
    print(f"    _LON_NORM_MIN   = {lon_norm_min:.4f}   # (giá trị normalize tương ứng)")
    print(f"    _LON_NORM_MAX   = {lon_norm_max:.4f}")
    print(f"    _LAT_NORM_MIN   = {lat_norm_min:.4f}")
    print(f"    _LAT_NORM_MAX   = {lat_norm_max:.4f}")
    print(f"\n  Copy đúng 4 giá trị _LON_NORM_MIN/MAX/_LAT_NORM_MIN/MAX ở trên vào")
    print(f"  trajectoriesWithMe_unet_training.py (dòng 112-115) rồi gửi lại cho Claude")
    print(f"  để cập nhật code chính xác — KHÔNG dùng số ước lượng.")


if __name__ == "__main__":
    main()
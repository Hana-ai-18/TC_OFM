# # """
# # scripts/visual_evaluate_model_Me.py
# # ====================================
# # TC-Diffusion 72h Forecast Visualisation  ── FIXED VERSION

# # FIXES:
# # 1. Proper DDPM sampling (không dùng physics model đơn giản)
# # 2. Coordinate system đúng: LONG → X (East=Right), LAT → Y (North=Up)
# # 3. Anchor: trajectory bắt đầu từ current position
# # 4. detect_pred_len() dùng đúng key 'denoiser.pos_enc'
# # 5. Scale và offset tính lại đúng để track khớp satellite image
# # """

# # import os
# # import sys
# # import random
# # import argparse

# # import numpy as np
# # import torch
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt
# # import cv2
# # from datetime import datetime

# # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# # sys.path.insert(0, project_root)

# # from Model.flow_matching_model import TCFlowMatching
# # from Model.data.loader import data_loader
# # from Model.data.trajectoriesWithMe_unet_training import seq_collate


# # # ── Seed ─────────────────────────────────────────────────────────────────────

# # def set_seed(s=42):
# #     random.seed(s); np.random.seed(s)
# #     torch.manual_seed(s); torch.cuda.manual_seed_all(s)
# #     torch.backends.cudnn.deterministic = True
# #     torch.backends.cudnn.benchmark     = False
# #     print(f"🔒 Seed fixed = {s}\n")


# # # ── Device helpers ────────────────────────────────────────────────────────────

# # def move_batch(batch, device):
# #     out = list(batch)
# #     for i, x in enumerate(out):
# #         if torch.is_tensor(x):
# #             out[i] = x.to(device)
# #         elif isinstance(x, dict):
# #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# #                       for k, v in x.items()}
# #     return tuple(out)


# # # ── Checkpoint helpers ────────────────────────────────────────────────────────

# # def detect_pred_len(ckpt_path):
# #     """
# #     Đọc pred_len từ pos_enc shape trong checkpoint.
# #     KEY PHẢI KHỚP với DeterministicDenoiser: 'denoiser.pos_enc'
# #     """
# #     ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
# #     sd = ck.get('model_state_dict', ck.get('model_state', ck))

# #     # Thử đúng key trước
# #     for key in ['denoiser.pos_enc', 'denoiser.pos_encoding']:
# #         if key in sd:
# #             return sd[key].shape[1]

# #     # Fallback: tìm bất kỳ pos_enc nào
# #     for k, v in sd.items():
# #         if 'pos_enc' in k and v.dim() == 3:
# #             return v.shape[1]

# #     print("  Không tìm thấy pos_enc key, dùng pred_len=12")
# #     return 12


# # # ── Denormalization ────────────────────────────────────────────────────────────

# # def denorm(norm_traj):
# #     """
# #     [N, 2] normalised → [N, 2] real (0.1° units)
# #     LONG = norm * 50 + 1800  (0.1°E)
# #     LAT  = norm * 50          (0.1°N)
# #     """
# #     r = np.zeros_like(norm_traj)
# #     r[:, 0] = norm_traj[:, 0] * 50.0 + 1800.0   # LONG 0.1°E
# #     r[:, 1] = norm_traj[:, 1] * 50.0             # LAT  0.1°N
# #     return r


# # def real_to_deg(pts_01):
# #     """Convert 0.1° units → degrees"""
# #     return pts_01 / 10.0


# # # ── Pixel mapping ─────────────────────────────────────────────────────────────

# # def to_pixels(coords_deg, ref_deg, ppu, cx, cy):
# #     """
# #     [N, 2] degrees (LONG, LAT) → screen pixels
# #     LONG increases East → screen X increases right (+)
# #     LAT  increases North → screen Y increases UP, but screen Y is inverted → (-)
# #     """
# #     d_long = coords_deg[:, 0] - ref_deg[0]   # dương = Đông
# #     d_lat  = coords_deg[:, 1] - ref_deg[1]   # dương = Bắc

# #     px = cx + d_long * ppu   # East  → right
# #     py = cy - d_lat  * ppu   # North → up (Y inverted on screen)
# #     return px, py


# # # ── Satellite image ───────────────────────────────────────────────────────────

# # def load_himawari(him_path, year, name, timestamp):
# #     name  = name.strip().upper()
# #     exact = os.path.join(him_path, str(year), name, f"{timestamp}.png")
# #     if os.path.exists(exact):
# #         img = cv2.imread(exact)
# #         if img is not None:
# #             print(f"🛰️  Loaded: {exact}")
# #             return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# #     d = os.path.join(him_path, str(year), name)
# #     if os.path.exists(d):
# #         pngs = sorted(f for f in os.listdir(d) if f.endswith('.png'))
# #         if pngs:
# #             tgt  = datetime.strptime(timestamp, '%Y%m%d%H')
# #             best = min(pngs, key=lambda f: abs(
# #                 (datetime.strptime(f[:-4], '%Y%m%d%H') - tgt).total_seconds()))
# #             path = os.path.join(d, best)
# #             img  = cv2.imread(path)
# #             if img is not None:
# #                 print(f"🛰️  Nearest: {path}")
# #                 return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# #     print("  Himawari image not found – black background")
# #     return np.zeros((1000, 1000, 3), dtype=np.uint8)


# # # ── Main ──────────────────────────────────────────────────────────────────────

# # def visualize_forecast(args):
# #     set_seed(42)
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #     print(f"{'='*65}")
# #     print(f"  TC-Diffusion Forecast  |  {args.tc_name}  @  {args.tc_date}")
# #     print(f"{'='*65}\n")

# #     # ── Auto-detect pred_len ──────────────────────────────────────────────
# #     detected = detect_pred_len(args.model_path)
# #     if args.pred_len != detected:
# #         print(f"  pred_len: {args.pred_len} → {detected} (from checkpoint)")
# #         args.pred_len = detected

# #     # ── Load model ────────────────────────────────────────────────────────
# #     model = TCFlowMatching(pred_len=args.pred_len, obs_len=args.obs_len).to(device)
# #     ck    = torch.load(args.model_path, map_location=device, weights_only=False)
# #     sd    = ck.get('model_state_dict', ck.get('model_state', ck))
# #     model.load_state_dict(sd)
# #     model.eval()
# #     print(" Model loaded\n")

# #     # ── Load dataset ──────────────────────────────────────────────────────
# #     dset, _ = data_loader(
# #         args,
# #         {'root': args.TC_data_path, 'type': args.dset_type},
# #         test=True,
# #         test_year=args.test_year,
# #     )
# #     print(f" Dataset: {len(dset)} samples\n")

# #     # ── Find typhoon sequence ─────────────────────────────────────────────
# #     t_name  = args.tc_name.strip().upper()
# #     t_date  = str(args.tc_date).strip()
# #     target  = None

# #     for i in range(len(dset)):
# #         item = dset[i]
# #         info = item[-1]
# #         if (t_name in str(info['old'][1]).strip().upper()
# #                 and t_date == str(info['tydate'][args.obs_len]).strip()):
# #             target = item
# #             print(f" Found: {info['old'][1]}  @  {info['tydate'][args.obs_len]}\n")
# #             break

# #     if target is None:
# #         print(f" '{t_name} @ {t_date}' not found. Check --tc_name and --tc_date.")
# #         # Gợi ý các sample có sẵn
# #         print("Available samples (first 10):")
# #         for i in range(min(10, len(dset))):
# #             info = dset[i][-1]
# #             print(f"  [{i}] {info['old'][1]} @ {info['tydate'][args.obs_len]}")
# #         return

# #     # ── Build batch ───────────────────────────────────────────────────────
# #     batch = move_batch(seq_collate([target]), device)

# #     # ── Inference: proper DDPM sampling ──────────────────────────────────
# #     print(" Running DDPM reverse diffusion sampling...")
# #     # pred_traj_t, pred_Me_t = model.sample(batch)  # [T_pred, B, 2]
# #     pred_traj_t, pred_Me_t, _ = model.sample(batch)
# #     print(" Sampling done\n")

# #     # ── Extract arrays ────────────────────────────────────────────────────
# #     obs_n  = batch[0][:, 0, :].cpu().numpy()    # [T_obs,  2] normalized
# #     gt_n   = batch[1][:, 0, :].cpu().numpy()    # [T_pred, 2] normalized
# #     pred_n = pred_traj_t[:, 0, :].cpu().numpy() # [T_pred, 2] normalized

# #     obs_r  = denorm(obs_n)   # [T_obs,  2] 0.1° units
# #     gt_r   = denorm(gt_n)    # [T_pred, 2] 0.1° units
# #     pred_r = denorm(pred_n)  # [T_pred, 2] 0.1° units

# #     # Convert sang degrees để hiển thị
# #     obs_deg  = real_to_deg(obs_r)   # [T_obs,  2] degrees
# #     gt_deg   = real_to_deg(gt_r)    # [T_pred, 2] degrees
# #     pred_deg = real_to_deg(pred_r)  # [T_pred, 2] degrees

# #     # ── Error report ──────────────────────────────────────────────────────
# #     # Error tính theo 0.1° units, 1 unit ≈ 11.1 km
# #     errors_km = np.linalg.norm(gt_r - pred_r, axis=1) * 11.1

# #     print(" Track errors:")
# #     for i, e in enumerate(errors_km):
# #         mark = "  ◀" if (i + 1) in [4, 8, 12] else ""
# #         print(f"   +{(i+1)*6:3d}h : {e:6.1f} km{mark}")
# #     print(f"\n   Mean : {errors_km.mean():6.1f} km")
# #     n = len(errors_km)
# #     if n >= 4:  print(f"   24h  : {errors_km[3]:6.1f} km")
# #     if n >= 8:  print(f"   48h  : {errors_km[7]:6.1f} km")
# #     if n >= 12: print(f"   72h  : {errors_km[11]:6.1f} km")
# #     print()

# #     # ── Coordinate ranges ─────────────────────────────────────────────────
# #     ref_r   = obs_r[-1]                        # last observed (0.1° units)
# #     ref_deg = real_to_deg(ref_r)               # degrees

# #     print(f"  Current position: LONG={ref_deg[0]:.2f}°E, LAT={ref_deg[1]:.2f}°N")
# #     print(f"   GT 72h: LONG={gt_deg[-1,0]:.2f}°E, LAT={gt_deg[-1,1]:.2f}°N")
# #     print(f"   PD 72h: LONG={pred_deg[-1,0]:.2f}°E, LAT={pred_deg[-1,1]:.2f}°N\n")

# #     # ── Satellite background ──────────────────────────────────────────────
# #     sat = load_himawari(args.himawari_path, args.test_year, t_name, t_date)

# #     SZ  = 1000
# #     sat = cv2.resize(sat, (SZ, SZ))
# #     cx, cy = SZ / 2.0, SZ / 2.0

# #     # ── Scale: tính ppu (pixels per 0.1°) ────────────────────────────────
# #     # Gom tất cả điểm để tính span
# #     all_pts_deg = np.vstack([obs_deg, gt_deg, pred_deg])
# #     span_long   = all_pts_deg[:, 0].max() - all_pts_deg[:, 0].min()
# #     span_lat    = all_pts_deg[:, 1].max() - all_pts_deg[:, 1].min()
# #     span_max    = max(span_long, span_lat, 5.0)  # tối thiểu 5°

# #     # Dùng 65% canvas để hiển thị toàn bộ track
# #     ppu = (SZ * 0.65) / span_max  # pixels per degree

# #     # ── Convert to screen pixels ──────────────────────────────────────────
# #     def px(pts_deg):
# #         return to_pixels(pts_deg, ref_deg, ppu, cx, cy)

# #     ox, oy  = px(obs_deg)

# #     # Ground truth bắt đầu từ current position
# #     gt_full  = np.vstack([ref_deg.reshape(1, -1), gt_deg])
# #     gx, gy   = px(gt_full)

# #     # Prediction bắt đầu từ current position
# #     pred_full = np.vstack([ref_deg.reshape(1, -1), pred_deg])
# #     bx, by    = px(pred_full)

# #     # ── Plot ──────────────────────────────────────────────────────────────
# #     fig, ax = plt.subplots(figsize=(16, 16), facecolor='black')
# #     ax.set_facecolor('black')
# #     ax.imshow(sat, extent=[0, SZ, SZ, 0], alpha=0.60, zorder=0)

# #     # 1. Observed (cyan)
# #     ax.plot(ox, oy, 'o-', color='#00FFFF', linewidth=2.5, markersize=5,
# #             markeredgecolor='white', markeredgewidth=1.2,
# #             label=f'Observed ({args.obs_len * 6}h)', zorder=8, alpha=0.9)

# #     # 2. Ground truth (đỏ)
# #     ax.plot(gx, gy, 'o-', color='#FF2222', linewidth=4, markersize=8,
# #             markeredgecolor='white', markeredgewidth=2,
# #             label=f'Actual track ({args.pred_len * 6}h)', zorder=9, alpha=0.95)

# #     # 3. Prediction (xanh lá)
# #     ax.plot(bx, by, 'o-', color='#00FF44', linewidth=4, markersize=8,
# #             markeredgecolor='#004400', markeredgewidth=2,
# #             label=f'Predicted track ({args.pred_len * 6}h)', zorder=10, alpha=0.95)

# #     # 4. Error connectors tại 24h / 48h / 72h
# #     for step_idx, label_h in [(4, 24), (8, 48), (12, 72)]:
# #         si = step_idx  # index trong full array (0=NOW, 1=+6h, ...)
# #         if si < len(gx) and si < len(bx):
# #             ax.plot([gx[si], bx[si]], [gy[si], by[si]],
# #                     '--', color='#FFD700', linewidth=1.8, alpha=0.65, zorder=7)
# #             # Ghi khoảng cách error
# #             if step_idx - 1 < len(errors_km):
# #                 mid_x = (gx[si] + bx[si]) / 2
# #                 mid_y = (gy[si] + by[si]) / 2
# #                 ax.text(mid_x, mid_y, f'{errors_km[step_idx-1]:.0f}km',
# #                         fontsize=8, color='#FFD700', ha='center',
# #                         bbox=dict(fc='black', alpha=0.6, ec='none', pad=1), zorder=18)

# #     # 5. Time labels trên track xanh (prediction)
# #     for i in range(len(bx)):
# #         h = i * 6
# #         if i == 0:
# #             lbl, col, fs = 'NOW', 'white', 11
# #         elif h % 12 == 0:
# #             e_km = errors_km[i - 1] if i > 0 and i - 1 < len(errors_km) else 0
# #             lbl  = f'+{h}h\n{e_km:.0f}km'
# #             col  = '#AAFF66'
# #             fs   = 9
# #         else:
# #             continue
# #         ax.text(bx[i], by[i] - 28, lbl,
# #                 fontsize=fs, color=col, ha='center', fontweight='bold',
# #                 bbox=dict(boxstyle='round,pad=0.4', facecolor='black',
# #                           alpha=0.82, edgecolor=col, linewidth=1.5),
# #                 zorder=16)

# #     # 6. Hướng mũi tên cho track dự đoán (để thấy hướng di chuyển)
# #     for i in range(0, len(bx) - 1, 2):
# #         dx = bx[i+1] - bx[i]
# #         dy = by[i+1] - by[i]
# #         if abs(dx) + abs(dy) > 5:
# #             ax.annotate('', xy=(bx[i+1], by[i+1]), xytext=(bx[i], by[i]),
# #                        arrowprops=dict(arrowstyle='->', color='#00FF44',
# #                                       lw=1.5, mutation_scale=15),
# #                        zorder=11)

# #     # 7. Current position ★
# #     ax.scatter([cx], [cy], color='#FFD700', marker='*', s=900,
# #                edgecolors='#FF4400', linewidths=3, zorder=25,
# #                label='Current position')

# #     # ── Title ─────────────────────────────────────────────────────────────
# #     dt_str = datetime.strptime(t_date, '%Y%m%d%H').strftime('%d %b %Y  %H:%M UTC')
# #     fh     = args.pred_len * 6
# #     mean_e = errors_km.mean()
# #     last_e = errors_km[-1]

# #     ax.set_title(
# #         f"🌀  {t_name}  –  {fh}h TC-Diffusion Forecast\n"
# #         f"📅  {dt_str}    │    Mean: {mean_e:.0f} km    │    {fh}h: {last_e:.0f} km",
# #         fontsize=17, fontweight='bold', color='white', pad=18,
# #         bbox=dict(boxstyle='round,pad=0.9', facecolor='#000000',
# #                   alpha=0.92, edgecolor='#00FFFF', linewidth=2.5),
# #     )

# #     # ── Legend ────────────────────────────────────────────────────────────
# #     ax.legend(loc='upper right', fontsize=12, framealpha=0.92,
# #               facecolor='#111111', edgecolor='#00FFFF',
# #               labelcolor='white', title='Track Legend',
# #               title_fontsize=13)

# #     # ── Info panel (lower-left) ───────────────────────────────────────────
# #     lines = [
# #         "Model : TC-Diffusion (DDPM)",
# #         f"Obs   : {args.obs_len} × 6h = {args.obs_len*6}h",
# #         f"Pred  : {args.pred_len} × 6h = {fh}h",
# #         f"Ref   : {ref_deg[0]:.1f}°E  {ref_deg[1]:.1f}°N",
# #         "",
# #         "Track Errors (km):",
# #     ]
# #     for i, e in enumerate(errors_km):
# #         h = (i + 1) * 6
# #         if h in [12, 24, 48, 72] and h <= fh:
# #             lines.append(f"  {h:3d}h : {e:6.1f}")
# #     lines.append(f"  Mean : {mean_e:6.1f}")

# #     ax.text(0.02, 0.02, '\n'.join(lines),
# #             transform=ax.transAxes, fontsize=10, va='bottom',
# #             family='monospace', color='#88FF88',
# #             bbox=dict(boxstyle='round,pad=0.6', facecolor='black',
# #                       alpha=0.88, edgecolor='white', linewidth=1.5),
# #             zorder=20)

# #     # ── Compass rose (nhỏ, góc dưới phải) ────────────────────────────────
# #     ax.annotate('N', xy=(0.96, 0.12), xytext=(0.96, 0.08),
# #                 xycoords='axes fraction',
# #                 fontsize=12, color='white', ha='center', fontweight='bold',
# #                 arrowprops=dict(arrowstyle='->', color='white', lw=2),
# #                 zorder=30)

# #     ax.set_xlim(0, SZ); ax.set_ylim(SZ, 0)
# #     ax.axis('off')
# #     plt.tight_layout()

# #     # ── Save ──────────────────────────────────────────────────────────────
# #     out = f"forecast_{fh}h_{t_name}_{t_date}.png"
# #     plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='black')
# #     plt.close()
# #     print(f" Saved → {out}\n")


# # # ── CLI ───────────────────────────────────────────────────────────────────────

# # if __name__ == '__main__':
# #     p = argparse.ArgumentParser(description='TC-FlowMatching Forecast Visualisation (FIXED)')
# #     p.add_argument('--model_path',    required=True,  help='Path to best_model.pth')
# #     p.add_argument('--TC_data_path',  required=True,  help='TCND_vn root directory')
# #     p.add_argument('--himawari_path', required=True,  help='Himawari image directory')
# #     p.add_argument('--tc_name',       default='WIPHA')
# #     p.add_argument('--tc_date',       default='2019073106',
# #                    help='Thời điểm bắt đầu dự báo (obs_len cuối = thời điểm này)')
# #     p.add_argument('--test_year',     type=int,   default=2019)
# #     p.add_argument('--obs_len',       type=int,   default=8)
# #     p.add_argument('--pred_len',      type=int,   default=12,
# #                    help='Tự động detect từ checkpoint nếu khác')
# #     p.add_argument('--dset_type',     default='test')
# #     p.add_argument('--batch_size',    type=int,   default=1)
# #     p.add_argument('--delim',         default=' ')
# #     p.add_argument('--skip',          type=int,   default=1)
# #     p.add_argument('--min_ped',       type=int,   default=1)
# #     p.add_argument('--threshold',     type=float, default=0.002)
# #     p.add_argument('--other_modal',   default='gph')
# #     visualize_forecast(p.parse_args())

# # # # __________________ new version: scripts/visual_evaluate_model_Me.py ____
# # # """
# # # scripts/visual_evaluate_model_Me.py
# # # ====================================
# # # TC-FlowMatching 72h Forecast Visualisation  ── FIXED VERSION

# # # FIXES:
# # # 1. Proper DDPM sampling (không dùng physics model đơn giản)
# # # 2. Coordinate system đúng: LONG → X (East=Right), LAT → Y (North=Up)
# # # 3. Anchor: trajectory bắt đầu từ current position
# # # 4. detect_pred_len() dùng đúng key 'denoiser.pos_enc'
# # # 5. Scale và offset tính lại đúng để track khớp satellite image
# # # """

# # # import os
# # # import sys
# # # import random
# # # import argparse

# # # import numpy as np
# # # import torch
# # # import matplotlib


# # # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# # # sys.path.insert(0, project_root)

# # # from Model.flow_matching_model import TCFlowMatching
# # # matplotlib.use('Agg')
# # # import matplotlib.pyplot as plt
# # # import cv2
# # # from datetime import datetime

# # # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# # # sys.path.insert(0, project_root)


# # # from Model.data.loader import data_loader
# # # from Model.data.trajectoriesWithMe_unet_training import seq_collate


# # # # ── Seed ─────────────────────────────────────────────────────────────────────

# # # def set_seed(s=42):
# # #     random.seed(s); np.random.seed(s)
# # #     torch.manual_seed(s); torch.cuda.manual_seed_all(s)
# # #     torch.backends.cudnn.deterministic = True
# # #     torch.backends.cudnn.benchmark     = False
# # #     print(f"🔒 Seed fixed = {s}\n")


# # # # ── Device helpers ────────────────────────────────────────────────────────────

# # # def move_batch(batch, device):
# # #     out = list(batch)
# # #     for i, x in enumerate(out):
# # #         if torch.is_tensor(x):
# # #             out[i] = x.to(device)
# # #         elif isinstance(x, dict):
# # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # #                       for k, v in x.items()}
# # #     return tuple(out)


# # # # ── Checkpoint helpers ────────────────────────────────────────────────────────

# # # def detect_pred_len(ckpt_path):
# # #     """
# # #     Đọc pred_len từ pos_enc shape trong checkpoint.
# # #     KEY PHẢI KHỚP với DeterministicDenoiser: 'denoiser.pos_enc'
# # #     """
# # #     ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
# # #     sd = ck.get('model_state_dict', ck.get('model_state', ck))

# # #     # Thử đúng key trước
# # #     for key in ['denoiser.pos_enc', 'denoiser.pos_encoding']:
# # #         if key in sd:
# # #             return sd[key].shape[1]

# # #     # Fallback: tìm bất kỳ pos_enc nào
# # #     for k, v in sd.items():
# # #         if 'pos_enc' in k and v.dim() == 3:
# # #             return v.shape[1]

# # #     print("  Không tìm thấy pos_enc key, dùng pred_len=12")
# # #     return 12


# # # # ── Denormalization ────────────────────────────────────────────────────────────

# # # def denorm(norm_traj):
# # #     """
# # #     [N, 2] normalised → [N, 2] real (0.1° units)
# # #     LONG = norm * 50 + 1800  (0.1°E)
# # #     LAT  = norm * 50          (0.1°N)
# # #     """
# # #     r = np.zeros_like(norm_traj)
# # #     r[:, 0] = norm_traj[:, 0] * 50.0 + 1800.0   # LONG 0.1°E
# # #     r[:, 1] = norm_traj[:, 1] * 50.0             # LAT  0.1°N
# # #     return r


# # # def real_to_deg(pts_01):
# # #     """Convert 0.1° units → degrees"""
# # #     return pts_01 / 10.0


# # # # ── Pixel mapping ─────────────────────────────────────────────────────────────

# # # def to_pixels(coords_deg, ref_deg, ppu, cx, cy):
# # #     """
# # #     [N, 2] degrees (LONG, LAT) → screen pixels
# # #     LONG increases East → screen X increases right (+)
# # #     LAT  increases North → screen Y increases UP, but screen Y is inverted → (-)
# # #     """
# # #     d_long = coords_deg[:, 0] - ref_deg[0]   # dương = Đông
# # #     d_lat  = coords_deg[:, 1] - ref_deg[1]   # dương = Bắc

# # #     px = cx + d_long * ppu   # East  → right
# # #     py = cy - d_lat  * ppu   # North → up (Y inverted on screen)
# # #     return px, py


# # # # ── Satellite image ───────────────────────────────────────────────────────────

# # # def load_himawari(him_path, year, name, timestamp):
# # #     name  = name.strip().upper()
# # #     exact = os.path.join(him_path, str(year), name, f"{timestamp}.png")
# # #     if os.path.exists(exact):
# # #         img = cv2.imread(exact)
# # #         if img is not None:
# # #             print(f"🛰️  Loaded: {exact}")
# # #             return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # #     d = os.path.join(him_path, str(year), name)
# # #     if os.path.exists(d):
# # #         pngs = sorted(f for f in os.listdir(d) if f.endswith('.png'))
# # #         if pngs:
# # #             tgt  = datetime.strptime(timestamp, '%Y%m%d%H')
# # #             best = min(pngs, key=lambda f: abs(
# # #                 (datetime.strptime(f[:-4], '%Y%m%d%H') - tgt).total_seconds()))
# # #             path = os.path.join(d, best)
# # #             img  = cv2.imread(path)
# # #             if img is not None:
# # #                 print(f"🛰️  Nearest: {path}")
# # #                 return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # #     print("  Himawari image not found – black background")
# # #     return np.zeros((1000, 1000, 3), dtype=np.uint8)


# # # # ── Main ──────────────────────────────────────────────────────────────────────

# # # def visualize_forecast(args):
# # #     set_seed(42)
# # #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # #     print(f"{'='*65}")
# # #     print(f"  TC-FlowMatching Forecast  |  {args.tc_name}  @  {args.tc_date}")
# # #     print(f"{'='*65}\n")

# # #     # ── Auto-detect pred_len ──────────────────────────────────────────────
# # #     detected = detect_pred_len(args.model_path)
# # #     if args.pred_len != detected:
# # #         print(f"  pred_len: {args.pred_len} → {detected} (from checkpoint)")
# # #         args.pred_len = detected

# # #     # ── Load model ────────────────────────────────────────────────────────
# # #     model = TCFlowMatching(pred_len=args.pred_len, obs_len=args.obs_len).to(device)
# # #     ck    = torch.load(args.model_path, map_location=device, weights_only=False)
# # #     sd      = ck.get('model_state_dict', ck.get('model_state', ck))
# # #     missing, unexpected = model.load_state_dict(sd, strict=False)
# # #     if missing:
# # #         print(f"  Missing keys (new layers, random init): {len(missing)}")
# # #     if unexpected:
# # #         print(f"  Unexpected keys (old layers, ignored): {len(unexpected)}")
# # #     model.eval()
# # #     print(" Model loaded\n")

# # #     # ── Load dataset ──────────────────────────────────────────────────────
# # #     dset, _ = data_loader(
# # #         args,
# # #         {'root': args.TC_data_path, 'type': args.dset_type},
# # #         test=True,
# # #         test_year=args.test_year,
# # #     )
# # #     print(f" Dataset: {len(dset)} samples\n")

# # #     # ── Find typhoon sequence ─────────────────────────────────────────────
# # #     t_name  = args.tc_name.strip().upper()
# # #     t_date  = str(args.tc_date).strip()
# # #     target  = None

# # #     for i in range(len(dset)):
# # #         item = dset[i]
# # #         info = item[-1]
# # #         if (t_name in str(info['old'][1]).strip().upper()
# # #                 and t_date == str(info['tydate'][args.obs_len]).strip()):
# # #             target = item
# # #             print(f" Found: {info['old'][1]}  @  {info['tydate'][args.obs_len]}\n")
# # #             break

# # #     if target is None:
# # #         print(f" '{t_name} @ {t_date}' not found. Check --tc_name and --tc_date.")
# # #         # Gợi ý các sample có sẵn
# # #         print("Available samples (first 10):")
# # #         for i in range(min(10, len(dset))):
# # #             info = dset[i][-1]
# # #             print(f"  [{i}] {info['old'][1]} @ {info['tydate'][args.obs_len]}")
# # #         return

# # #     # ── Build batch ───────────────────────────────────────────────────────
# # #     batch = move_batch(seq_collate([target]), device)

# # #     # ── Inference: proper DDPM sampling ──────────────────────────────────
# # #     print(" Running DDPM reverse diffusion sampling...")
# # #     pred_traj_t, pred_Me_t = model.sample(batch)  # [T_pred, B, 2]
# # #     print(" Sampling done\n")

# # #     # ── Extract arrays ────────────────────────────────────────────────────
# # #     obs_n  = batch[0][:, 0, :].cpu().numpy()    # [T_obs,  2] normalized
# # #     gt_n   = batch[1][:, 0, :].cpu().numpy()    # [T_pred, 2] normalized
# # #     pred_n = pred_traj_t[:, 0, :].cpu().numpy() # [T_pred, 2] normalized

# # #     obs_r  = denorm(obs_n)   # [T_obs,  2] 0.1° units
# # #     gt_r   = denorm(gt_n)    # [T_pred, 2] 0.1° units
# # #     pred_r = denorm(pred_n)  # [T_pred, 2] 0.1° units

# # #     # Convert sang degrees để hiển thị
# # #     obs_deg  = real_to_deg(obs_r)   # [T_obs,  2] degrees
# # #     gt_deg   = real_to_deg(gt_r)    # [T_pred, 2] degrees
# # #     pred_deg = real_to_deg(pred_r)  # [T_pred, 2] degrees

# # #     # ── Error report ──────────────────────────────────────────────────────
# # #     # Error tính theo 0.1° units, 1 unit ≈ 11.1 km
# # #     errors_km = np.linalg.norm(gt_r - pred_r, axis=1) * 11.1

# # #     print(" Track errors:")
# # #     for i, e in enumerate(errors_km):
# # #         mark = "  ◀" if (i + 1) in [4, 8, 12] else ""
# # #         print(f"   +{(i+1)*6:3d}h : {e:6.1f} km{mark}")
# # #     print(f"\n   Mean : {errors_km.mean():6.1f} km")
# # #     n = len(errors_km)
# # #     if n >= 4:  print(f"   24h  : {errors_km[3]:6.1f} km")
# # #     if n >= 8:  print(f"   48h  : {errors_km[7]:6.1f} km")
# # #     if n >= 12: print(f"   72h  : {errors_km[11]:6.1f} km")
# # #     print()

# # #     # ── Coordinate ranges ─────────────────────────────────────────────────
# # #     ref_r   = obs_r[-1]                        # last observed (0.1° units)
# # #     ref_deg = real_to_deg(ref_r)               # degrees

# # #     print(f"  Current position: LONG={ref_deg[0]:.2f}°E, LAT={ref_deg[1]:.2f}°N")
# # #     print(f"   GT 72h: LONG={gt_deg[-1,0]:.2f}°E, LAT={gt_deg[-1,1]:.2f}°N")
# # #     print(f"   PD 72h: LONG={pred_deg[-1,0]:.2f}°E, LAT={pred_deg[-1,1]:.2f}°N\n")

# # #     # ── Satellite background ──────────────────────────────────────────────
# # #     sat = load_himawari(args.himawari_path, args.test_year, t_name, t_date)

# # #     SZ  = 1000
# # #     sat = cv2.resize(sat, (SZ, SZ))
# # #     cx, cy = SZ / 2.0, SZ / 2.0

# # #     # ── Scale: tính ppu (pixels per 0.1°) ────────────────────────────────
# # #     # Gom tất cả điểm để tính span
# # #     all_pts_deg = np.vstack([obs_deg, gt_deg, pred_deg])
# # #     span_long   = all_pts_deg[:, 0].max() - all_pts_deg[:, 0].min()
# # #     span_lat    = all_pts_deg[:, 1].max() - all_pts_deg[:, 1].min()
# # #     span_max    = max(span_long, span_lat, 5.0)  # tối thiểu 5°

# # #     # Dùng 65% canvas để hiển thị toàn bộ track
# # #     ppu = (SZ * 0.65) / span_max  # pixels per degree

# # #     # ── Convert to screen pixels ──────────────────────────────────────────
# # #     def px(pts_deg):
# # #         return to_pixels(pts_deg, ref_deg, ppu, cx, cy)

# # #     ox, oy  = px(obs_deg)

# # #     # Ground truth bắt đầu từ current position
# # #     gt_full  = np.vstack([ref_deg.reshape(1, -1), gt_deg])
# # #     gx, gy   = px(gt_full)

# # #     # Prediction bắt đầu từ current position
# # #     pred_full = np.vstack([ref_deg.reshape(1, -1), pred_deg])
# # #     bx, by    = px(pred_full)

# # #     # ── Plot ──────────────────────────────────────────────────────────────
# # #     fig, ax = plt.subplots(figsize=(16, 16), facecolor='black')
# # #     ax.set_facecolor('black')
# # #     ax.imshow(sat, extent=[0, SZ, SZ, 0], alpha=0.60, zorder=0)

# # #     # 1. Observed (cyan)
# # #     ax.plot(ox, oy, 'o-', color='#00FFFF', linewidth=2.5, markersize=5,
# # #             markeredgecolor='white', markeredgewidth=1.2,
# # #             label=f'Observed ({args.obs_len * 6}h)', zorder=8, alpha=0.9)

# # #     # 2. Ground truth (đỏ)
# # #     ax.plot(gx, gy, 'o-', color='#FF2222', linewidth=4, markersize=8,
# # #             markeredgecolor='white', markeredgewidth=2,
# # #             label=f'Actual track ({args.pred_len * 6}h)', zorder=9, alpha=0.95)

# # #     # 3. Prediction (xanh lá)
# # #     ax.plot(bx, by, 'o-', color='#00FF44', linewidth=4, markersize=8,
# # #             markeredgecolor='#004400', markeredgewidth=2,
# # #             label=f'Predicted track ({args.pred_len * 6}h)', zorder=10, alpha=0.95)

# # #     # 4. Error connectors tại 24h / 48h / 72h
# # #     for step_idx, label_h in [(4, 24), (8, 48), (12, 72)]:
# # #         si = step_idx  # index trong full array (0=NOW, 1=+6h, ...)
# # #         if si < len(gx) and si < len(bx):
# # #             ax.plot([gx[si], bx[si]], [gy[si], by[si]],
# # #                     '--', color='#FFD700', linewidth=1.8, alpha=0.65, zorder=7)
# # #             # Ghi khoảng cách error
# # #             if step_idx - 1 < len(errors_km):
# # #                 mid_x = (gx[si] + bx[si]) / 2
# # #                 mid_y = (gy[si] + by[si]) / 2
# # #                 ax.text(mid_x, mid_y, f'{errors_km[step_idx-1]:.0f}km',
# # #                         fontsize=8, color='#FFD700', ha='center',
# # #                         bbox=dict(fc='black', alpha=0.6, ec='none', pad=1), zorder=18)

# # #     # 5. Time labels trên track xanh (prediction)
# # #     for i in range(len(bx)):
# # #         h = i * 6
# # #         if i == 0:
# # #             lbl, col, fs = 'NOW', 'white', 11
# # #         elif h % 12 == 0:
# # #             e_km = errors_km[i - 1] if i > 0 and i - 1 < len(errors_km) else 0
# # #             lbl  = f'+{h}h\n{e_km:.0f}km'
# # #             col  = '#AAFF66'
# # #             fs   = 9
# # #         else:
# # #             continue
# # #         ax.text(bx[i], by[i] - 28, lbl,
# # #                 fontsize=fs, color=col, ha='center', fontweight='bold',
# # #                 bbox=dict(boxstyle='round,pad=0.4', facecolor='black',
# # #                           alpha=0.82, edgecolor=col, linewidth=1.5),
# # #                 zorder=16)

# # #     # 6. Hướng mũi tên cho track dự đoán (để thấy hướng di chuyển)
# # #     for i in range(0, len(bx) - 1, 2):
# # #         dx = bx[i+1] - bx[i]
# # #         dy = by[i+1] - by[i]
# # #         if abs(dx) + abs(dy) > 5:
# # #             ax.annotate('', xy=(bx[i+1], by[i+1]), xytext=(bx[i], by[i]),
# # #                        arrowprops=dict(arrowstyle='->', color='#00FF44',
# # #                                       lw=1.5, mutation_scale=15),
# # #                        zorder=11)

# # #     # 7. Current position ★
# # #     ax.scatter([cx], [cy], color='#FFD700', marker='*', s=900,
# # #                edgecolors='#FF4400', linewidths=3, zorder=25,
# # #                label='Current position')

# # #     # ── Title ─────────────────────────────────────────────────────────────
# # #     dt_str = datetime.strptime(t_date, '%Y%m%d%H').strftime('%d %b %Y  %H:%M UTC')
# # #     fh     = args.pred_len * 6
# # #     mean_e = errors_km.mean()
# # #     last_e = errors_km[-1]

# # #     ax.set_title(
# # #         f"  {t_name}  –  {fh}h TC-FlowMatching Forecast\n"
# # #         f"  {dt_str}    │    Mean: {mean_e:.0f} km    │    {fh}h: {last_e:.0f} km",
# # #         fontsize=17, fontweight='bold', color='white', pad=18,
# # #         bbox=dict(boxstyle='round,pad=0.9', facecolor='#000000',
# # #                   alpha=0.92, edgecolor='#00FFFF', linewidth=2.5),
# # #     )

# # #     # ── Legend ────────────────────────────────────────────────────────────
# # #     ax.legend(loc='upper right', fontsize=12, framealpha=0.92,
# # #               facecolor='#111111', edgecolor='#00FFFF',
# # #               labelcolor='white', title='Track Legend',
# # #               title_fontsize=13)

# # #     # ── Info panel (lower-left) ───────────────────────────────────────────
# # #     lines = [
# # #         "Model : TC-FlowMatching (DDPM)",
# # #         f"Obs   : {args.obs_len} × 6h = {args.obs_len*6}h",
# # #         f"Pred  : {args.pred_len} × 6h = {fh}h",
# # #         f"Ref   : {ref_deg[0]:.1f}°E  {ref_deg[1]:.1f}°N",
# # #         "",
# # #         "Track Errors (km):",
# # #     ]
# # #     for i, e in enumerate(errors_km):
# # #         h = (i + 1) * 6
# # #         if h in [12, 24, 48, 72] and h <= fh:
# # #             lines.append(f"  {h:3d}h : {e:6.1f}")
# # #     lines.append(f"  Mean : {mean_e:6.1f}")

# # #     ax.text(0.02, 0.02, '\n'.join(lines),
# # #             transform=ax.transAxes, fontsize=10, va='bottom',
# # #             family='monospace', color='#88FF88',
# # #             bbox=dict(boxstyle='round,pad=0.6', facecolor='black',
# # #                       alpha=0.88, edgecolor='white', linewidth=1.5),
# # #             zorder=20)

# # #     # ── Compass rose (nhỏ, góc dưới phải) ────────────────────────────────
# # #     ax.annotate('N', xy=(0.96, 0.12), xytext=(0.96, 0.08),
# # #                 xycoords='axes fraction',
# # #                 fontsize=12, color='white', ha='center', fontweight='bold',
# # #                 arrowprops=dict(arrowstyle='->', color='white', lw=2),
# # #                 zorder=30)

# # #     ax.set_xlim(0, SZ); ax.set_ylim(SZ, 0)
# # #     ax.axis('off')
# # #     plt.tight_layout()

# # #     # ── Save ──────────────────────────────────────────────────────────────
# # #     out = f"forecast_{fh}h_{t_name}_{t_date}.png"
# # #     plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='black')
# # #     plt.close()
# # #     print(f" Saved → {out}\n")


# # # # ── CLI ───────────────────────────────────────────────────────────────────────

# # # if __name__ == '__main__':
# # #     p = argparse.ArgumentParser(description='TC-FlowMatching Forecast Visualisation (FIXED)')
# # #     p.add_argument('--model_path',    required=True,  help='Path to best_model.pth')
# # #     p.add_argument('--TC_data_path',  required=True,  help='TCND_vn root directory')
# # #     p.add_argument('--himawari_path', required=True,  help='Himawari image directory')
# # #     p.add_argument('--tc_name',       default='WIPHA')
# # #     p.add_argument('--tc_date',       default='2019073106',
# # #                    help='Thời điểm bắt đầu dự báo (obs_len cuối = thời điểm này)')
# # #     p.add_argument('--test_year',     type=int,   default=2019)
# # #     p.add_argument('--obs_len',       type=int,   default=8)
# # #     p.add_argument('--pred_len',      type=int,   default=12,
# # #                    help='Tự động detect từ checkpoint nếu khác')
# # #     p.add_argument('--dset_type',     default='test')
# # #     p.add_argument('--batch_size',    type=int,   default=1)
# # #     p.add_argument('--delim',         default=' ')
# # #     p.add_argument('--skip',          type=int,   default=1)
# # #     p.add_argument('--min_ped',       type=int,   default=1)
# # #     p.add_argument('--threshold',     type=float, default=0.002)
# # #     p.add_argument('--other_modal',   default='gph')
# # #     visualize_forecast(p.parse_args())

# """
# scripts/visual_evaluate_model_Me.py  ── v9
# ==========================================
# TC-FlowMatching 72h Forecast Visualisation.

# FIXES v9:
# - Falls back to drawn terrain map when Himawari image not found
# - Correct coordinate system: LON→X (East=Right), LAT→Y (North=Up)
# - Anchor: trajectory starts from current (last observed) position
# - detect_pred_len() reads correct key 'net.pos_enc' from checkpoint
# - Case-study grid: straight-track×2 + recurvature×1 (WIPHA)
# """
# from __future__ import annotations

# import os
# import sys
# import random
# import argparse
# from datetime import datetime
# from typing import Optional

# import numpy as np

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, project_root)

# import torch
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.patheffects as pe
# from matplotlib.lines import Line2D

# try:
#     import cv2
#     HAS_CV2 = True
# except ImportError:
#     HAS_CV2 = False

# try:
#     import cartopy.crs as ccrs
#     import cartopy.feature as cfeature
#     HAS_CARTOPY = True
# except ImportError:
#     HAS_CARTOPY = False

# from Model.flow_matching_model import TCFlowMatching
# from Model.data.loader import data_loader
# from Model.data.trajectoriesWithMe_unet_training import seq_collate


# # ══════════════════════════════════════════════════════════════════════════════
# #  Utilities
# # ══════════════════════════════════════════════════════════════════════════════

# def set_seed(s: int = 42):
#     random.seed(s)
#     np.random.seed(s)
#     torch.manual_seed(s)
#     torch.cuda.manual_seed_all(s)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark     = False


# def move_batch(batch, device):
#     out = list(batch)
#     for i, x in enumerate(out):
#         if torch.is_tensor(x):
#             out[i] = x.to(device)
#         elif isinstance(x, dict):
#             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
#                       for k, v in x.items()}
#     return tuple(out)


# # ── Denorm ────────────────────────────────────────────────────────────────────

# def denorm(norm_traj: np.ndarray) -> np.ndarray:
#     """[N, 2] normalised → [N, 2] in 0.1° units (lon, lat)."""
#     r = np.zeros_like(norm_traj)
#     r[:, 0] = norm_traj[:, 0] * 50.0 + 1800.0
#     r[:, 1] = norm_traj[:, 1] * 50.0
#     return r


# def to_deg(pts_01: np.ndarray) -> np.ndarray:
#     return pts_01 / 10.0


# # ── Checkpoint helpers ────────────────────────────────────────────────────────

# def detect_pred_len(ckpt_path: str) -> int:
#     ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
#     sd = ck.get("model_state_dict", ck.get("model_state", ck))
#     for key in ["net.pos_enc", "denoiser.pos_enc", "pos_enc"]:
#         if key in sd:
#             return sd[key].shape[1]
#     for k, v in sd.items():
#         if "pos_enc" in k and hasattr(v, "dim") and v.dim() == 3:
#             return v.shape[1]
#     print("  pos_enc not found → default pred_len=12")
#     return 12


# # ══════════════════════════════════════════════════════════════════════════════
# #  Background image helpers
# # ══════════════════════════════════════════════════════════════════════════════

# def load_himawari(him_path: str, year: int, name: str, timestamp: str
#                   ) -> Optional[np.ndarray]:
#     """Try to load Himawari image. Returns None if not found."""
#     name = name.strip().upper()
#     exact = os.path.join(him_path, str(year), name, f"{timestamp}.png")
#     if os.path.exists(exact):
#         img = None
#         if HAS_CV2:
#             img = cv2.imread(exact)
#         if img is not None:
#             return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     d = os.path.join(him_path, str(year), name)
#     if os.path.exists(d):
#         pngs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
#         if pngs:
#             tgt  = datetime.strptime(timestamp, "%Y%m%d%H")
#             best = min(pngs, key=lambda f: abs(
#                 (datetime.strptime(f[:-4], "%Y%m%d%H") - tgt).total_seconds()))
#             path = os.path.join(d, best)
#             img  = cv2.imread(path) if HAS_CV2 else None
#             if img is not None:
#                 return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return None


# def draw_terrain_map(
#     ax: plt.Axes,
#     lon_range: tuple[float, float],
#     lat_range: tuple[float, float],
# ):
#     """
#     Draw a simple terrain background map when Himawari is unavailable.
#     Uses Cartopy if available, otherwise draws a basic ocean/land rectangle.
#     """
#     ax.set_facecolor("#1a3a5c")   # deep ocean blue

#     if HAS_CARTOPY:
#         # Use a GeoAxes-compatible approach via a secondary axis
#         # Since we can't replace the axis type, draw features manually
#         pass

#     # Draw ocean gradient
#     lon0, lon1 = lon_range
#     lat0, lat1 = lat_range
#     ax.set_xlim(lon0, lon1)
#     ax.set_ylim(lat0, lat1)

#     # Grid lines
#     for lon in np.arange(np.ceil(lon0 / 5) * 5, lon1, 5):
#         ax.axvline(lon, color="white", alpha=0.15, linewidth=0.5)
#     for lat in np.arange(np.ceil(lat0 / 5) * 5, lat1, 5):
#         ax.axhline(lat, color="white", alpha=0.15, linewidth=0.5)

#     # Simplified land masses for SCS region (polygon approximations)
#     land_patches = [
#         # Vietnam coastline (approx)
#         plt.Polygon([(102, 22), (108, 22), (109, 16), (106, 10), (104, 8),
#                      (103, 10), (102, 15)], closed=True,
#                     fc="#4a7c59", ec="#3a6b48", lw=0.5, alpha=0.8),
#         # Philippines (approx)
#         plt.Polygon([(118, 18), (122, 18), (126, 15), (126, 8), (122, 6),
#                      (119, 8), (117, 12)], closed=True,
#                     fc="#4a7c59", ec="#3a6b48", lw=0.5, alpha=0.8),
#         # Taiwan
#         plt.Polygon([(120, 25.5), (122, 25.5), (122, 22), (120, 22)], closed=True,
#                     fc="#4a7c59", ec="#3a6b48", lw=0.5, alpha=0.8),
#         # Hainan
#         plt.Polygon([(108.5, 20.3), (111.2, 20.3), (111.2, 18), (108.5, 18)],
#                     closed=True, fc="#4a7c59", ec="#3a6b48", lw=0.5, alpha=0.8),
#         # Southern China coast
#         plt.Polygon([(108, 22), (116, 24), (120, 24), (120, 22), (113, 21),
#                      (110, 21)], closed=True,
#                     fc="#4a7c59", ec="#3a6b48", lw=0.5, alpha=0.8),
#     ]
#     for patch in land_patches:
#         ax.add_patch(patch)

#     # Tick labels
#     xticks = np.arange(np.ceil(lon0 / 10) * 10, lon1 + 1, 10)
#     yticks = np.arange(np.ceil(lat0 / 5)  * 5,  lat1 + 1, 5)
#     ax.set_xticks(xticks)
#     ax.set_yticks(yticks)
#     ax.set_xticklabels([f"{x:.0f}°E" for x in xticks],
#                        color="white", fontsize=7)
#     ax.set_yticklabels([f"{y:.0f}°N" for y in yticks],
#                        color="white", fontsize=7)
#     ax.tick_params(colors="white", length=3)
#     for spine in ax.spines.values():
#         spine.set_edgecolor("white")


# # ══════════════════════════════════════════════════════════════════════════════
# #  Core plot function
# # ══════════════════════════════════════════════════════════════════════════════

# def plot_forecast(
#     ax:         plt.Axes,
#     obs_deg:    np.ndarray,    # [T_obs,  2] degrees (lon, lat)
#     gt_deg:     np.ndarray,    # [T_pred, 2] degrees
#     pred_deg:   np.ndarray,    # [T_pred, 2] degrees
#     title:      str,
#     sat_img:    Optional[np.ndarray],
#     lon_range:  tuple[float, float],
#     lat_range:  tuple[float, float],
#     errors_km:  Optional[np.ndarray] = None,
#     pred_deg2:  Optional[np.ndarray] = None,   # second model (Diffusion)
#     label2:     str = "Diffusion",
# ):
#     """Draw one forecast panel."""
#     ax.set_facecolor("#0d1b2a")

#     if sat_img is not None:
#         # Satellite background
#         ax.imshow(
#             sat_img,
#             extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
#             aspect="auto", alpha=0.55, zorder=0, origin="upper",
#         )
#         ax.set_xlim(*lon_range)
#         ax.set_ylim(*lat_range)
#     else:
#         draw_terrain_map(ax, lon_range, lat_range)

#     # Plotting helpers
#     pe_outline = [pe.withStroke(linewidth=3, foreground="black")]

#     # 1. Observed track (cyan)
#     ax.plot(obs_deg[:, 0], obs_deg[:, 1],
#             "o-", color="#00FFFF", lw=2, ms=5,
#             mec="white", mew=1.0, label="Observed", zorder=8,
#             path_effects=pe_outline)

#     # Current position ★
#     cur = obs_deg[-1]
#     ax.scatter([cur[0]], [cur[1]], s=400, marker="*",
#                color="#FFD700", edgecolors="#FF4400", lw=2, zorder=20,
#                label="Now")

#     # Prepend current position to gt & pred
#     gt_full   = np.vstack([cur.reshape(1, 2), gt_deg])
#     pred_full = np.vstack([cur.reshape(1, 2), pred_deg])

#     # 2. Ground truth (red)
#     ax.plot(gt_full[:, 0], gt_full[:, 1],
#             "o-", color="#FF3333", lw=3.5, ms=7,
#             mec="white", mew=1.5, label="Ground Truth", zorder=9,
#             path_effects=pe_outline)

#     # 3. FM+PINN prediction (green)
#     ax.plot(pred_full[:, 0], pred_full[:, 1],
#             "o-", color="#00FF66", lw=3.5, ms=7,
#             mec="#003300", mew=1.5, label="FM+PINN", zorder=10,
#             path_effects=pe_outline)

#     # 4. Second model (optional, orange)
#     if pred_deg2 is not None:
#         pred_full2 = np.vstack([cur.reshape(1, 2), pred_deg2])
#         ax.plot(pred_full2[:, 0], pred_full2[:, 1],
#                 "s--", color="#FF9900", lw=2.5, ms=6,
#                 mec="#552200", mew=1.0, label=label2, zorder=9,
#                 path_effects=pe_outline)

#     # 5. Error connectors at 24/48/72h
#     for step_idx, lh in [(4, 24), (8, 48), (12, 72)]:
#         si = step_idx  # index in full array (0=NOW)
#         if si < len(gt_full) and si < len(pred_full):
#             ax.plot([gt_full[si, 0], pred_full[si, 0]],
#                     [gt_full[si, 1], pred_full[si, 1]],
#                     "--", color="#FFD700", lw=1.5, alpha=0.7, zorder=7)
#             if errors_km is not None and step_idx - 1 < len(errors_km):
#                 mx = (gt_full[si, 0] + pred_full[si, 0]) / 2
#                 my = (gt_full[si, 1] + pred_full[si, 1]) / 2
#                 ax.text(mx, my, f"{errors_km[step_idx-1]:.0f}km",
#                         fontsize=7, color="#FFD700", ha="center", zorder=18,
#                         bbox=dict(fc="black", alpha=0.6, ec="none", pad=1))

#     # 6. Lead-time labels on FM+PINN track
#     for i, pt in enumerate(pred_full):
#         h = i * 6
#         if i == 0:
#             lbl, col, fs = "NOW", "white", 9
#         elif h % 24 == 0:
#             lbl  = f"+{h}h"
#             col  = "#AAFFAA"
#             fs   = 8
#         else:
#             continue
#         ax.text(pt[0], pt[1] + 0.5, lbl,
#                 fontsize=fs, color=col, ha="center", fontweight="bold",
#                 bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75,
#                           ec=col, lw=1.0), zorder=16)

#     # 7. Compass rose (small, bottom-right)
#     ax.annotate("N", xy=(0.96, 0.12), xytext=(0.96, 0.08),
#                 xycoords="axes fraction", fontsize=11, color="white",
#                 ha="center", fontweight="bold", zorder=30,
#                 arrowprops=dict(arrowstyle="->", color="white", lw=1.8))

#     # Error summary text
#     if errors_km is not None:
#         n = len(errors_km)
#         lines = [f"Mean: {errors_km.mean():.0f} km"]
#         for step, lh in [(3, 24), (7, 48), (11, 72)]:
#             if step < n:
#                 lines.append(f"{lh}h: {errors_km[step]:.0f} km")
#         ax.text(0.02, 0.02, "\n".join(lines),
#                 transform=ax.transAxes, fontsize=8, va="bottom",
#                 color="#88FF88", family="monospace",
#                 bbox=dict(boxstyle="round,pad=0.4", fc="black", alpha=0.8,
#                           ec="white", lw=1.0), zorder=20)

#     ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6,
#                  bbox=dict(fc="black", alpha=0.8, ec="#00FFFF", lw=1.5))
#     ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85,
#               facecolor="#111111", edgecolor="#00FFFF", labelcolor="white")


# # ══════════════════════════════════════════════════════════════════════════════
# #  Case-study grid  (2 straight + 1 recurvature)
# # ══════════════════════════════════════════════════════════════════════════════

# def build_case_study_grid(
#     args,
#     model: TCFlowMatching,
#     dset,
#     device: torch.device,
#     cases: list[dict],   # each: {name, date, label}
#     out_path: str,
#     him_path: str,
# ):
#     """
#     Produce a 3-row × 2-col figure:
#       col 0: FM+PINN forecast
#       col 1: error curve (lead-time vs km)
#     """
#     n_cases = len(cases)
#     fig, axes = plt.subplots(
#         n_cases, 2,
#         figsize=(18, 6 * n_cases),
#         facecolor="#0d1b2a",
#         gridspec_kw={"width_ratios": [2, 1]},
#     )
#     if n_cases == 1:
#         axes = axes[np.newaxis]

#     for row, case in enumerate(cases):
#         t_name = case["name"].strip().upper()
#         t_date = str(case["date"]).strip()
#         label  = case.get("label", t_name)

#         # ── Find sequence ─────────────────────────────────────────────────
#         target = None
#         for i in range(len(dset)):
#             item = dset[i]
#             info = item[-1]
#             if (t_name in str(info["old"][1]).strip().upper()
#                     and t_date == str(info["tydate"][args.obs_len]).strip()):
#                 target = item
#                 break

#         if target is None:
#             print(f"  ⚠  {t_name} @ {t_date} not found — skipping row {row}")
#             for c in range(2):
#                 axes[row, c].set_facecolor("#0d1b2a")
#                 axes[row, c].text(0.5, 0.5, f"NOT FOUND\n{t_name} @ {t_date}",
#                                   ha="center", va="center", color="red",
#                                   transform=axes[row, c].transAxes)
#             continue

#         batch = move_batch(seq_collate([target]), device)
#         with torch.no_grad():
#             pred_mean, _, _ = model.sample(
#                 batch, num_ensemble=50, ddim_steps=args.ode_steps)

#         obs_n  = batch[0][:, 0, :].cpu().numpy()
#         gt_n   = batch[1][:, 0, :].cpu().numpy()
#         pred_n = pred_mean[:, 0, :].cpu().numpy()

#         obs_deg  = to_deg(denorm(obs_n))
#         gt_deg   = to_deg(denorm(gt_n))
#         pred_deg = to_deg(denorm(pred_n))

#         errors_km = np.linalg.norm(denorm(gt_n) - denorm(pred_n), axis=1) * 11.1

#         # Geographic extent
#         all_deg = np.vstack([obs_deg, gt_deg, pred_deg])
#         margin  = 3.0
#         lon_range = (all_deg[:, 0].min() - margin, all_deg[:, 0].max() + margin)
#         lat_range = (all_deg[:, 1].min() - margin, all_deg[:, 1].max() + margin)

#         # Satellite image
#         sat = load_himawari(him_path, args.test_year, t_name, t_date)
#         if sat is not None and HAS_CV2:
#             sat_resized = cv2.resize(sat, (512, 512))
#         else:
#             sat_resized = None

#         # ── Left panel: map ───────────────────────────────────────────────
#         ax_map = axes[row, 0]
#         ax_map.set_facecolor("#0d1b2a")
#         dt_str = datetime.strptime(t_date, "%Y%m%d%H").strftime("%d %b %Y %H:%M UTC")
#         plot_forecast(
#             ax        = ax_map,
#             obs_deg   = obs_deg,
#             gt_deg    = gt_deg,
#             pred_deg  = pred_deg,
#             title     = f"[{label}] {t_name} — {dt_str}",
#             sat_img   = sat_resized,
#             lon_range = lon_range,
#             lat_range = lat_range,
#             errors_km = errors_km,
#         )

#         # ── Right panel: error curve ──────────────────────────────────────
#         ax_err = axes[row, 1]
#         ax_err.set_facecolor("#0d1b2a")
#         lead_h = np.arange(1, len(errors_km) + 1) * 6
#         ax_err.plot(lead_h, errors_km, "o-", color="#00FF66", lw=2.5,
#                     ms=6, label="FM+PINN", zorder=5)
#         ax_err.fill_between(lead_h, 0, errors_km, alpha=0.15, color="#00FF66")

#         # CLIPER baseline
#         obs_r  = denorm(obs_n)
#         v_cliper = obs_r[-1] - obs_r[-2] if len(obs_r) >= 2 else np.zeros(2)
#         gt_r = denorm(gt_n)
#         cliper_preds = np.array([obs_r[-1] + (k + 1) * v_cliper
#                                  for k in range(len(gt_r))])
#         cliper_err = np.linalg.norm(cliper_preds - gt_r, axis=1) * 11.1
#         ax_err.plot(lead_h, cliper_err[:len(lead_h)], "s--",
#                     color="#FF6666", lw=2, ms=5, label="CLIPER", zorder=4)

#         # Horizontal guides
#         for yl in [100, 200, 300]:
#             ax_err.axhline(yl, color="white", alpha=0.1, lw=0.8)

#         ax_err.set_xlabel("Forecast lead time (h)", color="white", fontsize=9)
#         ax_err.set_ylabel("Track error (km)", color="white", fontsize=9)
#         ax_err.set_title(f"Lead-time error — {t_name}", color="white",
#                          fontsize=10, fontweight="bold")
#         ax_err.legend(fontsize=8, facecolor="#111111", edgecolor="#00FFFF",
#                       labelcolor="white")
#         ax_err.tick_params(colors="white")
#         for spine in ax_err.spines.values():
#             spine.set_edgecolor("white")
#         ax_err.set_facecolor("#0d1b2a")
#         ax_err.yaxis.label.set_color("white")

#     plt.tight_layout(pad=1.5)
#     plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1b2a")
#     plt.close()
#     print(f"  📊  Case study → {out_path}")


# # ══════════════════════════════════════════════════════════════════════════════
# #  Single TC forecast
# # ══════════════════════════════════════════════════════════════════════════════

# def visualize_forecast(args):
#     set_seed(42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print(f"{'='*65}")
#     print(f"  TC-FM v9 Forecast  |  {args.tc_name}  @  {args.tc_date}")
#     print(f"{'='*65}\n")

#     detected = detect_pred_len(args.model_path)
#     if args.pred_len != detected:
#         print(f"  pred_len: {args.pred_len} → {detected}")
#         args.pred_len = detected

#     model = TCFlowMatching(pred_len=args.pred_len, obs_len=args.obs_len).to(device)
#     ck    = torch.load(args.model_path, map_location=device, weights_only=False)
#     sd    = ck.get("model_state_dict", ck.get("model_state", ck))
#     model.load_state_dict(sd, strict=False)
#     model.eval()
#     print("  Model loaded\n")

#     dset, _ = data_loader(
#         args,
#         {"root": args.TC_data_path, "type": args.dset_type},
#         test=True, test_year=args.test_year,
#     )
#     print(f"  Dataset: {len(dset)} samples\n")

#     t_name = args.tc_name.strip().upper()
#     t_date = str(args.tc_date).strip()
#     target = None
#     for i in range(len(dset)):
#         item = dset[i]
#         info = item[-1]
#         if (t_name in str(info["old"][1]).strip().upper()
#                 and t_date == str(info["tydate"][args.obs_len]).strip()):
#             target = item
#             print(f"  Found: {info['old'][1]} @ {info['tydate'][args.obs_len]}\n")
#             break

#     if target is None:
#         print(f"  '{t_name} @ {t_date}' not found.")
#         print("  Available (first 10):")
#         for i in range(min(10, len(dset))):
#             info = dset[i][-1]
#             print(f"    [{i}] {info['old'][1]} @ {info['tydate'][args.obs_len]}")
#         return

#     batch = move_batch(seq_collate([target]), device)
#     with torch.no_grad():
#         pred_mean, _, _ = model.sample(
#             batch, num_ensemble=50, ddim_steps=args.ode_steps)

#     obs_n  = batch[0][:, 0, :].cpu().numpy()
#     gt_n   = batch[1][:, 0, :].cpu().numpy()
#     pred_n = pred_mean[:, 0, :].cpu().numpy()

#     obs_deg  = to_deg(denorm(obs_n))
#     gt_deg   = to_deg(denorm(gt_n))
#     pred_deg = to_deg(denorm(pred_n))

#     errors_km = np.linalg.norm(denorm(gt_n) - denorm(pred_n), axis=1) * 11.1

#     print("  Track errors:")
#     for i, e in enumerate(errors_km):
#         mark = "  ◀" if (i + 1) in [4, 8, 12] else ""
#         print(f"    +{(i+1)*6:3d}h : {e:6.1f} km{mark}")
#     print(f"    Mean  : {errors_km.mean():.1f} km\n")

#     all_deg   = np.vstack([obs_deg, gt_deg, pred_deg])
#     margin    = 3.0
#     lon_range = (all_deg[:, 0].min() - margin, all_deg[:, 0].max() + margin)
#     lat_range = (all_deg[:, 1].min() - margin, all_deg[:, 1].max() + margin)

#     # Try Himawari, fallback to terrain map
#     sat = load_himawari(args.himawari_path, args.test_year, t_name, t_date)
#     sat_resized = None
#     if sat is not None and HAS_CV2:
#         sat_resized = cv2.resize(sat, (512, 512))

#     fig, ax = plt.subplots(figsize=(14, 12), facecolor="#0d1b2a")
#     dt_str  = datetime.strptime(t_date, "%Y%m%d%H").strftime("%d %b %Y %H:%M UTC")
#     plot_forecast(
#         ax        = ax,
#         obs_deg   = obs_deg,
#         gt_deg    = gt_deg,
#         pred_deg  = pred_deg,
#         title     = f"🌀 {t_name} — {args.pred_len * 6}h TC-FM Forecast\n{dt_str}",
#         sat_img   = sat_resized,
#         lon_range = lon_range,
#         lat_range = lat_range,
#         errors_km = errors_km,
#     )

#     fh  = args.pred_len * 6
#     out = f"forecast_{fh}h_{t_name}_{t_date}.png"
#     plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="#0d1b2a")
#     plt.close()
#     print(f"  Saved → {out}\n")


# # ══════════════════════════════════════════════════════════════════════════════
# #  Case-study mode (3 typhoons)
# # ══════════════════════════════════════════════════════════════════════════════

# def visualize_case_study(args):
#     set_seed(42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     detected = detect_pred_len(args.model_path)
#     args.pred_len = detected

#     model = TCFlowMatching(pred_len=args.pred_len, obs_len=args.obs_len).to(device)
#     ck    = torch.load(args.model_path, map_location=device, weights_only=False)
#     sd    = ck.get("model_state_dict", ck.get("model_state", ck))
#     model.load_state_dict(sd, strict=False)
#     model.eval()

#     dset, _ = data_loader(
#         args,
#         {"root": args.TC_data_path, "type": "test"},
#         test=True, test_year=args.test_year,
#     )

#     cases = [
#         {"name": args.straight1_name, "date": args.straight1_date,
#          "label": "Straight-track Case 1"},
#         {"name": args.straight2_name, "date": args.straight2_date,
#          "label": "Straight-track Case 2"},
#         {"name": "WIPHA",             "date": args.recurv_date,
#          "label": "Recurvature Case — WIPHA"},
#     ]

#     out_path = os.path.join(args.output_dir, "case_study_grid.png")
#     os.makedirs(args.output_dir, exist_ok=True)
#     build_case_study_grid(
#         args, model, dset, device, cases,
#         out_path=out_path,
#         him_path=args.himawari_path,
#     )


# # ══════════════════════════════════════════════════════════════════════════════
# #  CLI
# # ══════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--model_path",      required=True)
#     p.add_argument("--TC_data_path",    required=True)
#     p.add_argument("--himawari_path",   default="")
#     p.add_argument("--output_dir",      default="outputs")
#     p.add_argument("--mode",            default="single",
#                    choices=["single", "case_study"])
#     # Single mode
#     p.add_argument("--tc_name",         default="WIPHA")
#     p.add_argument("--tc_date",         default="2019073106")
#     p.add_argument("--dset_type",       default="test")
#     # Case-study mode
#     p.add_argument("--straight1_name",  default="BEBINCA")
#     p.add_argument("--straight1_date",  default="2018090806")
#     p.add_argument("--straight2_name",  default="MANGKHUT")
#     p.add_argument("--straight2_date",  default="2018091312")
#     p.add_argument("--recurv_date",     default="2019073106")
#     # Model params
#     p.add_argument("--test_year",       type=int, default=2019)
#     p.add_argument("--obs_len",         type=int, default=8)
#     p.add_argument("--pred_len",        type=int, default=12)
#     p.add_argument("--ode_steps",       type=int, default=10)
#     p.add_argument("--batch_size",      type=int, default=1)
#     p.add_argument("--delim",           default=" ")
#     p.add_argument("--skip",            type=int, default=1)
#     p.add_argument("--min_ped",         type=int, default=1)
#     p.add_argument("--threshold",       type=float, default=0.002)
#     p.add_argument("--other_modal",     default="gph")

#     args = p.parse_args()
#     if args.mode == "single":
#         visualize_forecast(args)
#     else:
#         visualize_case_study(args)
"""
scripts/visual_evaluate_model_Me.py  ── v9-fixed
=================================================
TC-FlowMatching 72h Forecast Visualisation.

Single mode:
    python scripts/visual_evaluate_model_Me.py \
        --mode single \
        --model_path runs/v9/best_model.pth \
        --TC_data_path /path/to/TCND_vn \
        --tc_name WIPHA \
        --tc_date 2019073106

Case-study grid (2 straight + 1 recurvature):
    python scripts/visual_evaluate_model_Me.py \
        --mode case_study \
        --model_path runs/v9/best_model.pth \
        --TC_data_path /path/to/TCND_vn \
        --output_dir outputs
"""
from __future__ import annotations

import os
import sys
import random
import argparse
from datetime import datetime
from typing import Optional

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from Model.flow_matching_model import TCFlowMatching
from Model.data.loader import data_loader
from Model.data.trajectoriesWithMe_unet_training import seq_collate


# ══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def move_batch(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in x.items()}
    return tuple(out)


# ── Denormalisation ───────────────────────────────────────────────────────────

def denorm(norm_traj: np.ndarray) -> np.ndarray:
    """[N, 2] normalised → [N, 2] in 0.1-degree units (lon, lat)."""
    r = np.zeros_like(norm_traj)
    r[:, 0] = norm_traj[:, 0] * 50.0 + 1800.0
    r[:, 1] = norm_traj[:, 1] * 50.0
    return r


def to_deg(pts_01: np.ndarray) -> np.ndarray:
    """0.1-degree units → degrees."""
    return pts_01 / 10.0


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def detect_pred_len(ckpt_path: str) -> int:
    """Read pred_len from pos_enc shape in checkpoint."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck.get("model_state", ck))
    # VelocityField stores pos_enc under 'net.pos_enc'
    for key in ["net.pos_enc", "denoiser.pos_enc", "pos_enc"]:
        if key in sd:
            return sd[key].shape[1]
    for k, v in sd.items():
        if "pos_enc" in k and hasattr(v, "dim") and v.dim() == 3:
            return v.shape[1]
    print("  pos_enc not found in checkpoint → using pred_len=12")
    return 12


# ══════════════════════════════════════════════════════════════════════════════
#  Background image helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_himawari(him_path: str, year: int, name: str, timestamp: str
                  ) -> Optional[np.ndarray]:
    """Try to load Himawari satellite image. Returns None if unavailable."""
    if not him_path:
        return None
    name  = name.strip().upper()
    exact = os.path.join(him_path, str(year), name, f"{timestamp}.png")
    if os.path.exists(exact) and HAS_CV2:
        img = cv2.imread(exact)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    d = os.path.join(him_path, str(year), name)
    if os.path.exists(d):
        pngs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        if pngs:
            tgt  = datetime.strptime(timestamp, "%Y%m%d%H")
            best = min(pngs, key=lambda f: abs(
                (datetime.strptime(f[:-4], "%Y%m%d%H") - tgt).total_seconds()))
            path = os.path.join(d, best)
            if HAS_CV2:
                img = cv2.imread(path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def draw_terrain_map(
    ax: plt.Axes,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
):
    """Simple ocean+land background when Himawari is unavailable."""
    ax.set_facecolor("#1a3a5c")
    ax.set_xlim(*lon_range)
    ax.set_ylim(*lat_range)

    # Grid lines
    for lon in np.arange(np.ceil(lon_range[0] / 5) * 5, lon_range[1], 5):
        ax.axvline(lon, color="white", alpha=0.15, linewidth=0.5)
    for lat in np.arange(np.ceil(lat_range[0] / 5) * 5, lat_range[1], 5):
        ax.axhline(lat, color="white", alpha=0.15, linewidth=0.5)

    # Simplified land patches for SCS region
    land_patches = [
        plt.Polygon(
            [(102, 22), (108, 22), (109, 16), (106, 10), (104, 8), (103, 10), (102, 15)],
            closed=True, fc="#4a7c59", ec="#3a6b48", lw=0.5, alpha=0.8,
        ),
        plt.Polygon(
            [(118, 18), (122, 18), (126, 15), (126, 8), (122, 6), (119, 8), (117, 12)],
            closed=True, fc="#4a7c59", ec="#3a6b48", lw=0.5, alpha=0.8,
        ),
        plt.Polygon(
            [(120, 25.5), (122, 25.5), (122, 22), (120, 22)],
            closed=True, fc="#4a7c59", ec="#3a6b48", lw=0.5, alpha=0.8,
        ),
        plt.Polygon(
            [(108.5, 20.3), (111.2, 20.3), (111.2, 18), (108.5, 18)],
            closed=True, fc="#4a7c59", ec="#3a6b48", lw=0.5, alpha=0.8,
        ),
        plt.Polygon(
            [(108, 22), (116, 24), (120, 24), (120, 22), (113, 21), (110, 21)],
            closed=True, fc="#4a7c59", ec="#3a6b48", lw=0.5, alpha=0.8,
        ),
    ]
    for patch in land_patches:
        ax.add_patch(patch)

    # Axis labels
    xticks = np.arange(np.ceil(lon_range[0] / 10) * 10, lon_range[1] + 1, 10)
    yticks = np.arange(np.ceil(lat_range[0] / 5)  * 5,  lat_range[1] + 1, 5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{x:.0f}E" for x in xticks], color="white", fontsize=7)
    ax.set_yticklabels([f"{y:.0f}N" for y in yticks], color="white", fontsize=7)
    ax.tick_params(colors="white", length=3)
    for spine in ax.spines.values():
        spine.set_edgecolor("white")


# ══════════════════════════════════════════════════════════════════════════════
#  Core plot function
# ══════════════════════════════════════════════════════════════════════════════

def plot_forecast(
    ax:         plt.Axes,
    obs_deg:    np.ndarray,   # [T_obs, 2]  degrees (lon, lat)
    gt_deg:     np.ndarray,   # [T_pred, 2] degrees
    pred_deg:   np.ndarray,   # [T_pred, 2] degrees
    title:      str,
    sat_img:    Optional[np.ndarray],
    lon_range:  tuple[float, float],
    lat_range:  tuple[float, float],
    errors_km:  Optional[np.ndarray] = None,
    pred_deg2:  Optional[np.ndarray] = None,  # second model (e.g. Diffusion)
    label2:     str = "Diffusion",
):
    """Draw one forecast panel on the given Axes."""
    ax.set_facecolor("#0d1b2a")

    if sat_img is not None:
        ax.imshow(
            sat_img,
            extent=[lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
            aspect="auto", alpha=0.55, zorder=0, origin="upper",
        )
        ax.set_xlim(*lon_range)
        ax.set_ylim(*lat_range)
    else:
        draw_terrain_map(ax, lon_range, lat_range)

    outline = [pe.withStroke(linewidth=3, foreground="black")]
    cur     = obs_deg[-1]

    # 1. Observed track (cyan)
    ax.plot(obs_deg[:, 0], obs_deg[:, 1],
            "o-", color="#00FFFF", lw=2, ms=5,
            mec="white", mew=1.0, label="Observed",
            zorder=8, path_effects=outline)

    # Current position star
    ax.scatter([cur[0]], [cur[1]], s=400, marker="*",
               color="#FFD700", edgecolors="#FF4400", lw=2, zorder=20,
               label="Now")

    gt_full   = np.vstack([cur.reshape(1, 2), gt_deg])
    pred_full = np.vstack([cur.reshape(1, 2), pred_deg])

    # 2. Ground truth (red)
    ax.plot(gt_full[:, 0], gt_full[:, 1],
            "o-", color="#FF3333", lw=3.5, ms=7,
            mec="white", mew=1.5, label="Ground Truth",
            zorder=9, path_effects=outline)

    # 3. FM+PINN prediction (green)
    ax.plot(pred_full[:, 0], pred_full[:, 1],
            "o-", color="#00FF66", lw=3.5, ms=7,
            mec="#003300", mew=1.5, label="FM+PINN",
            zorder=10, path_effects=outline)

    # 4. Optional second model (orange)
    if pred_deg2 is not None:
        pred_full2 = np.vstack([cur.reshape(1, 2), pred_deg2])
        ax.plot(pred_full2[:, 0], pred_full2[:, 1],
                "s--", color="#FF9900", lw=2.5, ms=6,
                mec="#552200", mew=1.0, label=label2,
                zorder=9, path_effects=outline)

    # 5. Error connectors at 24/48/72h
    for step_idx, lh in [(4, 24), (8, 48), (12, 72)]:
        si = step_idx
        if si < len(gt_full) and si < len(pred_full):
            ax.plot([gt_full[si, 0], pred_full[si, 0]],
                    [gt_full[si, 1], pred_full[si, 1]],
                    "--", color="#FFD700", lw=1.5, alpha=0.7, zorder=7)
            if errors_km is not None and step_idx - 1 < len(errors_km):
                mx = (gt_full[si, 0] + pred_full[si, 0]) / 2
                my = (gt_full[si, 1] + pred_full[si, 1]) / 2
                ax.text(mx, my, f"{errors_km[step_idx-1]:.0f}km",
                        fontsize=7, color="#FFD700", ha="center", zorder=18,
                        bbox=dict(fc="black", alpha=0.6, ec="none", pad=1))

    # 6. Lead-time labels on FM+PINN track
    for i, pt in enumerate(pred_full):
        h = i * 6
        if i == 0:
            lbl, col, fs = "NOW", "white", 9
        elif h % 24 == 0:
            lbl, col, fs = f"+{h}h", "#AAFFAA", 8
        else:
            continue
        ax.text(pt[0], pt[1] + 0.5, lbl,
                fontsize=fs, color=col, ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75,
                          ec=col, lw=1.0), zorder=16)

    # 7. Compass rose
    ax.annotate("N", xy=(0.96, 0.12), xytext=(0.96, 0.08),
                xycoords="axes fraction", fontsize=11, color="white",
                ha="center", fontweight="bold", zorder=30,
                arrowprops=dict(arrowstyle="->", color="white", lw=1.8))

    # Error summary text
    if errors_km is not None:
        lines = [f"Mean: {errors_km.mean():.0f} km"]
        for step, lh in [(3, 24), (7, 48), (11, 72)]:
            if step < len(errors_km):
                lines.append(f"{lh}h: {errors_km[step]:.0f} km")
        ax.text(0.02, 0.02, "\n".join(lines),
                transform=ax.transAxes, fontsize=8, va="bottom",
                color="#88FF88", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="black", alpha=0.8,
                          ec="white", lw=1.0), zorder=20)

    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6,
                 bbox=dict(fc="black", alpha=0.8, ec="#00FFFF", lw=1.5))
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85,
              facecolor="#111111", edgecolor="#00FFFF", labelcolor="white")


# ══════════════════════════════════════════════════════════════════════════════
#  Single TC forecast
# ══════════════════════════════════════════════════════════════════════════════

def visualize_forecast(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*65}")
    print(f"  TC-FM v9 Forecast  |  {args.tc_name}  @  {args.tc_date}")
    print(f"{'='*65}\n")

    detected = detect_pred_len(args.model_path)
    if args.pred_len != detected:
        print(f"  pred_len: {args.pred_len} → {detected} (from checkpoint)")
        args.pred_len = detected

    model = TCFlowMatching(pred_len=args.pred_len, obs_len=args.obs_len).to(device)
    ck    = torch.load(args.model_path, map_location=device, weights_only=False)
    sd    = ck.get("model_state_dict", ck.get("model_state", ck))
    model.load_state_dict(sd, strict=False)
    model.eval()
    print("  Model loaded\n")

    dset, _ = data_loader(
        args,
        {"root": args.TC_data_path, "type": args.dset_type},
        test=True, test_year=args.test_year,
    )
    print(f"  Dataset: {len(dset)} samples\n")

    t_name = args.tc_name.strip().upper()
    t_date = str(args.tc_date).strip()
    target = None
    for i in range(len(dset)):
        item = dset[i]
        info = item[-1]
        if (t_name in str(info["old"][1]).strip().upper()
                and t_date == str(info["tydate"][args.obs_len]).strip()):
            target = item
            print(f"  Found: {info['old'][1]} @ {info['tydate'][args.obs_len]}\n")
            break

    if target is None:
        print(f"  '{t_name} @ {t_date}' not found.")
        print("  Available (first 10):")
        for i in range(min(10, len(dset))):
            info = dset[i][-1]
            print(f"    [{i}] {info['old'][1]} @ {info['tydate'][args.obs_len]}")
        return

    batch = move_batch(seq_collate([target]), device)
    with torch.no_grad():
        pred_mean, _, _ = model.sample(
            batch, num_ensemble=50, ddim_steps=args.ode_steps)

    obs_n  = batch[0][:, 0, :].cpu().numpy()     # [T_obs, 2]
    gt_n   = batch[1][:, 0, :].cpu().numpy()     # [T_pred, 2]
    pred_n = pred_mean[:, 0, :].cpu().numpy()    # [T_pred, 2]

    obs_deg  = to_deg(denorm(obs_n))
    gt_deg   = to_deg(denorm(gt_n))
    pred_deg = to_deg(denorm(pred_n))

    errors_km = np.linalg.norm(denorm(gt_n) - denorm(pred_n), axis=1) * 11.1

    print("  Track errors:")
    for i, e in enumerate(errors_km):
        mark = "  ◀" if (i + 1) in [4, 8, 12] else ""
        print(f"    +{(i+1)*6:3d}h : {e:6.1f} km{mark}")
    print(f"    Mean  : {errors_km.mean():.1f} km\n")

    all_deg   = np.vstack([obs_deg, gt_deg, pred_deg])
    margin    = 3.0
    lon_range = (all_deg[:, 0].min() - margin, all_deg[:, 0].max() + margin)
    lat_range = (all_deg[:, 1].min() - margin, all_deg[:, 1].max() + margin)

    sat = load_himawari(args.himawari_path, args.test_year, t_name, t_date)
    sat_resized = None
    if sat is not None and HAS_CV2:
        sat_resized = cv2.resize(sat, (512, 512))

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="#0d1b2a")
    dt_str  = datetime.strptime(t_date, "%Y%m%d%H").strftime("%d %b %Y %H:%M UTC")
    plot_forecast(
        ax        = ax,
        obs_deg   = obs_deg,
        gt_deg    = gt_deg,
        pred_deg  = pred_deg,
        title     = f"  {t_name} — {args.pred_len * 6}h TC-FM Forecast\n{dt_str}",
        sat_img   = sat_resized,
        lon_range = lon_range,
        lat_range = lat_range,
        errors_km = errors_km,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    fh  = args.pred_len * 6
    out = os.path.join(args.output_dir, f"forecast_{fh}h_{t_name}_{t_date}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="#0d1b2a")
    plt.close()
    print(f"  Saved → {out}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Case-study grid (2 straight + 1 recurvature)
# ══════════════════════════════════════════════════════════════════════════════

def build_case_study_grid(
    args,
    model:  TCFlowMatching,
    dset,
    device: torch.device,
    cases:  list[dict],
    out_path: str,
    him_path: str,
):
    """
    Produce a (n_cases × 2) figure:
      col 0: FM+PINN forecast map
      col 1: lead-time error curve vs CLIPER
    """
    n_cases = len(cases)
    fig, axes = plt.subplots(
        n_cases, 2,
        figsize=(18, 6 * n_cases),
        facecolor="#0d1b2a",
        gridspec_kw={"width_ratios": [2, 1]},
    )
    if n_cases == 1:
        axes = axes[np.newaxis]

    for row, case in enumerate(cases):
        t_name = case["name"].strip().upper()
        t_date = str(case["date"]).strip()
        label  = case.get("label", t_name)

        target = None
        for i in range(len(dset)):
            item = dset[i]
            info = item[-1]
            if (t_name in str(info["old"][1]).strip().upper()
                    and t_date == str(info["tydate"][args.obs_len]).strip()):
                target = item
                break

        if target is None:
            print(f"  {t_name} @ {t_date} not found — skipping row {row}")
            for c in range(2):
                axes[row, c].set_facecolor("#0d1b2a")
                axes[row, c].text(0.5, 0.5, f"NOT FOUND\n{t_name} @ {t_date}",
                                  ha="center", va="center", color="red",
                                  transform=axes[row, c].transAxes)
            continue

        batch = move_batch(seq_collate([target]), device)
        with torch.no_grad():
            pred_mean, _, _ = model.sample(
                batch, num_ensemble=50, ddim_steps=args.ode_steps)

        obs_n  = batch[0][:, 0, :].cpu().numpy()
        gt_n   = batch[1][:, 0, :].cpu().numpy()
        pred_n = pred_mean[:, 0, :].cpu().numpy()

        obs_deg  = to_deg(denorm(obs_n))
        gt_deg   = to_deg(denorm(gt_n))
        pred_deg = to_deg(denorm(pred_n))
        errors_km = np.linalg.norm(denorm(gt_n) - denorm(pred_n), axis=1) * 11.1

        all_deg   = np.vstack([obs_deg, gt_deg, pred_deg])
        margin    = 3.0
        lon_range = (all_deg[:, 0].min() - margin, all_deg[:, 0].max() + margin)
        lat_range = (all_deg[:, 1].min() - margin, all_deg[:, 1].max() + margin)

        sat = load_himawari(him_path, args.test_year, t_name, t_date)
        sat_resized = None
        if sat is not None and HAS_CV2:
            sat_resized = cv2.resize(sat, (512, 512))

        # ── Left: map ──────────────────────────────────────────────────────
        ax_map = axes[row, 0]
        dt_str = datetime.strptime(t_date, "%Y%m%d%H").strftime("%d %b %Y %H:%M UTC")
        plot_forecast(
            ax        = ax_map,
            obs_deg   = obs_deg,
            gt_deg    = gt_deg,
            pred_deg  = pred_deg,
            title     = f"[{label}] {t_name} — {dt_str}",
            sat_img   = sat_resized,
            lon_range = lon_range,
            lat_range = lat_range,
            errors_km = errors_km,
        )

        # ── Right: error curve ─────────────────────────────────────────────
        ax_err = axes[row, 1]
        ax_err.set_facecolor("#0d1b2a")
        lead_h = np.arange(1, len(errors_km) + 1) * 6
        ax_err.plot(lead_h, errors_km, "o-", color="#00FF66", lw=2.5,
                    ms=6, label="FM+PINN", zorder=5)
        ax_err.fill_between(lead_h, 0, errors_km, alpha=0.15, color="#00FF66")

        # CLIPER baseline: extrapolate last observed velocity
        obs_r = denorm(obs_n)
        gt_r  = denorm(gt_n)
        v_cliper = obs_r[-1] - obs_r[-2] if len(obs_r) >= 2 else np.zeros(2)
        cliper_preds = np.array([obs_r[-1] + (k + 1) * v_cliper
                                  for k in range(len(gt_r))])
        cliper_err = np.linalg.norm(cliper_preds - gt_r, axis=1) * 11.1
        ax_err.plot(lead_h, cliper_err[:len(lead_h)], "s--",
                    color="#FF6666", lw=2, ms=5, label="CLIPER", zorder=4)

        for yl in [100, 200, 300]:
            ax_err.axhline(yl, color="white", alpha=0.1, lw=0.8)

        ax_err.set_xlabel("Lead time (h)", color="white", fontsize=9)
        ax_err.set_ylabel("Track error (km)", color="white", fontsize=9)
        ax_err.set_title(f"Error — {t_name}", color="white",
                          fontsize=10, fontweight="bold")
        ax_err.legend(fontsize=8, facecolor="#111111",
                       edgecolor="#00FFFF", labelcolor="white")
        ax_err.tick_params(colors="white")
        for spine in ax_err.spines.values():
            spine.set_edgecolor("white")

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1b2a")
    plt.close()
    print(f"  Case study → {out_path}")


def visualize_case_study(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detected = detect_pred_len(args.model_path)
    args.pred_len = detected

    model = TCFlowMatching(pred_len=args.pred_len, obs_len=args.obs_len).to(device)
    ck    = torch.load(args.model_path, map_location=device, weights_only=False)
    sd    = ck.get("model_state_dict", ck.get("model_state", ck))
    model.load_state_dict(sd, strict=False)
    model.eval()

    dset, _ = data_loader(
        args,
        {"root": args.TC_data_path, "type": "test"},
        test=True, test_year=args.test_year,
    )

    cases = [
        {"name": args.straight1_name, "date": args.straight1_date,
         "label": "Straight-track 1"},
        {"name": args.straight2_name, "date": args.straight2_date,
         "label": "Straight-track 2"},
        {"name": "WIPHA",             "date": args.recurv_date,
         "label": "Recurvature — WIPHA"},
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "case_study_grid.png")
    build_case_study_grid(
        args, model, dset, device, cases,
        out_path=out_path, him_path=args.himawari_path,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",      required=True)
    p.add_argument("--TC_data_path",    required=True)
    p.add_argument("--himawari_path",   default="",
                   help="Path to Himawari images (optional; draws terrain map if absent)")
    p.add_argument("--output_dir",      default="outputs")
    p.add_argument("--mode",            default="single",
                   choices=["single", "case_study"])
    # Single mode
    p.add_argument("--tc_name",         default="WIPHA")
    p.add_argument("--tc_date",         default="2019073106")
    p.add_argument("--dset_type",       default="test")
    # Case-study mode
    p.add_argument("--straight1_name",  default="BEBINCA")
    p.add_argument("--straight1_date",  default="2018090806")
    p.add_argument("--straight2_name",  default="MANGKHUT")
    p.add_argument("--straight2_date",  default="2018091312")
    p.add_argument("--recurv_date",     default="2019073106")
    # Model params
    p.add_argument("--test_year",       type=int,   default=2019)
    p.add_argument("--obs_len",         type=int,   default=8)
    p.add_argument("--pred_len",        type=int,   default=12)
    p.add_argument("--ode_steps",       type=int,   default=10)
    p.add_argument("--batch_size",      type=int,   default=1)
    p.add_argument("--num_workers",     type=int,   default=0)
    p.add_argument("--delim",           default=" ")
    p.add_argument("--skip",            type=int,   default=1)
    p.add_argument("--min_ped",         type=int,   default=1)
    p.add_argument("--threshold",       type=float, default=0.002)
    p.add_argument("--other_modal",     default="gph")

    args = p.parse_args()
    if args.mode == "single":
        visualize_forecast(args)
    else:
        visualize_case_study(args)
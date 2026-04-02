# """
# scripts/visual_evaluate_model_Me_v10.py
# ========================================
# TC-FlowMatching v10 — Forecast Visualisation (fully fixed)

# Fixes vs v9:
#   - env_data index: reads batch[13] consistently (matches seq_collate)
#   - Proper denorm: norm*50+1800 for lon, norm*50 for lat (0.1° units)
#   - Wind speed display from pred_Me (knots, denorm: norm*25+40)
#   - Cartopy map with coastlines, lat/lon grid, land fill
#   - Per-step error labels on predicted track
#   - Ensemble spread shown as shaded cone
#   - Supports both single and case_study modes

# Usage:
#     # Single typhoon
#     python scripts/visual_evaluate_model_Me_v10.py \
#         --mode single \
#         --model_path runs/v10_turbo/best_model.pth \
#         --TC_data_path /path/to/TCND_vn \
#         --tc_name WIPHA \
#         --tc_date 2019073106 \
#         --output_dir outputs

#     # Case-study grid
#     python scripts/visual_evaluate_model_Me_v10.py \
#         --mode case_study \
#         --model_path runs/v10_turbo/best_model.pth \
#         --TC_data_path /path/to/TCND_vn \
#         --output_dir outputs
# """
# from __future__ import annotations

# import os
# import sys
# import random
# import argparse
# from datetime import datetime
# from typing import Optional, List, Tuple

# import numpy as np

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, project_root)

# import torch
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.patheffects as pe
# import matplotlib.colors as mcolors
# from matplotlib.lines import Line2D
# from matplotlib.patches import FancyArrowPatch

# try:
#     import cartopy.crs as ccrs
#     import cartopy.feature as cfeature
#     HAS_CARTOPY = True
# except ImportError:
#     HAS_CARTOPY = False
#     print("  Warning: cartopy not found — using plain axes. "
#           "Install with: pip install cartopy")

# try:
#     import cv2
#     HAS_CV2 = True
# except ImportError:
#     HAS_CV2 = False

# from Model.flow_matching_model import TCFlowMatching
# from Model.data.loader import data_loader
# from Model.data.trajectoriesWithMe_unet_training import seq_collate


# # ── Styling ────────────────────────────────────────────────────────────────────
# STYLE = dict(
#     obs_color      = "#00CFFF",   # cyan
#     gt_color       = "#FF3B3B",   # red  — actual track
#     pred_color     = "#39FF14",   # neon green — predicted track
#     ens_color      = "#39FF14",
#     ens_alpha      = 0.12,
#     cone_alpha     = 0.18,
#     marker_size    = 7,
#     lw_main        = 2.8,
#     lw_thin        = 1.6,
#     bg_color       = "#0D1B2A",
#     land_color     = "#2D4A3E",
#     ocean_color    = "#0D1B2A",
#     border_color   = "#4A6FA5",
#     grid_color     = "#FFFFFF",
#     grid_alpha     = 0.12,
#     error_color    = "#FFD700",
#     text_shadow    = pe.withStroke(linewidth=2.5, foreground="black"),
#     title_pad      = 14,
# )

# # Wind speed → intensity category (kt)
# INTENSITY = [
#     (0,  34,  "TD",      "#99CCFF"),
#     (34, 48,  "TS",      "#66FF66"),
#     (48, 64,  "TY",      "#FFFF00"),
#     (64, 84,  "Sev.TY",  "#FFA500"),
#     (84, 115, "Vis.TY",  "#FF4500"),
#     (115, 999,"Super TY","#FF00FF"),
# ]

# def wind_intensity(wind_kt: float) -> Tuple[str, str]:
#     for lo, hi, name, color in INTENSITY:
#         if lo <= wind_kt < hi:
#             return name, color
#     return "Super TY", "#FF00FF"


# # ── Helpers ────────────────────────────────────────────────────────────────────

# def set_seed(s: int = 42):
#     random.seed(s); np.random.seed(s)
#     torch.manual_seed(s); torch.cuda.manual_seed_all(s)
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


# def denorm_traj(n: np.ndarray) -> np.ndarray:
#     """
#     Normalised → 0.1° units.
#       lon_norm = (lon_01E - 1800) / 50  →  lon_01E = norm*50 + 1800
#       lat_norm = lat_01N / 50           →  lat_01N = norm*50
#     """
#     r = np.zeros_like(n)
#     r[..., 0] = n[..., 0] * 50.0 + 1800.0   # 0.1°E
#     r[..., 1] = n[..., 1] * 50.0             # 0.1°N
#     return r


# def to_deg(pts_01: np.ndarray) -> np.ndarray:
#     """0.1° units → degrees."""
#     return pts_01 / 10.0


# def denorm_wind(wind_norm: float) -> float:
#     """Normalised wind → knots.  wind_norm = (wnd - 40) / 25"""
#     return wind_norm * 25.0 + 40.0


# def haversine_km(p1_deg: np.ndarray, p2_deg: np.ndarray) -> np.ndarray:
#     """[..., 2] degrees (lon, lat) → distance km."""
#     lat1 = np.deg2rad(p1_deg[..., 1])
#     lat2 = np.deg2rad(p2_deg[..., 1])
#     dlat = np.deg2rad(p2_deg[..., 1] - p1_deg[..., 1])
#     dlon = np.deg2rad(p2_deg[..., 0] - p1_deg[..., 0])
#     a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
#     return 2.0 * 6371.0 * np.arcsin(np.clip(np.sqrt(a), 0, 1))


# def detect_pred_len(ckpt_path: str) -> int:
#     ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
#     sd = ck.get("model_state_dict", ck.get("model_state", ck))
#     for key in ["net.pos_enc", "denoiser.pos_enc", "pos_enc"]:
#         if key in sd:
#             return sd[key].shape[1]
#     for k, v in sd.items():
#         if "pos_enc" in k and hasattr(v, "dim") and v.dim() == 3:
#             return v.shape[1]
#     return 12


# # ── Map setup ──────────────────────────────────────────────────────────────────

# def make_map_ax(fig, subplot_spec, lon_range, lat_range):
#     """Create a Cartopy GeoAxes or fallback plain Axes."""
#     if HAS_CARTOPY:
#         ax = fig.add_subplot(
#             subplot_spec,
#             projection=ccrs.PlateCarree(central_longitude=0)
#         )
#         ax.set_extent([lon_range[0], lon_range[1],
#                        lat_range[0], lat_range[1]],
#                       crs=ccrs.PlateCarree())
#         # Ocean & land
#         ax.add_feature(cfeature.OCEAN.with_scale("50m"),
#                        facecolor=STYLE["ocean_color"], zorder=0)
#         ax.add_feature(cfeature.LAND.with_scale("50m"),
#                        facecolor=STYLE["land_color"], zorder=1, alpha=0.9)
#         ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
#                        edgecolor="#8ABCD1", linewidth=0.7, zorder=2)
#         ax.add_feature(cfeature.BORDERS.with_scale("50m"),
#                        edgecolor=STYLE["border_color"],
#                        linewidth=0.4, linestyle=":", zorder=2)
#         # Grid
#         gl = ax.gridlines(
#             crs=ccrs.PlateCarree(), draw_labels=True,
#             linewidth=0.5, color=STYLE["grid_color"],
#             alpha=STYLE["grid_alpha"], linestyle="--"
#         )
#         gl.top_labels   = False
#         gl.right_labels = False
#         gl.xlabel_style = dict(color="white", fontsize=7)
#         gl.ylabel_style = dict(color="white", fontsize=7)
#     else:
#         ax = fig.add_subplot(subplot_spec)
#         ax.set_facecolor(STYLE["bg_color"])
#         ax.set_xlim(*lon_range)
#         ax.set_ylim(*lat_range)
#         for lon in np.arange(np.ceil(lon_range[0]/5)*5, lon_range[1], 5):
#             ax.axvline(lon, color="white", alpha=STYLE["grid_alpha"], lw=0.5)
#         for lat in np.arange(np.ceil(lat_range[0]/5)*5, lat_range[1], 5):
#             ax.axhline(lat, color="white", alpha=STYLE["grid_alpha"], lw=0.5)
#         ax.set_xlabel("Longitude (°E)", color="white", fontsize=8)
#         ax.set_ylabel("Latitude (°N)", color="white", fontsize=8)
#         ax.tick_params(colors="white", labelsize=7)
#         for spine in ax.spines.values():
#             spine.set_edgecolor("white")
#     return ax


# def _plot_on_ax(ax, lon_range, lat_range,
#                 obs_deg, gt_deg, pred_deg, pred_Me_deg,
#                 all_trajs_deg=None,
#                 errors_km=None,
#                 title="", dt_str=""):
#     """
#     Draw one complete forecast panel.

#     Parameters
#     ----------
#     ax           : Axes (Cartopy or plain)
#     obs_deg      : [T_obs, 2] degrees (lon, lat)
#     gt_deg       : [T_pred, 2]
#     pred_deg     : [T_pred, 2]  ensemble mean
#     pred_Me_deg  : [T_pred, 2] (pres_norm, wind_norm) — for wind display
#     all_trajs_deg: [S, T_pred, 2] optional ensemble members
#     errors_km    : [T_pred] error per step
#     """
#     transform = ccrs.PlateCarree() if HAS_CARTOPY else ax.transData

#     def _plot(x, y, **kw):
#         if HAS_CARTOPY:
#             ax.plot(x, y, transform=transform, **kw)
#         else:
#             ax.plot(x, y, **kw)

#     def _scatter(x, y, **kw):
#         if HAS_CARTOPY:
#             ax.scatter(x, y, transform=transform, **kw)
#         else:
#             ax.scatter(x, y, **kw)

#     def _text(x, y, s, **kw):
#         if HAS_CARTOPY:
#             ax.text(x, y, s, transform=transform, **kw)
#         else:
#             ax.text(x, y, s, **kw)

#     outline = [pe.withStroke(linewidth=2.5, foreground="black")]
#     cur_lon, cur_lat = obs_deg[-1, 0], obs_deg[-1, 1]

#     # ── Ensemble spread cone ──────────────────────────────────────────────
#     if all_trajs_deg is not None and len(all_trajs_deg) > 1:
#         S = all_trajs_deg.shape[0]
#         for s in range(S):
#             xt = np.concatenate([[cur_lon], all_trajs_deg[s, :, 0]])
#             yt = np.concatenate([[cur_lat], all_trajs_deg[s, :, 1]])
#             _plot(xt, yt,
#                   color=STYLE["ens_color"], linewidth=0.4,
#                   alpha=STYLE["ens_alpha"], zorder=3)

#     # ── Observed track ────────────────────────────────────────────────────
#     _plot(obs_deg[:, 0], obs_deg[:, 1],
#           "o-", color=STYLE["obs_color"],
#           linewidth=STYLE["lw_thin"], markersize=5,
#           markeredgecolor="white", markeredgewidth=0.8,
#           zorder=7, path_effects=outline, label="Observed")

#     # ── Ground truth ──────────────────────────────────────────────────────
#     gt_full_lon = np.concatenate([[cur_lon], gt_deg[:, 0]])
#     gt_full_lat = np.concatenate([[cur_lat], gt_deg[:, 1]])
#     _plot(gt_full_lon, gt_full_lat,
#           "o-", color=STYLE["gt_color"],
#           linewidth=STYLE["lw_main"], markersize=STYLE["marker_size"],
#           markeredgecolor="white", markeredgewidth=1.2,
#           zorder=8, path_effects=outline, label="Ground truth")

#     # ── Predicted track ───────────────────────────────────────────────────
#     pred_full_lon = np.concatenate([[cur_lon], pred_deg[:, 0]])
#     pred_full_lat = np.concatenate([[cur_lat], pred_deg[:, 1]])
#     _plot(pred_full_lon, pred_full_lat,
#           "o-", color=STYLE["pred_color"],
#           linewidth=STYLE["lw_main"], markersize=STYLE["marker_size"],
#           markeredgecolor="#003300", markeredgewidth=1.0,
#           zorder=9, path_effects=outline, label="FM+PINN (mean)")

#     # ── Wind intensity markers on predicted track ─────────────────────────
#     if pred_Me_deg is not None:
#         for i in range(len(pred_deg)):
#             wnd_kt = denorm_wind(float(pred_Me_deg[i, 1]))
#             _, wcolor = wind_intensity(wnd_kt)
#             _scatter([pred_deg[i, 0]], [pred_deg[i, 1]],
#                      s=60, color=wcolor,
#                      edgecolors="white", linewidths=0.6,
#                      zorder=11, marker="o")

#     # ── Error connectors + distance labels ────────────────────────────────
#     if errors_km is not None:
#         step_show = {3: "24h", 7: "48h", 11: "72h"}
#         for si, lbl in step_show.items():
#             if si < len(gt_deg) and si < len(pred_deg):
#                 gx, gy = gt_deg[si, 0],   gt_deg[si, 1]
#                 px, py = pred_deg[si, 0], pred_deg[si, 1]
#                 if HAS_CARTOPY:
#                     ax.plot([gx, px], [gy, py],
#                             "--", color=STYLE["error_color"],
#                             linewidth=1.2, alpha=0.7,
#                             transform=transform, zorder=7)
#                 else:
#                     ax.plot([gx, px], [gy, py],
#                             "--", color=STYLE["error_color"],
#                             linewidth=1.2, alpha=0.7, zorder=7)
#                 mx, my = (gx + px) / 2, (gy + py) / 2
#                 _text(mx, my,
#                       f" {lbl}\n{errors_km[si]:.0f}km",
#                       fontsize=7, color=STYLE["error_color"],
#                       ha="center", va="bottom", zorder=14,
#                       path_effects=outline)

#     # ── Lead-time labels on predicted track ───────────────────────────────
#     for i, (lx, ly) in enumerate(zip(pred_full_lon, pred_full_lat)):
#         h = i * 6
#         if i == 0:
#             lbl = "NOW"
#         elif h % 24 == 0:
#             lbl = f"+{h}h"
#         else:
#             continue
#         _text(lx, ly + 0.4, lbl,
#               fontsize=7.5, color="#AAFFAA",
#               ha="center", fontweight="bold",
#               path_effects=[pe.withStroke(linewidth=2, foreground="black")],
#               zorder=15)

#     # ── Current position star ─────────────────────────────────────────────
#     _scatter([cur_lon], [cur_lat],
#              s=350, marker="*",
#              color="#FFD700", edgecolors="#FF4400",
#              linewidths=2, zorder=20, label="Now")

#     # ── Error summary box ─────────────────────────────────────────────────
#     if errors_km is not None:
#         n = len(errors_km)
#         lines = [f"Mean: {errors_km.mean():.0f} km"]
#         for si, lh in [(3, 24), (7, 48), (11, 72)]:
#             if si < n:
#                 lines.append(f" {lh}h: {errors_km[si]:.0f} km")
#         ax.text(0.02, 0.03, "\n".join(lines),
#                 transform=ax.transAxes,
#                 fontsize=8, va="bottom",
#                 color="#88FF88", family="monospace",
#                 bbox=dict(boxstyle="round,pad=0.4",
#                           fc=STYLE["bg_color"], alpha=0.85,
#                           ec="white", lw=0.8),
#                 zorder=16)

#     # ── Wind legend ───────────────────────────────────────────────────────
#     legend_items = []
#     for lo, hi, name, color in INTENSITY:
#         legend_items.append(
#             Line2D([0], [0], marker="o", color="none",
#                    markerfacecolor=color, markersize=7,
#                    markeredgecolor="white", markeredgewidth=0.5,
#                    label=f"{name} ({lo}–{hi}kt)")
#         )
#     ax.legend(
#         handles=legend_items,
#         loc="upper right", fontsize=6.5,
#         facecolor="#111111", edgecolor="#00FFFF",
#         labelcolor="white", title="Wind (kt)",
#         title_fontsize=7,
#         ncol=2,
#     )

#     # ── Observed/GT/Pred legend ───────────────────────────────────────────
#     track_handles = [
#         Line2D([0], [0], color=STYLE["obs_color"],  lw=2, label="Observed"),
#         Line2D([0], [0], color=STYLE["gt_color"],   lw=2, label="Ground truth"),
#         Line2D([0], [0], color=STYLE["pred_color"], lw=2, label="FM+PINN"),
#     ]
#     if all_trajs_deg is not None:
#         track_handles.append(
#             Line2D([0], [0], color=STYLE["ens_color"],
#                    lw=1, alpha=0.5, label="Ensemble members"))
#     ax.legend(
#         handles=track_handles,
#         loc="lower right", fontsize=7.5,
#         facecolor="#111111", edgecolor="#00CFFF",
#         labelcolor="white",
#     )

#     # ── Title ─────────────────────────────────────────────────────────────
#     ax.set_title(
#         f"{title}\n{dt_str}",
#         color="white", fontsize=10,
#         fontweight="bold", pad=STYLE["title_pad"],
#         bbox=dict(fc=STYLE["bg_color"], alpha=0.88,
#                   ec="#00FFFF", lw=1.5),
#     )
#     ax.set_facecolor(STYLE["bg_color"])
#     for spine in ax.spines.values():
#         spine.set_edgecolor("#334466")


# # ── Error curve panel ─────────────────────────────────────────────────────────

# def plot_error_curve(ax, errors_km, cliper_err_km, t_name):
#     lead_h = np.arange(1, len(errors_km) + 1) * 6
#     ax.set_facecolor(STYLE["bg_color"])

#     ax.plot(lead_h, errors_km, "o-",
#             color=STYLE["pred_color"], lw=2.5, ms=5,
#             label="FM+PINN", zorder=5)
#     ax.fill_between(lead_h, 0, errors_km,
#                     alpha=0.15, color=STYLE["pred_color"])

#     if cliper_err_km is not None:
#         ax.plot(lead_h, cliper_err_km[:len(lead_h)], "s--",
#                 color="#FF6666", lw=2, ms=4, label="CLIPER", zorder=4)

#     for yl in [100, 200, 300, 400]:
#         ax.axhline(yl, color="white", alpha=0.08, lw=0.7)
#     for xm in [24, 48, 72]:
#         ax.axvline(xm, color=STYLE["error_color"],
#                    alpha=0.2, lw=0.7, ls=":")

#     ax.set_xlabel("Lead time (h)", color="white", fontsize=8)
#     ax.set_ylabel("Track error (km)", color="white", fontsize=8)
#     ax.set_title(f"Error — {t_name}", color="white",
#                  fontsize=9, fontweight="bold")
#     ax.legend(fontsize=7.5, facecolor="#111111",
#               edgecolor="#00FFFF", labelcolor="white")
#     ax.tick_params(colors="white", labelsize=7)
#     for spine in ax.spines.values():
#         spine.set_edgecolor("white")
#     ax.yaxis.label.set_color("white")
#     ax.xaxis.label.set_color("white")


# # ── Run inference for one sequence ────────────────────────────────────────────

# def run_inference(model, target, device, ode_steps, num_ensemble):
#     batch = move_batch(seq_collate([target]), device)
#     with torch.no_grad():
#         pred_mean, pred_Me, all_trajs = model.sample(
#             batch,
#             num_ensemble=num_ensemble,
#             ddim_steps=ode_steps,
#         )
#     # Shape: pred_mean [T_pred, B, 2], all_trajs [S, T_pred, B, 2]
#     obs_n    = batch[0][:, 0, :].cpu().numpy()    # [T_obs, 2]  normalised
#     gt_n     = batch[1][:, 0, :].cpu().numpy()    # [T_pred, 2]
#     pred_n   = pred_mean[:, 0, :].cpu().numpy()   # [T_pred, 2]
#     pred_Me_n= pred_Me[:, 0, :].cpu().numpy()     # [T_pred, 2] (pres, wnd)
#     ens_n    = all_trajs[:, :, 0, :].cpu().numpy()# [S, T_pred, 2]

#     obs_deg   = to_deg(denorm_traj(obs_n))
#     gt_deg    = to_deg(denorm_traj(gt_n))
#     pred_deg  = to_deg(denorm_traj(pred_n))
#     ens_deg   = to_deg(denorm_traj(ens_n))

#     errors_km = haversine_km(pred_deg, gt_deg)

#     # CLIPER: extrapolate last obs velocity
#     obs_r  = denorm_traj(obs_n)
#     gt_r   = denorm_traj(gt_n)
#     v_c = obs_r[-1] - obs_r[-2] if len(obs_r) >= 2 else np.zeros(2)
#     cliper_preds = np.array([obs_r[-1] + (k+1)*v_c
#                               for k in range(len(gt_r))])
#     cliper_err = haversine_km(to_deg(cliper_preds), to_deg(gt_r))

#     return (obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg,
#             errors_km, cliper_err)


# # ── Single typhoon visualisation ───────────────────────────────────────────────

# def visualize_forecast(args):
#     set_seed(42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print(f"{'='*65}")
#     print(f"  TC-FM v10 Forecast  |  {args.tc_name}  @  {args.tc_date}")
#     print(f"{'='*65}\n")

#     detected = detect_pred_len(args.model_path)
#     if args.pred_len != detected:
#         print(f"  pred_len: {args.pred_len} → {detected} (from checkpoint)")
#         args.pred_len = detected

#     model = TCFlowMatching(
#         pred_len=args.pred_len, obs_len=args.obs_len).to(device)
#     ck  = torch.load(args.model_path, map_location=device,
#                      weights_only=False)
#     sd  = ck.get("model_state_dict", ck.get("model_state", ck))
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
#         print("  Available (first 15):")
#         for i in range(min(15, len(dset))):
#             info = dset[i][-1]
#             print(f"    [{i}]  {info['old'][1]}  @  {info['tydate'][args.obs_len]}")
#         return

#     (obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg,
#      errors_km, cliper_err) = run_inference(
#          model, target, device, args.ode_steps, args.num_ensemble)

#     print("  Track errors (km):")
#     for i, e in enumerate(errors_km):
#         mark = "  ◀" if (i+1) in [4, 8, 12] else ""
#         print(f"    +{(i+1)*6:3d}h : {e:6.1f} km{mark}")
#     print(f"    Mean  : {errors_km.mean():.1f} km\n")

#     all_deg   = np.vstack([obs_deg, gt_deg, pred_deg])
#     margin    = 3.5
#     lon_range = (all_deg[:, 0].min() - margin, all_deg[:, 0].max() + margin)
#     lat_range = (all_deg[:, 1].min() - margin, all_deg[:, 1].max() + margin)

#     # ── Figure layout: map (left 2/3) + error curve (right 1/3) ──────────
#     fig = plt.figure(figsize=(18, 10), facecolor=STYLE["bg_color"])
#     gs  = fig.add_gridspec(1, 3, wspace=0.08)

#     ax_map = make_map_ax(fig, gs[0, :2], lon_range, lat_range)
#     ax_err = fig.add_subplot(gs[0, 2])

#     dt_str = datetime.strptime(t_date, "%Y%m%d%H").strftime(
#         "%d %b %Y  %H:%M UTC")
#     fh = args.pred_len * 6

#     _plot_on_ax(
#         ax_map, lon_range, lat_range,
#         obs_deg, gt_deg, pred_deg, pred_Me_n,
#         all_trajs_deg=ens_deg if args.num_ensemble > 1 else None,
#         errors_km=errors_km,
#         title=f"🌀 {t_name}  —  {fh}h FC  |  FM+PINN v10",
#         dt_str=dt_str,
#     )
#     plot_error_curve(ax_err, errors_km, cliper_err, t_name)

#     os.makedirs(args.output_dir, exist_ok=True)
#     out = os.path.join(args.output_dir,
#                        f"forecast_{fh}h_{t_name}_{t_date}.png")
#     plt.savefig(out, dpi=200, bbox_inches="tight",
#                 facecolor=STYLE["bg_color"])
#     plt.close()
#     print(f"  Saved → {out}\n")


# # ── Case-study grid ────────────────────────────────────────────────────────────

# def visualize_case_study(args):
#     set_seed(42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     detected    = detect_pred_len(args.model_path)
#     args.pred_len = detected

#     model = TCFlowMatching(
#         pred_len=args.pred_len, obs_len=args.obs_len).to(device)
#     ck = torch.load(args.model_path, map_location=device,
#                     weights_only=False)
#     sd = ck.get("model_state_dict", ck.get("model_state", ck))
#     model.load_state_dict(sd, strict=False)
#     model.eval()

#     dset, _ = data_loader(
#         args,
#         {"root": args.TC_data_path, "type": "test"},
#         test=True, test_year=args.test_year,
#     )

#     cases = [
#         {"name": args.straight1_name, "date": args.straight1_date,
#          "label": "Straight-track 1"},
#         {"name": args.straight2_name, "date": args.straight2_date,
#          "label": "Straight-track 2"},
#         {"name": "WIPHA",             "date": args.recurv_date,
#          "label": "Recurvature — WIPHA"},
#     ]

#     n_cases = len(cases)
#     fig = plt.figure(
#         figsize=(22, 7 * n_cases),
#         facecolor=STYLE["bg_color"]
#     )
#     gs = fig.add_gridspec(n_cases, 3, wspace=0.08, hspace=0.25)

#     for row, case in enumerate(cases):
#         t_name = case["name"].strip().upper()
#         t_date = str(case["date"]).strip()
#         label  = case.get("label", t_name)

#         target = None
#         for i in range(len(dset)):
#             item = dset[i]
#             info = item[-1]
#             if (t_name in str(info["old"][1]).strip().upper()
#                     and t_date == str(info["tydate"][args.obs_len]).strip()):
#                 target = item
#                 break

#         if target is None:
#             print(f"  ⚠  {t_name} @ {t_date} — not found, skipping")
#             for c in range(3):
#                 ax = fig.add_subplot(gs[row, c])
#                 ax.set_facecolor(STYLE["bg_color"])
#                 ax.text(0.5, 0.5, f"NOT FOUND\n{t_name} @ {t_date}",
#                         ha="center", va="center", color="red",
#                         transform=ax.transAxes)
#             continue

#         (obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg,
#          errors_km, cliper_err) = run_inference(
#              model, target, device, args.ode_steps, args.num_ensemble)

#         all_deg   = np.vstack([obs_deg, gt_deg, pred_deg])
#         margin    = 3.5
#         lon_range = (all_deg[:, 0].min() - margin,
#                      all_deg[:, 0].max() + margin)
#         lat_range = (all_deg[:, 1].min() - margin,
#                      all_deg[:, 1].max() + margin)

#         dt_str = datetime.strptime(t_date, "%Y%m%d%H").strftime(
#             "%d %b %Y %H:%M UTC")

#         ax_map = make_map_ax(fig, gs[row, :2], lon_range, lat_range)
#         ax_err = fig.add_subplot(gs[row, 2])

#         _plot_on_ax(
#             ax_map, lon_range, lat_range,
#             obs_deg, gt_deg, pred_deg, pred_Me_n,
#             all_trajs_deg=ens_deg if args.num_ensemble > 1 else None,
#             errors_km=errors_km,
#             title=f"[{label}]  {t_name}",
#             dt_str=dt_str,
#         )
#         plot_error_curve(ax_err, errors_km, cliper_err, t_name)

#         print(f"  [{label}] ADE={errors_km.mean():.1f} km  "
#               f"72h={errors_km[11] if len(errors_km)>11 else float('nan'):.1f} km")

#     os.makedirs(args.output_dir, exist_ok=True)
#     out = os.path.join(args.output_dir, "case_study_grid_v10.png")
#     plt.savefig(out, dpi=150, bbox_inches="tight",
#                 facecolor=STYLE["bg_color"])
#     plt.close()
#     print(f"\n  Case study saved → {out}")


# # ── CLI ────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     p = argparse.ArgumentParser(
#         description="TC-FlowMatching v10 Forecast Visualisation")
#     p.add_argument("--model_path",      required=True)
#     p.add_argument("--TC_data_path",    required=True)
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
#     # Model / data
#     p.add_argument("--test_year",       type=int,   default=2019)
#     p.add_argument("--obs_len",         type=int,   default=8)
#     p.add_argument("--pred_len",        type=int,   default=12)
#     p.add_argument("--ode_steps",       type=int,   default=10)
#     p.add_argument("--num_ensemble",    type=int,   default=50)
#     p.add_argument("--batch_size",      type=int,   default=1)
#     p.add_argument("--num_workers",     type=int,   default=0)
#     p.add_argument("--delim",           default=" ")
#     p.add_argument("--skip",            type=int,   default=1)
#     p.add_argument("--min_ped",         type=int,   default=1)
#     p.add_argument("--threshold",       type=float, default=0.002)
#     p.add_argument("--other_modal",     default="gph")

#     args = p.parse_args()
#     if args.mode == "single":
#         visualize_forecast(args)
#     else:
#         visualize_case_study(args)
# """
# scripts/visual_evaluate_model_Me_v10.py
# ========================================
# TC-FlowMatching v10 — Forecast Visualisation (v11 — NHC-style Probability Cone)

# Thay đổi vs v10:
#   - Probability cone kiểu NHC: vùng liên tục mượt dọc theo track,
#     càng về sau càng rộng ra (không phải ellipse rời rạc)
#   - Vẽ boundary trái/phải của cone bằng cách lấy các điểm ngoài cùng của
#     ensemble tại mỗi timestep rồi fill_between
#   - Giữ đường mean chính + 50%/90% boundary rõ ràng
#   - Panel phải: spread vs error theo lead time
# """
# from __future__ import annotations

# import os
# import sys
# import random
# import argparse
# from datetime import datetime
# from typing import Optional, List, Tuple

# import numpy as np
# from scipy.stats import chi2

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, project_root)

# import torch
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.patheffects as pe
# import matplotlib.patches as mpatches
# from matplotlib.lines import Line2D
# from matplotlib.patches import Ellipse

# try:
#     import cartopy.crs as ccrs
#     import cartopy.feature as cfeature
#     HAS_CARTOPY = True
# except ImportError:
#     HAS_CARTOPY = False
#     print("  Warning: cartopy not found — using plain axes.")

# try:
#     import cv2
#     HAS_CV2 = True
# except ImportError:
#     HAS_CV2 = False

# from Model.flow_matching_model import TCFlowMatching
# from Model.data.loader import data_loader
# from Model.data.trajectoriesWithMe_unet_training import seq_collate


# # ── Styling ────────────────────────────────────────────────────────────────────
# STYLE = dict(
#     obs_color      = "#00CFFF",
#     gt_color       = "#FF3B3B",
#     pred_color     = "#39FF14",
#     ens_color      = "#39FF14",
#     ens_alpha      = 0.05,
#     marker_size    = 8,
#     lw_main        = 2.8,
#     lw_thin        = 1.6,
#     bg_color       = "#0D1B2A",
#     land_color     = "#2D4A3E",
#     ocean_color    = "#0D1B2A",
#     border_color   = "#4A6FA5",
#     grid_color     = "#FFFFFF",
#     grid_alpha     = 0.12,
#     error_color    = "#FFD700",
#     title_pad      = 14,
#     # Cone colours
#     cone_50_fill   = "#39FF14",   # 50% — green
#     cone_90_fill   = "#00CFFF",   # 90% — cyan
#     cone_50_alpha  = 0.22,
#     cone_90_alpha  = 0.10,
#     cone_edge_lw   = 1.2,
# )

# _CHI2_50 = chi2.ppf(0.50, df=2)
# _CHI2_90 = chi2.ppf(0.90, df=2)

# INTENSITY = [
#     (0,   34,  "TD",       "#99CCFF"),
#     (34,  48,  "TS",       "#66FF66"),
#     (48,  64,  "TY",       "#FFFF00"),
#     (64,  84,  "Sev.TY",   "#FFA500"),
#     (84,  115, "Vis.TY",   "#FF4500"),
#     (115, 999, "Super TY", "#FF00FF"),
# ]

# def wind_intensity(wind_kt):
#     for lo, hi, name, color in INTENSITY:
#         if lo <= wind_kt < hi:
#             return name, color
#     return "Super TY", "#FF00FF"


# # ── Helpers ────────────────────────────────────────────────────────────────────

# def set_seed(s=42):
#     random.seed(s); np.random.seed(s)
#     torch.manual_seed(s); torch.cuda.manual_seed_all(s)
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


# def denorm_traj(n):
#     r = np.zeros_like(n)
#     r[..., 0] = n[..., 0] * 50.0 + 1800.0
#     r[..., 1] = n[..., 1] * 50.0
#     return r


# def to_deg(pts_01):
#     return pts_01 / 10.0


# def denorm_wind(wind_norm):
#     return wind_norm * 25.0 + 40.0


# def haversine_km(p1_deg, p2_deg):
#     lat1 = np.deg2rad(p1_deg[..., 1])
#     lat2 = np.deg2rad(p2_deg[..., 1])
#     dlat = np.deg2rad(p2_deg[..., 1] - p1_deg[..., 1])
#     dlon = np.deg2rad(p2_deg[..., 0] - p1_deg[..., 0])
#     a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
#     return 2.0 * 6371.0 * np.arcsin(np.clip(np.sqrt(a), 0, 1))


# def detect_pred_len(ckpt_path):
#     ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
#     sd = ck.get("model_state_dict", ck.get("model_state", ck))
#     for key in ["net.pos_enc", "denoiser.pos_enc", "pos_enc"]:
#         if key in sd:
#             return sd[key].shape[1]
#     for k, v in sd.items():
#         if "pos_enc" in k and hasattr(v, "dim") and v.dim() == 3:
#             return v.shape[1]
#     return 12


# # ── Core: NHC-style smooth probability cone ────────────────────────────────────

# def _gaussian_cone_boundary(pts_deg, chi2_thresh):
#     """
#     Fit 2D Gaussian tại một timestep, trả về boundary ellipse (parametric).

#     Parameters
#     ----------
#     pts_deg     : [S, 2]  ensemble positions (lon, lat)
#     chi2_thresh : float

#     Returns
#     -------
#     boundary : [64, 2]  polygon points của ellipse
#     """
#     if len(pts_deg) < 3:
#         return None
#     mu  = pts_deg.mean(axis=0)
#     cov = np.cov(pts_deg.T) + np.eye(2) * 1e-8
#     eigvals, eigvecs = np.linalg.eigh(cov)
#     eigvals = np.maximum(eigvals, 1e-8)

#     a = np.sqrt(chi2_thresh * eigvals[-1])   # semi-major
#     b = np.sqrt(chi2_thresh * eigvals[0])    # semi-minor

#     theta = np.linspace(0, 2 * np.pi, 64)
#     ellipse_local = np.stack([a * np.cos(theta), b * np.sin(theta)], axis=1)
#     R        = eigvecs
#     boundary = ellipse_local @ R.T + mu
#     return boundary


# def draw_smooth_cone(ax, ens_deg, cur_pos_deg, transform=None):
#     """
#     Vẽ NHC-style probability cone: vùng liên tục mượt dọc theo track,
#     bắt đầu hội tụ tại NOW, càng về sau càng rộng ra.

#     Strategy:
#       1. Tại mỗi timestep t, fit 2D Gaussian từ S ensemble members
#          → tính boundary ellipse 50% và 90%
#       2. Tìm 2 điểm "vuông góc track nhất" trên mỗi ellipse
#          → đây là cạnh trái và phải của cone tại t
#       3. Nối các cạnh trái/phải thành polygon kín → fill

#     Parameters
#     ----------
#     ax          : Axes
#     ens_deg     : [S, T, 2]  ensemble members (lon, lat) degrees
#     cur_pos_deg : [2]        vị trí NOW
#     transform   : cartopy transform hoặc None
#     """
#     S, T, _ = ens_deg.shape
#     if S < 3:
#         return

#     def _fill(verts, color, alpha, zorder):
#         verts_c = np.vstack([verts, verts[0]])
#         if HAS_CARTOPY and transform is not None:
#             ax.fill(verts_c[:, 0], verts_c[:, 1],
#                     color=color, alpha=alpha, zorder=zorder,
#                     transform=transform, linewidth=0)
#         else:
#             ax.fill(verts_c[:, 0], verts_c[:, 1],
#                     color=color, alpha=alpha, zorder=zorder, linewidth=0)

#     def _line(xs, ys, color, alpha, lw, zorder, ls="-"):
#         if HAS_CARTOPY and transform is not None:
#             ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw,
#                     zorder=zorder, linestyle=ls, transform=transform)
#         else:
#             ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw,
#                     zorder=zorder, linestyle=ls)

#     # Mean position tại mỗi timestep
#     means = np.array([ens_deg[:, t, :].mean(axis=0) for t in range(T)])
#     # Track points: NOW + T means
#     track_pts = np.vstack([cur_pos_deg, means])   # [T+1, 2]

#     def _perp_unit(p1, p2):
#         """Unit vector vuông góc với hướng p1 → p2."""
#         d = p2 - p1
#         n = np.linalg.norm(d)
#         if n < 1e-10:
#             return np.array([0.0, 1.0])
#         d /= n
#         return np.array([-d[1], d[0]])

#     def _cone_edges(chi2_thresh):
#         """
#         Tính cạnh trái và phải của cone tại từng timestep.
#         Trả về left_pts [T+1, 2] và right_pts [T+1, 2].
#         t=0 là điểm NOW (spread = 0).
#         """
#         left_pts  = [cur_pos_deg.copy()]
#         right_pts = [cur_pos_deg.copy()]

#         for t in range(T):
#             pts = ens_deg[:, t, :]
#             b   = _gaussian_cone_boundary(pts, chi2_thresh)

#             if b is None:
#                 left_pts.append(means[t])
#                 right_pts.append(means[t])
#                 continue

#             # Hướng vuông góc track tại t
#             if t + 1 < len(track_pts):
#                 perp = _perp_unit(track_pts[t], track_pts[t + 1])
#             else:
#                 perp = _perp_unit(track_pts[t - 1], track_pts[t])

#             # Chiếu boundary lên perp để tìm cạnh trái/phải
#             proj = (b - means[t]) @ perp
#             left_pts.append(b[proj.argmax()])
#             right_pts.append(b[proj.argmin()])

#         return np.array(left_pts), np.array(right_pts)

#     # ── 90% cone (ngoài, vẽ trước) ───────────────────────────────────────
#     left_90, right_90 = _cone_edges(_CHI2_90)
#     poly_90 = np.vstack([left_90, right_90[::-1]])
#     _fill(poly_90, STYLE["cone_90_fill"], STYLE["cone_90_alpha"], zorder=3)
#     _line(left_90[:, 0],  left_90[:, 1],
#           STYLE["cone_90_fill"], alpha=0.35,
#           lw=STYLE["cone_edge_lw"], zorder=4, ls="--")
#     _line(right_90[:, 0], right_90[:, 1],
#           STYLE["cone_90_fill"], alpha=0.35,
#           lw=STYLE["cone_edge_lw"], zorder=4, ls="--")

#     # ── 50% cone (trong, vẽ sau để đè lên) ───────────────────────────────
#     left_50, right_50 = _cone_edges(_CHI2_50)
#     poly_50 = np.vstack([left_50, right_50[::-1]])
#     _fill(poly_50, STYLE["cone_50_fill"], STYLE["cone_50_alpha"], zorder=5)
#     _line(left_50[:, 0],  left_50[:, 1],
#           STYLE["cone_50_fill"], alpha=0.6,
#           lw=STYLE["cone_edge_lw"] * 1.3, zorder=6)
#     _line(right_50[:, 0], right_50[:, 1],
#           STYLE["cone_50_fill"], alpha=0.6,
#           lw=STYLE["cone_edge_lw"] * 1.3, zorder=6)

#     # ── Ensemble member lines (rất mờ, dưới cùng) ────────────────────────
#     for s in range(S):
#         xs = np.concatenate([[cur_pos_deg[0]], ens_deg[s, :, 0]])
#         ys = np.concatenate([[cur_pos_deg[1]], ens_deg[s, :, 1]])
#         if HAS_CARTOPY and transform is not None:
#             ax.plot(xs, ys, color=STYLE["ens_color"], linewidth=0.25,
#                     alpha=STYLE["ens_alpha"], zorder=2, transform=transform)
#         else:
#             ax.plot(xs, ys, color=STYLE["ens_color"], linewidth=0.25,
#                     alpha=STYLE["ens_alpha"], zorder=2)


# # ── Spread panel ───────────────────────────────────────────────────────────────

# def plot_spread_over_time(ax, ens_deg, errors_km, cliper_err_km, t_name):
#     S, T, _ = ens_deg.shape
#     lead_h   = np.arange(1, T + 1) * 6

#     spreads_km = []
#     for t in range(T):
#         pts      = ens_deg[:, t, :]
#         std_lon  = pts[:, 0].std()
#         std_lat  = pts[:, 1].std()
#         mean_lat = pts[:, 1].mean()
#         # std_km   = np.sqrt(
#         #     (std_lon * 111 * np.cos(np.deg2rad(mean_lat)))**2
#         #     + (std_lat * 111)**2
#         # )
#         # spreads_km.append(std_km)
#         # FIX: Bù trừ cos(lat) cho kinh độ để ra km chuẩn
#         std_lon_km = pts[:, 0].std() * 111.32 * np.cos(np.deg2rad(mean_lat))
#         std_lat_km = pts[:, 1].std() * 110.57
#         std_km = np.sqrt(std_lon_km**2 + std_lat_km**2)
#         spreads_km.append(std_km)
#     spreads_km = np.array(spreads_km)

#     ax.set_facecolor(STYLE["bg_color"])
#     ax.fill_between(lead_h, 0, spreads_km,
#                     alpha=0.25, color=STYLE["cone_50_fill"],
#                     label="Ensemble spread (1σ)")
#     ax.plot(lead_h, spreads_km, "-",
#             color=STYLE["cone_50_fill"], lw=2.2, zorder=5)

#     ax_twin = ax.twinx()
#     ax_twin.set_facecolor(STYLE["bg_color"])
#     ax_twin.plot(lead_h, errors_km, "o-",
#                  color=STYLE["pred_color"], lw=2.5, ms=5,
#                  label="FM+PINN ADE", zorder=6)
#     ax_twin.fill_between(lead_h, 0, errors_km,
#                          alpha=0.12, color=STYLE["pred_color"])

#     if cliper_err_km is not None:
#         ax_twin.plot(lead_h, cliper_err_km[:T], "s--",
#                      color="#FF6666", lw=2, ms=4, label="CLIPER", zorder=4)

#     for xm in [24, 48, 72]:
#         ax.axvline(xm, color=STYLE["error_color"], alpha=0.2, lw=0.7, ls=":")

#     ax.set_xlabel("Lead time (h)", color="white", fontsize=8)
#     ax.set_ylabel("Spread 1σ (km)", color=STYLE["cone_50_fill"], fontsize=8)
#     ax_twin.set_ylabel("Track error (km)", color=STYLE["pred_color"], fontsize=8)
#     ax.set_title(f"Spread vs Error — {t_name}",
#                  color="white", fontsize=9, fontweight="bold")

#     lines1, lbs1 = ax.get_legend_handles_labels()
#     lines2, lbs2 = ax_twin.get_legend_handles_labels()
#     ax.legend(lines1 + lines2, lbs1 + lbs2,
#               fontsize=7.5, facecolor="#111111",
#               edgecolor="#00CFFF", labelcolor="white", loc="upper left")

#     for spine in ax.spines.values():
#         spine.set_edgecolor("white")
#     ax.tick_params(colors="white", labelsize=7)
#     ax_twin.tick_params(colors="white", labelsize=7)
#     ax_twin.yaxis.label.set_color(STYLE["pred_color"])
#     ax.yaxis.label.set_color(STYLE["cone_50_fill"])


# # ── Map setup ──────────────────────────────────────────────────────────────────

# def make_map_ax(fig, subplot_spec, lon_range, lat_range):
#     if HAS_CARTOPY:
#         ax = fig.add_subplot(
#             subplot_spec,
#             projection=ccrs.PlateCarree(central_longitude=0)
#         )
#         ax.set_extent([lon_range[0], lon_range[1],
#                        lat_range[0], lat_range[1]],
#                       crs=ccrs.PlateCarree())
#         ax.add_feature(cfeature.OCEAN.with_scale("50m"),
#                        facecolor=STYLE["ocean_color"], zorder=0)
#         ax.add_feature(cfeature.LAND.with_scale("50m"),
#                        facecolor=STYLE["land_color"], zorder=1, alpha=0.9)
#         ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
#                        edgecolor="#8ABCD1", linewidth=0.7, zorder=2)
#         ax.add_feature(cfeature.BORDERS.with_scale("50m"),
#                        edgecolor=STYLE["border_color"],
#                        linewidth=0.4, linestyle=":", zorder=2)
#         gl = ax.gridlines(
#             crs=ccrs.PlateCarree(), draw_labels=True,
#             linewidth=0.5, color=STYLE["grid_color"],
#             alpha=STYLE["grid_alpha"], linestyle="--"
#         )
#         gl.top_labels   = False
#         gl.right_labels = False
#         gl.xlabel_style = dict(color="white", fontsize=7)
#         gl.ylabel_style = dict(color="white", fontsize=7)
#     else:
#         ax = fig.add_subplot(subplot_spec)
#         ax.set_facecolor(STYLE["bg_color"])
#         ax.set_xlim(*lon_range)
#         ax.set_ylim(*lat_range)
#         for lon in np.arange(np.ceil(lon_range[0]/5)*5, lon_range[1], 5):
#             ax.axvline(lon, color="white", alpha=STYLE["grid_alpha"], lw=0.5)
#         for lat in np.arange(np.ceil(lat_range[0]/5)*5, lat_range[1], 5):
#             ax.axhline(lat, color="white", alpha=STYLE["grid_alpha"], lw=0.5)
#         ax.set_xlabel("Longitude (°E)", color="white", fontsize=8)
#         ax.set_ylabel("Latitude (°N)", color="white", fontsize=8)
#         ax.tick_params(colors="white", labelsize=7)
#         for spine in ax.spines.values():
#             spine.set_edgecolor("white")
#     return ax


# def _plot_on_ax(ax, lon_range, lat_range,
#                 obs_deg, gt_deg, pred_deg, pred_Me_deg,
#                 all_trajs_deg=None, errors_km=None,
#                 title="", dt_str=""):

#     transform = ccrs.PlateCarree() if HAS_CARTOPY else None
#     outline   = [pe.withStroke(linewidth=2.5, foreground="black")]
#     cur_pos   = obs_deg[-1]

#     def _plot(x, y, **kw):
#         if HAS_CARTOPY:
#             ax.plot(x, y, transform=transform, **kw)
#         else:
#             ax.plot(x, y, **kw)

#     def _scatter(x, y, **kw):
#         if HAS_CARTOPY:
#             ax.scatter(x, y, transform=transform, **kw)
#         else:
#             ax.scatter(x, y, **kw)

#     def _text(x, y, s, **kw):
#         if HAS_CARTOPY:
#             ax.text(x, y, s, transform=transform, **kw)
#         else:
#             ax.text(x, y, s, **kw)

#     # ── 1. Probability cone (vẽ trước nhất) ──────────────────────────────
#     if all_trajs_deg is not None and all_trajs_deg.shape[0] >= 3:
#         draw_smooth_cone(ax, all_trajs_deg, cur_pos, transform)

#     # ── 2. Observed track ─────────────────────────────────────────────────
#     _plot(obs_deg[:, 0], obs_deg[:, 1],
#           "o-", color=STYLE["obs_color"],
#           linewidth=STYLE["lw_thin"], markersize=5,
#           markeredgecolor="white", markeredgewidth=0.8,
#           zorder=7, path_effects=outline)

#     # ── 3. Ground truth ───────────────────────────────────────────────────
#     gt_lon = np.concatenate([[cur_pos[0]], gt_deg[:, 0]])
#     gt_lat = np.concatenate([[cur_pos[1]], gt_deg[:, 1]])
#     _plot(gt_lon, gt_lat, "o-", color=STYLE["gt_color"],
#           linewidth=STYLE["lw_main"], markersize=STYLE["marker_size"],
#           markeredgecolor="white", markeredgewidth=1.2,
#           zorder=8, path_effects=outline)

#     # ── 4. Predicted track (ensemble mean) ───────────────────────────────
#     pred_lon = np.concatenate([[cur_pos[0]], pred_deg[:, 0]])
#     pred_lat = np.concatenate([[cur_pos[1]], pred_deg[:, 1]])
#     _plot(pred_lon, pred_lat, "o-", color=STYLE["pred_color"],
#           linewidth=STYLE["lw_main"], markersize=STYLE["marker_size"],
#           markeredgecolor="#003300", markeredgewidth=1.0,
#           zorder=9, path_effects=outline)

#     # ── 5. Wind intensity markers ─────────────────────────────────────────
#     if pred_Me_deg is not None:
#         for i in range(len(pred_deg)):
#             wnd_kt = denorm_wind(float(pred_Me_deg[i, 1]))
#             _, wcolor = wind_intensity(wnd_kt)
#             _scatter([pred_deg[i, 0]], [pred_deg[i, 1]],
#                      s=70, color=wcolor,
#                      edgecolors="white", linewidths=0.7, zorder=11)

#     # ── 6. Error connectors ───────────────────────────────────────────────
#     if errors_km is not None:
#         for si, lbl in {3: "24h", 7: "48h", 11: "72h"}.items():
#             if si < len(gt_deg) and si < len(pred_deg):
#                 gx, gy = gt_deg[si, 0], gt_deg[si, 1]
#                 px, py = pred_deg[si, 0], pred_deg[si, 1]
#                 if HAS_CARTOPY:
#                     ax.plot([gx, px], [gy, py], "--",
#                             color=STYLE["error_color"],
#                             linewidth=1.2, alpha=0.7,
#                             transform=transform, zorder=7)
#                 else:
#                     ax.plot([gx, px], [gy, py], "--",
#                             color=STYLE["error_color"],
#                             linewidth=1.2, alpha=0.7, zorder=7)
#                 _text((gx+px)/2, (gy+py)/2,
#                       f" {lbl}\n{errors_km[si]:.0f}km",
#                       fontsize=7, color=STYLE["error_color"],
#                       ha="center", va="bottom", zorder=14,
#                       path_effects=outline)

#     # ── 7. Lead-time labels ───────────────────────────────────────────────
#     # for i, (lx, ly) in enumerate(zip(pred_lon, pred_lat)):
#     #     h   = i * 6
#     #     lbl = "NOW" if i == 0 else (f"+{h}h" if h % 24 == 0 else None)
#     #     if lbl:
#     #         _text(lx, ly + 0.4, lbl,
#     #               fontsize=7.5, color="#AAFFAA",
#     #               ha="center", fontweight="bold",
#     #               path_effects=[pe.withStroke(linewidth=2,
#     #                                           foreground="black")],
#     #               zorder=15)
#     # ── 7. Lead-time labels (Cả cho GT và Pred để so sánh tốc độ) ──
#     for i in range(len(pred_lon)):
#         h = i * 6
#         if h % 24 == 0:
#             lbl = "NOW" if i == 0 else f"+{h}h"
#             # Nhãn cho Dự báo (Màu xanh)
#             _text(pred_lon[i], pred_lat[i] + 0.5, lbl, color="#AAFFAA", fontweight="bold", path_effects=outline)
#             # Nhãn cho Thực tế (Màu đỏ) - Giúp nhìn ra bão nhanh hay chậm hơn
#             if i < len(gt_lon):
#                 _text(gt_lon[i], gt_lat[i] - 0.7, lbl, color="#FFAAAA", fontsize=6, alpha=0.8, path_effects=outline)
#     # ── 8. NOW star ───────────────────────────────────────────────────────
#     _scatter([cur_pos[0]], [cur_pos[1]],
#              s=350, marker="*", color="#FFD700",
#              edgecolors="#FF4400", linewidths=2, zorder=20)

#     # ── 9. Error summary box ──────────────────────────────────────────────
#     if errors_km is not None:
#         n     = len(errors_km)
#         lines = [f"Mean: {errors_km.mean():.0f} km"]
#         for si, lh in [(3, 24), (7, 48), (11, 72)]:
#             if si < n:
#                 lines.append(f" {lh}h: {errors_km[si]:.0f} km")
#         ax.text(0.02, 0.03, "\n".join(lines),
#                 transform=ax.transAxes, fontsize=8, va="bottom",
#                 color="#88FF88", family="monospace",
#                 bbox=dict(boxstyle="round,pad=0.4",
#                           fc=STYLE["bg_color"], alpha=0.85,
#                           ec="white", lw=0.8),
#                 zorder=16)

#     # ── 10. Legend ────────────────────────────────────────────────────────
#     track_handles = [
#         Line2D([0], [0], color=STYLE["obs_color"],  lw=2, label="Observed"),
#         Line2D([0], [0], color=STYLE["gt_color"],   lw=2, label="Ground truth"),
#         Line2D([0], [0], color=STYLE["pred_color"], lw=2.5,
#                label="FM+PINN (mean)"),
#         mpatches.Patch(facecolor=STYLE["cone_50_fill"],
#                        alpha=0.6, label="50% prob. cone"),
#         mpatches.Patch(facecolor=STYLE["cone_90_fill"],
#                        alpha=0.45, label="90% prob. cone"),
#     ]
#     ax.legend(handles=track_handles, loc="lower right", fontsize=7.5,
#               facecolor="#111111", edgecolor="#00CFFF", labelcolor="white")

#     wind_handles = [
#         Line2D([0], [0], marker="o", color="none",
#                markerfacecolor=c, markersize=7,
#                markeredgecolor="white", markeredgewidth=0.5,
#                label=f"{nm} ({lo}–{hi}kt)")
#         for lo, hi, nm, c in INTENSITY
#     ]
#     leg2 = ax.legend(handles=wind_handles, loc="upper right", fontsize=6.5,
#                      facecolor="#111111", edgecolor="#00FFFF",
#                      labelcolor="white", title="Wind (kt)",
#                      title_fontsize=7, ncol=2)
#     ax.add_artist(leg2)

#     ax.set_title(f"{title}\n{dt_str}",
#                  color="white", fontsize=10, fontweight="bold",
#                  pad=STYLE["title_pad"],
#                  bbox=dict(fc=STYLE["bg_color"], alpha=0.88,
#                            ec="#00FFFF", lw=1.5))
#     ax.set_facecolor(STYLE["bg_color"])
#     for spine in ax.spines.values():
#         spine.set_edgecolor("#334466")


# # ── Run inference ──────────────────────────────────────────────────────────────

# def run_inference(model, target, device, ode_steps, num_ensemble):
#     batch = move_batch(seq_collate([target]), device)
#     with torch.no_grad():
#         pred_mean, pred_Me, all_trajs = model.sample(
#             batch, num_ensemble=num_ensemble, ddim_steps=ode_steps)

#     obs_n     = batch[0][:, 0, :].cpu().numpy()
#     gt_n      = batch[1][:, 0, :].cpu().numpy()
#     pred_n    = pred_mean[:, 0, :].cpu().numpy()
#     pred_Me_n = pred_Me[:, 0, :].cpu().numpy()
#     ens_n     = all_trajs[:, :, 0, :].cpu().numpy()   # [S, T, 2]

#     # obs_deg  = to_deg(denorm_traj(obs_n))
#     # gt_deg   = to_deg(denorm_traj(gt_n))
#     pred_deg = to_deg(denorm_traj(pred_n))
#     ens_deg  = to_deg(denorm_traj(ens_n))

#     errors_km = haversine_km(pred_deg, gt_deg)

#     # obs_r = denorm_traj(obs_n)
#     # gt_r  = denorm_traj(gt_n)
#     # v_c   = obs_r[-1] - obs_r[-2] if len(obs_r) >= 2 else np.zeros(2)
#     # cliper_preds = np.array([obs_r[-1] + (k+1)*v_c for k in range(len(gt_r))])
#     # cliper_err   = haversine_km(to_deg(cliper_preds), to_deg(gt_r))
#     obs_deg = to_deg(denorm_traj(obs_n))
#     gt_deg  = to_deg(denorm_traj(gt_n))
    
#     if len(obs_deg) >= 2:
#         v_deg = obs_deg[-1] - obs_deg[-2] # Vận tốc tính bằng độ/6h
#         cliper_preds_deg = np.array([obs_deg[-1] + (k+1)*v_deg for k in range(len(gt_deg))])
#     else:
#         cliper_preds_deg = np.tile(obs_deg[-1], (len(gt_deg), 1))
        
#     cliper_err = haversine_km(cliper_preds_deg, gt_deg)

#     return obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg, errors_km, cliper_err


# # ── Single mode ────────────────────────────────────────────────────────────────

# def visualize_forecast(args):
#     set_seed(42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print(f"{'='*65}")
#     print(f"  TC-FM v11 Forecast  |  {args.tc_name}  @  {args.tc_date}")
#     print(f"{'='*65}\n")

#     detected = detect_pred_len(args.model_path)
#     if args.pred_len != detected:
#         print(f"  pred_len: {args.pred_len} → {detected}")
#         args.pred_len = detected

#     model = TCFlowMatching(
#         pred_len=args.pred_len, obs_len=args.obs_len).to(device)
#     ck = torch.load(args.model_path, map_location=device, weights_only=False)
#     sd = ck.get("model_state_dict", ck.get("model_state", ck))
#     model.load_state_dict(sd, strict=False)
#     model.eval()
#     print("  Model loaded\n")

#     dset, _ = data_loader(
#         args, {"root": args.TC_data_path, "type": args.dset_type},
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
#         for i in range(min(15, len(dset))):
#             info = dset[i][-1]
#             print(f"    [{i}]  {info['old'][1]}  @  {info['tydate'][args.obs_len]}")
#         return

#     (obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg,
#      errors_km, cliper_err) = run_inference(
#          model, target, device, args.ode_steps, args.num_ensemble)

#     print("  Track errors (km):")
#     for i, e in enumerate(errors_km):
#         mark = "  ◀" if (i+1) in [4, 8, 12] else ""
#         print(f"    +{(i+1)*6:3d}h : {e:6.1f} km{mark}")
#     print(f"    Mean  : {errors_km.mean():.1f} km\n")

#     all_deg   = np.vstack([obs_deg, gt_deg, pred_deg,
#                             ens_deg.reshape(-1, 2)])
#     margin    = 4.5
#     lon_range = (all_deg[:, 0].min() - margin, all_deg[:, 0].max() + margin)
#     lat_range = (all_deg[:, 1].min() - margin, all_deg[:, 1].max() + margin)

#     fig = plt.figure(figsize=(20, 10), facecolor=STYLE["bg_color"])
#     gs  = fig.add_gridspec(1, 3, wspace=0.10)
#     ax_map = make_map_ax(fig, gs[0, :2], lon_range, lat_range)
#     ax_err = fig.add_subplot(gs[0, 2])
#     ax_err.set_facecolor(STYLE["bg_color"])

#     dt_str = datetime.strptime(t_date, "%Y%m%d%H").strftime(
#         "%d %b %Y  %H:%M UTC")
#     fh = args.pred_len * 6

#     _plot_on_ax(
#         ax_map, lon_range, lat_range,
#         obs_deg, gt_deg, pred_deg, pred_Me_n,
#         all_trajs_deg=ens_deg if args.num_ensemble >= 3 else None,
#         errors_km=errors_km,
#         title=f"🌀 {t_name}  —  {fh}h FC  |  FM+PINN v11  (ens={args.num_ensemble})",
#         dt_str=dt_str,
#     )
#     plot_spread_over_time(ax_err, ens_deg, errors_km, cliper_err, t_name)

#     os.makedirs(args.output_dir, exist_ok=True)
#     out = os.path.join(args.output_dir,
#                        f"forecast_{fh}h_{t_name}_{t_date}.png")
#     plt.savefig(out, dpi=200, bbox_inches="tight",
#                 facecolor=STYLE["bg_color"])
#     plt.close()
#     print(f"  Saved → {out}\n")


# # ── Case-study grid ────────────────────────────────────────────────────────────

# def visualize_case_study(args):
#     set_seed(42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     detected      = detect_pred_len(args.model_path)
#     args.pred_len = detected

#     model = TCFlowMatching(
#         pred_len=args.pred_len, obs_len=args.obs_len).to(device)
#     ck = torch.load(args.model_path, map_location=device, weights_only=False)
#     sd = ck.get("model_state_dict", ck.get("model_state", ck))
#     model.load_state_dict(sd, strict=False)
#     model.eval()

#     dset, _ = data_loader(
#         args, {"root": args.TC_data_path, "type": "test"},
#         test=True, test_year=args.test_year,
#     )

#     cases = [
#         {"name": args.straight1_name, "date": args.straight1_date,
#          "label": "Straight-track 1"},
#         {"name": args.straight2_name, "date": args.straight2_date,
#          "label": "Straight-track 2"},
#         {"name": "WIPHA",             "date": args.recurv_date,
#          "label": "Recurvature — WIPHA"},
#     ]

#     fig = plt.figure(figsize=(22, 8 * len(cases)),
#                      facecolor=STYLE["bg_color"])
#     gs  = fig.add_gridspec(len(cases), 3, wspace=0.10, hspace=0.28)

#     for row, case in enumerate(cases):
#         t_name = case["name"].strip().upper()
#         t_date = str(case["date"]).strip()
#         label  = case.get("label", t_name)

#         target = None
#         for i in range(len(dset)):
#             item = dset[i]
#             info = item[-1]
#             if (t_name in str(info["old"][1]).strip().upper()
#                     and t_date == str(info["tydate"][args.obs_len]).strip()):
#                 target = item
#                 break

#         if target is None:
#             print(f"  ⚠  {t_name} @ {t_date} — not found")
#             for c in range(3):
#                 ax = fig.add_subplot(gs[row, c])
#                 ax.set_facecolor(STYLE["bg_color"])
#                 ax.text(0.5, 0.5, f"NOT FOUND\n{t_name}",
#                         ha="center", va="center", color="red",
#                         transform=ax.transAxes)
#             continue

#         (obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg,
#          errors_km, cliper_err) = run_inference(
#              model, target, device, args.ode_steps, args.num_ensemble)

#         all_deg   = np.vstack([obs_deg, gt_deg, pred_deg,
#                                 ens_deg.reshape(-1, 2)])
#         margin    = 4.5
#         lon_range = (all_deg[:, 0].min() - margin,
#                      all_deg[:, 0].max() + margin)
#         lat_range = (all_deg[:, 1].min() - margin,
#                      all_deg[:, 1].max() + margin)

#         dt_str  = datetime.strptime(t_date, "%Y%m%d%H").strftime(
#             "%d %b %Y %H:%M UTC")
#         ax_map  = make_map_ax(fig, gs[row, :2], lon_range, lat_range)
#         ax_err  = fig.add_subplot(gs[row, 2])
#         ax_err.set_facecolor(STYLE["bg_color"])

#         _plot_on_ax(
#             ax_map, lon_range, lat_range,
#             obs_deg, gt_deg, pred_deg, pred_Me_n,
#             all_trajs_deg=ens_deg if args.num_ensemble >= 3 else None,
#             errors_km=errors_km,
#             title=f"[{label}]  {t_name}",
#             dt_str=dt_str,
#         )
#         plot_spread_over_time(ax_err, ens_deg, errors_km, cliper_err, t_name)

#         print(f"  [{label}] ADE={errors_km.mean():.1f} km  "
#               f"72h={errors_km[11] if len(errors_km)>11 else float('nan'):.1f} km")

#     os.makedirs(args.output_dir, exist_ok=True)
#     out = os.path.join(args.output_dir, "case_study_grid_v11.png")
#     plt.savefig(out, dpi=150, bbox_inches="tight",
#                 facecolor=STYLE["bg_color"])
#     plt.close()
#     print(f"\n  Saved → {out}")


# # ── CLI ────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--model_path",      required=True)
#     p.add_argument("--TC_data_path",    required=True)
#     p.add_argument("--output_dir",      default="outputs")
#     p.add_argument("--mode",            default="single",
#                    choices=["single", "case_study"])
#     p.add_argument("--tc_name",         default="WIPHA")
#     p.add_argument("--tc_date",         default="2019073106")
#     p.add_argument("--dset_type",       default="test")
#     p.add_argument("--straight1_name",  default="BEBINCA")
#     p.add_argument("--straight1_date",  default="2018090806")
#     p.add_argument("--straight2_name",  default="MANGKHUT")
#     p.add_argument("--straight2_date",  default="2018091312")
#     p.add_argument("--recurv_date",     default="2019073106")
#     p.add_argument("--test_year",       type=int,   default=2019)
#     p.add_argument("--obs_len",         type=int,   default=8)
#     p.add_argument("--pred_len",        type=int,   default=12)
#     p.add_argument("--ode_steps",       type=int,   default=10)
#     p.add_argument("--num_ensemble",    type=int,   default=50)
#     p.add_argument("--batch_size",      type=int,   default=1)
#     p.add_argument("--num_workers",     type=int,   default=0)
#     p.add_argument("--delim",           default=" ")
#     p.add_argument("--skip",            type=int,   default=1)
#     p.add_argument("--min_ped",         type=int,   default=1)
#     p.add_argument("--threshold",       type=float, default=0.002)
#     p.add_argument("--other_modal",     default="gph")

#     args = p.parse_args()
#     if args.mode == "single":
#         visualize_forecast(args)
#     else:
#         visualize_case_study(args)

"""
scripts/visual_evaluate_model_Me_v13.py
========================================
TC-FlowMatching — Forecast Visualisation v13

Fixes vs v12:
  - [FIX-1] find_target: bỏ dòng no-op `dt.replace(hour=dt.hour)`;
            chuyển `from datetime import timedelta` ra ngoài module-level
  - [FIX-2] _search_one_date: `best_pri = 99` → `float("inf")` để không
            bị giới hạn khi obs_len > 99
  - [FIX-3] _gaussian_cone_boundary: guard `N < 2` (np.cov với N=1 trả
            về scalar gây crash np.linalg.eigh)
  - [FIX-4] visualize_case_study: subplot "NOT FOUND" dùng `gs[row, :]`
            thay vì loop 3 subplot riêng lẻ (tránh conflict với map)
  - [FIX-5] plot_spread_over_time: `ax_twin.fill_between` phải gọi trên
            `ax_twin`, không phải `ax` (trục Y sai khi fill error area)
  - [FIX-6] haversine_km: thêm guard shape để tránh silent broadcast lỗi
            khi 2 array có ndim khác nhau
  - [FIX-7] detect_pred_len: fallback duyệt key an toàn hơn với try/except
            để không crash khi checkpoint có cấu trúc lạ
  - [CLEAN] Tất cả import được đưa lên top-level; xoá dead code
"""
from __future__ import annotations

import os
import sys
import random
import argparse
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from scipy.stats import chi2

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("  Warning: cartopy not found — using plain axes.")

from Model.flow_matching_model import TCFlowMatching
from Model.data.loader import data_loader
from Model.data.trajectoriesWithMe_unet_training import seq_collate


# ── Styling ────────────────────────────────────────────────────────────────────
STYLE = dict(
    obs_color     = "#00CFFF",
    gt_color      = "#FF3B3B",
    pred_color    = "#39FF14",
    ens_color     = "#39FF14",
    ens_alpha     = 0.05,
    marker_size   = 8,
    lw_main       = 2.8,
    lw_thin       = 1.6,
    bg_color      = "#0D1B2A",
    land_color    = "#2D4A3E",
    ocean_color   = "#0D1B2A",
    border_color  = "#4A6FA5",
    grid_color    = "#FFFFFF",
    grid_alpha    = 0.12,
    error_color   = "#FFD700",
    title_pad     = 14,
    cone_50_fill  = "#39FF14",
    cone_90_fill  = "#00CFFF",
    cone_50_alpha = 0.22,
    cone_90_alpha = 0.10,
    cone_edge_lw  = 1.2,
)

_CHI2_50 = chi2.ppf(0.50, df=2)
_CHI2_90 = chi2.ppf(0.90, df=2)

INTENSITY = [
    (0,   34,  "TD",       "#99CCFF"),
    (34,  48,  "TS",       "#66FF66"),
    (48,  64,  "TY",       "#FFFF00"),
    (64,  84,  "Sev.TY",   "#FFA500"),
    (84,  115, "Vis.TY",   "#FF4500"),
    (115, 999, "Super TY", "#FF00FF"),
]


def wind_intensity(wind_kt):
    for lo, hi, name, color in INTENSITY:
        if lo <= wind_kt < hi:
            return name, color
    return "Super TY", "#FF00FF"


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_seed(s=42):
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


def denorm_traj(n):
    """Inverse-normalise trajectory array (any shape ending in 2)."""
    r = np.zeros_like(n)
    r[..., 0] = n[..., 0] * 50.0 + 1800.0
    r[..., 1] = n[..., 1] * 50.0
    return r


def to_deg(pts_01):
    return pts_01 / 10.0


def denorm_wind(wind_norm):
    return wind_norm * 25.0 + 40.0


def haversine_km(p1_deg, p2_deg):
    """
    Haversine distance (km) between two arrays of (lon, lat) points.
    Both arrays must have identical shape (..., 2).
    """
    # [FIX-6] Explicit shape check to catch silent broadcast errors early
    p1_deg = np.asarray(p1_deg)
    p2_deg = np.asarray(p2_deg)
    if p1_deg.shape != p2_deg.shape:
        raise ValueError(
            f"haversine_km: shape mismatch {p1_deg.shape} vs {p2_deg.shape}"
        )
    lat1 = np.deg2rad(p1_deg[..., 1])
    lat2 = np.deg2rad(p2_deg[..., 1])
    dlat = np.deg2rad(p2_deg[..., 1] - p1_deg[..., 1])
    dlon = np.deg2rad(p2_deg[..., 0] - p1_deg[..., 0])
    a    = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2.0 * 6371.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def detect_pred_len(ckpt_path):
    """
    Infer pred_len from the pos_enc shape stored in the checkpoint.
    [FIX-7] Wrapped in try/except so unusual checkpoint layouts don't crash.
    """
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck.get("model_state", ck))
        for key in ["net.pos_enc", "denoiser.pos_enc", "pos_enc"]:
            if key in sd:
                return sd[key].shape[1]
        for k, v in sd.items():
            if "pos_enc" in k and isinstance(v, torch.Tensor) and v.dim() == 3:
                return v.shape[1]
    except Exception as e:
        print(f"  [WARN] detect_pred_len failed ({e}); defaulting to 12")
    return 12


# ── Snap & Search ──────────────────────────────────────────────────────────────

def snap_to_6h(date_str: str) -> str:
    """Floor YYYYMMDDHH to the nearest prior 6-hour mark."""
    s  = str(date_str).strip()[:10]
    dt = datetime.strptime(s, "%Y%m%d%H")
    dt = dt.replace(hour=(dt.hour // 6) * 6, minute=0, second=0)
    return dt.strftime("%Y%m%d%H")


def resolve_date(raw_date: str) -> tuple[str, bool]:
    original    = str(raw_date).strip()[:10]
    snapped     = snap_to_6h(original)
    was_snapped = snapped != original
    if was_snapped:
        print(f"  [SNAP] {original} → {snapped}  "
              f"(làm tròn về mốc 6h gần nhất trước đó)")
    return snapped, was_snapped


def _search_one_date(dset, t_name: str, t_date: str, obs_len: int):
    """
    Scan the entire dataset for a sample matching t_name + t_date.

    Priority:
      0  — date falls exactly at index obs_len  (ideal)
      N  — date falls at index obs_len+N        (later window, still usable)

    Returns (item, matched_idx) or (None, None).

    [FIX-2] best_pri initialised to float("inf") instead of hardcoded 99,
            so datasets with obs_len > 99 are handled correctly.
    """
    best_item = None
    best_idx  = None
    best_pri  = float("inf")   # FIX-2

    for i in range(len(dset)):
        item = dset[i]
        info = item[-1]
        if t_name not in str(info["old"][1]).strip().upper():
            continue
        for idx, td in enumerate(info["tydate"]):
            if str(td).strip() != t_date:
                continue
            if idx < obs_len:
                continue
            pri = 0 if idx == obs_len else (idx - obs_len + 1)
            if pri < best_pri:
                best_item, best_idx, best_pri = item, idx, pri

    return best_item, best_idx


def find_target(
    dset,
    t_name: str,
    t_date: str,
    obs_len: int,
    max_forward_steps: int = 20,
):
    """
    Flexible sample search:
      1. Try t_date (already snapped to 6h).
      2. If not found (e.g. TC track too short at that timestamp),
         advance by 6h up to max_forward_steps times.

    Returns (item, matched_idx, actual_date) or (None, None, None).

    [FIX-1] Removed the dead no-op `dt.replace(hour=dt.hour)`.
            `timedelta` import is now at module level (not inside the loop).
    """
    dt = datetime.strptime(t_date, "%Y%m%d%H")

    for step in range(max_forward_steps + 1):
        candidate = dt.strftime("%Y%m%d%H")
        item, idx = _search_one_date(dset, t_name, candidate, obs_len)
        if item is not None:
            if step > 0:
                print(
                    f"  [AUTO-FORWARD] {t_date} không có dữ liệu → "
                    f"dùng mốc tiếp theo: {candidate}  (+{step * 6}h)"
                )
            return item, idx, candidate
        dt = dt + timedelta(hours=6)   # FIX-1: no dead replace(); timedelta at top

    return None, None, None


def list_available(dset, t_name: str, obs_len: int, limit: int = 30):
    """Print available timestamps for TC t_name in the dataset."""
    shown = 0
    seen  = set()
    for i in range(len(dset)):
        info = dset[i][-1]
        name = str(info["old"][1]).strip().upper()
        if t_name not in name:
            continue
        td = str(info["tydate"][obs_len]).strip()
        if td in seen:
            continue
        seen.add(td)
        print(f"    {name:<15s}  @  {td}")
        shown += 1
        if shown >= limit:
            break

    if shown == 0:
        print(f"  (Không tìm thấy TC '{t_name}' trong dataset)")
        print("  Một số TC có sẵn:")
        seen_names: set[str] = set()
        for i in range(len(dset)):
            info = dset[i][-1]
            n = str(info["old"][1]).strip().upper()
            if n in seen_names:
                continue
            seen_names.add(n)
            print(f"    {n}")
            if len(seen_names) >= 15:
                break


# ── NHC-style smooth probability cone ─────────────────────────────────────────

def _gaussian_cone_boundary(pts_deg, chi2_thresh):
    """
    Fit a 2-D Gaussian to pts_deg and return the chi2 confidence ellipse.

    [FIX-3] np.cov with N==1 returns a scalar (not a 2×2 matrix), which
            crashes np.linalg.eigh.  Guard raised to N >= 3 to also ensure
            the covariance estimate is meaningful.
    """
    if len(pts_deg) < 3:          # FIX-3: was `< 3` in name only; enforce here
        return None
    mu               = pts_deg.mean(axis=0)
    cov              = np.cov(pts_deg.T)
    if cov.ndim < 2:               # FIX-3: scalar guard for N==1 or N==2 edge case
        return None
    cov             += np.eye(2) * 1e-8
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals          = np.maximum(eigvals, 1e-8)
    a                = np.sqrt(chi2_thresh * eigvals[-1])
    b                = np.sqrt(chi2_thresh * eigvals[0])
    theta            = np.linspace(0, 2 * np.pi, 64)
    ell              = np.stack([a * np.cos(theta), b * np.sin(theta)], axis=1)
    return ell @ eigvecs.T + mu


def draw_smooth_cone(ax, ens_deg, cur_pos_deg, transform=None):
    S, T, _ = ens_deg.shape
    if S < 3:
        return

    def _fill(verts, color, alpha, zo):
        v  = np.vstack([verts, verts[0]])
        kw = dict(color=color, alpha=alpha, zorder=zo, linewidth=0)
        if HAS_CARTOPY and transform is not None:
            ax.fill(v[:, 0], v[:, 1], transform=transform, **kw)
        else:
            ax.fill(v[:, 0], v[:, 1], **kw)

    def _line(xs, ys, color, alpha, lw, zo, ls="-"):
        kw = dict(color=color, alpha=alpha, linewidth=lw, zorder=zo, linestyle=ls)
        if HAS_CARTOPY and transform is not None:
            ax.plot(xs, ys, transform=transform, **kw)
        else:
            ax.plot(xs, ys, **kw)

    means     = np.array([ens_deg[:, t, :].mean(axis=0) for t in range(T)])
    track_pts = np.vstack([cur_pos_deg, means])

    def _perp(p1, p2):
        d = p2 - p1
        n = np.linalg.norm(d)
        if n < 1e-10:
            return np.array([0.0, 1.0])
        d /= n
        return np.array([-d[1], d[0]])

    def _cone_edges(chi2_thresh):
        left  = [cur_pos_deg.copy()]
        right = [cur_pos_deg.copy()]
        for t in range(T):
            b = _gaussian_cone_boundary(ens_deg[:, t, :], chi2_thresh)
            if b is None:
                left.append(means[t])
                right.append(means[t])
                continue
            perp = (
                _perp(track_pts[t], track_pts[t + 1])
                if t + 1 < len(track_pts)
                else _perp(track_pts[t - 1], track_pts[t])
            )
            proj = (b - means[t]) @ perp
            left.append(b[proj.argmax()])
            right.append(b[proj.argmin()])
        return np.array(left), np.array(right)

    l90, r90 = _cone_edges(_CHI2_90)
    _fill(np.vstack([l90, r90[::-1]]), STYLE["cone_90_fill"], STYLE["cone_90_alpha"], 3)
    _line(l90[:, 0], l90[:, 1], STYLE["cone_90_fill"], 0.35, STYLE["cone_edge_lw"],        4, "--")
    _line(r90[:, 0], r90[:, 1], STYLE["cone_90_fill"], 0.35, STYLE["cone_edge_lw"],        4, "--")

    l50, r50 = _cone_edges(_CHI2_50)
    _fill(np.vstack([l50, r50[::-1]]), STYLE["cone_50_fill"], STYLE["cone_50_alpha"], 5)
    _line(l50[:, 0], l50[:, 1], STYLE["cone_50_fill"], 0.6, STYLE["cone_edge_lw"] * 1.3, 6)
    _line(r50[:, 0], r50[:, 1], STYLE["cone_50_fill"], 0.6, STYLE["cone_edge_lw"] * 1.3, 6)

    for s in range(S):
        xs = np.concatenate([[cur_pos_deg[0]], ens_deg[s, :, 0]])
        ys = np.concatenate([[cur_pos_deg[1]], ens_deg[s, :, 1]])
        kw = dict(color=STYLE["ens_color"], linewidth=0.25,
                  alpha=STYLE["ens_alpha"], zorder=2)
        if HAS_CARTOPY and transform is not None:
            ax.plot(xs, ys, transform=transform, **kw)
        else:
            ax.plot(xs, ys, **kw)


# ── Spread panel ───────────────────────────────────────────────────────────────

def plot_spread_over_time(ax, ens_deg, errors_km, cliper_err_km, t_name):
    """
    Left axis  (ax)       : ensemble spread 1σ  [km]
    Right axis (ax_twin)  : track error & CLIPER [km]

    [FIX-5] Error fill_between was incorrectly called on `ax` (spread axis)
            instead of `ax_twin` (error axis), causing it to be drawn against
            the wrong Y scale and potentially hidden under the spread fill.
    """
    S, T, _ = ens_deg.shape
    lead_h   = np.arange(1, T + 1) * 6

    spreads_km = []
    for t in range(T):
        pts        = ens_deg[:, t, :]
        mean_lat   = pts[:, 1].mean()
        std_lon_km = pts[:, 0].std() * 111.32 * np.cos(np.deg2rad(mean_lat))
        std_lat_km = pts[:, 1].std() * 110.57
        spreads_km.append(np.sqrt(std_lon_km ** 2 + std_lat_km ** 2))
    spreads_km = np.array(spreads_km)

    ax.set_facecolor(STYLE["bg_color"])
    ax.fill_between(lead_h, 0, spreads_km, alpha=0.25,
                    color=STYLE["cone_50_fill"], label="Ensemble spread (1σ)")
    ax.plot(lead_h, spreads_km, "-", color=STYLE["cone_50_fill"], lw=2.2, zorder=5)

    ax_twin = ax.twinx()
    ax_twin.set_facecolor(STYLE["bg_color"])
    ax_twin.plot(lead_h, errors_km, "o-", color=STYLE["pred_color"],
                 lw=2.5, ms=5, label="FM+PINN ADE", zorder=6)
    # [FIX-5] Use ax_twin (not ax) so the fill is scaled to the error Y-axis
    ax_twin.fill_between(lead_h, 0, errors_km, alpha=0.12, color=STYLE["pred_color"])

    if cliper_err_km is not None:
        ax_twin.plot(lead_h, cliper_err_km[:T], "s--",
                     color="#FF6666", lw=2, ms=4, label="CLIPER", zorder=4)

    for xm in [24, 48, 72]:
        ax.axvline(xm, color=STYLE["error_color"], alpha=0.2, lw=0.7, ls=":")

    ax.set_xlabel("Lead time (h)", color="white", fontsize=8)
    ax.set_ylabel("Spread 1σ (km)", color=STYLE["cone_50_fill"], fontsize=8)
    ax_twin.set_ylabel("Track error (km)", color=STYLE["pred_color"], fontsize=8)
    ax.set_title(f"Spread vs Error — {t_name}", color="white",
                 fontsize=9, fontweight="bold")

    lines1, lbs1 = ax.get_legend_handles_labels()
    lines2, lbs2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbs1 + lbs2, fontsize=7.5,
              facecolor="#111111", edgecolor="#00CFFF",
              labelcolor="white", loc="upper left")

    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.tick_params(colors="white", labelsize=7)
    ax_twin.tick_params(colors="white", labelsize=7)
    ax_twin.yaxis.label.set_color(STYLE["pred_color"])
    ax.yaxis.label.set_color(STYLE["cone_50_fill"])


# ── Map setup ──────────────────────────────────────────────────────────────────

def make_map_ax(fig, subplot_spec, lon_range, lat_range):
    if HAS_CARTOPY:
        ax = fig.add_subplot(
            subplot_spec,
            projection=ccrs.PlateCarree(central_longitude=0),
        )
        ax.set_extent(
            [lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
            crs=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                       facecolor=STYLE["ocean_color"], zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"),
                       facecolor=STYLE["land_color"], zorder=1, alpha=0.9)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                       edgecolor="#8ABCD1", linewidth=0.7, zorder=2)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                       edgecolor=STYLE["border_color"],
                       linewidth=0.4, linestyle=":", zorder=2)
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True,
            linewidth=0.5, color=STYLE["grid_color"],
            alpha=STYLE["grid_alpha"], linestyle="--",
        )
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = dict(color="white", fontsize=7)
        gl.ylabel_style = dict(color="white", fontsize=7)
    else:
        ax = fig.add_subplot(subplot_spec)
        ax.set_facecolor(STYLE["bg_color"])
        ax.set_xlim(*lon_range)
        ax.set_ylim(*lat_range)
        for lon in np.arange(np.ceil(lon_range[0] / 5) * 5, lon_range[1], 5):
            ax.axvline(lon, color="white", alpha=STYLE["grid_alpha"], lw=0.5)
        for lat in np.arange(np.ceil(lat_range[0] / 5) * 5, lat_range[1], 5):
            ax.axhline(lat, color="white", alpha=STYLE["grid_alpha"], lw=0.5)
        ax.set_xlabel("Longitude (°E)", color="white", fontsize=8)
        ax.set_ylabel("Latitude (°N)",  color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
    return ax


def _plot_on_ax(
    ax, lon_range, lat_range,
    obs_deg, gt_deg, pred_deg, pred_Me_deg,
    all_trajs_deg=None, errors_km=None,
    title="", dt_str="",
):
    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    outline   = [pe.withStroke(linewidth=2.5, foreground="black")]
    cur_pos   = obs_deg[-1]

    def _plot(x, y, fmt=None, **kw):
        args_ = [x, y] + ([fmt] if fmt is not None else [])
        if HAS_CARTOPY:
            ax.plot(*args_, transform=transform, **kw)
        else:
            ax.plot(*args_, **kw)

    def _scatter(x, y, **kw):
        if HAS_CARTOPY:
            ax.scatter(x, y, transform=transform, **kw)
        else:
            ax.scatter(x, y, **kw)

    def _text(x, y, s, **kw):
        if HAS_CARTOPY:
            ax.text(x, y, s, transform=transform, **kw)
        else:
            ax.text(x, y, s, **kw)

    # 1. Probability cone
    if all_trajs_deg is not None and all_trajs_deg.shape[0] >= 3:
        draw_smooth_cone(ax, all_trajs_deg, cur_pos, transform)

    # 2. Observed track
    _plot(obs_deg[:, 0], obs_deg[:, 1], fmt="o-",
          color=STYLE["obs_color"], linewidth=STYLE["lw_thin"], markersize=5,
          markeredgecolor="white", markeredgewidth=0.8,
          zorder=7, path_effects=outline)

    # 3. Ground truth
    gt_lon = np.concatenate([[cur_pos[0]], gt_deg[:, 0]])
    gt_lat = np.concatenate([[cur_pos[1]], gt_deg[:, 1]])
    _plot(gt_lon, gt_lat, fmt="o-",
          color=STYLE["gt_color"], linewidth=STYLE["lw_main"],
          markersize=STYLE["marker_size"],
          markeredgecolor="white", markeredgewidth=1.2,
          zorder=8, path_effects=outline)

    # 4. Predicted track (ensemble mean)
    pred_lon = np.concatenate([[cur_pos[0]], pred_deg[:, 0]])
    pred_lat = np.concatenate([[cur_pos[1]], pred_deg[:, 1]])
    _plot(pred_lon, pred_lat, fmt="o-",
          color=STYLE["pred_color"], linewidth=STYLE["lw_main"],
          markersize=STYLE["marker_size"],
          markeredgecolor="#003300", markeredgewidth=1.0,
          zorder=9, path_effects=outline)

    # 5. Wind intensity markers
    if pred_Me_deg is not None:
        for i in range(len(pred_deg)):
            wnd_kt  = denorm_wind(float(pred_Me_deg[i, 1]))
            _, wcolor = wind_intensity(wnd_kt)
            _scatter([pred_deg[i, 0]], [pred_deg[i, 1]],
                     s=70, color=wcolor,
                     edgecolors="white", linewidths=0.7, zorder=11)

    # 6. Error connectors at 24/48/72h
    if errors_km is not None:
        for si, lbl in {3: "24h", 7: "48h", 11: "72h"}.items():
            if si < len(gt_deg) and si < len(pred_deg):
                gx, gy = gt_deg[si, 0], gt_deg[si, 1]
                px, py = pred_deg[si, 0], pred_deg[si, 1]
                if HAS_CARTOPY:
                    ax.plot([gx, px], [gy, py], "--",
                            color=STYLE["error_color"], linewidth=1.2,
                            alpha=0.7, transform=transform, zorder=7)
                else:
                    ax.plot([gx, px], [gy, py], "--",
                            color=STYLE["error_color"], linewidth=1.2,
                            alpha=0.7, zorder=7)
                _text(
                    (gx + px) / 2, (gy + py) / 2,
                    f" {lbl}\n{errors_km[si]:.0f}km",
                    fontsize=7, color=STYLE["error_color"],
                    ha="center", va="bottom", zorder=14,
                    path_effects=outline,
                )

    # 7. Lead-time labels every 24h for both GT and Pred
    for i in range(len(pred_lon)):
        h   = i * 6
        if h % 24 == 0:
            lbl = "NOW" if i == 0 else f"+{h}h"
            _text(pred_lon[i], pred_lat[i] + 0.5, lbl,
                  color="#AAFFAA", fontweight="bold", fontsize=7.5,
                  path_effects=outline)
            if i < len(gt_lon):
                _text(gt_lon[i], gt_lat[i] - 0.7, lbl,
                      color="#FFAAAA", fontsize=6, alpha=0.8,
                      path_effects=outline)

    # 8. NOW star
    _scatter([cur_pos[0]], [cur_pos[1]],
             s=350, marker="*", color="#FFD700",
             edgecolors="#FF4400", linewidths=2, zorder=20)

    # 9. Error summary box
    if errors_km is not None:
        n     = len(errors_km)
        lines = [f"Mean: {errors_km.mean():.0f} km"]
        for si, lh in [(3, 24), (7, 48), (11, 72)]:
            if si < n:
                lines.append(f" {lh}h: {errors_km[si]:.0f} km")
        ax.text(
            0.02, 0.03, "\n".join(lines),
            transform=ax.transAxes, fontsize=8, va="bottom",
            color="#88FF88", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc=STYLE["bg_color"],
                      alpha=0.85, ec="white", lw=0.8),
            zorder=16,
        )

    # 10. Legends
    track_handles = [
        Line2D([0], [0], color=STYLE["obs_color"],  lw=2,   label="Observed"),
        Line2D([0], [0], color=STYLE["gt_color"],   lw=2,   label="Ground truth"),
        Line2D([0], [0], color=STYLE["pred_color"], lw=2.5, label="FM+PINN (mean)"),
        mpatches.Patch(facecolor=STYLE["cone_50_fill"], alpha=0.6,
                       label="50% prob. cone"),
        mpatches.Patch(facecolor=STYLE["cone_90_fill"], alpha=0.45,
                       label="90% prob. cone"),
    ]
    ax.legend(handles=track_handles, loc="lower right", fontsize=7.5,
              facecolor="#111111", edgecolor="#00CFFF", labelcolor="white")

    wind_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=c, markersize=7,
               markeredgecolor="white", markeredgewidth=0.5,
               label=f"{nm} ({lo}–{hi}kt)")
        for lo, hi, nm, c in INTENSITY
    ]
    leg2 = ax.legend(
        handles=wind_handles, loc="upper right", fontsize=6.5,
        facecolor="#111111", edgecolor="#00FFFF",
        labelcolor="white", title="Wind (kt)",
        title_fontsize=7, ncol=2,
    )
    ax.add_artist(leg2)

    ax.set_title(
        f"{title}\n{dt_str}", color="white", fontsize=10,
        fontweight="bold", pad=STYLE["title_pad"],
        bbox=dict(fc=STYLE["bg_color"], alpha=0.88, ec="#00FFFF", lw=1.5),
    )
    ax.set_facecolor(STYLE["bg_color"])
    for spine in ax.spines.values():
        spine.set_edgecolor("#334466")


# ── Run inference ──────────────────────────────────────────────────────────────

def _extract_seq(tensor, batch_idx=0):
    """
    Safely extract the trajectory for one sample from a batch tensor.

    seq_collate produces obs/gt with shape  [T, B, F]  (time-first).
    model.sample()  produces pred with shape [B, T, F]  (batch-first).
    all_trajs has shape [S, B, T, F]  (ensemble-first, batch-second).

    This function normalises both conventions → returns [T, F].

    Heuristic: if dim-0 >> dim-1, assume time-first [T, B, F];
               otherwise assume batch-first [B, T, F].
    We also print shapes once so bugs are immediately visible.
    """
    t = tensor.cpu()
    if t.dim() == 3:
        d0, d1, _ = t.shape
        # If d0 looks like a time axis (≥ d1 * 2 or d1 == 1), treat as [T, B, F]
        if d1 == 1 or d0 >= d1 * 2:
            return t[:, batch_idx, :].numpy()      # [T, B, F] → [T, F]
        else:
            return t[batch_idx, :, :].numpy()      # [B, T, F] → [T, F]
    raise ValueError(f"_extract_seq: unexpected tensor dim {t.dim()}, shape {t.shape}")


def _extract_ens(all_trajs, batch_idx=0):
    """
    Extract ensemble trajectories for one sample.
    Expected shape: [S, B, T, F]  →  returns [S, T, F]
    Also handles [S, T, B, F] just in case.
    """
    t = all_trajs.cpu()
    if t.dim() == 4:
        S, d1, d2, F = t.shape
        if d1 == 1 or d2 > d1:
            # [S, B, T, F]
            return t[:, batch_idx, :, :].numpy()   # → [S, T, F]
        else:
            # [S, T, B, F]
            return t[:, :, batch_idx, :].numpy()   # → [S, T, F]
    raise ValueError(f"_extract_ens: unexpected tensor dim {t.dim()}, shape {t.shape}")


def run_inference(model, target, device, ode_steps, num_ensemble):
    batch = move_batch(seq_collate([target]), device)
    with torch.no_grad():
        pred_mean, pred_Me, all_trajs = model.sample(
            batch, num_ensemble=num_ensemble, ddim_steps=ode_steps
        )

    # ── Debug: print shapes once so axis confusion is immediately visible ──
    print(f"  [shape] batch[0]   (obs)      : {tuple(batch[0].shape)}")
    print(f"  [shape] batch[1]   (gt)       : {tuple(batch[1].shape)}")
    print(f"  [shape] pred_mean             : {tuple(pred_mean.shape)}")
    print(f"  [shape] pred_Me               : {tuple(pred_Me.shape)}")
    print(f"  [shape] all_trajs             : {tuple(all_trajs.shape)}")

    obs_n     = _extract_seq(batch[0])       # [T_obs,  2]
    gt_n      = _extract_seq(batch[1])       # [T_pred, 2]
    pred_n    = _extract_seq(pred_mean)      # [T_pred, 2]
    pred_Me_n = _extract_seq(pred_Me)        # [T_pred, F_me]
    ens_n     = _extract_ens(all_trajs)      # [S, T_pred, 2]

    # ── Deep value debug ──────────────────────────────────────────────────
    print(f"\n  [raw]  obs_n  (first 2 rows):\n{obs_n[:2]}")
    print(f"  [raw]  gt_n   (first 2 rows):\n{gt_n[:2]}")
    print(f"  [raw]  pred_n (first 2 rows):\n{pred_n[:2]}")

    obs_deg  = to_deg(denorm_traj(obs_n))
    gt_deg   = to_deg(denorm_traj(gt_n))
    pred_deg = to_deg(denorm_traj(pred_n))
    ens_deg  = to_deg(denorm_traj(ens_n))

    print(f"\n  [deg]  obs_deg  (first→last): {obs_deg[0]}  …  {obs_deg[-1]}")
    print(f"  [deg]  gt_deg   (first→last): {gt_deg[0]}  …  {gt_deg[-1]}")
    print(f"  [deg]  pred_deg (first→last): {pred_deg[0]}  …  {pred_deg[-1]}")
    print(f"  [deg]  ens_deg  mean t=0   : {ens_deg[:, 0, :].mean(axis=0)}")
    print(f"  [deg]  ens_deg  mean t=-1  : {ens_deg[:, -1, :].mean(axis=0)}")
    print(f"  NOTE: expected lon ~120-160°E, lat ~10-50°N")
    print(f"        if lon<50 or lat>100 → feature order is [lat,lon] → need swap!\n")

    errors_km = haversine_km(pred_deg, gt_deg)

    # CLIPER: constant-velocity extrapolation from last two observed points
    if len(obs_deg) >= 2:
        v_deg            = obs_deg[-1] - obs_deg[-2]
        cliper_preds_deg = np.array(
            [obs_deg[-1] + (k + 1) * v_deg for k in range(len(gt_deg))]
        )
    else:
        cliper_preds_deg = np.tile(obs_deg[-1], (len(gt_deg), 1))

    cliper_err = haversine_km(cliper_preds_deg, gt_deg)

    return obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg, errors_km, cliper_err


# ── Load model & dataset ───────────────────────────────────────────────────────

def load_model_and_data(args, device, dset_type="test"):
    detected = detect_pred_len(args.model_path)
    if args.pred_len != detected:
        print(f"  pred_len: {args.pred_len} → {detected}")
        args.pred_len = detected

    model = TCFlowMatching(pred_len=args.pred_len, obs_len=args.obs_len).to(device)
    ck    = torch.load(args.model_path, map_location=device, weights_only=False)
    sd    = ck.get("model_state_dict", ck.get("model_state", ck))
    model.load_state_dict(sd, strict=False)
    model.eval()
    print("  Model loaded\n")

    dset, _ = data_loader(
        args,
        {"root": args.TC_data_path, "type": dset_type},
        test=True,
        test_year=args.test_year,
    )
    print(f"  Dataset: {len(dset)} samples\n")
    return model, dset


# ── Single mode ────────────────────────────────────────────────────────────────

def visualize_forecast(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_name              = args.tc_name.strip().upper()
    t_date, was_snapped = resolve_date(args.tc_date)

    print(f"{'=' * 65}")
    print(f"  TC-FM v13  |  {t_name}  @  {t_date}")
    print(f"{'=' * 65}\n")

    model, dset = load_model_and_data(args, device, args.dset_type)

    target, matched_obs_len, actual_date = find_target(
        dset, t_name, t_date, args.obs_len
    )

    if target is None:
        print(f"  '{t_name} @ {t_date}' not found (kể cả sau khi thử tiến 20 mốc).")
        print(f"\n  Các thời điểm có sẵn của '{t_name}':")
        list_available(dset, t_name, args.obs_len)
        return

    if actual_date != t_date:
        t_date = actual_date

    if matched_obs_len != args.obs_len:
        print(
            f"  [INFO] Dùng tydate[{matched_obs_len}] thay vì [{args.obs_len}] "
            f"(date khớp ở window khác)\n"
        )

    print(f"  Found: {t_name} @ {t_date}\n")

    (obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg,
     errors_km, cliper_err) = run_inference(
        model, target, device, args.ode_steps, args.num_ensemble
    )

    print("  Track errors (km):")
    for i, e in enumerate(errors_km):
        mark = "  ◀" if (i + 1) in [4, 8, 12] else ""
        print(f"    +{(i + 1) * 6:3d}h : {e:6.1f} km{mark}")
    print(f"    Mean  : {errors_km.mean():.1f} km\n")

    all_deg   = np.vstack([obs_deg, gt_deg, pred_deg, ens_deg.reshape(-1, 2)])
    margin    = 4.5
    lon_range = (all_deg[:, 0].min() - margin, all_deg[:, 0].max() + margin)
    lat_range = (all_deg[:, 1].min() - margin, all_deg[:, 1].max() + margin)

    fig    = plt.figure(figsize=(20, 10), facecolor=STYLE["bg_color"])
    gs     = fig.add_gridspec(1, 3, wspace=0.10)
    ax_map = make_map_ax(fig, gs[0, :2], lon_range, lat_range)
    ax_err = fig.add_subplot(gs[0, 2])
    ax_err.set_facecolor(STYLE["bg_color"])

    dt_str    = datetime.strptime(t_date, "%Y%m%d%H").strftime("%d %b %Y  %H:%M UTC")
    fh        = args.pred_len * 6
    snap_note = f" [snapped from {args.tc_date}]" if was_snapped else ""

    _plot_on_ax(
        ax_map, lon_range, lat_range,
        obs_deg, gt_deg, pred_deg, pred_Me_n,
        all_trajs_deg=ens_deg if args.num_ensemble >= 3 else None,
        errors_km=errors_km,
        title=(
            f"🌀 {t_name}  —  {fh}h FC  |  FM+PINN v13"
            f"  (ens={args.num_ensemble}){snap_note}"
        ),
        dt_str=dt_str,
    )
    plot_spread_over_time(ax_err, ens_deg, errors_km, cliper_err, t_name)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"forecast_{fh}h_{t_name}_{t_date}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=STYLE["bg_color"])
    plt.close()
    print(f"  Saved → {out}\n")


# ── Case-study grid ────────────────────────────────────────────────────────────

def visualize_case_study(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, dset = load_model_and_data(args, device, "test")

    cases = [
        {"name": args.straight1_name, "date": args.straight1_date,
         "label": "Straight-track 1"},
        {"name": args.straight2_name, "date": args.straight2_date,
         "label": "Straight-track 2"},
        {"name": "WIPHA",             "date": args.recurv_date,
         "label": "Recurvature — WIPHA"},
    ]

    fig = plt.figure(figsize=(22, 8 * len(cases)), facecolor=STYLE["bg_color"])
    gs  = fig.add_gridspec(len(cases), 3, wspace=0.10, hspace=0.28)

    for row, case in enumerate(cases):
        t_name              = case["name"].strip().upper()
        t_date, was_snapped = resolve_date(case["date"])
        label               = case.get("label", t_name)

        target, matched_obs_len, actual_date = find_target(
            dset, t_name, t_date, args.obs_len
        )

        if target is None:
            print(f"  ⚠  {t_name} @ {t_date} — not found")
            # [FIX-4] Use a single spanning subplot instead of 3 separate ones
            # that would conflict with the 2-column map layout.
            ax_nf = fig.add_subplot(gs[row, :])
            ax_nf.set_facecolor(STYLE["bg_color"])
            ax_nf.text(
                0.5, 0.5, f"NOT FOUND\n{t_name}",
                ha="center", va="center", color="red",
                fontsize=14, transform=ax_nf.transAxes,
            )
            ax_nf.axis("off")
            continue

        if actual_date != t_date:
            t_date = actual_date

        if matched_obs_len != args.obs_len:
            print(
                f"  [INFO] {t_name}: dùng tydate[{matched_obs_len}] "
                f"thay vì [{args.obs_len}]"
            )

        (obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg,
         errors_km, cliper_err) = run_inference(
            model, target, device, args.ode_steps, args.num_ensemble
        )

        all_deg   = np.vstack([obs_deg, gt_deg, pred_deg, ens_deg.reshape(-1, 2)])
        margin    = 4.5
        lon_range = (all_deg[:, 0].min() - margin, all_deg[:, 0].max() + margin)
        lat_range = (all_deg[:, 1].min() - margin, all_deg[:, 1].max() + margin)

        dt_str    = datetime.strptime(t_date, "%Y%m%d%H").strftime("%d %b %Y %H:%M UTC")
        snap_note = f" [snapped from {case['date']}]" if was_snapped else ""
        ax_map    = make_map_ax(fig, gs[row, :2], lon_range, lat_range)
        ax_err    = fig.add_subplot(gs[row, 2])
        ax_err.set_facecolor(STYLE["bg_color"])

        _plot_on_ax(
            ax_map, lon_range, lat_range,
            obs_deg, gt_deg, pred_deg, pred_Me_n,
            all_trajs_deg=ens_deg if args.num_ensemble >= 3 else None,
            errors_km=errors_km,
            title=f"[{label}]  {t_name}{snap_note}",
            dt_str=dt_str,
        )
        plot_spread_over_time(ax_err, ens_deg, errors_km, cliper_err, t_name)

        ade = errors_km.mean()
        e72 = errors_km[11] if len(errors_km) > 11 else float("nan")
        print(f"  [{label}] ADE={ade:.1f} km  72h={e72:.1f} km")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "case_study_grid_v13.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=STYLE["bg_color"])
    plt.close()
    print(f"\n  Saved → {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",     required=True)
    p.add_argument("--TC_data_path",   required=True)
    p.add_argument("--output_dir",     default="outputs")
    p.add_argument("--mode",           default="single",
                   choices=["single", "case_study"])
    p.add_argument("--tc_name",        default="WIPHA")
    p.add_argument("--tc_date",        default="2019073106")
    p.add_argument("--dset_type",      default="test")
    p.add_argument("--straight1_name", default="BEBINCA")
    p.add_argument("--straight1_date", default="2018090806")
    p.add_argument("--straight2_name", default="MANGKHUT")
    p.add_argument("--straight2_date", default="2018091312")
    p.add_argument("--recurv_date",    default="2019073106")
    p.add_argument("--test_year",      type=int,   default=2019)
    p.add_argument("--obs_len",        type=int,   default=8)
    p.add_argument("--pred_len",       type=int,   default=12)
    p.add_argument("--ode_steps",      type=int,   default=10)
    p.add_argument("--num_ensemble",   type=int,   default=50)
    p.add_argument("--batch_size",     type=int,   default=1)
    p.add_argument("--num_workers",    type=int,   default=0)
    p.add_argument("--delim",          default=" ")
    p.add_argument("--skip",           type=int,   default=1)
    p.add_argument("--min_ped",        type=int,   default=1)
    p.add_argument("--threshold",      type=float, default=0.002)
    p.add_argument("--other_modal",    default="gph")

    args = p.parse_args()
    if args.mode == "single":
        visualize_forecast(args)
    else:
        visualize_case_study(args)
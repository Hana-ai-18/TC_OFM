"""
scripts/visual_evaluate_model_Me_v10.py
========================================
TC-FlowMatching v10 — Forecast Visualisation (fully fixed)

Fixes vs v9:
  - env_data index: reads batch[13] consistently (matches seq_collate)
  - Proper denorm: norm*50+1800 for lon, norm*50 for lat (0.1° units)
  - Wind speed display from pred_Me (knots, denorm: norm*25+40)
  - Cartopy map with coastlines, lat/lon grid, land fill
  - Per-step error labels on predicted track
  - Ensemble spread shown as shaded cone
  - Supports both single and case_study modes

Usage:
    # Single typhoon
    python scripts/visual_evaluate_model_Me_v10.py \
        --mode single \
        --model_path runs/v10_turbo/best_model.pth \
        --TC_data_path /path/to/TCND_vn \
        --tc_name WIPHA \
        --tc_date 2019073106 \
        --output_dir outputs

    # Case-study grid
    python scripts/visual_evaluate_model_Me_v10.py \
        --mode case_study \
        --model_path runs/v10_turbo/best_model.pth \
        --TC_data_path /path/to/TCND_vn \
        --output_dir outputs
"""
from __future__ import annotations

import os
import sys
import random
import argparse
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("  Warning: cartopy not found — using plain axes. "
          "Install with: pip install cartopy")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from Model.flow_matching_model import TCFlowMatching
from Model.data.loader import data_loader
from Model.data.trajectoriesWithMe_unet_training import seq_collate


# ── Styling ────────────────────────────────────────────────────────────────────
STYLE = dict(
    obs_color      = "#00CFFF",   # cyan
    gt_color       = "#FF3B3B",   # red  — actual track
    pred_color     = "#39FF14",   # neon green — predicted track
    ens_color      = "#39FF14",
    ens_alpha      = 0.12,
    cone_alpha     = 0.18,
    marker_size    = 7,
    lw_main        = 2.8,
    lw_thin        = 1.6,
    bg_color       = "#0D1B2A",
    land_color     = "#2D4A3E",
    ocean_color    = "#0D1B2A",
    border_color   = "#4A6FA5",
    grid_color     = "#FFFFFF",
    grid_alpha     = 0.12,
    error_color    = "#FFD700",
    text_shadow    = pe.withStroke(linewidth=2.5, foreground="black"),
    title_pad      = 14,
)

# Wind speed → intensity category (kt)
INTENSITY = [
    (0,  34,  "TD",      "#99CCFF"),
    (34, 48,  "TS",      "#66FF66"),
    (48, 64,  "TY",      "#FFFF00"),
    (64, 84,  "Sev.TY",  "#FFA500"),
    (84, 115, "Vis.TY",  "#FF4500"),
    (115, 999,"Super TY","#FF00FF"),
]

def wind_intensity(wind_kt: float) -> Tuple[str, str]:
    for lo, hi, name, color in INTENSITY:
        if lo <= wind_kt < hi:
            return name, color
    return "Super TY", "#FF00FF"


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_seed(s: int = 42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
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


def denorm_traj(n: np.ndarray) -> np.ndarray:
    """
    Normalised → 0.1° units.
      lon_norm = (lon_01E - 1800) / 50  →  lon_01E = norm*50 + 1800
      lat_norm = lat_01N / 50           →  lat_01N = norm*50
    """
    r = np.zeros_like(n)
    r[..., 0] = n[..., 0] * 50.0 + 1800.0   # 0.1°E
    r[..., 1] = n[..., 1] * 50.0             # 0.1°N
    return r


def to_deg(pts_01: np.ndarray) -> np.ndarray:
    """0.1° units → degrees."""
    return pts_01 / 10.0


def denorm_wind(wind_norm: float) -> float:
    """Normalised wind → knots.  wind_norm = (wnd - 40) / 25"""
    return wind_norm * 25.0 + 40.0


def haversine_km(p1_deg: np.ndarray, p2_deg: np.ndarray) -> np.ndarray:
    """[..., 2] degrees (lon, lat) → distance km."""
    lat1 = np.deg2rad(p1_deg[..., 1])
    lat2 = np.deg2rad(p2_deg[..., 1])
    dlat = np.deg2rad(p2_deg[..., 1] - p1_deg[..., 1])
    dlon = np.deg2rad(p2_deg[..., 0] - p1_deg[..., 0])
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2.0 * 6371.0 * np.arcsin(np.clip(np.sqrt(a), 0, 1))


def detect_pred_len(ckpt_path: str) -> int:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck.get("model_state", ck))
    for key in ["net.pos_enc", "denoiser.pos_enc", "pos_enc"]:
        if key in sd:
            return sd[key].shape[1]
    for k, v in sd.items():
        if "pos_enc" in k and hasattr(v, "dim") and v.dim() == 3:
            return v.shape[1]
    return 12


# ── Map setup ──────────────────────────────────────────────────────────────────

def make_map_ax(fig, subplot_spec, lon_range, lat_range):
    """Create a Cartopy GeoAxes or fallback plain Axes."""
    if HAS_CARTOPY:
        ax = fig.add_subplot(
            subplot_spec,
            projection=ccrs.PlateCarree(central_longitude=0)
        )
        ax.set_extent([lon_range[0], lon_range[1],
                       lat_range[0], lat_range[1]],
                      crs=ccrs.PlateCarree())
        # Ocean & land
        ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                       facecolor=STYLE["ocean_color"], zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"),
                       facecolor=STYLE["land_color"], zorder=1, alpha=0.9)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                       edgecolor="#8ABCD1", linewidth=0.7, zorder=2)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                       edgecolor=STYLE["border_color"],
                       linewidth=0.4, linestyle=":", zorder=2)
        # Grid
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True,
            linewidth=0.5, color=STYLE["grid_color"],
            alpha=STYLE["grid_alpha"], linestyle="--"
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
        for lon in np.arange(np.ceil(lon_range[0]/5)*5, lon_range[1], 5):
            ax.axvline(lon, color="white", alpha=STYLE["grid_alpha"], lw=0.5)
        for lat in np.arange(np.ceil(lat_range[0]/5)*5, lat_range[1], 5):
            ax.axhline(lat, color="white", alpha=STYLE["grid_alpha"], lw=0.5)
        ax.set_xlabel("Longitude (°E)", color="white", fontsize=8)
        ax.set_ylabel("Latitude (°N)", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
    return ax


def _plot_on_ax(ax, lon_range, lat_range,
                obs_deg, gt_deg, pred_deg, pred_Me_deg,
                all_trajs_deg=None,
                errors_km=None,
                title="", dt_str=""):
    """
    Draw one complete forecast panel.

    Parameters
    ----------
    ax           : Axes (Cartopy or plain)
    obs_deg      : [T_obs, 2] degrees (lon, lat)
    gt_deg       : [T_pred, 2]
    pred_deg     : [T_pred, 2]  ensemble mean
    pred_Me_deg  : [T_pred, 2] (pres_norm, wind_norm) — for wind display
    all_trajs_deg: [S, T_pred, 2] optional ensemble members
    errors_km    : [T_pred] error per step
    """
    transform = ccrs.PlateCarree() if HAS_CARTOPY else ax.transData

    def _plot(x, y, **kw):
        if HAS_CARTOPY:
            ax.plot(x, y, transform=transform, **kw)
        else:
            ax.plot(x, y, **kw)

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

    outline = [pe.withStroke(linewidth=2.5, foreground="black")]
    cur_lon, cur_lat = obs_deg[-1, 0], obs_deg[-1, 1]

    # ── Ensemble spread cone ──────────────────────────────────────────────
    if all_trajs_deg is not None and len(all_trajs_deg) > 1:
        S = all_trajs_deg.shape[0]
        for s in range(S):
            xt = np.concatenate([[cur_lon], all_trajs_deg[s, :, 0]])
            yt = np.concatenate([[cur_lat], all_trajs_deg[s, :, 1]])
            _plot(xt, yt,
                  color=STYLE["ens_color"], linewidth=0.4,
                  alpha=STYLE["ens_alpha"], zorder=3)

    # ── Observed track ────────────────────────────────────────────────────
    _plot(obs_deg[:, 0], obs_deg[:, 1],
          "o-", color=STYLE["obs_color"],
          linewidth=STYLE["lw_thin"], markersize=5,
          markeredgecolor="white", markeredgewidth=0.8,
          zorder=7, path_effects=outline, label="Observed")

    # ── Ground truth ──────────────────────────────────────────────────────
    gt_full_lon = np.concatenate([[cur_lon], gt_deg[:, 0]])
    gt_full_lat = np.concatenate([[cur_lat], gt_deg[:, 1]])
    _plot(gt_full_lon, gt_full_lat,
          "o-", color=STYLE["gt_color"],
          linewidth=STYLE["lw_main"], markersize=STYLE["marker_size"],
          markeredgecolor="white", markeredgewidth=1.2,
          zorder=8, path_effects=outline, label="Ground truth")

    # ── Predicted track ───────────────────────────────────────────────────
    pred_full_lon = np.concatenate([[cur_lon], pred_deg[:, 0]])
    pred_full_lat = np.concatenate([[cur_lat], pred_deg[:, 1]])
    _plot(pred_full_lon, pred_full_lat,
          "o-", color=STYLE["pred_color"],
          linewidth=STYLE["lw_main"], markersize=STYLE["marker_size"],
          markeredgecolor="#003300", markeredgewidth=1.0,
          zorder=9, path_effects=outline, label="FM+PINN (mean)")

    # ── Wind intensity markers on predicted track ─────────────────────────
    if pred_Me_deg is not None:
        for i in range(len(pred_deg)):
            wnd_kt = denorm_wind(float(pred_Me_deg[i, 1]))
            _, wcolor = wind_intensity(wnd_kt)
            _scatter([pred_deg[i, 0]], [pred_deg[i, 1]],
                     s=60, color=wcolor,
                     edgecolors="white", linewidths=0.6,
                     zorder=11, marker="o")

    # ── Error connectors + distance labels ────────────────────────────────
    if errors_km is not None:
        step_show = {3: "24h", 7: "48h", 11: "72h"}
        for si, lbl in step_show.items():
            if si < len(gt_deg) and si < len(pred_deg):
                gx, gy = gt_deg[si, 0],   gt_deg[si, 1]
                px, py = pred_deg[si, 0], pred_deg[si, 1]
                if HAS_CARTOPY:
                    ax.plot([gx, px], [gy, py],
                            "--", color=STYLE["error_color"],
                            linewidth=1.2, alpha=0.7,
                            transform=transform, zorder=7)
                else:
                    ax.plot([gx, px], [gy, py],
                            "--", color=STYLE["error_color"],
                            linewidth=1.2, alpha=0.7, zorder=7)
                mx, my = (gx + px) / 2, (gy + py) / 2
                _text(mx, my,
                      f" {lbl}\n{errors_km[si]:.0f}km",
                      fontsize=7, color=STYLE["error_color"],
                      ha="center", va="bottom", zorder=14,
                      path_effects=outline)

    # ── Lead-time labels on predicted track ───────────────────────────────
    for i, (lx, ly) in enumerate(zip(pred_full_lon, pred_full_lat)):
        h = i * 6
        if i == 0:
            lbl = "NOW"
        elif h % 24 == 0:
            lbl = f"+{h}h"
        else:
            continue
        _text(lx, ly + 0.4, lbl,
              fontsize=7.5, color="#AAFFAA",
              ha="center", fontweight="bold",
              path_effects=[pe.withStroke(linewidth=2, foreground="black")],
              zorder=15)

    # ── Current position star ─────────────────────────────────────────────
    _scatter([cur_lon], [cur_lat],
             s=350, marker="*",
             color="#FFD700", edgecolors="#FF4400",
             linewidths=2, zorder=20, label="Now")

    # ── Error summary box ─────────────────────────────────────────────────
    if errors_km is not None:
        n = len(errors_km)
        lines = [f"Mean: {errors_km.mean():.0f} km"]
        for si, lh in [(3, 24), (7, 48), (11, 72)]:
            if si < n:
                lines.append(f" {lh}h: {errors_km[si]:.0f} km")
        ax.text(0.02, 0.03, "\n".join(lines),
                transform=ax.transAxes,
                fontsize=8, va="bottom",
                color="#88FF88", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4",
                          fc=STYLE["bg_color"], alpha=0.85,
                          ec="white", lw=0.8),
                zorder=16)

    # ── Wind legend ───────────────────────────────────────────────────────
    legend_items = []
    for lo, hi, name, color in INTENSITY:
        legend_items.append(
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=color, markersize=7,
                   markeredgecolor="white", markeredgewidth=0.5,
                   label=f"{name} ({lo}–{hi}kt)")
        )
    ax.legend(
        handles=legend_items,
        loc="upper right", fontsize=6.5,
        facecolor="#111111", edgecolor="#00FFFF",
        labelcolor="white", title="Wind (kt)",
        title_fontsize=7,
        ncol=2,
    )

    # ── Observed/GT/Pred legend ───────────────────────────────────────────
    track_handles = [
        Line2D([0], [0], color=STYLE["obs_color"],  lw=2, label="Observed"),
        Line2D([0], [0], color=STYLE["gt_color"],   lw=2, label="Ground truth"),
        Line2D([0], [0], color=STYLE["pred_color"], lw=2, label="FM+PINN"),
    ]
    if all_trajs_deg is not None:
        track_handles.append(
            Line2D([0], [0], color=STYLE["ens_color"],
                   lw=1, alpha=0.5, label="Ensemble members"))
    ax.legend(
        handles=track_handles,
        loc="lower right", fontsize=7.5,
        facecolor="#111111", edgecolor="#00CFFF",
        labelcolor="white",
    )

    # ── Title ─────────────────────────────────────────────────────────────
    ax.set_title(
        f"{title}\n{dt_str}",
        color="white", fontsize=10,
        fontweight="bold", pad=STYLE["title_pad"],
        bbox=dict(fc=STYLE["bg_color"], alpha=0.88,
                  ec="#00FFFF", lw=1.5),
    )
    ax.set_facecolor(STYLE["bg_color"])
    for spine in ax.spines.values():
        spine.set_edgecolor("#334466")


# ── Error curve panel ─────────────────────────────────────────────────────────

def plot_error_curve(ax, errors_km, cliper_err_km, t_name):
    lead_h = np.arange(1, len(errors_km) + 1) * 6
    ax.set_facecolor(STYLE["bg_color"])

    ax.plot(lead_h, errors_km, "o-",
            color=STYLE["pred_color"], lw=2.5, ms=5,
            label="FM+PINN", zorder=5)
    ax.fill_between(lead_h, 0, errors_km,
                    alpha=0.15, color=STYLE["pred_color"])

    if cliper_err_km is not None:
        ax.plot(lead_h, cliper_err_km[:len(lead_h)], "s--",
                color="#FF6666", lw=2, ms=4, label="CLIPER", zorder=4)

    for yl in [100, 200, 300, 400]:
        ax.axhline(yl, color="white", alpha=0.08, lw=0.7)
    for xm in [24, 48, 72]:
        ax.axvline(xm, color=STYLE["error_color"],
                   alpha=0.2, lw=0.7, ls=":")

    ax.set_xlabel("Lead time (h)", color="white", fontsize=8)
    ax.set_ylabel("Track error (km)", color="white", fontsize=8)
    ax.set_title(f"Error — {t_name}", color="white",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7.5, facecolor="#111111",
              edgecolor="#00FFFF", labelcolor="white")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")


# ── Run inference for one sequence ────────────────────────────────────────────

def run_inference(model, target, device, ode_steps, num_ensemble):
    batch = move_batch(seq_collate([target]), device)
    with torch.no_grad():
        pred_mean, pred_Me, all_trajs = model.sample(
            batch,
            num_ensemble=num_ensemble,
            ddim_steps=ode_steps,
        )
    # Shape: pred_mean [T_pred, B, 2], all_trajs [S, T_pred, B, 2]
    obs_n    = batch[0][:, 0, :].cpu().numpy()    # [T_obs, 2]  normalised
    gt_n     = batch[1][:, 0, :].cpu().numpy()    # [T_pred, 2]
    pred_n   = pred_mean[:, 0, :].cpu().numpy()   # [T_pred, 2]
    pred_Me_n= pred_Me[:, 0, :].cpu().numpy()     # [T_pred, 2] (pres, wnd)
    ens_n    = all_trajs[:, :, 0, :].cpu().numpy()# [S, T_pred, 2]

    obs_deg   = to_deg(denorm_traj(obs_n))
    gt_deg    = to_deg(denorm_traj(gt_n))
    pred_deg  = to_deg(denorm_traj(pred_n))
    ens_deg   = to_deg(denorm_traj(ens_n))

    errors_km = haversine_km(pred_deg, gt_deg)

    # CLIPER: extrapolate last obs velocity
    obs_r  = denorm_traj(obs_n)
    gt_r   = denorm_traj(gt_n)
    v_c = obs_r[-1] - obs_r[-2] if len(obs_r) >= 2 else np.zeros(2)
    cliper_preds = np.array([obs_r[-1] + (k+1)*v_c
                              for k in range(len(gt_r))])
    cliper_err = haversine_km(to_deg(cliper_preds), to_deg(gt_r))

    return (obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg,
            errors_km, cliper_err)


# ── Single typhoon visualisation ───────────────────────────────────────────────

def visualize_forecast(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*65}")
    print(f"  TC-FM v10 Forecast  |  {args.tc_name}  @  {args.tc_date}")
    print(f"{'='*65}\n")

    detected = detect_pred_len(args.model_path)
    if args.pred_len != detected:
        print(f"  pred_len: {args.pred_len} → {detected} (from checkpoint)")
        args.pred_len = detected

    model = TCFlowMatching(
        pred_len=args.pred_len, obs_len=args.obs_len).to(device)
    ck  = torch.load(args.model_path, map_location=device,
                     weights_only=False)
    sd  = ck.get("model_state_dict", ck.get("model_state", ck))
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
        print("  Available (first 15):")
        for i in range(min(15, len(dset))):
            info = dset[i][-1]
            print(f"    [{i}]  {info['old'][1]}  @  {info['tydate'][args.obs_len]}")
        return

    (obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg,
     errors_km, cliper_err) = run_inference(
         model, target, device, args.ode_steps, args.num_ensemble)

    print("  Track errors (km):")
    for i, e in enumerate(errors_km):
        mark = "  ◀" if (i+1) in [4, 8, 12] else ""
        print(f"    +{(i+1)*6:3d}h : {e:6.1f} km{mark}")
    print(f"    Mean  : {errors_km.mean():.1f} km\n")

    all_deg   = np.vstack([obs_deg, gt_deg, pred_deg])
    margin    = 3.5
    lon_range = (all_deg[:, 0].min() - margin, all_deg[:, 0].max() + margin)
    lat_range = (all_deg[:, 1].min() - margin, all_deg[:, 1].max() + margin)

    # ── Figure layout: map (left 2/3) + error curve (right 1/3) ──────────
    fig = plt.figure(figsize=(18, 10), facecolor=STYLE["bg_color"])
    gs  = fig.add_gridspec(1, 3, wspace=0.08)

    ax_map = make_map_ax(fig, gs[0, :2], lon_range, lat_range)
    ax_err = fig.add_subplot(gs[0, 2])

    dt_str = datetime.strptime(t_date, "%Y%m%d%H").strftime(
        "%d %b %Y  %H:%M UTC")
    fh = args.pred_len * 6

    _plot_on_ax(
        ax_map, lon_range, lat_range,
        obs_deg, gt_deg, pred_deg, pred_Me_n,
        all_trajs_deg=ens_deg if args.num_ensemble > 1 else None,
        errors_km=errors_km,
        title=f"🌀 {t_name}  —  {fh}h FC  |  FM+PINN v10",
        dt_str=dt_str,
    )
    plot_error_curve(ax_err, errors_km, cliper_err, t_name)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir,
                       f"forecast_{fh}h_{t_name}_{t_date}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight",
                facecolor=STYLE["bg_color"])
    plt.close()
    print(f"  Saved → {out}\n")


# ── Case-study grid ────────────────────────────────────────────────────────────

def visualize_case_study(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detected    = detect_pred_len(args.model_path)
    args.pred_len = detected

    model = TCFlowMatching(
        pred_len=args.pred_len, obs_len=args.obs_len).to(device)
    ck = torch.load(args.model_path, map_location=device,
                    weights_only=False)
    sd = ck.get("model_state_dict", ck.get("model_state", ck))
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

    n_cases = len(cases)
    fig = plt.figure(
        figsize=(22, 7 * n_cases),
        facecolor=STYLE["bg_color"]
    )
    gs = fig.add_gridspec(n_cases, 3, wspace=0.08, hspace=0.25)

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
            print(f"  ⚠  {t_name} @ {t_date} — not found, skipping")
            for c in range(3):
                ax = fig.add_subplot(gs[row, c])
                ax.set_facecolor(STYLE["bg_color"])
                ax.text(0.5, 0.5, f"NOT FOUND\n{t_name} @ {t_date}",
                        ha="center", va="center", color="red",
                        transform=ax.transAxes)
            continue

        (obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg,
         errors_km, cliper_err) = run_inference(
             model, target, device, args.ode_steps, args.num_ensemble)

        all_deg   = np.vstack([obs_deg, gt_deg, pred_deg])
        margin    = 3.5
        lon_range = (all_deg[:, 0].min() - margin,
                     all_deg[:, 0].max() + margin)
        lat_range = (all_deg[:, 1].min() - margin,
                     all_deg[:, 1].max() + margin)

        dt_str = datetime.strptime(t_date, "%Y%m%d%H").strftime(
            "%d %b %Y %H:%M UTC")

        ax_map = make_map_ax(fig, gs[row, :2], lon_range, lat_range)
        ax_err = fig.add_subplot(gs[row, 2])

        _plot_on_ax(
            ax_map, lon_range, lat_range,
            obs_deg, gt_deg, pred_deg, pred_Me_n,
            all_trajs_deg=ens_deg if args.num_ensemble > 1 else None,
            errors_km=errors_km,
            title=f"[{label}]  {t_name}",
            dt_str=dt_str,
        )
        plot_error_curve(ax_err, errors_km, cliper_err, t_name)

        print(f"  [{label}] ADE={errors_km.mean():.1f} km  "
              f"72h={errors_km[11] if len(errors_km)>11 else float('nan'):.1f} km")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "case_study_grid_v10.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg_color"])
    plt.close()
    print(f"\n  Case study saved → {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="TC-FlowMatching v10 Forecast Visualisation")
    p.add_argument("--model_path",      required=True)
    p.add_argument("--TC_data_path",    required=True)
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
    # Model / data
    p.add_argument("--test_year",       type=int,   default=2019)
    p.add_argument("--obs_len",         type=int,   default=8)
    p.add_argument("--pred_len",        type=int,   default=12)
    p.add_argument("--ode_steps",       type=int,   default=10)
    p.add_argument("--num_ensemble",    type=int,   default=50)
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
"""
visual_evaluate_model.py
==========================
Visualize TC-FlowMatching (FM) and baseline (ST-Trans/LSTM/GRU/RNN)
predictions for a specific storm/timestamp, with a clean paper-style
map background (Cartopy NaturalEarthFeature, land/ocean/coastline) and
an ensemble spread cone.

Modes:
  --mode single       One storm, ONE model (default FM), full ensemble
                       spread cone + error summary box. Matches the
                       "RITA — 72h FC | FM" reference layout: one map,
                       Wind(kt) legend, Mean ADE / spread text box.
  --mode multi_model  Same storm, MULTIPLE architectures overlaid
                       (FM vs ST-Trans/LSTM/GRU/RNN), one line each.
  --mode multi_seed   Same storm, SAME architecture, MULTIPLE seeds
                       overlaid — for checking seed-to-seed stability.
  --mode case_study   Grid of several storms at once (2-3 rows), each
                       row = 1 map + 1 spread-vs-time panel.

Rebuilt from scratch (see chat history) after the version running on
Kaggle was found to be an older snapshot using a lower-resolution map
background. This version consolidates every fix discussed across prior
sessions: full-resolution Cartopy background (10m NaturalEarthFeature,
falls back to 50m then flat colors if offline), single-map layout for
--mode single (inset zoom panel removed per explicit request), real
6-hourly lead-time labels, Wind(kt) intensity legend, and a probability
cone drawn from the actual ensemble spread (not a synthetic circle).
"""
from __future__ import annotations

import os
import re
import sys
import random
import argparse
from datetime import datetime
from typing import Optional, Dict, List, Tuple

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
    print("  Warning: cartopy not found — using plain axes (no coastlines).")

from Model.flow_matching_model import TCFlowMatching
from Model.paper_baseline_model import PaperBaseline
from Model.st_trans_model import STTrans
from Model.data.loader import data_loader
from Model.data.trajectoriesWithMe_unet_training import seq_collate


# ─────────────────────────────────────────────────────────────────────────────
#  Styling — paper/white background, matches the reference image
# ─────────────────────────────────────────────────────────────────────────────
STYLE = dict(
    obs_color     = "#000000",   # black — observed history track
    gt_color      = "#1F5FBF",   # blue — Actual Track
    pred_color    = "#D62728",   # red  — Predicted
    ens_color     = "#D62728",
    ens_alpha     = 0.05,
    marker_size   = 6,
    lw_main       = 2.0,
    lw_thin       = 1.3,
    bg_color      = "#FFFFFF",
    land_color    = "#FFFFFF",   # flat-color fallback only (Cartopy offline)
    ocean_color   = "#EAF3FB",   # flat-color fallback only
    border_color  = "#BBBBBB",
    grid_color    = "#CCCCCC",
    grid_alpha    = 0.5,
    error_color   = "#B8860B",   # dark goldenrod — readable on white bg
    title_pad     = 14,
    cone_50_fill  = "#D62728",
    cone_90_fill  = "#1F77B4",
    cone_50_alpha = 0.18,
    cone_90_alpha = 0.10,
    cone_edge_lw  = 1.2,
    text_color    = "#000000",
    panel_edge    = "#888888",
)

MODEL_COLORS = {
    "FM":       "#D62728",
    "ST-Trans": "#FF7F0E",
    "LSTM":     "#2CA02C",
    "GRU":      "#9467BD",
    "RNN":      "#8C564B",
}

SEED_COLORS = {
    "0": "#D62728", "1": "#1F77B4", "2": "#2CA02C",
    "3": "#9467BD", "4": "#FF7F0E", "5": "#8C564B",
}
_SEED_COLOR_FALLBACK = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD",
                        "#FF7F0E", "#8C564B", "#17BECF", "#BCBD22"]


def _seed_color(seed_label: str, idx: int) -> str:
    return SEED_COLORS.get(str(seed_label),
                           _SEED_COLOR_FALLBACK[idx % len(_SEED_COLOR_FALLBACK)])


_CHI2_50 = chi2.ppf(0.50, df=2)
_CHI2_90 = chi2.ppf(0.90, df=2)

# Wind intensity bins (kt) — Saffir-Simpson-ish, matches the reference
# image's "Wind (kt)" legend exactly (TD/TS/TY/Sev.TY/Vis.TY/Super TY).
INTENSITY = [
    (0,   34,  "TD",       "#6699CC"),
    (34,  48,  "TS",       "#33AA33"),
    (48,  64,  "TY",       "#CCAA00"),
    (64,  84,  "Sev.TY",   "#FF8C00"),
    (84,  115, "Vis.TY",   "#E03C00"),
    (115, 999, "Super TY", "#B000B0"),
]


def wind_intensity(wind_kt: float) -> Tuple[str, str]:
    for lo, hi, name, color in INTENSITY:
        if lo <= wind_kt < hi:
            return name, color
    return "Super TY", "#FF00FF"


# ─────────────────────────────────────────────────────────────────────────────
#  Generic helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
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


def denorm_traj(n: np.ndarray) -> np.ndarray:
    """Inverse-normalize a trajectory array (any shape ending in 2:
    [..., 0]=lon-code, [..., 1]=lat-code) back to the model's internal
    degree-like units (still needs to_deg() afterward for true degrees).
    """
    r = np.zeros_like(n)
    r[..., 0] = n[..., 0] * 50.0 + 1800.0
    r[..., 1] = n[..., 1] * 50.0
    return r


def to_deg(pts_01: np.ndarray) -> np.ndarray:
    return pts_01 / 10.0


def denorm_wind(wind_norm: float) -> float:
    return wind_norm * 25.0 + 40.0


def haversine_km(p1_deg: np.ndarray, p2_deg: np.ndarray) -> np.ndarray:
    """Haversine distance (km) between two arrays of (lon, lat) points.
    Both arrays must have identical shape (..., 2)."""
    p1_deg = np.asarray(p1_deg)
    p2_deg = np.asarray(p2_deg)
    if p1_deg.shape != p2_deg.shape:
        raise ValueError(
            f"haversine_km: shape mismatch {p1_deg.shape} vs {p2_deg.shape}")
    lat1 = np.deg2rad(p1_deg[..., 1])
    lat2 = np.deg2rad(p2_deg[..., 1])
    dlat = np.deg2rad(p2_deg[..., 1] - p1_deg[..., 1])
    dlon = np.deg2rad(p2_deg[..., 0] - p1_deg[..., 0])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2.0 * 6371.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def detect_pred_len(ckpt_path: str) -> int:
    """Infer pred_len from a checkpoint — prefer model_cfg (authoritative,
    matches exact train-time args), fallback to a position-embedding
    tensor's shape, final fallback to 12 (the project default)."""
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return 12
    if isinstance(ck, dict):
        cfg = ck.get("model_cfg")
        if isinstance(cfg, dict) and "pred_len" in cfg:
            return int(cfg["pred_len"])
        sd = ck.get("model", ck.get("model_state_dict", ck.get("model_state", ck)))
    else:
        sd = ck
    if isinstance(sd, dict):
        for key in ["net.pos_enc", "denoiser.pos_enc", "pos_enc"]:
            if key in sd and hasattr(sd[key], "shape"):
                return int(sd[key].shape[1])
        for k, v in sd.items():
            if "pos_enc" in k and hasattr(v, "dim") and v.dim() == 3:
                return int(v.shape[1])
    return 12


# ─────────────────────────────────────────────────────────────────────────────
#  Date snapping & sample search
# ─────────────────────────────────────────────────────────────────────────────

def snap_to_6h(date_str: str) -> str:
    """Floor YYYYMMDDHH to the nearest earlier multiple of 6h."""
    s = str(date_str).strip()[:10]
    dt = datetime.strptime(s, "%Y%m%d%H")
    dt = dt.replace(hour=(dt.hour // 6) * 6, minute=0, second=0)
    return dt.strftime("%Y%m%d%H")


def resolve_date(raw_date: str) -> Tuple[str, bool]:
    original = str(raw_date).strip()[:10]
    snapped = snap_to_6h(original)
    was_snapped = snapped != original
    if was_snapped:
        print(f"  [SNAP] {original} -> {snapped} (rounded to nearest earlier 6h)")
    return snapped, was_snapped


def find_target(dset, t_name: str, t_date: str, obs_len: int):
    """
    Search dataset for a sample matching (storm name, date) at the
    requested obs_len boundary. Searches the FULL tydate array (not
    just index obs_len), preferring an exact match at tydate[obs_len]
    but accepting any idx >= obs_len (need at least obs_len prior
    observed steps).

    Returns (item, matched_obs_len, actual_date) or (None, None, None).
    """
    best_item, best_obs_len, best_date = None, None, None
    best_priority = 10 ** 9

    for i in range(len(dset)):
        item = dset[i]
        info = item[-1]
        name = str(info["old"][1]).strip().upper()
        if t_name not in name:
            continue
        tydates = info.get("tydate")
        if tydates is None:
            continue
        for idx, td in enumerate(tydates):
            if str(td).strip() != t_date:
                continue
            if idx < obs_len:
                continue
            priority = 0 if idx == obs_len else (idx - obs_len + 1)
            if priority < best_priority:
                best_item, best_obs_len, best_date = item, idx, str(td).strip()
                best_priority = priority

    return best_item, best_obs_len, best_date


def list_available(dset, t_name: str, obs_len: int, limit: int = 30):
    """Print available (storm, date) pairs for t_name, for when
    find_target() fails and the user needs to pick a valid date."""
    seen = []
    for i in range(len(dset)):
        item = dset[i]
        info = item[-1]
        name = str(info["old"][1]).strip().upper()
        if t_name not in name:
            continue
        tydates = info.get("tydate")
        if tydates is None or obs_len >= len(tydates):
            continue
        seen.append(str(tydates[obs_len]).strip())
    seen = sorted(set(seen))
    if not seen:
        print(f"  (no sequences found for '{t_name}' with obs_len={obs_len})")
        return
    print(f"  Available timestamps for '{t_name}' (showing up to {limit}):")
    for d in seen[:limit]:
        print(f"    {d}")
    if len(seen) > limit:
        print(f"    ... and {len(seen) - limit} more")


# ─────────────────────────────────────────────────────────────────────────────
#  Map setup — Cartopy background, full 10m-resolution NaturalEarthFeature
# ─────────────────────────────────────────────────────────────────────────────

def make_map_ax(fig, subplot_spec, lon_range, lat_range, use_satellite_bg=True):
    """
    Creates a map axis with land/ocean/coastline drawn via Cartopy's
    NaturalEarthFeature at "10m" resolution (falls back to "50m", then
    to flat STYLE colors, if offline/uncached) — this is what produces
    the clean, readable coastline in the reference image, as opposed to
    a blurry stock_img() raster or a flat single-color background.

    use_satellite_bg=True (default) uses the detailed land/ocean fill;
    False skips straight to flat colors (kept for flexibility, not used
    by default anywhere in this file).
    """
    if HAS_CARTOPY:
        ax = fig.add_subplot(
            subplot_spec, projection=ccrs.PlateCarree(central_longitude=0))
        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                       crs=ccrs.PlateCarree())
        drew_bg = False
        if use_satellite_bg:
            for scale in ("10m", "50m"):
                try:
                    ax.add_feature(cfeature.NaturalEarthFeature(
                        "physical", "land", scale,
                        facecolor="#E8E4D8", edgecolor="none"), zorder=1)
                    ax.add_feature(cfeature.NaturalEarthFeature(
                        "physical", "ocean", scale,
                        facecolor="#C8DCF0", edgecolor="none"), zorder=0)
                    drew_bg = True
                    break
                except Exception as e:
                    print(f"  ⚠ Cartopy NaturalEarthFeature '{scale}' land/ocean "
                          f"failed to load ({type(e).__name__}: {e}) — "
                          f"likely no internet access to download/cache Natural "
                          f"Earth assets on this machine. Trying next scale...")
                    continue
        if not drew_bg:
            print("  ⚠ Falling back to flat ocean/land colors (no coastline "
                  "detail) — Natural Earth land/ocean fill could not be "
                  "loaded at any resolution.")
            ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                           facecolor=STYLE["ocean_color"], zorder=0)
            ax.add_feature(cfeature.LAND.with_scale("50m"),
                           facecolor=STYLE["land_color"], zorder=1, alpha=0.9)
        try:
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                           edgecolor="#4D4D4D", linewidth=0.8, zorder=2)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                           edgecolor=STYLE["border_color"],
                           linewidth=0.4, linestyle=":", zorder=2)
        except Exception as e:
            print(f"  ⚠ Cartopy COASTLINE/BORDERS failed to load "
                  f"({type(e).__name__}: {e}) — map will have NO coastline "
                  f"outline. This is almost always a network/cache issue: "
                  f"cartopy needs internet on its FIRST run on this machine "
                  f"to download Natural Earth shapefiles (~a few MB), then "
                  f"caches them locally for offline use afterward. If "
                  f"running on Kaggle, make sure 'Internet' is turned ON in "
                  f"the notebook's session settings (right sidebar), or "
                  f"pre-download the cartopy data cache into the working "
                  f"directory before running this script.")
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color=STYLE["grid_color"],
                          alpha=STYLE["grid_alpha"], linestyle="--")
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = dict(color=STYLE["text_color"], fontsize=7)
        gl.ylabel_style = dict(color=STYLE["text_color"], fontsize=7)
    else:
        ax = fig.add_subplot(subplot_spec)
        ax.set_facecolor(STYLE["bg_color"])
        ax.set_xlim(*lon_range)
        ax.set_ylim(*lat_range)
        for lon in np.arange(np.ceil(lon_range[0] / 5) * 5, lon_range[1], 5):
            ax.axvline(lon, color=STYLE["grid_color"], alpha=STYLE["grid_alpha"], lw=0.5)
        for lat in np.arange(np.ceil(lat_range[0] / 5) * 5, lat_range[1], 5):
            ax.axhline(lat, color=STYLE["grid_color"], alpha=STYLE["grid_alpha"], lw=0.5)
        ax.set_xlabel("Longitude (°E)", color=STYLE["text_color"], fontsize=8)
        ax.set_ylabel("Latitude (°N)",  color=STYLE["text_color"], fontsize=8)
        ax.tick_params(colors=STYLE["text_color"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["panel_edge"])
    return ax


# ─────────────────────────────────────────────────────────────────────────────
#  Probability cone — drawn from the ACTUAL ensemble spread (PCA ellipses
#  at each lead time, chained together), not a synthetic fixed-width band.
# ─────────────────────────────────────────────────────────────────────────────

def _ellipse_points(center: np.ndarray, cov: np.ndarray, chi2_val: float,
                     n_pts: int = 24) -> np.ndarray:
    """
    Returns n_pts points on the boundary of the confidence ellipse
    {x : (x-center)^T cov^-1 (x-center) = chi2_val} — standard
    elliptical confidence-region construction from a 2x2 covariance
    matrix (eigendecomposition gives ellipse axis lengths/orientation).
    Falls back to a small circle if cov is near-singular (e.g. K<3
    ensemble members collapsed to nearly one point) to avoid NaN/inf.
    """
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-8, None)
        axis_lengths = np.sqrt(eigvals * chi2_val)
    except np.linalg.LinAlgError:
        axis_lengths = np.array([1e-3, 1e-3])
        eigvecs = np.eye(2)
    theta = np.linspace(0, 2 * np.pi, n_pts)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # [n_pts, 2]
    ellipse = circle * axis_lengths  # scale by axis lengths
    ellipse = ellipse @ eigvecs.T    # rotate into eigenvector frame
    return ellipse + center


def draw_smooth_cone(ax, all_trajs_deg: np.ndarray, cur_pos: np.ndarray,
                      transform=None):
    """
    Draws the 50%/90% probability cone as a series of chained elliptical
    confidence regions, one per lead time, filled between consecutive
    time steps — this produces the widening, roughly triangular
    "cone of uncertainty" shape seen in the reference image, grounded in
    the ACTUAL ensemble covariance at each step (not a fixed synthetic
    width) so a genuinely tight ensemble (e.g. spread~4km, N=1 before
    the multi-step ODE fix) shows a correspondingly thin cone, and a
    wide one (spread~50km+, N=10) shows a visibly wider cone.

    all_trajs_deg: [K, T, 2] array (K ensemble members, T lead times).
    cur_pos: [2] the "NOW" anchor point (last observed position) — the
    cone starts collapsed at this point (zero width at t=0) since all
    ensemble members share the same observed history.
    """
    K, T, _ = all_trajs_deg.shape
    if K < 3:
        return  # not enough members for a meaningful covariance estimate

    def _fill(ax, poly_lo, poly_hi, **kw):
        # Build a closed ring: outer boundary forward, inner boundary
        # backward, so fill() traces a proper annulus/band between the
        # two ellipse chains at consecutive time steps.
        ring = np.vstack([poly_lo, poly_hi[::-1]])
        if transform is not None:
            ax.fill(ring[:, 0], ring[:, 1], transform=transform, **kw)
        else:
            ax.fill(ring[:, 0], ring[:, 1], **kw)

    prev_ellipse_50 = np.tile(cur_pos, (24, 1))
    prev_ellipse_90 = np.tile(cur_pos, (24, 1))

    for t in range(T):
        members_t = all_trajs_deg[:, t, :]           # [K, 2]
        center_t  = members_t.mean(axis=0)
        cov_t     = np.cov(members_t.T)               # [2, 2]
        if cov_t.shape != (2, 2) or not np.all(np.isfinite(cov_t)):
            continue

        ell_50 = _ellipse_points(center_t, cov_t, _CHI2_50)
        ell_90 = _ellipse_points(center_t, cov_t, _CHI2_90)

        # Fill the band between the previous and current 90% ellipse
        # (outer, drawn first so 50% band layers on top), then the 50%
        # band — this chains adjacent time steps into one continuous
        # cone instead of a stack of disjoint ellipses.
        _fill(ax, prev_ellipse_90, ell_90,
              color=STYLE["cone_90_fill"], alpha=STYLE["cone_90_alpha"],
              zorder=3, edgecolor="none")
        _fill(ax, prev_ellipse_50, ell_50,
              color=STYLE["cone_50_fill"], alpha=STYLE["cone_50_alpha"],
              zorder=4, edgecolor="none")

        prev_ellipse_50, prev_ellipse_90 = ell_50, ell_90


def _inset_range(pred_deg: np.ndarray, ens_deg: Optional[np.ndarray]):
    """Tight (lon_range, lat_range) zoomed on the forecast region only —
    kept for potential future use (e.g. a --show_inset flag), not called
    by default anywhere in this rebuild (inset panel removed per the
    explicit request to match the reference image's single-map layout)."""
    pts = pred_deg if ens_deg is None else np.vstack([pred_deg, ens_deg.reshape(-1, 2)])
    lon_span = pts[:, 0].max() - pts[:, 0].min()
    lat_span = pts[:, 1].max() - pts[:, 1].min()
    margin_lon = max(lon_span * 0.30, 0.3)
    margin_lat = max(lat_span * 0.30, 0.3)
    lon_range = (pts[:, 0].min() - margin_lon, pts[:, 0].max() + margin_lon)
    lat_range = (pts[:, 1].min() - margin_lat, pts[:, 1].max() + margin_lat)
    return lon_range, lat_range


# ─────────────────────────────────────────────────────────────────────────────
#  Core track-plotting routine — shared by single/multi_model/case_study
# ─────────────────────────────────────────────────────────────────────────────

def _plot_on_ax(ax, lon_range, lat_range,
                 obs_deg, gt_deg, pred_deg, pred_Me_deg,
                 all_trajs_deg=None, errors_km=None,
                 title="", dt_str="", pred_label="FM (mean)",
                 ref_spread_km=None):
    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    outline = [pe.withStroke(linewidth=2.5, foreground="white")]
    cur_pos = obs_deg[-1]

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

    # 1. Probability cone (from actual ensemble spread)
    if all_trajs_deg is not None and all_trajs_deg.shape[0] >= 3:
        draw_smooth_cone(ax, all_trajs_deg, cur_pos, transform)

    # 2. Observed track (history leading up to "NOW")
    _plot(obs_deg[:, 0], obs_deg[:, 1], fmt="o-",
          color=STYLE["obs_color"], linewidth=STYLE["lw_thin"], markersize=5,
          markeredgecolor="white", markeredgewidth=0.8,
          zorder=7, path_effects=outline)

    # 3. Ground truth (Actual Track)
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
          markeredgecolor="white", markeredgewidth=1.0,
          zorder=9, path_effects=outline)

    # 5. Wind intensity markers along the predicted track
    if pred_Me_deg is not None:
        for i in range(len(pred_deg)):
            wnd_kt = denorm_wind(float(pred_Me_deg[i, 1]))
            _, wcolor = wind_intensity(wnd_kt)
            _scatter([pred_deg[i, 0]], [pred_deg[i, 1]],
                     s=70, color=wcolor,
                     edgecolors="white", linewidths=0.7, zorder=11)

    # 6. Error connectors at 24h/48h/72h (dashed line + label)
    if errors_km is not None:
        for si, lbl in {3: "24h", 7: "48h", 11: "72h"}.items():
            if si < len(gt_deg) and si < len(pred_deg) and si < len(errors_km):
                gx, gy = gt_deg[si, 0], gt_deg[si, 1]
                px, py = pred_deg[si, 0], pred_deg[si, 1]
                if HAS_CARTOPY:
                    ax.plot([gx, px], [gy, py], "--",
                            color=STYLE["error_color"], linewidth=1.2,
                            alpha=0.8, transform=transform, zorder=7)
                else:
                    ax.plot([gx, px], [gy, py], "--",
                            color=STYLE["error_color"], linewidth=1.2,
                            alpha=0.8, zorder=7)
                _text((gx + px) / 2, (gy + py) / 2,
                      f" {lbl}\n{errors_km[si]:.0f}km",
                      fontsize=7, color=STYLE["error_color"],
                      ha="center", va="bottom", zorder=14,
                      path_effects=outline)

    # 7. Lead-time labels every 24h, for both GT and Pred
    for i in range(len(pred_lon)):
        h = i * 6
        if h % 24 == 0:
            lbl = "NOW" if i == 0 else f"+{h}h"
            _text(pred_lon[i], pred_lat[i] + 0.5, lbl,
                  color=STYLE["pred_color"], fontweight="bold", fontsize=7.5,
                  path_effects=outline)
            if i < len(gt_lon):
                _text(gt_lon[i], gt_lat[i] - 0.7, lbl,
                      color=STYLE["gt_color"], fontsize=6, alpha=0.9,
                      path_effects=outline)

    # 8. "NOW" star at the current position
    _scatter([cur_pos[0]], [cur_pos[1]],
             s=350, marker="*", color="#FFD700",
             edgecolors="black", linewidths=1.5, zorder=20)

    # 9. Error/spread summary text box
    if errors_km is not None:
        n = len(errors_km)
        lines = [f"Mean ADE: {errors_km.mean():.0f} km"]
        for si, lh in [(3, 24), (7, 48), (11, 72)]:
            if si < n:
                lines.append(f" {lh}h: {errors_km[si]:.0f} km")

        if all_trajs_deg is not None and all_trajs_deg.shape[0] >= 3:
            lines.append("")
            lines.append("Spread (1σ):")
            for si, lh in [(3, 24), (7, 48), (11, 72)]:
                if si < all_trajs_deg.shape[1]:
                    members_at_t = all_trajs_deg[:, si, :]
                    mean_at_t = members_at_t.mean(axis=0, keepdims=True)
                    d_to_mean = haversine_km(
                        members_at_t, np.repeat(mean_at_t, members_at_t.shape[0], axis=0))
                    this_spread = d_to_mean.std()
                    ref_str = ""
                    if ref_spread_km and lh in ref_spread_km:
                        ref_str = f" (ref: {ref_spread_km[lh]:.0f})"
                    lines.append(f" {lh}h: {this_spread:.0f} km{ref_str}")

        ax.text(0.02, 0.03, "\n".join(lines),
                transform=ax.transAxes, fontsize=8, va="bottom",
                color=STYLE["text_color"], family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          alpha=0.9, ec=STYLE["panel_edge"], lw=0.8),
                zorder=16)

    # 10. Legends
    track_handles = [
        Line2D([0], [0], color=STYLE["obs_color"], lw=2, label="Observed"),
        Line2D([0], [0], color=STYLE["gt_color"], lw=2, label="Ground truth"),
        Line2D([0], [0], color=STYLE["pred_color"], lw=2.5, label=f"Predicted ({pred_label})"),
        mpatches.Patch(facecolor=STYLE["cone_50_fill"], alpha=0.5, label="50% prob. cone"),
        mpatches.Patch(facecolor=STYLE["cone_90_fill"], alpha=0.35, label="90% prob. cone"),
    ]
    ax.legend(handles=track_handles, loc="lower right", fontsize=7.5,
              facecolor="white", edgecolor=STYLE["panel_edge"],
              labelcolor=STYLE["text_color"], framealpha=0.92)

    wind_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=c, markersize=7,
               markeredgecolor="black", markeredgewidth=0.5,
               label=f"{nm} ({lo}\u2013{hi}kt)")
        for lo, hi, nm, c in INTENSITY
    ]
    leg2 = ax.legend(handles=wind_handles, loc="upper right", fontsize=6.5,
                     facecolor="white", edgecolor=STYLE["panel_edge"],
                     labelcolor=STYLE["text_color"], title="Wind (kt)",
                     title_fontsize=7, ncol=2, framealpha=0.92)
    ax.add_artist(leg2)

    ax.set_title(f"{title}\n{dt_str}", color=STYLE["text_color"], fontsize=10,
                 fontweight="bold", pad=STYLE["title_pad"],
                 bbox=dict(fc="white", alpha=0.9, ec=STYLE["panel_edge"], lw=1.2))
    ax.set_facecolor(STYLE["bg_color"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["panel_edge"])


def plot_spread_over_time(ax, ens_deg, errors_km, cliper_err, t_name):
    """
    Right-panel diagnostic: ensemble spread (1sigma, km) vs. forecast
    lead time, overlaid with the actual track error and a CLIPER
    (constant-velocity extrapolation) baseline error — lets a reader see
    at a glance whether spread grows commensurately with actual error
    (well-calibrated) or stays flat while error balloons (underdispersed
    — the "co cụm" issue this whole line of work has been diagnosing).
    """
    T = ens_deg.shape[1]
    lead_h = np.arange(1, T + 1) * 6
    spread_km = np.zeros(T)
    for t in range(T):
        members_t = ens_deg[:, t, :]
        mean_t = members_t.mean(axis=0, keepdims=True)
        d_to_mean = haversine_km(members_t, np.repeat(mean_t, members_t.shape[0], axis=0))
        spread_km[t] = d_to_mean.std()

    spread_color = STYLE["cone_50_fill"]
    ax.plot(lead_h, spread_km, "o-", color=spread_color, linewidth=1.8,
            markersize=4, label="Ensemble spread (1σ)")
    ax.set_ylabel("Spread 1σ (km)", color=spread_color, fontsize=8)
    ax.tick_params(axis="y", colors=spread_color, labelsize=7)

    ax_twin = ax.twinx()
    n = min(len(errors_km), T)
    ax_twin.plot(lead_h[:n], errors_km[:n], "s-", color=STYLE["pred_color"],
                 linewidth=1.8, markersize=4, label="Track error (FM)")
    if cliper_err is not None:
        n2 = min(len(cliper_err), T)
        ax_twin.plot(lead_h[:n2], cliper_err[:n2], "^--", color="#888888",
                     linewidth=1.2, markersize=3.5, alpha=0.8, label="CLIPER baseline")
    ax_twin.set_ylabel("Track error (km)", color=STYLE["pred_color"], fontsize=8)
    ax_twin.tick_params(axis="y", colors=STYLE["pred_color"], labelsize=7)

    ax.set_xlabel("Forecast Lead Time (h)", color=STYLE["text_color"], fontsize=8)
    ax.set_title(f"Spread vs Error — {t_name}", color=STYLE["text_color"],
                 fontsize=9, fontweight="bold")

    lines1, lbs1 = ax.get_legend_handles_labels()
    lines2, lbs2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbs1 + lbs2, fontsize=7.5,
              facecolor="white", edgecolor=STYLE["panel_edge"],
              labelcolor=STYLE["text_color"], loc="upper left", framealpha=0.92)

    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["panel_edge"])
    ax.tick_params(colors=STYLE["text_color"], labelsize=7)
    ax_twin.tick_params(colors=STYLE["text_color"], labelsize=7)


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading — generic across all 5 architectures
# ─────────────────────────────────────────────────────────────────────────────

def load_model_generic(model_path: str, model_type: str, device,
                        obs_len: int = 8, pred_len: int = 12):
    """
    Loads any of FM/ST-Trans/LSTM/GRU/RNN from a checkpoint. Prefers the
    checkpoint's own saved model_cfg (authoritative architecture, from
    train_flowmatching.py/train_st_trans.py/train_paper_baseline.py's
    --seed-era checkpoint format) over CLI-default-matching fallback
    values, same convention as evaluate_multi_model.py's loaders.
    """
    ck = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = ck.get("model_cfg") if isinstance(ck, dict) else None

    if model_type == "fm":
        model = TCFlowMatching(**(model_cfg or dict(pred_len=pred_len, obs_len=obs_len))).to(device)
        state = ck.get("model", ck.get("model_state_dict", ck.get("model_state", ck)))
    elif model_type == "st_trans":
        if model_cfg:
            model = STTrans(**model_cfg).to(device)
        else:
            model = STTrans(obs_len=obs_len, pred_len=pred_len).to(device)
        state = ck.get("model_state", ck.get("model"))
    else:  # lstm / gru / rnn
        if model_cfg:
            model = PaperBaseline(**model_cfg).to(device)
        else:
            model = PaperBaseline(model_type=model_type, obs_len=obs_len,
                                   pred_len=pred_len).to(device)
        state = ck.get("model_state", ck.get("model"))

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ {model_type.upper()} load_state_dict: "
              f"{len(missing)} missing, {len(unexpected)} unexpected keys")
    model.eval()
    return model


def _infer_seed_label(ckpt_path: str, idx: int) -> str:
    """Prefer the checkpoint's own saved 'seed' field; fall back to
    parsing 'seedN' from the path, then to a positional index."""
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "seed" in ck:
            return str(ck["seed"])
    except Exception:
        pass
    m = re.search(r"seed[_-]?(\d+)", ckpt_path, re.IGNORECASE)
    if m:
        return m.group(1)
    return str(idx)


# ─────────────────────────────────────────────────────────────────────────────
#  Inference — single model, single storm/timestamp
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, target, device, ode_steps: int, num_ensemble: int):
    """
    FM-specific inference path — used by --mode single, which prints a
    detailed per-lead-time error breakdown (see visualize_forecast()).
    Returns (obs_deg, gt_deg, pred_deg, pred_Me_deg, ens_deg, errors_km,
    cliper_err).
    """
    batch = move_batch(seq_collate([target]), device)
    with torch.no_grad():
        pred_mean, pred_Me, all_trajs = model.sample(
            batch, num_ensemble=num_ensemble, ddim_steps=ode_steps)

    obs_n     = batch[0][:, 0, :].cpu().numpy()
    gt_n      = batch[1][:, 0, :].cpu().numpy()
    pred_n    = pred_mean[:, 0, :].cpu().numpy()
    pred_Me_n = pred_Me[:, 0, :].cpu().numpy()
    ens_n     = all_trajs[:, :, 0, :].cpu().numpy()   # [K, T, 2]

    obs_deg  = to_deg(denorm_traj(obs_n))
    gt_deg   = to_deg(denorm_traj(gt_n))
    pred_deg = to_deg(denorm_traj(pred_n))
    ens_deg  = to_deg(denorm_traj(ens_n))

    errors_km = haversine_km(pred_deg, gt_deg)

    # CLIPER: constant-velocity extrapolation from the last two observed points
    if len(obs_deg) >= 2:
        v_deg = obs_deg[-1] - obs_deg[-2]
        cliper_preds_deg = np.array(
            [obs_deg[-1] + (k + 1) * v_deg for k in range(len(gt_deg))])
    else:
        cliper_preds_deg = np.tile(obs_deg[-1], (len(gt_deg), 1))
    cliper_err = haversine_km(cliper_preds_deg, gt_deg)

    return obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg, errors_km, cliper_err


def run_inference_generic(model, target, device, model_type: str,
                           ode_steps: int = 10, num_ensemble: int = 1):
    """
    Generalized inference for ANY of the 5 architectures — used by
    --mode multi_model/multi_seed, where several models/seeds are
    compared and a terse (obs_deg, gt_deg, pred_deg, ens_deg, errors_km)
    tuple suffices.

    ddim_steps is only meaningful for FM (the ODE integration step
    count) — NOT passed for the recurrent/ST-Trans baselines, which
    have no such parameter and would error if it were forwarded.
    num_ensemble likewise only applies to FM; baselines are
    deterministic (num_ensemble forced to 1 regardless of what's passed).
    """
    batch = move_batch(seq_collate([target]), device)
    with torch.no_grad():
        if model_type == "fm":
            pred_mean, _, all_trajs = model.sample(
                batch, num_ensemble=num_ensemble, ddim_steps=ode_steps)
        else:
            pred_mean, _, all_trajs = model.sample(batch, num_ensemble=1)

    obs_n  = batch[0][:, 0, :].cpu().numpy()
    gt_n   = batch[1][:, 0, :].cpu().numpy()
    pred_n = pred_mean[:, 0, :].cpu().numpy()
    ens_n  = all_trajs[:, :, 0, :].cpu().numpy() if all_trajs is not None else None

    obs_deg  = to_deg(denorm_traj(obs_n))
    gt_deg   = to_deg(denorm_traj(gt_n))
    pred_deg = to_deg(denorm_traj(pred_n))
    ens_deg  = to_deg(denorm_traj(ens_n)) if ens_n is not None else None

    # If the checkpoint's pred_len doesn't match the ground-truth length
    # (e.g. a mismatched --pred_len passed at load time), truncate both
    # to the shorter length rather than crashing on a shape mismatch —
    # but WARN loudly, since silently truncating changes what "72h" means.
    T_min = min(pred_deg.shape[0], gt_deg.shape[0])
    if pred_deg.shape[0] != gt_deg.shape[0]:
        print(f"  ⚠ LENGTH MISMATCH: pred has {pred_deg.shape[0]} steps, "
              f"gt has {gt_deg.shape[0]} steps — truncating both to "
              f"{T_min} steps ({T_min * 6}h). Verify --pred_len matches "
              f"this checkpoint's actual training config.")
        pred_deg = pred_deg[:T_min]
        gt_deg = gt_deg[:T_min]
        if ens_deg is not None and ens_deg.shape[1] != T_min:
            ens_deg = ens_deg[:, :T_min]

    errors_km = haversine_km(pred_deg, gt_deg)
    return obs_deg, gt_deg, pred_deg, ens_deg, errors_km


# ─────────────────────────────────────────────────────────────────────────────
#  Load model + dataset together (--mode single/case_study convenience)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_data(args, device, dset_type: str = "test"):
    detected = detect_pred_len(args.model_path)
    if args.pred_len != detected:
        print(f"  pred_len: {args.pred_len} -> {detected} (detected from checkpoint)")
        args.pred_len = detected

    model = TCFlowMatching(pred_len=args.pred_len, obs_len=args.obs_len).to(device)
    ck = torch.load(args.model_path, map_location=device, weights_only=False)
    sd = ck.get("model", ck.get("model_state_dict", ck.get("model_state", ck)))
    model.load_state_dict(sd, strict=False)
    model.eval()
    print("  Model loaded\n")

    dset, _ = data_loader(args, {"root": args.TC_data_path, "type": dset_type},
                          test=True, test_year=args.test_year)
    print(f"  Dataset: {len(dset)} samples\n")
    return model, dset


# ─────────────────────────────────────────────────────────────────────────────
#  --mode single — ONE storm, ONE model, full ensemble spread
#  (matches the reference "RITA — 72h FC | FM" layout: single map,
#  no inset zoom panel — removed per explicit request)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_forecast(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_name = args.tc_name.strip().upper()
    t_date, was_snapped = resolve_date(args.tc_date)

    print(f"{'=' * 65}")
    print(f"  TC-FM Visualize  |  {t_name}  @  {t_date}")
    print(f"{'=' * 65}\n")

    model, dset = load_model_and_data(args, device, args.dset_type)

    target, matched_obs_len, actual_date = find_target(dset, t_name, t_date, args.obs_len)
    if target is None:
        print(f"  '{t_name} @ {t_date}' not found.")
        list_available(dset, t_name, args.obs_len)
        return
    if actual_date != t_date:
        t_date = actual_date
    if matched_obs_len != args.obs_len:
        print(f"  [INFO] Using tydate[{matched_obs_len}] instead of "
              f"[{args.obs_len}] (date matched at a different window offset)\n")
    print(f"  Found: {t_name} @ {t_date}\n")

    (obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg,
     errors_km, cliper_err) = run_inference(
         model, target, device, args.ode_steps, args.num_ensemble)

    print("  Track errors (km):")
    for i, e in enumerate(errors_km):
        mark = "  <" if (i + 1) in [4, 8, 12] else ""
        print(f"    +{(i + 1) * 6:3d}h : {e:6.1f} km{mark}")
    print(f"    Mean  : {errors_km.mean():.1f} km\n")

    all_deg = np.vstack([obs_deg, gt_deg, pred_deg, ens_deg.reshape(-1, 2)])
    margin = 4.5
    lon_range = (all_deg[:, 0].min() - margin, all_deg[:, 0].max() + margin)
    lat_range = (all_deg[:, 1].min() - margin, all_deg[:, 1].max() + margin)

    dt_str = datetime.strptime(t_date, "%Y%m%d%H").strftime("%d %b %Y  %H:%M UTC")
    fh = args.pred_len * 6
    snap_note = f" [snapped from {args.tc_date}]" if was_snapped else ""

    # Main map (full context) + inset zoom on the predicted region only —
    # matches the reference "RITA — 72h FC | FM" image exactly (2 maps
    # side by side, no separate spread-vs-time panel).
    fig = plt.figure(figsize=(18, 13), facecolor=STYLE["bg_color"])
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 2, 1], wspace=0.12)
    ax_map = make_map_ax(fig, gs[0, :2], lon_range, lat_range)
    ax_inset = make_map_ax(fig, gs[0, 2], *_inset_range(pred_deg, ens_deg))

    _plot_on_ax(
        ax_map, lon_range, lat_range,
        obs_deg, gt_deg, pred_deg, pred_Me_n,
        all_trajs_deg=ens_deg if args.num_ensemble >= 3 else None,
        errors_km=errors_km,
        title=(f"{t_name}  \u2014  {fh}h FC  |  FM"
               f"  (ens={args.num_ensemble}, ode_steps={args.ode_steps}){snap_note}"),
        dt_str=dt_str,
    )
    _plot_on_ax(
        ax_inset, *_inset_range(pred_deg, ens_deg),
        obs_deg, gt_deg, pred_deg, pred_Me_n,
        all_trajs_deg=ens_deg if args.num_ensemble >= 3 else None,
        errors_km=None,   # don't repeat the error summary box in the inset — cone only
        title="Zoom: Predicted + Spread",
        dt_str="",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"forecast_{fh}h_{t_name}_{t_date}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=STYLE["bg_color"])
    plt.close()
    print(f"  Saved -> {out}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  --mode multi_model — same storm, multiple ARCHITECTURES overlaid
# ─────────────────────────────────────────────────────────────────────────────

def plot_multi_model_comparison(obs_deg, gt_deg, preds_by_model, errors_by_model,
                                 t_name: str, output_path: str):
    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    cur_pos = obs_deg[-1]

    all_pts = [obs_deg, gt_deg] + list(preds_by_model.values())
    all_deg = np.vstack(all_pts)
    margin = 3.0
    lon_range = (all_deg[:, 0].min() - margin, all_deg[:, 0].max() + margin)
    lat_range = (all_deg[:, 1].min() - margin, all_deg[:, 1].max() + margin)

    fig = plt.figure(figsize=(9, 9), facecolor=STYLE["bg_color"])
    ax = make_map_ax(fig, 111, lon_range, lat_range)
    outline = [pe.withStroke(linewidth=2.5, foreground="white")]

    def _plot(x, y, **kw):
        if HAS_CARTOPY:
            ax.plot(x, y, transform=transform, **kw)
        else:
            ax.plot(x, y, **kw)

    _plot(obs_deg[:, 0], obs_deg[:, 1], marker="o", color=STYLE["obs_color"],
          linewidth=STYLE["lw_thin"], markersize=STYLE["marker_size"],
          zorder=6, label="Observed")

    gt_lon = np.concatenate([[cur_pos[0]], gt_deg[:, 0]])
    gt_lat = np.concatenate([[cur_pos[1]], gt_deg[:, 1]])
    _plot(gt_lon, gt_lat, marker="o", color=STYLE["gt_color"],
          linewidth=2.2, markersize=STYLE["marker_size"],
          zorder=7, label="Ground truth", path_effects=outline)

    for model_name, pred_deg in preds_by_model.items():
        color = MODEL_COLORS.get(model_name, "#333333")
        p_lon = np.concatenate([[cur_pos[0]], pred_deg[:, 0]])
        p_lat = np.concatenate([[cur_pos[1]], pred_deg[:, 1]])
        ade = errors_by_model[model_name].mean()
        _plot(p_lon, p_lat, marker="o", color=color, linewidth=1.8,
              markersize=4.5, alpha=0.9, zorder=8,
              label=f"{model_name} (ADE={ade:.0f}km)", path_effects=outline)

    _plot([cur_pos[0]], [cur_pos[1]], marker="*", color="#FFD700",
          markersize=16, zorder=20)

    ax.legend(loc="lower right", fontsize=8, facecolor="white",
              edgecolor=STYLE["panel_edge"], labelcolor=STYLE["text_color"],
              framealpha=0.92)
    ax.set_title(f"{t_name} — Architecture comparison", color=STYLE["text_color"],
                 fontsize=11, fontweight="bold", pad=STYLE["title_pad"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> {output_path}")


def visualize_multi_model(args):
    """
    Compares FM against up to 4 baselines (ST-Trans/LSTM/GRU/RNN) on the
    SAME storm/window — each checkpoint CLI arg is optional, only the
    models that are given a checkpoint get plotted.

    Does NOT use args.model_path — each model reads its own checkpoint
    via args.fm_checkpoint/args.st_trans_checkpoint/args.lstm_checkpoint/
    args.gru_checkpoint/args.rnn_checkpoint (each optional).
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_name = args.tc_name.strip().upper()
    t_date, was_snapped = resolve_date(args.tc_date)

    print(f"{'=' * 65}")
    print(f"  TC-FM Visualize — Multi-model comparison  |  {t_name}  @  {t_date}")
    print(f"{'=' * 65}\n")

    jobs = []
    if args.fm_checkpoint:
        jobs.append(("FM", "fm", args.fm_checkpoint))
    if args.st_trans_checkpoint:
        jobs.append(("ST-Trans", "st_trans", args.st_trans_checkpoint))
    if args.lstm_checkpoint:
        jobs.append(("LSTM", "lstm", args.lstm_checkpoint))
    if args.gru_checkpoint:
        jobs.append(("GRU", "gru", args.gru_checkpoint))
    if args.rnn_checkpoint:
        jobs.append(("RNN", "rnn", args.rnn_checkpoint))

    if not jobs:
        print("  ERROR: no checkpoints given — pass at least one of "
              "--fm_checkpoint/--st_trans_checkpoint/--lstm_checkpoint/"
              "--gru_checkpoint/--rnn_checkpoint")
        return

    dset, _ = data_loader(args, {"root": args.TC_data_path, "type": args.dset_type},
                          test=True, test_year=args.test_year)
    print(f"  Dataset: {len(dset)} samples\n")

    target, matched_obs_len, actual_date = find_target(dset, t_name, t_date, args.obs_len)
    if target is None:
        print(f"  '{t_name} @ {t_date}' not found.")
        list_available(dset, t_name, args.obs_len)
        return
    if actual_date != t_date:
        t_date = actual_date
    print(f"  Found: {t_name} @ {t_date}\n")

    preds_by_model, errors_by_model = {}, {}
    obs_deg = gt_deg = None
    for display_name, kind, ckpt in jobs:
        print(f"  Loading {display_name}: {ckpt}")
        model = load_model_generic(ckpt, kind, device,
                                   obs_len=args.obs_len, pred_len=args.pred_len)
        od, gd, pd_, ens, err = run_inference_generic(
            model, target, device, kind, ode_steps=args.ode_steps,
            num_ensemble=(args.num_ensemble if kind == "fm" else 1))
        obs_deg, gt_deg = od, gd
        preds_by_model[display_name] = pd_
        errors_by_model[display_name] = err
        print(f"    {display_name}: ADE={err.mean():.1f}km")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"track_multi_{t_name}_{t_date}.png")
    plot_multi_model_comparison(obs_deg, gt_deg, preds_by_model, errors_by_model,
                                t_name, out)


# ─────────────────────────────────────────────────────────────────────────────
#  --mode multi_seed — same storm, SAME architecture, multiple seeds
# ─────────────────────────────────────────────────────────────────────────────

def _plot_multiseed_on_ax(ax, obs_deg, gt_deg, preds_by_seed, errors_by_seed,
                           title="", dt_str=""):
    """
    Draws observed/ground-truth tracks plus MULTIPLE seed predictions on
    one axis — the best-ADE seed drawn BOLD (thick line, full opacity,
    on top / highest zorder) and the other seeds drawn FAINT (thin line,
    low opacity, underneath), so a reader's eye is drawn to the best
    prediction while still seeing how much the others disagree.

    Deliberately mirrors _plot_on_ax()'s conventions (same NOW star,
    same +24h/+48h/+72h labels, same Wind(kt) legend, same white text
    outline for readability on the map background) so a --mode
    multi_seed figure looks visually consistent with a --mode single
    figure — this is NOT the ensemble-cone version (draw_smooth_cone is
    for K samples of ONE model, not for comparing separate seeds/runs).
    """
    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    outline = [pe.withStroke(linewidth=2.5, foreground="white")]
    cur_pos = obs_deg[-1]

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

    # 1. Observed track
    _plot(obs_deg[:, 0], obs_deg[:, 1], fmt="o-",
          color=STYLE["obs_color"], linewidth=STYLE["lw_thin"], markersize=5,
          markeredgecolor="white", markeredgewidth=0.8,
          zorder=7, path_effects=outline)

    # 2. Ground truth (Actual Track)
    gt_lon = np.concatenate([[cur_pos[0]], gt_deg[:, 0]])
    gt_lat = np.concatenate([[cur_pos[1]], gt_deg[:, 1]])
    _plot(gt_lon, gt_lat, fmt="o-",
          color=STYLE["gt_color"], linewidth=STYLE["lw_main"],
          markersize=STYLE["marker_size"],
          markeredgecolor="white", markeredgewidth=1.2,
          zorder=8, path_effects=outline)

    # 3. Rank seeds by ADE (ascending — best first), draw worst-to-best
    # so the BEST seed's line/markers end up on top (highest zorder).
    ranked = sorted(preds_by_seed.keys(), key=lambda s: errors_by_seed[s].mean())
    best_seed = ranked[0]

    for rank, seed_label in enumerate(reversed(ranked)):
        pred_deg = preds_by_seed[seed_label]
        ade = errors_by_seed[seed_label].mean()
        is_best = (seed_label == best_seed)
        p_lon = np.concatenate([[cur_pos[0]], pred_deg[:, 0]])
        p_lat = np.concatenate([[cur_pos[1]], pred_deg[:, 1]])

        if is_best:
            _plot(p_lon, p_lat, fmt="o-", color=STYLE["pred_color"],
                  linewidth=STYLE["lw_main"], markersize=STYLE["marker_size"],
                  alpha=1.0, markeredgecolor="white", markeredgewidth=1.0,
                  zorder=12, path_effects=outline,
                  label=f"seed={seed_label} (best, ADE={ade:.0f}km)")
        else:
            # [FIX] Previously linewidth=1.0 with markersize=3.5 at
            # alpha=0.35 made overlapping seeds look like scattered dots
            # rather than a readable line (markers visually dominated
            # the thin line beneath them, especially where seeds' tracks
            # nearly coincide). Thicker line + smaller markers reads as
            # a continuous faint track instead.
            _plot(p_lon, p_lat, fmt="-", color=STYLE["pred_color"],
                  linewidth=1.8, alpha=0.4, zorder=9,
                  solid_capstyle="round")
            _plot(p_lon, p_lat, fmt="o", color=STYLE["pred_color"],
                  markersize=2.5, alpha=0.5, zorder=9,
                  label=f"seed={seed_label} (ADE={ade:.0f}km)")

    # 4. Lead-time labels every 24h (based on the BEST seed's track)
    best_pred = preds_by_seed[best_seed]
    best_lon = np.concatenate([[cur_pos[0]], best_pred[:, 0]])
    best_lat = np.concatenate([[cur_pos[1]], best_pred[:, 1]])
    for i in range(len(best_lon)):
        h = i * 6
        if h % 24 == 0:
            lbl = "NOW" if i == 0 else f"+{h}h"
            _text(best_lon[i], best_lat[i] + 0.5, lbl,
                  color=STYLE["pred_color"], fontweight="bold", fontsize=7.5,
                  path_effects=outline)
            if i < len(gt_lon):
                _text(gt_lon[i], gt_lat[i] - 0.7, lbl,
                      color=STYLE["gt_color"], fontsize=6, alpha=0.9,
                      path_effects=outline)

    # 5. "NOW" star
    _scatter([cur_pos[0]], [cur_pos[1]], s=350, marker="*", color="#FFD700",
             edgecolors="black", linewidths=1.5, zorder=20)

    # 6. Error summary text box (best seed's ADE breakdown)
    n = len(errors_by_seed[best_seed])
    err_best = errors_by_seed[best_seed]
    lines = [f"Best seed: {best_seed}  (Mean ADE: {err_best.mean():.0f} km)"]
    for si, lh in [(3, 24), (7, 48), (11, 72)]:
        if si < n:
            lines.append(f" {lh}h: {err_best[si]:.0f} km")
    lines.append("")
    lines.append("All seeds (Mean ADE):")
    for seed_label in ranked:
        tag = " *" if seed_label == best_seed else ""
        lines.append(f" seed={seed_label}: {errors_by_seed[seed_label].mean():.0f} km{tag}")

    ax.text(0.02, 0.03, "\n".join(lines),
            transform=ax.transAxes, fontsize=8, va="bottom",
            color=STYLE["text_color"], family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      alpha=0.9, ec=STYLE["panel_edge"], lw=0.8),
            zorder=16)

    # 7. Legend — track legend only (Observed/GT/best-seed/other-seeds).
    # No Wind(kt) legend here: this function draws no wind-colored
    # markers (unlike _plot_on_ax, which shows per-step wind intensity
    # for a single model's prediction) — showing that legend would
    # falsely imply wind markers are present on this plot.
    track_handles = [
        Line2D([0], [0], color=STYLE["obs_color"], lw=2, label="Observed"),
        Line2D([0], [0], color=STYLE["gt_color"], lw=2, label="Actual Track"),
        Line2D([0], [0], color=STYLE["pred_color"], lw=2.5, alpha=1.0,
               label=f"Predicted (best seed={best_seed})"),
        Line2D([0], [0], color=STYLE["pred_color"], lw=1.0, alpha=0.35,
               label="Predicted (other seeds)"),
    ]
    ax.legend(handles=track_handles, loc="lower right", fontsize=7.5,
              facecolor="white", edgecolor=STYLE["panel_edge"],
              labelcolor=STYLE["text_color"], framealpha=0.92)

    ax.set_title(f"{title}\n{dt_str}", color=STYLE["text_color"], fontsize=10,
                 fontweight="bold", pad=STYLE["title_pad"],
                 bbox=dict(fc="white", alpha=0.9, ec=STYLE["panel_edge"], lw=1.2))
    ax.set_facecolor(STYLE["bg_color"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["panel_edge"])


def _inset_range_multiseed(preds_by_seed: dict):
    """Same idea as _inset_range(), but zoomed on the union of ALL
    seeds' predicted tracks (not a single model's ensemble)."""
    pts = np.vstack(list(preds_by_seed.values()))
    lon_span = pts[:, 0].max() - pts[:, 0].min()
    lat_span = pts[:, 1].max() - pts[:, 1].min()
    margin_lon = max(lon_span * 0.30, 0.3)
    margin_lat = max(lat_span * 0.30, 0.3)
    lon_range = (pts[:, 0].min() - margin_lon, pts[:, 0].max() + margin_lon)
    lat_range = (pts[:, 1].min() - margin_lat, pts[:, 1].max() + margin_lat)
    return lon_range, lat_range


def plot_multi_seed_comparison(obs_deg, gt_deg, preds_by_seed, errors_by_seed,
                               t_name: str, output_path: str, model_type: str,
                               fh: int, dt_str: str, snap_note: str = ""):
    """
    [FIX] Single full-width map only — NO inset zoom panel (removed per
    explicit request: the inset was making the layout cluttered, and the
    text boxes from the two panels were overlapping). The lowest-ADE
    seed is drawn bold/opaque, the rest faint — see
    _plot_multiseed_on_ax()'s docstring for the exact rule.
    """
    all_pts = [obs_deg, gt_deg] + list(preds_by_seed.values())
    all_deg = np.vstack(all_pts)
    margin = 4.5
    lon_range = (all_deg[:, 0].min() - margin, all_deg[:, 0].max() + margin)
    lat_range = (all_deg[:, 1].min() - margin, all_deg[:, 1].max() + margin)

    fig = plt.figure(figsize=(14, 12), facecolor=STYLE["bg_color"])
    ax_map = make_map_ax(fig, 111, lon_range, lat_range)

    title = f"{t_name}  \u2014  {fh}h FC  |  {model_type.upper()}  (seeds: {len(preds_by_seed)}){snap_note}"

    _plot_multiseed_on_ax(ax_map, obs_deg, gt_deg, preds_by_seed, errors_by_seed,
                          title=title, dt_str=dt_str)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=STYLE["bg_color"])
    plt.close()
    print(f"  Saved -> {output_path}")


def visualize_multi_seed(args):
    """
    Compares multiple seeds of the SAME architecture (default FM,
    switch via --model_type) on one storm/window — illustrates
    stability across random init, NOT a substitute for the mean±std
    statistical tables in generate_paper_report.py (this is a single
    forecast instance, not an aggregate quantity).

    Does NOT use args.model_path — each seed reads its own checkpoint
    via args.seed_checkpoints (a list, at least 1 required).
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_name = args.tc_name.strip().upper()
    t_date, was_snapped = resolve_date(args.tc_date)

    print(f"{'=' * 65}")
    print(f"  TC-FM Visualize — Multi-seed comparison ({args.model_type.upper()})"
          f"  |  {t_name}  @  {t_date}")
    print(f"{'=' * 65}\n")

    if not args.seed_checkpoints:
        print("  ERROR: --seed_checkpoints needs at least 1 checkpoint "
              "(recommended >=2 for a meaningful comparison)")
        return

    dset, _ = data_loader(args, {"root": args.TC_data_path, "type": args.dset_type},
                          test=True, test_year=args.test_year)
    print(f"  Dataset: {len(dset)} samples\n")

    target, matched_obs_len, actual_date = find_target(dset, t_name, t_date, args.obs_len)
    if target is None:
        print(f"  '{t_name} @ {t_date}' not found.")
        list_available(dset, t_name, args.obs_len)
        return
    if actual_date != t_date:
        t_date = actual_date
    print(f"  Found: {t_name} @ {t_date}\n")

    preds_by_seed, errors_by_seed = {}, {}
    obs_deg = gt_deg = None
    for idx, ckpt in enumerate(args.seed_checkpoints):
        seed_label = _infer_seed_label(ckpt, idx)
        print(f"  Loading seed={seed_label}: {ckpt}")
        model = load_model_generic(ckpt, args.model_type, device,
                                   obs_len=args.obs_len, pred_len=args.pred_len)
        od, gd, pd_, ens, err = run_inference_generic(
            model, target, device, args.model_type, ode_steps=args.ode_steps,
            num_ensemble=(args.num_ensemble if args.model_type == "fm" else 1))
        obs_deg, gt_deg = od, gd
        preds_by_seed[seed_label] = pd_
        errors_by_seed[seed_label] = err
        print(f"    seed={seed_label}: ADE={err.mean():.1f}km")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir,
                       f"track_multiseed_{args.model_type}_{t_name}_{t_date}.png")
    dt_str = datetime.strptime(t_date, "%Y%m%d%H").strftime("%d %b %Y  %H:%M UTC")
    fh = args.pred_len * 6
    snap_note = f" [snapped from {args.tc_date}]" if was_snapped else ""
    plot_multi_seed_comparison(obs_deg, gt_deg, preds_by_seed, errors_by_seed,
                               t_name, out, args.model_type, fh, dt_str, snap_note)


# ─────────────────────────────────────────────────────────────────────────────
#  --mode case_study — grid of several storms, each row = 1 map + 1 spread panel
# ─────────────────────────────────────────────────────────────────────────────

def visualize_case_study(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, dset = load_model_and_data(args, device, "test")

    cases = [
        {"name": args.straight1_name, "date": args.straight1_date,
         "label": "Straight-track 1"},
        {"name": args.straight2_name, "date": args.straight2_date,
         "label": "Straight-track 2"},
        {"name": "WIPHA", "date": args.recurv_date,
         "label": "Recurvature — WIPHA"},
    ]

    fig = plt.figure(figsize=(22, 8 * len(cases)), facecolor=STYLE["bg_color"])
    gs = fig.add_gridspec(len(cases), 3, wspace=0.10, hspace=0.28)

    for row, case in enumerate(cases):
        t_name = case["name"].strip().upper()
        t_date, was_snapped = resolve_date(case["date"])
        label = case.get("label", t_name)

        target, matched_obs_len, actual_date = find_target(dset, t_name, t_date, args.obs_len)

        if target is None:
            print(f"  ⚠  {t_name} @ {t_date} — not found")
            ax_nf = fig.add_subplot(gs[row, :])
            ax_nf.set_facecolor(STYLE["bg_color"])
            ax_nf.text(0.5, 0.5, f"NOT FOUND\n{t_name}",
                      ha="center", va="center", color="red",
                      fontsize=14, transform=ax_nf.transAxes)
            ax_nf.axis("off")
            continue

        if actual_date != t_date:
            t_date = actual_date
        if matched_obs_len != args.obs_len:
            print(f"  [INFO] {t_name}: using tydate[{matched_obs_len}] "
                  f"instead of [{args.obs_len}]")

        (obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg,
         errors_km, cliper_err) = run_inference(
            model, target, device, args.ode_steps, args.num_ensemble)

        all_deg = np.vstack([obs_deg, gt_deg, pred_deg, ens_deg.reshape(-1, 2)])
        margin = 4.5
        lon_range = (all_deg[:, 0].min() - margin, all_deg[:, 0].max() + margin)
        lat_range = (all_deg[:, 1].min() - margin, all_deg[:, 1].max() + margin)

        dt_str = datetime.strptime(t_date, "%Y%m%d%H").strftime("%d %b %Y %H:%M UTC")
        snap_note = f" [snapped from {case['date']}]" if was_snapped else ""
        ax_map = make_map_ax(fig, gs[row, :2], lon_range, lat_range)
        ax_err = fig.add_subplot(gs[row, 2])
        ax_err.set_facecolor(STYLE["bg_color"])

        _plot_on_ax(
            ax_map, lon_range, lat_range,
            obs_deg, gt_deg, pred_deg, pred_Me_n,
            all_trajs_deg=ens_deg if args.num_ensemble >= 3 else None,
            errors_km=errors_km,
            title=f"[{label}]  {t_name}  (ode_steps={args.ode_steps}){snap_note}",
            dt_str=dt_str,
        )
        plot_spread_over_time(ax_err, ens_deg, errors_km, cliper_err, t_name)

        ade = errors_km.mean()
        e72 = errors_km[11] if len(errors_km) > 11 else float("nan")
        print(f"  [{label}] ADE={ade:.1f} km  72h={e72:.1f} km")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "case_study_grid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=STYLE["bg_color"])
    plt.close()
    print(f"\n  Saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--model_path", default=None,
                   help="FM checkpoint — REQUIRED only for --mode single/"
                        "case_study (both are hardwired to exactly one FM "
                        "checkpoint via load_model_and_data()). NOT used by "
                        "--mode multi_model (use --fm_checkpoint/"
                        "--st_trans_checkpoint/--lstm_checkpoint/"
                        "--gru_checkpoint/--rnn_checkpoint instead) or "
                        "--mode multi_seed (use --seed_checkpoints instead) "
                        "— safe to leave blank for those two modes.")
    p.add_argument("--TC_data_path", required=True)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--mode", default="single",
                   choices=["single", "case_study", "multi_model", "multi_seed"])
    p.add_argument("--tc_name", default="WIPHA")
    p.add_argument("--tc_date", default="2019073106")
    p.add_argument("--dset_type", default="test")
    p.add_argument("--test_year", type=int, default=None)
    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12,
                   help="Only affects --mode single (visualize_forecast), "
                        "auto-corrected via detect_pred_len() if it doesn't "
                        "match the loaded checkpoint. Other modes don't read "
                        "this parameter.")
    p.add_argument("--num_ensemble", type=int, default=20)
    p.add_argument("--ode_steps", type=int, default=10)

    # --mode case_study: 3 fixed cases (2 straight-track + 1 recurvature)
    p.add_argument("--straight1_name", default="DANAS")
    p.add_argument("--straight1_date", default="2019072000")
    p.add_argument("--straight2_name", default="BEBINCA")
    p.add_argument("--straight2_date", default="2018081312")
    p.add_argument("--recurv_date", default="2019073106")

    # --mode multi_model: each checkpoint optional, only given models are plotted
    p.add_argument("--fm_checkpoint", default=None)
    p.add_argument("--st_trans_checkpoint", default=None)
    p.add_argument("--lstm_checkpoint", default=None)
    p.add_argument("--gru_checkpoint", default=None)
    p.add_argument("--rnn_checkpoint", default=None)

    # --mode multi_seed: multiple checkpoints of the SAME architecture
    p.add_argument("--seed_checkpoints", nargs="+", default=None,
                   help="List of checkpoint paths, one per seed, of the "
                        "SAME architecture (see --model_type). Example: "
                        "--seed_checkpoints runs/fm_seed0/best_model.pth "
                        "runs/fm_seed1/best_model.pth")
    p.add_argument("--model_type", default="fm",
                   choices=["fm", "st_trans", "lstm", "gru", "rnn"],
                   help="Architecture used for --mode multi_seed (default fm)")

    args = p.parse_args()

    if args.mode == "single":
        if not args.model_path:
            print("  ERROR: --model_path required for --mode single")
            sys.exit(1)
        visualize_forecast(args)
    elif args.mode == "case_study":
        if not args.model_path:
            print("  ERROR: --model_path required for --mode case_study")
            sys.exit(1)
        visualize_case_study(args)
    elif args.mode == "multi_model":
        visualize_multi_model(args)
    else:
        visualize_multi_seed(args)
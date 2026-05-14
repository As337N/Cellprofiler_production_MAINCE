"""
III_QC_collage.py
=================
Cell Painting plate collage builder + interactive HTML QC report.

Outputs
-------
  <plate>_QC.jpg          High-resolution collage (scale, q=80)
  <plate>_QC_web.jpg      Compressed thumbnail embedded in the HTML (~scale*0.4)
  <cohort>_QC_report.html Self-contained interactive report (Plotly embedded)

Usage
-----
    python III_QC_collage.py -i /output/QC/Images -o /output/QC/Collages
    python III_QC_collage.py -i /data -o /out --cohort MyCohort --n-sigma 3

Requirements
------------
    pip install pandas numpy pillow tifffile opencv-python scipy
    # Plotly JS is fetched once from CDN and embedded automatically.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import median_abs_deviation


# ── Fonts ──────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_FONT_PATHS = {
    "bold": [
        _SCRIPT_DIR / "fonts" / "DejaVuSans-Bold.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
        Path("/Library/Fonts/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans-Bold.ttf"),
    ],
    "regular": [
        _SCRIPT_DIR / "fonts" / "DejaVuSans.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("/Library/Fonts/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans.ttf"),
    ],
}
_font_cache: dict = {}


def _font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    key = (size, bold)
    if key not in _font_cache:
        for path in _FONT_PATHS["bold" if bold else "regular"]:
            if path.exists():
                try:
                    _font_cache[key] = ImageFont.truetype(str(path), size)
                    break
                except (OSError, IOError):
                    continue
        else:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def _text_h(draw, text: str, font) -> int:
    try:
        return draw.textbbox((0, 0), text, font=font)[3]
    except AttributeError:
        return font.size if hasattr(font, "size") else 14


# ── QC constants ───────────────────────────────────────────────────────────────

# Absolute thresholds (floor / ceiling — always enforced regardless of n_sigma).
# See ThresholdEngine docstring for per-metric rationale.
THRESHOLDS: dict[str, tuple] = {
    "PowerLogLogSlope": (-2.5, -1.0),
    "MaxIntensity":     (None,  0.95),
    "FocusScore":       (0.005, None),
}
THRESHOLDS_LOCAL_FOCUS: dict[str, tuple] = {
    "DNA":           (0.80,  None),
    "Syto":          (0.05,  None),
    "Golgi_c":       (0.03,  None),
    "ER_c":          (0.08,  None),
    "Mito_c":        (0.005, None),
    "Brightfield_c": (0.001, None),
}

CHANNELS       = ["DNA", "Syto", "Golgi_c", "ER_c", "Mito_c", "Brightfield_c"]
CHANNELS_EXTRA = []   # channels with focus metrics only

CHANNEL_LABELS = {
    "DNA": "DNA", "Syto": "Sy", "Golgi_c": "Go",
    "ER_c": "ER", "Mito_c": "Mi", "Brightfield_c": "BF",
}
CHANNEL_COLORS = {
    "DNA": (100, 160, 255), "Syto": (80, 220, 120), "Golgi_c": (255, 180, 60),
    "ER_c": (180, 90, 255), "Mito_c": (255, 80, 80), "Brightfield_c": (160, 160, 160),
}

ILLUM_METRICS  = ["PowerLogLogSlope", "MaxIntensity"]
FOCUS_METRICS  = ["FocusScore", "LocalFocusScore"]
BORDER_METRIC  = "PowerLogLogSlope"
METRIC_LABELS  = {
    "PowerLogLogSlope": "Slope", "MaxIntensity": "MaxInt",
    "FocusScore": "Focus",       "LocalFocusScore": "LocalFoc",
}

COL_PASS    = (75,  215,  95)
COL_FAIL    = (255,  65,  65)
COL_NODATA  = (110, 110, 120)

COUNT_COLS = {
    "Raw": "Count_Raw_nuclei", "Filtered": "Count_Nuclei",
    "Cells": "Count_Cells",    "Artifacts": "Count_Illum_artifacts",
}
AREA_COL           = "ImageQuality_TotalArea_Brightfield_c"
DEFAULT_IMAGE_AREA = 1_166_400   # 1080×1080 px fallback


def _miq_col(metric: str, channel: str) -> str:
    if metric == "LocalFocusScore":
        return f"ImageQuality_LocalFocusScore_{channel}_10"
    return f"ImageQuality_{metric}_{channel}"


METRIC_COLS: dict[str, list[str]] = {
    mk: [_miq_col(mk, ch) for ch in CHANNELS]
    for mk in ("PowerLogLogSlope", "MaxIntensity", "FocusScore", "LocalFocusScore")
}
for mk in ("FocusScore", "LocalFocusScore"):
    METRIC_COLS[mk] += [_miq_col(mk, ch) for ch in CHANNELS_EXTRA]

COL_TO_CHANNEL: dict[str, str] = {
    col: ch
    for mk, cols in METRIC_COLS.items()
    for col in cols
    for ch in CHANNELS + CHANNELS_EXTRA
    if ch in col
}


# ── ThresholdEngine ────────────────────────────────────────────────────────────

class ThresholdEngine:
    """
    Hybrid threshold evaluator: absolute bounds + adaptive MAD-based outlier detection.

    Absolute bounds (always enforced)
    ----------------------------------
    PowerLogLogSlope  (-2.5, -1.0)   Log-log power spectrum slope; primary blur metric.
    MaxIntensity      (None, 0.95)   Saturation guard.
    FocusScore        (0.005, None)  Loose — catches blank/fully out-of-focus only.
    LocalFocusScore   per-channel    Main focus metric; thresholds vary by signal density.

    Adaptive bounds (MAD-based, fitted per plate)
    ----------------------------------------------
    For each metric×channel, computes median ± n_sigma * MAD across all wells.
    A well is flagged as an adaptive outlier if it falls outside this range AND
    also fails the absolute bound in the same direction.
    This prevents flagging wells that are statistically unusual but still within
    biologically valid absolute limits.

    Parameters
    ----------
    n_sigma : float
        Number of MAD-equivalent sigmas for the adaptive band (default 3.0).
    """

    def __init__(self, n_sigma: float = 3.0):
        self.n_sigma   = n_sigma
        self._adaptive: dict[str, dict[str, tuple]] = {}   # {metric: {col: (lo, hi)}}

    def fit(self, plate_qc: dict) -> None:
        """Compute adaptive bounds from plate data. Call once per plate."""
        self._adaptive = {}
        for mk, cols in METRIC_COLS.items():
            self._adaptive[mk] = {}
            for col in cols:
                vals = [
                    m[col] for m in plate_qc.values()
                    if col in m and m[col] is not None and not np.isnan(m[col])
                ]
                if len(vals) < 5:
                    continue
                arr    = np.array(vals)
                median = float(np.median(arr))
                mad    = float(median_abs_deviation(arr, scale="normal"))
                self._adaptive[mk][col] = (
                    median - self.n_sigma * mad,
                    median + self.n_sigma * mad,
                )

    def passes(self, value, metric_key: str, channel: str = "") -> bool | None:
        """
        Returns True (pass), False (fail), or None (no data).
        Fails if the value breaks the absolute bound OR is an adaptive outlier
        that also violates the absolute bound direction.
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None

        # Absolute check
        if metric_key == "LocalFocusScore":
            abs_lo, abs_hi = THRESHOLDS_LOCAL_FOCUS.get(channel, (None, None))
        else:
            abs_lo, abs_hi = THRESHOLDS.get(metric_key, (None, None))

        abs_fail = (
            (abs_lo is not None and value < abs_lo) or
            (abs_hi is not None and value > abs_hi)
        )

        # Adaptive check — only flags if within an absolute limit direction
        col      = _miq_col(metric_key, channel) if channel else None
        adp_fail = False
        if col and col in self._adaptive.get(metric_key, {}):
            adp_lo, adp_hi = self._adaptive[metric_key][col]
            if value < adp_lo and (abs_lo is not None and value < abs_lo):
                adp_fail = True
            if value > adp_hi and (abs_hi is not None and value > abs_hi):
                adp_fail = True

        return not (abs_fail or adp_fail)

    def val_color(self, value, metric_key: str, channel: str = "") -> tuple:
        p = self.passes(value, metric_key, channel)
        return COL_NODATA if p is None else (COL_PASS if p else COL_FAIL)

    def adaptive_bounds(self, metric_key: str, col: str) -> tuple | None:
        """Return (lo, hi) adaptive bounds for a column, or None if not fitted."""
        return self._adaptive.get(metric_key, {}).get(col)


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_qc_tsv(tsv_path) -> dict:
    """Load CellProfiler Image.txt TSV → {plate: {well: {col: value}}}."""
    if tsv_path is None:
        return {}
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        print(f"[warn] QC file not found: {tsv_path}")
        return {}

    df        = pd.read_csv(tsv_path, sep="\t")
    plate_col = "Metadata_Plate" if "Metadata_Plate" in df.columns else None
    well_col  = "Metadata_Well"  if "Metadata_Well"  in df.columns else None
    if well_col is None:
        print("[warn] Metadata_Well not found — QC enrichment disabled.")
        return {}

    mean_cols = list(dict.fromkeys(
        c for cols in METRIC_COLS.values() for c in cols if c in df.columns
    ))
    sum_cols  = list(dict.fromkeys(
        c for c in df.columns
        if c.startswith("Count_") or c == AREA_COL
    ))
    gkeys = [plate_col, well_col] if plate_col else [well_col]
    grp   = df.groupby(gkeys)

    agg_mean = grp[mean_cols].mean().reset_index() if mean_cols else None
    agg_sum  = grp[sum_cols].sum().reset_index()   if sum_cols  else None

    if agg_mean is not None and agg_sum is not None:
        agg = agg_mean.merge(agg_sum, on=gkeys, how="left")
    else:
        agg = agg_mean or agg_sum

    all_cols = mean_cols + [c for c in sum_cols if c not in mean_cols]
    result   = defaultdict(dict)
    for _, row in agg.iterrows():
        plate = str(row[plate_col]).strip() if plate_col else "Plate"
        well  = str(row[well_col]).strip().upper()
        result[plate][well] = {c: row[c] for c in all_cols if c in row.index}

    print(f"[qc] {sum(len(v) for v in result.values())} wells "
          f"across {len(result)} plate(s).")
    return result


def load_platemap(platemap_path) -> dict:
    """Load platemap CSV → {plate: {well: compound}}."""
    if platemap_path is None:
        return {}
    platemap_path = Path(platemap_path)
    if not platemap_path.exists():
        print(f"[warn] Platemap not found: {platemap_path}")
        return {}

    df           = pd.read_csv(platemap_path)
    well_col     = next((c for c in df.columns if "Well"        in c), None)
    compound_col = next((c for c in df.columns if "Compound"    in c
                         or "Perturbation" in c), None)
    plate_col    = next((c for c in df.columns if "Plate"       in c), None)

    if not well_col or not compound_col:
        print(f"[warn] Platemap missing Well/Compound column. Found: {list(df.columns)}")
        return {}

    result = defaultdict(dict)
    for _, row in df.iterrows():
        well  = str(row[well_col]).strip().upper()
        cmpd  = str(row[compound_col]).strip()
        raw_p = str(row[plate_col]).strip() if plate_col else "Plate"
        plate = raw_p if raw_p.startswith("P") else f"P{raw_p}"
        result[plate][well] = cmpd

    print(f"[platemap] {sum(len(v) for v in result.values())} well→compound mappings.")
    return result


# ── Image helpers ──────────────────────────────────────────────────────────────

def well_label(row: int, col: int) -> str:
    return f"{chr(ord('A') + row - 1)}{col:02d}"


def load_and_downscale(path: Path, scale: float = 0.5) -> np.ndarray:
    with tiff.TiffFile(path) as tf:
        img = tf.pages[0].asarray()
    if img.dtype != np.uint8:
        p_max = img.max()
        img = (img / p_max * 255).astype(np.uint8) if p_max > 0 else img.astype(np.uint8)
    if scale != 1.0:
        img = cv2.resize(img,
                         (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def build_well_montage(args: tuple) -> tuple:
    (r, c), sites, scale, spw = args
    tiles = [load_and_downscale(sites[s], scale)
             for s in range(1, spw + 1) if s in sites]
    if not tiles:
        return (r, c), None
    n     = max(1, int(np.ceil(np.sqrt(len(tiles)))))
    blank = np.zeros_like(tiles[0])
    while len(tiles) % n:
        tiles.append(blank)
    rows = [np.concatenate(tiles[i:i + n], axis=1) for i in range(0, len(tiles), n)]
    return (r, c), np.concatenate(rows, axis=0)


def _make_tile(tile: np.ndarray, border_rgb: tuple, border_w: int = 6) -> np.ndarray:
    img    = tile if tile.ndim == 3 else np.stack([tile] * 3, axis=-1)
    canvas = Image.fromarray(img.copy())
    draw   = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, img.shape[1] - 1, img.shape[0] - 1],
                   outline=border_rgb, width=border_w)
    return np.array(canvas)


# ── QC helpers ─────────────────────────────────────────────────────────────────

def _slope_range(plate_qc: dict) -> tuple:
    cols = METRIC_COLS[BORDER_METRIC]
    vals = [m[col] for m in plate_qc.values() for col in cols
            if (col in m) and m[col] is not None and not np.isnan(m[col])]
    lo, hi = THRESHOLDS[BORDER_METRIC]
    return (min(vals) if vals else lo, max(vals) if vals else hi)


def _slope_to_rgb(well_metrics: dict | None, engine: ThresholdEngine) -> tuple:
    results = [
        engine.passes(well_metrics.get(col) if well_metrics else None, BORDER_METRIC)
        for col in METRIC_COLS[BORDER_METRIC]
    ]
    results = [r for r in results if r is not None]
    if not results:
        return (45, 45, 55)
    n = sum(results)
    if n == len(results): return (55, 205, 80)
    if n == 0:            return (215, 35, 35)
    return (255, 145, 0)


def _artifact_density(metrics: dict) -> float | None:
    n = metrics.get(COUNT_COLS["Artifacts"])
    if n is None or np.isnan(n):
        return None
    area = metrics.get(AREA_COL) or DEFAULT_IMAGE_AREA
    return (n / area) * 1000 if area > 0 else None


def _count_summary(metrics: dict) -> list[tuple[str, str]]:
    rows = [(lbl, "—" if (v := metrics.get(col)) is None or
             (isinstance(v, float) and np.isnan(v)) else str(int(round(v))))
            for lbl, col in COUNT_COLS.items()]
    d = _artifact_density(metrics)
    rows.append(("Art/kpx²", "—" if d is None else f"{d:.2f}"))
    return rows


def _well_passes_all(metrics: dict, engine: ThresholdEngine) -> bool:
    return all(
        engine.passes(metrics.get(col), mk, COL_TO_CHANNEL.get(col, "")) is not False
        for mk, cols in METRIC_COLS.items() for col in cols
    )


def _well_passes_group(metrics: dict, group: list, engine: ThresholdEngine) -> bool:
    return all(
        engine.passes(metrics.get(col), mk, COL_TO_CHANNEL.get(col, "")) is not False
        for mk in group for col in METRIC_COLS[mk]
    )


# ── Rendering ──────────────────────────────────────────────────────────────────

def _make_band(width: int, well_labels_in_row: list, plate_qc: dict,
               band_height: int, font_size: int, tile_width: int,
               plate_map: dict | None, engine: ThresholdEngine) -> np.ndarray:
    """
    QC band below each plate row. Two-pass: first measures needed height,
    then draws. Absorbs the old _measure_band_height function.
    """
    ALL_METRICS = ILLUM_METRICS + FOCUS_METRICS
    fw = _font(font_size + 2, bold=True)
    fl = _font(font_size - 1, bold=True)
    fv = _font(font_size - 1, bold=False)

    # ── Pass 1: measure ────────────────────────────────────────────────────────
    dummy = Image.new("RGB", (tile_width, 10))
    dd    = ImageDraw.Draw(dummy)
    max_y = 0
    for wlabel in well_labels_in_row:
        metrics  = plate_qc.get(wlabel)
        compound = (plate_map or {}).get(wlabel, "")
        y = 7 + _text_h(dd, wlabel, fw) + 3
        if compound:
            y += _text_h(dd, compound, fv) + 4
        if metrics:
            for group in (ILLUM_METRICS, FOCUS_METRICS):
                yc = y
                for mk in group:
                    yc += _text_h(dd, f"{METRIC_LABELS[mk]}:", fl) + 1
                    for col in METRIC_COLS[mk]:
                        v = metrics.get(col)
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            yc += _text_h(dd, " xx: 0.000", fv) + 1
                    yc += 3
                max_y = max(max_y, yc)
            yc = y + _text_h(dd, "Objects:", fl) + 2
            yc += len(_count_summary(metrics)) * (_text_h(dd, " xx: 000", fv) + 1)
            max_y = max(max_y, yc)
        else:
            max_y = max(max_y, y + 10)

    band_height = max(band_height, max_y + 16)

    # ── Pass 2: draw ───────────────────────────────────────────────────────────
    band = Image.new("RGB", (width, band_height), color=(12, 12, 18))
    draw = ImageDraw.Draw(band)

    for idx, wlabel in enumerate(well_labels_in_row):
        metrics = plate_qc.get(wlabel)
        x0, x1  = idx * tile_width, idx * tile_width + tile_width - 1

        pass_vals = [
            engine.passes(metrics.get(col) if metrics else None, mk,
                          COL_TO_CHANNEL.get(col, ""))
            for mk in ALL_METRICS for col in METRIC_COLS[mk]
        ]
        pass_vals = [p for p in pass_vals if p is not None]

        if pass_vals:
            frac   = sum(pass_vals) / len(pass_vals)
            tint   = (20, 55, 25) if frac == 1.0 else (55, 15, 15) if frac == 0 else (50, 35, 10)
            accent = COL_PASS if frac == 1.0 else (COL_FAIL if frac == 0 else (255, 155, 0))
        else:
            tint, accent = (18, 18, 24), (60, 70, 90)

        draw.rectangle([x0, 0, x1, band_height - 1], fill=tint)
        draw.rectangle([x0, 0, x1, 5], fill=accent)

        wc      = tuple(min(255, int(c * 1.4 + 50)) for c in accent)
        draw.text((x0 + 4, 7), wlabel, fill=wc, font=fw)
        y_start = 7 + _text_h(draw, wlabel, fw) + 3

        compound = (plate_map or {}).get(wlabel, "")
        if compound:
            max_chars = max(6, (tile_width - 8) // max(1, font_size - 3))
            disp = compound if len(compound) <= max_chars else compound[:max_chars - 1] + "…"
            draw.text((x0 + 4, y_start), disp, fill=(200, 220, 180), font=fv)
            y_start += _text_h(draw, disp, fv) + 4

        if not metrics:
            draw.line([x1, 0, x1, band_height - 1], fill=(28, 28, 36), width=1)
            continue

        col_w   = (tile_width - 10) // 3
        x_left  = x0 + 4
        x_right = x0 + 4 + col_w
        x_count = x0 + 4 + col_w * 2

        for col_x, group in ((x_left, ILLUM_METRICS), (x_right, FOCUS_METRICS)):
            y = y_start
            for mk in group:
                lbl = METRIC_LABELS[mk]
                draw.text((col_x, y), f"{lbl}:", fill=(170, 195, 225), font=fl)
                y += _text_h(draw, f"{lbl}:", fl) + 1
                for col in METRIC_COLS[mk]:
                    v = metrics.get(col)
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        continue
                    ch    = COL_TO_CHANNEL.get(col, col)
                    short = CHANNEL_LABELS.get(ch, ch[:2])
                    vc    = engine.val_color(v, mk, ch)
                    txt   = f" {short}: {v:.3f}"
                    draw.text((col_x, y + 1), txt, fill=(0, 0, 0), font=fv)
                    draw.text((col_x, y),     txt, fill=vc,         font=fv)
                    y += _text_h(draw, txt, fv) + 1
                y += 3

        y = y_start
        draw.text((x_count, y), "Objects:", fill=(170, 195, 225), font=fl)
        y += _text_h(draw, "Objects:", fl) + 2
        for lbl, val in _count_summary(metrics):
            vc = COL_NODATA
            if lbl == "Art/kpx²":
                try:
                    fv_ = float(val)
                    vc  = COL_FAIL if fv_ > 0.05 else ((255, 190, 0) if fv_ > 0.02 else COL_PASS)
                except ValueError:
                    pass
            else:
                vc = (200, 210, 220)
            txt = f" {lbl}: {val}"
            draw.text((x_count, y + 1), txt, fill=(0, 0, 0), font=fv)
            draw.text((x_count, y),     txt, fill=vc,         font=fv)
            y += _text_h(draw, txt, fv) + 1

        draw.line([x1, 0, x1, band_height - 1], fill=(28, 28, 36), width=1)

    return np.array(band)


def make_header(width: int, title: str, font_size: int = 20) -> np.ndarray:
    header = Image.new("RGB", (width, 58), color=(10, 12, 22))
    draw   = ImageDraw.Draw(header)
    draw.text((12, 14), title, fill=(180, 220, 255), font=_font(font_size, bold=True))
    draw.rectangle([0, 55, width - 1, 57], fill=(40, 60, 100))
    legend = [("Pass", COL_PASS), ("Fail", COL_FAIL),
              ("Both pass", (55, 205, 80)), ("Mixed", (255, 145, 0)),
              ("Both fail", (215, 35, 35))]
    x  = width - 12
    fl = _font(font_size - 5, bold=False)
    for lbl, col in reversed(legend):
        try:
            tw = draw.textbbox((0, 0), lbl, font=fl)[2]
        except AttributeError:
            tw = len(lbl) * 7
        x -= tw + 4
        draw.text((x, 20), lbl, fill=col, font=fl)
        x -= 16
        draw.rectangle([x, 20, x + 12, 34], fill=col)
        x -= 10
    return np.array(header)


def make_report_footer(width: int, plate_name: str, plate_qc: dict,
                        engine: ThresholdEngine, font_size: int = 16,
                        plate_map: dict | None = None) -> np.ndarray:
    n_wells   = len(plate_qc)
    n_pass    = sum(1 for m in plate_qc.values() if _well_passes_all(m, engine))
    n_illum_p = sum(1 for m in plate_qc.values() if _well_passes_group(m, ILLUM_METRICS, engine))
    n_focus_p = sum(1 for m in plate_qc.values() if _well_passes_group(m, FOCUS_METRICS, engine))
    pct       = lambda n: f"{100 * n / n_wells:.1f}%" if n_wells else "—"

    stats = {mk: {ch: [] for ch in CHANNELS + CHANNELS_EXTRA} for mk in METRIC_COLS}
    for m in plate_qc.values():
        for mk, cols in METRIC_COLS.items():
            for col in cols:
                ch = COL_TO_CHANNEL.get(col, col)
                v  = m.get(col)
                if v is not None and not np.isnan(v):
                    stats[mk][ch].append(v)

    failing = []
    for wl, m in sorted(plate_qc.items()):
        reasons = [
            f"{METRIC_LABELS.get(mk, mk)}/{CHANNEL_LABELS.get(ch_key, '?')}"
            for mk, cols in METRIC_COLS.items()
            for col in cols
            if (v := m.get(col)) is not None and not np.isnan(v)
            and engine.passes(v, mk, ch_key := COL_TO_CHANNEL.get(col, "")) is False
        ]
        if reasons:
            failing.append((wl, reasons))

    ft  = _font(font_size + 6, bold=True)
    fs  = _font(font_size + 1, bold=True)
    fb  = _font(font_size - 1, bold=True)
    fr  = _font(font_size - 1, bold=False)

    def _draw_footer(draw, measure_only=False):
        y = 16

        def line(txt, fill, font, x=16):
            nonlocal y
            if not measure_only:
                draw.text((x, y + 1), txt, fill=(0, 0, 0), font=font)
                draw.text((x, y),     txt, fill=fill,       font=font)
            y += _text_h(draw, txt, font) + 4

        def rule(color=(40, 60, 110)):
            nonlocal y
            if not measure_only:
                draw.rectangle([16, y, width - 16, y + 2], fill=color)
            y += 6

        line(f"  QC REPORT — {plate_name}", (180, 210, 255), ft)
        rule((50, 75, 140))

        oc = COL_PASS if n_pass / max(n_wells, 1) >= 0.8 else \
             (COL_FAIL if n_pass / max(n_wells, 1) < 0.5 else (255, 190, 0))
        line(f"  Overall:       {n_pass}/{n_wells} wells pass all metrics  ({pct(n_pass)})", oc, fs)

        ic = COL_PASS if n_illum_p / max(n_wells, 1) >= 0.8 else \
             (COL_FAIL if n_illum_p / max(n_wells, 1) < 0.5 else (255, 190, 0))
        fc_ = COL_PASS if n_focus_p / max(n_wells, 1) >= 0.8 else \
              (COL_FAIL if n_focus_p / max(n_wells, 1) < 0.5 else (255, 190, 0))
        line(f"  Illumination:  {n_illum_p}/{n_wells} pass ({pct(n_illum_p)})  [Slope + MaxInt]", ic, fb)
        line(f"  Focus:         {n_focus_p}/{n_wells} pass ({pct(n_focus_p)})  [FocusScore + LocalFocusScore]", fc_, fb)
        rule(); y += 4

        for grp_lbl, grp_col, grp_metrics in (
            ("ILLUMINATION METRICS", (255, 200, 80), ILLUM_METRICS),
            ("FOCUS METRICS",        (100, 180, 255), FOCUS_METRICS),
        ):
            line(f"  {grp_lbl}", grp_col, fs)
            line(f"    {'Metric':<22}  {'Channel':<8}  {'Threshold':<14}  {'Pass':>12}  {'mean ± SD':<22}",
                 (110, 140, 175), fb)

            for mk in grp_metrics:
                ml = METRIC_LABELS.get(mk, mk)
                first = True
                for col in METRIC_COLS[mk]:
                    ch       = COL_TO_CHANNEL.get(col, col)
                    vals     = stats[mk].get(ch, [])
                    ch_short = CHANNEL_LABELS.get(ch, ch[:4])
                    ch_color = CHANNEL_COLORS.get(ch, (150, 170, 200))

                    if mk == "LocalFocusScore":
                        lo, hi = THRESHOLDS_LOCAL_FOCUS.get(ch, (None, None))
                    else:
                        lo, hi = THRESHOLDS.get(mk, (None, None))
                    tstr = (f"> {lo}" if hi is None else f"< {hi}" if lo is None
                            else f"{lo} to {hi}")

                    if vals:
                        np_ = sum(1 for v in vals if engine.passes(v, mk, ch))
                        pp  = 100 * np_ / len(vals)
                        pc  = COL_PASS if pp >= 80 else (COL_FAIL if pp < 50 else (255, 190, 0))
                        stat_str = f"{np_}/{len(vals)} ({pp:.0f}%)   μ={np.mean(vals):.3f}±{np.std(vals):.3f}"
                    else:
                        pc, stat_str = COL_NODATA, "—"

                    m_col   = f"    {ml:<22}" if first else f"    {'':22}"
                    row_txt = f"{m_col}  {ch_short:<8}  {tstr:<14}  "
                    first   = False

                    if not measure_only:
                        x = 16
                        draw.text((x, y + 1), row_txt, fill=(0, 0, 0), font=fr)
                        draw.text((x, y),     row_txt, fill=(130, 160, 200), font=fr)
                        try:
                            x += draw.textbbox((0, 0), row_txt, font=fr)[2]
                        except AttributeError:
                            x += len(row_txt) * (font_size - 4)
                        draw.text((x, y + 1), f"{ch_short}  ", fill=(0, 0, 0), font=fr)
                        draw.text((x, y),     f"{ch_short}  ", fill=ch_color,   font=fr)
                        try:
                            x += draw.textbbox((0, 0), f"{ch_short}  ", font=fr)[2]
                        except AttributeError:
                            x += (len(ch_short) + 2) * (font_size - 4)
                        draw.text((x, y + 1), stat_str, fill=(0, 0, 0), font=fr)
                        draw.text((x, y),     stat_str, fill=pc,         font=fr)
                    y += _text_h(draw, row_txt, fr) + 2
                y += 4
            y += 4; rule(); y += 4

        line(f"  Failing wells  ({len(failing)}):", (200, 215, 235), fs)
        if failing:
            for i in range(0, len(failing), 5):
                seg = failing[i:i + 5]
                line("    " + "   ".join(f"{wl} [{', '.join(r)}]" for wl, r in seg),
                     COL_FAIL, fr)
        else:
            line("    None — all wells pass.", COL_PASS, fr)
        rule(); y += 6

        # Object counts table
        line("  OBJECT COUNTS PER WELL", (180, 210, 255), fs)
        line(f"    {'Well':<6}  {'Compound':<28}  {'Raw':>6}  {'Filtered':>8}  "
             f"{'Cells':>6}  {'Artifacts':>9}  {'Art/kpx²':>9}",
             (110, 140, 175), fb)
        for wl in sorted(plate_qc.keys()):
            m        = plate_qc[wl]
            compound = (plate_map or {}).get(wl, "—")
            fmt      = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) \
                                  else str(int(round(v)))
            density  = _artifact_density(m)
            dens_str = "—" if density is None else f"{density:.3f}"
            dc       = (COL_FAIL if density is not None and density > 0.05
                        else (255, 190, 0) if density is not None and density > 0.02
                        else COL_PASS if density is not None else COL_NODATA)
            row_txt  = (f"    {wl:<6}  {compound[:28]:<28}  "
                        f"{fmt(m.get(COUNT_COLS['Raw'])):>6}  "
                        f"{fmt(m.get(COUNT_COLS['Filtered'])):>8}  "
                        f"{fmt(m.get(COUNT_COLS['Cells'])):>6}  "
                        f"{fmt(m.get(COUNT_COLS['Artifacts'])):>9}  ")
            if not measure_only:
                x = 16
                draw.text((x, y + 1), row_txt, fill=(0, 0, 0), font=fr)
                draw.text((x, y),     row_txt, fill=(160, 180, 200), font=fr)
                try:
                    x += draw.textbbox((0, 0), row_txt, font=fr)[2]
                except AttributeError:
                    x += len(row_txt) * (font_size - 4)
                draw.text((x, y + 1), dens_str, fill=(0, 0, 0), font=fr)
                draw.text((x, y),     dens_str, fill=dc,         font=fr)
            y += _text_h(draw, row_txt, fr) + 2
        rule(); y += 6

        # Threshold reference
        line("  Thresholds applied:", (190, 205, 225), fs)
        for mk, (lo, hi) in THRESHOLDS.items():
            tstr = (f"> {lo}" if hi is None else f"< {hi}" if lo is None else f"{lo} to {hi}")
            line(f"    {METRIC_LABELS.get(mk, mk)}: {tstr}", (150, 170, 195), fr)
        line(f"  Adaptive: median ± {engine.n_sigma} σ (MAD) — per plate",
             (150, 170, 195), fr)
        y += 16
        return y

    dummy = Image.new("RGB", (width, 10))
    total_h = _draw_footer(ImageDraw.Draw(dummy), measure_only=True) + 20
    footer  = Image.new("RGB", (width, total_h), color=(10, 12, 22))
    fdraw   = ImageDraw.Draw(footer)
    fdraw.rectangle([0, 0, width - 1, 5], fill=(40, 60, 120))
    _draw_footer(fdraw, measure_only=False)
    return np.array(footer)


# ── HTML report ────────────────────────────────────────────────────────────────

def _fetch_plotly_js() -> str:
    """Download Plotly JS once and return as string for embedding."""
    url = "https://cdn.plot.ly/plotly-2.35.2.min.js"
    cache = Path(__file__).parent / ".plotly_cache.js"
    if cache.exists():
        return cache.read_text()
    print("[html] Downloading Plotly JS for embedding (one-time)…")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            js = r.read().decode()
        cache.write_text(js)
        return js
    except Exception as e:
        print(f"[warn] Could not fetch Plotly: {e}. HTML will use CDN fallback.")
        return f'/* CDN fallback */\ndocument.write(\'<script src="{url}"></script>\');'


def _collage_to_b64(collage_arr: np.ndarray, web_scale: float = 0.2) -> str:
    """Resize collage for web and return as base64 JPEG string."""
    h, w    = collage_arr.shape[:2]
    web_img = Image.fromarray(collage_arr).resize(
        (max(1, int(w * web_scale)), max(1, int(h * web_scale))),
        Image.LANCZOS
    )
    buf = io.BytesIO()
    web_img.save(buf, format="JPEG", quality=72,
                 optimize=True, progressive=True, subsampling=2)
    return base64.b64encode(buf.getvalue()).decode()


def _round_floats(obj, decimals: int = 3):
    if isinstance(obj, float):
        return round(obj, decimals) if not np.isnan(obj) else None
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    return obj


def _make_report_collage(collage_arr: np.ndarray, plate_qc: dict,
                          plate_map: dict, plate_rows: int = 8,
                          plate_cols: int = 12) -> np.ndarray:
    """
    Generate a clean, readable collage for the HTML report.
    Layout: 8x12 grid of well labels, colour-coded by pass/fail,
    with compound name and cell count. No microscopy images — pure QC grid.
    Each cell is 120x80 px, readable at web resolution.
    """
    CW, CH   = 120, 80          # cell width / height in px
    W        = CW * plate_cols
    H        = CH * plate_rows
    img      = Image.new("RGB", (W, H), color=(10, 12, 22))
    draw     = ImageDraw.Draw(img)

    fwell = _font(15, bold=True)
    fcmpd = _font(10, bold=False)
    fval  = _font(10, bold=False)

    slope_cols = METRIC_COLS["PowerLogLogSlope"]

    for r in range(plate_rows):
        for c in range(plate_cols):
            wl      = well_label(r + 1, c + 1)
            metrics = plate_qc.get(wl, {})
            compound = (plate_map or {}).get(wl, "")
            x0, y0  = c * CW, r * CH
            x1, y1  = x0 + CW - 1, y0 + CH - 1

            # Background: pass/fail by slope
            results = [
                p for col in slope_cols
                if (v := metrics.get(col)) is not None and not np.isnan(v)
                and (p := _passes_absolute(v, "PowerLogLogSlope")) is not None
            ]
            n_pass = sum(results) if results else 0
            if not results:
                bg = (20, 22, 35)
                border = (40, 45, 70)
            elif n_pass == len(results):
                bg     = (15, 42, 20)
                border = (55, 180, 70)
            elif n_pass == 0:
                bg     = (42, 12, 12)
                border = (200, 35, 35)
            else:
                bg     = (42, 32, 10)
                border = (220, 130, 0)

            draw.rectangle([x0, y0, x1, y1], fill=bg, outline=border, width=2)

            # Well label
            draw.text((x0 + 5, y0 + 4), wl, fill=(200, 220, 255), font=fwell)

            # Compound name (truncated)
            if compound and compound.lower() not in ("", "nan", "dmso"):
                disp = compound[:14] + "…" if len(compound) > 14 else compound
                draw.text((x0 + 5, y0 + 22), disp, fill=(160, 200, 140), font=fcmpd)
            elif compound.upper() == "DMSO":
                draw.text((x0 + 5, y0 + 22), "DMSO", fill=(120, 140, 180), font=fcmpd)

            # Cell count
            cells = metrics.get(COUNT_COLS["Cells"])
            if cells is not None and not (isinstance(cells, float) and np.isnan(cells)):
                draw.text((x0 + 5, y0 + 36), f"n={int(cells)}", fill=(160, 175, 200), font=fval)

            # Slope value (DNA channel only for brevity)
            dna_slope_col = f"ImageQuality_PowerLogLogSlope_DNA"
            sv = metrics.get(dna_slope_col)
            if sv is not None and not (isinstance(sv, float) and np.isnan(sv)):
                sc = (75, 215, 95) if -2.5 <= sv <= -1.0 else (255, 65, 65)
                draw.text((x0 + 5, y0 + 50), f"sl={sv:.2f}", fill=sc, font=fval)

    return np.array(img)


def _passes_absolute(value, metric_key: str, channel: str = "") -> bool | None:
    """Standalone absolute-only pass check used by the report collage."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if metric_key == "LocalFocusScore":
        lo, hi = THRESHOLDS_LOCAL_FOCUS.get(channel, (None, None))
    else:
        lo, hi = THRESHOLDS.get(metric_key, (None, None))
    if lo is not None and value < lo: return False
    if hi is not None and value > hi: return False
    return True


def generate_html(cohort_name: str, plates_data: list[dict],
                  output_path: Path, web_scale: float = 0.2) -> None:
    """
    Generate self-contained HTML QC report.

    plates_data: list of dicts, one per plate:
        {name, collage_arr, plate_qc, plate_map, pass_rate,
         n_wells, n_pass, n_illum_pass, n_focus_pass}
    """
    plotly_js = _fetch_plotly_js()

    html_cols = (
        [c for cols in METRIC_COLS.values() for c in cols] +
        list(COUNT_COLS.values()) + [AREA_COL]
    )

    payload = {}
    for pd_ in plates_data:
        pname = pd_["name"]
        pqc   = pd_["plate_qc"]
        pmap  = pd_["plate_map"] or {}

        # Generate readable QC grid for the HTML (not the microscopy collage)
        report_grid = _make_report_collage(pd_["collage_arr"], pqc, pmap)

        payload[pname] = {
            "collage_b64": _collage_to_b64(report_grid, 1.0),   # already small, no extra scale
            "pass_rate":   round(pd_["pass_rate"], 1),
            "n_wells":     pd_["n_wells"],
            "n_pass":      pd_["n_pass"],
            "n_illum":     pd_["n_illum_pass"],
            "n_focus":     pd_["n_focus_pass"],
            "wells": _round_floats({
                well: {
                    "compound": pmap.get(well, ""),
                    **{c: m.get(c) for c in html_cols if c in m}
                }
                for well, m in pqc.items()
            }),
        }

    data_json    = json.dumps(payload)

    # Build per-metric tab specs — all channels for Slope and LocalFocus
    slope_specs = json.dumps([
        {"col": f"ImageQuality_PowerLogLogSlope_{ch}",
         "title": f"Slope — {CHANNEL_LABELS.get(ch, ch)}",
         "cmin": -2.5, "cmax": -1.0, "cs": "RdYlGn"}
        for ch in CHANNELS
    ])
    focus_specs = json.dumps([
        {"col": f"ImageQuality_LocalFocusScore_{ch}_10",
         "title": f"LocalFocus — {CHANNEL_LABELS.get(ch, ch)}",
         "cmin": 0, "cmax": 3, "cs": "Blues"}
        for ch in CHANNELS
    ])
    maxint_specs = json.dumps([
        {"col": f"ImageQuality_MaxIntensity_{ch}",
         "title": f"MaxInt — {CHANNEL_LABELS.get(ch, ch)}",
         "cmin": 0, "cmax": 1, "cs": "RdYlGn_r"}
        for ch in CHANNELS
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{cohort_name} — Cell Painting QC Report</title>
<script>{plotly_js}</script>
<style>
  :root {{
    --bg:     #0d0f1a;
    --panel:  #13162a;
    --border: #1e2540;
    --text:   #c8d8f0;
    --muted:  #6a7a9a;
    --pass:   #4bd760;
    --fail:   #ff4444;
    --warn:   #ffbe00;
    --accent: #3a7bd5;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text);
          font-family: 'Segoe UI', system-ui, sans-serif;
          font-size: 13px; line-height: 1.5; }}
  h1 {{ font-size: 1.5rem; color: #a8c8ff; padding: 24px 32px 4px; }}
  h2 {{ font-size: 1.05rem; color: #8ab0e0; margin-bottom: 12px; letter-spacing: 0.03em; }}
  .subtitle {{ color: var(--muted); padding: 0 32px 20px; font-size: 0.82rem; }}
  .container {{ max-width: 1500px; margin: 0 auto; padding: 0 32px 56px; }}
  .section {{ margin-bottom: 44px; }}

  /* Summary table */
  .summary-table {{ width: 100%; border-collapse: collapse; }}
  .summary-table th, .summary-table td {{
    padding: 8px 14px; border: 1px solid var(--border); text-align: center;
  }}
  .summary-table th {{ background: #1a2040; color: #8ab0e0;
                        font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .summary-table tr:hover {{ background: #161c35; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
            font-size: 0.75rem; font-weight: bold; }}
  .badge-pass {{ background: #1a4020; color: var(--pass); }}
  .badge-warn {{ background: #3a2e00; color: var(--warn); }}
  .badge-fail {{ background: #3a0f0f; color: var(--fail); }}

  /* Plate browser */
  .browser-controls {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
                        background: var(--panel); border: 1px solid var(--border);
                        border-radius: 8px; padding: 14px 18px; margin-bottom: 14px; }}
  .browser-controls label {{ color: var(--muted); font-size: 0.8rem; white-space: nowrap; }}
  #plate-slider {{ flex: 1; min-width: 120px; accent-color: var(--accent); cursor: pointer; }}
  #plate-select {{ background: #1a2040; color: var(--text); border: 1px solid var(--border);
                   border-radius: 4px; padding: 5px 10px; font-size: 0.88rem; cursor: pointer; }}
  .plate-name-display {{ font-size: 1rem; color: #a8c8ff; font-weight: bold;
                          min-width: 110px; text-align: center; }}
  .collage-wrap {{ background: #080a14; border: 1px solid var(--border);
                   border-radius: 8px; padding: 6px; overflow: auto; }}
  .collage-wrap img {{ display: block; border-radius: 4px; max-width: 100%; height: auto; }}

  /* Tabs */
  .tabs {{ display: flex; gap: 2px; flex-wrap: wrap;
            margin-bottom: 0; border-bottom: 2px solid var(--border); }}
  .tab {{ padding: 7px 16px; cursor: pointer; color: var(--muted); font-size: 0.8rem;
           border-radius: 4px 4px 0 0; transition: background 0.12s; white-space: nowrap; }}
  .tab:hover {{ background: #181d35; color: var(--text); }}
  .tab.active {{ background: var(--panel); color: var(--text);
                  border: 1px solid var(--border); border-bottom: 2px solid var(--bg); }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}
  .tab-group-label {{ padding: 7px 12px 7px 6px; color: var(--muted);
                       font-size: 0.72rem; text-transform: uppercase;
                       letter-spacing: 0.08em; align-self: center; white-space: nowrap; }}

  /* Metric channel grid */
  .channel-grid {{ display: grid;
                   grid-template-columns: repeat(3, 1fr);
                   gap: 12px; padding: 12px;
                   background: var(--panel); border: 1px solid var(--border);
                   border-radius: 0 0 8px 8px; }}
  .channel-card {{ background: #0d1020; border: 1px solid var(--border);
                   border-radius: 6px; padding: 4px; }}

  /* Cell counts */
  .counts-grid {{ display: grid;
                  grid-template-columns: repeat(3, 1fr);
                  gap: 16px; }}
  .count-card {{ background: var(--panel); border: 1px solid var(--border);
                 border-radius: 8px; padding: 8px; }}

  /* Compound filter */
  .compound-toolbar {{ display: flex; align-items: center; gap: 10px;
                        flex-wrap: wrap; margin-bottom: 12px; }}
  .compound-toolbar label {{ color: var(--muted); font-size: 0.8rem; white-space: nowrap; }}
  .cmpd-btn {{ padding: 4px 10px; border-radius: 14px; border: 1px solid var(--border);
               background: #1a2040; color: var(--text); font-size: 0.78rem;
               cursor: pointer; transition: all 0.12s; white-space: nowrap; }}
  .cmpd-btn:hover {{ border-color: var(--accent); color: #a8c8ff; }}
  .cmpd-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .cmpd-btn.ctrl {{ background: #20192a; border-color: #3a2550; color: #c090e0; }}
  .cmpd-btn.ctrl:hover {{ background: #2e2040; }}

  /* Flagged wells */
  .flag-controls {{ display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }}
  #flag-filter {{ background: #1a2040; color: var(--text); border: 1px solid var(--border);
                  border-radius: 4px; padding: 6px 10px; width: 320px; font-size: 0.85rem; }}
  #flag-filter::placeholder {{ color: var(--muted); }}
  .flag-table {{ width: 100%; border-collapse: collapse; }}
  .flag-table th, .flag-table td {{
    padding: 7px 10px; border: 1px solid var(--border); text-align: left; font-size: 0.8rem;
  }}
  .flag-table th {{ background: #1a2040; color: #8ab0e0;
                    text-transform: uppercase; font-size: 0.72rem; }}
  .flag-table tr.hidden {{ display: none; }}
  .flag-table tr:hover {{ background: #161c35; }}
</style>
</head>
<body>
<h1>{cohort_name} — Cell Painting QC Report</h1>
<p class="subtitle">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
  &nbsp;·&nbsp; {len(plates_data)} plate(s)
  &nbsp;·&nbsp; Thresholds: absolute + adaptive MAD (3&sigma;)
</p>

<div class="container">

<!-- 1. COHORT SUMMARY -->
<div class="section">
  <h2>Cohort Summary</h2>
  <table class="summary-table">
    <thead>
      <tr><th>Plate</th><th>Wells</th><th>Overall pass</th>
          <th>Illumination</th><th>Focus</th><th>Flagged wells</th></tr>
    </thead>
    <tbody id="summary-tbody"></tbody>
  </table>
</div>

<!-- 2. PLATE BROWSER -->
<div class="section">
  <h2>Plate Browser</h2>
  <div class="browser-controls">
    <label>Plate</label>
    <input type="range" id="plate-slider" min="0" step="1" value="0">
    <span class="plate-name-display" id="plate-name-display"></span>
    <select id="plate-select"></select>
  </div>
  <div class="collage-wrap">
    <img id="collage-img" src="" alt="Plate QC grid">
  </div>
</div>

<!-- 3. QC METRICS -->
<div class="section">
  <h2>QC Metrics</h2>
  <div class="tabs" id="metrics-tabs"></div>
  <div id="metrics-contents"></div>
</div>

<!-- 4. CELL COUNTS -->
<div class="section">
  <h2>Cell Counts</h2>
  <div class="counts-grid" id="counts-grid"></div>
</div>

<!-- 5. FLAGGED WELLS -->
<div class="section">
  <h2>Flagged Wells</h2>
  <div class="flag-controls">
    <input id="flag-filter" type="text" placeholder="Filter by plate, well, compound, or metric…">
  </div>

  <!-- Compound filter buttons -->
  <div class="compound-toolbar" id="compound-toolbar">
    <label>Compounds:</label>
    <button class="cmpd-btn ctrl" id="btn-hide-all">Hide all</button>
    <button class="cmpd-btn ctrl" id="btn-show-all">Show all</button>
  </div>

  <table class="flag-table">
    <thead>
      <tr><th>Plate</th><th>Well</th><th>Compound</th><th>Failing metrics</th></tr>
    </thead>
    <tbody id="flag-tbody"></tbody>
  </table>
</div>

</div><!-- /container -->

<script>
const DATA   = {data_json};
const PLATES = Object.keys(DATA);

// All metric tab specs, grouped — fixed scales per metric across all channels
const SLOPE_SPECS  = {slope_specs};
const FOCUS_SPECS  = {focus_specs};
const MAXINT_SPECS = {maxint_specs};

const METRIC_GROUPS = [
  {{ label: 'Slope', specs: SLOPE_SPECS }},
  {{ label: 'LocalFocus', specs: FOCUS_SPECS }},
  {{ label: 'MaxIntensity', specs: MAXINT_SPECS }},
];

// ── Helpers ───────────────────────────────────────────────────────────────────
function badge(pct) {{
  const cls = pct >= 80 ? 'pass' : pct >= 50 ? 'warn' : 'fail';
  return `<span class="badge badge-${{cls}}">${{pct}}%</span>`;
}}

// ── 1. Summary table ──────────────────────────────────────────────────────────
(function() {{
  const tbody = document.getElementById('summary-tbody');
  PLATES.forEach(p => {{
    const d = DATA[p];
    tbody.insertAdjacentHTML('beforeend', `<tr>
      <td><strong>${{p}}</strong></td>
      <td>${{d.n_wells}}</td>
      <td>${{badge(d.pass_rate)}} ${{d.n_pass}}/${{d.n_wells}}</td>
      <td>${{badge(Math.round(d.n_illum/d.n_wells*100))}} ${{d.n_illum}}/${{d.n_wells}}</td>
      <td>${{badge(Math.round(d.n_focus/d.n_wells*100))}} ${{d.n_focus}}/${{d.n_wells}}</td>
      <td>${{d.n_wells - d.n_pass}}</td>
    </tr>`);
  }});
}})();

// ── 2. Plate browser ──────────────────────────────────────────────────────────
const slider   = document.getElementById('plate-slider');
const pSelect  = document.getElementById('plate-select');
const nameDisp = document.getElementById('plate-name-display');
const img      = document.getElementById('collage-img');

slider.max = PLATES.length - 1;
PLATES.forEach((p, i) =>
  pSelect.insertAdjacentHTML('beforeend', `<option value="${{i}}">${{p}}</option>`)
);

function renderPlate(idx) {{
  const name = PLATES[idx];
  img.src              = `data:image/jpeg;base64,${{DATA[name].collage_b64}}`;
  nameDisp.textContent = name;
  slider.value         = idx;
  pSelect.value        = idx;
  renderMetrics(name);
  renderCounts(name);
}}

slider.oninput  = () => renderPlate(+slider.value);
pSelect.onchange = () => renderPlate(+pSelect.value);

// ── 3. QC Metrics — channel grid with fixed scale per metric ──────────────────
const ROWS = ['A','B','C','D','E','F','G','H'];
const COLS = Array.from({{length:12}}, (_,i) => String(i+1).padStart(2,'0'));

function wellMatrix(plateData, colName) {{
  return ROWS.map(r => COLS.map(c => plateData.wells[r+c]?.[colName] ?? null));
}}

function makeHeatmap(plateData, spec) {{
  const z    = wellMatrix(plateData, spec.col);
  const text = ROWS.map(r => COLS.map(c => {{
    const w = r + c;
    const v = plateData.wells[w]?.[colName = spec.col];
    const cmpd = plateData.wells[w]?.compound || '';
    const label = spec.col.split('_').slice(2, -1).join(' ');
    return `<b>${{w}}</b><br>${{cmpd}}<br>${{label}}: ${{v != null ? v.toFixed(3) : 'N/A'}}`;
  }}));
  return {{
    type: 'heatmap', z, text, hoverinfo: 'text',
    x: COLS, y: ROWS, colorscale: spec.cs,
    zmin: spec.cmin, zmax: spec.cmax,
    colorbar: {{ thickness: 10, len: 0.9, tickfont: {{ size: 9 }} }},
  }};
}}

const tabsEl     = document.getElementById('metrics-tabs');
const contentsEl = document.getElementById('metrics-contents');

function renderMetrics(plateName) {{
  tabsEl.innerHTML     = '';
  contentsEl.innerHTML = '';
  const pd = DATA[plateName];
  let firstGroup = true;

  METRIC_GROUPS.forEach((grp, gi) => {{
    // Group label in tab bar
    tabsEl.insertAdjacentHTML('beforeend',
      `<span class="tab-group-label">${{grp.label}}</span>`);

    grp.specs.forEach((spec, ci) => {{
      const isFirst = firstGroup && ci === 0;
      const tabId   = `tab-g${{gi}}-c${{ci}}`;
      const divId   = `cnt-g${{gi}}-c${{ci}}`;

      tabsEl.insertAdjacentHTML('beforeend',
        `<div class="tab${{isFirst?' active':''}}" data-tab="${{divId}}"
              data-group="${{gi}}">${{spec.title.split('—')[1]?.trim() || spec.title}}</div>`);

      // One shared content div per group (reuse for channel switching)
      if (ci === 0) {{
        contentsEl.insertAdjacentHTML('beforeend',
          `<div class="tab-content${{isFirst?' active':''}}" id="grp-${{gi}}">
             <div class="channel-grid" id="chgrid-${{gi}}"></div>
           </div>`);
      }}

      // Render all channels for this group as a grid
      const gridId = `chgrid-${{gi}}`;
      if (ci === 0) {{
        // Populate grid after DOM is inserted
        setTimeout(() => {{
          const grid = document.getElementById(gridId);
          if (!grid) return;
          grp.specs.forEach((s, si) => {{
            const cid = `ch-${{gi}}-${{si}}`;
            const card = document.createElement('div');
            card.className = 'channel-card';
            card.innerHTML = `<div id="${{cid}}"></div>`;
            grid.appendChild(card);
            const layout = {{
              paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
              font: {{ color:'#c8d8f0', size:10 }},
              margin: {{ t:28, b:36, l:36, r:16 }},
              height: 240,
              title: {{ text: s.title, font: {{ size:11, color:'#8ab0e0' }}, x:0.5 }},
              xaxis: {{ tickfont:{{size:9}} }},
              yaxis: {{ tickfont:{{size:9}} }},
            }};
            Plotly.react(cid, [makeHeatmap(pd, s)], layout,
                         {{responsive:true, displayModeBar:false}});
          }});
        }}, 0);
      }}
      if (isFirst) firstGroup = false;
    }});
  }});

  // Tab group switching — show the right group content div
  tabsEl.querySelectorAll('.tab').forEach(tab => {{
    tab.onclick = () => {{
      const gi = tab.dataset.group;
      tabsEl.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      contentsEl.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      // Activate all tabs of this group
      tabsEl.querySelectorAll(`.tab[data-group="${{gi}}"]`).forEach(t => t.classList.add('active'));
      document.getElementById(`grp-${{gi}}`).classList.add('active');
    }};
  }});
}}

// ── 4. Cell counts — larger grid, full labels ─────────────────────────────────
const COUNT_SPECS = [
  {{ col:'Count_Cells',            title:'Cells',           cs:'Viridis' }},
  {{ col:'Count_Nuclei',           title:'Nuclei (filtered)',cs:'Viridis' }},
  {{ col:'Count_Illum_artifacts',  title:'Illum. Artifacts',cs:'YlOrRd'  }},
];

function renderCounts(plateName) {{
  const grid = document.getElementById('counts-grid');
  grid.innerHTML = '';
  const pd = DATA[plateName];

  COUNT_SPECS.forEach(spec => {{
    const cid  = `cnt-${{spec.col.replace(/[^a-z0-9]/gi,'_')}}`;
    const card = document.createElement('div');
    card.className = 'count-card';
    card.innerHTML = `<div id="${{cid}}"></div>`;
    grid.appendChild(card);

    const z    = wellMatrix(pd, spec.col);
    const text = ROWS.map(r => COLS.map(c => {{
      const w = r + c;
      const v = pd.wells[w]?.[spec.col];
      const cmpd = pd.wells[w]?.compound || '';
      return `<b>${{w}}</b><br>${{cmpd}}<br>${{spec.title}}: ${{v != null ? Math.round(v) : 'N/A'}}`;
    }}));

    const layout = {{
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
      font: {{ color:'#c8d8f0', size:11 }},
      margin: {{ t:40, b:50, l:50, r:20 }},
      height: 340,
      title: {{ text: spec.title, font: {{ size:13, color:'#a8c8ff' }}, x:0.5 }},
      xaxis: {{ title:'Column', tickfont:{{size:10}}, tickvals: COLS,
                ticktext: COLS.map(c => parseInt(c)) }},
      yaxis: {{ title:'Row', tickfont:{{size:10}} }},
    }};
    Plotly.react(cid,
      [{{ type:'heatmap', z, text, hoverinfo:'text',
          x:COLS, y:ROWS, colorscale:spec.cs,
          colorbar:{{ thickness:14, len:0.85, tickfont:{{size:10}} }} }}],
      layout, {{responsive:true, displayModeBar:false}});
  }});
}}

// ── 5. Flagged wells + compound filter buttons ────────────────────────────────
const ABSOLUTE_CHECKS = [
  {{ cols: {json.dumps(METRIC_COLS["PowerLogLogSlope"])}, lo: -2.5, hi: -1.0, label: 'Slope' }},
  {{ cols: {json.dumps(METRIC_COLS["MaxIntensity"])},     lo: null,  hi: 0.95, label: 'MaxInt' }},
  {{ cols: {json.dumps(METRIC_COLS["FocusScore"])},       lo: 0.005, hi: null, label: 'Focus' }},
];

let activeCompounds = new Set();   // empty = show all
let hideAllMode     = false;

(function buildFlagTable() {{
  const tbody   = document.getElementById('flag-tbody');
  const toolbar = document.getElementById('compound-toolbar');
  const allCmpds = new Set();

  PLATES.forEach(p => {{
    const pd = DATA[p];
    Object.entries(pd.wells).forEach(([well, m]) => {{
      const failing = [];
      ABSOLUTE_CHECKS.forEach(chk => {{
        chk.cols.forEach(col => {{
          const v = m[col];
          if (v == null) return;
          const lo_fail = chk.lo != null && v < chk.lo;
          const hi_fail = chk.hi != null && v > chk.hi;
          if (lo_fail || hi_fail) {{
            const ch = col.split('_').pop();
            failing.push(`${{chk.label}}/${{ch}}`);
          }}
        }});
      }});

      const cmpd = m.compound || '';
      if (cmpd) allCmpds.add(cmpd);

      if (failing.length > 0) {{
        tbody.insertAdjacentHTML('beforeend',
          `<tr data-plate="${{p}}" data-well="${{well}}"
               data-compound="${{cmpd}}" data-metrics="${{failing.join(',')}}">
            <td>${{p}}</td><td>${{well}}</td>
            <td>${{cmpd || '—'}}</td>
            <td>${{[...new Set(failing)].join(', ')}}</td>
          </tr>`);
      }}
    }});
  }});

  // Build compound buttons
  [...allCmpds].sort().forEach(cmpd => {{
    const btn = document.createElement('button');
    btn.className    = 'cmpd-btn';
    btn.textContent  = cmpd;
    btn.dataset.cmpd = cmpd;
    btn.onclick = () => {{
      hideAllMode = false;
      btn.classList.toggle('active');
      if (btn.classList.contains('active')) activeCompounds.add(cmpd);
      else activeCompounds.delete(cmpd);
      applyFilters();
    }};
    toolbar.appendChild(btn);
  }});

  document.getElementById('btn-hide-all').onclick = () => {{
    hideAllMode = true;
    activeCompounds.clear();
    toolbar.querySelectorAll('.cmpd-btn:not(.ctrl)').forEach(b => b.classList.remove('active'));
    applyFilters();
  }};
  document.getElementById('btn-show-all').onclick = () => {{
    hideAllMode = false;
    activeCompounds.clear();
    toolbar.querySelectorAll('.cmpd-btn:not(.ctrl)').forEach(b => b.classList.remove('active'));
    applyFilters();
  }};
}})();

function applyFilters() {{
  const q = document.getElementById('flag-filter').value.toLowerCase();
  document.querySelectorAll('#flag-tbody tr').forEach(tr => {{
    const text = [tr.dataset.plate, tr.dataset.well,
                  tr.dataset.compound, tr.dataset.metrics].join(' ').toLowerCase();
    const cmpd  = tr.dataset.compound;
    const textOk  = !q || text.includes(q);
    let   cmpdOk;
    if (hideAllMode) {{
      cmpdOk = false;
    }} else if (activeCompounds.size === 0) {{
      cmpdOk = true;   // no filter active → show all
    }} else {{
      cmpdOk = activeCompounds.has(cmpd);
    }}
    tr.classList.toggle('hidden', !(textOk && cmpdOk));
  }});
}}

document.getElementById('flag-filter').oninput = applyFilters;

// ── Init ──────────────────────────────────────────────────────────────────────
renderPlate(0);
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    size_mb = output_path.stat().st_size / 1e6
    print(f"[html] → {output_path}  ({size_mb:.1f} MB)")


# ── Collage class ──────────────────────────────────────────────────────────────

class Collage:
    def __init__(self, input_dir, output_path,
                 qc_tsv=None, platemap=None,
                 cohort_name: str = "Cohort",
                 plate_rows: int = 8, plate_cols: int = 12,
                 sites_per_well: int = 9, scale: float = 0.5,
                 workers: int = 8, band_height: int = 280,
                 font_size: int = 18, n_sigma: float = 3.0,
                 jpeg_quality: int = 80, web_scale: float = 0.2):

        self.input_dir    = Path(input_dir)
        self.output_path  = Path(output_path)
        self.cohort_name  = cohort_name
        self.plate_rows   = plate_rows
        self.plate_cols   = plate_cols
        self.sites_per_well = sites_per_well
        self.scale        = scale
        self.workers      = workers
        self.band_height  = band_height
        self.font_size    = font_size
        self.jpeg_quality = jpeg_quality
        self.web_scale    = web_scale
        self.engine       = ThresholdEngine(n_sigma=n_sigma)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Auto-detect QC TSV
        self.qc = load_qc_tsv(qc_tsv)
        if not self.qc:
            for name in ("Image.txt", "image.txt", "Image.tsv", "image.tsv"):
                p = self.input_dir / name
                if p.exists():
                    print(f"[qc] Auto-detected: {p}")
                    self.qc = load_qc_tsv(p)
                    break

        # Auto-detect platemap — explicit path or first platemap_*.csv in input dir
        self.platemap = load_platemap(platemap)
        if not self.platemap:
            candidates = sorted(self.input_dir.glob("platemap_*.csv"))
            if candidates:
                print(f"[platemap] Auto-detected: {candidates[0]}")
                self.platemap = load_platemap(candidates[0])

        # Process plates and accumulate HTML data
        self._html_plates: list[dict] = []
        for plate_name, files in sorted(self._group_by_plate().items()):
            print(f"\n[plate] {plate_name}")
            wells    = self._group_well_imgs(files)
            montages = self._build_montages_parallel(wells)
            self._render_plate(plate_name, montages)

        # Generate HTML after all plates are processed
        html_name = f"{self.cohort_name}_QC_report.html"
        generate_html(
            cohort_name  = self.cohort_name,
            plates_data  = self._html_plates,
            output_path  = self.output_path / html_name,
            web_scale    = self.web_scale,
        )

    def _group_by_plate(self) -> dict:
        plates = defaultdict(list)
        for f in self.input_dir.glob("*.tiff"):
            m = re.search(r"_(P\d+)", f.name)
            if m:
                plates[m.group(1)].append(f)
        return plates

    def _group_well_imgs(self, files: list) -> dict:
        wells = defaultdict(dict)
        for f in files:
            r, c, s = parse_name(f.name)
            wells[(r, c)][s] = f
        return wells

    def _build_montages_parallel(self, wells: dict) -> dict:
        montages = {}
        args     = [((r, c), sites, self.scale, self.sites_per_well)
                    for (r, c), sites in wells.items()]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = {ex.submit(build_well_montage, a): a[0] for a in args}
            for fut in as_completed(futures):
                (r, c), mont = fut.result()
                if mont is not None:
                    montages[(r, c)] = mont
        return montages

    def _render_plate(self, plate_name: str, montages: dict) -> None:
        if not montages:
            print(f"  No images for {plate_name}, skipping.")
            return

        plate_qc  = (self.qc.get(plate_name)
                     or self.qc.get(plate_name.lstrip("P"))
                     or (next(iter(self.qc.values())) if len(self.qc) == 1 else {}))
        plate_map = (self.platemap.get(plate_name)
                     or self.platemap.get(plate_name.lstrip("P"))
                     or (next(iter(self.platemap.values())) if len(self.platemap) == 1 else {}))

        # Fit adaptive thresholds for this plate
        if plate_qc:
            self.engine.fit(plate_qc)

        tile_h, tile_w = next(iter(montages.values())).shape[:2]
        blank          = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        rows_imgs      = []

        for r in range(1, self.plate_rows + 1):
            row_labels = [well_label(r, c) for c in range(1, self.plate_cols + 1)]
            row_tiles  = [
                _make_tile(montages.get((r, c), blank),
                           _slope_to_rgb(plate_qc.get(well_label(r, c)), self.engine))
                for c in range(1, self.plate_cols + 1)
            ]
            row_img = np.concatenate(row_tiles, axis=1)
            rows_imgs.append(row_img)
            rows_imgs.append(_make_band(
                width              = row_img.shape[1],
                well_labels_in_row = row_labels,
                plate_qc           = plate_qc,
                band_height        = self.band_height,
                font_size          = self.font_size,
                tile_width         = tile_w,
                plate_map          = plate_map,
                engine             = self.engine,
            ))

        collage  = np.concatenate(rows_imgs, axis=0)
        s_min, s_max = _slope_range(plate_qc)
        header   = make_header(
            collage.shape[1],
            title=f"{plate_name}  ·  Cell Painting QC  "
                  f"|  Slope range: {s_min:.2f} – {s_max:.2f}  "
                  f"|  Thresholds: absolute + {self.engine.n_sigma}σ MAD")
        collage  = np.concatenate([header, collage], axis=0)

        if plate_qc:
            footer  = make_report_footer(collage.shape[1], plate_name, plate_qc,
                                          self.engine, self.font_size, plate_map)
            collage = np.concatenate([collage, footer], axis=0)

        # High-res JPEG
        out_jpg = self.output_path / f"{plate_name}_QC.jpg"
        Image.fromarray(collage).save(str(out_jpg), format="JPEG",
                                      quality=self.jpeg_quality,
                                      optimize=True, subsampling=2)
        jpg_mb = out_jpg.stat().st_size / 1e6
        print(f"  → {out_jpg}  ({collage.shape[1]}×{collage.shape[0]} px, "
              f"JPEG q={self.jpeg_quality})  {jpg_mb:.1f} MB")

        # Accumulate HTML data
        n_wells     = len(plate_qc)
        n_pass      = sum(1 for m in plate_qc.values() if _well_passes_all(m, self.engine))
        n_illum     = sum(1 for m in plate_qc.values() if _well_passes_group(m, ILLUM_METRICS, self.engine))
        n_focus     = sum(1 for m in plate_qc.values() if _well_passes_group(m, FOCUS_METRICS, self.engine))
        self._html_plates.append({
            "name":        plate_name,
            "collage_arr": collage,
            "plate_qc":    plate_qc,
            "plate_map":   plate_map,
            "pass_rate":   100 * n_pass / n_wells if n_wells else 0,
            "n_wells":     n_wells,
            "n_pass":      n_pass,
            "n_illum_pass": n_illum,
            "n_focus_pass": n_focus,
        })


def parse_name(fname: str) -> tuple:
    name     = fname.replace(".tiff", "")
    rc, site, _ = name.split("-")
    return int(rc[:3]), int(rc[3:6]), int(site)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Cell Painting QC collage + HTML report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i",  "--input",       required=True)
    p.add_argument("-o",  "--output",      required=True)
    p.add_argument("--cohort",             default="Cohort",
                   help="Cohort name — used in HTML title and filename.")
    p.add_argument("--qc",                 default=None)
    p.add_argument("--platemap",           default=None)
    p.add_argument("--rows",               type=int,   default=8)
    p.add_argument("--cols",               type=int,   default=12)
    p.add_argument("--sites",              type=int,   default=9)
    p.add_argument("--scale",              type=float, default=0.5)
    p.add_argument("--workers",            type=int,   default=8)
    p.add_argument("--band-height",        type=int,   default=280)
    p.add_argument("--font",               type=int,   default=18)
    p.add_argument("--n-sigma",            type=float, default=3.0,
                   help="MAD-sigma for adaptive outlier detection.")
    p.add_argument("--jpeg-quality",       type=int,   default=80)
    p.add_argument("--web-scale",          type=float, default=0.2,
                   help="Scale factor for collage thumbnails in HTML.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Collage(
        input_dir      = args.input,
        output_path    = args.output,
        cohort_name    = args.cohort,
        qc_tsv         = args.qc,
        platemap       = args.platemap,
        plate_rows     = args.rows,
        plate_cols     = args.cols,
        sites_per_well = args.sites,
        scale          = args.scale,
        workers        = args.workers,
        band_height    = args.band_height,
        font_size      = args.font,
        n_sigma        = args.n_sigma,
        jpeg_quality   = args.jpeg_quality,
        web_scale      = args.web_scale,
    )
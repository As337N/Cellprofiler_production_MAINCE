"""
III_QC_collage.py
=================
Cell Painting plate collage builder + interactive HTML QC report.

Outputs
-------
  <cohort>_QC_report.html Self-contained interactive report (Plotly embedded)
                          Includes: plate overview grid, well montages for flagged
                          wells, QC heatmaps (all channels), cell count maps.

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
FOCUS_METRICS  = ["FocusScore", "FocusScore"]
BORDER_METRIC  = "PowerLogLogSlope"
METRIC_LABELS  = {
    "PowerLogLogSlope": "Slope", "MaxIntensity": "MaxInt",
    "FocusScore": "Focus",       "FocusScore": "LocalFoc",
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
    if metric == "FocusScore":
        return f"ImageQuality_FocusScore_{channel}"
    return f"ImageQuality_{metric}_{channel}"


METRIC_COLS: dict[str, list[str]] = {
    mk: [_miq_col(mk, ch) for ch in CHANNELS]
    for mk in ("PowerLogLogSlope", "MaxIntensity", "FocusScore", "FocusScore")
}
for mk in ("FocusScore", "FocusScore"):
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
    FocusScore   per-channel    Main focus metric; thresholds vary by signal density.

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
        if metric_key == "FocusScore":
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
    """Load CellProfiler Image.txt TSV -> {plate: {well: {col: value}}}."""
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
    """Load platemap CSV -> {plate: {well: compound}}."""
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

    print(f"[platemap] {sum(len(v) for v in result.values())} well->compound mappings.")
    return result


# ── MFI channel definitions ───────────────────────────────────────────────────
# Hoechst (DNA) → Nuclei.txt  |  Syto, ER, Golgi, Mito → Cells.txt
# Column format: Intensity_MeanIntensity_<Channel>

MFI_COLS_CELLS = {
    "Syto":  ("Intensity_MeanIntensity_Syto",  "#50DC78"),
    "ER":    ("Intensity_MeanIntensity_ER",     "#B45AFF"),
    "Golgi": ("Intensity_MeanIntensity_Golgi",  "#FFB43C"),
    "Mito":  ("Intensity_MeanIntensity_Mito",   "#FF5050"),
}
MFI_COLS_NUCLEI = {
    "Hoechst": ("Intensity_MeanIntensity_Hoechst", "#64A0FF"),
}
MFI_CHANNEL_ORDER = ["Hoechst", "Syto", "ER", "Golgi", "Mito"]


def _load_object_tsv(tsv_path: Path, channel_map: dict, label: str) -> "pd.DataFrame | None":
    """Load one CellProfiler object TSV (Cells.txt or Nuclei.txt)."""
    if not tsv_path.exists():
        print(f"[mfi] {label} not found: {tsv_path}")
        return None
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    print(f"[mfi] Loaded {len(df):,} rows from {tsv_path.name}  ({label})")
    if "Metadata_Well" not in df.columns:
        print(f"[mfi] Metadata_Well missing in {tsv_path.name} — skipping.")
        return None
    df["Metadata_Well"]  = df["Metadata_Well"].astype(str).str.strip().str.upper()
    df["Metadata_Plate"] = (df["Metadata_Plate"].astype(str).str.strip()
                            if "Metadata_Plate" in df.columns else "Plate")
    mfi_cols = [col for col, _ in channel_map.values() if col in df.columns]
    if not mfi_cols:
        print(f"[mfi] No MFI columns found in {tsv_path.name} — skipping.")
        return None
    keep = (["Metadata_Plate", "Metadata_Well"]
            + (["ImageNumber"] if "ImageNumber" in df.columns else [])
            + mfi_cols)
    return df[keep].copy()


def load_mfi_data(cells_path: "Path | None",
                  nuclei_path: "Path | None") -> "tuple[dict, dict]":
    """
    Load Cells.txt and Nuclei.txt and return (source_dfs, channel_map).
    source_dfs : { channel_label: DataFrame }
    channel_map: { channel_label: (col_name, hex_colour) }  ordered by MFI_CHANNEL_ORDER
    """
    source_dfs:  dict = {}
    channel_map: dict = {}

    if cells_path is not None:
        df_cells = _load_object_tsv(cells_path, MFI_COLS_CELLS, "Cells")
        if df_cells is not None:
            for ch, (col, color) in MFI_COLS_CELLS.items():
                if col in df_cells.columns:
                    keep = (["Metadata_Plate", "Metadata_Well"]
                            + (["ImageNumber"] if "ImageNumber" in df_cells.columns else [])
                            + [col])
                    source_dfs[ch]  = df_cells[keep].copy()
                    channel_map[ch] = (col, color)

    if nuclei_path is not None:
        df_nuclei = _load_object_tsv(nuclei_path, MFI_COLS_NUCLEI, "Nuclei")
        if df_nuclei is not None:
            for ch, (col, color) in MFI_COLS_NUCLEI.items():
                if col in df_nuclei.columns:
                    keep = (["Metadata_Plate", "Metadata_Well"]
                            + (["ImageNumber"] if "ImageNumber" in df_nuclei.columns else [])
                            + [col])
                    source_dfs[ch]  = df_nuclei[keep].copy()
                    channel_map[ch] = (col, color)

    if not channel_map:
        print("[mfi] No usable MFI data found.")

    ordered = {ch: channel_map[ch] for ch in MFI_CHANNEL_ORDER if ch in channel_map}
    return source_dfs, ordered


def _aggregate_mfi_per_well(source_dfs: dict,
                             channel_map: dict,
                             plate_name: str) -> "dict[str, dict[str, list]]":
    """
    Aggregate per-object MFI → { well: { channel: [per_image_median, ...] } }.
    Groups objects by ImageNumber first (one value per image/site), then collects
    those per-image medians into a list per well — what the boxplots display.
    """
    result: dict = defaultdict(lambda: defaultdict(list))
    for ch, (col, _) in channel_map.items():
        df = source_dfs.get(ch)
        if df is None:
            continue
        mask     = df["Metadata_Plate"].str.strip() == plate_name.strip()
        plate_df = df[mask].copy()
        if plate_df.empty:
            if df["Metadata_Plate"].nunique() == 1:
                plate_df = df.copy()
            else:
                continue
        if "ImageNumber" in plate_df.columns:
            per_image = (plate_df
                         .groupby(["Metadata_Plate", "Metadata_Well", "ImageNumber"])[col]
                         .median()
                         .reset_index())
        else:
            per_image = plate_df
        for _, row in per_image.iterrows():
            well = str(row["Metadata_Well"]).strip().upper()
            v    = row[col]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                result[well][ch].append(float(v))
    return {well: dict(ch_vals) for well, ch_vals in result.items()}


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
        line(f"  Focus:         {n_focus_p}/{n_wells} pass ({pct(n_focus_p)})  [FocusScore + FocusScore]", fc_, fb)
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

                    if mk == "FocusScore":
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


def _collage_to_b64(collage_arr: np.ndarray, web_scale: float = 1.0,
                    quality: int = 72) -> str:
    """Resize array and return as base64 JPEG string."""
    h, w = collage_arr.shape[:2]
    if web_scale != 1.0:
        pil = Image.fromarray(collage_arr).resize(
            (max(1, int(w * web_scale)), max(1, int(h * web_scale))),
            Image.LANCZOS)
    else:
        pil = Image.fromarray(collage_arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality,
             optimize=True, progressive=True, subsampling=2)
    return base64.b64encode(buf.getvalue()).decode()


def _well_montage_b64(montages: dict, well_key: tuple,
                      scale_factor: float = 0.30) -> str | None:
    """
    Build a web-ready b64 JPEG for a single well montage (already assembled
    as a 3x3 site grid at self.scale). scale_factor shrinks it further.
    Returns None if the well has no montage.
    """
    mont = montages.get(well_key)
    if mont is None:
        return None
    h, w   = mont.shape[:2]
    target = (max(1, int(w * scale_factor)), max(1, int(h * scale_factor)))
    pil    = Image.fromarray(mont if mont.ndim == 3
                             else np.stack([mont] * 3, axis=-1))
    pil    = pil.resize(target, Image.LANCZOS)
    buf    = io.BytesIO()
    pil.save(buf, format="JPEG", quality=75, optimize=True, subsampling=2)
    return base64.b64encode(buf.getvalue()).decode()


def _make_overview_grid(montages: dict, plate_qc: dict, plate_map: dict,
                        plate_rows: int = 8, plate_cols: int = 12,
                        well_px: int = 90) -> tuple:
    """
    Build a plate overview image: real microscopy thumbnails in an 8x12 grid.
    Each well thumbnail is well_px x well_px.
    Returns (image_array, cell_w, cell_h) — cell dimensions needed for SVG overlay.
    well_px=90 -> ~1080x720 px total.
    """
    CW = CH = well_px
    W  = CW * plate_cols
    H  = CH * plate_rows

    canvas = Image.new("RGB", (W, H), color=(8, 10, 20))
    fdraw  = ImageDraw.Draw(canvas)
    flabel = _font(9, bold=True)
    slope_cols = METRIC_COLS["PowerLogLogSlope"]

    for r in range(plate_rows):
        for c in range(plate_cols):
            wl      = well_label(r + 1, c + 1)
            metrics = plate_qc.get(wl, {})
            x0, y0  = c * CW, r * CH

            mont = montages.get((r + 1, c + 1))
            if mont is not None:
                thumb = Image.fromarray(
                    mont if mont.ndim == 3 else np.stack([mont] * 3, axis=-1)
                ).resize((CW, CH), Image.LANCZOS)
                canvas.paste(thumb, (x0, y0))
            else:
                fdraw.rectangle([x0, y0, x0 + CW - 1, y0 + CH - 1],
                                fill=(15, 18, 30))

            # Border colour by slope pass/fail
            results = [
                p for col in slope_cols
                if (v := metrics.get(col)) is not None
                and not (isinstance(v, float) and np.isnan(v))
                and (p := _passes_absolute(v, "PowerLogLogSlope")) is not None
            ]
            n_pass = sum(results) if results else -1
            border = ((40, 45, 70)   if n_pass < 0 else
                      (55, 200, 70)  if n_pass == len(results) else
                      (210, 35, 35)  if n_pass == 0 else
                      (220, 130, 0))
            fdraw.rectangle([x0, y0, x0 + CW - 1, y0 + CH - 1],
                            outline=border, width=2)
            fdraw.text((x0 + 2, y0 + 1), wl, fill=(220, 230, 255), font=flabel)

    return np.array(canvas), CW, CH


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
    if metric_key == "FocusScore":
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
         n_wells, n_pass, n_illum_pass, n_focus_pass, mfi_data}
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
        adp   = pd_.get("engine_adaptive", {})

        # QC summary grid (text-based, always included)
        report_grid = _make_report_collage(pd_["collage_arr"], pqc, pmap)

        # Per-well flag classification: absolute vs adaptive
        well_flags = {}
        for well, m in pqc.items():
            abs_fails, adp_fails = [], []
            for mk, cols in METRIC_COLS.items():
                for col in cols:
                    v  = m.get(col)
                    ch = COL_TO_CHANNEL.get(col, "")
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        continue
                    # Absolute
                    if mk == "FocusScore":
                        lo, hi = THRESHOLDS_LOCAL_FOCUS.get(ch, (None, None))
                    else:
                        lo, hi = THRESHOLDS.get(mk, (None, None))
                    abs_fail = (lo is not None and v < lo) or (hi is not None and v > hi)
                    # Adaptive
                    adp_col = adp.get(mk, {}).get(col)
                    adp_fail = False
                    if adp_col:
                        adp_lo, adp_hi = adp_col
                        adp_fail = (v < adp_lo and lo is not None and v < lo) or                                    (v > adp_hi and hi is not None and v > hi)
                    ch_lbl = CHANNEL_LABELS.get(ch, ch)
                    met_lbl = METRIC_LABELS.get(mk, mk)
                    tag = f"{met_lbl}/{ch_lbl}"
                    if abs_fail:
                        abs_fails.append(tag)
                    elif adp_fail:
                        adp_fails.append(tag)
            if abs_fails or adp_fails:
                well_flags[well] = {"abs": abs_fails, "adp": adp_fails}

        payload[pname] = {
            "collage_b64":  _collage_to_b64(report_grid, 1.0),
            "overview_b64": _collage_to_b64(pd_["overview_arr"], 1.0, quality=75),
            "overview_cw":  pd_["overview_cw"],
            "overview_ch":  pd_["overview_ch"],
            "flagged_b64":  pd_.get("flagged_b64", {}),
            "well_flags":   well_flags,
            "pass_rate":    round(pd_["pass_rate"], 1),
            "n_wells":      pd_["n_wells"],
            "n_pass":       pd_["n_pass"],
            "n_illum":      pd_["n_illum_pass"],
            "n_focus":      pd_["n_focus_pass"],
            "mfi":          pd_.get("mfi_data", {}),
            "wells": _round_floats({
                well: {
                    "compound": pmap.get(well, ""),
                    **{c: m.get(c) for c in html_cols if c in m}
                }
                for well, m in pqc.items()
            }),
        }

    data_json = json.dumps(payload)

    # ── Compute cohort-wide scale for every metric column ─────────────────────
    # Uses p2/p98 percentiles so extreme outliers don't compress the scale.
    # Falls back to absolute threshold bounds if insufficient data.
    def _cohort_range(col: str, fallback_lo: float, fallback_hi: float,
                      plo: float = 2, phi: float = 98) -> tuple:
        vals = [
            m for pd_ in plates_data
            for m in pd_["plate_qc"].values()
            if (v := m.get(col)) is not None
            and not (isinstance(v, float) and np.isnan(v))
            for m in [v]
        ]
        if len(vals) < 10:
            return fallback_lo, fallback_hi
        lo = float(np.percentile(vals, plo))
        hi = float(np.percentile(vals, phi))
        # Never exceed the absolute threshold bounds (clip outward slightly)
        lo = min(lo, fallback_lo) if fallback_lo is not None else lo
        hi = max(hi, fallback_hi) if fallback_hi is not None else hi
        return round(lo, 3), round(hi, 3)

    # RdBu: red=bad(low) white=borderline blue=good(high) for Slope/Focus
    # RdBu_r: red=bad(high) white=borderline blue=good(low) for MaxInt
    slope_specs = json.dumps([
        {"col": col,
         "title": f"Slope — {CHANNEL_LABELS.get(ch, ch)}",
         "cmin": _cohort_range(col, -2.5, -1.0)[0],
         "cmax": _cohort_range(col, -2.5, -1.0)[1],
         "cs": "RdBu"}
        for ch in CHANNELS
        for col in [f"ImageQuality_PowerLogLogSlope_{ch}"]
    ])
    focus_specs = json.dumps([
        {"col": col,
         "title": f"LocalFocus — {CHANNEL_LABELS.get(ch, ch)}",
         "cmin": _cohort_range(col, 0, None)[0],
         "cmax": _cohort_range(col, 0, None)[1],
         "cs": "RdBu"}
        for ch in CHANNELS
        for col in [f"ImageQuality_FocusScore_{ch}"] 
    ])
    maxint_specs = json.dumps([
        {"col": col,
         "title": f"MaxInt — {CHANNEL_LABELS.get(ch, ch)}",
         "cmin": _cohort_range(col, 0, 1)[0],
         "cmax": _cohort_range(col, 0, 1)[1],
         "cs": "RdBu_r"}
        for ch in CHANNELS
        for col in [f"ImageQuality_MaxIntensity_{ch}"]
    ])

    # Count columns: cohort-wide p2/p98 so all plates share the same colour scale
    count_ranges_json = json.dumps({
        col: list(_cohort_range(col, 0, None))
        for col in ["Count_Cells", "Count_Nuclei", "Count_Illum_artifacts"]
    })

    # ── MFI data: build per-plate { channel: { well: [vals] } } ─────────────────
    # mfi_data in plates_data is now { well: { channel: [per_image_medians] } }
    # Restructure to { plate: { channel: { well: [vals] } } } for JS injection
    mfi_payload: dict = {}
    for pd_ in plates_data:
        pname    = pd_["name"]
        mfi_raw  = pd_.get("mfi_data", {})   # { well: { ch: [vals] } }
        by_ch: dict = {}
        for well, ch_vals in mfi_raw.items():
            for ch, vals in ch_vals.items():
                by_ch.setdefault(ch, {})[well] = vals
        mfi_payload[pname] = by_ch

    # Ordered channel list and colours from MFI_COLS_*
    _all_mfi_channels: list = [ch for ch in MFI_CHANNEL_ORDER
                                if any(ch in by_ch for by_ch in mfi_payload.values())]
    _mfi_colors: dict = {**{ch: c for ch, (_, c) in MFI_COLS_NUCLEI.items()},
                          **{ch: c for ch, (_, c) in MFI_COLS_CELLS.items()}}

    mfi_payload_json  = json.dumps(mfi_payload)
    mfi_channels_json = json.dumps(_all_mfi_channels)
    mfi_colors_json   = json.dumps({ch: _mfi_colors.get(ch, "#8ab0d0")
                                    for ch in _all_mfi_channels})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{cohort_name} — Cell Painting QC Report</title>
<script>{plotly_js}</script>
<style>
  :root {{
    --bg:     #0d0f1a; --panel: #13162a; --border: #1e2540;
    --text:   #c8d8f0; --muted: #6a7a9a;
    --pass:   #4bd760; --fail:  #ff4444; --warn: #ffbe00; --accent: #3a7bd5;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text);
          font-family: 'Segoe UI', system-ui, sans-serif; font-size: 13px; line-height: 1.5; }}
  h1 {{ font-size: 1.5rem; color: #a8c8ff; padding: 24px 32px 4px; }}
  h2 {{ font-size: 1.05rem; color: #8ab0e0; margin-bottom: 6px; letter-spacing: 0.03em; }}
  .subtitle {{ color: var(--muted); padding: 0 32px 20px; font-size: 0.82rem; }}
  .container {{ max-width: 1600px; margin: 0 auto; padding: 0 32px 56px; }}
  .section {{ margin-bottom: 44px; }}

  /* ── Feature audit label ── */
  .feature-audit {{
    background: linear-gradient(135deg, #0d1830 0%, #111d38 100%);
    border: 1px solid #2a4070;
    border-left: 3px solid #3a7bd5;
    border-radius: 6px; padding: 10px 16px; margin-bottom: 16px;
    font-size: 0.77rem; color: #8ab4d8; line-height: 1.8;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }}
  .feature-audit .audit-row {{
    display: flex; align-items: baseline; gap: 6px; flex-wrap: wrap;
  }}
  .feature-audit .audit-label {{
    color: #a8c8f0; font-weight: 600; font-size: 0.78rem;
    min-width: 90px; flex-shrink: 0;
  }}
  .feature-audit .audit-source {{
    color: #4a8fd8; font-weight: 600; font-size: 0.75rem;
  }}
  .feature-audit .thresh {{
    color: #7ac0f0; font-family: monospace; font-size: 0.72rem;
    background: rgba(58,123,213,0.12); border-radius: 3px;
    padding: 0 4px;
  }}
  .feature-audit .thresh-val {{
    color: #f0c060; font-family: monospace; font-size: 0.72rem;
    background: rgba(200,140,0,0.12); border-radius: 3px;
    padding: 0 4px;
  }}
  .feature-audit .audit-sep {{
    color: #2a4070; margin: 4px 0;
    border: none; border-top: 1px solid #1e3055;
  }}

  /* Summary */
  .summary-table {{ width: 100%; border-collapse: collapse; }}
  .summary-table th, .summary-table td {{ padding: 8px 14px; border: 1px solid var(--border); text-align: center; }}
  .summary-table th {{ background: #1a2040; color: #8ab0e0; font-size: 0.75rem; text-transform: uppercase; }}
  .summary-table tr:hover {{ background: #161c35; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: bold; }}
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
  .plate-name-display {{ font-size: 1rem; color: #a8c8ff; font-weight: bold; min-width: 110px; text-align: center; }}

  /* Well detail panel */
  .browser-body {{ display: grid; grid-template-columns: 260px 1fr; gap: 14px; margin-bottom: 14px; }}
  .well-info-panel {{ background: var(--panel); border: 1px solid var(--border);
                      border-radius: 8px; padding: 14px; overflow-y: auto; max-height: 480px; }}
  .well-info-panel h3 {{ font-size: 0.95rem; color: #a8c8ff; margin-bottom: 8px; }}
  .well-compound {{ font-size: 0.82rem; color: #90c870; margin-bottom: 10px; }}
  .metric-row {{ display: flex; justify-content: space-between; align-items: center;
                 padding: 3px 0; border-bottom: 1px solid #1a1f35; font-size: 0.78rem; }}
  .metric-label {{ color: var(--muted); }}
  .metric-val {{ font-weight: bold; font-family: monospace; }}
  .metric-val.fail-abs  {{ color: var(--fail); }}
  .metric-val.fail-adp  {{ color: var(--warn); }}
  .metric-val.pass      {{ color: var(--pass); }}
  .fail-tag {{ font-size: 0.68rem; padding: 1px 5px; border-radius: 8px; margin-left: 4px; }}
  .fail-tag.abs {{ background: #3a0f0f; color: var(--fail); }}
  .fail-tag.adp {{ background: #3a2e00; color: var(--warn); }}
  .no-well {{ color: var(--muted); font-size: 0.82rem; font-style: italic; padding: 8px 0; }}
  .well-montage {{ background: var(--panel); border: 1px solid var(--border);
                   border-radius: 8px; padding: 8px; display: flex;
                   align-items: center; justify-content: center; min-height: 200px; }}
  .well-montage img {{ max-width: 100%; max-height: 440px; border-radius: 4px; }}
  .well-montage .no-img {{ color: var(--muted); font-size: 0.82rem; text-align: center; }}
  .section-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em;
                    color: var(--muted); margin: 10px 0 4px; }}

  /* Overview */
  .overview-wrap {{ position: relative; background: #080a14; border: 1px solid var(--border);
                    border-radius: 8px; overflow: hidden; cursor: crosshair; }}
  .overview-wrap img {{ display: block; width: 100%; height: auto; }}
  .overview-wrap svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; }}
  .well-cell {{ fill: transparent; }}
  .well-cell:hover {{ fill: rgba(255,255,255,0.12); }}
  .well-cell.selected {{ fill: rgba(58,123,213,0.35); stroke: #3a7bd5; stroke-width: 2; }}

  /* Tabs */
  .tabs {{ display: flex; gap: 2px; flex-wrap: wrap; margin-bottom: 0; border-bottom: 2px solid var(--border); }}
  .tab {{ padding: 7px 16px; cursor: pointer; color: var(--muted); font-size: 0.8rem;
           border-radius: 4px 4px 0 0; transition: background 0.12s; white-space: nowrap; }}
  .tab:hover {{ background: #181d35; color: var(--text); }}
  .tab.active {{ background: var(--panel); color: var(--text);
                  border: 1px solid var(--border); border-bottom: 2px solid var(--bg); }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}
  .channel-grid {{ display: grid; grid-template-columns: repeat(3, 1fr);
                   gap: 10px; padding: 10px; background: var(--panel);
                   border: 1px solid var(--border); border-radius: 0 0 8px 8px; }}
  .channel-card {{ background: #0d1020; border: 1px solid var(--border);
                   border-radius: 6px; padding: 4px; }}

  /* Cell counts */
  .counts-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
  .count-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 8px; }}

  /* Compound filter */
  .compound-toolbar {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }}
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
  .flag-table th, .flag-table td {{ padding: 7px 10px; border: 1px solid var(--border); text-align: left; font-size: 0.8rem; }}
  .flag-table th {{ background: #1a2040; color: #8ab0e0; text-transform: uppercase; font-size: 0.72rem; }}
  .flag-table tr.hidden {{ display: none; }}
  .flag-table tr:hover {{ background: #161c35; }}

  /* MFI section */
  .mfi-channel-section {{ margin-bottom: 18px; border: 1px solid var(--border);
                          border-radius: 8px; overflow: hidden; }}
  .mfi-channel-header {{ background: #0d0f1e; padding: 8px 14px;
                         display: flex; align-items: center; gap: 9px;
                         border-bottom: 1px solid var(--border); }}
  .mfi-ch-dot {{ width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }}
  .mfi-channel-header h3 {{ font-size: 0.85rem; color: #a8c8ff; margin: 0; font-weight: 700; letter-spacing: 0.03em; }}
  /* Two-column: left=boxplots, right=platemap */
  .mfi-body {{ display: grid; grid-template-columns: 360px 1fr; }}
  .mfi-boxplot-col {{ display: flex; flex-direction: column;
                      border-right: 1px solid var(--border); background: #090b14; }}
  .mfi-plot-box {{ width: 100%; height: 250px; }}
  .mfi-plot-divider {{ height: 1px; background: var(--border); }}
  .mfi-platemap-col {{ padding: 14px 18px; display: flex; flex-direction: column;
                       align-items: flex-start; justify-content: center;
                       background: var(--bg); }}
  .mfi-platemap-title {{ font-size: 0.72rem; color: var(--muted);
                          margin-bottom: 10px; letter-spacing: 0.03em; }}
  /* Platemap grid: 1 label col + 12 well cols × 1 label row + 8 well rows */
  .mfi-platemap-grid {{ display: grid;
                        grid-template-columns: 20px repeat(12, 34px);
                        grid-template-rows: 20px repeat(8, 34px);
                        gap: 3px; }}
  .mfi-grid-label {{ display: flex; align-items: center; justify-content: center;
                     color: #4a5a78; font-size: 10px; font-weight: 600; user-select: none; }}
  .mfi-well-cell {{ width: 34px; height: 34px; border-radius: 4px;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 8px; color: rgba(255,255,255,0.5);
                    cursor: pointer; border: 1.5px solid transparent;
                    transition: transform 0.1s, border-color 0.1s;
                    position: relative; }}
  .mfi-well-cell:hover {{ transform: scale(1.18); z-index: 10;
                           border-color: rgba(255,255,255,0.6) !important; }}
  .mfi-well-cell.dimmed {{ opacity: 0.12; }}
  .mfi-well-cell.lit    {{ opacity: 1 !important; border-color: rgba(255,255,255,0.5); }}
  /* Compound filter bar inside MFI */
  .mfi-filter-bar {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
                     padding: 8px 0 14px; }}
  .mfi-filter-btn {{ padding: 3px 10px; border-radius: 12px;
                     border: 1.5px solid #263050; background: #131628;
                     color: #8898c0; cursor: pointer; font-size: 11px;
                     transition: background 0.12s, border-color 0.12s; white-space: nowrap; }}
  .mfi-filter-btn:hover  {{ border-color: #4060a0; color: #c0d0f0; }}
  .mfi-filter-btn.active {{ background: #1a3060; border-color: #4080d8; color: #fff; font-weight: 600; }}
  /* Tooltip */
  #mfi-tooltip {{ position: fixed; background: #141c34; border: 1px solid #304080;
                   border-radius: 7px; padding: 8px 13px; font-size: 11px;
                   color: #d0e0ff; pointer-events: none; display: none; z-index: 9999;
                   line-height: 1.65; max-width: 210px; box-shadow: 0 4px 18px rgba(0,0,0,0.6); }}
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
          <th>Illumination</th><th>Focus</th><th>Flagged wells</th><th>MFI Δ Hoechst</th><th>MFI Δ Syto</th></tr>
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

  <!-- Well detail row: info panel (left) + montage (right) -->
  <div class="browser-body">
    <div class="well-info-panel" id="well-info">
      <p class="no-well">Click a well in the overview below to inspect it.</p>
    </div>
    <div class="well-montage" id="well-montage">
      <span class="no-img">Select a well to see its site montage.</span>
    </div>
  </div>

  <!-- Overview: microscopy thumbnails, clickable via SVG overlay -->
  <div class="overview-wrap" id="overview-wrap">
    <img id="overview-img" src="" alt="Plate overview">
    <svg id="overview-svg" viewBox="0 0 1 1" preserveAspectRatio="none"></svg>
  </div>
</div>

<!-- 3. QC METRICS -->
<div class="section">
  <h2>QC Metrics</h2>
  <div class="feature-audit">
    <div class="audit-row">
      <span class="audit-label">Illumination</span>
      <span class="thresh">ImageQuality_PowerLogLogSlope_*</span>
      <span class="thresh">ImageQuality_MaxIntensity_*</span>
      <span class="audit-source">← Image.txt</span>
    </div>
    <div class="audit-row">
      <span class="audit-label">Focus</span>
      <span class="thresh">ImageQuality_FocusScore_*</span>
      <span class="audit-source">← Image.txt</span>
    </div>
    <hr class="audit-sep">
    <div class="audit-row">
      <span class="audit-label">Thresholds</span>
      Slope <span class="thresh-val">[−2.5, −1.0]</span>
      &nbsp;·&nbsp; MaxInt <span class="thresh-val">≤ 0.95</span>
      &nbsp;·&nbsp; Focus <span class="thresh-val">≥ 0.005</span>
      &nbsp;·&nbsp; LocalFocus per-channel
      &nbsp;·&nbsp; Adaptive: median ± 3σ MAD per plate
    </div>
  </div>
  <div class="compound-toolbar" id="compound-toolbar" style="margin-bottom:10px;">
    <label>Filter by compound:</label>
    <button class="cmpd-btn ctrl" id="btn-show-all">All</button>
    <button class="cmpd-btn ctrl" id="btn-hide-all">None</button>
  </div>
  <div class="tabs" id="metrics-tabs"></div>
  <div id="metrics-contents"></div>
</div>

<!-- 4. CELL COUNTS -->
<div class="section">
  <h2>Cell Counts</h2>
  <div class="feature-audit">
    <div class="audit-row">
      <span class="audit-label">Cells</span>
      <span class="thresh">Count_Cells</span>
      <span class="audit-source">← Image.txt</span>
    </div>
    <div class="audit-row">
      <span class="audit-label">Ratio</span>
      <span class="thresh">Count_Cells / Count_Nuclei</span>
      &nbsp;
      <span style="color:#4bd760;font-size:0.78rem;">■ 1.00</span>
      <span style="color:#ffbe00;font-size:0.78rem;">■ 0.99-0.95</span>
      <span style="color:#ff4444;font-size:0.78rem;">■ &lt;0.95</span>
    </div>
    <div class="audit-row">
      <span class="audit-label">Illum. Artifacts</span>
      <span class="thresh">Count_Illum_artifacts</span>
      <span class="audit-source">← Image.txt</span>
      &nbsp;—
      <span style="color:#ffffff;font-size:0.78rem;">■ 0</span>
      <span style="color:#cc1111;font-size:0.78rem;">■ ≥500</span>
    </div>
  </div>
  <div class="counts-grid" id="counts-grid"></div>
</div>

<!-- 5. MFI (Median Fluorescence Intensity) -->
<div class="section">
  <h2>Median Fluorescence Intensity (MFI)</h2>
  <div class="feature-audit">
    <div class="audit-row">
      <span class="audit-label">Hoechst</span>
      <span class="thresh">Intensity_MeanIntensity_Hoechst</span>
      <span class="audit-source">← Nuclei.txt</span>
    </div>
    <div class="audit-row">
      <span class="audit-label">Syto / ER / Golgi / Mito</span>
      <span class="thresh">Intensity_MeanIntensity_*</span>
      <span class="audit-source">← Cells.txt</span>
    </div>
    <hr class="audit-sep">
    <div class="audit-row">
      Per-image median of all objects → one value per site per well · boxplots show distribution across sites
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
    <label style="color:var(--muted);font-size:0.8rem;">Plate</label>
    <select id="mfi-plate-select" style="background:#1a2040;color:var(--text);border:1px solid var(--border);
            border-radius:4px;padding:5px 10px;font-size:0.88rem;cursor:pointer;"></select>
  </div>
  <div class="mfi-filter-bar" id="mfi-filter-bar">
    <span style="color:var(--muted);font-size:0.8rem;">Highlight compound:</span>
    <button class="mfi-filter-btn active" id="mfi-btn-all">All</button>
    <button class="mfi-filter-btn" id="mfi-btn-none">None</button>
  </div>
  <div id="mfi-content"></div>
  <div id="mfi-tooltip"></div>
</div>

<!-- 6. FLAGGED WELLS -->
<div class="section">
  <h2>Flagged Wells</h2>
  <div class="flag-controls">
    <input id="flag-filter" type="text" placeholder="Filter by plate, well, compound, or metric…">
  </div>
  <table class="flag-table">
    <thead>
      <tr><th>Plate</th><th>Well</th><th>Compound</th><th>Absolute fails</th><th>Adaptive outliers</th></tr>
    </thead>
    <tbody id="flag-tbody"></tbody>
  </table>
</div>

</div><!-- /container -->
<script>
const DATA   = {data_json};
const COUNT_RANGES = {count_ranges_json};
const MFI_DATA     = {mfi_payload_json};
const MFI_CHANNELS = {mfi_channels_json};
const MFI_COLORS   = {mfi_colors_json};
const PLATES = Object.keys(DATA);

const SLOPE_SPECS  = {slope_specs};
const FOCUS_SPECS  = {focus_specs};
const MAXINT_SPECS = {maxint_specs};
const METRIC_GROUPS = [
  {{ label: 'Slope',        specs: SLOPE_SPECS  }},
  {{ label: 'LocalFocus',   specs: FOCUS_SPECS  }},
  {{ label: 'MaxIntensity', specs: MAXINT_SPECS }},
];

// ROWS: A at top (index 0) -> H at bottom (index 7)
// Plotly heatmap y-axis goes bottom->top, so we reverse for display.
const ROW_LABELS  = ['A','B','C','D','E','F','G','H'];
const ROWS_PLOTLY = ['H','G','F','E','D','C','B','A'];  // reversed for Plotly
const COLS = Array.from({{length:12}}, (_,i) => String(i+1).padStart(2,'0'));
const PLATE_ROWS = 8, PLATE_COLS = 12;

// MFI_COLORS and MFI_DATA/MFI_CHANNELS injected above from Python
function _chColor(ch) {{
  // First try exact key match in injected MFI_COLORS, then substring fallback
  if (MFI_COLORS[ch]) return MFI_COLORS[ch];
  const fallbacks = {{
    'DNA':'#6495ed','Hoechst':'#6495ed','DAPI':'#6495ed',
    'Syto':'#4bc870','Golgi':'#ffc04d','ER':'#b06aff',
    'Mito':'#ff5555','BF':'#909090',
  }};
  for (const [k,v] of Object.entries({{...MFI_COLORS,...fallbacks}}))
    if (ch.toLowerCase().includes(k.toLowerCase())) return v;
  return '#8ab0d0';
}}

// ── Helpers ───────────────────────────────────────────────────────────────────
function badge(pct) {{
  const cls = pct >= 80 ? 'pass' : pct >= 50 ? 'warn' : 'fail';
  return `<span class="badge badge-${{cls}}">${{pct}}%</span>`;
}}
function fmt3(v) {{ return v != null ? v.toFixed(3) : '—'; }}
function fmt5(v) {{ return v != null ? v.toFixed(5) : '—'; }}

// Median of array (ignoring nulls)
function arrMedian(arr) {{
  const a = arr.filter(x => x != null && !isNaN(x)).sort((a,b)=>a-b);
  if (!a.length) return null;
  const m = Math.floor(a.length/2);
  return a.length%2 ? a[m] : (a[m-1]+a[m])/2;
}}

// ── 1. Summary ────────────────────────────────────────────────────────────────
(function() {{
  const tbody = document.getElementById('summary-tbody');
  PLATES.forEach(p => {{
    const d = DATA[p];
    // Compute median MFI Δ for Hoechst/DNA and Syto across wells
    const mfiDelta = (ch) => {{
      const objKey = `Nuclei_${{ch}}`;
      const vals = Object.values(d.mfi||{{}}).map(m=>m[objKey]).filter(x=>x!=null);
      if (!vals.length) return '—';
      return arrMedian(vals).toFixed(5);
    }};
    tbody.insertAdjacentHTML('beforeend', `<tr>
      <td><strong>${{p}}</strong></td><td>${{d.n_wells}}</td>
      <td>${{badge(d.pass_rate)}} ${{d.n_pass}}/${{d.n_wells}}</td>
      <td>${{badge(Math.round(d.n_illum/d.n_wells*100))}} ${{d.n_illum}}/${{d.n_wells}}</td>
      <td>${{badge(Math.round(d.n_focus/d.n_wells*100))}} ${{d.n_focus}}/${{d.n_wells}}</td>
      <td>${{d.n_wells - d.n_pass}}</td>
      <td style="font-family:monospace;">${{mfiDelta('DNA')||mfiDelta('Hoechst')}}</td>
      <td style="font-family:monospace;">${{mfiDelta('Syto')||mfiDelta('RNA')}}</td>
    </tr>`);
  }});
}})();

// ── 2. Plate browser ──────────────────────────────────────────────────────────
const slider   = document.getElementById('plate-slider');
const pSelect  = document.getElementById('plate-select');
const nameDisp = document.getElementById('plate-name-display');

slider.max = PLATES.length - 1;
PLATES.forEach((p, i) =>
  pSelect.insertAdjacentHTML('beforeend', `<option value="${{i}}">${{p}}</option>`)
);

let currentPlate = null;
let selectedWell = null;

function renderPlate(idx) {{
  const name    = PLATES[idx];
  currentPlate  = name;
  selectedWell  = null;
  nameDisp.textContent = name;
  slider.value  = idx;
  pSelect.value = idx;

  const pd = DATA[name];
  document.getElementById('overview-img').src =
    `data:image/jpeg;base64,${{pd.overview_b64}}`;
  buildOverlaySVG(name);

  document.getElementById('well-info').innerHTML =
    '<p class="no-well">Click a well in the overview below to inspect it.</p>';
  document.getElementById('well-montage').innerHTML =
    '<span class="no-img">Select a well to see its site montage.</span>';

  renderMetrics(name);
  renderCounts(name);
}}

// ── Fix #2: corrected SVG overlay (A=top, H=bottom) ──────────────────────────
function buildOverlaySVG(plateName) {{
  const pd  = DATA[plateName];
  const cw  = pd.overview_cw;
  const ch  = pd.overview_ch;
  const svg = document.getElementById('overview-svg');

  const imgW = cw * PLATE_COLS;
  const imgH = ch * PLATE_ROWS;
  svg.setAttribute('viewBox', `0 0 ${{imgW}} ${{imgH}}`);
  svg.innerHTML = '';

  // ROW_LABELS[0]='A' -> ri=0 -> y0=0 (top). Correct: A01 maps to top-left.
  ROW_LABELS.forEach((r, ri) => {{
    COLS.forEach((c, ci) => {{
      const well = r + c;
      const x0   = ci * cw, y0 = ri * ch;
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('class', 'well-cell');
      rect.setAttribute('id', `sv-${{well}}`);
      rect.setAttribute('x', x0); rect.setAttribute('y', y0);
      rect.setAttribute('width', cw); rect.setAttribute('height', ch);
      rect.setAttribute('data-well', well);
      rect.onclick = () => selectWell(plateName, well);
      svg.appendChild(rect);
    }});
  }});
}}

function selectWell(plateName, well) {{
  if (selectedWell) {{
    const prev = document.getElementById(`sv-${{selectedWell}}`);
    if (prev) prev.classList.remove('selected');
  }}
  selectedWell = well;
  const rect = document.getElementById(`sv-${{well}}`);
  if (rect) rect.classList.add('selected');

  renderWellInfo(plateName, well);
  renderWellMontage(plateName, well);
}}

// ── Well info panel ───────────────────────────────────────────────────────────
const ALL_METRIC_SPECS = [...SLOPE_SPECS, ...FOCUS_SPECS, ...MAXINT_SPECS];

function renderWellInfo(plateName, well) {{
  const pd     = DATA[plateName];
  const m      = pd.wells[well] || {{}};
  const flags  = pd.well_flags[well] || {{}};
  const absSet = new Set(flags.abs || []);
  const adpSet = new Set(flags.adp || []);
  const cmpd   = m.compound || '—';

  let html = `<h3>${{well}}</h3>
    <div class="well-compound">${{cmpd}}</div>`;

  html += `<div class="section-label">Cell counts</div>`;
  [['Cells','Count_Cells'],['Nuclei','Count_Nuclei'],
   ['Raw nuclei','Count_Raw_nuclei'],['Artifacts','Count_Illum_artifacts']].forEach(([lbl,col]) => {{
    const v = m[col];
    html += `<div class="metric-row">
      <span class="metric-label">${{lbl}}</span>
      <span class="metric-val">${{v != null ? Math.round(v) : '—'}}</span>
    </div>`;
  }});

  // MFI for this well
  const mfi = (pd.mfi||{{}})[well];
  if (mfi && Object.keys(mfi).length) {{
    html += `<div class="section-label">MFI (Nuclei)</div>`;
    Object.entries(mfi).forEach(([key, v]) => {{
      html += `<div class="metric-row">
        <span class="metric-label">${{key}}</span>
        <span class="metric-val" style="color:${{_chColor(key)}}">${{v!=null?v.toFixed(5):'—'}}</span>
      </div>`;
    }});
  }}

  ALL_METRIC_SPECS.forEach(spec => {{
    const v      = m[spec.col];
    if (v == null) return;
    const metKey = spec.title.split('—')[0].trim();
    const chKey  = spec.title.split('—')[1]?.trim() || '';
    const tag    = `${{metKey}}/${{chKey}}`;
    const isFail = absSet.has(tag) || adpSet.has(tag);
    const cls    = isFail ? 'fail-abs' : 'pass';
    html += `<div class="metric-row">
      <span class="metric-label">${{spec.title}}</span>
      <span class="metric-val ${{cls}}">${{fmt3(v)}}</span>
    </div>`;
  }});

  document.getElementById('well-info').innerHTML = html;
}}

function renderWellMontage(plateName, well) {{
  const pd  = DATA[plateName];
  const b64 = pd.flagged_b64[well];
  const el  = document.getElementById('well-montage');
  if (b64) {{
    el.innerHTML = `<img src="data:image/jpeg;base64,${{b64}}" alt="Well ${{well}} montage">`;
  }} else {{
    const flags = pd.well_flags[well];
    el.innerHTML = flags
      ? '<span class="no-img">Flagged well — montage not pre-generated.</span>'
      : '<span class="no-img">Well passes all QC thresholds — no image preloaded.</span>';
  }}
}}

slider.oninput   = () => renderPlate(+slider.value);
pSelect.onchange = () => renderPlate(+pSelect.value);

// ── Compound filter state ────────────────────────────────────────────────────
let activeCompounds = new Set();
let hideAllMode     = false;
function compoundVisible(cmpd) {{
  if (hideAllMode)              return false;
  if (activeCompounds.size===0) return true;
  return activeCompounds.has(cmpd);
}}

// ── 3. QC Metrics ─────────────────────────────────────────────────────────────
function wellMatrix(plateData, colName) {{
  return ROWS_PLOTLY.map(r => COLS.map(c => {{
    const w    = r + c;
    const well = plateData.wells[w];
    if (!well) return null;
    if (!compoundVisible(well.compound || '')) return null;
    return well[colName] ?? null;
  }}));
}}

function isFiltered() {{
  return hideAllMode || activeCompounds.size > 0;
}}

function makeHeatmap(plateData, spec) {{
  const z    = wellMatrix(plateData, spec.col);
  const text = ROWS_PLOTLY.map(r => COLS.map(c => {{
    const w    = r + c;
    const v    = plateData.wells[w]?.[spec.col];
    const cmpd = plateData.wells[w]?.compound || '';
    const vis  = compoundVisible(cmpd);
    return vis
      ? `<b>${{w}}</b><br>${{cmpd}}<br>${{spec.title}}: ${{v != null ? v.toFixed(3) : 'N/A'}}`
      : `<b>${{w}}</b><br>${{cmpd}}<br>(hidden)`;
  }}));
  return {{
    type:'heatmap', z, text, hoverinfo:'text',
    x:COLS, y:ROWS_PLOTLY, colorscale: spec.cs,
    zmin: spec.cmin, zmax: spec.cmax,
    colorbar:{{ thickness:10, len:0.85, tickfont:{{size:9}} }},
  }};
}}

function heatmapLayout(title, extraY) {{
  const filtered = isFiltered();
  const gridcolor = filtered ? 'rgba(0,0,0,0)' : 'rgba(80,90,120,0.4)';
  return {{
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#0a0c18',
    font:{{color:'#c8d8f0', size:10}},
    margin:{{t:32,b:42,l:42,r:16}}, height:300,
    title:{{text:title, font:{{size:11,color:'#8ab0e0'}}, x:0.5}},
    xaxis:{{ tickfont:{{size:9}}, showgrid:!filtered, gridcolor, zeroline:false }},
    yaxis:{{ tickfont:{{size:9}}, showgrid:!filtered, gridcolor, zeroline:false,
             title: extraY ? {{text:'Row', font:{{size:9}}}} : undefined }},
  }};
}}

const tabsEl     = document.getElementById('metrics-tabs');
const contentsEl = document.getElementById('metrics-contents');

function renderMetrics(plateName) {{
  tabsEl.innerHTML = ''; contentsEl.innerHTML = '';
  const pd = DATA[plateName];
  let firstGroup = true;
  METRIC_GROUPS.forEach((grp, gi) => {{
    const isFirst = firstGroup;
    tabsEl.insertAdjacentHTML('beforeend',
      `<div class="tab${{isFirst?' active':''}}" data-group="${{gi}}">${{grp.label}}</div>`);
    contentsEl.insertAdjacentHTML('beforeend',
      `<div class="tab-content${{isFirst?' active':''}}" id="grp-${{gi}}">
         <div class="channel-grid" id="chgrid-${{gi}}"></div></div>`);
    setTimeout(() => {{
      const grid = document.getElementById(`chgrid-${{gi}}`);
      if (!grid) return;
      grp.specs.forEach((s, si) => {{
        const cid  = `ch-${{gi}}-${{si}}`;
        const card = document.createElement('div');
        card.className = 'channel-card';
        card.innerHTML = `<div id="${{cid}}"></div>`;
        grid.appendChild(card);
        Plotly.react(cid, [makeHeatmap(pd, s)],
          heatmapLayout(s.title, si % 3 === 0),
          {{responsive:true, displayModeBar:false}});
      }});
    }}, 0);
    if (isFirst) firstGroup = false;
  }});
  tabsEl.querySelectorAll('.tab').forEach(tab => {{
    tab.onclick = () => {{
      const gi = tab.dataset.group;
      tabsEl.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      contentsEl.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      document.querySelector(`.tab[data-group="${{gi}}"]`).classList.add('active');
      document.getElementById(`grp-${{gi}}`).classList.add('active');
    }};
  }});
}}

// ── 4. Cell counts — Cells heatmap + Cells/Nuclei ratio ──────────────────────
function renderCounts(plateName) {{
  const grid = document.getElementById('counts-grid');
  grid.innerHTML = '';
  const pd = DATA[plateName];

  // Card 1: Cells count heatmap
  (function() {{
    const cid  = `cnt-cells`;
    const card = document.createElement('div');
    card.className = 'count-card';
    card.innerHTML = `<div id="${{cid}}"></div>`;
    grid.appendChild(card);
    const z    = ROWS_PLOTLY.map(r => COLS.map(c => {{
      const w = r+c, well = pd.wells[w];
      if (!well || !compoundVisible(well.compound||'')) return null;
      return well['Count_Cells'] ?? null;
    }}));
    const text = ROWS_PLOTLY.map(r => COLS.map(c => {{
      const w = r+c, v = pd.wells[w]?.['Count_Cells'];
      return `<b>${{w}}</b><br>${{pd.wells[w]?.compound||''}}<br>Cells: ${{v!=null?Math.round(v):'N/A'}}`;
    }}));
    const filtered = isFiltered();
    const gridcolor = filtered ? 'rgba(0,0,0,0)' : 'rgba(80,90,120,0.4)';
    const cRange = COUNT_RANGES['Count_Cells'] || [null,null];
    Plotly.react(cid,
      [{{type:'heatmap',z,text,hoverinfo:'text',x:COLS,y:ROWS_PLOTLY,colorscale:'Viridis',
         zmin:cRange[0],zmax:cRange[1],colorbar:{{thickness:14,len:0.85,tickfont:{{size:10}}}}}}],
      {{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#0a0c18',
        font:{{color:'#c8d8f0',size:11}},margin:{{t:40,b:50,l:50,r:20}},height:340,
        title:{{text:'Cells',font:{{size:13,color:'#a8c8ff'}},x:0.5}},
        xaxis:{{title:'Column',tickfont:{{size:10}},tickvals:COLS,ticktext:COLS.map(c=>parseInt(c)),
                showgrid:!filtered,gridcolor,zeroline:false}},
        yaxis:{{title:'Row',tickfont:{{size:10}},showgrid:!filtered,gridcolor,zeroline:false}},
      }},{{responsive:true,displayModeBar:false}});
  }})();

  // Card 2: Cells/Nuclei ratio with custom colour scale (1=green, 0.99-095=yellow, <0.95=red)
  (function() {{
    const cid  = `cnt-ratio`;
    const card = document.createElement('div');
    card.className = 'count-card';
    card.innerHTML = `<div id="${{cid}}"></div>`;
    grid.appendChild(card);

    const ratioCS = [
      [0,    '#ff4444'],
      [0.94, '#ff4444'],
      [0.95, '#ffbe00'],
      [0.99, '#ffbe00'],
      [1.0,  '#4bd760'],
    ];

    const z    = ROWS_PLOTLY.map(r => COLS.map(c => {{
      const w = r+c, well = pd.wells[w];
      if (!well || !compoundVisible(well.compound||'')) return null;
      const cells  = well['Count_Cells'];
      const nuclei = well['Count_Nuclei'];
      if (cells == null || nuclei == null || nuclei === 0) return null;
      return Math.min(1, cells / nuclei);
    }}));
    const text = ROWS_PLOTLY.map(r => COLS.map(c => {{
      const w    = r+c, well = pd.wells[w];
      const cells  = well?.['Count_Cells'];
      const nuclei = well?.['Count_Nuclei'];
      const ratio  = (cells != null && nuclei != null && nuclei > 0)
                     ? (cells/nuclei).toFixed(4) : 'N/A';
      return `<b>${{w}}</b><br>${{well?.compound||''}}<br>` +
             `Ratio: ${{ratio}}<br>Cells: ${{cells!=null?Math.round(cells):'—'}} · ` +
             `Nuclei: ${{nuclei!=null?Math.round(nuclei):'—'}}`;
    }}));
    const filtered = isFiltered();
    const gridcolor = filtered ? 'rgba(0,0,0,0)' : 'rgba(80,90,120,0.4)';
    Plotly.react(cid,
      [{{type:'heatmap',z,text,hoverinfo:'text',x:COLS,y:ROWS_PLOTLY,
         colorscale:ratioCS, zmin:0.93, zmax:1.0,
         colorbar:{{thickness:14,len:0.85,tickfont:{{size:10}},
                    tickvals:[0.93, 0.95, 0.99, 1.0],
                    ticktext:['<0.95','0.95','0.99','1.00']}}}}],
      {{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#0a0c18',
        font:{{color:'#c8d8f0',size:11}},margin:{{t:40,b:50,l:50,r:20}},height:340,
        title:{{text:'Cells / Nuclei ratio',font:{{size:13,color:'#a8c8ff'}},x:0.5}},
        xaxis:{{title:'Column',tickfont:{{size:10}},tickvals:COLS,ticktext:COLS.map(c=>parseInt(c)),
                showgrid:!filtered,gridcolor,zeroline:false}},
        yaxis:{{title:'Row',tickfont:{{size:10}},showgrid:!filtered,gridcolor,zeroline:false}},
      }},{{responsive:true,displayModeBar:false}});
  }})();

  // Card 3: Illumination Artifacts count heatmap (white=0, dark red=500+)
  (function() {{
    const cid  = `cnt-artifacts`;
    const card = document.createElement('div');
    card.className = 'count-card';
    card.innerHTML = `<div id="${{cid}}"></div>`;
    grid.appendChild(card);

    // White (0 artifacts) -> dark red (500+ artifacts)
    const artifactCS = [
      [0,   '#ffffff'],
      [0.1, '#ffcccc'],
      [0.3, '#ff6666'],
      [0.6, '#cc2222'],
      [1.0, '#660000'],
    ];

    const z    = ROWS_PLOTLY.map(r => COLS.map(c => {{
      const w = r+c, well = pd.wells[w];
      if (!well || !compoundVisible(well.compound||'')) return null;
      return well['Count_Illum_artifacts'] ?? null;
    }}));
    const text = ROWS_PLOTLY.map(r => COLS.map(c => {{
      const w    = r+c, well = pd.wells[w];
      const n    = well?.['Count_Illum_artifacts'];
      const cmpd = well?.compound || '';
      return `<b>${{w}}</b><br>${{cmpd}}<br>Illum. Artifacts: ${{n!=null?Math.round(n):'N/A'}}`;
    }}));
    const filtered2 = isFiltered();
    const gridcolor2 = filtered2 ? 'rgba(0,0,0,0)' : 'rgba(80,90,120,0.4)';
    Plotly.react(cid,
      [{{type:'heatmap',z,text,hoverinfo:'text',x:COLS,y:ROWS_PLOTLY,
         colorscale:artifactCS, zmin:0, zmax:500,
         colorbar:{{thickness:14,len:0.85,tickfont:{{size:10}},
                    tickvals:[0,100,250,500],ticktext:['0','100','250','≥500']}}}}],
      {{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#0a0c18',
        font:{{color:'#c8d8f0',size:11}},margin:{{t:40,b:50,l:50,r:20}},height:340,
        title:{{text:'Illum. Artifacts',font:{{size:13,color:'#a8c8ff'}},x:0.5}},
        xaxis:{{title:'Column',tickfont:{{size:10}},tickvals:COLS,ticktext:COLS.map(c=>parseInt(c)),
                showgrid:!filtered2,gridcolor:gridcolor2,zeroline:false}},
        yaxis:{{title:'Row',tickfont:{{size:10}},showgrid:!filtered2,gridcolor:gridcolor2,zeroline:false}},
      }},{{responsive:true,displayModeBar:false}});
  }})();
}}

// ── 5. MFI section (V2: boxplots + interactive platemap per channel) ──────────

PLATES.forEach(p =>
  document.getElementById('mfi-plate-select').insertAdjacentHTML(
    'beforeend', `<option value="${{p}}">${{p}}</option>`)
);
document.getElementById('mfi-plate-select').onchange = () => renderMFI();

// ── MFI state ──────────────────────────────────────────────────────────────────
let mfiActivePlate  = PLATES[0] || '';
let mfiActiveFilter = 'All';

// Assign a distinct colour per compound (for platemap highlighting)
const MFI_COMPOUND_PALETTE = [
  '#4A90D0','#E8A830','#50C878','#E05858','#A060D0',
  '#30C0C0','#E87050','#80B040','#C050A0','#5080E0',
  '#D0A040','#60C0A0','#D06060','#8060D0','#40A0C0',
];
const mfiCompoundColors = {{}};
(function() {{
  const allCmpds = new Set();
  PLATES.forEach(p => {{
    Object.values(DATA[p].wells || {{}}).forEach(m => {{ if (m.compound) allCmpds.add(m.compound); }});
  }});
  [...allCmpds].sort().forEach((c, i) => {{
    mfiCompoundColors[c] = MFI_COMPOUND_PALETTE[i % MFI_COMPOUND_PALETTE.length];
  }});
}})();

// ── Stats helpers ──────────────────────────────────────────────────────────────
function mfiQuantile(sorted, q) {{
  const pos = q * (sorted.length - 1);
  const lo  = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}}
function mfiBoxStats(vals) {{
  if (!vals || vals.length === 0) return null;
  const s   = [...vals].sort((a, b) => a - b);
  const q1  = mfiQuantile(s, 0.25);
  const med = mfiQuantile(s, 0.50);
  const q3  = mfiQuantile(s, 0.75);
  const iqr = q3 - q1;
  const whislo = s.find(v => v >= q1 - 1.5 * iqr) ?? s[0];
  const whishi = [...s].reverse().find(v => v <= q3 + 1.5 * iqr) ?? s[s.length - 1];
  return {{ q1, median: med, q3, whislo, whishi, n: s.length }};
}}
function mfiHexAlpha(hex, alpha) {{
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${{r}},${{g}},${{b}},${{alpha.toFixed(3)}})`;
}}

// ── Boxplot rendering ──────────────────────────────────────────────────────────
function mfiRenderBoxplot(divId, plateName, channel, groupBy) {{
  const wellData    = (MFI_DATA[plateName] || {{}})[channel] || {{}};
  const color       = MFI_COLORS[channel] || '#8ab0d0';
  const categories  = groupBy === 'row' ? ROW_LABELS : COLS.map(String);
  const wellCmpd    = (plateName && DATA[plateName]) ? DATA[plateName].wells : {{}};

  const allVals    = Object.values(wellData).flat();
  const allSorted  = [...allVals].sort((a,b) => a-b);
  const globalMed  = allSorted.length ? mfiQuantile(allSorted, 0.5) : null;

  const boxTraces = [];
  const ptX = [], ptY = [];

  categories.forEach(cat => {{
    const wells = Object.keys(wellData).filter(well => {{
      const match = groupBy === 'row' ? well[0] === cat : well.slice(1) === cat.padStart(2,'0');
      if (!match) return false;
      if (mfiActiveFilter === 'All')  return true;
      if (mfiActiveFilter === 'None') return false;
      return (wellCmpd[well]?.compound || '') === mfiActiveFilter;
    }});
    let vals = [];
    wells.forEach(w => {{ vals = vals.concat(wellData[w] || []); }});
    const stats = mfiBoxStats(vals);
    if (!stats) return;
    boxTraces.push({{
      type:'box', name: String(cat), x:[String(cat)],
      lowerfence:[stats.whislo], q1:[stats.q1], median:[stats.median],
      q3:[stats.q3], upperfence:[stats.whishi],
      marker:{{color, size:3, opacity:0.7}}, line:{{color, width:1.5}},
      fillcolor: mfiHexAlpha(color, 0.18), showlegend:false, boxmean:false,
    }});
    const sample = vals.length > 150 ? [...vals].sort(()=>Math.random()-0.5).slice(0,150) : vals;
    sample.forEach(v => {{ ptX.push(String(cat)); ptY.push(v); }});
  }});

  const scatterTrace = {{
    type:'scatter', mode:'markers', x:ptX, y:ptY,
    marker:{{color:'#000', size:3, opacity:0.45}},
    showlegend:false, hoverinfo:'none',
  }};

  const yMin = ptY.length ? Math.min(...ptY, globalMed??0)*0.97 : 0;
  const yMax = ptY.length ? Math.max(...ptY, globalMed??0)*1.03 : 1;
  const shapes = globalMed != null ? [{{
    type:'line', xref:'paper', yref:'y', x0:0, x1:1, y0:globalMed, y1:globalMed,
    line:{{color:'#FF4040', width:1.5, dash:'dot'}},
  }}] : [];
  const annotations = globalMed != null ? [{{
    xref:'paper', x:0.99, xanchor:'right', yref:'y', y:globalMed, yanchor:'bottom',
    text:`median: ${{globalMed.toFixed(4)}}`, showarrow:false,
    font:{{color:'#FF8888', size:8}},
  }}] : [];

  Plotly.react(divId, [...boxTraces, scatterTrace], {{
    paper_bgcolor:'#090b14', plot_bgcolor:'#090b14',
    margin:{{t:28,b:34,l:50,r:14}},
    title:{{text:`${{channel}} — by ${{groupBy==='row'?'Row':'Column'}}`,
            font:{{color:'#8a9ab8',size:10}}, x:0.03, xanchor:'left'}},
    xaxis:{{type:'category', color:'#5a6a88', tickfont:{{size:9,color:'#7a8aaa'}},
             gridcolor:'#161c2c', tickmode:'array', tickvals:categories.map(String)}},
    yaxis:{{title:{{text:'MFI',font:{{color:'#5a6a88',size:9}},standoff:4}},
             color:'#5a6a88', tickfont:{{size:9,color:'#7a8aaa'}},
             gridcolor:'#161c2c', zeroline:false, range:[yMin, yMax]}},
    shapes, annotations,
  }}, {{responsive:true, displayModeBar:false}});
}}

// ── Platemap rendering ─────────────────────────────────────────────────────────
function mfiRenderPlatemap(channel, plateName) {{
  const grid     = document.getElementById(`mfi-grid-${{channel}}`);
  const wellData = (MFI_DATA[plateName] || {{}})[channel] || {{}};
  const color    = MFI_COLORS[channel] || '#8ab0d0';
  const wellCmpd = (DATA[plateName]?.wells || {{}});
  grid.innerHTML = '';

  // Per-well medians for opacity
  const wellMeds = {{}};
  Object.entries(wellData).forEach(([well, vals]) => {{
    const s = [...vals].sort((a,b)=>a-b);
    wellMeds[well] = s.length ? mfiQuantile(s, 0.5) : 0;
  }});
  const medVals = Object.values(wellMeds);
  const minMed  = medVals.length ? Math.min(...medVals) : 0;
  const maxMed  = medVals.length ? Math.max(...medVals) : 1;

  function opacityFor(well) {{
    const m = wellMeds[well];
    if (m === undefined) return 0.1;
    if (maxMed === minMed) return 0.6;
    return 0.18 + 0.82*(m-minMed)/(maxMed-minMed);
  }}

  // Corner spacer
  grid.appendChild(Object.assign(document.createElement('div'), {{}}));

  // Column headers
  COLS.forEach(c => {{
    const lbl = document.createElement('div');
    lbl.className = 'mfi-grid-label';
    lbl.textContent = parseInt(c);
    grid.appendChild(lbl);
  }});

  // Rows
  ROW_LABELS.forEach(row => {{
    const rowLbl = document.createElement('div');
    rowLbl.className = 'mfi-grid-label';
    rowLbl.textContent = row;
    grid.appendChild(rowLbl);

    COLS.forEach(col => {{
      const well     = row + col;
      const compound = wellCmpd[well]?.compound || '';
      const cell     = document.createElement('div');
      cell.className = 'mfi-well-cell';
      cell.id        = `mfi-cell-${{channel}}-${{well}}`;

      let bg;
      if (mfiActiveFilter === 'None') {{
        bg = 'rgba(20,24,40,1)';
        cell.classList.remove('lit','dimmed');
      }} else if (mfiActiveFilter === 'All') {{
        bg = mfiHexAlpha(color, opacityFor(well));
        cell.classList.remove('lit','dimmed');
      }} else {{
        if (compound === mfiActiveFilter) {{
          bg = mfiHexAlpha(mfiCompoundColors[compound]||color, 0.85);
          cell.classList.add('lit'); cell.classList.remove('dimmed');
        }} else {{
          bg = 'rgba(16,20,34,1)';
          cell.classList.add('dimmed'); cell.classList.remove('lit');
        }}
      }}
      cell.style.background = bg;

      // Tooltip
      cell.addEventListener('mouseenter', e => {{
        const vals = wellData[well] || [];
        const s    = [...vals].sort((a,b)=>a-b);
        const med  = s.length ? mfiQuantile(s,0.5) : null;
        const q1   = s.length ? mfiQuantile(s,0.25): null;
        const q3   = s.length ? mfiQuantile(s,0.75): null;
        const fmt  = v => v!=null ? v.toFixed(4) : '—';
        const cc   = mfiCompoundColors[compound] || '#8898c0';
        document.getElementById('mfi-tooltip').innerHTML =
          `<b style="font-size:12px">${{well}}</b> ` +
          `<span style="color:${{cc}};font-size:10px">${{compound}}</span><br>` +
          `Channel: <span style="color:${{color}}">${{channel}}</span><br>` +
          `Median MFI: <b>${{fmt(med)}}</b><br>` +
          `IQR: ${{fmt(q1)}} – ${{fmt(q3)}}<br>` +
          `n images: ${{vals.length}}`;
        document.getElementById('mfi-tooltip').style.display = 'block';
        document.getElementById('mfi-tooltip').style.left = (e.clientX+15)+'px';
        document.getElementById('mfi-tooltip').style.top  = (e.clientY-12)+'px';
      }});
      cell.addEventListener('mousemove', e => {{
        document.getElementById('mfi-tooltip').style.left = (e.clientX+15)+'px';
        document.getElementById('mfi-tooltip').style.top  = (e.clientY-12)+'px';
      }});
      cell.addEventListener('mouseleave', () => {{
        document.getElementById('mfi-tooltip').style.display='none';
      }});
      cell.addEventListener('click', () => mfiSetFilter(compound));

      grid.appendChild(cell);
    }});
  }});
}}

// ── Filter ─────────────────────────────────────────────────────────────────────
function mfiSetFilter(compound) {{
  mfiActiveFilter = compound;
  document.querySelectorAll('.mfi-filter-btn').forEach(btn => {{
    btn.classList.toggle('active', btn.textContent.trim() === compound ||
                                   (compound==='All' && btn.id==='mfi-btn-all') ||
                                   (compound==='None' && btn.id==='mfi-btn-none'));
  }});
  const plateName = document.getElementById('mfi-plate-select').value;
  MFI_CHANNELS.forEach(ch => {{
    mfiRenderBoxplot(`mfi-box-row-${{ch}}`, plateName, ch, 'row');
    mfiRenderBoxplot(`mfi-box-col-${{ch}}`, plateName, ch, 'col');
    mfiRenderPlatemap(ch, plateName);
  }});
}}

document.getElementById('mfi-btn-all').onclick  = () => mfiSetFilter('All');
document.getElementById('mfi-btn-none').onclick = () => mfiSetFilter('None');

// ── Main render ───────────────────────────────────────────────────────────────
function renderMFI() {{
  const plateName = document.getElementById('mfi-plate-select').value;
  mfiActivePlate  = plateName;
  mfiActiveFilter = 'All';
  document.querySelectorAll('.mfi-filter-btn:not(#mfi-btn-all):not(#mfi-btn-none)').forEach(b=>b.remove());
  document.getElementById('mfi-btn-all').classList.add('active');
  document.getElementById('mfi-btn-none').classList.remove('active');

  // Rebuild compound filter buttons for this plate
  const bar = document.getElementById('mfi-filter-bar');
  const cmpds = new Set();
  Object.values(DATA[plateName]?.wells || {{}}).forEach(m => {{ if (m.compound) cmpds.add(m.compound); }});
  [...cmpds].sort().forEach(c => {{
    const btn = document.createElement('button');
    btn.className   = 'mfi-filter-btn';
    btn.textContent = c;
    btn.style.borderColor = (mfiCompoundColors[c]||'#4060a0')+'AA';
    btn.onclick = () => mfiSetFilter(c);
    bar.appendChild(btn);
  }});

  const container = document.getElementById('mfi-content');
  container.innerHTML = '';

  if (!MFI_CHANNELS.length) {{
    container.innerHTML = '<p style="color:var(--muted);padding:16px;">No MFI data found — check that Nuclei.txt/Cells.txt exist in the measurements directory.</p>';
    return;
  }}

  MFI_CHANNELS.forEach(ch => {{
    const color   = MFI_COLORS[ch] || '#8ab0d0';
    const wellData = (MFI_DATA[plateName]||{{}})[ch]||{{}};
    if (!Object.keys(wellData).length) return;

    const allVals  = Object.values(wellData).flat();
    const allSorted = [...allVals].sort((a,b)=>a-b);
    const plateMed  = allSorted.length ? mfiQuantile(allSorted,0.5) : null;

    const sec = document.createElement('div');
    sec.className = 'mfi-channel-section';
    sec.innerHTML = `
      <div class="mfi-channel-header">
        <span class="mfi-ch-dot" style="background:${{color}}"></span>
        <h3>${{ch}} <span style="color:var(--muted);font-size:0.78rem;font-weight:normal;">
          — plate median: ${{plateMed!=null?plateMed.toFixed(5):'—'}}</span></h3>
      </div>
      <div class="mfi-body">
        <div class="mfi-boxplot-col">
          <div class="mfi-plot-box" id="mfi-box-row-${{ch}}"></div>
          <div class="mfi-plot-divider"></div>
          <div class="mfi-plot-box" id="mfi-box-col-${{ch}}"></div>
        </div>
        <div class="mfi-platemap-col">
          <div class="mfi-platemap-title">Platemap — median MFI</div>
          <div class="mfi-platemap-grid" id="mfi-grid-${{ch}}"></div>
        </div>
      </div>`;
    container.appendChild(sec);

    setTimeout(() => mfiRenderBoxplot(`mfi-box-row-${{ch}}`, plateName, ch, 'row'), 0);
    setTimeout(() => mfiRenderBoxplot(`mfi-box-col-${{ch}}`, plateName, ch, 'col'), 0);
    setTimeout(() => mfiRenderPlatemap(ch, plateName), 0);
  }});
}}

// ── 6. Flagged wells ──────────────────────────────────────────────────────────
(function buildCompoundToolbar() {{
  const toolbar  = document.getElementById('compound-toolbar');
  const allCmpds = new Set();
  PLATES.forEach(p => {{
    Object.values(DATA[p].wells).forEach(m => {{
      if (m.compound) allCmpds.add(m.compound);
    }});
  }});
  [...allCmpds].sort().forEach(cmpd => {{
    const btn = document.createElement('button');
    btn.className = 'cmpd-btn'; btn.textContent = cmpd; btn.dataset.cmpd = cmpd;
    btn.onclick = () => {{
      hideAllMode = false;
      btn.classList.toggle('active');
      btn.classList.contains('active') ? activeCompounds.add(cmpd) : activeCompounds.delete(cmpd);
      applyGlobalFilter();
    }};
    toolbar.appendChild(btn);
  }});
  document.getElementById('btn-show-all').onclick = () => {{
    hideAllMode = false; activeCompounds.clear();
    toolbar.querySelectorAll('.cmpd-btn:not(.ctrl)').forEach(b => b.classList.remove('active'));
    applyGlobalFilter();
  }};
  document.getElementById('btn-hide-all').onclick = () => {{
    hideAllMode = true; activeCompounds.clear();
    toolbar.querySelectorAll('.cmpd-btn:not(.ctrl)').forEach(b => b.classList.remove('active'));
    applyGlobalFilter();
  }};
}})();

function applyGlobalFilter() {{
  if (!currentPlate) return;
  renderMetrics(currentPlate);
  renderCounts(currentPlate);
  applyFlagFilter();
}}

(function buildFlagTable() {{
  const tbody = document.getElementById('flag-tbody');
  PLATES.forEach(p => {{
    const pd = DATA[p];
    Object.entries(pd.well_flags || {{}}).forEach(([well, flags]) => {{
      const m      = pd.wells[well] || {{}};
      const cmpd   = m.compound || '';
      const absTxt = (flags.abs||[]).join(', ') || '—';
      const adpTxt = (flags.adp||[]).join(', ') || '—';
      tbody.insertAdjacentHTML('beforeend',
        `<tr data-plate="${{p}}" data-well="${{well}}"
             data-compound="${{cmpd}}" data-metrics="${{[...(flags.abs||[]),...(flags.adp||[])].join(',')}}">
          <td>${{p}}</td><td>${{well}}</td><td>${{cmpd||'—'}}</td>
          <td style="color:var(--fail)">${{absTxt}}</td>
          <td style="color:var(--warn)">${{adpTxt}}</td>
        </tr>`);
    }});
  }});
}})();

function applyFlagFilter() {{
  const q = document.getElementById('flag-filter').value.toLowerCase();
  document.querySelectorAll('#flag-tbody tr').forEach(tr => {{
    const text   = [tr.dataset.plate,tr.dataset.well,tr.dataset.compound,tr.dataset.metrics].join(' ').toLowerCase();
    const cmpdOk = compoundVisible(tr.dataset.compound);
    tr.classList.toggle('hidden', !(!q||text.includes(q)) || !cmpdOk);
  }});
}}
document.getElementById('flag-filter').oninput = applyFlagFilter;

// ── Init ──────────────────────────────────────────────────────────────────────
renderPlate(0);
renderMFI();
</script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
    size_mb = output_path.stat().st_size / 1e6
    print(f"[html] -> {output_path}  ({size_mb:.1f} MB)")


# ── Collage class ──────────────────────────────────────────────────────────────
class Collage:
    def __init__(self, input_dir, output_path,
                 qc_tsv=None, platemap=None,
                 cohort_name: str = "Cohort",
                 plate_rows: int = 8, plate_cols: int = 12,
                 sites_per_well: int = 9, scale: float = 0.5,
                 workers: int = 8, band_height: int = 280,
                 font_size: int = 18, n_sigma: float = 3.0,
                 web_scale: float = 0.2):

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
        self.web_scale    = web_scale
        self.engine       = ThresholdEngine(n_sigma=n_sigma)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Auto-detect QC TSV
        self.qc = load_qc_tsv(qc_tsv)
        if not self.qc:
            # Search candidate paths in order of priority
            _qc_candidates = [
                self.input_dir / "Measurements" / "Image.txt",
                self.input_dir / "measurements" / "Image.txt",
                self.input_dir / "Images_MEASUREMENTS" / "Image.txt",
                self.input_dir / "Image.txt",
            ]
            for _p in _qc_candidates:
                if _p.exists():
                    print(f"[qc] Auto-detected: {_p}")
                    self.qc = load_qc_tsv(_p)
                    if self.qc:
                        break
            if not self.qc:
                _searched = ", ".join(str(p) for p in _qc_candidates)
                print(f"[warn] No QC measurements found. Searched: {_searched}")
                print(f"[warn] Use --qc to specify the path explicitly.")
        

        # Auto-detect Cells.txt / Nuclei.txt for MFI
        _meas_candidates = [
            self.input_dir / "Measurements",
            self.input_dir / "measurements",
            self.input_dir / "Images_MEASUREMENTS",
            self.input_dir,
        ]
        _cells_path  = None
        _nuclei_path = None
        for _md in _meas_candidates:
            if (_md / "Cells.txt").exists() or (_md / "Nuclei.txt").exists():
                _cells_path  = _md / "Cells.txt"  if (_md / "Cells.txt").exists()  else None
                _nuclei_path = _md / "Nuclei.txt" if (_md / "Nuclei.txt").exists() else None
                break
        self._source_dfs, self._channel_map = load_mfi_data(
            cells_path=_cells_path, nuclei_path=_nuclei_path)

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

    def _lookup_plate(self, mapping: dict, plate_name: str, label: str = "") -> dict:
        """
        Robust plate key lookup. Tries multiple normalizations of plate_name
        against the keys in mapping, and logs a clear warning on miss.

        Tries (in order):
          1. Exact match:           "P4"  -> key "P4"
          2. Strip leading P:       "P4"  -> key "4"
          3. Add leading P:         "4"   -> key "P4"
          4. Zero-padded variants:  "P04" -> key "P4" or "04" -> "4"
          5. Case-insensitive match
          6. Single-plate fallback: if mapping has exactly one entry, use it.
        """
        if not mapping:
            return {}

        # Build a normalised-key -> original-key lookup for the mapping
        def _norm(k: str) -> str:
            """Lowercase, strip leading zeros after optional P prefix."""
            k = str(k).strip().lower()
            if k.startswith("p"):
                num = k[1:].lstrip("0") or "0"
                return f"p{num}"
            return k.lstrip("0") or "0"

        norm_map: dict[str, str] = {_norm(k): k for k in mapping}
        candidates = [
            plate_name,                          # "P4" exact
            plate_name.lstrip("P").lstrip("p"),  # "4"
            f"P{plate_name.lstrip('P').lstrip('p')}",  # ensure "P" prefix
        ]

        for cand in candidates:
            # Exact
            if cand in mapping:
                return mapping[cand]
            # Normalised
            nk = _norm(cand)
            if nk in norm_map:
                return mapping[norm_map[nk]]

        # Single-entry fallback
        if len(mapping) == 1:
            key = next(iter(mapping))
            print(f"  [warn] {label}: no key matched '{plate_name}' "
                  f"(available: {list(mapping)[:5]}). "
                  f"Using sole entry '{key}'.")
            return mapping[key]

        print(f"  [warn] {label}: no key matched '{plate_name}'. "
              f"Available keys: {list(mapping)[:10]}. "
              f"Run with --qc to check column 'Metadata_Plate' values.")
        return {}

    def _render_plate(self, plate_name: str, montages: dict) -> None:
        if not montages:
            print(f"  No images for {plate_name}, skipping.")
            return

        plate_qc  = self._lookup_plate(self.qc,      plate_name, label="QC")
        plate_map = self._lookup_plate(self.platemap, plate_name, label="platemap")

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

        print(f"  [plate] {plate_name}  {collage.shape[1]}×{collage.shape[0]} px — overview + HTML only")

        # Accumulate HTML data
        n_wells = len(plate_qc)
        n_pass  = sum(1 for m in plate_qc.values() if _well_passes_all(m, self.engine))
        n_illum = sum(1 for m in plate_qc.values() if _well_passes_group(m, ILLUM_METRICS, self.engine))
        n_focus = sum(1 for m in plate_qc.values() if _well_passes_group(m, FOCUS_METRICS, self.engine))

        # Overview grid: real microscopy thumbnails for all wells
        overview_arr, ov_cw, ov_ch = _make_overview_grid(
            montages, plate_qc, plate_map,
            plate_rows=self.plate_rows, plate_cols=self.plate_cols)

        # Flagged well montages (absolute OR adaptive failures)
        flagged_b64 = {}
        for wl, m in plate_qc.items():
            is_flagged = any(
                self.engine.passes(m.get(col), mk, COL_TO_CHANNEL.get(col, "")) is False
                for mk, cols in METRIC_COLS.items() for col in cols
            )
            if is_flagged:
                r_idx = ord(wl[0]) - ord("A") + 1
                c_idx = int(wl[1:])
                b64   = _well_montage_b64(montages, (r_idx, c_idx),
                                          scale_factor=0.33)
                if b64:
                    flagged_b64[wl] = b64

        self._html_plates.append({
            "name":          plate_name,
            "collage_arr":   collage,
            "overview_arr":  overview_arr,
            "overview_cw":   ov_cw,
            "overview_ch":   ov_ch,
            "plate_qc":      plate_qc,
            "plate_map":     plate_map,
            "flagged_b64":   flagged_b64,
            "engine_adaptive": self.engine._adaptive,
            "mfi_data":      _aggregate_mfi_per_well(self._source_dfs, self._channel_map, plate_name) if self._channel_map else {},
            "pass_rate":     100 * n_pass / n_wells if n_wells else 0,
            "n_wells":       n_wells,
            "n_pass":        n_pass,
            "n_illum_pass":  n_illum,
            "n_focus_pass":  n_focus,
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
        web_scale      = args.web_scale,
    )
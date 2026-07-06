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

# NOTE: `render_report` is imported lazily inside the method that uses it (see
# the call site below), NOT here at module level. qc.report imports helpers and
# constants from THIS module at its own module level, so a top-level import here
# creates a circular import: when you run this script, execution reaches this
# line, jumps into qc.report, which tries to read names from this module that
# aren't defined yet (they're further down). The deferred import runs only when
# the report is actually generated, by which point this module is fully loaded.

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
    "MedianIntensity":     (None,  0.95),
}
THRESHOLDS_LOCAL_FOCUS: dict[str, tuple] = {
    "Hoechst":           (0.80,  None),
    "Syto":          (0.05,  None),
    "Golgi":       (0.03,  None),
    "ER":          (0.08,  None),
    "Mito":        (0.005, None),
    "Brightfield": (0.001, None),
}

CHANNELS       = ["Hoechst", "Syto", "Golgi", "ER", "Mito", "Brightfield"]
CHANNELS_EXTRA = []   # channels with focus metrics only

CHANNEL_LABELS = {
    "Hoechst": "Hoechst", "Syto": "Sy", "Golgi": "Go",
    "ER": "ER", "Mito": "Mi", "Brightfield": "BF",
}
CHANNEL_COLORS = {
    "Hoechst": (100, 160, 255), "Syto": (80, 220, 120), "Golgi": (255, 180, 60),
    "ER": (180, 90, 255), "Mito": (255, 80, 80), "Brightfield": (160, 160, 160),
}

ILLUM_METRICS  = ["PowerLogLogSlope", "MedianIntensity"]
BORDER_METRIC  = "PowerLogLogSlope"
METRIC_LABELS  = {
    "PowerLogLogSlope": "Focus", "MedianIntensity": "MaxInt",
}

COL_PASS    = (75,  215,  95)
COL_FAIL    = (255,  65,  65)
COL_NODATA  = (110, 110, 120)

COUNT_COLS = {
    "Raw": "Count_Raw_nuclei", "Filtered": "Count_Nuclei",
    "Cells": "Count_Cells",    "Artifacts": "Count_Illum_artifacts_filtered",
}
AREA_COL           = "ImageQuality_TotalArea_Brightfield"
DEFAULT_IMAGE_AREA = 1_166_400   # 1080×1080 px fallback


def _miq_col(metric: str, channel: str) -> str:
    if metric == "FocusScore":
        return f"ImageQuality_FocusScore_{channel}"
    return f"ImageQuality_{metric}_{channel}"


METRIC_COLS: dict[str, list[str]] = {
    mk: [_miq_col(mk, ch) for ch in CHANNELS]
    for mk in ("PowerLogLogSlope", "MedianIntensity", "FocusScore", "FocusScore")
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
    MedianIntensity      (None, 0.95)   Saturation guard.
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
    pct_cols  = list(dict.fromkeys(
        c for c in df.columns
        if c.startswith("ImageQuality_PercentMaximal_") or 
           c.startswith("ImageQuality_PercentMinimal_")
    ))
    sum_cols  = list(dict.fromkeys(
        c for c in df.columns
        if c.startswith("Count_") or c == AREA_COL
    ))
    gkeys = [plate_col, well_col] if plate_col else [well_col]
    grp   = df.groupby(gkeys)

    agg_mean = grp[mean_cols].mean().reset_index() if mean_cols else None
    agg_pct  = grp[pct_cols].mean().reset_index()  if pct_cols  else None
    agg_sum  = grp[sum_cols].sum().reset_index()   if sum_cols  else None

    agg = agg_mean
    if agg_pct is not None:
        agg = agg.merge(agg_pct, on=gkeys, how="left") if agg is not None else agg_pct
    if agg_sum is not None:
        agg = agg.merge(agg_sum, on=gkeys, how="left") if agg is not None else agg_sum

    all_cols = (mean_cols + pct_cols +
                [c for c in sum_cols if c not in mean_cols and c not in pct_cols])
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
# Hoechst (Hoechst) → Nuclei.txt  |  Syto, ER, Golgi, Mito → Cells.txt
# Column format: Intensity_MedianIntensity_<Channel>

MFI_COLS_CELLS = {
    "Syto":  ("Intensity_MedianIntensity_Syto",  "#50DC78"),
    "ER":    ("Intensity_MedianIntensity_ER",     "#B45AFF"),
    "Golgi": ("Intensity_MedianIntensity_Golgi",  "#FFB43C"),
    "Mito":  ("Intensity_MedianIntensity_Mito",   "#FF5050"),
}
MFI_COLS_NUCLEI = {
    "Hoechst": ("Intensity_MedianIntensity_Hoechst", "#64A0FF"),
}
MFI_CHANNEL_ORDER = ["Hoechst", "Syto", "ER", "Golgi", "Mito"]

RADIUS_COLS = {
    "Cells":  "AreaShape_MedianRadius",
    "Nuclei": "AreaShape_MedianRadius",
}

MFI_IMG_COLS = {
    "Hoechst": "ImageQuality_MedianIntensity_Hoechst",
    "Syto":    "ImageQuality_MedianIntensity_Syto",
    "ER":      "ImageQuality_MedianIntensity_ER",
    "Golgi":   "ImageQuality_MedianIntensity_Golgi",
    "Mito":    "ImageQuality_MedianIntensity_Mito",
}

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

    print(f"[DEBUG] cells_path: {cells_path}, nuclei_path: {nuclei_path}")

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
            print(f"[mfi] Nuclei.txt columns: {[c for c in df_nuclei.columns if 'Hoechst' in c or 'Intensity' in c][:10]}")
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
        def _norm_plate(s):
            return str(s).strip().lstrip("P").lstrip("0") or "0"

        mask = df["Metadata_Plate"].astype(str).apply(_norm_plate) == _norm_plate(plate_name)
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

def load_mfi_img(image_txt_path: "Path | None") -> "dict[str, dict[str, float]]":
    """
    Load per-well median MFI from Image.txt.
    Returns { well: { channel: median_across_sites } }
    """
    if image_txt_path is None or not image_txt_path.exists():
        print(f"[mfi_img] Image.txt not found: {image_txt_path}")
        return {}
    df = pd.read_csv(image_txt_path, sep="\t", low_memory=False)
    if "Metadata_Well" not in df.columns:
        return {}
    df["Metadata_Well"] = df["Metadata_Well"].astype(str).str.strip().str.upper()

    result = {}
    for well, grp in df.groupby("Metadata_Well"):
        result[well] = {}
        for ch, col in MFI_IMG_COLS.items():
            if col in grp.columns:
                vals = grp[col].dropna().values
                if len(vals):
                    result[well][ch] = float(np.median(vals))
    return result

def load_radius_data(cells_path: "Path | None",
                     nuclei_path: "Path | None",
                     plate_name: str) -> "dict[str, dict[str, float]]":
    """
    Load AreaShape_MedianRadius from Cells.txt and Nuclei.txt.
    Returns { 'Cells': { well: median_radius }, 'Nuclei': { well: median_radius } }
    """
    col = "AreaShape_MedianRadius"
    result = {}

    def _norm_plate(s):
        return str(s).strip().lstrip("P").lstrip("0") or "0"

    for label, path in [("Cells", cells_path), ("Nuclei", nuclei_path)]:
        if path is None or not path.exists():
            continue
        df = pd.read_csv(path, sep="\t", low_memory=False)
        if "Metadata_Well" not in df.columns or col not in df.columns:
            print(f"[radius] {col} not found in {path.name} — skipping.")
            continue
        df["Metadata_Well"]  = df["Metadata_Well"].astype(str).str.strip().str.upper()
        df["Metadata_Plate"] = (df["Metadata_Plate"].astype(str).str.strip()
                                if "Metadata_Plate" in df.columns else "Plate")
        mask = df["Metadata_Plate"].apply(_norm_plate) == _norm_plate(plate_name)
        plate_df = df[mask].copy()
        if plate_df.empty:
            if df["Metadata_Plate"].nunique() == 1:
                plate_df = df.copy()
            else:
                continue
        well_medians = (plate_df.groupby("Metadata_Well")[col]
                        .median()
                        .dropna()
                        .to_dict())
        result[label] = {str(w).strip().upper(): float(v)
                         for w, v in well_medians.items()}
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
    ALL_METRICS = ILLUM_METRICS
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
            yc = y
            for mk in ILLUM_METRICS:
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
        x_count = x0 + 4 + col_w * 2

        y = y_start
        for mk in ILLUM_METRICS:
            lbl = METRIC_LABELS[mk]
            draw.text((x_left, y), f"{lbl}:", fill=(170, 195, 225), font=fl)
            y += _text_h(draw, f"{lbl}:", fl) + 1
            for col in METRIC_COLS[mk]:
                v = metrics.get(col)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                ch    = COL_TO_CHANNEL.get(col, col)
                short = ch
                vc    = engine.val_color(v, mk, ch)
                txt   = f" {short}: {v:.3f}"
                draw.text((x_left, y + 1), txt, fill=(0, 0, 0), font=fv)
                draw.text((x_left, y),     txt, fill=vc,         font=fv)
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
        line(f"  Illumination:  {n_illum_p}/{n_wells} pass ({pct(n_illum_p)})  [Slope + MaxInt]", ic, fb)
        rule(); y += 4

        for grp_lbl, grp_col, grp_metrics in (
            ("ILLUMINATION METRICS", (255, 200, 80), ILLUM_METRICS),
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
                    ch_short = ch
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

            # Slope value (Hoechst channel only for brevity)
            Hoechst_slope_col = f"ImageQuality_PowerLogLogSlope_Hoechst"
            sv = metrics.get(Hoechst_slope_col)
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


def _mad_scores(col: str, plates_data: list) -> tuple:
    """Cohort-wide MAD score per well: (val - cohort_median) / cohort_MAD"""
    vals = []
    for pd_ in plates_data:
        for well_m in pd_["plate_qc"].values():
            v = well_m.get(col)
            if v is not None and not np.isnan(v):
                vals.append(v)
    if len(vals) < 4:
        return {}, None, None
    arr     = np.array(vals)
    coh_med = float(np.median(arr))
    coh_mad = float(np.median(np.abs(arr - coh_med))) or 1.0
    scores  = {}
    for pd_ in plates_data:
        pname = pd_["name"]
        plate_vals = []
        for well_m in pd_["plate_qc"].values():
            v = well_m.get(col)
            if v is not None and not np.isnan(v):
                plate_vals.append(v)
        pl_med = float(np.median(plate_vals)) if plate_vals else coh_med
        pl_mad = float(np.median(np.abs(np.array(plate_vals) - pl_med))) or 1.0
        scores[pname] = {}
        for well, well_m in pd_["plate_qc"].items():
            v = well_m.get(col)
            if v is None or np.isnan(v):
                continue
            scores[pname][well] = {
                "v":    round(float(v), 5),
                "z_pl": round((float(v) - pl_med)  / pl_mad,  2),
                "z_co": round((float(v) - coh_med) / coh_mad, 2),
            }
    return scores, round(coh_med, 5), round(coh_mad, 5)


def _median_cs(col: str, plates_data: list) -> tuple:
    """Traffic-light colorscale centered on cohort median ± 2/3 MAD."""
    vals = []
    for pd_ in plates_data:
        for well_m in pd_["plate_qc"].values():
            v = well_m.get(col)
            if v is not None and not np.isnan(v):
                vals.append(v)
    if len(vals) < 4:
        return "RdBu_r", None, None
    arr  = np.array(vals)
    med  = float(np.median(arr))
    mad  = float(np.median(np.abs(arr - med)))
    lo2  = med - 2*mad
    lo3  = med - 3*mad
    hi2  = med + 2*mad
    hi3  = med + 3*mad
    cmin = max(float(arr.min()), lo3 * 0.95)
    cmax = min(float(arr.max()), hi3 * 1.05)
    rng  = cmax - cmin
    if rng == 0:
        return "RdBu_r", cmin, cmax
    def pos(v):
        return round(max(0.0, min(1.0, (v - cmin) / rng)), 4)
    cs = [
        [0.0,      "#ff4444"],
        [pos(lo3), "#ff4444"],
        [pos(lo3), "#ffbe00"],
        [pos(lo2), "#ffbe00"],
        [pos(lo2), "#4bd760"],
        [pos(hi2), "#4bd760"],
        [pos(hi2), "#ffbe00"],
        [pos(hi3), "#ffbe00"],
        [pos(hi3), "#ff4444"],
        [1.0,      "#ff4444"],
    ]
    return cs, round(cmin, 5), round(cmax, 5)


def generate_html(cohort_name: str, plates_data: list[dict],
                  output_path: Path, web_scale: float = 0.2) -> None:
    """
    Generate self-contained HTML QC report.

    plates_data: list of dicts, one per plate:
        {name, collage_arr, plate_qc, plate_map, pass_rate,
         n_wells, n_pass, n_illum_pass, mfi_data}
    """
    plotly_js = _fetch_plotly_js()

    html_cols = (
            [c for cols in METRIC_COLS.values() for c in cols] +
            list(COUNT_COLS.values()) + [AREA_COL] +
            [f"ImageQuality_PercentMaximal_{ch}" for ch in CHANNELS] +
            [f"ImageQuality_PercentMinimal_{ch}" for ch in CHANNELS]
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
                    ch_lbl = ch
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
            "wells": _round_floats({
                well: {
                    "compound": pmap.get(well, ""),
                    **{c: m.get(c) for c in html_cols if c in m}
                }
                for well, m in pqc.items()
            }),
        }

    data_json = json.dumps(payload)

    # MAD scores para tooltip
    _qc_cols_for_scores = (
        [f"ImageQuality_PowerLogLogSlope_{ch}" for ch in CHANNELS] +
        [f"ImageQuality_PercentMaximal_{ch}"      for ch in CHANNELS] +
        [f"ImageQuality_PercentMinimal_{ch}"      for ch in CHANNELS] +
        [f"ImageQuality_MedianIntensity_{ch}"  for ch in CHANNELS]
    )
    calc_mad_scores  = {}
    for col in _qc_cols_for_scores:
        sc, cmed, cmad = _mad_scores(col, plates_data)
        calc_mad_scores [col] = sc
    mad_scores_json = json.dumps(calc_mad_scores)

    # Red well counts per metric group per plate
    def _red_wells(plates_data, cols, thresholds):
        """Count wells with at least one red value across cols."""
        result = {}
        for pd_ in plates_data:
            pname = pd_["name"]
            count = 0
            for well, m in pd_["plate_qc"].items():
                for col in cols:
                    v = m.get(col)
                    if v is None:
                        continue
                    lo, hi = thresholds.get(col, (None, None))
                    if (lo is not None and v < lo) or (hi is not None and v > hi):
                        count += 1
                        break  # one red col is enough to flag the well
            result[pname] = count
        return result

    slope_red   = _red_wells(plates_data,
        [f"ImageQuality_PowerLogLogSlope_{ch}" for ch in CHANNELS],
        {col: (-2.7, -1.3) for ch in CHANNELS
        for col in [f"ImageQuality_PowerLogLogSlope_{ch}"]})

    pctmax_red  = _red_wells(plates_data,
        [f"ImageQuality_PercentMaximal_{ch}" for ch in CHANNELS],
        {col: (None, 1.0) for ch in CHANNELS
        for col in [f"ImageQuality_PercentMaximal_{ch}"]})

    pctmin_red  = _red_wells(plates_data,
        [f"ImageQuality_PercentMinimal_{ch}" for ch in CHANNELS],
        {col: (None, 5.0) for ch in CHANNELS
        for col in [f"ImageQuality_PercentMinimal_{ch}"]})

    # medint_red uses MAD-score below — skip _red_wells call

    # Para MedianInt usamos MAD-score cohort > 3
    medint_red = {}
    for pd_ in plates_data:
        pname = pd_["name"]
        count = 0
        for well in pd_["plate_qc"]:
            for ch in CHANNELS:
                col = f"ImageQuality_MedianIntensity_{ch}"
                sc  = calc_mad_scores.get(col, {}).get(pname, {}).get(well)
                if sc and abs(sc["z_co"]) > 3:
                    count += 1
                    break
        medint_red[pname] = count

    red_counts_json = json.dumps({
        "slope":   slope_red,
        "pctmax":  pctmax_red,
        "pctmin":  pctmin_red,
        "medint":  medint_red,
    })

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

    SLOPE_CS = [
        [0.0,  "#ff4444"],   # -2.5  rojo
        [0.15, "#ff4444"],   # -2.35 rojo  (límite -2.7)
        [0.15, "#ffbe00"],   # -2.35 amarillo
        [0.35, "#ffbe00"],   # -2.15 amarillo (límite -2.5)
        [0.35, "#4bd760"],   # -2.15 verde
        [0.65, "#4bd760"],   # -1.85 verde  (límite -1.5)
        [0.65, "#ffbe00"],   # -1.85 amarillo
        [0.85, "#ffbe00"],   # -1.65 amarillo
        [0.85, "#ff4444"],   # -1.65 rojo
        [1.0,  "#ff4444"],   # -1.0  rojo
    ]

    slope_specs = json.dumps([
        {"col": col,
        "title": f"Slope — {ch}",
        "cmin": -3.0,
        "cmax": -1.0,
        "cs": SLOPE_CS}
        for ch in CHANNELS
        for col in [f"ImageQuality_PowerLogLogSlope_{ch}"]
    ])

    PCT_MAX_CS = [
        [0.0,  "#4bd760"],
        [0.05, "#4bd760"],  # 0.1% — verde hasta aquí
        [0.05, "#ffbe00"],
        [0.25, "#ffbe00"],  # 1%   — amarillo hasta aquí
        [0.25, "#ff4444"],
        [1.0,  "#ff4444"],  # > 1% — rojo
    ]
    PCT_MIN_CS = [
        [0.0,  "#4bd760"],
        [0.2,  "#4bd760"],  # 1%   — verde hasta aquí
        [0.2,  "#ffbe00"],
        [0.5,  "#ffbe00"],  # 5%   — amarillo hasta aquí
        [0.5,  "#ff4444"],
        [1.0,  "#ff4444"],
    ]

    pct_max_specs = json.dumps([
        {"col": col,
        "title": f"PctMaximal — {ch}",
        "cmin": 0,
        "cmax": 2.0,
        "cs": PCT_MAX_CS}
        for ch in CHANNELS
        for col in [f"ImageQuality_PercentMaximal_{ch}"]
    ])

    pct_min_specs = json.dumps([
        {"col": col,
        "title": f"PctMinimal — {ch}",
        "cmin": 0,
        "cmax": 10.0,
        "cs": PCT_MIN_CS}
        for ch in CHANNELS
        for col in [f"ImageQuality_PercentMinimal_{ch}"]
    ])

    mEDIANint_specs = json.dumps([
        {"col": col,
        "title": f"MedianInt — {ch}",
        "cmin": _median_cs(col, plates_data)[1],
        "cmax": _median_cs(col, plates_data)[2],
        "cs":   _median_cs(col, plates_data)[0]}
        for ch in CHANNELS
        for col in [f"ImageQuality_MedianIntensity_{ch}"]
    ])

    # Count columns: cohort-wide p2/p98 so all plates share the same colour scale
    count_ranges_json = json.dumps({
        col: list(_cohort_range(col, 0, None))
        for col in ["Count_Cells", "Count_Nuclei", "Count_Illum_artifacts_filtered"]
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
    mfi_img_payload = { pd_["name"]: pd_.get("mfi_img", {}) for pd_ in plates_data }
    mfi_img_json    = json.dumps(mfi_img_payload)
    radius_payload = { pd_["name"]: pd_.get("radius_data", {}) for pd_ in plates_data }
    radius_json    = json.dumps(radius_payload)

    html = f""""""


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
            _p = self.input_dir / "Measurements" / "Image.txt"
            if _p.exists():
                print(f"[qc] Auto-detected: {_p}")
                self.qc = load_qc_tsv(_p)
            if not self.qc:
                print(f"[warn] No QC measurements found. Searched: {_p}")
                print(f"[warn] Use --qc to specify the path explicitly.")
        

        # Auto-detect Cells.txt / Nuclei.txt for MFI
        _md = self.input_dir / "Measurements"
        self._measurements_dir = _md
        self._source_dfs, self._channel_map = load_mfi_data(
            cells_path  = _md / "Cells.txt"  if (_md / "Cells.txt").exists()  else None,
            nuclei_path = _md / "Nuclei.txt" if (_md / "Nuclei.txt").exists() else None,
        )
        self._mfi_img = load_mfi_img(_md / "Image.txt" if (_md / "Image.txt").exists() else None)

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

        # Generate HTML after all plates are processed.
        # Deferred import (breaks the qc.report <-> III_QC_collage circular import):
        # by now this module is fully initialized, so qc.report can safely pull
        # its helpers/constants from it.
        from qc.report import render_report

        html_name = f"{self.cohort_name}_QC_report.html"
        render_report(
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
            "mfi_img":  self._mfi_img,
            "radius_data":   load_radius_data(
                                 cells_path  = self._measurements_dir / "Cells.txt"  if (self._measurements_dir / "Cells.txt").exists()  else None,
                                 nuclei_path = self._measurements_dir / "Nuclei.txt" if (self._measurements_dir / "Nuclei.txt").exists() else None,
                                 plate_name  = plate_name,
                             ),
            "pass_rate":     100 * n_pass / n_wells if n_wells else 0,
            "n_wells":       n_wells,
            "n_pass":        n_pass,
            "n_illum_pass":  n_illum,
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
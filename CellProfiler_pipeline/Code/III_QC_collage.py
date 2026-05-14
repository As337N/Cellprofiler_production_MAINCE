"""
III_QC_collage.py
=================
Cell Painting plate collage builder — QC focused on illumination artefacts and focus.

Visual design
-------------
  Tile border   : PowerLogLogSlope quality  (green=good / orange=borderline / red=bad)
  Band 1 — ILLUM: PowerLogLogSlope + MaxIntensity  (artefacts + saturation)
  Band 2 — FOCUS: FocusScore + LocalFocusScore

  Values are coloured  GREEN (pass threshold)  /  RED (fail threshold).

Cell Painting QC thresholds (defaults):
  PowerLogLogSlope  : -2.5 to -1.0   (primary blur metric, log-log power spectrum)
  MaxIntensity      : < 0.95          (saturation guard)
  FocusScore        : > 0.005         (loose — catches only blank/extreme blur in fluorescence)
  LocalFocusScore   : > 0.8           (main per-channel focus metric; typical good range 1.5–3.0 for DNA)

Footer report:
  - Overall pass rate
  - Per-metric stats (pass %, mean ± SD) for Hoechst and Syto
  - Compound name per well (from platemap_*.csv, auto-detected)
  - Object counts: Raw nuclei / Filtered nuclei / Final cells / Artifacts
  - Artifact density per well (artifacts / image area)
  - Failing wells with reasons

Usage
-----
    python III_QC_collage.py -i /output/QC/Images -o /output/QC/Collages
    # --qc optional: auto-detected from -i if Image.txt present
    # --platemap optional: auto-detected from -i if platemap_*.csv present

Requirements
------------
    pip install pandas numpy pillow tifffile imageio scikit-image
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import resize

# ── Fonts ─────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_FONT_SEARCH = {
    "bold": [
        _SCRIPT_DIR / "fonts" / "DejaVuSans-Bold.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans-Bold.ttf"),
        Path("/Library/Fonts/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans-Bold.ttf"),
    ],
    "regular": [
        _SCRIPT_DIR / "fonts" / "DejaVuSans.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans.ttf"),
        Path("/Library/Fonts/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans.ttf"),
    ],
}
_font_cache = {}

def _font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    for path in _FONT_SEARCH["bold" if bold else "regular"]:
        if path.exists():
            try:
                f = ImageFont.truetype(str(path), size)
                _font_cache[key] = f
                return f
            except (OSError, IOError):
                continue
    f = ImageFont.load_default()
    _font_cache[key] = f
    return f

def _text_h(draw, text, font):
    try:
        return draw.textbbox((0, 0), text, font=font)[3]
    except AttributeError:
        return font.size if hasattr(font, 'size') else 14


# ── QC config ─────────────────────────────────────────────────────────────────

# ── QC Thresholds ────────────────────────────────────────────────────────────
# metric_key → (min, max), None = no bound
#
# PowerLogLogSlope  (-2.5, -1.0)
#   Log-log power spectrum slope. Recommended as the primary blur metric for
#   most imaging modalities (Field 1997; Bray et al. 2012). A more negative
#   slope indicates more low-frequency content (blur). Values outside this
#   range indicate either extremely blurry (-2.5) or over-sharpened / noisy
#   (-1.0) images. Source: CellProfiler MeasureImageQuality documentation.
#
# MaxIntensity  (None, 0.95)
#   Maximum pixel intensity (0–1 scale after CellProfiler rescaling). Values
#   ≥ 0.95 indicate saturation artefacts that compress dynamic range and
#   distort morphological measurements. Source: Bray et al. 2012.
#
# FocusScore  (0.005, None)
#   Normalised variance of the full image (variance / mean intensity).
#   Optimised for brightfield / DIC autofocusing where global intensity is
#   constant (Sun 2004). In fluorescence Cell Painting images the mean
#   intensity is very low (dark background + small bright objects), so the
#   ratio is structurally small even for perfectly focused images — typical
#   good-quality fluorescence values are 0.01–0.10. The threshold is set
#   deliberately loose (0.005) to catch only extreme cases (blank / fully
#   out-of-focus images) without penalising normal fluorescence images.
#   Use LocalFocusScore for channel-level focus decisions.
#   Source: CellProfiler docs — "less useful for comparison of different
#   fields of view … for distinguishing extremely blurry images it performs
#   well."
#
# LocalFocusScore  (0.8, None)
#   Mean normalised variance across non-overlapping image tiles. More robust
#   than FocusScore for comparing different fields of view (CellProfiler
#   docs). In Cell Painting:
#     DNA (Hoechst): well-focused nuclei → typically 1.5–3.0
#     Fluorescence channels: 0.3–1.5 depending on staining density
#     BF / sparse channels: 0.05–0.3
#   Threshold 0.8 separates acceptably focused images from blurry / poorly
#   segmented ones across the majority of Cell Painting channels. Channels
#   with low staining density (Mito, BF) may systematically fall below this
#   value and should be inspected visually before exclusion.
#   Source: CellProfiler docs + empirical Cell Painting QC practice
#   (Bray et al. 2012, J Biomol Screen).

# Metrics with a single global threshold (channel-independent)
THRESHOLDS = {
    "PowerLogLogSlope": (-2.5,  -1.0),   # primary blur metric
    "MaxIntensity":     (None,   0.95),   # saturation guard
    "FocusScore":       (0.005,  None),   # loose — blank/extreme blur only
}

# LocalFocusScore: per-channel thresholds (min, None).
# Rationale: signal density varies by channel — DNA (Hoechst) produces
# high local variance due to compact bright nuclei; sparse/diffuse channels
# (Syto, Mito, BF) are structurally lower and need calibrated thresholds
# to avoid rejecting well-focused images with low staining density.
#
#   DNA        0.80 — compact nuclei, high variance expected
#   Syto       0.05 — diffuse cytoplasmic stain, lower variance
#   Golgi_c    0.03 — punctate/sparse signal
#   ER_c       0.08 — best non-DNA channel in typical Cell Painting (μ≈0.13)
#   Mito_c     0.005 — very low signal density
#   Brightfield_c 0.001 — BF channel near-zero in fluorescence acquisitions
THRESHOLDS_LOCAL_FOCUS: dict[str, tuple] = {
    "DNA":           (0.80,  None),
    "Syto":          (0.05,  None),
    "Golgi_c":       (0.03,  None),
    "ER_c":          (0.08,  None),
    "Mito_c":        (0.005, None),
    "Brightfield_c": (0.001, None),
}

# Channel names match CellProfiler ImageQuality column suffixes exactly.
# DNA = Hoechst nucleus channel (has full ImageQuality_* metrics incl. Slope/MaxInt).
# Brightfield_c = brightfield channel (has Slope/MaxInt too in this pipeline).
# Note: Hoechst only appears as Intensity_* columns (not ImageQuality_*).
CHANNELS = ["DNA", "Syto", "Golgi_c", "ER_c", "Mito_c", "Brightfield_c"]
CHANNELS_FOCUS_ONLY = []   # all channels have full metrics in this pipeline

# Short display labels per channel (used in band and footer)
CHANNEL_LABELS = {
    "DNA":          "DNA",
    "Syto":         "Sy",
    "Golgi_c":      "Go",
    "ER_c":         "ER",
    "Mito_c":       "Mi",
    "Brightfield_c":"BF",
}

# Per-channel accent colours (used in footer table headers)
CHANNEL_COLORS = {
    "DNA":          (100, 160, 255),   # blue
    "Syto":         (80,  220, 120),   # green
    "Golgi_c":      (255, 180,  60),   # amber
    "ER_c":         (180,  90, 255),   # purple
    "Mito_c":       (255,  80,  80),   # red
    "Brightfield_c":(160, 160, 160),   # grey
}

# Column lists: metric_key → [col_ch1, col_ch2, ...]  (N channels)
def _miq_col(metric: str, channel: str) -> str:
    """Build ImageQuality column name from metric and channel."""
    suffix = {
        "LocalFocusScore": f"ImageQuality_LocalFocusScore_{channel}_10",
    }.get(metric, f"ImageQuality_{metric}_{channel}")
    return suffix

METRIC_COLS: dict[str, list[str]] = {
    mk: [_miq_col(mk, ch) for ch in CHANNELS]
    for mk in ("PowerLogLogSlope", "MaxIntensity", "FocusScore", "LocalFocusScore")
}
# Add Brightfield to focus-only metrics
for mk in ("FocusScore", "LocalFocusScore"):
    METRIC_COLS[mk] += [_miq_col(mk, ch) for ch in CHANNELS_FOCUS_ONLY]

# Convenience: map each column back to its channel label
COL_TO_CHANNEL: dict[str, str] = {}
for mk, cols in METRIC_COLS.items():
    for col in cols:
        for ch in CHANNELS + CHANNELS_FOCUS_ONLY:
            if ch in col:
                COL_TO_CHANNEL[col] = ch

# The two metric groups shown in the band
ILLUM_METRICS = ["PowerLogLogSlope", "MaxIntensity"]
FOCUS_METRICS = ["FocusScore", "LocalFocusScore"]

# Primary metric driving tile border colour (uses CHANNELS only, not BF)
BORDER_METRIC = "PowerLogLogSlope"

# Display labels
METRIC_LABELS = {
    "PowerLogLogSlope": "Slope",
    "MaxIntensity":     "MaxInt",
    "FocusScore":       "Focus",
    "LocalFocusScore":  "LocalFoc",
}

# Colours
COL_PASS   = (75,  215,  95)
COL_FAIL   = (255,  65,  65)
COL_NODATA = (110, 110, 120)
COL_ILLUM_HDR = (255, 200,  80)   # amber header for illumination band
COL_FOCUS_HDR = (100, 180, 255)   # blue header for focus band


def _passes(value, metric_key: str, channel: str = ""):
    """
    Return True/False/None for a metric value.
    LocalFocusScore uses per-channel thresholds from THRESHOLDS_LOCAL_FOCUS;
    all other metrics use the global THRESHOLDS dict.
    channel: CellProfiler channel name (e.g. "DNA", "Syto") — required for
             LocalFocusScore, ignored for other metrics.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if metric_key == "LocalFocusScore":
        lo, hi = THRESHOLDS_LOCAL_FOCUS.get(channel, (None, None))
    else:
        lo, hi = THRESHOLDS.get(metric_key, (None, None))
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True

def _val_color(value, metric_key: str, channel: str = "") -> tuple:
    p = _passes(value, metric_key, channel)
    return COL_NODATA if p is None else (COL_PASS if p else COL_FAIL)


# ── TSV loader ────────────────────────────────────────────────────────────────

def load_qc_tsv(tsv_path) -> dict:
    if tsv_path is None:
        return {}
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        print(f"[warn] QC file not found: {tsv_path}")
        return {}

    df = pd.read_csv(tsv_path, sep="\t")
    print(f"[qc] Loaded {len(df)} rows from {tsv_path.name}")

    plate_col = "Metadata_Plate" if "Metadata_Plate" in df.columns else None
    well_col  = "Metadata_Well"  if "Metadata_Well"  in df.columns else None
    if well_col is None:
        print("[warn] Metadata_Well not found — enrichment disabled.")
        return {}

    # Image quality metrics → mean across sites (per-image properties)
    mean_cols = [c for cols in METRIC_COLS.values() for c in cols if c in df.columns]
    mean_cols = list(dict.fromkeys(mean_cols))

    # Object counts → sum across sites (total objects in the well)
    # TotalArea → sum across sites (total area imaged per well)
    sum_cols = [c for c in df.columns
                if c.startswith("Count_") or c == "ImageQuality_TotalArea_Brightfield_c"]
    sum_cols = list(dict.fromkeys(sum_cols))

    gkeys = [plate_col, well_col] if plate_col else [well_col]
    grp   = df.groupby(gkeys)

    agg_mean = grp[mean_cols].mean().reset_index() if mean_cols else None
    agg_sum  = grp[sum_cols].sum().reset_index()   if sum_cols  else None

    # Merge mean and sum aggregations on group keys
    if agg_mean is not None and agg_sum is not None:
        agg = agg_mean.merge(agg_sum, on=gkeys, how="left")
    elif agg_mean is not None:
        agg = agg_mean
    else:
        agg = agg_sum

    all_cols = mean_cols + [c for c in sum_cols if c not in mean_cols]

    result = defaultdict(dict)
    for _, row in agg.iterrows():
        plate = str(row[plate_col]).strip() if plate_col else "Plate"
        well  = str(row[well_col]).strip().upper()
        result[plate][well] = {c: row[c] for c in all_cols if c in row.index}

    total = sum(len(v) for v in result.values())
    print(f"[qc] {total} wells across {len(result)} plate(s) ready.")
    return result


# ── Platemap loader ───────────────────────────────────────────────────────────

def load_platemap(platemap_path) -> dict:
    """
    Load a platemap CSV (platemap_*.csv) and return a nested dict:
      { plate: { well: compound_name } }
    Columns expected: Metadata_Well, Metadata_Perturbation, Metadata_Plate (optional).
    """
    if platemap_path is None:
        return {}
    platemap_path = Path(platemap_path)
    if not platemap_path.exists():
        print(f"[warn] Platemap not found: {platemap_path}")
        return {}

    df = pd.read_csv(platemap_path)
    print(f"[platemap] Loaded {len(df)} wells from {platemap_path.name}")

    well_col     = next((c for c in df.columns if "Well"        in c), None)
    compound_col = next((c for c in df.columns if "Compound"    in c
                         or "Perturbation" in c), None)
    plate_col    = next((c for c in df.columns if "Plate"       in c), None)

    if well_col is None or compound_col is None:
        print(f"[warn] Platemap missing Well or Compound/Perturbation column — skipping.")
        print(f"[warn] Found columns: {list(df.columns)}")
        return {}

    result = defaultdict(dict)
    for _, row in df.iterrows():
        well     = str(row[well_col]).strip().upper()
        compound = str(row[compound_col]).strip()
        # Normalise plate key: accept both "19" and "P19" → store as "P19"
        raw_plate = str(row[plate_col]).strip() if plate_col else "Plate"
        plate     = raw_plate if raw_plate.startswith("P") else f"P{raw_plate}"
        result[plate][well] = compound

    total = sum(len(v) for v in result.values())
    print(f"[platemap] {total} well→compound mappings across {len(result)} plate(s).")
    return result



def _slope_range(plate_qc: dict) -> tuple:
    cols = METRIC_COLS[BORDER_METRIC]
    vals = [v for m in plate_qc.values()
            for col in cols
            if (v := m.get(col)) is not None and not np.isnan(v)]
    lo, hi = THRESHOLDS[BORDER_METRIC]
    return (min(vals) if vals else lo, max(vals) if vals else hi)


def _slope_to_rgb(well_metrics: dict | None) -> tuple:
    """
    Colour tile border by PowerLogLogSlope pass/fail across ALL channels:
      All pass  → bright green
      Mixed     → orange
      All fail  → deep red
      No data   → dark grey
    """
    results = []
    for col in METRIC_COLS[BORDER_METRIC]:
        v = well_metrics.get(col) if well_metrics else None
        p = _passes(v, BORDER_METRIC)
        if p is not None:
            results.append(p)
    if not results:
        return (45, 45, 55)
    n_pass = sum(results)
    if n_pass == len(results):
        return (55, 205, 80)
    elif n_pass == 0:
        return (215, 35, 35)
    else:
        return (255, 145, 0)


# ── Object count helpers ──────────────────────────────────────────────────────

# Maps friendly label → CellProfiler Count column name
COUNT_COLS = {
    "Raw":       "Count_Raw_nuclei",
    "Filtered":  "Count_Nuclei",       # nuclei after filtering
    "Cells":     "Count_Cells",
    "Artifacts": "Count_Illum_artifacts",
}

# Image area column (pixels²) — used for artifact density
AREA_COL = "ImageQuality_TotalArea_Brightfield_c"
# Fallback fixed area if column absent (1080×1080 default)
DEFAULT_IMAGE_AREA = 1166400


def _artifact_density(metrics: dict) -> float | None:
    """
    Artifacts per 1000 px² across the whole well.
    Both Count_Illum_artifacts and TotalArea are summed across sites in
    load_qc_tsv, so this correctly reflects the full well, not a single site.
    """
    n = metrics.get(COUNT_COLS["Artifacts"])
    if n is None or np.isnan(n):
        return None
    area = metrics.get(AREA_COL) or DEFAULT_IMAGE_AREA
    if area <= 0:
        return None
    return (n / area) * 1000


def _count_summary(metrics: dict) -> list[tuple[str, str]]:
    """
    Return list of (label, value_str) for object counts and artifact density.
    Counts are well-level totals (sum across all sites).
    """
    rows = []
    for label, col in COUNT_COLS.items():
        v = metrics.get(col)
        rows.append((label, "—" if (v is None or (isinstance(v, float) and np.isnan(v)))
                     else str(int(round(v)))))
    density = _artifact_density(metrics)
    rows.append(("Art/kpx²", "—" if density is None else f"{density:.2f}"))
    return rows



def well_label(row: int, col: int) -> str:
    return f"{chr(ord('A') + row - 1)}{col:02d}"

def load_and_downscale(path: Path, scale: float = 0.5) -> np.ndarray:
    img = tiff.imread(path)
    if img.dtype != np.uint8:
        p_max = img.max()
        img = (img / p_max * 255).astype(np.uint8) if p_max > 0 else img.astype(np.uint8)
    if scale != 1.0:
        img = resize(img, (int(img.shape[0] * scale), int(img.shape[1] * scale)),
                     preserve_range=True, anti_aliasing=True).astype(np.uint8)
    return img

def parse_name(fname: str) -> tuple:
    name = fname.replace(".tiff", "")
    rc, site, _ = name.split("-")
    return int(rc[:3]), int(rc[3:6]), int(site)

def build_well_montage(args: tuple) -> tuple:
    (r, c), sites, scale, spw = args
    tiles = [load_and_downscale(sites[s], scale)
             for s in range(1, spw + 1) if s in sites]
    if not tiles:
        return (r, c), None
    n = max(1, int(np.ceil(np.sqrt(len(tiles)))))
    blank = np.zeros_like(tiles[0])
    while len(tiles) % n:
        tiles.append(blank)
    rows = [np.concatenate(tiles[i:i + n], axis=1) for i in range(0, len(tiles), n)]
    return (r, c), np.concatenate(rows, axis=0)

def _make_tile(tile: np.ndarray, border_rgb: tuple, metrics=None,
               border_w: int = 6, font_size: int = 14) -> np.ndarray:
    """Clean tile with focus-quality coloured border only."""
    img    = tile if tile.ndim == 3 else np.stack([tile] * 3, axis=-1)
    canvas = Image.fromarray(img.copy())
    draw   = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, img.shape[1] - 1, img.shape[0] - 1],
                   outline=border_rgb, width=border_w)
    return np.array(canvas)


# ── Band builder (shared for ILLUM and FOCUS) ─────────────────────────────────

def _measure_band_height(well_labels_in_row: list, plate_qc: dict,
                          font_size: int, tile_width: int,
                          plate_map: dict | None = None) -> int:
    """Dry-run to compute the minimum band height needed so nothing is clipped."""
    dummy = Image.new("RGB", (tile_width, 10))
    draw  = ImageDraw.Draw(dummy)
    font_well = _font(font_size + 2, bold=True)
    font_lbl  = _font(font_size - 1, bold=True)
    font_val  = _font(font_size - 1, bold=False)

    max_y = 0
    for wlabel in well_labels_in_row:
        metrics  = plate_qc.get(wlabel)
        compound = (plate_map or {}).get(wlabel, "")

        try:
            well_h = draw.textbbox((0, 0), wlabel, font=font_well)[3]
        except AttributeError:
            well_h = font_size + 2
        y = 7 + well_h + 3

        if compound:
            try:
                cmp_h = draw.textbbox((0, 0), compound, font=font_val)[3]
            except AttributeError:
                cmp_h = font_size - 1
            y += cmp_h + 4

        if metrics:
            for group in (ILLUM_METRICS, FOCUS_METRICS):
                yc = y
                for mk in group:
                    lbl = METRIC_LABELS.get(mk, mk)
                    yc += _text_h(draw, f"{lbl}:", font_lbl) + 1
                    for col in METRIC_COLS[mk]:
                        v = metrics.get(col)
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            continue
                        yc += _text_h(draw, " xx: 0.000", font_val) + 1
                    yc += 3
                max_y = max(max_y, yc)
            yc = y
            yc += _text_h(draw, "Objects:", font_lbl) + 2
            for _ in _count_summary(metrics):
                yc += _text_h(draw, " xx: 000", font_val) + 1
            max_y = max(max_y, yc)
        else:
            max_y = max(max_y, y + 10)

    return max_y + 16


def _make_band(width: int, well_labels_in_row: list, plate_qc: dict,
               band_height: int, font_size: int, tile_width: int,
               plate_map: dict | None = None) -> np.ndarray:
    """
    Single unified QC band. band_height is the *minimum*; auto-expands to fit
    all channels (DNA, Syto, Golgi_c, ER_c, Mito_c, Brightfield_c).
    """
    ALL_METRICS = ILLUM_METRICS + FOCUS_METRICS

    needed      = _measure_band_height(well_labels_in_row, plate_qc,
                                       font_size, tile_width, plate_map)
    band_height = max(band_height, needed)

    band      = Image.new("RGB", (width, band_height), color=(12, 12, 18))
    draw      = ImageDraw.Draw(band)
    font_well = _font(font_size + 2, bold=True)
    font_lbl  = _font(font_size - 1, bold=True)
    font_val  = _font(font_size - 1, bold=False)

    for idx, wlabel in enumerate(well_labels_in_row):
        metrics = plate_qc.get(wlabel)
        x0, x1  = idx * tile_width, idx * tile_width + tile_width - 1

        # Overall pass fraction → background tint + accent
        pass_vals = []
        for mk in ALL_METRICS:
            for col in METRIC_COLS[mk]:
                v  = metrics.get(col) if metrics else None
                ch = COL_TO_CHANNEL.get(col, "")
                p  = _passes(v, mk, ch)
                if p is not None:
                    pass_vals.append(p)

        if pass_vals:
            frac   = sum(pass_vals) / len(pass_vals)
            tint   = (20, 55, 25) if frac == 1.0 else (55, 15, 15) if frac == 0.0 else (50, 35, 10)
            accent = COL_PASS     if frac == 1.0 else (COL_FAIL     if frac == 0.0 else (255, 155, 0))
        else:
            tint, accent = (18, 18, 24), (60, 70, 90)

        draw.rectangle([x0, 0, x1, band_height - 1], fill=tint)
        draw.rectangle([x0, 0, x1, 5], fill=accent)

        # Well label
        wc = tuple(min(255, int(c * 1.4 + 50)) for c in accent)
        draw.text((x0 + 4, 7), wlabel, fill=wc, font=font_well)
        try:
            well_h = draw.textbbox((0, 0), wlabel, font=font_well)[3]
        except AttributeError:
            well_h = font_size + 2
        y_start = 7 + well_h + 3

        # Compound name (from platemap)
        compound = (plate_map or {}).get(wlabel, "")
        if compound:
            # Truncate to fit tile width
            max_chars = max(6, (tile_width - 8) // max(1, (font_size - 3)))
            disp = compound if len(compound) <= max_chars else compound[:max_chars - 1] + "…"
            draw.text((x0 + 4, y_start), disp, fill=(200, 220, 180), font=font_val)
            try:
                cmp_h = draw.textbbox((0, 0), disp, font=font_val)[3]
            except AttributeError:
                cmp_h = font_size - 1
            y_start += cmp_h + 4

        if not metrics:
            draw.line([x1, 0, x1, band_height - 1], fill=(28, 28, 36), width=1)
            continue

        # Three-column layout: ILLUM left, FOCUS center, COUNTS right
        col_w   = (tile_width - 10) // 3
        x_left  = x0 + 4
        x_right = x0 + 4 + col_w
        x_count = x0 + 4 + col_w * 2

        for col_x, group in ((x_left, ILLUM_METRICS), (x_right, FOCUS_METRICS)):
            y = y_start
            for mk in group:
                lbl = METRIC_LABELS.get(mk, mk)
                draw.text((col_x, y), f"{lbl}:", fill=(170, 195, 225), font=font_lbl)
                y += _text_h(draw, f"{lbl}:", font_lbl) + 1

                for col in METRIC_COLS[mk]:
                    v = metrics.get(col)
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        continue
                    ch    = COL_TO_CHANNEL.get(col, col)
                    short = CHANNEL_LABELS.get(ch, ch[:2])
                    vc    = _val_color(v, mk, ch)
                    txt   = f" {short}: {v:.3f}"
                    draw.text((col_x, y + 1), txt, fill=(0, 0, 0), font=font_val)
                    draw.text((col_x, y),     txt, fill=vc,         font=font_val)
                    y += _text_h(draw, txt, font_val) + 1

                y += 3

        # Counts column (right)
        y = y_start
        draw.text((x_count, y), "Objects:", fill=(170, 195, 225), font=font_lbl)
        y += _text_h(draw, "Objects:", font_lbl) + 2
        for lbl, val in _count_summary(metrics):
            # Artifact density gets a special colour if high
            if lbl == "Art/kpx²":
                try:
                    fv = float(val)
                    vc = COL_FAIL if fv > 0.05 else ((255, 190, 0) if fv > 0.02 else COL_PASS)
                except ValueError:
                    vc = COL_NODATA
            else:
                vc = (200, 210, 220)
            txt = f" {lbl}: {val}"
            draw.text((x_count, y + 1), txt, fill=(0, 0, 0), font=font_val)
            draw.text((x_count, y),     txt, fill=vc,         font=font_val)
            y += _text_h(draw, txt, font_val) + 1

        draw.line([x1, 0, x1, band_height - 1], fill=(28, 28, 36), width=1)

    return np.array(band)


# ── Header ────────────────────────────────────────────────────────────────────

def make_header(width: int, title: str, font_size: int = 20) -> np.ndarray:
    header = Image.new("RGB", (width, 58), color=(10, 12, 22))
    draw   = ImageDraw.Draw(header)
    draw.text((12, 14), title, fill=(180, 220, 255), font=_font(font_size, bold=True))
    draw.rectangle([0, 55, width - 1, 57], fill=(40, 60, 100))

    legend = [
        ("Pass",       COL_PASS),
        ("Fail",       COL_FAIL),
        ("Both pass",  (55, 205, 80)),
        ("Mixed",      (255, 145, 0)),
        ("Both fail",  (215, 35, 35)),
    ]
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


# ── Report footer ─────────────────────────────────────────────────────────────

def _well_passes_all(metrics: dict) -> bool:
    for mk, cols in METRIC_COLS.items():
        for col in cols:
            v  = metrics.get(col)
            ch = COL_TO_CHANNEL.get(col, "")
            if v is not None and not np.isnan(v) and not _passes(v, mk, ch):
                return False
    return True

def _well_passes_group(metrics: dict, group: list) -> bool:
    for mk in group:
        for col in METRIC_COLS[mk]:
            v  = metrics.get(col)
            ch = COL_TO_CHANNEL.get(col, "")
            if v is not None and not np.isnan(v) and not _passes(v, mk, ch):
                return False
    return True


def make_report_footer(width: int, plate_name: str,
                        plate_qc: dict, font_size: int = 16,
                        plate_map: dict | None = None) -> np.ndarray:
    n_wells   = len(plate_qc)
    n_pass    = sum(1 for m in plate_qc.values() if _well_passes_all(m))
    n_illum_p = sum(1 for m in plate_qc.values() if _well_passes_group(m, ILLUM_METRICS))
    n_focus_p = sum(1 for m in plate_qc.values() if _well_passes_group(m, FOCUS_METRICS))
    pct       = lambda n: f"{100 * n / n_wells:.1f}%" if n_wells else "—"

    # Per-metric stats: {metric: {channel: [values]}}
    all_chs = CHANNELS + CHANNELS_FOCUS_ONLY
    stats = {mk: {ch: [] for ch in all_chs} for mk in METRIC_COLS}
    for m in plate_qc.values():
        for mk, cols in METRIC_COLS.items():
            for col in cols:
                ch = COL_TO_CHANNEL.get(col, col)
                v  = m.get(col)
                if v is not None and not np.isnan(v):
                    stats[mk][ch].append(v)

    # Failing wells
    failing = []
    for wl, m in sorted(plate_qc.items()):
        reasons = []
        for mk, cols in METRIC_COLS.items():
            for col in cols:
                ch_key = COL_TO_CHANNEL.get(col, "")
                ch     = CHANNEL_LABELS.get(ch_key, "?")
                v      = m.get(col)
                if v is not None and not np.isnan(v) and not _passes(v, mk, ch_key):
                    reasons.append(f"{METRIC_LABELS.get(mk, mk)}/{ch}")
        if reasons:
            failing.append((wl, reasons))

    # ── Build footer image ────────────────────────────────────────────────────
    font_title = _font(font_size + 6, bold=True)
    font_sec   = _font(font_size + 1, bold=True)
    font_b     = _font(font_size - 1, bold=True)
    font_r     = _font(font_size - 1, bold=False)

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

        # Title
        line(f"  QC REPORT — {plate_name}", (180, 210, 255), font_title)
        rule((50, 75, 140))

        # Overall
        oc = COL_PASS if n_pass / max(n_wells, 1) >= 0.8 else \
             (COL_FAIL if n_pass / max(n_wells, 1) < 0.5 else (255, 190, 0))
        line(f"  Overall:       {n_pass}/{n_wells} wells pass all metrics  ({pct(n_pass)})", oc, font_sec)

        # Illumination vs Focus sub-totals
        ic = COL_PASS if n_illum_p / max(n_wells, 1) >= 0.8 else \
             (COL_FAIL if n_illum_p / max(n_wells, 1) < 0.5 else (255, 190, 0))
        fc = COL_PASS if n_focus_p / max(n_wells, 1) >= 0.8 else \
             (COL_FAIL if n_focus_p / max(n_wells, 1) < 0.5 else (255, 190, 0))
        line(f"  Illumination:  {n_illum_p}/{n_wells} wells pass  ({pct(n_illum_p)})   "
             f"[Slope + MaxInt]", ic, font_b)
        line(f"  Focus:         {n_focus_p}/{n_wells} wells pass  ({pct(n_focus_p)})   "
             f"[FocusScore + LocalFocusScore]", fc, font_b)
        rule()
        y += 4

        # Per-metric detail — grouped
        for group_label, group_color, group_metrics in (
            ("ILLUMINATION METRICS", COL_ILLUM_HDR, ILLUM_METRICS),
            ("FOCUS METRICS",        COL_FOCUS_HDR, FOCUS_METRICS),
        ):
            line(f"  {group_label}", group_color, font_sec)

            # Table header
            hdr = f"    {'Metric':<22}  {'Channel':<8}  {'Threshold':<14}  {'Pass':>12}  {'mean ± SD':<22}"
            line(hdr, (110, 140, 175), font_b)

            for mk in group_metrics:
                metric_lbl = METRIC_LABELS.get(mk, mk)

                first = True
                for col in METRIC_COLS[mk]:
                    ch   = COL_TO_CHANNEL.get(col, col)
                    vals = stats[mk].get(ch, [])
                    ch_short = CHANNEL_LABELS.get(ch, ch[:4])
                    ch_color = CHANNEL_COLORS.get(ch, (150, 170, 200))

                    # Per-channel threshold string for LocalFocusScore
                    if mk == "LocalFocusScore":
                        lo, hi = THRESHOLDS_LOCAL_FOCUS.get(ch, (None, None))
                    else:
                        lo, hi = THRESHOLDS.get(mk, (None, None))
                    tstr = (f"> {lo}" if hi is None else
                            f"< {hi}" if lo is None else f"{lo} to {hi}")

                    if vals:
                        np_ = sum(1 for v in vals if _passes(v, mk, ch))
                        pp  = 100 * np_ / len(vals)
                        pc  = COL_PASS if pp >= 80 else (COL_FAIL if pp < 50 else (255, 190, 0))
                        stat_str = f"{np_}/{len(vals)} ({pp:.0f}%)   μ={np.mean(vals):.3f}±{np.std(vals):.3f}"
                    else:
                        pc, stat_str = COL_NODATA, "—"

                    m_col = f"    {metric_lbl:<22}" if first else f"    {'':22}"
                    row_txt = f"{m_col}  {ch_short:<8}  {tstr:<14}  "
                    first = False

                    if not measure_only:
                        x = 16
                        draw.text((x, y + 1), row_txt, fill=(0, 0, 0), font=font_r)
                        draw.text((x, y),     row_txt, fill=(130, 160, 200), font=font_r)
                        try:
                            x += draw.textbbox((0, 0), row_txt, font=font_r)[2]
                        except AttributeError:
                            x += len(row_txt) * (font_size - 4)
                        # Channel name in its accent colour
                        draw.text((x, y + 1), f"{ch_short}  ", fill=(0, 0, 0), font=font_r)
                        draw.text((x, y),     f"{ch_short}  ", fill=ch_color,   font=font_r)
                        try:
                            x += draw.textbbox((0, 0), f"{ch_short}  ", font=font_r)[2]
                        except AttributeError:
                            x += (len(ch_short) + 2) * (font_size - 4)
                        draw.text((x, y + 1), stat_str, fill=(0, 0, 0), font=font_r)
                        draw.text((x, y),     stat_str, fill=pc,         font=font_r)
                    y += _text_h(draw, row_txt, font_r) + 2

                y += 4

            y += 4
            rule()
            y += 4

        # Failing wells
        line(f"  Failing wells  ({len(failing)}):", (200, 215, 235), font_sec)
        if failing:
            for i in range(0, len(failing), 5):
                seg = failing[i:i + 5]
                txt = "    " + "   ".join(
                    f"{wl} [{', '.join(r)}]" for wl, r in seg)
                line(txt, COL_FAIL, font_r)
        else:
            line("    None — all wells pass all thresholds.", COL_PASS, font_r)

        rule()
        y += 6

        # ── Object counts + artifact density per well ─────────────────────────
        line("  OBJECT COUNTS PER WELL", (180, 210, 255), font_sec)
        hdr_counts = (f"    {'Well':<6}  {'Compound':<28}  "
                      f"{'Raw':>6}  {'Filtered':>8}  {'Cells':>6}  "
                      f"{'Artifacts':>9}  {'Art/kpx²':>9}")
        line(hdr_counts, (110, 140, 175), font_b)

        for wl in sorted(plate_qc.keys()):
            m        = plate_qc[wl]
            compound = (plate_map or {}).get(wl, "—")
            raw      = m.get(COUNT_COLS["Raw"])
            filt     = m.get(COUNT_COLS["Filtered"])
            cells    = m.get(COUNT_COLS["Cells"])
            arts     = m.get(COUNT_COLS["Artifacts"])
            density  = _artifact_density(m)

            def _fmt(v):
                return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else str(int(round(v)))

            dens_str = "—" if density is None else f"{density:.3f}"
            # Colour artifact density
            if density is None:
                dc = COL_NODATA
            elif density > 0.05:
                dc = COL_FAIL
            elif density > 0.02:
                dc = (255, 190, 0)
            else:
                dc = COL_PASS

            row_txt = (f"    {wl:<6}  {compound[:28]:<28}  "
                       f"{_fmt(raw):>6}  {_fmt(filt):>8}  {_fmt(cells):>6}  "
                       f"{_fmt(arts):>9}  ")

            if not measure_only:
                x = 16
                draw.text((x, y + 1), row_txt, fill=(0, 0, 0), font=font_r)
                draw.text((x, y),     row_txt, fill=(160, 180, 200), font=font_r)
                try:
                    x += draw.textbbox((0, 0), row_txt, font=font_r)[2]
                except AttributeError:
                    x += len(row_txt) * (font_size - 4)
                draw.text((x, y + 1), dens_str, fill=(0, 0, 0), font=font_r)
                draw.text((x, y),     dens_str, fill=dc,         font=font_r)
            y += _text_h(draw, row_txt, font_r) + 2

        rule()
        y += 6

        # Threshold reference box
        line("  Thresholds applied:", (190, 205, 225), font_sec)
        for mk, (lo, hi) in THRESHOLDS.items():
            tstr = (f"> {lo}" if hi is None else f"< {hi}" if lo is None else f"{lo} to {hi}")
            line(f"    {METRIC_LABELS.get(mk, mk)}: {tstr}", (150, 170, 195), font_r)
            for col in METRIC_COLS[mk]:
                ch = COL_TO_CHANNEL.get(col, col)
                ch_lbl = CHANNEL_LABELS.get(ch, ch)
                line(f"      {ch_lbl}: {col}", (110, 130, 155), font_r)

        y += 16
        return y

    # Measure height
    _dummy = Image.new("RGB", (width, 10))
    _dd    = ImageDraw.Draw(_dummy)
    total_h = _draw_footer(_dd, measure_only=True) + 20

    # Real draw
    footer = Image.new("RGB", (width, total_h), color=(10, 12, 22))
    fdraw  = ImageDraw.Draw(footer)
    fdraw.rectangle([0, 0, width - 1, 5], fill=(40, 60, 120))
    _draw_footer(fdraw, measure_only=False)

    return np.array(footer)


# ── Collage class ─────────────────────────────────────────────────────────────

class Collage:
    def __init__(self, input_dir, output_path,
                 qc_tsv=None,
                 platemap=None,
                 plate_rows=8, plate_cols=12,
                 sites_per_well=9, scale=0.5, workers=8,
                 band_height=280, font_size=18):

        self.input_dir      = Path(input_dir)
        self.output_path    = Path(output_path)
        self.plate_rows     = plate_rows
        self.plate_cols     = plate_cols
        self.sites_per_well = sites_per_well
        self.scale          = scale
        self.workers        = workers
        self.band_height    = band_height
        self.font_size      = font_size
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.qc = load_qc_tsv(qc_tsv)
        if not self.qc:
            for name in ("Image.txt", "image.txt", "Image.tsv", "image.tsv"):
                p = self.input_dir / name
                if p.exists():
                    print(f"[qc] Auto-detected: {p}")
                    self.qc = load_qc_tsv(p)
                    break

        # Load platemap — explicit path or auto-detect platemap_*.csv in input dir
        self.platemap = load_platemap(platemap)
        if not self.platemap:
            candidates = sorted(self.input_dir.glob("platemap_*.csv"))
            if candidates:
                print(f"[platemap] Auto-detected: {candidates[0]}")
                self.platemap = load_platemap(candidates[0])

        for plate_name, files in sorted(self._group_by_plate().items()):
            print(f"\n[plate] {plate_name}")
            wells    = self._group_well_imgs(files)
            montages = self._build_montages_parallel(wells)
            self._render_plate(plate_name, montages)

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
        args = [((r, c), sites, self.scale, self.sites_per_well)
                for (r, c), sites in wells.items()]
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            for fut in as_completed([ex.submit(build_well_montage, a) for a in args]):
                (r, c), mont = fut.result()
                if mont is not None:
                    montages[(r, c)] = mont
        return montages

    def _render_plate(self, plate_name: str, montages: dict):
        if not montages:
            print(f"  No images for {plate_name}, skipping.")
            return

        plate_qc = (self.qc.get(plate_name)
                    or self.qc.get(plate_name.lstrip("P"))
                    or (next(iter(self.qc.values())) if len(self.qc) == 1 else {}))

        # Resolve platemap for this plate (try exact name, then first available)
        plate_map = (self.platemap.get(plate_name)
                     or self.platemap.get(plate_name.lstrip("P"))
                     or (next(iter(self.platemap.values())) if len(self.platemap) == 1 else {}))

        tile_h, tile_w = next(iter(montages.values())).shape[:2]
        blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        rows_imgs = []

        for r in range(1, self.plate_rows + 1):
            row_labels = [well_label(r, c) for c in range(1, self.plate_cols + 1)]
            row_tiles  = []

            for c in range(1, self.plate_cols + 1):
                wl      = well_label(r, c)
                tile    = montages.get((r, c), blank)
                metrics = plate_qc.get(wl)

                border_rgb = _slope_to_rgb(metrics)
                row_tiles.append(_make_tile(tile, border_rgb))

            row_img = np.concatenate(row_tiles, axis=1)
            rows_imgs.append(row_img)

            # Single unified QC band (ILLUM + FOCUS + COUNTS)
            rows_imgs.append(_make_band(
                width=row_img.shape[1],
                well_labels_in_row=row_labels,
                plate_qc=plate_qc,
                band_height=self.band_height,
                font_size=self.font_size,
                tile_width=tile_w,
                plate_map=plate_map,
            ))

        collage = np.concatenate(rows_imgs, axis=0)

        # Slope range for header info
        s_min, s_max = _slope_range(plate_qc)
        header = make_header(
            collage.shape[1],
            title=f"{plate_name}  ·  Cell Painting QC  "
                  f"|  Slope range: {s_min:.2f} – {s_max:.2f}  "
                  f"|  Threshold: {THRESHOLDS[BORDER_METRIC][0]} to {THRESHOLDS[BORDER_METRIC][1]}")
        collage = np.concatenate([header, collage], axis=0)

        if plate_qc:
            footer = make_report_footer(
                collage.shape[1], plate_name, plate_qc,
                font_size=self.font_size,
                plate_map=plate_map)
            collage = np.concatenate([collage, footer], axis=0)

        # PNG — lossless, max compression
        """out_png = self.output_path / f"{plate_name}_QC.png"
        imageio.imwrite(str(out_png), collage, format="png", compress_level=9)
        png_mb = out_png.stat().st_size / 1e6"""

        # JPEG — lossy quality=95, much smaller, visually identical
        out_jpg = self.output_path / f"{plate_name}_QC.jpg"
        Image.fromarray(collage).save(str(out_jpg), format="JPEG", quality=95,
                                      optimize=True, subsampling=0)
        jpg_mb = out_jpg.stat().st_size / 1e6

        print(f"  → {out_jpg}  ({collage.shape[1]}×{collage.shape[0]} px, JPEG q=95)  {jpg_mb:.1f} MB")


def parse_args():
    p = argparse.ArgumentParser(
        description="Cell Painting QC collage — illumination artefacts + focus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input",       required=True)
    p.add_argument("-o", "--output",      required=True)
    p.add_argument("--qc",                default=None,
                   help="CellProfiler Image.txt TSV (auto-detected if omitted).")
    p.add_argument("--platemap",          default=None,
                   help="Platemap CSV (platemap_*.csv, auto-detected if omitted).")
    p.add_argument("--rows",              type=int,   default=8)
    p.add_argument("--cols",              type=int,   default=12)
    p.add_argument("--sites",             type=int,   default=9)
    p.add_argument("--scale",             type=float, default=0.5)
    p.add_argument("--workers",           type=int,   default=8)
    p.add_argument("--band-height",       type=int,   default=280,
                   help="Minimum band height px; auto-expands to fit content.")
    p.add_argument("--font",              type=int,   default=18)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Collage(
        input_dir      = args.input,
        output_path    = args.output,
        qc_tsv         = args.qc,
        platemap       = args.platemap,
        plate_rows     = args.rows,
        plate_cols     = args.cols,
        sites_per_well = args.sites,
        scale          = args.scale,
        workers        = args.workers,
        band_height    = args.band_height,
        font_size      = args.font,
    )
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
  PowerLogLogSlope  : -2.5 to -1.0   (illumination quality / blur)
  MaxIntensity      : < 0.95          (saturation)
  FocusScore        : > 0.1
  LocalFocusScore   : > 0.1

Footer report:
  - Overall pass rate
  - Per-metric stats (pass %, mean ± SD) for Hoechst and Syto
  - Failing wells with reasons

Usage
-----
    python III_QC_collage.py -i /output/QC/Images -o /output/QC/Collages
    # --qc optional: auto-detected from -i if Image.txt present

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

# Thresholds: metric_key → (min, max), None = no bound
THRESHOLDS = {
    "PowerLogLogSlope": (-2.5, -1.0),
    "MaxIntensity":     (None,  0.95),
    "FocusScore":       (0.1,   None),
    "LocalFocusScore":  (0.1,   None),
}

# Column pairs: metric_key → (hoechst_col, syto_col)
METRIC_COLS = {
    "PowerLogLogSlope": (
        "ImageQuality_PowerLogLogSlope_Hoechst",
        "ImageQuality_PowerLogLogSlope_Syto",
    ),
    "MaxIntensity": (
        "ImageQuality_MaxIntensity_Hoechst",
        "ImageQuality_MaxIntensity_Syto",
    ),
    "FocusScore": (
        "ImageQuality_FocusScore_Hoechst",
        "ImageQuality_FocusScore_Syto",
    ),
    "LocalFocusScore": (
        "ImageQuality_LocalFocusScore_Hoechst_10",
        "ImageQuality_LocalFocusScore_Syto_10",
    ),
}

# The two bands
ILLUM_METRICS = ["PowerLogLogSlope", "MaxIntensity"]
FOCUS_METRICS = ["FocusScore", "LocalFocusScore"]

# Primary metric driving tile border colour
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


def _passes(value, metric_key: str):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    lo, hi = THRESHOLDS.get(metric_key, (None, None))
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True

def _val_color(value, metric_key: str) -> tuple:
    p = _passes(value, metric_key)
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

    all_cols = [c for cols in METRIC_COLS.values() for c in cols if c in df.columns]
    gkeys    = [plate_col, well_col] if plate_col else [well_col]
    agg      = df.groupby(gkeys)[all_cols].mean().reset_index()

    result = defaultdict(dict)
    for _, row in agg.iterrows():
        plate = str(row[plate_col]).strip() if plate_col else "Plate"
        well  = str(row[well_col]).strip().upper()
        result[plate][well] = {c: row[c] for c in all_cols if c in row.index}

    total = sum(len(v) for v in result.values())
    print(f"[qc] {total} wells across {len(result)} plate(s) ready.")
    return result


# ── Slope → tile border colour ────────────────────────────────────────────────

def _slope_range(plate_qc: dict) -> tuple:
    col_h, col_s = METRIC_COLS[BORDER_METRIC]
    vals = [v for m in plate_qc.values()
            for col in (col_h, col_s)
            if (v := m.get(col)) is not None and not np.isnan(v)]
    lo, hi = THRESHOLDS[BORDER_METRIC]
    return (min(vals) if vals else lo, max(vals) if vals else hi)


def _slope_to_rgb(vh, vs) -> tuple:
    """
    Colour tile border by PowerLogLogSlope pass/fail:
      Both pass  → bright green
      One fails  → orange
      Both fail  → deep red
      No data    → dark grey
    """
    results = []
    for v, mk in ((vh, BORDER_METRIC), (vs, BORDER_METRIC)):
        p = _passes(v, mk)
        if p is not None:
            results.append(p)
    if not results:
        return (45, 45, 55)
    n_pass = sum(results)
    if n_pass == len(results):
        return (55, 205, 80)    # all pass → green
    elif n_pass == 0:
        return (215, 35, 35)    # all fail → red
    else:
        return (255, 145, 0)    # mixed    → orange


# ── Image helpers ─────────────────────────────────────────────────────────────

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

def _make_band(width: int, well_labels_in_row: list, plate_qc: dict,
               band_height: int, font_size: int, tile_width: int) -> np.ndarray:
    """
    Single unified QC band. Layout per well cell:
      - Top accent bar (pass/fail colour)
      - Well label (compact, top-left)
      - Metrics in two columns: left=ILLUM, right=FOCUS
        Each metric: label line + Hoechst + Syto values coloured by threshold
    """
    ALL_METRICS = ILLUM_METRICS + FOCUS_METRICS

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
            col_h, col_s = METRIC_COLS[mk]
            for col in (col_h, col_s):
                v = metrics.get(col) if metrics else None
                p = _passes(v, mk)
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
        y_start = 7 + well_h + 5

        if not metrics:
            draw.line([x1, 0, x1, band_height - 1], fill=(28, 28, 36), width=1)
            continue

        # Two-column layout: ILLUM left, FOCUS right
        col_w   = (tile_width - 8) // 2
        x_left  = x0 + 4
        x_right = x0 + 4 + col_w

        for col_x, group in ((x_left, ILLUM_METRICS), (x_right, FOCUS_METRICS)):
            y = y_start
            for mk in group:
                col_h, col_s = METRIC_COLS[mk]
                vh = metrics.get(col_h)
                vs = metrics.get(col_s)
                if (vh is None or np.isnan(vh)) and (vs is None or np.isnan(vs)):
                    continue

                lbl = METRIC_LABELS.get(mk, mk)
                draw.text((col_x, y), f"{lbl}:", fill=(170, 195, 225), font=font_lbl)
                y += _text_h(draw, f"{lbl}:", font_lbl) + 1

                for v, ch in ((vh, "Hoechst"), (vs, "Syto")):
                    if v is None or np.isnan(v):
                        continue
                    vc  = _val_color(v, mk)
                    txt = f" {ch[:1]}: {v:.3f}"
                    draw.text((col_x, y + 1), txt, fill=(0, 0, 0), font=font_val)
                    draw.text((col_x, y),     txt, fill=vc,         font=font_val)
                    y += _text_h(draw, txt, font_val) + 1

                y += 4

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
    for mk, (col_h, col_s) in METRIC_COLS.items():
        for col in (col_h, col_s):
            v = metrics.get(col)
            if v is not None and not np.isnan(v) and not _passes(v, mk):
                return False
    return True

def _well_passes_group(metrics: dict, group: list) -> bool:
    for mk in group:
        col_h, col_s = METRIC_COLS[mk]
        for col in (col_h, col_s):
            v = metrics.get(col)
            if v is not None and not np.isnan(v) and not _passes(v, mk):
                return False
    return True


def make_report_footer(width: int, plate_name: str,
                        plate_qc: dict, font_size: int = 16) -> np.ndarray:
    n_wells   = len(plate_qc)
    n_pass    = sum(1 for m in plate_qc.values() if _well_passes_all(m))
    n_illum_p = sum(1 for m in plate_qc.values() if _well_passes_group(m, ILLUM_METRICS))
    n_focus_p = sum(1 for m in plate_qc.values() if _well_passes_group(m, FOCUS_METRICS))
    pct       = lambda n: f"{100 * n / n_wells:.1f}%" if n_wells else "—"

    # Per-metric stats
    stats = {mk: {"Hoechst": [], "Syto": []} for mk in METRIC_COLS}
    for m in plate_qc.values():
        for mk, (col_h, col_s) in METRIC_COLS.items():
            for col, ch in ((col_h, "Hoechst"), (col_s, "Syto")):
                v = m.get(col)
                if v is not None and not np.isnan(v):
                    stats[mk][ch].append(v)

    # Failing wells
    failing = []
    for wl, m in sorted(plate_qc.items()):
        reasons = []
        for mk, (col_h, col_s) in METRIC_COLS.items():
            for col, ch in ((col_h, "H"), (col_s, "S")):
                v = m.get(col)
                if v is not None and not np.isnan(v) and not _passes(v, mk):
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
            hdr = f"    {'Metric (CellProfiler column)':<48}  {'Threshold':<14}  {'Hoechst':>10}  {'mean ± SD':<20}  {'Syto':>10}  {'mean ± SD':<20}"
            line(hdr, (110, 140, 175), font_b)

            for mk in group_metrics:
                lo, hi = THRESHOLDS.get(mk, (None, None))
                tstr = (f"> {lo}" if hi is None else
                        f"< {hi}" if lo is None else f"{lo} to {hi}")
                col_h, col_s = METRIC_COLS[mk]
                cp_name = col_h.replace("ImageQuality_", "IQ_").replace("_Hoechst", "").replace("_10", "")
                disp = f"{METRIC_LABELS.get(mk, mk)} ({cp_name})"
                lbl = f"    {disp:<48}  {tstr:<14}  "

                row_parts  = [lbl]
                row_colors = [(130, 160, 200)]

                for ch in ("Hoechst", "Syto"):
                    vals = stats[mk][ch]
                    if vals:
                        np_ = sum(1 for v in vals if _passes(v, mk))
                        pp  = 100 * np_ / len(vals)
                        pc  = COL_PASS if pp >= 80 else (COL_FAIL if pp < 50 else (255, 190, 0))
                        row_parts.append(
                            f"{np_}/{len(vals)} ({pp:.0f}%)        "
                            f"μ={np.mean(vals):.3f}±{np.std(vals):.3f}  ")
                        row_colors.append(pc)
                    else:
                        row_parts.append("—                                   ")
                        row_colors.append(COL_NODATA)

                if not measure_only:
                    x = 16
                    for part, col in zip(row_parts, row_colors):
                        draw.text((x, y + 1), part, fill=(0, 0, 0), font=font_r)
                        draw.text((x, y),     part, fill=col,        font=font_r)
                        try:
                            x += draw.textbbox((0, 0), part, font=font_r)[2]
                        except AttributeError:
                            x += len(part) * (font_size - 4)
                y += _text_h(draw, lbl, font_r) + 4

            y += 6
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

        # Threshold reference box
        line("  Thresholds applied:", (190, 205, 225), font_sec)
        for mk, (lo, hi) in THRESHOLDS.items():
            tstr  = (f"> {lo}" if hi is None else f"< {hi}" if lo is None else f"{lo} to {hi}")
            col_h, col_s = METRIC_COLS[mk]
            cp_h  = col_h.replace("ImageQuality_", "ImageQuality_")  # keep full name here
            cp_s  = col_s.replace("ImageQuality_", "ImageQuality_")
            line(f"    {METRIC_LABELS.get(mk, mk)}: {tstr}", (150, 170, 195), font_r)
            line(f"      Hoechst col: {col_h}", (110, 130, 155), font_r)
            line(f"      Syto col:    {col_s}", (110, 130, 155), font_r)

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

                col_h, col_s = METRIC_COLS[BORDER_METRIC]
                border_rgb = _slope_to_rgb(
                    metrics.get(col_h) if metrics else None,
                    metrics.get(col_s) if metrics else None,
                )
                row_tiles.append(_make_tile(tile, border_rgb))

            row_img = np.concatenate(row_tiles, axis=1)
            rows_imgs.append(row_img)

            # Single unified QC band (ILLUM + FOCUS)
            rows_imgs.append(_make_band(
                width=row_img.shape[1],
                well_labels_in_row=row_labels,
                plate_qc=plate_qc,
                band_height=self.band_height,
                font_size=self.font_size,
                tile_width=tile_w,
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
                font_size=self.font_size)
            collage = np.concatenate([collage, footer], axis=0)

        # PNG — lossless, max compression
        out_png = self.output_path / f"{plate_name}_QC.png"
        imageio.imwrite(str(out_png), collage, format="png", compress_level=9)
        png_mb = out_png.stat().st_size / 1e6

        # JPEG — lossy quality=95, much smaller, visually identical
        out_jpg = self.output_path / f"{plate_name}_QC.jpg"
        Image.fromarray(collage).save(str(out_jpg), format="JPEG", quality=95,
                                      optimize=True, subsampling=0)
        jpg_mb = out_jpg.stat().st_size / 1e6

        print(f"  → {out_png}  ({collage.shape[1]}×{collage.shape[0]} px)  {png_mb:.1f} MB")
        print(f"  → {out_jpg}  (JPEG q=95)  {jpg_mb:.1f} MB")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Cell Painting QC collage — illumination artefacts + focus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input",       required=True)
    p.add_argument("-o", "--output",      required=True)
    p.add_argument("--qc",                default=None,
                   help="CellProfiler Image.txt TSV (auto-detected if omitted).")
    p.add_argument("--rows",              type=int,   default=8)
    p.add_argument("--cols",              type=int,   default=12)
    p.add_argument("--sites",             type=int,   default=9)
    p.add_argument("--scale",             type=float, default=0.5)
    p.add_argument("--workers",           type=int,   default=8)
    p.add_argument("--band-height",       type=int,   default=170)
    p.add_argument("--font",              type=int,   default=18)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Collage(
        input_dir      = args.input,
        output_path    = args.output,
        qc_tsv         = args.qc,
        plate_rows     = args.rows,
        plate_cols     = args.cols,
        sites_per_well = args.sites,
        scale          = args.scale,
        workers        = args.workers,
        band_height    = args.band_height,
        font_size      = args.font,
    )
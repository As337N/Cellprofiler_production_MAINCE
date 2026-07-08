from __future__ import annotations

def _miq_col(metric: str, channel: str) -> str:
    if metric == "FocusScore":
        return f"ImageQuality_FocusScore_{channel}"
    return f"ImageQuality_{metric}_{channel}"


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
METRIC_LABELS  = {
    "PowerLogLogSlope": "Focus", "MedianIntensity": "MaxInt",
}


COUNT_COLS = {
    "Raw": "Count_Raw_nuclei", "Filtered": "Count_Nuclei",
    "Cells": "Count_Cells",    "Artifacts": "Count_Illum_artifacts_filtered",
}
AREA_COL           = "ImageQuality_TotalArea_Brightfield"
DEFAULT_IMAGE_AREA = 1_166_400   # 1080×1080 px fallback

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

COL_THRESHOLDS = {'COL_PASS': (75,  215,  95),
                  'COL_FAIL': (255,  65,  65),
                  'COL_NODATA': (110, 110, 120)}

# ── MFI channel definitions ───────────────────────────────────────────────────
# Hoechst (Hoechst) → Nuclei.txt  |  Syto, ER, Golgi, Mito → Cells.txt
# Column format: Intensity_MedianIntensity_<Channel>

_MFI = {
    "Hoechst": ("nuclei", "#64A0FF"),
    "Syto":    ("cells",  "#50DC78"),
    "ER":      ("cells",  "#B45AFF"),
    "Golgi":   ("cells",  "#FFB43C"),
    "Mito":    ("cells",  "#FF5050"),
}

_obj = lambda ch: f"Intensity_MedianIntensity_{ch}"
_img = lambda ch: f"ImageQuality_MedianIntensity_{ch}"

MFI_CHANNEL_ORDER = list(_MFI)
MFI_COLS_CELLS  = {ch: (_obj(ch), c) for ch, (s, c) in _MFI.items() if s == "cells"}
MFI_COLS_NUCLEI = {ch: (_obj(ch), c) for ch, (s, c) in _MFI.items() if s == "nuclei"}
MFI_IMG_COLS    = {ch: _img(ch) for ch in _MFI}

MFI_DATA = {'MFI_COLS_CELLS': MFI_COLS_CELLS, 'MFI_COLS_NUCLEI': MFI_COLS_NUCLEI,
            'MFI_CHANNEL_ORDER': MFI_CHANNEL_ORDER, 'MFI_IMG_COLS': MFI_IMG_COLS}
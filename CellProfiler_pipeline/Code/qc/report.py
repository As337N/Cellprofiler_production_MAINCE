"""
qc/report.py — HTML report builder.

Phase 1 of the refactor: this module replaces the monolithic ``generate_html``
f-string. The data-computation logic is copied verbatim from the original so the
generated report is functionally equivalent to the pre-refactor baseline. The
only structural change is the *output* stage:

  * The 13 separate ``json.dumps(...)`` variables are grouped into a single
    ``window.__QC__`` object (see ``build_payload``).
  * CSS / JS / HTML live in ``qc/assets/`` and are stitched in via marker
    replacement (``__CSS__``, ``__JS__``, ``__PLOTLY__``, ``__COHORT__``,
    ``__GENERATED__``, ``__PAYLOAD__``).

SCAFFOLDING NOTE (remove in Phase 2):
    Helpers and constants are still imported from the main script. Once the
    builder is verified against the baseline, these move to stats.py / montage.py
    / thresholds.py in small, individually-verified commits.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import urllib.request

from qc import config

# ── Phase-1 scaffolding import (to be replaced by proper module imports) ──────
from III_QC_collage import (
    _collage_to_b64, _make_report_collage,
    _round_floats, _mad_scores, _median_cs,
)

# SLOPE_CS is defined locally in the original generate_html (not a global), so
# it lives here rather than being imported. Values copied verbatim.
SLOPE_CS = [
    [0.0,  "#ff4444"],   # -2.5  red
    [0.15, "#ff4444"],   # -2.35 red   (limit -2.7)
    [0.15, "#ffbe00"],   # -2.35 yellow
    [0.35, "#ffbe00"],   # -2.15 yellow (limit -2.5)
    [0.35, "#4bd760"],   # -2.15 green
    [0.65, "#4bd760"],   # -1.85 green  (limit -1.5)
    [0.65, "#ffbe00"],   # -1.85 yellow
    [0.85, "#ffbe00"],   # -1.65 yellow
    [0.85, "#ff4444"],   # -1.65 red
    [1.0,  "#ff4444"],   # -1.0  red
]

_ASSETS = Path(__file__).resolve().parent / "assets"


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

# ─────────────────────────────────────────────────────────────────────────────
# Payload construction  (logic copied verbatim from the original generate_html)
# ─────────────────────────────────────────────────────────────────────────────
def build_payload(cohort_name: str, plates_data: list[dict]) -> dict:
    """Compute every piece of report data and return it grouped as window.__QC__.

    The internal calculations are identical to the original generate_html; only
    the final grouping differs. Returns a plain dict ready for json.dumps.
    """
    html_cols = (
        [c for cols in config.METRIC_COLS.values() for c in cols] +
        list(config.COUNT_COLS.values()) + [config.AREA_COL] +
        [f"ImageQuality_PercentMaximal_{ch}" for ch in config.CHANNELS] +
        [f"ImageQuality_PercentMinimal_{ch}" for ch in config.CHANNELS]
    )

    # ── Per-plate payload (collages, flags, per-well metrics) ────────────────
    payload = {}
    for pd_ in plates_data:
        pname = pd_["name"]
        pqc   = pd_["plate_qc"]
        pmap  = pd_["plate_map"] or {}
        adp   = pd_.get("engine_adaptive", {})

        report_grid = _make_report_collage(pd_["collage_arr"], pqc, pmap)

        well_flags = {}
        for well, m in pqc.items():
            abs_fails, adp_fails = [], []
            for mk, cols in config.METRIC_COLS.items():
                for col in cols:
                    v  = m.get(col)
                    ch = config.COL_TO_CHANNEL.get(col, "")
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        continue
                    if mk == "FocusScore":
                        lo, hi = config.THRESHOLDS_LOCAL_FOCUS.get(ch, (None, None))
                    else:
                        lo, hi = config.THRESHOLDS.get(mk, (None, None))
                    abs_fail = (lo is not None and v < lo) or (hi is not None and v > hi)
                    adp_col = adp.get(mk, {}).get(col)
                    adp_fail = False
                    if adp_col:
                        adp_lo, adp_hi = adp_col
                        adp_fail = (v < adp_lo and lo is not None and v < lo) or \
                                   (v > adp_hi and hi is not None and v > hi)
                    ch_lbl = ch
                    met_lbl = config.METRIC_LABELS.get(mk, mk)
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
            # Per-site QC: { well: { field: { col: val } } } — same columns as the
            # well-info panel, so the site panel can reuse the well-level specs.
            "site_data": _round_floats({
                well: {
                    str(field): {c: fvals.get(c) for c in html_cols if c in fvals}
                    for field, fvals in fields.items()
                }
                for well, fields in (pd_.get("plate_qc_sites") or {}).items()
            }),
        }

    # ── MAD scores (tooltip) ─────────────────────────────────────────────────
    _qc_cols_for_scores = (
        [f"ImageQuality_PowerLogLogSlope_{ch}" for ch in config.CHANNELS] +
        [f"ImageQuality_PercentMaximal_{ch}"      for ch in config.CHANNELS] +
        [f"ImageQuality_PercentMinimal_{ch}"      for ch in config.CHANNELS] +
        [f"ImageQuality_MedianIntensity_{ch}"  for ch in config.CHANNELS]
    )
    calc_mad_scores = {}
    for col in _qc_cols_for_scores:
        sc, cmed, cmad = _mad_scores(col, plates_data)
        calc_mad_scores[col] = sc

    # ── Red well counts per metric group per plate ───────────────────────────
    def _red_wells(plates_data, cols, thresholds):
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
                        break
            result[pname] = count
        return result

    slope_red = _red_wells(plates_data,
        [f"ImageQuality_PowerLogLogSlope_{ch}" for ch in config.CHANNELS],
        {col: (-2.7, -1.3) for ch in config.CHANNELS
         for col in [f"ImageQuality_PowerLogLogSlope_{ch}"]})
    pctmax_red = _red_wells(plates_data,
        [f"ImageQuality_PercentMaximal_{ch}" for ch in config.CHANNELS],
        {col: (None, 1.0) for ch in config.CHANNELS
         for col in [f"ImageQuality_PercentMaximal_{ch}"]})
    pctmin_red = _red_wells(plates_data,
        [f"ImageQuality_PercentMinimal_{ch}" for ch in config.CHANNELS],
        {col: (None, 5.0) for ch in config.CHANNELS
         for col in [f"ImageQuality_PercentMinimal_{ch}"]})

    medint_red = {}
    for pd_ in plates_data:
        pname = pd_["name"]
        count = 0
        for well in pd_["plate_qc"]:
            for ch in config.CHANNELS:
                col = f"ImageQuality_MedianIntensity_{ch}"
                sc  = calc_mad_scores.get(col, {}).get(pname, {}).get(well)
                if sc and abs(sc["z_co"]) > 3:
                    count += 1
                    break
        medint_red[pname] = count

    red_counts = {
        "slope":  slope_red,
        "pctmax": pctmax_red,
        "pctmin": pctmin_red,
        "medint": medint_red,
    }

    # ── Cohort-wide colour ranges for count columns ──────────────────────────
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
        return lo, hi

    # ── Metric specs (heatmap colour scales) ─────────────────────────────────
    slope_specs = [
        {"col": col, "title": f"Slope — {ch}",
         "cmin": -3.0, "cmax": -1.0, "cs": SLOPE_CS}
        for ch in config.CHANNELS
        for col in [f"ImageQuality_PowerLogLogSlope_{ch}"]
    ]
    PCT_MAX_CS = [
        [0.0, "#4bd760"], [0.05, "#4bd760"], [0.05, "#ffbe00"],
        [0.25, "#ffbe00"], [0.25, "#ff4444"], [1.0, "#ff4444"],
    ]
    PCT_MIN_CS = [
        [0.0, "#4bd760"], [0.2, "#4bd760"], [0.2, "#ffbe00"],
        [0.5, "#ffbe00"], [0.5, "#ff4444"], [1.0, "#ff4444"],
    ]
    pct_max_specs = [
        {"col": col, "title": f"PctMaximal — {ch}",
         "cmin": 0, "cmax": 2.0, "cs": PCT_MAX_CS}
        for ch in config.CHANNELS
        for col in [f"ImageQuality_PercentMaximal_{ch}"]
    ]
    pct_min_specs = [
        {"col": col, "title": f"PctMinimal — {ch}",
         "cmin": 0, "cmax": 10.0, "cs": PCT_MIN_CS}
        for ch in config.CHANNELS
        for col in [f"ImageQuality_PercentMinimal_{ch}"]
    ]
    medianint_specs = [
        {"col": col, "title": f"MedianInt — {ch}",
         "cmin": _median_cs(col, plates_data)[1],
         "cmax": _median_cs(col, plates_data)[2],
         "cs":   _median_cs(col, plates_data)[0]}
        for ch in config.CHANNELS
        for col in [f"ImageQuality_MedianIntensity_{ch}"]
    ]
    count_ranges = {
        col: list(_cohort_range(col, 0, None))
        for col in ["Count_Cells", "Count_Nuclei", "Count_Illum_artifacts_filtered"]
    }

    # ── MFI: { plate: { channel: { well: [vals] } } } ────────────────────────
    mfi_payload: dict = {}
    for pd_ in plates_data:
        pname   = pd_["name"]
        mfi_raw = pd_.get("mfi_data", {})
        by_ch: dict = {}
        for well, ch_vals in mfi_raw.items():
            for ch, vals in ch_vals.items():
                by_ch.setdefault(ch, {})[well] = vals
        mfi_payload[pname] = by_ch

    _all_mfi_channels = [ch for ch in config.MFI_DATA['MFI_CHANNEL_ORDER']
                         if any(ch in by_ch for by_ch in mfi_payload.values())]
    _mfi_colors = {**{ch: c for ch, (_, c) in config.MFI_DATA['MFI_COLS_NUCLEI'].items()},
                   **{ch: c for ch, (_, c) in config.MFI_DATA['MFI_COLS_CELLS'].items()}}
    mfi_colors = {ch: _mfi_colors.get(ch, "#8ab0d0") for ch in _all_mfi_channels}
    mfi_img_payload = {pd_["name"]: pd_.get("mfi_img", {}) for pd_ in plates_data}
    radius_payload  = {pd_["name"]: pd_.get("radius_data", {}) for pd_ in plates_data}

    # ── Group everything into window.__QC__ ──────────────────────────────────
    return {
        "data":        payload,
        "countRanges": count_ranges,
        "mfi": {
            "data":     mfi_payload,
            "channels": _all_mfi_channels,
            "colors":   mfi_colors,
            "img":      mfi_img_payload,
        },
        "radius":    radius_payload,
        "specs": {
            "slope":     slope_specs,
            "pctMax":    pct_max_specs,
            "pctMin":    pct_min_specs,
            "medianInt": medianint_specs,
        },
        "madScores": calc_mad_scores,
        "redCounts": red_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML assembly
# ─────────────────────────────────────────────────────────────────────────────
def _read_asset(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


def render_report(cohort_name: str, plates_data: list[dict],
                  output_path: Path, web_scale: float = 0.2) -> None:
    """Build the self-contained HTML QC report from assets + computed payload."""
    plotly_js = _fetch_plotly_js()
    qc_payload = build_payload(cohort_name, plates_data)

    payload_js = "window.__QC__ = " + json.dumps(qc_payload) + ";"
    generated  = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = _read_asset("report.html")
    # Replace order matters: do the short text markers (__COHORT__, __GENERATED__)
    # FIRST, while the template still only contains our own markup. The large
    # injected blobs (plotly, payload, js) go LAST, so if any of them happen to
    # contain a literal like "__COHORT__" (e.g. a compound name), it is treated
    # as data and not accidentally substituted.
    html = html.replace("__COHORT__",    cohort_name)
    html = html.replace("__GENERATED__", generated)
    html = html.replace("__CSS__",       _read_asset("report.css"))
    html = html.replace("__PLOTLY__",    plotly_js)
    html = html.replace("__PAYLOAD__",   payload_js)
    html = html.replace("__JS__",        _read_asset("report.js"))
    # UTIF.js: decodificador TIFF para el visor de imágenes opcional. Se embebe
    # como asset para que el reporte sea autocontenido (no requiere internet).
    html = html.replace("__UTIF__",      _read_asset("utif.js"))

    Path(output_path).write_text(html, encoding="utf-8")
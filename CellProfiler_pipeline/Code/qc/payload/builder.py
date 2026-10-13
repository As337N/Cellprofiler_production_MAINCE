"""
qc/payload/builder.py — DataFrame/stats results → window.__QC__ dict.

This is the ONLY module that materialises the nested dict consumed by report.js.
Everything upstream (io, stats, aggregate, scores) works in Polars; here the
computed results are assembled into the exact structure the frontend expects.

Replaces report.build_payload. The statistics are no longer recomputed inline:
they come from qc.stats (mad_scores, median_colorscale, red counts) and the
ThresholdEngine adaptive bands. The image blobs come from qc.render. This split
is what breaks the III_QC_collage <-> report circular import.

Payload keys (unchanged, frontend contract):
    n_rows, n_cols, data, countRanges,
    mfi{data,channels,colors,img}, radius,
    specs{slope,debris,low_density,medianInt},
    madScores, redCounts, robustMetrics
"""
from __future__ import annotations

import numpy as np

from qc import profiles
from qc.render.collage import _make_report_collage
from qc.render.overview import _collage_to_b64, _round_floats
from qc.stats import scores as S
from qc.stats.colors import color_spec_payload

# Fixed absolute colour scales for percentage heatmaps (verbatim from report.py).
_PCT_MAX_CS = [
    [0.0, "#4bd760"], [0.05, "#4bd760"], [0.05, "#ffbe00"],
    [0.25, "#ffbe00"], [0.25, "#ff4444"], [1.0, "#ff4444"],
]
def _cohort_range(col: str, plates_data: list, fallback_lo, fallback_hi,
                  plo: float = 2, phi: float = 98) -> tuple:
    """Percentile-based cohort colour range for count columns.

    Falls back to (fallback_lo, fallback_hi) when fewer than 10 valid values
    are available. Verbatim behaviour from report._cohort_range.
    """
    vals = [v for pd_ in plates_data
            for m in pd_["plate_qc"].values()
            if (v := m.get(col)) is not None
            and not (isinstance(v, float) and np.isnan(v))]
    if len(vals) < 10:
        return fallback_lo, fallback_hi
    return float(np.percentile(vals, plo)), float(np.percentile(vals, phi))


def build_payload(cohort_name: str, plates_data: list[dict],
                  profile: profiles.DatasetProfile,) -> dict:
    """Assemble window.__QC__ from precomputed render + stats results.

    Parameters
    ----------
    cohort_name : str
        Cohort label (kept for signature parity; not embedded here).
    plates_data : list of dict
        One entry per plate, carrying the rendered arrays and per-well QC. Each
        entry provides: name, plate_qc, plate_map, engine_adaptive, collage_arr,
        overview_arr/cw/ch, flagged_b64, pass_rate, n_wells, n_pass,
        n_illum_pass, plate_qc_sites, mfi_data, mfi_img, radius_data, plate_rows,
        plate_cols.
    profile : DatasetProfile
        Active dataset profile (channels, column templates, labels).

    Returns
    -------
    dict ready for json.dumps as window.__QC__.
    """
    channels = list(profile.channels)
    metric_cols = profile.metric_cols()
    col_to_channel = profile.col_to_channel()

    html_cols = (
        [c for cols in metric_cols.values() for c in cols] +
        list(profiles.COUNT_COLS.values()) + [profiles.AREA_COL] +
        [f"ImageQuality_PercentMaximal_{ch}" for ch in channels] +
        ["Ratio_P10/P90"]
    )

    # ── Per-plate payload (collages, flags, per-well metrics) ────────────────
    data: dict = {}
    
    for pd_ in plates_data:
        pname = pd_["name"]
        pqc = pd_["plate_qc"]
        pmap = pd_["plate_map"] or {}
        adp = pd_.get("engine_adaptive", {})

        report_grid = _make_report_collage(
            pd_["collage_arr"], pqc, pmap,
            plate_rows=pd_.get("plate_rows", profile.plate_rows),
            plate_cols=pd_.get("plate_cols", profile.plate_cols),
            robust_verdict=pd_.get("robust_verdict", {}),
        )

        well_flags = _compute_well_flags(pqc, adp, metric_cols, col_to_channel)

        # Sitio responsable del peor valor por columna de percentil (para dirigir
        # la búsqueda de outliers): {well: {col: field}}. Viene del colapso
        # peor-sitio en el pipeline; se serializa como str(field) para casar con
        # las claves de site_data.
        worst_sites = pd_.get("plate_worst_sites", {}) or {}

        data[pname] = {
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
                well: {"compound": pmap.get(well, ""),
                       **{c: m.get(c) for c in html_cols if c in m},
                       **({"_worst": {c: str(f) for c, f in worst_sites[well].items()}}
                          if well in worst_sites and worst_sites[well] else {})}
                for well, m in pqc.items()
            }),
            "site_data": _round_floats({
                well: {
                    str(field): {c: fvals.get(c) for c in html_cols if c in fvals}
                    for field, fvals in fields.items()
                }
                for well, fields in (pd_.get("plate_qc_sites") or {}).items()
            }),
        }

    # ── MAD scores (tooltip) — from stats.scores ─────────────────────────────
    # MAD scores solo para métricas que se colorean/etiquetan por z (robustas) y
    # PercentMaximal (tooltip). Percentile_10 se colorea por ratio, no por MAD.
    qc_cols_for_scores = (
        [f"ImageQuality_PowerLogLogSlope_{ch}" for ch in channels] +
        [f"ImageQuality_PercentMaximal_{ch}" for ch in channels] +
        [f"ImageQuality_MedianIntensity_{ch}" for ch in channels]
    )
    calc_mad_scores = {col: S.mad_scores(col, plates_data)[0]
                       for col in qc_cols_for_scores}

    # ── Red well counts — from stats.scores ──────────────────────────────────
    robust_red = {m: S.red_wells_zscore(m, channels, calc_mad_scores, plates_data)
                  for m in profiles.ROBUST_METRICS}
    from qc.stats.colors import ABSOLUTE_CUTS
    r_p10_90_fail = ABSOLUTE_CUTS["Ratio_P10/P90"][1]  
    r_p10_90_red_thresholds = {"Ratio_P10/P90": (None, r_p10_90_fail)}
    red_counts = {
        "slope":  robust_red["PowerLogLogSlope"],
        "medint": robust_red["MedianIntensity"],
        "debris": S.red_wells_absolute(
            plates_data, [f"ImageQuality_PercentMaximal_{ch}" for ch in channels],
            {f"ImageQuality_PercentMaximal_{ch}": (None, 1.0) for ch in channels}),
        "low_density": S.red_wells_absolute(
            plates_data, list(r_p10_90_red_thresholds.keys()), r_p10_90_red_thresholds),
    }

    # ── Heatmap colour specs — robust ones from stats.scores.median_colorscale
    cs_cache: dict = {}
    def cs(col):
        if col not in cs_cache:
            cs_cache[col] = S.median_colorscale(col, plates_data)
        return cs_cache[col]

    slope_specs = [{"col": col, "title": f"Slope — {ch}",
                    "cs": cs(col)[0], "cmin": cs(col)[1], "cmax": cs(col)[2]}
                   for ch in channels
                   for col in [f"ImageQuality_PowerLogLogSlope_{ch}"]]
    medianint_specs = [{"col": col, "title": f"MedianInt — {ch}",
                        "cs": cs(col)[0], "cmin": cs(col)[1], "cmax": cs(col)[2]}
                       for ch in channels
                       for col in [f"ImageQuality_MedianIntensity_{ch}"]]
    pct_max_specs = [{"col": col, "title": f"PctMaximal — {ch}",
                      "cmin": 0, "cmax": 2.0, "cs": _PCT_MAX_CS}
                     for ch in channels
                     for col in [f"ImageQuality_PercentMaximal_{ch}"]]
    r_p10_p90_specs = [{"col": "Ratio_P10/P90", "title": "Low density — P10/P90"}]

    count_ranges = {
        col: list(_cohort_range(col, plates_data, 0, None))
        for col in ["Count_Cells", "Count_Nuclei", "Count_Illum_artifacts_filtered"]
    }

    # ── MFI: { plate: { channel: { well: [vals] } } } ────────────────────────
    mfi_payload: dict = {}
    for pd_ in plates_data:
        by_ch: dict = {}
        for well, ch_vals in pd_.get("mfi_data", {}).items():
            for ch, vals in ch_vals.items():
                by_ch.setdefault(ch, {})[well] = vals
        mfi_payload[pd_["name"]] = by_ch

    all_mfi_channels = [ch for ch in channels
                        if any(ch in by_ch for by_ch in mfi_payload.values())]
    mfi_colors = {ch: (profile.mfi_color(ch) or "#8ab0d0")
                  for ch in all_mfi_channels}

    # Cohort-wide median MFI per channel (all sites of all wells of all plates).
    # Used by the frontend to express each row/column's drift vs the cohort.
    mfi_cohort_median = {}
    for ch in all_mfi_channels:
        vals = [v for by_ch in mfi_payload.values()
                for well_vals in by_ch.get(ch, {}).values()
                for v in well_vals]
        if vals:
            mfi_cohort_median[ch] = round(float(np.median(vals)), 6)
    mfi_img_payload = {pd_["name"]: pd_.get("mfi_img", {}) for pd_ in plates_data}
    # Per-site MFI for the CSV export (object + image), keyed by real ImageNumber.
    mfi_sites_payload = {pd_["name"]: pd_.get("mfi_sites", {}) for pd_ in plates_data}
    mfi_img_sites_payload = {pd_["name"]: pd_.get("mfi_img_sites", {})
                             for pd_ in plates_data}
    radius_payload = {pd_["name"]: pd_.get("radius_data", {}) for pd_ in plates_data}

    n_rows = plates_data[0].get("plate_rows", profile.plate_rows)
    n_cols = plates_data[0].get("plate_cols", profile.plate_cols)

    # PCA over well features to flag deviant plates (empty when not enough data).
    from qc.stats.pca import compute_pca
    pca_payload = compute_pca(plates_data, profile)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "data": data,
        "countRanges": count_ranges,
        "mfi": {
            "data": mfi_payload,
            "channels": all_mfi_channels,
            "colors": mfi_colors,
            "img": mfi_img_payload,
            "sites": mfi_sites_payload,
            "imgSites": mfi_img_sites_payload,
            "cohortMedian": mfi_cohort_median,
        },
        "radius": radius_payload,
        "specs": {
            "slope": slope_specs,
            "debris": pct_max_specs,
            "low_density": r_p10_p90_specs,
            "medianInt": medianint_specs,
        },
        "madScores": calc_mad_scores,
        "redCounts": red_counts,
        "robustMetrics": [f"ImageQuality_{m}_" for m in profiles.ROBUST_METRICS],
        "colorSpec": color_spec_payload(
            [f"ImageQuality_{m}_" for m in profiles.ROBUST_METRICS]),
        "pca": pca_payload,
    }


def _compute_well_flags(pqc: dict, adp: dict, metric_cols: dict,
                        col_to_channel: dict) -> dict:
    """Per-well absolute/adaptive failure tags. Verbatim from report.build_payload.

    A metric fails 'abs' if it breaks the absolute bound; 'adp' if it breaks the
    adaptive band AND the absolute bound in the same direction. abs takes
    precedence (a well already failing abs is not double-tagged as adp).
    """
    well_flags: dict = {}
    for well, m in pqc.items():
        abs_fails, adp_fails = [], []
        for mk, cols in metric_cols.items():
            for col in cols:
                v = m.get(col)
                ch = col_to_channel.get(col, "")
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                if mk == "FocusScore":
                    lo, hi = profiles.THRESHOLDS_LOCAL_FOCUS.get(ch, (None, None))
                else:
                    lo, hi = profiles.THRESHOLDS.get(mk, (None, None))
                abs_fail = (lo is not None and v < lo) or (hi is not None and v > hi)
                adp_col = adp.get(mk, {}).get(col)
                adp_fail = False
                if adp_col:
                    adp_lo, adp_hi = adp_col
                    adp_fail = (v < adp_lo and lo is not None and v < lo) or \
                               (v > adp_hi and hi is not None and v > hi)
                tag = f"{profiles.METRIC_LABELS.get(mk, mk)}/{ch}"
                if abs_fail:
                    abs_fails.append(tag)
                elif adp_fail:
                    adp_fails.append(tag)
        if abs_fails or adp_fails:
            well_flags[well] = {"abs": abs_fails, "adp": adp_fails}
    return well_flags
"""
qc/render/collage.py — Plate collage assembly (labels, pass/fail bands, header,
footer) plus the report QC grid.

Phase 0 moved _make_report_collage here verbatim. Phase 4 completes the module
with the full drawing body of the old Collage._render_plate: _make_band,
make_header, make_report_footer, _slope_to_rgb/_slope_range, the well pass/fail
helpers and the per-well MFI aggregation. render_plate_collage() is the
orchestrator that assembles a full plate image from montages + per-well QC.

All statistics come through the ThresholdEngine (which carries the active
DatasetProfile); no import from III_QC_collage. Instance-derived vocabulary
(channels, metric columns, col→channel) is read from engine.profile so a single
profile drives both stats and drawing.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl
from PIL import Image, ImageDraw

from qc import profiles
from qc.render.montage import _font, _text_h, _make_tile, well_label
from qc.stats.engine import ThresholdEngine

BORDER_METRIC = "PowerLogLogSlope"


def _make_report_collage(collage_arr: np.ndarray, plate_qc: dict,
                         plate_map: dict, plate_rows: int = 8,
                         plate_cols: int = 12,
                         robust_verdict: dict | None = None) -> np.ndarray:
    # robust_verdict: {well: {metric: bool_fail}} del peor sitio (Slope/MedInt),
    # de site_flags.well_robust_verdict[plate]. El borde del tile refleja el peor
    # sitio; si no se provee, el tile queda neutro (sin veredicto robusto).
    robust_verdict = robust_verdict or {}
    CW, CH   = 120, 80
    W        = CW * plate_cols
    H        = CH * plate_rows
    img      = Image.new("RGB", (W, H), color=(10, 12, 22))
    draw     = ImageDraw.Draw(img)

    fwell = _font(15, bold=True)
    fcmpd = _font(10, bold=False)
    fval  = _font(10, bold=False)

    slope_cols = [f"ImageQuality_{BORDER_METRIC}_{ch}"
                  for ch in _slope_channels(plate_qc)]

    for r in range(plate_rows):
        for c in range(plate_cols):
            wl       = well_label(r + 1, c + 1)
            metrics  = plate_qc.get(wl, {})
            compound = (plate_map or {}).get(wl, "")
            x0, y0   = c * CW, r * CH
            x1, y1   = x0 + CW - 1, y0 + CH - 1

            # Borde del tile = veredicto del PEOR sitio para Slope (BORDER_METRIC),
            # vía z por sitio (placa/cohorte). Verde pasa, rojo falla; gris sin dato.
            wv = robust_verdict.get(wl)
            if not wv or BORDER_METRIC not in wv:
                bg, border = (20, 22, 35), (40, 45, 70)
                slope_fail = None
            elif wv[BORDER_METRIC]:
                bg, border = (42, 12, 12), (200, 35, 35)
                slope_fail = True
            else:
                bg, border = (15, 42, 20), (55, 180, 70)
                slope_fail = False

            draw.rectangle([x0, y0, x1, y1], fill=bg, outline=border, width=2)
            draw.text((x0 + 5, y0 + 4), wl, fill=(200, 220, 255), font=fwell)

            if compound and compound.lower() not in ("", "nan", "dmso"):
                disp = compound[:14] + "\u2026" if len(compound) > 14 else compound
                draw.text((x0 + 5, y0 + 22), disp, fill=(160, 200, 140), font=fcmpd)
            elif compound.upper() == "DMSO":
                draw.text((x0 + 5, y0 + 22), "DMSO", fill=(120, 140, 180), font=fcmpd)

            cells = metrics.get(profiles.COUNT_COLS["Cells"])
            if cells is not None and not (isinstance(cells, float) and np.isnan(cells)):
                draw.text((x0 + 5, y0 + 36), f"n={int(cells)}", fill=(160, 175, 200), font=fval)

            sv = metrics.get("ImageQuality_PowerLogLogSlope_Hoechst")
            if sv is not None and not (isinstance(sv, float) and np.isnan(sv)):
                # Color del texto sl= según el veredicto robusto del pozo (no umbral fijo).
                sc = (160, 175, 200) if slope_fail is None else \
                     ((255, 65, 65) if slope_fail else (75, 215, 95))
                draw.text((x0 + 5, y0 + 50), f"sl={sv:.2f}", fill=sc, font=fval)

    return np.array(img)


def _slope_channels(plate_qc: dict) -> list:
    """Channels that actually have a slope column in the data (order-agnostic)."""
    chans = []
    for m in plate_qc.values():
        for col in m:
            if col.startswith(f"ImageQuality_{BORDER_METRIC}_"):
                ch = col.rsplit("_", 1)[-1]
                if ch not in chans:
                    chans.append(ch)
        if chans:
            break
    return chans


def _artifact_density(metrics: dict) -> float | None:
    n = metrics.get(profiles.COUNT_COLS["Artifacts"])
    if n is None or np.isnan(n):
        return None
    area = metrics.get(profiles.AREA_COL) or profiles.DEFAULT_IMAGE_AREA
    return (n / area) * 1000 if area > 0 else None

def _count_summary(metrics: dict) -> list[tuple[str, str]]:
    rows = [(lbl, "—" if (v := metrics.get(col)) is None or
             (isinstance(v, float) and np.isnan(v)) else str(int(round(v))))
            for lbl, col in profiles.COUNT_COLS.items()]
    d = _artifact_density(metrics)
    rows.append(("Art/kpx²", "—" if d is None else f"{d:.2f}"))
    return rows

def _slope_range(plate_qc: dict) -> tuple:
    # Rango de la escala de Slope, tomado de los datos. El veredicto de Slope es
    # z por sitio (no un umbral absoluto); aquí solo se necesita un rango para la
    # escala, con un fallback neutro cuando no hay valores.
    prefix = f"ImageQuality_{BORDER_METRIC}_"
    vals = [v for m in plate_qc.values() for col, v in m.items()
            if col.startswith(prefix) and v is not None and not np.isnan(v)]
    return (min(vals), max(vals)) if vals else (-3.0, -1.0)

def _slope_to_rgb(well: str, engine: ThresholdEngine) -> tuple:
    """Color del tile por el veredicto robusto de Slope del pozo (peor sitio)."""
    rv = engine.well_robust_passes(well, BORDER_METRIC)
    if rv is None:
        return (45, 45, 55)
    return (55, 205, 80) if rv else (215, 35, 35)

def _metric_well_fail(well: str, metrics: dict, mk: str, engine: ThresholdEngine) -> bool:
    """True si el pozo FALLA la métrica mk. Robustas (Slope/MedInt): veredicto por
    peor sitio (well_robust_passes). No robustas: borde absoluto por valor."""
    if mk in engine._ROBUST:
        return engine.well_robust_passes(well, mk) is False
    return any(
        engine.passes(metrics.get(col), mk,
                      engine.profile.col_to_channel().get(col, "")) is False
        for col in engine.profile.metric_cols()[mk])

def _well_passes_all(metrics: dict, engine: ThresholdEngine, well: str = "") -> bool:
    return not any(_metric_well_fail(well, metrics, mk, engine)
                   for mk in engine.profile.metric_cols())

def _well_passes_group(metrics: dict, group: list, engine: ThresholdEngine,
                       well: str = "") -> bool:
    return not any(_metric_well_fail(well, metrics, mk, engine) for mk in group)

def _make_band(width: int, well_labels_in_row: list, plate_qc: dict,
               band_height: int, font_size: int, tile_width: int,
               plate_map: dict | None, engine: ThresholdEngine) -> np.ndarray:
    """
    QC band below each plate row. Two-pass: first measures needed height,
    then draws. Absorbs the old _measure_band_height function.
    """
    ALL_METRICS = profiles.ILLUM_METRICS
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
            for mk in profiles.ILLUM_METRICS:
              yc += _text_h(dd, f"{profiles.METRIC_LABELS[mk]}:", fl) + 1
              for col in engine.profile.metric_cols()[mk]:
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

        pass_vals = []
        for mk in ALL_METRICS:
            if mk in engine._ROBUST:
                rv = engine.well_robust_passes(wlabel, mk)
                if rv is not None:
                    pass_vals.append(rv)
            else:
                for col in engine.profile.metric_cols()[mk]:
                    p = engine.passes(metrics.get(col) if metrics else None, mk,
                                      engine.profile.col_to_channel().get(col, ""))
                    if p is not None:
                        pass_vals.append(p)

        if pass_vals:
            frac   = sum(pass_vals) / len(pass_vals)
            tint   = (20, 55, 25) if frac == 1.0 else (55, 15, 15) if frac == 0 else (50, 35, 10)
            accent = profiles.COL_THRESHOLDS['COL_PASS'] if frac == 1.0 else (profiles.COL_THRESHOLDS['COL_FAIL'] if frac == 0 else (255, 155, 0))
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
        for mk in profiles.ILLUM_METRICS:
            lbl = profiles.METRIC_LABELS[mk]
            draw.text((x_left, y), f"{lbl}:", fill=(170, 195, 225), font=fl)
            y += _text_h(draw, f"{lbl}:", fl) + 1
            for col in engine.profile.metric_cols()[mk]:
                v = metrics.get(col)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                ch    = engine.profile.col_to_channel().get(col, col)
                short = ch
                if mk in engine._ROBUST:
                    rv = engine.well_robust_passes(wlabel, mk)
                    vc = (profiles.COL_THRESHOLDS["COL_NODATA"] if rv is None else
                          (profiles.COL_THRESHOLDS["COL_PASS"] if rv
                           else profiles.COL_THRESHOLDS["COL_FAIL"]))
                else:
                    vc = engine.val_color(v, mk, ch)
                txt   = f" {short}: {v:.3f}"
                draw.text((x_left, y + 1), txt, fill=(0, 0, 0), font=fv)
                draw.text((x_left, y),     txt, fill=vc,         font=fv)
                y += _text_h(draw, txt, fv) + 1
            y += 3

        y = y_start
        draw.text((x_count, y), "Objects:", fill=(170, 195, 225), font=fl)
        y += _text_h(draw, "Objects:", fl) + 2
        for lbl, val in _count_summary(metrics):
            vc = profiles.COL_THRESHOLDS['COL_NODATA']
            if lbl == "Art/kpx²":
                try:
                    fv_ = float(val)
                    vc  = profiles.COL_THRESHOLDS['COL_FAIL'] if fv_ > 0.05 else ((255, 190, 0) if fv_ > 0.02 else profiles.COL_THRESHOLDS['COL_PASS'])
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
    legend = [("Pass", profiles.COL_THRESHOLDS['COL_PASS']), ("Fail", profiles.COL_THRESHOLDS['COL_FAIL']),
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
    n_pass    = sum(1 for wl, m in plate_qc.items() if _well_passes_all(m, engine, wl))
    n_illum_p = sum(1 for wl, m in plate_qc.items()
                    if _well_passes_group(m, profiles.ILLUM_METRICS, engine, wl))
    pct       = lambda n: f"{100 * n / n_wells:.1f}%" if n_wells else "—"

    stats = {mk: {ch: [] for ch in list(engine.profile.channels)} for mk in engine.profile.metric_cols()}
    for m in plate_qc.values():
        for mk, cols in engine.profile.metric_cols().items():
            for col in cols:
                ch = engine.profile.col_to_channel().get(col, col)
                v  = m.get(col)
                if v is not None and not np.isnan(v):
                    stats[mk][ch].append(v)

    failing = []
    for wl, m in sorted(plate_qc.items()):
        reasons = []
        for mk, cols in engine.profile.metric_cols().items():
            if mk in engine._ROBUST:
                # Robustas: veredicto por peor sitio; una razón por métrica.
                if engine.well_robust_passes(wl, mk) is False:
                    reasons.append(profiles.METRIC_LABELS.get(mk, mk))
            else:
                for col in cols:
                    v = m.get(col)
                    ch_key = engine.profile.col_to_channel().get(col, "")
                    if (v is not None and not np.isnan(v)
                            and engine.passes(v, mk, ch_key) is False):
                        reasons.append(
                            f"{profiles.METRIC_LABELS.get(mk, mk)}/"
                            f"{profiles.CHANNEL_LABELS.get(ch_key, '?')}")
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

        oc = profiles.COL_THRESHOLDS['COL_PASS'] if n_pass / max(n_wells, 1) >= 0.8 else \
             (profiles.COL_THRESHOLDS['COL_FAIL'] if n_pass / max(n_wells, 1) < 0.5 else (255, 190, 0))
        line(f"  Overall:       {n_pass}/{n_wells} wells pass all metrics  ({pct(n_pass)})", oc, fs)

        ic = profiles.COL_THRESHOLDS['COL_PASS'] if n_illum_p / max(n_wells, 1) >= 0.8 else \
             (profiles.COL_THRESHOLDS['COL_FAIL'] if n_illum_p / max(n_wells, 1) < 0.5 else (255, 190, 0))
        line(f"  Illumination:  {n_illum_p}/{n_wells} pass ({pct(n_illum_p)})  [Slope + MaxInt]", ic, fb)
        rule(); y += 4

        for grp_lbl, grp_col, grp_metrics in (
            ("ILLUMINATION METRICS", (255, 200, 80), profiles.ILLUM_METRICS),
        ):
            line(f"  {grp_lbl}", grp_col, fs)
            line(f"    {'Metric':<22}  {'Channel':<8}  {'Threshold':<14}  {'Pass':>12}  {'mean ± SD':<22}",
                 (110, 140, 175), fb)

            for mk in grp_metrics:
                ml = profiles.METRIC_LABELS.get(mk, mk)
                first = True
                for col in engine.profile.metric_cols()[mk]:
                    ch       = engine.profile.col_to_channel().get(col, col)
                    vals     = stats[mk].get(ch, [])
                    ch_short = ch
                    ch_color = profiles.CHANNEL_COLORS.get(ch, (150, 170, 200))

                    if mk == "FocusScore":
                        lo, hi = profiles.THRESHOLDS_LOCAL_FOCUS.get(ch, (None, None))
                    else:
                        lo, hi = profiles.THRESHOLDS.get(mk, (None, None))
                    tstr = (f"> {lo}" if hi is None else f"< {hi}" if lo is None
                            else f"{lo} to {hi}")

                    if mk in engine._ROBUST:
                        # Conteo por POZO (veredicto peor sitio), no por valor.
                        verds = [engine.well_robust_passes(wl, mk)
                                 for wl in plate_qc]
                        verds = [x for x in verds if x is not None]
                        tstr = f"|z|≤{ '3' if mk=='PowerLogLogSlope' else '5' } (site)"
                        if verds:
                            np_ = sum(verds)
                            pp  = 100 * np_ / len(verds)
                            pc  = profiles.COL_THRESHOLDS['COL_PASS'] if pp >= 80 else (profiles.COL_THRESHOLDS['COL_FAIL'] if pp < 50 else (255, 190, 0))
                            stat_str = (f"{np_}/{len(verds)} ({pp:.0f}%)   "
                                        f"μ={np.mean(vals):.3f}±{np.std(vals):.3f}"
                                        if vals else f"{np_}/{len(verds)} ({pp:.0f}%)")
                        else:
                            pc, stat_str = profiles.COL_THRESHOLDS['COL_NODATA'], "—"
                    elif vals:
                        np_ = sum(1 for v in vals if engine.passes(v, mk, ch))
                        pp  = 100 * np_ / len(vals)
                        pc  = profiles.COL_THRESHOLDS['COL_PASS'] if pp >= 80 else (profiles.COL_THRESHOLDS['COL_FAIL'] if pp < 50 else (255, 190, 0))
                        stat_str = f"{np_}/{len(vals)} ({pp:.0f}%)   μ={np.mean(vals):.3f}±{np.std(vals):.3f}"
                    else:
                        pc, stat_str = profiles.COL_THRESHOLDS['COL_NODATA'], "—"

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
                     profiles.COL_THRESHOLDS['COL_FAIL'], fr)
        else:
            line("    None — all wells pass.", profiles.COL_THRESHOLDS['COL_PASS'], fr)
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
            dc       = (profiles.COL_THRESHOLDS['COL_FAIL'] if density is not None and density > 0.05
                        else (255, 190, 0) if density is not None and density > 0.02
                        else profiles.COL_THRESHOLDS['COL_PASS'] if density is not None else profiles.COL_THRESHOLDS['COL_NODATA'])
            row_txt  = (f"    {wl:<6}  {compound[:28]:<28}  "
                        f"{fmt(m.get(profiles.COUNT_COLS['Raw'])):>6}  "
                        f"{fmt(m.get(profiles.COUNT_COLS['Filtered'])):>8}  "
                        f"{fmt(m.get(profiles.COUNT_COLS['Cells'])):>6}  "
                        f"{fmt(m.get(profiles.COUNT_COLS['Artifacts'])):>9}  ")
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
        for mk, (lo, hi) in profiles.THRESHOLDS.items():
            tstr = (f"> {lo}" if hi is None else f"< {hi}" if lo is None else f"{lo} to {hi}")
            line(f"    {profiles.METRIC_LABELS.get(mk, mk)}: {tstr}", (150, 170, 195), fr)
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

def _aggregate_mfi_per_well(source_dfs: dict,
                             channel_map: dict,
                             plate_name: str) -> "dict[str, dict[str, list]]":
    """
    Aggregate per-object MFI → { well: { channel: [per_image_median, ...] } }.
    Groups objects by ImageNumber first (one value per image/site), then collects
    those per-image medians into a list per well — what the boxplots display.
    """
    def _norm_plate(s: str) -> str:
        return str(s).strip().lstrip("P").lstrip("0") or "0"

    target = _norm_plate(plate_name)
    result: dict = defaultdict(lambda: defaultdict(list))

    for ch, (col, _) in channel_map.items():
        df = source_dfs.get(ch)
        if df is None:
            continue

        norm_expr = (
            pl.col("Metadata_Plate").cast(pl.Utf8).str.strip_chars()
              .str.strip_chars_start("P").str.strip_chars_start("0")
              .replace("", "0")
        )
        plate_df = df.filter(norm_expr == target)

        if plate_df.is_empty():
            if df["Metadata_Plate"].n_unique() == 1:
                plate_df = df
            else:
                continue

        if "ImageNumber" in plate_df.columns:
            per_image = plate_df.group_by(
                ["Metadata_Plate", "Metadata_Well", "ImageNumber"]
            ).agg(pl.col(col).median())
        else:
            per_image = plate_df

        for row in per_image.iter_rows(named=True):
            well = str(row["Metadata_Well"]).strip().upper()
            v    = row[col]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                result[well][ch].append(float(v))

    return {well: dict(ch_vals) for well, ch_vals in result.items()}


def _aggregate_mfi_sites(source_dfs: dict, channel_map: dict,
                         plate_name: str) -> "dict":
    """Per-object MFI keyed by real ImageNumber, for the CSV export.

    Returns { well: { channel: { ImageNumber: median } } }. Same per-image median
    as _aggregate_mfi_per_well, but retains the ImageNumber instead of appending
    into an order-dependent list — so the CSV's site_index is the real site and
    object vs image values align per site. Does not replace
    _aggregate_mfi_per_well (the boxplots keep consuming its plain value lists).
    """
    def _norm_plate(s: str) -> str:
        return str(s).strip().lstrip("P").lstrip("0") or "0"

    target = _norm_plate(plate_name)
    result: dict = {}

    for ch, (col, _) in channel_map.items():
        df = source_dfs.get(ch)
        if df is None:
            continue
        norm_expr = (
            pl.col("Metadata_Plate").cast(pl.Utf8).str.strip_chars()
              .str.strip_chars_start("P").str.strip_chars_start("0")
              .replace("", "0")
        )
        plate_df = df.filter(norm_expr == target)
        if plate_df.is_empty():
            if df["Metadata_Plate"].n_unique() == 1:
                plate_df = df
            else:
                continue
        if "ImageNumber" not in plate_df.columns:
            continue
        per_image = plate_df.group_by(
            ["Metadata_Well", "ImageNumber"]
        ).agg(pl.col(col).median())
        for row in per_image.iter_rows(named=True):
            well = str(row["Metadata_Well"]).strip().upper()
            img = int(row["ImageNumber"])
            v = row[col]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                result.setdefault(well, {}).setdefault(ch, {})[img] = float(v)
    return result


def render_plate_collage(plate_name: str, montages: dict, plate_qc: dict,
                         plate_map: dict, engine: ThresholdEngine,
                         *, plate_rows: int, plate_cols: int,
                         band_height: int, font_size: int) -> np.ndarray:
    """Assemble a full plate collage: tiled montages + per-row info bands +
    header + optional footer.

    Verbatim port of the drawing portion of Collage._render_plate. Assumes the
    engine has already been fitted on plate_qc. Returns the composed RGB array.
    """
    tile_h, tile_w = next(iter(montages.values())).shape[:2]
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    rows_imgs = []

    for r in range(1, plate_rows + 1):
        row_labels = [well_label(r, c) for c in range(1, plate_cols + 1)]
        row_tiles = [
            _make_tile(montages.get((r, c), blank),
                       _slope_to_rgb(well_label(r, c), engine))
            for c in range(1, plate_cols + 1)
        ]
        row_img = np.concatenate(row_tiles, axis=1)
        rows_imgs.append(row_img)
        rows_imgs.append(_make_band(
            width=row_img.shape[1], well_labels_in_row=row_labels,
            plate_qc=plate_qc, band_height=band_height, font_size=font_size,
            tile_width=tile_w, plate_map=plate_map, engine=engine,
        ))

    collage = np.concatenate(rows_imgs, axis=0)
    s_min, s_max = _slope_range(plate_qc)
    header = make_header(
        collage.shape[1],
        title=f"{plate_name}  \u00b7  Cell Painting QC  "
              f"|  Slope range: {s_min:.2f} \u2013 {s_max:.2f}  "
              f"|  Thresholds: absolute + {engine.n_sigma}\u03c3 MAD")
    collage = np.concatenate([header, collage], axis=0)

    if plate_qc:
        footer = make_report_footer(collage.shape[1], plate_name, plate_qc,
                                    engine, font_size, plate_map)
        collage = np.concatenate([collage, footer], axis=0)

    return collage
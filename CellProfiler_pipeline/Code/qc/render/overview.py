"""
qc/render/overview.py — Grid de miniaturas de placa + helpers base64.

Fase 0: movido VERBATIM desde III_QC_collage. BORDER_METRIC y el pass/fail por
slope se resuelven vía stats.expressions.passes_absolute_scalar, de modo que
render no importa III_QC_collage (rompe ese acoplamiento).
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw

from qc.render.montage import _font, well_label

BORDER_METRIC = "PowerLogLogSlope"


def _collage_to_b64(collage_arr: np.ndarray, web_scale: float = 1.0,
                    quality: int = 60) -> str:
    """Resize array and return as base64 JPEG string."""
    h, w = collage_arr.shape[:2]
    if web_scale != 1.0:
        pil = Image.fromarray(collage_arr).resize(
            (max(1, int(w * web_scale)), max(1, int(h * web_scale))),
            Image.LANCZOS)
    else:
        pil = Image.fromarray(collage_arr)
    buf = io.BytesIO()
    pil.save(buf, format="WEBP", quality=quality,
             optimize=True, progressive=True, subsampling=2)
    return base64.b64encode(buf.getvalue()).decode()


def _well_montage_b64(montages: dict, well_key: tuple,
                      scale_factor: float = 0.30) -> "str | None":
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
                        well_px: int = 90, robust_verdict: dict | None = None) -> tuple:
    # robust_verdict: {well: {metric: bool_fail}} del peor sitio; el borde del
    # thumbnail refleja el veredicto de Slope (BORDER_METRIC). Sin él → neutro.
    robust_verdict = robust_verdict or {}
    CW = CH = well_px
    W  = CW * plate_cols
    H  = CH * plate_rows

    canvas = Image.new("RGB", (W, H), color=(8, 10, 20))
    fdraw  = ImageDraw.Draw(canvas)
    flabel = _font(9, bold=True)
    # Slope columns are discovered from the data (any BORDER_METRIC column), so
    # this stays profile-independent and follows whatever channel set is present.
    _prefix = f"ImageQuality_{BORDER_METRIC}_"
    slope_cols = sorted({col for m in plate_qc.values() for col in m
                         if col.startswith(_prefix)})

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

            wv = robust_verdict.get(wl)
            if not wv or BORDER_METRIC not in wv:
                border = (40, 45, 70)          # sin veredicto → neutro
            elif wv[BORDER_METRIC]:
                border = (210, 35, 35)         # falla (peor sitio)
            else:
                border = (55, 200, 70)         # pasa
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
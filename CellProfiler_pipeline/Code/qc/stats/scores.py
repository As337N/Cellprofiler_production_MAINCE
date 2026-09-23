"""
qc/stats/scores.py — Puntajes y escalas de color a nivel cohorte (Polars).

Porta _mad_scores, _median_cs y _red_wells (hoy en III_QC_collage/report) a
Polars vectorizado. Firma de entrada/salida en dict (compatible con
test_baseline.py y con report.js) mientras el pipeline no sea DataFrame
end-to-end; el puente dict→DataFrame→dict se elimina en Fase 3.

Cada función construye UN DataFrame long [plate, well, value] y computa
median/MAD por grupo con expresiones, en vez de los dobles/triples loops numpy.
"""
from __future__ import annotations

import polars as pl

from qc.stats import expressions as X

_MAD_TO_SIGMA = 1.4826


def _long_frame(plates_data: list, col: str) -> pl.DataFrame:
    """Aplana plates_data a [plate, well, value] para una columna, sin nulls/NaN."""
    rows = []
    for pd_ in plates_data:
        pname = pd_["name"]
        for well, m in pd_["plate_qc"].items():
            v = m.get(col)
            if v is not None and not (isinstance(v, float) and v != v):
                rows.append((pname, well, float(v)))
    if not rows:
        return pl.DataFrame(schema={"plate": pl.Utf8, "well": pl.Utf8, "value": pl.Float64})
    return pl.DataFrame(rows, schema=["plate", "well", "value"], orient="row")


def mad_scores(col: str, plates_data: list) -> tuple[dict, float | None, float | None]:
    """Por well: valor + z contra placa (z_pl) y contra cohorte (z_co).

    Réplica de _mad_scores. Cohort median/MAD en un agg global; plate median/MAD
    con group_by(plate); z vectorizado. Salida idéntica (dict + redondeos).
    """
    df = _long_frame(plates_data, col)
    if df.height < 4:
        return {}, None, None

    coh_med = float(df["value"].median())
    coh_mad = float((df["value"] - coh_med).abs().median()) or 1.0

    # median/MAD por placa en una pasada; join de vuelta.
    per_plate = df.group_by("plate").agg(
        pl.col("value").median().alias("_pmed"),
        (pl.col("value") - pl.col("value").median()).abs().median().alias("_pmad"),
    )
    df = df.join(per_plate, on="plate")
    df = df.with_columns(
        pl.when(pl.col("_pmad") == 0).then(1.0).otherwise(pl.col("_pmad")).alias("_pmad")
    ).with_columns(
        ((pl.col("value") - pl.col("_pmed")) / pl.col("_pmad")).round(2).alias("z_pl"),
        ((pl.col("value") - coh_med) / coh_mad).round(2).alias("z_co"),
        pl.col("value").round(5).alias("v"),
    )

    scores: dict = {}
    for row in df.iter_rows(named=True):
        scores.setdefault(row["plate"], {})[row["well"]] = {
            "v": row["v"], "z_pl": row["z_pl"], "z_co": row["z_co"],
        }
    return scores, round(coh_med, 5), round(coh_mad, 5)


def median_colorscale(col: str, plates_data: list) -> tuple:
    """Colorscale semáforo centrada en mediana cohorte ± 2/3·MAD. Porta _median_cs.

    Devuelve (cs, cmin, cmax) — cs es la lista Plotly de [pos, color], o
    ("RdBu_r", None, None) si hay <4 valores.
    """
    df = _long_frame(plates_data, col)
    if df.height < 4:
        return "RdBu_r", None, None

    arr = df["value"]
    med = float(arr.median())
    mad = float((arr - med).abs().median())
    lo2, lo3 = med - 2 * mad, med - 3 * mad
    hi2, hi3 = med + 2 * mad, med + 3 * mad
    cmin = max(float(arr.min()), lo3 * 0.95)
    cmax = min(float(arr.max()), hi3 * 1.05)
    rng = cmax - cmin
    if rng == 0:
        return "RdBu_r", cmin, cmax

    def pos(v):
        return round(max(0.0, min(1.0, (v - cmin) / rng)), 4)

    cs = [
        [0.0, "#ff4444"], [pos(lo3), "#ff4444"], [pos(lo3), "#ffbe00"],
        [pos(lo2), "#ffbe00"], [pos(lo2), "#4bd760"], [pos(hi2), "#4bd760"],
        [pos(hi2), "#ffbe00"], [pos(hi3), "#ffbe00"], [pos(hi3), "#ff4444"],
        [1.0, "#ff4444"],
    ]
    return cs, round(cmin, 5), round(cmax, 5)


def red_wells_absolute(plates_data: list, cols: list, thresholds: dict) -> dict:
    """Conteo de wells rojos por bordes ABSOLUTOS. Porta report._red_wells.

    Un well es rojo si CUALQUIER col rompe su (lo, hi). Usado por PctMax/PctMin.
    """
    result = {}
    for pd_ in plates_data:
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
        result[pd_["name"]] = count
    return result


def red_wells_zscore(metric: str, channels: list, mad_score_cache: dict,
                     plates_data: list, z_thresh: float = 3.0) -> dict:
    """Conteo de wells rojos por z-score de cohorte. Porta report._zscore_red.

    Un well es rojo si |z_co| > z_thresh en cualquier canal. Usado por las
    métricas robustas (Slope, MedInt). mad_score_cache: {col: mad_scores(col)[0]}.
    """
    result = {}
    for pd_ in plates_data:
        pname = pd_["name"]
        count = 0
        for well in pd_["plate_qc"]:
            for ch in channels:
                col = f"ImageQuality_{metric}_{ch}"
                sc = mad_score_cache.get(col, {}).get(pname, {}).get(well)
                if sc and abs(sc["z_co"]) > z_thresh:
                    count += 1
                    break
        result[pname] = count
    return result

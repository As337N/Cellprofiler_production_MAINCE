"""
qc/stats/site_flags.py — Flag de sitios que fallan algún criterio de calidad.

Produce un parquet con SOLO los sitios fallidos, todas sus mediciones, y dos
columnas de resumen:
  · fail            : bool (siempre True en las filas emitidas).
  · failed_criteria : list[str] de los criterios rotos, con canal cuando aplica.

Criterios (a nivel de SITIO)
----------------------------
Robustas (z-score robusto de cohorte, la ÚNICA fuente de verdad para estas):
  · slope:<ch>        |z| > 3 sobre Intensity/ImageQuality_PowerLogLogSlope_<ch>
  · medint:<ch>       |z| > 3 sobre ImageQuality_MedianIntensity_<ch>
El z se calcula sobre la distribución de SITIOS de la cohorte (no de wells):
    z = (x - mediana_cohorte) / MAD_cohorte,  MAD = mediana(|x - mediana|)
misma fórmula que scores.mad_scores (MAD-score crudo, sin factor 1.4826), pero
agrupando por sitio. El valor crudo se conserva para auditoría.

Absolutos (a nivel de sitio):
  · pctmax:<ch>       ImageQuality_PercentMaximal_<ch> > 1.0
  · percentile10:<ch> Intensity_Percentile_10_<ch>    < 0.015
  · cells_nuclei_ratio  Count_Cells / Count_Nuclei    < 0.95
  · low_cell_count      Count_Cells                    < 70

Un sitio se emite si failed_criteria no está vacío. Para las métricas por-canal,
se lista cada canal culpable por separado (p. ej. "pctmax:Hoechst").
"""
from __future__ import annotations

from qc import profiles

# Umbrales absolutos
PCTMAX_FAIL = 1.0            # >  fail
PERCENTILE10_FAIL = 0.015    # <  fail
CELLS_NUCLEI_RATIO_FAIL = 0.95   # <  fail
LOW_CELL_COUNT_FAIL = 70     # <  fail
ZSCORE_FAIL = 3.0            # |z| > fail (robustas)

ROBUST_METRICS = profiles.ROBUST_METRICS          # ("PowerLogLogSlope", "MedianIntensity")
_CRIT_NAME = {"PowerLogLogSlope": "slope", "MedianIntensity": "medint"}

CELLS_COL = profiles.COUNT_COLS["Cells"]          # Count_Cells
NUCLEI_COL = profiles.COUNT_COLS["Filtered"]      # Count_Nuclei


def _robust_col(metric: str, ch: str) -> str:
    return f"ImageQuality_{metric}_{ch}"


def _percentile10_col(ch: str) -> str:
    return f"Intensity_Percentile_10_{ch}"


def _pctmax_col(ch: str) -> str:
    return f"ImageQuality_PercentMaximal_{ch}"


def _valid(v) -> bool:
    return v is not None and not (isinstance(v, float) and v != v)


def cohort_site_mad(site_qc: dict, col: str):
    """(mediana, MAD) de una columna sobre TODOS los sitios de la cohorte.

    MAD = mediana(|x - mediana|), con guardia MAD==0 -> 1.0 (igual que
    scores.mad_scores). Devuelve (None, None) si <4 valores válidos.
    """
    from statistics import median
    vals = []
    for wells in site_qc.values():
        for fields in wells.values():
            for sv in fields.values():
                v = sv.get(col)
                if _valid(v):
                    vals.append(float(v))
    if len(vals) < 4:
        return None, None
    med = median(vals)
    mad = median([abs(v - med) for v in vals]) or 1.0
    return med, mad


def robust_site_refs(site_qc: dict, profile: profiles.DatasetProfile) -> dict:
    """{col: (mediana, MAD)} de cohorte por sitio para cada métrica robusta×canal."""
    refs = {}
    for metric in ROBUST_METRICS:
        for ch in profile.channels:
            col = _robust_col(metric, ch)
            med, mad = cohort_site_mad(site_qc, col)
            if med is not None:
                refs[col] = (med, mad)
    return refs


def site_failed_criteria(site_vals: dict, profile: profiles.DatasetProfile,
                         robust_refs: dict) -> list[str]:
    """Lista de criterios que este sitio rompe (vacía si pasa todo)."""
    failed = []

    # Robustas por z de sitio
    for metric in ROBUST_METRICS:
        crit = _CRIT_NAME.get(metric, metric.lower())
        for ch in profile.channels:
            col = _robust_col(metric, ch)
            ref = robust_refs.get(col)
            v = site_vals.get(col)
            if ref is None or not _valid(v):
                continue
            med, mad = ref
            z = (float(v) - med) / mad
            if abs(z) > ZSCORE_FAIL:
                failed.append(f"{crit}:{ch}")

    # PercentMaximal > 1.0
    for ch in profile.channels:
        v = site_vals.get(_pctmax_col(ch))
        if _valid(v) and float(v) > PCTMAX_FAIL:
            failed.append(f"pctmax:{ch}")

    # Percentile10 < 0.015
    for ch in profile.channels:
        v = site_vals.get(_percentile10_col(ch))
        if _valid(v) and float(v) < PERCENTILE10_FAIL:
            failed.append(f"percentile10:{ch}")

    # Cells/Nuclei ratio < 0.95
    cells = site_vals.get(CELLS_COL)
    nuclei = site_vals.get(NUCLEI_COL)
    if _valid(cells) and _valid(nuclei) and float(nuclei) > 0:
        if float(cells) / float(nuclei) < CELLS_NUCLEI_RATIO_FAIL:
            failed.append("cells_nuclei_ratio")

    # Bajo conteo de células < 70
    if _valid(cells) and float(cells) < LOW_CELL_COUNT_FAIL:
        failed.append("low_cell_count")

    return failed


def flagged_sites_rows(site_qc: dict, profile: profiles.DatasetProfile) -> list[dict]:
    """Filas (dict) de SOLO los sitios fallidos: metadatos + todas las mediciones
    + fail + failed_criteria. El orden de columnas se resuelve al escribir.
    """
    robust_refs = robust_site_refs(site_qc, profile)
    rows = []
    for plate, wells in site_qc.items():
        for well, fields in wells.items():
            for field, sv in fields.items():
                failed = site_failed_criteria(sv, profile, robust_refs)
                if not failed:
                    continue
                row = {"Metadata_Plate": plate, "Metadata_Well": well,
                       "Metadata_Field": field}
                row.update(sv)                       # todas las mediciones crudas
                row["fail"] = True
                row["failed_criteria"] = failed
                rows.append(row)
    return rows


def write_flagged_sites_parquet(site_qc: dict, profile: profiles.DatasetProfile,
                                out_path) -> int:
    """Escribe el parquet de sitios fallidos. Devuelve el nº de filas escritas.

    Columnas: Metadata_Plate/Well/Field, todas las mediciones presentes (unión de
    columnas sobre las filas), fail (bool), failed_criteria (list[str]).
    """
    import polars as pl

    rows = flagged_sites_rows(site_qc, profile)
    if not rows:
        # Parquet vacío con esquema mínimo, para no romper consumidores.
        empty = pl.DataFrame(schema={
            "Metadata_Plate": pl.Utf8, "Metadata_Well": pl.Utf8,
            "Metadata_Field": pl.Int64, "fail": pl.Boolean,
            "failed_criteria": pl.List(pl.Utf8)})
        empty.write_parquet(out_path)
        return 0

    # Orden de columnas: metadatos, mediciones (ordenadas), resumen al final.
    meas_cols = sorted({k for r in rows for k in r
                        if k not in ("Metadata_Plate", "Metadata_Well",
                                     "Metadata_Field", "fail", "failed_criteria")})
    ordered = ["Metadata_Plate", "Metadata_Well", "Metadata_Field",
               *meas_cols, "fail", "failed_criteria"]
    # Normaliza cada fila a todas las columnas (None donde falte la medición).
    norm = [{c: r.get(c) for c in ordered} for r in rows]
    df = pl.DataFrame(norm, schema=ordered)
    df.write_parquet(out_path)
    return df.height
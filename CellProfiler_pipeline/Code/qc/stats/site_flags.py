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
RATIO_P1090_FAIL = 0.5    # >  fail
CELLS_NUCLEI_RATIO_FAIL = 0.95   # <  fail
LOW_CELL_COUNT_FAIL = 70     # <  fail

# Corte de z robusto por métrica (|z_pl| > k O |z_co| > k = fail). MedianIntensity
# usa 5 sigma (más tolerante a corrimientos de intensidad por la perturbación);
# Slope usa 3 sigma.
ZSCORE_FAIL = {"PowerLogLogSlope": 3.0, "MedianIntensity": 5.0}
ZSCORE_FAIL_DEFAULT = 3.0

# Robustez del MAD frente a dispersión casi nula. Un MAD diminuto convierte
# variaciones triviales en z gigantes (falsos positivos). Se aplica un PISO:
#   MAD_efectivo = max(MAD, MAD_FLOOR_FRAC*|mediana|, MAD_FLOOR_ABS)
# El z por PLACA solo se calcula con la MAD real si la placa tiene > MIN_PLATE_SITES
# sitios válidos; con <= ese número se usa el piso (opción 1) y se registra la
# placa/columna para una nota en el reporte.
MIN_PLATE_SITES = 15
MAD_FLOOR_FRAC = 0.02        # 2% de |mediana|
MAD_FLOOR_ABS = 1e-9         # mínimo absoluto para el caso mediana ~ 0

ROBUST_METRICS = profiles.ROBUST_METRICS          # ("PowerLogLogSlope", "MedianIntensity")
_CRIT_NAME = {"PowerLogLogSlope": "slope", "MedianIntensity": "medint"}

CELLS_COL = profiles.COUNT_COLS["Cells"]          # Count_Cells
NUCLEI_COL = profiles.COUNT_COLS["Filtered"]      # Count_Nuclei


def _zcut(metric: str) -> float:
    return ZSCORE_FAIL.get(metric, ZSCORE_FAIL_DEFAULT)


def _robust_col(metric: str, ch: str) -> str:
    return f"ImageQuality_{metric}_{ch}"


def _pctmax_col(ch: str) -> str:
    return f"ImageQuality_PercentMaximal_{ch}"


def _valid(v) -> bool:
    return v is not None and not (isinstance(v, float) and v != v)


def _mad_floor(med: float) -> float:
    return max(MAD_FLOOR_FRAC * abs(med), MAD_FLOOR_ABS)


def _mad_ref(vals: list, apply_floor: bool = True):
    """(mediana, MAD_efectivo, floored) de una lista.

    MAD = mediana(|x-mediana|). Si apply_floor, MAD_efectivo = max(MAD, piso) y
    floored indica si el piso llegó a dominar (MAD real < piso). None si <4 valores.
    """
    from statistics import median
    if len(vals) < 4:
        return None, None, False
    med = median(vals)
    mad_raw = median([abs(v - med) for v in vals])
    if apply_floor:
        floor = _mad_floor(med)
        if mad_raw < floor:
            return med, floor, True
        return med, mad_raw or floor, False
    return med, (mad_raw or 1.0), False


def cohort_site_mad(site_qc: dict, col: str):
    """(mediana, MAD_efectivo, floored) sobre TODOS los sitios de la cohorte."""
    vals = [float(sv[col]) for wells in site_qc.values()
            for fields in wells.values() for sv in fields.values()
            if col in sv and _valid(sv[col])]
    return _mad_ref(vals)


def plate_site_mad(plate_fields: dict, col: str):
    """(mediana, MAD_efectivo, floored) sobre los sitios de UNA placa.

    Si la placa tiene <= MIN_PLATE_SITES sitios válidos, el MAD real es
    inestable: se fuerza el piso (floored=True) para no inflar el z. Devuelve
    (None, None, False) con <4 valores.
    """
    vals = [float(sv[col]) for fields in plate_fields.values()
            for sv in fields.values()
            if col in sv and _valid(sv[col])]
    if len(vals) < 4:
        return None, None, False
    if len(vals) <= MIN_PLATE_SITES:
        from statistics import median
        med = median(vals)
        return med, _mad_floor(med), True     # pocos sitios → piso
    return _mad_ref(vals)


def robust_site_refs(site_qc: dict, profile: profiles.DatasetProfile) -> dict:
    """Referencias de z robusto por sitio, placa y cohorte.

    Returns {
      "cohort": {col: (med, mad, floored)},
      "plate":  {plate: {col: (med, mad, floored)}},
      "floored": [ (scope, plate_or_None, col), ... ]   # para la nota del reporte
    }
    """
    cols = [_robust_col(m, ch) for m in ROBUST_METRICS for ch in profile.channels]
    cohort, plate, floored = {}, {}, []
    for col in cols:
        med, mad, fl = cohort_site_mad(site_qc, col)
        if med is not None:
            cohort[col] = (med, mad, fl)
            if fl:
                floored.append(("cohort", None, col))
    for pname, wells in site_qc.items():
        plate[pname] = {}
        for col in cols:
            med, mad, fl = plate_site_mad(wells, col)
            if med is not None:
                plate[pname][col] = (med, mad, fl)
                if fl:
                    floored.append(("plate", pname, col))
    return {"cohort": cohort, "plate": plate, "floored": floored}


def robust_site_fail(metric: str, ch: str, value, plate: str,
                     refs: dict) -> bool:
    """True si el sitio falla la métrica robusta por z de PLACA o de COHORTE.

    Doble guardia: |z_pl| > k O |z_co| > k, con k = _zcut(metric) (3 Slope, 5
    MedInt). refs es la salida de robust_site_refs. Si falta una referencia, esa
    mitad de la guardia simplemente no dispara (no rompe).
    """
    if not _valid(value):
        return False
    col = _robust_col(metric, ch)
    k = _zcut(metric)
    v = float(value)
    # Cohorte
    cref = refs.get("cohort", {}).get(col)
    if cref is not None:
        med, mad, _ = cref
        if abs((v - med) / mad) > k:
            return True
    # Placa
    pref = refs.get("plate", {}).get(plate, {}).get(col)
    if pref is not None:
        med, mad, _ = pref
        if abs((v - med) / mad) > k:
            return True
    return False


def well_robust_verdict(site_qc: dict, profile: profiles.DatasetProfile,
                        robust_refs: dict | None = None) -> dict:
    """Veredicto por-pozo del PEOR sitio para las métricas robustas.

    Un pozo "falla" una métrica si CUALQUIER sitio suyo la falla (doble guardia
    z placa/cohorte). Es la regla única que el collage y el overview usan para el
    borde del tile (el tile refleja el peor sitio).

    Returns {plate: {well: {metric: bool}}}. metric ∈ ROBUST_METRICS.
    Los módulos de render consultan este dict precomputado — no recalculan z.
    """
    if robust_refs is None:
        robust_refs = robust_site_refs(site_qc, profile)
    out: dict = {}
    for plate, wells in site_qc.items():
        out[plate] = {}
        for well, fields in wells.items():
            wv = {}
            for metric in ROBUST_METRICS:
                failed = False
                for ch in profile.channels:
                    for sv in fields.values():
                        if robust_site_fail(metric, ch, sv.get(_robust_col(metric, ch)),
                                            plate, robust_refs):
                            failed = True
                            break
                    if failed:
                        break
                wv[metric] = failed
            out[plate][well] = wv
    return out


def site_failed_criteria(site_vals: dict, profile: profiles.DatasetProfile,
                         robust_refs: dict, plate: str) -> list[str]:
    """Lista de criterios que este sitio rompe (vacía si pasa todo).

    robust_refs: salida de robust_site_refs (cohorte + placa).
    plate: nombre de la placa del sitio (para el z por placa).
    """
    failed = []

    # Robustas: doble guardia z por placa O cohorte (3 sigma Slope, 5 MedInt).
    for metric in ROBUST_METRICS:
        crit = _CRIT_NAME.get(metric, metric.lower())
        for ch in profile.channels:
            v = site_vals.get(_robust_col(metric, ch))
            if robust_site_fail(metric, ch, v, plate, robust_refs):
                failed.append(f"{crit}:{ch}")

    # PercentMaximal > 1.0
    for ch in profile.channels:
        v = site_vals.get(_pctmax_col(ch))
        if _valid(v) and float(v) > PCTMAX_FAIL:
            failed.append(f"pctmax:{ch}")

    # Ratio_P10/P90 > 0.5
    v = site_vals.get("Ratio_P10/P90")
    if _valid(v) and float(v) > RATIO_P1090_FAIL:
        failed.append(f"Ratio_P10/P90")

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


def floored_note(robust_refs: dict) -> str | None:
    """Nota para el reporte: qué placas usaron el piso de MAD por tener <=
    MIN_PLATE_SITES sitios (z por placa menos fiable). None si ninguna.
    """
    floored = robust_refs.get("floored", [])
    plates = sorted({p for scope, p, _ in floored if scope == "plate" and p})
    if not plates:
        return None
    return (f"Per-plate robust z used a MAD floor for {len(plates)} plate(s) with "
            f"≤{MIN_PLATE_SITES} sites ({', '.join(plates)}): site-level plate "
            f"z-scores there are less sensitive. Cohort z is unaffected.")


def flagged_sites_rows(site_qc: dict, profile: profiles.DatasetProfile) -> list[dict]:
    """Filas (dict) de SOLO los sitios fallidos: metadatos + todas las mediciones
    + fail + failed_criteria. El orden de columnas se resuelve al escribir.
    """
    robust_refs = robust_site_refs(site_qc, profile)
    rows = []
    for plate, wells in site_qc.items():
        for well, fields in wells.items():
            for field, sv in fields.items():
                failed = site_failed_criteria(sv, profile, robust_refs, plate)
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
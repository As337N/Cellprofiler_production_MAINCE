"""
qc/stats/aggregate.py — Colapso site → well (Polars).

Porta loaders._classify_cols y collapse_sites_to_wells. La política de colapso
por columna:

  · métricas (Slope, MedInt)            → mean   (promedio de sitios)
  · percentages (PctMaximal/PctMinimal) → worst  (peor sitio)  ← cambio clave
  · Count_*/AREA_COL                     → sum
  · desconocidas                         → mean   (default seguro)

Cambio de diseño (peor-sitio para percentages)
----------------------------------------------
PercentMaximal e Ratio_P10/P90 son detectores de ANOMALÍA PUNTUAL: un
solo sitio con debris (p. ej. Hoechst = 1.89) o dominado por fondo negro señala un
problema real de imagen. Promediar sitios diluía ese pico y lo escondía. La unidad
de detección es el SITIO, así que el well se resume por su PEOR sitio, no por la
media. El escalar del well pasa a ser ese peor valor -el contrato escalar aguas
abajo (scores, platemaps) no cambia- y ademas se retiene qué sitio lo produjo,
para dirigir la búsqueda de outliers.

"Peor" depende de la dirección del fallo de la métrica:
  · PercentMaximal          -> el MAXIMO (más píxeles saturados = peor).
  · Ratio_P10/P90           -> el MAXIMO (ratio cercano a uno indica fondo
                               negro predominante = peor).
La dirección se declara en PCT_WORST_DIRECTION.

collapse_sites_to_wells conserva su firma (dict escalar) para no romper llamadas
existentes. collapse_sites_to_wells_with_sources añade, por well, el sitio
responsable de cada columna colapsada-por-peor: {well: {col: field}}.
"""
from __future__ import annotations

from qc import profiles

# Direccion de "peor" para las métricas de percentil. +1 -> peor es el máximo;
# -1 -> peor es el mínimo.
#   PercentMaximal: peor = máximo (más saturación).
#   Ratio_P10/P90: peor = MÍNIMO (piso de intensidad más bajo = fondo
#     negro predominante). 
PCT_WORST_DIRECTION = {
    "ImageQuality_PercentMaximal_": +1,
    "Ratio_P10/P90": +1,
}


def _pct_direction(col: str) -> int:
    for prefix, d in PCT_WORST_DIRECTION.items():
        if col.startswith(prefix):
            return d
    return +1


def _is_pct(col: str) -> bool:
    return (col.startswith("ImageQuality_PercentMaximal_") or
            col.startswith("Ratio_P10/P90"))


def classify_columns(cols, profile: profiles.DatasetProfile):
    """(mean_cols, pct_cols, sum_cols). Réplica de loaders._classify_cols.

    pct_cols son las métricas de percentil, que ahora se colapsan por peor-sitio
    (no por mean). El nombre se conserva por compatibilidad de llamadas.
    """
    metric_cols = {c for vals in profile.metric_cols().values() for c in vals}
    mean_cols, pct_cols, sum_cols = [], [], []
    for c in cols:
        if c in metric_cols:
            mean_cols.append(c)
        elif _is_pct(c):
            pct_cols.append(c)
        elif c.startswith("Count_") or c == profiles.AREA_COL:
            sum_cols.append(c)
        else:
            mean_cols.append(c)
    return mean_cols, pct_cols, sum_cols


def _valid_site_values(fields: dict, col: str) -> list:
    """[(field, value), ...] para una columna, omitiendo null/NaN.

    fields: {field: {col: val}}. field puede ser int o str; se conserva tal cual
    para que el 'responsable' reportado coincida con la clave de sitio del resto
    del payload (site_data usa str(field)).
    """
    out = []
    for field, fv in fields.items():
        if col in fv:
            v = fv[col]
            if v is not None and not (isinstance(v, float) and v != v):
                out.append((field, float(v)))
    return out


def collapse_sites_to_wells(site_qc: dict, profile: profiles.DatasetProfile) -> dict:
    """{plate:{well:{field:{col:val}}}} -> {plate:{well:{col:val}}}.

    Métricas promediadas, counts/area sumados, percentages por PEOR sitio,
    ignorando null/NaN. Un well sin valor válido para una columna la omite.
    Firma y tipo de salida idénticos al original (dict escalar).
    """
    collapsed, _ = collapse_sites_to_wells_with_sources(site_qc, profile)
    return collapsed


def collapse_sites_to_wells_with_sources(site_qc: dict,
                                         profile: profiles.DatasetProfile):
    """Como collapse_sites_to_wells, pero devuelve también los responsables.

    Returns
    -------
    (collapsed, worst_sites)
        collapsed   : {plate: {well: {col: val}}}  - escalar por well.
        worst_sites : {plate: {well: {col: field}}} - SOLO para columnas de
                      percentil, el sitio (field) que produjo el peor valor.
                      Vacío para wells/cols sin datos de percentil.
    """
    result: dict = {}
    worst: dict = {}
    for plate, wells in site_qc.items():
        result[plate] = {}
        worst[plate] = {}
        for well, fields in wells.items():
            all_cols = {c for fv in fields.values() for c in fv}
            mean_cols, pct_cols, sum_cols = classify_columns(all_cols, profile)

            collapsed: dict = {}
            well_worst: dict = {}

            for col in mean_cols:
                vals = [v for _, v in _valid_site_values(fields, col)]
                if vals:
                    collapsed[col] = sum(vals) / len(vals)

            for col in pct_cols:
                fv = _valid_site_values(fields, col)
                if not fv:
                    continue
                direction = _pct_direction(col)
                if direction > 0:
                    worst_field, worst_val = max(fv, key=lambda t: t[1])
                else:
                    worst_field, worst_val = min(fv, key=lambda t: t[1])
                collapsed[col] = worst_val
                well_worst[col] = worst_field

            for col in sum_cols:
                vals = [v for _, v in _valid_site_values(fields, col)]
                if vals:
                    collapsed[col] = sum(vals)

            result[plate][well] = collapsed
            if well_worst:
                worst[plate][well] = well_worst
    return result, worst
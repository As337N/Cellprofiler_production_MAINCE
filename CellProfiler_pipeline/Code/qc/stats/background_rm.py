"""
qc/stats/background.py — Referencia de cohorte para el "piso de intensidad".

Punto 4 (rediseñado): PercentMinimal se reemplaza POR COMPLETO por
Intensity_Percentile_10_<canal> (percentil 10 de intensidad de la imagen por
canal, tabla Image). Un campo dominado por fondo negro tiene el p10 muy bajo:
el grueso de la imagen está pegado al mínimo de intensidad.

El color de esta métrica es por RATIO vs la mediana de cohorte del canal (modo
low_ratio en colors.py): value/ref, donde ratio bajo = piso deprimido = fondo.
Este módulo solo calcula esas medianas de cohorte por columna, que viajan al
payload (colorSpec.refs) para que front y back coloreen igual.

No hay métrica derivada ni columna extra: la métrica visible es el propio
Intensity_Percentile_10 por canal, reemplazando a PercentMinimal 1-a-1.
"""
from __future__ import annotations

from qc import profiles

PERCENTILE10_TPL = "Intensity_Percentile_10_{channel}"
_EXCLUDED_CHANNELS = ("Brightfield",)


def signal_channels(profile: profiles.DatasetProfile) -> list[str]:
    return [ch for ch in profile.channels if ch not in _EXCLUDED_CHANNELS]


def p10_col(ch: str) -> str:
    return PERCENTILE10_TPL.format(channel=ch)


def p10_columns(profile: profiles.DatasetProfile) -> list[str]:
    """Columnas Percentile_10 por canal (todos los canales, incl. Brightfield)."""
    return [p10_col(ch) for ch in profile.channels]


def cohort_p10_medians(site_qc: dict, profile: profiles.DatasetProfile) -> dict:
    """{col: mediana_cohorte} del p10 por canal, sobre TODOS los sitios de todas
    las placas. Referencia para el color por ratio. Omite null/NaN. Solo incluye
    columnas presentes en los datos.
    """
    from statistics import median
    cols = p10_columns(profile)
    acc = {c: [] for c in cols}
    for wells in site_qc.values():
        for fields in wells.values():
            for site_vals in fields.values():
                for c in cols:
                    v = site_vals.get(c)
                    if v is not None and not (isinstance(v, float) and v != v):
                        acc[c].append(float(v))
    return {c: median(vals) for c, vals in acc.items() if vals}
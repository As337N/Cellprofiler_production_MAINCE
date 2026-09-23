"""
qc/stats/expressions.py — Expresiones Polars puras y reutilizables.

Esta es la base directa de tu notebook (test_load_statistics_V3). La idea:
la estadística deja de ser loops Python+numpy sobre dicts y pasa a ser pl.Expr
componibles que Polars vectoriza y paraleliza sobre todos los cores.

Todas las funciones aquí devuelven pl.Expr — NO tocan datos. Se aplican con
with_columns / agg / select según el nivel. Son puras y triviales de testear:
das una Series de entrada, comparas la de salida.
"""
from __future__ import annotations

import polars as pl

# Constante MAD→sigma para consistencia con scipy median_abs_deviation(scale="normal")
_MAD_TO_SIGMA = 1.4826


def median(col: str) -> pl.Expr:
    return pl.col(col).median()


def mad(col: str, *, scaled: bool = True) -> pl.Expr:
    """Median Absolute Deviation. scaled=True lo normaliza a ~sigma gaussiana.

    Reemplaza los np.median(np.abs(arr - med)) dispersos en _mad_scores,
    _median_cs y ThresholdEngine.fit.
    """
    m = pl.col(col).median()
    raw = (pl.col(col) - m).abs().median()
    return (_MAD_TO_SIGMA * raw) if scaled else raw


def robust_z(col: str, *, alias: str | None = None) -> pl.Expr:
    """(x - median) / (1.4826 * MAD). Tu robust_z_score del notebook, generalizado.

    Se aplica dentro de un group_by (por placa o cohorte) para que median/MAD
    se computen por grupo automáticamente — sin el doble loop de _mad_scores.
    """
    name = alias or f"{col}__z"
    return ((pl.col(col) - pl.col(col).median()) / mad(col)).alias(name)


def passes_absolute(col: str, lo: float | None, hi: float | None) -> pl.Expr:
    """True si lo<=x<=hi (bordes None = sin límite). Vectoriza _passes_absolute.

    Devuelve una expresión booleana; null se propaga como null (= sin dato),
    preservando la semántica actual (True/False/None).
    """
    expr = pl.lit(True)
    if lo is not None:
        expr = expr & (pl.col(col) >= lo)
    if hi is not None:
        expr = expr & (pl.col(col) <= hi)
    return pl.when(pl.col(col).is_null()).then(None).otherwise(expr)


def quantile(col: str, q: float) -> pl.Expr:
    """Cuantil nativo (celda 9 del notebook: q10/q90 para bandas de intensidad)."""
    return pl.col(col).quantile(q)


# ── Forma escalar (Python) ───────────────────────────────────────────────────
# Copiada VERBATIM de III_QC_collage._passes_absolute. El render (overview,
# collage) la necesita a nivel de valor suelto, no de columna. Se importa de
# aquí para no duplicarla y para que render no dependa de III_QC_collage.
# En Fase 2 se prefiere passes_absolute (vectorizada) donde se opere sobre
# DataFrames; esta queda para el path de dibujo píxel-a-píxel.
def passes_absolute_scalar(value, metric_key: str, channel: str = ""):
    """Returns True (pass), False (fail), or None (no data). Solo bordes absolutos."""
    import numpy as np
    from qc import profiles
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if metric_key == "FocusScore":
        lo, hi = profiles.THRESHOLDS_LOCAL_FOCUS.get(channel, (None, None))
    else:
        lo, hi = profiles.THRESHOLDS.get(metric_key, (None, None))
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True
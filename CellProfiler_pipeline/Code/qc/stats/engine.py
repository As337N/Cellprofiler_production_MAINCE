"""
qc/stats/engine.py — ThresholdEngine: absoluto + adaptativo MAD (Polars).

Réplica exacta de III_QC_collage.ThresholdEngine. Semántica idéntica:
  · Absoluto: bordes fijos por métrica/canal (profile.thresholds / local_focus).
  · Adaptativo: median ± n_sigma·MAD por placa; solo marca si además viola el
    borde absoluto en la MISMA dirección.

Cambio: fit() era un doble loop Python (métrica × columna × wells) con
median_abs_deviation. Ahora las bandas se calculan con expresiones Polars sobre
un DataFrame long por placa. passes() conserva la lógica escalar (se usa
punto-a-punto en el render).

Firma dict-compatible: fit(plate_qc) recibe {well: {col: val}}. En Fase 3, fit
tomará un DataFrame directamente.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm as _norm

from qc import profiles

# Factor de escala MAD→sigma idéntico a scipy median_abs_deviation(scale="normal"):
# 1 / norm.ppf(0.75) = 1.4826022185... (no el 1.4826 truncado). Se usa la misma
# constante que el original para reproducir sus bandas bit a bit.
_MAD_TO_SIGMA = 1.0 / _norm.ppf(0.75)


class ThresholdEngine:
    def __init__(self, profile: profiles.DatasetProfile, n_sigma: float = 3.0):
        self.profile = profile
        self.n_sigma = n_sigma
        self._adaptive: dict[str, dict[str, tuple]] = {}   # {metric: {col: (lo, hi)}}

    def fit(self, plate_qc: dict) -> None:
        """Bandas adaptativas por placa. Antes: doble loop numpy; ahora Polars.

        Para cada métrica×columna: median ± n_sigma·MAD sobre los wells de la
        placa, descartando columnas con <5 valores (guard idéntico al original).
        """
        self._adaptive = {}
        metric_cols = self.profile.metric_cols()

        # DataFrame long [col, value] con todos los valores de esta placa.
        for mk, cols in metric_cols.items():
            self._adaptive[mk] = {}
            for col in cols:
                vals = [m[col] for m in plate_qc.values()
                        if col in m and m[col] is not None
                        and not (isinstance(m[col], float) and np.isnan(m[col]))]
                if len(vals) < 5:
                    continue
                s = pl.Series(vals, dtype=pl.Float64)
                median = float(s.median())
                mad = _MAD_TO_SIGMA * float((s - median).abs().median())
                self._adaptive[mk][col] = (median - self.n_sigma * mad,
                                           median + self.n_sigma * mad)

    def passes(self, value, metric_key: str, channel: str = "") -> bool | None:
        """True (pass) / False (fail) / None (no data). Lógica idéntica al original."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None

        if metric_key == "FocusScore":
            abs_lo, abs_hi = profiles.THRESHOLDS_LOCAL_FOCUS.get(channel, (None, None))
        else:
            abs_lo, abs_hi = profiles.THRESHOLDS.get(metric_key, (None, None))

        abs_fail = ((abs_lo is not None and value < abs_lo) or
                    (abs_hi is not None and value > abs_hi))

        col = self.profile.image_col(metric_key, channel) if channel else None
        adp_fail = False
        if col and col in self._adaptive.get(metric_key, {}):
            adp_lo, adp_hi = self._adaptive[metric_key][col]
            if value < adp_lo and (abs_lo is not None and value < abs_lo):
                adp_fail = True
            if value > adp_hi and (abs_hi is not None and value > abs_hi):
                adp_fail = True

        return not (abs_fail or adp_fail)

    def val_color(self, value, metric_key: str, channel: str = "") -> tuple:
        """RGB accent for a value: pass/fail/no-data. Verbatim from the original."""
        p = self.passes(value, metric_key, channel)
        if p is None:
            return profiles.COL_THRESHOLDS["COL_NODATA"]
        return (profiles.COL_THRESHOLDS["COL_PASS"] if p
                else profiles.COL_THRESHOLDS["COL_FAIL"])

    def adaptive_bounds(self, metric_key: str, col: str) -> tuple | None:
        """Return (lo, hi) adaptive bounds for a column, or None if not fitted."""
        return self._adaptive.get(metric_key, {}).get(col)

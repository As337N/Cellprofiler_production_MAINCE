"""
qc/stats/colors.py — Autoridad ÚNICA del color semántico de una métrica QC.

Problema que resuelve
---------------------
Había dos motores de color desincronizados para las MISMAS métricas:

  · Platemaps (heatmaps): las métricas robustas (Slope, MedInt) se coloreaban por
    BANDAS DISCRETAS de |z_cohorte| (verde <=1, ambar 1-3, rojo >3).
  · Plate Browser / paneles de well y sitio / bordes de verdict: las MISMAS
    métricas se coloreaban interpolando el VALOR CRUDO sobre una escala continua.

Resultado: G06 Slope-Syto (z_co = -0.89, claramente verde en el platemap) salía
ROJO en el Plate Browser. Dos reglas distintas para el mismo dato.

Diseño
------
ColorResolver es la única regla. Por métrica declara un MODO:

  · "zband"    -> color por |z_cohorte| en bandas discretas (robustas).
  · "absolute" -> color por el valor crudo contra cortes fijos (percentiles).

Y expone:
  · color_for(spec, value, z_co)  -> hex del color de UN valor (un sitio, un well).
  · worst_color(colors)           -> peor color de una lista (rojo>ambar>verde).

La MISMA especificación (modo + cortes + colores) se serializa al payload
(color_spec_payload) para que report.js implemente un ÚNICO resolveColor() con
esta regla, en vez de tres copias divergentes de specColor()/madMatrix().

Colores canónicos (idénticos a los ya usados en el front):
    verde  #4bd760   ambar #ffbe00   rojo #ff4444
"""
from __future__ import annotations

GREEN = "#4bd760"
AMBER = "#ffbe00"
RED = "#ff4444"

# Peor -> mejor. worst_color escoge el primero presente.
_PRIORITY = (RED, AMBER, GREEN)

# Bandas de |z| para métricas robustas (Slope, MedInt). Coincide con MAD_BAND_CS
# del front: verde |z|<=1, ambar 1<|z|<=3, rojo |z|>3.
ZBAND_GREEN_MAX = 1.0
ZBAND_AMBER_MAX = 3.0

# Cortes absolutos "ALTO = malo" (valor crudo, peor-sitio).
# PercentMaximal: verde <0.1, ambar 0.1-1, rojo >1.
ABSOLUTE_CUTS = {
    "ImageQuality_PercentMaximal_": (0.1, 1.0),
    "Ratio_P10/P90": (0.4, 0.5),
}


def _absolute_prefix(col: str):
    for prefix in ABSOLUTE_CUTS:
        if col.startswith(prefix):
            return prefix
    return None


class ColorResolver:
    """Regla única de color. Sin estado por-dataset: opera sobre spec + valor."""

    def __init__(self, robust_prefixes, cohort_refs=None):
        # robust_prefixes: prefijos de métricas robustas (zband).
        # cohort_refs: ya no se usa (cortes absolutos); se acepta por compatibilidad.
        self.robust_prefixes = tuple(robust_prefixes)
        self.cohort_refs = dict(cohort_refs or {})

    # ── Modo de una columna ──────────────────────────────────────────────────
    def mode(self, col: str) -> str:
        if any(col.startswith(p) for p in self.robust_prefixes):
            return "zband"
        return "absolute"

    # ── Color de un valor individual ─────────────────────────────────────────
    def color_for(self, col: str, value, z_co=None, ref=None):
        """Hex del color de un valor, según el modo de la columna.

        zband        usa |z_co| (ignora value); None -> None.
        absolute_low usa value contra cortes BAJO=malo (fail, warn): verde >warn,
                     ambar fail..warn, rojo <fail. Para Ratio_P10/P90.
        absolute     usa value contra cortes ALTO=malo (lo, hi).
        """
        mode = self.mode(col)
        if mode == "zband":
            if z_co is None:
                return None
            az = abs(z_co)
            if az <= ZBAND_GREEN_MAX:
                return GREEN
            if az <= ZBAND_AMBER_MAX:
                return AMBER
            return RED
        # absolute (ALTO = malo)
        if value is None or (isinstance(value, float) and value != value):
            return None
        prefix = _absolute_prefix(col)
        if prefix is None:
            return None
        lo, hi = ABSOLUTE_CUTS[prefix]
        if value < lo:
            return GREEN
        if value <= hi:
            return AMBER
        return RED

    # ── Peor color de un conjunto ────────────────────────────────────────────
    @staticmethod
    def worst_color(colors):
        present = {c for c in colors if c}
        for c in _PRIORITY:
            if c in present:
                return c
        return None


def color_spec_payload(robust_prefixes, cohort_refs=None) -> dict:
    """Especificación serializable para que report.js aplique la MISMA regla.

    cohort_refs se ignora (los cortes de Percentile_10 son absolutos ahora); se
    mantiene el parámetro por compatibilidad con la firma existente.
    """
    return {
        "robustPrefixes": list(robust_prefixes),
        "zband": {"greenMax": ZBAND_GREEN_MAX, "amberMax": ZBAND_AMBER_MAX},
        "absolute": {p: {"lo": lo, "hi": hi} for p, (lo, hi) in ABSOLUTE_CUTS.items()},
        "colors": {"green": GREEN, "amber": AMBER, "red": RED},
    }
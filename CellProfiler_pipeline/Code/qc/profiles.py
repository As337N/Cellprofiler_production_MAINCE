"""
qc/profiles.py — Perfil de dataset construido desde el .env.

Lo que realmente varía entre tu dataset interno y JUMP (confirmado con el .env
de producción) es SOLO tres cosas:

  1. CHANNEL_DICT     — orden de canales y su índice numérico en el filename.
                        Los NOMBRES de canal son idénticos en ambos
                        (Hoechst/Syto/Golgi/ER/Mito/Brightfield); cambia el orden
                        y qué número le toca a cada uno.
  2. FILENAME_REGEX   — cómo extraer Row/Column/Field/Channel del nombre de
                        archivo. Interno: '001001-1-...'; JUMP: 'r01c01f01p01-ch3'.
  3. WELLS_PER_PLATE  — 96 (8×12) o 384 (16×24). Define geometría de placa y si
                        el platemap es de 96 o 96×4 pozos.

Thresholds, criterios de QC y nombres de canal son COMPARTIDOS — no se
parametrizan por dataset. El perfil se arma leyendo el mismo .env que ya usa tu
orquestador, así no hay una segunda fuente de config que mantener sincronizada.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field


# ── Constantes compartidas entre datasets (antes en config.py) ───────────────
# Nombres de canal canónicos y todo lo que NO depende del dataset.
THRESHOLDS = {"MedianIntensity": (None, 0.95)}
THRESHOLDS_LOCAL_FOCUS = {
    "Hoechst": (0.80, None), "Syto": (0.05, None), "Golgi": (0.03, None),
    "ER": (0.08, None), "Mito": (0.005, None), "Brightfield": (0.001, None),
}
ROBUST_METRICS = ("PowerLogLogSlope", "MedianIntensity")
ILLUM_METRICS  = ("PowerLogLogSlope", "MedianIntensity")

CHANNEL_LABELS = {"Hoechst": "Hoechst", "Syto": "Sy", "Golgi": "Go",
                  "ER": "ER", "Mito": "Mi", "Brightfield": "BF"}
CHANNEL_COLORS = {"Hoechst": (100, 160, 255), "Syto": (80, 220, 120),
                  "Golgi": (255, 180, 60), "ER": (180, 90, 255),
                  "Mito": (255, 80, 80), "Brightfield": (160, 160, 160)}
METRIC_LABELS  = {"PowerLogLogSlope": "Focus", "MedianIntensity": "MaxInt"}

COUNT_COLS = {"Raw": "Count_Raw_nuclei", "Filtered": "Count_Nuclei",
              "Cells": "Count_Cells", "Artifacts": "Count_Illum_artifacts_filtered"}
AREA_COL = "ImageQuality_TotalArea_Brightfield"
DEFAULT_IMAGE_AREA = 1_166_400   # 1080×1080 px fallback

# RGB accent colours for pass/fail/no-data used across the drawn collage.
COL_THRESHOLDS = {"COL_PASS": (75, 215, 95),
                  "COL_FAIL": (255, 65, 65),
                  "COL_NODATA": (110, 110, 120)}

# MFI: canal → (objeto_fuente, color). Compartido (nombres de canal iguales).
_MFI = {"Hoechst": ("nuclei", "#64A0FF"), "Syto": ("cells", "#50DC78"),
        "ER": ("cells", "#B45AFF"), "Golgi": ("cells", "#FFB43C"),
        "Mito": ("cells", "#FF5050")}

IMAGE_METRIC_TPL = "ImageQuality_{metric}_{channel}"
OBJECT_INTENSITY_TPL = "Intensity_MedianIntensity_{channel}"
# MFI por imagen: misma forma que la métrica MedianIntensity (config._img).
IMAGE_INTENSITY_TPL = "ImageQuality_MedianIntensity_{channel}"


# ── Geometría de placa según WELLS_PER_PLATE ─────────────────────────────────
_PLATE_GEOMETRY = {
    96:  (8, 12),
    384: (16, 24),
}


@dataclass(frozen=True)
class DatasetProfile:
    """Todo lo específico del dataset, derivado del .env."""
    cohort: str
    channel_dict: dict           # {channel_name: index_str}, del .env CHANNEL_DICT
    filename_regex: re.Pattern   # compilada, del .env FILENAME_REGEX
    wells_per_plate: int         # 96 o 384

    # ── Derivados de geometría ────────────────────────────────────────────────
    @property
    def plate_rows(self) -> int:
        return _PLATE_GEOMETRY[self.wells_per_plate][0]

    @property
    def plate_cols(self) -> int:
        return _PLATE_GEOMETRY[self.wells_per_plate][1]

    # ── Orden de canales (del CHANNEL_DICT, por índice) ──────────────────────
    @property
    def channels(self) -> tuple[str, ...]:
        """Canales ordenados por su índice numérico en el .env."""
        return tuple(sorted(self.channel_dict, key=lambda c: int(self.channel_dict[c])))

    # ── Nombres de columna (compartidos, dependen solo de channels) ──────────
    def image_col(self, metric: str, channel: str) -> str:
        return IMAGE_METRIC_TPL.format(metric=metric, channel=channel)

    def object_col(self, channel: str) -> str:
        return OBJECT_INTENSITY_TPL.format(channel=channel)

    def image_intensity_col(self, channel: str) -> str:
        """Columna de MFI por imagen (tabla Image). Ex-config._img."""
        return IMAGE_INTENSITY_TPL.format(channel=channel)

    # ── MFI: fuente de objeto y color por canal (del _MFI compartido) ────────
    def mfi_source(self, channel: str) -> str | None:
        """'cells' | 'nuclei' | None — de qué tabla sale la intensidad del canal."""
        entry = _MFI.get(channel)
        return entry[0] if entry else None

    def mfi_color(self, channel: str) -> str | None:
        entry = _MFI.get(channel)
        return entry[1] if entry else None

    def metric_cols(self, metrics=("PowerLogLogSlope", "MedianIntensity", "FocusScore")):
        return {m: [self.image_col(m, ch) for ch in self.channels] for m in metrics}

    def col_to_channel(self):
        return {col: ch for cols in self.metric_cols().values()
                for col in cols for ch in self.channels if ch in col}

    # ── Parseo de filename (reemplaza IMAGE_REGEX + parse_name hardcodeados) ──
    def parse_name(self, fname: str) -> tuple:
        """(row, column, field) desde el nombre de archivo, usando la regex del .env."""
        m = self.filename_regex.match(fname)
        if not m:
            raise ValueError(f"No se pudo parsear el nombre de imagen: {fname!r}")
        gd = m.groupdict()
        return int(gd["Row"]), int(gd["Column"]), int(gd["Field"])

    def channel_from_index(self, index: str) -> str | None:
        """Nombre de canal a partir del índice numérico del filename."""
        for ch, idx in self.channel_dict.items():
            if str(idx) == str(index):
                return ch
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Construcción desde el .env (punto de control externo, orquestador)
# ─────────────────────────────────────────────────────────────────────────────
def profile_from_env() -> DatasetProfile:
    """Arma el perfil desde las variables de entorno del .env.

    Requiere: CHANNEL_DICT (JSON), FILENAME_REGEX, WELLS_PER_PLATE, COHORT.
    Si usas python-dotenv, carga el .env antes (load_dotenv()).

    Diseñado para no romper si faltan comillas: FILENAME_REGEX en el .env va
    entre comillas simples que el shell suele conservar; se limpian aquí.
    """
    cohort = os.getenv("COHORT", "Cohort")

    raw_channels = os.getenv("CHANNEL_DICT")
    if not raw_channels:
        raise ValueError("Falta CHANNEL_DICT en el entorno (.env).")
    channel_dict = json.loads(raw_channels.strip().strip("'\""))

    raw_regex = os.getenv("FILENAME_REGEX")
    if not raw_regex:
        raise ValueError("Falta FILENAME_REGEX en el entorno (.env).")
    filename_regex = re.compile(raw_regex.strip().strip("'\""))

    wells = int(os.getenv("WELLS_PER_PLATE", "96"))
    if wells not in _PLATE_GEOMETRY:
        raise ValueError(f"WELLS_PER_PLATE={wells} no soportado. "
                         f"Opciones: {sorted(_PLATE_GEOMETRY)}")

    return DatasetProfile(cohort=cohort, channel_dict=channel_dict,
                          filename_regex=filename_regex, wells_per_plate=wells)


def profile_from_values(*, cohort: str, channel_dict: dict, filename_regex: str,
                        wells_per_plate: int) -> DatasetProfile:
    """Construcción explícita (para tests o llamada programática sin .env)."""
    return DatasetProfile(cohort=cohort, channel_dict=channel_dict,
                          filename_regex=re.compile(filename_regex),
                          wells_per_plate=wells_per_plate)

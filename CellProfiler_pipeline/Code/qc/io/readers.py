"""
qc/io/readers.py — TableStore: lectura única de las tablas de medidas.

Problema que resuelve (el I/O que pediste optimizar)
----------------------------------------------------
En loaders.py, Cells.parquet/Nuclei.parquet se leen:
  · una vez en load_mfi_data (MFI),
  · otra vez en load_mfi_img (Image),
  · y OTRA VEZ por CADA placa dentro de load_radius_data (loop en _render_plate).
Con N placas, Cells se lee N+1 veces.

TableStore lee cada tabla UNA sola vez como LazyFrame (scan_parquet → pushdown de
columnas/filas), cachea, normaliza Well/Plate una vez, y deriva MFI / radius /
img / qc-sites de esa fuente. El radius se computa para TODAS las placas de una
pasada (group_by [plate, well]) y se indexa por placa — se elimina el factor ×N.

Los parquet viven en UN solo directorio por cohorte (Cells/Image/Nuclei);
TableStore filtra por Metadata_Plate, no itera subdirectorios.

Inerte hasta Fase 4 (decisión B): nadie lo importa aún en el pipeline viejo.
Se verifica contra loaders.py con datos sintéticos del schema real.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from qc import profiles
from qc.profiles import DatasetProfile

TABLES = ("Image", "Cells", "Nuclei")
_RADIUS_COL = "AreaShape_MedianRadius"


class TableStore:
    """Localiza, lee una vez y cachea los LazyFrames de las tablas de medidas."""

    def __init__(self, measurements_dir: Path, profile: DatasetProfile):
        self.dir = Path(measurements_dir)
        self.profile = profile
        self._lf_cache: dict[str, pl.LazyFrame | None] = {}

    # ── Localización (ex-Collage._table) ─────────────────────────────────────
    def path(self, table: str) -> Path | None:
        """Ruta de una tabla: prefiere .parquet, cae a .txt."""
        for ext in (".parquet", ".txt"):
            p = self.dir / f"{table}{ext}"
            if p.exists():
                return p
        return None

    # ── Lectura única + normalización ────────────────────────────────────────
    def scan(self, table: str) -> pl.LazyFrame | None:
        """LazyFrame cacheado de la tabla, con Well/Plate ya normalizados.

        scan_parquet difiere la lectura hasta el collect(), permitiendo que
        Polars solo lea del disco las columnas/filas que cada consumidor use.
        Para .txt cae a scan_csv (tab-separated).
        """
        if table in self._lf_cache:
            return self._lf_cache[table]

        p = self.path(table)
        if p is None:
            self._lf_cache[table] = None
            return None

        lf = (pl.scan_parquet(p) if p.suffix == ".parquet"
              else pl.scan_csv(p, separator="\t", infer_schema_length=10_000))

        # Normalización de Well (upper, sin espacios) y Plate (str, sin espacios).
        # Se hace en el LazyFrame, así se aplica una vez y se comparte.
        schema_names = lf.collect_schema().names()
        if "Metadata_Well" in schema_names:
            lf = lf.with_columns(
                pl.col("Metadata_Well").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
            )
        if "Metadata_Plate" in schema_names:
            lf = lf.with_columns(
                pl.col("Metadata_Plate").cast(pl.Utf8).str.strip_chars()
            )
        else:
            lf = lf.with_columns(pl.lit("Plate").alias("Metadata_Plate"))

        self._lf_cache[table] = lf
        return lf

    def columns(self, table: str) -> list[str]:
        lf = self.scan(table)
        return [] if lf is None else lf.collect_schema().names()

    # ── MFI (ex-load_mfi_data) ───────────────────────────────────────────────
    def mfi_sources(self) -> tuple[dict, dict]:
        """(source_dfs, channel_map) desde Cells/Nuclei, leídos una vez.

        Réplica de load_mfi_data pero con las columnas MFI derivadas del perfil
        (no de config.MFI_DATA). channel_map ordenado por profile.channels.
        source_dfs[channel] = DataFrame[Metadata_Plate, Metadata_Well, (ImageNumber?), col]
        """
        source_dfs: dict = {}
        channel_map: dict = {}

        for table, source in (("Cells", "cells"), ("Nuclei", "nuclei")):
            lf = self.scan(table)
            if lf is None:
                continue
            cols = lf.collect_schema().names()
            if "Metadata_Well" not in cols:
                continue
            has_img = "ImageNumber" in cols
            for ch in self.profile.channels:
                if self.profile.mfi_source(ch) != source:
                    continue
                col = self.profile.object_col(ch)
                if col not in cols:
                    continue
                keep = (["Metadata_Plate", "Metadata_Well"]
                        + (["ImageNumber"] if has_img else []) + [col])
                source_dfs[ch] = lf.select(keep).collect()
                channel_map[ch] = (col, self.profile.mfi_color(ch))

        ordered = {ch: channel_map[ch] for ch in self.profile.channels
                   if ch in channel_map}
        if not ordered:
            print("[mfi] No usable MFI data found.")
        return source_dfs, ordered

    # ── MFI por imagen (ex-load_mfi_img) ─────────────────────────────────────
    def mfi_img(self) -> dict:
        """{well: {channel: median_across_sites}} desde la tabla Image."""
        lf = self.scan("Image")
        if lf is None:
            return {}
        cols = lf.collect_schema().names()
        if "Metadata_Well" not in cols:
            return {}
        img_cols = {ch: self.profile.image_intensity_col(ch)
                    for ch in self.profile.channels
                    if self.profile.image_intensity_col(ch) in cols}
        if not img_cols:
            return {}
        agg = (lf.group_by("Metadata_Well")
                 .agg([pl.col(col).median().alias(ch) for ch, col in img_cols.items()])
                 .collect())
        out: dict = {}
        for row in agg.iter_rows(named=True):
            out[row["Metadata_Well"]] = {ch: float(row[ch]) for ch in img_cols
                                         if row[ch] is not None}
        return out

    def mfi_img_sites(self) -> dict:
        """Per-site image MFI: {well: {ImageNumber: {channel: median}}}.

        Like mfi_img but grouped by [Well, ImageNumber] so each site keeps its own
        whole-image median, keyed by the real ImageNumber. Used to align
        object-vs-image MFI per site in the CSV export. Does not replace mfi_img,
        which stays a single per-well median for the SNR/Δ views.
        """
        lf = self.scan("Image")
        if lf is None:
            return {}
        cols = lf.collect_schema().names()
        if "Metadata_Well" not in cols or "ImageNumber" not in cols:
            return {}
        img_cols = {ch: self.profile.image_intensity_col(ch)
                    for ch in self.profile.channels
                    if self.profile.image_intensity_col(ch) in cols}
        if not img_cols:
            return {}
        agg = (lf.group_by(["Metadata_Well", "ImageNumber"])
                 .agg([pl.col(col).median().alias(ch) for ch, col in img_cols.items()])
                 .collect())
        out: dict = {}
        for row in agg.iter_rows(named=True):
            well = str(row["Metadata_Well"]).strip().upper()
            img = int(row["ImageNumber"])
            out.setdefault(well, {})[img] = {
                ch: float(row[ch]) for ch in img_cols if row[ch] is not None}
        return out

    # ── Radius para TODAS las placas de una vez (ex-load_radius_data ×N) ──────
    def radius_all_plates(self) -> dict:
        """{plate_norm: {'Cells': {well: r}, 'Nuclei': {well: r}}}.

        Antes: load_radius_data releía Cells/Nuclei una vez por placa. Aquí se
        computa la mediana por [plate, well] en UNA pasada por tabla y se indexa
        por placa normalizada (schema.normalize_plate). El consumidor pide
        result[norm(plate)] sin releer nada.
        """
        from qc import schema
        result: dict = {}
        for label in ("Cells", "Nuclei"):
            lf = self.scan(label)
            if lf is None:
                continue
            cols = lf.collect_schema().names()
            if "Metadata_Well" not in cols or _RADIUS_COL not in cols:
                print(f"[radius] {_RADIUS_COL} not found in {label} — skipping.")
                continue
            agg = (lf.group_by(["Metadata_Plate", "Metadata_Well"])
                     .agg(pl.col(_RADIUS_COL).median().alias("_med"))
                     .drop_nulls("_med")
                     .collect())
            for row in agg.iter_rows(named=True):
                pkey = schema.normalize_plate(row["Metadata_Plate"])
                (result.setdefault(pkey, {}).setdefault(label, {})
                       [row["Metadata_Well"]]) = float(row["_med"])
        return result

    # ── Site-level QC from the Image table (ex-load_qc_tsv_sites) ─────────────
    def qc_sites(self) -> dict:
        """Per-site QC values grouped by [plate, well, field].

        Returns {plate: {well: {field: {col: value}}}}. Single source of truth:
        the well level is derived on demand
        (stats.aggregate.collapse_sites_to_wells) instead of re-reading.
        Aggregation policy matches loaders.load_qc_tsv_sites: metrics/percentages
        → mean, counts/area → sum.
        """
        lf = self.scan("Image")
        if lf is None:
            return {}
        cols = lf.collect_schema().names()
        if "Metadata_Well" not in cols or "Metadata_Field" not in cols:
            print("[warn] Metadata_Well/Field not found — site-level QC disabled.")
            return {}

        metric_cols = list(dict.fromkeys(
            c for cs in self.profile.metric_cols().values() for c in cs if c in cols))
        # PercentMinimal se reemplaza por completo por Intensity_Percentile_10
        # (piso de intensidad de la imagen por canal, tabla Image): detecta campos
        # dominados por fondo negro, que PercentMinimal no distinguía. Ambos
        # (PercentMaximal, Percentile_10) se colapsan por peor-sitio.
        pct_cols = list(dict.fromkeys(
            c for c in cols
            if c.startswith("ImageQuality_PercentMaximal_") or
               c.startswith("Intensity_Percentile_10_")))
        sum_cols = list(dict.fromkeys(
            c for c in cols if c.startswith("Count_") or c == profiles.AREA_COL))

        agg_exprs = (
            [pl.col(c).mean() for c in metric_cols] +
            [pl.col(c).mean() for c in pct_cols] +
            [pl.col(c).sum() for c in sum_cols
             if c not in metric_cols and c not in pct_cols])
        if not agg_exprs:
            print("[warn] No QC columns matched — nothing to aggregate.")
            return {}

        all_cols = (metric_cols + pct_cols +
                    [c for c in sum_cols if c not in metric_cols and c not in pct_cols])
        agg = (lf.group_by(["Metadata_Plate", "Metadata_Well", "Metadata_Field"])
                 .agg(agg_exprs).collect())

        result: dict = {}
        for row in agg.iter_rows(named=True):
            plate = str(row["Metadata_Plate"]).strip()
            well = str(row["Metadata_Well"]).strip().upper()
            field = int(row["Metadata_Field"])
            (result.setdefault(plate, {}).setdefault(well, {})
                   [field]) = {c: row[c] for c in all_cols if c in row}
        return result
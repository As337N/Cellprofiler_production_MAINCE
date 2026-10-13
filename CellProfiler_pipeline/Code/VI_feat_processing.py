"""
Perfilado Cell Painting siguiendo el flujo de Broad (profiling handbook / recipe):

    células -> [QC de outliers opcional] -> agregación POR POCILLO -> anotación (platemap)
            -> normalización POR PLACA -> feature selection a nivel de batch

Todo en polars salvo las llamadas a pycytominer, que requieren pandas.
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import polars as pl
import polars.selectors as cs
from pycytominer import aggregate, feature_select, normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Prefijos estándar de pycytominer / CellProfiler
OBJECT_PREFIXES = {"Cells": "Cells", "Cytoplasm": "Cytoplasm", "Nuclei": "Nuclei"}
PLATE_WELL = ["Metadata_Plate", "Metadata_Well"]
PERTURBATION = "Metadata_Perturbation"


class OutlierHandler:
    """(Opcional) Detecta y corrige outliers por célula con score MAD y winsorización."""

    def __init__(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        well_col: str = "Metadata_Well",
        mad_thresh: float = 5.0,
        winsor_percentile: float = 0.95,
    ):
        if not (0 < winsor_percentile < 1):
            raise ValueError(
                f"winsor_percentile debe estar entre 0 y 1, recibido: {winsor_percentile}"
            )
        self.df = df
        self.feature_cols = feature_cols
        self.well_col = well_col
        self.mad_thresh = mad_thresh
        self.winsor_percentile = winsor_percentile
        self.mad_score: pl.DataFrame | None = None
        self.outlier_mask: pl.DataFrame | None = None

    def _check_outliers_computed(self) -> None:
        if self.outlier_mask is None:
            raise RuntimeError("Run detect_outliers() first")

    def detect_outliers(self) -> pl.DataFrame:
        feats = pl.col(self.feature_cols)
        centered = feats - feats.median()
        mad = centered.abs().median()
        # MAD == 0 (features dispersas) -> sin score, no se marca como outlier
        score = pl.when(mad > 0).then(centered / mad).otherwise(None)

        self.mad_score = self.df.select(score)
        self.outlier_mask = self.mad_score.select(
            pl.all().abs().gt(self.mad_thresh).fill_null(False)
        )
        return self.outlier_mask

    def report(self, top_n: int = 20) -> None:
        self._check_outliers_computed()
        per_cell = pl.concat(
            [
                self.df.select(self.well_col),
                self.outlier_mask.select(
                    pl.sum_horizontal(pl.all().cast(pl.UInt32)).alias("n_outlier_features")
                ),
                self.mad_score.select(pl.max_horizontal(pl.all().abs()).alias("max_mad")),
            ],
            how="horizontal",
        ).filter(pl.col("n_outlier_features") > 0)

        report_well = (
            per_cell.group_by(self.well_col)
            .agg(
                n_cells=pl.len(),
                mean_outlier_features=pl.col("n_outlier_features").mean(),
                max_mad=pl.col("max_mad").max(),
            )
            .sort("max_mad", descending=True)
            .head(top_n)
        )
        logger.info("Top outlier wells:\n%s", report_well)

    def winsorize(self) -> pl.DataFrame:
        self._check_outliers_computed()
        feats = pl.col(self.feature_cols)
        upper = feats.quantile(self.winsor_percentile, interpolation="linear")
        lower = feats.quantile(1 - self.winsor_percentile, interpolation="linear")
        self.df = self.df.with_columns(feats.clip(lower, upper))
        return self.df


class PyCytoPipe:
    def __init__(
        self,
        input_p: List[Path],
        output_p: Path,
        cohort: str,
        metadata_cols: List[str],
        platemap_p: Path,
        agg_operation: str = "median",
        norm_samples: str = f'{PERTURBATION} == "DMSO"',
        corr_method: str = "pearson",
        corr_threshold: float = 0.9,
        blocklist_p: Path | None = None,
        outlier_qc: bool = False,
        mad_thresh: float = 5.0,
        winsor_percentile: float = 0.95,
    ):
        self.input_p = input_p
        self.output_p = output_p
        self.cohort = cohort
        self.metadata_cols = metadata_cols
        self.platemap_p = platemap_p
        self.agg_operation = agg_operation
        self.norm_samples = norm_samples
        self.corr_method = corr_method
        self.corr_threshold = corr_threshold
        self.blocklist_p = blocklist_p
        self.outlier_qc = outlier_qc
        self.mad_thresh = mad_thresh
        self.winsor_percentile = winsor_percentile

        self.out_agg = output_p / f"{cohort}_aggregated.csv"
        self.out_norm = output_p / f"{cohort}_normalized.csv"
        self.out_red = output_p / f"{cohort}_normalized_feature_select.csv"

    # ------------------------------------------------------------------ lectura
    def _read_object_tables(self, files: dict[str, Path]) -> pl.DataFrame:
        """Lee Cells/Cytoplasm/Nuclei de UNA carpeta y los une por metadata."""
        meta = set(self.metadata_cols)
        tables = {
            name: pl.read_csv(path, separator="\t", glob=False, infer_schema_length=10_000)
            for name, path in files.items()
        }
        for name, df in tables.items():
            absent = meta - set(df.columns)
            if absent:
                raise ValueError(f"{name}: faltan columnas de metadata {sorted(absent)}")

        # Prefijo (Cells_/Cytoplasm_/Nuclei_) en toda columna presente en >1 tabla
        counts = Counter(c for df in tables.values() for c in df.columns if c not in meta)
        shared = {c for c, n in counts.items() if n > 1}
        tables = {
            name: df.rename(
                {c: f"{OBJECT_PREFIXES[name]}_{c}" for c in shared & set(df.columns)}
            )
            for name, df in tables.items()
        }
        return (
            tables["Cells"]
            .join(tables["Cytoplasm"], on=self.metadata_cols, how="inner")
            .join(tables["Nuclei"], on=self.metadata_cols, how="inner")
        )

    def _prepare_cohort_df(self) -> pl.DataFrame:
        by_dir: dict[Path, dict[str, Path]] = defaultdict(dict)
        for f in self.input_p:
            if f.stem in OBJECT_PREFIXES:
                by_dir[f.parent][f.stem] = f
        if not by_dir:
            raise FileNotFoundError(
                f"No se encontró ningún archivo de {sorted(OBJECT_PREFIXES)}"
            )

        frames = []
        for folder, files in by_dir.items():
            missing = sorted(OBJECT_PREFIXES.keys() - files.keys())
            if missing:
                raise FileNotFoundError(f"Faltan archivos en {folder}: {missing}")
            logger.info("Leyendo %s", folder)
            frames.append(self._read_object_tables(files))

        df = pl.concat(frames, how="vertical_relaxed")
        # polars distingue NaN de null; unificamos a null
        df = df.with_columns(cs.float().fill_nan(None))
        # Llaves de plate/well siempre como string (polars no une dtypes distintos)
        return df.with_columns(pl.col(PLATE_WELL).cast(pl.Utf8))

    def _load_platemap(self) -> pl.DataFrame:
        platemap = pl.read_csv(self.platemap_p)
        required = set(PLATE_WELL) | {PERTURBATION}
        missing = required - set(platemap.columns)
        if missing:
            raise ValueError(f"Platemap falta columnas: {sorted(missing)}")
        dups = platemap.group_by(PLATE_WELL).len().filter(pl.col("len") > 1)
        if dups.height:
            raise ValueError(
                f"Platemap tiene {dups.height} combinaciones (Plate, Well) duplicadas"
            )
        return platemap.with_columns(pl.col(PLATE_WELL).cast(pl.Utf8))

    # ----------------------------------------------------------- pasos (pandas
    # solo dentro de estos tres métodos, porque pycytominer lo requiere)
    def _aggregate_wells(self, cells: pl.DataFrame, feat_cols: List[str]) -> pl.DataFrame:
        pdf = cells.select(PLATE_WELL + feat_cols).to_pandas()
        agg = aggregate(
            population_df=pdf,
            strata=PLATE_WELL,
            features=feat_cols,
            operation=self.agg_operation,
            output_type="pandas",
        )
        return pl.from_pandas(agg)

    def _normalize_by_plate(self, profiles: pl.DataFrame, feat_cols: List[str]) -> pl.DataFrame:
        frames = []
        for plate_df in profiles.partition_by("Metadata_Plate", maintain_order=True):
            plate = plate_df["Metadata_Plate"][0]
            pdf = plate_df.to_pandas()
            n_ref = len(pdf.query(self.norm_samples))
            if n_ref == 0:
                raise ValueError(
                    f"Placa {plate}: ningún pocillo cumple la referencia de normalización "
                    f"({self.norm_samples})"
                )
            logger.info("Normalizando placa %s (%d pocillos de referencia)", plate, n_ref)
            norm = normalize(
                profiles=pdf,
                features=feat_cols,
                samples=self.norm_samples,
                method="mad_robustize",
                output_type="pandas",
            )
            frames.append(pl.from_pandas(norm))
        return pl.concat(frames, how="vertical_relaxed")

    def _select_features(self, profiles: pl.DataFrame, feat_cols: List[str]) -> pl.DataFrame:
        operations = ["drop_na_columns", "variance_threshold", "correlation_threshold"]
        kwargs = {}
        if self.blocklist_p is not None:
            operations.append("blocklist")
            kwargs["blocklist_file"] = str(self.blocklist_p)

        red = feature_select(
            profiles=profiles.to_pandas(),
            features=feat_cols,
            operation=operations,
            corr_threshold=self.corr_threshold,
            corr_method=self.corr_method,
            output_type="pandas",
            **kwargs,
        )
        return pl.from_pandas(red)

    # ---------------------------------------------------------------- pipeline
    def run_analysis(self) -> None:
        logger.info("Processing cohort: %s", self.cohort)
        self.output_p.mkdir(parents=True, exist_ok=True)

        cells = self._prepare_cohort_df()
        meta_present = [c for c in self.metadata_cols if c in cells.columns]
        feat_cols = [c for c in cells.select(cs.numeric()).columns if c not in meta_present]

        # --- Paso adicional (no forma parte del flujo de Broad) ---
        if self.outlier_qc:
            logger.info("QC de outliers activado")
            qc = OutlierHandler(
                df=cells,
                feature_cols=feat_cols,
                mad_thresh=self.mad_thresh,
                winsor_percentile=self.winsor_percentile,
            )
            qc.detect_outliers()
            qc.report()
            cells = qc.winsorize()
            del qc  # libera mad_score y outlier_mask

        # 1) Agregación por pocillo
        cell_counts = cells.group_by(PLATE_WELL).agg(pl.len().alias("Metadata_Cell_Count"))
        agg_df = self._aggregate_wells(cells, feat_cols)
        del cells

        # 2) Anotación con el platemap
        annotated = agg_df.join(cell_counts, on=PLATE_WELL, how="left").join(
            self._load_platemap(), on=PLATE_WELL, how="left"
        )
        n_unmatched = annotated[PERTURBATION].null_count()
        if n_unmatched:
            logger.warning(
                "%d pocillos sin match en el platemap (se excluyen del resto del análisis)",
                n_unmatched,
            )
            annotated = annotated.filter(pl.col(PERTURBATION).is_not_null())

        feat_agg = [c for c in agg_df.columns if not c.startswith("Metadata_")]

        # 3) Normalización por placa
        norm_df = self._normalize_by_plate(annotated, feat_agg)

        # 4) Feature selection a nivel de batch
        red_df = self._select_features(norm_df, feat_agg)

        annotated.write_csv(self.out_agg)
        norm_df.write_csv(self.out_norm)
        red_df.write_csv(self.out_red)
        logger.info("Finished cohort %s", self.cohort)


def parse_args():
    p = argparse.ArgumentParser("PyCytominer feature processing (flujo Broad)")
    p.add_argument("-i", "--input", required=True, help="Input root directory")
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("-c", "--cohort", required=True, help="Analyzed cohort")
    p.add_argument("-m", "--platemap", required=True, help="Path to platemap CSV")
    p.add_argument(
        "--metadata-cols",
        nargs="+",
        default=["ImageNumber", "ObjectNumber", "Metadata_Field",
                 "Metadata_Plate", "Metadata_Well"],
        help="Metadata column names",
    )
    p.add_argument("--agg-operation", choices=["median", "mean"], default="mean")
    p.add_argument("--norm-samples", default=f'{PERTURBATION} == "DMSO"',
                   help="Query pandas que define los pocillos de referencia por placa")
    p.add_argument("--corr-method", choices=["pearson", "spearman", "kendall"],
                   default="pearson")
    p.add_argument("--corr-threshold", type=float, default=0.9)
    p.add_argument("--blocklist", type=Path, default=None,
                   help="Archivo de blocklist de features (opcional)")
    p.add_argument("--outlier-qc", action="store_true",
                   help="Paso adicional: MAD + winsorización a nivel de célula")
    p.add_argument("--mad-thresh", type=float, default=5.0)
    p.add_argument("--winsor-percentile", type=float, default=0.95)
    p.add_argument("-w", "--workers", type=int, default=None, help="(sin uso por ahora)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_files = sorted(Path(args.input).rglob("*.txt"))

    PyCytoPipe(
        input_p=input_files,
        output_p=Path(args.output),
        cohort=args.cohort,
        metadata_cols=args.metadata_cols,
        platemap_p=Path(args.platemap),
        agg_operation=args.agg_operation,
        norm_samples=args.norm_samples,
        corr_method=args.corr_method,
        corr_threshold=args.corr_threshold,
        blocklist_p=args.blocklist,
        outlier_qc=args.outlier_qc,
        mad_thresh=args.mad_thresh,
        winsor_percentile=args.winsor_percentile,
    ).run_analysis()
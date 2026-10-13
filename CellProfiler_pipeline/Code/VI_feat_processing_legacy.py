import logging
import re
from pathlib import Path
from typing import List, Optional
import pandas as pd
import polars as pl
from pycytominer import feature_select, normalize, aggregate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

def get_well(row: int, col: int) -> str:
    return f"{chr(64 + int(row))}{int(col):02d}"

class OutlierHandler:
    """Detect and correct outliers using MAD-based scoring and winsorization."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        well_col: str = "Metadata_Well",
        mad_thresh: float = 5.0,
        winsor_percentile: float = 0.95,
    ):
        if not (0 < winsor_percentile < 1):
            raise ValueError(
                f"winsor_percentile debe estar entre 0 y 1, recibido: {winsor_percentile}"
            )
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.well_col = well_col
        self.mad_thresh = mad_thresh
        self.winsor_percentile = winsor_percentile
        self.mad_score: pd.DataFrame | None = None
        self.outlier_mask: pd.DataFrame | None = None

    def _check_outliers_computed(self) -> None:
        if self.outlier_mask is None:
            raise RuntimeError("Run detect_outliers() first")

    def detect_outliers(self) -> pd.DataFrame:
        median = self.df[self.feature_cols].median()
        mad = (self.df[self.feature_cols] - median).abs().median()
        self.mad_score = (self.df[self.feature_cols] - median) / (mad + 1e-8)
        self.outlier_mask = self.mad_score.abs() > self.mad_thresh
        return self.outlier_mask

    def report(self, top_n: int = 20) -> None:
        self._check_outliers_computed()
        rep = self.df.loc[self.outlier_mask.any(axis=1), [self.well_col]].copy()
        rep["n_outlier_features"] = self.outlier_mask.loc[rep.index].sum(axis=1)
        rep["max_mad"] = self.mad_score.loc[rep.index].abs().max(axis=1)
        report_well = (
            rep.groupby(self.well_col)
            .agg(
                n_cells=("n_outlier_features", "count"),
                mean_outlier_features=("n_outlier_features", "mean"),
                max_mad=("max_mad", "max"),
            )
            .sort_values("max_mad", ascending=False)
            .head(top_n)
        )
        logger.info("Top outlier wells:\n%s", report_well)

    def winsorize(self) -> pd.DataFrame:
        self._check_outliers_computed()
        upper = self.df[self.feature_cols].quantile(self.winsor_percentile)
        lower = self.df[self.feature_cols].quantile(1 - self.winsor_percentile)
        self.df[self.feature_cols] = self.df[self.feature_cols].clip(
            lower=lower, upper=upper, axis=1
        )
        return self.df

class PyCytoPipe:
    def __init__(self, input_p: List[Path], output_p: Path, cohort: str, 
                 metadata_cols: List[str], platemap_p: Path,):
        self.input_p = input_p
        self.output_p = output_p
        self.cohort = cohort
        self.metadata_cols = metadata_cols
        self.platemap_p = platemap_p

        self.out_agg = output_p / f"{cohort}_aggregated.csv"
        self.out_norm = output_p / f"{cohort}_normalized.csv"
        self.out_red = output_p / f"{cohort}_reduced.csv"

    def _prepare_cohort_df(self) -> pl.DataFrame:
        relevant_csv = ["Cells", "Cytoplasm", "Nuclei"]

        dfs = {
            i.stem: pl.read_csv(i, separator="\t", glob=False)
            for i in self.input_p
            if i.stem in relevant_csv
        }

        missing = [k for k in relevant_csv if k not in dfs]
        if missing:
            raise FileNotFoundError(f"Faltan archivos: {missing}")

        common_cols = (
            set(dfs["Cells"].columns)
            & set(dfs["Cytoplasm"].columns)
            & set(dfs["Nuclei"].columns)
        ) - set(self.metadata_cols)

        prefixes = {"Cells": "cells", "Cytoplasm": "cyto", "Nuclei": "nuclei"}
        dfs = {
            name: df.rename({c: f"{prefixes[name]}_{c}" for c in common_cols})
            for name, df in dfs.items()
        }

        return (
            dfs["Cells"]
            .join(dfs["Cytoplasm"], on=self.metadata_cols, how="inner")
            .join(dfs["Nuclei"], on=self.metadata_cols, how="inner")
        )

    def _load_platemap(self) -> pd.DataFrame:
        """Carga el platemap y valida columnas mínimas requeridas."""
        platemap = pd.read_csv(self.platemap_p)
        required = {"Metadata_Plate", "Metadata_Perturbation"}
        missing = required - set(platemap.columns)
        if missing:
            raise ValueError(f"Platemap falta columnas: {missing}")
        return platemap

    def run_analysis(self):
        meta_cols = ["Metadata_Well", "Metadata_Plate"]
        logger.info(f"Processing cohort: {self.cohort}")
        self.output_p.mkdir(parents=True, exist_ok=True)

        df = self._prepare_cohort_df().to_pandas()

        meta_cols_present = [c for c in self.metadata_cols if c in df.columns]
        feat_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in meta_cols_present
        ]

        qc = OutlierHandler(df=df, feature_cols=feat_cols, well_col="Metadata_Well")
        qc.detect_outliers()
        qc.report()
        df_corr = qc.winsorize()

        platemap = self._load_platemap()
        for col in meta_cols:
            if col in df_corr.columns and col in platemap.columns:
                if df_corr[col].dtype != platemap[col].dtype:
                    logger.warning(f"Tipo incompatible en '{col}': df_corr={df_corr[col].dtype}, platemap={platemap[col].dtype}. Convirtiendo a string.")
                    df_corr[col] = df_corr[col].astype(str)
                    platemap[col] = platemap[col].astype(str)

        df_corr = df_corr.merge(platemap, on=meta_cols, how="left")

        agg_df = aggregate(
            population_df=df_corr,
            strata=["Metadata_Plate", "Metadata_Perturbation"],
            features=feat_cols,
            operation="median",
            output_type="pandas",)

        feat_agg = [c for c in agg_df.columns if c not in ["Metadata_Plate", "Metadata_Perturbation"]]

        norm_df = normalize(
            profiles=agg_df,
            features=feat_agg,
            samples='Metadata_Perturbation == "DMSO"',
            method="mad_robustize",
            output_type="pandas",)

        red_df = feature_select(
            profiles=norm_df,
            features=feat_agg,
            output_type="pandas",
            operation=["correlation_threshold", "variance_threshold"],
            corr_threshold=0.9,
            corr_method="spearman",)

        agg_df.to_csv(self.out_agg, index=False)
        norm_df.to_csv(self.out_norm, index=False)
        red_df.to_csv(self.out_red, index=False)

        logger.info(f"Finished cohort {self.cohort}")

def parse_args():
    import argparse
    p = argparse.ArgumentParser("PyCytoMiner feature processing")
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
    p.add_argument("-w", "--workers", type=int, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    input_files = list(Path(args.input).rglob("*.txt"))

    pipe = PyCytoPipe(
        input_p=input_files,
        output_p=Path(args.output),
        cohort=args.cohort,
        metadata_cols=args.metadata_cols,
        platemap_p=Path(args.platemap),)
    pipe.run_analysis()
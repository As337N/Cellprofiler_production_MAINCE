from pycytominer import annotate, feature_select, normalize, aggregate
import pandas as pd
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import re
import os
import numpy as np

class OutlierHandler:
    def __init__(
        self,
        df,
        feature_cols,
        compound_col="Metadata_Compound",
        well_col="Metadata_Well",
        mad_thresh=5.0,
        p95=0.95,):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.compound_col = compound_col
        self.well_col = well_col
        self.mad_thresh = mad_thresh
        self.p95 = p95

        self.mad_score = None
        self.outlier_mask = None

    def detect_outliers(self):
        median = self.df[self.feature_cols].median()
        mad = (self.df[self.feature_cols] - median).abs().median()

        self.mad_score = (self.df[self.feature_cols] - median) / (mad + 1e-8)
        self.outlier_mask = self.mad_score.abs() > self.mad_thresh

        return self.outlier_mask

    def report(self, top_n=20):
        if self.outlier_mask is None:
            raise RuntimeError("Run detect_outliers() first")

        rep = self.df.loc[self.outlier_mask.any(axis=1), [
            self.well_col,
            self.compound_col
        ]].copy()

        rep["n_outlier_features"] = self.outlier_mask.sum(axis=1)
        rep["max_mad"] = self.mad_score.abs().max(axis=1)

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

        report_compound = (
            rep.groupby(self.compound_col)
            .agg(
                n_cells=("n_outlier_features", "count"),
                mean_outlier_features=("n_outlier_features", "mean"),
                max_mad=("max_mad", "max"),
            )
            .sort_values("max_mad", ascending=False)
            .head(top_n)
        )

        print("\n=== OUTLIERS POR WELL ===")
        print(report_well)

        print("\n=== OUTLIERS POR COMPOUND ===")
        print(report_compound)

        return report_well, report_compound

    def winsorize(self):
        def _cap(group):
            p = group[self.feature_cols].quantile(self.p95)
            vals = group[self.feature_cols].clip(upper=p, axis=1)
            group[self.feature_cols] = vals.astype("float32")
            return group

        self.df = (
            self.df
            .groupby(self.compound_col, group_keys=False)
            .apply(_cap)
        )

        return self.df

def get_well(row, col):
  row_corr = chr(64 + int(row))
  return f"{row_corr}{col}"

try:
    print(version("pycytominer"))
except PackageNotFoundError:
    print("pycytominer no está instalado como paquete.")


Cohort = "C1"
save_path = Path("/workspace/Output")
regex_plate = re.compile(r"_p(?P<Plate>\d)_")
metadata_cols = ["Row", "Column", "Timepoint", "Field", "Object No", "X", "Y", "Bounding Box", "Position X [µm]", "Position Y [µm]", "Compound", "Concentration", "Cell Type", "Cell Count", "Nuclei Selected - Object No in Nuclei"]

for p in Path("/workspace_data").iterdir():
    #print(p / "platemap.csv <<<<")
    metadata_df = pd.read_csv(p / "platemap.csv")
    mp_tsv = p / "Evaluation1" / "Objects_Population - Nuclei Selected.txt"
    plate_re = regex_plate.search(str(p))
    meta_plate = plate_re.groupdict()
    plate = meta_plate["Plate"]
    out_path = save_path / f"plate_{plate}"

    os.makedirs(out_path, exist_ok=True)

    out_agg = os.path.join(out_path, f"{Cohort}_{plate}_pycyt_agg.csv")
    out_norm = os.path.join(out_path, f"{Cohort}_{plate}_pycyt_norm.csv")
    out_red = os.path.join(out_path, f"{Cohort}_{plate}_pycyt_red.csv")

    df = pd.read_csv(mp_tsv, sep="\t", skiprows=9)
    df["Metadata_Well"] = (
    (df["Row"] + 64).astype(int).apply(chr) +
    df["Column"].astype(int).astype(str).str.zfill(2))

    enriched_df = metadata_df.merge(df,  on="Metadata_Well", how="left")
    
    enriched_df = enriched_df.drop(columns=metadata_cols)
    enriched_df = enriched_df.dropna(axis="columns", how="all")
    enriched_df = enriched_df.copy()

    print(list(enriched_df.columns)[:15])
    print(enriched_df)

    feat_cols = enriched_df.select_dtypes(include="number").columns.tolist()

    qc = OutlierHandler(
        df=enriched_df,
        feature_cols=feat_cols,
        compound_col="Metadata_Compound",
        well_col="Metadata_Well",
        mad_thresh=5.0,
        p95=0.95,
    )

    qc.detect_outliers()
    qc.report()

    enriched_df_corr = qc.winsorize()

    # 1) Single cell aggregation
    agg_df = aggregate(
        population_df=enriched_df_corr,
        strata=["Metadata_Compound"],
        features=feat_cols,
        operation="median",
        output_type="pandas",          # ← queremos un DataFrame
    )

    print(f"df agregado: ")
    print(agg_df)

    # Metadatos que realmente quedaron después de agregar
    feat_agg = enriched_df.select_dtypes(include="number").columns.tolist()
    meta_agg = [c for c in agg_df.columns if c not in feat_agg]

    # 2) Normalize
    norm_df = normalize(
        profiles=agg_df,
        features=feat_agg,
        samples='Metadata_Compound == "DMSO"',
        output_type="pandas",
        method="mad_robustize",
    )

    print(f"df normalizado: ")
    print(norm_df)

    # 3) Feature reduction
    red_df = feature_select(
        profiles=norm_df,
        features=feat_agg,
        output_type="pandas",
        operation=["correlation_threshold", "variance_threshold"],
        corr_threshold=0.9,
        corr_method="spearman",
    )

    agg_df.to_csv(out_agg, index=False)
    norm_df.to_csv(out_norm, index=False)
    red_df.to_csv(out_red, index=False)

    print(red_df.head())
    print(f"Proceso terminado plate {plate}")

"""
VII_Reproducibility.py
======================
Reproducibility analysis of morphological profiles.

Metrics computed
----------------
1. Percent Replicating  — median pairwise Spearman vs. 95th-percentile null
2. Mean Average Precision (mAP) — via copairs
3. Cosine similarity    — mean / std between replicates per compound
4. Spearman correlation — mean / std between replicates per compound
5. Cohen's d            — effect size separating same-compound vs. cross-compound
                          similarity distributions (per metric)

Usage
-----
python VII_Reproducibility.py -i <input_dir> -o <output_dir> -c <cohort>

Input
-----
Searches <input_dir> for:
  - *_red*.csv      → reduced / feature-selected profiles
  - *cohort_norm*.csv → cohort-normalised profiles

Each CSV must contain a "Metadata_Perturbation" column plus numeric feature columns.

Output
------
<output_dir>/
  <cohort>_reproducibility_report.csv   — per-compound summary of all metrics
  <cohort>_map_per_sample.csv           — per-well average precision scores
  <cohort>_cohens_d.csv                 — Cohen's d per similarity metric
"""

import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import polars as pl
import pandas as pd
import argparse

import scipy
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from numba import njit


# copairs is optional — gracefully skip mAP if not installed
try:
    from copairs.map import average_precision as copairs_average_precision
    COPAIRS_AVAILABLE = True
except ImportError:
    COPAIRS_AVAILABLE = False
    print("[WARNING] 'copairs' not installed. mAP metric will be skipped.")

DEFAULT_LABEL_COL    = "Metadata_Perturbation"
DEFAULT_RANDOM_STATE = 42
DEFAULT_NULL_REPS    = 5_000


# ── Numba helpers (Spearman via rank-Pearson) ─────────────────────────────────

@njit
def _pearson_on_ranks(x: np.ndarray, y: np.ndarray) -> float:
    n  = x.shape[0]
    mx = x.mean()
    my = y.mean()
    num = denx = deny = 0.0
    for k in range(n):
        dx = x[k] - mx
        dy = y[k] - my
        num  += dx * dy
        denx += dx * dx
        deny += dy * dy
    denom = np.sqrt(denx * deny)
    return num / denom if denom > 0 else 0.0


@njit
def _pairwise_spearman_from_ranks(ranks: np.ndarray) -> np.ndarray:
    n, _ = ranks.shape
    out  = np.empty(n * (n - 1) // 2, dtype=np.float32)
    idx  = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            out[idx] = _pearson_on_ranks(ranks[i], ranks[j])
            idx += 1
    return out


def _to_rank_matrix(X: np.ndarray) -> np.ndarray:
    """Row-wise rank matrix (float32) needed by Numba functions."""
    order = X.argsort(axis=1)
    return order.argsort(axis=1).astype(np.float32)


def _median_pairwise_spearman(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return np.nan
    corrs = _pairwise_spearman_from_ranks(_to_rank_matrix(X))
    return float(np.median(corrs))


# ── Metric 1 — Percent Replicating ───────────────────────────────────────────

def percent_replicating(
    df: pl.DataFrame,
    feature_cols: List[str],
    n_null: int = DEFAULT_NULL_REPS,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[dict, float]:
    """
    For each compound compute:
      - observed  : median pairwise Spearman among its replicates
      - null_95   : 95th percentile of null distribution
                    (random sets of n_reps profiles from *other* compounds)
      - replicates: 1 if observed > null_95 else 0

    Returns
    -------
    results : dict  {compound: {"observed", "null_95", "replicates"}}
    percent  : float  fraction of replicating compounds (0–1)
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RANDOM_STATE)

    compounds = df[DEFAULT_LABEL_COL].unique().to_list()

    feats_by_compound: dict[str, np.ndarray] = {
        c: df.filter(pl.col(DEFAULT_LABEL_COL) == c)
             .select(feature_cols)
             .to_numpy()
        for c in compounds
    }

    # Infer n_reps from the minimum replicate count (floor at 2)
    n_reps = max(2, min(X.shape[0] for X in feats_by_compound.values()))

    results: dict = {}
    for current in compounds:
        others = [c for c in compounds if c != current]
        if len(others) < n_reps:
            print(f"  [WARNING] Skipping null for '{current}': "
                  f"not enough other compounds ({len(others)} < {n_reps})")
            results[current] = {"observed": float("nan"), "null_95": float("nan"), "replicates": 0}
            continue

        # Observed
        observed = _median_pairwise_spearman(feats_by_compound[current])

        # Null distribution
        null_vals = np.empty(n_null, dtype=np.float32)
        for i in range(n_null):
            sampled = rng.choice(others, size=n_reps, replace=False)
            rows = [
                feats_by_compound[c][rng.integers(0, feats_by_compound[c].shape[0])]
                for c in sampled
            ]
            null_vals[i] = _median_pairwise_spearman(np.stack(rows))

        null_95 = float(np.percentile(null_vals, 95))
        results[current] = {
            "observed":   observed,
            "null_95":    null_95,
            "replicates": int(observed > null_95),
        }

    vals    = [v["replicates"] for v in results.values()]
    percent = float(np.mean(vals)) if vals else float("nan")
    return results, percent


# ── Metric 2 — Mean Average Precision (mAP) ──────────────────────────────────

def mean_average_precision(
    df: pl.DataFrame,
    feature_cols: List[str],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Returns
    -------
    map_per_sample  : pd.DataFrame with Metadata_Perturbation + average_precision
    map_per_compound: pd.Series indexed by Metadata_Perturbation
    Both are None if copairs is not installed.
    """
    if not COPAIRS_AVAILABLE:
        return None, None

    pdf = df.to_pandas()
    meta_cols = [c for c in pdf.columns if c.startswith("Metadata_")]
    meta  = pdf[meta_cols]
    feats = pdf[feature_cols].fillna(0).values

    result = copairs_average_precision(
        meta,
        feats,
        pos_sameby=["Metadata_Perturbation"],
        pos_diffby=[],
        neg_sameby=[],
        neg_diffby=["Metadata_Perturbation"],
        batch_size=20_000,
    )

    map_per_sample   = meta.copy()
    map_per_sample["average_precision"] = result["average_precision"]
    map_per_compound = (
        result.groupby("Metadata_Perturbation")["average_precision"]
              .mean()
              .sort_values(ascending=False)
    )
    return map_per_sample, map_per_compound


# ── Metric 3 — Cosine Similarity ─────────────────────────────────────────────

def cosine_similarity_per_compound(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
) -> pl.DataFrame:
    """Mean / std cosine similarity between replicates per compound."""
    cos_sim = cosine_similarity(X_scaled)
    same_label = labels[:, None] == labels[None, :]
    off_diag   = ~np.eye(len(labels), dtype=bool)
    mean_global = cos_sim[same_label & off_diag].mean()
    print(f"  [cosine]   global mean (all replicates): {mean_global:.4f}")

    rows = []
    for compound in label_names:
        idx = np.where(labels == compound)[0]
        if len(idx) < 2:
            continue
        sub      = cos_sim[np.ix_(idx, idx)]
        sim_vals = sub[~np.eye(len(idx), dtype=bool)]
        rows.append({
            DEFAULT_LABEL_COL: compound,
            "mean_cosine":     float(sim_vals.mean()),
            "std_cosine":      float(sim_vals.std()),
        })
    return pl.DataFrame(rows)


# ── Metric 4 — Spearman Correlation ──────────────────────────────────────────

def spearman_per_compound(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
) -> pl.DataFrame:
    """Mean / std Spearman correlation between replicates per compound."""
    rows      = []
    all_corrs = []

    for compound in label_names:
        idx = np.where(labels == compound)[0]
        if len(idx) < 2:
            continue
        corrs = [
            spearmanr(X_scaled[idx[i]], X_scaled[idx[j]]).correlation  
            for i in range(len(idx))
            for j in range(i + 1, len(idx))
        ]
        all_corrs.extend(corrs)
        rows.append({
            DEFAULT_LABEL_COL: compound,
            "mean_spearman":   float(np.mean(corrs)),
            "std_spearman":    float(np.std(corrs)),
        })

    if all_corrs:
        print(f"  [spearman] global mean (all replicates): {np.mean(all_corrs):.4f}")

    return pl.DataFrame(rows)


# ── Metric 5 — Cohen's d ─────────────────────────────────────────────────────

def cohens_d_similarity_metrics(
    X_scaled: np.ndarray,
    labels: np.ndarray,
) -> pl.DataFrame:
    """
    Computes Cohen's d for cosine and Spearman similarity distributions:
      - signal: same-compound pairs
      - noise : cross-compound pairs

    Returns a DataFrame with one row per metric.
    """
    n = len(labels)
    same_mask = labels[:, None] == labels[None, :]
    off_diag  = ~np.eye(n, dtype=bool)

    # --- cosine ---
    cos_sim    = cosine_similarity(X_scaled)
    same_cos   = cos_sim[same_mask  & off_diag]
    cross_cos  = cos_sim[~same_mask & off_diag]

    # --- spearman (vectorised via rank trick) ---
    ranks     = _to_rank_matrix(X_scaled)
    sp_matrix = np.corrcoef(ranks)          # Pearson of ranks ≈ Spearman
    same_sp   = sp_matrix[same_mask  & off_diag]
    cross_sp  = sp_matrix[~same_mask & off_diag]

    def _cohens_d(signal: np.ndarray, noise: np.ndarray) -> dict:
        n1, n2 = len(signal), len(noise)
        m1, m2 = signal.mean(), noise.mean()
        s1, s2 = signal.std(ddof=1), noise.std(ddof=1)
        if n1 + n2 > 2:
            pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        else:
            pooled = 0.0
        d = abs(m1 - m2) / pooled if pooled > 0 else 0.0
        return {"mean_signal": float(m1), "mean_noise": float(m2),
                "pooled_std": float(pooled), "cohens_d": float(d)}

    rows = [
        {"metric": "cosine",   **_cohens_d(same_cos, cross_cos)},
        {"metric": "spearman", **_cohens_d(same_sp,  cross_sp)},
    ]
    return pl.DataFrame(rows)


# ── Main analysis orchestrator ────────────────────────────────────────────────

class Reproducibility:
    """
    Runs all reproducibility metrics on a morphological profile CSV
    and saves reports to disk.

    Parameters
    ----------
    path_profiles : Path   CSV with Metadata_Perturbation + numeric features
    saving_path   : Path   Output directory
    cohort        : str    Prefix for output files
    n_null        : int    Number of null permutations for Percent Replicating
    random_state  : int    Seed
    """

    def __init__(
        self,
        path_profiles: Path,
        saving_path: Path,
        cohort: str,
        n_null: int       = DEFAULT_NULL_REPS,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        self.saving_path  = saving_path
        self.cohort       = cohort
        self.n_null       = n_null
        self.random_state = random_state
        self.rng          = np.random.default_rng(random_state)

        self.saving_path.mkdir(parents=True, exist_ok=True)

        self.df_raw, self.feature_cols, self.X_scaled, self.labels = (
            self._load_and_prepare(path_profiles)
        )
        self.label_names = np.unique(self.labels)

        self._run_all()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_and_prepare(self, path: Path):
        df = pl.read_csv(path)

        # Feature columns: numeric, not Metadata_*
        feature_cols = [
            c for c in df.columns
            if not c.startswith("Metadata_")
            and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]

        labels   = df[DEFAULT_LABEL_COL].to_numpy()
        X        = df.select(feature_cols).to_numpy().astype(float)
        X_scaled = StandardScaler().fit_transform(X)

        return df, feature_cols, X_scaled, labels

    # ── Run all metrics ───────────────────────────────────────────────────────

    def _run_all(self):
        print("\n── Percent Replicating ──────────────────────────────")
        pr_results, pr_percent = percent_replicating(
            self.df_raw, self.feature_cols, n_null=self.n_null, rng=self.rng
        )
        print(f"  Percent Replicating: {pr_percent * 100:.1f}%")

        print("\n── Mean Average Precision (mAP) ─────────────────────")
        map_per_sample, map_per_compound = mean_average_precision(
            self.df_raw, self.feature_cols
        )
        if map_per_compound is not None:
            print(f"  Global mAP: {map_per_compound.mean():.4f}")
        else:
            print("  mAP skipped (copairs not available)")

        print("\n── Cosine Similarity ────────────────────────────────")
        df_cosine = cosine_similarity_per_compound(
            self.X_scaled, self.labels, self.label_names
        )

        print("\n── Spearman Correlation ─────────────────────────────")
        df_spearman = spearman_per_compound(
            self.X_scaled, self.labels, self.label_names
        )

        print("\n── Cohen's d ────────────────────────────────────────")
        df_cohens = cohens_d_similarity_metrics(self.X_scaled, self.labels)
        print(df_cohens)

        # ── Assemble per-compound report ──────────────────────────────────────
        pr_rows = [
            {
                DEFAULT_LABEL_COL: c,
                "pr_observed":    v["observed"],
                "pr_null_95":     v["null_95"],
                "pr_replicates":  v["replicates"],
            }
            for c, v in pr_results.items()
        ]
        df_pr = pl.DataFrame(pr_rows)

        report = (
            pl.DataFrame({DEFAULT_LABEL_COL: list(self.label_names)})
            .join(df_pr,       on=DEFAULT_LABEL_COL, how="left")
            .join(df_cosine,   on=DEFAULT_LABEL_COL, how="left")
            .join(df_spearman, on=DEFAULT_LABEL_COL, how="left")
        )

        if map_per_compound is not None:
            df_map = pl.from_pandas(
                map_per_compound.reset_index().rename(
                    columns={"average_precision": "mAP"}
                )
            )
            report = report.join(df_map, on=DEFAULT_LABEL_COL, how="left")

        # Sort by cosine similarity descending
        sort_col = "mean_cosine" if "mean_cosine" in report.columns else report.columns[1]
        report = report.sort(sort_col, descending=True)

        # ── Save outputs ──────────────────────────────────────────────────────
        prefix = self.saving_path / self.cohort

        report_path = Path(f"{prefix}_reproducibility_report.csv")
        report.write_csv(report_path)
        print(f"\n  Report saved → {report_path}")

        cohens_path = Path(f"{prefix}_cohens_d.csv")
        df_cohens.write_csv(cohens_path)
        print(f"  Cohen's d saved → {cohens_path}")

        if map_per_sample is not None:
            map_path = Path(f"{prefix}_map_per_sample.csv")
            map_per_sample.to_csv(map_path, index=False)
            print(f"  mAP per sample saved → {map_path}")

        self.report    = report
        self.df_cohens = df_cohens


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducibility analysis of morphological profiles."
    )
    parser.add_argument("-i", "--input",  required=True, type=Path,
                        dest="input_dir",   help="Directory containing input CSV profiles.")
    parser.add_argument("-o", "--output", required=True, type=Path,
                        dest="saving_path", help="Directory for output reports.")
    parser.add_argument("-c", "--cohort", required=True, type=str,
                        help="Name of the analyzed cohort.")
    parser.add_argument("--null-reps", type=int, default=DEFAULT_NULL_REPS,
                        dest="n_null",
                        help=f"Null permutations for Percent Replicating "
                             f"(default: {DEFAULT_NULL_REPS}).")
    return parser.parse_args()


def _process_dataset(csv_path: Path, saving_path: Path, tag: str, cohort: str, n_null: int):
    print(f"\n{'─'*50}\nProcessing: {tag}\n{'─'*50}")
    Reproducibility(csv_path, saving_path, cohort, n_null=n_null)


def main():
    args = parse_args()
    args.saving_path.mkdir(parents=True, exist_ok=True)

    # ── Reduced profiles ─────────────────────────────────────────────────────
    red_files = sorted(args.input_dir.glob("**/*_red*.csv"))
    if red_files:
        print(f"\nFound {len(red_files)} reduced profile file(s).")
        import polars as pl
        combined = pl.concat([pl.read_csv(p) for p in red_files], how="diagonal")
        tmp = args.saving_path / "combined_red.csv"
        combined.write_csv(tmp)
        _process_dataset(tmp, args.saving_path / "combined_red",
                         "combined reduced profiles", args.cohort, args.n_null)
    else:
        print(f"[WARNING] No reduced files ('_red') found in {args.input_dir}")

    # ── Cohort norm profiles ──────────────────────────────────────────────────
    norm_files = sorted(args.input_dir.glob("**/*cohort_norm*.csv"))
    if norm_files:
        print(f"\nFound {len(norm_files)} cohort norm file(s).")
        import polars as pl
        combined = pl.concat([pl.read_csv(p) for p in norm_files], how="diagonal")
        tmp = args.saving_path / "combined_norm.csv"
        combined.write_csv(tmp)
        _process_dataset(tmp, args.saving_path / "combined_norm",
                         "combined norm profiles", args.cohort, args.n_null)
    else:
        print(f"[WARNING] No cohort norm files ('cohort_norm') found in {args.input_dir}")


if __name__ == "__main__":
    main()
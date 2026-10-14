"""
VIII_Reproducibility.py
======================
Reproducibility analysis of morphological profiles.

Metrics computed
----------------
1. Percent Replicating  — median pairwise Spearman vs. 95th-percentile null
2. Mean Average Precision (mAP) — implemented in this file (equivalent to copairs),
                          with permutation p-values and Benjamini-Hochberg correction
3. Cosine similarity    — mean / std between replicates per compound
4. Spearman correlation — mean / std between replicates per compound
5. Cohen's d            — effect size separating same-compound vs. cross-compound
                          similarity distributions (per metric)

Usage
-----
python VIII_Reproducibility.py -i <input_dir> -o <output_dir> -c <cohort>

Input
-----
Searches <input_dir> for the outputs of pycyto_pipe_broad.py (.csv and .csv.gz):
  - **/*_normalized_feature_select_batch.csv*  → feature-selected profiles
  - **/*_normalized.csv*                       → normalized profiles
(override the patterns with --reduced-glob / --norm-glob, e.g. for the negcon profiles)
Each CSV must contain "Metadata_Perturbation" and numeric feature columns.
Multiple files are concatenated using only the columns they share.

Output
------
<output_dir>/combined_red|combined_norm/
  <cohort>_reproducibility_report.csv   — per-compound summary of all metrics
  <cohort>_map_per_sample.csv           — per-profile average precision
  <cohort>_cohens_d.csv                 — Cohen's d per similarity metric
"""

from __future__ import annotations

import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Sequence, Union

import numpy as np
import polars as pl
import polars.selectors as cs

from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from numba import njit

DEFAULT_LABEL_COL     = "Metadata_Perturbation"
DEFAULT_RANDOM_STATE  = 42
DEFAULT_NULL_REPS     = 5_000
DEFAULT_MAP_NULL_SIZE = 10_000
DEFAULT_MAP_THRESHOLD = 0.05


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

    compounds = df[DEFAULT_LABEL_COL].unique(maintain_order=True).to_list()  # fixed order -> reproducible results

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


# ── mAP engine (copairs-equivalent, vectorized) ──────────────────────────────
#
# - Cosine distance on GPU (CuPy) when available; NumPy fallback.
# - Non-interpolated AP identical to copairs (ties: positives first, as in copairs).
# - Pairs defined with vectorized masks: a pair must match on ALL *_sameby
#   columns and differ on ALL *_diffby columns.
# - p-values: permutation test with a per-group null (average of its queries'
#   nulls) and Benjamini-Hochberg correction.
#   Reference: Kalinin et al. 2025, Nat. Commun. 16:5181.

# ----------------------------------------------------------------------
# Backend: CuPy (GPU) if available, otherwise NumPy (CPU)
# ----------------------------------------------------------------------
try:
    import cupy as _cp
    _GPU = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _cp = None
    _GPU = False


def _xp():
    return _cp if _GPU else np


def _to_host(a) -> np.ndarray:
    return _cp.asnumpy(a) if (_GPU and isinstance(a, _cp.ndarray)) else np.asarray(a)


# ----------------------------------------------------------------------
# 1. Cosine distance (full matrix)
# ----------------------------------------------------------------------
def cosine_distance_matrix(feats: np.ndarray) -> np.ndarray:
    """NxN cosine distance matrix (1 - similarity), returned on the host (NumPy)."""
    xp = _xp()
    x = xp.asarray(feats.astype(np.float32))
    norm = xp.linalg.norm(x, axis=1, keepdims=True)
    norm = xp.where(norm == 0, 1, norm)
    xn = x / norm
    sim = xp.clip(xn @ xn.T, -1.0, 1.0)
    return _to_host(1.0 - sim)


# ----------------------------------------------------------------------
# 2. Per-query pair masks (vectorized)
# ----------------------------------------------------------------------
def _encode(meta: pl.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    """Encode each metadata column as dense integers (fast comparisons)."""
    codes = {}
    for c in cols:
        if meta[c].null_count():
            raise ValueError(f"Metadata column '{c}' contains null values")
        codes[c] = meta.select(pl.col(c).cast(pl.Utf8).rank("dense"))[c].to_numpy()
    return codes


def _mask(codes, q: int, sameby: list[str], diffby: list[str], n: int) -> np.ndarray:
    m = np.ones(n, dtype=bool)
    for c in sameby:
        m &= codes[c] == codes[c][q]
    for c in diffby:
        m &= codes[c] != codes[c][q]
    m[q] = False
    return m


# ----------------------------------------------------------------------
# 3. Non-interpolated AP (equivalent to copairs.compute.ap_contiguous)
# ----------------------------------------------------------------------
def _average_precision(rel_k: np.ndarray) -> np.ndarray:
    """rel_k: binary matrix (n_queries x n_ranked), ordered by rank."""
    tp = np.cumsum(rel_k, axis=1)
    num_pos = tp[:, -1]
    k = np.arange(1, rel_k.shape[1] + 1)
    pr_k = tp / k
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pr_k * rel_k).sum(axis=1) / num_pos


# ----------------------------------------------------------------------
# 4. Permutation null (per (n_pos, n_total) configuration) and BH
# ----------------------------------------------------------------------
def _null_ap(null_size: int, num_pos: int, total: int, seed: int,
             chunk: int = 1000) -> np.ndarray:
    """Null AP distribution: random rankings with num_pos hits out of total.
    Generated in chunks to bound memory (chunk x total)."""
    rng = np.random.default_rng(seed)
    out = np.empty(null_size)
    for start in range(0, null_size, chunk):
        n = min(chunk, null_size - start)
        mat = np.zeros((n, total), dtype=np.int8)
        mat[:, :num_pos] = 1
        rng.permuted(mat, axis=1, out=mat)
        out[start:start + n] = _average_precision(mat)
    return out


def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg."""
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# ----------------------------------------------------------------------
# 5. Main API
# ----------------------------------------------------------------------
def compute_average_precision(
    df: pl.DataFrame,
    feature_cols: list[str],
    pos_sameby: list[str],
    pos_diffby: list[str] | None = None,
    neg_sameby: list[str] | None = None,
    neg_diffby: list[str] | None = None,
) -> pl.DataFrame:
    """
    Per-profile (query) AP. Returns only the queries with >=1 positive and >=1
    negative, with the metadata columns + average_precision, n_pos_pairs and
    n_total_pairs.
    """
    pos_sameby, pos_diffby = list(pos_sameby or []), list(pos_diffby or [])
    neg_sameby, neg_diffby = list(neg_sameby or []), list(neg_diffby or [])

    meta = df.select([c for c in df.columns if c not in feature_cols])
    n = df.height
    dist = cosine_distance_matrix(df.select(feature_cols).to_numpy())

    codes = _encode(meta, sorted(set(pos_sameby + pos_diffby + neg_sameby + neg_diffby)))

    ap_scores, n_pos, n_tot, keep = [], [], [], []
    for q in range(n):
        pm = _mask(codes, q, pos_sameby, pos_diffby, n)
        nm = _mask(codes, q, neg_sameby, neg_diffby, n)
        k_pos, k_neg = int(pm.sum()), int(nm.sum())
        if k_pos == 0 or k_neg == 0:
            continue
        d = np.concatenate([dist[q, pm], dist[q, nm]])
        rel = np.concatenate([np.ones(k_pos, np.int8), np.zeros(k_neg, np.int8)])
        order = np.argsort(d, kind="stable")  # ascending distance; ties: positives first
        ap_scores.append(_average_precision(rel[order][None, :])[0])
        n_pos.append(k_pos)
        n_tot.append(k_pos + k_neg)
        keep.append(q)

    if not keep:
        raise ValueError("No query has both positives and negatives: check the pair design")

    return (
        meta.with_row_index("__row")
        .filter(pl.col("__row").is_in(keep))
        .drop("__row")
        .with_columns(
            pl.Series("average_precision", ap_scores),
            pl.Series("n_pos_pairs", n_pos),
            pl.Series("n_total_pairs", n_tot),
        )
    )


def compute_mean_average_precision(
    ap_df: pl.DataFrame,
    sameby: list[str],
    null_size: int = 10_000,
    seed: int = 0,
    threshold: float = 0.05,
) -> pl.DataFrame:
    """
    AP -> mAP per `sameby` group, with a permutation p-value and BH correction.

    The null of each group is the average of its queries' AP nulls (each with
    its own (n_pos, n_total) configuration), as in copairs.
    """
    ap = ap_df["average_precision"].to_numpy()
    confs = np.column_stack([ap_df["n_pos_pairs"].to_numpy(), ap_df["n_total_pairs"].to_numpy()])

    uniq, inv = np.unique(confs, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    ss = np.random.SeedSequence(seed).spawn(len(uniq))
    nulls = np.stack([
        _null_ap(null_size, int(p), int(t), int(s.generate_state(1)[0]))
        for (p, t), s in zip(uniq, ss)
    ])  # (n_configs, null_size)

    work = ap_df.select(sameby).with_columns(
        pl.Series("__ap", ap), pl.Series("__cfg", inv)
    )
    groups = work.group_by(sameby, maintain_order=True).agg(
        pl.col("__ap").mean().alias("mean_average_precision"),
        pl.col("__ap").count().alias("n"),
        pl.col("__cfg").alias("__cfgs"),
    )

    pvals = np.empty(groups.height)
    for i, (m_ap, cfgs) in enumerate(zip(groups["mean_average_precision"], groups["__cfgs"])):
        counts = np.bincount(np.asarray(cfgs), minlength=len(uniq))
        null = (counts @ nulls) / counts.sum()
        pvals[i] = (1 + np.sum(null > m_ap)) / (null_size + 1)

    p_corr = _bh_correct(pvals)
    return groups.drop("__cfgs").with_columns(
        pl.Series("p_value", pvals),
        pl.Series("corrected_p_value", p_corr),
        pl.Series("below_threshold", p_corr < threshold),
    )


# ── Metric 2 — Mean Average Precision (mAP) ──────────────────────────────────

def mean_average_precision(
    df: pl.DataFrame,
    feature_cols: List[str],
    pos_diffby: Sequence[str] = (),
    neg_sameby: Sequence[str] = (),
    neg_diffby: Optional[Sequence[str]] = None,
    null_size: int = DEFAULT_MAP_NULL_SIZE,
    threshold: float = DEFAULT_MAP_THRESHOLD,
    seed: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    mAP with the copairs pair design:
      positives = same Metadata_Perturbation (and different on `pos_diffby`)
      negatives = pairs that match on `neg_sameby` and differ on `neg_diffby`
                  (default: different Metadata_Perturbation -> replicate retrieval
                  against all other compounds, DMSO included).

    Returns
    -------
    ap_per_sample   : per-profile AP (metadata + average_precision, n_pos_pairs, n_total_pairs)
    map_per_compound: per-compound mAP, p_value, corrected_p_value (BH) and below_threshold
    """
    neg_diffby = [DEFAULT_LABEL_COL] if neg_diffby is None else list(neg_diffby)
    meta_cols  = [c for c in df.columns if c.startswith("Metadata_")]

    # Replace NaN/null feature values with 0 before computing distances
    data = (
        df.select(meta_cols + feature_cols)
          .with_columns(cs.float().fill_nan(None))
          .with_columns(pl.col(feature_cols).fill_null(0.0))
    )

    ap_per_sample = compute_average_precision(
        data, feature_cols,
        pos_sameby=[DEFAULT_LABEL_COL], pos_diffby=list(pos_diffby),
        neg_sameby=list(neg_sameby),    neg_diffby=neg_diffby,
    )
    map_per_compound = compute_mean_average_precision(
        ap_per_sample, [DEFAULT_LABEL_COL],
        null_size=null_size, seed=seed, threshold=threshold,
    ).sort("mean_average_precision", descending=True)

    return ap_per_sample, map_per_compound


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
    Runs all reproducibility metrics on a morphological profile table
    and saves reports to disk.

    Parameters
    ----------
    profiles      : Path | pl.DataFrame   CSV (or DataFrame) with Metadata_Perturbation + features
    saving_path   : Path    Output directory
    cohort        : str     Prefix for output files
    n_null        : int     Number of null permutations for Percent Replicating
    random_state  : int     Seed
    map_pos_diffby, map_neg_sameby, map_neg_diffby : mAP pair design
                    (default: positives = same compound; negatives = other compounds)
    map_null_size : int     Number of permutations in the mAP null
    map_threshold : float   Corrected (BH) p-value cutoff for "retrieved"
    """

    def __init__(
        self,
        profiles: Union[Path, pl.DataFrame],
        saving_path: Path,
        cohort: str,
        n_null: int       = DEFAULT_NULL_REPS,
        random_state: int = DEFAULT_RANDOM_STATE,
        map_pos_diffby: Sequence[str]           = (),
        map_neg_sameby: Sequence[str]           = (),
        map_neg_diffby: Optional[Sequence[str]] = None,
        map_null_size: int   = DEFAULT_MAP_NULL_SIZE,
        map_threshold: float = DEFAULT_MAP_THRESHOLD,
    ):
        self.saving_path    = saving_path
        self.cohort         = cohort
        self.n_null         = n_null
        self.random_state   = random_state
        self.rng            = np.random.default_rng(random_state)
        self.map_pos_diffby = map_pos_diffby
        self.map_neg_sameby = map_neg_sameby
        self.map_neg_diffby = map_neg_diffby
        self.map_null_size  = map_null_size
        self.map_threshold  = map_threshold

        self.saving_path.mkdir(parents=True, exist_ok=True)

        self.df_raw, self.feature_cols, self.X_scaled, self.labels = (
            self._load_and_prepare(profiles)
        )
        self.label_names = np.unique(self.labels)

        self._run_all()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_and_prepare(self, profiles: Union[Path, pl.DataFrame]):
        df = profiles if isinstance(profiles, pl.DataFrame) else pl.read_csv(profiles)

        # Feature columns: numeric, not Metadata_*
        feature_cols = [
            c for c in df.columns
            if not c.startswith("Metadata_")
            and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]

        n_null = int(df.select(feature_cols).null_count().sum_horizontal().item())
        if n_null:
            raise ValueError(
                f"{n_null} null values in the features; check the profiles "
                f"(were files with different feature sets concatenated?)"
            )

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
            self.df_raw, self.feature_cols,
            pos_diffby=self.map_pos_diffby,
            neg_sameby=self.map_neg_sameby,
            neg_diffby=self.map_neg_diffby,
            null_size=self.map_null_size,
            threshold=self.map_threshold,
            seed=self.random_state,
        )
        print(f"  Global mAP: {map_per_compound['mean_average_precision'].mean():.4f}")
        print(f"  Compounds with significant mAP (BH p < {self.map_threshold}): "
              f"{map_per_compound['below_threshold'].mean() * 100:.1f}%")

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

        df_map = map_per_compound.select(
            DEFAULT_LABEL_COL,
            pl.col("mean_average_precision").alias("mAP"),
            pl.col("p_value").alias("map_p_value"),
            pl.col("corrected_p_value").alias("map_corrected_p_value"),
            pl.col("below_threshold").alias("map_significant"),
        )

        report = (
            pl.DataFrame({DEFAULT_LABEL_COL: list(self.label_names)})
            .join(df_pr,       on=DEFAULT_LABEL_COL, how="left")
            .join(df_map,      on=DEFAULT_LABEL_COL, how="left")
            .join(df_cosine,   on=DEFAULT_LABEL_COL, how="left")
            .join(df_spearman, on=DEFAULT_LABEL_COL, how="left")
        )

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

        map_path = Path(f"{prefix}_map_per_sample.csv")
        map_per_sample.write_csv(map_path)
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
    parser.add_argument("--reduced-glob", default="**/*_normalized_feature_select_batch.csv*",
                        help="Glob pattern for the feature-selected profiles.")
    parser.add_argument("--norm-glob", default="**/*_normalized.csv*",
                        help="Glob pattern for the normalized profiles.")

    m = parser.add_argument_group("mAP")
    m.add_argument("--map-null-size", type=int, default=DEFAULT_MAP_NULL_SIZE,
                   help=f"Permutations in the mAP null (default: {DEFAULT_MAP_NULL_SIZE}).")
    m.add_argument("--map-threshold", type=float, default=DEFAULT_MAP_THRESHOLD,
                   help="Corrected (BH) p-value cutoff.")
    m.add_argument("--map-pos-diffby", nargs="*", default=[],
                   help="Columns on which positives must DIFFER (e.g. Metadata_Plate).")
    m.add_argument("--map-neg-sameby", nargs="*", default=[],
                   help="Columns on which negatives must MATCH (e.g. Metadata_Plate).")
    m.add_argument("--map-neg-diffby", nargs="*", default=[DEFAULT_LABEL_COL],
                   help="Columns on which negatives must DIFFER.")
    return parser.parse_args()


def _combine_profiles(paths: List[Path]) -> pl.DataFrame:
    """Concatenate profiles using only the columns common to all files."""
    frames = [pl.read_csv(p) for p in paths]
    common = [c for c in frames[0].columns if all(c in f.columns for f in frames[1:])]
    dropped = {c for f in frames for c in f.columns} - set(common)
    if dropped:
        print(f"  [WARNING] Dropped {len(dropped)} columns not shared by all files")
    return pl.concat([f.select(common) for f in frames], how="vertical_relaxed")


def _process_dataset(profiles: pl.DataFrame, saving_path: Path, tag: str,
                     cohort: str, args: argparse.Namespace):
    print(f"\n{'─'*50}\nProcessing: {tag}\n{'─'*50}")
    Reproducibility(
        profiles, saving_path, cohort, n_null=args.n_null,
        map_pos_diffby=args.map_pos_diffby,
        map_neg_sameby=args.map_neg_sameby,
        map_neg_diffby=args.map_neg_diffby,
        map_null_size=args.map_null_size,
        map_threshold=args.map_threshold,
    )


def main():
    args = parse_args()
    args.saving_path.mkdir(parents=True, exist_ok=True)

    # ── Feature-selected profiles ────────────────────────────────────────────
    red_files = sorted(args.input_dir.glob(args.reduced_glob))
    if red_files:
        print(f"\nFound {len(red_files)} feature-selected profile file(s).")
        _process_dataset(_combine_profiles(red_files), args.saving_path / "combined_red",
                         "combined feature-selected profiles", args.cohort, args)
    else:
        print(f"[WARNING] No files matching '{args.reduced_glob}' in {args.input_dir}")

    # ── Normalized profiles ──────────────────────────────────────────────────
    norm_files = sorted(args.input_dir.glob(args.norm_glob))
    if norm_files:
        print(f"\nFound {len(norm_files)} normalized profile file(s).")
        _process_dataset(_combine_profiles(norm_files), args.saving_path / "combined_norm",
                         "combined norm profiles", args.cohort, args)
    else:
        print(f"[WARNING] No files matching '{args.norm_glob}' in {args.input_dir}")


if __name__ == "__main__":
    main()
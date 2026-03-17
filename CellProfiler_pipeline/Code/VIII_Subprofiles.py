"""
VIII_Subprofiles.py
===================
Subprofile-based morphological clustering using similarity graphs.

Implements an unsupervised, soft-clustering pipeline for Cell Painting
morphological profiles, inspired by Pahl & Ziegler (2023) but extended with:
  - Spearman rank correlation as primary similarity metric (configurable)
  - FDR-controlled similarity threshold derived from empirical null distribution
  - Graph-based community detection via Leiden algorithm (soft memberships)
  - Combined sign + magnitude criterion for dominant feature identification
  - Export format designed for multi-cohort integration (IX_Integration)

Usage
-----
python VIII_Subprofiles.py \\
    -i <input_dir> \\
    -o <output_dir> \\
    -c <cohort_name> \\
    [--fraction 0.85] \\
    [--magnitude 0.5] \\
    [--fdr 0.05] \\
    [--null-reps 1000] \\
    [--metric spearman] \\
    [--secondary-threshold 0.6] \\
    [--reference-col Metadata_Reference]

Input
-----
Searches <input_dir> for *_red*.csv or *cohort_norm*.csv files.
Each CSV must contain:
  - Metadata_Perturbation : compound identifier
  - Metadata_Reference (optional): boolean/str column marking reference compounds
  - Numeric feature columns (no Metadata_ prefix)

Output (designed for IX_Integration compatibility)
------
<cohort>_similarity_matrix.parquet     — full pairwise Spearman matrix
<cohort>_graph_edges.csv               — edges passing FDR threshold
<cohort>_community_assignments.csv     — per-compound cluster assignments + memberships
<cohort>_subprofiles.parquet           — dominant features + median profiles per cluster
<cohort>_subprofile_report.csv         — human-readable summary
<cohort>_reference_validation.csv      — similarity of reference compounds to clusters
<cohort>_graph.html                    — interactive Plotly network visualization
"""

import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

from pathlib import Path
from typing import Optional, List, Tuple, Dict
import warnings
import argparse

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

# Optional: igraph + leidenalg for community detection
try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    warnings.warn(
        "[WARNING] 'igraph' or 'leidenalg' not installed. "
        "Community detection will fall back to connected components only. "
        "Install with: pip install igraph leidenalg",
        stacklevel=2,
    )

DEFAULT_LABEL_COL      = "Metadata_Perturbation"
DEFAULT_REF_COL        = "Metadata_Reference"
DEFAULT_FRACTION       = 0.85
DEFAULT_MAGNITUDE      = 0.5
DEFAULT_FDR            = 0.05
DEFAULT_NULL_REPS      = 1_000
DEFAULT_METRIC         = "spearman"
DEFAULT_SEC_THRESHOLD  = 0.60
DEFAULT_RANDOM_STATE   = 42

SUPPORTED_METRICS = ("spearman", "pearson", "cosine")


# ── Similarity functions ──────────────────────────────────────────────────────

def _spearman_sim(u: np.ndarray, v: np.ndarray) -> float:
    """Spearman rank correlation, clipped to [0, 1]."""
    r, _ = spearmanr(u, v)
    return float(np.clip(r, 0.0, 1.0)) if not np.isnan(r) else 0.0


def _pearson_sim(u: np.ndarray, v: np.ndarray) -> float:
    """
    Pearson correlation with ±25 clipping (Pahl convention).
    Clipping reduces the influence of extreme feature outliers.
    """
    u = np.clip(u, -25.0, 25.0)
    v = np.clip(v, -25.0, 25.0)
    r, _ = pearsonr(u, v)
    return float(np.clip(r, 0.0, 1.0)) if not np.isnan(r) else 0.0


def _cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity, clipped to [0, 1]."""
    d = cosine_dist(u, v)
    return float(np.clip(1.0 - d, 0.0, 1.0))


METRIC_FN = {
    "spearman": _spearman_sim,
    "pearson":  _pearson_sim,
    "cosine":   _cosine_sim,
}


def compute_similarity_matrix(
    X: np.ndarray,
    metric: str = DEFAULT_METRIC,
) -> np.ndarray:
    """
    Compute full pairwise similarity matrix.

    Parameters
    ----------
    X      : (n_samples, n_features) array, already scaled
    metric : one of 'spearman', 'pearson', 'cosine'

    Returns
    -------
    S : (n_samples, n_samples) symmetric similarity matrix, diagonal = 1.0
    """
    if metric not in METRIC_FN:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {SUPPORTED_METRICS}.")

    fn  = METRIC_FN[metric]
    n   = X.shape[0]
    S   = np.eye(n, dtype=np.float32)

    for i in range(n - 1):
        for j in range(i + 1, n):
            s      = fn(X[i], X[j])
            S[i, j] = s
            S[j, i] = s

    return S


# ── FDR threshold from empirical null ────────────────────────────────────────

def fdr_threshold(
    S: np.ndarray,
    labels: np.ndarray,
    fdr: float = DEFAULT_FDR,
    n_null: int = DEFAULT_NULL_REPS,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Derive a similarity threshold using FDR correction against an empirical
    null distribution of cross-compound similarities.

    Strategy
    --------
    1. For each same-compound pair (i, j where label[i] == label[j], i != j),
       record the observed similarity S[i,j].
    2. Build a null distribution by randomly sampling cross-compound pairs
       of the same size as the same-compound set.
    3. Perform Benjamini-Hochberg FDR correction across all same-compound
       similarities using null-derived p-values.
    4. The threshold is the minimum similarity among pairs that survive FDR.

    Parameters
    ----------
    S      : pairwise similarity matrix
    labels : compound labels per row/col
    fdr    : false discovery rate threshold (default 0.05)
    n_null : size of null distribution sample
    rng    : numpy random generator

    Returns
    -------
    threshold   : float, similarity cutoff
    same_sims   : observed same-compound similarities
    null_sims   : sampled null similarities
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RANDOM_STATE)

    n = S.shape[0]
    same_mask = labels[:, None] == labels[None, :]
    off_diag  = ~np.eye(n, dtype=bool)

    # Upper triangle only to avoid double-counting
    triu = np.triu(np.ones((n, n), dtype=bool), k=1)
    same_pairs  = np.where(same_mask  & triu)
    cross_pairs = np.where(~same_mask & triu)

    same_sims  = S[same_pairs]
    cross_sims = S[cross_pairs]

    if len(same_sims) == 0:
        warnings.warn("[FDR] No same-compound pairs found. Cannot compute threshold.")
        return 0.0, same_sims, cross_sims

    # Sample null of equal size to same-compound pairs
    null_idx  = rng.integers(0, len(cross_sims), size=min(n_null, len(cross_sims)))
    null_sims = cross_sims[null_idx]

    # p-value for each same-compound similarity:
    # fraction of null that is >= observed (one-sided, higher = more similar)
    p_values = np.array([
        float((null_sims >= s).sum()) / len(null_sims)
        for s in same_sims
    ])
    # Avoid p=0 (conservative)
    p_values = np.clip(p_values, 1.0 / len(null_sims), 1.0)

    reject, _, _, _ = multipletests(p_values, alpha=fdr, method="fdr_bh")

    if reject.any():
        threshold = float(same_sims[reject].min())
    else:
        warnings.warn(
            f"[FDR] No same-compound pairs survived FDR={fdr}. "
            "Consider relaxing --fdr or checking profile quality. "
            "Falling back to 95th percentile of null distribution."
        )
        threshold = float(np.percentile(null_sims, 95))

    print(f"  [FDR threshold] {threshold:.4f}  "
          f"(FDR={fdr}, {reject.sum()}/{len(reject)} same-compound pairs retained)")

    return threshold, same_sims, null_sims


# ── Graph construction ────────────────────────────────────────────────────────

def build_similarity_graph(
    S: np.ndarray,
    compound_names: np.ndarray,
    threshold: float,
) -> Tuple["ig.Graph", List[tuple]]:
    """
    Build an undirected weighted graph where nodes are compounds and edges
    connect pairs with similarity >= threshold.

    Parameters
    ----------
    S              : pairwise similarity matrix (aggregated per compound)
    compound_names : array of compound identifiers
    threshold      : minimum similarity for an edge

    Returns
    -------
    G     : igraph.Graph with 'name' vertex attribute and 'weight' edge attribute
    edges : list of (name_i, name_j, similarity) tuples
    """
    n     = len(compound_names)
    edges = []
    edge_weights = []

    for i in range(n - 1):
        for j in range(i + 1, n):
            if S[i, j] >= threshold:
                edges.append((compound_names[i], compound_names[j], float(S[i, j])))
                edge_weights.append(float(S[i, j]))

    if not LEIDEN_AVAILABLE:
        return None, edges

    G = ig.Graph()
    G.add_vertices(n)
    G.vs["name"] = list(compound_names)

    ig_edges = [(i, j) for i in range(n - 1) for j in range(i + 1, n)
                if S[i, j] >= threshold]
    ig_weights = [S[i, j] for i in range(n - 1) for j in range(i + 1, n)
                  if S[i, j] >= threshold]

    G.add_edges(ig_edges)
    G.es["weight"] = ig_weights

    print(f"  [Graph] {n} nodes, {len(ig_edges)} edges above threshold {threshold:.4f}")
    return G, edges


# ── Community detection (Leiden) ─────────────────────────────────────────────

def detect_communities(
    G: "ig.Graph",
    random_state: int = DEFAULT_RANDOM_STATE,
) -> List[List[int]]:
    """
    Run Leiden community detection on the weighted similarity graph.

    Leiden (Traag et al., 2019) is preferred over Louvain because it
    guarantees well-connected communities and is less sensitive to
    resolution parameter initialization.

    Parameters
    ----------
    G            : igraph.Graph with 'weight' edge attribute
    random_state : seed for reproducibility

    Returns
    -------
    List of communities, each a list of vertex indices.
    """
    if not LEIDEN_AVAILABLE:
        # Fallback: connected components
        components = G.clusters() if G is not None else []
        return [list(c) for c in components]

    partition = leidenalg.find_partition(
        G,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=random_state,
        n_iterations=-1,  # run until convergence
    )
    communities = [list(c) for c in partition]
    print(f"  [Leiden] {len(communities)} communities detected")
    return communities


# ── Aggregate similarity per compound ────────────────────────────────────────

def aggregate_to_compounds(
    S_replicate: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
) -> np.ndarray:
    """
    Aggregate the replicate-level similarity matrix to compound level
    by taking the median similarity across all replicate pairs.

    Returns
    -------
    S_compound : (n_compounds, n_compounds) median similarity matrix
    """
    n = len(label_names)
    S_compound = np.eye(n, dtype=np.float32)

    for i, ci in enumerate(label_names):
        for j, cj in enumerate(label_names):
            if i >= j:
                continue
            idx_i = np.where(labels == ci)[0]
            idx_j = np.where(labels == cj)[0]
            vals  = S_replicate[np.ix_(idx_i, idx_j)].ravel()
            med   = float(np.median(vals))
            S_compound[i, j] = med
            S_compound[j, i] = med

    return S_compound


# ── Dominant feature identification ──────────────────────────────────────────

def dominant_features(
    X_compound: np.ndarray,
    feature_cols: List[str],
    fraction: float = DEFAULT_FRACTION,
    magnitude: float = DEFAULT_MAGNITUDE,
) -> List[str]:
    """
    Identify features that consistently define a morphological cluster,
    using a combined sign-consistency + magnitude criterion.

    A feature is dominant if:
      (a) >= `fraction` of replicates share the same sign (Pahl criterion)
      AND
      (b) The median absolute value >= `magnitude` (new criterion)

    The magnitude criterion prevents features with negligible effect sizes
    from being selected solely because they consistently point in one direction
    (e.g., all slightly positive but biologically irrelevant).

    Parameters
    ----------
    X_compound  : (n_replicates, n_features) array for one compound/cluster
    feature_cols: list of feature names
    fraction    : minimum fraction with same sign (default 0.85, Pahl 2023)
    magnitude   : minimum median |zscore| (default 0.5)

    Returns
    -------
    List of dominant feature names.
    """
    n      = X_compound.shape[0]
    result = []

    for k, feat in enumerate(feature_cols):
        vals         = X_compound[:, k]
        count_plus   = int((vals >= 0.0).sum())
        count_minus  = int((vals  < 0.0).sum())
        sign_frac    = max(count_plus, count_minus) / n
        median_abs   = float(np.median(np.abs(vals)))

        if sign_frac >= fraction and median_abs >= magnitude:
            result.append(feat)

    return result


# ── Subprofile computation ────────────────────────────────────────────────────

def compute_subprofile(
    X_cluster: np.ndarray,
    feature_cols: List[str],
    dom_feats: List[str],
    cluster_name: str,
) -> pd.DataFrame:
    """
    Compute the median subprofile over dominant features for a cluster.

    Returns a single-row DataFrame with dominant feature values
    and cluster identifier, ready for similarity scoring.
    """
    feat_idx  = [feature_cols.index(f) for f in dom_feats]
    med_vals  = np.median(X_cluster[:, feat_idx], axis=0)
    df        = pd.DataFrame([med_vals], columns=dom_feats)
    df[DEFAULT_LABEL_COL] = cluster_name
    return df


# ── Soft membership scoring ───────────────────────────────────────────────────

def soft_memberships(
    X_compound: np.ndarray,          # (1, n_dom_feats) — median profile of compound
    subprofiles: List[pd.DataFrame], # one per cluster
    metric_fn,
    secondary_threshold: float = DEFAULT_SEC_THRESHOLD,
) -> Dict[str, float]:
    """
    Compute similarity of a compound's median profile to each cluster subprofile.
    Returns a dict {cluster_name: similarity} for all clusters above
    secondary_threshold (captures polypharmacology).
    """
    memberships = {}
    for sp in subprofiles:
        cl_name  = sp[DEFAULT_LABEL_COL].values[0]
        dom_cols = [c for c in sp.columns if c != DEFAULT_LABEL_COL]
        if len(dom_cols) == 0:
            continue
        # Project compound profile onto dominant features of this cluster
        try:
            comp_vec = X_compound[dom_cols].values.flatten()
            cl_vec   = sp[dom_cols].values.flatten()
            sim      = metric_fn(comp_vec, cl_vec)
        except (KeyError, ValueError):
            sim = 0.0
        if sim >= secondary_threshold:
            memberships[cl_name] = round(sim, 4)

    return dict(sorted(memberships.items(), key=lambda x: x[1], reverse=True))


# ── Barcode plot (Axel Pahl style) ───────────────────────────────────────────

def plot_barcode(
    X_cluster: np.ndarray,
    feature_cols: List[str],
    dom_feats: List[str],
    cluster_name: str,
    save_path: Path,
    labels: Optional[np.ndarray] = None,
    compartment_order: Optional[List[str]] = None,
) -> None:
    """
    Barcode plot in the style of Pahl & Ziegler (2023).

    One horizontal strip per compound (median across its replicates).
    Compounds are stacked vertically, labelled on the left.
    Features are ordered by compartment (Cells → Cytoplasm → Nuclei).
    Colour: blue (−) → white (0) → red (+).

    Parameters
    ----------
    X_cluster       : (n_replicates × n_features) z-scored array for the cluster
    feature_cols    : ordered list of all feature names
    dom_feats       : subset that passed dominant-feature selection
    cluster_name    : title string (e.g. "Cluster_00")
    save_path       : output PNG path
    labels          : per-row compound name array (len == n_replicates).
                      If None, all rows are treated as one compound.
    compartment_order : compartment keywords in desired left-to-right order;
                        defaults to ["Cells", "Cytoplasm", "Nuclei"]
    """
    if compartment_order is None:
        compartment_order = ["Cells", "Cytoplasm", "Nuclei"]

    # ── 1. Sort features by compartment then name ─────────────────────────
    def _compartment(feat: str) -> int:
        fl = feat.lower()
        for i, comp in enumerate(compartment_order):
            if comp.lower() in fl:
                return i
        return len(compartment_order)

    sorted_idx   = sorted(range(len(feature_cols)),
                          key=lambda k: (_compartment(feature_cols[k]),
                                         feature_cols[k]))
    sorted_feats = [feature_cols[i] for i in sorted_idx]
    dom_set      = set(dom_feats)

    # ── 2. Compute per-compound median (collapse replicates) ──────────────
    if labels is not None:
        compound_names = list(dict.fromkeys(labels))   # unique, order-preserved
        medians_per_cpd = np.array([
            np.median(X_cluster[labels == cpd], axis=0)
            for cpd in compound_names
        ])                                             # (n_compounds, n_features)
    else:
        compound_names  = [cluster_name]
        medians_per_cpd = np.median(X_cluster, axis=0, keepdims=True)

    medians_sorted = medians_per_cpd[:, sorted_idx]   # (n_compounds, n_features)

    # ── 3. Colormap + shared normalisation ───────────────────────────────
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "bwr_pahl", ["#2166ac", "#f7f7f7", "#d6604d"]
    )
    vmax = max(2.0, float(np.nanpercentile(np.abs(medians_sorted), 95)))
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    # ── 4. Figure geometry ────────────────────────────────────────────────
    n_feats    = len(sorted_feats)
    n_cpd      = len(compound_names)
    row_height = 0.28            # inches per compound strip
    top_pad    = 0.50            # title
    bot_pad    = 0.45            # compartment labels + feature indices
    fig_width  = max(8, n_feats / 50)
    fig_height = top_pad + n_cpd * row_height + bot_pad

    fig = plt.figure(figsize=(fig_width, fig_height))

    # One axes per compound row
    strip_h  = row_height / fig_height          # axes height in figure fraction
    strip_top = 1.0 - top_pad / fig_height      # where first strip starts

    axes = []
    for i in range(n_cpd):
        bottom = strip_top - (i + 1) * strip_h
        ax = fig.add_axes([0.0, bottom, 0.88, strip_h * 0.82])
        axes.append(ax)

    # ── 5. Draw barcode strips ────────────────────────────────────────────
    for row_i, (ax, cpd) in enumerate(zip(axes, compound_names)):
        vals = medians_sorted[row_i]

        # Build colour image: shape (1, n_feats, 4)
        rgba = cmap(norm(vals))[np.newaxis, :, :]
        ax.imshow(rgba, aspect="auto", extent=[-0.5, n_feats - 0.5, 0, 1],
                  interpolation="nearest")

        # Dim non-dominant features with a white overlay
        for x, feat in enumerate(sorted_feats):
            if feat not in dom_set:
                ax.axvspan(x - 0.5, x + 0.5, ymin=0, ymax=1,
                           color="white", alpha=0.72, zorder=2)

        ax.set_xlim(-0.5, n_feats - 0.5)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Compound label on the left
        ax.text(-0.01, 0.5, cpd,
                ha="right", va="center", fontsize=7,
                transform=ax.transAxes, clip_on=False)

    # ── 6. Compartment labels, dividers and feature indices ───────────────
    # Use the bottom axes as reference for annotation
    ax_bot = axes[-1]

    comp_positions: Dict[str, List[int]] = {c: [] for c in compartment_order}
    for x, feat in enumerate(sorted_feats):
        for comp in compartment_order:
            if comp.lower() in feat.lower():
                comp_positions[comp].append(x)
                break

    for comp in compartment_order:
        xs = comp_positions.get(comp, [])
        if not xs:
            continue
        x_start = min(xs)
        x_end   = max(xs)
        mid     = (x_start + x_end) / 2

        # "1"-based index at start and end of compartment
        ax_bot.text(x_start, -0.15, str(x_start + 1),
                    ha="center", va="top", fontsize=6,
                    color="#555555", transform=ax_bot.transData, clip_on=False)
        ax_bot.text(x_end, -0.15, str(x_end + 1),
                    ha="center", va="top", fontsize=6,
                    color="#555555", transform=ax_bot.transData, clip_on=False)

        # Compartment name centred below
        ax_bot.text(mid, -0.55, comp,
                    ha="center", va="top", fontsize=8,
                    color="#333333", transform=ax_bot.transData, clip_on=False)

        # Vertical divider between compartments
        if x_start > 0:
            for ax in axes:
                ax.axvline(x_start - 0.5, color="#888888",
                           linewidth=0.6, linestyle="--", zorder=3)

    # ── 7. Title ──────────────────────────────────────────────────────────
    fig.text(0.01, 1.0 - (top_pad * 0.18) / fig_height,
             f"{cluster_name}  —  {n_cpd} compounds  |  "
             f"{len(dom_feats)} / {n_feats} dominant features",
             ha="left", va="top", fontsize=9, fontweight="bold",
             color="#111111")

    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"    Barcode plot → {save_path}")


# ── Interactive graph visualization ──────────────────────────────────────────

def plot_similarity_graph(
    S_compound: np.ndarray,
    compound_names: np.ndarray,
    community_map: Dict[str, int],
    edges: List[tuple],
    threshold: float,
    save_path: Path,
    cohort: str,
):
    """
    Interactive Plotly network visualization of the similarity graph.
    Node color = community. Edge width = similarity strength.
    Hovering shows compound name + community + top memberships.
    """
    try:
        import igraph as ig
        n = len(compound_names)
        G_layout = ig.Graph()
        G_layout.add_vertices(n)
        ig_edges = [(list(compound_names).index(e[0]),
                     list(compound_names).index(e[1])) for e in edges]
        G_layout.add_edges(ig_edges)
        layout = G_layout.layout("fr")  # Fruchterman-Reingold
        pos    = {compound_names[i]: layout[i] for i in range(n)}
    except Exception:
        # Fallback: circular layout
        angles = np.linspace(0, 2 * np.pi, len(compound_names), endpoint=False)
        pos = {name: [np.cos(a), np.sin(a)]
               for name, a in zip(compound_names, angles)}

    n_communities = max(community_map.values()) + 1 if community_map else 1
    cmap = [f"hsl({int(360 * i / max(n_communities, 1))},70%,55%)"
            for i in range(n_communities)]

    # Edge traces
    edge_traces = []
    for (ci, cj, sim) in edges:
        x0, y0 = pos[ci]
        x1, y1 = pos[cj]
        width   = 1 + 5 * (sim - threshold) / (1.0 - threshold + 1e-6)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color="rgba(150,150,150,0.5)"),
            hoverinfo="none",
            showlegend=False,
        ))

    # Node trace per community
    node_traces = []
    communities_present = sorted(set(community_map.values()))
    for comm_id in communities_present:
        members = [c for c, cid in community_map.items() if cid == comm_id]
        node_traces.append(go.Scatter(
            x=[pos[c][0] for c in members],
            y=[pos[c][1] for c in members],
            mode="markers+text",
            marker=dict(size=18, color=cmap[comm_id % len(cmap)],
                        line=dict(width=1.5, color="white")),
            text=members,
            textposition="top center",
            textfont=dict(size=9),
            name=f"Cluster {comm_id}",
            hovertemplate="<b>%{text}</b><br>Cluster: " + str(comm_id) + "<extra></extra>",
        ))

    # Isolated nodes (no edges)
    connected = {e[0] for e in edges} | {e[1] for e in edges}
    isolated  = [c for c in compound_names if c not in connected]
    if isolated:
        node_traces.append(go.Scatter(
            x=[pos[c][0] for c in isolated],
            y=[pos[c][1] for c in isolated],
            mode="markers+text",
            marker=dict(size=14, color="lightgray",
                        line=dict(width=1, color="gray")),
            text=isolated,
            textposition="top center",
            textfont=dict(size=8),
            name="Isolated",
            hovertemplate="<b>%{text}</b><br>No significant connections<extra></extra>",
        ))

    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        title=dict(text=f"{cohort} — Morphological Similarity Graph "
                        f"(threshold={threshold:.3f})", font=dict(size=16)),
        showlegend=True,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1100, height=850,
        plot_bgcolor="rgba(250,250,250,0.9)",
        legend=dict(title="Community", font=dict(size=11)),
    )
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    with open(save_path, "w") as f:
        f.write(html)
    print(f"  [Graph plot] saved → {save_path}")


# ── Main orchestrator ─────────────────────────────────────────────────────────

class Subprofiles:
    """
    Full subprofile analysis pipeline for a single cohort.

    Parameters
    ----------
    path_profiles        : CSV with Metadata_Perturbation + numeric features
    saving_path          : output directory
    cohort               : cohort name prefix
    fraction             : sign-consistency threshold for dominant features
    magnitude            : minimum |zscore| for dominant features
    fdr                  : FDR level for similarity threshold
    n_null               : null permutations for FDR estimation
    metric               : similarity metric ('spearman', 'pearson', 'cosine')
    secondary_threshold  : minimum similarity for secondary cluster assignment
    reference_col        : column marking reference/control compounds (optional)
    random_state         : seed
    """

    def __init__(
        self,
        path_profiles: Path,
        saving_path: Path,
        cohort: str,
        fraction: float           = DEFAULT_FRACTION,
        magnitude: float          = DEFAULT_MAGNITUDE,
        fdr: float                = DEFAULT_FDR,
        n_null: int               = DEFAULT_NULL_REPS,
        metric: str               = DEFAULT_METRIC,
        secondary_threshold: float = DEFAULT_SEC_THRESHOLD,
        reference_col: str        = DEFAULT_REF_COL,
        random_state: int         = DEFAULT_RANDOM_STATE,
    ):
        self.saving_path         = saving_path
        self.cohort              = cohort
        self.fraction            = fraction
        self.magnitude           = magnitude
        self.fdr                 = fdr
        self.n_null              = n_null
        self.metric              = metric
        self.metric_fn           = METRIC_FN[metric]
        self.secondary_threshold = secondary_threshold
        self.reference_col       = reference_col
        self.random_state        = random_state
        self.rng                 = np.random.default_rng(random_state)

        self.saving_path.mkdir(parents=True, exist_ok=True)

        (self.df_raw, self.feature_cols,
         self.X_scaled, self.labels,
         self.label_names, self.ref_compounds) = self._load(path_profiles)

        self._run()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self, path: Path):
        df = pl.read_csv(path)

        feature_cols = [
            c for c in df.columns
            if not c.startswith("Metadata_")
            and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]

        labels      = df[DEFAULT_LABEL_COL].to_numpy()
        label_names = np.unique(labels)
        X           = df.select(feature_cols).to_numpy().astype(float)
        X_scaled    = StandardScaler().fit_transform(X)

        # Reference compounds (optional)
        ref_compounds = set()
        if self.reference_col in df.columns:
            mask = df[self.reference_col].to_numpy()
            ref_compounds = set(labels[mask.astype(bool)])
            print(f"  [References] {len(ref_compounds)} reference compounds identified")

        return df, feature_cols, X_scaled, labels, label_names, ref_compounds

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def _run(self):
        prefix = self.saving_path / self.cohort

        # ── 1. Replicate-level similarity matrix ──────────────────────────────
        print("\n── Step 1: Pairwise similarity matrix ───────────────────────")
        S_rep = compute_similarity_matrix(self.X_scaled, metric=self.metric)

        # Save full matrix with compound labels for IX_Integration
        # Use numeric column names to avoid duplicates from biological replicates
        df_sim = pd.DataFrame(S_rep, columns=[f"rep_{i}" for i in range(S_rep.shape[1])])
        df_sim.insert(0, "compound", self.labels)
        df_sim.to_parquet(f"{prefix}_similarity_matrix.parquet", index=False)
        print(f"  Similarity matrix saved ({S_rep.shape}) → "
              f"{prefix}_similarity_matrix.parquet")

        # ── 2. FDR threshold ──────────────────────────────────────────────────
        print("\n── Step 2: FDR-derived similarity threshold ─────────────────")
        threshold, same_sims, null_sims = fdr_threshold(
            S_rep, self.labels, fdr=self.fdr,
            n_null=self.n_null, rng=self.rng,
        )
        self.threshold = threshold

        # ── 3. Compound-level aggregation ─────────────────────────────────────
        print("\n── Step 3: Aggregating replicates to compound level ─────────")
        S_cpd = aggregate_to_compounds(S_rep, self.labels, self.label_names)

        # ── 4. Graph construction ─────────────────────────────────────────────
        print("\n── Step 4: Building similarity graph ────────────────────────")
        G, edges = build_similarity_graph(S_cpd, self.label_names, threshold)

        # Save edges
        df_edges = pd.DataFrame(edges, columns=["compound_i", "compound_j", "similarity"])
        df_edges.to_csv(f"{prefix}_graph_edges.csv", index=False)
        print(f"  Edges saved → {prefix}_graph_edges.csv")

        # ── 5. Community detection ────────────────────────────────────────────
        print("\n── Step 5: Community detection (Leiden) ─────────────────────")
        if G is not None and LEIDEN_AVAILABLE:
            communities = detect_communities(G, random_state=self.random_state)
        else:
            # All connected components as fallback
            print("  [WARNING] Falling back to connected components.")
            communities = self._connected_components(edges)

        community_map = {}
        for cid, members in enumerate(communities):
            for m in members:
                name = (self.label_names[m]
                        if isinstance(m, (int, np.integer))
                        else m)
                community_map[name] = cid

        # Isolated compounds get their own singleton cluster
        for name in self.label_names:
            if name not in community_map:
                community_map[name] = max(community_map.values(), default=-1) + 1

        # ── 6. Dominant features + subprofiles per community ──────────────────
        print("\n── Step 6: Dominant features and subprofiles ────────────────")
        subprofiles   = []
        subprof_meta  = []

        # Build compound-median profile matrix
        X_cpd_median = {}
        for name in self.label_names:
            idx = np.where(self.labels == name)[0]
            X_cpd_median[name] = np.median(self.X_scaled[idx], axis=0)

        n_communities = max(community_map.values()) + 1
        for cid in range(n_communities):
            members = [n for n, c in community_map.items() if c == cid]
            if len(members) < 2:
                # Singleton: generate barcode anyway (one compound = one row)
                cpd      = members[0]
                idx_s    = np.where(self.labels == cpd)[0]
                X_s      = self.X_scaled[idx_s]
                dom_s    = dominant_features(X_s, self.feature_cols,
                                             fraction=self.fraction,
                                             magnitude=self.magnitude)
                cl_name_s = cpd  # use compound name as title
                print(f"  Singleton [{cpd}]: {len(dom_s)} dominant features")
                plot_barcode(
                    X_cluster    = X_s,
                    feature_cols = self.feature_cols,
                    dom_feats    = dom_s,
                    cluster_name = cl_name_s,
                    save_path    = Path(f"{prefix}_{cpd}_barcode.png"),
                    labels       = self.labels[idx_s],
                )
                subprof_meta.append({
                    "cluster_id":       cid,
                    "n_members":        1,
                    "members":          cpd,
                    "n_dominant_feats": len(dom_s),
                    "dominant_feats":   "|".join(dom_s),
                    "singleton":        True,
                })
                continue

            # Stack replicate profiles for all cluster members
            idx_all  = np.concatenate([
                np.where(self.labels == m)[0] for m in members
            ])
            X_cl     = self.X_scaled[idx_all]
            dom_feats = dominant_features(
                X_cl, self.feature_cols,
                fraction=self.fraction,
                magnitude=self.magnitude,
            )

            cl_name = f"Cluster_{cid:02d}"
            print(f"  {cl_name}: {len(members)} members, "
                  f"{len(dom_feats)} dominant features")

            # Barcode plot always — even if no dominant features (shown as all dimmed)
            plot_barcode(
                X_cluster    = X_cl,
                feature_cols = self.feature_cols,
                dom_feats    = dom_feats,
                cluster_name = cl_name,
                save_path    = Path(f"{prefix}_{cl_name}_barcode.png"),
                labels       = self.labels[idx_all],
            )

            if dom_feats:
                sp = compute_subprofile(X_cl, self.feature_cols,
                                        dom_feats, cl_name)
                subprofiles.append(sp)

            subprof_meta.append({
                "cluster_id":       cid,
                "n_members":        len(members),
                "members":          "|".join(members),
                "n_dominant_feats": len(dom_feats),
                "dominant_feats":   "|".join(dom_feats),
                "singleton":        False,
            })

        # Save subprofiles as parquet (wide format, one row per cluster)
        if subprofiles:
            # Align columns across subprofiles (fill missing with NaN)
            all_cols  = sorted({c for sp in subprofiles
                                for c in sp.columns if c != DEFAULT_LABEL_COL})
            sp_aligned = []
            for sp in subprofiles:
                for c in all_cols:
                    if c not in sp.columns:
                        sp[c] = np.nan
                sp_aligned.append(sp[[DEFAULT_LABEL_COL] + all_cols])
            df_subprofiles = pd.concat(sp_aligned, ignore_index=True)
            df_subprofiles.to_parquet(f"{prefix}_subprofiles.parquet", index=False)
            print(f"  Subprofiles saved → {prefix}_subprofiles.parquet")

        # ── Combined barcode panel (always — includes singletons) ──────────
        print("\n── Step 6b: Combined barcode panel ──────────────────────────")
        self._plot_combined_barcodes(
            subprof_meta  = subprof_meta,
            community_map = community_map,
            prefix        = prefix,
        )

        # ── 7. Soft memberships (polypharmacology) ────────────────────────────
        print("\n── Step 7: Soft cluster memberships ─────────────────────────")
        records = []
        for name in self.label_names:
            idx       = np.where(self.labels == name)[0]
            med_prof  = pd.DataFrame(
                np.median(self.X_scaled[idx], axis=0).reshape(1, -1),
                columns=self.feature_cols,
            )
            primary   = community_map.get(name, -1)
            if subprofiles:
                memberships = soft_memberships(
                    med_prof, subprofiles,
                    self.metric_fn, self.secondary_threshold,
                )
            else:
                memberships = {}

            is_ref = name in self.ref_compounds
            records.append({
                DEFAULT_LABEL_COL:  name,
                "primary_cluster":  primary,
                "primary_cluster_id": f"Cluster_{primary:02d}",
                "n_secondary":      max(0, len(memberships) - 1),
                "all_memberships":  str(memberships),
                "is_reference":     is_ref,
                # Fields useful for IX_Integration
                "cohort":           self.cohort,
                "fdr_threshold":    round(self.threshold, 6),
                "metric":           self.metric,
            })

        df_assignments = pd.DataFrame(records)
        df_assignments.to_csv(f"{prefix}_community_assignments.csv", index=False)
        print(f"  Assignments saved → {prefix}_community_assignments.csv")

        # ── 8. Reference validation (unsupervised) ────────────────────────────
        if self.ref_compounds and subprofiles:
            print("\n── Step 8: Reference compound validation ────────────────")
            ref_records = []
            for name in self.ref_compounds:
                idx      = np.where(self.labels == name)[0]
                med_prof = pd.DataFrame(
                    np.median(self.X_scaled[idx], axis=0).reshape(1, -1),
                    columns=self.feature_cols,
                )
                mems = soft_memberships(
                    med_prof, subprofiles,
                    self.metric_fn, self.secondary_threshold,
                )
                ref_records.append({
                    DEFAULT_LABEL_COL: name,
                    "assigned_cluster": community_map.get(name, -1),
                    "memberships":      str(mems),
                })
            pd.DataFrame(ref_records).to_csv(
                f"{prefix}_reference_validation.csv", index=False
            )
            print(f"  Reference validation saved → {prefix}_reference_validation.csv")

        # ── 9. Human-readable report ──────────────────────────────────────────
        df_report = pd.DataFrame(subprof_meta)
        df_report.to_csv(f"{prefix}_subprofile_report.csv", index=False)
        print(f"\n  Subprofile report saved → {prefix}_subprofile_report.csv")

        # ── 10. Interactive graph visualization ───────────────────────────────
        print("\n── Step 10: Interactive graph visualization ─────────────────")
        plot_similarity_graph(
            S_cpd, self.label_names, community_map, edges,
            threshold=self.threshold,
            save_path=Path(f"{prefix}_graph.html"),
            cohort=self.cohort,
        )

    # ── Combined barcode panel ─────────────────────────────────────────────────

    def _plot_combined_barcodes(
        self,
        subprof_meta: List[Dict],
        community_map: Dict[str, int],
        prefix: Path,
        compartment_order: Optional[List[str]] = None,
    ) -> None:
        """
        Stacked barcode panel — one row per compound (all compounds, including
        singletons). Each row shows ONLY its dominant features (variable width,
        Pahl style). A companion CSV with the dominant-feature vectors is also
        saved.
        """
        if compartment_order is None:
            compartment_order = ["Cells", "Cytoplasm", "Nuclei"]

        def _compartment(feat: str) -> int:
            fl = feat.lower()
            for i, comp in enumerate(compartment_order):
                if comp.lower() in fl:
                    return i
            return len(compartment_order)

        def _sort_feats(feats):
            return sorted(feats,
                          key=lambda f: (_compartment(f), f))

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "bwr_pahl", ["#2166ac", "#f7f7f7", "#d6604d"]
        )

        # ── Build one entry per compound ──────────────────────────────────────
        # Each entry: compound name, dominant features (sorted), median z-scores
        entries = []   # list of dicts
        for meta in subprof_meta:
            members   = meta["members"].split("|")
            dom_feats = [f for f in meta["dominant_feats"].split("|") if f]                         if meta["dominant_feats"] else []
            dom_feats = _sort_feats(dom_feats)

            for cpd in members:
                idx = np.where(self.labels == cpd)[0]
                med = np.median(self.X_scaled[idx], axis=0)

                # Per-compound dominant features: re-compute so singletons
                # also get their own dom_feats correctly
                cpd_dom = dominant_features(
                    self.X_scaled[idx], self.feature_cols,
                    fraction=self.fraction, magnitude=self.magnitude,
                )
                cpd_dom = _sort_feats(cpd_dom)

                feat_idx   = [self.feature_cols.index(f) for f in cpd_dom]
                dom_values = med[feat_idx] if feat_idx else np.array([])

                entries.append({
                    "compound":  cpd,
                    "dom_feats": cpd_dom,
                    "dom_vals":  dom_values,
                    "cluster":   f"Cluster_{meta['cluster_id']:02d}",
                    "singleton": meta["singleton"],
                })

        if not entries:
            print("  [WARNING] No compounds to plot in combined barcode panel.")
            return

        # ── Global colour scale ───────────────────────────────────────────────
        all_vals = np.concatenate([e["dom_vals"] for e in entries if len(e["dom_vals"]) > 0])
        vmax = max(2.0, float(np.nanpercentile(np.abs(all_vals), 95))) if len(all_vals) else 2.0
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

        # ── Figure layout ─────────────────────────────────────────────────────
        n_cpd      = len(entries)
        row_h      = 0.32           # inches per compound row
        label_w    = 2.8            # inches for compound name column
        bar_w      = 8.0            # inches for barcode area
        fig_w      = label_w + bar_w + 0.6   # + colorbar
        fig_h      = n_cpd * row_h + 0.9     # + compartment labels

        fig = plt.figure(figsize=(fig_w, fig_h))

        top_pad  = 0.55 / fig_h
        bot_pad  = 0.55 / fig_h
        left_pad = label_w / fig_w
        right_pad = 0.55 / fig_w
        plot_h   = (1.0 - top_pad - bot_pad) / n_cpd
        plot_w   = 1.0 - left_pad - right_pad - 0.06  # leave room for colorbar

        axes = []
        for i, entry in enumerate(entries):
            n_dom = max(len(entry["dom_feats"]), 1)
            bottom = 1.0 - top_pad - (i + 1) * plot_h
            ax = fig.add_axes([left_pad, bottom + plot_h * 0.08,
                               plot_w * n_dom / max(len(e["dom_feats"]) or 1
                                                    for e in entries),
                               plot_h * 0.84])
            axes.append(ax)

        # ── Draw each compound row ────────────────────────────────────────────
        max_dom = max((len(e["dom_feats"]) for e in entries), default=1)

        for ax, entry in zip(axes, entries):
            dom_feats = entry["dom_feats"]
            dom_vals  = entry["dom_vals"]
            n_dom     = len(dom_feats)

            ax.set_xlim(-0.5, max(n_dom - 0.5, 0.5))
            ax.set_ylim(0, 1)
            ax.axis("off")

            if n_dom == 0:
                ax.text(0.5, 0.5, "no dominant features",
                        ha="center", va="center", fontsize=5.5,
                        color="#aaaaaa", transform=ax.transAxes)
            else:
                rgba = cmap(norm(dom_vals))[np.newaxis, :, :]
                ax.imshow(rgba, aspect="auto",
                          extent=[-0.5, n_dom - 0.5, 0, 1],
                          interpolation="nearest")

            # Compound label to the left (in figure coords)
            fig.text(left_pad - 0.01,
                     ax.get_position().y0 + ax.get_position().height / 2,
                     entry["compound"],
                     ha="right", va="center", fontsize=7.5,
                     color="#111111")

            # Feature count to the right of the bar
            fig.text(left_pad + plot_w * n_dom / max_dom + 0.005,
                     ax.get_position().y0 + ax.get_position().height / 2,
                     f"n={n_dom}",
                     ha="left", va="center", fontsize=6,
                     color="#666666")

        # ── Compartment labels below the last row ─────────────────────────────
        # Use the entry with most features to anchor the labels
        anchor_entry = max(entries, key=lambda e: len(e["dom_feats"]))
        anchor_ax    = axes[entries.index(anchor_entry)]

        comp_positions: Dict[str, List[int]] = {c: [] for c in compartment_order}
        for x, feat in enumerate(anchor_entry["dom_feats"]):
            for comp in compartment_order:
                if comp.lower() in feat.lower():
                    comp_positions[comp].append(x)
                    break

        last_ax = axes[-1]
        for comp in compartment_order:
            xs = comp_positions.get(comp, [])
            if not xs:
                continue
            mid = (min(xs) + max(xs)) / 2
            last_ax.text(mid, -0.5, comp,
                         ha="center", va="top", fontsize=7.5,
                         color="#333333", transform=last_ax.transData,
                         clip_on=False)
            if min(xs) > 0:
                last_ax.axvline(min(xs) - 0.5, color="#999999",
                                linewidth=0.6, linestyle="--",
                                ymin=-0.2, ymax=1.1, clip_on=False, zorder=3)

        # ── Global colorbar ───────────────────────────────────────────────────
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb_ax = fig.add_axes([1.0 - right_pad - 0.035, bot_pad + 0.05,
                              0.018, 1.0 - top_pad - bot_pad - 0.1])
        cb = fig.colorbar(sm, cax=cb_ax)
        cb.set_label("median z-score", fontsize=6.5, labelpad=4)
        cb.ax.tick_params(labelsize=5.5)

        fig.suptitle(f"{self.cohort} — Morphological Profiles (dominant features only)",
                     fontsize=9, fontweight="bold",
                     x=left_pad + plot_w / 2, y=1.0 - top_pad * 0.3,
                     ha="center")

        out_png = Path(f"{prefix}_barcodes_all.png")
        plt.savefig(out_png, dpi=200, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"  Combined barcode panel → {out_png}")

        # ── CSV with dominant-feature vectors ─────────────────────────────────
        rows = []
        for entry in entries:
            row = {"compound": entry["compound"], "cluster": entry["cluster"],
                   "n_dominant_features": len(entry["dom_feats"])}
            for feat, val in zip(entry["dom_feats"], entry["dom_vals"]):
                row[feat] = round(float(val), 6)
            rows.append(row)
        df_csv = pd.DataFrame(rows)
        out_csv = Path(f"{prefix}_barcode_vectors.csv")
        df_csv.to_csv(out_csv, index=False)
        print(f"  Barcode vectors CSV  → {out_csv}")

    # ── Fallback: connected components without igraph ─────────────────────────

    def _connected_components(self, edges: List[tuple]) -> List[list]:
        graph: Dict[str, set] = {n: set() for n in self.label_names}
        for ci, cj, _ in edges:
            graph[ci].add(cj)
            graph[cj].add(ci)

        visited: set = set()
        components   = []

        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for node in self.label_names:
            if node not in visited:
                comp = []
                dfs(node, comp)
                components.append(comp)

        return components


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Subprofile-based morphological clustering via similarity graphs."
    )
    p.add_argument("-i", "--input",  required=True, type=Path, dest="input_dir")
    p.add_argument("-o", "--output", required=True, type=Path, dest="saving_path")
    p.add_argument("-c", "--cohort", required=True, type=str)
    p.add_argument("--fraction",    type=float, default=DEFAULT_FRACTION,
                   help=f"Sign-consistency fraction for dominant features (default {DEFAULT_FRACTION})")
    p.add_argument("--magnitude",   type=float, default=DEFAULT_MAGNITUDE,
                   help=f"Min |zscore| for dominant features (default {DEFAULT_MAGNITUDE})")
    p.add_argument("--fdr",         type=float, default=DEFAULT_FDR,
                   help=f"FDR level for similarity threshold (default {DEFAULT_FDR})")
    p.add_argument("--null-reps",   type=int,   default=DEFAULT_NULL_REPS, dest="n_null")
    p.add_argument("--metric",      type=str,   default=DEFAULT_METRIC,
                   choices=SUPPORTED_METRICS)
    p.add_argument("--secondary-threshold", type=float, default=DEFAULT_SEC_THRESHOLD,
                   dest="secondary_threshold",
                   help=f"Min similarity for secondary cluster assignment (default {DEFAULT_SEC_THRESHOLD})")
    p.add_argument("--reference-col", type=str, default=DEFAULT_REF_COL,
                   dest="reference_col")
    return p.parse_args()


def _process(csv_path: Path, saving_path: Path, tag: str, args: argparse.Namespace):
    print(f"\n{'─'*55}\nProcessing: {tag}\n{'─'*55}")
    Subprofiles(
        path_profiles        = csv_path,
        saving_path          = saving_path,
        cohort               = args.cohort,
        fraction             = args.fraction,
        magnitude            = args.magnitude,
        fdr                  = args.fdr,
        n_null               = args.n_null,
        metric               = args.metric,
        secondary_threshold  = args.secondary_threshold,
        reference_col        = args.reference_col,
    )


def main():
    args = parse_args()
    args.saving_path.mkdir(parents=True, exist_ok=True)

    # Prefer full profiles (cohort_norm) over reduced for this analysis
    norm_files = sorted(args.input_dir.glob("**/*_normalized*.csv") or args.input_dir.glob("**/*cohort_norm*.csv"))
    if norm_files:
        print(f"\nFound {len(norm_files)} cohort norm file(s).")
        combined = pl.concat([pl.read_csv(p) for p in norm_files], how="diagonal")
        tmp = args.saving_path / "combined_norm.csv"
        combined.write_csv(tmp)
        _process(tmp, args.saving_path / "subprofiles_norm",
                 "cohort norm profiles (full)", args)
    else:
        # Fallback to reduced
        red_files = sorted(args.input_dir.glob("**/*_red*.csv"))
        if red_files:
            print(f"\nFound {len(red_files)} reduced profile file(s) (fallback).")
            combined = pl.concat([pl.read_csv(p) for p in red_files], how="diagonal")
            tmp = args.saving_path / "combined_red.csv"
            combined.write_csv(tmp)
            _process(tmp, args.saving_path / "subprofiles_red",
                     "reduced profiles (fallback)", args)
        else:
            print(f"[ERROR] No profile files found in {args.input_dir}")


if __name__ == "__main__":
    main()
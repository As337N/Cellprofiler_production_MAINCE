"""
IX_MorphologicalMap.py
======================
Single-cohort morphological map from VIII_Subprofiles outputs.

Reads precomputed similarity matrix, community assignments, subprofiles,
and reproducibility metrics to build an interactive compound-level
similarity graph annotated with per-node confidence metrics.

Usage
-----
python IX_MorphologicalMap.py \\
    -i <cohort_root_dir> \\
    -o <output_dir> \\
    -c <cohort_name> \\
    [--fdr 0.05] \\
    [--recompute-threshold] \\
    [--null-reps 1000] \\
    [--metric spearman] \\
    [--secondary-threshold 0.60] \\
    [--prefix-filenames]

Input (expected directory structure)
-------------------------------------
<cohort_root_dir>/
  Subprofiles/
    similarity_matrix.parquet          (or <cohort>_similarity_matrix.parquet)
    community_assignments.csv
    subprofiles.parquet
    subprofile_report.csv
  Reproducibility/
    reproducibility_report.csv         (or <cohort>_reproducibility_report.csv)

Output
------
<output_dir>/
  <cohort>_morphological_map.html      — interactive compound-level graph
  <cohort>_compound_graph_edges.csv    — compound-level edges
  <cohort>_compound_nodes.csv          — node attributes + reproducibility
"""

import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import warnings
import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine as cosine_dist
from statsmodels.stats.multitest import multipletests

try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    warnings.warn(
        "[WARNING] 'igraph' or 'leidenalg' not installed. "
        "Community detection will fall back to connected components.",
        stacklevel=2,
    )

DEFAULT_LABEL_COL     = "Metadata_Perturbation"
DEFAULT_FDR           = 0.05
DEFAULT_NULL_REPS     = 1_000
DEFAULT_METRIC        = "spearman"
DEFAULT_SEC_THRESHOLD = 0.60
DEFAULT_RANDOM_STATE  = 42

SUPPORTED_METRICS = ("spearman", "pearson", "cosine")

# Fixed filenames (can be prefixed with cohort name via --prefix-filenames)
FNAME_SIMILARITY   = "similarity_matrix.parquet"
FNAME_ASSIGNMENTS  = "community_assignments.csv"
FNAME_SUBPROFILES  = "subprofiles.parquet"
FNAME_SP_REPORT    = "subprofile_report.csv"
FNAME_REPRO        = "reproducibility_report.csv"


# ── Similarity functions ──────────────────────────────────────────────────────

def _spearman_sim(u: np.ndarray, v: np.ndarray) -> float:
    r, _ = spearmanr(u, v)
    return float(np.clip(r, 0.0, 1.0)) if not np.isnan(r) else 0.0

def _pearson_sim(u: np.ndarray, v: np.ndarray) -> float:
    u = np.clip(u, -25.0, 25.0)
    v = np.clip(v, -25.0, 25.0)
    r, _ = pearsonr(u, v)
    return float(np.clip(r, 0.0, 1.0)) if not np.isnan(r) else 0.0

def _cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.clip(1.0 - cosine_dist(u, v), 0.0, 1.0))

METRIC_FN = {
    "spearman": _spearman_sim,
    "pearson":  _pearson_sim,
    "cosine":   _cosine_sim,
}


# ── File discovery ────────────────────────────────────────────────────────────

def resolve_path(directory: Path, fname: str, cohort: str, prefixed: bool) -> Path:
    """
    Resolve a file path, optionally prefixing with cohort name.

    Parameters
    ----------
    directory : parent directory
    fname     : base filename (e.g. 'similarity_matrix.parquet')
    cohort    : cohort name prefix
    prefixed  : if True, look for '<cohort>_<fname>' first, fall back to '<fname>'
    """
    if prefixed:
        prefixed_path = directory / f"{cohort}_{fname}"
        if prefixed_path.exists():
            return prefixed_path
        warnings.warn(
            f"Prefixed file '{prefixed_path.name}' not found, "
            f"falling back to '{fname}'."
        )
    plain_path = directory / fname
    if not plain_path.exists():
        raise FileNotFoundError(
            f"Could not find '{fname}' (or '{cohort}_{fname}') in {directory}. "
            f"Make sure VIII_Subprofiles and VII_Reproducibility have been run."
        )
    return plain_path


# ── Similarity aggregation to compound level ──────────────────────────────────

def aggregate_similarity_to_compounds(
    S_rep: np.ndarray,
    rep_labels: np.ndarray,
    compound_names: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate replicate-level similarity matrix to compound level.

    For each compound pair (i, j):
      - compound similarity = median of all cross-replicate similarities
    For each compound (diagonal):
      - intra-compound similarity = median of all pairwise replicate similarities
        (used as reproducibility metric)

    Returns
    -------
    S_cpd         : (n_compounds × n_compounds) median similarity matrix
    repro_sims    : (n_compounds,) median intra-compound replicate similarity
    """
    n = len(compound_names)
    S_cpd      = np.eye(n, dtype=np.float32)
    repro_sims = np.full(n, np.nan, dtype=np.float32)

    for i, ci in enumerate(compound_names):
        idx_i = np.where(rep_labels == ci)[0]

        # Intra-compound reproducibility
        if len(idx_i) >= 2:
            vals = []
            for a in range(len(idx_i)):
                for b in range(a + 1, len(idx_i)):
                    vals.append(float(S_rep[idx_i[a], idx_i[b]]))
            repro_sims[i] = float(np.median(vals))
        else:
            repro_sims[i] = np.nan  # single replicate: undefined

        for j, cj in enumerate(compound_names):
            if i >= j:
                continue
            idx_j = np.where(rep_labels == cj)[0]
            cross  = S_rep[np.ix_(idx_i, idx_j)].ravel()
            med    = float(np.median(cross))
            S_cpd[i, j] = med
            S_cpd[j, i] = med

    return S_cpd, repro_sims


# ── FDR threshold ─────────────────────────────────────────────────────────────

def compute_fdr_threshold(
    S_rep: np.ndarray,
    rep_labels: np.ndarray,
    fdr: float = DEFAULT_FDR,
    n_null: int = DEFAULT_NULL_REPS,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Derive FDR-controlled similarity threshold from replicate-level matrix.

    Signal : same-compound replicate pairs (upper triangle only)
    Null   : random cross-compound pairs of equal sample size

    Returns threshold (float).
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RANDOM_STATE)

    n         = S_rep.shape[0]
    triu      = np.triu(np.ones((n, n), dtype=bool), k=1)
    same      = rep_labels[:, None] == rep_labels[None, :]
    diff      = ~same

    same_sims  = S_rep[same  & triu]
    cross_sims = S_rep[diff  & triu]

    if len(same_sims) == 0:
        warnings.warn("[FDR] No same-compound pairs found. Returning 0.")
        return 0.0

    null_idx  = rng.integers(0, len(cross_sims),
                              size=min(n_null, len(cross_sims)))
    null_sims = cross_sims[null_idx]

    p_values = np.clip(
        np.array([(null_sims >= s).sum() / len(null_sims) for s in same_sims]),
        1.0 / len(null_sims), 1.0,
    )
    reject, _, _, _ = multipletests(p_values, alpha=fdr, method="fdr_bh")

    if reject.any():
        threshold = float(same_sims[reject].min())
    else:
        warnings.warn(
            f"[FDR] No same-compound pairs survived FDR={fdr}. "
            "Falling back to 95th percentile of null distribution."
        )
        threshold = float(np.percentile(null_sims, 95))

    print(f"  [FDR threshold] {threshold:.4f}  "
          f"(FDR={fdr}, {reject.sum()}/{len(reject)} same-compound pairs retained)")
    return threshold


# ── Graph construction ────────────────────────────────────────────────────────

def build_compound_graph(
    S_cpd: np.ndarray,
    compound_names: np.ndarray,
    threshold: float,
) -> tuple:
    """
    Build compound-level similarity graph.

    Returns
    -------
    G     : igraph.Graph (or None if igraph unavailable)
    edges : list of dicts with compound_i, compound_j, similarity
    """
    n     = len(compound_names)
    edges = []

    for i in range(n - 1):
        for j in range(i + 1, n):
            if S_cpd[i, j] >= threshold:
                edges.append({
                    "compound_i": compound_names[i],
                    "compound_j": compound_names[j],
                    "similarity": float(S_cpd[i, j]),
                })

    print(f"  [Graph] {n} nodes, {len(edges)} edges "
          f"(threshold={threshold:.4f})")

    if not LEIDEN_AVAILABLE:
        return None, edges

    G = ig.Graph()
    G.add_vertices(n)
    G.vs["name"] = list(compound_names)

    cname_list = list(compound_names)
    ig_edges   = [(cname_list.index(e["compound_i"]),
                   cname_list.index(e["compound_j"])) for e in edges]
    ig_weights = [e["similarity"] for e in edges]

    G.add_edges(ig_edges)
    G.es["weight"] = ig_weights

    return G, edges


# ── Community detection ───────────────────────────────────────────────────────

def detect_communities(
    G: "ig.Graph",
    compound_names: np.ndarray,
    edges: List[dict],
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, int]:
    """
    Run Leiden community detection. Falls back to connected components.
    Returns {compound_name: community_id} dict.
    """
    if LEIDEN_AVAILABLE and G is not None:
        partition    = leidenalg.find_partition(
            G, leidenalg.ModularityVertexPartition,
            weights="weight", seed=random_state, n_iterations=-1,
        )
        communities = [list(c) for c in partition]
        community_map = {}
        for cid, members in enumerate(communities):
            for m in members:
                community_map[compound_names[m]] = cid
        print(f"  [Leiden] {len(communities)} communities detected")
    else:
        # Connected components fallback
        graph: dict[str, set] = {n: set() for n in compound_names}
        for e in edges:
            graph[e["compound_i"]].add(e["compound_j"])
            graph[e["compound_j"]].add(e["compound_i"])
        visited, community_map, cid = set(), {}, 0
        def dfs(node):
            visited.add(node)
            community_map[node] = cid
            for nb in graph[node]:
                if nb not in visited:
                    dfs(nb)
        for node in compound_names:
            if node not in visited:
                dfs(node)
                cid += 1

    # Assign singletons (isolated nodes)
    max_cid = max(community_map.values(), default=-1)
    for name in compound_names:
        if name not in community_map:
            max_cid += 1
            community_map[name] = max_cid

    return community_map


# ── Node attribute assembly ───────────────────────────────────────────────────

def build_node_table(
    compound_names: np.ndarray,
    S_cpd: np.ndarray,
    repro_sims: np.ndarray,
    community_map: Dict[str, int],
    df_assignments: pd.DataFrame,
    df_repro: Optional[pd.DataFrame],
    threshold: float,
    cohort: str,
) -> pd.DataFrame:
    """
    Assemble per-node attribute table combining:
      - Graph position info (community, degree)
      - Reproducibility metrics (median replicate similarity,
        percent-replicating flag, mAP if available)
      - VIII assignments (primary cluster, memberships)

    Reproducibility metrics per node
    ---------------------------------
    median_replicate_sim : float [0–1]
        Median pairwise Spearman similarity among replicates of this compound.
        Higher = more reproducible morphological profile.
        NaN if only one replicate available.

    pr_replicates : int {0, 1}
        Whether the compound passed the Percent Replicating threshold
        (from VII_Reproducibility). 1 = replicating, 0 = not replicating.
        Used to visually flag unreliable nodes.

    mAP : float [0–1] (optional)
        Mean Average Precision from VII_Reproducibility, if available.
        Higher = compound is more consistently retrievable among similar profiles.

    node_confidence : str {'high', 'medium', 'low'}
        Derived confidence label:
          high   : pr_replicates=1 AND median_replicate_sim >= threshold
          medium : pr_replicates=1 OR  median_replicate_sim >= threshold
          low    : pr_replicates=0 AND median_replicate_sim <  threshold (or NaN)
    """
    cname_list = list(compound_names)
    n = len(compound_names)

    # Degree (number of edges above threshold)
    degrees = np.array([
        sum(1 for j in range(n) if i != j and S_cpd[i, j] >= threshold)
        for i in range(n)
    ])

    # Build base from assignments
    assign_idx = (
        df_assignments.set_index(DEFAULT_LABEL_COL)
        if DEFAULT_LABEL_COL in df_assignments.columns
        else df_assignments
    )

    # Reproducibility from VII
    repro_idx = None
    if df_repro is not None and DEFAULT_LABEL_COL in df_repro.columns:
        repro_idx = df_repro.set_index(DEFAULT_LABEL_COL)

    rows = []
    for i, name in enumerate(compound_names):
        row: dict = {
            DEFAULT_LABEL_COL:     name,
            "cohort":              cohort,
            "community_ix":        community_map.get(name, -1),
            "community_id":        f"Cluster_{community_map.get(name, -1):02d}",
            "graph_degree":        int(degrees[i]),
            "median_replicate_sim": (
                round(float(repro_sims[i]), 4)
                if not np.isnan(repro_sims[i]) else None
            ),
        }

        # From VIII community_assignments
        if name in assign_idx.index:
            a = assign_idx.loc[name]
            row["viii_primary_cluster"]  = a.get("primary_cluster_id", None)
            row["n_secondary_clusters"]  = a.get("n_secondary", 0)
            row["all_memberships"]       = a.get("all_memberships", "{}")
            row["is_reference"]          = bool(a.get("is_reference", False))
        else:
            row["viii_primary_cluster"]  = None
            row["n_secondary_clusters"]  = 0
            row["all_memberships"]       = "{}"
            row["is_reference"]          = False

        # From VII reproducibility
        pr_flag = None
        map_val = None
        if repro_idx is not None and name in repro_idx.index:
            r = repro_idx.loc[name]
            pr_flag = int(r["pr_replicates"]) if "pr_replicates" in r.index else None
            map_val = (round(float(r["mAP"]), 4)
                       if "mAP" in r.index and not pd.isna(r["mAP"]) else None)

        row["pr_replicates"] = pr_flag
        row["mAP"]           = map_val

        # Derived confidence label
        sim_ok = (
            not np.isnan(repro_sims[i]) and repro_sims[i] >= threshold
        )
        pr_ok  = pr_flag == 1
        if pr_ok and sim_ok:
            confidence = "high"
        elif pr_ok or sim_ok:
            confidence = "medium"
        else:
            confidence = "low"
        row["node_confidence"] = confidence

        rows.append(row)

    return pd.DataFrame(rows)


# ── Interactive morphological map ─────────────────────────────────────────────

def plot_morphological_map(
    S_cpd: np.ndarray,
    compound_names: np.ndarray,
    edges: List[dict],
    df_nodes: pd.DataFrame,
    threshold: float,
    cohort: str,
    save_path: Path,
):
    """
    Interactive Plotly compound-level morphological map.

    Visual encoding
    ---------------
    Node size     : median_replicate_sim (larger = more reproducible)
                    NaN nodes use a fixed minimum size
    Node border   : solid thick  = high confidence
                    solid thin   = medium confidence
                    dashed proxy = low confidence (lighter color)
    Edge width    : proportional to similarity above threshold
    Edge color    : similarity intensity (light → dark)

    Color modes (dropdown buttons)
    --------------------------------
    1. By community   : Leiden/connected-component cluster
    2. By confidence  : high / medium / low reproducibility
    3. By VIII cluster: primary cluster from VIII_Subprofiles

    Hover info
    ----------
    Compound name, community, VIII cluster, median_replicate_sim,
    pr_replicates, mAP, confidence, degree.
    """
    # ── Layout ────────────────────────────────────────────────────────────────
    try:
        n_nodes  = len(compound_names)
        G_lay    = ig.Graph()
        G_lay.add_vertices(n_nodes)
        cname_list = list(compound_names)
        G_lay.add_edges([
            (cname_list.index(e["compound_i"]),
             cname_list.index(e["compound_j"]))
            for e in edges
        ])
        layout = G_lay.layout("fr", niter=500)
        pos    = {compound_names[i]: layout[i] for i in range(n_nodes)}
    except Exception:
        angles = np.linspace(0, 2 * np.pi, len(compound_names), endpoint=False)
        pos    = {n: [np.cos(a), np.sin(a)]
                  for n, a in zip(compound_names, angles)}

    node_idx = df_nodes.set_index(DEFAULT_LABEL_COL)

    # ── Edge trace ────────────────────────────────────────────────────────────
    sim_range = max(e["similarity"] for e in edges) - threshold if edges else 1.0
    edge_traces = []
    for e in edges:
        x0, y0 = pos[e["compound_i"]]
        x1, y1 = pos[e["compound_j"]]
        norm_sim = (e["similarity"] - threshold) / (sim_range + 1e-6)
        width    = 0.8 + 4.0 * norm_sim
        alpha    = 0.25 + 0.45 * norm_sim
        color    = f"rgba(100,100,180,{alpha:.2f})"
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="none",
            showlegend=False,
        ))

    # ── Node attributes ───────────────────────────────────────────────────────
    xs  = [pos[c][0] for c in compound_names]
    ys  = [pos[c][1] for c in compound_names]

    def node_size(name):
        sim = node_idx.loc[name, "median_replicate_sim"] if name in node_idx.index else None
        if sim is None or pd.isna(sim):
            return 10
        return int(10 + 16 * float(sim))

    def node_opacity(name):
        conf = node_idx.loc[name, "node_confidence"] if name in node_idx.index else "low"
        return {"high": 0.95, "medium": 0.75, "low": 0.45}.get(conf, 0.6)

    def border_width(name):
        conf = node_idx.loc[name, "node_confidence"] if name in node_idx.index else "low"
        return {"high": 2.5, "medium": 1.5, "low": 0.8}.get(conf, 1.0)

    sizes    = [node_size(c)    for c in compound_names]
    opacities= [node_opacity(c) for c in compound_names]
    borders  = [border_width(c) for c in compound_names]

    def hover(name):
        if name not in node_idx.index:
            return f"<b>{name}</b>"
        r = node_idx.loc[name]
        sim_str = f"{r['median_replicate_sim']:.3f}" if pd.notna(r.get('median_replicate_sim')) else "n/a"
        map_str = f"{r['mAP']:.3f}" if pd.notna(r.get('mAP')) else "n/a"
        pr_str  = str(int(r['pr_replicates'])) if pd.notna(r.get('pr_replicates')) else "n/a"
        return (
            f"<b>{name}</b><br>"
            f"Community: {r.get('community_id','?')}<br>"
            f"VIII cluster: {r.get('viii_primary_cluster','?')}<br>"
            f"Degree: {int(r.get('graph_degree', 0))}<br>"
            f"Median replicate sim: {sim_str}<br>"
            f"PR replicates: {pr_str}<br>"
            f"mAP: {map_str}<br>"
            f"Confidence: {r.get('node_confidence','?')}"
        )

    hover_texts = [hover(c) for c in compound_names]

    # ── Color palettes ────────────────────────────────────────────────────────
    communities  = [node_idx.loc[c, "community_ix"] if c in node_idx.index else -1
                    for c in compound_names]
    n_comm       = max(communities) + 2
    comm_palette = (px.colors.qualitative.Plotly * 5)[:n_comm]
    comm_colors  = [comm_palette[c % len(comm_palette)] for c in communities]

    conf_palette = {"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"}
    conf_colors  = [
        conf_palette.get(
            node_idx.loc[c, "node_confidence"] if c in node_idx.index else "low",
            "#aaaaaa"
        )
        for c in compound_names
    ]

    viii_clusters = [
        str(node_idx.loc[c, "viii_primary_cluster"])
        if c in node_idx.index else "?"
        for c in compound_names
    ]
    unique_viii = sorted(set(viii_clusters))
    viii_palette = (px.colors.qualitative.Dark24 * 3)
    viii_cmap   = {cl: viii_palette[i % len(viii_palette)]
                   for i, cl in enumerate(unique_viii)}
    viii_colors = [viii_cmap[cl] for cl in viii_clusters]

    texts = [c for c in compound_names]

    def make_node_trace(colors, trace_name):
        return go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=colors,
                opacity=opacities,
                line=dict(width=borders, color="white"),
            ),
            text=texts,
            textposition="top center",
            textfont=dict(size=8),
            hovertext=hover_texts,
            hoverinfo="text",
            name=trace_name,
            visible=True,
        )

    node_comm  = make_node_trace(comm_colors,  "By community")
    node_conf  = make_node_trace(conf_colors,  "By confidence")
    node_viii  = make_node_trace(viii_colors,  "By VIII cluster")

    # ── Confidence legend traces (dummy, for legend only) ─────────────────────
    legend_traces = []
    for label, color in conf_palette.items():
        legend_traces.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=color),
            name=f"Confidence: {label}",
            showlegend=True, visible=False,
        ))

    all_traces = edge_traces + [node_comm, node_conf, node_viii] + legend_traces

    n_edge_traces = len(edge_traces)
    def visibility(active_node: int):
        v = [True] * n_edge_traces  # edges always visible
        v += [i == active_node for i in range(3)]  # node traces
        # legend traces: only visible when confidence mode is active
        v += [active_node == 1] * len(legend_traces)
        return v

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=dict(
            text=f"{cohort} — Morphological Map  "
                 f"(threshold={threshold:.3f}, "
                 f"nodes={len(compound_names)}, "
                 f"edges={len(edges)})",
            font=dict(size=16),
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.01, y=1.09, xanchor="left",
            buttons=[
                dict(label="Color: Community",
                     method="update",
                     args=[{"visible": visibility(0)},
                           {"title.text": f"{cohort} — Morphological Map | Color: Community"}]),
                dict(label="Color: Confidence",
                     method="update",
                     args=[{"visible": visibility(1)},
                           {"title.text": f"{cohort} — Morphological Map | Color: Confidence"}]),
                dict(label="Color: VIII Cluster",
                     method="update",
                     args=[{"visible": visibility(2)},
                           {"title.text": f"{cohort} — Morphological Map | Color: VIII Cluster"}]),
            ],
        )],
        annotations=[
            dict(
                text=(
                    "Node size = replicate reproducibility  |  "
                    "Border width = confidence level  |  "
                    "Edge opacity = similarity strength"
                ),
                xref="paper", yref="paper",
                x=0.5, y=-0.03, showarrow=False,
                font=dict(size=10, color="gray"),
            )
        ],
        showlegend=True,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1200, height=900,
        plot_bgcolor="rgba(248,248,248,0.95)",
        legend=dict(font=dict(size=10), x=1.01, y=1.0),
    )

    # Start with community coloring
    fig.data[n_edge_traces + 1].visible = False  # conf
    fig.data[n_edge_traces + 2].visible = False  # viii
    for lt in legend_traces:
        lt.visible = False

    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    with open(save_path, "w") as f:
        f.write(html)
    print(f"  [Map] Saved → {save_path}")


# ── Main orchestrator ─────────────────────────────────────────────────────────

class MorphologicalMap:
    """
    Single-cohort morphological map from VIII + VII outputs.

    Parameters
    ----------
    cohort_root      : root directory of the cohort (contains Subprofiles/, Reproducibility/)
    saving_path      : output directory
    cohort           : cohort name (used as filename prefix)
    fdr              : FDR level for threshold computation
    recompute_threshold : if True, recompute FDR threshold from similarity matrix;
                         if False, read from community_assignments.csv
    n_null           : null permutations for FDR
    metric           : similarity metric (for display metadata only when not recomputing)
    secondary_threshold : soft membership cutoff
    prefix_filenames : if True, look for <cohort>_<filename> first
    random_state     : seed
    """

    def __init__(
        self,
        cohort_root: Path,
        saving_path: Path,
        cohort: str,
        fdr: float                  = DEFAULT_FDR,
        recompute_threshold: bool   = True,
        n_null: int                 = DEFAULT_NULL_REPS,
        metric: str                 = DEFAULT_METRIC,
        secondary_threshold: float  = DEFAULT_SEC_THRESHOLD,
        prefix_filenames: bool      = True,
        random_state: int           = DEFAULT_RANDOM_STATE,
    ):
        self.cohort_root         = cohort_root
        self.saving_path         = saving_path
        self.cohort              = cohort
        self.fdr                 = fdr
        self.recompute_threshold = recompute_threshold
        self.n_null              = n_null
        self.metric              = metric
        self.secondary_threshold = secondary_threshold
        self.prefix_filenames    = prefix_filenames
        self.random_state        = random_state
        self.rng                 = np.random.default_rng(random_state)

        self.saving_path.mkdir(parents=True, exist_ok=True)
        self._run()

    def _resolve(self, subdir: str, fname: str) -> Path:
        """
        Try paths in order:
        1. <cohort_root>/<subdir>/combined_norm/<cohort>_<fname>
        2. <cohort_root>/<subdir>/combined_red/<cohort>_<fname>
        3. <cohort_root>/<subdir>/<cohort>_<fname>  (original)
        4. <cohort_root>/<cohort>_<fname>            (flat fallback)
        """
        subdirs_to_try = [
            self.cohort_root / subdir / "combined_norm",
            self.cohort_root / subdir / "combined_red",
            self.cohort_root / subdir,
            self.cohort_root,
        ]
        for d in subdirs_to_try:
            if not d.exists():
                continue
            candidate = d / f"{self.cohort}_{fname}"
            if candidate.exists():
                return candidate
            candidate = d / fname
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not find '{fname}' (or '{self.cohort}_{fname}') "
            f"in any of: {[str(d) for d in subdirs_to_try]}. "
            f"Make sure VIII_Subprofiles and VII_Reproducibility have been run."
        )

    def _run(self):
        prefix = self.saving_path / self.cohort

        # ── 1. Load VIII outputs ──────────────────────────────────────────────
        print("\n── Step 1: Loading VIII outputs ─────────────────────────────")

        S_path = self._resolve("Subprofiles", FNAME_SIMILARITY)
        print(f"  Similarity matrix: {S_path}")
        df_sim = pd.read_parquet(S_path)

        # VIII format: first column is 'compound', remaining are rep_0, rep_1, ...
        if "compound" in df_sim.columns:
            rep_labels = np.array(df_sim["compound"])
            S_rep      = df_sim.drop(columns=["compound"]).values.astype(np.float32)
        else:
            # Legacy format: index contains replicate labels
            rep_labels = np.array(df_sim.index)
            S_rep      = df_sim.values.astype(np.float32)
        compound_names = np.unique(rep_labels)
        print(f"  Replicate matrix: {S_rep.shape}  "
              f"({len(compound_names)} unique compounds)")

        df_assignments = pd.read_csv(
            self._resolve("Subprofiles", FNAME_ASSIGNMENTS)
        )
        print(f"  Assignments: {len(df_assignments)} rows")

        try:
            df_subprofiles = pd.read_parquet(
                self._resolve("Subprofiles", FNAME_SUBPROFILES)
            )
            print(f"  Subprofiles: {len(df_subprofiles)} clusters")
        except FileNotFoundError:
            df_subprofiles = None
            print("  [WARNING] subprofiles.parquet not found — "
                  "soft membership scoring skipped")

        # ── 2. Load VII reproducibility ───────────────────────────────────────
        print("\n── Step 2: Loading VII reproducibility ──────────────────────")
        try:
            # Reproducibility está al mismo nivel que Subprofiles, no dentro de cohort_root
            repro_root = self.cohort_root.parent.parent
            repro_candidates = [
                repro_root / "Reproducibility" / "combined_norm" / f"{self.cohort}_reproducibility_report.csv",
                repro_root/ "Reproducibility" / "combined_red"  / f"{self.cohort}_reproducibility_report.csv"
            ]
            repro_path = next((p for p in repro_candidates if p.exists()), None)
            if repro_path is None:
                raise FileNotFoundError(
                    f"reproducibility_report.csv not found. Tried:\n" +
                    "\n".join(f"  {p}" for p in repro_candidates)
                )
            df_repro = pd.read_csv(repro_path)
            print(f"  Reproducibility report: {repro_path}")
            print(f"  {len(df_repro)} compounds loaded")
        except FileNotFoundError as e:
            df_repro = None
            print(f"  [WARNING] {e}")

        # ── 3. Aggregate to compound level ────────────────────────────────────
        print("\n── Step 3: Aggregating to compound level ─────────────────────")
        S_cpd, repro_sims = aggregate_similarity_to_compounds(
            S_rep, rep_labels, compound_names
        )
        print(f"  Compound matrix: {S_cpd.shape}")
        n_with_repro = int((~np.isnan(repro_sims)).sum())
        print(f"  Compounds with ≥2 replicates: {n_with_repro}/{len(compound_names)}")

        # ── 4. Similarity threshold ───────────────────────────────────────────
        print("\n── Step 4: Similarity threshold ─────────────────────────────")
        if self.recompute_threshold:
            print("  Recomputing FDR threshold from similarity matrix...")
            threshold = compute_fdr_threshold(
                S_rep, rep_labels,
                fdr=self.fdr, n_null=self.n_null, rng=self.rng,
            )
        else:
            # Read from assignments (stored by VIII)
            if "fdr_threshold" in df_assignments.columns:
                threshold = float(df_assignments["fdr_threshold"].iloc[0])
                print(f"  Using stored FDR threshold: {threshold:.4f}")
            else:
                warnings.warn(
                    "fdr_threshold column not found in assignments. "
                    "Recomputing from matrix."
                )
                threshold = compute_fdr_threshold(
                    S_rep, rep_labels,
                    fdr=self.fdr, n_null=self.n_null, rng=self.rng,
                )

        # ── 5. Build compound-level graph ─────────────────────────────────────
        print("\n── Step 5: Building compound-level graph ─────────────────────")
        G, edges = build_compound_graph(S_cpd, compound_names, threshold)

        df_edges = pd.DataFrame(edges)
        df_edges.to_csv(f"{prefix}_compound_graph_edges.csv", index=False)
        print(f"  Edges saved → {prefix}_compound_graph_edges.csv")

        # ── 6. Community detection ────────────────────────────────────────────
        print("\n── Step 6: Community detection ──────────────────────────────")
        community_map = detect_communities(
            G, compound_names, edges,
            random_state=self.random_state,
        )

        # ── 7. Node attribute table ───────────────────────────────────────────
        print("\n── Step 7: Building node attribute table ─────────────────────")
        df_nodes = build_node_table(
            compound_names  = compound_names,
            S_cpd           = S_cpd,
            repro_sims      = repro_sims,
            community_map   = community_map,
            df_assignments  = df_assignments,
            df_repro        = df_repro,
            threshold       = threshold,
            cohort          = self.cohort,
        )
        df_nodes.to_csv(f"{prefix}_compound_nodes.csv", index=False)
        print(f"  Node table saved → {prefix}_compound_nodes.csv")
        print(f"  Confidence breakdown: "
              f"{df_nodes['node_confidence'].value_counts().to_dict()}")

        # ── 8. Interactive morphological map ──────────────────────────────────
        print("\n── Step 8: Generating morphological map ──────────────────────")
        plot_morphological_map(
            S_cpd          = S_cpd,
            compound_names = compound_names,
            edges          = edges,
            df_nodes       = df_nodes,
            threshold      = threshold,
            cohort         = self.cohort,
            save_path      = Path(f"{prefix}_morphological_map.html"),
        )

        print(f"\n{'─'*55}")
        print(f"  Done. Outputs in {self.saving_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-cohort morphological map from VIII+VII outputs."
    )
    p.add_argument("-i", "--input",  required=True, type=Path, dest="cohort_root",
                   help="Cohort root directory (contains Subprofiles/ and Reproducibility/).")
    p.add_argument("-o", "--output", required=True, type=Path, dest="saving_path")
    p.add_argument("-c", "--cohort", required=True, type=str)
    p.add_argument("--fdr",          type=float, default=DEFAULT_FDR,
                   help=f"FDR level for threshold (default {DEFAULT_FDR}). "
                        "Only used when --recompute-threshold is set.")
    p.add_argument("--recompute-threshold", action="store_true",
                   dest="recompute_threshold",
                   help="Recompute FDR threshold from similarity matrix "
                        "(default: read from community_assignments.csv).")
    p.add_argument("--null-reps",    type=int,   default=DEFAULT_NULL_REPS,
                   dest="n_null")
    p.add_argument("--metric",       type=str,   default=DEFAULT_METRIC,
                   choices=SUPPORTED_METRICS,
                   help="Similarity metric label (informational when not recomputing).")
    p.add_argument("--secondary-threshold", type=float, default=DEFAULT_SEC_THRESHOLD,
                   dest="secondary_threshold")
    p.add_argument("--prefix-filenames", action="store_true", default=True,
                   dest="prefix_filenames",
                   help="Look for <cohort>_<filename> instead of <filename> (default: True).")
    return p.parse_args()


def main():
    args = parse_args()
    args.saving_path.mkdir(parents=True, exist_ok=True)
    MorphologicalMap(
        cohort_root         = args.cohort_root,
        saving_path         = args.saving_path,
        cohort              = args.cohort,
        fdr                 = args.fdr,
        recompute_threshold = args.recompute_threshold,
        n_null              = args.n_null,
        metric              = args.metric,
        secondary_threshold = args.secondary_threshold,
        prefix_filenames    = args.prefix_filenames,
    )


if __name__ == "__main__":
    main()
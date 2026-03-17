import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import argparse

from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
import plotly.graph_objects as go

DEFAULT_LABEL_COL    = "Metadata_Perturbation"
DEFAULT_RANDOM_STATE = 42


# ── Metric functions ────────────────────────────────────────────────────────
# Each metric function has the signature:
#   fn(X_scaled, labels, label_names) -> pl.DataFrame
# and must return a DataFrame with at least a "Metadata_Compound" column.
# A global summary scalar can optionally be printed inside the function.

def metric_cosine_similarity(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
) -> pl.DataFrame:
    """Mean / std cosine similarity between replicas per compound."""
    cos_sim = cosine_similarity(X_scaled)

    same_label  = labels[:, None] == labels[None, :]
    off_diag    = ~np.eye(len(labels), dtype=bool)
    mean_global = cos_sim[same_label & off_diag].mean()
    print(f"  [cosine]   mean (all replicas): {mean_global:.4f}")

    results = []
    for compound in label_names:
        idx = np.where(labels == compound)[0]
        if len(idx) < 2:
            continue
        sub      = cos_sim[np.ix_(idx, idx)]
        sim_vals = sub[~np.eye(len(idx), dtype=bool)]
        results.append({
            "Metadata_Compound": compound,
            "mean_cosine":       float(sim_vals.mean()),
            "std_cosine":        float(sim_vals.std()),
        })

    return pl.DataFrame(results)


def metric_spearman_correlation(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
) -> pl.DataFrame:
    """Mean Spearman correlation between replicas per compound."""
    results = []
    all_corrs = []

    for compound in label_names:
        idx = np.where(labels == compound)[0]
        if len(idx) < 2:
            continue

        corrs = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                r, _ = spearmanr(X_scaled[idx[i]], X_scaled[idx[j]])
                corrs.append(r)

        all_corrs.extend(corrs)
        results.append({
            "Metadata_Compound":  compound,
            "mean_spearman":      float(np.mean(corrs)),
            "std_spearman":       float(np.std(corrs)),
        })

    if all_corrs:
        print(f"  [spearman] mean (all replicas): {np.mean(all_corrs):.4f}")

    return pl.DataFrame(results)


# ── Main class ───────────────────────────────────────────────────────────────

class Clustering:
    """
    Loads morphological profiles, runs PCA / UMAP / t-SNE embeddings,
    saves figures, computes silhouette scores, and a configurable set
    of replica similarity metrics.

    Parameters
    ----------
    metric_fns : List of callables, optional
        Functions with signature (X_scaled, labels, label_names) -> pl.DataFrame.
        Each must return a DataFrame with a "Metadata_Compound" column.
        Defaults to [metric_cosine_similarity, metric_spearman_correlation].
    """

    DEFAULT_METRICS: List[Callable] = [
        metric_cosine_similarity,
        metric_spearman_correlation,
    ]

    def __init__(
        self,
        path_profiles: Path,
        saving_path: Path,
        cohort: str,
        label_col: str        = DEFAULT_LABEL_COL,
        drop_cols: tuple      = (DEFAULT_LABEL_COL,),
        random_state: int     = DEFAULT_RANDOM_STATE,
        metric_fns: Optional[Callable] = None,
    ):
        self.saving_path  = saving_path
        self.cohort = cohort
        self.label_col    = label_col
        self.drop_cols    = drop_cols
        self.random_state = random_state
        self.metric_fns   = metric_fns if metric_fns is not None else self.DEFAULT_METRICS

        self.saving_path.mkdir(parents=True, exist_ok=True)

        self.X_scaled, self.labels = self._load_and_prepare(path_profiles)
        self.label_codes, self.label_names, self.cmap = self._encode_labels()

        self.X_pca, self.X_umap, self.X_tsne = self._run_embeddings()
        self._save_figures()
        self._save_figures_3d()

        self.df_report = self._build_report()

    # ── Data loading ─────────────────────────────────────────────────────

    def _load_and_prepare(self, path_profiles: Path):
        df     = pl.read_csv(path_profiles)
        labels = df[self.label_col].to_numpy()
        X      = (df
                  .drop(list(self.drop_cols))
                  .select(pl.col(pl.Float64, pl.Float32, pl.Int64, pl.Int32))
                  .to_numpy()
                  .astype(float))
        return StandardScaler().fit_transform(X), labels

    def _encode_labels(self):
        label_names, label_codes = np.unique(self.labels, return_inverse=True)
        cmap = plt.get_cmap("tab20", len(label_names))
        return label_codes, label_names, cmap

    # ── Embeddings ───────────────────────────────────────────────────────

    def _run_embeddings(self):
        MIN_SAMPLES = 10
        n = self.X_scaled.shape[0]
        X_pca  = self._run_pca()
        X_umap = self._run_umap() if n >= MIN_SAMPLES else None
        X_tsne = self._run_tsne() if n >= MIN_SAMPLES else None

        if X_umap is None: print(f"[WARNING] UMAP omitido: solo {n} muestras")
        if X_tsne is None: print(f"[WARNING] t-SNE omitido: solo {n} muestras")

        return X_pca, X_umap, X_tsne

    def _run_pca(self):
        return PCA(n_components=3, random_state=self.random_state).fit_transform(self.X_scaled)

    def _run_umap(self):
        return umap.UMAP(
            n_neighbors=20, min_dist=0.5, n_components=3,
            random_state=self.random_state,
        ).fit_transform(self.X_scaled)

    def _run_tsne(self):
        n = self.X_scaled.shape[0]
        return TSNE(
            n_components=3, perplexity=min(30, n - 1),
            learning_rate="auto", init="pca",
            random_state=self.random_state,
        ).fit_transform(self.X_scaled)

    # ── Figures ──────────────────────────────────────────────────────────

    def _plot_embedding(self, X: np.ndarray, title: str, filename: Path):
        fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
        for i, label in enumerate(self.label_names):
            mask = self.label_codes == i
            ax.scatter(X[mask, 0], X[mask, 1], color=self.cmap(i),
                       s=40, alpha=0.9, label=label)

        ax.legend(title="Perturbation", bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=7, title_fontsize=8, framealpha=0.7)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.tight_layout()
        fig.savefig(filename, transparent=True, bbox_inches="tight")
        plt.close(fig)

    def _save_figures(self):
        embeddings = {
            "PCA":   (self.X_pca,  "pca_embedding.png"),
            "UMAP":  (self.X_umap, "umap_embedding.png"),
            "t-SNE": (self.X_tsne, "tsne_embedding.png"),
        }
        n_labels  = len(np.unique(self.label_codes))
        for title, (X, fname) in embeddings.items():
            if X is None:
                continue
            self._plot_embedding(X, title, self.saving_path / fname)
            n_samples = X.shape[0]
            if 2 <= n_labels <= n_samples - 1:
                score = silhouette_score(X, self.label_codes)
                print(f"  Silhouette {title:<6}: {score:.3f}")
            else:
                print(f"  Silhouette {title:<6}: omitido (n_labels={n_labels}, n_samples={n_samples})")

    # ── 3D Interactive Figures ────────────────────────────────────────────

    def _plot_embedding_3d_interactive(self, emb: np.ndarray, title: str, fname: Path):
        """Genera figura 3D interactiva con Plotly y la guarda como HTML autocontenido."""
        fig = go.Figure()

        for i, name in enumerate(self.label_names):
            mask  = self.label_codes == i
            color = (
                f"rgb({int(self.cmap(i)[0]*255)},"
                f"{int(self.cmap(i)[1]*255)},"
                f"{int(self.cmap(i)[2]*255)})"
            )
            fig.add_trace(go.Scatter3d(
                x=emb[mask, 0],
                y=emb[mask, 1],
                z=emb[mask, 2],
                mode="markers",
                name=name,
                marker=dict(size=4, color=color, opacity=0.85),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}"
                    "<extra></extra>"
                ),
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            scene=dict(
                xaxis_title="Dim 1",
                yaxis_title="Dim 2",
                zaxis_title="Dim 3",
                bgcolor="rgba(245,245,245,0.8)",
            ),
            legend=dict(title="Perturbation", itemsizing="constant", font=dict(size=10)),
            margin=dict(l=0, r=0, t=40, b=0),
            width=900, height=700,
        )

        html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
        with open(fname, "w") as f:
            f.write(html_str)
        print(f"  3D interactive saved → {fname}")

    def _save_figures_3d(self):
        embeddings_3d = {
            "PCA 3D":   (self.X_pca,  "pca_3d.html"),
            "UMAP 3D":  (self.X_umap, "umap_3d.html"),
            "t-SNE 3D": (self.X_tsne, "tsne_3d.html"),
        }
        n_labels = len(np.unique(self.label_codes))
        for title, (X, fname) in embeddings_3d.items():
            if X is None:
                continue
            self._plot_embedding_3d_interactive(X, title, self.saving_path / f"{self.cohort}_{fname}")
            n_samples = X.shape[0]
            if 2 <= n_labels <= n_samples - 1:
                score = silhouette_score(X, self.label_codes)
                print(f"  Silhouette {title:<6}: {score:.3f}")

    # ── Metrics report ───────────────────────────────────────────────────

    def _build_report(self) -> pl.DataFrame:
        """
        Runs all metric_fns and joins their results into a single DataFrame
        keyed on Metadata_Compound, sorted by mean_cosine if present.
        """
        base = pl.DataFrame({"Metadata_Compound": list(self.label_names)})

        for fn in self.metric_fns:
            print(f"\nRunning metric: {fn.__name__}")
            result = fn(self.X_scaled, self.labels, self.label_names)
            if "Metadata_Compound" not in result.columns:
                raise ValueError(
                    f"Metric '{fn.__name__}' must return a DataFrame with 'Metadata_Compound' column."
                )
            base = base.join(result, on="Metadata_Compound", how="left")

        sort_col = "mean_cosine" if "mean_cosine" in base.columns else base.columns[1]
        return base.sort(sort_col)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clustering and embedding of morphological profiles."
    )
    parser.add_argument("-i", "--input",  required=True, type=Path,
                        dest="input_dir",   help="Directory containing input CSV profiles.")
    parser.add_argument("-o", "--output", required=True, type=Path,
                        dest="saving_path", help="Directory for output figures and reports.")
    parser.add_argument("-c", "--cohort", required=True, type=str,
                        help="Name of the analyzed cohort.")
    return parser.parse_args()


def _process_dataset(tmp_csv: Path, saving_path: Path, tag: str, cohort: str):
    print(f"\n{'─'*50}\nProcessing: {tag}\n{'─'*50}")
    cl = Clustering(tmp_csv, saving_path, cohort)
    out = saving_path / "similarity_report.csv"
    cl.df_report.write_csv(out)
    print(f"Report saved → {out}")


def main():
    args = parse_args()
    args.saving_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Reduced profiles (*_red*.csv) ─────────────────────────────────
    red_files = sorted(args.input_dir.glob("**/*_red*.csv"))
    if red_files:
        print(f"\nFound {len(red_files)} reduced profile file(s):")
        dfs = []
        for p in red_files:
            print(f"  Loading: {p.name}")
            dfs.append(pl.read_csv(p))
        combined = pl.concat(dfs, how="diagonal")
        print(f"Combined reduced shape: {combined.shape}")
        tmp = args.saving_path / "combined_red.csv"
        combined.write_csv(tmp)
        _process_dataset(tmp, args.saving_path / "combined_red", "combined reduced profiles", args.cohort)
    else:
        print(f"[WARNING] No reduced files ('_red') found in {args.input_dir}")

    # ── 2. Cohort norm profiles (*cohort_norm*.csv) ───────────────────────
    norm_files = sorted(args.input_dir.glob("**/*cohort_norm*.csv"))
    if norm_files:
        print(f"\nFound {len(norm_files)} cohort norm file(s):")
        dfs = []
        for p in norm_files:
            print(f"  Loading: {p.name}")
            dfs.append(pl.read_csv(p))
        combined = pl.concat(dfs, how="diagonal")
        print(f"Combined norm shape: {combined.shape}")
        tmp = args.saving_path / "combined_norm.csv"
        combined.write_csv(tmp)
        _process_dataset(tmp, args.saving_path / "combined_norm", "combined norm profiles (all wells)", args.cohort)
    else:
        print(f"[WARNING] No cohort norm files ('cohort_norm') found in {args.input_dir}")


if __name__ == "__main__":
    main()
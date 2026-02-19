import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import argparse

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap

DEFAULT_LABEL_COL  = "Metadata_Perturbation"
DEFAULT_DROP_COLS  = ("Metadata_Perturbation", "Metadata_Plate")
DEFAULT_RANDOM_STATE = 42

class Clustering:
    """
    Loads morphological profiles, runs PCA / UMAP / t-SNE embeddings,
    saves figures, computes silhouette scores, and replica cosine similarity.
    """

    def __init__(
        self,
        path_profiles: Path,
        saving_path: Path,
        label_col: str = DEFAULT_LABEL_COL,
        drop_cols: tuple = DEFAULT_DROP_COLS,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        self.saving_path   = saving_path
        self.label_col     = label_col
        self.drop_cols     = drop_cols
        self.random_state  = random_state

        self.saving_path.mkdir(parents=True, exist_ok=True)

        self.X_scaled, self.labels = self._load_and_prepare(path_profiles)
        self.label_codes, self.label_names, self.cmap = self._encode_labels()

        self.X_pca, self.X_umap, self.X_tsne = self._run_embeddings()

        self._save_figures()
        self.df_cosine = self._replica_similarity()

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

    def _run_embeddings(self):
        MIN_SAMPLES_TSNE = 10
        MIN_SAMPLES_UMAP = 10
        n = self.X_scaled.shape[0]
        X_pca  = self._run_pca()
        X_umap = self._run_umap() if n >= MIN_SAMPLES_UMAP else None
        X_tsne = self._run_tsne() if n >= MIN_SAMPLES_TSNE else None

        if X_umap is None: print(f"[WARNING] UMAP omitido: solo {n} muestras")
        if X_tsne is None: print(f"[WARNING] t-SNE omitido: solo {n} muestras")

        return X_pca, X_umap, X_tsne

    def _run_pca(self):
        return PCA(n_components=2, random_state=self.random_state).fit_transform(self.X_scaled)

    def _run_umap(self):
        return umap.UMAP(
            n_neighbors=5,
            min_dist=0.1,
            n_components=2,
            random_state=self.random_state,
        ).fit_transform(self.X_scaled)

    def _run_tsne(self):
        return TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=self.random_state,
        ).fit_transform(self.X_scaled)

    def _plot_embedding(self, X: np.ndarray, title: str, filename: Path):
        fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

        for i, label in enumerate(self.label_names):
            mask = self.label_codes == i
            ax.scatter(
                X[mask, 0], X[mask, 1],
                color=self.cmap(i),
                s=40, alpha=0.9,
                label=label
            )
            cx, cy = X[mask, 0].mean(), X[mask, 1].mean()
            ax.annotate(
                label, (cx, cy),
                fontsize=7,
                fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5, ec="none")
            )

        ax.legend(
            title="Perturbation",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=7,
            title_fontsize=8,
            framealpha=0.7
        )
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
        for title, (X, fname) in embeddings.items():
            if X is None:
                continue
            self._plot_embedding(X, title, self.saving_path / fname)
            score = silhouette_score(X, self.label_codes)
            print(f"Silhouette {title:<6}: {score:.3f}")

    def _replica_similarity(self) -> pl.DataFrame:
        cos_sim = cosine_similarity(self.X_scaled)

        same_label = self.labels[:, None] == self.labels[None, :]
        off_diag   = ~np.eye(len(self.labels), dtype=bool)
        mean_global = cos_sim[same_label & off_diag].mean()
        print(f"Mean cosine similarity (all replicas): {mean_global:.4f}")
        results = []
        for compound in np.unique(self.labels):
            idx = np.where(self.labels == compound)[0]
            if len(idx) < 2:
                continue

            sub      = cos_sim[np.ix_(idx, idx)]
            sim_vals = sub[~np.eye(len(idx), dtype=bool)]
            results.append({
                "Metadata_Compound": compound,
                "mean_cosine":       float(sim_vals.mean()),
                "std_cosine":        float(sim_vals.std()),
            })

        return pl.DataFrame(results).sort("mean_cosine")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clustering and embedding of morphological profiles."
    )
    parser.add_argument("-i", "--input",  required=True, type=Path,
                        dest="input_dir", help="Directory containing input CSV profiles.")
    parser.add_argument("-o", "--output", required=True, type=Path,
                        dest="saving_path", help="Directory for output figures.")
    return parser.parse_args()


def main():
    args     = parse_args()
    csv_files = sorted(args.input_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return

    for csv_path in csv_files:
        stem        = csv_path.stem
        saving_path = args.saving_path / stem

        print(f"\n{'─' * 50}")
        print(f"Processing: {csv_path.name}")
        print(f"{'─' * 50}")

        clustering = Clustering(csv_path, saving_path)
        output_csv = saving_path / f"{stem}_cosine_similarity.csv"
        clustering.df_cosine.write_csv(output_csv)
        print(f"Cosine similarity table saved to {output_csv}")


if __name__ == "__main__":
    main()
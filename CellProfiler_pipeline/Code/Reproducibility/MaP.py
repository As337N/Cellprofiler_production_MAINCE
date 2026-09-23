"""
map_polars.py
=============
Evaluación de reproducibilidad de réplicas (phenotypic activity) y
consistencia (phenotypic consistency) usando mean Average Precision (mAP),
siguiendo el marco de Kalinin et al. 2025 (Nat. Commun. 16:5181) / copairs.

- Entrada: polars.DataFrame (metadata + features en el mismo frame).
- Distancias: coseno, con aceleración CUDA vía CuPy si está disponible;
  fallback transparente a NumPy.
- AP: replica exactamente el estimador no interpolado de copairs
  (ap_contiguous), con test de permutación cacheado para los p-values.

El bloque de diseño (block design) se especifica igual que en copairs:
    pos_sameby / pos_diffby  -> definen qué pares son "correct matches"
    neg_sameby / neg_diffby  -> definen qué pares son "incorrect matches"
"""

from __future__ import annotations
from pathlib import Path
from itertools import combinations
import numpy as np
import polars as pl

# ----------------------------------------------------------------------
# Backend: CuPy (GPU) si existe, si no NumPy (CPU)
# ----------------------------------------------------------------------
try:
    import cupy as _cp
    _GPU = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _cp = None
    _GPU = False


def _xp():
    """Devuelve el módulo de arreglos activo (cupy o numpy)."""
    return _cp if _GPU else np


def _to_device(a: np.ndarray):
    return _cp.asarray(a) if _GPU else a


def _to_host(a):
    return _cp.asnumpy(a) if (_GPU and isinstance(a, _cp.ndarray)) else np.asarray(a)


# ----------------------------------------------------------------------
# 1. Distancia coseno (matriz completa) en el backend activo
# ----------------------------------------------------------------------
def cosine_distance_matrix(feats: np.ndarray) -> np.ndarray:
    """
    Matriz NxN de distancias coseno (1 - similitud). El cálculo pesado
    (feats @ feats.T) corre en GPU si CuPy está disponible.
    """
    xp = _xp()
    x = _to_device(feats.astype(np.float32))
    norm = xp.linalg.norm(x, axis=1, keepdims=True)
    norm = xp.where(norm == 0, 1, norm)
    xn = x / norm
    sim = xn @ xn.T
    sim = xp.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    return dist  # queda en device; se indexa allí y se baja al final


# ----------------------------------------------------------------------
# 2. Construcción de pares positivos / negativos por metadata
# ----------------------------------------------------------------------
def _build_pairs(meta: pl.DataFrame, sameby, diffby):
    """
    Devuelve un set de pares (i, j) i<j que comparten TODAS las columnas de
    `sameby` y difieren en TODAS las de `diffby`. Estrategia: agrupar por
    `sameby` (barato en polars) y filtrar por diffby dentro de cada grupo.
    Para diffby global (sin sameby) se compara por bloques de grupo.
    """
    n = meta.height
    idx = np.arange(n)
    sameby = list(sameby or [])
    diffby = list(diffby or [])

    pairs = set()

    if sameby:
        # agrupamos filas por combinación de valores sameby
        groups = (
            meta.with_row_index("__i")
            .group_by(sameby)
            .agg(pl.col("__i"))
        )
        for row in groups.iter_rows(named=True):
            members = np.asarray(row["__i"])
            if members.size < 2:
                continue
            for a, b in combinations(members.tolist(), 2):
                pairs.add((a, b))
    else:
        # sin sameby: todos los pares candidatos (se filtra por diffby abajo)
        for a, b in combinations(idx.tolist(), 2):
            pairs.add((a, b))

    if diffby:
        cols = {c: meta[c].to_numpy() for c in diffby}
        filtered = set()
        for a, b in pairs:
            if all(cols[c][a] != cols[c][b] for c in diffby):
                filtered.add((a, b))
        pairs = filtered

    return pairs


# ----------------------------------------------------------------------
# 3. AP no interpolado (equivalente a copairs.compute.ap_contiguous)
# ----------------------------------------------------------------------
def _average_precision(rel_k: np.ndarray) -> np.ndarray:
    """rel_k: matriz binaria (n_queries x n_ranked) ordenada por relevancia."""
    tp = np.cumsum(rel_k, axis=1)
    num_pos = tp[:, -1]
    k = np.arange(1, rel_k.shape[1] + 1)
    pr_k = tp / k
    with np.errstate(invalid="ignore", divide="ignore"):
        ap = (pr_k * rel_k).sum(axis=1) / num_pos
    return ap


# ----------------------------------------------------------------------
# 4. Test de permutación para p-values (cacheado por (num_pos, total))
# ----------------------------------------------------------------------
def _random_binary_matrix(n, m, k, rng):
    mat = np.zeros((n, m), dtype=np.int8)
    mat[:, :k] = 1
    rng.permuted(mat, axis=1, out=mat)
    return mat


def _null_ap(num_perm, num_pos, total, seed):
    rng = np.random.default_rng(seed)
    rel = _random_binary_matrix(num_perm, total, num_pos, rng)
    return _average_precision(rel)


def _p_values(ap_scores, confs, null_size, seed):
    """confs: array (n, 2) con [num_pos, total] por query. p = fracción del
    nulo >= AP observado (convención permutación del paper)."""
    p = np.empty(len(ap_scores))
    cache = {}
    rng = np.random.default_rng(seed)
    for i, (ap, (num_pos, total)) in enumerate(zip(ap_scores, confs)):
        key = (int(num_pos), int(total))
        if key not in cache:
            cache[key] = _null_ap(null_size, key[0], key[1], rng.integers(1 << 31))
        null = cache[key]
        p[i] = (1 + np.sum(null >= ap)) / (1 + len(null))
    return p


def _bh_correct(pvals):
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
# 5. API principal
# ----------------------------------------------------------------------
def average_precision(
    df: pl.DataFrame,
    feature_cols: list[str],
    pos_sameby: list[str],
    pos_diffby: list[str] | None = None,
    neg_sameby: list[str] | None = None,
    neg_diffby: list[str] | None = None,
) -> pl.DataFrame:
    """
    Calcula AP por perfil (query). Positivos = réplicas de la misma
    perturbación; negativos = controles (definido por la metadata).

    Ejemplo phenotypic activity:
        pos_sameby=['Metadata_pert'], pos_diffby=['Metadata_well'] (opcional),
        neg_sameby=[], neg_diffby=['Metadata_pert']   # controles vs pert
    """
    meta = df.select([c for c in df.columns if c not in feature_cols])
    feats = df.select(feature_cols).to_numpy()
    n = df.height

    dist = cosine_distance_matrix(feats)  # en device

    pos = _build_pairs(meta, pos_sameby, pos_diffby)
    neg = _build_pairs(meta, neg_sameby, neg_diffby)

    # relevancia por query: 1 = par positivo, 0 = par negativo
    pos_map = {i: [] for i in range(n)}
    neg_map = {i: [] for i in range(n)}
    for a, b in pos:
        pos_map[a].append(b); pos_map[b].append(a)
    for a, b in neg:
        neg_map[a].append(b); neg_map[b].append(a)

    ap_scores, confs, keep = [], [], []
    for q in range(n):
        p_idx = np.asarray(pos_map[q], dtype=int)
        ngt = np.asarray(neg_map[q], dtype=int)
        if p_idx.size == 0 or ngt.size == 0:
            continue
        cand = np.concatenate([p_idx, ngt])
        rel = np.concatenate([np.ones(p_idx.size, np.int8),
                              np.zeros(ngt.size, np.int8)])
        d = _to_host(dist[q, _to_device(cand)]) if _GPU else dist[q, cand]
        order = np.argsort(d, kind="stable")  # menor distancia primero
        rel_sorted = rel[order][None, :]
        ap = _average_precision(rel_sorted)[0]
        ap_scores.append(ap)
        confs.append((p_idx.size, cand.size))
        keep.append(q)

    out = meta[keep].with_columns([
        pl.Series("average_precision", ap_scores),
        pl.Series("n_pos_pairs", [c[0] for c in confs]),
        pl.Series("n_total_pairs", [c[1] for c in confs]),
    ])
    out = out.with_columns(pl.Series("__conf", list(map(list, confs))))
    return out


def mean_average_precision(
    ap_df: pl.DataFrame,
    sameby: list[str],
    null_size: int = 10000,
    seed: int = 0,
    threshold: float = 0.05,
) -> pl.DataFrame:
    """
    Agrega AP -> mAP por grupo `sameby`, calcula p-value por permutación
    y corrección BH. `percent retrieved` = fracción con p_corr < threshold.
    """
    confs = np.asarray(ap_df["__conf"].to_list())
    ap = ap_df["average_precision"].to_numpy()

    grouped = (
        ap_df.group_by(sameby)
        .agg([
            pl.col("average_precision").mean().alias("mean_average_precision"),
            pl.col("average_precision").count().alias("n"),
            pl.col("__conf").first().alias("conf"),
        ])
    )
    g_ap = grouped["mean_average_precision"].to_numpy()
    g_conf = np.asarray(grouped["conf"].to_list())
    pvals = _p_values(g_ap, g_conf, null_size, seed)
    p_corr = _bh_correct(pvals)

    return grouped.drop("conf").with_columns([
        pl.Series("p_value", pvals),
        pl.Series("corrected_p_value", p_corr),
        pl.Series("below_threshold", p_corr < threshold),
    ])


if __name__ == "__main__":
    print(f"Backend GPU activo: {_GPU}")
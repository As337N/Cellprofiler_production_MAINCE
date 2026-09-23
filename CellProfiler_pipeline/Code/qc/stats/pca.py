"""
qc/stats/pca.py — PCA over well-level features to flag deviant plates.

Builds one PCA fitted on all wells (every numeric QC feature, standardised), and
projects both wells and per-plate centroids into the same 2-D space. A plate
whose centroid sits far from the cluster is a candidate outlier — a systematic
shift affecting the whole plate (illumination, incubation, a staining batch).

Design (agreed):
  · One point per well; each plate drawn as the centroid of its wells, so wells
    and plates share identical axes (comparable).
  · All numeric features per well, z-scored (a slope ~-2 and a count ~2000 must
    not let magnitude dominate the components).
  · NaNs imputed with each feature's median (PCA admits no NaN; median is a
    conservative fill that keeps every well and feature).
  · numpy-only (SVD). No sklearn dependency; SVD-PCA is mathematically identical.

Returns a payload-ready dict; the pipeline injects it under window.__QC__.pca.
"""
from __future__ import annotations

import numpy as np

from qc import profiles

# Minimum wells and plates below which a PCA is not meaningful.
_MIN_WELLS = 6
_MIN_PLATES = 3


def _feature_columns(plates_data: list, profile: profiles.DatasetProfile) -> list:
    """Ordered, de-duplicated list of numeric feature columns present in the data.

    Union across all wells so a column missing from one plate still participates
    (its absent values become NaN → imputed). Metadata/compound fields excluded.
    """
    channels = list(profile.channels)
    candidate = (
        [c for cols in profile.metric_cols().values() for c in cols] +
        [f"ImageQuality_PercentMaximal_{ch}" for ch in channels] +
        [f"Intensity_Percentile_10_{ch}" for ch in channels] +
        list(profiles.COUNT_COLS.values()) + [profiles.AREA_COL]
    )
    seen, present = set(), set()
    for pd_ in plates_data:
        for m in pd_["plate_qc"].values():
            present.update(k for k, v in m.items() if isinstance(v, (int, float)))
    out = []
    for c in candidate:
        if c in present and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def compute_pca(plates_data: list, profile: profiles.DatasetProfile) -> dict:
    """Fit PCA on wells; project wells + plate centroids. See module docstring.

    Returns {} when there aren't enough wells/plates/features for a meaningful
    projection (the frontend then hides the section).
    """
    features = _feature_columns(plates_data, profile)
    if len(features) < 2:
        return {}

    # ── Build the well matrix [n_wells × n_features] with row provenance ──────
    rows, well_plate, well_id = [], [], []
    for pd_ in plates_data:
        pname = pd_["name"]
        for well, m in pd_["plate_qc"].items():
            rows.append([_num(m.get(c)) for c in features])
            well_plate.append(pname)
            well_id.append(well)

    X = np.asarray(rows, dtype=float)
    if X.shape[0] < _MIN_WELLS:
        return {}
    n_plates = len(set(well_plate))
    if n_plates < _MIN_PLATES:
        return {}

    # ── Impute NaN with per-feature median, then drop zero-variance columns ───
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)          # column all-NaN → 0
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])

    std = X.std(axis=0)
    keep = std > 1e-12
    if keep.sum() < 2:
        return {}
    X = X[:, keep]
    features = [f for f, k in zip(features, keep) if k]

    # ── Standardise (z-score), then PCA via SVD ──────────────────────────────
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    Z = (X - mean) / std

    # SVD of the centred matrix: columns of Vt are the principal axes.
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    n_comp = min(2, Vt.shape[0])
    scores = Z @ Vt[:n_comp].T                       # [n_wells × n_comp]
    var = (S ** 2)
    explained = (var / var.sum())[:n_comp]

    # ── Assemble wells + per-plate centroids ─────────────────────────────────
    well_points = []
    for i in range(scores.shape[0]):
        well_points.append({
            "plate": well_plate[i],
            "well": well_id[i],
            "pc1": round(float(scores[i, 0]), 4),
            "pc2": round(float(scores[i, 1]), 4) if n_comp > 1 else 0.0,
        })

    plate_centroids = []
    for pname in dict.fromkeys(well_plate):           # preserve order
        idx = [i for i, p in enumerate(well_plate) if p == pname]
        c1 = float(np.mean(scores[idx, 0]))
        c2 = float(np.mean(scores[idx, 1])) if n_comp > 1 else 0.0
        plate_centroids.append({
            "plate": pname,
            "pc1": round(c1, 4),
            "pc2": round(c2, 4),
            "n_wells": len(idx),
        })

    # Deviation score per plate: distance of its centroid from the global
    # centroid, in PC units — a simple, interpretable "how far out" measure.
    if plate_centroids:
        gc1 = np.mean([p["pc1"] for p in plate_centroids])
        gc2 = np.mean([p["pc2"] for p in plate_centroids])
        for p in plate_centroids:
            p["deviation"] = round(float(
                np.hypot(p["pc1"] - gc1, p["pc2"] - gc2)), 4)

    return {
        "wells": well_points,
        "plates": plate_centroids,
        "explained": [round(float(e) * 100, 1) for e in explained],
        "n_features": len(features),
        "features": features,
    }


def _num(v):
    """Coerce to float or NaN (so missing/None participate as imputable gaps)."""
    if v is None:
        return np.nan
    if isinstance(v, float) and np.isnan(v):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan
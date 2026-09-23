"""
qc/schema.py — Canonical vocabulary for the pipeline.

A single place for metadata column names, aggregation levels and plate-id
normalization. The rest of the code must not hard-code "Metadata_Well" or
"Metadata_Field": it imports them from here. If CellProfiler changes a header or
a cohort arrives with different naming, only this file changes.
"""
from __future__ import annotations

from enum import Enum

# ── Metadata columns (identity of each row) ──────────────────────────────────
PLATE = "Metadata_Plate"
WELL = "Metadata_Well"
FIELD = "Metadata_Field"
IMAGE = "ImageNumber"

META_KEYS = (PLATE, WELL, FIELD)


class Level(str, Enum):
    """Aggregation levels. Each is a distinct group_by over the same LazyFrame."""
    SITE = "site"       # group_by(PLATE, WELL, FIELD)
    WELL = "well"       # group_by(PLATE, WELL)
    PLATE = "plate"     # group_by(PLATE)
    COHORT = "cohort"   # no group_by

    def group_keys(self) -> list:
        return {
            Level.SITE: [PLATE, WELL, FIELD],
            Level.WELL: [PLATE, WELL],
            Level.PLATE: [PLATE],
            Level.COHORT: [],
        }[self]


class AggPolicy(str, Enum):
    """Per-column reduction when collapsing site → well."""
    MEAN = "mean"   # metrics: Slope, MedianIntensity, FocusScore, Percent*
    SUM = "sum"     # Count_*, AREA_COL


# ── Plate-id normalization (single source of truth) ──────────────────────────
# Previously duplicated in 4+ places (loaders._norm_plate, the Polars
# strip_chars_start expression, Collage._lookup_plate._norm). Unified here into
# two variants with shared semantics: "P04"→"4", "4"→"4", "P4"→"4", ""→"0".

def normalize_plate(raw) -> str:
    """Scalar (Python) form. Replaces loaders._norm_plate and Collage._norm."""
    s = str(raw).strip()
    if s[:1] in ("P", "p"):
        s = s[1:]
    return s.lstrip("0") or "0"


def normalize_plate_expr(col: str = PLATE):
    """Vectorized (pl.Expr) form for the Polars path — same result as the scalar
    version without pulling data into Python. Replaces the duplicated
    strip_chars_start expression. Local import so callers of the scalar form
    don't require polars.
    """
    import polars as pl
    return (
        pl.col(col).cast(pl.Utf8).str.strip_chars()
          .str.strip_chars_start("Pp")
          .str.strip_chars_start("0")
          .replace("", "0")
    )

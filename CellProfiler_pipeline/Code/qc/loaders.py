from collections import defaultdict
from pathlib import Path
import polars as pl
from qc import config

# ── Aggregation policy ─────────────────────────────────────────────────────────
# Site-level values collapse to well-level with a per-column rule:
#   · mean_cols (metrics: Slope, MedianIntensity, FocusScore…) → mean over sites
#   · pct_cols  (PercentMaximal / PercentMinimal)              → mean over sites
#   · sum_cols  (Count_*, AREA_COL)                            → sum over sites
# These mirror exactly the aggregation performed inside load_qc_tsv_sites so that
# a collapsed well is identical to what a native per-well group_by would produce.

def _classify_cols(cols, AREA_COL, METRIC_COLS):
    """Split a list of column names into (mean, pct, sum) aggregation buckets."""
    metric_cols = {c for vals in METRIC_COLS.values() for c in vals}
    mean_cols, pct_cols, sum_cols = [], [], []
    for c in cols:
        if c in metric_cols:
            mean_cols.append(c)
        elif (c.startswith("ImageQuality_PercentMaximal_") or
              c.startswith("ImageQuality_PercentMinimal_")):
            pct_cols.append(c)
        elif c.startswith("Count_") or c == AREA_COL:
            sum_cols.append(c)
        else:
            # Unknown columns default to mean (safe for intensity-like values).
            mean_cols.append(c)
    return mean_cols, pct_cols, sum_cols


def collapse_sites_to_wells(site_qc: dict, AREA_COL, METRIC_COLS) -> dict:
    """Collapse {plate: {well: {field: {col: val}}}} → {plate: {well: {col: val}}}.

    Each column is aggregated across a well's sites using the policy above
    (metrics/percentages averaged, counts/area summed). Nulls and NaNs are
    ignored per column; a well with no valid value for a column omits it.

    This is the single source of truth for per-well QC: load once at site level,
    derive well level on demand instead of re-reading the TSV.
    """
    result: dict = {}
    for plate, wells in site_qc.items():
        result[plate] = {}
        for well, fields in wells.items():
            # Union of columns present across this well's sites.
            all_cols = {c for fv in fields.values() for c in fv}
            mean_cols, pct_cols, sum_cols = _classify_cols(
                all_cols, AREA_COL, METRIC_COLS)

            collapsed: dict = {}
            for col in mean_cols + pct_cols:
                vals = [fv[col] for fv in fields.values()
                        if col in fv and fv[col] is not None
                        and not (isinstance(fv[col], float) and _isnan(fv[col]))]
                if vals:
                    collapsed[col] = sum(vals) / len(vals)
            for col in sum_cols:
                vals = [fv[col] for fv in fields.values()
                        if col in fv and fv[col] is not None
                        and not (isinstance(fv[col], float) and _isnan(fv[col]))]
                if vals:
                    collapsed[col] = sum(vals)
            result[plate][well] = collapsed
    return result


def _isnan(x) -> bool:
    return x != x


# ── Data loaders ───────────────────────────────────────────────────────────────
def load_qc_tsv_sites(tsv_path, AREA_COL, METRIC_COLS) -> dict:
    """Per-site QC values, sin colapsar sites.

    Agrupa por [plate, well, field]. Es la fuente única de verdad: el nivel de
    well se deriva bajo demanda con collapse_sites_to_wells(), evitando releer el
    TSV y garantizando que ambos niveles no diverjan.

    Returns {plate: {well: {field: {col: value}}}}.
    """
    if tsv_path is None:
        return {}
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        print(f"[warn] QC file not found: {tsv_path}")
        return {}

    df        = pl.read_csv(tsv_path, separator="\t")
    plate_col = "Metadata_Plate" if "Metadata_Plate" in df.columns else None
    well_col  = "Metadata_Well"  if "Metadata_Well"  in df.columns else None
    field_col = "Metadata_Field" if "Metadata_Field" in df.columns else None
    if well_col is None:
        print("[warn] Metadata_Well not found — site-level QC disabled.")
        return {}
    if field_col is None:
        print("[warn] Metadata_Field not found — site-level QC disabled.")
        return {}

    mean_cols = list(dict.fromkeys(
        c for cols in METRIC_COLS.values() for c in cols if c in df.columns
    ))
    pct_cols  = list(dict.fromkeys(
        c for c in df.columns
        if c.startswith("ImageQuality_PercentMaximal_") or
           c.startswith("ImageQuality_PercentMinimal_")
    ))
    sum_cols  = list(dict.fromkeys(
        c for c in df.columns
        if c.startswith("Count_") or c == AREA_COL
    ))

    gkeys = ([plate_col] if plate_col else []) + [well_col, field_col]

    agg_exprs = (
        [pl.col(c).mean() for c in mean_cols] +
        [pl.col(c).mean() for c in pct_cols] +
        [pl.col(c).sum()  for c in sum_cols if c not in mean_cols and c not in pct_cols]
    )
    if not agg_exprs:
        print("[warn] No QC columns matched — nothing to aggregate.")
        return {}

    agg = df.group_by(gkeys).agg(agg_exprs)

    all_cols = (
        mean_cols + pct_cols +
        [c for c in sum_cols if c not in mean_cols and c not in pct_cols]
    )

    result: dict = {}
    for row in agg.iter_rows(named=True):
        plate = str(row[plate_col]).strip() if plate_col else "Plate"
        well  = str(row[well_col]).strip().upper()
        field = int(row[field_col])
        (result.setdefault(plate, {})
               .setdefault(well, {})[field]) = {c: row[c] for c in all_cols if c in row}

    n_sites = sum(len(fields) for p in result.values() for fields in p.values())
    print(f"[qc-sites] {n_sites} sites across "
          f"{sum(len(v) for v in result.values())} wells, {len(result)} plate(s).")
    return result


def load_platemap(platemap_path) -> dict:
    """Load platemap CSV -> {plate: {well: compound}}."""
    if platemap_path is None:
        return {}
    platemap_path = Path(platemap_path)
    if not platemap_path.exists():
        print(f"[warn] Platemap not found: {platemap_path}")
        return {}

    df           = pl.read_csv(platemap_path)
    well_col     = next((c for c in df.columns if "Well"     in c), None)
    compound_col = next((c for c in df.columns if "Compound" in c
                         or "Perturbation" in c), None)
    plate_col    = next((c for c in df.columns if "Plate"    in c), None)
    if not well_col or not compound_col:
        print(f"[warn] Platemap missing Well/Compound column. Found: {list(df.columns)}")
        return {}

    result = defaultdict(dict)
    for row in df.iter_rows(named=True):
        well  = str(row[well_col]).strip().upper()
        cmpd  = str(row[compound_col]).strip()
        raw_p = str(row[plate_col]).strip() if plate_col else "Plate"
        plate = raw_p if raw_p.startswith("P") else f"P{raw_p}"
        result[plate][well] = cmpd

    print(f"[platemap] {sum(len(v) for v in result.values())} well->compound mappings.")
    return result

def _load_object_tsv(tsv_path: Path, channel_map: dict, label: str) -> "pl.DataFrame | None":
    """Load one CellProfiler object TSV (Cells.txt or Nuclei.txt)."""
    if not tsv_path.exists():
        print(f"[mfi] {label} not found: {tsv_path}")
        return None

    df = pl.read_csv(tsv_path, separator="\t", low_memory=False)
    print(f"[mfi] Loaded {len(df):,} rows from {tsv_path.name}  ({label})")

    if "Metadata_Well" not in df.columns:
        print(f"[mfi] Metadata_Well missing in {tsv_path.name} — skipping.")
        return None

    df = df.with_columns(
        pl.col("Metadata_Well").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    )
    if "Metadata_Plate" in df.columns:
        df = df.with_columns(pl.col("Metadata_Plate").cast(pl.Utf8).str.strip_chars())
    else:
        df = df.with_columns(pl.lit("Plate").alias("Metadata_Plate"))

    mfi_cols = [col for col, _ in channel_map.values() if col in df.columns]
    if not mfi_cols:
        print(f"[mfi] No MFI columns found in {tsv_path.name} — skipping.")
        return None

    keep = (["Metadata_Plate", "Metadata_Well"]
            + (["ImageNumber"] if "ImageNumber" in df.columns else [])
            + mfi_cols)
    return df.select(keep)

def load_mfi_data(cells_path: "Path | None",
                  nuclei_path: "Path | None") -> "tuple[dict, dict]":
    """
    Load Cells.txt and Nuclei.txt and return (source_dfs, channel_map).
    source_dfs : { channel_label: DataFrame }
    channel_map: { channel_label: (col_name, hex_colour) }  ordered by MFI_CHANNEL_ORDER
    """
    source_dfs:  dict = {}
    channel_map: dict = {}
    print(f"[DEBUG] cells_path: {cells_path}, nuclei_path: {nuclei_path}")

    if cells_path is not None:
        df_cells = _load_object_tsv(cells_path, config.MFI_DATA['MFI_COLS_CELLS'], "Cells")
        if df_cells is not None:
            for ch, (col, color) in config.MFI_DATA['MFI_COLS_CELLS'].items():
                if col in df_cells.columns:
                    keep = (["Metadata_Plate", "Metadata_Well"]
                            + (["ImageNumber"] if "ImageNumber" in df_cells.columns else [])
                            + [col])
                    source_dfs[ch]  = df_cells.select(keep)
                    channel_map[ch] = (col, color)

    if nuclei_path is not None:
        df_nuclei = _load_object_tsv(nuclei_path, config.MFI_DATA['MFI_COLS_NUCLEI'], "Nuclei")
        if df_nuclei is not None:
            print(f"[mfi] Nuclei.txt columns: {[c for c in df_nuclei.columns if 'Hoechst' in c or 'Intensity' in c][:10]}")
            for ch, (col, color) in config.MFI_DATA['MFI_COLS_NUCLEI'].items():
                if col in df_nuclei.columns:
                    keep = (["Metadata_Plate", "Metadata_Well"]
                            + (["ImageNumber"] if "ImageNumber" in df_nuclei.columns else [])
                            + [col])
                    source_dfs[ch]  = df_nuclei.select(keep)
                    channel_map[ch] = (col, color)

    if not channel_map:
        print("[mfi] No usable MFI data found.")

    ordered = {ch: channel_map[ch] for ch in config.MFI_DATA['MFI_CHANNEL_ORDER'] if ch in channel_map}
    return source_dfs, ordered

def load_mfi_img(image_txt_path: "Path | None") -> "dict[str, dict[str, float]]":
    """
    Load per-well median MFI from Image.txt.
    Returns { well: { channel: median_across_sites } }
    """
    if image_txt_path is None or not image_txt_path.exists():
        print(f"[mfi_img] Image.txt not found: {image_txt_path}")
        return {}

    df = pl.read_csv(image_txt_path, separator="\t", low_memory=False)
    if "Metadata_Well" not in df.columns:
        return {}

    df = df.with_columns(
        pl.col("Metadata_Well").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    )

    img_cols = {ch: col for ch, col in config.MFI_DATA['MFI_IMG_COLS'].items()
                if col in df.columns}
    if not img_cols:
        return {}

    agg = df.group_by("Metadata_Well").agg(
        [pl.col(col).median().alias(ch) for ch, col in img_cols.items()]
    )

    result = {}
    for row in agg.iter_rows(named=True):
        well = row["Metadata_Well"]
        result[well] = {ch: float(row[ch]) for ch in img_cols
                        if row[ch] is not None}
    return result


def load_radius_data(cells_path: "Path | None",
                     nuclei_path: "Path | None",
                     plate_name: str) -> "dict[str, dict[str, float]]":
    """
    Load AreaShape_MedianRadius from Cells.txt and Nuclei.txt.
    Returns { 'Cells': { well: median_radius }, 'Nuclei': { well: median_radius } }
    """
    col = "AreaShape_MedianRadius"
    result = {}

    def _norm_plate(s: str) -> str:
        return str(s).strip().lstrip("P").lstrip("0") or "0"

    target = _norm_plate(plate_name)

    for label, path in [("Cells", cells_path), ("Nuclei", nuclei_path)]:
        if path is None or not path.exists():
            continue
        df = pl.read_csv(path, separator="\t", low_memory=False)
        if "Metadata_Well" not in df.columns or col not in df.columns:
            print(f"[radius] {col} not found in {path.name} — skipping.")
            continue

        df = df.with_columns(
            pl.col("Metadata_Well").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
        )
        if "Metadata_Plate" in df.columns:
            df = df.with_columns(pl.col("Metadata_Plate").cast(pl.Utf8).str.strip_chars())
        else:
            df = df.with_columns(pl.lit("Plate").alias("Metadata_Plate"))

        norm_expr = (
            pl.col("Metadata_Plate").cast(pl.Utf8).str.strip_chars()
              .str.strip_chars_start("P").str.strip_chars_start("0")
              .replace("", "0")
        )
        plate_df = df.filter(norm_expr == target)

        if plate_df.is_empty():
            if df["Metadata_Plate"].n_unique() == 1:
                plate_df = df
            else:
                continue

        agg = (plate_df.group_by("Metadata_Well")
               .agg(pl.col(col).median().alias("_med"))
               .drop_nulls("_med"))
        result[label] = {row["Metadata_Well"]: float(row["_med"])
                         for row in agg.iter_rows(named=True)}
    return result
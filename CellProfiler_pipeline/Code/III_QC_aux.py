#!/usr/bin/env python3
"""Consolida los .txt de cada batch_i_j en un parquet por tipo de objeto."""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import polars as pl

META_PREFIXES = ("Metadata_", "FileName_", "PathName_", "URL_",
                 "ObjectNumber_", "Group_")  # ajusta a tu pipeline

def harmonize(frames: list[pl.DataFrame]) -> list[pl.DataFrame]:
    dtypes = {}
    for df in frames:
        for c, t in df.schema.items():
            dtypes.setdefault(c, set()).add(t)
    target = {}
    for c, ts in dtypes.items():
        if len(ts) == 1:
            continue
        target[c] = pl.Utf8 if c.startswith(META_PREFIXES) else pl.Float64
    out = []
    for df in frames:
        exprs = [pl.col(c).cast(t, strict=False) for c, t in target.items()
                 if c in df.columns and df.schema[c] != t]
        out.append(df.with_columns(exprs) if exprs else df)
    return out

TABLES = ("Cells", "Image", "Nuclei")
BATCH_RE = re.compile(r"^batch_(?P<start>\d+)_(?P<end>\d+)$")

def field_map(dirs: list[Path]) -> pl.DataFrame | None:
    """(Batch, ImageNumber) -> Metadata_Field, leído de Cells/Nuclei."""
    frames = []
    for d in dirs:
        for t in ("Cells", "Nuclei"):
            f = d / f"{t}.txt"
            if not f.exists():
                continue
            try:
                df = pl.read_csv(f, separator="\t", infer_schema_length=10000,
                                 columns=["ImageNumber", "Metadata_Field"])
            except Exception as e:
                print(f"[WARN] {f}: sin ImageNumber/Metadata_Field ({e})", file=sys.stderr)
                continue
            frames.append(df.with_columns(pl.lit(d.name).alias("Batch")))
    if not frames:
        return None
    return pl.concat(frames).unique(["Batch", "ImageNumber"])


def add_field(df: pl.DataFrame, fmap: pl.DataFrame | None) -> pl.DataFrame:
    if "Metadata_Field" in df.columns:
        return df
    if fmap is None:
        print("[WARN] No se pudo reconstruir Metadata_Field", file=sys.stderr)
        return df

    df = df.with_columns(pl.col("ImageNumber").cast(pl.Int64))
    fmap = fmap.with_columns(pl.col("ImageNumber").cast(pl.Int64))
    df = df.join(fmap, on=["Batch", "ImageNumber"], how="left")

    # Imágenes sin objetos: completar por complemento dentro de (Plate, Well)
    key = ["Metadata_Plate", "Metadata_Well"]
    expected = set(range(1, 10))            # sitios por pozo: ajusta si no son 9
    known = (df.filter(pl.col("Metadata_Field").is_not_null())
               .group_by(key).agg(pl.col("Metadata_Field").alias("seen")))
    missing = (df.filter(pl.col("Metadata_Field").is_null())
                 .group_by(key).agg(pl.col("ImageNumber").alias("imgs"),
                                    pl.col("Batch").alias("batches")))
    fixes, ambiguous = [], 0
    for r in missing.join(known, on=key, how="left").iter_rows(named=True):
        comp = sorted(expected - set(r["seen"] or []))
        if len(r["imgs"]) == 1 and len(comp) == 1:
            fixes.append((r["batches"][0], r["imgs"][0], comp[0]))
        else:
            ambiguous += len(r["imgs"])
    if fixes:
        fix = pl.DataFrame(fixes, schema=["Batch", "ImageNumber", "_f"], orient="row")
        df = (df.join(fix, on=["Batch", "ImageNumber"], how="left")
                .with_columns(pl.coalesce("Metadata_Field", "_f").alias("Metadata_Field"))
                .drop("_f"))
    print(f"[INFO] Metadata_Field reconstruido; {len(fixes)} por complemento, "
          f"{ambiguous} sin resolver")
    return df
    
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="Directorio que contiene las carpetas batch_i_j")
    p.add_argument("--output", type=Path, default=None,
                   help="Directorio donde escribir los parquet "
                        "(por defecto: <input>/../Measurements)")
    p.add_argument("--tables", nargs="+", default=list(TABLES))
    p.add_argument("--keep-batch-column", action="store_true",
                   help="Añade una columna Batch con el tag de origen")

    args = p.parse_args()
    args.input = args.input.resolve()
    if args.output is None:
        args.output = args.input.parent / "Measurements"
    return args


def batch_dirs(root: Path) -> list[Path]:
    found = []
    for d in root.iterdir():
        m = BATCH_RE.match(d.name)
        if d.is_dir() and m:
            found.append((int(m.group("start")), d))
    return [d for _, d in sorted(found)]

KEY_COLS = {"ImageNumber", "ObjectNumber"}

def read_table(f: Path) -> pl.DataFrame:
    header = pl.read_csv(f, separator="\t", n_rows=0).columns
    overrides = {c: pl.Float64 for c in header
                 if not c.startswith(META_PREFIXES) and c not in KEY_COLS}
    return pl.read_csv(
        f, separator="\t",
        dtypes=overrides,
        null_values=["nan", "NaN", "inf", "-inf", ""],
    )

def consolidate(table: str, dirs: list[Path]) -> pl.DataFrame | None:
    frames = []
    for d in dirs:
        f = d / f"{table}.txt"
        if not f.exists():
            print(f"[WARN] {f} no existe, salto", file=sys.stderr)
            continue
        df = read_table(f)
        if df.height == 0:
            print(f"[WARN] {f} está vacío, salto", file=sys.stderr)
            continue
        df = df.with_columns(pl.lit(d.name).alias("Batch"))
        frames.append(df)
    from collections import defaultdict
    seen = defaultdict(dict)
    for df, d in zip(frames, (x for x in dirs)):  # frames puede ser más corto si saltaste alguno
        for c, t in df.schema.items():
            seen[c][d.name] = t
    for c, per_batch in seen.items():
        if len(set(per_batch.values())) > 1:
            print(f"[WARN] {table}.{c}: {per_batch}", file=sys.stderr)
    if not frames:
        return None
    return pl.concat(harmonize(frames), how="diagonal")


def main():
    args = parse_args()

    if not args.input.is_dir():
        sys.exit(f"[ERROR] No existe el directorio de entrada: {args.input}")

    dirs = batch_dirs(args.input)
    if not dirs:
        sys.exit(f"[ERROR] No se encontraron carpetas batch_i_j en {args.input}")

    print(f"[INFO] {len(dirs)} batches identified: {', '.join(d.name for d in dirs)}")
    print(f"[INFO] Output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    fmap = field_map(dirs) if "Image" in args.tables else None

    rc = 0
    for table in args.tables:
        df = consolidate(table, dirs)
        if df is None:
            print(f"[ERROR] Without data for: {table}", file=sys.stderr)
            rc = 1
            continue
        if table == "Image":
            df = add_field(df, fmap)
        if not args.keep_batch_column:
            df = df.drop("Batch")
        dest = args.output / f"{table}.parquet"
        df.write_parquet(dest, compression="zstd")
        print(f"[INFO] {table}: {df.height} rows, {df.width} columns → {dest}")

    sys.exit(rc)


if __name__ == "__main__":
    main()
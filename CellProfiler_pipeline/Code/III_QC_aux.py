#!/usr/bin/env python3
"""Consolida los .txt de cada batch_i_j en un parquet por tipo de objeto."""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path


import polars as pl

TABLES = ("Cells", "Image", "Nuclei")
BATCH_RE = re.compile(r"^batch_(?P<start>\d+)_(?P<end>\d+)$")


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


def consolidate(table: str, dirs: list[Path], keep_batch: bool) -> pl.DataFrame | None:
    frames = []
    for d in dirs:
        f = d / f"{table}.txt"
        if not f.exists():
            print(f"[WARN] {f} no existe, salto", file=sys.stderr)
            continue
        df = pl.read_csv(f, separator="\t", infer_schema_length=10000)
        if df.height == 0:
            print(f"[WARN] {f} está vacío, salto", file=sys.stderr)
            continue
        if keep_batch:
            df = df.with_columns(pl.lit(d.name).alias("Batch"))
        frames.append(df)

    if not frames:
        return None
    return pl.concat(frames, how="diagonal")


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

    rc = 0
    for table in args.tables:
        df = consolidate(table, dirs, args.keep_batch_column)
        if df is None:
            print(f"[ERROR] Without data for: {table}", file=sys.stderr)
            rc = 1
            continue
        dest = args.output / f"{table}.parquet"
        df.write_parquet(dest, compression="zstd")
        print(f"[INFO] {table}: {df.height} rows, {df.width} columns → {dest}")

    sys.exit(rc)


if __name__ == "__main__":
    main()
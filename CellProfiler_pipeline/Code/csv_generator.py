from __future__ import annotations
from pathlib import Path
import re
import random
import logging
from typing import Union, List, Dict, Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import polars as pl
from tqdm import tqdm
import os, json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PLATE_REGEX = re.compile(r"_P(?P<Plate>\d{2})_")


def get_well(row: Union[str, int], col: Union[str, int]) -> str:
    """
        Convert numeric row and column indices into well notation.

        Parameters
        ----------
        row : str | int
            Row index (1-based).
        col : str | int
            Column index.

        Returns
        -------
        str
            Well identifier (e.g. 'A01').
    """
    row_corr = chr(64 + int(row))
    return f"{row_corr}{int(col):02d}"


def _parse_single_file(
    path: Path,
    illum: bool,
    masks: bool,
    compiled_regex: re.Pattern,
    channels: Dict[int, str],
    output: Path | None,) -> dict | None:
    """
        Parse metadata from a single image file.

        Parameters
        ----------
        path : Path
            Path to the image file.
        illum : bool
            Whether to include illumination correction fields.
        masks : bool
            Whether to include RNA mask fields.
        output : Path | None
            Output directory used to build derived paths.

        Returns
        -------
        dict | None
            Parsed metadata row, or None if the file does not match criteria.
    """
    match = compiled_regex.match(path.name)
    if not match:
        return None

    meta = match.groupdict()

    plate_match = PLATE_REGEX.search(str(path))
    if not plate_match:
        return None

    plate_meta = plate_match.groupdict()
    ch_num = int(meta["Channel"])

    row = {
        "Image_FileName": path.name,
        "Image_PathName": str(path.parent),
        "Metadata_Plate": plate_meta["Plate"],
        "Metadata_Well": get_well(meta["Row"], meta["Column"]),
        "Metadata_Field": int(meta["Field"]),
        "Metadata_Channel": ch_num,
        "Metadata_Channel_name": channels.get(ch_num, str(ch_num)),
    }

    if illum and output is not None:
        row["FileName_Illum"] = f"Illum_{channels.get(ch_num, ch_num)}.npy"
        row["PathName_Illum"] = str(output.parents[0] / "Illum_files")

    if masks and output is not None:
        row["Image_FileName_RNA_mask"] = None
        row["Image_PathName_RNA_mask"] = str(output.parents[0] / "Cellpose_seg" / f"P_{plate_meta['Plate']}")
    return row


def parse_image_dataset(
    root: Union[str, Path],
    channels: dict,
    compiled_regex: re.Pattern,
    illum: bool = False,
    masks: bool = False,
    output: Union[str, Path, None] = None,
    max_workers: int | None = None,
    plates2process: list | None = None,
    chunk_size: int = 2000,) -> pl.DataFrame:
    """
        Parse a microscopy image dataset into a structured Polars DataFrame using
        multiprocessing and chunked execution for memory efficiency.

        Parameters
        ----------
        root : str | Path
            Root directory containing plate subfolders.
        illum : bool, default False
            Whether to include illumination correction metadata.
        masks : bool, default False
            Whether to include RNA mask metadata.
        output : str | Path | None, default None
            Output directory used to build derived paths.
        max_workers : int | None, default None
            Number of worker processes.
        chunk_size : int, default 2000
            Number of files processed per batch to limit memory usage.

        Returns
        -------
        pl.DataFrame
            Long-format metadata table.
    """
    root = Path(root)
    output = Path(output) if output is not None else None
    files = [
        path
        for plate in root.iterdir()
        if plate.is_dir()
        and plate.name != "Output"
        and (plates2process is None or any(f"_P{p}_" in plate.name for p in plates2process))
        for path in plate.rglob("*.tif*")
    ]

    logger.info(f"Found {len(files)} image files")

    rows: List[dict] = []
    worker = partial(_parse_single_file, illum=illum, masks=masks, output=output, compiled_regex=compiled_regex, channels=channels)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for i in tqdm(range(0, len(files), chunk_size), desc="Parsing images"):
            batch = files[i:i + chunk_size]
            for result in ex.map(worker, batch, chunksize=64):
                if result is not None:
                    rows.append(result)

    logger.info(f"Parsed {len(rows)} valid image records")
    return pl.DataFrame(rows)


def _aux_pivot(df: pl.DataFrame, values_col: str, idx: list[str], channel_names: list[str]) -> pl.DataFrame:
    """
        Pivot helper to reshape long to wide format.

        Parameters
        ----------
        df : pl.DataFrame
            Input long-format DataFrame.
        values_col : str
            Column to pivot.
        idx : list[str]
            Index columns.
        channel_names : list[str]
            Channel identifiers.

        Returns
        -------
        pl.DataFrame
            Pivoted DataFrame.
    """
    df_aux = df.pivot(
        index=idx,
        on="Metadata_Channel_name",
        values=values_col,
        aggregate_function="first",
    )
    rename_map = {ch: f"{values_col}_{ch}" for ch in channel_names}    
    return df_aux.rename(rename_map)

def pivot_df(df: pl.DataFrame, illum: bool = False, masks: bool = False) -> pl.DataFrame:
    idx = ["Metadata_Plate", "Metadata_Well", "Metadata_Field"]
    channel_names = df["Metadata_Channel_name"].unique().to_list()

    df_file = _aux_pivot(df, "Image_FileName", idx, channel_names)
    df_path = _aux_pivot(df, "Image_PathName", idx, channel_names)

    df_final = df_file.join(df_path, on=idx, how="inner")

    if illum:
        df_illum_file = _aux_pivot(df, "FileName_Illum", idx, channel_names)
        df_illum_path = _aux_pivot(df, "PathName_Illum", idx, channel_names)

        df_final = (
            df_final
            .join(df_illum_file, on=idx, how="inner")
            .join(df_illum_path, on=idx, how="inner")
        )

    if masks:
        df_masks = (
            df
            .group_by(idx)
            .agg([pl.first("Image_FileName_RNA_mask").alias("Image_FileName_RNA_mask"),
                  pl.first("Image_PathName_RNA_mask").alias("Image_PathName_RNA_mask")]))

        df_final = df_final.join(df_masks, on=idx, how="left")
        df_final = df_final.with_columns([
            pl.when(pl.col("Image_FileName_RNA_mask").is_null())
            .then(
                pl.col(f"Image_FileName_Syto")
                .str.replace(r"\.tiff?$", ".png")
            )
            .otherwise(pl.col("Image_FileName_RNA_mask"))
            .alias("Image_FileName_RNA_mask")])

    ordered_cols = (
        [f"Image_FileName_{ch}" for ch in channel_names]
        + [f"Image_PathName_{ch}" for ch in channel_names]
        + ([f"FileName_Illum_{ch}" for ch in channel_names]
           + [f"PathName_Illum_{ch}" for ch in channel_names] if illum else [])
        + (["Image_FileName_RNA_mask", "Image_PathName_RNA_mask"] if masks else [])
        + idx
    )

    return df_final.select(ordered_cols)

def prepare_CSVs(
    path: Union[str, Path],
    output: Union[str, Path],
    name_csv: str,
    channels: dict,
    compiled_regex: re.Pattern,
    illum: bool = False,
    masks: bool = False,
    max_workers: int | None = None,
    plates2process: list | None = None,) -> None:
    """
        Run the full metadata extraction pipeline and save final CSV output.

        Parameters
        ----------
        path : str | Path
            Input dataset root directory.
        output : str | Path
            Output directory.
        illum : bool, default False
            Whether to include illumination metadata.
        masks : bool, default False
            Whether to include RNA mask metadata.
        max_workers : int | None, default None
            Number of worker processes.
    """
    #print(f"[DEBUG masks] masks: {masks}")
    logger.info("Starting metadata parsing")
    df = parse_image_dataset(
        path,
        illum=illum,
        masks=masks,
        output=output,
        max_workers=max_workers,
        plates2process=plates2process,
        channels=channels,
        compiled_regex=compiled_regex
    )

    #print("[DEBUG] columnas reales:", df.columns)

    logger.info("Pivoting metadata table")
    df_wide = pivot_df(df, illum=illum, masks=masks)

    filepath = Path(output) / f"{name_csv}.csv"
    df_wide.write_csv(filepath)
    logger.info(f"Final metadata CSV written to: {filepath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="High-performance metadata extraction pipeline")
    parser.add_argument("-i", "--input", required=True, help="Input dataset root")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--name_csv", required=True, help="Name of the CSV file to generate")
    parser.add_argument("--illum", action="store_true", help="Include illumination metadata")
    parser.add_argument("--masks", action="store_true", help="Include RNA mask metadata")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--plates2process", nargs='*', help="List of plates to be processed")
    parser.add_argument("--channel_dict", required=True, help='JSON Name→Number, e.g. {"Mito":"1",...}')
    parser.add_argument("--filename_regex", required=True, help="Regex with the following groups: Row/Column/Field/Plane/Channel")

    args = parser.parse_args()

    print(f"[DEBUG] channel_dict repr: {repr(args.channel_dict)}")

    channels = {int(v): k for k, v in json.loads(args.channel_dict).items()}
    compiled_regex = re.compile(args.filename_regex)

    prepare_CSVs(
        path=args.input,
        output=args.output,
        name_csv=args.name_csv,
        illum=args.illum,
        masks=args.masks,
        max_workers=args.workers,
        plates2process=args.plates2process,
        channels=channels,
        compiled_regex=compiled_regex
    )
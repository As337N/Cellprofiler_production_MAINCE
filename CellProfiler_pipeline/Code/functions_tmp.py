from pathlib import Path
import re
import polars as pl
import random
from typing import Union, List

def get_well(row, col):
  row_corr = chr(64 + int(row))
  return f"{row_corr}{col}"

def split_and_save_by_groups(df: pl.DataFrame, col: str, k: int,
    out_dir: Union[str, Path],) -> List[pl.DataFrame]:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = df.select(pl.col(col).unique()).to_series().to_list()
    random.shuffle(groups)

    chunk_size = (len(groups) + k - 1) // k
    group_chunks = [groups[i:i + chunk_size] for i in range(0, len(groups), chunk_size)]

    dfs_out = []

    for idx, chunk in enumerate(group_chunks):
        subdf = df.filter(pl.col(col).is_in(chunk))
        filename = f"{idx:03d}.csv"
        filepath = out_dir / filename

        subdf.write_csv(filepath)
        dfs_out.append(subdf)

    while len(dfs_out) < k:
        empty_df = pl.DataFrame(schema=df.schema)
        filename = f"{len(dfs_out):03d}.parquet"
        filepath = out_dir / filename
        empty_df.write_parquet(filepath)
        dfs_out.append(empty_df)

    return dfs_out

def parse_image_dataset(root: Union[str, Path], illum: bool = False, masks: bool = False, output: str = None) -> pl.DataFrame:
    root = Path(root)
    print(f"root: {root}")

    regex = re.compile(
        r"^\d(?P<Row>\d{2})\d(?P<Column>\d{2})-(?P<Field>\d)-\d{3}(?P<Plane>\d{3})(?P<Channel>\d{3}).tif"
    )
    regex_plate = re.compile(r"_P(?P<Plate>\d{2})_")

    channels = {1: "Mito", 2: "Golgi", 3: "Brightfield", 4: "Syto", 5: "ER", 6: "Hoechst"}
    rows = []

    for plate in root.iterdir():
        if plate.is_dir() and plate.name != "Output":
                
            for path in plate.rglob("*.tif*"):
                m = regex.match(path.name)
                if not m:
                    continue

                meta = m.groupdict()
                
                # solo plane == 2
                if int(meta["Plane"]) != 2:
                    continue

                meta["Well"] = get_well(meta["Row"], meta["Column"])

                parent_name = path.parent.name
                parent_id, _, channel_name = parent_name.partition("w")
                print(f"[DEBUG] path: {path}")
                plate_re = regex_plate.search(str(path))
                print(f"[DEBUG] plate_re: {plate_re}")
                meta_plate = plate_re.groupdict()
                ch_num = int(meta["Channel"])

                row = {
                    "Image_FileName": path.name,
                    "Image_PathName": str(path.parent),
                    "Metadata_Plate": meta_plate["Plate"],
                    "Metadata_Well": meta["Well"],
                    "Metadata_Field": int(meta["Field"]),
                    "Metadata_Channel": ch_num,
                    "Metadata_Parent_id": parent_id,
                    "Metadata_Channel_name": channels.get(ch_num, str(ch_num)),
                }

                if illum:
                    row["FileName_Illum"] = f"Illum_{channels.get(ch_num, ch_num)}.npy"
                    row["PathName_Illum"] = f"{output}/Illum_files"

                if masks and ch_num == 4:
                    row["Image_FileName_RNA_mask"] = f"{path.stem}.png",
                    row["Image_PathName_RNA_mask"] = f"{output}/Cellpose_seg"

                rows.append(row)

    return pl.DataFrame(rows)

def _aux_pivot(df, values_col, idx, channel_names):
    df_aux = df.pivot(
        index=idx,
        on="Metadata_Channel_name",
        values=values_col,
        aggregate_function="first"
    )
    cpa_name = {ch: f"{values_col}_{ch}" for ch in channel_names}
    return df_aux.rename(cpa_name)

def pivot_df(df: pl.DataFrame, illum: bool=False, masks: bool=False) -> pl.DataFrame:
    idx = ["Metadata_Plate", "Metadata_Well", "Metadata_Field", "Metadata_Parent_id"]
    #print(df)
    channel_names = df["Metadata_Channel_name"].unique().to_list()
    print(f"Channel names: {channel_names}")

    ###### 1) Detectar si hay None / null
    null_count = df["Metadata_Channel_name"].null_count()
    if null_count > 0:
        #print(f"[DEBUG] Hay {null_count} valores None en Metadata_Channel_name")
        # Ver qué filas son
        print(
            df.filter(pl.col("Metadata_Channel_name").is_null())
              .select([
                  "Metadata_Plate",
                  "Metadata_Well",
                  "Metadata_Field",
                  "Metadata_Parent_id",
                  "Metadata_Channel_name",
              ])
              .unique()
        )
    ######

    df_file = _aux_pivot(df=df, 
                         values_col="Image_FileName",
                         idx=idx,
                         channel_names=channel_names)
    df_path = _aux_pivot(df=df, 
                         values_col="Image_PathName",
                         idx=idx,
                         channel_names=channel_names)

    if illum:
        df_illum_file = _aux_pivot(df=df,
                                   values_col="FileName_Illum",
                                   idx=idx,
                                   channel_names=channel_names)
        df_illum_path = _aux_pivot(df=df,
                                   values_col="PathName_Illum",
                                   idx=idx,
                                   channel_names=channel_names)

        df_final = (df_file
                    .join(df_path, on=idx, how="inner")
                    .join(df_illum_file, on=idx, how="inner")
                    .join(df_illum_path, on=idx, how="inner"))

        ordered_cols = (
            [f"Image_FileName_{ch}" for ch in channel_names] +
            [f"Image_PathName_{ch}" for ch in channel_names] +
            [f"FileName_Illum_{ch}" for ch in channel_names] +
            [f"PathName_Illum_{ch}" for ch in channel_names] +
            ["Image_FileName_RNA_mask", "Image_PathName_RNA_mask", "Metadata_Well", "Metadata_Field", "Metadata_Plate"]
        )

    else:
        df_final = df_file.join(df_path, on=idx)
        ordered_cols = (
            [f"Image_FileName_{ch}" for ch in channel_names] +
            [f"Image_PathName_{ch}" for ch in channel_names] +
            ["Metadata_Well", "Metadata_Field", "Metadata_Plate"]
        )

    return df_final.select(ordered_cols)

def prepare_CSVs(path, output, illum:bool=False, masks=False):
    df = parse_image_dataset(path, illum=illum, masks=masks, output=output)
    df_wide = pivot_df(df, illum=illum)

    df_agg = df_wide.group_by(
                pl.col("Metadata_Well"))

    #print(df_agg)

    filepath = output / f"metadata.csv"
    df_wide.write_csv(filepath)    
    print(f"Archivo guardado en: {filepath}")

if __name__=="__main__":
    path = "/workspace_data/BBBC047/"
    output = Path("/workspace_data/BBBC047_output")
    """out_folders = ["Illum_pipelines", "MP_pipelines", "Illum_files", "Profiles", "Illum_CSVs", "Prof_CSVs"]

    for folder in out_folders:
        (output / folder).mkdir(exist_ok=True)
    output.mkdir(exist_ok=True, parents=True)"""
    
    prepare_CSVs(path=path, output=output)

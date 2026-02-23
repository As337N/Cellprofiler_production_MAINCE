import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import tifffile as tiff
import imageio.v2 as imageio
from skimage.transform import resize
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
import re
import csv

_SCRIPT_DIR = Path(__file__).resolve().parent

# Font candidates in priority order:
#   1. fonts/ directory next to this script  (recommended for Docker: COPY fonts/ ./fonts/)
#   2. Common paths on Debian/Ubuntu
#   3. Common paths on Alpine  (apk add ttf-dejavu)
#   4. macOS
#   5. Windows
_FONT_SEARCH: dict[str, list[Path]] = {
    "bold": [
        _SCRIPT_DIR / "fonts" / "DejaVuSans-Bold.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans-Bold.ttf"),
        Path("/Library/Fonts/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans-Bold.ttf"),
    ],
    "regular": [
        _SCRIPT_DIR / "fonts" / "DejaVuSans.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans.ttf"),
        Path("/Library/Fonts/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans.ttf"),
    ],
}

_font_cache: dict[tuple, ImageFont.FreeTypeFont] = {}


def _font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    """
    Resolve a font with progressive fallback:
        1. Font bundled in fonts/ next to this script
        2. System font paths (Debian, Alpine, macOS, Windows)
        3. Pillow's built-in bitmap font (always available, no size scaling)
    """
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]

    candidates = _FONT_SEARCH["bold" if bold else "regular"]
    for path in candidates:
        if path.exists():
            try:
                f = ImageFont.truetype(str(path), size)
                _font_cache[key] = f
                return f
            except (OSError, IOError):
                continue

    print(
        f"WARNING: TrueType font not found (size={size}, bold={bold}). "
        f"Falling back to Pillow default bitmap font (text will be small). "
        f"For better quality place DejaVuSans[-Bold].ttf in {_SCRIPT_DIR / 'fonts'}/",
        flush=True,
    )
    f = ImageFont.load_default()
    _font_cache[key] = f
    return f


def draw_label(img, text, pos=(5, 5),
               color=(120, 200, 255),
               shadow=(0, 0, 0),
               size=70):
    """Draw shadowed text onto a numpy array (returns a copy)."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    font = _font(size)
    x, y = pos
    draw.text((x + 1, y + 1), text, fill=shadow, font=font)
    draw.text((x, y), text, fill=color, font=font)
    return np.array(pil)


def well_label(row, col):
    return f"{chr(ord('A') + row - 1)}{col:02d}"


def load_and_downscale(path, scale=0.5):
    img = tiff.imread(path)
    if img.dtype != np.uint8:
        p_max = img.max()
        img = (img / p_max * 255).astype(np.uint8) if p_max > 0 else img.astype(np.uint8)
    if scale != 1.0:
        new_shape = (int(img.shape[0] * scale), int(img.shape[1] * scale))
        img = resize(img, new_shape, preserve_range=True,
                     anti_aliasing=True).astype(np.uint8)
    return img


def parse_name(fname):
    """Parse filename and return (row, col, site) as integers."""
    name = fname.replace(".tiff", "")
    rc, site, _ = name.split("-")
    row = int(rc[:3])
    col = int(rc[3:6])
    site = int(site)
    return row, col, site


def load_metadata(csv_path):
    """
    Read the metadata CSV and return a nested dict:
        meta[plate][well_label] = {
            'perturbation':   str,
            'object_count':   int   | None,
            'mean_intensity': float | None,
            'std_intensity':  float | None,
            'iqr_intensity':  float | None,
        }

    Expected CSV columns (case-insensitive, common aliases accepted):
        plate          -> Plate / PlateID / plate_id
        well           -> Well              (format A01, optional if Row+Col present)
        row            -> Row
        col            -> Col / Column
        perturbation   -> Perturbation / Treatment / Compound / Drug
        object_count   -> ObjectCount / Objects / NumObjects / CellCount
        mean_intensity -> MeanIntensity / Mean_Intensity / Mean
        std_intensity  -> StdIntensity / Std_Intensity / Std / SD / StdDev  (optional)
        iqr_intensity  -> IQR / IQR_Intensity                                (optional)
    """
    if csv_path is None:
        return {}

    aliases = {
        "plate":          ["plate", "plateid", "plate_id"],
        "well":           ["well"],
        "row":            ["row"],
        "col":            ["col", "column"],
        "perturbation":   ["perturbation", "treatment", "compound", "drug"],
        "object_count":   ["objectcount", "objects", "numobjects", "num_objects",
                           "cell_count", "cellcount"],
        "mean_intensity": ["meanintensity", "mean_intensity", "mean"],
        "std_intensity":  ["stdintensity", "std_intensity", "std", "sd",
                           "stdev", "stddev"],
        "iqr_intensity":  ["iqr", "iqr_intensity"],
    }

    def find_col(headers, key):
        for h in headers:
            if h.lower().replace(" ", "_") in aliases[key]:
                return h
        return None

    meta = defaultdict(dict)
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []

        c_plate = find_col(headers, "plate")
        c_well  = find_col(headers, "well")
        c_row   = find_col(headers, "row")
        c_col   = find_col(headers, "col")
        c_pert  = find_col(headers, "perturbation")
        c_obj   = find_col(headers, "object_count")
        c_mean  = find_col(headers, "mean_intensity")
        c_std   = find_col(headers, "std_intensity")
        c_iqr   = find_col(headers, "iqr_intensity")

        if c_plate is None:
            raise ValueError("CSV: plate column not found (Plate / PlateID).")
        if c_well is None and (c_row is None or c_col is None):
            raise ValueError("CSV: need either a 'Well' column or both 'Row' and 'Col' columns.")

        for row in reader:
            plate = row[c_plate].strip()

            if c_well:
                wlabel = row[c_well].strip().upper()
            else:
                wlabel = well_label(int(row[c_row]), int(row[c_col]))

            def _float(col_key):
                if col_key and row.get(col_key, "").strip():
                    try:
                        return float(row[col_key])
                    except ValueError:
                        pass
                return None

            def _int(col_key):
                v = _float(col_key)
                return int(v) if v is not None else None

            meta[plate][wlabel] = {
                "perturbation":   row[c_pert].strip() if c_pert else "",
                "object_count":   _int(c_obj),
                "mean_intensity": _float(c_mean),
                "std_intensity":  _float(c_std),
                "iqr_intensity":  _float(c_iqr),
            }

    return meta


def build_well_montage(args):
    (r, c), sites, scale, sites_per_well = args

    tiles = []
    for s in range(1, sites_per_well + 1):
        if s in sites:
            tiles.append(load_and_downscale(sites[s], scale))

    if not tiles:
        return (r, c), None

    n = max(1, int(np.ceil(np.sqrt(len(tiles)))))
    blank = np.zeros_like(tiles[0])
    while len(tiles) % n != 0:
        tiles.append(blank)

    rows_img = [np.concatenate(tiles[i:i + n], axis=1)
                for i in range(0, len(tiles), n)]
    well_montage = np.concatenate(rows_img, axis=0)

    label = well_label(r, c)
    well_montage = draw_label(well_montage, label,
                              size=max(20, int(well_montage.shape[0] * 0.07)))
    return (r, c), well_montage


def make_margin_band(width, well_labels_in_row, meta_row,
                     band_height=60, font_size=18, tile_width=None):
    """
    Build a dark horizontal band with per-well metadata text, aligned to each
    well tile. Inserted between plate rows so it never overlaps image pixels.
    """
    band = Image.new("RGB", (width, band_height), color=(18, 18, 18))
    draw = ImageDraw.Draw(band)
    font_b = _font(font_size, bold=True)
    font_r = _font(font_size, bold=False)

    for idx, wlabel in enumerate(well_labels_in_row):
        x_offset = idx * tile_width if tile_width else 0
        info = meta_row.get(wlabel)
        if info is None:
            continue

        lines = []
        if info.get("perturbation"):
            lines.append(("bold", info["perturbation"]))

        objs = info.get("object_count")
        if objs is not None:
            lines.append(("reg", f"n={objs}"))

        mean = info.get("mean_intensity")
        std  = info.get("std_intensity")
        iqr  = info.get("iqr_intensity")
        if mean is not None:
            if std is not None:
                lines.append(("reg", f"u={mean:.1f}+/-{std:.1f}"))
            elif iqr is not None:
                lines.append(("reg", f"u={mean:.1f} IQR={iqr:.1f}"))
            else:
                lines.append(("reg", f"u={mean:.1f}"))

        y = 2
        for style, text in lines:
            font = font_b if style == "bold" else font_r
            draw.text((x_offset + 4, y + 1), text, fill=(0, 0, 0), font=font)
            draw.text((x_offset + 4, y), text,
                      fill=(200, 230, 255) if style == "bold" else (180, 220, 180),
                      font=font)
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                line_h = bbox[3] - bbox[1]
            except AttributeError:
                line_h = font_size
            y += line_h + 2
            if y + line_h > band_height:
                break

    return np.array(band)


class Collage:
    def __init__(self, input_dir, output_path,
                 plate_rows=8, plate_cols=12,
                 sites_per_well=9, scale=0.5, workers=8,
                 metadata_csv=None,
                 margin_height=60, margin_font_size=16):

        self.input_dir        = Path(input_dir)
        self.output_path      = Path(output_path)
        self.plate_rows       = plate_rows
        self.plate_cols       = plate_cols
        self.sites_per_well   = sites_per_well
        self.scale            = scale
        self.workers          = workers
        self.margin_height    = margin_height
        self.margin_font_size = margin_font_size
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.meta = load_metadata(metadata_csv)
        if self.meta:
            total = sum(len(v) for v in self.meta.values())
            print(f"Metadata loaded: {total} records across {len(self.meta)} plate(s).")

        self.plates = self._group_by_plate()

        for plate_name, files in self.plates.items():
            print(f"Processing plate: {plate_name}")
            wells         = self._group_well_imgs(files)
            well_montages = self._create_well_montages_parallel(wells)
            self._create_plate_montage(plate_name, well_montages)

    def _group_by_plate(self):
        plates = defaultdict(list)
        for f in self.input_dir.glob("*.tiff"):
            match = re.search(r"_(P\d+)", f.name)
            if match:
                plates[match.group(1)].append(f)
            else:
                print(f"Warning: no plate suffix found in {f.name}, skipping.")
        return plates

    def _group_well_imgs(self, files):
        wells = defaultdict(dict)
        for f in files:
            r, c, s = parse_name(f.name)
            wells[(r, c)][s] = f
        return wells

    def _create_well_montages_parallel(self, wells):
        well_montages = {}
        args = [
            ((r, c), sites, self.scale, self.sites_per_well)
            for (r, c), sites in wells.items()
        ]
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            futures = [ex.submit(build_well_montage, a) for a in args]
            for fut in as_completed(futures):
                (r, c), montage = fut.result()
                if montage is not None:
                    well_montages[(r, c)] = montage
        return well_montages

    def _create_plate_montage(self, plate_name, well_montages):
        if not well_montages:
            print(f"  No images found for {plate_name}, skipping.")
            return

        sample_tile = next(iter(well_montages.values()))
        tile_h, tile_w = sample_tile.shape[:2]
        blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

        plate_meta = self.meta.get(plate_name, {})
        plate_rows_imgs = []

        for r in range(1, self.plate_rows + 1):
            row_labels = [well_label(r, c) for c in range(1, self.plate_cols + 1)]
            row_tiles  = []

            for c in range(1, self.plate_cols + 1):
                tile = well_montages.get((r, c), blank)
                if tile.ndim == 2:
                    tile = np.stack([tile] * 3, axis=-1)
                row_tiles.append(tile)

            plate_row_img = np.concatenate(row_tiles, axis=1)
            plate_rows_imgs.append(plate_row_img)

            if plate_meta:
                band = make_margin_band(
                    width=plate_row_img.shape[1],
                    well_labels_in_row=row_labels,
                    meta_row=plate_meta,
                    band_height=self.margin_height,
                    font_size=self.margin_font_size,
                    tile_width=tile_w,
                )
                plate_rows_imgs.append(band)

        collage = np.concatenate(plate_rows_imgs, axis=0)
        out_file = self.output_path / f"{plate_name}.png"
        imageio.imwrite(str(out_file), collage, format="png", compress_level=6)
        print(f"  -> {out_file}  ({collage.shape[1]}x{collage.shape[0]} px)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Plate collage builder — Cell Painting QC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input",   required=True,
                   help="Directory containing the plate .tiff files.")
    p.add_argument("-o", "--output",  required=True,
                   help="Output directory for the collage .png files.")
    p.add_argument("--rows",    type=int,   default=8,
                   help="Number of plate rows.")
    p.add_argument("--cols",    type=int,   default=12,
                   help="Number of plate columns.")
    p.add_argument("--sites",   type=int,   default=9,
                   help="Sites per well.")
    p.add_argument("--scale",   type=float, default=0.5,
                   help="Downscale factor applied to each site image.")
    p.add_argument("--workers", type=int,   default=8,
                   help="Parallel worker processes for well montage building.")
    p.add_argument("--metadata", default=None,
                   help=(
                       "Path to the per-well metadata CSV. "
                       "Expected columns (case-insensitive): "
                       "Plate, Well (or Row+Col), Perturbation/Treatment, "
                       "ObjectCount, MeanIntensity, StdIntensity/IQR."
                   ))
    p.add_argument("--margin-height", type=int, default=60,
                   help="Height in pixels of the metadata band between plate rows.")
    p.add_argument("--margin-font",   type=int, default=16,
                   help="Font size used inside the metadata band.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Collage(
        input_dir        = args.input,
        output_path      = args.output,
        plate_rows       = args.rows,
        plate_cols       = args.cols,
        sites_per_well   = args.sites,
        scale            = args.scale,
        workers          = args.workers,
        metadata_csv     = args.metadata,
        margin_height    = args.margin_height,
        margin_font_size = args.margin_font,
    )
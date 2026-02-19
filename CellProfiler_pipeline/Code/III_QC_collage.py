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

def draw_label(img, text, pos=(5, 5),
               color=(120, 200, 255),
               shadow=(0, 0, 0),
               size=70):

    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        size
    )

    x, y = pos
    draw.text((x+1, y+1), text, fill=shadow, font=font)
    draw.text((x, y), text, fill=color, font=font)

    return np.array(pil)

def well_label(row, col):
    letter = chr(ord("A") + row - 1)
    return f"{letter}{col:02d}"

def load_and_downscale(path, scale=0.5):
    img = tiff.imread(path)

    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)

    if scale != 1:
        new_shape = (
            int(img.shape[0] * scale),
            int(img.shape[1] * scale)
        )
        img = resize(
            img,
            new_shape,
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.uint8)

    return img

def parse_name(fname):
    name = fname.replace(".tiff", "")
    rc, site, _ = name.split("-")
    row = int(rc[:3])
    col = int(rc[3:6])
    site = int(site)
    return row, col, site

def build_well_montage(args):
    (r, c), sites, scale, sites_per_well = args

    tiles = []
    for s in range(1, sites_per_well + 1):
        if s not in sites:
            continue
        img = load_and_downscale(sites[s], scale)
        tiles.append(img)

    n = int(np.sqrt(len(tiles)))
    rows = [np.concatenate(tiles[i:i+n], axis=1)
            for i in range(0, len(tiles), n)]

    well_montage = np.concatenate(rows, axis=0)
    label = well_label(r, c)
    well_montage = draw_label(well_montage, label)

    return (r, c), well_montage

class Collage:
    def __init__(self, input_dir, output_path,
                 plate_rows=8, plate_cols=12,
                 sites_per_well=9, scale=0.5, workers=8):
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path)
        self.plate_rows = plate_rows
        self.plate_cols = plate_cols
        self.sites_per_well = sites_per_well
        self.scale = scale
        self.workers = workers

        self.plates = self._group_by_plate()

        for plate_name, files in self.plates.items():
            print(f"Processing plate: {plate_name}")
            wells = self._group_well_imgs(files)
            well_montages = self._create_well_montages_parallel(wells)
            self._create_plate_montage(plate_name, well_montages)

    def _group_by_plate(self):
        """Agrupa archivos por plate detectando el sufijo _PXX en el nombre."""
        plates = defaultdict(list)
        for f in self.input_dir.glob("*.tiff"):
            match = re.search(r'_(P\d+)', f.name)
            if match:
                plate_name = match.group(1)
                plates[plate_name].append(f)
            else:
                print(f"Advertencia: no se encontró sufijo de plate en {f.name}, se omite.")
        return plates

    def _group_well_imgs(self, files):
        """Agrupa imágenes de una lista de archivos por pocillo (well)."""
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
                well_montages[(r, c)] = montage
        return well_montages

    def _create_plate_montage(self, plate_name, well_montages):
        plate_rows = []
        sample_tile = next(iter(well_montages.values()))
        blank = np.zeros_like(sample_tile)
        for r in range(1, self.plate_rows + 1):
            row_tiles = []
            for c in range(1, self.plate_cols + 1):
                tile = well_montages.get((r, c), blank)
                row_tiles.append(tile)
            plate_rows.append(np.concatenate(row_tiles, axis=1))
        collage = np.concatenate(plate_rows, axis=0)
        out_file = self.output_path / f"{plate_name}.png"
        imageio.imwrite(out_file, collage, format="png", compress_level=6)
        print(f"Plate montage done for plate: {plate_name}")

def parse_args():
    p = argparse.ArgumentParser("Plate collage builder")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=12)
    p.add_argument("--sites", type=int, default=9)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    Collage(
        input_dir=args.input,
        output_path=args.output,
        plate_rows=args.rows,
        plate_cols=args.cols,
        sites_per_well=args.sites,
        scale=args.scale,
        workers=args.workers
    )

"""
qc/render/montage.py — Montaje de imágenes por well (I/O paralelo).

Fase 0: funciones movidas VERBATIM desde III_QC_collage. Esta es la parte que
SÍ conviene en ThreadPoolExecutor — decodificar TIFF y redimensionar con
cv2/tifffile libera el GIL, así que los hilos corren de verdad en paralelo.
El cómputo numérico NO va aquí: va a stats/ con Polars.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
from PIL import Image, ImageDraw, ImageFont


# ── Fonts (verbatim) ─────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent.parent   # raíz del paquete qc/
_FONT_PATHS = {
    "bold": [
        _SCRIPT_DIR / "fonts" / "DejaVuSans-Bold.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
        Path("/Library/Fonts/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans-Bold.ttf"),
    ],
    "regular": [
        _SCRIPT_DIR / "fonts" / "DejaVuSans.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("/Library/Fonts/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans.ttf"),
    ],
}
_font_cache: dict = {}


def _font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    key = (size, bold)
    if key not in _font_cache:
        for path in _FONT_PATHS["bold" if bold else "regular"]:
            if path.exists():
                try:
                    _font_cache[key] = ImageFont.truetype(str(path), size)
                    break
                except (OSError, IOError):
                    continue
        else:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def _text_h(draw, text: str, font) -> int:
    try:
        return draw.textbbox((0, 0), text, font=font)[3]
    except AttributeError:
        return font.size if hasattr(font, "size") else 14


# ── Image helpers (verbatim) ─────────────────────────────────────────────────
def well_label(row: int, col: int) -> str:
    return f"{chr(ord('A') + row - 1)}{col:02d}"


def load_and_downscale(path: Path, scale: float = 0.5) -> np.ndarray:
    with tiff.TiffFile(path) as tf:
        img = tf.pages[0].asarray()
    if img.dtype != np.uint8:
        p_max = img.max()
        img = (img / p_max * 255).astype(np.uint8) if p_max > 0 else img.astype(np.uint8)
    if scale != 1.0:
        img = cv2.resize(img,
                         (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def build_well_montage(args: tuple) -> tuple:
    (r, c), sites, scale, spw = args
    tiles = [load_and_downscale(sites[s], scale)
             for s in range(1, spw + 1) if s in sites]
    if not tiles:
        return (r, c), None
    n     = max(1, int(np.ceil(np.sqrt(len(tiles)))))
    blank = np.zeros_like(tiles[0])
    while len(tiles) % n:
        tiles.append(blank)
    rows = [np.concatenate(tiles[i:i + n], axis=1) for i in range(0, len(tiles), n)]
    return (r, c), np.concatenate(rows, axis=0)


def _make_tile(tile: np.ndarray, border_rgb: tuple, border_w: int = 6) -> np.ndarray:
    img    = tile if tile.ndim == 3 else np.stack([tile] * 3, axis=-1)
    canvas = Image.fromarray(img.copy())
    draw   = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, img.shape[1] - 1, img.shape[0] - 1],
                   outline=border_rgb, width=border_w)
    return np.array(canvas)


# ── Paralelismo (ex-Collage._build_montages_parallel) ────────────────────────
def build_montages_parallel(wells: dict, *, scale: float, sites_per_well: int,
                            workers: int = 8) -> dict:
    montages: dict = {}
    args = [((r, c), sites, scale, sites_per_well) for (r, c), sites in wells.items()]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(build_well_montage, a): a[0] for a in args}
        for fut in as_completed(futures):
            (r, c), mont = fut.result()
            if mont is not None:
                montages[(r, c)] = mont
    return montages

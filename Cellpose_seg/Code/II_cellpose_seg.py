from pathlib import Path
import re
from cellpose import models
from tqdm import tqdm
import warnings
import time

import argparse
from concurrent.futures import ThreadPoolExecutor
from skimage.io import imread, imsave
import numpy as np
import torch
import tifffile as tiff
import os
import shutil
from cellpose import models

MODEL_DIR = Path(__file__).parent / "sam-model"
MODEL_PATH = MODEL_DIR / "cpsam"


def ensure_cpsam_weights() -> Path:
    """Devuelve la ruta a los pesos cpsam, descargándolos si no están."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    print(f"[INFO] Pesos cpsam no encontrados en {MODEL_PATH}, descargando…")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Cellpose descarga a CELLPOSE_LOCAL_MODELS_PATH o ~/.cellpose/models
    os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(MODEL_DIR))
    models.CellposeModel(gpu=False)   # fuerza la descarga

    cache = Path(os.environ["CELLPOSE_LOCAL_MODELS_PATH"]) / "cpsam"
    if cache.exists() and cache != MODEL_PATH:
        shutil.copy2(cache, MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"La descarga no dejó los pesos en {MODEL_PATH}. "
            f"Revisa la salida a internet del contenedor o monta los pesos manualmente."
        )
    return MODEL_PATH

# Harmony / JUMP: r09c22f02p01-ch4sk1fk1fl1.tiff
"""DEFAULT_CHANNEL_REGEX = (
    r"^r(?P<Row>\d{2})c(?P<Column>\d{2})f(?P<Field>\d{2})p(?P<Plane>\d{2})"
    r"-ch(?P<Channel>\d)(?:sk\d+fk\d+fl\d+)?\.tiff?$"
)"""

# Dataset custom: ..._002004.tif  ->  r"00200(?P<Channel>\d)\.tiff?$"
DEFAULT_CHANNEL_REGEX = r"00200(?P<Channel>\d)\.tiff?$"

def apply_plate_illumination_correction(
    image_paths: list[Path],
    illumination_npy: Path,
    device: str = "cuda",
    batch_size: int = 32,) -> list[np.ndarray]:
    """
        Apply illumination correction to a full plate (≈864 images) using PyTorch + GPU.

        Steps:
        1) Check dimension consistency.
        2) Divide image / illumination.
        3) Clamp values: <0 → 0, > original max → original max.
        4) Return corrected images as NumPy arrays.

        Parameters
        ----------
        image_paths : list[Path]
            List of image paths for the full plate.
        illumination_npy : Path
            Path to illumination correction .npy file.
        device : str, default="cuda"
            Device to use: "cuda" or "cpu".
        batch_size : int, default=32
            Batch size for GPU processing.

        Returns
        -------
        list[np.ndarray]
            List of corrected images.
    """

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    illum = np.load(illumination_npy)
    illum_t = torch.from_numpy(illum).float().to(device)

    corrected_images = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Illumination correction"):
        batch_paths = image_paths[i:i + batch_size]

        imgs = [tiff.imread(p) for p in batch_paths]
        imgs_np = np.stack(imgs, axis=0)

        if imgs_np.shape[1:] != illum.shape:
            raise ValueError(
                f"Shape mismatch: images {imgs_np.shape[1:]} vs illumination {illum.shape}"
            )

        imgs_t = torch.from_numpy(imgs_np).float().to(device)

        eps = 1e-6
        corrected = imgs_t / (illum_t + eps)

        img_max = imgs_t.amax(dim=(1, 2), keepdim=True)
        corrected = torch.clamp(corrected, min=0.0)
        corrected = torch.minimum(corrected, img_max)

        corrected_images.extend(corrected.cpu().numpy().astype(imgs_np.dtype))

    return corrected_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cellpose segmentation of RNA channel per plate."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--rna_channel", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--regex", type=str, default=r"_P(?P<Plate>\d{2})_")
    parser.add_argument(
        "--channel-regex",
        type=str,
        default=DEFAULT_CHANNEL_REGEX,
        help="Regex con un grupo nombrado 'Channel' que capture el índice de canal.",
    )
    return parser.parse_args()


def load_images(images_path):
    with ThreadPoolExecutor() as ex:
        return list(ex.map(imread, images_path))


def save_masks(masks, paths, output_dir, max_workers=None):
    def _save(mask, path):
        imsave(
            output_dir / f"{path.stem}.png",
            mask.astype(np.uint16)
        )
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_save, mask, path) for mask, path in zip(masks, paths)]
        for f in futures:
            f.result()


class CellposeRnaSeg():
    def __init__(self, input_path, output_path, rna_channel, batch_size, regex, channel_regex):
        self.model = models.CellposeModel(gpu=True, pretrained_model=str(ensure_cpsam_weights()))
        self.input_path = input_path
        self.output_path = output_path
        self.plate_regex = re.compile(regex)
        self.channel_regex = re.compile(channel_regex)
        if "Channel" not in self.channel_regex.groupindex:
            raise ValueError(
                f"--channel-regex debe contener un grupo nombrado 'Channel': {channel_regex}"
            )
        self.rna_channel = rna_channel
        self.batch_size = batch_size
        self.name_image_directories = "untreated_data"

    def _get_plate(self, path: Path) -> str:
        m = self.plate_regex.search(str(path))
        return m.group("Plate") if m else "unknown"

    def _is_rna_image(self, path: Path) -> bool:
        m = self.channel_regex.search(path.name)
        return m is not None and int(m.group("Channel")) == self.rna_channel

    def _mk_plate_dir(self, path_images):
        plate = self._get_plate(path_images)
        path_masks = self.output_path / f"P_{plate}"
        path_masks.mkdir(parents=True, exist_ok=True)
        return plate, path_masks

    def _process_plate(self, img_paths, corrected_images, path_masks):
        """
            Segment a full plate using pre-corrected images already in memory.

            Parameters
            ----------
            img_paths : list[Path]
                Original image paths (used only for naming output masks).
            corrected_images : list[np.ndarray]
                Illumination-corrected images.
            path_masks : Path
                Output directory for masks.
        """
        if len(img_paths) != len(corrected_images):
            raise ValueError(
                f"Mismatch: {len(img_paths)} paths vs {len(corrected_images)} corrected images"
            )

        for i in range(0, len(corrected_images), self.batch_size):
            batch_paths = img_paths[i:i + self.batch_size]
            batch_imgs = corrected_images[i:i + self.batch_size]

            masks, _, _ = self.model.eval(
                batch_imgs,
                diameter=95,
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )

            save_masks(masks, batch_paths, path_masks)

    def run(self, illumination_npy: Path):
        for p in self.input_path.iterdir():
            if not p.is_dir():
                continue

            path_images = p / self.name_image_directories
            if not path_images.exists():
                print(f"[WARN] No existe {path_images}, salto plate")
                continue

            plate, path_masks = self._mk_plate_dir(path_images)

            img_paths = sorted(f for f in path_images.iterdir() if self._is_rna_image(f))

            if not img_paths:
                all_files = [f.name for f in path_images.iterdir()][:5]
                print(f"[WARN] Plate {plate}: 0 imágenes para canal {self.rna_channel} "
                      f"con patrón '{self.channel_regex.pattern}'. "
                      f"Ejemplos en la carpeta: {all_files}")
                continue

            print(f"[INFO Cellpose segmentation] Plate {plate}: {len(img_paths)} images")

            corrected_images = apply_plate_illumination_correction(
                image_paths=img_paths,
                illumination_npy=illumination_npy,
                batch_size=self.batch_size,
            )
            self._process_plate(img_paths, corrected_images, path_masks)


def main():
    args = parse_args()
    pipeline = CellposeRnaSeg(
        input_path=args.input_path,
        output_path=args.output_path,
        regex=args.regex,
        channel_regex=args.channel_regex,
        rna_channel=args.rna_channel,
        batch_size=args.batch_size
    )
    pipeline.run(illumination_npy=Path("/output/CellProfiler_files/Illum_files/Illum_Syto.npy"))


if __name__=="__main__":
    start_time = time.perf_counter()

    warnings.filterwarnings(
        "ignore",
        message=".*low contrast image.*"
    )

    warnings.filterwarnings(
        "ignore",
        message=".*Resizing is deprecated.*"
    )

    main()

    elapsed = time.perf_counter() - start_time
    print(f"Total execution time for cellpose segmentation: {elapsed/3600:.2f} hours")
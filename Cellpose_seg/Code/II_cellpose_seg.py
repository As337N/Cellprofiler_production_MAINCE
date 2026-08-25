from pathlib import Path
import re
from cellpose import models
from tqdm import tqdm
import warnings
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from skimage.io import imsave
import numpy as np
import torch
import tifffile as tiff


MODEL_PATH = Path(__file__).parent / "sam-model" / "cpsam"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"cpsam weights not found at {MODEL_PATH}")

# Harmony / JUMP: r09c22f02p01-ch4sk1fk1fl1.tiff
"""DEFAULT_CHANNEL_REGEX = (
    r"^r(?P<Row>\d{2})c(?P<Column>\d{2})f(?P<Field>\d{2})p(?P<Plane>\d{2})"
    r"-ch(?P<Channel>\d)(?:sk\d+fk\d+fl\d+)?\.tiff?$"
)"""

# Dataset custom: ..._002004.tif  ->  r"00200(?P<Channel>\d)\.tiff?$"
DEFAULT_CHANNEL_REGEX = r"00200(?P<Channel>\d)\.tiff?$"


class PreparePlate():
    def __init__(self, image_paths, illumination_npy, batch_size: int = 32):
        self.image_paths = image_paths
        self.illumination_npy = illumination_npy
        self.batch_size = batch_size

    def illum_correction(self) -> list[np.ndarray]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        illum = np.load(self.illumination_npy)
        illum_t = torch.from_numpy(illum).float().to(device)

        corrected_images = []

        for i in tqdm(range(0, len(self.image_paths), self.batch_size), desc="Illumination correction"):
            batch_paths = self.image_paths[i:i + self.batch_size]

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


class CellPose():
    def __init__(self, path_model, max_workers, batch_size):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.model = models.CellposeModel(gpu=True, pretrained_model=str(path_model))

    def _save_masks(self, masks, paths, output_dir):
        def _save(mask, path):
            imsave(
                output_dir / f"{path.stem}.png",
                mask.astype(np.uint16)
            )
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(_save, mask, path) for mask, path in zip(masks, paths)]
            for f in futures:
                f.result()

    def process_plate(self, img_paths, corrected_images, path_masks):
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

            self._save_masks(masks, batch_paths, path_masks)


class CellSegmentation():
    def __init__(self, input_path, output_path, rna_channel, batch_size, regex, channel_regex, max_workers):
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
        self.max_workers = max_workers

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

    def run(self, illumination_npy: Path):
        cellpose_seg = CellPose(
            path_model=MODEL_PATH,
            max_workers=self.max_workers,
            batch_size=self.batch_size,
        )

        for p in self.input_path.iterdir():
            if not p.is_dir():
                continue

            path_images = p / self.name_image_directories
            if not path_images.exists():
                print(f"[WARN] {path_images} does not exists, skipping")
                continue

            plate, path_masks = self._mk_plate_dir(path_images)

            img_paths = sorted(f for f in path_images.iterdir() if self._is_rna_image(f))

            if not img_paths:
                print(f"[WARN] Plate {plate}: 0 images for channel {self.rna_channel} "
                      f"regex: '{self.channel_regex.pattern}'. "
                      f"Example files: {[f.name for f in path_images.iterdir()][:5]}")
                continue

            print(f"[INFO Cellpose segmentation] Plate {plate}: {len(img_paths)} images")

            prep = PreparePlate(
                image_paths=img_paths,
                illumination_npy=illumination_npy,
                batch_size=self.batch_size,
            )
            corrected_images = prep.illum_correction()

            cellpose_seg.process_plate(
                img_paths=img_paths,
                corrected_images=corrected_images,
                path_masks=path_masks,
            )


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


if __name__ == "__main__":
    start_time = time.perf_counter()
    warnings.filterwarnings("ignore", message=".*low contrast image.*")
    warnings.filterwarnings("ignore", message=".*Resizing is deprecated.*")

    args = parse_args()
    segmentation = CellSegmentation(
        input_path=args.input_path,
        output_path=args.output_path,
        regex=args.regex,
        channel_regex=args.channel_regex,
        rna_channel=args.rna_channel,
        batch_size=args.batch_size,
        max_workers=2,
    )
    segmentation.run(illumination_npy=Path("/output/CellProfiler_files/Illum_files/Illum_Syto.npy"))

    elapsed = time.perf_counter() - start_time
    print(f"Total execution time for cellpose segmentation: {elapsed/3600:.2f} hours")
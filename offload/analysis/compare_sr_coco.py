import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offload.server.sr import create_lowres_sr_engine


def parse_args():
    parser = argparse.ArgumentParser(description="Compare x4 SR variants on the first COCO validation images.")
    parser.add_argument("--num-images", type=int, default=10, help="Number of initial COCO images to compare.")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize used before downsampling.")
    parser.add_argument("--downscale", type=int, default=4, help="Downsample factor before x4 restoration.")
    parser.add_argument("--sr-batch-size", type=int, default=10, help="Batch size used for SR inference.")
    parser.add_argument("--weights-dir", type=str, default="~/cjpark/weights/realesrgan", help="Directory with Real-ESRGAN checkpoints.")
    parser.add_argument("--dtype", type=str, default="fp16", help="SR dtype: fp16, bf16, or fp32.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, default: cuda if available else cpu.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory. Default: logs/sr_compare_<timestamp>.")
    return parser.parse_args()


def resolve_output_dir(out_dir: str | None) -> Path:
    if out_dir is not None:
        path = Path(out_dir).expanduser()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("logs") / f"sr_compare_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_coco_images(num_images: int, image_size: int) -> tuple[list[str], np.ndarray]:
    import fiftyone.zoo as foz

    dataset = foz.load_zoo_dataset("coco-2017", split="validation")
    filepaths = dataset.values("filepath")[:num_images]

    names = []
    images = []
    for filepath in filepaths:
        path = Path(filepath)
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {filepath}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        names.append(path.name)
        images.append(image_rgb)

    return names, np.stack(images, axis=0)


def make_lowres_variants(images_rgb: np.ndarray, downscale: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = images_rgb.shape[1:3]
    lr_size = (w // downscale, h // downscale)

    lowres = np.stack([
        cv2.resize(image, lr_size, interpolation=cv2.INTER_AREA)
        for image in images_rgb
    ], axis=0)
    lowres_display = np.stack([
        cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
        for image in lowres
    ], axis=0)
    bicubic = np.stack([
        cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        for image in lowres
    ], axis=0)
    return lowres_display, bicubic


def run_sr_in_chunks(lowres_rgb: np.ndarray, target_hw: tuple[int, int], model_name: str, weights_dir: str, dtype: str, device: torch.device, batch_size: int) -> np.ndarray:
    sr_engine = create_lowres_sr_engine({
        "model": model_name,
        "weights_dir": weights_dir,
        "dtype": dtype,
        "tile": 0,
        "tile_pad": 10,
        "pre_pad": 0,
    }, device)

    outputs = []
    for start in range(0, len(lowres_rgb), batch_size):
        chunk = lowres_rgb[start:start + batch_size]
        outputs.append(sr_engine.upscale(chunk, target_hw=target_hw))

    del sr_engine
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return np.concatenate(outputs, axis=0)


def make_residual_rgb(reference_rgb: np.ndarray, candidate_rgb: np.ndarray) -> np.ndarray:
    absdiff = cv2.absdiff(reference_rgb, candidate_rgb)
    gray = cv2.cvtColor(absdiff, cv2.COLOR_RGB2GRAY)
    return np.repeat(gray[..., None], 3, axis=2)


def draw_grid(rows: list[list[np.ndarray]], col_titles: list[str], row_titles: list[str], output_path: Path):
    assert len(rows) == len(row_titles)
    assert all(len(row) == len(col_titles) for row in rows)

    cell_h, cell_w = rows[0][0].shape[:2]
    header_h = 42
    row_label_w = 180
    pad = 8

    total_h = header_h + pad + len(rows) * (cell_h + pad) + pad
    total_w = row_label_w + pad + len(col_titles) * (cell_w + pad) + pad
    canvas = np.full((total_h, total_w, 3), 245, dtype=np.uint8)

    for col_idx, title in enumerate(col_titles):
        x = row_label_w + pad + col_idx * (cell_w + pad)
        cv2.putText(
            canvas,
            title,
            (x + 6, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    for row_idx, (row_name, images) in enumerate(zip(row_titles, rows)):
        y = header_h + pad + row_idx * (cell_h + pad)
        cv2.putText(
            canvas,
            row_name,
            (10, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        for col_idx, image in enumerate(images):
            x = row_label_w + pad + col_idx * (cell_w + pad)
            canvas[y:y + cell_h, x:x + cell_w] = image

    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    if args.image_size % args.downscale != 0:
        raise ValueError(f"image_size={args.image_size} must be divisible by downscale={args.downscale}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = resolve_output_dir(args.out_dir)

    names, originals = load_coco_images(args.num_images, args.image_size)
    lowres_display, bicubic = make_lowres_variants(originals, args.downscale)
    lowres_native = np.stack([
        cv2.resize(image, (args.image_size // args.downscale, args.image_size // args.downscale), interpolation=cv2.INTER_AREA)
        for image in originals
    ], axis=0)

    target_hw = (args.image_size, args.image_size)
    general_x4v3 = run_sr_in_chunks(
        lowres_native, target_hw, "realesr_general_x4v3",
        args.weights_dir, args.dtype, device, args.sr_batch_size
    )
    x4plus = run_sr_in_chunks(
        lowres_native, target_hw, "realesrgan_x4plus",
        args.weights_dir, args.dtype, device, args.sr_batch_size
    )

    comparison_rows = []
    residual_rows = []
    zero_tile = np.zeros_like(originals[0])

    for idx, name in enumerate(names):
        comparison_rows.append([
            originals[idx],
            lowres_display[idx],
            bicubic[idx],
            general_x4v3[idx],
            x4plus[idx],
        ])
        residual_rows.append([
            zero_tile.copy(),
            make_residual_rgb(originals[idx], lowres_display[idx]),
            make_residual_rgb(originals[idx], bicubic[idx]),
            make_residual_rgb(originals[idx], general_x4v3[idx]),
            make_residual_rgb(originals[idx], x4plus[idx]),
        ])

    col_titles = [
        "Original",
        "1/4 Downsample",
        "Bicubic x4",
        "RealESR general x4v3",
        "RealESRGAN x4plus",
    ]

    draw_grid(comparison_rows, col_titles, names, out_dir / "comparison_grid.png")
    draw_grid(residual_rows, col_titles, names, out_dir / "residual_grid.png")

    print(f"[SR Compare] Saved comparison grid to {out_dir / 'comparison_grid.png'}")
    print(f"[SR Compare] Saved residual grid to {out_dir / 'residual_grid.png'}")


if __name__ == "__main__":
    main()

import csv
import json
import os
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from offload.server.sr import create_lowres_sr_engine


def normalize_dataset_name(name: str) -> str:
    value = str(name).strip().lower()
    aliases = {
        "imagenet": "imagenet-1k",
        "imnet-1k": "imagenet-1k",
        "imagenet-1k": "imagenet-1k",
        "coco": "coco",
        "coco2017": "coco",
        "coco-2017": "coco",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported dataset: {name}")
    return aliases[value]


def default_data_root_for_dataset(dataset_name: str) -> str:
    normalized = normalize_dataset_name(dataset_name)
    if normalized == "imagenet-1k":
        return "~/data/imagenet_val"
    return ""


def resolve_output_dir(prefix: str, out_dir: str | None) -> Path:
    if out_dir is not None:
        path = Path(out_dir).expanduser()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("logs") / "analysis" / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        formatted_rows = []
        for row in rows:
            formatted = {}
            for key, value in row.items():
                if isinstance(value, Real) and not isinstance(value, bool):
                    if isinstance(value, int):
                        formatted[key] = value
                    else:
                        formatted[key] = f"{float(value):+0.6e}"
                else:
                    formatted[key] = value
            formatted_rows.append(formatted)
        writer.writerows(formatted_rows)


def resize_longest_side_and_pad(image: Image.Image, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    width, height = image.size
    scale = image_size / max(height, width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resampling = Image.Resampling if hasattr(Image, "Resampling") else Image
    resized = image.resize((resized_width, resized_height), resample=resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))

    pad_left = (image_size - resized_width) // 2
    pad_top = (image_size - resized_height) // 2
    canvas.paste(resized, (pad_left, pad_top))

    image_np = np.asarray(canvas, dtype=np.uint8)
    valid_mask = np.zeros((image_size, image_size), dtype=np.bool_)
    valid_mask[pad_top:pad_top + resized_height, pad_left:pad_left + resized_width] = True

    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
    mask_tensor = torch.from_numpy(valid_mask)
    return image_tensor, mask_tensor


class _ImageNetAnalysisDataset(Dataset):
    def __init__(self, root: str, image_size: int):
        self.root = os.path.expanduser(root)
        self.image_size = image_size
        self.dataset = datasets.ImageFolder(root=self.root, transform=None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        path, _ = self.dataset.samples[idx]
        image = Image.open(path).convert("RGB")
        image_tensor, valid_mask = resize_longest_side_and_pad(image, self.image_size)
        _, label = self.dataset[idx]
        return image_tensor, {
            "sample_key": f"{idx:06d}:{Path(path).name}",
            "path": path,
            "label": int(label),
            "valid_mask": valid_mask,
        }


class _COCOAnalysisDataset(Dataset):
    def __init__(self, image_size: int):
        import fiftyone.zoo as foz

        self.image_size = image_size
        dataset = foz.load_zoo_dataset("coco-2017", split="validation")
        self.filepaths = dataset.values("filepath")

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        path = self.filepaths[idx]
        image = Image.open(path).convert("RGB")
        image_tensor, valid_mask = resize_longest_side_and_pad(image, self.image_size)
        return image_tensor, {
            "sample_key": f"{idx:06d}:{Path(path).name}",
            "path": path,
            "label": -1,
            "valid_mask": valid_mask,
        }


def _collate_analysis_batch(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    metadata = [item[1] for item in batch]
    return {
        "images": images,
        "valid_masks": torch.stack([item["valid_mask"] for item in metadata], dim=0),
        "sample_keys": [item["sample_key"] for item in metadata],
        "paths": [item["path"] for item in metadata],
        "labels": [item["label"] for item in metadata],
    }


def build_analysis_loader(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    image_size: int,
    num_workers: int,
) -> DataLoader:
    normalized = normalize_dataset_name(dataset_name)
    if normalized == "imagenet-1k":
        dataset = _ImageNetAnalysisDataset(root=data_root, image_size=image_size)
    else:
        dataset = _COCOAnalysisDataset(image_size=image_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_analysis_batch,
    )


def _bchw_to_bhwc_numpy(images_bchw: torch.Tensor) -> np.ndarray:
    return images_bchw.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def _bhwc_numpy_to_bchw(images_bhwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images_bhwc).permute(0, 3, 1, 2).contiguous()


def make_lowres_and_bicubic(images_bchw: torch.Tensor, downscale: int) -> tuple[torch.Tensor, torch.Tensor]:
    images_bhwc = _bchw_to_bhwc_numpy(images_bchw)
    h, w = images_bhwc.shape[1:3]
    lr_hw = (w // downscale, h // downscale)

    lowres = []
    bicubic = []
    for image in images_bhwc:
        lowres_image = cv2.resize(image, lr_hw, interpolation=cv2.INTER_AREA)
        bicubic_image = cv2.resize(lowres_image, (w, h), interpolation=cv2.INTER_CUBIC)
        lowres.append(lowres_image)
        bicubic.append(bicubic_image)

    lowres_np = np.stack(lowres, axis=0)
    bicubic_np = np.stack(bicubic, axis=0)
    return _bhwc_numpy_to_bchw(lowres_np), _bhwc_numpy_to_bchw(bicubic_np)


def run_sr_model_in_chunks(
    lowres_bchw: torch.Tensor,
    target_hw: tuple[int, int],
    model_name: str,
    weights_dir: str,
    dtype: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    sr_engine = create_lowres_sr_engine({
        "model": model_name,
        "weights_dir": weights_dir,
        "dtype": dtype,
        "tile": 0,
        "tile_pad": 10,
        "pre_pad": 0,
    }, device)

    lowres_bhwc = _bchw_to_bhwc_numpy(lowres_bchw)
    outputs = []
    for start in range(0, len(lowres_bhwc), batch_size):
        chunk = lowres_bhwc[start:start + batch_size]
        outputs.append(sr_engine.upscale(chunk, target_hw=target_hw))

    del sr_engine
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return _bhwc_numpy_to_bchw(np.concatenate(outputs, axis=0))

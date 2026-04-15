#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from appcorr.models.dinov3.hub.backbones import dinov3_vitb16


DEFAULT_DATA_ROOT = "~/data/imagenet_val"
DEFAULT_WEIGHTS = "~/cjpark/weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_LAYERS_TO_ANALYZE = 12
LAYER_STAGE_VARIANTS = ("attn_out", "ffn_out", "output")
TOP_ENERGY_RANKS = (1, 4, 8, 16, 64, 128, 192)
_TORCHVISION_NMS_STUB_LIB = None


def _purge_torchvision_modules() -> None:
    to_delete = [name for name in sys.modules if name == "torchvision" or name.startswith("torchvision.")]
    for name in to_delete:
        del sys.modules[name]


def _ensure_torchvision_nms_stub() -> None:
    global _TORCHVISION_NMS_STUB_LIB
    if _TORCHVISION_NMS_STUB_LIB is not None:
        return
    _TORCHVISION_NMS_STUB_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_NMS_STUB_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")


try:
    from torchvision import datasets, transforms
except RuntimeError as exc:
    if "operator torchvision::nms does not exist" not in str(exc):
        raise
    _purge_torchvision_modules()
    _ensure_torchvision_nms_stub()
    from torchvision import datasets, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load 256x256 ImageNet-1k images, downsample-then-upsample them, "
            f"run DINOv3 base for token projection plus attention/ffn/layer outputs through "
            f"layer{NUM_LAYERS_TO_ANALYZE}, and analyze "
            "token spectrum imbalance with SVD."
        )
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT, help="ImageNet-1k val root.")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Local DINOv3 base checkpoint path.")
    parser.add_argument("--image-size", type=int, default=256, help="Model input size.")
    parser.add_argument(
        "--downsample-size",
        type=int,
        default=128,
        help="Intermediate low-resolution size before bicubic upsampling back to image-size.",
    )
    parser.add_argument(
        "--no-downsample",
        action="store_true",
        help="Use the original image as-is without any downsample/upsample degradation.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--max-samples", type=int, default=64, help="How many images to analyze.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model/input dtype.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="dinov3_base_token_spectrum_summary.json",
        help="Where to save the summary JSON.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[dtype_arg]
    if device.type == "cpu" and dtype == torch.float16:
        raise RuntimeError("float16 on CPU is not supported for this experiment. Use float32 or bfloat16.")
    return dtype


@dataclass
class RunningCovariance:
    count: int
    mean: torch.Tensor
    m2: torch.Tensor

    @classmethod
    def create(cls, dim: int) -> "RunningCovariance":
        return cls(
            count=0,
            mean=torch.zeros(dim, dtype=torch.float64),
            m2=torch.zeros((dim, dim), dtype=torch.float64),
        )

    def update(self, batch_tokens: torch.Tensor) -> None:
        x = batch_tokens.detach().to(device="cpu", dtype=torch.float64)
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D token matrix, got shape {tuple(x.shape)}")
        if x.shape[0] == 0:
            return

        batch_count = int(x.shape[0])
        batch_mean = x.mean(dim=0)
        centered = x - batch_mean
        batch_m2 = centered.transpose(0, 1) @ centered

        if self.count == 0:
            self.count = batch_count
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        correction = torch.outer(delta, delta) * (self.count * batch_count / total_count)
        self.m2 += batch_m2 + correction
        self.mean += delta * (batch_count / total_count)
        self.count = total_count


def build_loader(data_root: str, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    dataset = datasets.ImageFolder(root=str(Path(data_root).expanduser()), transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def downsample_then_upsample(images: torch.Tensor, downsample_size: int) -> torch.Tensor:
    lowres = F.interpolate(images, size=(downsample_size, downsample_size), mode="area")
    return F.interpolate(
        lowres,
        size=images.shape[-2:],
        mode="bicubic",
        align_corners=False,
    )


def extract_stages(model: torch.nn.Module, images: torch.Tensor) -> dict[str, torch.Tensor]:
    x, (h_tokens, w_tokens) = model.prepare_tokens_with_masks(images)
    rope = model.rope_embed(H=h_tokens, W=w_tokens) if model.rope_embed is not None else None

    num_prefix_tokens = 1 + model.n_storage_tokens
    stages = {
        "token_projection": x[:, num_prefix_tokens:, :],
    }

    hidden = x
    num_layers = min(NUM_LAYERS_TO_ANALYZE, len(model.blocks))
    for layer_idx in range(num_layers):
        branch_outputs = model.blocks[layer_idx].forward_with_branch_outputs(hidden, rope)
        hidden = branch_outputs["out"]
        layer_name = f"layer{layer_idx + 1}"
        stages[f"{layer_name}_attn_out"] = branch_outputs["attn_out"][:, num_prefix_tokens:, :]
        stages[f"{layer_name}_ffn_out"] = branch_outputs["ffn_out"][:, num_prefix_tokens:, :]
        stages[f"{layer_name}_output"] = branch_outputs["out"][:, num_prefix_tokens:, :]

    return stages


def summarize_spectrum(accumulator: RunningCovariance) -> dict[str, object]:
    if accumulator.count == 0:
        raise ValueError("Cannot summarize an empty accumulator.")

    scatter = 0.5 * (accumulator.m2 + accumulator.m2.transpose(0, 1))
    energy = torch.linalg.eigvalsh(scatter).flip(0).clamp_min(0)
    singular_values = torch.sqrt(energy)
    energy_sum = energy.sum().clamp_min(1e-12)
    energy_ratio = energy / energy_sum
    max_rank = min(accumulator.count, accumulator.mean.numel())

    entropy = -(energy_ratio * energy_ratio.clamp_min(1e-12).log()).sum()
    effective_rank = torch.exp(entropy)
    stable_rank = energy_sum / energy.max().clamp_min(1e-12)
    condition_number = singular_values[0] / singular_values[-1].clamp_min(1e-12)

    summary = {
        "num_tokens_total": int(accumulator.count),
        "embed_dim": int(accumulator.mean.numel()),
        "effective_rank": float(effective_rank.item()),
        "effective_rank_ratio": float((effective_rank / max_rank).item()),
        "stable_rank": float(stable_rank.item()),
        "condition_number": float(condition_number.item()),
        "normalized_singular_values_head": [
            float(v.item()) for v in (singular_values / singular_values[0].clamp_min(1e-12))[:16]
        ],
    }
    for rank in TOP_ENERGY_RANKS:
        summary[f"top{rank}_energy_ratio"] = float(energy_ratio[: min(rank, energy_ratio.numel())].sum().item())
    return summary


def print_stage_summary(stage_name: str, summary: dict[str, object]) -> None:
    print(
        f"{stage_name:>16} | tokens={summary['num_tokens_total']:>6d} | dim={summary['embed_dim']:>4d} | "
        f"top1={summary['top1_energy_ratio']:.4f} | top4={summary['top4_energy_ratio']:.4f} | "
        f"top16={summary['top16_energy_ratio']:.4f} | top64={summary['top64_energy_ratio']:.4f} | "
        f"top128={summary['top128_energy_ratio']:.4f} | top192={summary['top192_energy_ratio']:.4f} | "
        f"erank={summary['effective_rank']:.2f} | "
        f"erank_ratio={summary['effective_rank_ratio']:.4f} | stable_rank={summary['stable_rank']:.2f}"
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    data_root = Path(args.data_root).expanduser()
    weights_path = Path(args.weights).expanduser()
    if not data_root.is_dir():
        raise FileNotFoundError(f"ImageNet root not found: {data_root}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 base weights not found: {weights_path}")

    if not args.no_downsample and args.downsample_size >= args.image_size:
        raise ValueError("--downsample-size must be smaller than --image-size.")

    loader = build_loader(
        data_root=str(data_root),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = dinov3_vitb16(pretrained=True, weights=str(weights_path))
    model.eval()
    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    embed_dim = int(model.embed_dim)

    num_layers_to_analyze = min(NUM_LAYERS_TO_ANALYZE, len(model.blocks))
    stage_names = ["token_projection"]
    for layer_idx in range(1, num_layers_to_analyze + 1):
        for variant in LAYER_STAGE_VARIANTS:
            stage_names.append(f"layer{layer_idx}_{variant}")
    stage_accumulators: dict[str, RunningCovariance] = {
        stage_name: RunningCovariance.create(embed_dim) for stage_name in stage_names
    }

    num_images = 0
    tokens_per_image = None
    total_images = min(len(loader.dataset), args.max_samples)

    degradation_mode = "identity" if args.no_downsample else "downsample_then_upsample"

    with tqdm(total=total_images, desc="Accumulating spectra", unit="img", dynamic_ncols=True) as pbar:
        with torch.inference_mode():
            for images, _ in loader:
                if num_images >= args.max_samples:
                    break

                remaining = args.max_samples - num_images
                if images.shape[0] > remaining:
                    images = images[:remaining]

                images = images.to(device=device, dtype=dtype, non_blocking=True)
                model_inputs = images if args.no_downsample else downsample_then_upsample(images, args.downsample_size)
                stages = extract_stages(model, model_inputs)

                for stage_name, tokens in stages.items():
                    if tokens_per_image is None:
                        tokens_per_image = int(tokens.shape[1])
                    stage_accumulators[stage_name].update(tokens.reshape(-1, tokens.shape[-1]))

                num_images += images.shape[0]
                pbar.update(images.shape[0])

    results = {
        "config": {
            "data_root": str(data_root),
            "weights": str(weights_path),
            "image_size": args.image_size,
            "downsample_size": None if args.no_downsample else args.downsample_size,
            "degradation_mode": degradation_mode,
            "batch_size": args.batch_size,
            "max_samples": num_images,
            "device": str(device),
            "dtype": str(dtype),
            "tokens_per_image": tokens_per_image,
            "token_subset": "patch_only",
            "num_layers_analyzed": num_layers_to_analyze,
            "stage_representations": ["token_projection", *LAYER_STAGE_VARIANTS],
            "accumulation_mode": "streaming_covariance",
        },
        "stages": {},
    }

    print("DINOv3 base patch-token spectrum imbalance (projection + attn/ffn/layer outputs)")
    print(
        f"images={num_images} | input={args.image_size} | lowres={'none' if args.no_downsample else args.downsample_size} | "
        f"patch_tokens_per_image={tokens_per_image}"
    )

    for stage_name in stage_names:
        summary = summarize_spectrum(stage_accumulators[stage_name])
        results["stages"][stage_name] = summary
        print_stage_summary(stage_name, summary)

    out_json = Path(args.out_json).expanduser()
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {out_json}")


if __name__ == "__main__":
    main()

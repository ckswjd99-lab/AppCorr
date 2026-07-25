"""Render real pi0-FAST language-to-vision pruning scores on a saved LIBERO batch.

The final panel is a pixel-space proxy for the progressive feature computation: unselected
patches show the low-resolution base while selected patches show the exact model input. The
actual implementation corrects SigLIP/Gemma feature tokens rather than constructing this image.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import numpy as np
import torch

from appcorr.models.pi0fast.progressive_model import (
    Pi0FastProgressiveModel,
    configure_policy_precision,
    install_gemma_scaling_fix,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-file", required=True)
    parser.add_argument("--checkpoint", default="lerobot/pi0fast-libero")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keep", type=float, default=0.5)
    parser.add_argument("--base-factor", type=int, default=4)
    parser.add_argument("--llm-vision-weight", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def display_image(tensor):
    image = tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    if image.min() < 0:
        image = (image + 1.0) / 2.0
    return np.clip(image, 0.0, 1.0)


def add_grid(axis, grid_size, color="white", alpha=0.28, linewidth=0.35):
    for position in range(grid_size + 1):
        coordinate = position / grid_size
        axis.plot(
            [coordinate, coordinate],
            [0, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            transform=axis.transAxes,
        )
        axis.plot(
            [0, 1],
            [coordinate, coordinate],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            transform=axis.transAxes,
        )


def selected_overlay(
    axis,
    selected,
    grid_size,
    image_shape,
    color="#00ffd0",
    linewidth=0.8,
):
    height, width = image_shape[:2]
    patch_height = height / grid_size
    patch_width = width / grid_size
    for token_index in selected:
        row, column = divmod(int(token_index), grid_size)
        axis.add_patch(
            Rectangle(
                (column * patch_width - 0.5, row * patch_height - 0.5),
                patch_width,
                patch_height,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
            )
        )


def corrected_proxy(exact, base, selected, grid_size):
    proxy = base.copy()
    height, width = exact.shape[:2]
    for token_index in selected:
        row, column = divmod(int(token_index), grid_size)
        y0 = round(row * height / grid_size)
        y1 = round((row + 1) * height / grid_size)
        x0 = round(column * width / grid_size)
        x1 = round((column + 1) * width / grid_size)
        proxy[y0:y1, x0:x1] = exact[y0:y1, x0:x1]
    return proxy


def short_task(batch):
    task = batch.get("task", [""])
    text = task[0] if isinstance(task, list) else str(task)
    text = text.removeprefix("Task: ")
    return text.split(", State:", 1)[0].strip()


def main():
    args = parse_args()
    device = torch.device(args.device)
    install_gemma_scaling_fix()

    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

    saved = torch.load(args.batch_file, map_location=device, weights_only=False)
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in saved.items()
    }

    policy = PI0FastPolicy.from_pretrained(args.checkpoint).to(device).eval()
    configure_policy_precision(policy, "float32")
    progressive = Pi0FastProgressiveModel.from_policy(
        policy,
        device,
        precision="float32",
    )

    with torch.inference_mode():
        progressive._partial_tokens_from_batch(
            batch,
            keep=args.keep,
            base_factor=args.base_factor,
            correct_text=True,
            score_mode="vit_llm_language",
            llm_vision_weight=args.llm_vision_weight,
        )
        images, image_masks = policy._preprocess_images(batch)
        base_images = [
            progressive._base_pixel(image, args.base_factor)
            for image in images
        ]

    components = progressive.last_pscore_components
    real_indices = [
        index for index, mask in enumerate(image_masks) if bool(mask.any().item())
    ]
    if not real_indices:
        raise RuntimeError("Saved batch contains no real camera images")

    token_count = next(
        components["per_image"][index]["combined"].numel()
        for index in real_indices
    )
    grid_size = int(round(token_count ** 0.5))
    if grid_size * grid_size != token_count:
        raise RuntimeError(f"Expected a square patch grid, got {token_count} tokens")

    figure, axes = plt.subplots(
        len(real_indices),
        4,
        figsize=(17.5, 4.7 * len(real_indices)),
        squeeze=False,
        constrained_layout=True,
    )
    task = short_task(saved)
    figure.suptitle(
        "pi0-FAST language → vision pruning on a real LIBERO observation\n"
        f'Instruction: “{task}”',
        fontsize=16,
        fontweight="bold",
    )

    for row_index, image_index in enumerate(real_indices):
        item = components["per_image"][image_index]
        exact = display_image(images[image_index][0])
        base = display_image(base_images[image_index][0])
        attention = item["llm_language"].float().cpu().reshape(grid_size, grid_size).numpy()
        combined = item["combined"].float().cpu().reshape(grid_size, grid_size).numpy()
        selected = item["selected"].cpu().tolist()
        proxy = corrected_proxy(exact, base, selected, grid_size)

        source_axis, attention_axis, selection_axis, proxy_axis = axes[row_index]
        source_axis.imshow(exact)
        source_axis.set_title(f"Camera {image_index + 1}: exact model input")
        add_grid(source_axis, grid_size)

        attention_axis.imshow(exact, alpha=0.42)
        positive = attention[attention > 0]
        if positive.size and positive.max() > positive.min():
            normalization = colors.LogNorm(
                vmin=max(float(np.percentile(positive, 2)), np.finfo(float).tiny),
                vmax=float(positive.max()),
            )
        else:
            normalization = None
        heatmap = attention_axis.imshow(
            attention,
            cmap="magma",
            alpha=0.78,
            interpolation="nearest",
            extent=(-0.5, exact.shape[1] - 0.5, exact.shape[0] - 0.5, -0.5),
            norm=normalization,
        )
        attention_axis.set_title(
            "Actual language → vision attention\n"
            f"min={attention.min():.2e}  max={attention.max():.2e}"
        )
        figure.colorbar(heatmap, ax=attention_axis, fraction=0.046, pad=0.02)

        selection_axis.imshow(exact)
        selection_axis.imshow(
            np.ma.masked_where(
                ~np.isin(np.arange(token_count), selected).reshape(grid_size, grid_size),
                combined,
            ),
            cmap="viridis",
            alpha=0.62,
            interpolation="nearest",
            extent=(-0.5, exact.shape[1] - 0.5, exact.shape[0] - 0.5, -0.5),
        )
        selected_overlay(selection_axis, selected, grid_size, exact.shape)
        selection_axis.set_title(
            f"Actual fused-score selection\n{len(selected)} / {token_count} tokens"
        )

        proxy_axis.imshow(proxy)
        selected_overlay(proxy_axis, selected, grid_size, exact.shape)
        proxy_axis.set_title(
            "What 50% correction targets\n"
            "(pixel-space proxy of feature correction)"
        )

        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])

    figure.supxlabel(
        "Attention is averaged over Gemma layers, heads, and valid instruction/BOS queries. "
        "Selection uses geometric fusion with the SigLIP pscore (weight=1.0). "
        "Cyan outlines are the exact top-k token indices.",
        fontsize=10,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"saved={output}")
    print(f"instruction={task}")
    print(
        f"query_group={components['llm_query_group']} "
        f"llm_vision_weight={components['llm_vision_weight']}"
    )
    for image_index in real_indices:
        item = components["per_image"][image_index]
        attention = item["llm_language"].float()
        print(
            f"camera={image_index + 1} selected={item['selected'].numel()}/{token_count} "
            f"attention_min={attention.min().item():.8e} "
            f"attention_mean={attention.mean().item():.8e} "
            f"attention_max={attention.max().item():.8e}"
        )


if __name__ == "__main__":
    main()

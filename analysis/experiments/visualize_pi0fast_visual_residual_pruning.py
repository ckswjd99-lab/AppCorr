"""Render L2-to-L0 visual-residual energy × average-attention pruning on a LIBERO batch.

For patch i, the experimental score is

    sum_pixels,channels((L0 - upsample(L2))^2) * mean_SigLIP_attention_received

L0 is the exact pi0-FAST model input. L2 is its 1/4-resolution Gaussian-style base, restored to
the model resolution with the same bilinear path used by the progressive model. This differs from
the current production SigLIP pscore, whose residual term comes from hidden-state block updates.
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
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--values-output",
        help="Optional torch file for the exact component tensors and selected indices.",
    )
    return parser.parse_args()


def display_image(tensor):
    image = tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    if image.min() < 0:
        image = (image + 1.0) / 2.0
    return np.clip(image, 0.0, 1.0)


def short_task(batch):
    task = batch.get("task", [""])
    text = task[0] if isinstance(task, list) else str(task)
    return text.removeprefix("Task: ").split(", State:", 1)[0].strip()


def add_grid(axis, grid_size, color="white", alpha=0.3, linewidth=0.35):
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


def selected_overlay(axis, selected, grid_size, image_shape):
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
                edgecolor="#00ffd0",
                linewidth=0.8,
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


def positive_log_norm(values):
    positive = values[values > 0]
    if not positive.size or positive.max() <= positive.min():
        return None
    return colors.LogNorm(
        vmin=max(float(np.percentile(positive, 2)), np.finfo(float).tiny),
        vmax=float(positive.max()),
    )


def heatmap_panel(figure, axis, image, values, title, cmap):
    axis.imshow(image, alpha=0.4)
    rendered = axis.imshow(
        values,
        cmap=cmap,
        alpha=0.8,
        interpolation="nearest",
        extent=(-0.5, image.shape[1] - 0.5, image.shape[0] - 0.5, -0.5),
        norm=positive_log_norm(values),
    )
    axis.set_title(
        f"{title}\nmin={values.min():.2e}  max={values.max():.2e}"
    )
    figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.02)


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

    # Run the stock progressive scorer once to populate the real SigLIP received-attention.
    # Its selected indices are deliberately ignored below: this experiment ranks the new energy
    # metric without changing the production selector.
    with torch.inference_mode():
        progressive._partial_tokens_from_batch(
            batch,
            keep=args.keep,
            base_factor=args.base_factor,
            correct_text=True,
            score_mode="vit",
        )
        images, image_masks = policy._preprocess_images(batch)
        base_images = [
            progressive._base_pixel(image, args.base_factor)
            for image in images
        ]

    real_indices = [
        index for index, mask in enumerate(image_masks) if bool(mask.any().item())
    ]
    values_dump = {
        "instruction": short_task(saved),
        "metric": "patch_sum((L0 - upsample(L2))^2) * mean_siglip_received_attention",
        "keep": args.keep,
        "base_factor": args.base_factor,
        "per_image": [],
    }
    per_image = []
    for image_index in real_indices:
        average_attention = progressive.cache_feature[
            f"img{image_index}_avg_attn"
        ][0].float()
        token_count = average_attention.numel()
        grid_size = int(round(token_count ** 0.5))
        if grid_size * grid_size != token_count:
            raise RuntimeError(f"Expected a square patch grid, got {token_count} tokens")
        residual_energy = progressive._patch_visual_residual_energy(
            images[image_index],
            base_images[image_index],
            token_count,
        )[0]
        pscore = residual_energy * average_attention
        selected_count = max(
            1,
            min(int(round(args.keep * token_count)), token_count),
        )
        selected = torch.topk(pscore, selected_count).indices.sort().values
        item = {
            "visual_residual_patch_energy": residual_energy.detach().cpu(),
            "average_attention": average_attention.detach().cpu(),
            "pscore": pscore.detach().cpu(),
            "selected": selected.detach().cpu(),
        }
        per_image.append(item)
        values_dump["per_image"].append(item)

    token_count = per_image[0]["pscore"].numel()
    grid_size = int(round(token_count ** 0.5))
    if grid_size * grid_size != token_count:
        raise RuntimeError(f"Expected a square patch grid, got {token_count} tokens")

    figure, axes = plt.subplots(
        len(real_indices),
        5,
        figsize=(21.5, 4.5 * len(real_indices)),
        squeeze=False,
        constrained_layout=True,
    )
    instruction = values_dump["instruction"]
    figure.suptitle(
        "pi0-FAST L2→L0 visual-residual energy × attention pruning on real LIBERO\n"
        f'Instruction: “{instruction}”',
        fontsize=16,
        fontweight="bold",
    )

    for row_index, (image_index, item) in enumerate(zip(real_indices, per_image)):
        exact = display_image(images[image_index][0])
        base = display_image(base_images[image_index][0])
        energy = item["visual_residual_patch_energy"].reshape(
            grid_size,
            grid_size,
        ).numpy()
        attention = item["average_attention"].reshape(grid_size, grid_size).numpy()
        pscore = item["pscore"].reshape(grid_size, grid_size).numpy()
        selected = item["selected"].tolist()
        proxy = corrected_proxy(exact, base, selected, grid_size)
        (
            source_axis,
            energy_axis,
            attention_axis,
            selection_axis,
            proxy_axis,
        ) = axes[row_index]

        source_axis.imshow(exact)
        source_axis.set_title(f"Camera {image_index + 1}: exact model input")
        add_grid(source_axis, grid_size)
        heatmap_panel(
            figure,
            energy_axis,
            exact,
            energy,
            "Visual residual energy  Σ(L0−L2)²",
            "viridis",
        )
        heatmap_panel(
            figure,
            attention_axis,
            exact,
            attention,
            "SigLIP avg. received attention",
            "magma",
        )

        selection_axis.imshow(exact)
        selection_axis.imshow(
            np.ma.masked_where(
                ~np.isin(np.arange(token_count), selected).reshape(
                    grid_size,
                    grid_size,
                ),
                pscore,
            ),
            cmap="plasma",
            alpha=0.64,
            interpolation="nearest",
            extent=(-0.5, exact.shape[1] - 0.5, exact.shape[0] - 0.5, -0.5),
            norm=positive_log_norm(pscore),
        )
        selected_overlay(selection_axis, selected, grid_size, exact.shape)
        selection_axis.set_title(
            "New pscore = visual energy × avg. attention\n"
            f"{len(selected)} / {token_count} tokens"
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
        "Visual energy is measured per 14×14 patch between the real L0 input and its "
        "1/4-resolution L2 base. Attention comes from the real L2 SigLIP approximate pass. "
        "Cyan outlines are top-k under the new metric; production selection is unchanged.",
        fontsize=10,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    values_output = (
        Path(args.values_output)
        if args.values_output
        else output.with_suffix(".pt")
    )
    torch.save(values_dump, values_output)
    print(f"saved_figure={output}")
    print(f"saved_values={values_output}")
    print(f"instruction={instruction}")
    for image_index, item in zip(real_indices, per_image):
        energy = item["visual_residual_patch_energy"]
        attention = item["average_attention"]
        score = item["pscore"]
        print(
            f"camera={image_index + 1} selected={item['selected'].numel()}/{token_count} "
            f"energy_mean={energy.mean().item():.8e} "
            f"attention_mean={attention.mean().item():.8e} "
            f"pscore_min={score.min().item():.8e} "
            f"pscore_mean={score.mean().item():.8e} "
            f"pscore_max={score.max().item():.8e}"
        )


if __name__ == "__main__":
    main()

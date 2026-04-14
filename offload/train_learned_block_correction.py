import argparse
import contextlib
import json
import os
import sys
from typing import Any, Dict

import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())

from offload.common import ExperimentConfig
from offload.mobile.dataset import get_dataset_loader
from offload.server.model import get_model_executor
from appcorr.models.dinov3.layers.learned_correction import (
    collect_learned_block_state_dict,
    get_learned_block_layers,
    iter_learned_block_modules,
)


def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path, "r") as f:
        config_data = json.load(f)

    if "image_shape" in config_data:
        config_data["image_shape"] = tuple(config_data["image_shape"])
    if "patch_size" in config_data:
        config_data["patch_size"] = tuple(config_data["patch_size"])
    return ExperimentConfig(**config_data)


def get_vit_backbone(executor) -> torch.nn.Module:
    if hasattr(executor.model, "backbone"):
        return executor.model.backbone
    if hasattr(executor, "_get_vit_backbone"):
        return executor._get_vit_backbone()
    raise ValueError("Unable to locate ViT backbone from executor.")


def normalize_images(images: torch.Tensor, executor, device: torch.device, model_dtype: torch.dtype) -> torch.Tensor:
    images = images.to(device=device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    else:
        images = images.float()
    images = (images - executor.norm_mean) / executor.norm_std
    return images.to(dtype=model_dtype)


def build_old_input(images: torch.Tensor, source_level: int) -> torch.Tensor:
    if int(source_level) <= 0:
        return images
    input_dtype = images.dtype
    images = images.float()
    down = torch.nn.functional.interpolate(
        images,
        scale_factor=2 ** (-int(source_level)),
        mode="bicubic",
        align_corners=False,
    )
    return torch.nn.functional.interpolate(
        down,
        scale_factor=2 ** int(source_level),
        mode="bicubic",
        align_corners=False,
    ).to(dtype=input_dtype)


def flatten_cosine(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.float().reshape(pred.shape[0], -1)
    target_flat = target.float().reshape(target.shape[0], -1)
    return F.cosine_similarity(pred_flat, target_flat, dim=1).mean()


def mean_sample_l2(x: torch.Tensor) -> torch.Tensor:
    flat = x.float().reshape(x.shape[0], -1)
    return torch.linalg.vector_norm(flat, dim=1).mean()


def resolve_checkpoint_load_path(appcorr_options: Dict[str, Any]) -> str:
    path = (
        appcorr_options.get("learned_checkpoint_load_path")
        or appcorr_options.get("learned_checkpoint_path")
        or appcorr_options.get("learned_checkpoint_save_path")
    )
    return os.path.expanduser(path) if path else ""


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        param.numel()
        for param in module.parameters()
        if (param.requires_grad or not trainable_only)
    )


def format_params(num_params: int) -> str:
    return f"{num_params:,} ({num_params / 1_000_000:.3f}M)"


def format_layer_list(layer_indices: list[int]) -> str:
    if not layer_indices:
        return "[]"
    if len(layer_indices) <= 12:
        return "[" + ", ".join(str(idx) for idx in layer_indices) + "]"
    head = ", ".join(str(idx) for idx in layer_indices[:6])
    tail = ", ".join(str(idx) for idx in layer_indices[-3:])
    return f"[{head}, ..., {tail}]"


def build_dataloader(config: ExperimentConfig, data_root: str):
    dataset_kwargs = getattr(config, "dataset_kwargs", {})
    image_size = config.image_shape[0] if config.image_shape else 256
    dataset_loader = get_dataset_loader(
        config.dataset_name,
        data_root,
        batch_size=config.batch_size,
        image_size=image_size,
        **dataset_kwargs,
    )
    return dataset_loader.get_loader()


def resolve_data_root(config: ExperimentConfig, data_root: str | None) -> str:
    if config.dataset_name == "coco2017":
        if data_root:
            print(f"[LearnedCorrection] Ignoring --data for coco2017 and loading via FiftyOne.")
        return ""
    if not data_root:
        raise ValueError(f"--data is required for dataset '{config.dataset_name}'.")
    return data_root


def select_source_level(appcorr_options: Dict[str, Any]) -> int:
    levels = [int(level) for level in appcorr_options.get("pyramid_levels", [0])]
    for level in levels:
        if level > 0:
            return level
    return levels[0] if levels else 0


def resolve_checkpoint_path(appcorr_options: Dict[str, Any], training: bool) -> str:
    if training and appcorr_options.get("learned_checkpoint_save_path"):
        return os.path.expanduser(appcorr_options["learned_checkpoint_save_path"])
    if appcorr_options.get("learned_checkpoint_path"):
        return os.path.expanduser(appcorr_options["learned_checkpoint_path"])
    return os.path.expanduser(
        appcorr_options.get(
            "learned_checkpoint_load_path",
            "checkpoints/learned_block_correction.pt",
        )
    )


def save_checkpoint(
    vit_backbone: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    save_path: str,
    epoch: int,
    step: int,
    appcorr_options: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "state_dict": collect_learned_block_state_dict(vit_backbone),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "appcorr_options": appcorr_options,
    }
    torch.save(payload, save_path)
    print(f"[LearnedCorrection] checkpoint save path: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate learned block correction.")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--data", type=str, default=None, help="Dataset root. Not used for coco2017/FiftyOne.")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device is not None:
        config.device = args.device

    appcorr_options = config.get_appcorr_options()
    if appcorr_options["correction_mode"] != "learned_block":
        raise ValueError("Set appcorr_kwargs.correction_mode to 'learned_block' for learned correction training/eval.")

    device_str = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[LearnedCorrection] Using device: {device}")

    executor = get_model_executor(config.model_name, device)
    executor.load_model(config.model_name, config)
    vit_backbone = get_vit_backbone(executor)
    selected_layers = get_learned_block_layers(appcorr_options, max_layers=len(vit_backbone.blocks))
    if not selected_layers:
        raise ValueError("No valid learned_correction_layers were selected.")

    selected_blocks = []
    trainable_params = []
    for layer_idx in selected_layers:
        block = vit_backbone.blocks[layer_idx]
        predictor = getattr(block, "learned_block_delta", None)
        if predictor is None:
            raise RuntimeError(f"learned_block_delta was not attached to layer {layer_idx}.")
        selected_blocks.append((layer_idx, block, predictor))
        trainable_params.extend(list(predictor.parameters()))

    for param in executor.model.parameters():
        param.requires_grad_(False)
    for _, module in iter_learned_block_modules(vit_backbone):
        for param in module.parameters():
            param.requires_grad_(True)

    model_param_count = count_parameters(executor.model)
    learned_param_count = sum(count_parameters(predictor) for _, _, predictor in selected_blocks)
    trainable_param_count = count_parameters(executor.model, trainable_only=True)

    training = bool(appcorr_options.get("learned_train", False))
    executor.model.eval()
    for _, _, predictor in selected_blocks:
        predictor.train(training)

    optimizer = None
    if training:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=appcorr_options["learned_train_lr"],
            weight_decay=appcorr_options["learned_train_weight_decay"],
        )

    data_root = resolve_data_root(config, args.data)
    dataloader = build_dataloader(config, data_root)
    source_level = select_source_level(appcorr_options)
    model_dtype = next(vit_backbone.parameters()).dtype if device.type == "cuda" else torch.float32
    checkpoint_path = resolve_checkpoint_path(appcorr_options, training=training)
    checkpoint_load_path = resolve_checkpoint_load_path(appcorr_options)
    if training and optimizer is not None and checkpoint_load_path:
        if os.path.exists(checkpoint_load_path):
            checkpoint = torch.load(checkpoint_load_path, map_location="cpu")
            optimizer_state = checkpoint.get("optimizer") if isinstance(checkpoint, dict) else None
            if optimizer_state:
                try:
                    optimizer.load_state_dict(optimizer_state)
                    print(f"[LearnedCorrection] Restored optimizer state from {checkpoint_load_path}")
                except ValueError:
                    print(
                        f"[LearnedCorrection] Skipping optimizer state from {checkpoint_load_path} "
                        "because the trainable parameter set changed."
                    )
        else:
            print(f"[LearnedCorrection] No checkpoint at {checkpoint_load_path}; training from scratch.")

    print(f"[LearnedCorrection] source level: {source_level}")
    print(f"[LearnedCorrection] mode: {'train' if training else 'eval'}")
    print(
        "[LearnedCorrection] "
        f"learned_layers={format_layer_list(selected_layers)} "
        f"num_learned_layers={len(selected_layers)} "
        f"model_params={format_params(model_param_count)} "
        f"learned_module_params={format_params(learned_param_count)} "
        f"trainable_params={format_params(trainable_param_count)}"
    )

    global_step = 0
    num_epochs = appcorr_options["learned_train_epochs"] if training else 1
    step_limit = appcorr_options["learned_train_steps_per_epoch"]
    aux_enabled = (
        appcorr_options["learned_loss_weight_attn"] > 0.0
        or appcorr_options["learned_loss_weight_ffn"] > 0.0
    )
    last_saved_step = -1
    log_prefix = "[LearnedCorrection][train-vs-exact]" if training else "[LearnedCorrection][eval-vs-exact]"

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_dx_err = 0.0
        epoch_dx_rel_err = 0.0
        epoch_dx_cos = 0.0
        epoch_none_dx_err = 0.0
        epoch_improve_vs_none = 0.0
        epoch_da_err = 0.0
        epoch_da_rel_err = 0.0
        epoch_dm_err = 0.0
        epoch_dm_rel_err = 0.0

        for batch_idx, (images, _) in enumerate(dataloader):
            if step_limit > 0 and batch_idx >= step_limit:
                break

            x_new_img = normalize_images(images, executor, device, model_dtype)
            x_old_img = build_old_input(x_new_img, source_level)

            with torch.no_grad():
                x_old, hw_tuple = vit_backbone.prepare_tokens_with_masks(x_old_img, None)
                x_new, _ = vit_backbone.prepare_tokens_with_masks(x_new_img, None)
                rope = vit_backbone.rope_embed(H=hw_tuple[0], W=hw_tuple[1]) if vit_backbone.rope_embed is not None else None

                cached_old_outputs = {}
                cached_new_outputs = {}
                x_old_layer = x_old
                x_new_layer = x_new
                max_layer = selected_layers[-1]
                selected_layer_set = set(selected_layers)
                for layer_idx in range(max_layer + 1):
                    block = vit_backbone.blocks[layer_idx]
                    old_outputs = block.forward_with_branch_outputs(x_old_layer, rope=rope)
                    new_outputs = block.forward_with_branch_outputs(x_new_layer, rope=rope)
                    if layer_idx in selected_layer_set:
                        cached_old_outputs[layer_idx] = old_outputs
                        cached_new_outputs[layer_idx] = new_outputs
                    x_old_layer = old_outputs["out"]
                    x_new_layer = new_outputs["out"]

            predictor_context = contextlib.nullcontext() if training else torch.no_grad()
            with predictor_context:
                total_loss = None
                batch_layer_count = 0
                batch_dx_err = 0.0
                batch_dx_rel_err = 0.0
                batch_dx_cos = 0.0
                batch_none_dx_err = 0.0
                batch_improve_vs_none = 0.0
                batch_dA_err = 0.0
                batch_dA_rel_err = 0.0
                batch_dM_err = 0.0
                batch_dM_rel_err = 0.0

                for layer_idx, block, _predictor in selected_blocks:
                    old_outputs = cached_old_outputs[layer_idx]
                    new_outputs = cached_new_outputs[layer_idx]
                    pred_outputs = block.predict_learned_block_delta(
                        x_old=old_outputs["x"],
                        x_new=new_outputs["x"],
                        attn_out_old=old_outputs["attn_out"],
                        ln1_old=old_outputs["ln1"],
                        ln2_old=old_outputs["ln2"],
                        h_old=old_outputs["h"],
                    )

                    dA_true = new_outputs["attn_out"] - old_outputs["attn_out"]
                    dM_true = new_outputs["ffn_out"] - old_outputs["ffn_out"]
                    dx_out_true = new_outputs["out"] - old_outputs["out"]

                    layer_loss = appcorr_options["learned_loss_weight_dx"] * F.mse_loss(
                        pred_outputs["dx_out_hat"].float(),
                        dx_out_true.float(),
                    )

                    if appcorr_options["learned_loss_weight_attn"] > 0.0:
                        layer_loss = layer_loss + appcorr_options["learned_loss_weight_attn"] * F.mse_loss(
                            pred_outputs["dA_hat"].float(),
                            dA_true.float(),
                        )

                    if appcorr_options["learned_loss_weight_ffn"] > 0.0:
                        layer_loss = layer_loss + appcorr_options["learned_loss_weight_ffn"] * F.mse_loss(
                            pred_outputs["dM_hat"].float(),
                            dM_true.float(),
                        )

                    dx_cos = flatten_cosine(pred_outputs["dx_out_hat"], dx_out_true)
                    if appcorr_options["learned_loss_weight_cosine"] > 0.0:
                        layer_loss = layer_loss + appcorr_options["learned_loss_weight_cosine"] * (1.0 - dx_cos)

                    total_loss = layer_loss if total_loss is None else total_loss + layer_loss
                    batch_layer_count += 1

                    dx_err_t = mean_sample_l2(pred_outputs["dx_out_hat"] - dx_out_true)
                    dx_true_norm_t = mean_sample_l2(dx_out_true).clamp_min(1e-12)
                    dx_rel_err_t = dx_err_t / dx_true_norm_t
                    none_dx_err_t = dx_true_norm_t
                    improve_vs_none_t = 1.0 - (dx_err_t / none_dx_err_t)

                    dA_err_t = mean_sample_l2(pred_outputs["dA_hat"] - dA_true)
                    dA_true_norm_t = mean_sample_l2(dA_true).clamp_min(1e-12)
                    dA_rel_err_t = dA_err_t / dA_true_norm_t

                    dM_err_t = mean_sample_l2(pred_outputs["dM_hat"] - dM_true)
                    dM_true_norm_t = mean_sample_l2(dM_true).clamp_min(1e-12)
                    dM_rel_err_t = dM_err_t / dM_true_norm_t

                    batch_dx_err += float(dx_err_t.item())
                    batch_dx_rel_err += float(dx_rel_err_t.item())
                    batch_dx_cos += float(dx_cos.item())
                    batch_none_dx_err += float(none_dx_err_t.item())
                    batch_improve_vs_none += float(improve_vs_none_t.item())
                    batch_dA_err += float(dA_err_t.item())
                    batch_dA_rel_err += float(dA_rel_err_t.item())
                    batch_dM_err += float(dM_err_t.item())
                    batch_dM_rel_err += float(dM_rel_err_t.item())

                if batch_layer_count == 0 or total_loss is None:
                    continue

                loss = total_loss / batch_layer_count

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            dx_err = batch_dx_err / batch_layer_count
            dx_rel_err = batch_dx_rel_err / batch_layer_count
            none_dx_err = batch_none_dx_err / batch_layer_count
            improve_vs_none = batch_improve_vs_none / batch_layer_count
            dx_cos_value = batch_dx_cos / batch_layer_count
            dA_err = batch_dA_err / batch_layer_count
            dA_rel_err = batch_dA_rel_err / batch_layer_count
            dM_err = batch_dM_err / batch_layer_count
            dM_rel_err = batch_dM_rel_err / batch_layer_count

            epoch_loss += float(loss.item())
            epoch_steps += 1
            epoch_dx_err += dx_err
            epoch_dx_rel_err += dx_rel_err
            epoch_dx_cos += dx_cos_value
            epoch_none_dx_err += none_dx_err
            epoch_improve_vs_none += improve_vs_none
            epoch_da_err += dA_err
            epoch_da_rel_err += dA_rel_err
            epoch_dm_err += dM_err
            epoch_dm_rel_err += dM_rel_err
            global_step += 1

            if global_step % appcorr_options["learned_log_interval"] == 0:
                message = (
                    f"{log_prefix} "
                    f"epoch={epoch + 1} step={global_step} "
                    f"layers={batch_layer_count} "
                    f"loss={loss.item():.6f} "
                    f"||dx_out_hat-dx_out_true||={dx_err:.6f} "
                    f"rel_dx_err={dx_rel_err:.6f} "
                    f"none_dx_err={none_dx_err:.6f} "
                    f"improve_vs_none={improve_vs_none:.6f} "
                    f"cos(dx_out_hat,dx_out_true)={dx_cos_value:.6f}"
                )
                if aux_enabled:
                    message += (
                        f" ||dA_hat-dA_true||={dA_err:.6f}"
                        f" dA_rel_err={dA_rel_err:.6f}"
                        f" ||dM_hat-dM_true||={dM_err:.6f}"
                        f" dM_rel_err={dM_rel_err:.6f}"
                    )
                print(message)

        if epoch_steps == 0:
            print("[LearnedCorrection] No steps were executed.")
            break

        summary = (
            f"{log_prefix} "
            f"epoch={epoch + 1} "
            f"avg_loss={epoch_loss / epoch_steps:.6f} "
            f"avg_||dx_out_hat-dx_out_true||={epoch_dx_err / epoch_steps:.6f} "
            f"avg_rel_dx_err={epoch_dx_rel_err / epoch_steps:.6f} "
            f"avg_none_dx_err={epoch_none_dx_err / epoch_steps:.6f} "
            f"avg_improve_vs_none={epoch_improve_vs_none / epoch_steps:.6f} "
        )
        summary += f"avg_cos(dx_out_hat,dx_out_true)={epoch_dx_cos / epoch_steps:.6f}"
        if aux_enabled:
            summary += (
                f" avg_||dA_hat-dA_true||={epoch_da_err / epoch_steps:.6f}"
                f" avg_dA_rel_err={epoch_da_rel_err / epoch_steps:.6f}"
                f" avg_||dM_hat-dM_true||={epoch_dm_err / epoch_steps:.6f}"
                f" avg_dM_rel_err={epoch_dm_rel_err / epoch_steps:.6f}"
            )
        print(summary)

        if training and ((epoch + 1) % appcorr_options["learned_save_every"] == 0):
            save_checkpoint(
                vit_backbone=vit_backbone,
                optimizer=optimizer,
                save_path=checkpoint_path,
                epoch=epoch + 1,
                step=global_step,
                appcorr_options=appcorr_options,
            )
            last_saved_step = global_step

    if training and epoch_steps > 0 and global_step != last_saved_step:
        save_checkpoint(
            vit_backbone=vit_backbone,
            optimizer=optimizer,
            save_path=checkpoint_path,
            epoch=num_epochs,
            step=global_step,
            appcorr_options=appcorr_options,
        )


if __name__ == "__main__":
    main()

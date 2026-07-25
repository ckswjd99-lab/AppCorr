"""Offline stock-vs-partial parity check for the pi0-FAST official policy path.

This intentionally compares generated FAST token IDs before detokenization. It uses one loaded
PI0FastPolicy, one preprocessed batch, and one precision context for both paths, avoiding the
standalone-vs-lerobot-eval precision mismatch that can otherwise hide rollout regressions.

Examples:
    TORCHDYNAMO_DISABLE=1 python analysis/experiments/pi0fast_partial_token_parity.py
    TORCHDYNAMO_DISABLE=1 python analysis/experiments/pi0fast_partial_token_parity.py \
        --precision amp_bf16 --allow-mismatch
"""

import argparse
from contextlib import nullcontext
import time

import torch

from appcorr.models.pi0fast.progressive_model import (
    Pi0FastProgressiveModel,
    configure_policy_precision,
    install_gemma_scaling_fix,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="lerobot/pi0fast-libero")
    parser.add_argument("--dataset", default="HuggingFaceVLA/libero")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--batch-file",
        help="Use a torch-saved preprocessed batch instead of loading a dataset sample.",
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("float32", "amp_bf16"), default="float32")
    parser.add_argument("--base-factor", type=int, default=4)
    parser.add_argument("--correct-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--score-mode",
        choices=("vit", "vit_llm_vision", "vit_llm_language"),
        default="vit",
    )
    parser.add_argument("--llm-vision-weight", type=float, default=1.0)
    parser.add_argument(
        "--score-report-keep",
        type=float,
        default=0.5,
        help="Top-k fraction used only for ViT-vs-fused score overlap diagnostics.",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Report mismatches without returning a failing exit status (useful for AMP diagnosis).",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="On mismatch, compare full-correction vision and per-layer Gemma tensors to stock.",
    )
    return parser.parse_args()


def observation_from_sample(sample):
    return {
        "observation.images.image": sample["observation.images.image"][None],
        "observation.images.image2": sample["observation.images.image2"][None],
        "observation.state": sample["observation.state"][None],
        "task": [sample["task"]],
    }


def padded_tokens(tokens, length):
    result = torch.zeros(
        tokens.shape[0],
        length,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    count = min(tokens.shape[1], length)
    result[:, :count] = tokens[:, :count]
    return result


def score_diagnostics(progressive, keep):
    components = progressive.last_pscore_components
    if components["mode"] == "vit":
        return
    query_group = components["llm_query_group"]
    for image_index, item in enumerate(components["per_image"]):
        if item is None:
            continue
        vit = item["vit"].float()
        llm = item["llm_attention"].float()
        combined = item["combined"].float()
        count = max(1, min(int(round(keep * vit.numel())), vit.numel()))
        vit_top = set(torch.topk(vit, count).indices.cpu().tolist())
        fused_top = set(torch.topk(combined, count).indices.cpu().tolist())
        pearson = torch.corrcoef(torch.stack([vit, llm]))[0, 1].item()

        def ranks(values):
            result = torch.empty_like(values)
            result[values.argsort()] = torch.arange(
                values.numel(),
                dtype=values.dtype,
                device=values.device,
            )
            return result

        spearman = torch.corrcoef(torch.stack([ranks(vit), ranks(llm)]))[0, 1].item()
        overlap = len(vit_top & fused_top) / count
        print(
            f"score image={image_index} report_keep={keep:g} "
            f"llm_query_group={query_group} "
            f"vit_llm_pearson={pearson:.6f} "
            f"vit_llm_spearman={spearman:.6f} "
            f"topk_overlap={overlap:.1%}",
            flush=True,
        )


def precision_context(device, precision):
    if precision == "amp_bf16":
        return torch.autocast(device_type=device.type)
    return nullcontext()


def stock_tokens(policy, batch, capture_hidden=False):
    from lerobot.policies.pi0_fast.modeling_pi0_fast import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    captured = []
    captured_layer_input = []
    captured_layer_outputs = None
    handles = []
    if capture_hidden:
        def capture_lm_head_input(_module, inputs):
            captured.append(inputs[0].detach().clone())

        language_model = policy.model.paligemma_with_expert.paligemma.language_model
        captured_layer_outputs = [[] for _ in language_model.layers]

        def capture_first_layer_input(_module, inputs):
            captured_layer_input.append(inputs[0].detach().clone())

        def make_layer_output_hook(layer_index):
            def capture_layer_output(_module, _inputs, output):
                captured_layer_outputs[layer_index].append(output.detach().clone())

            return capture_layer_output

        handles.append(
            policy.model.paligemma_with_expert.paligemma.lm_head.register_forward_pre_hook(
                capture_lm_head_input
            )
        )
        handles.append(
            language_model.layers[0].register_forward_pre_hook(
                capture_first_layer_input
            )
        )
        for layer_index, layer in enumerate(language_model.layers):
            handles.append(
                layer.register_forward_hook(make_layer_output_hook(layer_index))
            )
    try:
        images, image_masks = policy._preprocess_images(batch)
        tokens = policy.model.sample_actions_fast_kv_cache(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            max_decoding_steps=policy.config.max_decoding_steps,
            temperature=policy.config.temperature,
        )
    finally:
        for handle in handles:
            handle.remove()
    trace = {
        "lm_head_inputs": captured,
        "layer_inputs": captured_layer_input,
        "layer_outputs": captured_layer_outputs,
    }
    return tokens, trace


def diagnose_full_correction(policy, progressive, batch, base_factor):
    from lerobot.policies.pi0_fast.modeling_pi0_fast import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    images, image_masks = policy._preprocess_images(batch)
    corrected_features = []
    corrected_raw_features = []
    cache = {}
    for image_index, image in enumerate(images):
        tag = f"diagnose_img{image_index}"
        exact = policy.model.paligemma_with_expert.embed_image(image)
        progressive.siglip.approx_forward(
            progressive._base_pixel(image, base_factor),
            cache,
            tag,
        )
        token_count = cache[f"{tag}_layer0_kv"].shape[2]
        corrected, cache = progressive.siglip.correct_forward(
            image,
            torch.arange(token_count, device=image.device),
            cache,
            tag,
        )
        corrected_raw_features.append(progressive._proj_raw(corrected))
        corrected_projected = progressive._project(corrected)
        corrected_features.append(corrected_projected)
        difference = (exact.float() - corrected_projected.float()).abs()
        print(
            f"diagnose vision[{image_index}] exact="
            f"{torch.equal(exact, corrected_projected)} "
            f"max_diff={difference.max().item():.8g} "
            f"mean_diff={difference.mean().item():.8g}",
            flush=True,
        )

    text_ids = batch[OBS_LANGUAGE_TOKENS]
    text_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
    bos = torch.full(
        (1, 1),
        progressive.tok.bos_token_id,
        device=progressive.device,
        dtype=text_ids.dtype,
    )
    text_ids = torch.cat([text_ids, bos], dim=1)
    text_mask = torch.cat(
        [
            text_mask,
            torch.ones((1, 1), dtype=text_mask.dtype, device=progressive.device),
        ],
        dim=1,
    )

    tokens_per_image = corrected_features[0].shape[1]
    raw_images = torch.cat(corrected_raw_features, dim=1)
    text_embeddings = progressive._language_at_layer_scale(text_ids)
    stock_hidden = torch.cat([raw_images, text_embeddings], dim=1)
    image_token_count = tokens_per_image * len(images)
    prefix_length = stock_hidden.shape[1]
    padding = torch.cat(
        [
            torch.ones(
                1,
                tokens_per_image,
                dtype=torch.bool,
                device=progressive.device,
            )
            if image_masks[index].any()
            else torch.zeros(
                1,
                tokens_per_image,
                dtype=torch.bool,
                device=progressive.device,
            )
            for index in range(len(images))
        ]
        + [text_mask.bool()],
        dim=1,
    )
    mask_2d = progressive.m._create_custom_attention_mask_fast(
        [("image", image_token_count), ("language", text_ids.shape[1])],
        padding,
        1,
    )
    mask_4d = progressive.m._prepare_attention_masks_4d(
        mask_2d,
        dtype=stock_hidden.dtype,
    )
    positions = torch.cumsum(padding.long(), dim=1)[0] - 1
    cos, sin = progressive._rope(positions)
    active_vision = torch.cat(
        [
            torch.arange(
                index * tokens_per_image,
                (index + 1) * tokens_per_image,
                device=progressive.device,
            )
            for index in range(len(images))
            if image_masks[index].any()
        ]
    )
    active_indices = torch.cat(
        [
            active_vision,
            torch.arange(
                image_token_count,
                prefix_length,
                device=progressive.device,
            ),
        ]
    ).sort().values

    for layer_index, stock_layer in enumerate(progressive.lm.layers):
        stock_hidden = stock_layer(
            stock_hidden,
            attention_mask=mask_4d,
            position_embeddings=(cos, sin),
        )
        corrected_hidden = progressive.cache_feature[
            f"llm{layer_index}_corrected_x"
        ]
        stock_active = stock_hidden[:, active_indices]
        difference = (stock_active.float() - corrected_hidden.float()).abs()
        print(
            f"diagnose llm[{layer_index}] exact="
            f"{torch.equal(stock_active, corrected_hidden)} "
            f"max_diff={difference.max().item():.8g} "
            f"mean_diff={difference.mean().item():.8g}",
            flush=True,
        )
    corrected_final = progressive.lm_norm(
        progressive.cache_feature["_x"][:, active_indices]
    )
    stock_final = progressive.lm_norm(stock_hidden[:, active_indices])
    difference = (stock_final.float() - corrected_final.float()).abs()
    print(
        f"diagnose prefix_final exact={torch.equal(stock_final, corrected_final)} "
        f"max_diff={difference.max().item():.8g} "
        f"mean_diff={difference.mean().item():.8g} "
        f"prefix_length={prefix_length}",
        flush=True,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    install_gemma_scaling_fix()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

    policy = PI0FastPolicy.from_pretrained(args.checkpoint).to(device).eval()
    model_precision = "float32" if args.precision == "float32" else "inherit"
    configure_policy_precision(policy, model_precision)
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    progressive = Pi0FastProgressiveModel.from_policy(
        policy,
        device,
        precision=model_precision,
    )
    progressive.capture_debug = args.diagnose
    vision_attention = (
        policy.model.paligemma_with_expert.paligemma.model.vision_tower
        .vision_model.encoder.layers[0].self_attn
    )
    text_attention = (
        policy.model.paligemma_with_expert.paligemma.language_model
        .layers[0].self_attn
    )
    print(
        "attention_backend "
        f"vision={vision_attention.config._attn_implementation} "
        f"text={text_attention.config._attn_implementation}",
        flush=True,
    )
    dataset = None
    if args.batch_file is None:
        dataset = LeRobotDataset(args.dataset, episodes=[args.episode])

    mismatches = 0
    stock_times_ms = []
    partial_times_ms = []
    stock_peak_deltas_mb = []
    partial_peak_deltas_mb = []
    with torch.inference_mode():
        if args.batch_file is not None:
            sample_indices = [0]
        else:
            sample_indices = range(
                0,
                args.num_samples * args.sample_stride,
                args.sample_stride,
            )
        for sample_index in sample_indices:
            if args.batch_file is not None:
                saved_batch = torch.load(args.batch_file, map_location=device, weights_only=False)
                batch = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in saved_batch.items()
                }
            else:
                batch = preprocessor(observation_from_sample(dataset[sample_index]))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                stock_base_memory = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            stock_started = time.perf_counter()
            with precision_context(device, args.precision):
                reference_tokens, stock_trace = stock_tokens(
                    policy,
                    batch,
                    capture_hidden=args.diagnose,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                stock_peak_deltas_mb.append(
                    (
                        torch.cuda.max_memory_allocated(device) - stock_base_memory
                    )
                    / (1024 ** 2)
                )
            stock_times_ms.append((time.perf_counter() - stock_started) * 1000)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                partial_base_memory = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            partial_started = time.perf_counter()
            with precision_context(device, args.precision):
                partial_tokens = progressive._partial_tokens_from_batch(
                    batch,
                    keep=1.0,
                    base_factor=args.base_factor,
                    correct_text=args.correct_text,
                    score_mode=args.score_mode,
                    llm_vision_weight=args.llm_vision_weight,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                partial_peak_deltas_mb.append(
                    (
                        torch.cuda.max_memory_allocated(device) - partial_base_memory
                    )
                    / (1024 ** 2)
                )
            partial_times_ms.append((time.perf_counter() - partial_started) * 1000)
            score_diagnostics(progressive, args.score_report_keep)

            partial_tokens = padded_tokens(partial_tokens, reference_tokens.shape[1])
            reference_actions = policy.detokenize_actions(
                reference_tokens,
                action_horizon=progressive.n_action_steps,
                action_dim=progressive.action_dim,
            )
            partial_actions = policy.detokenize_actions(
                partial_tokens,
                action_horizon=progressive.n_action_steps,
                action_dim=progressive.action_dim,
            )
            token_equal = torch.equal(reference_tokens, partial_tokens)
            action_diff = (reference_actions.float() - partial_actions.float()).abs()
            max_action_diff = action_diff.max().item()
            mean_action_diff = action_diff.mean().item()
            mismatches += int(not token_equal or max_action_diff != 0.0)
            print(
                f"sample={sample_index} token_equal={token_equal} "
                f"action_max_diff={max_action_diff:.8g} "
                f"action_mean_diff={mean_action_diff:.8g}",
                flush=True,
            )
            if args.diagnose and (not token_equal or max_action_diff != 0.0):
                mismatch_positions = (
                    reference_tokens != partial_tokens
                ).nonzero(as_tuple=False)
                if mismatch_positions.numel():
                    first_mismatch = int(mismatch_positions[0, 1].item())
                    print(
                        f"diagnose first_token_mismatch={first_mismatch} "
                        f"stock={int(reference_tokens[0, first_mismatch])} "
                        f"partial={int(partial_tokens[0, first_mismatch])}",
                        flush=True,
                    )
                for step, (stock_hidden, partial_hidden) in enumerate(
                    zip(
                        stock_trace["lm_head_inputs"],
                        progressive.debug_fast_hidden,
                        strict=True,
                    )
                ):
                    hidden_difference = (
                        stock_hidden.float() - partial_hidden.float()
                    ).abs()
                    print(
                        f"diagnose fast[{step}] exact="
                        f"{torch.equal(stock_hidden, partial_hidden)} "
                        f"max_diff={hidden_difference.max().item():.8g} "
                        f"mean_diff={hidden_difference.mean().item():.8g}",
                        flush=True,
                    )
                    if hidden_difference.max().item() != 0.0:
                        break
                stock_prefix_input = stock_trace["layer_inputs"][0]
                prefix_input_difference = (
                    stock_prefix_input.float()
                    - progressive.debug_corrected_prefix.float()
                ).abs()
                print(
                    "diagnose model_prefix_input exact="
                    f"{torch.equal(stock_prefix_input, progressive.debug_corrected_prefix)} "
                    f"max_diff={prefix_input_difference.max().item():.8g} "
                    f"mean_diff={prefix_input_difference.mean().item():.8g}",
                    flush=True,
                )
                for layer_index, layer_outputs in enumerate(
                    stock_trace["layer_outputs"]
                ):
                    stock_prefix_output = layer_outputs[0]
                    corrected_prefix_output = progressive.cache_feature[
                        f"llm{layer_index}_corrected_x"
                    ]
                    active_stock_output = stock_prefix_output[
                        :, progressive.debug_active_indices
                    ]
                    layer_difference = (
                        active_stock_output.float()
                        - corrected_prefix_output.float()
                    ).abs()
                    print(
                        f"diagnose hooked_llm[{layer_index}] exact="
                        f"{torch.equal(active_stock_output, corrected_prefix_output)} "
                        f"max_diff={layer_difference.max().item():.8g} "
                        f"mean_diff={layer_difference.mean().item():.8g}",
                        flush=True,
                    )
                    if layer_difference.max().item() != 0.0:
                        break
                diagnose_full_correction(
                    policy,
                    progressive,
                    batch,
                    args.base_factor,
                )

    print(
        f"PARITY precision={args.precision} score_mode={args.score_mode} "
        f"samples={len(sample_indices)} mismatches={mismatches}",
        flush=True,
    )
    print(
        "PERF "
        f"stock_mean_ms={sum(stock_times_ms) / len(stock_times_ms):.3f} "
        f"partial_mean_ms={sum(partial_times_ms) / len(partial_times_ms):.3f}",
        flush=True,
    )
    if stock_peak_deltas_mb:
        print(
            "MEMORY_DELTA "
            f"stock_peak_mb={max(stock_peak_deltas_mb):.1f} "
            f"partial_peak_mb={max(partial_peak_deltas_mb):.1f}",
            flush=True,
        )
    if mismatches and not args.allow_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

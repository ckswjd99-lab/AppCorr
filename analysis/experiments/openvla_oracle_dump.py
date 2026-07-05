"""
openvla_oracle_dump.py

Phase 0 of the progressive-VLA-prefill plan (see /home/nxclab/.claude/plans/async-stargazing-mango.md).

Loads stock OpenVLA (a LIBERO-finetuned checkpoint), grabs one real camera frame from a LIBERO
environment, runs it through the *unmodified* vision_backbone -> projector -> language_model
prefill, and dumps every intermediate tensor we'll need as ground truth for later phases:
per-block DINOv2/SigLIP hidden states (via forward hooks), the concatenated patch features,
the projected embeddings, the full multimodal `inputs_embeds`, and the prefill `past_key_values`
(all 32 Llama layers) plus per-layer hidden states.

Run (from repo root, in the `openvla` conda env, with LIBERO's software-EGL env vars set):

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 MUJOCO_EGL_ALLOW_ANY_DEVICE=1 USE_TF=0 USE_TORCH=1 \
    python analysis/experiments/openvla_oracle_dump.py \
        --checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --task-suite libero_spatial --task-id 0
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Dump stock OpenVLA prefill intermediates as an oracle.")
    parser.add_argument("--checkpoint", type=str, default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--task-suite", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--center-crop", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "analysis" / "logs" / "openvla_oracle" / "oracle.pt"),
    )
    return parser.parse_args()


def get_one_libero_frame(task_suite_name: str, task_id: int):
    """Grabs a single real camera frame + task instruction from a LIBERO environment."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)

    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(init_states[0])
    # A few no-op steps so objects settle, matching run_libero_eval.py's num_steps_wait.
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    img = obs["agentview_image"][::-1, ::-1]  # matches libero_utils.get_libero_image's 180-degree rotation
    env.close()
    return img, task.language


def center_crop_and_resize(image: Image.Image, crop_scale: float) -> Image.Image:
    """Mirrors experiments/robot/openvla_utils.py's crop_and_resize (PIL/numpy version we wrote this session)."""
    import math

    orig_width, orig_height = image.size
    new_height = orig_height * math.sqrt(crop_scale)
    new_width = orig_width * math.sqrt(crop_scale)
    top = (orig_height - new_height) / 2
    left = (orig_width - new_width) / 2
    cropped = image.crop((left, top, left + new_width, top + new_height))
    return cropped.resize((224, 224), Image.BILINEAR)


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[oracle] Grabbing one real frame from {args.task_suite} task {args.task_id}...")
    img_np, task_description = get_one_libero_frame(args.task_suite, args.task_id)
    image = Image.fromarray(img_np).convert("RGB")
    if args.center_crop:
        image = center_crop_and_resize(image, 0.9).convert("RGB")

    print(f"[oracle] Loading {args.checkpoint}...")
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.checkpoint,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()

    # === Forward hooks on every DINOv2 / SigLIP block, capturing per-block output hidden states ===
    # Note: the HF-exported model (modeling_prismatic.py::PrismaticVisionBackbone) names the two
    # towers `featurizer` ("alpha" = DINOv2) and `fused_featurizer` ("beta" = SigLIP), and expects
    # a single channel-stacked `pixel_values` tensor [B, 6, H, W] rather than a dict of two tensors
    # (that dict format is only used by the training-time `prismatic` repo's DinoSigLIPViTBackbone).
    vision_backbone = vla.vision_backbone
    dino_featurizer = vision_backbone.featurizer
    siglip_featurizer = vision_backbone.fused_featurizer

    captured = {"dino_blocks": {}, "siglip_blocks": {}}

    def make_hook(store: dict, idx: int):
        def hook(module, inputs, output):
            store[idx] = output.detach().to(torch.float32).cpu()

        return hook

    handles = []
    for i, blk in enumerate(dino_featurizer.blocks):
        handles.append(blk.register_forward_hook(make_hook(captured["dino_blocks"], i)))
    for i, blk in enumerate(siglip_featurizer.blocks):
        handles.append(blk.register_forward_hook(make_hook(captured["siglip_blocks"], i)))

    projector_io = {}

    def projector_hook(module, inputs, output):
        projector_io["input"] = inputs[0].detach().to(torch.float32).cpu()
        projector_io["output"] = output.detach().to(torch.float32).cpu()

    handles.append(vla.projector.register_forward_hook(projector_hook))

    # === Build prompt exactly as get_vla_action() does ===
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    print(f"[oracle] Prompt: {prompt!r}")
    print(f"[oracle] input_ids shape: {inputs['input_ids'].shape}, pixel_values shape: {inputs['pixel_values'].shape}")

    with torch.no_grad():
        outputs = vla(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs["pixel_values"],
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    for h in handles:
        h.remove()

    n_dino_blocks_run = len(captured["dino_blocks"])
    n_siglip_blocks_run = len(captured["siglip_blocks"])
    print(f"[oracle] DINOv2 blocks executed: {n_dino_blocks_run} / {len(dino_featurizer.blocks)}")
    print(f"[oracle] SigLIP blocks executed: {n_siglip_blocks_run} / {len(siglip_featurizer.blocks)}")
    print(f"[oracle] projector input/output shapes: {projector_io['input'].shape} -> {projector_io['output'].shape}")
    print(f"[oracle] LLM hidden_states: {len(outputs.hidden_states)} layers, shape {outputs.hidden_states[0].shape}")
    print(f"[oracle] LLM past_key_values: {len(outputs.past_key_values)} layers")
    k0, v0 = outputs.past_key_values[0]
    print(f"[oracle] layer-0 K/V shape: {tuple(k0.shape)} / {tuple(v0.shape)}")

    dump = {
        "checkpoint": args.checkpoint,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "task_description": task_description,
        "prompt": prompt,
        "input_ids": inputs["input_ids"].detach().cpu(),
        "attention_mask": inputs.get("attention_mask").detach().cpu() if inputs.get("attention_mask") is not None else None,
        "pixel_values": inputs["pixel_values"].detach().to(torch.float32).cpu(),
        "dino_block_outputs": captured["dino_blocks"],
        "siglip_block_outputs": captured["siglip_blocks"],
        "dino_num_pretokens": getattr(dino_featurizer, "num_prefix_tokens", None),
        "siglip_num_pretokens": getattr(siglip_featurizer, "num_prefix_tokens", None),
        "projector_input": projector_io["input"],
        "projector_output": projector_io["output"],
        "llm_hidden_states": [h.detach().to(torch.float32).cpu() for h in outputs.hidden_states],
        "llm_past_key_values": [(k.detach().to(torch.float32).cpu(), v.detach().to(torch.float32).cpu()) for k, v in outputs.past_key_values],
        "logits": outputs.logits.detach().to(torch.float32).cpu(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dump, out_path)
    print(f"[oracle] Saved oracle dump to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

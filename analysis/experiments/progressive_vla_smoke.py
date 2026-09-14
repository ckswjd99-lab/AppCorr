"""
progressive_vla_smoke.py

Phase 3 smoke test (see /home/nxclab/.claude/plans/async-stargazing-mango.md): drives
`OpenVLAProgressiveModel` (appcorr/models/openvla/progressive_model.py) through a real progressive
correction schedule on a real LIBERO camera frame, and reports the *decoded 7-DoF action* at each
correction ratio against the ground-truth action from the real, unmodified `predict_action()` --
directly answering: does a partially corrected prefill produce a meaningful action, on real image
data, end to end through both vision towers and the LLM?

Patch correction order: ranked by per-patch pixel-space residual magnitude between the true and
blurred image (largest-residual-first), matching AppCorr's own design principle
(dinov3_classifier.py's `_apply_image_residual_token_pruning`) rather than a random or raster order.

Run (from repo root, in the `openvla` conda env, with LIBERO's software-EGL env vars set):
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 MUJOCO_EGL_ALLOW_ANY_DEVICE=1 USE_TF=0 USE_TORCH=1 \
    python analysis/experiments/progressive_vla_smoke.py \
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
import torch.nn.functional as F
from PIL import Image

from appcorr.models.openvla.progressive_model import OpenVLAProgressiveModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--task-suite", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--blur-factor", type=int, default=4, help="Downsample factor simulating the low-res base layer.")
    parser.add_argument("--ratios", type=str, default="0.1,0.25,0.5,0.75,1.0", help="Cumulative correction ratios.")
    return parser.parse_args()


def get_one_libero_frame(task_suite_name: str, task_id: int):
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
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    img = obs["agentview_image"][::-1, ::-1]
    env.close()
    return img, task.language


def blur_downup(x: torch.Tensor, factor: int) -> torch.Tensor:
    B, C, H, W = x.shape
    low = F.interpolate(x, size=(H // factor, W // factor), mode="bilinear", align_corners=False)
    return F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)


def rank_patches_by_residual(true_px: torch.Tensor, blur_px: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
    """Ranks the 16x16=256 patches by mean-abs pixel residual (true vs. blurred), descending.
    Uses only the DINOv2 channel group (first 3 channels) -- both towers share the same 224px/14
    patch grid, so one residual ranking is used for both."""
    residual = (true_px[:, :3] - blur_px[:, :3]).abs()  # [1, 3, 224, 224]
    B, C, H, W = residual.shape
    gh, gw = H // patch_size, W // patch_size
    residual = residual.view(B, C, gh, patch_size, gw, patch_size)
    per_patch = residual.mean(dim=(1, 3, 5)).view(gh * gw)  # [256]
    return torch.argsort(per_patch, descending=True)


def action_str(a: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:+.3f}" for v in a) + "]"


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ratios = [float(r) for r in args.ratios.split(",")]

    print(f"[smoke] Grabbing one real frame from {args.task_suite} task {args.task_id}...")
    img_np, task_description = get_one_libero_frame(args.task_suite, args.task_id)
    image = Image.fromarray(img_np).convert("RGB")

    print(f"[smoke] Loading {args.checkpoint}...")
    model = OpenVLAProgressiveModel(args.checkpoint, device, unnorm_key=args.task_suite)
    model.start_session(image, task_description, center_crop=True)
    print(f"[smoke] Prompt task: {task_description!r}")

    true_px = model.reference_pixel_values.to(device=device, dtype=torch.bfloat16)

    print("\n[smoke] Ground truth: running the real, unmodified predict_action() on the full-res image...")
    with torch.no_grad():
        gt_action = model.vla.predict_action(
            input_ids=model.input_ids, pixel_values=true_px, unnorm_key=args.task_suite, do_sample=False
        )
    print(f"    ground truth action: {action_str(gt_action)}")

    blur_px = blur_downup(true_px.float(), factor=args.blur_factor).to(dtype=torch.bfloat16)
    patch_rank = rank_patches_by_residual(true_px.float(), blur_px.float())
    num_patches = patch_rank.shape[0]

    print(f"\n[smoke] === Round 0: approx() on {1.0/args.blur_factor:.0%}-scale blurred image (0% corrected) ===")
    model.approx_forward(blur_px)
    action0 = model.decode_action()
    err0 = np.linalg.norm(action0 - gt_action)
    print(f"    action: {action_str(action0)}  |  L2 error vs ground truth: {err0:.4f}")

    already_corrected = 0
    results = [(0.0, err0)]
    for ratio in ratios:
        target_count = max(1, int(round(num_patches * ratio)))
        new_patch_idx = patch_rank[already_corrected:target_count]
        if new_patch_idx.numel() == 0:
            continue
        already_corrected = target_count

        print(f"\n[smoke] === Round: correct_forward, cumulative {ratio:.0%} of patches (by residual rank) ===")
        model.correct_forward(true_px, new_patch_idx)
        action = model.decode_action()
        err = np.linalg.norm(action - gt_action)
        print(f"    action: {action_str(action)}  |  L2 error vs ground truth: {err:.4f}")
        results.append((ratio, err))

    print("\n[smoke] === Summary: action L2 error vs. correction ratio ===")
    print("    ratio    L2 error")
    for ratio, err in results:
        print(f"    {ratio:5.0%}    {err:.4f}")


if __name__ == "__main__":
    main()

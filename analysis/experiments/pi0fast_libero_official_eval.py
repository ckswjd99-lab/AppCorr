"""
pi0fast_libero_official_eval.py

Parity-guaranteed LIBERO rollout eval for pi0-FAST progressive vision. Instead of reimplementing the
rollout loop, this is a thin launcher around lerobot's OWN `lerobot-eval` (`eval_main`): the ONLY
thing swapped for the progressive condition is `PI0FastPaliGemma.embed_image`, replaced with a
SigLIP low-res-base-approx + per-group patch-correct forward. So:

  - STOCK  (no PROG_KEEP)  == lerobot's official eval, byte-for-byte. This is the checkpoint's real
                             baseline (~40% on libero_spatial task 0; model card = 82.5% averaged
                             over all tasks).
  - PROG   (PROG_KEEP set) == the identical official pipeline with only the vision encoder made
                             progressive -> stock-vs-prog is apples-to-apples in ONE harness, so
                             equivalence to the official eval holds BY CONSTRUCTION.

Requires the double-scaling fix (installed on import of progressive_model). Two hard requirements
learned the hard way: `--policy.use_amp=true` (autocast; without it the SigLIP Float-vs-BFloat16
error fires) and `--env.task_ids=[N]` to build ONE task env at a time (building all 10 at once
deadlocks EGL on the B200). LIBERO renders at 360x360, max_steps=280 for libero_spatial (lerobot's
own env config).

Env vars: TASK_ID (default 0), N_EP (default 5), PROG_KEEP (unset=stock; e.g. "0.5"), PROG_BASE
(base downsample factor, default 4). First CLI arg = output dir.

Run:
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 TORCHDYNAMO_DISABLE=1 TASK_ID=0 N_EP=10 \
        python analysis/experiments/pi0fast_libero_official_eval.py /tmp/out_stock
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 TORCHDYNAMO_DISABLE=1 TASK_ID=0 N_EP=10 PROG_KEEP=0.5 \
        python analysis/experiments/pi0fast_libero_official_eval.py /tmp/out_prog50
"""

import os
import sys

import torch
import torch.nn.functional as F

from appcorr.models.pi0fast.progressive_model import install_gemma_scaling_fix

install_gemma_scaling_fix()

_KEEP = os.environ.get("PROG_KEEP")
if _KEEP is not None:
    keep = float(_KEEP)
    base_factor = int(os.environ.get("PROG_BASE", "4"))
    import lerobot.policies.pi0_fast.modeling_pi0_fast as MOD
    from appcorr.models.pi0fast.siglip_vision import ApproxCorrectSiglipBackbone

    _st = {}

    def _prog_embed_image(self, image):
        """Progressive replacement for PI0FastPaliGemma.embed_image: SigLIP low-res base approx +
        correct only `keep` fraction of patches (top-to-bottom), then projector / sqrt(hidden)
        (== get_image_features). Called once per image; cache reset every 3 calls."""
        if "fork" not in _st:
            vm = self.paligemma.model.vision_tower.vision_model
            _st["fork"] = ApproxCorrectSiglipBackbone(vm).to(image.device)
            _st["proj"] = self.paligemma.model.multi_modal_projector
            _st["scale"] = self.paligemma.config.text_config.hidden_size ** 0.5
            _st["ctr"] = 0
        i = _st["ctr"] % 3
        if i == 0:
            _st["cache"] = {}
        cache = _st["cache"]
        tag = f"e{i}"
        h, w = image.shape[-2:]
        base = F.interpolate(
            F.interpolate(image, size=(h // base_factor, w // base_factor), mode="bilinear", align_corners=False),
            size=(h, w), mode="bilinear", align_corners=False)
        _st["fork"].approx_forward(base, cache, tag)
        npatch = cache[f"{tag}_layer0_kv"].shape[2]
        k = max(1, int(round(keep * npatch)))
        feat, _ = _st["fork"].correct_forward(image, torch.arange(k, device=image.device), cache, tag)
        _st["ctr"] += 1
        return _st["proj"](feat) / _st["scale"]

    MOD.PI0FastPaliGemma.embed_image = _prog_embed_image
    print(f"[eval] PROGRESSIVE vision: keep={keep} base_factor={base_factor}", flush=True)
else:
    print("[eval] STOCK vision (== official lerobot-eval)", flush=True)

from lerobot.scripts.lerobot_eval import eval_main

sys.argv = [
    "lerobot-eval",
    "--policy.path=lerobot/pi0fast-libero",
    "--policy.device=cuda",
    "--policy.use_amp=true",
    "--env.type=libero",
    "--env.task=libero_spatial",
    f"--env.task_ids=[{os.environ.get('TASK_ID', '0')}]",
    "--eval.batch_size=1",
    f"--eval.n_episodes={os.environ.get('N_EP', '5')}",
    f"--output_dir={sys.argv[1] if len(sys.argv) > 1 else '/tmp/pi0fast_eval_out'}",
]
eval_main()

"""
pi0fast_libero_partial_token_eval.py

DINOv3-style PARTIAL-TOKEN-correct progressive vision for pi0-FAST, evaluated in the parity-guaranteed
official lerobot-eval harness (same construction as pi0fast_libero_official_eval.py: swap only the
vision path, keep lerobot's LLM/env/rollout).

One approx + one correct, pscore-selected (all inside the SigLIP ViT -- no LLM attention needed):
  1. approx: SigLIP encode a low-res base of each image, with pscore=True.
  2. pscore_i = residual_i * avg_attn_i  (ProgVFM contrib_i / DINOv3 patch_attn_prob):
       - avg_attn_i = attention patch i RECEIVES, meaned over heads, query patches, layers.
       - residual_i = L2 norm of patch i's per-block output update, meaned over layers.
  3. select the top `PTC_KEEP` fraction of patches per image by pscore.
  4. correct: SigLIP re-encodes ONLY those patches at full res (others keep the base KV).
  5. project -> get_image_features scale, inject via embed_image, run lerobot's stock FAST decode.

At PTC_KEEP=1.0 every patch is corrected => identical to stock (parity check).

Env: TASK_ID (0), N_EP (5), PTC_KEEP (0.5), PTC_BASE (base downsample factor, 4). arg1 = output dir.
Run (autocast + one-task-at-a-time EGL, same as the official eval):
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 TORCHDYNAMO_DISABLE=1 TASK_ID=0 N_EP=10 PTC_KEEP=0.5 \
        python analysis/experiments/pi0fast_libero_partial_token_eval.py /tmp/out_ptc50
"""

import os
import sys

import torch
import torch.nn.functional as F

from appcorr.models.pi0fast.progressive_model import install_gemma_scaling_fix
from appcorr.models.pi0fast.siglip_vision import ApproxCorrectSiglipBackbone

install_gemma_scaling_fix()

KEEP = float(os.environ.get("PTC_KEEP", "0.5"))
BASE_FACTOR = int(os.environ.get("PTC_BASE", "4"))


def _base_pixel(px, factor):
    h, w = px.shape[-2:]
    low = F.interpolate(px, size=(h // factor, w // factor), mode="bilinear", align_corners=False)
    return F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)


import lerobot.policies.pi0_fast.modeling_pi0_fast as MOD

_orig_predict = MOD.PI0FastPolicy.predict_action_chunk


def _partial_predict(self, batch, **kwargs):
    if not hasattr(self, "_ptc"):
        pg = self.model.paligemma_with_expert.paligemma
        dev = next(self.parameters()).device
        self._ptc = {
            "fork": ApproxCorrectSiglipBackbone(pg.model.vision_tower.vision_model).to(dev),
            "proj": pg.model.multi_modal_projector,
            "scale": pg.config.text_config.hidden_size ** 0.5,
        }
    fork, proj, scale = self._ptc["fork"], self._ptc["proj"], self._ptc["scale"]

    images, _img_masks = self._preprocess_images(batch)   # list of true [1,3,H,W]
    corrected = []
    kept_counts = []
    for img in images:
        cache = {}
        fork.approx_forward(_base_pixel(img, BASE_FACTOR), cache, "v", pscore=True)
        pscore = fork.get_pscore(cache, "v")[0]           # [N]
        npatch = pscore.shape[0]
        k = max(1, min(int(round(KEEP * npatch)), npatch))
        keep_idx = torch.topk(pscore.float(), k, largest=True).indices
        feat, _ = fork.correct_forward(img, keep_idx, cache, "v")
        corrected.append(proj(feat) / scale)              # == get_image_features
        kept_counts.append(k)

    queue = list(corrected)
    orig_embed = self.model.paligemma_with_expert.embed_image
    self.model.paligemma_with_expert.embed_image = lambda im: queue.pop(0)
    try:
        return _orig_predict(self, batch, **kwargs)
    finally:
        self.model.paligemma_with_expert.embed_image = orig_embed


MOD.PI0FastPolicy.predict_action_chunk = _partial_predict
print(f"[eval] PARTIAL-TOKEN progressive: pscore=residual*avg_attn, keep={KEEP} base_factor={BASE_FACTOR}", flush=True)

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
    f"--output_dir={sys.argv[1] if len(sys.argv) > 1 else '/tmp/pi0fast_ptc_out'}",
]
eval_main()

"""
pi0fast_libero_partial_token_eval.py

DINOv3-style PARTIAL-TOKEN correct on BOTH the SigLIP ViT and the Gemma LLM for pi0-FAST, evaluated
in the parity-guaranteed official lerobot-eval harness. Unlike the vision-only variant, this patches
PI0FastPolicy.predict_action_chunk to run Pi0FastProgressiveModel._partial_from_batch, which:
  - vision: SigLIP base approx (+pscore = residual * avg_attn) -> correct top-`keep` patches/image;
  - LLM:    bidirectional approx on the base vision features + text -> correct ONLY the selected
            vision tokens' K/V + the text permanent group (non-selected vision keep base LLM K/V) ->
            FAST-decode. At keep=1.0 this reproduces stock lerobot exactly (verified |diff|=0).

So stock vs partial-token is apples-to-apples in one harness (equivalence by construction), and the
LLM -- not just the ViT -- does partial-token recompute.

Env: TASK_ID (0), N_EP (5), PTC_KEEP (0.5), PTC_BASE (4), PTC_CORRECT_TEXT (1). arg1 = output dir.
Run:
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 TORCHDYNAMO_DISABLE=1 TASK_ID=0 N_EP=10 PTC_KEEP=0.5 \
        python analysis/experiments/pi0fast_libero_partial_token_eval.py /tmp/out_ptc50
"""

import os
import sys

from appcorr.models.pi0fast.progressive_model import install_gemma_scaling_fix

install_gemma_scaling_fix()

KEEP = float(os.environ.get("PTC_KEEP", "0.5"))
BASE_FACTOR = int(os.environ.get("PTC_BASE", "4"))
CORRECT_TEXT = os.environ.get("PTC_CORRECT_TEXT", "1") not in {"0", "false", "False"}

import lerobot.policies.pi0_fast.modeling_pi0_fast as MOD


def _ptc_predict(self, batch, **kwargs):
    if not hasattr(self, "_orch"):
        from appcorr.models.pi0fast.progressive_model import Pi0FastProgressiveModel
        self._orch = Pi0FastProgressiveModel.from_policy(self, next(self.parameters()).device)
    return self._orch._partial_from_batch(batch, keep=KEEP, base_factor=BASE_FACTOR,
                                          correct_text=CORRECT_TEXT)


MOD.PI0FastPolicy.predict_action_chunk = _ptc_predict
print(f"[eval] PARTIAL-TOKEN (ViT + LLM): keep={KEEP} base={BASE_FACTOR} correct_text={CORRECT_TEXT}", flush=True)

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

"""CPU gates for GLM-5.3-Flash that need `transformers >= 5.16` -- run WITHOUT pytest.

The pytest suites (`tests/test_glm53_tower.py`, `tests/test_glm53_prompt.py`) are the primary
gate, but they run in the `appcorr` env, whose transformers is 5.13.0 and has no
`models/glm5_next` -- so two of their checks skip there. The env that HAS glm5_next
(`/NHNHOME/share/cjpark/backup/env/appcorr-vllm-main`, transformers 5.16.1) has no pytest
installed and a broken `bin/pip`. This script is those checks, standalone:

  --mode port    `appcorr/models/glm53/vision/stock.py`'s port of the tower against
                 transformers' own `Glm5NextVisionModel` on the SAME checkpoint weights. This is
                 what pins the fork to stock: every identity in `tests/test_glm53_tower.py` is
                 relative to the port, and the port is only worth that if it IS stock.
  --mode prompt  the template / `Glm53Axis.build_inputs` / `Glm53Composer.prompt_text` /
                 `low_res_inputs` checks of `tests/test_glm53_prompt.py`.

    OMP_NUM_THREADS=8 PYTHONPATH=$PWD HF_HUB_OFFLINE=1 HF_HOME=/NHNHOME/huggingface \\
    /NHNHOME/share/cjpark/backup/env/appcorr-vllm-main/bin/python3.11 \\
        analysis/experiments/glm53_tower_gate.py --mode port --out analysis/results/glm53/gate_port.json

(`OMP_NUM_THREADS=8` is not cosmetic: the tower's per-layer GEMMs are small, and at this box's
default 72 threads the OpenMP fan-out dominates -- a full pytest pass went from "still running at
15 minutes" to minutes. Same lesson as the serving hot path.)
"""
import argparse, json, math, os, sys, time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

MODEL_ID = "zai-org/GLM-5.3-Flash"
IMAGE, START, END, THINK_OPEN, THINK_CLOSE = 154854, 154830, 154831, 154841, 154842


def _pixels(seed: int, grid=(1, 8, 12), dtype=torch.float32):
    from appcorr.models.glm53.vision.backbone import vision_config
    v = vision_config(MODEL_ID)
    t, h, w = grid
    dim = int(v.in_channels) * int(v.temporal_patch_size) * int(v.patch_size) ** 2
    g = torch.Generator().manual_seed(seed)
    return torch.randn(t * h * w, dim, generator=g).to(dtype), torch.tensor([list(grid)])


def gate_port(a):
    """The port vs transformers' `Glm5NextVisionModel`, same weights, same two epsilons.

    Not asserted bitwise: HF's attention goes through `ALL_ATTENTION_FUNCTIONS` (sdpa on a
    `[1, H, T, D]` layout split by `torch.split`) while the port calls
    `scaled_dot_product_attention` per segment, and HF's RMSNorm may be a fused kernel from the
    hub. Same arithmetic, possibly a different reduction order -- so the criterion is fp32 GEMM
    noise (rel-L2 < 1e-6) and the number is printed either way.
    """
    from appcorr.models.glm53.vision.backbone import (
        ApproxCorrectGlm5NextVisionTower, load_stock_vision_tower)
    from appcorr.models.glm53.vision.stock import tower_eps

    t0 = time.perf_counter()
    port = load_stock_vision_tower(MODEL_ID, device="cpu", dtype=torch.float32, prefer_hf=False)
    hf = load_stock_vision_tower(MODEL_ID, device="cpu", dtype=torch.float32, prefer_hf=True)
    assert type(hf).__name__ == "Glm5NextVisionModel", type(hf).__name__
    px, grid = _pixels(1)
    with torch.no_grad():
        want = hf(px, grid_thw=grid).pooler_output
        got = port(px, grid)
        fork = ApproxCorrectGlm5NextVisionTower(port).reference_forward(px, grid)
    rel = float((got - want).norm() / want.norm())
    rel_fork = float((fork - want).norm() / want.norm())
    row = {"shape": list(got.shape), "eps_port": tower_eps(port), "eps_hf": tower_eps(hf),
           "port_vs_hf_relL2": rel, "port_vs_hf_max_abs": float((got - want).abs().max()),
           "fork_vs_hf_relL2": rel_fork,
           "fork_vs_hf_max_abs": float((fork - want).abs().max()),
           "fork_vs_port_bitwise": bool(torch.equal(fork, got)),
           "wall_s": time.perf_counter() - t0}
    fails = []
    if rel >= 1e-6:
        fails.append(f"port vs HF rel-L2 {rel:.3g} >= 1e-6")
    if rel_fork >= 1e-6:
        fails.append(f"fork vs HF rel-L2 {rel_fork:.3g} >= 1e-6")
    if not row["fork_vs_port_bitwise"]:
        fails.append("fork reference_forward is not bitwise the port's forward")
    return {"_mode": "port", "_model": MODEL_ID, "row": row, "fails": fails}


def gate_prompt(a):
    """The prompt half: template shape, the `</think>` edit on both sides, positions, low-res."""
    import numpy as np
    import torch.nn as nn
    from types import SimpleNamespace
    from PIL import Image
    from transformers import AutoProcessor

    from appcorr.models.glm53.axis import Glm53Axis
    from appcorr.models.glm53.vision.backbone import resolve_snapshot
    from appcorr.models.qwen_vl_axis import QwenVLStreamingAxis

    proc = AutoProcessor.from_pretrained(resolve_snapshot(MODEL_ID))

    class _PromptAxis(Glm53Axis):
        """The real class and the real MRO (so `super()` in `build_inputs` reaches
        `Glm46VAxis` -> `QwenVLStreamingAxis`), without a 45-layer model behind it."""
        def __init__(self, processor):
            nn.Module.__init__(self)
            self.processor = processor
            self.cfg = SimpleNamespace(image_token_id=IMAGE, image_start_token_id=START,
                                       image_end_token_id=END,
                                       vision_config=SimpleNamespace(spatial_merge_size=2))
            self.image_token_id = IMAGE
            self.positions_mode = "fast"

    axis = _PromptAxis(proc)

    def img(seed=1, h=120, w=160):
        return Image.fromarray((np.random.RandomState(seed).rand(h, w, 3) * 255).astype("uint8"))

    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": "What is this?"}]}]
    text = proc.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    row, fails = {"prompt_text": text}, []

    def chk(name, cond, detail=""):
        row[name] = bool(cond)
        if not cond:
            fails.append(f"{name} {detail}".strip())

    chk("tmpl_starts_gmask_sop", text.startswith("[gMASK]<sop>"))
    chk("tmpl_has_flanked_placeholder",
        "<|begin_of_image|><|image|><|end_of_image|>" in text)
    chk("tmpl_ends_assistant_think", text.endswith("<|assistant|><think>"))
    chk("tmpl_has_no_nothink_switch", "/nothink" not in text and "enable_thinking" not in text)

    inputs = axis.build_inputs(img(), "What is this?")
    ids = inputs["input_ids"][0]
    row["seq"] = int(ids.shape[0])
    row["grid"] = [int(v) for v in inputs["image_grid_thw"][0]]
    chk("build_inputs_appends_think_close",
        int(ids[-1]) == THINK_CLOSE and int(ids[-2]) == THINK_OPEN, f"tail={ids[-3:].tolist()}")
    chk("per_token_fields_extended",
        all(inputs[k].shape[1] == ids.shape[0] for k in ("attention_mask", "mm_token_type_ids")))
    lo, n = axis._image_token_run(inputs["input_ids"])
    row["image_run"] = [lo, lo + n]
    t, h, w = row["grid"]
    chk("placeholder_expanded_by_processor", n == t * (h // 2) * (w // 2), f"{n} vs grid {row['grid']}")
    chk("sentinels_flank_the_run", int(ids[lo - 1]) == START and int(ids[lo + n]) == END)
    chk("trailing_text_rows_ge_2", ids.shape[0] - (lo + n) >= 2)

    think_on = axis.build_inputs(img(), "What is this?", think=True)
    chk("think_True_keeps_the_open_block",
        int(think_on["input_ids"][0, -1]) == THINK_OPEN
        and think_on["input_ids"].shape[1] == ids.shape[0] - 1)

    # the composer makes the same edit on the TEXT
    from appcorr.vllm_stream.client import Glm53Composer

    class _PromptComposer(Glm53Composer):
        """Real class, real MRO (`prompt_text` calls `super().prompt_text`), no engine."""
        def __init__(self, hf):
            self.hf = hf
            self.image_pad_id = IMAGE

    comp_text = _PromptComposer(proc).prompt_text("What is this?")
    ids_text = proc.tokenizer(comp_text, add_special_tokens=False)["input_ids"]
    ids_base = proc.tokenizer(text, add_special_tokens=False)["input_ids"]
    chk("composer_matches_axis_edit",
        comp_text == text + "</think>" and ids_text == ids_base + [THINK_CLOSE])

    # positions
    pos, delta = axis._positions({}, image_run=(0, 1), grid_thw=(1, 2, 2))
    chk("family_default_uses_mrope", QwenVLStreamingAxis.uses_mrope is True)
    chk("glm53_uses_mrope_false", axis.uses_mrope is False)
    chk("positions_are_none", pos is None and delta == 0, f"{pos!r}/{delta}")

    # the low-res knob is max_image_tokens, not `size`
    full = axis.build_inputs(img(2, 480, 640), "What is this?")
    small = axis.low_res_inputs(img(2, 480, 640), "What is this?", max_image_tokens=64)
    nf = int((full["input_ids"][0] == IMAGE).sum())
    ns = int((small["input_ids"][0] == IMAGE).sum())
    row["tokens_full"], row["tokens_low"] = nf, ns
    row["grid_full"] = [int(v) for v in full["image_grid_thw"][0]]
    row["grid_low"] = [int(v) for v in small["image_grid_thw"][0]]
    chk("low_res_shrinks_the_grid", ns < nf and ns <= 64, f"{nf} -> {ns}")

    # driver / gate plumbing
    from analysis.experiments.qwen_vllm_accuracy import FAMILIES, FAMILY_MAX_PX, clean_text
    from analysis.experiments.vllm_stream_gate import composer_for
    chk("driver_family_registered", "glm53" in FAMILIES)
    chk("family_max_px", FAMILY_MAX_PX["glm53"] == 6_272_000, str(FAMILY_MAX_PX.get("glm53")))
    chk("clean_text_strips_box",
        clean_text("glm53", "<|begin_of_box|>C<|end_of_box|>") == "C")
    chk("composer_for_glm53", composer_for(MODEL_ID).__name__ == "Glm53Composer")
    chk("composer_for_glm46v", composer_for("zai-org/GLM-4.6V-FP8").__name__ == "Glm46VComposer")
    try:
        composer_for("some-org/Brand-New-VL-9B")
        chk("composer_for_raises_on_unknown", False, "returned a composer")
    except ValueError:
        chk("composer_for_raises_on_unknown", True)

    return {"_mode": "prompt", "_model": MODEL_ID, "row": row, "fails": fails}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["port", "prompt"], required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"port": gate_port, "prompt": gate_prompt}[a.mode](a)
    print(json.dumps(res, indent=1, default=str))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, default=str)
        print(f"wrote {a.out}")
    print("PASS" if not res["fails"] else "FAIL: " + "; ".join(res["fails"]))
    raise SystemExit(1 if res["fails"] else 0)


if __name__ == "__main__":
    main()

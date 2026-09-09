"""CPU bitwise gate for the streaming-axis throughput levers (2026-09-08): tiny randomly
initialised Qwen2.5-VL and Qwen3.5 models (real HF classes, real forks, fp32, CPU) run the
streaming arm through the CURRENT tree and through a REFERENCE tree (`--ref-root`, the code
before the levers), and every consumed output must be `torch.equal`:

    image_embeds (the rows the LLM consumed)   final-position logits   decode_start_pos
    chunk boundaries                            corrected_groups

for groups in {1, 4}, keep in {1.0, 0.5}, two grids each (one that tiles the 2.5-VL window
exactly, one that needs the padding path), plus `positions_mode="check"` (closed-form M-RoPE vs
transformers' get_rope_index) on the current tree. No weights, no GPU, ~20 s. This is the
cheap half of the gate -- the GPU half (bf16, real towers, real images) is
qwen_axis_snapshot.py; bitwise here and there is what "the lever is a no-op on numerics" means.

  PYTHONPATH=. python analysis/experiments/qwen_axis_cpu_unittest.py \
      --ref-root /NHNHOME/share/cjpark/AppCorr-vllm
"""
import argparse, importlib, os, sys, tempfile

import torch

torch.manual_seed(0)
torch.set_num_threads(8)


def import_ref(ref_root):
    """Import the reference tree's `appcorr` package under the name `appcorr_ref` (a symlink in
    a temp dir; the forks only use relative imports, so the two trees do not mix)."""
    d = tempfile.mkdtemp(prefix="appcorr_ref_")
    os.symlink(os.path.join(ref_root, "appcorr"), os.path.join(d, "appcorr_ref"))
    sys.path.insert(0, d)
    return {
        "qwen25vl": importlib.import_module("appcorr_ref.models.qwen25vl.unified").Qwen25VLAxis,
        "qwen35": importlib.import_module("appcorr_ref.models.qwen35.unified").Qwen35Axis,
    }


def tiny_models():
    from transformers import (Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration,
                              Qwen3_5Config, Qwen3_5ForConditionalGeneration)
    vc = dict(depth=4, hidden_size=64, num_heads=4, intermediate_size=128, out_hidden_size=96,
              patch_size=14, spatial_merge_size=2, window_size=56, fullatt_block_indexes=[3],
              temporal_patch_size=2, in_channels=3)
    tc = dict(hidden_size=96, intermediate_size=192, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, vocab_size=1000, max_position_embeddings=4096,
              rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                               "mrope_section": [4, 4, 4]},
              bos_token_id=1, eos_token_id=2)
    cfg = Qwen2_5_VLConfig(vision_config=vc, text_config=tc, image_token_id=5,
                           vision_start_token_id=3, vision_end_token_id=4, vocab_size=1000,
                           bos_token_id=1, eos_token_id=2)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(1)
    m25 = Qwen2_5_VLForConditionalGeneration(cfg).eval()

    vc3 = dict(depth=4, hidden_size=64, num_heads=4, intermediate_size=128, out_hidden_size=96,
               patch_size=16, spatial_merge_size=2, temporal_patch_size=2, in_channels=3,
               num_position_embeddings=64)
    tc3 = dict(hidden_size=96, intermediate_size=192, num_hidden_layers=2, num_attention_heads=4,
               num_key_value_heads=2, vocab_size=1000, max_position_embeddings=4096,
               linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
               linear_value_head_dim=16, linear_conv_kernel_dim=4,
               layer_types=["linear_attention", "full_attention"],
               rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                "mrope_section": [4, 4, 4], "partial_rotary_factor": 1.0},
               bos_token_id=1, eos_token_id=2)
    cfg3 = Qwen3_5Config(vision_config=vc3, text_config=tc3, image_token_id=5,
                         vision_start_token_id=3, vision_end_token_id=4, bos_token_id=1,
                         eos_token_id=2)
    cfg3._attn_implementation = "sdpa"
    torch.manual_seed(2)
    m35 = Qwen3_5ForConditionalGeneration(cfg3).eval()
    return {"qwen25vl": m25, "qwen35": m35}


def make_inputs(model, h, w, seed):
    """One request: 3 leading text tokens, the image run, 5 trailing text tokens."""
    vcfg = model.config.vision_config
    m = vcfg.spatial_merge_size
    n_rows = h * w
    n_tok = n_rows // (m * m)
    ids = torch.tensor([[7, 8, 3] + [5] * n_tok + [4, 9, 10, 11, 12]])
    g = torch.Generator().manual_seed(seed)
    dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2
    px_full = torch.randn(n_rows, dim, generator=g)
    px_base = px_full + 0.3 * torch.randn(n_rows, dim, generator=g)
    inputs = {"input_ids": ids, "mm_token_type_ids": (ids == 5).long(),
              "pixel_values": px_full, "image_grid_thw": torch.tensor([[1, h, w]])}
    return inputs, px_base


def run(axis, inputs, px_base, groups, keep):
    lg, kv, st = axis.streaming_forward(inputs, px_base, groups, keep=keep)
    return {"image_embeds": st["image_embeds"], "logits": lg,
            "decode_start_pos": st["decode_start_pos"], "chunks": st["chunks"],
            "corrected_groups": st["corrected_groups"], "rope_delta": st["rope_delta"]}


def same(a, b):
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and torch.equal(a, b)
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-root", required=True)
    a = ap.parse_args()

    from appcorr.models.qwen25vl.unified import Qwen25VLAxis
    from appcorr.models.qwen35.unified import Qwen35Axis
    new_cls = {"qwen25vl": Qwen25VLAxis, "qwen35": Qwen35Axis}
    ref_cls = import_ref(a.ref_root)
    models = tiny_models()

    fails = 0
    for fam in ("qwen25vl", "qwen35"):
        model = models[fam]
        new = new_cls[fam](model, None)
        ref = ref_cls[fam](model, None)
        for (h, w) in ((8, 12), (6, 10)):
            for seed in (0, 1):
                inputs, px_base = make_inputs(model, h, w, seed)
                for groups in (1, 4):
                    for keep in (1.0, 0.5):
                        with torch.no_grad():
                            r = run(ref, inputs, px_base, groups, keep)
                            new.positions_mode = "check"
                            n = run(new, inputs, px_base, groups, keep)
                        bad = [k for k in r if not same(r[k], n[k])]
                        tag = f"{fam} grid {h}x{w} seed {seed} g={groups} keep={keep}"
                        if bad:
                            fails += 1
                            print(f"FAIL {tag}: differ on {bad}")
                            for k in bad:
                                if isinstance(r[k], torch.Tensor):
                                    print(f"    {k}: max|d| {(r[k] - n[k]).abs().max().item():.3e}")
                                else:
                                    print(f"    {k}: ref {r[k]} new {n[k]}")
                        else:
                            print(f"ok   {tag}: chunks {n['chunks']} corrected {n['corrected_groups']}")
    # the levers' own switches: slow forms in the NEW tree must equal the fast forms too
    for fam in ("qwen25vl", "qwen35"):
        model = models[fam]
        inputs, px_base = make_inputs(model, 6, 10, 3)
        fast = new_cls[fam](model, None)
        slow = new_cls[fam](model, None)
        slow.correct_rows_only = False
        slow.positions_mode = "reference"
        with torch.no_grad():
            for keep in (1.0, 0.5):
                f, s = run(fast, inputs, px_base, 4, keep), run(slow, inputs, px_base, 4, keep)
                bad = [k for k in f if not same(f[k], s[k])]
                if bad:
                    fails += 1
                    print(f"FAIL {fam} fast-vs-slow keep={keep}: {bad}")
                else:
                    print(f"ok   {fam} fast-vs-slow keep={keep}")
    # FLOPs identity (FLOPs-first rule): the counter's per-stage numbers must not move. Same
    # counter (the new tree's `appcorr.flops`, which patches SDPA globally and hooks Linear/Conv
    # under the roots) around the old axis and the new one; only the model code differs.
    from appcorr import flops as flops_mod
    for fam in ("qwen25vl", "qwen35"):
        model = models[fam]
        inputs, px_base = make_inputs(model, 6, 10, 3)
        for groups in (1, 4):
            for keep in (1.0, 0.5):
                got = {}
                for tag, cls in (("ref", ref_cls[fam]), ("new", new_cls[fam])):
                    with flops_mod.session(model.model.visual, model.model.language_model,
                                           enabled=True) as fl:
                        axis = cls(model, None, flop_counter=fl)
                        with fl.request(0), torch.no_grad():
                            axis.streaming_forward(inputs, px_base, groups, keep=keep)
                        rq = fl.requests[-1]
                        got[tag] = {"total": rq.total, "critical": rq.critical,
                                    "by_stage": dict(rq.by_stage())}
                if got["ref"] != got["new"]:
                    fails += 1
                    print(f"FAIL {fam} flops g={groups} keep={keep}: ref {got['ref']} new {got['new']}")
                else:
                    st = " ".join(f"{k}={v/1e6:.2f}M" for k, v in got["new"]["by_stage"].items())
                    print(f"ok   {fam} flops g={groups} keep={keep}: total {got['new']['total']/1e6:.2f}M "
                          f"critical {got['new']['critical']/1e6:.2f}M [{st}]")
    print("QWEN_AXIS_CPU_GATE", "FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

"""Gate for the progressive-selection interleaved walk on the real Gemma 3 4B.

The construction identity: keep=1.0, g=1 -- everything arrives in one band, everything is
corrected with the full-depth signal -- must reproduce the stock full-res forward. Feature-space
rel-L2 (the contract's own recommendation), bf16 noise scale expected.

Direction: g=4, keep=0.25 must land strictly between the floor and the ceiling in feature space.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from datasets import load_dataset
from qwen_vl_prefill.datasets_eval import get_spec
from flops_report_gemma3 import l2_from_native, patch_energy
from appcorr.models.gemma3.unified import Gemma3UnifiedAxis


def main() -> int:
    dev = "cuda:0"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it", dtype=torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    spec = get_spec("chartqa")
    ds = spec.load(load_dataset)
    axis = Gemma3UnifiedAxis(model.model).eval()
    patch = int(model.config.vision_config.patch_size)

    ok = True
    for i in (0, 700):
        img, prompt, _ = spec.prepare(ds[i], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        enc = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": img},
                                          {"type": "text", "text": prompt}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt").to(dev)
        px, ids, tti = enc["pixel_values"].to(torch.bfloat16), enc["input_ids"], enc.get("token_type_ids")
        deg = l2_from_native(img, 2, 896)
        px2 = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": deg},
                                          {"type": "text", "text": prompt}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt")["pixel_values"].to(dev, torch.bfloat16)
        energy = patch_energy(px, px2, patch)

        with torch.no_grad():
            ref = axis.full_forward(px, ids, tti)
            floor = axis.full_forward(px2, ids, tti)
            # Interleaved walks return the PRE-finish hidden state by contract (the driver applies
            # `llm_finish`); full_forward returns a finished one. Comparing them raw measures the
            # final RMSNorm, not the schedule -- rel-L2 came out 231 that way.
            e1, _ = axis.interleaved_forward_progressive(px, px2, ids, tti, energy, 1.0, 1)
            e4, _ = axis.interleaved_forward_progressive(px, px2, ids, tti, energy, 0.25, 4)
            e1, e4 = axis.llm_finish(e1), axis.llm_finish(e4)

        def rel(x):
            return ((x - ref).norm() / ref.norm()).item()
        r_fl, r_1, r_4 = rel(floor), rel(e1), rel(e4)
        g1 = r_1 < 5e-3            # bf16 noise scale on a 61-stage axis
        g4 = r_1 < r_4 < r_fl
        ok &= g1 and g4
        print(f"  sample {i}: rel-L2  floor {r_fl:.4f}  g=4,k=.25 {r_4:.4f}  g=1,k=1.0 {r_1:.6f}"
              f"   [{'PASS' if g1 else 'FAIL'} identity] [{'PASS' if g4 else 'FAIL'} ordering]")
    print("\n" + ("ALL GATES PASS" if ok else "GATE FAILURE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

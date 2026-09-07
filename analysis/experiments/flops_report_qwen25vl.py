"""Critical backbone FLOPs for Qwen2.5-VL, interleaved g=4 at three recompute rates.

Driven in process rather than through the offload server, because the Qwen offload configs name
dataset "realworldqa" and `offload/mobile/dataset.get_dataset_loader` has no loader for it -- every
run aborts at handshake with "Unknown dataset name" before a batch moves. The forks under
`appcorr/models/qwen25vl/` are the same code the executor would have driven.

Two things about this model shape the numbers and neither is visible from a parameter count:

  * The vision encoder is SHARED across 32B and 72B -- depth 32, width 1280, window 112, with full
    attention only at layers 7/15/23/31. So the two sizes differ only in their LLM half, and the
    windowed layers are cheap relative to their layer count.
  * `mrope` gives image tokens 3D positions, but the FLOPs do not care: the count follows the token
    counts and widths, which the processor fixes.

    python analysis/experiments/flops_report_qwen25vl.py [--model-path Qwen/Qwen2.5-VL-32B-Instruct]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from appcorr import flops
from qwen_vl_prefill.datasets_eval import get_spec

DATASETS = ("realworldqa", "gqa", "refcoco")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-32B-Instruct")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keeps", type=float, nargs="+", default=[0.25, 0.30, 0.50])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dev, dt = a.device, torch.bfloat16
    tok = os.environ.get("HF_TOKEN")
    print(f"[qwen] loading {a.model_path}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model_path, dtype=dt, token=tok).eval().to(dev)
    proc = AutoProcessor.from_pretrained(a.model_path, token=tok)

    inner = model.model
    vis, lm = inner.visual, inner.language_model
    vc, tc = model.config.vision_config, model.config.text_config
    v_layers = int(getattr(vc, "depth", getattr(vc, "num_hidden_layers", 32)))
    l_layers = int(tc.num_hidden_layers)
    print(f"[qwen] axis {v_layers}+{l_layers}={v_layers + l_layers} stages", flush=True)

    for ds_name in a.datasets:
        spec = get_spec(ds_name)
        ds = spec.load(load_dataset)
        idxs = list(range(0, len(ds), max(1, len(ds) // a.samples)))[:a.samples]

        # The whole point of the FLOPs question is the SCHEDULE, and the schedule is the same one
        # `flops_analytic` models: an approximate pass to the first bound, then per round a
        # correction at the frontier plus an advance. Rather than re-deriving Qwen's interleaved
        # driver here, the stock forward is measured (the ceiling, which is what the table's
        # denominator needs) and the arms are left to the analytic model, whose ceiling agreement
        # is what licenses it.
        with flops.session(vis, lm, enabled=True) as fl:
            for i in idxs:
                img, prompt, _ = spec.prepare(ds[i], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
                msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                     {"type": "text", "text": prompt}]}]
                text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                enc = proc(text=[text], images=[img], return_tensors="pt").to(dev)
                with fl.request(i, n_patch=int(enc["pixel_values"].shape[0]),
                                seq=int(enc["input_ids"].shape[1])):
                    with fl.stage("full"):
                        model(**{k: v for k, v in enc.items()}, use_cache=False)
        agg = fl.aggregate()
        npatch = sum(int(r.meta["n_patch"]) for r in fl.requests) / len(idxs)
        seq = sum(int(r.meta["seq"]) for r in fl.requests) / len(idxs)
        print(f"\n══ {ds_name} (n={len(idxs)}) ══")
        print(f"  mean n_patch={npatch:.0f}  seq={seq:.0f}  n_img_tok={npatch/4:.0f}")
        print(f"  full inference (ceiling)  {agg['mean_total_gflops']:10.1f} GFLOPs/instruction")
        print(f"  ANALYTIC-INPUT  (\"{'qwen25vl_32b' if '32B' in a.model_path else 'qwen25vl_72b'}\", "
              f"\"{ds_name}\", {npatch:.0f}, {seq:.0f}, {npatch/4:.0f}),")


if __name__ == "__main__":
    main()

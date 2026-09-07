"""Streaming prefill against the REAL Qwen3.5-35B-A3B, end to end through the multimodal path.

The synthetic gate proved the chunk-boundary algebra on a toy config. What it cannot prove:
that the shipped checkpoint's GatedDeltaNet fast path (flash-linear-attention / causal-conv1d,
which the toy fell back from), the real MoE router, and the vision-token embedding splice all
survive chunking. This runs the actual 35B in bf16 on one GPU.

bf16 makes bit-exactness unattainable (reduction order is shape-dependent and chunking changes
shapes), so the gate compares GREEDY TOKENS over a 24-token continuation, plus logit deltas for
diagnosis. Divergent decode from identical prefill state is the failure being hunted; identical
decode through 4 chunk boundaries is what "streaming is lossless" means operationally.
"""
import glob, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoProcessor, AutoModelForImageTextToText
from appcorr.models.qwen35.llm.streaming import stream_prefill

MID = "Qwen/Qwen3.5-35B-A3B"


def main() -> int:
    from PIL import Image
    import numpy as np
    torch.manual_seed(0)
    proc = AutoProcessor.from_pretrained(MID)
    model = AutoModelForImageTextToText.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    # A real image-text request through the real chat template.
    img = Image.fromarray((np.random.RandomState(0).rand(448, 448, 3) * 255).astype("uint8"))
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "Describe this image in one sentence."}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to("cuda:0")
    ids = inputs["input_ids"]
    T = ids.shape[1]
    print(f"  prompt: {T} tokens")

    # Embed once (vision tower runs here); stream the EMBEDS so chunk boundaries can fall inside
    # the image-token run, which is where AppCorr's transmission rounds would put them.
    with torch.no_grad():
        embeds = model.get_input_embeddings()(ids)
        pix = inputs.get("pixel_values")
        if pix is not None:
            vis = model.model.visual(pix.to(model.dtype), grid_thw=inputs["image_grid_thw"]).pooler_output
            mask = (ids == model.config.image_token_id)
            print(f"  image tokens: {int(mask.sum())} of {T}")
            embeds = embeds.masked_scatter(mask.unsqueeze(-1), vis.to(embeds.dtype))

    def greedy(boundaries, n=24):
        with torch.no_grad():
            logits, cache = stream_prefill(model, inputs_embeds=embeds, boundaries=boundaries)
            toks = []
            cur = logits[:, -1].argmax(-1, keepdim=True)
            pos = T
            for _ in range(n):
                toks.append(int(cur))
                out = model(input_ids=cur, past_key_values=cache, use_cache=True,
                            position_ids=torch.tensor([[pos]], device=cur.device))
                cache = out.past_key_values
                cur = out.logits[:, -1].argmax(-1, keepdim=True)
                pos += 1
            return toks, logits[:, -1]

    n_img = int((ids == model.config.image_token_id).sum())
    img_start = int((ids == model.config.image_token_id).nonzero()[0, 1])
    # Boundaries INSIDE the image-token run -- 4 arrival rounds, the AppCorr shape.
    quarter = n_img // 4
    b_stream = [0] + [img_start + quarter * k for k in (1, 2, 3)] + [T]
    print(f"  streaming boundaries (inside image run): {b_stream}")

    ref_toks, ref_logit = greedy([0, T])
    st_toks, st_logit = greedy(b_stream)

    same = ref_toks == st_toks
    dlog = (st_logit - ref_logit).abs().max().item()
    print(f"\n  one-shot  : {proc.tokenizer.decode(ref_toks)!r}")
    print(f"  streamed  : {proc.tokenizer.decode(st_toks)!r}")
    print(f"  {'PASS' if same else 'FAIL'}  greedy tokens identical over 24 steps")
    print(f"  ----  final-position max|logit diff| = {dlog:.3e} (bf16; diagnostic, not gated)")
    print("\n" + ("ALL GATES PASS" if same else "GATE FAILURE"))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())

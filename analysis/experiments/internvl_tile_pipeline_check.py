"""Is tile-by-tile streaming of InternVL lossless?

The claim to test: because InternViT encodes each 448x448 tile independently (tiles are the batch
dimension, attention never crosses them) and the LLM is causal over a sequence in which each tile's
tokens form a contiguous block, one can encode-and-prefill tile 0, then tile 1, ... overlapping each
tile's compute with the next tile's transmission, and arrive at exactly the monolithic result.

If that holds, approx-then-correct has nothing to offer THIS model on ordering grounds: streaming is
already lossless, so there is no staleness to correct. Worth knowing where a technique is not needed.

Three checks, each able to fail on its own:

1. **Tile independence.** Encode all tiles together vs one at a time. Any difference means attention
   or normalisation crosses tiles and the whole idea collapses.
2. **Contiguity.** Each tile's LLM tokens must occupy one contiguous span, or "prefill tile k" is not
   expressible -- LLaVA-OneVision fails this: it reassembles crops into one grid and flattens
   raster-order, interleaving them.
3. **Chunked prefill equivalence.** Feed the sequence to the LLM in tile-sized chunks through a
   growing KV cache and compare logits against one monolithic pass.

    python analysis/experiments/internvl_tile_pipeline_check.py [--model ...] [--dtype float32]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, "/NHNHOME/share/cjpark/AppCorr-internvl")


def rep(name, got, ref, rtol):
    scale = max(ref.float().abs().max().item(), 1e-9)
    err = (got.float() - ref.float()).abs().max().item()
    ok = err / scale <= rtol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<50} rel {err/scale:.2e}  (abs {err:.3e})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL3-2B-hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    rtol = 1e-4 if dtype is torch.float32 else 5e-2
    tok = os.environ.get("HF_TOKEN")
    full = AutoModelForImageTextToText.from_pretrained(a.model, dtype=dtype, token=tok).eval().to(a.device)
    proc = AutoProcessor.from_pretrained(a.model, token=tok)
    model = full.model

    ds = load_dataset("lmms-lab/RealWorldQA", split="test")
    ex = ds[0]
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": ex["question"]}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = proc(images=ex["image"].convert("RGB"), text=prompt, return_tensors="pt").to(a.device)
    px = enc["pixel_values"].to(dtype)
    tiles = px.shape[0]
    print(f"{a.model}  {tiles} tiles, {a.dtype}")

    ok = True
    with torch.no_grad():
        # --- 1. tile independence -------------------------------------------------------------
        allat = model.get_image_features(pixel_values=px).pooler_output
        one_by_one = torch.cat(
            [model.get_image_features(pixel_values=px[i:i + 1]).pooler_output for i in range(tiles)]
        )
        ok &= rep("encode tiles one at a time == all together", one_by_one, allat, rtol)

        # --- 2. contiguity of each tile's LLM tokens ------------------------------------------
        ids = enc["input_ids"][0]
        img_pos = (ids == model.config.image_token_id).nonzero(as_tuple=True)[0]
        per_tile = allat.shape[1]
        contiguous = bool((img_pos[1:] - img_pos[:-1] == 1).all())
        print(f"  {'PASS' if contiguous else 'FAIL'}  image tokens form ONE contiguous span"
              f"{'':<12} {img_pos.numel()} tokens = {tiles} x {per_tile}")
        ok &= contiguous and img_pos.numel() == tiles * per_tile

        # --- 3. chunked prefill through a growing cache ---------------------------------------
        embeds = model.get_input_embeddings()(ids.unsqueeze(0))
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[img_pos] = True
        embeds = embeds.masked_scatter(mask.unsqueeze(0).unsqueeze(-1), allat.reshape(-1, allat.shape[-1]))

        ref = full.model.language_model(inputs_embeds=embeds, use_cache=False).last_hidden_state

        from transformers import DynamicCache
        cache = DynamicCache()
        # Chunk boundaries: everything before the image, then one chunk per tile, then the rest.
        bounds = [0, int(img_pos[0])] + \
                 [int(img_pos[0]) + (i + 1) * per_tile for i in range(tiles)] + [ids.numel()]
        bounds = sorted(set(b for b in bounds if 0 <= b <= ids.numel()))
        outs = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            if e <= s:
                continue
            pos = torch.arange(s, e, device=a.device).unsqueeze(0)
            o = full.model.language_model(inputs_embeds=embeds[:, s:e], past_key_values=cache,
                                          position_ids=pos, use_cache=True)
            cache = o.past_key_values
            outs.append(o.last_hidden_state)
        chunked = torch.cat(outs, dim=1)
        ok &= rep(f"chunked prefill ({len(bounds)-1} chunks) == monolithic", chunked, ref, rtol)

    print("\n" + ("TILE PIPELINING IS LOSSLESS" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

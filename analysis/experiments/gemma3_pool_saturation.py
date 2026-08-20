"""Does a patch selection survive Gemma 3's 16:1 pooling, or does it saturate?

Each LLM image token pools 16 vision patches, and the obvious mapping -- a token is corrected if ANY
of its patches was -- saturates fast: a uniformly random 50% patch selection marks 1-0.5^16 = 99.998%
of tokens. If real selections behave that way too, nothing saved in the vision half reaches the LLM
half and the unified axis is only half a technique.

The bet is that real selections are not uniform: the patch score is residual energy (times attention)
on an image degraded by an L2 pyramid, and that energy concentrates on edges and texture, which are
spatially clustered. Clustered patches share pooling groups, so fewer groups get touched.

This measures it on real images with the real score, against a random baseline at the same rate.
Reported per keep ratio:

    tokens hit   fraction of the 256 image tokens marked corrected
    saturation   tokens hit / (what an ideal, perfectly clustered selection would need)

An ideal selection of `k` patches touches ceil(k/16) tokens; random touches ~256*(1-(1-r)^16).

    python analysis/experiments/gemma3_pool_saturation.py --num-images 20
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, "/NHNHOME/share/cjpark/AppCorr-gemma3")


def l2_from_native(img, level, cap):
    from PIL import Image
    w, h = img.size
    short = min(min(w, h), cap)
    t = max(1, short // 2 ** level)
    if h <= w:
        th, tw = t, max(1, round(w / h * t))
    else:
        tw, th = t, max(1, round(h / w * t))
    return img.resize((tw, th), Image.BOX).resize((w, h), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--num-images", type=int, default=20)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor

    tok = os.environ.get("HF_TOKEN")
    proc = AutoProcessor.from_pretrained(a.model, token=tok)
    ds = load_dataset("lmms-lab/RealWorldQA", split="test")
    n = min(a.num_images, len(ds))
    patch = 14
    per_token = None

    ratios = [0.1, 0.25, 0.4, 0.55, 0.7]
    hit_real = {r: [] for r in ratios}
    hit_rand = {r: [] for r in ratios}

    for i in range(n):
        img = ds[i]["image"].convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": "x"}]}]
        enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt")
        px = enc["pixel_values"]
        cap = px.shape[-1]
        deg = l2_from_native(img, a.level, cap)
        msgs2 = [{"role": "user", "content": [{"type": "image", "image": deg},
                                              {"type": "text", "text": "x"}]}]
        px2 = proc.apply_chat_template(msgs2, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt")["pixel_values"]

        d = ((px.float() - px2.float()) ** 2).mean(dim=1)          # [B, H, W]
        b, H, W = d.shape
        e = d.reshape(b, H // patch, patch, W // patch, patch).mean(dim=(2, 4)).reshape(b, -1)
        n_patch = e.shape[1]
        n_tok = int(proc.image_processor.size["height"] // patch) ** 2 // 16 if per_token else 256
        per = n_patch // 256
        if per_token is None:
            per_token = per
            print(f"  {n_patch} patches -> 256 image tokens ({per} patches each)")

        for r in ratios:
            k = max(1, int(round(r * n_patch)))
            idx = e[0].topk(k).indices
            m = torch.zeros(n_patch, dtype=torch.bool); m[idx] = True
            hit_real[r].append(m.reshape(256, per).any(-1).sum().item())
            mr = torch.zeros(n_patch, dtype=torch.bool); mr[torch.randperm(n_patch)[:k]] = True
            hit_rand[r].append(mr.reshape(256, per).any(-1).sum().item())

    print(f"\n  {n} images, level L{a.level}, {per_token} patches per LLM image token\n")
    print(f"  {'keep':>6} {'ideal':>7} {'real score':>12} {'random':>12}   {'real/random':>12}")
    for r in ratios:
        ideal = -(-int(r * 4096) // per_token)
        real = sum(hit_real[r]) / len(hit_real[r])
        rand = sum(hit_rand[r]) / len(hit_rand[r])
        print(f"  {r:>5.0%} {ideal:>7} {real:>9.1f}/256 {rand:>9.1f}/256   {real/max(rand,1e-9):>11.2f}x")


if __name__ == "__main__":
    main()

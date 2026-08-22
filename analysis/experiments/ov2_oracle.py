"""floor / ceiling for LLaVA-OneVision-2, with the failures of the Gemma 3 run designed out.

Both arms here use stock `generate`, so there is no fork and no decode path to get wrong yet. What
CAN go wrong is the input, and every check below exists because something like it already bit us:

  * the image silently not reaching the model -- this processor takes a bare {"type": "image"}
    placeholder and the image as a separate argument; putting a PIL object in the message yields
    input_ids only, no pixel values, no error
  * the degraded image resampling to a DIFFERENT grid than the original, which would change the
    token count and confound floor against ceiling
  * a degradation that is effectively a no-op, which reads as "the model is robust"
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_vl_prefill.datasets_eval import get_spec
from experiments.ov2_degradation import l2_from_native, sampled_hw

MODEL = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"


def encode(proc, img, prompt, device):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = proc(text=[text], images=[img], return_tensors="pt", padding=True)
    assert "pixel_values" in enc, "the image did not reach the processor"
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}


@torch.no_grad()
def run_one(model, proc, img, prompt, arm, level, device, max_new_tokens, diag=None):
    enc = encode(proc, img, prompt, device)
    if arm == "floor":
        deg = l2_from_native(img, level, proc)
        enc2 = encode(proc, deg, prompt, device)
        assert enc2["pixel_values"].shape == enc["pixel_values"].shape, (
            f"degradation changed the sampled grid: {enc['image_grid_thw'].tolist()} vs "
            f"{enc2['image_grid_thw'].tolist()} -- floor and ceiling would not be comparable")
        if diag is not None:
            a, b = enc["pixel_values"].float(), enc2["pixel_values"].float()
            diag["pixel_rel_l2"] = float((a - b).norm() / a.norm().clamp_min(1e-9))
            diag["grid"] = tuple(enc["image_grid_thw"][0].tolist())
            diag["n_img_tok"] = int((enc["input_ids"][0] == model.config.image_token_id).sum())
        enc = enc2
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def load(device, dtype):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    tok = os.environ.get("HF_TOKEN")
    proc = AutoProcessor.from_pretrained(MODEL, token=tok, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=dtype, token=tok, trust_remote_code=True).eval().to(device)
    return model, proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chartqa")
    ap.add_argument("--arm", choices=["ceiling", "floor"], default="ceiling")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    from datasets import load_dataset
    model, proc = load(a.device, torch.bfloat16)
    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    n = len(ds) if a.full else min(a.num_samples, len(ds))
    idxs = list(range(len(ds)))[:n] if a.full else list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    print(f"[ov2] {a.dataset} arm={a.arm} level=L{a.level}  {n} of {len(ds)}", flush=True)

    total, correct, t0, per_sample = 0.0, 0, time.time(), []
    for k, i in enumerate(idxs, 1):
        img, prompt, gold = spec.prepare(ds[i], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        text = run_one(model, proc, img, prompt, a.arm, a.level, a.device, a.max_new_tokens)
        ok, sc = spec.score(text, gold)
        correct += int(ok); total += sc
        per_sample.append({"idx": i, "gold": gold, "pred": text, "score": sc})
        if k % 25 == 0 or k == n:
            el = time.time() - t0
            print(f"  [{k}/{n}] {el:.0f}s {el/k:.2f}s/ex  acc={total/k*100:.2f}%", flush=True)

    summary = {"model": MODEL, "dataset": a.dataset, "arm": a.arm, "level": a.level,
               "num_samples": n, "accuracy": total / n, "correct": correct}
    print("\n=== Final Summary: " + json.dumps(summary))
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump({"summary": summary, "per_sample": per_sample}, f)


if __name__ == "__main__":
    main()

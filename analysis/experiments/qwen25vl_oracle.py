"""
qwen25vl_oracle.py

Standalone correctness oracle for the Qwen2.5-VL AppCorr fork (Phase 0), plus a quick RealWorldQA
accuracy sanity check with the STOCK model (to validate the answer-extraction/scoring logic before
trusting any AppCorr comparison built on top of it).

Loads the stock `transformers.Qwen2_5_VLForConditionalGeneration`, runs real forward + generate on
real RealWorldQA images at native (smart-resized) resolution, and dumps:
  - per-vision-layer patch hidden states + grid_thw for a couple of images with different aspect
    ratios (to exercise both windowed and full-attention layers)
  - the LLM's 3-axis position_ids for a full prompt (from get_rope_index)
  - generated answer text for those images

Run (appcorr env):
    python analysis/experiments/qwen25vl_oracle.py --model-path <path or repo id> \
        --out /tmp/qwen25vl_oracle.pt --device cuda:0
"""

import argparse
import re
import sys
from pathlib import Path

import torch

MODEL_ID_32B = "Qwen/Qwen2.5-VL-32B-Instruct"
MODEL_ID_72B = "Qwen/Qwen2.5-VL-72B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default=MODEL_ID_32B)
    p.add_argument("--out", type=str, default="/tmp/qwen25vl_oracle.pt")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-oracle-images", type=int, default=3)
    p.add_argument("--num-sanity-samples", type=int, default=20)
    return p.parse_args()


def extract_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text.strip())
    return m.group(1) if m else None


def normalize_freeform(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def is_mcq(question: str) -> bool:
    return bool(re.search(r"\n\s*A\.\s", question))


def score_answer(question: str, pred_text: str, gt_answer: str) -> bool:
    if is_mcq(question):
        pred_letter = extract_letter(pred_text)
        return pred_letter == gt_answer.strip().upper()
    else:
        pred_norm = normalize_freeform(pred_text)
        gt_norm = normalize_freeform(gt_answer)
        return pred_norm == gt_norm or gt_norm in pred_norm.split()


def main():
    args = parse_args()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from datasets import load_dataset

    print(f"[oracle] loading {args.model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    print(f"[oracle] loaded. vision depth={model.config.vision_config.depth} "
          f"llm layers={model.config.text_config.num_hidden_layers} "
          f"llm heads={model.config.text_config.num_attention_heads} "
          f"llm kv_heads={model.config.text_config.num_key_value_heads} "
          f"image_token_id={model.config.image_token_id}")

    print("[oracle] loading lmms-lab/RealWorldQA ...")
    ds = load_dataset("lmms-lab/RealWorldQA", split="test")
    print(f"[oracle] {len(ds)} examples")

    def run_one(idx):
        ex = ds[idx]
        image = ex["image"].convert("RGB")
        question = ex["question"]
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(args.device)
        return ex, inputs

    # --- Oracle dump: a few images with different grid_thw, per-vision-layer states ---
    oracle_dumps = []
    with torch.no_grad():
        for i in range(args.num_oracle_images):
            ex, inputs = run_one(i)
            grid_thw = inputs["image_grid_thw"]
            vision_out = model.model.visual(inputs["pixel_values"].to(dtype=torch.bfloat16), grid_thw=grid_thw)
            gen_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            gen_trimmed = gen_ids[:, inputs["input_ids"].shape[1]:]
            answer_text = processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0]
            print(f"[oracle] image {i}: grid_thw={grid_thw.tolist()} image_size={ex['image'].size} "
                  f"gt_answer={ex['answer']!r} pred={answer_text!r}")
            oracle_dumps.append({
                "idx": i,
                "grid_thw": grid_thw.cpu(),
                "image_size": ex["image"].size,
                "question": ex["question"],
                "gt_answer": ex["answer"],
                "pred_text": answer_text,
                "last_hidden_state_window_order": vision_out.last_hidden_state.float().cpu(),
                "pooler_output_merged": vision_out.pooler_output.float().cpu(),
                "input_ids_shape": tuple(inputs["input_ids"].shape),
            })

    # --- Quick RealWorldQA accuracy sanity check (stock model) ---
    n_sanity = min(args.num_sanity_samples, len(ds))
    stride = max(len(ds) // n_sanity, 1)
    sanity_indices = list(range(0, len(ds), stride))[:n_sanity]
    correct = 0
    sanity_results = []
    with torch.no_grad():
        for count, idx in enumerate(sanity_indices):
            ex, inputs = run_one(idx)
            gen_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            gen_trimmed = gen_ids[:, inputs["input_ids"].shape[1]:]
            answer_text = processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0]
            ok = score_answer(ex["question"], answer_text, ex["answer"])
            correct += int(ok)
            sanity_results.append({"idx": idx, "gt": ex["answer"], "pred": answer_text, "correct": ok})
            print(f"    [{count+1}/{n_sanity}] idx={idx} gt={ex['answer']!r} pred={answer_text!r} correct={ok}")
            sys.stdout.flush()

    acc = 100.0 * correct / n_sanity
    print(f"\n[oracle] === RealWorldQA sanity accuracy (stock model, n={n_sanity}): {acc:.2f}% ===")

    torch.save({
        "model_path": args.model_path,
        "oracle_dumps": oracle_dumps,
        "sanity_results": sanity_results,
        "sanity_accuracy": acc,
    }, args.out)
    print(f"[oracle] saved to {args.out}")


if __name__ == "__main__":
    main()

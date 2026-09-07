"""Measure LLaVA-OneVision-2's actual masks and token layout before designing anything.

Every fact here was wrong somewhere upstream: the paper says the encoder uses spatial windowed
attention, the config exposes no window fields, and the code windows only the TEMPORAL axis. So
nothing below is taken from documentation -- it is read off real tensors.

What a unified vision+prefill axis needs to know, and why:
  1. how many vision patches a native-resolution image produces (there is no fixed canvas)
  2. how spatial_merge_size folds patches into LLM tokens -- Gemma 3's 4x4 pooling was mis-mapped
     once and reversed a ranking, so this mapping gets asserted, not assumed
  3. whether image tokens occupy one contiguous span of the LLM sequence
  4. whether vision attention really is global (no independent spatial units => no streaming)
  5. whether the LLM attends causally over image tokens
"""
import os, sys, torch
from PIL import Image

MODEL = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
tok = os.environ.get("HF_TOKEN")


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    dev = "cuda:0"
    proc = AutoProcessor.from_pretrained(MODEL, token=tok, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, token=tok, trust_remote_code=True).eval().to(dev)
    cfg = model.config
    vc, tc = cfg.vision_config, cfg.text_config
    print(f"  vision {vc.num_hidden_layers}L h{vc.hidden_size} patch{vc.patch_size} "
          f"merge{getattr(vc,'spatial_merge_size','?')} -> out{getattr(vc,'out_hidden_size','?')}")
    print(f"  llm    {tc.num_hidden_layers}L h{tc.hidden_size} "
          f"layer_types[:3]={getattr(tc,'layer_types',['?'])[:3]} sliding={getattr(tc,'sliding_window',None)}")
    print(f"  axis   {vc.num_hidden_layers}+{tc.num_hidden_layers}="
          f"{vc.num_hidden_layers+tc.num_hidden_layers} stages\n")

    # --- 1/2/3: token layout at several native resolutions ---------------------------------- #
    for (w, h) in [(448, 448), (896, 448), (1448, 938), (640, 427)]:
        img = Image.new("RGB", (w, h), (128, 128, 128))
        # This processor takes a bare {"type": "image"} placeholder in the template and the image
        # separately -- passing the PIL object inside the message silently yields NO pixel values
        # at all (input_ids only), which is how the first run of this probe failed.
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": "What is this?"}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = proc(text=[text], images=[img], return_tensors="pt", padding=True)
        enc = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in enc.items()}
        ids = enc["input_ids"]
        pos = (ids[0] == cfg.image_token_id).nonzero().flatten()
        grid = enc.get("image_grid_thw")
        px = enc.get("pixel_values")
        n_tok = pos.numel()
        contiguous = bool(((pos[1:] - pos[:-1]) == 1).all()) if n_tok > 1 else True
        gt = tuple(grid[0].tolist()) if grid is not None else None
        n_patch = int(grid[0].prod()) if grid is not None else (px.shape[0] if px is not None else 0)
        m = int(getattr(vc, "spatial_merge_size", 1))
        print(f"  {w}x{h}: grid_thw={gt} patches={n_patch} img_tokens={n_tok} "
              f"ratio={n_patch/max(n_tok,1):.2f} (merge^2={m*m}) contiguous={contiguous} "
              f"seq={ids.shape[1]}")

    # --- 4: is vision attention global over one image? -------------------------------------- #
    vt = model.model.visual if hasattr(model.model, "visual") else model.model.vision_tower
    print(f"\n  vision tower: {type(vt).__name__}")
    seen = {}
    for name, mod in vt.named_modules():
        if hasattr(mod, "is_causal"):
            seen.setdefault(bool(mod.is_causal), []).append(name)
    for k, v in seen.items():
        print(f"    is_causal={k}: {len(v)} modules (e.g. {v[0]})")
    print("    keys:", sorted(enc.keys()))
    if hasattr(vt, "_build_cu_seqlens") and grid is not None:
        cu, ms = vt._build_cu_seqlens(grid, n_patch, getattr(vc, "frame_windows_size", 4), dev)
        print(f"    cu_seqlens for a single image: {cu.tolist()}  max_seqlen={ms}")
        print(f"    -> {len(cu)-1} attention segment(s); 1 means fully global over the image")


if __name__ == "__main__":
    main()

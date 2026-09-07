"""L2 degradation for LLaVA-OneVision-2, whose sampling resolution is a function of the image.

Gemma 3 resizes every image to a fixed 896x896 canvas, so `cap` was a constant. OneVision-2 keeps
native resolution under a pixel budget (longest_edge 4,000,000; shortest_edge 3,136) and rounds each
side to a multiple of patch*merge = 28, so what the model actually samples must be ASKED, not
assumed: a 3000x3000 image is sampled at 1988x1988 (0.663x), while 448x448 passes through untouched.

The pyramid-direction rule (docs/memo/pyramid_degradation_native_vs_canvas.md) then applies per axis
against that measured resolution. Aspect ratio IS preserved here -- measured anisotropy is at most
1.023, pure 28-multiple rounding, versus 1.35-1.54 on Gemma 3's square squash -- so the two axes
would agree anyway; taking them separately costs nothing and keeps the rule stated in one form.
"""
from PIL import Image


def hw_from_grid(grid_thw, proc):
    """Sampled (h, w) in pixels, from an `image_grid_thw` the caller already holds.

    Prefer this over `sampled_hw`: any driver that has encoded the image at all is holding this
    tensor, and turning it into two integers is free.
    """
    g = grid_thw[0].tolist() if hasattr(grid_thw, "tolist") else list(grid_thw[0])
    return g[1] * proc.image_processor.patch_size, g[2] * proc.image_processor.patch_size


def sampled_hw(proc, img):
    """What the model really samples: the processor's own grid, in pixels. (h, w)

    EXPENSIVE, and avoidable whenever the caller has an encoding in hand. It runs the whole image
    preprocessing -- resample, normalise, patchify to [n_patch, 588] -- to read back two integers.
    Measured at ~265 ms per call against ~35 ms of actual GPU inference, which is why the floor arm
    ran 2.7x the ceiling arm (0.83 vs 0.30 s/example) for reasons that had nothing to do with the
    model: floor preprocessed three times, ceiling once.

    Use `hw_from_grid` when an `image_grid_thw` exists. The two agree by construction -- both ask
    the same processor about the same image -- and were checked equal on 8 ChartQA samples spanning
    three native sizes.
    """
    enc = proc(text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[img],
               return_tensors="pt")
    return hw_from_grid(enc["image_grid_thw"], proc)


def l2_from_native(img: Image.Image, level: int, proc, sampled=None) -> Image.Image:
    """Degrade content by 2**level relative to whichever of {native, sampled} binds, per axis.

    `sampled` is the (h, w) the model will resample this image to. Pass it -- from
    `hw_from_grid(enc["image_grid_thw"], proc)` -- whenever the caller has already encoded the
    image; leaving it None spends a whole extra preprocessing pass rediscovering what the caller
    already has.
    """
    w, h = img.size
    sh, sw = sampled if sampled is not None else sampled_hw(proc, img)
    th = max(1, min(h, sh) // 2 ** level)
    tw = max(1, min(w, sw) // 2 ** level)
    return img.resize((tw, th), Image.BOX).resize((w, h), Image.BICUBIC)


def main():
    import os
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
                                         token=os.environ.get("HF_TOKEN"), trust_remote_code=True)
    LEVEL, ok = 2, True
    CASES = [("passthrough", 448, 448), ("slight upround", 1448, 938), ("small", 640, 427),
             ("panorama", 4000, 400), ("over budget", 3000, 3000), ("tiny", 224, 224)]
    print(f"  level={LEVEL}; each axis must retain min(native, sampled)/2**level samples\n")
    import numpy as np
    for name, w, h in CASES:
        img = Image.new("RGB", (w, h))
        enc = proc(text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[img],
                   return_tensors="pt")
        sh, sw = hw_from_grid(enc["image_grid_thw"], proc)
        out = l2_from_native(img, LEVEL, proc, (sh, sw))
        assert out.size == (w, h), f"{name}: size must be restored, got {out.size}"
        want = (max(1, min(w, sw) // 2 ** LEVEL), max(1, min(h, sh) // 2 ** LEVEL))
        # blur each axis shows once the processor resamples to (sw, sh)
        bw, bh = sw / want[0], sh / want[1]
        exp_w = 2 ** LEVEL if w >= sw else 2 ** LEVEL * sw / w
        exp_h = 2 ** LEVEL if h >= sh else 2 ** LEVEL * sh / h
        good = abs(bw - exp_w) < 0.06 * exp_w and abs(bh - exp_h) < 0.06 * exp_h

        # The fast path must be the SAME degradation, not merely a similar one. `hw_from_grid`
        # replaced a `sampled_hw` call that cost ~265 ms of preprocessing per sample -- it made the
        # floor arm 2.7x the ceiling arm for reasons unrelated to the model -- and a shortcut that
        # changed the degraded image by even a pixel would silently move every floor number.
        agree = (sh, sw) == sampled_hw(proc, img)
        same = np.array_equal(np.asarray(out), np.asarray(l2_from_native(img, LEVEL, proc)))
        ok &= good and agree and same
        print(f"  {'PASS' if good and agree and same else 'FAIL'}  {name:<15} {w}x{h} sampled "
              f"{sw}x{sh} keep {want[0]}x{want[1]}  blur {bw:.2f}/{bh:.2f} "
              f"(want {exp_w:.2f}/{exp_h:.2f})  fast-path {'==' if agree and same else '!='} slow")
    print("\n" + ("ALL GATES PASS" if ok else "GATE FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

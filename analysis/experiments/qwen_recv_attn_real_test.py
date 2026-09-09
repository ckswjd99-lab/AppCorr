"""L4b follow-up: time the torch reference and the Triton column sum on REAL post-RoPE q/k captured
from the 122B tower (one image at --tokens), against random tensors of the same shapes and strides in
the same process. Separates "the kernel is slow on real data" from "the kernel is slow in the
model's context" (the gap-2 profile booked ~4.5 ms per call at 2109 tokens vs 0.4 ms on random data).

  PYTHONPATH=. python analysis/experiments/qwen_recv_attn_real_test.py --model Qwen/Qwen3.5-122B-A10B-FP8 --tokens 2100
"""
import argparse, os, statistics, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qwen_correct_profile import make_axis, image_at_tokens  # noqa: E402  (same dir)


def timed(fn, n=7):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B-FP8")
    ap.add_argument("--tokens", type=int, nargs="+", default=[2100])
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 13, 26])
    a = ap.parse_args()
    from transformers import AutoProcessor
    from qwen_vl_prefill.datasets_eval import get_spec
    from datasets import load_dataset
    from appcorr.models.vision_only import load_vision_only
    import appcorr.models.qwen35.vision.attention as attn_mod
    from appcorr.models.qwen35.vision import recv_attn_triton as rt

    proc = AutoProcessor.from_pretrained(a.model)
    model = load_vision_only(a.model, device="cuda:0")
    axis = make_axis("qwen35", model, proc)
    tower = axis.tower
    spec = get_spec("realworldqa"); ds = spec.load(load_dataset)
    img, _, _ = spec.prepare(ds[0], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
    side_px = proc.image_processor.patch_size * proc.image_processor.merge_size

    captured = {}
    cls = attn_mod.ApproxCorrectQwen35VisionAttention
    orig = cls._received_attention
    counter = {"i": 0}

    def capturing(self, q, k, segment_ranges):
        li = counter["i"]; counter["i"] += 1
        if li in a.layers:
            captured[li] = (q.detach().clone(), k.detach().clone(), list(segment_ranges), self.scaling)
        return orig(self, q, k, segment_ranges)
    cls._received_attention = capturing

    with torch.no_grad():
        for n_req in a.tokens:
            im = image_at_tokens(img, n_req, side_px)
            inputs = axis.build_inputs(im, "describe").to("cuda:0")
            grid = inputs["image_grid_thw"]
            try:
                gctx = tower.prepare_grid(grid, "cuda:0")
                ctx = tower.prepare_full_tokens(inputs["pixel_values"], grid, gctx)
            except (AttributeError, TypeError):
                ctx = tower.prepare_full_tokens(inputs["pixel_values"], grid)
            captured.clear(); counter["i"] = 0
            axis._approx_base(ctx, {}, collect_attn=True)
            print(f"\n#### {n_req} -> captured layers {sorted(captured)}; q shape/strides "
                  f"{tuple(captured[a.layers[0]][0].shape)} {captured[a.layers[0]][0].stride()} dtype {captured[a.layers[0]][0].dtype}")
            for li in sorted(captured):
                q, k, segs, scaling = captured[li]
                T = q.shape[0]
                s = (q[:256].float().transpose(0, 1) @ k.float().transpose(0, 1).transpose(-1, -2)) * scaling
                spread = (s.amax(-1) - s.amin(-1)).mean().item()
                attn_mod.RECV_ATTN_IMPL = "torch"
                stub = type("S", (), {"scaling": scaling})()
                t_torch = timed(lambda: orig(stub, q, k, segs))
                t_tri = timed(lambda: rt.received_attention(q, k, scaling, segs))
                # same shapes/strides, random data
                g = torch.Generator(device="cuda").manual_seed(li)
                qr = torch.randn(q.shape, device="cuda", generator=g).mul_(2.0).to(q.dtype)
                kr = torch.randn(k.shape, device="cuda", generator=g).mul_(2.0).to(k.dtype)
                t_torch_r = timed(lambda: orig(stub, qr, kr, segs))
                t_tri_r = timed(lambda: rt.received_attention(qr, kr, scaling, segs))
                ref = orig(stub, q, k, segs); tri = rt.received_attention(q, k, scaling, segs)
                print(f"layer {li:2d} T={T} segs={segs} score-spread/row {spread:7.1f} | real: torch {t_torch:6.2f} ms triton {t_tri:6.2f} ms"
                      f" | random: torch {t_torch_r:6.2f} triton {t_tri_r:6.2f} | exact {(ref == tri).float().mean().item():.3f}")
    cls._received_attention = orig
    print("QWEN_RECV_ATTN_REAL_TEST_DONE")


if __name__ == "__main__":
    main()

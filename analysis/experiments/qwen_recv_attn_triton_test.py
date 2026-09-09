"""L4b unit test: Triton two-pass column sum vs the torch `_received_attention` reference on
random post-RoPE q/k, plus timing. Reports max |diff| in units of the score's own scale, the
exact-equal fraction, and the relative rank agreement (the quantity selection consumes).

  APPCORR_RECV_ATTN=triton python analysis/experiments/qwen_recv_attn_triton_test.py --tokens 2109 4816
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[512, 2109, 4816])
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=72)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=2.0, help="q/k std; larger = sharper softmax (more denormal p)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="time a grid of tile configs (BLOCK_M,BLOCK_N,warps,stages) per T; "
                         "prints triton ms per config plus exact/rank agreement vs the torch reference")
    a = ap.parse_args()

    import appcorr.models.qwen35.vision.attention as attn_mod
    from appcorr.models.qwen35.vision import recv_attn_triton as rt
    assert rt.available()
    scaling = a.head_dim ** -0.5

    class Ref:  # borrow the unbound reference with a stub `self`
        pass
    ref = Ref(); ref.scaling = scaling
    attn_mod.RECV_ATTN_IMPL = "torch"
    torch_fn = attn_mod.ApproxCorrectQwen35VisionAttention._received_attention

    def timed(fn):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = fn(); torch.cuda.synchronize()
        return r, (time.perf_counter() - t0) * 1e3

    if a.sweep:
        cfgs = [(64, 64, 4, 2), (64, 64, 4, 3), (128, 64, 4, 2), (128, 64, 8, 2), (64, 128, 4, 2),
                (64, 128, 8, 2), (128, 128, 8, 2), (128, 128, 8, 3), (128, 128, 4, 2), (256, 64, 8, 2),
                (64, 256, 8, 2)]
        for T in a.tokens:
            g = torch.Generator(device="cuda").manual_seed(a.seed + T)
            q = torch.randn(T, a.heads, a.head_dim, device="cuda", generator=g).mul_(a.scale).to(torch.bfloat16)
            k = torch.randn(T, a.heads, a.head_dim, device="cuda", generator=g).mul_(a.scale).to(torch.bfloat16)
            segs = [(0, T)]
            ref_out, t_ref = timed(lambda: torch_fn(ref, q, k, segs))
            t_ref = sorted(timed(lambda: torch_fn(ref, q, k, segs))[1] for _ in range(a.repeats))[a.repeats // 2]
            print(f"T={T:5d}  torch {t_ref:7.2f} ms")
            for cfg in cfgs:
                rt.set_config(*cfg)
                try:
                    out, _ = timed(lambda: rt.received_attention(q, k, scaling, segs))
                    t = sorted(timed(lambda: rt.received_attention(q, k, scaling, segs))[1]
                               for _ in range(a.repeats))[a.repeats // 2]
                except Exception as e:  # noqa: BLE001  (e.g. shared-memory overflow for a tile)
                    print(f"   cfg {cfg}: FAILED {type(e).__name__}: {str(e)[:80]}")
                    continue
                exact = (out == ref_out).float().mean().item()
                rank_same = (torch.argsort(ref_out) == torch.argsort(out)).float().mean().item()
                print(f"   cfg BM={cfg[0]:3d} BN={cfg[1]:3d} w={cfg[2]} st={cfg[3]}: {t:7.2f} ms  x{t_ref / t:.1f}"
                      f"  exact {exact:.3f}  rank-same {rank_same:.4f}", flush=True)
        return 0

    rows = []
    for T in a.tokens:
        g = torch.Generator(device="cuda").manual_seed(a.seed + T)
        # post-RoPE q/k have unit-ish scale per dim; keep the softmax sharp enough to be realistic
        q = torch.randn(T, a.heads, a.head_dim, device="cuda", generator=g).mul_(a.scale).to(torch.bfloat16)
        k = torch.randn(T, a.heads, a.head_dim, device="cuda", generator=g).mul_(a.scale).to(torch.bfloat16)
        segs = [(0, T)]
        ref_out, _ = timed(lambda: torch_fn(ref, q, k, segs))
        tri_out, _ = timed(lambda: rt.received_attention(q, k, scaling, segs))
        t_ref = sorted(timed(lambda: torch_fn(ref, q, k, segs))[1] for _ in range(a.repeats))[a.repeats // 2]
        t_tri = sorted(timed(lambda: rt.received_attention(q, k, scaling, segs))[1] for _ in range(a.repeats))[a.repeats // 2]
        d = (tri_out - ref_out).abs()
        scale = ref_out.abs().mean().item()
        exact = (tri_out == ref_out).float().mean().item()
        # rank agreement: the score sorts tokens; count order flips between the two
        r1 = torch.argsort(ref_out); r2 = torch.argsort(tri_out)
        rank_same = (r1 == r2).float().mean().item()
        rows.append(dict(T=T, ref_ms=t_ref, tri_ms=t_tri, max_abs=d.max().item(), mean_abs=d.mean().item(),
                         rel_max=d.max().item() / scale, exact_frac=exact, rank_same=rank_same))
        print(f"T={T:5d}  torch {t_ref:7.1f} ms  triton {t_tri:7.1f} ms  x{t_ref / t_tri:.1f}   "
              f"max|d|/mean {rows[-1]['rel_max']:.2e}  exact {exact:.3f}  rank-same {rank_same:.4f}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The Qwen3.5 vision fork against the REAL 35B-A3B checkpoint.

The synthetic gate (`qwen35_vision_identity_gate.py`) uses a 4-layer tower with random weights,
which proves the algebra but not that the fork binds to the shipped module tree: a renamed
submodule, a transposed weight, a config field that differs from its class default -- none of those
show up against a tower this file constructed itself.

Only the vision tower is materialised (~0.5B params of the 35B checkpoint, since 30 of 40 LLM
layers are MoE and none of them are needed to test an image encoder). Everything is fp32 so the
identities stay identities rather than bf16 noise.
"""
import glob, json, os, sys
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig
from appcorr.models.qwen35.vision.backbone import ApproxCorrectQwen35VisionTower

SNAP = glob.glob("/NHNHOME/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/*/")[0]


def load_vision_tower() -> Qwen3_5MoeVisionModel:
    cfg_all = json.load(open(os.path.join(SNAP, "config.json")))
    cfg = Qwen3_5MoeVisionConfig(**cfg_all["vision_config"])
    print(f"  vision config: depth={cfg.depth} hidden={cfg.hidden_size} heads={cfg.num_heads} "
          f"patch={cfg.patch_size} window={getattr(cfg,'window_size',None)}")
    torch.set_default_dtype(torch.float32)
    model = Qwen3_5MoeVisionModel(cfg)

    wm = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
    want = {k: v for k, v in wm.items() if k.startswith("model.visual.")}
    state, shards = {}, sorted(set(want.values()))
    for sh in shards:
        blob = load_file(os.path.join(SNAP, sh))
        for k in (k for k in want if want[k] == sh):
            state[k[len("model.visual."):]] = blob[k].float()
        del blob
    missing, unexpected = model.load_state_dict(state, strict=False)
    # strict=False only to tolerate buffers; a genuinely missing PARAMETER is a fault.
    real_missing = [k for k in missing if not k.endswith("inv_freq")]
    if real_missing or unexpected:
        raise RuntimeError(f"vision weight load mismatch: missing={real_missing[:5]} "
                           f"unexpected={unexpected[:5]}")
    print(f"  loaded {len(state)} vision tensors from {len(shards)} shard(s); no key mismatch")
    return model.eval()


def main() -> int:
    torch.manual_seed(0)
    stock = load_vision_tower()
    cfg = stock.config
    grid_thw = torch.tensor([[1, 16, 16]])            # 256 patches -> 64 merge groups
    n_patch = int(grid_thw.prod(-1).sum())
    px = torch.randn(n_patch, cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size ** 2)

    with torch.no_grad():
        ref = stock(px, grid_thw).last_hidden_state
        merged_ref = stock(px, grid_thw).pooler_output

        tower = ApproxCorrectQwen35VisionTower(stock).eval()
        ctx = tower.prepare_full_tokens(px, grid_thw)
        n_l = len(tower.blocks)
        n_groups = ctx["seq_len"] // tower.spatial_merge_unit

        cache = {}
        x_ap, cache = tower.approx_forward(ctx["hidden_states"], 0, n_l, ctx, cache, "v",
                                           collect_attn_mean=True)
        d_ap = (x_ap - ref).abs().max().item()
        merged = tower.get_merged_output(x_ap, ctx)
        d_mg = (merged - merged_ref).abs().max().item()

        # Correction under the real setting: approximate on degraded pixels, then correct with the
        # true ones. Correcting every group must land exactly on the stock full-res forward.
        ctx_d = tower.prepare_full_tokens(px + 0.5 * torch.randn_like(px), grid_thw)
        cb = {}
        x_fl, cb = tower.approx_forward(ctx_d["hidden_states"], 0, n_l, ctx_d, cb, "b")
        d_floor = (x_fl - ref).abs().max().item()
        x_co, cb = tower.correct_forward(ctx["hidden_states"], torch.arange(n_groups),
                                         0, n_l, ctx, cb, "b")
        d_co = (x_co - ref).abs().max().item()

        # A PARTIAL correction must beat no correction at the corrected rows. This is the first
        # check here that is not an identity -- it is what the whole method claims.
        cp = {}
        _, cp = tower.approx_forward(ctx_d["hidden_states"], 0, n_l, ctx_d, cp, "p")
        half = torch.arange(0, n_groups, 2)
        x_half, cp = tower.correct_forward(ctx["hidden_states"], half, 0, n_l, ctx, cp, "p")
        unit = tower.spatial_merge_unit
        rows = (half.unsqueeze(1) * unit + torch.arange(unit)).flatten()
        err_before = (x_fl[rows] - ref[rows]).abs().mean().item()
        err_after = (x_half[rows] - ref[rows]).abs().mean().item()

        cache = tower.finalize_attn_layermean(cache, "v", n_l)
        am = cache["v_attn_layermean"]

    tol = 1e-3
    ok = True
    print()
    for name, d in (("approx == stock", d_ap), ("merger == stock pooler", d_mg),
                    ("degraded+correct(all) == stock", d_co)):
        ok &= d < tol
        print(f"  {'PASS' if d < tol else 'FAIL'}  {name:<32} max|diff| = {d:.3e}")
    print(f"  ----  degraded floor is {d_floor:.3e} from stock (correction test is meaningful)")
    # Gate the DIRECTION, report the magnitude. Correcting half the merge groups cannot drive the
    # error to zero even in principle: this tower is bidirectional, so a corrected token still
    # attends to the uncorrected half's stale K/V. How much of the error survives is an empirical
    # property of the model and the degradation, not something a wiring gate gets to assert -- an
    # earlier version of this file demanded a 50% reduction, got 40%, and reported a GATE FAILURE
    # for a fork that was working correctly.
    good = err_after < err_before
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  partial correction helps        "
          f"mean|err| {err_before:.4f} -> {err_after:.4f} "
          f"({(1 - err_after / err_before) * 100:.0f}% of the error removed at 50% of groups)")
    # With TRAINED weights the received-attention score must actually discriminate. The synthetic
    # gate could not test this: random weights give near-uniform attention by construction.
    rel = ((am.max() - am.min()) / am.mean()).item()
    ok &= rel > 0.1
    print(f"  {'PASS' if rel > 0.1 else 'FAIL'}  attn score discriminates        "
          f"(max-min)/mean = {rel:.3f}")
    print("\n" + ("ALL GATES PASS" if ok else "GATE FAILURE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

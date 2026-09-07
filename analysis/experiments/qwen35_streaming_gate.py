"""Is chunked prefill the same computation as one-shot prefill on Qwen3.5-MoE?

For a causal decoder this is an IDENTITY, not an approximation -- so it is gated as one. What makes
it worth checking rather than assuming is that 30 of 40 layers are `Qwen3_5MoeGatedDeltaNet`, a
recurrent linear-attention layer. Its entire history lives in a running state carried through the
cache, so if that state fails to cross a chunk boundary correctly the model still produces
plausible logits -- no error, no warning, just a wrong prefill. The full-attention layers would keep
working, which is exactly the kind of partial failure that survives a smoke test.

Everything runs in fp32. bf16 reduction order is shape-dependent, and chunking changes the shapes,
so a bf16 comparison could only ever be "close" -- which would hide precisely the failure this is
looking for.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
from appcorr.models.qwen35.llm.streaming import stream_prefill, assert_contiguous


def main() -> int:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)
    # Keep the real layer_types PATTERN (linear-heavy with periodic full attention); shrink
    # everything else. The identity does not depend on width or expert count, but it very much
    # depends on both layer kinds being present and interleaved.
    layer_types = ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"]
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=len(layer_types),
        num_attention_heads=8, num_key_value_heads=2, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=64, layer_types=layer_types,
    )
    model = Qwen3_5MoeForCausalLM(cfg).eval()
    for p in model.parameters():
        p.data.normal_(0, 0.02)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    n_lin = sum(t == "linear_attention" for t in layer_types)
    print(f"  model: {len(layer_types)} layers ({n_lin} linear_attention / "
          f"{len(layer_types) - n_lin} full_attention), {cfg.num_experts} experts")

    T = 96
    ids = torch.randint(0, cfg.vocab_size, (1, T), device=dev)

    with torch.no_grad():
        ref, _ = stream_prefill(model, input_ids=ids, boundaries=[0, T])

        cases = {
            "2 equal chunks":     [0, 48, T],
            "4 equal chunks":     [0, 24, 48, 72, T],
            "ragged chunks":      [0, 7, 31, 32, 90, T],
            "many small chunks":  list(range(0, T, 8)) + [T],
        }
        results = {}
        for name, b in cases.items():
            out, _ = stream_prefill(model, input_ids=ids, boundaries=b)
            # `stream_prefill` returns only the FINAL chunk's logits (see its docstring: a prefill
            # exists to produce the next token). So compare against the matching tail of the
            # one-shot run -- the whole tail, not just the last row, since a state that crossed the
            # boundary slightly wrong would still land on a plausible final position.
            results[name] = (out - ref[:, b[-2]:]).abs().max().item()

    ok = True
    tol = 1e-4
    print()
    for name, d in results.items():
        good = d < tol
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:<20} max|logit diff| vs one-shot = {d:.3e}")

    # A gate that only ever compares equal things proves nothing about its own sensitivity. Feed a
    # DIFFERENT sequence through the same path: if that also came out ~0, the comparison would be
    # measuring nothing (this repo has hit that exact trap -- three oracles agreeing to three
    # decimals because the score never reached the ranking code).
    with torch.no_grad():
        other, _ = stream_prefill(model, input_ids=torch.randint(0, cfg.vocab_size, (1, T), device=dev),
                                  boundaries=[0, 48, T])
    sens = (other - ref[:, 48:]).abs().max().item()
    good = sens > 1e-2
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {'sensitivity check':<20} different input differs by {sens:.3e}")

    # The contiguity contract must actually refuse what it says it refuses. Note what is NOT
    # tested: a "gap". In the boundary-list representation a gap is unexpressible -- consecutive
    # boundary pairs are contiguous by construction, so the only invalid shapes are a non-ascending
    # list, a start that is not 0, and an end that is not T. (The first draft of this gate listed
    # [0, 10, 20, T] as a "gap" and failed its own validator for correctly accepting a valid
    # partition.)
    bad = [("empty", []), ("overlap", [0, 50, 40, T]), ("not from 0", [10, 50, T]),
           ("short of T", [0, 50, T - 1])]
    for name, b in bad:
        try:
            assert_contiguous([x for x in b], T)
            print(f"  FAIL  rejects {name:<12} accepted an invalid partition")
            ok = False
        except ValueError:
            print(f"  PASS  rejects {name:<12} raised as designed")

    print("\n" + ("ALL GATES PASS" if ok else "GATE FAILURE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

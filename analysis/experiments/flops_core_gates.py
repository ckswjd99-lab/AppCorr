"""Gates for the FLOP accounting core, on a synthetic module so the arithmetic is checkable by hand.

A FLOP counter fails by returning a confidently wrong number, never by raising, so every claim it
makes is gated here against a value derived independently:

  1. **Arithmetic.** Linear, Conv and attention counts match closed forms computed from the shapes
     the test itself chose. If the hook reads the wrong dimension this fires.
  2. **GQA head count.** Attention is charged at the QUERY's head count. Reading the key's heads
     instead under-reports by the group factor -- 4x on Qwen3's 32/8 -- and is invisible in any
     end-to-end total that has no independent reference.
  3. **The critical rule**, on the four arm shapes this repo actually runs:
     no arrivals -> 100% critical (floor, ceiling); approx then one correction -> only the
     correction; g rounds -> only the last; and g=1 interleaved -> identical to one-shot.
  4. **Off is off.** Disabled, nothing is recorded AND `scaled_dot_product_attention` is the
     original function object -- not a wrapper that happens to skip work.
  5. **Scope.** A module outside the subtree passed to `session` is never counted, which is how
     heads and decode are excluded.
  6. **Re-entrancy.** Nested sessions restore attention exactly once; a leaked wrapper would keep
     counting into a dead counter for the rest of the process.

CPU only, seconds to run, no model download.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from appcorr import flops

OK = True


def check(name: str, got, want, tol: float = 0.0) -> None:
    global OK
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        good = abs(got - want) <= tol * max(abs(want), 1)
        # Format by magnitude, not by whether a tolerance was passed: `:,.0f` on a fraction rounds
        # 0.2 to "0", so a passing check printed "got 0 want 0" and read like a failure.
        fmt = (lambda v: f"{v:,}") if float(want).is_integer() and abs(want) >= 1 \
            else (lambda v: f"{v:.6g}")
        detail = f"got {fmt(got)} want {fmt(want)}"
    else:
        good = got == want
        detail = f"got {got!r} want {want!r}"
    OK &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name:<52} {detail}")


class Tiny(nn.Module):
    """One attention block with deliberately distinct dimensions, so a swapped axis cannot pass."""

    def __init__(self, d=64, heads=4, kv_heads=2, hidden=192):
        super().__init__()
        self.h, self.kv, self.hd = heads, kv_heads, d // heads
        self.q = nn.Linear(d, heads * self.hd, bias=False)
        self.k = nn.Linear(d, kv_heads * self.hd, bias=False)
        self.v = nn.Linear(d, kv_heads * self.hd, bias=False)
        self.o = nn.Linear(heads * self.hd, d, bias=False)
        self.fc1 = nn.Linear(d, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        b, s, d = x.shape
        q = self.q(x).view(b, s, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(b, s, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(b, s, self.kv, self.hd).transpose(1, 2)
        k = k.repeat_interleave(self.h // self.kv, dim=1)      # GQA expansion, as the real forks do
        v = v.repeat_interleave(self.h // self.kv, dim=1)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x + self.o(a.transpose(1, 2).reshape(b, s, -1))
        return x + self.fc2(self.fc1(x))


def main() -> None:
    torch.manual_seed(0)
    B, S, D, H, KV, HID = 2, 16, 64, 4, 2, 192
    HD = D // H
    m = Tiny(D, H, KV, HID).eval()
    x = torch.randn(B, S, D)

    lin = 2 * B * S * (D * H * HD + 2 * D * KV * HD + H * HD * D + D * HID + HID * D)
    attn = 2 * 2 * B * H * S * S * HD                      # QK^T and AV, at the QUERY's head count

    # --- 1 & 2: arithmetic, and GQA charged at the query's heads --------------------------------- #
    with flops.session(m, enabled=True) as fl:
        with torch.no_grad(), fl.request("r0"):
            m(x)
    r = fl.requests[0]
    got_lin = sum(b.linear for b in r.buckets.values())
    got_att = sum(b.attention for b in r.buckets.values())
    check("Linear FLOPs", got_lin, lin)
    check("attention FLOPs (GQA at query heads)", got_att, attn)
    check("attention NOT charged at kv heads", got_att != 2 * 2 * B * KV * S * S * HD, True)

    # --- Conv ------------------------------------------------------------------------------------ #
    conv = nn.Conv2d(3, 8, kernel_size=4, stride=4, bias=False)
    img = torch.randn(1, 3, 16, 16)
    with flops.session(conv, enabled=True) as fl2:
        with torch.no_grad(), fl2.request("c"):
            out = conv(img)
    check("Conv FLOPs", sum(b.conv for b in fl2.requests[0].buckets.values()),
          2 * out.numel() * 3 * 4 * 4)

    # --- 3: the critical rule on the four arm shapes --------------------------------------------- #
    print()
    with flops.session(m, enabled=True) as fl3:
        with torch.no_grad():
            # floor / ceiling: a single pass, no arrival ever opened
            with fl3.request("stock"):
                m(x)
            # one-shot: approximate pass overlaps, the single correction does not
            with fl3.request("oneshot") as rq:
                with fl3.arrival(0), fl3.stage("approx"):
                    m(x)
                with fl3.arrival(1), fl3.stage("correct"):
                    m(x)
            # interleaved g=4
            with fl3.request("g4"):
                with fl3.arrival(0), fl3.stage("approx"):
                    m(x)
                for rnd in range(4):
                    with fl3.arrival(1 + rnd), fl3.stage("correct"):
                        m(x)
            # g=1 must land exactly where one-shot did
            with fl3.request("g1"):
                with fl3.arrival(0), fl3.stage("approx"):
                    m(x)
                with fl3.arrival(1), fl3.stage("correct"):
                    m(x)
    by = {r.request_id: r for r in fl3.requests}
    one = lin + attn
    check("stock (no arrivals) is 100% critical", by["stock"].critical_fraction, 1.0)
    check("one-shot: critical == the correction only", by["oneshot"].critical, one)
    check("one-shot: approx counted but not critical", by["oneshot"].overlappable, one)
    check("g=4: critical == final round only", by["g4"].critical, one)
    check("g=4: total == approx + 4 rounds", by["g4"].total, 5 * one)
    check("g=4 critical fraction == 1/5 of what it did", by["g4"].critical_fraction, 0.2, 1e-9)
    check("g=1 interleaved == one-shot", by["g1"].critical, by["oneshot"].critical)

    # --- 3b: PREPARE_TOKENS is excluded from total/critical, not just relabelled ------------------ #
    # ADE20K's m2f segmentor re-runs its (expensive) token embed on every interleaved round --
    # `ADE20KWindowInterleavedPolicy` prepends `OpType.PREPARE_TOKENS` to every group's task -- which
    # measured at 6.75x the model's own ceiling and identical between keep=0.25 and keep=0.50, i.e.
    # scaling with round count rather than with what was actually corrected. Not backbone compute,
    # so it must not move `total`/`critical`, but `by_stage()` must still show it or the next one of
    # these goes undiagnosed the same way.
    with flops.session(m, enabled=True) as fl3b:
        with torch.no_grad():
            with fl3b.request("with_prep"):
                with fl3b.arrival(0), fl3b.stage("PREPARE_TOKENS"):
                    m(x)
                with fl3b.arrival(0), fl3b.stage("approx"):
                    m(x)
                with fl3b.arrival(1), fl3b.stage("correct"):
                    m(x)
    rq = fl3b.requests[0]
    check("PREPARE_TOKENS excluded from total", rq.total, 2 * one)
    check("PREPARE_TOKENS excluded from critical", rq.critical, one)
    check("PREPARE_TOKENS still visible in by_stage", rq.by_stage().get("PREPARE_TOKENS"), one)

    # --- 4: off is off --------------------------------------------------------------------------- #
    print()
    orig = torch.nn.functional.scaled_dot_product_attention
    with flops.session(m, enabled=False) as fl4:
        check("disabled: sdpa is the ORIGINAL function",
              torch.nn.functional.scaled_dot_product_attention is orig, True)
        with torch.no_grad(), fl4.request("off"):
            m(x)
    check("disabled: nothing recorded", fl4.requests[0].total, 0)
    check("disabled: sdpa restored", torch.nn.functional.scaled_dot_product_attention is orig, True)

    # --- 5: scope excludes what is not passed ---------------------------------------------------- #
    head = nn.Linear(D, 1000, bias=False)
    with flops.session(m, enabled=True) as fl5:          # head deliberately NOT in the subtree
        with torch.no_grad(), fl5.request("scoped"):
            head(m(x))
    check("head outside the subtree is not counted", fl5.requests[0].total, one)

    # --- 6: nested sessions restore exactly once ------------------------------------------------- #
    with flops.session(m, enabled=True) as outer:
        with flops.session(m, enabled=True) as inner:
            with torch.no_grad(), inner.request("i"):
                m(x)
        check("inner session closed, sdpa still wrapped",
              torch.nn.functional.scaled_dot_product_attention is not orig, True)
    check("outer session closed, sdpa restored",
          torch.nn.functional.scaled_dot_product_attention is orig, True)

    print("\n" + ("ALL GATES PASS" if OK else "GATE FAILED"))
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()

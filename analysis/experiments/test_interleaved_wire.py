"""CPU unit test for the interleaved wire path: the `correct` frame and `StreamSink.correct`.

No vLLM, no GPU, no server -- `wire.py` imports only numpy/torch and `bridge.py` imports only
`wire`, so the whole client half of docs/memo/vllm_interleaved_design.md §3.1/§3.3 is testable in
the `appcorr` env. What it pins down:

  * the frame round-trips EXACTLY: bf16 travels as its int16 bit pattern, so the embeds that come
    out the far side must compare bit-for-bit, not "close" (a corrected row is the LLM's input;
    a rounded one is a different prompt);
  * positions stay int64 and strictly increasing, and the window survives as a pair of ints;
  * `StreamSink.correct` refuses the states that would corrupt a request (before the prompt was
    pushed, after final), records a `{"kind": "correct", ...}` push record, and closes the sink
    on final.

    python analysis/experiments/test_interleaved_wire.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from appcorr.vllm_stream.wire import Frame, FrameParser          # noqa: E402
from appcorr.vllm_stream.bridge import BridgeError, StreamSink   # noqa: E402


class _FakeBridge:
    """Records what a sink hands the transport; no socket, no threads."""

    timeout_s = 1.0

    def __init__(self):
        self.calls = []

    def open(self, rid, embeds, mrope, mrope_delta, final, max_tokens, logprobs=None,
             sink=None, rec=None):
        self.calls.append(("open", rid, embeds, mrope, final, rec))

    def append(self, rid, embeds, mrope, mrope_delta, final, sink=None, rec=None):
        self.calls.append(("append", rid, embeds, mrope, final, rec))

    def correct(self, rid, positions, embeds, window, final, sink=None, rec=None):
        self.calls.append(("correct", rid, positions, embeds, tuple(window), final, rec))


def _mk(p=7, d=16, lo=5):
    """A round's payload: bf16 rows with awkward bit patterns and non-contiguous positions."""
    torch.manual_seed(0)
    embeds = (torch.randn(p, d) * 1e3).to(torch.bfloat16)
    embeds[0, 0] = torch.tensor(float("-0.0")).to(torch.bfloat16)   # sign bit alone
    positions = torch.tensor(sorted(torch.randperm(40)[:p].tolist()), dtype=torch.int64) + lo
    return positions, embeds


def test_frame_roundtrip():
    positions, embeds = _mk()
    window = (5, 60)
    f = Frame({"op": "correct", "rid": "r-0", "final": True, "window": [window[0], window[1]]})
    f.put_tensor("positions", positions).put_tensor("embeds", embeds)
    blob = f.encode()

    # through the incremental parser, byte by byte in two pieces, as the server reads it
    parser = FrameParser()
    assert parser.feed(blob[:3]) == []
    got = parser.feed(blob[3:])
    assert len(got) == 1, len(got)
    g = got[0]

    assert g.header["op"] == "correct" and g.header["rid"] == "r-0"
    assert g.header["final"] is True
    assert g.header["window"] == [5, 60] and all(isinstance(v, int) for v in g.header["window"])
    p2, e2 = g.get_tensor("positions"), g.get_tensor("embeds")
    assert p2.dtype == torch.int64, p2.dtype
    assert torch.equal(p2, positions), (p2, positions)
    assert bool((p2[1:] > p2[:-1]).all()), "positions must stay strictly increasing"
    assert e2.dtype == torch.bfloat16, e2.dtype
    assert e2.shape == embeds.shape, (e2.shape, embeds.shape)
    # bit pattern, not allclose: -0.0 and every mantissa must survive the int16 reinterpretation
    assert torch.equal(e2.view(torch.int16), embeds.view(torch.int16)), "bf16 bits changed"
    print(f"ok  frame round-trip: {tuple(embeds.shape)} bf16 rows, {p2.numel()} int64 positions, "
          f"window {tuple(g.header['window'])}, {len(blob)} bytes")


def test_sink_correct():
    br = _FakeBridge()
    sink = StreamSink(br, "r-1", max_tokens=8, logprobs=None)
    positions, embeds = _mk(p=4, d=8)

    try:
        sink.correct(positions, embeds, (0, 40), final=False)
    except BridgeError as e:
        assert "before the prompt was pushed" in str(e), e
    else:
        raise AssertionError("correct before the opening push must be refused")

    sink.push(torch.zeros(64, 8, dtype=torch.bfloat16), torch.zeros(3, 64, dtype=torch.int64),
              0, final=False)
    assert br.calls[-1][0] == "open" and br.calls[-1][4] is False
    assert sink.pushes[-1]["kind"] == "push", sink.pushes[-1]

    sink.correct(positions, embeds, (5, 60), final=False)
    op, rid, p_sent, e_sent, window, final, rec = br.calls[-1]
    assert op == "correct" and rid == "r-1" and window == (5, 60) and final is False
    assert torch.equal(p_sent, positions) and torch.equal(e_sent.view(torch.int16),
                                                          embeds.view(torch.int16))
    assert rec["kind"] == "correct" and rec["n"] == 4 and rec["window"] == [5, 60]
    assert rec["t_ack"] is None and rec["t_recv_server"] is None and "t_queued" in rec
    assert sink.pushes[-1] is rec and not sink.closed
    # a correct adds no PROMPT rows: the whole prompt went in with the opening push
    assert sink.num_tokens == 64, sink.num_tokens

    sink.correct(positions, embeds, (5, 60), final=True)
    assert sink.closed and sink.pushes[-1]["final"] is True
    for bad, args in (("correct", (positions, embeds, (5, 60), True)),):
        try:
            getattr(sink, bad)(*args)
        except BridgeError as e:
            assert "after final" in str(e), e
        else:
            raise AssertionError(f"{bad} after final must be refused")

    kinds = [r["kind"] for r in sink.pushes]
    assert kinds == ["push", "correct", "correct"], kinds
    print(f"ok  StreamSink.correct: {len(sink.pushes)} records {kinds}, closed={sink.closed}")


def test_sink_shape_guards():
    br = _FakeBridge()
    sink = StreamSink(br, "r-2", max_tokens=8, logprobs=None)
    sink.push(torch.zeros(16, 8, dtype=torch.bfloat16), None, None, final=False)
    positions, embeds = _mk(p=3, d=8)
    for name, pos, emb in (("row count", positions[:2], embeds),
                           ("dtype", positions.to(torch.int32), embeds),
                           ("rank", positions, embeds[0])):
        try:
            sink.correct(pos, emb, (0, 15), final=False)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"{name} mismatch must be refused")
    print("ok  StreamSink.correct shape/dtype guards")


if __name__ == "__main__":
    test_frame_roundtrip()
    test_sink_correct()
    test_sink_shape_guards()
    print("INTERLEAVED_WIRE_TESTS_PASS")

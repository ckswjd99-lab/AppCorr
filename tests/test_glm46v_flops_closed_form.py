"""GLM-4.6V closed-form FLOPs: config facts + a hand count on one small prompt.

CPU only, no model load: the architecture numbers come from the downloaded checkpoint's
`config.json` and its safetensors index, the FLOPs from arithmetic written out here by hand and
compared against `flops_analytic.Glm46VDecoder` / `Glm46VVision` and the axis cost hooks in
`appcorr/models/glm46v/unified.py`.

The hand count is deliberately written as flat arithmetic on the raw dimensions rather than by
re-calling the implementation's helpers -- a test that reuses the formula under test proves only
that Python is deterministic.

Sample: N = 100 prompt tokens, 64 image patch rows (= 16 merge groups = 16 image tokens; a
16x16-pixel-patch 8x8 grid).  Small on purpose: every term is checkable by hand.

    python -m pytest tests/test_glm46v_flops_closed_form.py -q
"""
import json
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "analysis", "experiments"))

from flops_analytic import (GLM46V_VISION, MODELS46, Glm46VDecoder,  # noqa: E402
                            glm46v_from_config, stage_bounds)
from appcorr.models.glm46v.unified import (Glm46VUnifiedCosts, llm_stage_costs,  # noqa: E402
                                           lm_head_cost, vision_merge_cost, vision_pre_cost,
                                           vision_stage_cost)

SNAPSHOT = ("/NHNHOME/huggingface/hub/models--zai-org--GLM-4.6V-FP8/snapshots/"
            "33172e26eb88482cf3d0a36fced01d05454734ec")

N = 100          # prompt tokens
ROWS = 64        # image patch rows
GROUPS = 16      # merge groups = image tokens = ROWS / 2**2


def _cfg():
    raw = json.load(open(os.path.join(SNAPSHOT, "config.json")))
    ns = types.SimpleNamespace(**{k: v for k, v in raw.items() if not isinstance(v, dict)})
    ns.text_config = types.SimpleNamespace(**raw["text_config"])
    ns.vision_config = types.SimpleNamespace(**raw["vision_config"])
    return ns


class _FakeAxis(Glm46VUnifiedCosts):
    def __init__(self, cfg):
        self.cfg = cfg


@unittest.skipUnless(os.path.isdir(SNAPSHOT), "GLM-4.6V-FP8 snapshot not present")
class Glm46VConfigFactsTest(unittest.TestCase):
    """Every number the closed form hard-codes, against the checkpoint itself."""

    def test_text_config(self):
        t = _cfg().text_config
        self.assertEqual(t.num_hidden_layers, 46)
        self.assertEqual(t.hidden_size, 4096)
        self.assertEqual((t.num_attention_heads, t.num_key_value_heads), (96, 8))
        self.assertEqual(t.head_dim, 128)
        self.assertTrue(t.attention_bias)          # q/k/v bias; o has none (glm4_moe.py:270-277)
        self.assertFalse(t.use_qk_norm)
        self.assertEqual(t.first_k_dense_replace, 1)
        self.assertEqual(t.intermediate_size, 10944)
        self.assertEqual((t.n_routed_experts, t.num_experts_per_tok), (128, 8))
        self.assertEqual((t.moe_intermediate_size, t.n_shared_experts), (1408, 1))
        self.assertEqual(t.routed_scaling_factor, 1.0)
        self.assertEqual(t.vocab_size, 151552)
        self.assertEqual(t.rope_parameters["mrope_section"], [8, 12, 12])
        self.assertEqual(t.rope_parameters["partial_rotary_factor"], 0.5)
        self.assertEqual(t.num_nextn_predict_layers, 0)     # no MTP layers

    def test_vision_config(self):
        v = _cfg().vision_config
        self.assertEqual(v.depth, 24)
        self.assertEqual(v.hidden_size, 1536)
        self.assertEqual(v.num_heads, 12)                   # head_dim 128
        self.assertEqual(v.out_hidden_size, 4096)           # block MLP hidden AND merger d_model
        self.assertEqual(v.intermediate_size, 10944)        # merger context_dim
        self.assertEqual((v.patch_size, v.temporal_patch_size, v.spatial_merge_size), (14, 2, 2))
        self.assertEqual(v.in_channels, 3)

    def test_checkpoint_tensor_shapes_match_the_closed_form(self):
        """The shapes the formula assumes, read off the safetensors index (no load)."""
        import struct
        idx = json.load(open(os.path.join(SNAPSHOT, "model.safetensors.index.json")))["weight_map"]

        def shape(key):
            with open(os.path.join(SNAPSHOT, idx[key]), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                return json.loads(fh.read(n))[key]["shape"]

        # q_proj is heads*dh with NO output gate (Qwen3.5's is 2x that)
        self.assertEqual(shape("model.language_model.layers.1.self_attn.q_proj.weight"),
                         [96 * 128, 4096])
        self.assertEqual(shape("model.language_model.layers.1.self_attn.k_proj.weight"),
                         [8 * 128, 4096])
        self.assertEqual(shape("model.language_model.layers.0.mlp.gate_proj.weight"),
                         [10944, 4096])                     # layer 0 dense
        self.assertEqual(shape("model.language_model.layers.1.mlp.experts.0.gate_proj.weight"),
                         [1408, 4096])                      # routed expert
        self.assertEqual(shape("model.language_model.layers.1.mlp.shared_experts.gate_proj.weight"),
                         [1408, 4096])                      # ONE shared expert of the same width
        # lm_head: 151552, not the port plan's 154880
        self.assertEqual(shape("lm_head.weight"), [151552, 4096])
        # vision: gated block MLP, Conv3d patch embed, 2x2 downsample, merger 4096 -> 10944
        self.assertEqual(shape("model.visual.blocks.0.mlp.gate_proj.weight"), [4096, 1536])
        self.assertEqual(shape("model.visual.patch_embed.proj.weight"), [1536, 3, 2, 14, 14])
        self.assertEqual(shape("model.visual.downsample.weight"), [4096, 1536, 2, 2])
        self.assertEqual(shape("model.visual.merger.proj.weight"), [4096, 4096])
        self.assertEqual(shape("model.visual.merger.gate_proj.weight"), [10944, 4096])

    def test_frozen_entry_matches_the_config(self):
        self.assertEqual(glm46v_from_config(os.path.join(SNAPSHOT, "config.json")),
                         MODELS46["glm46v"])


class Glm46VHandCountTest(unittest.TestCase):
    """N = 100 tokens, 64 image rows, every term written out."""

    dec = MODELS46["glm46v"]

    # -- decoder ------------------------------------------------------------------------------
    # per token, one layer:
    #   q  2 * 4096 * (96*128)      = 2 * 4096 * 12288
    #   k  2 * 4096 * (8*128)       = 2 * 4096 * 1024   (x2 for v)
    #   o  2 * (96*128) * 4096
    PROJ = 2 * 4096 * 12288 + 2 * (2 * 4096 * 1024) + 2 * 12288 * 4096
    # per layer, whole prefill: QK^T + AV at 96 query heads, Sq = Sk = 100, D = 128
    QUAD = 2 * 2 * 96 * 100 * 100 * 128
    # layer 0: dense SwiGLU gate/up [4096 x 10944] + down [10944 x 4096]
    DENSE = 2 * 4096 * 10944 + 2 * 4096 * 10944 + 2 * 10944 * 4096
    # layers 1-45: 8 routed experts x (gate + up + down at width 1408) + 128-way router
    #              + one shared expert of width 1408
    MOE = (8 * (2 * 4096 * 1408 + 2 * 4096 * 1408 + 2 * 1408 * 4096)
           + 2 * 4096 * 128
           + (2 * 4096 * 1408 + 2 * 4096 * 1408 + 2 * 1408 * 4096))

    def test_per_token_pieces(self):
        self.assertEqual(self.dec.proj_tok(), self.PROJ)
        self.assertEqual(self.dec.dense_tok(), self.DENSE)
        self.assertEqual(self.dec.moe_tok(), self.MOE)
        self.assertEqual(self.dec.attn(100, 100), self.QUAD)
        # sanity on the magnitudes: 12.3 GF/token of routed+shared MoE work, 0.22 GF of attention
        self.assertAlmostEqual(self.MOE / 1e9, 0.3125, places=3)
        self.assertAlmostEqual(self.PROJ / 1e9, 0.2181, places=3)

    def test_prefill_hand_count(self):
        hand = (100 * 46 * self.PROJ            # projections, every layer, every token
                + 46 * self.QUAD                # attention, every layer
                + 100 * self.DENSE              # layer 0's dense MLP
                + 100 * 45 * self.MOE)          # layers 1-45's MoE blocks
        self.assertEqual(self.dec.prefill_flops(100), hand)
        self.assertAlmostEqual(hand / 1e12, 2.4589, places=3)
        # Independent cross-check at a different altitude: a dense forward costs ~2 x active
        # params per token.  Active params per token (no embeddings, no lm_head) =
        # 46 x (q + k + v + o) + 1 x dense MLP + 45 x (router + 8 routed + 1 shared expert).
        active = (46 * (4096 * 12288 + 2 * 4096 * 1024 + 12288 * 4096)
                  + 3 * 4096 * 10944
                  + 45 * (4096 * 128 + 9 * 3 * 4096 * 1408))
        per_tok = hand / 100 - 46 * self.QUAD / 100      # strip the quadratic term
        self.assertAlmostEqual(per_tok / (2 * active), 1.0, places=6)
        self.assertAlmostEqual(active / 1e9, 12.15, places=1)     # "A12B" checks out

    def test_layer_costs_sum_to_the_prefill(self):
        cfg = _cfg() if os.path.isdir(SNAPSHOT) else None
        if cfg is None:
            self.skipTest("snapshot not present")
        costs = llm_stage_costs(cfg, 100)
        self.assertEqual(len(costs), 46)
        self.assertEqual(costs[0], float(100 * self.PROJ + self.QUAD + 100 * self.DENSE))
        self.assertEqual(costs[1], float(100 * self.PROJ + self.QUAD + 100 * self.MOE))
        self.assertEqual(len(set(costs[1:])), 1)            # layers 1-45 are identical
        self.assertAlmostEqual(sum(costs), self.dec.prefill_flops(100), delta=1.0)
        self.assertEqual(_FakeAxis(cfg)._llm_stage_costs(100), costs)

    def test_mlp_prefix_and_corrected_row(self):
        self.assertEqual(self.dec.mlp_prefix_tok(1), self.DENSE)
        self.assertEqual(self.dec.mlp_prefix_tok(3), self.DENSE + 2 * self.MOE)
        self.assertEqual(self.dec.mlp_prefix_tok(), self.DENSE + 45 * self.MOE)
        # one corrected row at position 99, full depth: projections + MLPs + 46 x attn(1, 100)
        hand = (46 * self.PROJ + self.DENSE + 45 * self.MOE
                + 46 * 2 * 2 * 96 * 1 * 100 * 128)
        self.assertEqual(self.dec.corrected_row_flops(99), hand)
        # ... and at a staged depth of 23 layers: 1 dense + 22 MoE
        hand23 = (23 * self.PROJ + self.DENSE + 22 * self.MOE
                  + 23 * 2 * 2 * 96 * 1 * 100 * 128)
        self.assertEqual(self.dec.corrected_row_flops(99, 23), hand23)
        self.assertEqual(stage_bounds(46, 2), [23, 46])

    def test_no_recurrent_term(self):
        self.assertEqual(self.dec.rescan_flops(1000), 0.0)
        self.assertEqual(self.dec.rescan_flops(1000, 12), 0.0)

    def test_lm_head_is_reported_not_added(self):
        self.assertEqual(self.dec.lm_head_flops(1), 2 * 4096 * 151552)
        if os.path.isdir(SNAPSHOT):
            self.assertEqual(lm_head_cost(_cfg(), 1), float(2 * 4096 * 151552))
        # and it is NOT inside the prefill
        self.assertLess(self.dec.lm_head_flops(1), 0.001 * self.dec.prefill_flops(100))

    # -- vision -------------------------------------------------------------------------------
    # one block over 64 rows: qkv [1536 x 3*1536], attn 12 heads x 64 x 64 x 128, proj, SwiGLU
    VBLOCK = (2 * 64 * 1536 * (3 * 1536)
              + 2 * 2 * 12 * 64 * 64 * 128
              + 2 * 64 * 1536 * 1536
              + 2 * 64 * 1536 * 4096 + 2 * 64 * 1536 * 4096 + 2 * 64 * 4096 * 1536)
    VPRE = 2 * 64 * 1536 * 3 * (2 * 14 * 14)
    VMERGE = (2 * (16 * 4096) * 1536 * (2 * 2)                       # 2x2 Conv2d downsample
              + 2 * 16 * 4096 * 4096                                 # merger proj
              + 2 * 16 * 4096 * 10944 + 2 * 16 * 4096 * 10944        # merger gate + up
              + 2 * 16 * 10944 * 4096)                               # merger down

    def test_vision_hand_count(self):
        v = GLM46V_VISION
        self.assertEqual(v.layer_flops(ROWS), self.VBLOCK)
        self.assertEqual(v.pre_flops(ROWS), self.VPRE)
        self.assertEqual(v.merge_flops(GROUPS), self.VMERGE)
        self.assertEqual(v.tower_flops(ROWS), self.VPRE + 24 * self.VBLOCK + self.VMERGE)
        # the gated MLP is 3 matmuls, not 2: an ungated form would be 1.55 GF lighter per block
        self.assertGreater(self.VBLOCK, 2 * 64 * 1536 * (3 * 1536) + 2 * 2 * 12 * 64 * 64 * 128
                           + 2 * 64 * 1536 * 1536 + 2 * 2 * 64 * 1536 * 4096)

    def test_vision_cost_hooks_match(self):
        if not os.path.isdir(SNAPSHOT):
            self.skipTest("snapshot not present")
        cfg = _cfg()
        self.assertEqual(vision_stage_cost(cfg, ROWS), float(self.VBLOCK))
        self.assertEqual(vision_pre_cost(cfg, ROWS), float(self.VPRE))
        self.assertEqual(vision_merge_cost(cfg, GROUPS), float(self.VMERGE))
        axis = _FakeAxis(cfg)
        self.assertEqual(axis._vision_stage_cost(ROWS), float(self.VBLOCK))
        self.assertEqual(axis._vision_pre_cost(ROWS), float(self.VPRE))
        self.assertEqual(axis._vision_merge_cost(GROUPS), float(self.VMERGE))

    def test_row_correction_is_exact_not_an_average(self):
        v = GLM46V_VISION
        self.assertAlmostEqual(v.row_layer_flops(ROWS) * ROWS, v.layer_flops(ROWS), delta=1.0)

    # -- the unified axis's stage list ---------------------------------------------------------
    def test_unified_axis_has_24_plus_46_stages(self):
        if not os.path.isdir(SNAPSHOT):
            self.skipTest("snapshot not present")
        cfg = _cfg()
        stages = ([vision_stage_cost(cfg, ROWS)] * 24) + llm_stage_costs(cfg, N)
        self.assertEqual(len(stages), 70)
        self.assertTrue(all(c > 0 for c in stages))


@unittest.skipUnless(os.path.isdir(SNAPSHOT), "GLM-4.6V-FP8 snapshot not present")
class AxisVsReportingClosedFormTest(unittest.TestCase):
    """The two closed forms in the tree must be ONE formula.

    `appcorr/models/glm46v/axis.py` carries the axis's bound-placing cost hooks (written in
    parallel with this half) and asks, in its own comment, that the reporting closed form be
    asserted equal to them -- the Qwen3.5 pair is checked the same way.  Equal here means to the
    float, not "close": both are exact integer arithmetic on the same config.
    """

    def setUp(self):
        try:
            from appcorr.models.glm46v.axis import Glm46VAxis
        except ImportError as e:                       # vision half not merged
            self.skipTest(f"axis.py not importable: {e}")
        self.axis_cls = Glm46VAxis
        self.obj = Glm46VAxis.__new__(Glm46VAxis)
        self.obj.cfg = _cfg()

    def test_llm_stage_costs_agree(self):
        a = self.axis_cls._llm_stage_costs(self.obj, N)
        self.assertEqual(a, llm_stage_costs(self.obj.cfg, N))              # == the mixin's
        self.assertEqual(a, _FakeAxis(self.obj.cfg)._llm_stage_costs(N))
        dec = MODELS46["glm46v"]                                           # == the report's
        self.assertAlmostEqual(sum(a), dec.prefill_flops(N), delta=1.0)
        for depth in (1, 12, 23, 46):
            self.assertAlmostEqual(
                sum(a[:depth]),
                depth * (N * dec.proj_tok() + dec.attn(N, N)) + N * dec.mlp_prefix_tok(depth),
                delta=1.0)

    def test_vision_stage_cost_agrees(self):
        for nr in (ROWS, 1024, 4096):
            a = self.axis_cls._vision_stage_cost(self.obj, nr)
            self.assertEqual(a, vision_stage_cost(self.obj.cfg, nr))
            self.assertEqual(a, float(GLM46V_VISION.layer_flops(nr)))

    def test_unified_bounds_are_the_same_on_both(self):
        from appcorr.models.qwen_vl_axis import QwenVLStreamingAxis
        cls = type("Glm46VMixinProbe", (Glm46VUnifiedCosts, QwenVLStreamingAxis), {})
        probe = cls.__new__(cls)
        probe.cfg = self.obj.cfg
        probe.tower = types.SimpleNamespace(blocks=[None] * 24)
        self.obj.tower = probe.tower
        for g in (1, 2, 4, 8):
            self.assertEqual(QwenVLStreamingAxis.unified_bounds(self.obj, g, 4096, 2048),
                             QwenVLStreamingAxis.unified_bounds(probe, g, 4096, 2048))
        self.assertEqual(len(QwenVLStreamingAxis.unified_stage_costs(self.obj, 4096, 2048)), 70)


class Glm46VScheduleReplayTest(unittest.TestCase):
    """The interleaved replay on a synthetic run: total = 1 approx pass + the rounds' rows."""

    dec = MODELS46["glm46v"]

    def test_keep1_g1(self):
        n, lo, n_img = 100, 20, 16          # rows 20..35 are the image, text suffix 36..98
        chunks = [("approx", 0, n - 1), ("correct", lo, n - 1, n_img + 63)]
        c = self.dec.interleaved_cost(n, lo, n_img, chunks)
        approx = self.dec.prefill_flops(n - 1)
        rows = (n_img + 63) * self.dec.corrected_row_flops((lo + n - 2) / 2.0)
        tail = self.dec.corrected_row_flops(n - 1)
        self.assertAlmostEqual(c["total"], approx + rows + tail, delta=1.0)
        self.assertAlmostEqual(c["crit"], rows + tail, delta=1.0)
        self.assertFalse(c["staged"])
        # no recurrent re-scan term, so total is exactly approx + corrections + the held-back row
        self.assertGreater(c["total"], approx)

    def test_staged_round_prices_at_its_depth(self):
        n, lo, n_img = 100, 20, 16
        chunks = [("approx", 0, n - 1),
                  ("correct", lo, lo + 8, 8, 0, 2),          # round 0 of 2 -> depth 23
                  ("correct", lo + 8, n - 1, 8 + 63, 1, 2)]  # round 1 -> full depth
        c = self.dec.interleaved_cost(n, lo, n_img, chunks)
        self.assertTrue(c["staged"])
        r0 = 8 * self.dec.corrected_row_flops((lo + lo + 8 - 1) / 2.0, 23)
        self.assertLess(r0, 8 * self.dec.corrected_row_flops((lo + lo + 8 - 1) / 2.0))


if __name__ == "__main__":
    unittest.main()

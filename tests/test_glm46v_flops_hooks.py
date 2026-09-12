"""GLM-4.6V's routed experts need their own `_SPECIAL_HOOKS` entry (CPU, no checkpoint).

The hooked FLOPs reports install `nn.Linear` / `nn.Conv*` forward hooks.  A MoE module that keeps
its expert weights as 3-D `nn.Parameter` stacks and calls `F.linear` per hit expert is invisible
to both -- the silent-undercount failure mode that cost the Qwen3.5-122B table a 7.3 GF/token
term (docs/memo/vllm_interleaved_design.md §7.10).

GLM-4.6V is that shape (`Glm4vMoeTextExperts`), and the existing `FP8Experts` entry does NOT
cover it: `FP8Experts` exists only under transformers' FineGrainedFP8 quantizer, while
GLM-4.6V-FP8 is compressed-tensors, which leaves the experts class alone.  This test pins both
halves of that claim.

    python -m pytest tests/test_glm46v_flops_hooks.py -q
"""
import types
import unittest

import torch

from appcorr.flops import hooks
from appcorr.flops.counter import FlopCounter

E, TOP_K, H, I, N_TOK = 8, 2, 32, 16, 7


def _experts():
    from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeTextExperts
    cfg = types.SimpleNamespace(num_local_experts=E, hidden_size=H, moe_intermediate_size=I,
                                hidden_act="silu", num_experts_per_tok=TOP_K,
                                _experts_implementation="eager")
    mod = Glm4vMoeTextExperts(cfg)
    with torch.no_grad():
        mod.gate_up_proj.normal_()
        mod.down_proj.normal_()
    return mod


def _batch():
    torch.manual_seed(0)
    return (torch.randn(N_TOK, H), torch.randint(0, E, (N_TOK, TOP_K)),
            torch.rand(N_TOK, TOP_K))


def _count(mod, batch):
    c = FlopCounter()
    h = hooks.install(c, [mod])
    with c.request("x"):
        mod(*batch)
    hooks.remove(h)
    return c.aggregate()["mean_total_gflops"] * 1e9, len(h)


class Glm46VExpertsHookTest(unittest.TestCase):
    def test_module_is_invisible_to_the_generic_hooks(self):
        mod = _experts()
        self.assertFalse(any(isinstance(m, (torch.nn.Linear, torch.nn.modules.conv._ConvNd))
                             for m in mod.modules()))
        self.assertIn("gate_up_proj", dict(mod.named_parameters()))

    def test_handler_counts_routed_experts_plus_router(self):
        mod, batch = _experts(), _batch()
        counted, n_handles = _count(mod, batch)
        expected = 2 * N_TOK * TOP_K * 3 * I * H + 2 * N_TOK * H * E
        self.assertEqual(n_handles, 1)
        self.assertAlmostEqual(counted, expected, delta=1.0)

    def test_without_the_entry_the_module_counts_zero(self):
        """The regression this entry prevents: no hook at all, so a confidently low number."""
        mod, batch = _experts(), _batch()
        saved = hooks._SPECIAL_HOOKS.pop("Glm4vMoeTextExperts")
        try:
            counted, n_handles = _count(mod, batch)
        finally:
            hooks._SPECIAL_HOOKS["Glm4vMoeTextExperts"] = saved
        self.assertEqual((counted, n_handles), (0.0, 0))

    def test_fp8experts_entry_does_not_apply_to_compressed_tensors(self):
        """`FP8Experts` is substituted by the FineGrainedFP8 quantizer only; the compressed-
        tensors quantizer that GLM-4.6V-FP8 uses never replaces the experts class, so the class
        name the hook table must carry is the model's own."""
        import inspect
        from transformers.quantizers import quantizer_compressed_tensors as q_ct
        src = inspect.getsource(q_ct.CompressedTensorsHfQuantizer)
        self.assertNotIn("FP8Experts", src)
        self.assertNotIn("experts_class", src)
        from transformers.integrations import finegrained_fp8 as fg
        self.assertIn("experts_class=FP8Experts", inspect.getsource(fg))

    def test_glm4moe_text_only_sibling_is_also_registered(self):
        self.assertIn("Glm4MoeExperts", hooks._SPECIAL_HOOKS)
        self.assertIs(hooks._SPECIAL_HOOKS["Glm4MoeExperts"],
                      hooks._SPECIAL_HOOKS["Glm4vMoeTextExperts"])


if __name__ == "__main__":
    unittest.main()

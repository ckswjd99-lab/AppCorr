"""Gate for the NVFP4 quantizer: byte-identical to TorchAO, or the fused version cannot be trusted.

A wrong scale swizzle raises nothing -- `torch._scaled_mm` accepts the buffer and returns numbers
that are simply wrong. So this compares `qdata` and `scale` for exact equality rather than a
tolerance, across the shapes the correction path actually uses, and must pass before any producer
logic (LayerNorm / SwiGLU) is folded into the kernel.
"""

import unittest

import torch


CORRECTION_K = (4096, 8192)          # block dim, and the SwiGLU hidden dim feeding mlp.w3
CORRECTION_M = (1027, 1280, 2068, 5120)   # ADE20K selected-token counts, aligned and not


def _setup():
    from offload.server.model.dinov3_precision import _configure_compile_environment

    _configure_compile_environment()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0),
    "NVFP4 needs compute capability 10.0+",
)
class TestNVFP4FusedQuantize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()

    def test_matches_torchao_bytewise(self):
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            NVFP4Tensor,
            per_tensor_amax_to_scale,
        )

        from appcorr.models.dinov3.layers.triton_kernels.nvfp4_fused import (
            quantize_nvfp4_swizzled,
        )

        dev = torch.device("cuda")
        for K in CORRECTION_K:
            for M in CORRECTION_M:
                with self.subTest(M=M, K=K):
                    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
                    ts = per_tensor_amax_to_scale(x.abs().amax())

                    ref = NVFP4Tensor.to_nvfp4(
                        x,
                        block_size=16,
                        per_tensor_scale=ts,
                        is_swizzled_scales=True,
                        use_triton_kernel=True,
                    )
                    qdata, scale = quantize_nvfp4_swizzled(x, ts)

                    self.assertEqual(tuple(qdata.shape), tuple(ref.qdata.shape))
                    self.assertTrue(
                        torch.equal(qdata, ref.qdata),
                        f"qdata differs at M={M} K={K}: "
                        f"{(qdata != ref.qdata).sum().item()} of {qdata.numel()} bytes",
                    )
                    # Compare the raw e4m3 bit patterns; float8 equality is not defined elementwise.
                    got_s = scale.reshape(-1).view(torch.uint8)
                    ref_s = ref.scale.reshape(-1).view(torch.uint8)
                    self.assertEqual(got_s.numel(), ref_s.numel())
                    self.assertTrue(
                        torch.equal(got_s, ref_s),
                        f"scale differs at M={M} K={K}: "
                        f"{(got_s != ref_s).sum().item()} of {got_s.numel()} bytes",
                    )

    def test_roundtrip_through_scaled_mm(self):
        """The real consumer is _scaled_mm; a buffer that matches byte-wise must also compute."""
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            NVFP4Tensor,
            per_tensor_amax_to_scale,
        )

        from appcorr.models.dinov3.layers.triton_kernels.nvfp4_fused import (
            quantize_nvfp4_swizzled,
        )

        dev = torch.device("cuda")
        M, K, N = 1280, 4096, 8192
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        xs = per_tensor_amax_to_scale(x.abs().amax())
        ws = per_tensor_amax_to_scale(w.abs().amax())

        wq = NVFP4Tensor.to_nvfp4(
            w, block_size=16, per_tensor_scale=ws, is_swizzled_scales=True
        ).t()
        qdata, scale = quantize_nvfp4_swizzled(x, xs)

        out = torch._scaled_mm(
            qdata.view(torch.float4_e2m1fn_x2),
            wq.qdata.view(torch.float4_e2m1fn_x2),
            scale.view(torch.float8_e4m3fn),
            wq.scale.t().view(torch.float8_e4m3fn),
            bias=None,
            out_dtype=torch.bfloat16,
        ) * (xs * ws).to(torch.bfloat16)

        ref = (x.float() @ w.float().t())
        rel = ((ref - out.float()).norm() / ref.norm()).item()
        self.assertLess(rel, 0.25, f"NVFP4 GEMM rel-L2 {rel:.4f} is beyond the format's ~0.14")


if __name__ == "__main__":
    unittest.main()

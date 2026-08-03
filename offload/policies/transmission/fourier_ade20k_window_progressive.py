from .ade20k_window_progressive import ADE20KWindowProgressiveLaplacianPolicy
from .fourier_laplacian_progressive import FourierLaplacianProgressivePolicy


class FourierADE20KWindowHybridPolicy(ADE20KWindowProgressiveLaplacianPolicy, FourierLaplacianProgressivePolicy):
    """
    DCT-base counterpart of ADE20KWindowProgressiveLaplacianPolicy, mirroring how
    FourierLaplacianHybridPolicy relates to COCOWindowProgressiveLaplacianPolicy: group 0 becomes
    FourierLaplacianProgressivePolicy's whole-image 2D DCT low-pass reconstruction instead of a
    Gaussian-pyramid downsample; the m2f crop-cover grouping of residuals into groups 1..N
    (ADE20KWindowProgressiveLaplacianPolicy's `_precompute_group_assignments` override +
    `num_correction_groups` stamping) is inherited unchanged.

    Pure composition, no new logic. Both parents share `ProgressiveLPyramidPolicy` as their common
    base, so MRO is: this class -> ADE20KWindowProgressiveLaplacianPolicy -> FourierLaplacianProgressivePolicy
    -> ProgressiveLPyramidPolicy -> LaplacianPyramidPolicy. ADE20KWindowProgressiveLaplacianPolicy's
    `encode()` (first in MRO) sets `self._active_config`/`self._active_image_hw` (needed by the
    crop_cover grouping) then calls `super().encode(...)` -- which resolves through THIS class's MRO
    to FourierLaplacianProgressivePolicy.encode (the DCT-based one), not
    ProgressiveLPyramidPolicy.encode (the spatial-pyramid one both parents would otherwise fall back
    to). `_precompute_group_assignments` is looked up dynamically via `self` from inside that DCT
    encode(), so it still resolves to ADE20KWindowProgressiveLaplacianPolicy's crop_cover override.
    """

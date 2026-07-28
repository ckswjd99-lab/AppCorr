from .raw import RawTransmissionPolicy
from .zlib import ZlibTransmissionPolicy
from .full_image import FullImageCompressionPolicy
from .laplacian import LaplacianPyramidPolicy
from .progressive import ProgressiveLPyramidPolicy
from .l2l1l0_progressive import L2L1L0ProgressiveLPyramidPolicy
from .coco_window_progressive import COCOWindowProgressiveLaplacianPolicy
from .ade20k_window_progressive import ADE20KWindowProgressiveLaplacianPolicy
from .nyu_appcorr_progressive import (
    NYUAppCorrLaplacianPolicy,
    NYUAppCorrProgressiveLaplacianPolicy,
    NYUAppCorrRawTransmissionPolicy,
)

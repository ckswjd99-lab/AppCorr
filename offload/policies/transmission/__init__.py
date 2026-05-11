from .raw import RawTransmissionPolicy
from .zlib import ZlibTransmissionPolicy
from .full_image import FullImageCompressionPolicy
from .laplacian import LaplacianPyramidPolicy
from .progressive import ProgressiveLPyramidPolicy
from .coco_window_progressive import COCOWindowProgressiveLaplacianPolicy
from .nyu_appcorr_progressive import (
    NYUAppCorrLaplacianPolicy,
    NYUAppCorrProgressiveLaplacianPolicy,
    NYUAppCorrRawTransmissionPolicy,
)

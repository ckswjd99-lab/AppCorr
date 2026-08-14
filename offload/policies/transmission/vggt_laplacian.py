"""Laplacian transmission for VGGT-Omega, anchored to the model's own input scale.

The generic `LaplacianPyramidPolicy` sizes every band from `mobile_resize_short_side` and a fixed
`config.image_shape`. Neither exists here: VGGT computes a per-frame canvas from the frame's own
aspect ratio (`vggt_preprocess.model_target_hw`), and Co3D frames come in 288 distinct native sizes,
so a fixed shape would be wrong for nearly all of them.

The one thing this class changes is therefore `_target_hw_for_level` -- but that one hook decides
what "level 2" means, and getting it wrong is not a small error. Anchoring bands to a canvas larger
than the model's input makes the degraded image survive the model's own downscale nearly intact:
that is exactly how COCO ended up with a floor-ceiling gap of 1e-4 and an approximation that was not
approximating. The rule:

    base = model canvas   if reaching it shrinks the frame in *both* dimensions
           native shape   otherwise
    band(level) = base / 2**level

The branch is on the scaling *direction*, deliberately, not on a size threshold. "Bigger than 512"
cannot be stated here: 512 is a token budget -- the canvas holds ~1024 patch tokens laid out along
each frame's own aspect ratio, so a 1898x1067 frame gets 688x384 and only a square frame ever gets
512x512 (4.5% of Co3D frames).

The two branches exist for different reasons, and both matter:

- **Scale-down**: reduce to the canvas first, *then* degrade. Detail above the canvas is discarded
  by the model regardless, so degrading it first would make L2 a quarter of something the model
  never sees.
- **Scale-up**: degrade natively, and only then scale up (on the server, in `preprocess_frames`).
  Degrading an already-enlarged frame degrades fabricated detail instead of real detail.

Requiring *both* dimensions to shrink is the conservative reading of that second point. On Co3Dv2 it
is indistinguishable from comparing pixel counts (identical on all 62235 frames; a single shape,
593x445 -> 592x448, is mixed and both readings agree), but the two can diverge on other aspect
ratios and only this one is safe.

It also keeps the transmission budget honest in the sense the user asked for: transmitted pixels are
never more than the model's canvas divided by 4**level, whatever the frame's native size.

Measured (Co3Dv2, 135 sequences x 8 frames, depth AbsRel / rotation):

    L0 0.0542 / 1.76 deg    L1 0.0526 / 1.77 deg    L2 0.0628 / 1.99 deg
    L3 0.0714 / 2.65 deg    L4 0.1210 / 3.93 deg

L1 is free -- the model discards that detail anyway, so anything above it is pure transmission
waste. L2 is the default floor: degraded measurably but not destroyed. L3 works unchanged by
listing it in `pyramid_levels`; L4 is past the collapse and is not a useful floor.
"""

import math
from typing import List, Tuple

from offload.common.protocol import ExperimentConfig, Patch
from .laplacian import LaplacianPyramidPolicy


class VGGTLaplacianPolicy(LaplacianPyramidPolicy):
    """Laplacian pyramid whose level-0 canvas is VGGT's per-frame model input shape."""

    def encode(self, images, config: ExperimentConfig):
        """Base-class encode, with each pyramid level stamped as its own scheduling group.

        `Patch.group_id` defaults to 0 and `LaplacianPyramidPolicy` never sets it, so a multi-level
        transmission arrives as several groups that all claim to be group 0. An approx-then-correct
        scheduler keys its phase off that field, sees group 0 twice, and issues APPROX again instead
        of CORRECT -- so SEND_RESPONSE is never reached and the client waits until its timeout with
        nothing in either log. Levels are yielded coarsest-first, so group 0 is the base and each
        residual level follows in order.
        """
        for group_idx, patches in enumerate(super().encode(images, config)):
            for patch in patches:
                patch.group_id = group_idx
            yield patches

    @staticmethod
    def _model_canvas_hw(config: ExperimentConfig, image_hw: Tuple[int, int]) -> Tuple[int, int]:
        from offload.server.model.vggt_preprocess import model_target_hw

        profile = config.get_input_profile_config()
        return model_target_hw(
            int(image_hw[0]),
            int(image_hw[1]),
            mode=profile.get("vggt_resize_mode", "balanced"),
            image_resolution=int(profile.get("vggt_resolution", 512)),
            patch_size=int(profile.get("vggt_patch_size", 16)),
        )

    @classmethod
    def _is_scale_down(cls, config: ExperimentConfig, image_hw: Tuple[int, int]) -> bool:
        """Does reaching the canvas mean shrinking the frame in *both* dimensions?

        The criterion is the scaling *direction*, not a size threshold. "Larger than 512" is not
        expressible here -- 512 is a token budget, not a side length, and the canvas follows each
        frame's own aspect ratio (a 1898x1067 frame gets 688x384, not 512x512). Comparing pixel
        counts happens to give the same answer on all 62235 Co3D frames, but it is the wrong thing
        to state, and it decides the ambiguous case wrongly in principle.

        Requiring *both* dimensions to shrink is the conservative reading. If either dimension would
        be enlarged, building the pyramid on the enlarged frame would degrade fabricated detail
        rather than real detail -- which is precisely the failure that made COCO's approximation stop
        approximating. The cost of being wrong the other way is only some transmission waste.
        """
        h, w = int(image_hw[0]), int(image_hw[1])
        ch, cw = cls._model_canvas_hw(config, (h, w))
        return ch <= h and cw <= w

    @classmethod
    def _base_hw(cls, config: ExperimentConfig, image_hw: Tuple[int, int]) -> Tuple[int, int]:
        """Level-0 geometry, i.e. the coordinates the pyramid is built in.

        Scale-down frames are reduced to the canvas *first* and degraded there, so level 2 is a
        genuine quarter of what the model consumes. Scale-up frames are degraded in their own
        coordinates and only scaled up afterwards, by `preprocess_frames` on the server.
        """
        h, w = int(image_hw[0]), int(image_hw[1])
        if cls._is_scale_down(config, (h, w)):
            return cls._model_canvas_hw(config, (h, w))
        return (h, w)

    @staticmethod
    def _target_hw_for_level(
        config: ExperimentConfig,
        lvl: int,
        image_hw: Tuple[int, int] | None = None,
    ) -> Tuple[int, int]:
        if image_hw is None:
            raise RuntimeError(
                "VGGTLaplacian needs each frame's native shape; set "
                "transmission_kwargs.preserve_input_shape = true."
            )
        cls = VGGTLaplacianPolicy
        base_h, base_w = cls._base_hw(config, image_hw)
        ph, pw = config.patch_size
        scale = 2 ** int(lvl)
        # Align up, never down: a band rounded below a patch multiple would lose a whole row of
        # patches, and `_create_patches_vectorized` asserts exact divisibility anyway.
        return (
            max(ph, int(math.ceil(base_h / scale / ph)) * ph),
            max(pw, int(math.ceil(base_w / scale / pw)) * pw),
        )

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

import numpy as np

from offload.common.protocol import ExperimentConfig, Patch
from .laplacian import LaplacianPyramidPolicy


class VGGTLaplacianPolicy(LaplacianPyramidPolicy):
    """Laplacian pyramid whose level-0 canvas is VGGT's per-frame model input shape."""

    def _create_patches_vectorized(self, patch_list, image, b_idx, lvl, config, dtype):
        """Base-class patching, plus a per-patch residual-energy hint on the residual levels.

        Without this the correction score is attention-only: it knows which tokens are *attended to*
        but not which ones are *wrong*, and the residual is the only signal for the latter. The
        transmitted band at a residual level is exactly that error, so its energy per patch is free
        to compute here and is what the mobile side is supposed to contribute.

        Only residual levels carry it. The base level is the image itself, not an error, so its
        energy would rank bright patches rather than mis-reconstructed ones.
        """
        start = len(patch_list)
        super()._create_patches_vectorized(patch_list, image, b_idx, lvl, config, dtype)
        if dtype != np.int16:
            return

        ph, pw = config.patch_size
        H, W, _ = image.shape
        gh, gw = H // ph, W // pw
        crops = (
            image.reshape(gh, ph, gw, pw, -1)
            .transpose(0, 2, 1, 3, 4)
            .reshape(gh * gw, -1)
            .astype(np.float32, copy=False)
        )
        energy = np.square(crops, dtype=np.float32).sum(axis=1, dtype=np.float32)
        for offset, patch in enumerate(patch_list[start:]):
            patch.pscore_hint = float(energy[offset])

    # --- correction grouping ---------------------------------------------------------------------
    # How the residual is split into rounds. One request is one scene of S views, so:
    #
    #   per_frame  round r carries every patch of one view. Views are completed one at a time.
    #   spatial    round r carries the same spatial band of *every* view at once.
    #
    # These are not interchangeable on this model. Correction has a threshold -- below ~40% of tokens
    # nothing recovers -- so `per_frame` concentrates each round's budget hard enough for the views
    # it touches to cross that threshold locally, while `spatial` spreads every round thinly across
    # all views. Which one wins is exactly what the option exists to measure.

    @staticmethod
    def _correction_groups(config: ExperimentConfig) -> int:
        return max(1, int(config.transmission_kwargs.get("correction_groups", 1)))

    @staticmethod
    def _correction_grouping(config: ExperimentConfig) -> str:
        mode = str(config.transmission_kwargs.get("correction_grouping", "per_frame"))
        if mode not in {"per_frame", "spatial"}:
            raise ValueError(
                f"correction_grouping must be 'per_frame' or 'spatial', got {mode!r}"
            )
        return mode

    @classmethod
    def _residual_round(cls, patch, config: ExperimentConfig, n_frames: int, grid: tuple | None) -> int:
        """Which correction round (0-based) a residual patch belongs to."""
        rounds = cls._correction_groups(config)
        if rounds <= 1:
            return 0
        # Every round must end up non-empty. A round with no patches is never transmitted, so the
        # scheduler waits for a group that never arrives: the client sends everything and both sides
        # sit until the 30-minute timeout with nothing in either log. Ceiling division does exactly
        # that whenever the units do not divide evenly -- 14 rows over 8 rounds is 2 rows each,
        # which fills 7 rounds and leaves the eighth empty (4 of 20 Co3D sequences). Spreading the
        # remainder over the first rounds keeps every round populated.
        def _bucket(index: int, units: int) -> int:
            if units < rounds:
                # Fewer units than rounds cannot fill them, whatever the assignment. Callers must
                # pick a finer unit before getting here; reaching this is a bug, not a data case.
                raise RuntimeError(
                    f"cannot split {units} units over {rounds} correction rounds without an "
                    "empty round"
                )
            base, extra = divmod(units, rounds)
            big = extra * (base + 1)                # first `extra` rounds take one unit more
            if index < big:
                return index // (base + 1)
            return extra + (index - big) // base

        if cls._correction_grouping(config) == "per_frame":
            return _bucket(int(patch.image_idx), n_frames)
        # spatial: split by patch row, so each round is a horizontal band of every view
        if not grid:
            return 0
        gh, gw = grid
        if gh >= rounds:
            return _bucket(int(patch.spatial_idx) // max(gw, 1), gh)
        # Not enough rows to give every round a band. Co3D has frames as small as a 5x6 grid, and at
        # G=8 that leaves rounds 5-7 with nothing to send -- which used to surface only as the
        # empty-round RuntimeError, i.e. a dead sweep 26 minutes in. Fall back to banding the
        # flattened grid, so the unit count is gh*gw instead of gh. The bands stop being whole rows
        # for these frames, which is a change in shape, not in kind: it is still one contiguous
        # spatial chunk of every view per round.
        return _bucket(int(patch.spatial_idx), max(gh * gw, 1))

    def encode(self, images, config: ExperimentConfig):
        """Base-class encode, with scheduling groups stamped on.

        `Patch.group_id` defaults to 0 and `LaplacianPyramidPolicy` never sets it, so a multi-level
        transmission arrives as several groups that all claim to be group 0. An approx-then-correct
        scheduler keys its phase off that field, sees group 0 twice, and issues APPROX again instead
        of CORRECT -- so SEND_RESPONSE is never reached and the client waits until its timeout with
        nothing in either log.

        Group 0 is the coarse base. The residual is then split into `correction_groups` rounds so
        correction can start before the whole residual has arrived.
        """
        image_list = self._as_image_list(images)
        n_frames = len(image_list)
        rounds = self._correction_groups(config)

        for level_idx, patches in enumerate(super().encode(images, config)):
            if level_idx == 0 or rounds <= 1:
                for patch in patches:
                    patch.group_id = level_idx
                yield patches
                continue

            # Grid comes from the frame's own shape: `patch.target_shape` is stamped by the client
            # *after* encode yields, so reading it here would silently give an empty tuple.
            ph, pw = config.patch_size
            grids = [
                (lambda hw: (hw[0] // ph, hw[1] // pw))(
                    self._target_hw_for_level(config, patches[0].res_level, img.shape[:2])
                )
                for img in image_list
            ] if patches else []

            by_round: List[list] = [[] for _ in range(rounds)]
            for patch in patches:
                grid = grids[patch.image_idx] if patch.image_idx < len(grids) else None
                by_round[self._residual_round(patch, config, n_frames, grid)].append(patch)
            for r, group in enumerate(by_round):
                if not group:
                    # Loud, because the quiet version is a deadlock: the scheduler waits for this
                    # group id forever and the failure surfaces only as a client timeout.
                    raise RuntimeError(
                        f"correction round {r} of {rounds} is empty for a frame grid {grids[:1]}; "
                        "the scheduler would wait for a group that is never sent"
                    )
                for patch in group:
                    patch.group_id = level_idx + r
                    patch.batch_group_total = len(group)
                yield group

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

"""
vla_patch_canvas.py

Transmission policy for VLA (OpenVLA) workloads: progressive patch-level canvas.

Encode (mobile side): group 0 = the full frame downsampled by `base_factor` (the low-res "approx"
base layer) + the task instruction text; groups 1..num_groups = true-resolution 14x14 pixel blocks.
`transmission_kwargs['grouping']` selects how groups 1..num_groups are formed from the per-patch
residual RMS score (AppCorr's `residual_rms` mobile pscore, the true frame vs. the upsampled base):

  "rank" (default, legacy): patches sorted by residual descending, then chopped into num_groups
      equal chunks -- group 1 is the globally most-changed patches, wherever they sit in the image.
      Fastest quality-per-byte for a single round, but group membership is scattered in *position*,
      so under interleaved multi-round correction a later group can contain a patch that is
      causally *before* (in the vision-token sequence) an already-corrected patch in an earlier
      group -- that earlier patch's output goes stale relative to it and is never revisited (see
      analysis/experiments/audit_progressive_semantics.py). Exact only when all groups arrive AND
      get corrected together as a single round; not exact across multiple interleaved rounds even
      at coverage=1.0.

  "sequential": groups are contiguous *position* bands (group 1 = the first num_patches/num_groups
      positions in raster order, group 2 = the next band, ...) -- causally monotonic, so by the
      time group g is corrected, every position it could attend to was already corrected in an
      earlier round. At coverage=1.0 this reproduces the Phase 3 audit's finding of exact
      multi-round convergence (no position is ever skipped, so nothing skipped needs revisiting).
      Within each band, patches are still ranked by residual and only the top `coverage` fraction
      of *that band* is sent/corrected -- so coverage < 1.0 reintroduces the same accepted
      approximation as "rank" mode (a skipped patch inside an arrived band stays stale forever
      once a later band is corrected); sequential grouping only guarantees exactness at
      coverage=1.0, it does not remove the fundamental multi-round staleness tradeoff below that.
      Trade-off vs "rank": the first (most latency-hidden) round is whatever content happens to
      sit in the first position band, not necessarily the most important content in the frame --
      worse quality-per-byte under coverage < 1.0 or a missed deadline, in exchange for exactness
      once everything arrives.

  "energy": same residual-descending priority order as "rank", but group *boundaries* are chosen
      so each group carries roughly equal total residual ENERGY (sum of squared per-patch RMS --
      a low-order proxy for post-compression bit cost, since a patch's encoded size roughly grows
      with its signal variance) instead of an equal patch COUNT. Since the order is
      residual-descending, group 1 ends up SMALL (a handful of high-residual patches already sums
      to a quarter of the total energy) while later groups are progressively LARGER (many
      low-residual patches needed to reach the same energy share) -- the intent being that every
      group costs roughly the same to transmit/compress, not the same to count. Shares "rank"'s
      scattered-position caveat (not causally monotonic; exact only as a single combined round).

Decode (server side): rebuilds the *cumulative canvas* -- upsampled base with all so-far-arrived
true patches pasted in. Faithful canvas semantics are REQUIRED for interleaved correction (the
approx continuation above the frontier consumes reconstructed stream rows -- see the semantics
audit in analysis/experiments/audit_progressive_semantics.py).
"""

from typing import Generator, List

import cv2
import numpy as np

from offload.common.protocol import ExperimentConfig, Patch

from ..interface import ITransmissionPolicy


class VLAPatchCanvasPolicy(ITransmissionPolicy):
    @staticmethod
    def _params(config: ExperimentConfig):
        tk = config.transmission_kwargs
        ph, pw = config.patch_size if not isinstance(config.patch_size, int) else (config.patch_size,) * 2
        return {
            "num_groups": max(int(tk.get("num_groups", 4)), 1),
            "coverage": float(tk.get("coverage", 1.0)),
            "base_factor": int(tk.get("base_factor", 4)),
            "grouping": str(tk.get("grouping", "rank")),
            "ph": ph,
            "pw": pw,
        }

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = images[None]
        assert images.shape[0] == 1, "VLAPatchCanvas currently supports batch_size 1 (one control step at a time)"
        img = np.ascontiguousarray(images[0])  # [H, W, 3] uint8
        H, W, C = img.shape
        p = self._params(config)
        text = str(config.transmission_kwargs.get("text", ""))

        # Group 0: downsampled base + instruction text.
        base = cv2.resize(img, (W // p["base_factor"], H // p["base_factor"]), interpolation=cv2.INTER_AREA)
        # Note: target_shape is intentionally NOT set (worker.py wraps batch_np into dicts when any
        # patch carries target_shape); the base dims are derived from config + base_factor on decode.
        base_patch = Patch(
            image_idx=0, spatial_idx=-1, data=np.ascontiguousarray(base).tobytes(),
            res_level=p["base_factor"], group_id=0, batch_group_total=1, text=text,
        )
        yield [base_patch]

        # Residual-RMS ranking against the upsampled base (mobile-side signal; no server feedback
        # in the static schedule).
        up = cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        resid = (img.astype(np.float32) - up)
        gh, gw = H // p["ph"], W // p["pw"]
        per_patch = np.sqrt(
            (resid ** 2).reshape(gh, p["ph"], gw, p["pw"], C).mean(axis=(1, 3, 4))
        ).reshape(-1)
        num_patches = gh * gw

        if p["grouping"] == "sequential":
            # Causally-monotonic position bands (see module docstring). Within each band, still
            # rank by residual and keep only the top `coverage` fraction of that band -- a no-op
            # at coverage=1.0, where every position in every band is sent.
            band_bounds = np.linspace(0, num_patches, p["num_groups"] + 1).round().astype(int)
            splits = []
            for gid in range(p["num_groups"]):
                lo, hi = band_bounds[gid], band_bounds[gid + 1]
                band_positions = np.arange(lo, hi)
                band_send_n = max(1, int(round(len(band_positions) * p["coverage"])))
                band_ranked = band_positions[np.argsort(-per_patch[band_positions])][:band_send_n]
                splits.append(np.sort(band_ranked))  # ascending position order within the group too
        elif p["grouping"] == "energy":
            # Residual-descending order (like "rank"), but group boundaries fall at equal shares
            # of cumulative *energy* (sum of squared RMS) rather than equal patch count -- see
            # module docstring.
            order = np.argsort(-per_patch)
            num_send = max(1, int(round(len(order) * p["coverage"])))
            chosen = order[:num_send]
            G = p["num_groups"]
            energy = per_patch[chosen].astype(np.float64) ** 2
            cum_energy = np.cumsum(energy)
            total_energy = float(cum_energy[-1]) if cum_energy.size else 0.0
            if total_energy <= 0.0 or len(chosen) <= G:
                splits = np.array_split(chosen, G)
            else:
                bounds = [0]
                for gid in range(1, G):
                    target = total_energy * gid / G
                    end = int(np.searchsorted(cum_energy, target, side="left")) + 1
                    end = max(end, bounds[-1] + 1)              # every group gets >=1 patch
                    end = min(end, len(chosen) - (G - gid))     # leave >=1 patch for each group after this one
                    bounds.append(end)
                bounds.append(len(chosen))
                splits = [chosen[bounds[i]:bounds[i + 1]] for i in range(G)]
        else:
            order = np.argsort(-per_patch)
            num_send = max(1, int(round(len(order) * p["coverage"])))
            chosen = order[:num_send]
            splits = np.array_split(chosen, p["num_groups"])

        for gid, group in enumerate(splits, start=1):
            group_patches = []
            for sp_idx in group.tolist():
                r, c = divmod(sp_idx, gw)
                block = img[r * p["ph"]:(r + 1) * p["ph"], c * p["pw"]:(c + 1) * p["pw"]]
                group_patches.append(Patch(
                    image_idx=0, spatial_idx=int(sp_idx), data=np.ascontiguousarray(block).tobytes(),
                    res_level=0, group_id=gid, batch_group_total=len(group),
                    pscore_hint=float(per_patch[sp_idx]),
                ))
            if group_patches:
                yield group_patches

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None) -> np.ndarray:
        H, W, C = config.image_shape
        p = self._params(config)
        if canvas is None:
            canvas = np.zeros((1, H, W, C), dtype=np.uint8)

        gw = W // p["pw"]
        base_shape = (H // p["base_factor"], W // p["base_factor"], C)
        for patch in patches:
            if patch.group_id == 0:
                base = np.frombuffer(patch.data, dtype=np.uint8).reshape(base_shape)
                canvas[0] = cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                r, c = divmod(patch.spatial_idx, gw)
                block = np.frombuffer(patch.data, dtype=np.uint8).reshape(p["ph"], p["pw"], C)
                canvas[0, r * p["ph"]:(r + 1) * p["ph"], c * p["pw"]:(c + 1) * p["pw"]] = block
        return canvas

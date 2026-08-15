"""Interleaved approx-then-correct for VGGT-Omega: one correction round per residual group.

The residual arrives in several groups instead of one, and each triggers a correction restricted to
the tokens it carried. Correction therefore starts before the whole residual has landed, rather than
waiting for all of it.

This differs from `GroupTrigger`, which interleaves over *layers* -- approximating the next chunk of
the network as each group arrives. Here the layer range is always the full depth and it is the
*token set* that grows, because VGGT's correction threshold is a property of how many tokens are
corrected, not how many layers.

Round structure, with `correction_groups = G`:

    group 0      LOAD -> PREPARE -> APPROX(0, depth)
    group 1..G-1 LOAD -> PREPARE -> CORRECT(0, depth)
    group G      LOAD -> PREPARE -> CORRECT(0, depth) -> HEAD -> SEND -> FREE

`PREPARE_TOKENS` runs every round on purpose: it re-embeds the image as decoded so far and, for a
correction round, corrects the patch-embed stack against the approximate pass's cache.
"""

from typing import Any, List, Optional

from offload.common.protocol import (
    ExperimentConfig,
    Instruction,
    OpType,
    Patch,
    Task,
    normalize_appcorr_kwargs,
)

from ..interface import ISchedulingPolicy


class VGGTInterleavedPolicy(ISchedulingPolicy):
    """One correction round per transmitted residual group."""

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self._request_id = None
        self._have_request = False

    # Relative per-stage cost, from the measured forward split (S=8, 688x384):
    #   patch-embed 28.7% / 24, frame blocks 35.2% / 24,
    #   inter-frame global 25.7% / 19, inter-frame register 1.2% / 5.
    #
    # The axis is all 48 stages -- 24 patch-embed blocks then 24 aggregator pairs -- because the
    # patch-embed stack is a full ViT and 28.7% of the work, not a projection. Stage costs range
    # from 1.20% (patch-embed) to 2.82% (a pair with a global inter-frame block), a 2.4x spread, so
    # equal *stage counts* are not equal work.
    _COST_PE = 28.7 / 24
    _COST_FRAME = 35.2 / 24
    _COST_INTER_GLOBAL = 25.7 / 19
    _COST_INTER_REGISTER = 1.2 / 5
    _REGISTER_BLOCKS = (2, 6, 9, 14, 20)      # VGGT-Omega's register_attention_block_indices
    _PE_STAGES = 24

    @staticmethod
    def _total_rounds(config: ExperimentConfig) -> int:
        """Number of residual groups; group ids run 1..rounds."""
        return max(1, int(config.transmission_kwargs.get("correction_groups", 1)))

    @classmethod
    def _pe_stages(cls, config: ExperimentConfig) -> int:
        return int(config.scheduler_kwargs.get("pe_stages", cls._PE_STAGES))

    @classmethod
    def _stage_costs(cls, total: int, config: ExperimentConfig) -> List[float]:
        """Cost of each of the `total` stages: patch-embed blocks first, then aggregator pairs."""
        pe = cls._pe_stages(config)
        reg = set(config.scheduler_kwargs.get("register_block_indices", cls._REGISTER_BLOCKS))
        costs = [cls._COST_PE] * min(pe, total)
        for i in range(total - len(costs)):
            costs.append(
                cls._COST_FRAME
                + (cls._COST_INTER_REGISTER if i in reg else cls._COST_INTER_GLOBAL)
            )
        return costs

    @classmethod
    def _layer_boundaries(cls, depth: int, rounds: int, config: ExperimentConfig) -> List[int]:
        """Block index after each round, split by equal cumulative cost.

        `layer_split` picks the rule: "compute" (default) balances measured work, "uniform" balances
        block count, "full" puts every boundary at the full depth.

        "full" is a control, not a deployable mode: it approximates the whole network up front and
        then corrects each group over all 48 stages, so the correction compute equals the one-shot
        case instead of the (G+1)/2G that real interleaving costs. It isolates the two ways
        interleaving can lose accuracy -- correcting each group only as deep as its round has
        reached, versus correcting the tokens in G separate passes at all -- by removing the first.
        """
        if rounds <= 1:
            return [depth]
        mode = str(config.scheduler_kwargs.get("layer_split", "compute"))
        if mode == "full":
            return [depth] * rounds
        if mode == "uniform":
            return [min(depth, round(depth * (r + 1) / rounds)) for r in range(rounds)]
        if mode != "compute":
            raise ValueError(f"layer_split must be 'compute', 'uniform' or 'full', got {mode!r}")

        costs = cls._stage_costs(depth, config)
        total = sum(costs)
        bounds, acc, target_idx = [], 0.0, 1
        for i, c in enumerate(costs):
            acc += c
            while target_idx <= rounds and acc >= total * target_idx / rounds - 1e-9:
                bounds.append(i + 1)
                target_idx += 1
        while len(bounds) < rounds:
            bounds.append(depth)
        return [min(b, depth) for b in bounds[:rounds]]

    def decide(
        self,
        buffer: List[Patch],
        config: ExperimentConfig,
        task_id_gen: Any,
        **kwargs,
    ) -> Optional[Task]:
        if not buffer:
            return None

        head = buffer[0]
        target = head.batch_group_total
        if len(buffer) < target:
            return None

        # Deliberately not `del buffer[:target]` -- SchedulerModule consumes `len(task.payload)`
        # after `decide` returns, and deleting here as well drops twice as many patches and
        # deadlocks the next group with nothing written to any log.
        payload = buffer[:target]
        group = head.group_id
        rounds = self._total_rounds(config)
        # Total *stages*, not aggregator blocks: 24 patch-embed + 24 aggregator pairs.
        depth = int(
            config.scheduler_kwargs.get(
                "total_layers", config.transmission_kwargs.get("total_layers", 48)
            )
        )
        options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        correction_on = options.get("enabled", False) or options.get("generated_from_client", False)

        t_id = next(task_id_gen)
        instructions = [Instruction(OpType.LOAD_INPUT), Instruction(OpType.PREPARE_TOKENS)]

        bounds = self._layer_boundaries(depth, rounds, config)

        if group == 0:
            # Approximate only as far as the first boundary; the rest of the depth is reached as
            # later rounds arrive, which is what lets computation overlap transmission.
            self._request_id, self._have_request = t_id, True
            instructions.append(Instruction(OpType.APPROX_FORWARD, {"layers": (0, bounds[0])}))
        else:
            prev = bounds[group - 1]
            here = bounds[min(group, rounds - 1)]
            # Correct everything computed so far against the tokens this round delivered, then push
            # the approximate frontier to this round's boundary.
            if correction_on and prev > 0:
                instructions.append(
                    Instruction(OpType.CORRECT_FORWARD, {"layers": (0, prev), "group_id": group})
                )
            if here > prev:
                instructions.append(
                    Instruction(OpType.APPROX_FORWARD, {"layers": (prev, here)})
                )
            if group >= rounds:
                # The correction above already covered [0, prev) and, on the last round, prev is the
                # full depth -- emitting another full-depth correction here would simply repeat it.
                if correction_on and prev < depth:
                    instructions.append(
                        Instruction(OpType.CORRECT_FORWARD,
                                    {"layers": (0, depth), "group_id": group})
                    )
                instructions += [
                    Instruction(OpType.HEAD_INFERENCE),
                    Instruction(OpType.SEND_RESPONSE),
                    Instruction(OpType.FREE_SESSION),
                ]

        return Task(
            task_id=t_id,
            request_id=self._request_id if self._have_request else t_id,
            payload=payload,
            instructions=instructions,
        )

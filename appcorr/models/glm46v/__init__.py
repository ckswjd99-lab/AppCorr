"""GLM-4.6V (106B-A12B, FP8) AppCorr fork.

Vision half only: the tower split (`vision/backbone.py`), the merge head and the progressive
axis (`axis.py`). The engine half (the `correct` op, the runner patch, the FLOP closed form)
is shared with the Qwen3.5 pair and lives where it always did.

See `docs/memo/glm46v_port_plan.md` for the architecture survey this implements.
"""

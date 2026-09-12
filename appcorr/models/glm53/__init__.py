"""GLM-5.3-Flash AppCorr fork (`Glm5NextForConditionalGeneration`).

Vision half plus the progressive axis: the tower split (`vision/backbone.py`, on top of the
GLM-4.6V tower), the shared merge head and `axis.py`. The engine half of GLM-5.3 -- the mHC layer
walk, the KDA side buffer and the sparse-indexer cache rewrite -- lives in
`appcorr/vllm_stream/{correct,runner_patch,glm53_indexer}.py` and is owned elsewhere.

See `docs/memo/glm53_vllm_survey.md` (the code map) and `docs/memo/glm53_correct_design.md`.
"""

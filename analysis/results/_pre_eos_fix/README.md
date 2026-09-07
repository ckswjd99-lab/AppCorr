# Gemma 3 results produced before the EOS fix

Every arm here decoded through `_generate_from_axis`, which stopped only on
`config.text_config.eos_token_id` (1) and missed `<end_of_turn>` (106) --  the id that actually
ends an assistant turn and one of the two `generation_config` lists. The loop therefore ran past
the answer and appended whatever came next, on 34-70% of samples depending on the arm.

`ceiling` is NOT here: it alone calls `generate`, so it was unaffected. That is precisely what makes
these unusable -- the bias runs one way, against every arm the ceiling is compared to.

Kept rather than deleted because the per-sample predictions are still the record of what the model
produced, and because the size of the effect was estimated from them: re-scoring the stored strings
at `<end_of_turn>` moved InfoVQA's floor +5.38pp and its corrected arm +1.93pp, TextVQA's arms
+0.5-0.8pp, and POPE/RealWorldQA not at all (their answers are one token).

Do not quote these. Regenerated results replace them under the normal paths.

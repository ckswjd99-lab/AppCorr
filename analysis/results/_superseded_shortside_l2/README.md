# Superseded: measured with the short-side degradation

These RealWorldQA arms were run before `l2_from_native` degraded per axis. On 92% of RealWorldQA
images the long axis was under-degraded by a median 1.54x, so the floor sits closer to the ceiling
than the pyramid rule intends and nothing between them is interpretable.

For the record, what was measured: ceiling 0.4288, floor 0.4275 (gap +0.13pp, CI [-2.22, +2.48]),
corrected_t 0.4497. The near-zero gap was read at the time as "Gemma 3 is insensitive to degradation
on coarse tasks" -- that reading is not supported, because the degradation itself was too mild.

The completed ChartQA sweep is NOT superseded: 2% of its images are affected, median ratio 1.00x.
POPE is unaffected (0%). TextVQA was affected (93%, median 1.14x) but was killed mid-run.

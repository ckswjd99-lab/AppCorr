# WildVision pairwise judge: design (not yet run)

Status: DESIGN. Dumps exist (`analysis/results/qwen25vl_bench/wildvision_{ceiling,floor,streaming}.jsonl`,
plus whatever B200's models produced); no judge has run. Decisions marked OPEN need the user.

## What WildVision measures and why it needs a judge

500 real user instructions, open-ended, no reference answers. The official protocol scores by
pairwise LLM judgment against a fixed opponent's answers (claude-3-sonnet in the original bench),
reporting win-rate-derived Elo-style scores. For OUR question -- "how much does degradation cost,
and how much does each arm recover" -- the fixed-opponent framing is unnecessary; what we need is
**paired comparison between arms on the same instruction**.

## Proposed protocol (option P, for "paired")

For each instruction i and arm pair (A, B) of interest:
  judge sees: instruction, image (FULL-RESOLUTION -- the judge must know what was actually asked
  about, so it judges answer quality w.r.t. ground truth, not w.r.t. the degraded view), answer_A,
  answer_B, in randomized position (A/B swap per sample, judged twice, position-debiased).
  verdict in {A, B, tie}. Report: win/tie/loss counts + net preference with a binomial interval.

Pairs that answer our questions (in priority order):
  1. floor vs ceiling      -- the degradation cost, judge-measured
  2. streaming vs ceiling  -- does the staleness-perturbation lift extend beyond scored MCQ/
                              grounding to open-ended quality? (The RefCOCO/MMVP result predicts
                              streaming >= ceiling; a judge tie or win here would be a third task
                              family for the mechanism.)
  3. streaming vs floor    -- the recovery, judge-measured

Coverage: restrict ALL pairs to the 446-index intersection (floor/streaming skip the identical 54
oversized images; ceiling covers 500 but comparisons need common population -- kept-set principle).

## Judge model (OPEN -- needs user decision)

Constraints: GH200 has 48G free disk, no API keys provisioned, B200 down for maintenance.
  (a) Qwen2.5-VL-32B judges itself -- ZERO download, but self-judgment bias is documented in the
      literature (models prefer their own phrasing). Mitigation: all three arms are the same
      model, so the bias is SHARED across arms and largely cancels in arm-vs-arm pairs. This makes
      option (a) far more defensible here than in cross-model judging. RECOMMENDED default.
  (b) A different local judge (e.g. Gemma3-4B, ~9GB): weaker judge, breaks the shared-bias
      argument's neatness, adds a download. Only worth it as a robustness check on (a).
  (c) API judge (GPT/Claude): matches the official bench closest; needs keys + spend approval.
  (d) Defer until B200 returns and the user weighs in. (Current state.)

## Mechanics (when green-lit)

- max_new_tokens for verdicts: small (judge outputs a letter + one sentence); temperature 0.
- Each pair judged twice (positions swapped); disagreement between the two orders = tie.
- Judge prompt: instruction + "Answer A" + "Answer B" + "Which answer better addresses the
  instruction for this image? Reply 'A', 'B', or 'tie'." -- keep it minimal; no rubric bloat.
- Driver: extend `qwen25vl_bench_eval.py`-style loop or a standalone ~100-line script reading the
  three dumps; ~446 x 3 pairs x 2 orders = 2676 judge calls; at ~1s each on the 32B, ~45min.
- Report per pair: W/T/L, net, binomial 95% CI; NO Elo (we are not ranking against a league).

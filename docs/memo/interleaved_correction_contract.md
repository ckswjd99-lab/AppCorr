# The interleaved correction contract

Every new model fork in this repo has re-derived interleaved correction from scratch and made the
same small set of mistakes. This memo is the contract. **Read it before writing an interleaved path,
and read the existing implementation it points at rather than reasoning from first principles.**

Reference implementations, in order of usefulness:

- `offload/server/model/vggt_omega.py` — `prepare_tokens`, the `refining` branch. Carries the
  "this round's tokens only" comment and the evidence for it.
- `offload/policies/scheduling/vggt_interleaved.py` — the round structure as a schedule.
- `appcorr/models/dinov3/layers/block.py` — `correct_partial_token`, and commit `ac0238f` for the
  increment-persistence fix.

---

## The four rules

### 1. A round corrects ITS OWN group, never the accumulated set

```python
tok = per_group[r]                      # right
tok = torch.cat(arrived).sort().values  # WRONG
```

Groups `0..r-1` are already corrected and *stay* corrected. Their state survives because:

- their recomputed K/V was scattered into the cache, so round `r`'s attention already reads correct
  keys and values for them;
- their corrected increment was persisted (rule 3), so reconstruction gives their corrected value;
- the approximate pass over the next layer range carried them forward — running "approx" layers on a
  token whose input is already correct produces the correct output for that token. *Approximate*
  describes the input, not the arithmetic.

**Why accumulating is not merely wasteful.** On the last round the accumulated set equals the full
selection, corrected over the full depth, so the schedule collapses into one-shot correction. In
VGGT this was caught only because interleaved and one-shot then agreed to 16 decimals. In the SAM 3
fork it was caught by a cost signature instead — interleaved ran 2.2x the one-shot time with the
same token count, `8+16+24+32 = 80` layer-corrections against one-shot's 32.

**It also inverts the cost claim.** Correcting only group `r` over depth `bounds[r]` costs
`(1/g)·Σ bounds[r]` token-layer units — at `g=4`, `(8+16+24+32)/4 = 20` against one-shot's 32.
Interleaved is *cheaper* than one-shot, which is the point (ProgVFM §3.3). Accumulating makes it
2.5x more expensive and quietly destroys the claim.

### 2. Distinguish what the STREAM carries from what a round RECOMPUTES

These are different sets and conflating them is the subtler half of rule 1.

| | contents | why |
|---|---|---|
| input stream | full-res at **every arrived token** (cumulative) | a corrected token's layer-0 value *is* its full-resolution patch embedding |
| `token_idx` | **this round's group only** | rule 1 |

The stream matters beyond the corrected indices because `correct()` reconstructs untouched positions
as `flat + increment`, reading `flat` at *all* positions.

**Never hand the round a stream built over the whole selection.** That places tokens that have not
arrived into the residual stream at full resolution — data from the future. It inflates every
interleaved arm against one-shot, which never shows the gap because its corrected set *is* the whole
selection. Symptom: interleaved beats the exact-forward ceiling. In the SAM 3 fork this reached
103.2% of the floor–ceiling gap and was briefly explained away as metric noise.

The opening approximate pass runs on the **pure approximate input**, not the mixed stream — at that
moment nothing has arrived.

### 3. Persist the corrected increment, every round

```python
new_increment[:, token_idx] = (x_attn_active - x_active) + mlp_out_new
cache_feature[f"{tag}_blocks_out_sum"] = new_increment.reshape(...)
```

Without it, a later round reconstructs every position it is not correcting from the *approximate*
increment and discards what earlier rounds fixed — interleaved keeps only its last round.

**One-shot correction cannot expose this**, which is why it survives into every new fork: there is no
later round to read a stale value, so the numbers look plausible and only interleaved is wrong.
Fixed in DINOv3 as `ac0238f` (VGGT 74.0% → 88.4% of the gap, ADE20K 80.8% → 89.7%); the CLIP fork
carried the pre-fix shape and SAM 3 inherited it from CLIP.

### 4. Coverage must equal the one-shot set, exactly

The union of the per-round groups, after restricting each to the tokens the patch score actually
selected, must equal the one-shot selection. Otherwise `g` changes the amount of computation and no
comparison across `g` means anything.

```python
keep_mask = torch.zeros(h * w, dtype=torch.bool, device=idx.device)
keep_mask[idx] = True
per_group = [g[keep_mask[g]] for g in per_group]
```

Indexing rounds from 1 silently drops group 0: the SAM 3 fork corrected 41.3% while reporting a 55%
keep ratio — *fewer* tokens than one-shot yet scoring higher, which is how it was found.

---

## Gates to run before reporting any interleaved number

1. **Coverage.** Union of per-round groups == one-shot selection, at every `g` you will report.
2. **`g=1` identity.** Driving the interleaved path with a single group must reproduce one-shot.
   With rules 1–2 correct this is the *same computation*, so expect equality, not "close enough".
   A residual difference means something is still wrong — do not attribute it to float
   reassociation without showing that reassociation is the only thing that differs.
3. **Cost signature.** Count layer-corrections and compare against `(1/g)·Σ bounds[r]`. Interleaved
   costing *more* than one-shot at equal token count means rule 1 is broken.
4. **Feature-space fidelity, not just the task metric.** Task metrics are not monotone in feature
   fidelity — a mask AP can exceed the ceiling by luck, and did. Relative L2 against the exact
   forward is zero only for the exact computation, so it cannot be beaten. Any arm that appears to
   beat the ceiling in feature space is a leak (rule 2), not a discovery.

## Why this keeps happening

Each rule is invisible in the arm that gets built first. One-shot correction exercises none of them:
its corrected set is the whole selection, so rule 1 is vacuous, rule 2 has no gap to leak through,
rule 3 has no later round, and rule 4 is trivially satisfied. Interleaving is where all four start to
matter at once — and by then the one-shot numbers look right, so the fork feels validated.

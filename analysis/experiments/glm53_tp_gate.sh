#!/usr/bin/env bash
# GLM-5.3-Flash TP=2 gate for the AppCorr stream server.  RUN ON B200-8, GPUs 2-3.
# NOT RUN BY ITS AUTHOR (CPU-only night).  Plan: docs/memo/glm53_tp_plan.md.
#
# WHY THERE IS NO TP=1 LEG.  The first version of this script opened with a TP=1 baseline so a
# non-exact TP=2 arm could be blamed on TP rather than on the model.  That leg is IMPOSSIBLE for
# this checkpoint: GLM-5.3-Flash is 328 GB of FP8 weights and does not fit on one B200, so
# `--tensor-parallel-size 1` OOMs before the first token and proves nothing.  The TP=2 legs are
# therefore self-sufficient and carry their own controls:
#
#   * arm A (vLLM's own image request) is the anchor INSIDE the same TP=2 engine, so an arm-B or
#     arm-C divergence is ours, not the all-reduce's;
#   * the TP=1-is-unchanged claim is made on the CPU instead, bitwise, by
#     `vllm_interleaved_axis_gate.py --tiny --family {glm53,glm46v}` and
#     `tests/test_glm53_tp_dispatch.py` -- see step 0.
#
# LEGS
#   0  cpu        preflight + the CPU evidence that TP=1 is untouched.  No GPU.
#   1  tp2-embeds arms A,B,C at TP=2, NO correction.  --gpu-mem 0.96 + --enforce-eager (what
#                 B200-8 measured for stock/streaming-only work).
#                   * A  stock image request -- the TP=2 engine itself.
#                   * B  the whole prompt pushed as prompt_embeds in ONE message.  Proves the
#                        composer and the SchedulerOutput/pickle path across the process
#                        boundary (plan §3).  It does NOT prove the runner patch: arm B never
#                        appends and GLM-5.3 is not an M-RoPE model, so neither `runner_patch`
#                        hook is on its path.
#                   * C  the same embeds in 4 chunks, one appended per engine step.  THIS is the
#                        (a) test: without `runner_patch._update_states` installed in each WORKER
#                        process, `st.prompt_embeds` never grows there and the request cannot
#                        finish.
#   2  tp2-kda    G-KDA (agent K) at TP=2.  Side buffers -> --gpu-mem 0.90 (K's sizing: 1.06 MiB
#                 per token per rank, allocated OUTSIDE vLLM's budget, so the budget must leave
#                 room for it).  Each rank owns a disjoint KDA head slice, so the ranks are
#                 EXPECTED to differ; the script prints max rel-L2 per rank.
#   3  tp2-mla    G-MLA (agent M) at TP=2, same --gpu-mem 0.90.  Everything it compares is
#                 REPLICATED (MLA latent = 1 kv head; the indexer is ReplicatedLinear /
#                 disable_tp), so it asserts the ranks agree bytewise -- the check that a
#                 correction cannot half-apply across ranks.
#
# PASS RULE.  Leg 1: arms B and C exact 8/8 against arm A on the generated token ids
# (`max_dlogprob_common` reported, not thresholded).  Leg 2: the layer-to-layer SPREAD, per rank
# (a uniform 1e-3 is bf16; one outlier layer is a wiring bug).  Leg 3: its own asserts, plus
# `rank_agreement.*.agree == true` on every snapshot.
#
# Legs are independent; run one at a time (one task per GPU pair).  `bash glm53_tp_gate.sh 0 1`
# runs legs 0 and 1; no argument runs 0 only, because everything below it wants the GPUs.

set -u -o pipefail

ROOT=/NHNHOME/share/cjpark/AppCorr-glm53
ENVDIR=/NHNHOME/share/cjpark/backup/env/appcorr-vllm-main
PY=$ENVDIR/bin/python3.11          # `bin/pip` in this env is broken; use the interpreter only
MODEL=zai-org/GLM-5.3-Flash
DEV=2,3                            # B200-8 GPUs 2-3 (coordinator's allocation)
OUT=$ROOT/analysis/results/glm53
LOG=$ROOT/logs/glm53_tp_gate
mkdir -p "$OUT" "$LOG"

export HF_HOME=/NHNHOME/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH=$ROOT
export TOKENIZERS_PARALLELISM=false
# The engine core stays in THIS process (the streaming scheduler is called by reference);
# orthogonal to tensor_parallel_size, which moves the WORKERS out -- plan §0.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# The interleaved path needs the triton GDN decode kernel: the default `cuda` path bypasses the
# capture/correct hook entirely and would silently never fire (correct.py::check_gdn_path).
export VLLM_GDN_DECODE_KERNEL=triton

# --- noexec-safe scratch ------------------------------------------------------------------------
# /tmp is mounted noexec on this box, which is how a sweep once "ran" for hours without starting:
# every JIT that writes a .so and dlopen()s it -- inductor, triton, flashinfer's workspace --
# fails or silently degrades there.  Everything that compiles goes under $HOME instead.
export APPCORR_SCRATCH=/home/nxclab/cjpark/glm53_tp_scratch
mkdir -p "$APPCORR_SCRATCH"/{tmp,flashinfer,inductor,triton}
export TMPDIR=$APPCORR_SCRATCH/tmp
export FLASHINFER_WORKSPACE_BASE=$APPCORR_SCRATCH/flashinfer
export TORCHINDUCTOR_CACHE_DIR=$APPCORR_SCRATCH/inductor
export TRITON_CACHE_DIR=$APPCORR_SCRATCH/triton

LEGS="${*:-0}"
have_leg () { case " $LEGS " in *" $1 "*) return 0;; *) return 1;; esac; }

# =================================================================================================
leg0 () {
  echo "== leg 0: preflight + the CPU evidence that TP=1 is unchanged =="
  CUDA_VISIBLE_DEVICES= "$PY" - <<'EOF' || return 1
import sys, vllm
from appcorr.vllm_stream import SUPPORTED_VLLM
print(f"vllm {vllm.__version__}; SUPPORTED_VLLM={SUPPORTED_VLLM}")
if vllm.__version__ not in SUPPORTED_VLLM:
    sys.exit(f"STOP: install() refuses vllm {vllm.__version__}.")
from appcorr.vllm_stream.tp_worker import (
    WORKER_EXTENSION_CLS, AppcorrWorkerExtension, INSTALLED)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
print("worker_extension_cls =", WORKER_EXTENSION_CLS)
print("tp_worker import installed the hooks:", INSTALLED,
      "| stream", getattr(GPUModelRunner, "_appcorr_stream_patched", False),
      "| correct", getattr(GPUModelRunner, "_appcorr_correct_patched", False))
# the extension must not collide with the worker's own attribute names (init_worker asserts)
from vllm.v1.worker.gpu_worker import Worker
clash = [a for a in dir(AppcorrWorkerExtension)
         if not a.startswith("__") and hasattr(Worker, a)]
assert not clash, f"worker_extension_cls name clash: {clash}"
print("no attribute clash with vllm's Worker")
from analysis.experiments.vllm_stream_gate import composer_for
assert composer_for("zai-org/GLM-5.3-Flash").__name__ == "Glm53Composer"
print("composer_for OK")
EOF
  echo "-- the TP=1-is-unchanged evidence (CPU, bitwise):"
  for fam in glm53 glm46v; do
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= "$PY" \
      "$ROOT/analysis/experiments/vllm_interleaved_axis_gate.py" --tiny --family $fam \
      --groups 4 --keeps 1.0 0.5 --out "$OUT/tp1_axis_$fam.json" 2>&1 | tail -3
  done
  echo "-- and the dispatch layer itself (fake two-rank executor, no CUDA):"
  echo "   run in the appcorr env, which has pytest:"
  echo "   OMP_NUM_THREADS=8 <appcorr python> -m pytest tests/test_glm53_tp_dispatch.py -q"
}

# =================================================================================================
leg1 () {
  local out=$OUT/gate_glm53_tp2_embeds.json log=$LOG/tp2_embeds.log
  echo "== leg 1: TP=2 pushed embeds / streaming, NO correction (GPUs $DEV) =="
  rm -f "$out"
  CUDA_VISIBLE_DEVICES=$DEV "$PY" "$ROOT/analysis/experiments/vllm_stream_gate.py" \
      --model "$MODEL" --arms A,B,C --chunks 4 --max-tokens 48 --n-images 8 \
      --gpu-mem 0.96 --enforce-eager \
      --tensor-parallel-size 2 \
      --out "$out" 2>&1 | tee "$log"
  echo "-- exit=${PIPESTATUS[0]}"
  grep -iE "Overriding VLLM_WORKER_MULTIPROC_METHOD|multiprocessing start method" "$log" | tail -2
  "$PY" - "$out" <<'EOF'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as e:
    print("  no result json:", e); raise SystemExit(0)
tally = {}
for name, row in r.items():
    if name == "_meta":
        continue
    for arm in ("B", "C", "C2"):
        cmp = row.get(f"{arm}_vs_A") or (row.get(arm) or {}).get("vs_A")
        if cmp:
            t = tally.setdefault(arm, [0, 0, 0.0])
            t[1] += 1
            t[0] += bool(cmp.get("exact"))
            t[2] = max(t[2], float(cmp.get("max_dlogprob_common") or 0.0))
for arm, (ok, n, dlp) in sorted(tally.items()):
    print(f"  arm {arm}: exact {ok}/{n}   max|dlogprob| {dlp:.3g}")
EOF
}

# =================================================================================================
leg2 () {
  echo "== leg 2: G-KDA at TP=2 (side buffers -> --gpu-mem 0.90) =="
  CUDA_VISIBLE_DEVICES=$DEV "$PY" "$ROOT/analysis/experiments/glm53_kda_gate.py" \
      --model "$MODEL" --gpu-mem 0.90 --max-model-len 8192 \
      --prompt-tokens 2048 --split 900 --tensor-parallel-size 2 \
      --out "$OUT/glm53_kda_gate_tp2.json" 2>&1 | tee "$LOG/tp2_kda.log" | tail -20
  echo "-- exit=${PIPESTATUS[0]}"
}

# =================================================================================================
leg3 () {
  echo "== leg 3: G-MLA at TP=2 (side buffers -> --gpu-mem 0.90) =="
  CUDA_VISIBLE_DEVICES=$DEV "$PY" "$ROOT/analysis/experiments/glm53_mla_gate.py" \
      --model "$MODEL" --gpu-mem 0.90 --max-model-len 8192 --max-num-seqs 256 \
      --g 4 --chunks 4 --n-images 8 --tensor-parallel-size 2 \
      --out "$OUT/mla_gate_tp2.json" 2>&1 | tee "$LOG/tp2_mla.log" | tail -25
  echo "-- exit=${PIPESTATUS[0]}"
}

for leg in $LEGS; do
  case $leg in
    0) leg0 ;;
    1) leg1 ;;
    2) leg2 ;;
    3) leg3 ;;
    *) echo "unknown leg $leg (0=cpu 1=tp2-embeds 2=tp2-kda 3=tp2-mla)"; exit 2 ;;
  esac
done

echo
echo "== done. results in $OUT, logs in $LOG =="
echo "Report: leg 1 arm B exact n/8 and arm C exact n/8 (+ the multiprocessing start method from"
echo "the log); leg 2 max rel-L2 per rank per quantity; leg 3 its asserts + rank_agreement.*.agree."

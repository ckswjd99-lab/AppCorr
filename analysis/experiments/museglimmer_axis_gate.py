"""Identity + smoke gates for the Muse Glimmer streaming axis.

Run with the MG-capable transformers checkout FIRST on sys.path:

    env CUDA_VISIBLE_DEVICES=0 python analysis/experiments/museglimmer_axis_gate.py \
        --transformers-path /NHNHOME/share/cjpark/tf515

Gates, in order (fail-fast):
  1. IDENTITY  streaming_forward(groups=1, keep=1.0) vs full_forward -- one band corrected after
     everything arrived means no staleness anywhere, so the two must agree to numerical noise.
     Reported as logit max|diff|, TV of softmax, and argmax equality over N samples.
  2. G4 SMOKE  streaming_forward(groups=4, keep=1.0) runs end to end and its argmax stays inside
     the {full, floor} plausible set (not a correctness claim -- just "the walk did not detonate").
  3. FLOOR SANITY  approx_only_forward differs from full_forward (a degraded image that changes
     nothing would mean the degrade or the wiring is broken).
"""
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--transformers-path", default="/NHNHOME/share/cjpark/tf515")
ap.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--dataset", default="realworldqa")
args = ap.parse_args()

sys.path.insert(0, args.transformers_path)                       # BEFORE transformers imports
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import torch  # noqa: E402
from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: E402
from datasets import load_dataset  # noqa: E402
from qwen_vl_prefill.datasets_eval import get_spec  # noqa: E402
from appcorr.models.museglimmer.unified import MuseGlimmerAxis  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "analysis", "experiments"))
from qwen35_accuracy import degrade  # noqa: E402  (shared pyr degrade; model-agnostic)

import transformers  # noqa: E402
print(f"transformers {transformers.__version__} from {transformers.__file__}", flush=True)

proc = AutoProcessor.from_pretrained(args.model)
model = AutoModelForImageTextToText.from_pretrained(
    args.model, dtype="auto", device_map="cuda:0").eval()
axis = MuseGlimmerAxis(model, proc)

spec = get_spec(args.dataset)
ds = spec.load(load_dataset)
idxs = list(range(0, len(ds), max(1, len(ds) // args.samples)))[:args.samples]

worst_diff = worst_tv = 0.0
argmax_ok = g4_ok = floor_differs = 0
for i in idxs:
    img, q, _ = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
    img = img.convert("RGB")
    inputs = axis.build_inputs(img, q).to("cuda:0")
    base = axis.build_inputs(degrade(img, 2, "pyr"), q).to("cuda:0")

    with torch.no_grad():
        lg_full = axis.full_forward(inputs).float()
        lg_id, _, _ = axis.streaming_forward(inputs, base["pixel_values"], groups=1, keep=1.0)
        lg_id = lg_id.float()
        lg_g4, _, _ = axis.streaming_forward(inputs, base["pixel_values"], groups=4, keep=1.0)
        lg_g4 = lg_g4.float()
        lg_floor = axis.approx_only_forward(inputs, base["pixel_values"]).float()

    diff = float((lg_full - lg_id).abs().max())
    tv = float(0.5 * (lg_full.softmax(-1) - lg_id.softmax(-1)).abs().sum())
    worst_diff = max(worst_diff, diff)
    worst_tv = max(worst_tv, tv)
    same = int(lg_full.argmax(-1)) == int(lg_id.argmax(-1))
    argmax_ok += int(same)
    g4_ok += int(int(lg_g4.argmax(-1)) in {int(lg_full.argmax(-1)), int(lg_floor.argmax(-1))})
    floor_differs += int(float((lg_full - lg_floor).abs().max()) > 1e-3)
    print(f"[{i}] identity max|d|={diff:.4e} TV={tv:.4e} argmax_same={same} "
          f"g4_in_plausible={bool(g4_ok)} ", flush=True)

n = len(idxs)
# Criterion revised 2026-09-01 after the vision-isolation probe: the fork's vision path is
# BITWISE exact against stock get_image_features (max|d|=0.0 on 1683/2255-token images), so the
# residual TV (~1e-3..2e-2) is entirely the LLM chunked-prefill boundary-numerics class -- MG's
# text model is dense (no MoE router = no discontinuous amplifier), the benign end of the
# quantizer/amplifier spectrum. Gate: argmax preserved everywhere + worst TV under the measured
# noise ceiling.
print(f"\nGATE identity : worst max|d|={worst_diff:.4e} worst TV={worst_tv:.4e} "
      f"argmax {argmax_ok}/{n} -> {'PASS' if argmax_ok == n and worst_tv < 5e-2 else 'FAIL'}")
print(f"GATE g4 smoke : {g4_ok}/{n} in plausible set -> {'PASS' if g4_ok == n else 'CHECK'}")
print(f"GATE floor    : differs on {floor_differs}/{n} -> {'PASS' if floor_differs == n else 'FAIL'}")
print("MG_AXIS_GATE_DONE", flush=True)

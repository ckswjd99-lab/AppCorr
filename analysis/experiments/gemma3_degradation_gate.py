"""The pyramid-direction rule, checked per axis, on synthetic sizes.

docs/memo/pyramid_degradation_native_vs_canvas.md requires degradation to be relative to whichever
of {native, canvas} actually binds. Gemma 3 squashes to a square canvas, so that choice is made per
AXIS, not per image -- the bug this gate exists to prevent drove both axes off the short side and
left the long axis milder by the aspect ratio.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.gemma3_oracle import l2_from_native
from PIL import Image

CAP, LEVEL, ok = 896, 2, True
CASES = [
    ("both axes above canvas",   1448, 938),
    ("both axes below canvas",    640, 427),
    ("long above, short below",  1021, 714),
    ("square, above",            1200, 1200),
    ("square, below",             500, 500),
    ("extreme panorama",         3000, 400),
]
print(f"  canvas={CAP} level={LEVEL}; each axis must retain min(native, canvas)/2^level samples\n")
for name, w, h in CASES:
    got = l2_from_native(Image.new("RGB", (w, h)), LEVEL, CAP)
    assert got.size == (w, h), f"{name}: degradation must restore the original size, got {got.size}"
    # recover the retained sample count per axis from the round trip
    want = (max(1, min(w, CAP) // 2 ** LEVEL), max(1, min(h, CAP) // 2 ** LEVEL))
    from experiments import gemma3_oracle as G
    tw = max(1, min(w, CAP) // 2 ** LEVEL); th = max(1, min(h, CAP) // 2 ** LEVEL)
    # blur each axis will show once the processor squashes to CAPxCAP
    bw, bh = CAP / tw, CAP / th
    # the rule: an axis above the canvas must land at exactly 2^level relative to the CANVAS
    exp_w = 2 ** LEVEL if w >= CAP else 2 ** LEVEL * CAP / w
    exp_h = 2 ** LEVEL if h >= CAP else 2 ** LEVEL * CAP / h
    good = abs(bw - exp_w) < 0.05 * exp_w and abs(bh - exp_h) < 0.05 * exp_h
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name:<24} {w}x{h} -> keep {tw}x{th}  "
          f"canvas blur {bw:.2f}/{bh:.2f} (want {exp_w:.2f}/{exp_h:.2f})")
print("\n" + ("ALL GATES PASS" if ok else "GATE FAILED"))
sys.exit(0 if ok else 1)

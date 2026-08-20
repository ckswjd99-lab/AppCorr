"""Fetch SA-Co/Gold images from Roboflow and restore their original relative paths.

The annotations on HuggingFace ship without images; Roboflow hosts them. Two things this has to get
right, both verified on the metaclip subset before the other six were downloaded:

**Only the 'a' project per subset is fetched.** a/b/c are three annotators over the *same* images --
identical image counts, different masks -- so downloading all three would triple the transfer for
nothing.

**Roboflow renames files.** `2/10002/metaclip_2_10002_ff3b....jpeg` comes back as
`metaclip_2_10002_ff3b..._jpeg.rf.<hash>.jpg`. The original nested path is recoverable because the
directory components are encoded in the stem. Verified 606/606 restored with zero missing and zero
extra against the annotation file list -- if that ever stops holding, the run would silently skip
images rather than fail, so `--verify` re-checks it per subset.

**Roboflow did not resize.** All 606 images came back at their annotated width/height. This matters
because a resize would misalign every mask while still producing plausible-looking numbers.

    RF_KEY=... python scripts/saco_gold_fetch_images.py            # all subsets
    RF_KEY=... python scripts/saco_gold_fetch_images.py --verify   # check what is on disk
"""

import argparse, json, os, re, shutil, subprocess, sys, tempfile, urllib.parse, urllib.request

ANN = "/NHNHOME/share/cjpark/data/saco_gold/annotations"
OUT = "/NHNHOME/share/cjpark/data/saco_gold/images"

# subset name in the annotation files -> Roboflow project slug (the 'a' annotator)
PROJECTS = {
    "metaclip":            "gold-metaclip-merged-a-release-test",
    "sa1b":                "gold-sa-1b-merged-a-release-test",
    "attributes":          "gold-attributes-merged-a-release-test",
    "crowded":             "gold-crowded-merged-a-release-test",
    "wiki_common":         "gold-wiki-common-merged-a-release-test",
    "fg_food":             "gold-fg-food-merged-a-release-test",
    "fg_sports_equipment": "gold-fg-sports-equipment-merged-a-release-test",
}


def restore(rf_name: str) -> str | None:
    """Roboflow basename -> path relative to OUT, or None if the pattern is unrecognised."""
    stem = re.sub(r"_jpe?g\.rf\.[0-9a-f]+\.jpg$", "", rf_name)
    stem = re.sub(r"_png\.rf\.[0-9a-f]+\.jpg$", "", stem)
    m = re.match(r"^metaclip_(\d+)_(\d+)_", stem)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{stem}.jpeg"
    if stem.startswith("sa_"):
        return f"{stem}.jpg"
    return None


def wanted(subset: str) -> set:
    gt = json.load(open(f"{ANN}/gold_{subset}_merged_a_release_test.json"))
    return {im["file_name"] for im in gt["images"]}


def fetch(subset: str, key: str) -> None:
    want = wanted(subset)
    have = {p for p in want if os.path.exists(os.path.join(OUT, p))}
    if len(have) == len(want):
        print(f"[{subset}] already complete ({len(want)} images)", flush=True)
        return

    url = (f"https://api.roboflow.com/sa-co-gold/{PROJECTS[subset]}/1/coco-segmentation?"
           + urllib.parse.urlencode({"api_key": key}))
    with urllib.request.urlopen(url, timeout=120) as r:
        link = json.load(r)["export"]["link"]

    with tempfile.TemporaryDirectory(dir="/NHNHOME/share/cjpark/data") as tmp:
        zp = os.path.join(tmp, "ds.zip")
        subprocess.run(["curl", "-sL", "-o", zp, link], check=True)
        subprocess.run(["unzip", "-o", "-q", zp, "-d", tmp], check=True)
        moved = unknown = 0
        for root, _, files in os.walk(tmp):
            for f in files:
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                rel = restore(f)
                if rel is None:
                    unknown += 1
                    continue
                dst = os.path.join(OUT, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(os.path.join(root, f), dst)
                moved += 1
    got = sum(1 for p in want if os.path.exists(os.path.join(OUT, p)))
    print(f"[{subset}] moved {moved}, unrecognised {unknown}, "
          f"on disk {got}/{len(want)}{'  ** INCOMPLETE **' if got != len(want) else ''}", flush=True)


def verify() -> int:
    bad = 0
    for subset in PROJECTS:
        want = wanted(subset)
        got = sum(1 for p in want if os.path.exists(os.path.join(OUT, p)))
        flag = "" if got == len(want) else "  ** MISSING **"
        bad += got != len(want)
        print(f"  {subset:<22} {got:>6}/{len(want):<6}{flag}")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--subsets", nargs="*", default=list(PROJECTS))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.verify:
        sys.exit(1 if verify() else 0)
    key = os.environ.get("RF_KEY")
    if not key:
        sys.exit("RF_KEY not set")
    for s in a.subsets:
        fetch(s, key)
    print("\nfinal state:")
    verify()
    print("SACO_GOLD_IMAGES_READY")

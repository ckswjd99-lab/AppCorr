# FourierLaplacianHybrid / FourierLaplacianProgressive — ~10% sanity sweep

DCT-based progressive transmission: instead of building the group-0 global approximation by
spatially downsampling the image (the standard Laplacian pyramid base), take a whole-image 2D DCT
and keep only the low-frequency corner (`pyramid_levels` sets the cutoff, e.g. `[2,0]` → keep
`ceil(dim/4)` coefficients per axis). This avoids the per-16×16-patch block-grid artifacts of the
earlier `FourierProgressive` policy (still present in `fourier_progressive.py`, superseded here) while
keeping the DCT's better energy compaction than a plain spatial blur. Everything past group 0 —
windowed pixel-residual correction for detection (`COCOWindowProgressiveLaplacianPolicy`, `Hybrid`
subclasses it) or grouped residual correction for the other tasks (`ProgressiveLPyramidPolicy`,
`Progressive` subclasses it) — is unchanged from the existing Laplacian-pyramid policies.

- `fourier_laplacian_hybrid.py` — `FourierLaplacianHybridPolicy` (COCO detection only, windowed
  residual correction).
- `fourier_laplacian_progressive.py` — `FourierLaplacianProgressivePolicy` (general grouped-residual
  correction: classification, segmentation).
- `nyu_appcorr_progressive.py` — `NYUAppCorrFourierLaplacianHybridPolicy` (NYU depth, fixed-grid
  variant of the Progressive class).

Branch `develop/fourier-progressive-transmission` (commits `d0a821d`, `c2835e0`), **not yet merged to
main**. Prior to this sweep, only COCO detection had been measured (`mAP 0.695` hybrid vs `0.690`
baseline at 1.03× latency, near-parity) — ImageNet-1k/ADE20K/NYUv2 had configs but no results.

## ~10%-scale sanity sweep (this run)

Every pair below is the DCT-hybrid config vs. its byte-identical Laplacian-pyramid baseline config
(only `transmission_policy_name` differs), run through `offload/run_local.sh` locally. Scale is ~10%
of each dataset (not full — a quick sanity check before deciding whether to commit to a full-dataset
run and merge):

| Task | metric | N (~10%) | baseline | hybrid (DCT) | Δ |
|---|---|---|---|---|---|
| ImageNet-1k (`imnet_*_g4.json`) | top1 acc | 5000 (strided, all 1000 classes) | 87.82% | 87.86% | **+0.04pp** |
| COCO detection (`coco_interleaved_static*.json`) | mAP | 500 | 0.6494 | 0.6520 | **+0.0026** |
| ADE20K M2F segmentation (`ade20k_m2f_*_appcorr.json`) | mIoU | 200 (`validation[:10%]`) | 53.55 | 55.67 | **+2.12pp** |
| NYUv2 depth (`nyu_*_appcorr.json`) | abs_rel (lower better) | 72 | 0.0901 | 0.0921 | −0.0020 |

**The DCT-hybrid base is not worse than the plain Laplacian-pyramid base on any of the 4 tasks** —
matches or slightly beats baseline on 3/4 (ImageNet, COCO, ADE20K), a small (0.002 abs_rel) regression
on NYU depth at only N=72. Consistent with the earlier COCO-only near-parity finding, now confirmed to
generalize across classification/detection/segmentation/depth. Recommendation: worth a full-dataset
confirmation run on all 4 (especially NYU, where N=72 is too small to trust the direction) before
merging `FourierLaplacianHybrid`/`FourierLaplacianProgressive` into main as a regular transmission-
policy option.

### Notes on running this sweep

- **ImageNet-1k**: `torchvision.datasets.ImageFolder` orders samples by class directory, so a front
  slice (`-nr N`) would only cover the first ~N/50 classes. Added `dataset_kwargs.sample_stride`
  (`ImageNetLoader.get_loader`, `offload/mobile/dataset.py`) — an evenly-spaced `Subset` across the
  full class-sorted dataset, so a 10%-scale run still spans all 1000 classes (stride=10 → 5000/50000,
  5 images/class evenly spaced).
- **ADE20K**: `dataset_kwargs.split` is passed straight through to HF `datasets.load_dataset`, so
  native slicing syntax (`"validation[:10%]"`) works with no code changes.
- **COCO**: `-nr 500` (front 10% of the 5000-image FiftyOne validation split; image order isn't
  category-sorted the way ImageFolder is, so this isn't class-biased the way a naive ImageNet front
  slice would be).
- **NYU**: `-nr 9` at batch_size=8 → 72/653 images (~11%).
- **Every task config was run with an explicit `--set device=cuda:0` override.** Two of the shipped
  NYU configs (`nyu_fourier_hybrid_appcorr.json`, `nyu_appcorr.json`) hardcode `"device": "cuda:1"`;
  under per-process `CUDA_VISIBLE_DEVICES` remapping (needed to run two configs on two physical GPUs
  concurrently) that would crash or silently target the wrong physical GPU, since only index 0 is
  visible inside each process. Forcing `device=cuda:0` on every run sidesteps this regardless of which
  physical GPU a given config's author had in mind.
- **COCO's FiftyOne backend cannot run two loaders concurrently.** `foz.load_zoo_dataset` starts a
  local MongoDB service keyed to a fixed `~/.fiftyone/var/lib/mongo` dbpath; launching two COCO configs
  at the same wall-clock time races to cold-start that service and one process's `mongod` exits with
  error 100 (lock contention) — full traceback lands in `offload/mobile/source.py`'s
  `get_dataset_loader` → `COCO2017Loader.__init__` → `foz.load_zoo_dataset`. The two COCO runs in this
  sweep had to be fully serialized (second one launched only after the first's entire process tree,
  including its server-side `mongod`, had exited) rather than run in parallel across GPUs like every
  other task pair.

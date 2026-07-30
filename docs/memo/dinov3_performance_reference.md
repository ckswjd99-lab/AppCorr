# DINOv3 task performance reference

Frequently reused accuracy anchors for the DINOv3 workloads. Add only results
measured on the complete validation split, and record enough of the input
semantics and runtime configuration to distinguish superficially similar
baselines.

## COCO 2017 detection — native-pyramid resolution anchors

Measured 2026-07-30 with DINOv3 ViT-7B + COCO DETR head on the complete
COCO val2017 split (5,000 images), batch size 1, BF16 model weights, and a
1024x1024 detector input grid. One warm-up request was excluded. The process
was isolated with `CUDA_VISIBLE_DEVICES=1` on an NVIDIA B200; the config's
`cuda:0` therefore referred to physical GPU 1.

The image pyramid is constructed from the native source image **before** model
resize:

- L2-only: native image is reduced twice with `cv2.pyrDown`, projected to
  256x256, then expanded to the detector's 1024x1024 input contract.
- L1-only: native image is reduced once, projected to 512x512, then expanded
  to 1024x1024.
- L0-only / Full: the native L0 image is projected directly to 1024x1024.

| Input | COCO mAP | AP50 | AP75 | Request latency | Full-inference event |
|---|---:|---:|---:|---:|---:|
| L2-only | 49.42 | 65.54 | 54.32 | 223.4 ms | 200.9 ms |
| L1-only | 59.32 | 76.53 | 65.72 | 257.0 ms | 201.7 ms |
| L0-only / Full | **63.11** | **80.70** | **69.92** | 285.7 ms | 202.3 ms |

Relative to L0/Full, L1 loses 3.80 mAP and L2 loses 13.70 mAP.

These are **content-resolution accuracy anchors**, not reduced-token compute
benchmarks. All three decoded images satisfy the same 1024x1024 detector input
contract, so the model executes essentially the same work; the nearly identical
~201–202 ms full-inference events confirm this. Request-latency differences
mainly reflect encoding, decoding, and local transfer volume.

### Reproduction

```bash
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=1

offload/run_local.sh offload/config/coco_approx_only_l2.json -nw 1 \
  --set dataset_kwargs.download_if_necessary=false
offload/run_local.sh offload/config/coco_approx_only_l1.json -nw 1 \
  --set dataset_kwargs.download_if_necessary=false
offload/run_local.sh offload/config/coco_sequential.json -nw 1 \
  --set dataset_kwargs.download_if_necessary=false
```

Local raw summaries, if retained on the experiment host:

- `logs/offload/coco_native_l2_fullval_20260730_143335/summary.json`
- `logs/offload/coco_native_l1_fullval_20260730_145623/summary.json`
- `logs/offload/coco_native_full_fullval_20260730_152154/summary.json`

The raw logs are ignored artifacts and are not the durable source of record;
the table above is.


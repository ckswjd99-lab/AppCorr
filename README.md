# 𓂀 Oudjat

**Oudjat** is a research prototype for **mobile → server vision inference offloading** built on DINOv3 backbones. It studies progressive transmission, scheduling, and an *approximate-then-correct* inference scheme: the server runs an approximate forward pass on a low-resolution base layer as soon as it arrives, then corrects only the most informative tokens once the residual layer is received.

The name follows the Eye of Horus / *oudjat* (𓂀): the eye that sees first at a glance, then refines. In ancient Egyptian mathematics, the six parts of the oudjat encoded the dyadic fractions 1/2, 1/4, 1/8, 1/16, 1/32, 1/64 — a fitting mascot for a Laplacian pyramid whose layers halve in resolution at each step.

## Layout

- `offload/` — the runtime. Server, mobile client, transmission/scheduling policies, model executors. Most engineering work happens here.
- `appcorr/` — vendored DINOv3 model code and detection/segmentation configs. Imported by `offload/server/model/*`. Treat as a library.
- `analysis/` — offline experiment scripts and log post-processing tools (no live pipeline required). See `analysis/README.md`.

## Running

A real deployment runs the two halves on different machines. Start the server first, then point the mobile client at it:

```bash
# Server host (GPU machine)
python -m offload.server.main \
    --recv-port 39998 \
    --send-port 39999

# Mobile host (client)
python -m offload.mobile.main \
    --ip <server-ip> \
    --recv-port 39998 \
    --send-port 39999 \
    --config offload/config/<config>.json \
    [--data <dataset-root>] [-nr N] [-nw N]
```

The server binds the two TCP ports and waits; the mobile client connects, performs a time-sync handshake, then streams encoded patches. `--ip` defaults to `127.0.0.1`, `--recv-port`/`--send-port` default to `39998`/`39999` on both sides — they must match. `-nr` limits the number of requests, `-nw` sets warm-up requests (default 1), `--data` overrides `dataset_kwargs.data_root` from the config.

### Local convenience wrapper

For single-host evaluation, `offload/run_local.sh` spawns both processes against one config:

```bash
offload/run_local.sh offload/config/<config>.json [-nr N] [-nw N] [-d DATA_ROOT] [-ns]
```

- `-nr N` — only push N requests/batches (omit to run the whole dataset).
- `-nw N` — number of warm-up requests before measurement (default 1).
- `-d PATH` — override `dataset_kwargs.data_root` for this run.
- `-ns` — profile the server with Nsight Systems (`temp_profile.nsys-rep`).

Environment overrides: `RECV_PORT` (default 39998, mobile → server), `SEND_PORT` (default 39999, server → mobile), `SERVER_STARTUP` (seconds to wait before launching mobile, default 2).

## Supported workloads

| Dataset       | Task                  | Executor                    | Configs        |
|---------------|-----------------------|-----------------------------|----------------|
| ImageNet-1k   | Classification        | `dinov3_classifier`         | `imnet_*.json` |
| COCO / LVIS   | Detection             | `dinov3_detector`           | `coco_*.json`  |
| ADE20K        | Segmentation (M2F)    | `dinov3_segmentor`          | `ade20k_*_m2f_*.json` |
| ADE20K        | Segmentation (linear) | `dinov3_segmentor_linhead`  | `ade20k_linhead_*.json` |
| NYU           | Depth estimation      | `dinov3_depther`            | `nyu_*.json`   |

## Architecture in one breath

Mobile encodes images into progressive Laplacian patches via an `ITransmissionPolicy` and streams them over TCP. The server's scheduler buffers patches and, when an `ISchedulingPolicy` says so, emits a `Task` of `Instruction(op_type, params)` ops. A three-thread worker (decoder → GPU launcher → reaper) executes the instructions on a DINOv3 executor, recording CUDA events for accurate timing without serializing the GPU. Results stream back over a second TCP channel.

## License

See `LICENSE`.

---

𓂀 *"Sees first at a glance, then refines."*

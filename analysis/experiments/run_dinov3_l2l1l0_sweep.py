"""Run the 1,024-image tail-full/L2-L1-L0 calibration matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = (
    REPO_ROOT
    / "analysis"
    / "experiments"
    / "dinov3_l2l1l0_eval.py"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="logs/dinov3_l2l1l0/calibration")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_jobs():
    jobs = [
        {
            "label": "full_sequential",
            "config": "offload/config/imnet_sequential.json",
        },
        {
            "label": "l2_approx_only",
            "config": "offload/config/imnet_approx_only_l2.json",
        },
    ]
    modes = [
        (
            "l2l0",
            "offload/config/imnet_interleaved_g4.json",
        ),
        (
            "l2l1l0",
            "offload/config/imnet_interleaved_l2l1l0_g4.json",
        ),
    ]
    for mode_name, config in modes:
        for final_full_layers in (0, 2, 3):
            for keep_ratio in (0.25, 0.4, 0.6, 1.0):
                jobs.append(
                    {
                        "label": (
                            f"{mode_name}_n{final_full_layers}_"
                            f"k{keep_ratio:g}"
                        ),
                        "config": config,
                        "final_full_layers": final_full_layers,
                        "keep_ratio": keep_ratio,
                        "energy_attn_pscore": True,
                    }
                )
    return jobs


def run_device_queue(device, jobs, args, output_dir):
    physical_device = device.removeprefix("cuda:")
    if not physical_device.isdigit():
        raise ValueError(
            "Each --devices entry must be a physical CUDA index, "
            f"got {device!r}"
        )
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = physical_device

    results = []
    for job in jobs:
        output_path = output_dir / f"{job['label']}.json"
        if output_path.exists() and not args.force:
            print(f"[sweep] resume-skip {output_path}", flush=True)
            results.append(str(output_path))
            continue

        command = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--config",
            job["config"],
            "--data-root",
            args.data_root,
            "--num-samples",
            str(args.num_samples),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            "cuda:0",
            "--label",
            job["label"],
            "--output-json",
            str(output_path),
        ]
        if "final_full_layers" in job:
            command.extend(
                [
                    "--final-full-layers",
                    str(job["final_full_layers"]),
                    "--token-keep-ratio",
                    str(job["keep_ratio"]),
                ]
            )
        if job.get("energy_attn_pscore"):
            command.append("--energy-attn-pscore")

        print(
            f"[sweep][{device}] start {job['label']}",
            flush=True,
        )
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=child_env,
            check=True,
        )
        results.append(str(output_path))
    return results


def main():
    args = parse_args()
    devices = [
        value.strip()
        for value in args.devices.split(",")
        if value.strip()
    ]
    if not devices:
        raise SystemExit("--devices must contain at least one device")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs()
    queues = [jobs[index::len(devices)] for index in range(len(devices))]
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(
                run_device_queue,
                device,
                device_jobs,
                args,
                output_dir,
            )
            for device, device_jobs in zip(devices, queues)
        ]
        result_files = []
        for future in futures:
            result_files.extend(future.result())

    index_path = output_dir / "calibration_index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "devices": devices,
                "results": sorted(result_files),
            },
            handle,
            indent=2,
        )
    print(f"[sweep] wrote {index_path}")


if __name__ == "__main__":
    main()

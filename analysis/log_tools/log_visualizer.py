import argparse
import json
import os
import re
import sys
from collections import defaultdict

PAPER_STAGES = ["Encode", "Transmission", "Decode", "Inference", "Refinement"]
PAPER_COLORS = {
    "Encode": "#4E79A7",
    "Transmission": "#F28E2B",
    "Decode": "#59A14F",
    "Inference": "#E15759",
    "Refinement": "#B07AA1",
}
LATENCY_MARKER_COLOR = "#8B0000"
PAPER_ROWS = ["Mobile", "Network", "Server (CPU)", "Server (GPU)"]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize offload events timeline")
    parser.add_argument("log_dir", type=str, help="Directory containing events.jsonl")
    parser.add_argument(
        "--max_sessions",
        type=int,
        default=10,
        help="Maximum number of sessions to analyze (default: 10)",
    )
    parser.add_argument(
        "--style",
        choices=("default", "paper", "both"),
        default="default",
        help="Visualization style to render (default: default)",
    )
    return parser.parse_args()


def import_matplotlib():
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for visualization, but the active environment "
            f"could not import it cleanly: {type(exc).__name__}: {exc}"
        ) from exc
    return plt, mpatches


def classify_event(event_name):
    server_gpu_events = {
        "LOAD_INPUT",
        "PREPARE_TOKENS",
        "FULL_INFERENCE",
        "APPROX_FORWARD",
        "CORRECT_FORWARD",
        "HEAD_INFERENCE",
        "DECIDE_EXIT",
        "Preprocess",
        "Preprocess::PinMemory",
        "Preprocess::ToDevice",
        "Preprocess::Slicing",
        "Preprocess::GroupMap",
        "Preprocess::Dindices",
    }

    if event_name.startswith("MOBILE_"):
        if event_name.startswith("MOBILE_SEND") or event_name in {"MOBILE_RECV", "MOBILE_RECEIVE"}:
            return "Network"
        return "Mobile"
    if event_name in {"SERVER_RECEIVE", "SERVER_SEND"}:
        return "Network"
    if event_name == "Decode":
        return "Server (CPU)"
    if event_name in server_gpu_events or event_name.startswith("Preprocess::"):
        return "Server (GPU)"

    parts = event_name.split("_")
    if parts[0] == "MOBILE":
        return "Mobile"
    if parts[0] == "SERVER":
        return "Server (GPU)"
    return "Server (GPU)"


def read_events(file_path):
    requests = []
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                requests.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return requests


COLOR_PALETTE = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#FFA07A",
    "#98D8C8",
    "#F06292",
    "#AED581",
    "#FFD54F",
    "#4DB6AC",
    "#7986CB",
    "#A1887F",
    "#90A4AE",
    "#E040FB",
    "#00BCD4",
    "#FF9800",
]


def get_color(event_name, color_map):
    if event_name not in color_map:
        color_map[event_name] = COLOR_PALETTE[len(color_map) % len(COLOR_PALETTE)]
    return color_map[event_name]


def normalize_events(request_data):
    events = request_data if isinstance(request_data, list) else request_data.get("events", [])
    valid_events = []

    for event in events:
        if "type" not in event:
            continue

        start_time = None
        end_time = None
        if "start" in event and "end" in event:
            start_time = event["start"]
            end_time = event["end"]
        elif "timestamp" in event and "duration" in event:
            start_time = event["timestamp"]
            end_time = event["timestamp"] + event["duration"]

        if start_time is None or end_time is None:
            continue

        normalized = event.copy()
        normalized["start"] = start_time
        normalized["end"] = end_time
        valid_events.append(normalized)

    valid_events.sort(key=lambda item: item["start"])
    return valid_events


def get_group_index(event_name, params=None):
    params = params or {}
    match = re.search(r"_G(\d+)$", event_name)
    if match:
        return int(match.group(1))
    if "group_id" in params:
        try:
            return int(params["group_id"])
        except (TypeError, ValueError):
            return None
    return None


def compute_uplink_segments(valid_events, base_time):
    server_recvs = sorted(
        [event for event in valid_events if event["type"] == "SERVER_RECEIVE"],
        key=lambda event: event["start"],
    )

    mobile_sends = {}
    for event in valid_events:
        if event["type"].startswith("MOBILE_SEND"):
            mobile_sends[event["type"]] = event["start"] - base_time

    uplink_segments = []
    prev_server_recv_time = None
    idx = 0
    while True:
        send_key = f"MOBILE_SEND_G{idx}"
        if send_key not in mobile_sends:
            if idx == 0 and "MOBILE_SEND" in mobile_sends:
                send_key = "MOBILE_SEND"
            else:
                break

        if idx < len(server_recvs):
            mobile_tx_start = mobile_sends[send_key]
            server_rx_end = server_recvs[idx]["end"] - base_time
            tx_start_eff = mobile_tx_start
            if prev_server_recv_time is not None:
                tx_start_eff = max(tx_start_eff, prev_server_recv_time)

            if server_rx_end > tx_start_eff:
                uplink_segments.append(
                    {
                        "stage": "Transmission",
                        "start": tx_start_eff,
                        "end": server_rx_end,
                        "duration": server_rx_end - tx_start_eff,
                        "group": idx,
                    }
                )
            prev_server_recv_time = server_rx_end
        idx += 1

    return uplink_segments


def merge_segments(segments, gap_tolerance=1e-6):
    if not segments:
        return []

    ordered = sorted(segments, key=lambda seg: (seg["start"], seg["end"]))
    merged = [ordered[0].copy()]

    for seg in ordered[1:]:
        last = merged[-1]
        if seg["start"] <= last["end"] + gap_tolerance:
            last["end"] = max(last["end"], seg["end"])
            last["duration"] = last["end"] - last["start"]
        else:
            merged.append(seg.copy())

    for seg in merged:
        seg["duration"] = seg["end"] - seg["start"]
    return merged


def map_paper_stage(event_name):
    if event_name == "MOBILE_LOAD" or event_name.startswith("MOBILE_ENCODE"):
        return "Encode"
    if event_name.startswith("MOBILE_SEND") or event_name in {"SERVER_RECEIVE", "SERVER_SEND"}:
        return "Transmission"
    if event_name == "Decode":
        return "Decode"
    if event_name == "CORRECT_FORWARD":
        return "Refinement"
    if event_name in {"SEND_RESPONSE", "FREE_SESSION", "EXIT_ALL", "MOBILE_RECEIVE"}:
        return None
    return "Inference"


def paper_row_for_stage(stage):
    if stage == "Encode":
        return "Mobile"
    if stage == "Transmission":
        return "Network"
    if stage == "Decode":
        return "Server (CPU)"
    return "Server (GPU)"


def build_paper_data(valid_events):
    if not valid_events:
        return None

    base_time = valid_events[0]["start"]
    raw_stage_segments = defaultdict(list)

    for event in valid_events:
        stage = map_paper_stage(event["type"])
        if stage is None or stage == "Transmission":
            continue
        start = event["start"] - base_time
        end = event["end"] - base_time
        if end <= start:
            continue
        raw_stage_segments[stage].append(
            {
                "stage": stage,
                "start": start,
                "end": end,
                "duration": end - start,
                "group": get_group_index(event["type"], event.get("params")),
                "source_type": event["type"],
            }
        )

    raw_stage_segments["Transmission"].extend(compute_uplink_segments(valid_events, base_time))

    stage_segments = {}
    for stage in PAPER_STAGES:
        if stage == "Inference":
            stage_segments[stage] = merge_segments(raw_stage_segments[stage], gap_tolerance=1e-4)
        else:
            stage_segments[stage] = merge_segments(raw_stage_segments[stage], gap_tolerance=1e-6)

    encode_rows = sorted(raw_stage_segments["Encode"], key=lambda seg: seg["start"])
    if encode_rows and len(encode_rows) > 1:
        first_source = encode_rows[0].get("source_type")
        second_source = encode_rows[1].get("source_type")
        if first_source == "MOBILE_LOAD" and second_source and second_source.startswith("MOBILE_ENCODE"):
            encode_rows[1]["duration"] += encode_rows[0]["duration"]
            encode_rows = encode_rows[1:]

    sequential_components = {
        "Encode": encode_rows,
        "Transmission": sorted(raw_stage_segments["Transmission"], key=lambda seg: seg["start"]),
        "Decode": sorted(raw_stage_segments["Decode"], key=lambda seg: seg["start"]),
        "Inference": sorted(stage_segments["Inference"], key=lambda seg: seg["start"]),
    }

    row_count = max((len(segments) for segments in sequential_components.values()), default=0)
    sequential_rows = []
    sequential_total = 0.0
    for row_idx in range(row_count):
        cursor = 0.0
        row_segments = []
        for stage in ["Encode", "Transmission", "Decode", "Inference"]:
            segments = sequential_components[stage]
            if row_idx >= len(segments):
                continue
            duration = segments[row_idx]["duration"]
            if duration <= 0:
                continue
            row_segments.append(
                {
                    "stage": stage,
                    "start": cursor,
                    "end": cursor + duration,
                    "duration": duration,
                }
            )
            cursor += duration
        sequential_total = max(sequential_total, cursor)
        sequential_rows.append(row_segments)

    sequential_stage_spans = {}
    cursor = 0.0
    for stage in ["Encode", "Transmission", "Decode", "Inference"]:
        total_duration = sum(segment["duration"] for segment in sequential_components[stage])
        sequential_stage_spans[stage] = {
            "stage": stage,
            "start": cursor,
            "end": cursor + total_duration,
            "duration": total_duration,
        }
        cursor += total_duration
    sequential_total = cursor

    actual_total = max(event["end"] for event in valid_events) - base_time
    return {
        "base_time": base_time,
        "actual_total": actual_total,
        "sequential_total": sequential_total,
        "stage_segments": stage_segments,
        "actual_transmission_segments": sorted(raw_stage_segments["Transmission"], key=lambda seg: seg["start"]),
        "sequential_rows": sequential_rows,
        "sequential_stage_spans": sequential_stage_spans,
    }


def plot_timeline(request_index, request_data, output_dir, color_map):
    plt, mpatches = import_matplotlib()
    valid_events = normalize_events(request_data)
    if not valid_events:
        print(f"Warning: No valid events found in request {request_index}")
        return

    base_time = valid_events[0]["start"]
    categories = ["Server (GPU)", "Server (CPU)", "Network", "Mobile"]
    y_pos = {cat: i for i, cat in enumerate(categories)}
    fig, ax = plt.subplots(figsize=(15, 5))

    decode_idx = 0
    for event in valid_events:
        name = event["type"]
        start = event["start"] - base_time
        end = event["end"] - base_time
        duration = max(end - start, 0.0001)
        category = classify_event(name)
        base_name = re.sub(r"_G\d+$", "", name)
        color = get_color(base_name, color_map)
        y = y_pos[category]
        display_name = name.replace("Preprocess::", "Prep:").replace("MOBILE_", "").replace("SERVER_", "")

        if name == "APPROX_FORWARD":
            layers = event.get("params", {}).get("layers", (0, 0))
            if event.get("params", {}).get("source_kind") == "global":
                display_name = "AP\nGLB"
            elif layers[1] == layers[0] + 1:
                display_name = f"AP\n{layers[0]}"
            else:
                display_name = f"AP\n({layers[0]}, {layers[1]})"
        elif name == "CORRECT_FORWARD":
            layers = event.get("params", {}).get("layers", (0, 0))
            gid = event.get("params", {}).get("group_id", "?")
            if layers[1] == layers[0] + 1:
                display_name = f"CO\nG{gid}\n{layers[0]}"
            else:
                display_name = f"CO\nG{gid}\n({layers[0]}, {layers[1]})"
        elif "Decode" in name:
            display_name = f"DEC\nG{decode_idx}"
            decode_idx += 1
        elif "ENCODE" in name or name.startswith("MOBILE_SEND"):
            match = re.search(r"[G_](\d+)$", name)
            gid = match.group(1) if match else event.get("params", {}).get("group_id", "?")
            display_name = f"ENC\nG{gid}"

        bar = ax.barh(
            y,
            width=duration,
            left=start,
            height=0.6,
            align="center",
            color=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )

        if duration > 0.005:
            text = ax.text(
                start + duration / 2,
                y,
                display_name,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
            text.set_clip_path(bar[0])
            text.set_clip_on(True)

    uplink_segments = compute_uplink_segments(valid_events, base_time)
    for idx, segment in enumerate(uplink_segments):
        net_dur = segment["duration"] if "duration" in segment else segment["end"] - segment["start"]
        ax.barh(
            y_pos["Network"],
            width=net_dur,
            left=segment["start"],
            height=0.3,
            align="center",
            color="gray",
            alpha=0.3,
            edgecolor="black",
            linestyle="--",
            zorder=1,
        )
        if net_dur > 0.01:
            ax.text(segment["start"] + net_dur / 2, y_pos["Network"] - 0.25, f"UL {idx}", ha="center", fontsize=7)

    total_mobile = sum(event["end"] - event["start"] for event in valid_events if classify_event(event["type"]) == "Mobile")
    total_cpu = sum(
        event["end"] - event["start"] for event in valid_events if classify_event(event["type"]) == "Server (CPU)"
    )
    total_gpu_no_correct = sum(
        event["end"] - event["start"]
        for event in valid_events
        if classify_event(event["type"]) == "Server (GPU)" and "CORRECT_FORWARD" not in event["type"]
    )
    total_ul = sum(segment["end"] - segment["start"] for segment in uplink_segments)

    sequential_time = total_mobile + total_ul + total_cpu + total_gpu_no_correct
    actual_time = max(event["end"] for event in valid_events) - base_time
    total_mobile_load = sum(event["end"] - event["start"] for event in valid_events if event["type"] == "MOBILE_LOAD")
    ideal_time = total_mobile_load + total_gpu_no_correct

    ax.axvline(x=sequential_time, color="red", linestyle="--", linewidth=2, label="Sequential Baseline", zorder=0)
    ax.text(
        sequential_time - 0.005,
        len(categories) - 0.7,
        f"Sequential baseline: {sequential_time:.3f}s",
        color="red",
        fontweight="bold",
        ha="right",
        va="top",
    )

    ax.axvline(x=actual_time, color="blue", linestyle="--", linewidth=2, label="AppCorr (Ours)", zorder=0)
    ax.text(
        actual_time - 0.005,
        len(categories) - 1.2,
        f"AppCorr (Ours): {actual_time:.3f}s",
        color="blue",
        fontweight="bold",
        ha="right",
        va="top",
    )

    ax.axvline(x=ideal_time, color="green", linestyle="--", linewidth=2, label="Ideal", zorder=0)
    ax.text(
        ideal_time - 0.005,
        len(categories) - 1.7,
        f"Ideal: {ideal_time:.3f}s",
        color="green",
        fontweight="bold",
        ha="right",
        va="top",
    )

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Request {request_index} Timeline Breakdown")
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)

    patches = [mpatches.Patch(color=color_map[key], label=key) for key in sorted(color_map.keys())]
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=8,
        fontsize="small",
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"request_{request_index:04d}_timeline.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_paper_figure(request_index, request_data, output_dir):
    plt, _ = import_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    valid_events = normalize_events(request_data)
    if not valid_events:
        print(f"Warning: No valid events found in request {request_index}")
        return

    paper_data = build_paper_data(valid_events)
    if paper_data is None:
        print(f"Warning: Failed to build paper data for request {request_index}")
        return

    approx_segments = paper_data["stage_segments"].get("Inference", [])
    actual_marker_time = max((segment["end"] for segment in approx_segments), default=paper_data["actual_total"])

    actual_height = 0.97
    sequential_height = actual_height
    fig_height = actual_height + sequential_height

    fig, (ax_seq, ax_actual) = plt.subplots(
        2,
        1,
        figsize=(3.33, fig_height),
        gridspec_kw={"height_ratios": [actual_height, sequential_height]},
        constrained_layout=False,
    )
    fig.subplots_adjust(top=0.9, bottom=0.23, left=0.20, right=0.98, hspace=0.42)

    actual_order = list(reversed(PAPER_ROWS))
    actual_y = {row: idx for idx, row in enumerate(actual_order)}
    for stage in PAPER_STAGES:
        row = paper_row_for_stage(stage)
        segments = paper_data["stage_segments"][stage]
        if stage == "Transmission":
            segments = paper_data["actual_transmission_segments"]
        for segment in segments:
            ax_actual.barh(
                actual_y[row],
                segment["end"] - segment["start"],
                left=segment["start"],
                height=0.62,
                color=PAPER_COLORS[stage],
                edgecolor="black",
                linewidth=0.45,
                zorder=2,
            )

    max_x = max(paper_data["actual_total"], paper_data["sequential_total"], 1e-6) * 1.03
    text_offset = 0.01 * max_x

    ax_actual.set_xlim(0, max_x)
    ax_actual.set_yticks(range(len(PAPER_ROWS)))
    ax_actual.set_yticklabels(actual_order)
    ax_actual.set_ylim(-0.95, len(PAPER_ROWS) - 0.45)
    ax_actual.grid(True, axis="x", linestyle="--", alpha=0.35, linewidth=0.6)
    ax_actual.tick_params(
        axis="x", which="both", bottom=True, top=False, labelbottom=True, direction="in", length=3, width=0.6
    )
    ax_actual.tick_params(axis="y", which="both", left=False, right=False, pad=0.0, width=0.6)
    ax_actual.axvline(actual_marker_time, color=LATENCY_MARKER_COLOR, linestyle="--", linewidth=1.1, zorder=0)
    ax_actual.text(
        max(0.0, actual_marker_time - text_offset),
        0.93,
        f"{actual_marker_time:.2f}s",
        transform=ax_actual.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
        color=LATENCY_MARKER_COLOR,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
    )

    for spine in ("left", "right", "top", "bottom"):
        ax_actual.spines[spine].set_visible(True)
        ax_actual.spines[spine].set_color("black")
        ax_actual.spines[spine].set_linewidth(0.6)

    seq_lane_order = list(reversed(PAPER_ROWS))
    seq_lane_y = {row: idx for idx, row in enumerate(seq_lane_order)}
    for stage in ["Encode", "Transmission", "Decode", "Inference"]:
        span = paper_data["sequential_stage_spans"][stage]
        if span["duration"] <= 0:
            continue
        row = paper_row_for_stage(stage)
        ax_seq.barh(
            seq_lane_y[row],
            span["end"] - span["start"],
            left=span["start"],
            height=0.62,
            color=PAPER_COLORS[stage],
            edgecolor="black",
            linewidth=0.45,
            zorder=2,
        )

    ax_seq.set_yticks(range(len(PAPER_ROWS)))
    ax_seq.set_yticklabels(seq_lane_order)
    ax_seq.set_xlim(0, max_x)
    ax_seq.set_ylim(-0.95, len(PAPER_ROWS) - 0.45)
    ax_seq.grid(True, axis="x", linestyle="--", alpha=0.35, linewidth=0.6)
    ax_seq.tick_params(
        axis="x", which="both", bottom=True, top=False, labelbottom=False, direction="in", length=3, width=0.6
    )
    ax_seq.tick_params(axis="y", which="both", left=False, right=False, pad=0.0, width=0.6)
    ax_seq.axvline(
        paper_data["sequential_total"], color=LATENCY_MARKER_COLOR, linestyle="--", linewidth=1.1, zorder=0
    )
    ax_seq.text(
        max(0.0, paper_data["sequential_total"] - text_offset),
        0.93,
        f"{paper_data['sequential_total']:.2f}s",
        transform=ax_seq.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
        color=LATENCY_MARKER_COLOR,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
    )

    for spine in ("left", "right", "top", "bottom"):
        ax_seq.spines[spine].set_visible(True)
        ax_seq.spines[spine].set_color("black")
        ax_seq.spines[spine].set_linewidth(0.6)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PAPER_COLORS[stage], edgecolor="black", linewidth=0.45, label=stage)
        for stage in PAPER_STAGES
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(PAPER_STAGES),
        frameon=False,
        handlelength=0.9,
        columnspacing=0.45,
        handletextpad=0.25,
        borderaxespad=0.1,
        prop={"size": 8},
    )

    seq_bbox = ax_seq.get_position()
    actual_bbox = ax_actual.get_position()
    fig.text(
        0.5,
        (seq_bbox.y0 + actual_bbox.y1) / 2 - 0.0,
        "(a) Baseline (Sequential)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.14,
        "(b) Ours",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    ax_actual.set_xlabel("(s)", fontsize=8, ha="right")
    ax_actual.xaxis.set_label_coords(1.0, -0.08)

    os.makedirs(output_dir, exist_ok=True)
    out_png_path = os.path.join(output_dir, f"request_{request_index:04d}_paper_timeline.png")
    out_pdf_path = os.path.join(output_dir, f"request_{request_index:04d}_paper_timeline.pdf")
    plt.savefig(out_png_path, dpi=300, pad_inches=0)
    plt.savefig(out_pdf_path, pad_inches=0)
    plt.close(fig)
    print(f"Saved: {out_png_path}")
    print(f"Saved: {out_pdf_path}")


def main():
    args = parse_args()
    log_dir = args.log_dir
    events_file = os.path.join(log_dir, "events.jsonl")
    if not os.path.exists(events_file):
        print(f"Error: {events_file} not found.")
        sys.exit(1)

    requests = read_events(events_file)
    original_count = len(requests)
    requests = requests[: args.max_sessions]
    print(f"Loaded {original_count} requests, analyzing first {len(requests)} from {events_file}")

    color_map = {}
    for idx, req in enumerate(requests):
        if args.style in {"default", "both"}:
            plot_timeline(idx, req, log_dir, color_map)
        if args.style in {"paper", "both"}:
            plot_paper_figure(idx, req, log_dir)

    print("All timelines generated successfully.")


if __name__ == "__main__":
    main()

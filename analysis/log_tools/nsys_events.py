"""Build an events.jsonl-compatible log from an Nsight Systems profile."""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


INSTR_PREFIX = "APPCORR_INSTR|"
EVENT_GROUP_SUFFIX_RE = re.compile(r"_G\d+$")
SUMMARY_BREAKDOWN_NOTE = (
    "Per-request event duration sums from events_nsys.jsonl; detail events are excluded, "
    "matched server GPU instruction events use Nsight GPU projection time, and values are "
    "not critical-path latency contributions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate events_nsys.jsonl from server_profile.nsys-rep and events.jsonl."
    )
    parser.add_argument("log_dir", type=Path, help="Directory containing events.jsonl")
    parser.add_argument("--profile", type=Path, default=None, help="Path to server_profile.nsys-rep or .sqlite")
    parser.add_argument("--events", type=Path, default=None, help="Path to source events.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Path to output events_nsys.jsonl")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path to summary.json to update with events_nsys duration breakdown. Default: LOG_DIR/summary.json",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Re-export the .nsys-rep to SQLite even when the SQLite file exists.",
    )
    return parser.parse_args()


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def resolve_profile_path(log_dir: Path, profile_arg: Path | None) -> Path:
    if profile_arg is not None:
        return profile_arg
    for candidate in (log_dir / "server_profile.nsys-rep", log_dir / "server_profile.sqlite"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No server_profile.nsys-rep or server_profile.sqlite found in {log_dir}")


def ensure_sqlite(profile_path: Path, force_export: bool) -> Path:
    if profile_path.suffix == ".sqlite":
        if not profile_path.exists():
            raise FileNotFoundError(profile_path)
        return profile_path

    if not profile_path.exists():
        raise FileNotFoundError(profile_path)

    sqlite_path = profile_path.with_suffix(".sqlite")
    if sqlite_path.exists() and not force_export:
        return sqlite_path

    if shutil.which("nsys") is None:
        raise RuntimeError("nsys is required to export .nsys-rep files to SQLite")

    subprocess.run(
        [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "--output",
            str(sqlite_path),
            str(profile_path),
        ],
        check=True,
    )
    return sqlite_path


def nvtx_text(row: sqlite3.Row) -> str | None:
    text = row["text"]
    if text:
        return str(text)
    string_value = row["string_value"]
    return str(string_value) if string_value else None


def parse_instr_name(name: str) -> dict[str, Any] | None:
    if not name.startswith(INSTR_PREFIX):
        return None

    fields: dict[str, str] = {}
    for part in name.split("|")[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value

    try:
        req_id = int(fields["req"])
        task_id = int(fields["task"])
        seq = int(fields["seq"])
        op = fields["op"]
    except (KeyError, TypeError, ValueError):
        return None

    return {
        "request_id": req_id,
        "task_id": task_id,
        "nsys_seq": seq,
        "op": op,
    }


def read_instr_ranges(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not table_exists(conn, "NVTX_EVENTS"):
        return []

    query = """
        SELECT
            n.start AS start,
            n.end AS end,
            n.text AS text,
            s.value AS string_value,
            n.globalTid AS global_tid
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s ON n.textId = s.id
        WHERE n.end IS NOT NULL
    """
    ranges: list[dict[str, Any]] = []
    for row in conn.execute(query):
        name = nvtx_text(row)
        if not name:
            continue
        parsed = parse_instr_name(name)
        if parsed is None:
            continue
        ranges.append(
            {
                "name": name,
                "start": int(row["start"]),
                "end": int(row["end"]),
                "global_tid": row["global_tid"],
                **parsed,
            }
        )
    ranges.sort(key=lambda item: (item["start"], item["end"]))
    return ranges


def read_runtime_calls(conn: sqlite3.Connection) -> list[dict[str, int]]:
    if not table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []

    query = """
        SELECT start, end, correlationId AS correlation_id, globalTid AS global_tid
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE correlationId IS NOT NULL
    """
    runtimes = [
        {
            "start": int(row["start"]),
            "end": int(row["end"]),
            "correlation_id": int(row["correlation_id"]),
            "global_tid": row["global_tid"],
        }
        for row in conn.execute(query)
    ]
    runtimes.sort(key=lambda item: (item["start"], item["end"]))
    return runtimes


def read_activity_table(
    conn: sqlite3.Connection,
    table_name: str,
    kind: str,
) -> list[dict[str, int | str]]:
    if not table_exists(conn, table_name):
        return []

    query = f"""
        SELECT start, end, correlationId AS correlation_id
        FROM {table_name}
        WHERE correlationId IS NOT NULL
    """
    activities = []
    for row in conn.execute(query):
        start = int(row["start"])
        end = int(row["end"])
        if end <= start:
            continue
        activities.append(
            {
                "kind": kind,
                "start": start,
                "end": end,
                "correlation_id": int(row["correlation_id"]),
            }
        )
    return activities


def read_gpu_activities(conn: sqlite3.Connection) -> dict[int, list[dict[str, int | str]]]:
    activities = []
    activities.extend(read_activity_table(conn, "CUPTI_ACTIVITY_KIND_KERNEL", "kernel"))
    activities.extend(read_activity_table(conn, "CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"))
    activities.extend(read_activity_table(conn, "CUPTI_ACTIVITY_KIND_MEMSET", "memset"))

    by_correlation: dict[int, list[dict[str, int | str]]] = defaultdict(list)
    for activity in activities:
        by_correlation[int(activity["correlation_id"])].append(activity)
    return by_correlation


def build_projection_map(
    instr_ranges: list[dict[str, Any]],
    runtimes: list[dict[str, int]],
    activities_by_correlation: dict[int, list[dict[str, int | str]]],
) -> dict[str, dict[str, Any]]:
    runtime_starts = [runtime["start"] for runtime in runtimes]
    projections: dict[str, dict[str, Any]] = {}

    for instr_range in instr_ranges:
        runtime_idx = bisect.bisect_left(runtime_starts, instr_range["start"])
        correlation_ids: set[int] = set()
        while runtime_idx < len(runtimes):
            runtime = runtimes[runtime_idx]
            if runtime["start"] > instr_range["end"]:
                break
            if (
                instr_range.get("global_tid") is not None
                and runtime.get("global_tid") != instr_range.get("global_tid")
            ):
                runtime_idx += 1
                continue
            correlation_ids.add(runtime["correlation_id"])
            runtime_idx += 1

        activities = []
        seen = set()
        for correlation_id in correlation_ids:
            for activity in activities_by_correlation.get(correlation_id, []):
                activity_key = (
                    activity["kind"],
                    activity["start"],
                    activity["end"],
                    activity["correlation_id"],
                )
                if activity_key in seen:
                    continue
                seen.add(activity_key)
                activities.append(activity)

        if not activities:
            projections[instr_range["name"]] = {
                "range": instr_range,
                "activity_count": 0,
                "unmatched_reason": "no_gpu_activity",
            }
            continue

        start = min(int(activity["start"]) for activity in activities)
        end = max(int(activity["end"]) for activity in activities)
        duration_by_kind = {"kernel": 0, "memcpy": 0, "memset": 0}
        for activity in activities:
            kind = str(activity["kind"])
            duration_by_kind[kind] = duration_by_kind.get(kind, 0) + int(activity["end"]) - int(activity["start"])

        projections[instr_range["name"]] = {
            "range": instr_range,
            "start": start,
            "end": end,
            "activity_count": len(activities),
            "duration_by_kind": duration_by_kind,
        }

    return projections


def read_events_jsonl(events_path: Path) -> list[dict[str, Any]]:
    rows = []
    with events_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def event_timestamp(event: dict[str, Any]) -> float | None:
    if "timestamp" in event:
        return float(event["timestamp"])
    if "start" in event:
        return float(event["start"])
    return None


def compute_timebase_offset(
    rows: list[dict[str, Any]],
    instr_ranges_by_name: dict[str, dict[str, Any]],
) -> float:
    offsets = []
    for row in rows:
        for event in row.get("events", []):
            if not isinstance(event, dict):
                continue
            meta = event.get("meta")
            if not isinstance(meta, dict):
                continue
            range_name = meta.get("nsys_range")
            instr_range = instr_ranges_by_name.get(range_name)
            if instr_range is None:
                continue
            timestamp = event_timestamp(event)
            if timestamp is None:
                continue
            offsets.append(timestamp - instr_range["start"] / 1_000_000_000.0)

    if not offsets:
        raise RuntimeError("No events with matching meta.nsys_range were found; cannot align Nsight timebase.")
    return float(median(offsets))


def set_event_timing(event: dict[str, Any], start: float, end: float) -> None:
    duration = max(end - start, 0.0)
    event["timestamp"] = start
    event["duration"] = duration
    if "start" in event or "end" in event:
        event["start"] = start
        event["end"] = end


def seconds(ns: int | float) -> float:
    return float(ns) / 1_000_000_000.0


def rewrite_events_with_nsys(
    rows: list[dict[str, Any]],
    projections: dict[str, dict[str, Any]],
    instr_ranges_by_name: dict[str, dict[str, Any]],
    timebase_offset: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "events_seen": 0,
        "events_with_nsys_range": 0,
        "events_projected": 0,
        "events_without_gpu_activity": 0,
        "events_without_nvtx_range": 0,
    }
    output_rows = []

    for row in rows:
        row_copy = copy.deepcopy(row)
        rewritten_events = []
        for event in row_copy.get("events", []):
            stats["events_seen"] += 1
            if not isinstance(event, dict):
                rewritten_events.append(event)
                continue

            meta = event.get("meta")
            if not isinstance(meta, dict):
                rewritten_events.append(event)
                continue

            range_name = meta.get("nsys_range")
            if not range_name:
                rewritten_events.append(event)
                continue

            stats["events_with_nsys_range"] += 1
            instr_range = instr_ranges_by_name.get(range_name)
            projection = projections.get(range_name)
            if instr_range is None or projection is None:
                meta["nsys_unmatched_reason"] = "nvtx_range_not_found"
                stats["events_without_nvtx_range"] += 1
                rewritten_events.append(event)
                continue

            if projection.get("activity_count", 0) <= 0:
                meta["nsys_unmatched_reason"] = projection.get("unmatched_reason", "no_gpu_activity")
                stats["events_without_gpu_activity"] += 1
                rewritten_events.append(event)
                continue

            start = seconds(projection["start"]) + timebase_offset
            end = seconds(projection["end"]) + timebase_offset
            set_event_timing(event, start, end)

            duration_by_kind = projection["duration_by_kind"]
            kernel_duration = seconds(duration_by_kind.get("kernel", 0))
            memcpy_duration = seconds(duration_by_kind.get("memcpy", 0))
            memset_duration = seconds(duration_by_kind.get("memset", 0))
            gpu_active_duration = kernel_duration + memcpy_duration + memset_duration
            meta.update(
                {
                    "nsys_timer": "gpu_projection",
                    "nsys_gpu_active_duration": gpu_active_duration,
                    "nsys_kernel_duration": kernel_duration,
                    "nsys_memcpy_duration": memcpy_duration,
                    "nsys_memset_duration": memset_duration,
                    "nsys_gpu_projection_duration": max(end - start, 0.0),
                    "nsys_activity_count": int(projection["activity_count"]),
                    "nsys_host_range_start": seconds(instr_range["start"]) + timebase_offset,
                    "nsys_host_range_duration": seconds(instr_range["end"] - instr_range["start"]),
                }
            )
            meta.pop("nsys_unmatched_reason", None)
            stats["events_projected"] += 1
            rewritten_events.append(event)

        if isinstance(row_copy.get("events"), list):
            row_copy["events"] = rewritten_events
        output_rows.append(row_copy)

    return output_rows, stats


def write_events_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def latency_event_type(event_type: str) -> str:
    return EVENT_GROUP_SUFFIX_RE.sub("", event_type)


def include_summary_event(event: dict[str, Any]) -> bool:
    if event.get("include_in_latency_stats") is False:
        return False
    if event.get("detail") is True:
        return False
    event_type = str(event.get("type", ""))
    if event_type.startswith(("Prepare::", "Preprocess::")):
        return False
    return True


def event_duration_seconds(event: dict[str, Any]) -> float:
    if "duration" in event:
        return max(float(event.get("duration") or 0.0), 0.0)
    if "start" in event and "end" in event:
        return max(float(event["end"]) - float(event["start"]), 0.0)
    return 0.0


def build_event_duration_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    accumulator: dict[str, dict[str, float | int]] = {}

    for row in rows:
        request_duration_map: dict[str, float] = {}
        for event in row.get("events", []):
            if not isinstance(event, dict) or not include_summary_event(event):
                continue
            event_type = latency_event_type(str(event.get("type", "")))
            request_duration_map[event_type] = request_duration_map.get(event_type, 0.0) + (
                event_duration_seconds(event) * 1000.0
            )

        for event_type, duration_ms in request_duration_map.items():
            stats = accumulator.setdefault(
                event_type,
                {"count": 0, "sum": 0.0, "min": float("inf"), "max": float("-inf")},
            )
            stats["count"] = int(stats["count"]) + 1
            stats["sum"] = float(stats["sum"]) + duration_ms
            stats["min"] = min(float(stats["min"]), duration_ms)
            stats["max"] = max(float(stats["max"]), duration_ms)

    breakdown: dict[str, dict[str, float | int]] = {}
    sorted_stats = sorted(accumulator.items(), key=lambda item: float(item[1]["sum"]), reverse=True)
    for event_type, stats in sorted_stats:
        count = int(stats["count"])
        if count <= 0:
            continue
        breakdown[event_type] = {
            "avg_ms": float(stats["sum"]) / count,
            "min_ms": float(stats["min"]),
            "max_ms": float(stats["max"]),
            "count": count,
        }
    return breakdown


def update_summary_json(
    summary_path: Path,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> bool:
    if not summary_path.exists():
        return False

    with summary_path.open("r") as f:
        summary = json.load(f)

    breakdown = build_event_duration_breakdown(rows)
    summary["latency_breakdown"] = breakdown
    summary["event_duration_breakdown"] = breakdown
    summary["latency_breakdown_note"] = SUMMARY_BREAKDOWN_NOTE
    summary["event_duration_breakdown_source"] = output_path.name
    summary["event_duration_breakdown_timer"] = "nsys_gpu_projection"

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=4)
    return True


def generate_events_nsys(
    profile_path: Path,
    events_path: Path,
    output_path: Path,
    summary_path: Path | None,
    force_export: bool,
) -> dict[str, int]:
    sqlite_path = ensure_sqlite(profile_path, force_export)
    rows = read_events_jsonl(events_path)

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        instr_ranges = read_instr_ranges(conn)
        if not instr_ranges:
            raise RuntimeError("No APPCORR_INSTR NVTX ranges found in the Nsight profile.")
        runtimes = read_runtime_calls(conn)
        activities_by_correlation = read_gpu_activities(conn)
    finally:
        conn.close()

    instr_ranges_by_name = {item["name"]: item for item in instr_ranges}
    projections = build_projection_map(instr_ranges, runtimes, activities_by_correlation)
    timebase_offset = compute_timebase_offset(rows, instr_ranges_by_name)
    output_rows, stats = rewrite_events_with_nsys(rows, projections, instr_ranges_by_name, timebase_offset)
    write_events_jsonl(output_rows, output_path)
    summary_updated = False
    if summary_path is not None:
        summary_updated = update_summary_json(summary_path, output_rows, output_path)

    stats["nvtx_instr_ranges"] = len(instr_ranges)
    stats["runtime_calls"] = len(runtimes)
    stats["correlated_activity_groups"] = len(activities_by_correlation)
    stats["timebase_offset_ms"] = int(round(timebase_offset * 1000.0))
    stats["summary_updated"] = int(summary_updated)
    return stats


def main() -> int:
    args = parse_args()
    log_dir = args.log_dir
    events_path = args.events or (log_dir / "events.jsonl")
    output_path = args.output or (log_dir / "events_nsys.jsonl")
    if args.summary is not None:
        summary_path = args.summary if args.summary.is_absolute() else log_dir / args.summary
    else:
        summary_path = log_dir / "summary.json"

    try:
        profile_path = resolve_profile_path(log_dir, args.profile)
        stats = generate_events_nsys(
            profile_path=profile_path,
            events_path=events_path,
            output_path=output_path,
            summary_path=summary_path,
            force_export=args.force_export,
        )
    except Exception as exc:
        print(f"[nsys_events] Error: {exc}", file=sys.stderr)
        return 1

    print(
        "[nsys_events] Wrote "
        f"{output_path} "
        f"(projected={stats['events_projected']}, "
        f"no_gpu={stats['events_without_gpu_activity']}, "
        f"missing_range={stats['events_without_nvtx_range']}, "
        f"summary_updated={stats['summary_updated']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

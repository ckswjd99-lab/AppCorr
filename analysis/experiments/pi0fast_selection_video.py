"""Write pi0-FAST policy-input videos with recomputed vision patches highlighted."""

import json
import math
from pathlib import Path


def render_selection_frame(trace_item):
    import cv2
    import numpy as np

    panels = []
    for camera_index, (image, selected) in enumerate(
        zip(trace_item["images"], trace_item["selected"])
    ):
        height, width = image.shape[:2]
        token_count = 256
        grid_size = int(round(token_count ** 0.5))
        selected_mask = np.zeros((height, width), dtype=bool)
        patch_height = height / grid_size
        patch_width = width / grid_size
        for token_index in selected:
            row, column = divmod(int(token_index), grid_size)
            y0 = round(row * patch_height)
            y1 = round((row + 1) * patch_height)
            x0 = round(column * patch_width)
            x1 = round((column + 1) * patch_width)
            selected_mask[y0:y1, x0:x1] = True
        panel = image.copy()
        panel[~selected_mask] = (
            panel[~selected_mask].astype(np.float32) * 0.3
        ).astype(np.uint8)
        panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        for token_index in selected:
            row, column = divmod(int(token_index), grid_size)
            y0 = round(row * patch_height)
            y1 = round((row + 1) * patch_height) - 1
            x0 = round(column * patch_width)
            x1 = round((column + 1) * patch_width) - 1
            cv2.rectangle(panel, (x0, y0), (x1, y1), (255, 255, 0), 1)
        cv2.putText(
            panel,
            f"camera {camera_index + 1}: {len(selected)}/{token_count}",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    if not panels:
        raise RuntimeError("Selection trace has no real camera images")
    body = np.concatenate(panels, axis=1)
    header = np.zeros((38, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"{trace_item['score_mode']} | cyan=recomputed, dim=approximate",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([header, body], axis=0)


def write_selection_videos(output_dir, selection_trace, action_horizon):
    """Match policy decisions to official videos and write side-by-side camera diagnostics."""
    if not selection_trace:
        print("[selection-video] no selection trace was captured", flush=True)
        return []
    import cv2

    info_path = Path(output_dir) / "eval_info.json"
    if not info_path.exists():
        print(f"[selection-video] missing {info_path}", flush=True)
        return []
    info = json.loads(info_path.read_text())
    source_videos = info["overall"]["video_paths"]
    trace_cursor = 0
    action_horizon = max(int(action_horizon or 1), 1)
    destinations = []
    for source_text in source_videos:
        source = Path(source_text)
        capture = cv2.VideoCapture(str(source))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        capture.release()
        decision_count = math.ceil(frame_count / action_horizon)
        episode_trace = selection_trace[
            trace_cursor:trace_cursor + decision_count
        ]
        trace_cursor += decision_count
        if len(episode_trace) != decision_count:
            raise RuntimeError(
                f"{source.name}: expected {decision_count} policy decisions, "
                f"found {len(episode_trace)}"
            )
        first_frame = render_selection_frame(episode_trace[0])
        destination = source.with_name(f"{source.stem}_selection.mp4")
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (first_frame.shape[1], first_frame.shape[0]),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open selection video {destination}")
        for frame_index in range(frame_count):
            decision_index = min(
                frame_index // action_horizon,
                decision_count - 1,
            )
            writer.write(render_selection_frame(episode_trace[decision_index]))
        writer.release()
        destinations.append(destination)
        print(f"[selection-video] saved {destination}", flush=True)
    if trace_cursor != len(selection_trace):
        raise RuntimeError(
            f"Selection trace has {len(selection_trace) - trace_cursor} unused decisions"
        )
    return destinations

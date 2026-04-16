import argparse
import json
import os
import socket
import signal
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a FiftyOne web viewer for COCO detection logs"
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Experiment log directory containing summary.json and detections/fiftyone_predictions.json",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional FiftyOne dataset name override for the temporary viewer dataset.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5151,
        help="Port for the FiftyOne app. Defaults to 5151.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind host for the FiftyOne app. Use 0.0.0.0 to allow external access.",
    )
    parser.add_argument(
        "--predictions-field",
        type=str,
        default="predictions",
        help="Field name to use for loaded detection predictions.",
    )
    parser.add_argument(
        "--ground-truth-field",
        type=str,
        default=None,
        help="Optional ground-truth field override. Defaults to the COCO zoo dataset default label field.",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Keep the generated FiftyOne dataset after the viewer exits.",
    )
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def _default_dataset_name(log_dir: Path) -> str:
    safe_name = log_dir.name.replace("-", "_")
    return f"log_{safe_name}_viewer"


def _get_public_host() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def _normalize_bbox(bbox):
    if len(bbox) != 4:
        raise ValueError(f"Expected [x, y, w, h] bbox, got {bbox!r}")
    return [float(v) for v in bbox]


def _infer_label_field(dataset, fallback="ground_truth"):
    field_name = getattr(dataset, "default_label_field", None)
    if isinstance(field_name, str) and field_name:
        return field_name

    try:
        import fiftyone as fo

        schema = dataset.get_field_schema()
        for name, field in schema.items():
            document_type = getattr(field, "document_type", None)
            if document_type in (fo.Detections, fo.Polylines, fo.Keypoints):
                return name
    except Exception:
        pass

    return fallback


def build_dataset(args):
    import fiftyone as fo
    import fiftyone.zoo as foz
    from fiftyone import ViewField as F

    log_dir = Path(args.log_dir).expanduser().resolve()
    detections_path = log_dir / "detections" / "fiftyone_predictions.json"
    summary_path = log_dir / "summary.json"

    if not detections_path.exists():
        raise FileNotFoundError(
            f"Detection export not found: {detections_path}. "
            "Run a COCO experiment after enabling the new detection logging."
        )

    detections_payload = _load_json(detections_path)
    summary = _load_json(summary_path) if summary_path.exists() else {}

    split = detections_payload.get("split", "validation")
    source_dataset_name = detections_payload.get("fiftyone_dataset_name")
    viewer_dataset_name = args.dataset_name or _default_dataset_name(log_dir)

    if fo.dataset_exists(viewer_dataset_name):
        fo.delete_dataset(viewer_dataset_name)

    source_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        dataset_name=source_dataset_name,
    )
    ground_truth_field = args.ground_truth_field or detections_payload.get("fiftyone_label_field")
    if not ground_truth_field:
        ground_truth_field = _infer_label_field(source_dataset)
    image_ids = detections_payload.get("processed_image_ids", [])
    sample_entries = detections_payload.get("samples", [])

    predictions_by_image = {
        int(sample["image_id"]): list(sample.get("predictions", []))
        for sample in sample_entries
    }
    filepaths_by_image = {
        int(sample["image_id"]): sample.get("filepath")
        for sample in sample_entries
        if sample.get("filepath")
    }

    samples = source_dataset.match(F("filepath").is_in(list(filepaths_by_image.values())))
    viewer_dataset = samples.clone(viewer_dataset_name, persistent=args.persist)
    schema = viewer_dataset.get_field_schema()
    if args.predictions_field not in schema:
        viewer_dataset.add_sample_field(
            args.predictions_field,
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )

    assigned_images = 0
    assigned_detections = 0
    for sample in viewer_dataset.iter_samples(progress=True, autosave=True):
        try:
            image_id = int(Path(sample.filepath).stem)
        except ValueError:
            sample[args.predictions_field] = fo.Detections(detections=[])
            continue

        detections = [
            fo.Detection(
                label=str(pred.get("label", pred.get("category_id", "unknown"))),
                confidence=float(pred.get("confidence", 0.0)),
                bounding_box=_normalize_bbox(pred.get("bounding_box", [])),
                category_id=int(pred["category_id"]) if "category_id" in pred else None,
            )
            for pred in predictions_by_image.get(int(image_id), [])
        ]
        sample[args.predictions_field] = fo.Detections(detections=detections)
        assigned_images += 1
        assigned_detections += len(detections)

    viewer_dataset.compute_metadata(overwrite=False)
    viewer_dataset.reload()

    return viewer_dataset, {
        "log_dir": str(log_dir),
        "summary": summary,
        "ground_truth_field": ground_truth_field,
        "predictions_field": args.predictions_field,
        "num_samples": len(viewer_dataset),
        "num_logged_images": len(image_ids),
        "assigned_images": assigned_images,
        "assigned_detections": assigned_detections,
    }


def main():
    args = parse_args()
    os.environ["FIFTYONE_APP_ADDRESS"] = args.host
    os.environ["FIFTYONE_APP_PORT"] = str(args.port)

    try:
        dataset, info = build_dataset(args)
    except Exception as exc:
        print(f"[coco_fiftyone_viewer] Failed to build viewer dataset: {exc}", file=sys.stderr)
        raise

    import fiftyone as fo

    session = fo.launch_app(dataset, address=args.host, port=args.port, auto=False)
    print(f"[coco_fiftyone_viewer] Viewer dataset: {dataset.name}")
    print(f"[coco_fiftyone_viewer] Log dir: {info['log_dir']}")
    print(
        f"[coco_fiftyone_viewer] Fields: GT='{info['ground_truth_field']}', "
        f"pred='{info['predictions_field']}'"
    )
    print(f"[coco_fiftyone_viewer] Samples in viewer: {info['num_samples']}")
    print(
        f"[coco_fiftyone_viewer] Loaded {info['assigned_detections']} predictions "
        f"across {info['assigned_images']} samples"
    )
    print(f"[coco_fiftyone_viewer] Listening on {args.host}:{args.port}")
    if args.host == "0.0.0.0":
        print(f"[coco_fiftyone_viewer] External URL: http://{_get_public_host()}:{args.port}")
    dataset_summary = info["summary"].get("dataset_summary", {})
    if dataset_summary:
        print(f"[coco_fiftyone_viewer] Dataset summary: {json.dumps(dataset_summary)}")
    print("[coco_fiftyone_viewer] Open the printed FiftyOne URL in your browser. Press Ctrl+C to stop.")

    def _handle_exit(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_exit)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        session.close()
        if not args.persist and fo.dataset_exists(dataset.name):
            fo.delete_dataset(dataset.name)


if __name__ == "__main__":
    main()

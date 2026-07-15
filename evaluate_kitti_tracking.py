# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate exported KITTI 2D tracks with official TrackEval metrics."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Sequence

from utils.kitti_trackeval import (
    EVALUATABLE_CLASSES,
    SUPPORTED_TRACKEVAL_COMMIT,
    build_metrics_report,
    discover_prediction_files,
    load_trackeval,
    normalize_classes,
    resolve_trackeval_root,
    run_trackeval,
    stage_trackeval_workspace,
    trackeval_revision,
    validate_evaluation_sequences,
    validate_tracker_name,
    write_json_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="KITTI Tracking root containing training/label_02 and image_02",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="One <sequence>.txt result file or a directory of result files",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Optional sequence subset when --predictions is a directory",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(EVALUATABLE_CLASSES),
        help="KITTI classes to evaluate: car pedestrian",
    )
    parser.add_argument(
        "--tracker-name",
        default="yolo_deepsort",
        help="Safe name used for TrackEval output folders",
    )
    parser.add_argument(
        "--trackeval-root",
        type=Path,
        help="Official TrackEval checkout (default: tools/TrackEval or TRACKEVAL_ROOT)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
        help="Root for official summaries, detailed CSV files, and metrics.json",
    )
    return parser


def _print_metrics_table(report: dict[str, Any]) -> None:
    print("\nCombined sequence metrics (percent except counts)")
    print(
        f"{'Class':<12} {'HOTA':>8} {'DetA':>8} {'AssA':>8} "
        f"{'MOTA':>8} {'IDF1':>8} {'FP':>8} {'FN':>8} {'IDSW':>8}"
    )
    for class_name, metrics in report["combined_sequences"].items():
        print(
            f"{class_name:<12} {metrics['HOTA']:>8.3f} {metrics['DetA']:>8.3f} "
            f"{metrics['AssA']:>8.3f} {metrics['MOTA']:>8.3f} "
            f"{metrics['IDF1']:>8.3f} {metrics['FP']:>8d} "
            f"{metrics['FN']:>8d} {metrics['IDSW']:>8d}"
        )


def run(args: argparse.Namespace) -> int:
    classes = normalize_classes(args.classes)
    tracker_name = validate_tracker_name(args.tracker_name)
    prediction_files = discover_prediction_files(args.predictions, args.sequences)
    sequences = validate_evaluation_sequences(
        args.dataset_root,
        prediction_files,
        classes,
    )
    trackeval_root = resolve_trackeval_root(
        args.trackeval_root,
        project_root=PROJECT_ROOT,
    )
    revision = trackeval_revision(trackeval_root)
    if revision is None:
        print(
            "WARNING: TrackEval Git revision could not be verified; use the "
            f"documented revision {SUPPORTED_TRACKEVAL_COMMIT} for reproducibility"
        )
    elif revision != SUPPORTED_TRACKEVAL_COMMIT:
        print(
            "WARNING: TrackEval checkout revision differs from the verified revision: "
            f"{revision} != {SUPPORTED_TRACKEVAL_COMMIT}"
        )

    print(
        f"Evaluating {len(sequences)} sequence(s) with TrackEval "
        f"({', '.join(item.sequence for item in sequences)})"
    )
    for item in sequences:
        print(
            f"  {item.sequence}: frames={item.frame_count}, "
            f"prediction_rows={item.prediction_rows}, "
            f"selected_class_rows={item.selected_class_rows}, "
            f"non_selected_class_rows={item.non_selected_class_rows}"
        )

    output_root = args.output_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="kitti_trackeval_") as temporary:
        workspace = stage_trackeval_workspace(temporary, sequences, tracker_name)
        trackeval_module = load_trackeval(trackeval_root)
        raw_results = run_trackeval(
            trackeval_module,
            workspace,
            output_root,
            tracker_name,
            classes,
        )

    report = build_metrics_report(
        raw_results,
        sequences,
        tracker_name,
        classes,
        trackeval_root=trackeval_root,
        revision=revision,
    )
    result_directory = output_root / tracker_name
    report_path = result_directory / "metrics.json"
    write_json_report(report_path, report)
    _print_metrics_table(report)
    print(f"\nCompact JSON report: {report_path}")
    print(f"Official TrackEval summaries and detailed CSV files: {result_directory}")
    print(
        "NOTE: These are development-subset metrics for the selected sequences, "
        "not KITTI test-server benchmark results."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

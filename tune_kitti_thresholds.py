# SPDX-License-Identifier: AGPL-3.0-only
"""Tune one KITTI class confidence without reading the locked holdout split."""

from __future__ import annotations

import argparse
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from utils.kitti_threshold_sweep import (
    KITTI_TO_MODEL_CLASS,
    choose_best_threshold,
    filter_prediction_file,
    load_sequence_split_preset,
    normalize_model_class,
    normalize_thresholds,
    paths_overlap,
    sha256_file,
    validate_export_manifest_if_present,
    validate_sequence_split,
    write_sweep_csv,
)
from utils.kitti_trackeval import (
    SUPPORTED_TRACKEVAL_COMMIT,
    compact_class_metrics,
    discover_prediction_files,
    load_trackeval,
    resolve_trackeval_root,
    run_trackeval,
    stage_trackeval_workspace,
    trackeval_revision,
    validate_evaluation_sequences,
    validate_tracker_name,
    write_json_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT_CONFIG = PROJECT_ROOT / "config" / "kitti_tracking_splits.json"
DEFAULT_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Directory containing source <sequence>.txt prediction files",
    )
    parser.add_argument(
        "--split-preset",
        choices=("smoke", "trackeval"),
        help=(
            "Load the short smoke or complete fixed split from "
            "config/kitti_tracking_splits.json"
        ),
    )
    parser.add_argument(
        "--tune-sequences",
        nargs="+",
        help="Explicit tuning sequences; requires --holdout-sequences",
    )
    parser.add_argument(
        "--holdout-sequences",
        nargs="+",
        help=(
            "Explicit locked sequences; recorded for the later rerun/evaluation "
            "but never read by this command; requires --tune-sequences"
        ),
    )
    parser.add_argument(
        "--class",
        dest="target_class",
        choices=tuple(KITTI_TO_MODEL_CLASS),
        default="pedestrian",
    )
    parser.add_argument(
        "--model-class",
        help=(
            "Target class name in the detector; defaults to COCO car/person. "
            "Use pedestrian for an 8-class KITTI-fine-tuned model."
        ),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Candidate score thresholds above the source-run confidence",
    )
    parser.add_argument(
        "--source-class-confidence",
        type=float,
        required=True,
        help=(
            "Confidence used for the target model class when source predictions "
            "were exported; this becomes the unchanged baseline"
        ),
    )
    parser.add_argument("--tracker-name", default="posthoc_threshold_sweep")
    parser.add_argument("--trackeval-root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/threshold_sweeps"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly reuse a nonempty experiment output directory",
    )
    return parser


def _evaluate_files(
    *,
    dataset_root: Path,
    prediction_directory: Path,
    sequences: Sequence[str],
    target_class: str,
    tracker_name: str,
    trackeval_module: Any,
    work_directory: Path,
    official_output_directory: Path,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    prediction_files = discover_prediction_files(prediction_directory, sequences)
    validated = validate_evaluation_sequences(
        dataset_root, prediction_files, (target_class,)
    )
    workspace = stage_trackeval_workspace(
        work_directory, validated, tracker_name
    )
    raw_results = run_trackeval(
        trackeval_module,
        workspace,
        official_output_directory,
        tracker_name,
        (target_class,),
    )
    metrics = compact_class_metrics(
        raw_results["COMBINED_SEQ"][target_class]
    )
    return metrics, validated


def _filter_sequence_set(
    source_files: dict[str, Path],
    sequences: Sequence[str],
    destination: Path,
    target_class: str,
    threshold: float,
) -> dict[str, int]:
    totals = {
        "all_rows_before": 0,
        "all_rows_after": 0,
        "target_rows_before": 0,
        "target_rows_retained": 0,
        "target_rows_removed": 0,
    }
    destination.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        counts = filter_prediction_file(
            source_files[sequence],
            destination / f"{sequence}.txt",
            target_class,
            threshold,
        )
        for key, value in counts.items():
            totals[key] += value
    return totals


def _print_sweep(candidate_results: Sequence[dict[str, Any]]) -> None:
    print("\nTune-only post-hoc threshold sweep")
    print(
        f"{'Threshold':>10} {'Kept':>8} {'HOTA':>8} {'DetA':>8} "
        f"{'AssA':>8} {'MOTA':>9} {'IDF1':>8} {'FP':>7} {'FN':>7}"
    )
    for result in candidate_results:
        metrics = result["metrics"]
        print(
            f"{result['threshold']:>10.3f} "
            f"{result['row_counts']['target_rows_retained']:>8d} "
            f"{metrics['HOTA']:>8.3f} {metrics['DetA']:>8.3f} "
            f"{metrics['AssA']:>8.3f} {metrics['MOTA']:>9.3f} "
            f"{metrics['IDF1']:>8.3f} {metrics['FP']:>7d} {metrics['FN']:>7d}"
        )


def run(args: argparse.Namespace) -> int:
    if args.split_preset is not None:
        if args.tune_sequences is not None or args.holdout_sequences is not None:
            raise ValueError(
                "--split-preset cannot be combined with explicit sequence sets"
            )
        tune_sequences, holdout_sequences = load_sequence_split_preset(
            DEFAULT_SPLIT_CONFIG, args.split_preset
        )
        split_preset = args.split_preset
    else:
        if args.tune_sequences is None or args.holdout_sequences is None:
            raise ValueError(
                "provide --split-preset or both --tune-sequences and "
                "--holdout-sequences"
            )
        tune_sequences, holdout_sequences = validate_sequence_split(
            args.tune_sequences, args.holdout_sequences
        )
        split_preset = None
    target_class = args.target_class.casefold()
    model_class = normalize_model_class(target_class, args.model_class)
    thresholds = normalize_thresholds(
        args.thresholds, source_threshold=args.source_class_confidence
    )
    tracker_name = validate_tracker_name(args.tracker_name)
    output_directory = args.output_dir.resolve() / target_class
    if paths_overlap(args.predictions, output_directory):
        raise ValueError(
            "--predictions and the class-specific output directory must not overlap"
        )
    if (
        output_directory.exists()
        and any(output_directory.iterdir())
        and not args.overwrite
    ):
        raise ValueError(
            f"experiment output is not empty: {output_directory}. Use a new "
            "--output-dir, or pass --overwrite to acknowledge reuse."
        )
    # Deliberately discover, hash, and validate tune predictions only. The locked
    # holdout files need not exist until the selected settings have been rerun.
    source_files = discover_prediction_files(args.predictions, tune_sequences)
    source_experiment = validate_export_manifest_if_present(
        args.predictions,
        source_files,
        tune_sequences,
        target_class=target_class,
        model_class=model_class,
        declared_source_confidence=float(args.source_class_confidence),
    )
    if not source_experiment["present"]:
        print(f"WARNING: {source_experiment['warning']}")
    else:
        for warning in source_experiment.get("warnings", ()):
            print(f"WARNING: {warning}")
    validate_evaluation_sequences(
        args.dataset_root, source_files, (target_class,)
    )

    trackeval_root = resolve_trackeval_root(
        args.trackeval_root, project_root=PROJECT_ROOT
    )
    revision = trackeval_revision(trackeval_root)
    if revision != SUPPORTED_TRACKEVAL_COMMIT:
        print(
            "WARNING: TrackEval revision is not the verified revision: "
            f"{revision!r} != {SUPPORTED_TRACKEVAL_COMMIT}"
        )
    trackeval_module = load_trackeval(trackeval_root)
    output_directory.mkdir(parents=True, exist_ok=True)

    candidate_results: list[dict[str, Any]] = []
    selected_predictions_directory = output_directory / "selected_tune_predictions"
    with tempfile.TemporaryDirectory(prefix="kitti_threshold_sweep_") as temporary:
        temporary_root = Path(temporary)
        for index, threshold in enumerate(thresholds):
            filtered_directory = temporary_root / "filtered" / f"candidate_{index:02d}"
            row_counts = _filter_sequence_set(
                source_files,
                tune_sequences,
                filtered_directory,
                target_class,
                threshold,
            )
            if (
                threshold == float(args.source_class_confidence)
                and row_counts["target_rows_removed"] != 0
            ):
                raise ValueError(
                    "source prediction rows exist below the declared "
                    f"--source-class-confidence {args.source_class_confidence:g}; "
                    "provide the actual source-run confidence"
                )
            metrics, _ = _evaluate_files(
                dataset_root=args.dataset_root,
                prediction_directory=filtered_directory,
                sequences=tune_sequences,
                target_class=target_class,
                tracker_name=f"{tracker_name}_{index:02d}",
                trackeval_module=trackeval_module,
                work_directory=temporary_root / "work" / f"candidate_{index:02d}",
                official_output_directory=temporary_root / "candidate_outputs",
            )
            candidate_results.append(
                {
                    "threshold": threshold,
                    "row_counts": row_counts,
                    "metrics": metrics,
                }
            )

        selected = choose_best_threshold(candidate_results)
        selected_threshold = float(selected["threshold"])
        tune_counts = _filter_sequence_set(
            source_files,
            tune_sequences,
            selected_predictions_directory,
            target_class,
            selected_threshold,
        )

    report = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "tune-only post-hoc KITTI prediction confidence sweep",
        "warning": (
            "Post-hoc filtering does not reproduce Deep SORT state changes. "
            "Rerun YOLO/Deep SORT with the selected model-class threshold before "
            "using it as a final result."
        ),
        "selection_protocol": {
            "split_preset": split_preset,
            "target_class": target_class,
            "model_class": model_class,
            "source_class_confidence": float(args.source_class_confidence),
            "objective": "highest tune HOTA, then IDF1, then MOTA, then lower threshold",
            "tune_sequences": list(tune_sequences),
            "holdout_sequences": list(holdout_sequences),
            "holdout_used_for_selection": False,
            "holdout_predictions_read": False,
            "holdout_ground_truth_read": False,
        },
        "source_experiment_manifest": source_experiment,
        "source_files": {
            sequence: {
                "path": str(source_files[sequence]),
                "sha256": sha256_file(source_files[sequence]),
            }
            for sequence in tune_sequences
        },
        "trackeval": {
            "revision": revision,
            "supported_revision": SUPPORTED_TRACKEVAL_COMMIT,
        },
        "candidates": candidate_results,
        "selected_threshold": selected_threshold,
        "selected_tune_metrics": selected["metrics"],
        "selected_tune_row_counts": tune_counts,
        "selected_tune_predictions": {
            "directory": str(selected_predictions_directory),
            "sequences": list(tune_sequences),
            "purpose": "diagnostic post-hoc tune artifacts; not final tracker output",
        },
        "locked_holdout_evaluation": {
            "status": "not_run",
            "sequences": list(holdout_sequences),
            "required_precondition": (
                "rerun YOLO and Deep SORT with the selected class threshold into "
                "a fresh prediction directory"
            ),
        },
        "rerun_argument": (
            f"--class-confidence {model_class}={selected_threshold:g}"
        ),
    }
    report_path = output_directory / "sweep.json"
    write_json_report(report_path, report)
    write_sweep_csv(output_directory / "sweep.csv", candidate_results)
    _print_sweep(candidate_results)
    print(f"\nSelected tune-only threshold: {selected_threshold:g}")
    print(
        "Required full rerun argument: "
        f"--class-confidence {model_class}={selected_threshold:g}"
    )
    print(
        "Locked holdout was not read. Rerun YOLO/Deep SORT first, then evaluate "
        "the fresh locked predictions separately."
    )
    print(f"Sweep report: {report_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

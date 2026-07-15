# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare and run official TrackEval KITTI 2D tracking evaluation.

TrackEval's KITTI adapter consumes native KITTI ``label_02`` files, but it
expects a small benchmark-specific directory layout and sequence-map file.
This module builds that layout in a temporary directory, validates tracker
results before evaluation, and converts TrackEval's nested NumPy results into
a compact JSON-friendly report.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from utils.kitti_tracking_dataset import KittiTrackingDataset, normalize_sequence_id
from utils.kitti_tracking_labels import KittiTrackingLabel, load_kitti_tracking_labels
from utils.kitti_tracking_results import (
    KittiTrackingPrediction,
    write_kitti_tracking_results,
)


SUPPORTED_TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
EVALUATABLE_CLASSES = ("car", "pedestrian")
_TRACKER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_WINDOWS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


@dataclass(frozen=True, slots=True)
class EvaluationSequence:
    """One validated sequence ready to be staged for TrackEval."""

    sequence: str
    frame_count: int
    ground_truth_path: Path
    prediction_path: Path
    prediction_rows: int
    selected_class_rows: int
    non_selected_class_rows: int


@dataclass(frozen=True, slots=True)
class TrackEvalWorkspace:
    """Paths in the temporary directory consumed by TrackEval."""

    ground_truth_root: Path
    trackers_root: Path
    sequence_map_path: Path


def validate_tracker_name(name: str) -> str:
    """Return a safe TrackEval folder name or raise a useful error."""

    if not isinstance(name, str) or not _TRACKER_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "tracker name must start with an alphanumeric character and contain "
            "only letters, digits, dots, underscores, or hyphens"
        )
    if name.endswith("."):
        raise ValueError("tracker name cannot end with a dot on Windows")
    windows_basename = name.split(".", 1)[0].upper()
    if windows_basename in _WINDOWS_RESERVED_BASENAMES:
        raise ValueError(f"tracker name uses the reserved Windows name {windows_basename}")
    return name


def normalize_classes(classes: Sequence[str]) -> tuple[str, ...]:
    """Validate and de-duplicate TrackEval KITTI class names."""

    normalized: list[str] = []
    for value in classes:
        class_name = str(value).strip().casefold()
        if class_name not in EVALUATABLE_CLASSES:
            raise ValueError(
                f"unsupported KITTI evaluation class {value!r}; "
                f"choose from {', '.join(EVALUATABLE_CLASSES)}"
            )
        if class_name not in normalized:
            normalized.append(class_name)
    if not normalized:
        raise ValueError("at least one evaluation class is required")
    return tuple(normalized)


def discover_prediction_files(
    predictions: str | Path,
    requested_sequences: Sequence[str | int] | None = None,
) -> dict[str, Path]:
    """Resolve a single result file or a directory of ``<sequence>.txt`` files."""

    prediction_path = Path(predictions)
    requested = (
        tuple(normalize_sequence_id(sequence) for sequence in requested_sequences)
        if requested_sequences
        else ()
    )
    if len(set(requested)) != len(requested):
        raise ValueError("--sequences contains duplicate sequence IDs")

    if prediction_path.is_file():
        if prediction_path.suffix.casefold() != ".txt":
            raise ValueError(f"prediction file must use a .txt extension: {prediction_path}")
        inferred = normalize_sequence_id(prediction_path.stem)
        if len(requested) > 1:
            raise ValueError("one prediction file can evaluate only one sequence")
        if requested and requested[0] != inferred:
            raise ValueError(
                f"prediction filename implies sequence {inferred}, but "
                f"--sequences requested {requested[0]}"
            )
        return {inferred: prediction_path.resolve()}

    if not prediction_path.is_dir():
        raise FileNotFoundError(
            f"predictions path is neither a result file nor a directory: {prediction_path}"
        )

    if requested:
        resolved: dict[str, Path] = {}
        for sequence in requested:
            candidate = prediction_path / f"{sequence}.txt"
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"prediction file for sequence {sequence} does not exist: {candidate}"
                )
            resolved[sequence] = candidate.resolve()
        return resolved

    resolved = {}
    for candidate in sorted(prediction_path.glob("*.txt")):
        try:
            sequence = normalize_sequence_id(candidate.stem)
        except ValueError:
            continue
        if sequence in resolved:
            raise ValueError(
                f"multiple prediction files normalize to sequence {sequence}: "
                f"{resolved[sequence]} and {candidate}"
            )
        resolved[sequence] = candidate.resolve()
    if not resolved:
        raise FileNotFoundError(
            f"no numeric KITTI sequence .txt files found in: {prediction_path}"
        )
    return resolved


def _validate_prediction_rows(
    labels: Sequence[KittiTrackingLabel],
    *,
    prediction_path: Path,
    frame_count: int,
    classes: Sequence[str],
) -> tuple[int, int]:
    seen: set[tuple[int, int]] = set()
    evaluated = 0
    ignored = 0
    for label in labels:
        if label.frame >= frame_count:
            raise ValueError(
                f"{prediction_path}: frame {label.frame} is outside the sequence "
                f"range 0..{frame_count - 1}"
            )
        key = label.frame, label.track_id
        if key in seen:
            raise ValueError(
                f"{prediction_path}: track ID {label.track_id} occurs more than once "
                f"in frame {label.frame}"
            )
        seen.add(key)

        if label.bbox_right <= label.bbox_left or label.bbox_bottom <= label.bbox_top:
            raise ValueError(
                f"{prediction_path}: prediction for track ID {label.track_id} in "
                f"frame {label.frame} must have a positive-area bounding box"
            )

        if label.score is None:
            raise ValueError(
                f"{prediction_path}: prediction rows must contain the eighteenth "
                "confidence score field"
            )
        if label.is_dont_care:
            raise ValueError(f"{prediction_path}: predictions cannot contain DontCare rows")

        object_type = label.type.casefold()
        if object_type not in EVALUATABLE_CLASSES:
            raise ValueError(
                f"{prediction_path}: unsupported prediction type {label.type!r}; "
                "KITTI 2D evaluation accepts only Car and Pedestrian predictions"
            )
        if object_type in classes:
            evaluated += 1
        else:
            ignored += 1
    return evaluated, ignored


def validate_evaluation_sequences(
    dataset_root: str | Path,
    prediction_files: Mapping[str, Path],
    classes: Sequence[str],
) -> tuple[EvaluationSequence, ...]:
    """Validate dataset synchronization and every prediction row."""

    normalized_classes = normalize_classes(classes)
    sequences: list[EvaluationSequence] = []
    for sequence, prediction_path in sorted(prediction_files.items()):
        dataset = KittiTrackingDataset(
            dataset_root,
            sequence,
            split="training",
            require_pointcloud=False,
            require_calibration=False,
            load_labels=True,
            require_labels=True,
            strict=True,
        )
        predictions = load_kitti_tracking_labels(prediction_path)
        evaluated, ignored = _validate_prediction_rows(
            predictions,
            prediction_path=prediction_path,
            frame_count=len(dataset),
            classes=normalized_classes,
        )
        sequences.append(
            EvaluationSequence(
                sequence=dataset.sequence,
                frame_count=len(dataset),
                ground_truth_path=dataset.label_path.resolve(),
                prediction_path=prediction_path.resolve(),
                prediction_rows=len(predictions),
                selected_class_rows=evaluated,
                non_selected_class_rows=ignored,
            )
        )
    if not sequences:
        raise ValueError("no prediction sequences were selected")
    return tuple(sequences)


def stage_trackeval_workspace(
    root: str | Path,
    sequences: Sequence[EvaluationSequence],
    tracker_name: str,
) -> TrackEvalWorkspace:
    """Create TrackEval's KITTI input layout without altering the source dataset."""

    tracker_name = validate_tracker_name(tracker_name)
    if not sequences:
        raise ValueError("at least one sequence is required for TrackEval staging")
    sequence_ids = [item.sequence for item in sequences]
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("TrackEval staging received duplicate sequence IDs")
    workspace_root = Path(root)
    ground_truth_root = workspace_root / "gt"
    ground_truth_labels = ground_truth_root / "label_02"
    tracker_data = workspace_root / "trackers" / tracker_name / "data"
    ground_truth_labels.mkdir(parents=True, exist_ok=True)
    tracker_data.mkdir(parents=True, exist_ok=True)

    sequence_map_lines: list[str] = []
    for item in sequences:
        shutil.copyfile(
            item.ground_truth_path,
            ground_truth_labels / f"{item.sequence}.txt",
        )
        source_predictions = load_kitti_tracking_labels(item.prediction_path)
        # TrackEval relabels IDs by allocating up to max(track_id) + 1. Dense
        # sequence-local IDs prevent excessive allocation and avoid float64
        # precision loss for valid but very large external tracker IDs.
        dense_ids = {
            track_id: dense_id
            for dense_id, track_id in enumerate(source_predictions.track_ids)
        }
        staged_predictions = (
            KittiTrackingPrediction(
                frame=prediction.frame,
                track_id=dense_ids[prediction.track_id],
                object_type=prediction.type,
                bbox_xyxy=prediction.bbox,
                score=float(prediction.score),
            )
            for prediction in source_predictions
        )
        write_kitti_tracking_results(
            tracker_data / f"{item.sequence}.txt", staged_predictions
        )
        sequence_map_lines.append(
            f"{item.sequence} empty 000000 {item.frame_count:06d}"
        )

    sequence_map_path = ground_truth_root / "evaluate_tracking.seqmap.training"
    sequence_map_path.write_text(
        "\n".join(sequence_map_lines) + "\n", encoding="utf-8", newline="\n"
    )
    return TrackEvalWorkspace(
        ground_truth_root=ground_truth_root,
        trackers_root=workspace_root / "trackers",
        sequence_map_path=sequence_map_path,
    )


def resolve_trackeval_root(
    explicit_root: str | Path | None,
    *,
    project_root: str | Path,
) -> Path:
    """Locate an official TrackEval source checkout."""

    if explicit_root is not None:
        resolved = Path(explicit_root).expanduser().resolve()
        if (resolved / "trackeval" / "__init__.py").is_file():
            return resolved
        raise FileNotFoundError(f"invalid --trackeval-root checkout: {resolved}")

    candidates: list[Path] = []
    environment_root = os.environ.get("TRACKEVAL_ROOT")
    if environment_root:
        candidates.append(Path(environment_root))
    candidates.append(Path(project_root) / "tools" / "TrackEval")

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "trackeval" / "__init__.py").is_file():
            return resolved
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "TrackEval source checkout was not found. Clone the official repository "
        f"at commit {SUPPORTED_TRACKEVAL_COMMIT} into tools/TrackEval or pass "
        f"--trackeval-root. Searched: {searched}"
    )


def trackeval_revision(trackeval_root: str | Path) -> str | None:
    """Return the checkout commit when Git metadata is available."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(trackeval_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def load_trackeval(trackeval_root: str | Path) -> Any:
    """Import TrackEval with a narrow compatibility bridge for NumPy 2.x."""

    root = Path(trackeval_root).resolve()
    if not (root / "trackeval" / "__init__.py").is_file():
        raise FileNotFoundError(f"invalid TrackEval source checkout: {root}")

    # Official TrackEval commit 12c8791 still uses aliases removed in NumPy
    # 1.24. Add them only in this evaluation process instead of downgrading the
    # main computer-vision environment or modifying third-party source.
    if "float" not in np.__dict__:
        setattr(np, "float", float)
    if "int" not in np.__dict__:
        setattr(np, "int", int)
    if "bool" not in np.__dict__:
        setattr(np, "bool", bool)

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    already_loaded = sys.modules.get("trackeval")
    if already_loaded is not None:
        loaded_path = Path(already_loaded.__file__).resolve()
        if root not in loaded_path.parents:
            raise RuntimeError(
                f"trackeval was already imported from {loaded_path}, not {root}"
            )
        return already_loaded

    # TrackEval imports optional dataset adapters eagerly and prints a warning
    # when pycocotools is absent. KITTI 2D does not need that dependency.
    with contextlib.redirect_stdout(io.StringIO()):
        module = importlib.import_module("trackeval")
    return module


def clear_previous_evaluation_outputs(output_directory: str | Path) -> None:
    """Remove only known generated files so a narrower rerun cannot look stale."""

    result_directory = Path(output_directory)
    generated_names = {"metrics.json", "trackeval_error.log"}
    for class_name in EVALUATABLE_CLASSES:
        generated_names.update(
            {
                f"{class_name}_summary.txt",
                f"{class_name}_detailed.csv",
                f"{class_name}_plot.pdf",
                f"{class_name}_plot.png",
            }
        )
    for name in generated_names:
        (result_directory / name).unlink(missing_ok=True)


def run_trackeval(
    trackeval_module: Any,
    workspace: TrackEvalWorkspace,
    output_root: str | Path,
    tracker_name: str,
    classes: Sequence[str],
) -> Mapping[str, Any]:
    """Run HOTA, CLEAR, and Identity metrics through TrackEval's Python API."""

    tracker_name = validate_tracker_name(tracker_name)
    normalized_classes = normalize_classes(classes)
    output_path = Path(output_root).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    result_directory = output_path / tracker_name
    result_directory.mkdir(parents=True, exist_ok=True)
    clear_previous_evaluation_outputs(result_directory)

    eval_config = {
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 1,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
        "LOG_ON_ERROR": str(output_path / tracker_name / "trackeval_error.log"),
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_EMPTY_CLASSES": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
    }
    dataset_config = {
        "GT_FOLDER": str(workspace.ground_truth_root),
        "TRACKERS_FOLDER": str(workspace.trackers_root),
        "OUTPUT_FOLDER": str(output_path),
        "TRACKERS_TO_EVAL": [tracker_name],
        "CLASSES_TO_EVAL": list(normalized_classes),
        "SPLIT_TO_EVAL": "training",
        "INPUT_AS_ZIP": False,
        "PRINT_CONFIG": False,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "TRACKER_DISPLAY_NAMES": [tracker_name],
    }

    try:
        evaluator = trackeval_module.Evaluator(eval_config)
        dataset = trackeval_module.datasets.Kitti2DBox(dataset_config)
        metrics = [
            trackeval_module.metrics.HOTA(),
            trackeval_module.metrics.CLEAR({"PRINT_CONFIG": False}),
            trackeval_module.metrics.Identity({"PRINT_CONFIG": False}),
        ]
        output_results, output_messages = evaluator.evaluate([dataset], metrics)
    except Exception as exc:
        raise RuntimeError(f"TrackEval evaluation failed: {exc}") from exc
    message = output_messages.get("Kitti2DBox", {}).get(tracker_name)
    if message != "Success":
        raise RuntimeError(f"TrackEval did not complete successfully: {message}")
    return output_results["Kitti2DBox"][tracker_name]


def _percentage(value: Any) -> float:
    numeric = float(np.mean(value)) if isinstance(value, np.ndarray) else float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"TrackEval returned a non-finite metric: {numeric}")
    return round(100.0 * numeric, 6)


def _integer(value: Any) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"TrackEval returned a non-integral count: {value!r}")
    return int(numeric)


def compact_class_metrics(result: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Extract the most useful class metrics with scores expressed as percentages."""

    hota = result["HOTA"]
    clear = result["CLEAR"]
    identity = result["Identity"]
    count = result["Count"]
    return {
        "HOTA": _percentage(hota["HOTA"]),
        "DetA": _percentage(hota["DetA"]),
        "AssA": _percentage(hota["AssA"]),
        "LocA": _percentage(hota["LocA"]),
        "DetRe": _percentage(hota["DetRe"]),
        "DetPr": _percentage(hota["DetPr"]),
        "AssRe": _percentage(hota["AssRe"]),
        "AssPr": _percentage(hota["AssPr"]),
        "MOTA": _percentage(clear["MOTA"]),
        "MOTP": _percentage(clear["MOTP"]),
        "Recall": _percentage(clear["CLR_Re"]),
        "Precision": _percentage(clear["CLR_Pr"]),
        "IDF1": _percentage(identity["IDF1"]),
        "IDR": _percentage(identity["IDR"]),
        "IDP": _percentage(identity["IDP"]),
        "TP": _integer(clear["CLR_TP"]),
        "FP": _integer(clear["CLR_FP"]),
        "FN": _integer(clear["CLR_FN"]),
        "IDSW": _integer(clear["IDSW"]),
        "Frag": _integer(clear["Frag"]),
        "MostlyTracked": _integer(clear["MT"]),
        "PartlyTracked": _integer(clear["PT"]),
        "MostlyLost": _integer(clear["ML"]),
        "PredictionDetections": _integer(count["Dets"]),
        "GroundTruthDetections": _integer(count["GT_Dets"]),
        "PredictionTracks": _integer(count["IDs"]),
        "GroundTruthTracks": _integer(count["GT_IDs"]),
    }


def build_metrics_report(
    raw_results: Mapping[str, Any],
    sequences: Sequence[EvaluationSequence],
    tracker_name: str,
    classes: Sequence[str],
    *,
    trackeval_root: str | Path,
    revision: str | None,
) -> dict[str, Any]:
    """Build a stable, JSON-friendly report from TrackEval API results."""

    normalized_classes = normalize_classes(classes)
    combined = raw_results["COMBINED_SEQ"]
    sequence_metrics: dict[str, dict[str, Any]] = {}
    for item in sequences:
        sequence_result = raw_results[item.sequence]
        sequence_metrics[item.sequence] = {
            class_name: compact_class_metrics(sequence_result[class_name])
            for class_name in normalized_classes
        }

    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": "KITTI Tracking 2D development subset",
        "scores_unit": "percent",
        "tracker": tracker_name,
        "classes": list(normalized_classes),
        "sequences": [
            {
                "sequence": item.sequence,
                "frames": item.frame_count,
                "prediction_rows": item.prediction_rows,
                "selected_class_rows": item.selected_class_rows,
                "non_selected_class_rows": item.non_selected_class_rows,
            }
            for item in sequences
        ],
        "trackeval": {
            "root": str(Path(trackeval_root).resolve()),
            "revision": revision,
            "supported_revision": SUPPORTED_TRACKEVAL_COMMIT,
        },
        "combined_sequences": {
            class_name: compact_class_metrics(combined[class_name])
            for class_name in normalized_classes
        },
        "per_sequence": sequence_metrics,
    }


def write_json_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """Atomically write the compact metrics report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "EVALUATABLE_CLASSES",
    "SUPPORTED_TRACKEVAL_COMMIT",
    "EvaluationSequence",
    "TrackEvalWorkspace",
    "build_metrics_report",
    "clear_previous_evaluation_outputs",
    "compact_class_metrics",
    "discover_prediction_files",
    "load_trackeval",
    "normalize_classes",
    "resolve_trackeval_root",
    "run_trackeval",
    "stage_trackeval_workspace",
    "trackeval_revision",
    "validate_evaluation_sequences",
    "validate_tracker_name",
    "write_json_report",
]

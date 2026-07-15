# SPDX-License-Identifier: AGPL-3.0-only
"""Helpers for leakage-safe post-hoc confidence threshold sweeps."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from utils.detection_thresholds import (
    confidence_for_class,
    model_inference_threshold,
)
from utils.kitti_tracking_dataset import normalize_sequence_id
from utils.kitti_tracking_labels import parse_kitti_tracking_label_line


KITTI_TO_MODEL_CLASS = {"car": "car", "pedestrian": "person"}
COMPATIBLE_MODEL_CLASSES = {
    "car": frozenset(("car",)),
    "pedestrian": frozenset(("person", "pedestrian")),
}
EXPORT_MANIFEST_NAME = "experiment_manifest.json"
SPLIT_PRESET_KEYS = {
    "trackeval": ("trackeval_training_minus_val", "trackeval_val"),
    "smoke": ("initial_smoke_tune", "initial_smoke_val"),
}


def normalize_model_class(target_class: str, model_class: str | None) -> str:
    """Validate the detector class used to produce one KITTI result class."""

    target = target_class.strip().casefold()
    if target not in KITTI_TO_MODEL_CLASS:
        raise ValueError("target class must be 'car' or 'pedestrian'")
    if model_class is None:
        normalized = KITTI_TO_MODEL_CLASS[target]
    else:
        normalized = model_class.strip().casefold()
        if not normalized:
            raise ValueError("--model-class must not be empty")
    allowed = COMPATIBLE_MODEL_CLASSES[target]
    if normalized not in allowed:
        choices = " or ".join(repr(value) for value in sorted(allowed))
        raise ValueError(
            f"--model-class {normalized!r} cannot produce KITTI {target.title()}; "
            f"use {choices}"
        )
    return normalized


def _manifest_confidence(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"export manifest {field} must be numeric, not boolean")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"export manifest {field} must be numeric") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"export manifest {field} must be between 0 and 1")
    return threshold


def validate_export_manifest_if_present(
    prediction_directory: str | Path,
    prediction_files: Mapping[str, Path],
    tune_sequences: Sequence[str],
    *,
    target_class: str,
    model_class: str,
    declared_source_confidence: float,
) -> dict[str, Any]:
    """Validate source prediction provenance when the fixed-split manifest exists.

    Only tune-sequence files are hashed. The returned threshold metadata keeps the
    detector's upstream inference floor separate from the target class's effective
    post-detection threshold.
    """

    directory = Path(prediction_directory).expanduser().resolve()
    manifest_path = directory / EXPORT_MANIFEST_NAME
    if not manifest_path.is_file():
        return {
            "present": False,
            "expected_path": str(manifest_path),
            "warning": (
                "No experiment_manifest.json was found. Source model, YOLO head, "
                "image size, tracker configuration, and prediction hashes could "
                "not be authenticated automatically."
            ),
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid export experiment manifest: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(
            f"export experiment manifest must contain a JSON object: {manifest_path}"
        )
    schema_version = manifest.get("schema_version")
    if schema_version not in {1, 2, 3}:
        raise ValueError(
            "unsupported export experiment manifest schema: "
            f"{schema_version!r}"
        )
    provenance_warnings: list[str] = []
    if schema_version == 1:
        provenance_warnings.append(
            "Legacy export manifest schema 1 predates exact-class Deep SORT "
            "association. Treat these predictions as non-final and regenerate "
            "them with the current schema before final reporting."
        )
    elif schema_version == 2:
        provenance_warnings.append(
            "Legacy export manifest schema 2 tracked detector classes that KITTI "
            "does not evaluate. It remains usable as a source screen, but regenerate "
            "with the current schema before final reporting."
        )

    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("export experiment manifest configuration must be an object")
    required_configuration_fields = {
        "dataset_root",
        "output_dir",
        "sequences",
        "model",
        "confidence",
        "class_confidence",
        "imgsz",
        "yolo_end2end",
        "device",
        "max_age",
        "n_init",
        "max_cosine_distance",
        "nn_budget",
        "embedder",
        "half",
        "embedder_gpu",
    }
    missing_configuration_fields = sorted(
        required_configuration_fields - set(configuration)
    )
    if missing_configuration_fields:
        raise ValueError(
            "export experiment manifest is missing configuration fields: "
            + ", ".join(missing_configuration_fields)
        )
    if schema_version >= 2:
        association_policy = configuration.get("association_policy")
        if association_policy != "deepsort-class-exact-match-v1":
            raise ValueError(
                "export experiment manifest has an unsupported association policy: "
                f"{association_policy!r}"
            )
    if schema_version >= 3:
        tracked_class_policy = configuration.get("tracked_class_policy")
        if tracked_class_policy != "kitti-evaluated-classes-v1":
            raise ValueError(
                "export experiment manifest has an unsupported tracked-class policy: "
                f"{tracked_class_policy!r}"
            )
        embedder_batch_size = configuration.get("embedder_batch_size")
        if (
            isinstance(embedder_batch_size, bool)
            or not isinstance(embedder_batch_size, int)
            or embedder_batch_size < 1
        ):
            raise ValueError(
                "export experiment manifest configuration.embedder_batch_size "
                "must be positive"
            )
    model = configuration.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("export experiment manifest configuration.model must be nonempty")
    imgsz = configuration.get("imgsz")
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
        raise ValueError("export experiment manifest configuration.imgsz must be positive")
    yolo_end2end = configuration.get("yolo_end2end")
    if yolo_end2end is not None and not isinstance(yolo_end2end, bool):
        raise ValueError(
            "export experiment manifest configuration.yolo_end2end must be true, "
            "false, or null"
        )

    global_confidence = _manifest_confidence(
        configuration.get("confidence"), "configuration.confidence"
    )
    raw_overrides = configuration.get("class_confidence")
    if not isinstance(raw_overrides, dict):
        raise ValueError(
            "export experiment manifest configuration.class_confidence must be an object"
        )
    overrides: dict[str, float] = {}
    for raw_name, raw_value in raw_overrides.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("export manifest class-confidence names must be nonempty")
        name = raw_name.strip().casefold()
        if name in overrides:
            raise ValueError(
                f"duplicate normalized class-confidence name in export manifest: {name!r}"
            )
        overrides[name] = _manifest_confidence(
            raw_value, f"configuration.class_confidence.{name}"
        )

    normalized_target = target_class.strip().casefold()
    normalized_model_class = normalize_model_class(normalized_target, model_class)
    alternate_target_names = (
        COMPATIBLE_MODEL_CLASSES[normalized_target] - {normalized_model_class}
    )
    conflicting_overrides = sorted(alternate_target_names & set(overrides))
    if conflicting_overrides:
        raise ValueError(
            f"--model-class {normalized_model_class!r} conflicts with export manifest "
            f"class override(s) for {', '.join(conflicting_overrides)}; use the exact "
            "detector class name used by the export"
        )

    effective_target_confidence = confidence_for_class(
        global_confidence, overrides, normalized_model_class
    )
    declared = _manifest_confidence(
        declared_source_confidence, "--source-class-confidence"
    )
    if not math.isclose(declared, effective_target_confidence, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"--source-class-confidence {declared:g} does not match export manifest "
            f"effective confidence {effective_target_confidence:g} for model class "
            f"{normalized_model_class!r} (global={global_confidence:g}, "
            f"class override={overrides.get(normalized_model_class)!r})"
        )

    raw_sequences = configuration.get("sequences")
    if not isinstance(raw_sequences, list):
        raise ValueError("export experiment manifest configuration.sequences must be a list")
    manifest_sequences = tuple(normalize_sequence_id(value) for value in raw_sequences)
    if len(set(manifest_sequences)) != len(manifest_sequences):
        raise ValueError("export experiment manifest contains duplicate sequences")
    missing_sequences = sorted(set(tune_sequences) - set(manifest_sequences))
    if missing_sequences:
        raise ValueError(
            "tune sequences are absent from export experiment manifest: "
            + ", ".join(missing_sequences)
        )

    model_provenance = manifest.get("model")
    if not isinstance(model_provenance, dict):
        raise ValueError("export experiment manifest model provenance must be an object")
    if model_provenance.get("argument") != model:
        raise ValueError(
            "export experiment manifest model provenance conflicts with configuration.model"
        )

    sequence_results = manifest.get("sequence_results")
    if not isinstance(sequence_results, dict):
        raise ValueError("export experiment manifest sequence_results must be an object")
    validated_hashes: dict[str, str] = {}
    for sequence in tune_sequences:
        entry = sequence_results.get(sequence)
        if not isinstance(entry, dict) or entry.get("status") != "completed":
            raise ValueError(
                f"export experiment manifest does not mark tune sequence {sequence} completed"
            )
        expected_hash = entry.get("output_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(
                f"export experiment manifest has no valid output hash for {sequence}"
            )
        source_file = prediction_files.get(sequence)
        if source_file is None:
            raise ValueError(f"prediction mapping is missing tune sequence {sequence}")
        actual_hash = sha256_file(source_file)
        if actual_hash.casefold() != expected_hash.casefold():
            raise ValueError(
                f"prediction file hash does not match export manifest for {sequence}"
            )
        validated_hashes[sequence] = actual_hash

    return {
        "present": True,
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "schema_version": manifest["schema_version"],
        "status": manifest.get("status"),
        "model": model_provenance,
        "configuration": configuration,
        "warnings": provenance_warnings,
        "thresholds": {
            "global_confidence": global_confidence,
            "class_confidence_overrides": dict(sorted(overrides.items())),
            "upstream_yolo_inference_floor": model_inference_threshold(
                global_confidence, overrides
            ),
            "target_model_class": normalized_model_class,
            "target_class_effective_confidence": effective_target_confidence,
        },
        "validated_tune_prediction_hashes": validated_hashes,
    }


def normalize_thresholds(
    values: Sequence[float], *, source_threshold: float
) -> tuple[float, ...]:
    """Validate candidates and include the observable source-run baseline."""

    if isinstance(source_threshold, bool):
        raise ValueError("source confidence threshold must be numeric, not boolean")
    source_floor = float(source_threshold)
    if not math.isfinite(source_floor) or not 0.0 <= source_floor <= 1.0:
        raise ValueError("source confidence threshold must be between 0 and 1")
    normalized = {source_floor}
    for value in values:
        if isinstance(value, bool):
            raise ValueError("confidence thresholds must be numeric, not boolean")
        threshold = float(value)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence thresholds must be finite values from 0 to 1")
        if threshold < source_floor:
            raise ValueError(
                f"candidate threshold {threshold:g} is below the source confidence "
                f"{source_floor:g}; detections below the source floor were never exported"
            )
        normalized.add(threshold)
    if len(normalized) == 1:
        raise ValueError(
            "at least one threshold above the source confidence is required"
        )
    return tuple(sorted(normalized))


def paths_overlap(first: str | Path, second: str | Path) -> bool:
    """Whether either resolved path contains the other (including equality)."""

    left = Path(first).resolve()
    right = Path(second).resolve()
    return left == right or left in right.parents or right in left.parents


def validate_sequence_split(
    tune_sequences: Sequence[str | int],
    holdout_sequences: Sequence[str | int],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Normalize two nonempty, duplicate-free, disjoint sequence sets."""

    tune = tuple(normalize_sequence_id(sequence) for sequence in tune_sequences)
    holdout = tuple(normalize_sequence_id(sequence) for sequence in holdout_sequences)
    if not tune or not holdout:
        raise ValueError("tune and holdout sequence sets must both be nonempty")
    if len(set(tune)) != len(tune):
        raise ValueError("tune sequence set contains duplicates")
    if len(set(holdout)) != len(holdout):
        raise ValueError("holdout sequence set contains duplicates")
    overlap = sorted(set(tune) & set(holdout))
    if overlap:
        raise ValueError(
            f"tune and holdout sequence sets overlap: {', '.join(overlap)}"
        )
    return tune, holdout


def load_sequence_split_preset(
    config_path: str | Path, preset: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load and validate one named tune/holdout split from the JSON config."""

    normalized_preset = preset.strip().casefold()
    if normalized_preset not in SPLIT_PRESET_KEYS:
        choices = ", ".join(sorted(SPLIT_PRESET_KEYS))
        raise ValueError(f"unknown split preset {preset!r}; expected one of: {choices}")
    path = Path(config_path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid split configuration JSON: {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"split configuration must contain a JSON object: {path}")
    tune_key, holdout_key = SPLIT_PRESET_KEYS[normalized_preset]
    tune = document.get(tune_key)
    holdout = document.get(holdout_key)
    if not isinstance(tune, list) or not isinstance(holdout, list):
        raise ValueError(
            f"split preset {normalized_preset!r} requires list fields "
            f"{tune_key!r} and {holdout_key!r}"
        )
    return validate_sequence_split(tune, holdout)


def filter_prediction_file(
    source: str | Path,
    destination: str | Path,
    target_class: str,
    threshold: float,
) -> dict[str, int]:
    """Filter one KITTI class inclusively while leaving the other class unchanged."""

    normalized_class = target_class.strip().casefold()
    if normalized_class not in KITTI_TO_MODEL_CLASS:
        raise ValueError("target class must be 'car' or 'pedestrian'")
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"KITTI tracking prediction file does not exist: {source_path}"
        )
    retained_lines: list[str] = []
    all_rows_before = 0
    target_before = 0
    target_retained = 0
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            label = parse_kitti_tracking_label_line(
                line, source=source_path, line_number=line_number
            )
            if label is None:
                continue
            all_rows_before += 1
            if label.score is None:
                raise ValueError(
                    f"{source_path}: all prediction rows must contain a score"
                )
            object_type = label.type.casefold()
            if object_type not in KITTI_TO_MODEL_CLASS:
                raise ValueError(
                    f"{source_path}: unsupported prediction type {label.type!r}; "
                    "expected Car or Pedestrian"
                )
            if object_type == normalized_class:
                target_before += 1
                if float(label.score) < float(threshold):
                    continue
                target_retained += 1
            # Preserve all 18 original tokens. This avoids perturbing 2D box
            # precision or replacing valid 3D result fields during a 2D sweep.
            retained_lines.append(line.split("#", 1)[0].strip())

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(f".{destination_path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for line in retained_lines:
                handle.write(line)
                handle.write("\n")
        temporary.replace(destination_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "all_rows_before": all_rows_before,
        "all_rows_after": len(retained_lines),
        "target_rows_before": target_before,
        "target_rows_retained": target_retained,
        "target_rows_removed": target_before - target_retained,
    }


def choose_best_threshold(
    candidate_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Choose by HOTA, IDF1, MOTA, then the least aggressive threshold."""

    if not candidate_results:
        raise ValueError("threshold selection requires candidate results")
    for result in candidate_results:
        if "threshold" not in result or "metrics" not in result:
            raise ValueError("each candidate result needs threshold and metrics")
        if int(result["metrics"].get("GroundTruthDetections", 0)) <= 0:
            raise ValueError("the tuning class has no ground-truth detections")

    best_hota = max(float(result["metrics"]["HOTA"]) for result in candidate_results)
    if best_hota <= 0.0:
        raise ValueError(
            "no threshold produced a true-positive HOTA score; this class cannot "
            "be tuned meaningfully on the selected sequences"
        )
    return max(
        candidate_results,
        key=lambda result: (
            float(result["metrics"]["HOTA"]),
            float(result["metrics"]["IDF1"]),
            float(result["metrics"]["MOTA"]),
            -float(result["threshold"]),
        ),
    )


def sha256_file(path: str | Path) -> str:
    """Return a stable SHA-256 content digest for provenance."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sweep_csv(
    path: str | Path, candidate_results: Sequence[Mapping[str, Any]]
) -> None:
    """Atomically write the compact tune-sweep table."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    metric_fields = ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "FP", "FN", "IDSW")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "threshold",
                    "target_rows_before",
                    "target_rows_retained",
                    *metric_fields,
                ),
            )
            writer.writeheader()
            for result in candidate_results:
                counts = result["row_counts"]
                metrics = result["metrics"]
                writer.writerow(
                    {
                        "threshold": result["threshold"],
                        "target_rows_before": counts["target_rows_before"],
                        "target_rows_retained": counts["target_rows_retained"],
                        **{field: metrics[field] for field in metric_fields},
                    }
                )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "COMPATIBLE_MODEL_CLASSES",
    "EXPORT_MANIFEST_NAME",
    "KITTI_TO_MODEL_CLASS",
    "SPLIT_PRESET_KEYS",
    "choose_best_threshold",
    "filter_prediction_file",
    "load_sequence_split_preset",
    "normalize_model_class",
    "normalize_thresholds",
    "paths_overlap",
    "sha256_file",
    "validate_export_manifest_if_present",
    "validate_sequence_split",
    "write_sweep_csv",
]

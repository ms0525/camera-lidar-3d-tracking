# SPDX-License-Identifier: AGPL-3.0-only
"""Export reproducible KITTI tracking predictions for a fixed sequence split.

Each sequence is executed in a fresh Python process so GPU memory and tracker
state cannot leak between sequences.  An atomic manifest makes long exports
resumable without silently mixing configurations or accepting stale outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from utils.detection_thresholds import parse_class_confidence_overrides
from utils.kitti_tracking_dataset import normalize_sequence_id


SCHEMA_VERSION = 3
MANIFEST_NAME = "experiment_manifest.json"
PROJECT_ROOT = Path(__file__).resolve().parent
SPLIT_CONFIG_PATH = PROJECT_ROOT / "config" / "kitti_tracking_splits.json"
TRACK_SEQUENCE_SCRIPT = PROJECT_ROOT / "track_sequence.py"
ASSOCIATION_POLICY = "deepsort-class-exact-match-v1"
TRACKED_CLASS_POLICY = "kitti-evaluated-classes-v1"
CODE_PROVENANCE_PATHS = {
    "export_kitti_tracking_split.py": Path(__file__).resolve(),
    "track_sequence.py": TRACK_SEQUENCE_SCRIPT,
    "utils/class_aware_deepsort.py": PROJECT_ROOT
    / "utils"
    / "class_aware_deepsort.py",
    "utils/bin_pointcloud_loader.py": PROJECT_ROOT
    / "utils"
    / "bin_pointcloud_loader.py",
    "utils/calib_loader.py": PROJECT_ROOT / "utils" / "calib_loader.py",
    "utils/detection_thresholds.py": PROJECT_ROOT
    / "utils"
    / "detection_thresholds.py",
    "utils/kitti_tracking_dataset.py": PROJECT_ROOT
    / "utils"
    / "kitti_tracking_dataset.py",
    "utils/kitti_tracking_labels.py": PROJECT_ROOT
    / "utils"
    / "kitti_tracking_labels.py",
    "utils/kitti_tracking_results.py": PROJECT_ROOT
    / "utils"
    / "kitti_tracking_results.py",
}
TRAINING_SEQUENCES = tuple(f"{index:04d}" for index in range(21))
PRESET_KEYS = {
    "trackeval_tune": "trackeval_training_minus_val",
    "trackeval_val": "trackeval_val",
    "smoke_tune": "initial_smoke_tune",
    "smoke_val": "initial_smoke_val",
}


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def _finite_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be numeric") from exc
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="yolo26s.pt")

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--split-preset",
        choices=("all", "trackeval_tune", "trackeval_val", "smoke_tune", "smoke_val"),
    )
    selection.add_argument(
        "--sequences",
        nargs="+",
        metavar="SEQUENCE",
        help="Explicit KITTI training sequence IDs, for example 0000 0012",
    )

    parser.add_argument("--confidence", type=_finite_float, default=0.28)
    parser.add_argument(
        "--class-confidence",
        action="append",
        default=[],
        metavar="CLASS=VALUE",
    )
    parser.add_argument("--imgsz", type=_positive_int, default=640)
    parser.add_argument(
        "--yolo-end2end",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-age", type=_nonnegative_int, default=4)
    parser.add_argument("--n-init", type=_positive_int, default=3)
    parser.add_argument("--max-cosine-distance", type=_finite_float, default=0.5)
    parser.add_argument("--nn-budget", type=_positive_int, default=80)
    parser.add_argument("--embedder", default="mobilenet")
    parser.add_argument("--embedder-batch-size", type=_positive_int, default=4)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--embedder-gpu", action="store_true")
    parser.add_argument(
        "--log-child-output",
        action="store_true",
        help=(
            "Write track_sequence.py stdout/stderr to one log per sequence attempt "
            "under OUTPUT_DIR/logs instead of printing every frame"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing exactly matching manifest",
    )
    return parser


def _load_split_document(path: Path = SPLIT_CONFIG_PATH) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid split configuration JSON: {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"split configuration must be a JSON object: {path}")
    return document


def _normalize_sequence_list(values: Sequence[str | int]) -> tuple[str, ...]:
    sequences = tuple(normalize_sequence_id(value) for value in values)
    if not sequences:
        raise ValueError("at least one KITTI sequence is required")
    duplicates = sorted({value for value in sequences if sequences.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate KITTI sequences: {', '.join(duplicates)}")
    unsupported = sorted(set(sequences) - set(TRAINING_SEQUENCES))
    if unsupported:
        raise ValueError(
            "KITTI tracking training sequences range from 0000 to 0020; "
            f"unsupported: {', '.join(unsupported)}"
        )
    return sequences


def select_sequences(
    args: argparse.Namespace, config_path: Path = SPLIT_CONFIG_PATH
) -> tuple[str, ...]:
    """Resolve exactly one explicit or repository-defined sequence selection."""

    preset = getattr(args, "split_preset", None)
    explicit = getattr(args, "sequences", None)
    if (preset is None) == (explicit is None):
        raise ValueError("provide exactly one of --split-preset or --sequences")
    if explicit is not None:
        return _normalize_sequence_list(explicit)

    document = _load_split_document(config_path)
    if preset == "all":
        tune_values = document.get("trackeval_training_minus_val")
        validation_values = document.get("trackeval_val")
        if not isinstance(tune_values, list) or not isinstance(validation_values, list):
            raise ValueError(
                "split configuration fields 'trackeval_training_minus_val' and "
                "'trackeval_val' must be lists"
            )
        raw_values: Any = [*tune_values, *validation_values]
        sequences = _normalize_sequence_list(raw_values)
        if set(sequences) != set(TRAINING_SEQUENCES):
            raise ValueError(
                "the TrackEval tune/validation lists must cover all 21 KITTI "
                "training sequences for --split-preset all"
            )
        return TRAINING_SEQUENCES

    key = PRESET_KEYS.get(str(preset))
    if key is None:
        raise ValueError(f"unknown split preset: {preset!r}")
    raw_values = document.get(key)
    if not isinstance(raw_values, list):
        raise ValueError(f"split configuration field {key!r} must be a list")
    return _normalize_sequence_list(raw_values)


def _normalize_text(value: Any, option: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(f"{option} must not be empty")
    return text


def normalized_configuration(
    args: argparse.Namespace, sequences: Sequence[str]
) -> dict[str, Any]:
    """Return the JSON-safe configuration which uniquely determines outputs."""

    confidence = float(args.confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("--confidence must be a finite value between 0 and 1")
    max_cosine_distance = float(args.max_cosine_distance)
    if not math.isfinite(max_cosine_distance) or max_cosine_distance < 0.0:
        raise ValueError("--max-cosine-distance must be finite and nonnegative")

    class_confidence = parse_class_confidence_overrides(args.class_confidence)
    model_argument = _normalize_text(args.model, "--model")
    assert model_argument is not None

    device = _normalize_text(args.device, "--device", allow_none=True)
    embedder = _normalize_text(args.embedder, "--embedder")
    assert embedder is not None
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"KITTI dataset root does not exist: {dataset_root}")

    return {
        "association_policy": ASSOCIATION_POLICY,
        "tracked_class_policy": TRACKED_CLASS_POLICY,
        "dataset_root": str(dataset_root),
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "sequences": list(sequences),
        "model": model_argument,
        "confidence": confidence,
        "class_confidence": dict(sorted(class_confidence.items())),
        "imgsz": int(args.imgsz),
        "yolo_end2end": args.yolo_end2end,
        "device": device,
        "max_age": int(args.max_age),
        "n_init": int(args.n_init),
        "max_cosine_distance": max_cosine_distance,
        "nn_budget": int(args.nn_budget),
        "embedder": embedder,
        "embedder_batch_size": int(args.embedder_batch_size),
        "half": bool(args.half),
        "embedder_gpu": bool(args.embedder_gpu),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> dict[str, Any]:
    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    opencv_distributions = (
        "opencv-python",
        "opencv-contrib-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    )
    opencv_packages = {
        name: version
        for name in opencv_distributions
        if (version := package_version(name)) is not None
    }

    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "ultralytics": package_version("ultralytics"),
        "torch": package_version("torch"),
        "deep_sort_realtime": package_version("deep-sort-realtime"),
        "numpy": package_version("numpy"),
        "opencv": opencv_packages,
    }


def code_provenance() -> dict[str, dict[str, Any]]:
    """Return byte-level provenance for project code that determines exports."""

    provenance: dict[str, dict[str, Any]] = {}
    for identifier, path in CODE_PROVENANCE_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"provenance source file does not exist: {path}")
        provenance[identifier] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return provenance


def model_provenance(model_argument: str) -> dict[str, Any]:
    candidate = Path(model_argument).expanduser()
    if not candidate.is_file():
        return {"argument": model_argument, "local_file": False}
    resolved = candidate.resolve()
    return {
        "argument": model_argument,
        "local_file": True,
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def atomic_write_manifest(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _new_manifest(configuration: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    output_dir = Path(configuration["output_dir"])
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": now,
        "updated_at_utc": now,
        "status": "pending",
        "configuration": configuration,
        "runtime": runtime_versions(),
        "code_provenance": code_provenance(),
        "model": model_provenance(configuration["model"]),
        "sequence_results": {
            sequence: {
                "status": "pending",
                "output_file": str(output_dir / f"{sequence}.txt"),
                "attempt_count": 0,
            }
            for sequence in configuration["sequences"]
        },
    }


def _load_resume_manifest(
    path: Path, configuration: dict[str, Any]
) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid experiment manifest: {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"experiment manifest must contain a JSON object: {path}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported experiment manifest schema: {manifest.get('schema_version')!r}"
        )
    if manifest.get("configuration") != configuration:
        raise ValueError(
            "--resume configuration does not exactly match experiment_manifest.json"
        )
    if manifest.get("code_provenance") != code_provenance():
        raise ValueError(
            "project code provenance changed since the experiment was created"
        )
    if manifest.get("runtime") != runtime_versions():
        raise ValueError(
            "runtime environment changed since the experiment was created"
        )
    expected_keys = set(configuration["sequences"])
    results = manifest.get("sequence_results")
    if not isinstance(results, dict) or set(results) != expected_keys:
        raise ValueError("experiment manifest sequence entries are inconsistent")
    output_dir = Path(configuration["output_dir"])
    for sequence in configuration["sequences"]:
        entry = results[sequence]
        expected_output = str(output_dir / f"{sequence}.txt")
        if not isinstance(entry, dict) or entry.get("output_file") != expected_output:
            raise ValueError(
                f"experiment manifest output entry is inconsistent for {sequence}"
            )

    stored_model = manifest.get("model")
    current_model = model_provenance(configuration["model"])
    if isinstance(stored_model, dict) and stored_model.get("local_file"):
        if stored_model != current_model:
            raise ValueError("local model file changed since the experiment was created")
    return manifest


def build_sequence_command(
    configuration: dict[str, Any], sequence: str, output_file: Path
) -> list[str]:
    command = [
        sys.executable,
        str(TRACK_SEQUENCE_SCRIPT),
        "--dataset-root",
        configuration["dataset_root"],
        "--sequence",
        sequence,
        "--headless",
        "--export-kitti",
        str(output_file),
        "--model",
        configuration["model"],
        "--confidence",
        repr(configuration["confidence"]),
        "--imgsz",
        str(configuration["imgsz"]),
        "--max-age",
        str(configuration["max_age"]),
        "--n-init",
        str(configuration["n_init"]),
        "--max-cosine-distance",
        repr(configuration["max_cosine_distance"]),
        "--nn-budget",
        str(configuration["nn_budget"]),
        "--embedder",
        configuration["embedder"],
        "--embedder-batch-size",
        str(configuration["embedder_batch_size"]),
    ]
    for name, value in configuration["class_confidence"].items():
        command.extend(("--class-confidence", f"{name}={value!r}"))
    if configuration["yolo_end2end"] is True:
        command.append("--yolo-end2end")
    elif configuration["yolo_end2end"] is False:
        command.append("--no-yolo-end2end")
    if configuration["device"] is not None:
        command.extend(("--device", configuration["device"]))
    if configuration["half"]:
        command.append("--half")
    if configuration["embedder_gpu"]:
        command.append("--embedder-gpu")
    return command


def _completed_output_matches(entry: dict[str, Any], output_file: Path) -> bool:
    expected_hash = entry.get("output_sha256")
    if (
        entry.get("status") != "completed"
        or not isinstance(expected_hash, str)
        or not output_file.is_file()
    ):
        return False
    try:
        return sha256_file(output_file) == expected_hash
    except OSError:
        return False


def _write_state(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = utc_now()
    atomic_write_manifest(path, manifest)


def _check_model_unchanged(manifest: dict[str, Any], model_argument: str) -> None:
    """Adopt a newly downloaded model once, then require byte identity."""

    stored = manifest.get("model")
    current = model_provenance(model_argument)
    if isinstance(stored, dict) and stored.get("local_file") and stored != current:
        raise ValueError("local model file changed during the experiment")
    if not isinstance(stored, dict) or not stored.get("local_file"):
        manifest["model"] = current


def run(args: argparse.Namespace) -> int:
    sequences = select_sequences(args)
    configuration = normalized_configuration(args, sequences)
    output_dir = Path(configuration["output_dir"])
    manifest_path = output_dir / MANIFEST_NAME

    if args.resume:
        if not output_dir.is_dir() or not manifest_path.is_file():
            raise ValueError(
                "--resume requires an existing output directory and experiment_manifest.json"
            )
        manifest = _load_resume_manifest(manifest_path, configuration)
    else:
        if output_dir.exists():
            raise ValueError(
                f"output directory already exists: {output_dir}; use --resume only "
                "for an existing matching experiment"
            )
        output_dir.mkdir(parents=True)
        manifest = _new_manifest(configuration)
        atomic_write_manifest(manifest_path, manifest)

    manifest["status"] = "running"
    _write_state(manifest_path, manifest)

    for sequence in sequences:
        entry = manifest["sequence_results"][sequence]
        output_file = output_dir / f"{sequence}.txt"
        if args.resume and _completed_output_matches(entry, output_file):
            print(f"Skipping completed sequence {sequence}: hash verified")
            continue

        try:
            _check_model_unchanged(manifest, configuration["model"])
        except ValueError as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            manifest["status"] = "failed"
            _write_state(manifest_path, manifest)
            print(f"Sequence {sequence} was not started: {exc}")
            return 1

        attempt_count = int(entry.get("attempt_count", 0)) + 1
        attempt_output_file = output_dir / (
            f".{sequence}.attempt-{attempt_count}.txt"
        )
        command = build_sequence_command(
            configuration, sequence, attempt_output_file
        )
        log_file: Path | None = None
        if getattr(args, "log_child_output", False):
            log_file = output_dir / "logs" / (
                f"{sequence}.attempt-{attempt_count}.log"
            )
        entry.update(
            {
                "status": "running",
                "started_at_utc": utc_now(),
                "completed_at_utc": None,
                "duration_seconds": None,
                "returncode": None,
                "output_sha256": None,
                "attempt_count": attempt_count,
                "attempt_output_file": str(attempt_output_file),
                "command": command,
            }
        )
        if log_file is not None:
            entry["log_file"] = str(log_file)
        else:
            entry.pop("log_file", None)
        manifest["status"] = "running"
        _write_state(manifest_path, manifest)
        print(f"Exporting KITTI sequence {sequence} -> {output_file}")
        if log_file is not None:
            print(f"Child output: {log_file}")
        started = time.perf_counter()
        try:
            if log_file is None:
                completed = subprocess.run(command, check=False)
            else:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with log_file.open("w", encoding="utf-8", newline="\n") as log:
                    completed = subprocess.run(
                        command,
                        check=False,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            returncode = int(completed.returncode)
        except KeyboardInterrupt:
            returncode = 130
            entry["error"] = "interrupted by user"
        except OSError as exc:
            returncode = 1
            entry["error"] = f"could not start track_sequence.py: {exc}"

        duration = time.perf_counter() - started
        entry["duration_seconds"] = round(duration, 6)
        entry["returncode"] = returncode
        entry["completed_at_utc"] = utc_now()
        try:
            _check_model_unchanged(manifest, configuration["model"])
        except ValueError as exc:
            returncode = 1
            entry["returncode"] = returncode
            entry["error"] = str(exc)

        if returncode == 0 and attempt_output_file.is_file():
            try:
                os.replace(attempt_output_file, output_file)
                entry["status"] = "completed"
                entry["output_sha256"] = sha256_file(output_file)
                entry.pop("error", None)
                _write_state(manifest_path, manifest)
                continue
            except OSError as exc:
                returncode = 1
                entry["returncode"] = returncode
                entry["error"] = f"could not finalize sequence output: {exc}"

        if returncode == 0:
            returncode = 1
            entry["returncode"] = returncode
            entry["error"] = (
                "track_sequence.py returned success without creating a new attempt output"
            )
        entry["status"] = "failed"
        manifest["status"] = "failed"
        _write_state(manifest_path, manifest)
        print(f"Sequence {sequence} failed; see {manifest_path}")
        return returncode

    manifest["status"] = "completed"
    manifest["completed_at_utc"] = utc_now()
    _write_state(manifest_path, manifest)
    print(f"Completed {len(sequences)} sequence exports. Manifest: {manifest_path}")
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

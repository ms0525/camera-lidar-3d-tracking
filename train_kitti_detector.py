# SPDX-License-Identifier: AGPL-3.0-only
"""Reproducible Ultralytics detector fine-tuning on the KITTI dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SUPPORTED_MODELS = ("yolo26s.pt", "yolo11s.pt")
DATASET_CONFIG = "kitti.yaml"
RUN_DATASET_CONFIG = "kitti_dataset.yaml"
MANIFEST_NAME = "training_request.json"
KITTI_CLASS_NAMES = (
    "car",
    "van",
    "truck",
    "pedestrian",
    "person_sitting",
    "cyclist",
    "tram",
    "misc",
)
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
EXPECTED_SPLIT_COUNTS = {"train": 5985, "val": 1496}
MANIFEST_SCHEMA_VERSION = 2


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


def _nonempty(value: str) -> str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an Ultralytics YOLO26 or YOLO11 detector using either "
            "an explicit KITTI dataset root or the built-in kitti.yaml config."
        )
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="yolo26s.pt",
        help="Pretrained detector checkpoint (default: %(default)s).",
    )
    parser.add_argument("--epochs", type=_positive_int, default=100)
    parser.add_argument("--imgsz", type=_positive_int, default=640)
    parser.add_argument("--batch", type=_positive_int, default=16)
    parser.add_argument(
        "--device",
        type=_nonempty,
        default="0",
        help=(
            "Ultralytics device selector, for example an accelerator index (0), "
            "a list (0,1), or cpu. Numeric selectors work for CUDA and ROCm."
        ),
    )
    parser.add_argument("--workers", type=_nonnegative_int, default=8)
    parser.add_argument("--patience", type=_nonnegative_int, default=50)
    parser.add_argument("--seed", type=_nonnegative_int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", type=_nonempty, default="kitti_yolo26s")
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset-root",
        type=Path,
        help=(
            "Existing KITTI detection dataset root containing images/train, "
            "images/val, labels/train, and labels/val. A run-local YAML with "
            "this absolute path is generated instead of changing global settings."
        ),
    )
    dataset_group.add_argument(
        "--datasets-dir",
        type=Path,
        help=(
            "Deprecated compatibility alias for the parent of an existing "
            "'kitti' directory. Prefer --dataset-root. This option does not "
            "change Ultralytics global settings or download data."
        ),
    )
    parser.add_argument(
        "--allow-cpu-training",
        action="store_true",
        help="Permit an explicitly requested --device cpu run (usually very slow).",
    )
    parser.add_argument(
        "--allow-custom-dataset-size",
        action="store_true",
        help=(
            "Permit paired KITTI-format splits whose image counts differ from "
            "the official 5,985 train / 1,496 val split."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Safely resume this exact --project/--name from weights/last.pt. "
            "The recorded model, configuration, and dataset fingerprint must match."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and exit without imports, downloads, output, or training.",
    )
    return parser


def _json_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _accelerator_backend(torch: Any) -> str:
    """Identify the runtime while respecting PyTorch's ROCm CUDA-compatible API."""

    torch_version = getattr(torch, "version", None)
    if getattr(torch_version, "hip", None):
        return "rocm"
    if getattr(torch_version, "cuda", None):
        return "cuda"
    if _mps_available(torch):
        return "mps"
    return "cpu"


def _hardware_manifest(torch: Any) -> dict[str, Any]:
    # PyTorch intentionally exposes both CUDA and ROCm accelerators through
    # torch.cuda. Record neutral fields plus the actual compiled runtime so a
    # ROCm GPU is not mistaken for a CUDA installation.
    accelerator_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if accelerator_available else 0
    backend = _accelerator_backend(torch)
    devices: list[dict[str, Any]] = []
    for index in range(device_count):
        device: dict[str, Any] = {
            "index": index,
            "name": str(torch.cuda.get_device_name(index)),
        }
        get_capability = getattr(torch.cuda, "get_device_capability", None)
        if backend == "cuda" and callable(get_capability):
            device["compute_capability"] = list(get_capability(index))
        get_properties = getattr(torch.cuda, "get_device_properties", None)
        if backend == "rocm" and callable(get_properties):
            properties = get_properties(index)
            gcn_arch_name = getattr(properties, "gcnArchName", None)
            if gcn_arch_name:
                device["gcn_arch_name"] = str(gcn_arch_name)
        devices.append(device)

    torch_version = getattr(torch, "version", None)
    cuda_version = getattr(torch_version, "cuda", None)
    hip_version = getattr(torch_version, "hip", None)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "backend": backend,
        "accelerator_available": accelerator_available,
        "accelerator_device_count": device_count,
        "accelerator_devices": devices,
        "cuda_runtime": cuda_version,
        "hip_runtime": hip_version,
        # Compatibility aliases retained for manifests created before ROCm
        # support. On ROCm these describe the torch.cuda compatibility API;
        # use backend/accelerator_* to identify the actual runtime and devices.
        "cuda_available": accelerator_available,
        "cuda_device_count": device_count,
        "cuda_devices": devices,
        "mps_available": _mps_available(torch),
    }


def _mps_available(torch: Any) -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def _validate_requested_device(args: argparse.Namespace, torch: Any) -> None:
    """Fail early when the requested accelerator cannot be honored."""

    device = args.device.strip().lower()
    if device == "cpu":
        if not args.allow_cpu_training:
            raise RuntimeError(
                "--device cpu requires --allow-cpu-training, even when an "
                "accelerator is available, because full detector training on CPU "
                "is very slow."
            )
        return
    if device == "mps":
        if not _mps_available(torch):
            raise RuntimeError("--device mps was requested, but PyTorch MPS is unavailable.")
        return

    is_accelerator_selector = bool(
        re.fullmatch(r"(?:cuda(?::\d+)?|\d+(?:,\d+)*|-1(?:,-1)*)", device)
    )
    if not is_accelerator_selector:
        raise RuntimeError(
            f"Unsupported --device selector {args.device!r}; use cpu, mps, an "
            "accelerator index such as 0, a list such as 0,1, or cuda:0. Numeric "
            "selectors target either CUDA or ROCm through PyTorch."
        )
    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            f"Accelerator device selector {args.device!r} was requested, but "
            "PyTorch reports no CUDA/ROCm accelerator is available. Select "
            "--device cpu --allow-cpu-training only for a deliberate CPU run."
        )


def _training_kwargs(
    args: argparse.Namespace, *, dataset_config: str = DATASET_CONFIG
) -> dict[str, Any]:
    return {
        "data": dataset_config,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "project": str(args.project),
        "name": args.name,
        # The manifest must exist before model.train(). Creating it would make
        # Ultralytics increment the directory unless this is explicitly true.
        "exist_ok": True,
    }


def _request_record(
    *,
    args: argparse.Namespace,
    argv: Sequence[str],
    ultralytics: Any,
    torch: Any,
) -> dict[str, Any]:
    return {
        "requested_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": [Path(sys.executable).name, Path(__file__).name, *argv],
        "arguments": _json_arguments(args),
        "versions": {
            "ultralytics": str(getattr(ultralytics, "__version__", "unknown")),
            "torch": str(getattr(torch, "__version__", "unknown")),
        },
        "hardware": _hardware_manifest(torch),
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    argv: Sequence[str],
    ultralytics: Any,
    torch: Any,
    train_kwargs: dict[str, Any],
    configuration: dict[str, Any],
    dataset_provenance: dict[str, Any],
    initialization_checkpoint: dict[str, Any],
) -> None:
    request = _request_record(
        args=args, argv=argv, ultralytics=ultralytics, torch=torch
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        **request,
        "configuration": configuration,
        "dataset": train_kwargs["data"],
        "dataset_provenance": dataset_provenance,
        "train_kwargs": train_kwargs,
        "checkpoints": {
            "initialization": initialization_checkpoint,
            "resume_requests": [],
        },
    }
    _atomic_write_json(path, manifest)


def _requested_dataset_root(args: argparse.Namespace) -> Path | None:
    if args.dataset_root is not None:
        return args.dataset_root
    if args.datasets_dir is not None:
        return args.datasets_dir / "kitti"
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_group(dataset_root: Path, paths: list[Path]) -> dict[str, Any]:
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in sorted(paths, key=lambda item: item.as_posix().casefold()):
        size = path.stat().st_size
        file_hash = _sha256_file(path)
        relative = path.relative_to(dataset_root).as_posix()
        aggregate.update(
            json.dumps(
                [relative, size, file_hash], separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        )
        aggregate.update(b"\n")
        total_bytes += size
    return {
        "count": len(paths),
        "total_bytes": total_bytes,
        "sha256": aggregate.hexdigest(),
    }


def _dataset_inventory(
    root: Path,
    split_files: dict[str, tuple[list[Path], list[Path]]],
) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split in ("train", "val"):
        images, labels = split_files[split]
        splits[split] = {
            "images": _inventory_group(root, images),
            "labels": _inventory_group(root, labels),
        }
    fingerprint_payload = {
        "class_names": list(KITTI_CLASS_NAMES),
        "splits": splits,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "root": str(root),
        **fingerprint_payload,
        "fingerprint_sha256": fingerprint,
    }


def _validate_dataset_root(
    root: Path, *, allow_custom_dataset_size: bool = False
) -> tuple[Path, dict[str, Any]]:
    """Resolve and validate the expected Ultralytics KITTI directory layout."""

    root = root.expanduser().resolve()
    required = (
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Invalid KITTI detection dataset root: {root}. Missing required "
            f"directories: {', '.join(missing)}"
        )

    split_files: dict[str, tuple[list[Path], list[Path]]] = {}
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_paths = [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        label_paths = [
            path
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        ]
        image_stems = {path.stem for path in image_paths}
        label_stems = {path.stem for path in label_paths}
        if not image_stems:
            raise ValueError(f"KITTI {split} image directory is empty: {image_dir}")
        if not label_stems:
            raise ValueError(f"KITTI {split} label directory is empty: {label_dir}")
        if len(image_stems) != len(image_paths):
            raise ValueError(
                f"KITTI {split} split contains multiple image files with the same stem."
            )
        if len(label_stems) != len(label_paths):
            raise ValueError(
                f"KITTI {split} split contains multiple label files with the same stem."
            )

        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)
        if missing_labels or orphan_labels:
            details: list[str] = []
            if missing_labels:
                details.append(
                    f"{len(missing_labels)} images without labels "
                    f"(for example {missing_labels[0]})"
                )
            if orphan_labels:
                details.append(
                    f"{len(orphan_labels)} labels without images "
                    f"(for example {orphan_labels[0]})"
                )
            raise ValueError(f"KITTI {split} split is not paired: {'; '.join(details)}")

        expected_count = EXPECTED_SPLIT_COUNTS[split]
        if not allow_custom_dataset_size and (
            len(image_paths) != expected_count or len(label_paths) != expected_count
        ):
            raise ValueError(
                f"KITTI {split} split has {len(image_paths):,} images and "
                f"{len(label_paths):,} labels; the official split requires "
                f"{expected_count:,} paired files. Pass "
                "--allow-custom-dataset-size only for an intentional custom subset."
            )
        split_files[split] = (image_paths, label_paths)

    return root, _dataset_inventory(root, split_files)


def _write_dataset_config(path: Path, dataset_root: Path) -> None:
    # JSON strings are valid YAML scalars and safely preserve spaces, colons,
    # Unicode, and Windows drive paths.
    root_scalar = json.dumps(dataset_root.as_posix(), ensure_ascii=False)
    lines = [
        "# Generated by train_kitti_detector.py; do not edit during a run.",
        f"path: {root_scalar}",
        "train: images/train",
        "val: images/val",
        "names:",
        *(f"  {index}: {name}" for index, name in enumerate(KITTI_CLASS_NAMES)),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _configuration_snapshot(
    args: argparse.Namespace, dataset_root: Path | None
) -> dict[str, Any]:
    """Return the immutable inputs that must match for a safe resume."""

    return {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device.strip().lower(),
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "project": str(args.project.expanduser().resolve()),
        "name": args.name,
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "dataset_config": DATASET_CONFIG if dataset_root is None else RUN_DATASET_CONFIG,
    }


def _checkpoint_provenance(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _resolve_initial_checkpoint(model: Any, requested: str) -> Path:
    candidates = [getattr(model, "ckpt_path", None), requested]
    for candidate in candidates:
        if candidate:
            path = Path(str(candidate)).expanduser()
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(
        f"Ultralytics loaded {requested!r}, but its resolved checkpoint file could "
        "not be found, so a reproducible SHA-256 cannot be recorded."
    )


def _dataset_provenance(
    *,
    dataset_root: Path | None,
    inventory: dict[str, Any] | None,
    dataset_config: str,
) -> dict[str, Any]:
    if dataset_root is None:
        marker = f"ultralytics-config:{DATASET_CONFIG}"
        return {
            "source": "ultralytics_builtin_config",
            "config": DATASET_CONFIG,
            "inventory": None,
            "fingerprint_sha256": hashlib.sha256(marker.encode("utf-8")).hexdigest(),
        }

    if inventory is None:  # pragma: no cover - defensive invariant
        raise RuntimeError("Explicit dataset root is missing its inventory")
    config_path = Path(dataset_config)
    return {
        "source": "explicit_root",
        "root": str(dataset_root),
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "inventory": inventory,
        "fingerprint_sha256": inventory["fingerprint_sha256"],
    }


def _load_resume_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot safely resume: the run manifest is missing: {path}"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot safely resume: invalid run manifest {path}: {exc}") from exc
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Cannot safely resume a run without the current provenance schema; "
            "start a new --name instead."
        )
    return manifest


def _validate_resume_manifest(
    manifest: dict[str, Any],
    *,
    configuration: dict[str, Any],
    dataset_provenance: dict[str, Any],
    run_dir: Path,
) -> None:
    stored_configuration = manifest.get("configuration")
    if not isinstance(stored_configuration, dict):
        raise ValueError("Cannot safely resume: manifest configuration is missing.")
    mismatches = [
        key
        for key, value in configuration.items()
        if stored_configuration.get(key) != value
    ]
    if mismatches:
        details = ", ".join(
            f"{key}={configuration[key]!r} (recorded {stored_configuration.get(key)!r})"
            for key in mismatches
        )
        raise ValueError(f"Cannot safely resume: configuration mismatch: {details}")

    stored_dataset = manifest.get("dataset_provenance")
    if not isinstance(stored_dataset, dict):
        raise ValueError("Cannot safely resume: dataset provenance is missing.")
    if stored_dataset.get("fingerprint_sha256") != dataset_provenance.get(
        "fingerprint_sha256"
    ):
        raise ValueError(
            "Cannot safely resume: dataset inventory fingerprint changed. Restore "
            "the original images/labels or start a new --name."
        )

    if dataset_provenance.get("source") == "explicit_root":
        config_path = (run_dir / RUN_DATASET_CONFIG).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Cannot safely resume: run-local dataset config is missing: {config_path}"
            )
        recorded_hash = stored_dataset.get("config_sha256")
        if recorded_hash != _sha256_file(config_path):
            raise ValueError(
                "Cannot safely resume: run-local dataset config changed. Restore it "
                "or start a new --name."
            )

    initialization = manifest.get("checkpoints", {}).get("initialization", {})
    if initialization.get("requested_model") != configuration["model"]:
        raise ValueError(
            "Cannot safely resume: initialization model does not match the request."
        )


def _append_resume_request(
    path: Path,
    manifest: dict[str, Any],
    *,
    args: argparse.Namespace,
    argv: Sequence[str],
    ultralytics: Any,
    torch: Any,
    checkpoint: dict[str, Any],
) -> None:
    checkpoints = manifest.setdefault("checkpoints", {})
    requests = checkpoints.setdefault("resume_requests", [])
    if not isinstance(requests, list):
        raise ValueError("Cannot safely resume: resume provenance history is invalid.")
    requests.append(
        {
            **_request_record(
                args=args, argv=argv, ultralytics=ultralytics, torch=torch
            ),
            "checkpoint": checkpoint,
        }
    )
    _atomic_write_json(path, manifest)


def run(argv: Sequence[str] | None = None) -> Any:
    """Parse ``argv`` and execute one training request.

    A valid dry run deliberately returns before importing Ultralytics or
    PyTorch, ensuring it cannot download weights/data or mutate settings.
    """

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    if args.dry_run:
        return None

    requested_dataset_root = _requested_dataset_root(args)
    if requested_dataset_root is None:
        dataset_root = None
        dataset_inventory = None
    else:
        dataset_root, dataset_inventory = _validate_dataset_root(
            requested_dataset_root,
            allow_custom_dataset_size=args.allow_custom_dataset_size,
        )

    # Keep heavyweight libraries lazy: argument validation and dry runs do not
    # initialize Ultralytics, an accelerator backend, or the user's global
    # Ultralytics settings.
    import torch
    import ultralytics

    _validate_requested_device(args, torch)

    run_dir = args.project.expanduser() / args.name
    manifest_path = run_dir / MANIFEST_NAME
    configuration = _configuration_snapshot(args, dataset_root)

    if args.resume:
        manifest = _load_resume_manifest(manifest_path)
        if dataset_root is None:
            dataset_config = DATASET_CONFIG
        else:
            dataset_config = str((run_dir / RUN_DATASET_CONFIG).resolve())
        dataset_provenance = _dataset_provenance(
            dataset_root=dataset_root,
            inventory=dataset_inventory,
            dataset_config=dataset_config,
        )
        _validate_resume_manifest(
            manifest,
            configuration=configuration,
            dataset_provenance=dataset_provenance,
            run_dir=run_dir,
        )
        last_checkpoint = (run_dir / "weights" / "last.pt").resolve()
        resume_checkpoint = _checkpoint_provenance(last_checkpoint)
        model = ultralytics.YOLO(str(last_checkpoint))
        _append_resume_request(
            manifest_path,
            manifest,
            args=args,
            argv=raw_argv,
            ultralytics=ultralytics,
            torch=torch,
            checkpoint=resume_checkpoint,
        )
        return model.train(resume=True)

    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Training run directory is not empty: {run_dir}. "
            "Choose a new --name to preserve it, or pass --resume for a "
            "configuration-matched interrupted run."
        )

    model = ultralytics.YOLO(args.model)
    initial_checkpoint_path = _resolve_initial_checkpoint(model, args.model)
    initialization_checkpoint = {
        "requested_model": args.model,
        **_checkpoint_provenance(initial_checkpoint_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)

    if dataset_root is None:
        dataset_config = DATASET_CONFIG
    else:
        dataset_config_path = (run_dir / RUN_DATASET_CONFIG).resolve()
        _write_dataset_config(dataset_config_path, dataset_root)
        dataset_config = str(dataset_config_path)

    train_kwargs = _training_kwargs(args, dataset_config=dataset_config)
    dataset_provenance = _dataset_provenance(
        dataset_root=dataset_root,
        inventory=dataset_inventory,
        dataset_config=dataset_config,
    )
    _write_manifest(
        manifest_path,
        args=args,
        argv=raw_argv,
        ultralytics=ultralytics,
        torch=torch,
        train_kwargs=train_kwargs,
        configuration=configuration,
        dataset_provenance=dataset_provenance,
        initialization_checkpoint=initialization_checkpoint,
    )

    return model.train(**train_kwargs)


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

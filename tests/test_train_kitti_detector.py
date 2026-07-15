# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from train_kitti_detector import (
    DATASET_CONFIG,
    KITTI_CLASS_NAMES,
    RUN_DATASET_CONFIG,
    build_parser,
    run,
)


class FakeSettings(dict):
    def __init__(self) -> None:
        super().__init__(datasets_dir="original-datasets")
        self.updates: list[dict[str, object]] = []

    def update(self, values):  # type: ignore[no-untyped-def]
        self.updates.append(dict(values))
        super().update(values)


class FakeCuda:
    def __init__(self, available: bool, gcn_arch_name: str | None = None) -> None:
        self.available = available
        self.gcn_arch_name = gcn_arch_name

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return 1 if self.available else 0

    def get_device_name(self, index: int) -> str:
        return f"Fake GPU {index}"

    def get_device_capability(self, index: int) -> tuple[int, int]:
        return (9, index)

    def get_device_properties(self, index: int):  # type: ignore[no-untyped-def]
        return types.SimpleNamespace(gcnArchName=self.gcn_arch_name)


def fake_modules(
    *,
    cuda: bool,
    mps: bool = False,
    hip: str | None = None,
    train_error: Exception | None = None,
):
    settings = FakeSettings()
    instances = []

    class FakeYOLO:
        def __init__(self, checkpoint: str) -> None:
            self.checkpoint = checkpoint
            candidate = Path(checkpoint)
            self.ckpt_path = candidate if candidate.is_file() else Path(__file__)
            self.train_kwargs = None
            instances.append(self)

        def train(self, **kwargs):  # type: ignore[no-untyped-def]
            self.train_kwargs = kwargs
            if train_error is not None:
                raise train_error
            return "trained"

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.__version__ = "99.1.2"
    ultralytics.YOLO = FakeYOLO
    ultralytics.settings = settings

    torch = types.ModuleType("torch")
    torch.__version__ = "88.2.1+fake"
    torch.cuda = FakeCuda(cuda, gcn_arch_name="gfx1102" if hip else None)
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    torch.version = types.SimpleNamespace(
        cuda=None if hip else ("13.0" if cuda else None),
        hip=hip,
    )
    return ultralytics, torch, settings, instances


def make_dataset(root: Path) -> Path:
    dataset_root = root / "kitti"
    for split, stem in (("train", "000001"), ("val", "000002")):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / f"{stem}.png").write_bytes(b"image")
        (label_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    return dataset_root


class TrainKittiDetectorTests(unittest.TestCase):
    def test_defaults_select_yolo26_and_expected_training_values(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.model, "yolo26s.pt")
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.imgsz, 640)
        self.assertEqual(args.batch, 16)
        self.assertFalse(args.allow_custom_dataset_size)
        self.assertFalse(args.resume)

    def test_parser_accepts_yolo11_and_rejects_invalid_numbers(self) -> None:
        args = build_parser().parse_args(["--model", "yolo11s.pt", "--workers", "0"])
        self.assertEqual(args.model, "yolo11s.pt")
        self.assertEqual(args.workers, 0)

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--epochs", "0"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--seed", "-1"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["--dataset-root", "kitti", "--datasets-dir", "datasets"]
            )

    def test_dry_run_has_no_import_output_or_filesystem_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "must-not-exist"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch.dict(sys.modules, {"ultralytics": None, "torch": None}):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = run(["--dry-run", "--project", str(project)])

            self.assertIsNone(result)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertFalse(project.exists())

    def test_cuda_guard_runs_before_manifest_or_model_creation(self) -> None:
        ultralytics, torch, _, instances = fake_modules(cuda=False)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "runs"
            with patch.dict(
                sys.modules, {"ultralytics": ultralytics, "torch": torch}
            ), self.assertRaisesRegex(RuntimeError, "--allow-cpu-training"):
                run(["--project", str(project)])

            self.assertFalse(project.exists())
            self.assertEqual(instances, [])

    def test_training_writes_manifest_and_uses_exact_run_directory(self) -> None:
        ultralytics, torch, settings, instances = fake_modules(cuda=True)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "runs"
            argv = [
                "--model",
                "yolo11s.pt",
                "--epochs",
                "3",
                "--imgsz",
                "320",
                "--batch",
                "2",
                "--device",
                "0",
                "--workers",
                "1",
                "--patience",
                "4",
                "--seed",
                "7",
                "--project",
                str(project),
                "--name",
                "experiment",
            ]
            with patch.dict(sys.modules, {"ultralytics": ultralytics, "torch": torch}):
                result = run(argv)

            self.assertEqual(result, "trained")
            self.assertEqual(len(instances), 1)
            self.assertEqual(instances[0].checkpoint, "yolo11s.pt")
            self.assertEqual(
                instances[0].train_kwargs,
                {
                    "data": DATASET_CONFIG,
                    "epochs": 3,
                    "imgsz": 320,
                    "batch": 2,
                    "device": "0",
                    "workers": 1,
                    "patience": 4,
                    "seed": 7,
                    "project": str(project),
                    "name": "experiment",
                    "exist_ok": True,
                },
            )
            self.assertEqual(settings.updates, [])

            manifest_path = project / "experiment" / "training_request.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["arguments"]["model"], "yolo11s.pt")
            self.assertEqual(manifest["command"][2:], argv)
            self.assertEqual(manifest["versions"]["ultralytics"], "99.1.2")
            self.assertEqual(manifest["versions"]["torch"], "88.2.1+fake")
            hardware = manifest["hardware"]
            self.assertEqual(hardware["backend"], "cuda")
            self.assertTrue(hardware["accelerator_available"])
            self.assertEqual(hardware["accelerator_device_count"], 1)
            self.assertEqual(
                hardware["accelerator_devices"][0]["compute_capability"], [9, 0]
            )
            self.assertEqual(hardware["cuda_runtime"], "13.0")
            self.assertIsNone(hardware["hip_runtime"])
            self.assertTrue(hardware["cuda_available"])
            self.assertTrue(manifest["requested_at_utc"].endswith("Z"))

    def test_rocm_manifest_records_hip_runtime_and_gcn_architecture(self) -> None:
        ultralytics, torch, _, instances = fake_modules(cuda=True, hip="7.13.0")
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "runs"
            with patch.dict(sys.modules, {"ultralytics": ultralytics, "torch": torch}):
                result = run(["--project", str(project), "--name", "rocm"])

            self.assertEqual(result, "trained")
            self.assertEqual(instances[0].train_kwargs["device"], "0")
            manifest = json.loads(
                (project / "rocm" / "training_request.json").read_text(
                    encoding="utf-8"
                )
            )
            hardware = manifest["hardware"]
            self.assertEqual(hardware["backend"], "rocm")
            self.assertEqual(hardware["hip_runtime"], "7.13.0")
            self.assertIsNone(hardware["cuda_runtime"])
            self.assertTrue(hardware["accelerator_available"])
            self.assertEqual(hardware["accelerator_device_count"], 1)
            device = hardware["accelerator_devices"][0]
            self.assertEqual(device["name"], "Fake GPU 0")
            self.assertEqual(device["gcn_arch_name"], "gfx1102")
            self.assertNotIn("compute_capability", device)

    def test_cpu_manifest_records_no_accelerator_runtime(self) -> None:
        ultralytics, torch, _, _ = fake_modules(cuda=False)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "runs"
            with patch.dict(sys.modules, {"ultralytics": ultralytics, "torch": torch}):
                run(
                    [
                        "--device",
                        "cpu",
                        "--allow-cpu-training",
                        "--project",
                        str(project),
                        "--name",
                        "cpu",
                    ]
                )

            manifest = json.loads(
                (project / "cpu" / "training_request.json").read_text(
                    encoding="utf-8"
                )
            )
            hardware = manifest["hardware"]
            self.assertEqual(hardware["backend"], "cpu")
            self.assertFalse(hardware["accelerator_available"])
            self.assertEqual(hardware["accelerator_device_count"], 0)
            self.assertEqual(hardware["accelerator_devices"], [])
            self.assertIsNone(hardware["cuda_runtime"])
            self.assertIsNone(hardware["hip_runtime"])

    def test_dataset_root_generates_absolute_run_local_yaml(self) -> None:
        ultralytics, torch, settings, instances = fake_modules(cuda=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = make_dataset(root / "data")
            project = root / "runs"
            with patch.dict(sys.modules, {"ultralytics": ultralytics, "torch": torch}):
                run(
                    [
                        "--project",
                        str(project),
                        "--dataset-root",
                        str(dataset_root),
                        "--allow-custom-dataset-size",
                    ]
                )

            yaml_path = (project / "kitti_yolo26s" / RUN_DATASET_CONFIG).resolve()
            self.assertTrue(yaml_path.is_file())
            self.assertEqual(instances[0].train_kwargs["data"], str(yaml_path))
            yaml_text = yaml_path.read_text(encoding="utf-8")
            self.assertIn(f'path: {json.dumps(dataset_root.resolve().as_posix())}', yaml_text)
            for index, name in enumerate(KITTI_CLASS_NAMES):
                self.assertIn(f"  {index}: {name}", yaml_text)
            self.assertEqual(settings.updates, [])

            manifest = json.loads(
                (project / "kitti_yolo26s" / "training_request.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["dataset"], str(yaml_path))
            self.assertEqual(manifest["train_kwargs"]["data"], str(yaml_path))
            inventory = manifest["dataset_provenance"]["inventory"]
            self.assertEqual(inventory["splits"]["train"]["images"]["count"], 1)
            self.assertEqual(inventory["splits"]["val"]["labels"]["count"], 1)
            self.assertEqual(len(inventory["fingerprint_sha256"]), 64)
            self.assertEqual(
                len(manifest["checkpoints"]["initialization"]["sha256"]), 64
            )

    def test_datasets_dir_alias_resolves_existing_kitti_child_without_settings(self) -> None:
        ultralytics, torch, settings, instances = fake_modules(cuda=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets_dir = root / "datasets"
            make_dataset(datasets_dir)
            with patch.dict(sys.modules, {"ultralytics": ultralytics, "torch": torch}):
                run(
                    [
                        "--project",
                        str(root / "runs"),
                        "--datasets-dir",
                        str(datasets_dir),
                        "--allow-custom-dataset-size",
                    ]
                )

            self.assertEqual(settings["datasets_dir"], "original-datasets")
            self.assertEqual(settings.updates, [])
            self.assertTrue(Path(instances[0].train_kwargs["data"]).is_absolute())

    def test_invalid_dataset_layout_is_rejected_before_output_or_model_creation(self) -> None:
        ultralytics, torch, _, instances = fake_modules(cuda=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "incomplete-kitti"
            (dataset_root / "images" / "train").mkdir(parents=True)
            project = root / "runs"
            with patch.dict(
                sys.modules, {"ultralytics": ultralytics, "torch": torch}
            ), self.assertRaisesRegex(FileNotFoundError, "labels.*train"):
                run(
                    [
                        "--project",
                        str(project),
                        "--dataset-root",
                        str(dataset_root),
                    ]
                )

            self.assertFalse(project.exists())
            self.assertEqual(instances, [])

    def test_unpaired_dataset_split_is_rejected(self) -> None:
        ultralytics, torch, _, instances = fake_modules(cuda=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = make_dataset(root)
            (dataset_root / "labels" / "val" / "000002.txt").unlink()
            with patch.dict(
                sys.modules, {"ultralytics": ultralytics, "torch": torch}
            ), self.assertRaisesRegex(ValueError, "val label directory is empty"):
                run(
                    [
                        "--dataset-root",
                        str(dataset_root),
                        "--allow-custom-dataset-size",
                    ]
                )

            self.assertEqual(instances, [])

    def test_nonempty_run_directory_is_not_overwritten(self) -> None:
        ultralytics, torch, _, instances = fake_modules(cuda=True)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "runs"
            run_dir = project / "existing"
            run_dir.mkdir(parents=True)
            (run_dir / "weights.pt").write_text("keep", encoding="utf-8")
            with patch.dict(
                sys.modules, {"ultralytics": ultralytics, "torch": torch}
            ), self.assertRaisesRegex(FileExistsError, "Choose a new --name"):
                run(["--project", str(project), "--name", "existing"])

            self.assertEqual(instances, [])
            self.assertEqual(
                (run_dir / "weights.pt").read_text(encoding="utf-8"), "keep"
            )

    def test_official_split_counts_are_enforced_unless_explicitly_overridden(self) -> None:
        ultralytics, torch, _, instances = fake_modules(cuda=True)
        with tempfile.TemporaryDirectory() as directory:
            dataset_root = make_dataset(Path(directory))
            with patch.dict(
                sys.modules, {"ultralytics": ultralytics, "torch": torch}
            ), self.assertRaisesRegex(ValueError, "official split requires 5,985"):
                run(["--dataset-root", str(dataset_root)])

            self.assertEqual(instances, [])

    def test_cpu_request_requires_opt_in_even_when_cuda_is_available(self) -> None:
        ultralytics, torch, _, instances = fake_modules(cuda=True)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                sys.modules, {"ultralytics": ultralytics, "torch": torch}
            ), self.assertRaisesRegex(RuntimeError, "--device cpu requires"):
                run(
                    [
                        "--device",
                        "cpu",
                        "--project",
                        str(Path(directory) / "runs"),
                    ]
                )
            self.assertEqual(instances, [])

    def test_accelerator_request_is_rejected_when_unavailable(self) -> None:
        ultralytics, torch, _, instances = fake_modules(cuda=False)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                sys.modules, {"ultralytics": ultralytics, "torch": torch}
            ), self.assertRaisesRegex(RuntimeError, "Accelerator device selector"):
                run(
                    [
                        "--device",
                        "0",
                        "--allow-cpu-training",
                        "--project",
                        str(Path(directory) / "runs"),
                    ]
                )
            self.assertEqual(instances, [])

    def test_available_mps_device_is_accepted_without_cpu_opt_in(self) -> None:
        ultralytics, torch, _, instances = fake_modules(cuda=False, mps=True)
        with tempfile.TemporaryDirectory() as directory:
            result = None
            with patch.dict(sys.modules, {"ultralytics": ultralytics, "torch": torch}):
                result = run(
                    [
                        "--device",
                        "mps",
                        "--project",
                        str(Path(directory) / "runs"),
                    ]
                )
            self.assertEqual(result, "trained")
            self.assertEqual(instances[0].train_kwargs["device"], "mps")

    def test_resume_uses_only_run_last_checkpoint_and_records_hash(self) -> None:
        ultralytics, torch, _, instances = fake_modules(cuda=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = make_dataset(root / "data")
            project = root / "runs"
            common = [
                "--dataset-root",
                str(dataset_root),
                "--allow-custom-dataset-size",
                "--project",
                str(project),
                "--name",
                "safe-resume",
            ]
            with patch.dict(sys.modules, {"ultralytics": ultralytics, "torch": torch}):
                run(common)
                last_checkpoint = project / "safe-resume" / "weights" / "last.pt"
                last_checkpoint.parent.mkdir()
                last_checkpoint.write_bytes(b"interrupted checkpoint")
                result = run([*common, "--resume"])

            self.assertEqual(result, "trained")
            self.assertEqual(len(instances), 2)
            self.assertEqual(instances[1].checkpoint, str(last_checkpoint.resolve()))
            self.assertEqual(instances[1].train_kwargs, {"resume": True})
            manifest = json.loads(
                (project / "safe-resume" / "training_request.json").read_text(
                    encoding="utf-8"
                )
            )
            resumes = manifest["checkpoints"]["resume_requests"]
            self.assertEqual(len(resumes), 1)
            self.assertEqual(resumes[0]["checkpoint"]["size_bytes"], 22)
            self.assertEqual(len(resumes[0]["checkpoint"]["sha256"]), 64)

    def test_resume_refuses_configuration_mismatch_without_loading_checkpoint(self) -> None:
        ultralytics, torch, _, instances = fake_modules(cuda=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "runs"
            initial = ["--project", str(project), "--name", "config", "--imgsz", "320"]
            with patch.dict(
                sys.modules, {"ultralytics": ultralytics, "torch": torch}
            ):
                run(initial)
                last_checkpoint = project / "config" / "weights" / "last.pt"
                last_checkpoint.parent.mkdir()
                last_checkpoint.write_bytes(b"last")
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    run(
                        [
                            "--project",
                            str(project),
                            "--name",
                            "config",
                            "--resume",
                        ]
                    )

            self.assertEqual(len(instances), 1)

    def test_resume_refuses_changed_dataset_inventory(self) -> None:
        ultralytics, torch, _, instances = fake_modules(cuda=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = make_dataset(root / "data")
            project = root / "runs"
            common = [
                "--dataset-root",
                str(dataset_root),
                "--allow-custom-dataset-size",
                "--project",
                str(project),
                "--name",
                "data-change",
            ]
            with patch.dict(sys.modules, {"ultralytics": ultralytics, "torch": torch}):
                run(common)
                last_checkpoint = project / "data-change" / "weights" / "last.pt"
                last_checkpoint.parent.mkdir()
                last_checkpoint.write_bytes(b"last")
                (dataset_root / "images" / "train" / "000001.png").write_bytes(
                    b"replacement image"
                )
                with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                    run([*common, "--resume"])

            self.assertEqual(len(instances), 1)

    def test_resume_requires_last_checkpoint(self) -> None:
        ultralytics, torch, _, instances = fake_modules(cuda=True)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "runs"
            common = ["--project", str(project), "--name", "missing-last"]
            with patch.dict(sys.modules, {"ultralytics": ultralytics, "torch": torch}):
                run(common)
                with self.assertRaisesRegex(FileNotFoundError, "Checkpoint does not exist"):
                    run([*common, "--resume"])

            self.assertEqual(len(instances), 1)


if __name__ == "__main__":
    unittest.main()

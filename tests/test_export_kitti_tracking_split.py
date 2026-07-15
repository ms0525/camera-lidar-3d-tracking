# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from export_kitti_tracking_split import (
    ASSOCIATION_POLICY,
    CODE_PROVENANCE_PATHS,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    TRACKED_CLASS_POLICY,
    build_parser,
    build_sequence_command,
    normalized_configuration,
    run,
    select_sequences,
)


def _arguments(root: Path, *extra: str) -> Namespace:
    return build_parser().parse_args(
        [
            "--dataset-root",
            str(root / "dataset"),
            "--output-dir",
            str(root / "predictions"),
            "--model",
            "remote-test-model.pt",
            *extra,
        ]
    )


def _successful_export(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    output = Path(command[command.index("--export-kitti") + 1])
    sequence = command[command.index("--sequence") + 1]
    output.write_text(f"prediction for {sequence}\n", encoding="ascii")
    return subprocess.CompletedProcess(command, 0)


class ExportKittiTrackingSplitTests(unittest.TestCase):
    def test_repository_presets_and_explicit_selection_are_normalized(self) -> None:
        all_sequences = select_sequences(
            Namespace(split_preset="all", sequences=None)
        )
        self.assertEqual(all_sequences, tuple(f"{index:04d}" for index in range(21)))

        tune = select_sequences(
            Namespace(split_preset="trackeval_tune", sequences=None)
        )
        validation = select_sequences(
            Namespace(split_preset="trackeval_val", sequences=None)
        )
        self.assertEqual(set(tune) | set(validation), set(all_sequences))
        self.assertFalse(set(tune) & set(validation))
        self.assertEqual(
            select_sequences(Namespace(split_preset=None, sequences=[0, "12"])),
            ("0000", "0012"),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_sequences(Namespace(split_preset=None, sequences=[0, "0000"]))

    def test_normalized_configuration_builds_all_passthrough_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            args = _arguments(
                root,
                "--sequences",
                "0",
                "--confidence",
                "0.2",
                "--class-confidence",
                "person=0.4",
                "--class-confidence",
                "car=0.3",
                "--imgsz",
                "960",
                "--no-yolo-end2end",
                "--device",
                "cpu",
                "--max-age",
                "8",
                "--n-init",
                "1",
                "--max-cosine-distance",
                "0.35",
                "--nn-budget",
                "120",
                "--embedder",
                "mobilenet",
                "--embedder-batch-size",
                "3",
                "--half",
                "--embedder-gpu",
            )
            configuration = normalized_configuration(args, ("0000",))
            command = build_sequence_command(
                configuration, "0000", root / "predictions" / "0000.txt"
            )

            self.assertEqual(configuration["class_confidence"], {"car": 0.3, "person": 0.4})
            self.assertEqual(
                configuration["association_policy"], ASSOCIATION_POLICY
            )
            self.assertEqual(
                configuration["tracked_class_policy"], TRACKED_CLASS_POLICY
            )
            self.assertEqual(configuration["embedder_batch_size"], 3)
            self.assertIn("--no-yolo-end2end", command)
            self.assertIn("--half", command)
            self.assertIn("--embedder-gpu", command)
            self.assertEqual(
                command[command.index("--embedder-batch-size") + 1], "3"
            )
            self.assertEqual(command[command.index("--device") + 1], "cpu")
            assignments = [
                command[index + 1]
                for index, token in enumerate(command)
                if token == "--class-confidence"
            ]
            self.assertEqual(assignments, ["car=0.3", "person=0.4"])

    def test_end_to_end_flag_is_only_forwarded_when_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace with spaces"
            (root / "dataset").mkdir(parents=True)

            default = normalized_configuration(
                _arguments(root, "--sequences", "0"), ("0000",)
            )
            default_command = build_sequence_command(
                default, "0000", root / "predictions" / "0000.txt"
            )
            self.assertNotIn("--yolo-end2end", default_command)
            self.assertNotIn("--no-yolo-end2end", default_command)
            self.assertIn(str((root / "dataset").resolve()), default_command)

            enabled = normalized_configuration(
                _arguments(root, "--sequences", "0", "--yolo-end2end"),
                ("0000",),
            )
            enabled_command = build_sequence_command(
                enabled, "0000", root / "predictions" / "0000.txt"
            )
            self.assertIn("--yolo-end2end", enabled_command)
            self.assertNotIn("--no-yolo-end2end", enabled_command)

    def test_model_argument_stays_stable_when_remote_weight_is_downloaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            previous = Path.cwd()
            try:
                os.chdir(root)
                args = _arguments(root, "--sequences", "0")
                before = normalized_configuration(args, ("0000",))
                Path("remote-test-model.pt").write_bytes(b"downloaded weights")
                after = normalized_configuration(args, ("0000",))
            finally:
                os.chdir(previous)

            self.assertEqual(before, after)
            self.assertEqual(after["model"], "remote-test-model.pt")

    def test_run_records_subprocess_outputs_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            args = _arguments(root, "--sequences", "0", "1")
            with patch(
                "export_kitti_tracking_split.subprocess.run",
                side_effect=_successful_export,
            ) as execute:
                self.assertEqual(run(args), 0)

            self.assertEqual(execute.call_count, 2)
            for call in execute.call_args_list:
                command = call.args[0]
                self.assertEqual(command[0], sys.executable)
                self.assertEqual(call.kwargs, {"check": False})
                self.assertIn("--headless", command)
                self.assertIn("--export-kitti", command)

            manifest = json.loads(
                (root / "predictions" / MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                manifest["configuration"]["association_policy"],
                ASSOCIATION_POLICY,
            )
            self.assertEqual(
                manifest["configuration"]["tracked_class_policy"],
                TRACKED_CLASS_POLICY,
            )
            self.assertEqual(
                manifest["configuration"]["embedder_batch_size"], 4
            )
            self.assertIn("python", manifest["runtime"])
            self.assertIn("ultralytics", manifest["runtime"])
            self.assertIn("torch", manifest["runtime"])
            self.assertIn("deep_sort_realtime", manifest["runtime"])
            self.assertIn("numpy", manifest["runtime"])
            self.assertIsInstance(manifest["runtime"]["opencv"], dict)
            self.assertEqual(
                set(manifest["code_provenance"]), set(CODE_PROVENANCE_PATHS)
            )
            for entry in manifest["code_provenance"].values():
                self.assertGreater(entry["size_bytes"], 0)
                self.assertEqual(len(entry["sha256"]), 64)
            for sequence in ("0000", "0001"):
                entry = manifest["sequence_results"][sequence]
                self.assertEqual(entry["status"], "completed")
                self.assertEqual(entry["attempt_count"], 1)
                self.assertEqual(len(entry["output_sha256"]), 64)

    def test_resume_skips_verified_files_and_reruns_only_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            initial = _arguments(root, "--sequences", "0", "1")
            with patch(
                "export_kitti_tracking_split.subprocess.run",
                side_effect=_successful_export,
            ):
                self.assertEqual(run(initial), 0)

            resume = _arguments(root, "--sequences", "0", "1", "--resume")
            with patch("export_kitti_tracking_split.subprocess.run") as execute:
                self.assertEqual(run(resume), 0)
                execute.assert_not_called()

            (root / "predictions" / "0001.txt").write_text(
                "modified\n", encoding="ascii"
            )
            with patch(
                "export_kitti_tracking_split.subprocess.run",
                side_effect=_successful_export,
            ) as execute:
                self.assertEqual(run(resume), 0)
            self.assertEqual(execute.call_count, 1)
            command = execute.call_args.args[0]
            self.assertEqual(command[command.index("--sequence") + 1], "0001")
            manifest = json.loads(
                (root / "predictions" / MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["sequence_results"]["0000"]["attempt_count"], 1)
            self.assertEqual(manifest["sequence_results"]["0001"]["attempt_count"], 2)

    def test_retry_does_not_accept_a_stale_final_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            initial = _arguments(root, "--sequences", "0")
            with patch(
                "export_kitti_tracking_split.subprocess.run",
                side_effect=_successful_export,
            ):
                self.assertEqual(run(initial), 0)

            final_output = root / "predictions" / "0000.txt"
            final_output.write_text("stale output\n", encoding="ascii")
            resume = _arguments(root, "--sequences", "0", "--resume")
            success_without_output = subprocess.CompletedProcess([], 0)
            with patch(
                "export_kitti_tracking_split.subprocess.run",
                return_value=success_without_output,
            ):
                self.assertEqual(run(resume), 1)

            self.assertEqual(final_output.read_text(encoding="ascii"), "stale output\n")
            manifest = json.loads(
                (root / "predictions" / MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertIn(
                "without creating a new attempt output",
                manifest["sequence_results"]["0000"]["error"],
            )

    def test_child_output_can_be_logged_per_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            args = _arguments(root, "--sequences", "0", "--log-child-output")

            def export_with_log(
                command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                log = kwargs["stdout"]
                log.write("frame progress\n")  # type: ignore[union-attr]
                return _successful_export(command, **kwargs)

            with patch(
                "export_kitti_tracking_split.subprocess.run",
                side_effect=export_with_log,
            ) as execute:
                self.assertEqual(run(args), 0)

            self.assertIs(execute.call_args.kwargs["stderr"], subprocess.STDOUT)
            log_file = root / "predictions" / "logs" / "0000.attempt-1.log"
            self.assertEqual(log_file.read_text(encoding="utf-8"), "frame progress\n")
            manifest = json.loads(
                (root / "predictions" / MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["sequence_results"]["0000"]["log_file"], str(log_file)
            )

    def test_resume_rejects_configuration_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            with patch(
                "export_kitti_tracking_split.subprocess.run",
                side_effect=_successful_export,
            ):
                self.assertEqual(run(_arguments(root, "--sequences", "0")), 0)

            changed = _arguments(
                root, "--sequences", "0", "--confidence", "0.5", "--resume"
            )
            with patch("export_kitti_tracking_split.subprocess.run") as execute:
                with self.assertRaisesRegex(ValueError, "exactly match"):
                    run(changed)
                execute.assert_not_called()

    def test_resume_rejects_old_manifest_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            initial = _arguments(root, "--sequences", "0")
            with patch(
                "export_kitti_tracking_split.subprocess.run",
                side_effect=_successful_export,
            ):
                self.assertEqual(run(initial), 0)

            manifest_path = root / "predictions" / MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = SCHEMA_VERSION - 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            resume = _arguments(root, "--sequences", "0", "--resume")
            with patch("export_kitti_tracking_split.subprocess.run") as execute:
                with self.assertRaisesRegex(ValueError, "unsupported.*schema"):
                    run(resume)
                execute.assert_not_called()

    def test_resume_rejects_code_provenance_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            initial = _arguments(root, "--sequences", "0")
            with patch(
                "export_kitti_tracking_split.subprocess.run",
                side_effect=_successful_export,
            ):
                self.assertEqual(run(initial), 0)

            changed_provenance = {
                "track_sequence.py": {"size_bytes": 1, "sha256": "0" * 64}
            }
            resume = _arguments(root, "--sequences", "0", "--resume")
            with (
                patch(
                    "export_kitti_tracking_split.code_provenance",
                    return_value=changed_provenance,
                ),
                patch("export_kitti_tracking_split.subprocess.run") as execute,
            ):
                with self.assertRaisesRegex(ValueError, "code provenance changed"):
                    run(resume)
                execute.assert_not_called()

    def test_resume_rejects_runtime_environment_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            initial = _arguments(root, "--sequences", "0")
            with patch(
                "export_kitti_tracking_split.subprocess.run",
                side_effect=_successful_export,
            ):
                self.assertEqual(run(initial), 0)

            manifest = json.loads(
                (root / "predictions" / MANIFEST_NAME).read_text(encoding="utf-8")
            )
            changed_runtime = dict(manifest["runtime"])
            changed_runtime["numpy"] = "changed-test-version"
            resume = _arguments(root, "--sequences", "0", "--resume")
            with (
                patch(
                    "export_kitti_tracking_split.runtime_versions",
                    return_value=changed_runtime,
                ),
                patch("export_kitti_tracking_split.subprocess.run") as execute,
            ):
                with self.assertRaisesRegex(ValueError, "runtime environment changed"):
                    run(resume)
                execute.assert_not_called()

    def test_failure_stops_and_remains_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            failure = subprocess.CompletedProcess([], 7)
            with patch(
                "export_kitti_tracking_split.subprocess.run", return_value=failure
            ) as execute:
                self.assertEqual(run(_arguments(root, "--sequences", "0", "1")), 7)
            self.assertEqual(execute.call_count, 1)
            manifest = json.loads(
                (root / "predictions" / MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["sequence_results"]["0000"]["status"], "failed")
            self.assertEqual(manifest["sequence_results"]["0001"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path

from scripts.audit_public_release import Finding, REQUIRED_PUBLIC_FILES, audit


def _write_text(root: Path, relative: str, content: str = "public test fixture\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_bytes(root: Path, relative: str, content: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _create_required_files(root: Path) -> None:
    for relative in REQUIRED_PUBLIC_FILES:
        _write_text(root, relative)


def _findings_by_path(findings: Iterable[Finding]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for finding in findings:
        result.setdefault(finding.path, set()).add(finding.category)
    return result


class PublicReleaseAuditTests(unittest.TestCase):
    def test_clean_candidates_allow_reviewed_media_and_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_required_files(root)

            # A null byte makes these unmistakably binary to the text scanner;
            # the audit's explicit reviewed-media allow-list is what permits them.
            png_fixture = b"\x89PNG\r\n\x1a\n\x00public-test-fixture"
            _write_bytes(root, "docs/assets/dashboard_overview.png", png_fixture)
            _write_bytes(root, "docs/assets/tracking_metrics_comparison.png", png_fixture)
            _write_text(root, ".env.example", "API_KEY=<your_api_key>\n")
            _write_text(root, "config.example", 'password = "changeme-before-use"\n')

            source, files, findings = audit(root)

            self.assertEqual(source, "filesystem candidates (root .gitignore applied)")
            self.assertGreaterEqual(len(files), len(REQUIRED_PUBLIC_FILES) + 4)
            self.assertEqual(findings, [])

    def test_missing_and_ignored_required_files_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_required_files(root)
            (root / "LICENSE").unlink()
            _write_text(root, ".gitignore", "/NOTICE\n")

            _, _, findings = audit(root)
            by_path = _findings_by_path(findings)

            self.assertEqual(by_path["LICENSE"], {"required"})
            self.assertEqual(by_path["NOTICE"], {"required"})
            details = {finding.path: finding.detail for finding in findings}
            self.assertIn("missing", details["LICENSE"])
            self.assertIn("not tracked/included", details["NOTICE"])

    def test_forbidden_models_kitti_media_archives_and_secrets_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_required_files(root)

            _write_bytes(root, "models/custom.pt", b"model fixture")
            _write_bytes(root, "training/velodyne/000001.bin", b"point cloud fixture")
            _write_bytes(root, "docs/unreviewed.png", b"\x89PNG\r\n\x1a\n\x00fixture")
            _write_bytes(root, "release.zip", b"archive fixture")
            _write_text(root, ".env", "SAFE_TEST_VALUE=fixture\n")

            # Build a credential only in the temporary fixture. Keeping its
            # signature split here prevents the repository audit from treating
            # this test source itself as a leaked credential.
            credential_name = "pass" + "word"
            credential_value = "correct-horse-battery-staple"
            _write_text(
                root,
                "config/private-settings.txt",
                f"{credential_name} = {credential_value}\n",
            )

            _, _, findings = audit(root)
            by_path = _findings_by_path(findings)

            self.assertEqual(by_path["models/custom.pt"], {"model"})
            self.assertEqual(by_path["training/velodyne/000001.bin"], {"kitti-data"})
            self.assertEqual(by_path["docs/unreviewed.png"], {"media"})
            self.assertEqual(by_path["release.zip"], {"archive"})
            self.assertEqual(by_path[".env"], {"secret-file"})
            self.assertEqual(by_path["config/private-settings.txt"], {"credential"})

    def test_filesystem_candidates_respect_root_gitignore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_required_files(root)
            _write_text(
                root,
                ".gitignore",
                "/private-models/\n/data/images/\n",
            )
            _write_bytes(root, "private-models/ignored.pt", b"ignored model")
            _write_bytes(root, "data/images/000000.png", b"ignored KITTI image")
            _write_bytes(root, "visible.pt", b"visible model")

            _, files, findings = audit(root)
            relative_files = {path.relative_to(root).as_posix() for path in files}
            by_path = _findings_by_path(findings)

            self.assertNotIn("private-models/ignored.pt", relative_files)
            self.assertNotIn("data/images/000000.png", relative_files)
            self.assertEqual(by_path, {"visible.pt": {"model"}})

    @unittest.skipUnless(shutil.which("git"), "Git executable is not available")
    def test_git_mode_checks_tracked_and_unignored_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_required_files(root)
            _write_text(root, ".gitignore", "/ignored*.pt\n")
            _write_bytes(root, "tracked-ignored.pt", b"tracked ignored model")

            subprocess.run(
                ["git", "init", "--quiet"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "add", "."],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "add", "--force", "tracked-ignored.pt"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            _write_bytes(root, "ignored-untracked.pt", b"ignored model")
            _write_bytes(root, "visible-untracked.pt", b"visible model")

            source, files, findings = audit(root)
            relative_files = {path.relative_to(root).as_posix() for path in files}
            by_path = _findings_by_path(findings)

            self.assertEqual(source, "Git-tracked and unignored files")
            self.assertIn("tracked-ignored.pt", relative_files)
            self.assertIn("visible-untracked.pt", relative_files)
            self.assertNotIn("ignored-untracked.pt", relative_files)
            self.assertEqual(
                by_path,
                {
                    "tracked-ignored.pt": {"model"},
                    "visible-untracked.pt": {"model"},
                },
            )


if __name__ == "__main__":
    unittest.main()

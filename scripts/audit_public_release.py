#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Fail when files intended for a public release contain private artifacts.

The audit examines Git-tracked and unignored files when run in a repository.
Before the repository is initialized, it applies the root ``.gitignore``
itself so the same command is useful while preparing a release.

This is deliberately a conservative, dependency-free guard. It complements
GitHub secret scanning; it is not a replacement for rotating a credential
that has already been committed or shared.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PUBLIC_FILES = (
    "LICENSE",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "app/requirements.txt",
    "docs/PUBLIC_RELEASE.md",
)

MODEL_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".engine",
    ".h5",
    ".onnx",
    ".pb",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
    ".torchscript",
    ".mlpackage",
    ".weights",
}
ARCHIVE_SUFFIXES = {
    ".7z",
    ".bz2",
    ".docx",
    ".gz",
    ".pptx",
    ".rar",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
}
DATA_SUFFIXES = {".npy", ".npz", ".pcd", ".ply"}
MEDIA_SUFFIXES = {
    ".avi",
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mkv",
    ".mov",
    ".mp4",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
APPROVED_PUBLIC_MEDIA = {
    "docs/assets/dashboard_overview.png",
    "docs/assets/tracking_metrics_comparison.png",
}
SECRET_FILE_NAMES = {
    ".env",
    "credentials.json",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
    "kubeconfig",
    "secrets.toml",
    "service-account.json",
    "service_account.json",
}
SECRET_FILE_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
SAFE_ENV_FILE_NAMES = {".env.example", ".env.sample", ".env.template"}

RAW_KITTI_PREFIXES = (
    "data/calib/",
    "data/images/",
    "data/labels/",
    "data/pointcloud/",
    "datasets/",
)
RAW_KITTI_DIRECTORY_PATTERN = re.compile(
    r"(?:^|/)(?:training|testing)/(?:calib|image_02|image_03|label_02|"
    r"velodyne|velodyne_points)(?:/|$)",
    re.IGNORECASE,
)

MAX_TEXT_SCAN_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class Finding:
    category: str
    path: str
    detail: str


@dataclass(frozen=True)
class IgnoreRule:
    pattern: str
    negated: bool
    anchored: bool
    directory_only: bool


@dataclass(frozen=True)
class SecretPattern:
    name: str
    expression: re.Pattern[str]


SECRET_PATTERNS = (
    SecretPattern(
        "private key",
        re.compile(r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----"),
    ),
    SecretPattern(
        "AWS access key",
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ),
    SecretPattern(
        "GitHub access token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"),
    ),
    SecretPattern(
        "GitLab access token",
        re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    ),
    SecretPattern(
        "Google API key",
        re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ),
    SecretPattern(
        "OpenAI-style API key",
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    SecretPattern(
        "Slack token",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    ),
    SecretPattern(
        "credential embedded in URL",
        re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@", re.IGNORECASE),
    ),
)

GENERIC_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?im)\b(?:api[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|"
    r"password)\b\s*[:=]\s*[\"']?(?P<value>[^\s\"'`,;#]{8,})"
)
PLACEHOLDER_FRAGMENTS = (
    "${",
    "$env:",
    "changeme",
    "dummy",
    "example",
    "getenv",
    "none",
    "not-set",
    "placeholder",
    "redacted",
    "replace",
    "secret_here",
    "your-",
    "your_",
    "xxxxx",
)


def _run_git(root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(arguments, 127, b"", b"git not found")


def _git_candidate_files(root: Path) -> list[Path] | None:
    probe = _run_git(root, ("rev-parse", "--is-inside-work-tree"))
    if probe.returncode != 0 or probe.stdout.strip() != b"true":
        return None

    result = _run_git(
        root,
        ("ls-files", "-z", "--cached", "--others", "--exclude-standard"),
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git ls-files failed: {message}")

    paths: list[Path] = []
    for encoded_path in result.stdout.split(b"\0"):
        if not encoded_path:
            continue
        relative = encoded_path.decode("utf-8", errors="surrogateescape")
        candidate = root / relative
        if candidate.is_file() or candidate.is_symlink():
            paths.append(candidate)
    return sorted(paths, key=lambda path: path.as_posix().lower())


def _load_ignore_rules(root: Path) -> list[IgnoreRule]:
    ignore_file = root / ".gitignore"
    if not ignore_file.is_file():
        return []

    rules: list[IgnoreRule] = []
    for raw_line in ignore_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        if not line:
            continue
        directory_only = line.endswith("/")
        anchored = line.startswith("/")
        line = line.strip("/")
        if line:
            rules.append(IgnoreRule(line, negated, anchored, directory_only))
    return rules


def _rule_matches(rule: IgnoreRule, relative_path: str, is_directory: bool) -> bool:
    parts = relative_path.strip("/").split("/")
    directory_count = len(parts) if is_directory else max(len(parts) - 1, 0)
    directory_prefixes = ["/".join(parts[:index]) for index in range(1, directory_count + 1)]

    if rule.directory_only:
        candidates = directory_prefixes
    else:
        candidates = [relative_path, *directory_prefixes]

    if rule.anchored or "/" in rule.pattern:
        return any(fnmatch.fnmatchcase(candidate, rule.pattern) for candidate in candidates)

    return any(
        fnmatch.fnmatchcase(component, rule.pattern)
        for candidate in candidates
        for component in candidate.split("/")
    )


def _is_ignored(relative_path: str, is_directory: bool, rules: Sequence[IgnoreRule]) -> bool:
    ignored = False
    for rule in rules:
        if _rule_matches(rule, relative_path, is_directory):
            ignored = not rule.negated
    return ignored


def _may_contain_reincluded_path(
    relative_directory: str,
    rules: Sequence[IgnoreRule],
) -> bool:
    """Return whether a negation rule could re-include a descendant.

    An unanchored negation may match below any directory, so it prevents safe
    pruning. An anchored root-file rule (for example ``!/.env.example``) cannot
    match below an unrelated ignored directory and should not force a walk of
    a large local virtual environment.
    """

    prefix = f"{relative_directory.strip('/')}/"
    for rule in rules:
        if not rule.negated:
            continue
        if not rule.anchored:
            return True
        if "/" in rule.pattern and rule.pattern.startswith(prefix):
            return True
    return False


def _candidate_files(root: Path) -> list[Path]:
    rules = _load_ignore_rules(root)
    candidates: list[Path] = []

    for directory, child_directories, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(root).as_posix()

        retained_directories: list[str] = []
        for child_name in child_directories:
            if child_name == ".git":
                continue
            relative = child_name if relative_directory == "." else f"{relative_directory}/{child_name}"
            ignored = _is_ignored(relative, True, rules)
            if not ignored or _may_contain_reincluded_path(relative, rules):
                retained_directories.append(child_name)
        child_directories[:] = retained_directories

        for file_name in file_names:
            path = directory_path / file_name
            relative = path.relative_to(root).as_posix()
            if not _is_ignored(relative, False, rules):
                candidates.append(path)

    return sorted(candidates, key=lambda path: path.as_posix().lower())


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _compound_suffix(path: Path) -> str:
    lower_name = path.name.lower()
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if lower_name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def _artifact_finding(path: Path, root: Path) -> Finding | None:
    relative = _relative(path, root)
    lower_relative = relative.lower()
    lower_name = path.name.lower()
    suffix = _compound_suffix(path)

    if lower_name in SAFE_ENV_FILE_NAMES:
        pass
    elif lower_name in SECRET_FILE_NAMES or suffix in SECRET_FILE_SUFFIXES:
        return Finding("secret-file", relative, "credential/private-key file must not be public")

    if any(lower_relative.startswith(prefix) for prefix in RAW_KITTI_PREFIXES):
        return Finding("kitti-data", relative, "raw KITTI sample/data path must not be public")
    if RAW_KITTI_DIRECTORY_PATTERN.search(lower_relative):
        return Finding("kitti-data", relative, "raw KITTI directory layout must not be public")

    if suffix in MODEL_SUFFIXES:
        return Finding("model", relative, f"model or binary artifact ({suffix}) must be supplied locally")
    if suffix in ARCHIVE_SUFFIXES or suffix.startswith(".tar."):
        return Finding("archive", relative, f"archive/Office artifact ({suffix}) is not allowed")
    if suffix in DATA_SUFFIXES:
        return Finding("dataset", relative, f"serialized data artifact ({suffix}) is not allowed")
    if suffix in MEDIA_SUFFIXES and relative not in APPROVED_PUBLIC_MEDIA:
        return Finding(
            "media",
            relative,
            "unreviewed media is not allowed; add only a deliberately reviewed public asset",
        )
    return None


def _read_text_for_scan(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_TEXT_SCAN_BYTES:
            return None
        content = path.read_bytes()
    except OSError:
        return None
    if b"\0" in content[:8192]:
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _looks_like_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized.startswith(("<", "{{", "%(", "os.environ", "settings.")):
        return True
    if set(normalized) <= {"*", "x", "-", "_"}:
        return True
    return any(fragment in normalized for fragment in PLACEHOLDER_FRAGMENTS)


def _secret_findings(path: Path, root: Path) -> Iterable[Finding]:
    text = _read_text_for_scan(path)
    if text is None:
        return
    relative = _relative(path, root)

    for pattern in SECRET_PATTERNS:
        match = pattern.expression.search(text)
        if match:
            yield Finding(
                "credential",
                relative,
                f"possible {pattern.name} at line {_line_number(text, match.start())}",
            )

    for match in GENERIC_CREDENTIAL_ASSIGNMENT.finditer(text):
        value = match.group("value")
        if not _looks_like_placeholder(value):
            yield Finding(
                "credential",
                relative,
                f"possible hard-coded credential at line {_line_number(text, match.start())}",
            )
            break


def audit(root: Path) -> tuple[str, list[Path], list[Finding]]:
    candidates = _git_candidate_files(root)
    if candidates is None:
        source = "filesystem candidates (root .gitignore applied)"
        files = _candidate_files(root)
    else:
        source = "Git-tracked and unignored files"
        files = candidates

    findings: list[Finding] = []
    included = {_relative(path, root) for path in files}
    for required in REQUIRED_PUBLIC_FILES:
        required_path = root / required
        if not required_path.is_file():
            findings.append(Finding("required", required, "required public-release file is missing"))
        elif required not in included:
            findings.append(Finding("required", required, "required public-release file is not tracked/included"))

    for path in files:
        artifact = _artifact_finding(path, root)
        if artifact is not None:
            findings.append(artifact)
            continue
        findings.extend(_secret_findings(path, root))

    unique_findings = sorted(
        set(findings),
        key=lambda item: (item.path.lower(), item.category, item.detail),
    )
    return source, files, unique_findings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit files intended for the public portfolio release.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="project root to inspect (default: the parent of this script's directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    root = arguments.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: project root does not exist: {root}", file=sys.stderr)
        return 2

    try:
        source, files, findings = audit(root)
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print("Public release audit")
    print(f"  Root:   {root}")
    print(f"  Source: {source}")
    print(f"  Files:  {len(files)} inspected")

    if findings:
        print(f"\nFAILED: {len(findings)} issue(s) found")
        for finding in findings:
            print(f"  - [{finding.category}] {finding.path}: {finding.detail}")
        print("\nKeep private artifacts outside Git or add precise root-level .gitignore rules.")
        return 1

    print("\nPASS: no forbidden datasets, models, archives, or likely credentials found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

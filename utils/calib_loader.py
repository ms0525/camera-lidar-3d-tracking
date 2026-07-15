# SPDX-License-Identifier: AGPL-3.0-only
"""Validated loader for KITTI calibration text files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


EXPECTED_VALUE_COUNTS = {
    "P0": 12,
    "P1": 12,
    "P2": 12,
    "P3": 12,
    "R0_rect": 9,
    "R_rect": 9,
    "Tr_velo_to_cam": 12,
    "Tr_velo_cam": 12,
    "Tr_imu_to_velo": 12,
    "Tr_imu_velo": 12,
}
REQUIRED_KEYS = ("P2", "R0_rect", "Tr_velo_to_cam")
KEY_ALIASES = {
    # KITTI tracking files use these shorter names; the object-detection files
    # bundled with this project use the canonical names above.
    "R_rect": "R0_rect",
    "Tr_velo_cam": "Tr_velo_to_cam",
    "Tr_imu_velo": "Tr_imu_to_velo",
}


def load_calibration(calib_path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a KITTI calibration file.

    Values remain flat arrays because KITTI keys have different shapes. Use
    :func:`utils.geometry.load_kitti_transforms` to obtain consistently shaped
    camera/LiDAR matrices.
    """

    path = Path(calib_path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: {path}")

    calibration: dict[str, np.ndarray] = {}
    with path.open("r", encoding="ascii") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            # KITTI object calibration files normally use ``key: values``.
            # The tracking archive mixes that form (P0..P3) with whitespace-
            # separated rows such as ``R_rect values`` and
            # ``Tr_velo_cam values``.
            if ":" in line:
                key, raw_values = line.split(":", 1)
                key = key.strip()
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"{path}:{line_number}: expected a calibration key and values"
                    )
                key, raw_values = parts
            try:
                values = np.asarray(
                    [float(value) for value in raw_values.split()], dtype=np.float64
                )
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: non-numeric calibration value for {key!r}"
                ) from exc

            expected = EXPECTED_VALUE_COUNTS.get(key)
            if expected is not None and values.size != expected:
                raise ValueError(
                    f"{path}:{line_number}: {key!r} has {values.size} values; "
                    f"expected {expected}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"{path}:{line_number}: {key!r} contains NaN or infinity")
            calibration[key] = values

    for source_key, canonical_key in KEY_ALIASES.items():
        if canonical_key not in calibration and source_key in calibration:
            calibration[canonical_key] = calibration[source_key].copy()

    missing = [key for key in REQUIRED_KEYS if key not in calibration]
    if missing:
        raise ValueError(f"{path}: missing required calibration keys: {', '.join(missing)}")
    return calibration

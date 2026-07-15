# SPDX-License-Identifier: AGPL-3.0-only
"""Parse and apply per-model-class detection confidence thresholds."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def parse_class_confidence_overrides(
    assignments: Sequence[str] | None,
) -> dict[str, float]:
    """Parse repeatable ``CLASS=VALUE`` CLI assignments into a normalized map."""

    overrides: dict[str, float] = {}
    for assignment in assignments or ():
        if "=" not in assignment:
            raise ValueError(
                f"invalid --class-confidence {assignment!r}; expected CLASS=VALUE"
            )
        raw_name, raw_value = assignment.split("=", 1)
        class_name = raw_name.strip().casefold()
        if not class_name:
            raise ValueError("--class-confidence requires a non-empty class name")
        if class_name in overrides:
            raise ValueError(
                f"duplicate --class-confidence override for class {class_name!r}"
            )
        try:
            threshold = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"confidence for class {class_name!r} must be numeric; got {raw_value!r}"
            ) from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"confidence for class {class_name!r} must be between 0 and 1"
            )
        overrides[class_name] = threshold
    return overrides


def confidence_for_class(
    default_threshold: float,
    overrides: Mapping[str, float] | None,
    class_name: str,
) -> float:
    """Return the effective post-detection threshold for one model class."""

    if not overrides:
        return float(default_threshold)
    return float(overrides.get(class_name.strip().casefold(), default_threshold))


def model_inference_threshold(
    default_threshold: float, overrides: Mapping[str, float] | None
) -> float:
    """Return the lowest threshold that must be requested from the detector."""

    values = [float(default_threshold)]
    if overrides:
        values.extend(float(value) for value in overrides.values())
    return min(values)


def model_class_names(names: Any) -> set[str]:
    """Normalize Ultralytics' dict/list class-name representation."""

    values = names.values() if isinstance(names, dict) else names
    try:
        return {str(value).strip().casefold() for value in values}
    except TypeError as exc:
        raise ValueError("model class names must be a mapping or iterable") from exc


def validate_overrides_for_model(
    overrides: Mapping[str, float], names: Any
) -> None:
    """Reject silent typos such as KITTI ``pedestrian`` for COCO ``person``."""

    available = model_class_names(names)
    unknown = sorted(set(overrides) - available)
    if unknown:
        raise ValueError(
            "--class-confidence names are not present in the model: "
            f"{', '.join(unknown)}. Use the exact model class names: COCO "
            "models use 'person', while a KITTI-fine-tuned model may use "
            "'pedestrian'."
        )


__all__ = [
    "confidence_for_class",
    "model_class_names",
    "model_inference_threshold",
    "parse_class_confidence_overrides",
    "validate_overrides_for_model",
]

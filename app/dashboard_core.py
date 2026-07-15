# SPDX-License-Identifier: AGPL-3.0-only
"""Pure rendering and navigation helpers for the Streamlit dashboard.

This module deliberately avoids importing Ultralytics, Deep SORT, Open3D, or
the KITTI loader.  The hosted portfolio preview can therefore run with a small
CPU-only dependency set, while the local dashboard adapts the existing live
pipeline into the view dataclasses below.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont


DEFAULT_FRAME_COUNT = 36
DEFAULT_X_RANGE = (-5.0, 80.0)
DEFAULT_Y_RANGE = (-40.0, 40.0)
DEFAULT_Z_RANGE = (-3.0, 5.0)

BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


@dataclass(frozen=True, slots=True)
class TrackView:
    """One model-local track rendered in the camera and LiDAR panels."""

    track_id: str
    class_name: str | None
    bbox_xyxy: tuple[float, float, float, float]
    center_velodyne: tuple[float, float, float] | None
    dimensions_lwh: tuple[float, float, float]
    color_rgb: tuple[float, float, float]
    source: str
    confidence: float = 1.0
    lidar_point_count: int = 0


@dataclass(frozen=True, slots=True)
class GroundTruthView:
    """Optional exact reference box used only when labels are available."""

    track_id: int
    object_type: str
    center_velodyne: tuple[float, float, float]
    rotation_velodyne: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    extent_lwh: tuple[float, float, float]
    bbox_xyxy: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class AnalysisView:
    """Model-specific annotations for one synchronized frame."""

    frame_name: str
    annotations: tuple[TrackView, ...] = ()
    ground_truth: tuple[GroundTruthView, ...] = ()
    camera_origin_velodyne: tuple[float, float, float] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DashboardFrame:
    """Shared camera/LiDAR data and the two model-specific analyses."""

    frame_index: int
    frame_count: int
    frame_name: str
    image_rgb: np.ndarray
    points_xyzi: np.ndarray | None
    yolo11: AnalysisView
    yolo26: AnalysisView
    source_label: str


@dataclass(frozen=True, slots=True)
class PlaybackStep:
    index: int
    playing: bool


def clamp_frame_index(index: int, frame_count: int) -> int:
    """Clamp an arbitrary index to the valid frame range.

    An empty source has no valid frame; returning zero keeps Streamlit state
    simple while the caller displays its source-validation error.
    """

    if frame_count <= 0:
        return 0
    return max(0, min(int(index), frame_count - 1))


def move_frame_index(index: int, delta: int, frame_count: int) -> int:
    """Move manually without wrapping at either end of the sequence."""

    return clamp_frame_index(int(index) + int(delta), frame_count)


def playback_step(index: int, frame_count: int, *, loop: bool) -> PlaybackStep:
    """Advance autoplay once and report whether playback should continue."""

    if frame_count <= 0:
        return PlaybackStep(0, False)
    current = clamp_frame_index(index, frame_count)
    if current < frame_count - 1:
        return PlaybackStep(current + 1, True)
    if loop and frame_count > 1:
        return PlaybackStep(0, True)
    return PlaybackStep(current, False)


def autoplay_interval_seconds(frames_per_second: float) -> float:
    if not math.isfinite(frames_per_second) or frames_per_second <= 0.0:
        raise ValueError("frames_per_second must be a finite positive number")
    return 1.0 / float(frames_per_second)


def seekable_frame_limit(
    frame_count: int,
    *,
    processed_through: int | None = None,
) -> int:
    """Return the largest one-based frame number currently safe to seek.

    Non-temporal sources expose every frame. A temporal live source exposes its
    cached frames plus one unseen frame so Deep SORT cannot be skipped ahead.
    Before the first live frame is processed, only frame 1 is available.
    """

    if frame_count <= 0:
        return 0
    if processed_through is None:
        return int(frame_count)
    return min(int(frame_count), max(1, int(processed_through) + 2))


def track_color(key: str) -> tuple[float, float, float]:
    """Return a stable, bright RGB color in Plotly's 0..1 convention."""

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    raw = np.frombuffer(digest[:3], dtype=np.uint8).astype(np.float64) / 255.0
    bright = 0.30 + 0.70 * raw
    return tuple(float(value) for value in bright)


def rgb_to_css(color_rgb: Sequence[float]) -> str:
    values = np.clip(np.asarray(color_rgb, dtype=np.float64), 0.0, 1.0)
    if values.shape != (3,):
        raise ValueError("color_rgb must contain exactly three values")
    red, green, blue = (int(round(value * 255.0)) for value in values)
    return f"#{red:02X}{green:02X}{blue:02X}"


def oriented_box_corners(
    center: Sequence[float],
    extent_lwh: Sequence[float],
    rotation: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return the eight world-space corners of an oriented 3D box."""

    center_array = np.asarray(center, dtype=np.float64)
    extent_array = np.asarray(extent_lwh, dtype=np.float64)
    rotation_array = np.asarray(rotation, dtype=np.float64)
    if center_array.shape != (3,):
        raise ValueError("center must have shape (3,)")
    if extent_array.shape != (3,) or np.any(extent_array <= 0.0):
        raise ValueError("extent_lwh must contain three positive values")
    if rotation_array.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")

    half = extent_array / 2.0
    local = np.asarray(
        (
            (-half[0], -half[1], -half[2]),
            (half[0], -half[1], -half[2]),
            (half[0], half[1], -half[2]),
            (-half[0], half[1], -half[2]),
            (-half[0], -half[1], half[2]),
            (half[0], -half[1], half[2]),
            (half[0], half[1], half[2]),
            (-half[0], half[1], half[2]),
        ),
        dtype=np.float64,
    )
    return local @ rotation_array.T + center_array


def box_line_coordinates(corners: np.ndarray) -> tuple[list[float | None], ...]:
    """Convert eight corners into Plotly line coordinates with separators."""

    corner_array = np.asarray(corners, dtype=np.float64)
    if corner_array.shape != (8, 3):
        raise ValueError("corners must have shape (8, 3)")
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for start, end in BOX_EDGES:
        for axis_values, axis in (
            (x_values, 0),
            (y_values, 1),
            (z_values, 2),
        ):
            axis_values.extend(
                (float(corner_array[start, axis]), float(corner_array[end, axis]), None)
            )
    return x_values, y_values, z_values


def downsample_lidar_for_plot(
    points_xyzi: np.ndarray | None,
    *,
    x_range: tuple[float, float] = DEFAULT_X_RANGE,
    y_range: tuple[float, float] = DEFAULT_Y_RANGE,
    z_range: tuple[float, float] = DEFAULT_Z_RANGE,
    budget: int = 15_000,
) -> np.ndarray:
    """Filter the display ROI and stride deterministically to a point budget."""

    if budget < 1:
        raise ValueError("budget must be at least one")
    if points_xyzi is None:
        return np.empty((0, 4), dtype=np.float32)
    points = np.asarray(points_xyzi)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points_xyzi must have shape (N, 3+) ")
    if points.shape[0] == 0:
        return points.copy()

    xyz = points[:, :3]
    finite = np.isfinite(xyz).all(axis=1)
    keep = (
        finite
        & (xyz[:, 0] >= x_range[0])
        & (xyz[:, 0] <= x_range[1])
        & (xyz[:, 1] >= y_range[0])
        & (xyz[:, 1] <= y_range[1])
        & (xyz[:, 2] >= z_range[0])
        & (xyz[:, 2] <= z_range[1])
    )
    filtered = points[keep]
    if len(filtered) <= budget:
        return filtered
    stride = max(1, math.ceil(len(filtered) / budget))
    return filtered[::stride][:budget]


def _prediction_hover(annotation: TrackView, model_label: str) -> str:
    class_name = annotation.class_name or "object"
    return (
        f"{model_label} · model-local ID {annotation.track_id}<br>"
        f"{class_name} · confidence {annotation.confidence:.2f}<br>"
        f"3D source: {annotation.source}<br>"
        f"associated LiDAR points: {annotation.lidar_point_count}"
    )


def build_lidar_figure(
    points_xyzi: np.ndarray | None,
    analysis: AnalysisView,
    model_label: str,
    *,
    max_points: int = 15_000,
    show_ground_truth: bool = False,
    show_labels: bool = True,
    x_range: tuple[float, float] = DEFAULT_X_RANGE,
    y_range: tuple[float, float] = DEFAULT_Y_RANGE,
    z_range: tuple[float, float] = DEFAULT_Z_RANGE,
    uirevision: str = "lidar-view",
    height: int = 500,
) -> go.Figure:
    """Build an interactive Plotly view in native KITTI Velodyne axes."""

    if height < 200:
        raise ValueError("height must be at least 200 pixels")
    sampled = downsample_lidar_for_plot(
        points_xyzi,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        budget=max_points,
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=sampled[:, 0] if len(sampled) else [],
            y=sampled[:, 1] if len(sampled) else [],
            z=sampled[:, 2] if len(sampled) else [],
            mode="markers",
            marker={
                "size": 1.35,
                "opacity": 0.76,
                "color": sampled[:, 2] if len(sampled) else [],
                "cmin": -2.5,
                "cmax": 2.0,
                "colorscale": (
                    (0.0, "#2563EB"),
                    (0.50, "#22D3EE"),
                    (1.0, "#FACC15"),
                ),
                "showscale": False,
            },
            hoverinfo="skip",
            name="LiDAR",
            showlegend=False,
        )
    )

    located = [item for item in analysis.annotations if item.center_velodyne is not None]
    centers: list[tuple[float, float, float]] = []
    center_colors: list[str] = []
    center_hover: list[str] = []
    label_positions: list[tuple[float, float, float]] = []
    center_labels: list[str] = []
    ray_x: list[float | None] = []
    ray_y: list[float | None] = []
    ray_z: list[float | None] = []
    identity = np.eye(3, dtype=np.float64)

    for annotation in located:
        assert annotation.center_velodyne is not None
        corners = oriented_box_corners(
            annotation.center_velodyne,
            annotation.dimensions_lwh,
            identity,
        )
        x_values, y_values, z_values = box_line_coordinates(corners)
        color = rgb_to_css(annotation.color_rgb)
        hover = _prediction_hover(annotation, model_label)
        figure.add_trace(
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="lines",
                line={
                    "color": color,
                    "width": 5,
                    "dash": "dash" if annotation.source.startswith("ray-fallback") else "solid",
                },
                text=[hover] * len(x_values),
                hovertemplate="%{text}<extra></extra>",
                name=f"Track {annotation.track_id}",
                showlegend=False,
            )
        )
        centers.append(annotation.center_velodyne)
        center_colors.append(color)
        center_hover.append(hover)
        class_name = (annotation.class_name or "object").title()
        label_positions.append(
            (
                annotation.center_velodyne[0],
                annotation.center_velodyne[1],
                annotation.center_velodyne[2] + annotation.dimensions_lwh[2] / 2.0 + 0.25,
            )
        )
        center_labels.append(
            f"{class_name} ID {annotation.track_id} Â· {annotation.confidence:.2f}"
        )
        if analysis.camera_origin_velodyne is not None:
            origin = analysis.camera_origin_velodyne
            ray_x.extend((origin[0], annotation.center_velodyne[0], None))
            ray_y.extend((origin[1], annotation.center_velodyne[1], None))
            ray_z.extend((origin[2], annotation.center_velodyne[2], None))

    if ray_x:
        figure.add_trace(
            go.Scatter3d(
                x=ray_x,
                y=ray_y,
                z=ray_z,
                mode="lines",
                line={"color": "rgba(148,163,184,0.38)", "width": 2},
                hoverinfo="skip",
                showlegend=False,
                name="Camera rays",
            )
        )

    if centers:
        center_array = np.asarray(centers, dtype=np.float64)
        figure.add_trace(
            go.Scatter3d(
                x=center_array[:, 0],
                y=center_array[:, 1],
                z=center_array[:, 2],
                mode="markers",
                marker={"size": 4, "color": center_colors, "symbol": "circle"},
                text=center_hover,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
                name="Track centers",
            )
        )
        if show_labels:
            label_array = np.asarray(label_positions, dtype=np.float64)
            figure.add_trace(
                go.Scatter3d(
                    x=label_array[:, 0],
                    y=label_array[:, 1],
                    z=label_array[:, 2],
                    mode="text",
                    text=center_labels,
                    textposition="top center",
                    textfont={"size": 11, "color": "#F8FAFC"},
                    hoverinfo="skip",
                    showlegend=False,
                    name="Track labels",
                )
            )

    if show_ground_truth and analysis.ground_truth:
        gt_x: list[float | None] = []
        gt_y: list[float | None] = []
        gt_z: list[float | None] = []
        gt_hover: list[str | None] = []
        gt_label_positions: list[tuple[float, float, float]] = []
        gt_labels: list[str] = []
        for annotation in analysis.ground_truth:
            corners = oriented_box_corners(
                annotation.center_velodyne,
                annotation.extent_lwh,
                annotation.rotation_velodyne,
            )
            x_values, y_values, z_values = box_line_coordinates(corners)
            hover = f"GT ID {annotation.track_id} · {annotation.object_type}"
            gt_x.extend(x_values)
            gt_y.extend(y_values)
            gt_z.extend(z_values)
            gt_hover.extend(hover if value is not None else None for value in x_values)
            gt_label_positions.append(
                (
                    annotation.center_velodyne[0],
                    annotation.center_velodyne[1],
                    annotation.center_velodyne[2] + annotation.extent_lwh[2] / 2.0 + 0.25,
                )
            )
            gt_labels.append(f"GT {annotation.object_type} ID {annotation.track_id}")
        figure.add_trace(
            go.Scatter3d(
                x=gt_x,
                y=gt_y,
                z=gt_z,
                mode="lines",
                line={"color": "#FF00FF", "width": 5, "dash": "dash"},
                text=gt_hover,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
                name="Ground truth",
            )
        )
        if show_labels:
            gt_label_array = np.asarray(gt_label_positions, dtype=np.float64)
            figure.add_trace(
                go.Scatter3d(
                    x=gt_label_array[:, 0],
                    y=gt_label_array[:, 1],
                    z=gt_label_array[:, 2],
                    mode="text",
                    text=gt_labels,
                    textposition="bottom center",
                    textfont={"size": 10, "color": "#FF5CFF"},
                    hoverinfo="skip",
                    showlegend=False,
                    name="Ground truth labels",
                )
            )

    if analysis.camera_origin_velodyne is not None:
        origin = analysis.camera_origin_velodyne
        figure.add_trace(
            go.Scatter3d(
                x=[origin[0]],
                y=[origin[1]],
                z=[origin[2]],
                mode="markers",
                marker={"size": 5, "color": "#3B82F6", "symbol": "diamond"},
                text=["Camera center"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
                name="Camera",
            )
        )

    notes: list[str] = []
    if sampled.size == 0:
        notes.append("LiDAR unavailable for this frame")
    if analysis.error:
        notes.append(analysis.error)
    annotations = [
        {
            "text": note,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.96 - position * 0.08,
            "showarrow": False,
            "font": {"color": "#FCA5A5", "size": 13},
            "bgcolor": "rgba(127,29,29,0.65)",
        }
        for position, note in enumerate(notes)
    ]
    axis_style = {
        "showbackground": True,
        "backgroundcolor": "#080C12",
        "gridcolor": "#263244",
        "zerolinecolor": "#475569",
        "color": "#CBD5E1",
    }
    figure.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 34, "b": 0},
        paper_bgcolor="#080C12",
        plot_bgcolor="#080C12",
        font={"color": "#CBD5E1"},
        title={
            "text": f"{model_label} · LiDAR-supported approximate 3D",
            "x": 0.02,
            "font": {"size": 14, "color": "#E2E8F0"},
        },
        annotations=annotations,
        uirevision=uirevision,
        scene={
            "xaxis": {**axis_style, "title": "Forward x (m)", "range": list(x_range)},
            "yaxis": {**axis_style, "title": "Left y (m)", "range": list(y_range)},
            "zaxis": {**axis_style, "title": "Up z (m)", "range": list(z_range)},
            "aspectmode": "data",
            "uirevision": uirevision,
            "camera": {
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
                "eye": {"x": -1.5, "y": -1.5, "z": 0.85},
                "projection": {"type": "perspective"},
            },
        },
    )
    return figure


def _pil_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=15)
    except TypeError:  # Pillow < 10.1 compatibility
        return ImageFont.load_default()


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    left, top = xy
    bounds = draw.textbbox((left, top), text, font=font, stroke_width=0)
    padding = 4
    draw.rectangle(
        (
            bounds[0] - padding,
            bounds[1] - padding,
            bounds[2] + padding,
            bounds[3] + padding,
        ),
        fill=(7, 12, 20),
        outline=color,
        width=1,
    )
    draw.text((left, top), text, fill=color, font=font)


def render_camera_view(
    image_rgb: np.ndarray,
    analysis: AnalysisView,
    model_label: str,
    *,
    frame_index: int,
    frame_count: int,
    show_ground_truth: bool = False,
    target_aspect_ratio: float | None = None,
    letterbox_color: Sequence[int] = (5, 8, 13),
) -> np.ndarray:
    """Draw tracked 2D boxes, optionally on a centered letterbox canvas."""

    array = np.asarray(image_rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    source_image = Image.fromarray(np.asarray(array, dtype=np.uint8))
    source_width, source_height = source_image.size
    offset_x = 0
    offset_y = 0
    image = source_image
    if target_aspect_ratio is not None:
        if not math.isfinite(target_aspect_ratio) or target_aspect_ratio <= 0.0:
            raise ValueError("target_aspect_ratio must be a finite positive number")
        color = np.asarray(letterbox_color, dtype=np.int64)
        if color.shape != (3,) or np.any(color < 0) or np.any(color > 255):
            raise ValueError("letterbox_color must contain three values from 0 to 255")
        source_aspect = source_width / source_height
        if source_aspect > target_aspect_ratio:
            canvas_width = source_width
            canvas_height = math.ceil(source_width / target_aspect_ratio)
        else:
            canvas_width = math.ceil(source_height * target_aspect_ratio)
            canvas_height = source_height
        offset_x = (canvas_width - source_width) // 2
        offset_y = (canvas_height - source_height) // 2
        fill = tuple(int(component) for component in color)
        image = Image.new("RGB", (canvas_width, canvas_height), fill)
        image.paste(source_image, (offset_x, offset_y))

    draw = ImageDraw.Draw(image)
    font = _pil_font()
    width, height = image.size

    if show_ground_truth:
        for annotation in analysis.ground_truth:
            if annotation.bbox_xyxy is None:
                continue
            x1, y1, x2, y2 = annotation.bbox_xyxy
            box = (
                int(round(x1)) + offset_x,
                int(round(y1)) + offset_y,
                int(round(x2)) + offset_x,
                int(round(y2)) + offset_y,
            )
            draw.rectangle(box, outline=(255, 0, 255), width=2)
            _draw_label(
                draw,
                (max(4, box[0] + 3), max(4, box[1] + 3)),
                f"GT {annotation.track_id} · {annotation.object_type}",
                (255, 0, 255),
                font,
            )

    for annotation in analysis.annotations:
        x1, y1, x2, y2 = annotation.bbox_xyxy
        box = (
            max(0, min(width - 1, int(round(x1)) + offset_x)),
            max(0, min(height - 1, int(round(y1)) + offset_y)),
            max(0, min(width - 1, int(round(x2)) + offset_x)),
            max(0, min(height - 1, int(round(y2)) + offset_y)),
        )
        color = tuple(
            int(round(value * 255.0))
            for value in np.clip(annotation.color_rgb, 0.0, 1.0)
        )
        draw.rectangle(box, outline=color, width=3)
        class_name = annotation.class_name or "object"
        label = (
            f"{model_label} ID {annotation.track_id} · {class_name} "
            f"{annotation.confidence:.2f}"
        )
        label_y = max(4, box[1] - 24)
        _draw_label(draw, (max(4, box[0] + 2), label_y), label, color, font)

    header = f"{model_label} · tracked 2D camera view"
    status = (
        analysis.error
        or f"frame {frame_index + 1}/{frame_count} · {len(analysis.annotations)} confirmed tracks"
    )
    draw.rectangle((0, 0, width, 31), fill=(5, 10, 18))
    draw.text((10, 8), header, fill=(226, 232, 240), font=font)
    status_width = draw.textbbox((0, 0), status, font=font)[2]
    draw.text(
        (max(10, width - status_width - 10), 8),
        status,
        fill=(248, 113, 113) if analysis.error else (148, 163, 184),
        font=font,
    )
    return np.asarray(image)


@dataclass(frozen=True, slots=True)
class _SyntheticObject:
    track_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    center: tuple[float, float, float]
    extent: tuple[float, float, float]


def _synthetic_objects(index: int, width: int, height: int) -> tuple[_SyntheticObject, ...]:
    wave = math.sin(index * 0.22)
    car_near_x = 18.0 + index * 0.18
    car_far_x = 39.0 - index * 0.14
    near_box = (
        width * (0.30 + index * 0.0025),
        height * (0.51 + wave * 0.004),
        width * (0.49 + index * 0.0025),
        height * 0.79,
    )
    far_box = (
        width * (0.58 - index * 0.0015),
        height * 0.53,
        width * (0.69 - index * 0.0010),
        height * 0.68,
    )
    person_box = (
        width * (0.74 + wave * 0.012),
        height * 0.48,
        width * (0.785 + wave * 0.012),
        height * 0.73,
    )
    cycle_box = (
        width * (0.12 + index * 0.002),
        height * 0.53,
        width * (0.205 + index * 0.002),
        height * 0.73,
    )
    return (
        _SyntheticObject(1, "car", near_box, (car_near_x, 2.8 - wave, -0.65), (4.2, 1.8, 1.6)),
        _SyntheticObject(2, "car", far_box, (car_far_x, -5.0 + wave, -0.72), (4.5, 1.9, 1.7)),
        _SyntheticObject(3, "person", person_box, (25.0, -6.8 - wave, -0.75), (0.8, 0.8, 1.75)),
        _SyntheticObject(4, "bicycle", cycle_box, (29.0, 7.5 - index * 0.08, -0.70), (1.8, 0.7, 1.55)),
    )


def _draw_synthetic_scene(index: int, width: int, height: int) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float64)[:, None]
    sky_top = np.asarray((17, 31, 52), dtype=np.float64)
    sky_bottom = np.asarray((104, 139, 168), dtype=np.float64)
    row_colors = sky_top + (sky_bottom - sky_top) * np.minimum(y / 0.64, 1.0)
    image = np.repeat(row_colors[:, None, :], width, axis=1).astype(np.uint8)
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    horizon = int(height * 0.43)

    draw.rectangle((0, horizon - 18, width, horizon + 28), fill=(38, 62, 66))
    for building in range(15):
        x1 = int(building * width / 14 - 20)
        building_width = 40 + (building * 17) % 55
        building_height = 25 + (building * 31) % 80
        shade = 32 + (building * 13) % 38
        draw.rectangle(
            (x1, horizon - building_height, x1 + building_width, horizon + 4),
            fill=(shade, shade + 7, shade + 12),
        )
        for window_x in range(x1 + 8, x1 + building_width - 5, 15):
            draw.rectangle(
                (window_x, horizon - building_height + 10, window_x + 4, horizon - building_height + 14),
                fill=(214, 192, 114),
            )

    draw.polygon(
        ((int(width * 0.35), horizon), (int(width * 0.65), horizon), (width, height), (0, height)),
        fill=(39, 44, 51),
    )
    draw.line((int(width * 0.35), horizon, 0, height), fill=(155, 162, 168), width=4)
    draw.line((int(width * 0.65), horizon, width, height), fill=(155, 162, 168), width=4)
    offset = (index * 24) % 90
    for lane in (-0.18, 0.18):
        vanishing_x = width * (0.5 + lane * 0.20)
        for dash in range(8):
            depth = (dash * 95 + offset) % 760
            fraction = depth / 760.0
            y1 = horizon + fraction * (height - horizon)
            y2 = min(height, y1 + 10 + fraction * 34)
            x1 = vanishing_x + lane * fraction * width
            half_width = 1 + int(fraction * 5)
            draw.polygon(
                ((x1 - half_width, y1), (x1 + half_width, y1), (x1 + half_width * 2, y2), (x1 - half_width * 2, y2)),
                fill=(236, 229, 191),
            )

    for obj in _synthetic_objects(index, width, height):
        x1, y1, x2, y2 = (int(round(value)) for value in obj.bbox_xyxy)
        if obj.class_name == "car":
            body = (40, 115, 180) if obj.track_id == 1 else (184, 74, 62)
            draw.rounded_rectangle((x1, y1 + (y2 - y1) // 3, x2, y2 - 6), radius=8, fill=body)
            draw.polygon(
                (
                    (x1 + (x2 - x1) // 4, y1 + (y2 - y1) // 3),
                    (x1 + (x2 - x1) * 2 // 5, y1 + 5),
                    (x1 + (x2 - x1) * 3 // 4, y1 + 8),
                    (x2 - 8, y1 + (y2 - y1) // 3),
                ),
                fill=(113, 161, 188),
            )
            wheel = max(3, (y2 - y1) // 11)
            draw.ellipse((x1 + 10, y2 - wheel * 2, x1 + 10 + wheel * 2, y2), fill=(12, 18, 24))
            draw.ellipse((x2 - 10 - wheel * 2, y2 - wheel * 2, x2 - 10, y2), fill=(12, 18, 24))
        elif obj.class_name == "person":
            cx = (x1 + x2) // 2
            head = max(4, (x2 - x1) // 3)
            draw.ellipse((cx - head, y1, cx + head, y1 + head * 2), fill=(221, 175, 137))
            draw.line((cx, y1 + head * 2, cx, y2 - 18), fill=(235, 179, 50), width=8)
            draw.line((cx, y2 - 18, x1 + 3, y2), fill=(25, 35, 48), width=4)
            draw.line((cx, y2 - 18, x2 - 3, y2), fill=(25, 35, 48), width=4)
        else:
            radius = max(6, (x2 - x1) // 5)
            draw.ellipse((x1, y2 - radius * 2, x1 + radius * 2, y2), outline=(25, 30, 36), width=3)
            draw.ellipse((x2 - radius * 2, y2 - radius * 2, x2, y2), outline=(25, 30, 36), width=3)
            draw.line((x1 + radius, y2 - radius, x2 - radius, y2 - radius), fill=(55, 185, 128), width=4)

    return np.asarray(canvas)


def _synthetic_point_cloud(index: int, objects: Sequence[_SyntheticObject]) -> np.ndarray:
    rng = np.random.default_rng(26_000 + index)
    ground_count = 10_000
    x = rng.uniform(-5.0, 80.0, ground_count)
    y = rng.uniform(-24.0, 24.0, ground_count)
    z = -1.72 + rng.normal(0.0, 0.025, ground_count)
    intensity = np.clip(0.18 + 0.55 * rng.random(ground_count), 0.0, 1.0)
    parts = [np.column_stack((x, y, z, intensity))]

    wall_count = 2_200
    wall_x = rng.uniform(5.0, 80.0, wall_count)
    wall_side = rng.choice((-1.0, 1.0), wall_count)
    wall_y = wall_side * rng.uniform(17.0, 28.0, wall_count)
    wall_z = rng.uniform(-1.5, 4.0, wall_count)
    wall_i = rng.uniform(0.15, 0.80, wall_count)
    parts.append(np.column_stack((wall_x, wall_y, wall_z, wall_i)))

    for obj in objects:
        center = np.asarray(obj.center, dtype=np.float64)
        extent = np.asarray(obj.extent, dtype=np.float64)
        count = 260 if obj.class_name == "car" else 120
        local = (rng.random((count, 3)) - 0.5) * extent
        # Favor visible front/side surfaces while keeping a recognizable volume.
        local[: count // 2, 0] = -extent[0] / 2.0 + rng.normal(0.0, 0.05, count // 2)
        xyz = center + local
        obj_i = rng.uniform(0.55, 1.0, count)
        parts.append(np.column_stack((xyz, obj_i)))

    return np.concatenate(parts).astype(np.float32, copy=False)


def _jitter_bbox(
    bbox: tuple[float, float, float, float], index: int, scale: float, phase: float
) -> tuple[float, float, float, float]:
    dx = math.sin(index * 0.31 + phase) * scale
    dy = math.cos(index * 0.27 + phase) * scale * 0.55
    return bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy


def _synthetic_analysis(
    model_key: str,
    model_label: str,
    index: int,
    objects: Sequence[_SyntheticObject],
    ground_truth: tuple[GroundTruthView, ...],
) -> AnalysisView:
    annotations: list[TrackView] = []
    for obj in objects:
        if model_key == "yolo11" and obj.class_name == "bicycle":
            continue
        if model_key == "yolo11" and obj.class_name == "person" and index % 11 in (0, 1):
            continue
        if model_key == "yolo26" and obj.class_name == "bicycle" and index % 8 in (0,):
            continue

        phase = obj.track_id * (0.71 if model_key == "yolo11" else 0.43)
        bbox_scale = 3.6 if model_key == "yolo11" else 2.2
        bbox = _jitter_bbox(obj.bbox_xyxy, index, bbox_scale, phase)
        center_noise = 0.34 if model_key == "yolo11" else 0.20
        center = (
            obj.center[0] + math.sin(index * 0.18 + phase) * center_noise,
            obj.center[1] + math.cos(index * 0.21 + phase) * center_noise,
            obj.center[2],
        )
        confidence = 0.68 + 0.06 * obj.track_id
        if model_key == "yolo26":
            confidence += 0.07
        confidence += 0.025 * math.sin(index * 0.17 + phase)
        annotations.append(
            TrackView(
                track_id=str(obj.track_id),
                class_name=obj.class_name,
                bbox_xyxy=bbox,
                center_velodyne=center,
                dimensions_lwh=obj.extent,
                color_rgb=track_color(f"{model_key}:{obj.track_id}"),
                source="synthetic-lidar",
                confidence=min(0.98, confidence),
                lidar_point_count=260 if obj.class_name == "car" else 120,
            )
        )
    return AnalysisView(
        frame_name=f"portfolio_{index:06d}.png",
        annotations=tuple(annotations),
        ground_truth=ground_truth,
        camera_origin_velodyne=(0.0, 0.0, 0.0),
    )


def build_synthetic_frame(
    index: int,
    *,
    frame_count: int = DEFAULT_FRAME_COUNT,
    width: int = 960,
    height: int = 420,
) -> DashboardFrame:
    """Create a deterministic, redistribution-safe portfolio preview frame.

    It exercises the exact dashboard controls and renderers but intentionally
    does not claim to be output from either YOLO model.  Local live mode is the
    path for real KITTI/model inference.
    """

    if frame_count < 1:
        raise ValueError("frame_count must be at least one")
    if index < 0 or index >= frame_count:
        raise IndexError(index)
    if width < 320 or height < 180:
        raise ValueError("synthetic preview dimensions are too small")

    objects = _synthetic_objects(index, width, height)
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ground_truth = tuple(
        GroundTruthView(
            track_id=obj.track_id,
            object_type=obj.class_name,
            center_velodyne=obj.center,
            rotation_velodyne=identity,
            extent_lwh=obj.extent,
            bbox_xyxy=obj.bbox_xyxy,
        )
        for obj in objects
    )
    image = _draw_synthetic_scene(index, width, height)
    points = _synthetic_point_cloud(index, objects)
    return DashboardFrame(
        frame_index=index,
        frame_count=frame_count,
        frame_name=f"portfolio_{index:06d}.png",
        image_rgb=image,
        points_xyzi=points,
        yolo11=_synthetic_analysis(
            "yolo11", "YOLO11", index, objects, ground_truth
        ),
        yolo26=_synthetic_analysis(
            "yolo26", "YOLO26", index, objects, ground_truth
        ),
        source_label="Synthetic portfolio preview",
    )

# SPDX-License-Identifier: AGPL-3.0-only
"""Generate deterministic visual assets used by the project README.

The tracking values below are the pretrained-control Car results documented in
the README for KITTI Tracking sequence 0000. They are local development scores,
not official KITTI test-server results.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard_core import (  # noqa: E402 - project root is added above
    BOX_EDGES,
    AnalysisView,
    build_synthetic_frame,
    downsample_lidar_for_plot,
    oriented_box_corners,
    render_camera_view,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "assets" / "tracking_metrics_comparison.png"
DEFAULT_DASHBOARD_OUTPUT = PROJECT_ROOT / "docs" / "assets" / "dashboard_overview.png"

DETECTORS = (
    "YOLO11s\ndefault",
    "YOLO26s\nend-to-end",
    "YOLO26s\none-to-many",
)

# Exact sequence-0000 Car values from README.md. TrackEval rates are percentages.
METRICS = {
    "HOTA": (47.811, 48.104, 48.724),
    "MOTA": (28.372, 33.953, 29.302),
    "IDF1": (54.462, 58.795, 58.525),
}

# Okabe-Ito colorblind-safe colors. Hatches keep the series distinguishable when
# the chart is printed in grayscale.
COLORS = {
    "HOTA": "#0072B2",
    "MOTA": "#E69F00",
    "IDF1": "#009E73",
}
HATCHES = {"HOTA": "///", "MOTA": "\\\\", "IDF1": "..."}


def generate_chart(output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Render the README tracking comparison and return its output path."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )

    figure, axes = plt.subplots(figsize=(9, 5), dpi=200)
    figure.patch.set_facecolor("white")
    axes.set_facecolor("white")

    x_positions = np.arange(len(DETECTORS), dtype=float)
    bar_width = 0.23
    offsets = (-bar_width, 0.0, bar_width)

    for (metric_name, values), offset in zip(METRICS.items(), offsets):
        bars = axes.bar(
            x_positions + offset,
            values,
            width=bar_width,
            color=COLORS[metric_name],
            edgecolor="#1A1A1A",
            linewidth=0.7,
            hatch=HATCHES[metric_name],
            label=metric_name,
            zorder=3,
        )
        axes.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in values],
            padding=3,
            fontsize=8.5,
            color="#111111",
        )

    axes.set_ylabel("TrackEval score (%)", fontweight="semibold")
    axes.set_xticks(x_positions)
    axes.set_xticklabels(DETECTORS, fontweight="semibold")
    axes.set_ylim(0, 68)
    axes.set_yticks(np.arange(0, 69, 10))
    axes.grid(axis="y", color="#D8D8D8", linewidth=0.8, zorder=0)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        handlelength=2.0,
    )

    figure.suptitle(
        "KITTI Tracking Sequence 0000 — Car Tracking Metrics",
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.02,
        "Pretrained controls · Local development result · Not official KITTI test-server scores",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.09, right=0.985, top=0.83, bottom=0.19)

    figure.savefig(
        output_path,
        dpi=200,
        facecolor="white",
        metadata={
            "Title": "KITTI Tracking Sequence 0000 pretrained control comparison",
            "Description": "Local development Car HOTA, MOTA, and IDF1 results.",
        },
    )
    plt.close(figure)
    return output_path


def _style_lidar_axes(axes: plt.Axes) -> None:
    """Apply the dashboard's dark LiDAR styling to a Matplotlib 3D panel."""

    background = "#080C12"
    grid = "#334155"
    foreground = "#CBD5E1"
    axes.set_facecolor(background)
    axes.set_xlim(-5.0, 68.0)
    axes.set_ylim(-26.0, 26.0)
    axes.set_zlim(-3.0, 5.0)
    axes.set_box_aspect((1.65, 1.15, 0.50))
    axes.view_init(elev=25.0, azim=-123.0)
    axes.set_xlabel("Forward x (m)", color=foreground, labelpad=6)
    axes.set_ylabel("Left y (m)", color=foreground, labelpad=6)
    axes.set_zlabel("Up z (m)", color=foreground, labelpad=4)
    axes.tick_params(colors=foreground, labelsize=7, pad=1)
    axes.grid(True, color=grid, linewidth=0.45)
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.set_pane_color((0.031, 0.047, 0.071, 1.0))
        axis._axinfo["grid"]["color"] = grid
        axis._axinfo["grid"]["linewidth"] = 0.45


def _render_lidar_panel(
    axes: plt.Axes,
    points_xyzi: np.ndarray | None,
    analysis: AnalysisView,
    model_label: str,
) -> None:
    """Render one static LiDAR panel using the same data as the live dashboard."""

    _style_lidar_axes(axes)
    sampled = downsample_lidar_for_plot(
        points_xyzi,
        x_range=(-5.0, 68.0),
        y_range=(-26.0, 26.0),
        z_range=(-3.0, 5.0),
        budget=7_500,
    )
    if len(sampled):
        heights = np.clip(sampled[:, 2], -2.5, 2.0)
        axes.scatter(
            sampled[:, 0],
            sampled[:, 1],
            sampled[:, 2],
            c=heights,
            cmap="viridis",
            vmin=-2.5,
            vmax=2.0,
            s=0.65,
            alpha=0.66,
            linewidths=0,
            rasterized=True,
        )

    identity = np.eye(3, dtype=np.float64)
    for annotation in analysis.annotations:
        if annotation.center_velodyne is None:
            continue
        corners = oriented_box_corners(
            annotation.center_velodyne,
            annotation.dimensions_lwh,
            identity,
        )
        color = np.clip(np.asarray(annotation.color_rgb), 0.0, 1.0)
        for start, end in BOX_EDGES:
            axes.plot(
                corners[[start, end], 0],
                corners[[start, end], 1],
                corners[[start, end], 2],
                color=color,
                linewidth=1.7,
                solid_capstyle="round",
            )
        center = np.asarray(annotation.center_velodyne, dtype=np.float64)
        axes.scatter(
            [center[0]], [center[1]], [center[2]], color=[color], s=20, depthshade=False
        )
        axes.text(
            center[0],
            center[1],
            center[2] + annotation.dimensions_lwh[2] * 0.72,
            f"ID {annotation.track_id}",
            color=color,
            fontsize=7,
            fontweight="bold",
        )

    axes.scatter([0.0], [0.0], [0.0], marker="D", color="#60A5FA", s=24)
    axes.set_title(
        f"{model_label} - LiDAR-supported approximate 3D",
        color="#0F172A",
        fontsize=12,
        fontweight="bold",
        pad=9,
    )


def generate_dashboard_preview(
    output_path: Path = DEFAULT_DASHBOARD_OUTPUT,
    *,
    frame_index: int = 12,
) -> Path:
    """Render a deterministic four-panel synthetic dashboard preview."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = build_synthetic_frame(frame_index)

    yolo11_camera = render_camera_view(
        frame.image_rgb,
        frame.yolo11,
        "YOLO11",
        frame_index=frame.frame_index,
        frame_count=frame.frame_count,
    )
    yolo26_camera = render_camera_view(
        frame.image_rgb,
        frame.yolo26,
        "YOLO26",
        frame_index=frame.frame_index,
        frame_count=frame.frame_count,
    )

    figure = plt.figure(figsize=(16, 10), dpi=160, facecolor="#F8FAFC")
    grid = figure.add_gridspec(
        2,
        2,
        left=0.025,
        right=0.975,
        top=0.875,
        bottom=0.09,
        wspace=0.08,
        hspace=0.20,
    )
    camera11_axes = figure.add_subplot(grid[0, 0])
    lidar11_axes = figure.add_subplot(grid[0, 1], projection="3d")
    camera26_axes = figure.add_subplot(grid[1, 0])
    lidar26_axes = figure.add_subplot(grid[1, 1], projection="3d")

    for axes, image_array, title in (
        (camera11_axes, yolo11_camera, "YOLO11 - tracked 2D camera view"),
        (camera26_axes, yolo26_camera, "YOLO26 - tracked 2D camera view"),
    ):
        axes.imshow(image_array)
        axes.set_title(title, fontsize=12, fontweight="bold", color="#0F172A", pad=9)
        axes.set_axis_off()

    _render_lidar_panel(lidar11_axes, frame.points_xyzi, frame.yolo11, "YOLO11")
    _render_lidar_panel(lidar26_axes, frame.points_xyzi, frame.yolo26, "YOLO26")

    figure.suptitle(
        "Synchronized Camera and LiDAR Tracking Dashboard",
        x=0.5,
        y=0.967,
        fontsize=22,
        fontweight="bold",
        color="#0F172A",
    )
    figure.text(
        0.5,
        0.915,
        f"Synthetic portfolio preview  |  Shared frame {frame.frame_index + 1}/{frame.frame_count}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="semibold",
        color="#2563EB",
    )
    figure.text(
        0.5,
        0.035,
        "Illustrative synthetic scene generated by the project renderer - not measured YOLO output or KITTI benchmark evidence",
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
    )
    figure.savefig(
        output_path,
        dpi=160,
        facecolor=figure.get_facecolor(),
        metadata={
            "Title": "Synthetic four-panel camera and LiDAR dashboard preview",
            "Description": "Illustrative project-renderer output; not measured model evidence.",
        },
    )
    plt.close(figure)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PNG output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dashboard-output",
        type=Path,
        default=DEFAULT_DASHBOARD_OUTPUT,
        help=f"Dashboard PNG output path (default: {DEFAULT_DASHBOARD_OUTPUT})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metric_output_path = generate_chart(args.output)
    dashboard_output_path = generate_dashboard_preview(args.dashboard_output)
    print(metric_output_path)
    print(dashboard_output_path)


if __name__ == "__main__":
    main()

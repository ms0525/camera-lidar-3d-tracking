# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import math
import unittest

import numpy as np

from app.dashboard_core import (
    AnalysisView,
    GroundTruthView,
    TrackView,
    autoplay_interval_seconds,
    box_line_coordinates,
    build_lidar_figure,
    build_synthetic_frame,
    clamp_frame_index,
    downsample_lidar_for_plot,
    move_frame_index,
    oriented_box_corners,
    playback_step,
    render_camera_view,
    rgb_to_css,
    seekable_frame_limit,
    track_color,
)


IDENTITY_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


class DashboardNavigationTests(unittest.TestCase):
    def test_clamp_and_manual_navigation_stop_at_sequence_ends(self) -> None:
        self.assertEqual(clamp_frame_index(-8, 5), 0)
        self.assertEqual(clamp_frame_index(2, 5), 2)
        self.assertEqual(clamp_frame_index(99, 5), 4)
        self.assertEqual(clamp_frame_index(7, 0), 0)
        self.assertEqual(clamp_frame_index(7, -3), 0)

        self.assertEqual(move_frame_index(0, -1, 5), 0)
        self.assertEqual(move_frame_index(2, 1, 5), 3)
        self.assertEqual(move_frame_index(4, 1, 5), 4)
        self.assertEqual(move_frame_index(3, -20, 5), 0)

    def test_playback_handles_empty_single_and_end_of_sequence(self) -> None:
        empty = playback_step(4, 0, loop=True)
        self.assertEqual((empty.index, empty.playing), (0, False))

        single_loop = playback_step(0, 1, loop=True)
        single_no_loop = playback_step(0, 1, loop=False)
        self.assertEqual((single_loop.index, single_loop.playing), (0, False))
        self.assertEqual((single_no_loop.index, single_no_loop.playing), (0, False))

        middle = playback_step(1, 4, loop=False)
        self.assertEqual((middle.index, middle.playing), (2, True))

        stopped = playback_step(3, 4, loop=False)
        wrapped = playback_step(3, 4, loop=True)
        self.assertEqual((stopped.index, stopped.playing), (3, False))
        self.assertEqual((wrapped.index, wrapped.playing), (0, True))

    def test_autoplay_interval_validates_fps(self) -> None:
        self.assertAlmostEqual(autoplay_interval_seconds(2.5), 0.4)
        self.assertAlmostEqual(autoplay_interval_seconds(0.5), 2.0)

        for invalid in (0.0, -1.0, math.inf, -math.inf, math.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    autoplay_interval_seconds(invalid)

    def test_live_seek_limit_handles_the_unprocessed_initial_frame(self) -> None:
        self.assertEqual(seekable_frame_limit(100), 100)
        self.assertEqual(seekable_frame_limit(100, processed_through=-1), 1)
        self.assertEqual(seekable_frame_limit(100, processed_through=0), 2)
        self.assertEqual(seekable_frame_limit(100, processed_through=8), 10)
        self.assertEqual(seekable_frame_limit(3, processed_through=8), 3)
        self.assertEqual(seekable_frame_limit(1, processed_through=-1), 1)
        self.assertEqual(seekable_frame_limit(0, processed_through=-1), 0)


class DashboardGeometryTests(unittest.TestCase):
    def test_lidar_downsampling_is_deterministic_and_respects_roi_and_budget(self) -> None:
        valid = np.asarray(
            [[float(index), 0.0, 0.0, index / 10.0] for index in range(7)],
            dtype=np.float32,
        )
        invalid = np.asarray(
            (
                (7.0, 0.0, 0.0, 0.7),
                (2.0, 1.1, 0.0, 0.8),
                (2.0, 0.0, 2.1, 0.9),
                (math.nan, 0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        points = np.concatenate((valid, invalid), axis=0)
        options = {
            "x_range": (0.0, 6.0),
            "y_range": (-1.0, 1.0),
            "z_range": (-2.0, 2.0),
            "budget": 3,
        }

        first = downsample_lidar_for_plot(points, **options)
        second = downsample_lidar_for_plot(points, **options)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:, 0], np.asarray((0.0, 3.0, 6.0)))
        self.assertLessEqual(len(first), options["budget"])
        self.assertTrue(np.isfinite(first[:, :3]).all())

        all_in_roi = downsample_lidar_for_plot(
            points,
            x_range=(0.0, 6.0),
            y_range=(-1.0, 1.0),
            z_range=(-2.0, 2.0),
            budget=99,
        )
        np.testing.assert_array_equal(all_in_roi, valid)
        self.assertEqual(downsample_lidar_for_plot(None).shape, (0, 4))
        with self.assertRaises(ValueError):
            downsample_lidar_for_plot(points, budget=0)

    def test_oriented_box_corners_and_line_coordinate_shape(self) -> None:
        angle = math.pi / 2.0
        rotation = (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
        center = np.asarray((10.0, -2.0, 1.0))
        corners = oriented_box_corners(center, (2.0, 4.0, 6.0), rotation)

        self.assertEqual(corners.shape, (8, 3))
        np.testing.assert_allclose(corners.mean(axis=0), center, atol=1e-12)
        np.testing.assert_allclose(np.ptp(corners, axis=0), (4.0, 2.0, 6.0), atol=1e-12)

        x_values, y_values, z_values = box_line_coordinates(corners)
        for values in (x_values, y_values, z_values):
            self.assertEqual(len(values), 36)
            self.assertEqual(sum(value is None for value in values), 12)

    def test_rgb_helpers_clip_convert_and_remain_stable(self) -> None:
        self.assertEqual(rgb_to_css((0.0, 0.5, 1.0)), "#0080FF")
        self.assertEqual(rgb_to_css((-1.0, 0.25, 2.0)), "#0040FF")
        with self.assertRaises(ValueError):
            rgb_to_css((0.0, 1.0))

        first = track_color("yolo11:17")
        self.assertEqual(first, track_color("yolo11:17"))
        self.assertNotEqual(first, track_color("yolo26:17"))
        self.assertTrue(all(0.3 <= component <= 1.0 for component in first))


class DashboardRenderingTests(unittest.TestCase):
    @staticmethod
    def _analysis() -> AnalysisView:
        prediction = TrackView(
            track_id="7",
            class_name="car",
            bbox_xyxy=(10.0, 34.0, 42.0, 57.0),
            center_velodyne=(12.0, -1.0, -0.5),
            dimensions_lwh=(4.2, 1.8, 1.6),
            color_rgb=(0.2, 0.8, 0.4),
            source="lidar-cluster",
            confidence=0.87,
            lidar_point_count=43,
        )
        ground_truth = GroundTruthView(
            track_id=11,
            object_type="Car",
            center_velodyne=(12.2, -0.9, -0.45),
            rotation_velodyne=IDENTITY_ROTATION,
            extent_lwh=(4.0, 1.7, 1.5),
            bbox_xyxy=(48.0, 35.0, 76.0, 58.0),
        )
        return AnalysisView(
            frame_name="000007.png",
            annotations=(prediction,),
            ground_truth=(ground_truth,),
            camera_origin_velodyne=(0.0, 0.0, 0.0),
        )

    def test_synthetic_frames_are_deterministic_and_models_differ(self) -> None:
        first = build_synthetic_frame(2, frame_count=6, width=320, height=180)
        second = build_synthetic_frame(2, frame_count=6, width=320, height=180)

        np.testing.assert_array_equal(first.image_rgb, second.image_rgb)
        np.testing.assert_array_equal(first.points_xyzi, second.points_xyzi)
        self.assertEqual(first.yolo11, second.yolo11)
        self.assertEqual(first.yolo26, second.yolo26)
        self.assertEqual(first.frame_name, "portfolio_000002.png")
        self.assertEqual(first.frame_count, 6)

        yolo11_by_id = {item.track_id: item for item in first.yolo11.annotations}
        yolo26_by_id = {item.track_id: item for item in first.yolo26.annotations}
        self.assertNotEqual(set(yolo11_by_id), set(yolo26_by_id))
        self.assertNotEqual(
            yolo11_by_id["1"].bbox_xyxy,
            yolo26_by_id["1"].bbox_xyxy,
        )
        self.assertGreater(
            yolo26_by_id["1"].confidence,
            yolo11_by_id["1"].confidence,
        )

    def test_camera_render_preserves_input_and_output_shape(self) -> None:
        image = np.full((72, 96, 3), 80, dtype=np.uint8)
        original = image.copy()

        rendered = render_camera_view(
            image,
            self._analysis(),
            "YOLO11",
            frame_index=6,
            frame_count=10,
            show_ground_truth=True,
        )

        self.assertEqual(rendered.shape, image.shape)
        self.assertEqual(rendered.dtype, np.uint8)
        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.array_equal(rendered, original))

    def test_camera_render_can_use_dashboard_colored_letterboxing(self) -> None:
        image = np.full((72, 96, 3), 80, dtype=np.uint8)
        background = (5, 8, 13)

        rendered = render_camera_view(
            image,
            self._analysis(),
            "YOLO11",
            frame_index=0,
            frame_count=1,
            target_aspect_ratio=2.0,
            letterbox_color=background,
        )

        self.assertEqual(rendered.shape, (72, 144, 3))
        np.testing.assert_array_equal(rendered[-1, 0], np.asarray(background))
        with self.assertRaises(ValueError):
            render_camera_view(
                image,
                self._analysis(),
                "YOLO11",
                frame_index=0,
                frame_count=1,
                target_aspect_ratio=0.0,
            )

    def test_lidar_figure_has_axes_ranges_prediction_and_ground_truth_traces(self) -> None:
        points = np.asarray(
            (
                (1.0, 0.0, -1.0, 0.1),
                (2.0, 0.5, -0.8, 0.2),
                (3.0, -0.5, -0.6, 0.3),
            ),
            dtype=np.float32,
        )
        x_range = (-2.0, 25.0)
        y_range = (-8.0, 8.0)
        z_range = (-3.0, 4.0)

        figure = build_lidar_figure(
            points,
            self._analysis(),
            "YOLO11",
            max_points=2,
            show_ground_truth=True,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            uirevision="test-view",
            height=360,
        )

        self.assertEqual(figure.layout.scene.xaxis.title.text, "Forward x (m)")
        self.assertEqual(figure.layout.scene.yaxis.title.text, "Left y (m)")
        self.assertEqual(figure.layout.scene.zaxis.title.text, "Up z (m)")
        self.assertEqual(tuple(figure.layout.scene.xaxis.range), x_range)
        self.assertEqual(tuple(figure.layout.scene.yaxis.range), y_range)
        self.assertEqual(tuple(figure.layout.scene.zaxis.range), z_range)
        self.assertEqual(figure.layout.scene.uirevision, "test-view")
        self.assertEqual(figure.layout.height, 360)

        traces = {trace.name: trace for trace in figure.data}
        expected = {
            "LiDAR",
            "Track 7",
            "Camera rays",
            "Track centers",
            "Track labels",
            "Ground truth",
            "Ground truth labels",
            "Camera",
        }
        self.assertTrue(expected.issubset(traces))
        self.assertEqual(len(traces["LiDAR"].x), 2)
        self.assertEqual(len(traces["Track 7"].x), 36)
        self.assertEqual(len(traces["Ground truth"].x), 36)
        self.assertEqual(traces["Ground truth"].line.color, "#FF00FF")
        self.assertEqual(tuple(traces["Track labels"].text), ("Car ID 7 Â· 0.87",))
        self.assertEqual(
            tuple(traces["Ground truth labels"].text),
            ("GT Car ID 11",),
        )

        hidden = build_lidar_figure(
            points,
            self._analysis(),
            "YOLO11",
            show_ground_truth=True,
            show_labels=False,
        )
        hidden_names = {trace.name for trace in hidden.data}
        self.assertNotIn("Track labels", hidden_names)
        self.assertNotIn("Ground truth labels", hidden_names)


if __name__ == "__main__":
    unittest.main()

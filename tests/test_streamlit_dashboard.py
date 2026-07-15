# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # The root CPU test environment intentionally stays lightweight.
    AppTest = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(
    AppTest is None,
    "install app/requirements.txt to run the Streamlit application smoke test",
)
class StreamlitDashboardTests(unittest.TestCase):
    def test_entrypoint_bootstraps_repository_root(self) -> None:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        command = (
            "from streamlit.testing.v1 import AppTest; "
            "app = AppTest.from_file('streamlit_app.py', default_timeout=60).run(); "
            "assert not app.exception, [item.value for item in app.exception]"
        )
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT / "app",
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_public_source_link_uses_deployment_revision_url(self) -> None:
        source_url = "https://example.com/owner/repository/tree/commit-sha"
        with patch.dict(os.environ, {"DASHBOARD_SOURCE_URL": source_url}):
            app = AppTest.from_file(
                str(PROJECT_ROOT / "app/streamlit_app.py"),
                default_timeout=60,
            ).run()

        self.assertEqual(list(app.exception), [])
        source_links = app.get("link_button")
        self.assertEqual(len(source_links), 1)
        self.assertEqual(source_links[0].label, "Source code and AGPL license")
        self.assertEqual(source_links[0].url, source_url)

    def test_public_preview_controls_and_four_panels(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "app/streamlit_app.py"),
            default_timeout=60,
        ).run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(
            [item.value for item in app.subheader],
            [
                "YOLO11 · Camera 2D",
                "YOLO11 · LiDAR 3D",
                "YOLO26 · Camera 2D",
                "YOLO26 · LiDAR 3D",
            ],
        )
        self.assertEqual(
            [getattr(item.proto, "shortcut", "") for item in app.button],
            ["left", "", "right"],
        )

        app.button[2].click().run()
        self.assertEqual(app.session_state["frame_index"], 1)
        self.assertEqual(app.session_state["frame_slider"], 2)

        frame_slider = next(item for item in app.slider if item.label == "Frame")
        frame_slider.set_value(4).run()
        self.assertEqual(app.session_state["frame_index"], 3)
        self.assertEqual(app.session_state["frame_slider"], 4)

        # A timer becoming due at the same moment as manual navigation must not
        # apply two moves. The callback stops playback before timer code runs.
        app.button[1].click().run()
        self.assertTrue(app.session_state["playing"])
        time.sleep(0.55)
        app.button[0].click().run()
        self.assertEqual(app.session_state["frame_index"], 2)
        self.assertFalse(app.session_state["playing"])
        self.assertEqual(list(app.exception), [])


if __name__ == "__main__":
    unittest.main()

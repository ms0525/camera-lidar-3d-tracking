# SPDX-License-Identifier: AGPL-3.0-only
"""Four-panel camera/LiDAR comparison dashboard for YOLO11 and YOLO26."""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st

# Streamlit Community Cloud executes this file directly, which makes ``app/``
# the first import location instead of the repository root. Add the root before
# importing project packages so the same entry point works locally and hosted.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard_core import (
    DEFAULT_FRAME_COUNT,
    DashboardFrame,
    autoplay_interval_seconds,
    build_lidar_figure,
    build_synthetic_frame,
    clamp_frame_index,
    move_frame_index,
    playback_step,
    render_camera_view,
    seekable_frame_limit,
)


PANEL_HEIGHT = 460
PANEL_CONTENT_HEIGHT = 360
CAMERA_PANEL_ASPECT_RATIO = 2.0
DASHBOARD_BACKGROUND_RGB = (5, 8, 13)


st.set_page_config(
    page_title="Camera + LiDAR Model Lab",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #05080d; }
      [data-testid="stHeader"] { background: rgba(5, 8, 13, 0.85); }
      [data-testid="stSidebar"] { background: #090e17; }
      .block-container { max-width: 1800px; padding-top: 1.5rem; }
      .dashboard-kicker {
        color: #38bdf8; font-size: .78rem; font-weight: 700;
        letter-spacing: .16em; text-transform: uppercase;
      }
      .dashboard-title { margin: .15rem 0 .2rem; font-size: 2.1rem; }
      .dashboard-subtitle { color: #94a3b8; margin-bottom: .8rem; }
      [data-testid="stMetric"] {
        background: #0b121e; border: 1px solid #1e293b;
        border-radius: .7rem; padding: .55rem .8rem;
      }
      [data-testid="stImage"], .stPlotlyChart {
        border-radius: .65rem; overflow: hidden;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _environment_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _environment_text(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@st.cache_data(show_spinner=False, max_entries=DEFAULT_FRAME_COUNT)
def _portfolio_frame(index: int) -> DashboardFrame:
    return build_synthetic_frame(index, frame_count=DEFAULT_FRAME_COUNT)


@st.cache_resource(show_spinner=False, max_entries=4)
def _load_yolo_resource(
    checkpoint: str,
    checkpoint_size: int,
    checkpoint_mtime_ns: int,
) -> Any:
    # File metadata is intentionally part of the cache key so replacing a
    # checkpoint at the same path cannot leave Streamlit serving stale weights.
    _ = checkpoint_size, checkpoint_mtime_ns
    from ultralytics import YOLO

    from app.live_runtime import SharedModelResource

    return SharedModelResource(YOLO(checkpoint), threading.RLock())


def _load_model_from_path(path: Path) -> Any:
    resolved = path.expanduser().resolve()
    metadata = resolved.stat()
    return _load_yolo_resource(
        str(resolved),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def _checkpoint_signature(path: Path) -> tuple[str, int, int]:
    resolved = path.expanduser().resolve()
    metadata = resolved.stat()
    return str(resolved), int(metadata.st_size), int(metadata.st_mtime_ns)


def _reset_navigation() -> None:
    st.session_state.frame_index = 0
    st.session_state.frame_slider = 1
    st.session_state.playing = False
    st.session_state.last_autoplay_tick = time.monotonic()
    st.session_state.navigation_requires_full_rerun = False


def _set_frame(index: int, frame_count: int) -> None:
    selected = clamp_frame_index(index, frame_count)
    st.session_state.frame_index = selected
    # This runs before the slider is instantiated in the current fragment.
    st.session_state.frame_slider = selected + 1


def _frame_slider_changed(frame_count: int) -> None:
    """Commit the widget value before either a fragment or full-app rerun."""

    selected = int(st.session_state.frame_slider) - 1
    st.session_state.frame_index = clamp_frame_index(selected, frame_count)
    st.session_state.playing = False
    st.session_state.last_autoplay_tick = time.monotonic()
    # A full rerun re-evaluates the fragment's run_every interval, ensuring a
    # manual seek also stops the periodic autoplay schedule immediately.
    st.session_state.navigation_requires_full_rerun = True


def _manual_navigation(delta: int, frame_count: int) -> None:
    """Apply a button/keyboard move before timer code runs in the fragment."""

    target = move_frame_index(st.session_state.frame_index, delta, frame_count)
    _set_frame(target, frame_count)
    st.session_state.playing = False
    st.session_state.last_autoplay_tick = time.monotonic()
    st.session_state.navigation_requires_full_rerun = True


def _toggle_autoplay() -> None:
    st.session_state.playing = not st.session_state.playing
    st.session_state.last_autoplay_tick = time.monotonic()
    st.session_state.navigation_requires_full_rerun = True


def _initialize_state(frame_count: int, source_fingerprint: str) -> None:
    defaults = {
        "frame_index": 0,
        "frame_slider": 1,
        "playing": False,
        "last_autoplay_tick": time.monotonic(),
        "source_fingerprint": source_fingerprint,
        "viewer_error": None,
        "navigation_requires_full_rerun": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.source_fingerprint != source_fingerprint:
        st.session_state.source_fingerprint = source_fingerprint
        st.session_state.viewer_error = None
        _reset_navigation()
    _set_frame(st.session_state.frame_index, frame_count)


def _live_controls() -> tuple[Any | None, str]:
    from app.live_runtime import LiveDashboardConfig, create_live_runtime

    st.sidebar.subheader("Local live source")
    dataset_root = st.sidebar.text_input(
        "KITTI Tracking root",
        value=_environment_text("DASHBOARD_DATASET_ROOT"),
        placeholder="Path to a KITTI Tracking root",
    )
    sequence = st.sidebar.text_input(
        "Sequence",
        value=_environment_text("DASHBOARD_SEQUENCE", "0000"),
        max_chars=4,
    )
    split = st.sidebar.selectbox("Dataset split", ("training", "testing"), index=0)
    default_yolo11 = _environment_text(
        "DASHBOARD_YOLO11_MODEL",
        str(PROJECT_ROOT / "yolo11s.pt") if (PROJECT_ROOT / "yolo11s.pt").is_file() else "",
    )
    default_yolo26 = _environment_text(
        "DASHBOARD_YOLO26_MODEL",
        str(PROJECT_ROOT / "yolo26s.pt") if (PROJECT_ROOT / "yolo26s.pt").is_file() else "",
    )
    yolo11_model = st.sidebar.text_input("YOLO11 checkpoint", value=default_yolo11)
    yolo26_model = st.sidebar.text_input("YOLO26 checkpoint", value=default_yolo26)

    with st.sidebar.expander("Live inference settings"):
        confidence = st.slider("Confidence", 0.05, 0.95, 0.28, 0.01)
        imgsz = st.select_slider("Inference size", options=(416, 512, 640, 768, 960), value=640)
        device = st.text_input(
            "Ultralytics device",
            value=_environment_text("DASHBOARD_DEVICE", "0"),
            help="Native Windows ROCm uses the same device selector as CUDA, normally 0.",
        )
        embedder_gpu = st.checkbox(
            "Run Deep SORT embedder on accelerator",
            value=_environment_bool("DASHBOARD_EMBEDDER_GPU", False),
        )
        end2end_label = st.selectbox(
            "YOLO26 head",
            ("Automatic", "End-to-end", "One-to-many"),
            index=0,
        )
        yolo26_end2end = {
            "Automatic": None,
            "End-to-end": True,
            "One-to-many": False,
        }[end2end_label]

    missing = [
        label
        for label, value in (
            ("KITTI root", dataset_root),
            ("sequence", sequence),
            ("YOLO11 checkpoint", yolo11_model),
            ("YOLO26 checkpoint", yolo26_model),
        )
        if not value
    ]
    if missing:
        return None, "Provide " + ", ".join(missing) + " in the sidebar."

    try:
        config = LiveDashboardConfig(
            dataset_root=Path(dataset_root),
            sequence=sequence,
            split=split,
            yolo11_model=Path(yolo11_model),
            yolo26_model=Path(yolo26_model),
            device=device or None,
            confidence=confidence,
            imgsz=int(imgsz),
            embedder_gpu=embedder_gpu,
            yolo26_end2end=yolo26_end2end,
        ).normalized()
    except (FileNotFoundError, ValueError) as exc:
        return None, str(exc)

    if st.sidebar.button("Reset live trackers", width="stretch"):
        st.session_state.pop("live_runtime", None)
        st.session_state.pop("live_runtime_config", None)
        st.session_state.pop("live_runtime_key", None)
        st.session_state.viewer_error = None
        _reset_navigation()
        st.rerun()

    try:
        runtime_key = (
            config,
            _checkpoint_signature(config.yolo11_model),
            _checkpoint_signature(config.yolo26_model),
        )
    except OSError as exc:
        return None, f"Could not read checkpoint metadata: {exc}"
    if (
        st.session_state.get("live_runtime") is None
        or st.session_state.get("live_runtime_key") != runtime_key
    ):
        try:
            with st.spinner("Loading YOLO11, YOLO26, and two Deep SORT trackers…"):
                yolo11_resource = _load_model_from_path(config.yolo11_model)
                yolo26_resource = _load_model_from_path(config.yolo26_model)
                runtime = create_live_runtime(config, yolo11_resource, yolo26_resource)
        except Exception as exc:
            return None, f"Live runtime initialization failed: {type(exc).__name__}: {exc}"
        st.session_state.live_runtime = runtime
        st.session_state.live_runtime_config = config
        st.session_state.live_runtime_key = runtime_key
        st.session_state.viewer_error = None
        _reset_navigation()
    return st.session_state.live_runtime, ""


st.markdown('<div class="dashboard-kicker">Multimodal perception prototype</div>', unsafe_allow_html=True)
st.markdown('<h1 class="dashboard-title">Camera + LiDAR Model Lab</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-subtitle">Synchronized YOLO11 and YOLO26 tracking views in 2D camera space and approximate 3D LiDAR space.</div>',
    unsafe_allow_html=True,
)

live_requested = _environment_bool("DASHBOARD_ENABLE_LIVE", False)
trusted_local = _environment_bool("DASHBOARD_TRUSTED_LOCAL", False)
live_enabled = live_requested and trusted_local
source_url = _environment_text("DASHBOARD_SOURCE_URL").strip()
if live_enabled:
    source_mode = st.sidebar.radio(
        "Dashboard source",
        ("Portfolio preview", "Local KITTI + models"),
        help="The preview has no heavyweight runtime; local mode runs real inference.",
    )
else:
    source_mode = "Portfolio preview"
    if live_requested:
        st.sidebar.warning(
            "Live mode was requested but remains locked. Use the localhost-only "
            "ROCm launcher, which sets the second trusted-local guard."
        )
    else:
        st.sidebar.info(
            "Public-safe preview mode is active. Local KITTI/model inference is disabled on this host."
        )

if source_url:
    if source_url.startswith(("https://", "http://")):
        st.sidebar.link_button(
            "Source code and AGPL license",
            source_url,
            width="stretch",
        )
    else:
        st.sidebar.warning(
            "DASHBOARD_SOURCE_URL must be an http(s) URL before this app is published."
        )
else:
    st.sidebar.caption(
        "AGPL-3.0-only. Public deployments must set DASHBOARD_SOURCE_URL to "
        "the source for the deployed revision."
    )

runtime = None
configuration_error = ""
if source_mode == "Local KITTI + models":
    runtime, configuration_error = _live_controls()
    frame_count = runtime.frame_count if runtime is not None else 0
    source_fingerprint = (
        "live:"
        + hashlib.sha256(
            repr(st.session_state.get("live_runtime_key")).encode("utf-8")
        ).hexdigest()[:20]
        if runtime is not None
        else "live:invalid"
    )
else:
    frame_count = DEFAULT_FRAME_COUNT
    source_fingerprint = "portfolio-synthetic-v1"

st.sidebar.divider()
show_3d_labels = st.sidebar.checkbox(
    "Show 3D labels",
    value=True,
    help="Show class, model-local track ID, and confidence above each 3D box.",
)
show_ground_truth = st.sidebar.checkbox(
    "Show ground truth",
    value=False,
    help="Magenta boxes are available in the synthetic preview and labeled KITTI training sequences.",
)
max_points = st.sidebar.slider(
    "LiDAR display points",
    min_value=3_000,
    max_value=30_000,
    value=12_000,
    step=1_000,
    help="Display-only downsampling; live LiDAR association still uses the complete scan.",
)
playback_fps = st.sidebar.slider(
    "Autoplay speed (frames/s)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5,
)
loop_playback = st.sidebar.checkbox("Loop autoplay", value=True)

if configuration_error:
    st.error(configuration_error)
    st.info(
        "Use the built-in portfolio preview now, or launch local mode with "
        "`scripts/run_streamlit_rocm.ps1` after checking the paths."
    )
    st.stop()
if frame_count < 1:
    st.error("The selected source contains no frames.")
    st.stop()

_initialize_state(frame_count, source_fingerprint)

if st.session_state.viewer_error:
    st.error(st.session_state.viewer_error)
    if st.button("Retry current frame"):
        st.session_state.viewer_error = None
        st.rerun()
    st.stop()

if source_mode == "Portfolio preview":
    st.info(
        "This hosted sequence is synthetic and redistribution-safe. It demonstrates the real "
        "dashboard workflow and illustrative model differences; it is not a YOLO benchmark or "
        "recorded KITTI inference result."
    )
else:
    st.success(
        f"Live local mode · {runtime.source_label} · models run once per new temporal frame"
    )

interval = autoplay_interval_seconds(playback_fps)
fragment_interval: float | None = interval if st.session_state.playing else None


@st.fragment(run_every=fragment_interval)
def _viewer() -> None:
    # Widget callbacks execute before the fragment body. Promote their rerun to
    # a full-app rerun first so run_every is rebuilt, and (for navigation)
    # playback is already stopped before the timer branch below can advance.
    if st.session_state.navigation_requires_full_rerun:
        st.session_state.navigation_requires_full_rerun = False
        st.rerun(scope="app")

    now = time.monotonic()
    if st.session_state.playing:
        elapsed = now - float(st.session_state.last_autoplay_tick)
        if elapsed >= interval * 0.75:
            step = playback_step(
                st.session_state.frame_index,
                frame_count,
                loop=loop_playback,
            )
            _set_frame(step.index, frame_count)
            st.session_state.playing = step.playing
            st.session_state.last_autoplay_tick = now
            if not step.playing:
                st.rerun(scope="app")

    left, play, position, next_column = st.columns((1.0, 1.2, 5.8, 1.0), vertical_alignment="center")
    with left:
        st.button(
            "◀ Previous",
            shortcut="Left",
            disabled=st.session_state.frame_index <= 0,
            width="stretch",
            on_click=_manual_navigation,
            args=(-1, frame_count),
        )
    with play:
        play_label = "⏸ Pause" if st.session_state.playing else "▶ Autoplay"
        st.button(
            play_label,
            width="stretch",
            on_click=_toggle_autoplay,
        )
    with next_column:
        st.button(
            "Next ▶",
            shortcut="Right",
            disabled=st.session_state.frame_index >= frame_count - 1,
            width="stretch",
            on_click=_manual_navigation,
            args=(1, frame_count),
        )
    with position:
        # Deep SORT is temporal. Live mode exposes cached frames and exactly one
        # unseen frame, preventing a seek from skipping hundreds of updates.
        slider_max = seekable_frame_limit(
            frame_count,
            processed_through=(runtime.processed_through if runtime is not None else None),
        )
        if slider_max == 1:
            # Streamlit rejects a slider whose minimum and maximum are both 1.
            # The initial frame is rendered below; Next then unlocks frame 2.
            st.caption("Frame 1 Â· initial temporal frame")
        else:
            st.slider(
                "Frame",
                min_value=1,
                max_value=slider_max,
                key="frame_slider",
                label_visibility="collapsed",
                on_change=_frame_slider_changed,
                args=(frame_count,),
            )

    st.caption(
        f"Frame {st.session_state.frame_index + 1:,} of {frame_count:,} · "
        "use ← / → anywhere on the page · autoplay uses the same synchronized index"
    )
    if runtime is not None and slider_max < frame_count:
        st.caption(
            f"Temporal live mode has unlocked frames 1–{slider_max:,}; process the next "
            "frame to extend the seek range. Cached backward navigation remains instant."
        )

    try:
        if source_mode == "Portfolio preview":
            frame = _portfolio_frame(st.session_state.frame_index)
        else:
            with st.spinner(
                "Processing unseen temporal frames through YOLO11, YOLO26, and Deep SORT…"
            ):
                frame = runtime.frame(st.session_state.frame_index)
    except Exception as exc:
        st.session_state.playing = False
        st.session_state.viewer_error = (
            f"Frame processing failed: {type(exc).__name__}: {exc}"
        )
        st.rerun(scope="app")
        return

    if st.session_state.playing and (frame.yolo11.error or frame.yolo26.error):
        st.session_state.playing = False
        st.rerun(scope="app")

    metrics = st.columns(4)
    metrics[0].metric("Source", frame.source_label)
    metrics[1].metric("YOLO11 tracks", len(frame.yolo11.annotations))
    metrics[2].metric("YOLO26 tracks", len(frame.yolo26.annotations))
    located = sum(item.center_velodyne is not None for item in frame.yolo11.annotations)
    located += sum(item.center_velodyne is not None for item in frame.yolo26.annotations)
    total_tracks = len(frame.yolo11.annotations) + len(frame.yolo26.annotations)
    metrics[3].metric("3D-supported", f"{located}/{total_tracks}")

    chart_config = {
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["toImage"],
    }
    model_rows = (
        ("yolo11", "YOLO11", frame.yolo11),
        ("yolo26", "YOLO26", frame.yolo26),
    )
    for model_key, model_label, analysis in model_rows:
        camera_column, lidar_column = st.columns(2, gap="medium")
        with camera_column:
            with st.container(border=True, height=PANEL_HEIGHT):
                st.subheader(f"{model_label} · Camera 2D")
                camera = render_camera_view(
                    frame.image_rgb,
                    analysis,
                    model_label,
                    frame_index=frame.frame_index,
                    frame_count=frame.frame_count,
                    show_ground_truth=show_ground_truth,
                    target_aspect_ratio=CAMERA_PANEL_ASPECT_RATIO,
                    letterbox_color=DASHBOARD_BACKGROUND_RGB,
                )
                st.image(camera, width="stretch")
                if analysis.error:
                    st.error(analysis.error)
        with lidar_column:
            with st.container(border=True, height=PANEL_HEIGHT):
                st.subheader(f"{model_label} · LiDAR 3D")
                figure = build_lidar_figure(
                    frame.points_xyzi,
                    analysis,
                    model_label,
                    max_points=max_points,
                    show_ground_truth=show_ground_truth,
                    show_labels=show_3d_labels,
                    uirevision=f"{source_fingerprint}:lidar",
                    height=PANEL_CONTENT_HEIGHT,
                )
                st.plotly_chart(
                    figure,
                    width="stretch",
                    height=PANEL_CONTENT_HEIGHT,
                    key=f"{model_key}_lidar_chart",
                    config=chart_config,
                )

    st.caption(
        "3D predictions are LiDAR-supported approximations: centers come from projected depth "
        "clusters, dimensions are class priors, and predicted yaw is fixed at zero. YOLO11 and "
        "YOLO26 track IDs are independent even when their numeric values match."
    )
    if source_url and source_url.startswith(("https://", "http://")):
        st.caption(
            "Licensed under GNU AGPL v3.0 only · "
            f"[Source code and license for this deployment]({source_url})"
        )
    else:
        st.caption("Licensed under GNU AGPL v3.0 only · See LICENSE in the repository.")


_viewer()

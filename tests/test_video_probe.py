"""Unit tests for ``qipedc2vsl400.video_probe`` (Task 4).

These tests never touch real video files. They install a fake
``cv2.VideoCapture`` (via :func:`unittest.mock.patch`) that returns scripted
property values, so the suite runs fully offline and deterministically.

Covers Requirements 4.3 (fps/num_frames/length/resolution + ``length_seconds``
formula and ``fps == 0`` guard) and 4.4 (``None`` on unreadable files, plus the
frame-count fallback when the container reports an unreliable frame count).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2

from qipedc2vsl400.video_probe import VideoProps, probe


class FakeVideoCapture:
    """A stand-in for :class:`cv2.VideoCapture`.

    Constructed with the property values the test wants returned. ``opened``
    controls :meth:`isOpened`; ``read_frames`` is the number of frames the
    fallback ``grab`` loop will yield before reporting end-of-stream.
    """

    def __init__(
        self,
        *,
        opened: bool = True,
        fps: float = 0.0,
        frame_count: float = 0.0,
        height: float = 0.0,
        read_frames: int = 0,
    ):
        self._opened = opened
        self._props = {
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_COUNT: frame_count,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_POS_FRAMES: 0,
        }
        self._read_frames = read_frames
        self._grabbed = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 (match cv2 API)
        return self._opened

    def get(self, prop_id):
        return self._props.get(prop_id, 0.0)

    def set(self, prop_id, value):
        self._props[prop_id] = value
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self._grabbed = int(value)
        return True

    def grab(self) -> bool:
        if self._grabbed >= self._read_frames:
            return False
        self._grabbed += 1
        return True

    def release(self) -> None:
        self.released = True


def _patch_capture(fake: FakeVideoCapture):
    """Patch ``cv2.VideoCapture`` to return *fake* regardless of arguments."""
    return patch("qipedc2vsl400.video_probe.cv2.VideoCapture", return_value=fake)


# --- normal clip (Req 4.3) --------------------------------------------------


def test_probe_normal_clip():
    fake = FakeVideoCapture(opened=True, fps=25.0, frame_count=65, height=720)
    with _patch_capture(fake):
        props = probe(Path("D0530.mp4"))

    assert props == VideoProps(
        fps=25.0, num_frames=65, length_seconds=2.6, resolution=720
    )
    assert fake.released is True


def test_probe_length_seconds_rounded_to_two_dp():
    # 100 / 30 = 3.333... -> rounds to 3.33
    fake = FakeVideoCapture(opened=True, fps=30.0, frame_count=100, height=1080)
    with _patch_capture(fake):
        props = probe(Path("clip.mp4"))

    assert props is not None
    assert props.length_seconds == 3.33
    assert props.resolution == 1080


# --- fps == 0 guard (Req 4.3) -----------------------------------------------


def test_probe_fps_zero_guard():
    # fps reported as 0 must not raise ZeroDivisionError; length -> 0.0.
    fake = FakeVideoCapture(opened=True, fps=0.0, frame_count=50, height=720)
    with _patch_capture(fake):
        props = probe(Path("noheader.mp4"))

    assert props is not None
    assert props.fps == 0.0
    assert props.num_frames == 50
    assert props.length_seconds == 0.0
    assert props.resolution == 720


# --- frame-count fallback (Req 4.4) -----------------------------------------


def test_probe_frame_count_fallback_when_unreliable():
    # Container reports 0 frames, so probe counts by reading the stream.
    fake = FakeVideoCapture(
        opened=True, fps=25.0, frame_count=0, height=720, read_frames=40
    )
    with _patch_capture(fake):
        props = probe(Path("badcount.mp4"))

    assert props is not None
    assert props.num_frames == 40
    assert props.length_seconds == round(40 / 25.0, 2)  # 1.6


# --- unreadable path (Req 4.4) ----------------------------------------------


def test_probe_unreadable_returns_none():
    fake = FakeVideoCapture(opened=False)
    with _patch_capture(fake):
        props = probe(Path("missing.mp4"))

    assert props is None
    # The capture is still released even when it never opened.
    assert fake.released is True


def test_probe_no_usable_signal_returns_none():
    # Opened but every property is empty and no frames can be read.
    fake = FakeVideoCapture(
        opened=True, fps=0.0, frame_count=0, height=0, read_frames=0
    )
    with _patch_capture(fake):
        props = probe(Path("empty.mp4"))

    assert props is None

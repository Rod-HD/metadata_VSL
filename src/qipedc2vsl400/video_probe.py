"""Probe QIPEDC video files for real clip properties (Requirement 4.3/4.4).

This module opens a video with :class:`cv2.VideoCapture` and reads the
properties the VSL400 metadata schema needs:

* ``fps``            -> ``cv2.CAP_PROP_FPS``
* ``num_frames``     -> ``cv2.CAP_PROP_FRAME_COUNT``
* ``resolution``     -> ``cv2.CAP_PROP_FRAME_HEIGHT`` (vertical pixel height)
* ``length_seconds`` -> ``round(num_frames / fps, 2)`` (guarding ``fps == 0``)

When the container's reported frame count is missing or unreliable (``0`` or
negative, which some codecs/containers report) the probe falls back to counting
frames by reading the stream. If the file cannot be opened or yields no usable
information at all, :func:`probe` returns ``None`` so callers can apply the
``on_missing_video`` policy (default skip-and-log).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoProps:
    """Measured properties of a single video clip.

    Mirrors the design's ``VideoProps`` model.

    Attributes:
        fps: Frames per second (``CAP_PROP_FPS``).
        num_frames: Total number of frames in the clip.
        length_seconds: Clip duration in seconds, ``round(num_frames / fps, 2)``.
        resolution: Vertical pixel height (e.g. ``720``).
    """

    fps: float
    num_frames: int
    length_seconds: float
    resolution: int


def _count_frames_by_reading(capture: "cv2.VideoCapture") -> int:
    """Count frames by reading the stream end-to-end.

    Used as a fallback when ``CAP_PROP_FRAME_COUNT`` is missing or unreliable.
    Rewinds to the first frame before counting when supported.
    """
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    except cv2.error:
        pass
    count = 0
    while True:
        grabbed = capture.grab()
        if not grabbed:
            break
        count += 1
    return count


def probe(video_path: Path) -> VideoProps | None:
    """Read real clip properties from *video_path*.

    Opens the file with :class:`cv2.VideoCapture` and reads ``fps``,
    ``num_frames`` and the frame height. ``length_seconds`` is computed as
    ``round(num_frames / fps, 2)`` with a guard for ``fps == 0`` (yielding
    ``0.0``). If the reported frame count is ``0`` or negative the frames are
    counted by reading the stream.

    Args:
        video_path: Path to the video file to probe.

    Returns:
        A :class:`VideoProps` instance, or ``None`` when the file cannot be
        opened or read (the caller then applies ``on_missing_video``).
    """
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None

        fps_raw = capture.get(cv2.CAP_PROP_FPS)
        fps = float(fps_raw) if fps_raw and fps_raw > 0 else 0.0

        frame_count_raw = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        try:
            num_frames = int(frame_count_raw)
        except (TypeError, ValueError):
            num_frames = 0

        # Fall back to counting frames when the container's count is unreliable.
        if num_frames <= 0:
            num_frames = _count_frames_by_reading(capture)

        height_raw = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        try:
            resolution = int(height_raw)
        except (TypeError, ValueError):
            resolution = 0

        # If we still have no usable signal at all, treat the file as unreadable.
        if num_frames <= 0 and resolution <= 0 and fps <= 0:
            return None

        if fps > 0:
            length_seconds = round(num_frames / fps, 2)
        else:
            length_seconds = 0.0

        return VideoProps(
            fps=fps,
            num_frames=num_frames,
            length_seconds=length_seconds,
            resolution=resolution,
        )
    finally:
        capture.release()

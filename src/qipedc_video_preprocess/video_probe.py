"""Probe thuộc tính video gốc (fps, số frame, độ phân giải) — Requirement 5.3.

Module mở video bằng :class:`cv2.VideoCapture` và đọc các thuộc tính mà bước cắt
cần để giữ nguyên fps và độ phân giải (w, h) của video gốc:

* ``fps``            -> ``cv2.CAP_PROP_FPS``
* ``num_frames``     -> ``cv2.CAP_PROP_FRAME_COUNT``
* ``resolution``     -> ``(CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT)``
* ``length_seconds`` -> ``round(num_frames / fps, 2)`` (chặn ``fps == 0``)

Khác với ``qipedc2vsl400.video_probe.VideoProps`` (chỉ giữ chiều cao), ``VideoProps``
ở đây giữ **cả chiều rộng lẫn chiều cao** vì Req 5.3 yêu cầu video con phải giữ nguyên
đúng cả ``width`` và ``height`` của video gốc.

Khi container báo số frame thiếu/không tin cậy (``0`` hoặc âm), probe đếm lại bằng cách
đọc luồng. Nếu file không mở được hoặc không có tín hiệu nào dùng được, :func:`probe`
trả về ``None`` để caller áp dụng chính sách xử lý video không đọc được.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoProps:
    """Thuộc tính đo được của một video clip.

    Mô hình ``VideoProps`` theo design.md cho package ``qipedc_video_preprocess``.
    Khác với phiên bản của ``qipedc2vsl400`` ở chỗ ``resolution`` là cặp
    ``(width, height)`` để giữ nguyên cả chiều rộng lẫn chiều cao khi cắt (Req 5.3).

    Attributes:
        fps: Số khung hình mỗi giây (``CAP_PROP_FPS``).
        num_frames: Tổng số frame trong clip.
        length_seconds: Thời lượng clip (giây), ``round(num_frames / fps, 2)``.
        resolution: Cặp ``(width, height)`` tính bằng pixel.
    """

    fps: float
    num_frames: int
    length_seconds: float
    resolution: tuple[int, int]

    @property
    def width(self) -> int:
        """Chiều rộng (pixel) — tiện ích truy cập ``resolution[0]``."""
        return self.resolution[0]

    @property
    def height(self) -> int:
        """Chiều cao (pixel) — tiện ích truy cập ``resolution[1]``."""
        return self.resolution[1]


def _count_frames_by_reading(capture: "cv2.VideoCapture") -> int:
    """Đếm frame bằng cách đọc hết luồng.

    Dùng làm phương án dự phòng khi ``CAP_PROP_FRAME_COUNT`` thiếu/không tin cậy.
    Tua về frame đầu trước khi đếm nếu được hỗ trợ.
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
    """Đọc thuộc tính thật của clip tại *video_path*.

    Mở file bằng :class:`cv2.VideoCapture` rồi đọc ``fps``, ``num_frames`` và
    cặp ``(width, height)``. ``length_seconds`` tính bằng ``round(num_frames / fps, 2)``
    với chặn ``fps == 0`` (trả ``0.0``). Nếu số frame container báo là ``0`` hoặc âm
    thì đếm lại bằng cách đọc luồng.

    Args:
        video_path: Đường dẫn tới file video cần probe.

    Returns:
        Một :class:`VideoProps`, hoặc ``None`` khi file không mở/đọc được.
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

        # Đếm lại frame khi số container báo không tin cậy.
        if num_frames <= 0:
            num_frames = _count_frames_by_reading(capture)

        width_raw = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        try:
            width = int(width_raw)
        except (TypeError, ValueError):
            width = 0
        if width < 0:
            width = 0

        height_raw = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        try:
            height = int(height_raw)
        except (TypeError, ValueError):
            height = 0
        if height < 0:
            height = 0

        # Không còn tín hiệu nào dùng được -> coi như file không đọc được.
        if num_frames <= 0 and width <= 0 and height <= 0 and fps <= 0:
            return None

        if fps > 0:
            length_seconds = round(num_frames / fps, 2)
        else:
            length_seconds = 0.0

        return VideoProps(
            fps=fps,
            num_frames=num_frames,
            length_seconds=length_seconds,
            resolution=(width, height),
        )
    finally:
        capture.release()

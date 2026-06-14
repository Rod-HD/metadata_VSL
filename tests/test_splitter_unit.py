"""Unit tests cho ``qipedc_video_preprocess.splitter.write_clip`` — lớp I/O biên.

Phủ các nhánh I/O của Bộ_Cắt_Video không thuộc lớp logic thuần (đã được
property-test ở ``test_preprocess_properties.py``):

* **Req 5.3** — :func:`write_clip` mở ``cv2.VideoWriter`` với đúng ``fps`` và
  ``(width, height)`` của video gốc, và ghi đúng số frame trong khoảng
  ``[start_frame, end_frame]`` (inclusive).
* **Req 6.4 / 9.3** — ghi vào cùng đường dẫn (ghi đè) không thêm hậu tố: tên file
  đầu ra truyền cho ``VideoWriter`` đúng bằng ``out_path`` đã cho.
* **Req 5.6** — lỗi mở/ghi clip → ghi log lỗi kèm ``video_id`` & số cách, trả về
  ``False``, không ném ra ngoài (caller tiếp tục video khác).
* **Req 5.7** — nếu CHÍNH việc ghi log thất bại, lỗi được để lan ra → ``write_clip``
  ném (lần chạy dừng), bảo đảm không lỗi nào bị mất âm thầm.

Toàn bộ dùng fake ``cv2.VideoCapture`` / ``cv2.VideoWriter`` (monkeypatch module
``cv2``) nên không cần video thật hay codec.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from qipedc_video_preprocess.segmenter import VariantSpan
from qipedc_video_preprocess.splitter import write_clip
from qipedc_video_preprocess.video_probe import VideoProps


# --------------------------------------------------------------------------- #
# Tiện ích test
# --------------------------------------------------------------------------- #
class CapturingHandler(logging.Handler):
    """Handler thu thập mọi :class:`logging.LogRecord` để kiểm tra trong test."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        self.records.append(record)

    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]


@pytest.fixture
def logger_and_handler() -> tuple[logging.Logger, CapturingHandler]:
    """Logger thật, cô lập, gắn :class:`CapturingHandler`."""
    handler = CapturingHandler()
    logger = logging.getLogger(f"test_splitter_unit.{id(handler)}")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, handler


class FakeCapture:
    """Thay ``cv2.VideoCapture``: ``read`` trả ``(True, <frame_index>)`` tuần tự.

    "Frame" là chính chỉ số frame của nó (một ``int``), đủ để
    :class:`FakeWriter` ghi nhận đã ghi bao nhiêu frame và những frame nào.
    ``set(CAP_PROP_POS_FRAMES, idx)`` định vị con trỏ đọc. ``total`` giới hạn số
    frame có thể đọc (mô phỏng độ dài video gốc).
    """

    def __init__(self, path: str, total: int = 10_000, opened: bool = True) -> None:
        self.path = path
        self._pos = 0
        self._total = total
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - khớp API cv2
        return self._opened

    def set(self, prop, value) -> bool:  # noqa: A003 - khớp API cv2
        self._pos = int(value)
        return True

    def read(self):
        if self._pos >= self._total:
            return False, None
        frame = self._pos
        self._pos += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class FakeWriter:
    """Thay ``cv2.VideoWriter``: ghi nhận tham số khởi tạo & các frame đã ghi."""

    instances: list["FakeWriter"] = []

    def __init__(self, path, fourcc, fps, size, opened: bool = True) -> None:
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self._opened = opened
        self.frames: list[int] = []
        self.released = False
        FakeWriter.instances.append(self)

    def isOpened(self) -> bool:  # noqa: N802 - khớp API cv2
        return self._opened

    def write(self, frame) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


def _install_fakes(
    monkeypatch,
    *,
    capture_total: int = 10_000,
    capture_opened: bool = True,
    writer_opened: bool = True,
) -> None:
    """Monkeypatch ``cv2`` để ``write_clip`` dùng fake capture/writer."""
    import cv2

    FakeWriter.instances.clear()

    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda path: FakeCapture(path, total=capture_total, opened=capture_opened),
    )
    monkeypatch.setattr(
        cv2,
        "VideoWriter",
        lambda path, fourcc, fps, size: FakeWriter(
            path, fourcc, fps, size, opened=writer_opened
        ),
    )
    monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *args: 0x7634706D, raising=False)


def make_props(fps: float = 25.0, width: int = 1280, height: int = 720) -> VideoProps:
    length_seconds = round(1000 / fps, 2) if fps > 0 else 0.0
    return VideoProps(
        fps=fps,
        num_frames=1000,
        length_seconds=length_seconds,
        resolution=(width, height),
    )


# --------------------------------------------------------------------------- #
# Req 5.3 — giữ nguyên fps/(w,h) và ghi đúng khoảng frame
# --------------------------------------------------------------------------- #
def test_write_clip_preserves_fps_and_resolution(monkeypatch, logger_and_handler):
    logger, _ = logger_and_handler
    _install_fakes(monkeypatch)

    props = make_props(fps=29.97, width=640, height=480)
    span = VariantSpan(variant_index=1, start_frame=10, end_frame=19)  # 10 frames

    ok = write_clip(
        "D:/projects/metadata_VSL/Dataset/raw_videos/W00202.mp4",
        "D:/projects/metadata_VSL/Dataset/processed_videos/split_variants/W00202_c1.mp4",
        span,
        props,
        logger,
    )

    assert ok is True
    assert len(FakeWriter.instances) == 1
    writer = FakeWriter.instances[0]

    # fps và (w, h) truyền vào VideoWriter đúng bằng giá trị video gốc (Req 5.3).
    assert writer.fps == 29.97
    assert writer.size == (640, 480)

    # Đúng các frame trong [start, end] inclusive được ghi: 10..19 → 10 frame.
    assert writer.frames == list(range(10, 20))
    assert writer.released is True


# --------------------------------------------------------------------------- #
# Req 6.4 / 9.3 — ghi đè cùng tên, không thêm hậu tố
# --------------------------------------------------------------------------- #
def test_write_clip_writes_to_exact_path_no_suffix(monkeypatch, logger_and_handler):
    logger, _ = logger_and_handler
    _install_fakes(monkeypatch)

    out = "D:/projects/metadata_VSL/Dataset/processed_videos/split_variants/W00381_c2.mp4"
    span = VariantSpan(variant_index=2, start_frame=0, end_frame=4)

    ok = write_clip(
        "D:/projects/metadata_VSL/Dataset/raw_videos/W00381.mp4",
        out,
        span,
        make_props(),
        logger,
    )

    assert ok is True
    writer = FakeWriter.instances[0]
    # Đường dẫn đầu ra truyền cho VideoWriter đúng bằng out_path — không hậu tố
    # phụ, nên chạy lại sẽ ghi đè cùng file (Req 6.4 / 9.3).
    assert Path(writer.path) == Path(out)


def test_write_clip_overwrite_same_name_across_runs(monkeypatch, logger_and_handler):
    """Chạy hai lần với cùng out_path → cùng đường dẫn đích, không sinh tên khác."""
    logger, _ = logger_and_handler
    out = "D:/projects/metadata_VSL/Dataset/processed_videos/split_variants/W1_c1.mp4"
    span = VariantSpan(variant_index=1, start_frame=0, end_frame=2)
    src = "D:/projects/metadata_VSL/Dataset/raw_videos/W1.mp4"

    _install_fakes(monkeypatch)
    assert write_clip(src, out, span, make_props(), logger) is True
    first_path = FakeWriter.instances[0].path

    _install_fakes(monkeypatch)
    assert write_clip(src, out, span, make_props(), logger) is True
    second_path = FakeWriter.instances[0].path

    assert Path(first_path) == Path(second_path) == Path(out)


# --------------------------------------------------------------------------- #
# Req 5.6 — lỗi ghi clip → log + tiếp tục (False), không ném
# --------------------------------------------------------------------------- #
def test_write_clip_capture_not_opened_logs_and_returns_false(
    monkeypatch, logger_and_handler
):
    logger, handler = logger_and_handler
    _install_fakes(monkeypatch, capture_opened=False)

    span = VariantSpan(variant_index=1, start_frame=0, end_frame=9)
    ok = write_clip(
        "D:/projects/metadata_VSL/Dataset/raw_videos/BAD.mp4",
        "D:/projects/metadata_VSL/Dataset/processed_videos/split_variants/BAD.mp4",
        span,
        make_props(),
        logger,
    )

    assert ok is False
    # Lỗi được ghi log kèm tên file (suy ra video_id) — Req 5.6.
    assert any("BAD" in m for m in handler.messages())


def test_write_clip_writer_not_opened_logs_and_returns_false(
    monkeypatch, logger_and_handler
):
    logger, handler = logger_and_handler
    _install_fakes(monkeypatch, writer_opened=False)

    span = VariantSpan(variant_index=3, start_frame=0, end_frame=9)
    ok = write_clip(
        "D:/projects/metadata_VSL/Dataset/raw_videos/W00202.mp4",
        "D:/projects/metadata_VSL/Dataset/processed_videos/split_variants/W00202_c3.mp4",
        span,
        make_props(),
        logger,
    )

    assert ok is False
    assert any("W00202_c3" in m for m in handler.messages())


def test_write_clip_no_frames_in_range_logs_and_returns_false(
    monkeypatch, logger_and_handler
):
    """Nguồn hết frame trước khi đọc được frame nào trong khoảng → False + log."""
    logger, handler = logger_and_handler
    # total=5 nhưng span bắt đầu ở frame 100 → read() trả về hết ngay.
    _install_fakes(monkeypatch, capture_total=5)

    span = VariantSpan(variant_index=1, start_frame=100, end_frame=110)
    ok = write_clip(
        "D:/projects/metadata_VSL/Dataset/raw_videos/W5.mp4",
        "D:/projects/metadata_VSL/Dataset/processed_videos/split_variants/W5_c1.mp4",
        span,
        make_props(),
        logger,
    )

    assert ok is False
    assert any("W5_c1" in m for m in handler.messages())


def test_write_clip_invalid_props_logs_and_returns_false(
    monkeypatch, logger_and_handler
):
    """fps/(w,h) không hợp lệ → không thể giữ nguyên thuộc tính → False + log."""
    logger, handler = logger_and_handler
    _install_fakes(monkeypatch)

    span = VariantSpan(variant_index=1, start_frame=0, end_frame=9)
    bad_props = make_props(fps=0.0)  # fps không hợp lệ
    ok = write_clip(
        "D:/projects/metadata_VSL/Dataset/raw_videos/W00202.mp4",
        "D:/projects/metadata_VSL/Dataset/processed_videos/split_variants/W00202_c1.mp4",
        span,
        bad_props,
        logger,
    )

    assert ok is False
    assert any("W00202_c1" in m for m in handler.messages())


# --------------------------------------------------------------------------- #
# Req 5.7 — lỗi ghi-log → để lan ra (dừng), không nuốt âm thầm
# --------------------------------------------------------------------------- #
class ExplodingLogger:
    """Logger giả mà ``error`` luôn ném — mô phỏng lỗi ghi log (Req 5.7)."""

    def error(self, *args, **kwargs):
        raise OSError("không ghi được log")


def test_write_clip_logging_failure_propagates(monkeypatch):
    """Khi clip lỗi VÀ việc ghi log cũng lỗi → write_clip ném (dừng), không nuốt."""
    # capture không mở được → đi vào nhánh logger.error(...) đầu tiên.
    _install_fakes(monkeypatch, capture_opened=False)

    span = VariantSpan(variant_index=1, start_frame=0, end_frame=9)
    with pytest.raises(OSError, match="không ghi được log"):
        write_clip(
            "D:/projects/metadata_VSL/Dataset/raw_videos/BAD.mp4",
            "D:/projects/metadata_VSL/Dataset/processed_videos/split_variants/BAD.mp4",
            span,
            make_props(),
            ExplodingLogger(),
        )

"""Unit tests cho ``qipedc_video_preprocess.segmenter`` — xử lý lỗi detector.

Phủ **Requirement 4.8**: nếu Bộ_Phát_Hiện_Số báo lỗi hoặc quá thời gian khi xử lý
một frame mẫu, THE Bộ_Phân_Đoạn SHALL coi frame đó là **không có số hợp lệ**, ghi
log lý do, và **tiếp tục** với các frame mẫu còn lại (không hủy cả lần chạy).

Hai điểm vào được kiểm:

* :func:`segmenter.segment_video` — dùng một :class:`FakeNumberDetector` ném lỗi
  trên một frame nhất định và một :class:`FakeCapture` thay cho
  ``cv2.VideoCapture`` (monkeypatch ``cv2.VideoCapture`` trong module segmenter)
  để không cần video thật.
* :func:`segmenter.refine_boundary` — dùng một **nguồn frame callable** (không cần
  OpenCV) với detector ném lỗi trên đúng một chỉ số frame được dò tới.

Mọi cảnh báo được bắt bằng một logger thật gắn :class:`CapturingHandler` để khẳng
định lý do lỗi đã được ghi log.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from qipedc_video_preprocess import segmenter
from qipedc_video_preprocess.config import PreprocessConfig
from qipedc_video_preprocess.discovery import VideoEntry
from qipedc_video_preprocess.number_detector import DetectionResult
from qipedc_video_preprocess.video_probe import VideoProps

# project_root giả lập nằm trên ổ D: (Req 1.3). Không chạm đĩa: capture được
# monkeypatch và refine_boundary dùng nguồn frame callable.
PROJECT_ROOT = Path("D:/projects/metadata_VSL")


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

    def messages_at(self, level: int) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno == level]


@pytest.fixture
def logger_and_handler() -> tuple[logging.Logger, CapturingHandler]:
    """Logger thật, cô lập, gắn :class:`CapturingHandler`."""
    handler = CapturingHandler()
    logger = logging.getLogger(f"test_segmenter_unit.{id(handler)}")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, handler


class FakeNumberDetector:
    """Bộ phát hiện số giả lập theo chỉ số frame.

    *frame* truyền vào ``detect`` chính là **chỉ số frame** (do
    :class:`FakeCapture` và nguồn frame callable trả về chỉ số). Detector tra
    ``numbers`` để lấy giá trị (``int`` hoặc ``None``); nếu chỉ số nằm trong
    ``raise_on`` thì **ném lỗi** để mô phỏng lỗi/timeout của OCR (Req 4.8).
    """

    def __init__(
        self,
        numbers: dict[int, int | None],
        raise_on: set[int] | None = None,
        exc: BaseException | None = None,
    ) -> None:
        self._numbers = numbers
        self._raise_on = set(raise_on or set())
        self._exc = exc or TimeoutError("OCR quá thời gian")
        self.calls: list[int] = []

    def detect(self, frame) -> DetectionResult:
        index = int(frame)
        self.calls.append(index)
        if index in self._raise_on:
            raise self._exc
        number = self._numbers.get(index)
        confidence = 0.0 if number is None else 0.9
        return DetectionResult(number=number, confidence=confidence)


class FakeCapture:
    """Thay thế ``cv2.VideoCapture``: ``read`` trả về chỉ số frame hiện tại.

    ``set(CAP_PROP_POS_FRAMES, idx)`` ghi nhận vị trí; ``read`` trả ``(True, idx)``
    nên "frame" chính là chỉ số — đủ để :class:`FakeNumberDetector` ánh xạ thành
    số. Không phụ thuộc OpenCV/video thật.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._pos = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - khớp API cv2
        return True

    def set(self, prop, value) -> bool:  # noqa: A003 - khớp API cv2
        self._pos = int(value)
        return True

    def read(self):
        return True, self._pos

    def release(self) -> None:
        self.released = True


def make_cfg(**overrides) -> PreprocessConfig:
    """Dựng ``PreprocessConfig`` thật với project_root trên D:.

    fps=1.0 + ``sample_interval_seconds=1.0`` cho bước lấy mẫu = 1, nên chỉ số
    frame mẫu trùng ``range(0, num_frames)`` — dễ suy luận trong test.
    """
    params = dict(project_root=PROJECT_ROOT, sample_interval_seconds=1.0)
    params.update(overrides)
    return PreprocessConfig(**params)


def make_entry(video_id: str = "W00202") -> VideoEntry:
    return VideoEntry(
        video_id=video_id,
        path=Path(f"D:/projects/metadata_VSL/Dataset/raw_videos/{video_id}.mp4"),
        source_dir="Dataset/raw_videos",
    )


def make_props(num_frames: int, fps: float = 1.0) -> VideoProps:
    return VideoProps(
        fps=fps,
        num_frames=num_frames,
        length_seconds=round(num_frames / fps, 2),
        resolution=(1280, 720),
    )


# --------------------------------------------------------------------------- #
# segment_video — Req 4.8: lỗi detector trên 1 frame → None + log + tiếp tục
# --------------------------------------------------------------------------- #
def test_segment_video_detector_error_on_one_frame_treated_as_no_number(
    monkeypatch, logger_and_handler
):
    """Detector ném lỗi trên frame mẫu 0 → coi như không có số, vẫn hoàn tất.

    Các frame còn lại đều "không có số" nên video phân loại là single; điều quan
    trọng là lần chạy KHÔNG bị hủy bởi lỗi và frame lỗi được ghi là ``None``.
    """
    import cv2

    logger, handler = logger_and_handler
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    # num_frames=4, fps=1, interval=1 -> sample indices [0,1,2,3].
    detector = FakeNumberDetector(
        numbers={0: None, 1: None, 2: None, 3: None},
        raise_on={0},
        exc=RuntimeError("OCR engine crashed"),
    )
    cfg = make_cfg()

    result = segmenter.segment_video(
        make_entry(), make_props(num_frames=4), detector, cfg, logger=logger
    )

    # Lần chạy hoàn tất (không ném) và frame lỗi được coi là không có số.
    assert result.observed_numbers == (None, None, None, None)
    assert result.kind == "single"
    assert result.variant_count == 1

    # Tất cả frame mẫu đều được thử (tiếp tục sau frame lỗi).
    assert detector.calls == [0, 1, 2, 3]

    # Lý do lỗi được ghi log (Req 4.8), nhắc tới frame lỗi và loại lỗi.
    warnings = handler.messages_at(logging.WARNING)
    frame0_warnings = [m for m in warnings if "frame 0" in m]
    assert len(frame0_warnings) == 1
    assert "RuntimeError" in frame0_warnings[0]
    assert "4.8" in frame0_warnings[0]


def test_segment_video_continues_detecting_after_error(
    monkeypatch, logger_and_handler
):
    """Sau frame lỗi, các frame còn lại VẪN được phát hiện số bình thường.

    detect: 0->5, 1->lỗi, 2->5, 3->5  =>  observed = (5, None, 5, 5). Chuỗi số
    không bắt đầu từ 1 nên phân loại manual_review, nhưng điểm cốt lõi là chuỗi
    quan sát giữ đúng số ở các frame không lỗi và ``None`` ở frame lỗi.
    """
    import cv2

    logger, handler = logger_and_handler
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    detector = FakeNumberDetector(
        numbers={0: 5, 1: 5, 2: 5, 3: 5},
        raise_on={1},
        exc=TimeoutError("OCR timeout"),
    )
    cfg = make_cfg()

    result = segmenter.segment_video(
        make_entry(), make_props(num_frames=4), detector, cfg, logger=logger
    )

    # Frame lỗi -> None; các frame khác giữ số đã đọc.
    assert result.observed_numbers == (5, None, 5, 5)
    assert detector.calls == [0, 1, 2, 3]

    # Chuỗi không bắt đầu từ 1 -> cần rà soát thủ công (run vẫn hoàn tất).
    assert result.kind == "manual_review"
    assert result.variant_count == 0

    frame1_warnings = [
        m for m in handler.messages_at(logging.WARNING) if "frame 1" in m
    ]
    assert len(frame1_warnings) == 1
    assert "TimeoutError" in frame1_warnings[0]
    assert "4.8" in frame1_warnings[0]


# --------------------------------------------------------------------------- #
# refine_boundary — Req 4.8: detector lỗi trên 1 chỉ số được dò → log + tiếp tục
# --------------------------------------------------------------------------- #
def test_refine_boundary_detector_error_logged_and_continues(logger_and_handler):
    """detect ném lỗi trên một frame được dò tới → coi như không mang số mới.

    Detector "đơn điệu": frame < 5 mang giá trị cũ (1), frame >= 5 mang
    ``target_number`` (2), NHƯNG ném lỗi tại frame 4. Tìm-nhị-phân dò các frame
    5, 3, 4; tại frame 4 detector ném lỗi → bị coi là không mang số mới (đúng,
    vì 4 < 5) → ghi log và tiếp tục, kết quả ranh giới vẫn đúng = 5.
    """
    logger, handler = logger_and_handler

    def detect_number(index: int) -> int:
        return 2 if index >= 5 else 1

    detector = FakeNumberDetector(
        numbers={i: detect_number(i) for i in range(0, 9)},
        raise_on={4},
        exc=RuntimeError("OCR engine crashed"),
    )

    # Nguồn frame callable: trả về chính chỉ số (không bao giờ None).
    frame_source = lambda index: index  # noqa: E731

    refined = segmenter.refine_boundary(
        detector,
        frame_source,
        coarse_lo=2,
        coarse_hi=8,
        target_number=2,
        cfg=make_cfg(),
        logger=logger,
    )

    # Ranh giới tinh chỉnh nằm trong (coarse_lo, coarse_hi] và đúng điểm chuyển.
    assert refined == 5

    # Lỗi tại frame 4 đã được dò tới và được ghi log (Req 4.8).
    assert 4 in detector.calls
    frame4_warnings = [
        m for m in handler.messages_at(logging.WARNING) if "frame 4" in m
    ]
    assert len(frame4_warnings) == 1
    assert "RuntimeError" in frame4_warnings[0]
    assert "4.8" in frame4_warnings[0]


def test_refine_boundary_unreadable_frame_logged_and_continues(logger_and_handler):
    """Nguồn frame trả ``None`` tại frame được dò → coi như không có số + log.

    Củng cố nhánh đọc-frame của Req 4.8 trong :func:`refine_boundary`: frame 4
    không đọc được (``None``) -> không mang số mới (đúng vì 4 < 5) -> ghi log và
    tiếp tục, ranh giới vẫn đúng = 5.
    """
    logger, handler = logger_and_handler

    detector = FakeNumberDetector(
        numbers={i: (2 if i >= 5 else 1) for i in range(0, 9)},
    )

    def frame_source(index: int):
        return None if index == 4 else index

    refined = segmenter.refine_boundary(
        detector,
        frame_source,
        coarse_lo=2,
        coarse_hi=8,
        target_number=2,
        cfg=make_cfg(),
        logger=logger,
    )

    assert refined == 5
    unreadable_warnings = [
        m
        for m in handler.messages_at(logging.WARNING)
        if "frame 4" in m and "không đọc được" in m
    ]
    assert len(unreadable_warnings) == 1
    assert "4.8" in unreadable_warnings[0]

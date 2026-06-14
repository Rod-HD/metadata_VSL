"""Unit tests cho ``qipedc_video_preprocess.number_detector`` — nhánh biên.

Phủ **Requirement 3.4**: khi frame nhận vào rỗng/không đọc được hoặc vùng
ROI_Góc_Trên_Trái có kích thước bằng 0, THE Bộ_Phát_Hiện_Số SHALL trả về kết quả
"không có số" (``DetectionResult(number=None, confidence=0.0)``) VÀ ghi log lý do.

Các test này xác nhận thêm rằng nhánh biên KHÔNG bao giờ khởi tạo
``easyocr.Reader`` (tức không cần ``easyocr``/``torch``): detector trả kết quả
ngay từ :func:`crop_roi` trước khi đụng tới engine OCR. Để bảo đảm điều đó một
cách tường minh, ``_ensure_reader`` được monkeypatch thành một hàm "nổ" — nếu nó
bị gọi, test sẽ thất bại.

``EasyOcrNumberDetector`` được dựng với ``models_dir`` nằm trên ổ ``D:`` (Req
1.6) và một logger thật gắn handler bắt bản ghi để khẳng định có log lý do.
Ngoài ra còn kiểm :func:`crop_roi` trực tiếp trả ``None`` cho các đầu vào biên.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from qipedc_video_preprocess.number_detector import (
    DetectionResult,
    EasyOcrNumberDetector,
    crop_roi,
)

# Thư mục trọng số giả lập nằm trên ổ D: (Req 1.6 — không ghi lên C:). Các test
# biên không bao giờ chạm tới thư mục này vì không khởi tạo engine OCR.
MODELS_DIR_ON_D = "D:/projects/metadata_VSL/Dataset/models/easyocr"


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
        """Danh sách thông điệp đã định dạng (bao gồm args)."""
        return [r.getMessage() for r in self.records]


@pytest.fixture
def logger_and_handler() -> tuple[logging.Logger, CapturingHandler]:
    """Logger thật, cô lập, gắn :class:`CapturingHandler`."""
    handler = CapturingHandler()
    logger = logging.getLogger(f"test_number_detector_unit.{id(handler)}")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, handler


@pytest.fixture
def detector(logger_and_handler, monkeypatch):
    """``EasyOcrNumberDetector`` với models_dir trên D: và engine OCR bị chặn.

    ``_ensure_reader`` được thay bằng hàm nổ: nếu detector cố khởi tạo engine
    (tức nhánh biên không hoạt động đúng), test sẽ thất bại ngay — chứng minh
    nhánh Req 3.4 không phụ thuộc easyocr/torch.
    """
    logger, handler = logger_and_handler
    det = EasyOcrNumberDetector(models_dir=MODELS_DIR_ON_D, log=logger)

    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "Nhánh biên (Req 3.4) không được khởi tạo engine OCR."
        )

    monkeypatch.setattr(det, "_ensure_reader", _boom)
    return det, handler


# --------------------------------------------------------------------------- #
# Req 3.4 — detect() trả "không có số" + log, không chạm engine OCR
# --------------------------------------------------------------------------- #
def test_detect_none_frame_returns_no_number_and_logs(detector):
    """frame=None → DetectionResult(None, 0.0) kèm log lý do."""
    det, handler = detector

    result = det.detect(None)

    assert result == DetectionResult(number=None, confidence=0.0)
    assert result.number is None
    assert result.confidence == 0.0
    # Có ít nhất một bản ghi log nêu lý do frame rỗng / ROI diện tích 0.
    assert any(
        "rỗng" in m or "diện tích 0" in m for m in handler.messages()
    )


def test_detect_empty_array_returns_no_number_and_logs(detector):
    """frame là mảng numpy rỗng (kích thước 0) → 'không có số' + log."""
    det, handler = detector

    empty = np.empty((0, 0, 3), dtype=np.uint8)
    result = det.detect(empty)

    assert result == DetectionResult(number=None, confidence=0.0)
    assert any(
        "rỗng" in m or "diện tích 0" in m for m in handler.messages()
    )


def test_detect_zero_area_roi_returns_no_number_and_logs(logger_and_handler, monkeypatch):
    """ROI diện tích 0 (roi=(0,0,0,0)) → 'không có số' + log, không gọi OCR."""
    logger, handler = logger_and_handler
    det = EasyOcrNumberDetector(
        models_dir=MODELS_DIR_ON_D, roi=(0.0, 0.0, 0.0, 0.0), log=logger
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("ROI diện tích 0 không được khởi tạo engine OCR.")

    monkeypatch.setattr(det, "_ensure_reader", _boom)

    # Frame hợp lệ (không rỗng) nhưng ROI ánh xạ ra vùng diện tích 0.
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = det.detect(frame)

    assert result == DetectionResult(number=None, confidence=0.0)
    assert any(
        "rỗng" in m or "diện tích 0" in m for m in handler.messages()
    )


# --------------------------------------------------------------------------- #
# crop_roi trực tiếp — củng cố Req 3.4 ở mức logic thuần
# --------------------------------------------------------------------------- #
def test_crop_roi_none_frame_returns_none():
    """frame=None → crop_roi trả None."""
    assert crop_roi(None, (0.0, 0.0, 0.22, 0.18)) is None


def test_crop_roi_empty_array_returns_none():
    """Mảng numpy rỗng (một chiều bằng 0) → crop_roi trả None."""
    empty = np.empty((0, 0, 3), dtype=np.uint8)
    assert crop_roi(empty, (0.0, 0.0, 0.22, 0.18)) is None


def test_crop_roi_zero_dim_frame_returns_none():
    """Frame có chiều rộng bằng 0 → crop_roi trả None."""
    zero_w = np.empty((10, 0, 3), dtype=np.uint8)
    assert crop_roi(zero_w, (0.0, 0.0, 0.22, 0.18)) is None


def test_crop_roi_zero_area_roi_returns_none():
    """ROI (0,0,0,0) trên frame hợp lệ → diện tích 0 → crop_roi trả None."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert crop_roi(frame, (0.0, 0.0, 0.0, 0.0)) is None


def test_crop_roi_degenerate_width_roi_returns_none():
    """ROI có x0==x1 (bề rộng pixel 0) → crop_roi trả None."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert crop_roi(frame, (0.5, 0.0, 0.5, 0.18)) is None


def test_crop_roi_valid_roi_returns_subarray():
    """ROI hợp lệ → crop_roi trả về đúng vùng con (không phải None)."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    cropped = crop_roi(frame, (0.0, 0.0, 0.25, 0.5))
    assert cropped is not None
    # 0.25*200=50 (rộng), 0.5*100=50 (cao).
    assert cropped.shape[:2] == (50, 50)

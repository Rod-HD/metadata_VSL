"""Bộ_Phát_Hiện_Số — OCR con số ở ROI góc trên bên trái.

Module này định nghĩa ``DetectionResult``, protocol ``NumberDetector`` và hàm
logic thuần ``crop_roi``. Phần ``interpret_ocr`` và ``EasyOcrNumberDetector``
(lớp biên gọi engine OCR) sẽ được bổ sung ở task sau. Xem design.md, Requirement 3.

Quy ước:

* ``DetectionResult.number`` ∈ ``{1..9}`` cho "có số", hoặc ``None`` cho
  "không có số"; khi ``number is None`` thì ``confidence`` bằng ``0.0``.
* ``crop_roi`` ánh xạ ROI tỉ lệ ``(x0, y0, x1, y1)`` (mỗi tọa độ 0.0–1.0) sang
  biên pixel ``(round(x0·w), round(y0·h), round(x1·w), round(y1·h))`` và cắt vùng
  tương ứng của ``frame``. Mọi biên pixel được chặn trong ``[0, w] × [0, h]``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence, runtime_checkable

logger = logging.getLogger(__name__)

# Tập chữ số hợp lệ cho con số "CÁCH": 1..9 (không nhận 0). Dùng cho cả
# ``interpret_ocr`` lẫn allowlist của EasyOCR.
_VALID_DIGITS = frozenset("123456789")


@dataclass(frozen=True)
class DetectionResult:
    """Kết quả phát hiện số trên một frame (Req 3).

    Attributes:
        number: Con số "CÁCH" đọc được, ``int`` trong ``{1..9}``; hoặc ``None``
            khi không có số hợp lệ ("không có số").
        confidence: Độ tin cậy của kết quả OCR. Bằng ``0.0`` khi ``number`` là
            ``None``.
    """

    number: int | None
    confidence: float


@runtime_checkable
class NumberDetector(Protocol):
    """Interface hẹp cho bộ phát hiện số trên một frame.

    Bọc engine OCR sau protocol này để có thể thay đổi engine (EasyOCR, ...) mà
    không ảnh hưởng phần còn lại của pipeline (Req 3).
    """

    def detect(self, frame) -> DetectionResult:
        """Phát hiện con số "CÁCH" trên *frame* và trả về :class:`DetectionResult`."""
        ...


def crop_roi(frame, roi: tuple[float, float, float, float]):
    """Cắt vùng ROI theo tỉ lệ ra khỏi *frame*.

    Ánh xạ ROI tỉ lệ ``(x0, y0, x1, y1)`` (mỗi tọa độ 0.0–1.0) sang biên pixel
    ``(round(x0·w), round(y0·h), round(x1·w), round(y1·h))`` với ``w`` là chiều
    rộng và ``h`` là chiều cao của frame. Mọi biên pixel được chặn trong
    ``[0, w] × [0, h]`` rồi dùng để cắt ``frame``.

    Args:
        frame: Mảng numpy theo bố cục ``(H, W, ...)`` (lấy từ OpenCV). Chiều cao
            là ``frame.shape[0]`` và chiều rộng là ``frame.shape[1]``.
        roi: ROI tỉ lệ ``(x0, y0, x1, y1)``, mỗi tọa độ trong ``[0, 1]``.

    Returns:
        Vùng ``frame`` đã cắt (một view numpy ``frame[py0:py1, px0:px1]``); hoặc
        ``None`` nếu frame rỗng hoặc vùng ROI có diện tích bằng 0 (Req 3.4).
    """
    # Frame rỗng: None hoặc không đủ chiều / có chiều bằng 0.
    if frame is None:
        return None
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        return None

    height = int(shape[0])
    width = int(shape[1])
    if width <= 0 or height <= 0:
        return None

    x0, y0, x1, y1 = roi

    # Tỉ lệ -> pixel theo round, rồi chặn trong [0, w] × [0, h].
    px0 = _clamp(round(x0 * width), 0, width)
    px1 = _clamp(round(x1 * width), 0, width)
    py0 = _clamp(round(y0 * height), 0, height)
    py1 = _clamp(round(y1 * height), 0, height)

    # ROI diện tích 0 (bề rộng hoặc bề cao pixel không dương) -> không có vùng.
    if px1 <= px0 or py1 <= py0:
        return None

    return frame[py0:py1, px0:px1]


def _clamp(value: int, low: int, high: int) -> int:
    """Chặn *value* vào đoạn ``[low, high]``."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def interpret_ocr(
    tokens: Sequence[tuple[str, float]],
    threshold: float,
) -> int | None:
    """Diễn giải kết quả OCR thành con số "CÁCH" (logic thuần, Req 3.2/3.3).

    Trả về một số nguyên ``n`` **khi và chỉ khi** trong *tokens* có **đúng một**
    token mà chuỗi nhận dạng (sau khi bỏ khoảng trắng hai đầu) là **một ký tự
    chữ số duy nhất** trong ``{1..9}`` **và** độ tin cậy ``>= threshold``. Trong
    mọi trường hợp còn lại trả về ``None`` ("không có số"):

    * không có token hợp lệ nào;
    * nhiều hơn một token chữ số hợp lệ;
    * chuỗi không phải đúng một chữ số (rỗng, nhiều ký tự, ký tự không phải số);
    * chữ số là ``0`` (ngoài khoảng ``[1, 9]``);
    * độ tin cậy ``< threshold``.

    Args:
        tokens: Danh sách các cặp ``(text, confidence)`` do engine OCR trả về.
        threshold: Ngưỡng tin cậy tối thiểu ``[0.0, 1.0]``.

    Returns:
        Con số ``int`` trong ``{1..9}`` nếu thỏa điều kiện trên; ngược lại
        ``None``.
    """
    qualifying: list[int] = []
    for text, confidence in tokens:
        digit = _as_single_digit(text)
        if digit is None:
            continue
        if confidence is None or float(confidence) < float(threshold):
            continue
        qualifying.append(digit)

    if len(qualifying) == 1:
        return qualifying[0]
    return None


def _as_single_digit(text: object) -> int | None:
    """Chuẩn hóa *text* và trả về chữ số ``1..9`` nếu nó là đúng một chữ số như
    vậy; ngược lại ``None``."""
    if text is None:
        return None
    stripped = str(text).strip()
    if len(stripped) != 1 or stripped not in _VALID_DIGITS:
        return None
    return int(stripped)


class EasyOcrNumberDetector:
    """Bộ phát hiện số dùng EasyOCR trên ROI góc trên bên trái (Req 3).

    Lớp này là lớp **biên** bọc engine EasyOCR. Quy trình ``detect``:

    1. Cắt ROI góc trên trái bằng :func:`crop_roi` (bỏ qua góc phải — Req 3.5).
    2. Chạy EasyOCR với ``allowlist='123456789'`` trên vùng đã cắt.
    3. Đưa kết quả qua :func:`interpret_ocr` để quyết định con số (Req 3.2/3.3).

    Frame rỗng hoặc ROI diện tích 0 → trả ``DetectionResult(None, 0.0)`` kèm log
    (Req 3.4).

    Ràng buộc ổ đĩa: trọng số EasyOCR **không** được ghi lên ``C:``. Constructor
    nhận thư mục trọng số (qua ``cfg`` hoặc ``models_dir``) và truyền tường minh
    ``model_storage_directory`` cho ``easyocr.Reader``; đồng thời đặt phòng thủ
    biến môi trường ``EASYOCR_MODULE_PATH`` về thư mục đó nếu chưa được đặt.
    ``easyocr`` được import **trễ** (bên trong lớp) để các test logic thuần và
    module khác không phụ thuộc ``torch``.
    """

    def __init__(
        self,
        cfg=None,
        *,
        models_dir: str | os.PathLike | None = None,
        roi: tuple[float, float, float, float] | None = None,
        confidence_threshold: float | None = None,
        languages: Iterable[str] = ("en",),
        gpu: bool = False,
        log: Optional[logging.Logger] = None,
    ) -> None:
        """Khởi tạo bộ phát hiện.

        Args:
            cfg: ``PreprocessConfig`` (tùy chọn) để lấy mặc định cho thư mục
                trọng số (``ocr_models_path``), ROI (``roi_top_left``) và ngưỡng
                tin cậy (``ocr_confidence_threshold``).
            models_dir: Ghi đè thư mục lưu trọng số EasyOCR. Bắt buộc phải có
                hoặc qua ``cfg`` hoặc qua tham số này để trọng số nằm trên ``D:``.
            roi: Ghi đè ROI tỉ lệ ``(x0, y0, x1, y1)``.
            confidence_threshold: Ghi đè ngưỡng tin cậy.
            languages: Danh sách ngôn ngữ cho EasyOCR (mặc định ``("en",)``).
            gpu: Có dùng GPU hay không (mặc định ``False``).
            log: Logger tùy chọn; mặc định dùng logger của module.
        """
        self._log = log if log is not None else logger

        # --- thư mục trọng số (bắt buộc phải nằm trên D:) ---
        resolved_models: Path | None = None
        if models_dir is not None:
            resolved_models = Path(models_dir)
        elif cfg is not None:
            resolved_models = Path(cfg.ocr_models_path)
        if resolved_models is None:
            raise ValueError(
                "EasyOcrNumberDetector cần thư mục trọng số: truyền 'cfg' "
                "(có ocr_models_path) hoặc 'models_dir'."
            )
        self._models_dir = resolved_models

        # --- ROI và ngưỡng tin cậy ---
        if roi is not None:
            self._roi = roi
        elif cfg is not None:
            self._roi = cfg.roi_top_left
        else:
            self._roi = (0.0, 0.0, 0.22, 0.18)

        if confidence_threshold is not None:
            self._threshold = float(confidence_threshold)
        elif cfg is not None:
            self._threshold = float(cfg.ocr_confidence_threshold)
        else:
            self._threshold = 0.5

        self._languages = tuple(languages)
        self._gpu = bool(gpu)
        self._reader = None  # khởi tạo trễ ở lần detect đầu tiên

    # ------------------------------------------------------------------ #
    # Engine OCR (khởi tạo trễ — không import easyocr ở cấp module)
    # ------------------------------------------------------------------ #
    def _ensure_reader(self):
        """Khởi tạo ``easyocr.Reader`` một lần, ghi trọng số vào thư mục trên
        ``D:`` (không bao giờ lên ``C:``)."""
        if self._reader is not None:
            return self._reader

        models_path = self._models_dir
        # Bảo đảm thư mục tồn tại để EasyOCR ghi/đọc trọng số trên D:.
        models_path.mkdir(parents=True, exist_ok=True)

        # Đặt phòng thủ EASYOCR_MODULE_PATH về thư mục trọng số nếu chưa đặt,
        # TRƯỚC khi import/khởi tạo Reader (Req 1.6 — không ghi lên C:).
        if not os.environ.get("EASYOCR_MODULE_PATH"):
            os.environ["EASYOCR_MODULE_PATH"] = str(models_path)

        # Lazy-import: chỉ phụ thuộc easyocr/torch khi thực sự cần OCR.
        import easyocr  # type: ignore

        self._reader = easyocr.Reader(
            list(self._languages),
            gpu=self._gpu,
            model_storage_directory=str(models_path),
        )
        return self._reader

    # ------------------------------------------------------------------ #
    # API NumberDetector
    # ------------------------------------------------------------------ #
    def detect(self, frame) -> DetectionResult:
        """Phát hiện con số "CÁCH" trên *frame* (Req 3.2/3.3/3.4)."""
        roi_img = crop_roi(frame, self._roi)
        if roi_img is None:
            self._log.debug(
                "Frame rỗng hoặc ROI diện tích 0 — trả 'không có số'."
            )
            return DetectionResult(number=None, confidence=0.0)

        reader = self._ensure_reader()
        # allowlist chỉ chữ số 1–9; không nhận diện ký tự khác.
        results = reader.readtext(roi_img, allowlist="123456789")

        tokens = _tokens_from_easyocr(results)
        number = interpret_ocr(tokens, self._threshold)
        if number is None:
            return DetectionResult(number=None, confidence=0.0)

        # Lấy độ tin cậy của token chữ số hợp lệ tương ứng (có đúng một).
        confidence = next(
            (
                float(conf)
                for text, conf in tokens
                if _as_single_digit(text) == number
                and conf is not None
                and float(conf) >= self._threshold
            ),
            0.0,
        )
        return DetectionResult(number=number, confidence=confidence)


def _tokens_from_easyocr(results: Iterable) -> list[tuple[str, float]]:
    """Chuyển kết quả ``easyocr.Reader.readtext`` thành danh sách
    ``(text, confidence)`` cho :func:`interpret_ocr`.

    EasyOCR (detail=1) trả mỗi phần tử dạng ``(bbox, text, confidence)``. Hàm
    này phòng thủ với các định dạng khác (ví dụ ``detail=0`` trả chuỗi).
    """
    tokens: list[tuple[str, float]] = []
    for item in results:
        if isinstance(item, str):
            tokens.append((item, 1.0))
            continue
        try:
            _bbox, text, confidence = item
        except (ValueError, TypeError):
            continue
        try:
            conf_value = float(confidence)
        except (TypeError, ValueError):
            conf_value = 0.0
        tokens.append((str(text), conf_value))
    return tokens

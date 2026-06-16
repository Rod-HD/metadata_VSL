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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence, runtime_checkable

logger = logging.getLogger(__name__)

# Tập chữ số hợp lệ cho con số "CÁCH": 1..9 (không nhận 0). Dùng cho cả
# ``interpret_ocr`` lẫn allowlist của EasyOCR.
_VALID_DIGITS = frozenset("123456789")

# Nhận dạng nhãn "CÁCH" do EasyOCR (model English) đọc lệch trên video QIPEDC.
# Biến thể quan sát thực tế:
#   "Cach/Cacl/Cack/Cac'" — chuẩn + mất dấu.
#   "Bach" — C→B khi logo che phần đầu.
#   "P█GH 1"/"c█CH 2" (D0105 vi-reader): block char giữa token → strip → "pgh"/"cch".
#   "'SacH 1'" (D0105 en-reader): bọc trong dấu nháy + C→S → strip nháy → "sach".
#   * Nhánh "cac|sac" + [a-z']: bao Cach/Cacl/Sach/SacH...
#   * Nhánh "bach": chỉ "Bach" (KHÔNG khớp "back").
#   * Nhánh "[cpbs][a-z]{0,2}[gc]h": pgh/cch/cgh/bgh sau strip garbage chars.
# Chuẩn hóa: bỏ dấu + strip non-alpha + strip nháy đầu/cuối trước khi khớp.
_CACH_RE = re.compile(r"^(?:[cs]ac[a-z']|bach|[cpbs][a-z]{0,2}[gc]h)(?:\b|\s|\d|$)", re.IGNORECASE)

# Bảng bỏ dấu tối thiểu cho các nguyên âm có dấu của "cách" và các ký tự tiếng Việt
# thường gặp khi OCR đọc nhãn CÁCH. Chỉ map ký tự dùng trong nhận dạng.
_DIACRITIC_MAP = str.maketrans(
    "áàảãạăắằẳẵặâấầẩẫậ",  # 17 biến thể của "a"
    "a" * 17,
)

# Fragment pattern: nhận dạng mảnh nhỏ của "CÁCH" khi OCR chỉ đọc được một phần.
# Điều kiện: bắt đầu bằng "c" (hoặc "C") + ít nhất 1 ký tự liên quan (a/á/ă/h/k/l...),
# hoặc token chứa ký tự tiếng Việt đặc trưng "á"/"ă"/"â" (chỉ có trong "cách").
# Không khớp với ký tự đơn lẻ hoặc từ ngắn không liên quan.
_CACH_FRAGMENT_RE = re.compile(
    r"^c[a-z]"          # "ca", "ch", "cl", "ck"... — bắt đầu "c" + 1 ký tự
    r"|^[a-z]?[áăâ]"    # có "á"/"ă"/"â" → chỉ tiếng Việt, rất đặc trưng
    r"|^ca$"            # token đúng "ca" (phổ biến khi OCR cắt)
    r"|^ch$"            # token "ch"
    r"|^[áă]",          # token bắt đầu bằng dấu thanh (dấu rời)
    re.IGNORECASE,
)

# Token nhiễu cố định xuất hiện ở góc trên trái của MỌI video QIPEDC (kể cả single):
# logo cuốn sách và chữ "QIPEDC" luôn có mặt. OCR đọc chúng thành các token dưới đây.
# Bất kỳ token nào NGOÀI danh sách này trong wide_roi → nghi ngờ có overlay "CÁCH".
_QIPEDC_NOISE_RE = re.compile(
    r"^q[i1][p|][e3][d][c]$"   # "QIPEDC" và biến thể OCR thường gặp
    r"|^qipedc?$"
    r"|^[il1|]{1,2}$"           # nét dọc đơn/đôi từ biểu tượng sách/tay
    r"|^[o0]{1,2}$"             # đốm tròn
    r"|^[-_~]{1,3}$",           # vạch ngang từ nét kẻ
    re.IGNORECASE,
)


def _is_cach_token(text: str) -> bool:
    """True nếu *text* trông như nhãn "CÁCH" đầy đủ (kể cả biến thể OCR mất dấu/có dấu)."""
    normalized = str(text).strip().lower().translate(_DIACRITIC_MAP)
    # Strip ký tự không phải a-z/0-9/space/nháy (OCR garbage như '█', D0105: "P█GH 1")
    # rồi strip thêm dấu nháy đầu/cuối (English-only reader bọc: "'SacH 1'").
    cleaned = re.sub(r"[^a-z0-9 ']", "", normalized).strip("'")
    return bool(_CACH_RE.match(cleaned))


def _is_cach_fragment(text: str) -> bool:
    """True nếu *text* là mảnh nhỏ của chữ "CÁCH" — dùng khi OCR chỉ đọc được một phần.

    Điều kiện lỏng hơn ``_is_cach_token``: chỉ cần 1 dấu hiệu nhỏ (fragment "ca",
    "ch", dấu "á"/"ă"...). Kết hợp với điều kiện "có số trong ROI" để tránh dương
    tính giả trên video không có overlay CÁCH.
    """
    t = str(text).strip()
    if len(t) < 1:
        return False
    # Khớp đầy đủ trước (không loại trừ)
    if _is_cach_token(t):
        return True
    return bool(_CACH_FRAGMENT_RE.match(t))


@dataclass(frozen=True)
class DetectionResult:
    """Kết quả phát hiện số trên một frame (Req 3).

    Attributes:
        number: Con số "CÁCH" đọc được, ``int`` trong ``{1..9}``; hoặc ``None``
            khi không có số hợp lệ ("không có số").
        confidence: Độ tin cậy của kết quả OCR. Bằng ``0.0`` khi ``number`` là
            ``None``.
        saw_cach: ``True`` nếu OCR vùng rộng nhìn thấy nhãn "CÁCH" (kể cả biến thể
            mất dấu "Cacl"/"Cack"/"Bach") trên frame, **dù có đọc được con số hay
            không**. Dùng để phân biệt "video không có overlay CÁCH" (single thật)
            với "có overlay CÁCH nhưng số không đọc chắc" (cần rà soát thủ công —
            tránh âm thầm bỏ sót video nhiều cách). Mặc định ``False`` để các nơi
            dựng ``DetectionResult`` cũ không phải đổi.
    """

    number: int | None
    confidence: float
    saw_cach: bool = False


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
    # Strip whitespace + dấu nháy đầu/cuối (English-only reader bọc token trong nháy)
    stripped = str(text).strip().strip("'")
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

    # ROI rộng bao logo + chữ "CÁCH" + số (layout lệch trái/phải vẫn trong vùng này).
    # Chiều cao y1=0.25 đủ cover logo (0–0.18) + lệch, không ăn vào content video bên dưới.
    # Chiều ngang x1=0.40 cover được số lệch phải hơn logo nhưng không lấn sang phụ đề.
    # (0.5×0.5 cũ quá rộng → OCR bắt content video → Pass 3 false positive → 980 manual_review)
    _DEFAULT_WIDE_ROI: tuple[float, float, float, float] = (0.0, 0.0, 0.40, 0.25)

    def __init__(
        self,
        cfg=None,
        *,
        models_dir: str | os.PathLike | None = None,
        roi: tuple[float, float, float, float] | None = None,
        wide_roi: tuple[float, float, float, float] | None = None,
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
            roi: Ghi đè ROI tỉ lệ ``(x0, y0, x1, y1)`` cho bước primary.
            wide_roi: ROI rộng cho fallback ``find_number_near_cach``; mặc định
                ``_DEFAULT_WIDE_ROI`` bao trùm logo + "CÁCH" + số.
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

        # --- ROI, wide_roi và ngưỡng tin cậy ---
        if roi is not None:
            self._roi = roi
        elif cfg is not None:
            self._roi = cfg.roi_top_left
        else:
            self._roi = (0.0, 0.0, 0.22, 0.18)

        self._wide_roi = wide_roi if wide_roi is not None else self._DEFAULT_WIDE_ROI

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
        """Phát hiện con số "CÁCH" trên *frame* (Req 3.2/3.3/3.4).

        Thử hai bước:
        1. **Primary**: OCR ROI cố định nhỏ (``self._roi``) với allowlist số —
           nhanh, đủ cho phần lớn video layout chuẩn.
        2. **Fallback**: luôn gọi :func:`find_number_near_cach` trên vùng rộng để
           xử lý layout lệch (logo shift) hoặc số đè lên logo (nét mảnh). Fallback
           cũng cho biết có nhìn thấy nhãn "CÁCH" hay không (``saw_cach``) — dùng
           để đánh dấu frame "có CÁCH nhưng không đọc được số" thay vì lặng lẽ coi
           là không có overlay.
        """
        roi_img = crop_roi(frame, self._roi)
        if roi_img is None:
            self._log.debug(
                "Frame rỗng hoặc ROI diện tích 0 — trả 'không có số'."
            )
            return DetectionResult(number=None, confidence=0.0, saw_cach=False)

        reader = self._ensure_reader()
        # allowlist chỉ chữ số 1–9; không nhận diện ký tự khác.
        results = reader.readtext(roi_img, allowlist="123456789")

        tokens = _tokens_from_easyocr(results)
        primary = interpret_ocr(tokens, self._threshold)

        # Luôn chạy fallback để phát hiện layout lệch (logo shift phải khiến
        # primary ROI bắt vào chữ "C"/"H" của "CÁCH" và đọc nhầm thành số).
        fallback, saw_cach = find_number_near_cach(frame, self._wide_roi, reader)

        # Chọn kết quả: ưu tiên fallback khi hai bên bất đồng — fallback đọc
        # toàn bộ text nên biết chắc số nằm sau "CÁCH", còn primary đọc blind.
        if primary is not None and fallback is not None and primary == fallback:
            number = primary  # cả hai đồng thuận: tự tin nhất
        elif fallback is not None:
            # Fallback tìm được (dù primary có hay không): tin fallback vì nó
            # neo vào chữ "CÁCH" thay vì crop mù.
            if primary != fallback:
                self._log.debug(
                    "Primary=%s nhưng fallback=%d (neo vào 'CÁCH') — dùng fallback.",
                    primary, fallback,
                )
            number = fallback
        else:
            number = primary  # chỉ primary có kết quả (hoặc cả hai None)

        if number is None:
            return DetectionResult(
                number=None, confidence=0.0, saw_cach=saw_cach
            )

        # Lấy độ tin cậy: dùng primary confidence nếu đồng thuận, ngưỡng nếu chỉ fallback.
        if primary == number:
            confidence = next(
                (
                    float(conf)
                    for text, conf in tokens
                    if _as_single_digit(text) == number
                    and conf is not None
                    and float(conf) >= self._threshold
                ),
                self._threshold,
            )
        else:
            confidence = self._threshold
        return DetectionResult(
            number=number, confidence=confidence, saw_cach=saw_cach
        )


def find_number_near_cach(
    frame,
    wide_roi: tuple[float, float, float, float],
    reader,
) -> tuple[int | None, bool]:
    """Tìm số cách bằng cách OCR vùng rộng và định vị token số ngay sau "CÁCH".

    Thay vì crop ROI cố định, hàm này thử **ba kỹ thuật** theo thứ tự (số đè lên
    logo / nét mảnh khiến mỗi video khó theo một kiểu khác nhau):

    1. **Token gộp** "CÁCH 1" → lấy số cuối token.
    2. **Token số bên phải**: token "CÁCH" đứng riêng → tìm token số gần nhất bên
       phải nó (OCR đã tách được số thành token riêng).
    3. **Zoom + ngưỡng + aspect** (:func:`_zoom_digit_right_of`): crop vùng số
       ngay phải "CÁCH", phóng to, thử nhiều ngưỡng sáng để OCR số mảnh; nếu OCR
       vẫn lưỡng lự "1" ↔ "2" thì dùng tỉ lệ rộng/cao của glyph làm tie-break
       (số "1" hẹp, số "2" rộng).

    Đồng thời trả cờ ``saw_cach``: ``True`` nếu nhìn thấy nhãn "CÁCH" trên frame
    (dù có đọc được số hay không). Cờ này cho phép phía gọi phân biệt "không có
    overlay" với "có CÁCH nhưng số không chắc" → tránh âm thầm bỏ sót.

    Không lọc theo confidence khi tìm "CÁCH" — confidence thấp là bình thường khi
    OCR chữ có dấu tiếng Việt bằng model English; điều quan trọng là vị trí.

    Args:
        frame: Mảng numpy (H, W, 3) từ OpenCV.
        wide_roi: ROI tỉ lệ (x0, y0, x1, y1) bao trùm khu vực logo + chữ "CÁCH"
            + số phía sau nó.
        reader: ``easyocr.Reader`` đã khởi tạo.

    Returns:
        Cặp ``(number, saw_cach)``: ``number`` là ``int`` trong ``{1..9}`` nếu
        tìm được, ngược lại ``None``; ``saw_cach`` là ``True`` nếu thấy nhãn
        "CÁCH" trên frame.
    """
    wide_img = crop_roi(frame, wide_roi)
    if wide_img is None:
        return None, False

    # OCR không có allowlist để nhận cả chữ "CÁCH" lẫn số.
    results = reader.readtext(wide_img)
    if not results:
        return None, False

    # Phân tách bbox, text, confidence từ kết quả EasyOCR (detail=1).
    parsed: list[tuple[list, str, float]] = []
    for item in results:
        try:
            bbox, text, conf = item
            parsed.append((bbox, str(text).strip(), float(conf)))
        except (ValueError, TypeError):
            continue

    def _extract_trailing_digit(text: str) -> int | None:
        """Lấy chữ số cuối token nếu EasyOCR gộp 'CÁCH 1' thành một text."""
        parts = text.strip().split()
        if len(parts) >= 2:
            return _as_single_digit(parts[-1])
        return None

    # --- Pass 1: tìm token "CÁCH" đầy đủ (logic cũ) ---
    # Đồng thời thu thập token số và token fragment để dùng ở Pass 2.
    digit_tokens: list[tuple[list, int]] = []  # (bbox, digit)
    fragment_tokens: list[tuple[list, str]] = []  # (bbox, text) — fragment cách

    saw_cach = False
    for bbox, text, conf in parsed:
        digit = _as_single_digit(text)
        if digit is not None:
            digit_tokens.append((bbox, digit))
        elif _is_cach_fragment(text) and not _is_cach_token(text):
            fragment_tokens.append((bbox, text))

        if not _is_cach_token(text):
            continue
        saw_cach = True

        # Trường hợp 1: token gộp "CÁCH 1" → lấy số cuối ngay.
        digit = _extract_trailing_digit(text)
        if digit is not None:
            return digit, True

        # Trường hợp 2: token "CÁCH" đứng riêng → tìm số token ngay bên phải.
        try:
            xs = [pt[0] for pt in bbox]
            cach_x_center = sum(xs) / len(xs)
            cach_width = max(xs) - min(xs)
        except (TypeError, IndexError):
            continue

        best_num: int | None = None
        best_dist = float("inf")
        for bbox2, text2, conf2 in parsed:
            digit2 = _as_single_digit(text2)
            if digit2 is None:
                continue
            try:
                num_xs = [pt[0] for pt in bbox2]
                num_x_center = sum(num_xs) / len(num_xs)
            except (TypeError, IndexError):
                continue
            dx = num_x_center - cach_x_center
            if 0 < dx < 2.5 * max(cach_width, 10) and dx < best_dist:
                best_dist = dx
                best_num = digit2
        if best_num is not None:
            return best_num, True

        # Trường hợp 3: số đè lên logo + nét mảnh nên KHÔNG được OCR tách thành
        # token ở độ phân giải gốc (vd "con chim"/"con khỉ"/"con sông": số "1"/"2"
        # nằm ngay trên cuốn sách). Zoom + đa ngưỡng + aspect tie-break.
        zoom_digit = _zoom_digit_right_of(wide_img, bbox, reader)
        if zoom_digit is not None:
            return zoom_digit, True

    # --- Pass 2: fragment + số — khi OCR không đọc được token "CÁCH" đầy đủ
    # nhưng thấy số trong ROI và ít nhất 1 fragment nhỏ của "CÁCH" gần đó.
    # Ý tưởng: số ở góc trên trái QIPEDC + dấu hiệu nhỏ của "cách" → 99% multi.
    if not saw_cach and digit_tokens and fragment_tokens:
        # Tìm cặp (fragment, số) gần nhau nhất trong wide_roi.
        img_w = wide_img.shape[1] if wide_img is not None else 1
        for frag_bbox, frag_text in fragment_tokens:
            try:
                fxs = [pt[0] for pt in frag_bbox]
                frag_x_center = sum(fxs) / len(fxs)
                frag_width = max(fxs) - min(fxs)
            except (TypeError, IndexError):
                continue
            for num_bbox, num_digit in digit_tokens:
                try:
                    nxs = [pt[0] for pt in num_bbox]
                    num_x_center = sum(nxs) / len(nxs)
                except (TypeError, IndexError):
                    continue
                dx = num_x_center - frag_x_center
                # Số phải nằm bên phải fragment (dx > 0) và không quá xa (< 3× rộng fragment).
                if 0 < dx < 3.0 * max(frag_width, img_w * 0.05):
                    saw_cach = True
                    return num_digit, True

    return None, saw_cach


def _glyph_width_ratio(gray, cv2) -> "float | None":
    """Tỉ lệ chiều rộng glyph so với tổng chiều rộng crop.

    Dùng khi glyph số bị logo che phần dưới nên contour không đủ để tính aspect.
    "1" hẹp (~0.34), "2" rộng hơn (~0.39) — ngưỡng split=0.365 hiệu chỉnh trên W00738.
    Trả None nếu không tìm thấy vùng trắng đủ rõ.
    """
    import numpy as np
    _, bw = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    col_sums = bw.sum(axis=0)
    active = np.where(col_sums > gray.shape[0] * 0.1)[0]
    if len(active) < 2:
        return None
    glyph_w = int(active[-1]) - int(active[0]) + 1
    return glyph_w / gray.shape[1]


def _zoom_digit_right_of(wide_img, cach_bbox, reader) -> int | None:
    """Vớt con số mảnh nằm ngay bên phải bbox "CÁCH" bằng zoom + ngưỡng + aspect.

    Dùng khi số bị đè lên logo / nét quá mảnh nên OCR ở độ phân giải gốc bỏ sót.
    Quy trình (hiệu chỉnh thực nghiệm trên "con chim"/"con khỉ"/"con sông"):

    1. Crop dải ngang **ngang hàng chữ "CÁCH"** ngay bên phải nó (không nới xuống
       dưới để loại biểu tượng hai bàn tay vàng nằm thấp hơn), phóng to 6×.
    2. OCR bản xám + vài ngưỡng nhị phân sáng (số màu trắng tách khỏi nền xanh),
       **vote** chữ số theo tổng confidence.
    3. Nếu cuộc vote lưỡng lự giữa "1" và "2" (cả hai cùng xuất hiện, hoặc số mảnh
       bị đọc nhầm), dùng **tỉ lệ rộng/cao** (aspect) của glyph nửa-trên làm
       tie-break: glyph "1" hẹp (aspect nhỏ), glyph "2" rộng hơn.

    Trả chữ số ``1..9`` nếu quyết được; ``None`` nếu không.

    Args:
        wide_img: Ảnh ``wide_roi`` đã crop (numpy ``(h, w, 3)``) — cùng hệ tọa độ
            với ``cach_bbox``.
        cach_bbox: bbox của token "CÁCH" (4 điểm ``[x, y]``) trong ``wide_img``.
        reader: ``easyocr.Reader`` đã khởi tạo.
    """
    try:
        import cv2  # lazy: chỉ cần khi thực sự zoom
    except ImportError:  # pragma: no cover - cv2 luôn có trong môi trường chạy
        return None
    import collections

    try:
        xs = [pt[0] for pt in cach_bbox]
        ys = [pt[1] for pt in cach_bbox]
    except (TypeError, IndexError):
        return None

    h, w = wide_img.shape[:2]
    x_right = int(max(xs))
    y_top = int(min(ys))
    y_bot = int(max(ys))
    cw = int(max(xs) - min(xs))
    ch = int(y_bot - y_top)
    if cw <= 0 or ch <= 0:
        return None

    # Dải A: bám sát bbox CÁCH (số nằm cạnh, không bị logo che).
    nx0 = _clamp(x_right - int(0.10 * cw), 0, w)
    nx1 = _clamp(x_right + int(1.7 * cw), 0, w)
    ny0_a = _clamp(y_top - int(0.10 * ch), 0, h)
    ny1_a = _clamp(y_bot + int(0.10 * ch), 0, h)
    if nx1 <= nx0 or ny1_a <= ny0_a:
        return None

    num_crop = wide_img[ny0_a:ny1_a, nx0:nx1]
    if num_crop.size == 0:
        return None

    big = cv2.resize(num_crop, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    # --- OCR vote trên dải A ---
    votes: "collections.Counter[int]" = collections.Counter()
    images = [gray]
    for thv in (160, 175, 190, 205):
        images.append(cv2.threshold(gray, thv, 255, cv2.THRESH_BINARY)[1])
    for img in images:
        for item in reader.readtext(img, allowlist="123456789"):
            try:
                _bbox, text, conf = item
            except (ValueError, TypeError):
                continue
            digit = _as_single_digit(text)
            if digit is not None and conf is not None and float(conf) >= 0.45:
                votes[digit] += float(conf)

    # Dải B: phần trên bbox CÁCH — số nhô lên trên logo (bị che phần dưới).
    # Dùng glyph_width_ratio: "1" hẹp (~0.33), "2" rộng (~0.37).
    # Ngưỡng 0.355 hiệu chỉnh thực nghiệm trên W00738.
    ny0_b = _clamp(y_top - int(0.8 * ch), 0, h)
    ny1_b = _clamp(y_top + int(0.5 * ch), 0, h)
    gwr = None
    if ny1_b > ny0_b and nx1 > nx0:
        crop_b = wide_img[ny0_b:ny1_b, nx0:nx1]
        if crop_b.size > 0:
            big_b = cv2.resize(crop_b, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_CUBIC)
            gray_b = cv2.cvtColor(big_b, cv2.COLOR_BGR2GRAY)
            gwr = _glyph_width_ratio(gray_b, cv2)

    if votes:
        relevant = {d: s for d, s in votes.items() if d in (1, 2)}
        if relevant:
            if 1 in relevant and 2 in relevant:
                hi = max(relevant[1], relevant[2])
                loval = min(relevant[1], relevant[2])
                if hi < 1.5 * loval:
                    aspect = _digit_aspect_ratio(gray, cv2)
                    if aspect is not None:
                        return 2 if aspect >= _ASPECT_SPLIT else 1
                    # OCR lưỡng lự, aspect không tính được → dùng glyph_width_ratio
                    if gwr is not None:
                        return 1 if gwr < 0.355 else 2
            # Chỉ có 1 hoặc chỉ có 2: tin OCR nhưng cross-check với gwr nếu có.
            ocr_result = max(relevant, key=relevant.get)
            # Nếu gwr mâu thuẫn rõ ràng với OCR → tin gwr (logo che làm OCR sai).
            if gwr is not None:
                gwr_says = 1 if gwr < 0.355 else 2
                if gwr_says != ocr_result:
                    # Mâu thuẫn: dùng gwr vì OCR dải A hay bị logo ảnh hưởng.
                    return gwr_says
            return ocr_result

    # OCR dải A thất bại hoàn toàn: thử aspect ratio.
    aspect = _digit_aspect_ratio(gray, cv2)
    if aspect is not None:
        if aspect <= _ASPECT_ONE_MAX:
            return 1
        if aspect >= _ASPECT_TWO_MIN:
            return 2

    # Fallback cuối: glyph_width_ratio dải B.
    if gwr is not None:
        return 1 if gwr < 0.355 else 2
    return None


# Ngưỡng aspect (rộng/cao của glyph số) phân biệt "1" với "2" — hiệu chỉnh trên
# video QIPEDC: "1" hẹp (~0.33–0.56), "2" rộng (~0.71–0.83). Vùng đệm ở giữa.
_ASPECT_ONE_MAX = 0.62
_ASPECT_TWO_MIN = 0.70
_ASPECT_SPLIT = 0.66  # điểm chia khi buộc phải chọn 1 hay 2


def _digit_aspect_ratio(gray, cv2) -> float | None:
    """Tỉ lệ rộng/cao của glyph số (contour lớn nhất ở nửa TRÊN ảnh xám đã zoom).

    Số nằm ngang hàng chữ "CÁCH" (nửa trên); biểu tượng bàn tay (nếu lọt vào) nằm
    nửa dưới nên bị loại bằng điều kiện tâm contour ở nửa trên. Trả ``None`` nếu
    không tìm được contour số đáng tin.
    """
    th = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    height = th.shape[0]
    best = None
    best_area = 0.0
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        # Loại contour nằm chủ yếu ở nửa dưới (bàn tay) và đốm quá nhỏ.
        if y + bh / 2 > height * 0.55:
            continue
        if bh < height * 0.20:
            continue
        area = cv2.contourArea(c)
        if area > best_area:
            best_area = area
            best = (bw, bh)
    if best is None or best[1] <= 0:
        return None
    return best[0] / best[1]


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

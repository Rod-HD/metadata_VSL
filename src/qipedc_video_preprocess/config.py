"""Cấu_Hình — cấu hình đường dẫn và tham số cho công đoạn tách video.

Module này định nghĩa :class:`PreprocessConfig` (dataclass bất biến) cùng
``resolve()`` và ``validate()``.

``PreprocessConfig`` là nguồn sự thật duy nhất cho mọi đường dẫn và tham số có
thể tinh chỉnh của công đoạn tách video nhiều cách. Mọi đường dẫn được biểu diễn
*tương đối với thư mục gốc dự án* và được phân giải thành đường dẫn tuyệt đối
trên ổ ``D:`` qua :meth:`PreprocessConfig.resolve`, bảo đảm không có artifact nào
nằm ngoài cây dự án hoặc trên ổ ``C:``.

Module này **không** import hay tham chiếu ``qipedc2vsl400`` để giữ công đoạn cô
lập hoàn toàn (Requirement 1.1).

Xem design.md — Requirements 1.3, 1.4, 3.6, 5.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Ổ đĩa bắt buộc cho mọi đầu ra (Req 1.3): thư mục gốc dự án và mọi đường dẫn đầu
# ra phải nằm trên ổ ``D:``.
REQUIRED_DRIVE = "D:"


@dataclass(frozen=True)
class PreprocessConfig:
    """Cấu hình bất biến cho công đoạn tách video nhiều cách.

    ``project_root`` là trường bắt buộc duy nhất và **phải** nằm trên ổ ``D:``;
    mọi trường còn lại có giá trị mặc định hợp lý nên một
    ``PreprocessConfig(project_root=Path("D:/.../metadata_VSL"))`` là dùng được
    ngay.
    """

    project_root: Path

    # --- nguồn video (Req 2.1) — thứ tự = thứ tự ưu tiên ---
    video_search_dirs: tuple[str, ...] = (
        "Dataset/processed_videos/resize_720p",
        "Dataset/raw_videos",
    )

    # --- bảng nhãn nguồn (Req 7) ---
    labels_glob: str = "Dataset/labels/*.xlsx"

    # --- đầu ra (Req 1.2, 5) ---
    split_output_dir: str = "Dataset/processed_videos/split_variants"
    new_labels_path: str = "Dataset/processed_videos/split_variants/labels_split.xlsx"
    log_dir: str = "Dataset/logs"
    # Thư mục gom video cần rà soát thủ công (segmenter không phân loại được,
    # xung đột tên, hoặc clip < 1 frame). File gốc được COPY vào đây để xem lại.
    manual_review_dir: str = "Dataset/processed_videos/manual_review"
    # Thư mục gom video được tách bằng SUY LUẬN (có nhãn "CÁCH" + điểm chuyển rõ
    # nhưng OCR không đọc trọn dãy số). Video đã được tách tự động NHƯNG cần người
    # kiểm lại ranh giới. File gốc được COPY vào đây để xem lại.
    inferred_review_dir: str = "Dataset/processed_videos/inferred_review"

    # --- OCR / ROI (Req 3.1, 3.6) ---
    # ROI góc trên trái theo tỉ lệ (x0, y0, x1, y1), mỗi giá trị trong [0.0, 1.0].
    # Giá trị này được HIỆU CHỈNH THỰC NGHIỆM trên video QIPEDC 1280×720: nó cô lập
    # đúng CON SỐ "CÁCH" (1/2/3) nằm ngay sau logo, bỏ qua logo QIPEDC (icon sách +
    # 2 bàn tay) và chữ "H" của "CÁCH". ROI rộng hơn (vd ôm cả logo) khiến EasyOCR
    # đọc nhiễu thành số 2 chữ số ("41"/"92") rồi bị loại → phân loại nhầm là một cách.
    roi_top_left: tuple[float, float, float, float] = (0.19, 0.05, 0.25, 0.22)
    ocr_confidence_threshold: float = 0.5  # Ngưỡng_Tin_Cậy (0.0–1.0)
    ocr_models_dir: str = "Dataset/models/easyocr"

    # --- phân đoạn (Req 4.1, 4.2) ---
    sample_interval_seconds: float = 1.0  # 0.1–5.0
    boundary_confirm_frames: int = 2  # số frame mẫu liên tiếp xác nhận số mới

    # --- cắt (Req 5.5) ---
    safety_margin_frames: int = 3  # số nguyên 0–60

    # ------------------------------------------------------------------ #
    # Phân giải đường dẫn
    # ------------------------------------------------------------------ #
    def resolve(self, relative_path: str) -> Path:
        """Phân giải đường dẫn tương đối thành đường dẫn tuyệt đối dưới
        ``project_root``.

        Nếu ``relative_path`` đã là tuyệt đối thì trả về nguyên (đã ``resolve``).
        Ngược lại nó được join lên :attr:`project_root` để kết quả nằm cùng ổ đĩa
        với dự án (``D:``). Mirror quy ước của
        ``qipedc2vsl400.config.Config.resolve``.
        """
        candidate = Path(relative_path)
        if candidate.is_absolute():
            return candidate.resolve()
        return (Path(self.project_root) / candidate).resolve()

    # --- convenience accessors (đường dẫn tuyệt đối trên D:) ---

    @property
    def split_output_path(self) -> Path:
        return self.resolve(self.split_output_dir)

    @property
    def new_labels_full_path(self) -> Path:
        return self.resolve(self.new_labels_path)

    @property
    def log_path(self) -> Path:
        return self.resolve(self.log_dir)

    @property
    def manual_review_path(self) -> Path:
        return self.resolve(self.manual_review_dir)

    @property
    def inferred_review_path(self) -> Path:
        return self.resolve(self.inferred_review_dir)

    @property
    def ocr_models_path(self) -> Path:
        return self.resolve(self.ocr_models_dir)

    def video_search_paths(self) -> tuple[Path, ...]:
        """Đường dẫn tuyệt đối của mọi thư mục tìm video đã cấu hình."""
        return tuple(self.resolve(d) for d in self.video_search_dirs)

    # ------------------------------------------------------------------ #
    # Kiểm tra hợp lệ (Req 1.3, 1.4, 3.6, 5.5)
    # ------------------------------------------------------------------ #
    def validate(self) -> list[str]:
        """Trả về danh sách lỗi cấu hình (rỗng nếu hợp lệ).

        Kiểm tra:

        * Mọi đường dẫn đầu ra (``split_output_dir``, ``new_labels_path``,
          ``log_dir``) resolve về một vị trí **nằm trong** cây ``project_root``
          **và** trên ổ đĩa ``D:`` (Req 1.3/1.4).
        * ``roi_top_left``: mỗi tọa độ ∈ [0, 1] với ``x0 < x1`` và ``y0 < y1``
          (Req 3.6).
        * ``ocr_confidence_threshold`` ∈ [0, 1] (Req 3.6).
        * ``sample_interval_seconds`` ∈ [0.1, 5.0] (Req 4.1).
        * ``safety_margin_frames`` là số nguyên ∈ [0, 60] (Req 5.5).

        Mỗi thông báo lỗi chỉ rõ giá trị/đường dẫn vi phạm.
        """
        errors: list[str] = []

        # --- project_root phải nằm trên ổ D: (Req 1.3) ---
        try:
            root = Path(self.project_root).resolve()
        except (OSError, ValueError) as exc:  # pragma: no cover - phòng thủ
            errors.append(
                f"project_root không phân giải được: {self.project_root!r} ({exc})"
            )
            root = None

        if root is not None and not _is_on_required_drive(root):
            errors.append(
                "project_root phải nằm trên ổ "
                f"'{REQUIRED_DRIVE}' nhưng lại là '{root}' "
                f"(ổ đĩa '{root.drive or '(không có)'}')."
            )

        # --- mọi đường dẫn đầu ra: trên D: VÀ trong cây project_root (Req 1.4) ---
        output_fields = (
            ("split_output_dir", self.split_output_dir),
            ("new_labels_path", self.new_labels_path),
            ("log_dir", self.log_dir),
            ("manual_review_dir", self.manual_review_dir),
            ("inferred_review_dir", self.inferred_review_dir),
        )
        for field_name, raw_value in output_fields:
            errors.extend(self._validate_output_path(field_name, raw_value, root))

        # --- ROI góc trên trái (Req 3.6) ---
        errors.extend(self._validate_roi())

        # --- ngưỡng tin cậy OCR (Req 3.6) ---
        conf = self.ocr_confidence_threshold
        if not _is_real_number(conf) or not (0.0 <= float(conf) <= 1.0):
            errors.append(
                "ocr_confidence_threshold phải nằm trong khoảng [0.0, 1.0] "
                f"nhưng nhận được {conf!r}."
            )

        # --- khoảng lấy mẫu frame (Req 4.1) ---
        interval = self.sample_interval_seconds
        if not _is_real_number(interval) or not (0.1 <= float(interval) <= 5.0):
            errors.append(
                "sample_interval_seconds phải nằm trong khoảng [0.1, 5.0] giây "
                f"nhưng nhận được {interval!r}."
            )

        # --- biên an toàn (Req 5.5) ---
        margin = self.safety_margin_frames
        if not _is_integer(margin):
            errors.append(
                "safety_margin_frames phải là số nguyên trong khoảng [0, 60] "
                f"nhưng nhận được {margin!r} (không phải số nguyên)."
            )
        elif not (0 <= int(margin) <= 60):
            errors.append(
                "safety_margin_frames phải là số nguyên trong khoảng [0, 60] "
                f"nhưng nhận được {margin!r}."
            )

        return errors

    # ------------------------------------------------------------------ #
    # Helpers nội bộ
    # ------------------------------------------------------------------ #
    def _validate_output_path(
        self, field_name: str, raw_value: str, root: Path | None
    ) -> list[str]:
        """Kiểm tra một đường dẫn đầu ra: trên ổ ``D:`` và trong cây dự án."""
        errors: list[str] = []
        try:
            resolved = self.resolve(raw_value)
        except (OSError, ValueError) as exc:  # pragma: no cover - phòng thủ
            errors.append(
                f"{field_name}={raw_value!r} không phân giải được ({exc})."
            )
            return errors

        if not _is_on_required_drive(resolved):
            errors.append(
                f"{field_name}={raw_value!r} resolve tới '{resolved}' nằm trên ổ "
                f"'{resolved.drive or '(không có)'}' thay vì ổ bắt buộc "
                f"'{REQUIRED_DRIVE}'."
            )

        if root is not None and not _is_within(resolved, root):
            errors.append(
                f"{field_name}={raw_value!r} resolve tới '{resolved}' nằm NGOÀI "
                f"cây thư mục gốc dự án '{root}'."
            )

        return errors

    def _validate_roi(self) -> list[str]:
        """Kiểm tra ROI góc trên trái (Req 3.6)."""
        errors: list[str] = []
        roi = self.roi_top_left

        if not isinstance(roi, (tuple, list)) or len(roi) != 4:
            errors.append(
                "roi_top_left phải là bộ 4 giá trị (x0, y0, x1, y1) "
                f"nhưng nhận được {roi!r}."
            )
            return errors

        x0, y0, x1, y1 = roi
        labels = (("x0", x0), ("y0", y0), ("x1", x1), ("y1", y1))
        for name, value in labels:
            if not _is_real_number(value) or not (0.0 <= float(value) <= 1.0):
                errors.append(
                    f"roi_top_left.{name} phải nằm trong khoảng [0.0, 1.0] "
                    f"nhưng nhận được {value!r}."
                )

        # Chỉ kiểm tra thứ tự khi tất cả là số hợp lệ để tránh lỗi so sánh.
        if all(_is_real_number(v) for _, v in labels):
            if not float(x0) < float(x1):
                errors.append(
                    "roi_top_left yêu cầu x0 < x1 nhưng "
                    f"x0={x0!r}, x1={x1!r}."
                )
            if not float(y0) < float(y1):
                errors.append(
                    "roi_top_left yêu cầu y0 < y1 nhưng "
                    f"y0={y0!r}, y1={y1!r}."
                )

        return errors


# ---------------------------------------------------------------------- #
# Helpers cấp module
# ---------------------------------------------------------------------- #
def _is_on_required_drive(path: Path) -> bool:
    """True nếu ``path`` nằm trên ổ đĩa bắt buộc (``D:``), so sánh không phân
    biệt hoa/thường."""
    return path.drive.upper() == REQUIRED_DRIVE.upper()


def _is_within(path: Path, root: Path) -> bool:
    """True nếu ``path`` nằm bên trong cây thư mục ``root`` (hoặc chính ``root``)."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_real_number(value: object) -> bool:
    """True nếu ``value`` là số thực (int/float) không phải bool/NaN."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    # loại NaN (NaN != NaN)
    return value == value


def _is_integer(value: object) -> bool:
    """True nếu ``value`` là số nguyên thực (int, không phải bool)."""
    return isinstance(value, int) and not isinstance(value, bool)

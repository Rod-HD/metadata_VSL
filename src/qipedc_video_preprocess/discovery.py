"""Bộ_Duyệt_Video — liệt kê, khử trùng và sắp xếp video nguồn.

Module này định nghĩa :class:`VideoEntry`, hàm logic thuần :func:`select_entries`,
hàm điều phối I/O :func:`discover_videos` và kiểm tra đọc được :func:`is_readable`.

Tách bạch trách nhiệm để dễ kiểm thử:

* :func:`select_entries` là **logic thuần** — nhận một mapping
  ``source_dir -> [filenames]`` (thứ tự khóa = thứ tự ưu tiên) và thực hiện lọc
  ``.mp4`` (không phân biệt hoa/thường), khử trùng ``video_id`` theo ưu tiên, rồi
  sắp xếp tăng dần theo mã ký tự. Không chạm filesystem nên property-test được mà
  không cần video thật.
* :func:`discover_videos` là **lớp biên** — liệt kê file thật theo thứ tự ưu tiên,
  cảnh báo khi thư mục thiếu, loại file không đọc được, rồi gọi
  :func:`select_entries` để khử trùng + sắp xếp.
* :func:`is_readable` mở video bằng :class:`cv2.VideoCapture` và xác nhận đọc được
  ít nhất một frame.

Xem design.md — Requirement 2 (2.1–2.6).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoEntry:
    """Một video nguồn đã được chọn để xử lý.

    Attributes:
        video_id: Tên file đã bỏ phần mở rộng (stem), ví dụ ``"W00202"``.
        path: Đường dẫn tới file được chọn (bản thuộc thư mục ưu tiên cao nhất).
        source_dir: Thư mục nguồn (đã cấu hình) chứa file được chọn.
    """

    video_id: str
    path: Path
    source_dir: str


def _is_mp4(filename: str) -> bool:
    """True nếu *filename* có phần mở rộng ``.mp4`` (không phân biệt hoa/thường).

    Dùng :class:`pathlib.PurePath` nên file ẩn không có đuôi như ``".mp4"`` (stem
    rỗng) sẽ **không** được coi là video hợp lệ.
    """
    return Path(filename).suffix.lower() == ".mp4"


def select_entries(
    listing_by_dir: Mapping[str, Iterable[str]],
) -> list[VideoEntry]:
    """Lọc, khử trùng và sắp xếp các video từ một mapping thư mục → tên file.

    Đây là **logic thuần** (không chạm filesystem):

    * Chỉ giữ file có đuôi ``.mp4`` không phân biệt hoa/thường (Req 2.1).
    * ``video_id`` (stem) chỉ xuất hiện **một lần**, lấy từ thư mục có thứ tự ưu
      tiên cao nhất chứa nó; thứ tự ưu tiên chính là thứ tự khóa của
      *listing_by_dir* (Req 2.2).
    * Kết quả được sắp xếp tăng dần theo ``video_id`` theo mã ký tự nên mọi lần
      chạy trên cùng input cho cùng thứ tự (Req 2.5).

    Args:
        listing_by_dir: Mapping có thứ tự ``source_dir -> iterable tên file``.
            Khóa đứng trước có ưu tiên cao hơn.

    Returns:
        Danh sách :class:`VideoEntry` đã khử trùng và sắp xếp. ``path`` được dựng
        bằng ``Path(source_dir) / filename`` (tương đối theo khóa mapping).
    """
    selected: dict[str, VideoEntry] = {}
    for source_dir, filenames in listing_by_dir.items():
        for filename in filenames:
            if not _is_mp4(filename):
                continue
            video_id = Path(filename).stem
            if video_id in selected:
                # Thư mục ưu tiên cao hơn đã có video_id này -> bỏ qua bản trùng.
                continue
            selected[video_id] = VideoEntry(
                video_id=video_id,
                path=Path(source_dir) / filename,
                source_dir=source_dir,
            )

    return sorted(selected.values(), key=lambda entry: entry.video_id)


def is_readable(path: Path) -> bool:
    """True nếu video tại *path* mở được và đọc được ít nhất một frame.

    Hiện thực Req 2.4: file không mở/đọc được sẽ bị loại khỏi danh sách xử lý.
    Mở bằng :class:`cv2.VideoCapture`; trả về ``False`` nếu không mở được, hoặc
    đọc frame đầu thất bại / frame rỗng.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        grabbed, frame = capture.read()
        return bool(grabbed) and frame is not None
    except cv2.error:
        return False
    finally:
        capture.release()


def discover_videos(
    cfg: "object", logger: logging.Logger
) -> list[VideoEntry]:
    """Duyệt các thư mục nguồn và trả về danh sách video cần xử lý.

    Trình tự (Req 2):

    1. Với mỗi thư mục trong ``cfg.video_search_dirs`` theo thứ tự ưu tiên, liệt
       kê các file ``.mp4`` nằm **trực tiếp** trong thư mục (không đệ quy, không
       phân biệt hoa/thường) (Req 2.1).
    2. Thư mục nguồn không tồn tại → ghi cảnh báo và tiếp tục thư mục khác
       (Req 2.3).
    3. File không mở/đọc được → loại khỏi danh sách, ghi log ``video_id`` kèm lý
       do, rồi tiếp tục (Req 2.4).
    4. Khử trùng ``video_id`` theo ưu tiên và sắp xếp tăng dần theo mã ký tự qua
       :func:`select_entries` (Req 2.2, 2.5).
    5. Không tìm thấy ``.mp4`` nào → ghi cảnh báo (Req 2.6); caller (Bộ_Tiền_Xử_Lý)
       chịu trách nhiệm kết thúc lần chạy mà không tạo đầu ra.

    Args:
        cfg: ``PreprocessConfig`` cung cấp ``video_search_dirs``,
            ``video_search_paths()`` và ``resolve()``.
        logger: Logger của lần chạy để ghi cảnh báo/thông tin.

    Returns:
        Danh sách :class:`VideoEntry` (đường dẫn tuyệt đối) đã khử trùng + sắp xếp.
    """
    listing_by_dir: dict[str, list[str]] = {}

    for source_dir, abs_dir in zip(cfg.video_search_dirs, cfg.video_search_paths()):
        if not abs_dir.is_dir():
            logger.warning(
                "Thư mục nguồn không tồn tại, bỏ qua: %s (cấu hình: %r)",
                abs_dir,
                source_dir,
            )
            continue

        readable_names: list[str] = []
        # Duyệt theo tên đã sắp xếp để I/O xác định (tái lập được).
        for child in sorted(abs_dir.iterdir(), key=lambda p: p.name):
            if not child.is_file():
                continue
            if not _is_mp4(child.name):
                continue
            if not is_readable(child):
                logger.warning(
                    "Loại video không đọc được: video_id=%s, path=%s "
                    "(lý do: không mở/đọc được frame nào)",
                    child.stem,
                    child,
                )
                continue
            readable_names.append(child.name)

        listing_by_dir[source_dir] = readable_names

    relative_entries = select_entries(listing_by_dir)

    # Phân giải đường dẫn tuyệt đối trên ổ D: trong cây dự án.
    entries = [
        VideoEntry(
            video_id=entry.video_id,
            path=cfg.resolve(str(entry.path)),
            source_dir=entry.source_dir,
        )
        for entry in relative_entries
    ]

    if not entries:
        logger.warning(
            "Không tìm thấy file .mp4 nào trong các thư mục nguồn đã cấu hình; "
            "không có video nào để xử lý."
        )
    else:
        logger.info("Đã duyệt %d video để xử lý.", len(entries))

    return entries

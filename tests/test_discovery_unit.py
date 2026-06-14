"""Unit tests for ``qipedc_video_preprocess.discovery`` — I/O edge cases.

Phủ các nhánh I/O biên của Bộ_Duyệt_Video (Requirement 2):

* **Req 2.3** — một thư mục nguồn được cấu hình không tồn tại → ghi một cảnh báo
  và tiếp tục với các thư mục còn lại.
* **Req 2.4** — một file video không mở/đọc được → loại khỏi danh sách xử lý,
  ghi log ``video_id`` kèm lý do, và tiếp tục duyệt các video còn lại.
* **Req 2.6** — không tìm thấy ``.mp4`` nào trong tất cả thư mục nguồn → ghi một
  cảnh báo cho biết không có video nào để xử lý và trả về danh sách rỗng.

Các test dùng ``tmp_path`` để tạo thư mục/file thật và một logger thật gắn handler
bắt bản ghi (``CapturingHandler``). Việc phân biệt file đọc được / không đọc được
được điều khiển bằng cách monkeypatch :func:`discovery.is_readable` (theo gợi ý ở
task), nên test không phụ thuộc vào việc encode video thật. Một test bổ sung kiểm
:func:`discovery.is_readable` trực tiếp trên file rác / file không tồn tại (không
mock) để chắc chắn hàm trả về ``False``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from qipedc_video_preprocess import discovery
from qipedc_video_preprocess.config import PreprocessConfig

# project_root giả lập nằm trên ổ D: (Req 1.3). Không bao giờ chạm đĩa vì các test
# ghi đè ``video_search_paths()`` để trỏ vào thư mục tmp thật.
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

    # --- helpers truy vấn ---
    def messages(self) -> list[str]:
        """Danh sách thông điệp đã định dạng (bao gồm args)."""
        return [r.getMessage() for r in self.records]

    def messages_at(self, level: int) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno == level]


@pytest.fixture
def logger_and_handler() -> tuple[logging.Logger, CapturingHandler]:
    """Logger thật, cô lập, gắn :class:`CapturingHandler`."""
    handler = CapturingHandler()
    logger = logging.getLogger(f"test_discovery_unit.{id(handler)}")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, handler


def make_cfg(
    search_dirs: tuple[str, ...], search_paths: tuple[Path, ...]
) -> PreprocessConfig:
    """Dựng ``PreprocessConfig`` thật (project_root trên D:) nhưng trỏ
    ``video_search_paths()`` vào các thư mục tmp thật.

    ``video_search_dirs`` giữ vai trò nhãn nguồn (khóa ưu tiên) còn
    ``video_search_paths()`` trả về thư mục tuyệt đối tmp. Ghi đè method bằng
    ``object.__setattr__`` vì :class:`PreprocessConfig` là dataclass ``frozen``;
    đây chính là cách "monkeypatch video_search_dirs/paths để trỏ vào tmp dirs".
    """
    cfg = PreprocessConfig(project_root=PROJECT_ROOT, video_search_dirs=search_dirs)
    object.__setattr__(cfg, "video_search_paths", lambda: tuple(search_paths))
    return cfg


def _touch(dir_path: Path, name: str) -> Path:
    """Tạo một file rỗng trong ``dir_path`` và trả về đường dẫn."""
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / name
    file_path.write_bytes(b"")
    return file_path


# --------------------------------------------------------------------------- #
# Req 2.3 — thư mục nguồn thiếu → cảnh báo và tiếp tục
# --------------------------------------------------------------------------- #
def test_missing_source_dir_warns_and_continues(
    tmp_path, monkeypatch, logger_and_handler
):
    """Một thư mục nguồn không tồn tại được cảnh báo; thư mục còn lại vẫn duyệt."""
    logger, handler = logger_and_handler

    existing_dir = tmp_path / "resize_720p"
    missing_dir = tmp_path / "raw_videos"  # cố tình KHÔNG tạo
    _touch(existing_dir, "W00001.mp4")

    # Mọi file đều coi là đọc được để cô lập đúng nhánh "thư mục thiếu".
    monkeypatch.setattr(discovery, "is_readable", lambda path: True)

    cfg = make_cfg(
        search_dirs=("resize_720p", "raw_videos"),
        search_paths=(existing_dir, missing_dir),
    )

    entries = discovery.discover_videos(cfg, logger)

    # Video trong thư mục tồn tại vẫn được duyệt.
    assert [e.video_id for e in entries] == ["W00001"]

    # Có đúng một cảnh báo cho thư mục thiếu, nhắc tới đường dẫn vi phạm.
    warnings = handler.messages_at(logging.WARNING)
    missing_warnings = [m for m in warnings if "không tồn tại" in m]
    assert len(missing_warnings) == 1
    assert str(missing_dir) in missing_warnings[0]


def test_all_source_dirs_missing_warns_each_and_returns_empty(
    tmp_path, monkeypatch, logger_and_handler
):
    """Khi mọi thư mục nguồn đều thiếu: cảnh báo từng thư mục và trả về rỗng."""
    logger, handler = logger_and_handler

    missing_a = tmp_path / "resize_720p"
    missing_b = tmp_path / "raw_videos"

    monkeypatch.setattr(discovery, "is_readable", lambda path: True)

    cfg = make_cfg(
        search_dirs=("resize_720p", "raw_videos"),
        search_paths=(missing_a, missing_b),
    )

    entries = discovery.discover_videos(cfg, logger)

    assert entries == []
    missing_warnings = [
        m for m in handler.messages_at(logging.WARNING) if "không tồn tại" in m
    ]
    assert len(missing_warnings) == 2


# --------------------------------------------------------------------------- #
# Req 2.4 — file hỏng/không đọc được → loại + log video_id và lý do
# --------------------------------------------------------------------------- #
def test_unreadable_file_excluded_and_logged(
    tmp_path, monkeypatch, logger_and_handler
):
    """File không đọc được bị loại khỏi kết quả và được ghi log kèm video_id."""
    logger, handler = logger_and_handler

    source_dir = tmp_path / "resize_720p"
    _touch(source_dir, "W00001.mp4")  # đọc được
    _touch(source_dir, "W00002.mp4")  # hỏng / không đọc được

    def fake_is_readable(path: Path) -> bool:
        return Path(path).stem != "W00002"

    monkeypatch.setattr(discovery, "is_readable", fake_is_readable)

    cfg = make_cfg(search_dirs=("resize_720p",), search_paths=(source_dir,))

    entries = discovery.discover_videos(cfg, logger)

    # Chỉ video đọc được còn lại; bản hỏng bị loại.
    assert [e.video_id for e in entries] == ["W00001"]

    # Có cảnh báo cho file hỏng nhắc tới video_id và lý do không đọc được.
    bad_warnings = [
        m for m in handler.messages_at(logging.WARNING) if "W00002" in m
    ]
    assert len(bad_warnings) == 1
    assert "không đọc được" in bad_warnings[0]


def test_all_files_unreadable_returns_empty_and_warns_no_videos(
    tmp_path, monkeypatch, logger_and_handler
):
    """Mọi file đều hỏng → kết quả rỗng và cảnh báo 'không có video' (giao 2.4+2.6)."""
    logger, handler = logger_and_handler

    source_dir = tmp_path / "resize_720p"
    _touch(source_dir, "W00001.mp4")
    _touch(source_dir, "W00002.mp4")

    monkeypatch.setattr(discovery, "is_readable", lambda path: False)

    cfg = make_cfg(search_dirs=("resize_720p",), search_paths=(source_dir,))

    entries = discovery.discover_videos(cfg, logger)

    assert entries == []
    # Mỗi file hỏng được log riêng.
    assert sum(1 for m in handler.messages() if "không đọc được" in m) == 2
    # Và vì không còn video nào, có cảnh báo không có video để xử lý (Req 2.6).
    assert any(
        "không có video nào để xử lý" in m
        for m in handler.messages_at(logging.WARNING)
    )


# --------------------------------------------------------------------------- #
# Req 2.6 — không có .mp4 nào → cảnh báo và trả về rỗng
# --------------------------------------------------------------------------- #
def test_no_mp4_files_anywhere_warns_and_returns_empty(
    tmp_path, monkeypatch, logger_and_handler
):
    """Thư mục tồn tại nhưng không có .mp4 → cảnh báo không có video & rỗng."""
    logger, handler = logger_and_handler

    dir_a = tmp_path / "resize_720p"
    dir_b = tmp_path / "raw_videos"
    # Có file nhưng không phải .mp4 (kể cả file dễ gây nhầm).
    _touch(dir_a, "notes.txt")
    _touch(dir_a, "clip.mov")
    _touch(dir_b, "readme.md")

    monkeypatch.setattr(discovery, "is_readable", lambda path: True)

    cfg = make_cfg(
        search_dirs=("resize_720p", "raw_videos"),
        search_paths=(dir_a, dir_b),
    )

    entries = discovery.discover_videos(cfg, logger)

    assert entries == []
    no_video_warnings = [
        m
        for m in handler.messages_at(logging.WARNING)
        if "không có video nào để xử lý" in m
    ]
    assert len(no_video_warnings) == 1


def test_empty_source_dirs_warns_and_returns_empty(
    tmp_path, monkeypatch, logger_and_handler
):
    """Thư mục nguồn tồn tại nhưng hoàn toàn rỗng → cảnh báo & trả về rỗng."""
    logger, handler = logger_and_handler

    dir_a = tmp_path / "resize_720p"
    dir_a.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(discovery, "is_readable", lambda path: True)

    cfg = make_cfg(search_dirs=("resize_720p",), search_paths=(dir_a,))

    entries = discovery.discover_videos(cfg, logger)

    assert entries == []
    assert any(
        "không có video nào để xử lý" in m
        for m in handler.messages_at(logging.WARNING)
    )


# --------------------------------------------------------------------------- #
# is_readable trực tiếp (không mock) — củng cố Req 2.4
# --------------------------------------------------------------------------- #
def test_is_readable_false_for_corrupt_file(tmp_path):
    """File .mp4 chứa byte rác không mở/đọc được → is_readable trả False."""
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"not a real video stream \x00\x01\x02\x03" * 8)
    assert discovery.is_readable(bad) is False


def test_is_readable_false_for_missing_file(tmp_path):
    """File không tồn tại → is_readable trả False."""
    missing = tmp_path / "does_not_exist.mp4"
    assert discovery.is_readable(missing) is False

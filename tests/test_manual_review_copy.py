"""Tests cho gom video manual_review vào folder riêng (``_copy_manual_review``).

Phủ:

* Copy đúng file gốc của từng ``video_id`` manual_review vào
  ``cfg.manual_review_path`` (giữ tên ``<video_id>.mp4``).
* File **được copy** (không move): file gốc vẫn còn.
* ``video_id`` không nằm trong tập đã duyệt → bỏ qua + log cảnh báo.
* Lỗi copy một file → log + tiếp tục (không hỏng cả lần chạy).
* ``cfg.manual_review_dir`` được ``validate()`` ràng buộc trên ổ ``D:`` + trong
  cây dự án (giống các đường dẫn đầu ra khác).
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from qipedc_video_preprocess.config import PreprocessConfig
from qipedc_video_preprocess.preprocess import _copy_manual_review


@dataclass
class _Entry:
    """Stand-in tối thiểu cho VideoEntry (chỉ cần ``path``)."""

    path: Path


class _CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]


def _logger() -> tuple[logging.Logger, _CapturingHandler]:
    handler = _CapturingHandler()
    logger = logging.getLogger(f"test_manual_review.{id(handler)}")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, handler


def _make_cfg(root: Path) -> PreprocessConfig:
    return PreprocessConfig(project_root=root)


def test_copies_originals_into_manual_review_dir():
    """Mỗi video_id manual_review → file <id>.mp4 trong manual_review_path; gốc còn."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = _make_cfg(root)

        src_dir = root / "src_videos"
        src_dir.mkdir()
        discovered = {}
        for vid in ("D0001", "D0002"):
            p = src_dir / f"{vid}.mp4"
            p.write_bytes(b"fake-video-" + vid.encode())
            discovered[vid] = _Entry(path=p)

        logger, _ = _logger()
        copied = _copy_manual_review({"D0001", "D0002"}, discovered, cfg, logger)

        assert copied == 2
        for vid in ("D0001", "D0002"):
            dest = cfg.manual_review_path / f"{vid}.mp4"
            assert dest.is_file(), f"{vid} chưa được copy"
            # nội dung khớp
            assert dest.read_bytes() == (b"fake-video-" + vid.encode())
            # file gốc vẫn còn (copy, không move)
            assert discovered[vid].path.is_file()


def test_skips_unknown_video_id_with_warning():
    """video_id không có trong discovered → bỏ qua + log cảnh báo, copied giảm."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = _make_cfg(root)

        src_dir = root / "src_videos"
        src_dir.mkdir()
        p = src_dir / "D0001.mp4"
        p.write_bytes(b"x")
        discovered = {"D0001": _Entry(path=p)}

        logger, handler = _logger()
        copied = _copy_manual_review({"D0001", "GHOST"}, discovered, cfg, logger)

        assert copied == 1
        assert (cfg.manual_review_path / "D0001.mp4").is_file()
        assert not (cfg.manual_review_path / "GHOST.mp4").exists()
        assert any("GHOST" in m for m in handler.messages())


def test_copy_error_is_logged_and_skipped(monkeypatch):
    """Lỗi copy một file → log lỗi + tiếp tục (copied đếm đúng phần thành công)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = _make_cfg(root)

        src_dir = root / "src_videos"
        src_dir.mkdir()
        good = src_dir / "GOOD.mp4"
        good.write_bytes(b"ok")
        bad = src_dir / "BAD.mp4"
        bad.write_bytes(b"bad")
        discovered = {"GOOD": _Entry(path=good), "BAD": _Entry(path=bad)}

        import shutil as _shutil

        real_copy = _shutil.copy2

        def flaky_copy(srcp, dstp, *a, **k):
            if "BAD" in str(srcp):
                raise OSError("disk full (giả lập)")
            return real_copy(srcp, dstp, *a, **k)

        monkeypatch.setattr(_shutil, "copy2", flaky_copy)

        logger, handler = _logger()
        copied = _copy_manual_review({"GOOD", "BAD"}, discovered, cfg, logger)

        assert copied == 1
        assert (cfg.manual_review_path / "GOOD.mp4").is_file()
        assert not (cfg.manual_review_path / "BAD.mp4").exists()
        assert any("BAD" in m or "lỗi copy" in m for m in handler.messages())


def test_manual_review_dir_validated_on_d_drive():
    """manual_review_dir nằm trong cây dự án trên D: → validate() không báo lỗi nó."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PreprocessConfig(project_root=Path(tmp))
        errors = cfg.validate()
        # Không có lỗi nào nhắc tới manual_review_dir.
        assert not any("manual_review_dir" in e for e in errors), errors


def test_manual_review_dir_outside_tree_flagged():
    """manual_review_dir trỏ ra ngoài cây dự án → validate() báo lỗi."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PreprocessConfig(
            project_root=Path(tmp),
            manual_review_dir="D:/somewhere_else/manual_review",
        )
        errors = cfg.validate()
        assert any("manual_review_dir" in e for e in errors), errors

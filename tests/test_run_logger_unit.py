"""Unit tests cho ``qipedc_video_preprocess.run_logger`` — Bộ_Ghi_Log (Req 8).

Phủ các nhánh I/O của bộ ghi log:

* **Req 8.5** — :func:`get_run_logger` tạo thư mục ``Dataset/logs/`` nếu chưa tồn
  tại trước khi ghi file log.
* **Req 8.1** — file log có tên với dấu thời gian đến giây và **duy nhất**, không
  ghi đè log của lần chạy trước (kể cả khi hai lần chạy rơi vào cùng một giây).
* **Req 8.6** — nếu mở/ghi file log thất bại, lỗi được **để lan ra** (raise) nên
  lần chạy dừng, bảo đảm không thông tin nào bị mất âm thầm.

Dùng ``tmp_path`` của pytest làm ``project_root``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from qipedc_video_preprocess.config import PreprocessConfig
from qipedc_video_preprocess.run_logger import get_run_logger


_LOG_NAME_RE = re.compile(r"^preprocess_\d{8}_\d{6}(?:_\d+)?\.log$")


def _cfg(tmp_path: Path) -> PreprocessConfig:
    return PreprocessConfig(project_root=tmp_path, log_dir="Dataset/logs")


def _close(logger: logging.Logger) -> Path:
    """Đóng handler và trả về đường dẫn file log đã ghi."""
    handler = logger.handlers[0]
    path = Path(handler.baseFilename)
    handler.close()
    logger.removeHandler(handler)
    return path


# --------------------------------------------------------------------------- #
# Req 8.5 — tạo thư mục logs/ nếu chưa tồn tại
# --------------------------------------------------------------------------- #
def test_get_run_logger_creates_log_dir(tmp_path):
    cfg = _cfg(tmp_path)
    log_dir = tmp_path / "Dataset" / "logs"
    assert not log_dir.exists()

    logger = get_run_logger(cfg)
    try:
        # Thư mục log đã được tạo (Req 8.5).
        assert log_dir.is_dir()
        # Ghi một dòng và xác nhận file tồn tại.
        logger.info("hello run")
    finally:
        path = _close(logger)

    assert path.exists()
    assert path.parent == log_dir


# --------------------------------------------------------------------------- #
# Req 8.1 — tên file log có timestamp và duy nhất, không ghi đè
# --------------------------------------------------------------------------- #
def test_get_run_logger_filename_has_timestamp(tmp_path):
    cfg = _cfg(tmp_path)
    logger = get_run_logger(cfg)
    path = _close(logger)
    assert _LOG_NAME_RE.match(path.name), path.name


def test_get_run_logger_unique_even_same_second(tmp_path, monkeypatch):
    """Hai lần chạy trong cùng một giây vẫn cho hai file log khác tên (Req 8.1)."""
    cfg = _cfg(tmp_path)

    # Ép datetime.now() trả về cùng một thời điểm cho mọi lần gọi → cùng timestamp,
    # buộc get_run_logger phải thêm hậu tố để giữ tên duy nhất.
    import qipedc_video_preprocess.run_logger as run_logger_mod

    class _FixedDateTime:
        @classmethod
        def now(cls):
            import datetime as _dt

            return _dt.datetime(2026, 6, 14, 12, 0, 0)

    monkeypatch.setattr(run_logger_mod, "datetime", _FixedDateTime)

    logger1 = get_run_logger(cfg)
    path1 = _close(logger1)

    logger2 = get_run_logger(cfg)
    path2 = _close(logger2)

    # Cùng timestamp nhưng tên file khác nhau và không ghi đè nhau.
    assert path1 != path2
    assert path1.exists()
    assert path2.exists()
    assert _LOG_NAME_RE.match(path1.name)
    assert _LOG_NAME_RE.match(path2.name)


# --------------------------------------------------------------------------- #
# Req 8.6 — lỗi mở file log → raise (dừng), không nuốt
# --------------------------------------------------------------------------- #
def test_get_run_logger_file_open_failure_raises(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)

    import qipedc_video_preprocess.run_logger as run_logger_mod

    def _boom(*args, **kwargs):
        raise OSError("không mở được file log")

    monkeypatch.setattr(run_logger_mod.logging, "FileHandler", _boom)

    with pytest.raises(OSError, match="không mở được file log"):
        get_run_logger(cfg)

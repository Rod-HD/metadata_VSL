"""Unit tests for ``qipedc2vsl400.qipedc_reader`` (Task 3.3).

These tests exercise the QIPEDC `.xlsx` reader and its validation helper using
fixtures generated programmatically with :mod:`openpyxl`, so they never depend
on the real ``Dataset/labels`` spreadsheets. Each test builds a temporary
``Dataset/labels`` tree under ``tmp_path`` and points a :class:`Config` at it.

Covers Requirements 3.1 (case-insensitive header load), 3.2 (multi-file combine
and duplicate VIDEO detection), 3.3 (skip-and-log rows missing VIDEO/LABEL),
3.4 (count logging) and 4.2 (float-label normalization, Vietnamese preserved).
"""

from __future__ import annotations

from pathlib import Path

import openpyxl

from qipedc2vsl400.config import Config
from qipedc2vsl400.qipedc_reader import (
    QipedcRow,
    normalize_label,
    read_rows,
    validate_rows,
)


# --- fixture helpers --------------------------------------------------------


def _write_xlsx(path: Path, header: list, rows: list[list]) -> None:
    """Create an ``.xlsx`` file at *path* with *header* row then *rows*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append(header)
    for row in rows:
        worksheet.append(row)
    workbook.save(path)
    workbook.close()


def _make_config(tmp_path: Path) -> Config:
    """A Config whose label glob points at ``tmp_path/Dataset/labels``."""
    return Config(project_root=tmp_path)


_CANONICAL_HEADER = ["STT", "ID", "VIDEO", "LABEL", "REGION", "TOPIC", "signer"]


# --- normalize_label (Req 4.2) ----------------------------------------------


def test_normalize_label_float_string_to_int():
    assert normalize_label("1.0") == "1"
    assert normalize_label("12.0") == "12"


def test_normalize_label_float_value_to_int():
    assert normalize_label(1.0) == "1"
    assert normalize_label(5) == "5"


def test_normalize_label_preserves_vietnamese_text():
    assert normalize_label("Chào") == "Chào"
    assert normalize_label("Số một") == "Số một"


def test_normalize_label_preserves_real_decimals():
    assert normalize_label("1.5") == "1.5"


def test_normalize_label_blank_to_none():
    assert normalize_label(None) is None
    assert normalize_label("   ") is None


# --- header case-insensitivity (Req 3.1) ------------------------------------


def test_header_case_insensitive_mapping(tmp_path):
    cfg = _make_config(tmp_path)
    labels = tmp_path / "Dataset" / "labels"
    # Mixed-case / spaced headers should still map to the canonical fields.
    header = ["stt", "Id", " VIDEO ", "Label", "region", "Topic", "SIGNER"]
    _write_xlsx(
        labels / "batch_1.xlsx",
        header,
        [[1, "A1", "D0530.mp4", "Chào", "Chung", "Số", None]],
    )

    rows = read_rows(cfg)

    assert len(rows) == 1
    row = rows[0]
    assert row.stt == "1"
    assert row.id == "A1"
    assert row.video == "D0530.mp4"
    assert row.label == "Chào"
    assert row.region == "Chung"
    assert row.topic == "Số"
    assert row.signer is None


# --- multi-file combine (Req 3.2) -------------------------------------------


def test_multi_file_combine(tmp_path):
    cfg = _make_config(tmp_path)
    labels = tmp_path / "Dataset" / "labels"
    _write_xlsx(
        labels / "batch_1.xlsx",
        _CANONICAL_HEADER,
        [[1, "A1", "D0530.mp4", "Một", "Chung", "Số", None]],
    )
    _write_xlsx(
        labels / "batch_1(1).xlsx",
        _CANONICAL_HEADER,
        [[2, "A2", "D0531.mp4", "Hai", "Nam", "Số", None]],
    )

    rows = read_rows(cfg)

    assert len(rows) == 2
    videos = {r.video for r in rows}
    assert videos == {"D0530.mp4", "D0531.mp4"}


# --- duplicate VIDEO detection (Req 3.2) ------------------------------------


def test_duplicate_video_detection(tmp_path):
    cfg = _make_config(tmp_path)
    labels = tmp_path / "Dataset" / "labels"
    _write_xlsx(
        labels / "batch_1.xlsx",
        _CANONICAL_HEADER,
        [[1, "A1", "D0530.mp4", "Một", "Chung", "Số", None]],
    )
    _write_xlsx(
        labels / "batch_1(1).xlsx",
        _CANONICAL_HEADER,
        [[2, "A2", "D0530.mp4", "Một", "Nam", "Số", None]],
    )

    rows = read_rows(cfg)
    valid, skipped = validate_rows(cfg, rows)

    # Both rows have video + label so both are valid; duplication is reported in
    # the log (no exception, run continues).
    assert len(valid) == 2
    assert skipped == []
    # Confirm both share the duplicate VIDEO value.
    assert [r.video for r in valid] == ["D0530.mp4", "D0530.mp4"]


# --- skip-and-log for missing required fields (Req 3.3) ---------------------


def test_skip_rows_missing_video_or_label(tmp_path):
    cfg = _make_config(tmp_path)
    labels = tmp_path / "Dataset" / "labels"
    _write_xlsx(
        labels / "batch_1.xlsx",
        _CANONICAL_HEADER,
        [
            [1, "A1", "D0530.mp4", "Một", "Chung", "Số", None],   # valid
            [2, "A2", None, "Hai", "Nam", "Số", None],            # missing VIDEO
            [3, "A3", "D0532.mp4", None, "Chung", "Số", None],    # missing LABEL
            [4, "A4", "   ", "   ", "Chung", "Số", None],         # both blank
        ],
    )

    rows = read_rows(cfg)
    valid, skipped = validate_rows(cfg, rows)

    assert len(valid) == 1
    assert valid[0].video == "D0530.mp4"
    assert len(skipped) == 3

    # A timestamped log file recording the run was written under Dataset/logs.
    log_files = list((tmp_path / "Dataset" / "logs").glob("*.log"))
    assert log_files, "expected a log file under Dataset/logs"
    log_text = log_files[0].read_text(encoding="utf-8")
    assert "skipped=3" in log_text
    assert "valid=1" in log_text


# --- float-label normalization end-to-end (Req 4.2) -------------------------


def test_float_label_normalization_via_reader(tmp_path):
    cfg = _make_config(tmp_path)
    labels = tmp_path / "Dataset" / "labels"
    _write_xlsx(
        labels / "batch_1.xlsx",
        _CANONICAL_HEADER,
        [
            [1, "A1", "D0001.mp4", 1.0, "Chung", "Số", None],       # float number
            [2, "A2", "D0002.mp4", "2.0", "Chung", "Số", None],     # float string
            [3, "A3", "D0003.mp4", "Chào bạn", "Chung", "Chữ", None],  # Vietnamese
        ],
    )

    rows = read_rows(cfg)
    by_video = {r.video: r.label for r in rows}

    assert by_video["D0001.mp4"] == "1"
    assert by_video["D0002.mp4"] == "2"
    assert by_video["D0003.mp4"] == "Chào bạn"


# --- empty glob (no files) --------------------------------------------------


def test_read_rows_no_files(tmp_path):
    cfg = _make_config(tmp_path)
    (tmp_path / "Dataset" / "labels").mkdir(parents=True, exist_ok=True)
    assert read_rows(cfg) == []

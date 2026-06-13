"""Read and parse the QIPEDC ``*.xlsx`` label files (Requirement 3).

This module loads every spreadsheet matching ``cfg.qipedc_labels_glob``
(``Dataset/labels/*.xlsx``) with :mod:`openpyxl`, matches the header row
case-insensitively to the known QIPEDC columns
(``STT, ID, VIDEO, LABEL, REGION, TOPIC, signer``) and returns a flat
``list[QipedcRow]``.

Subtask 3.1 covers :func:`read_rows` and the :func:`normalize_label` helper that
canonicalises Excel float-formatted number glosses (e.g. ``"1.0"`` -> ``"1"``)
while leaving real words / Vietnamese text untouched. ``validate_rows`` (3.2)
and the test suite (3.3) are implemented separately.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openpyxl

# Known QIPEDC header columns, matched case-insensitively.
_FIELDS: tuple[str, ...] = ("stt", "id", "video", "label", "region", "topic", "signer")

# A string that is purely a (signed) integer or decimal number, e.g. "1", "1.0",
# "12.50", "-3.0". Used to decide whether a label is a stray Excel float string
# that should be canonicalised rather than a real word.
_FLOAT_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")


@dataclass
class QipedcRow:
    """One parsed QIPEDC source row.

    Mirrors the design's ``QipedcRow`` model. All value fields are optional
    because a spreadsheet cell may be blank; ``source_file`` and ``row_index``
    record provenance for logging and duplicate reporting.
    """

    stt: str | None
    id: str | None
    video: str | None  # e.g. "D0530.mp4"
    label: str | None  # gloss (Vietnamese); float-number labels normalized
    region: str | None
    topic: str | None
    signer: str | None
    source_file: str
    row_index: int


def _num_to_str(value: int | float) -> str:
    """Render a number as its canonical string.

    Integer-valued floats lose the trailing ``.0`` (``1.0`` -> ``"1"``); genuine
    decimals are preserved (``1.5`` -> ``"1.5"``).
    """
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value)


def _coerce_cell(value: Any) -> str | None:
    """Convert an arbitrary openpyxl cell value to a clean string or ``None``.

    Numeric cells are normalised through :func:`_num_to_str` so integral floats
    (common for ``STT``/``ID`` columns) do not carry a spurious ``.0``. Blank or
    whitespace-only strings become ``None``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return _num_to_str(value)
    text = str(value).strip()
    return text or None


def normalize_label(value: Any) -> str | None:
    """Canonicalise a QIPEDC ``LABEL`` value.

    Excel stores number signs as floats, so a gloss like the number *one* may
    arrive as the float ``1.0`` or the string ``"1.0"``. This helper converts
    such integer-valued numbers to their canonical integer string (``"1"``)
    while leaving real words and Vietnamese text (with diacritics) intact.

    Args:
        value: Raw cell value (``str``, ``int``, ``float`` or ``None``).

    Returns:
        The normalised label string, or ``None`` for empty/blank input.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return _num_to_str(value)
    text = str(value).strip()
    if not text:
        return None
    if _FLOAT_RE.match(text):
        # A bare numeric string from Excel; canonicalise integral values.
        return _num_to_str(float(text))
    return text


def _label_files(cfg) -> list[str]:
    """Return the sorted list of ``*.xlsx`` files matching the configured glob."""
    pattern = str(Path(cfg.project_root) / cfg.qipedc_labels_glob)
    return sorted(glob.glob(pattern))


def _build_header_map(header_row: tuple[Any, ...] | None) -> dict[str, int]:
    """Map each known QIPEDC field name to its column index in the header row.

    Matching is case-insensitive and whitespace-tolerant.
    """
    mapping: dict[str, int] = {}
    if not header_row:
        return mapping
    for idx, cell in enumerate(header_row):
        if cell is None:
            continue
        name = str(cell).strip().lower()
        if name in _FIELDS and name not in mapping:
            mapping[name] = idx
    return mapping


def _is_blank_row(row: tuple[Any, ...] | None) -> bool:
    if not row:
        return True
    return all(
        cell is None or (isinstance(cell, str) and not cell.strip()) for cell in row
    )


def _build_row(
    row: tuple[Any, ...],
    header_map: dict[str, int],
    source_file: str,
    row_index: int,
) -> QipedcRow:
    def get(field: str) -> Any:
        idx = header_map.get(field)
        if idx is None or idx >= len(row):
            return None
        return row[idx]

    return QipedcRow(
        stt=_coerce_cell(get("stt")),
        id=_coerce_cell(get("id")),
        video=_coerce_cell(get("video")),
        label=normalize_label(get("label")),
        region=_coerce_cell(get("region")),
        topic=_coerce_cell(get("topic")),
        signer=_coerce_cell(get("signer")),
        source_file=source_file,
        row_index=row_index,
    )


def _read_one_file(path: str) -> list[QipedcRow]:
    """Parse a single ``.xlsx`` file into ``QipedcRow`` objects."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook.active
        rows: list[QipedcRow] = []
        header_map: dict[str, int] | None = None
        for sheet_row_index, row in enumerate(
            worksheet.iter_rows(values_only=True), start=1
        ):
            if header_map is None:
                # The first (non-empty) row is the header.
                if _is_blank_row(row):
                    continue
                header_map = _build_header_map(row)
                continue
            if _is_blank_row(row):
                continue
            rows.append(_build_row(row, header_map, path, sheet_row_index))
        return rows
    finally:
        workbook.close()


def read_rows(cfg) -> list[QipedcRow]:
    """Read every QIPEDC label spreadsheet and return combined rows.

    Loads all files matching ``cfg.qipedc_labels_glob`` (sorted for
    determinism), matching each file's header row case-insensitively to the
    columns ``STT, ID, VIDEO, LABEL, REGION, TOPIC, signer``. Float-formatted
    number labels are normalised via :func:`normalize_label`.

    Args:
        cfg: A :class:`~qipedc2vsl400.config.Config` instance.

    Returns:
        A flat ``list[QipedcRow]`` across all matched files.
    """
    rows: list[QipedcRow] = []
    for path in _label_files(cfg):
        rows.extend(_read_one_file(path))
    return rows


import logging
from collections import Counter
from datetime import datetime

# Module logger; handlers are attached on demand by ``_get_file_logger`` so that
# importing this module never creates files as a side effect.
_LOGGER_NAME = "qipedc2vsl400.qipedc_reader"


def _get_file_logger(cfg) -> logging.Logger:
    """Return a logger that writes to a timestamped file under ``cfg.log_dir``.

    The log directory (``Dataset/logs/`` by default) is created if needed. A
    fresh :class:`~logging.FileHandler` is attached per call so each validation
    run is captured, and propagation is disabled to avoid duplicate console
    output from the root logger.
    """
    log_dir = cfg.log_path
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"qipedc_reader_{timestamp}.log"

    logger = logging.getLogger(f"{_LOGGER_NAME}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    return logger


def _find_duplicate_videos(rows: list[QipedcRow]) -> dict[str, int]:
    """Return a mapping of ``VIDEO`` value -> occurrence count for duplicates.

    Only rows with a non-empty ``video`` are considered. The returned dict only
    contains values that appear more than once.
    """
    counts = Counter(
        row.video for row in rows if row.video is not None and row.video != ""
    )
    return {video: count for video, count in counts.items() if count > 1}


def validate_rows(cfg, rows: list[QipedcRow]) -> tuple[list[QipedcRow], list[QipedcRow]]:
    """Split parsed rows into ``(valid, skipped)`` and log a run summary.

    A row is **valid** only when it has both a non-empty ``video`` and a
    non-empty ``label`` (Requirement 3.3); every other row is collected into
    ``skipped`` and logged, never aborting the run. Duplicate ``VIDEO`` values
    are detected and reported (Requirement 3.2). Counts (total read, valid,
    skipped, duplicates) are written to a log file under ``cfg.log_dir``
    (Requirement 3.4).

    Args:
        cfg: A :class:`~qipedc2vsl400.config.Config` instance (used for
            ``log_dir`` resolution).
        rows: The combined rows returned by :func:`read_rows`.

    Returns:
        A tuple ``(valid, skipped)`` of ``QipedcRow`` lists.
    """
    valid: list[QipedcRow] = []
    skipped: list[QipedcRow] = []

    logger = _get_file_logger(cfg)

    for row in rows:
        has_video = bool(row.video and row.video.strip())
        has_label = bool(row.label and row.label.strip())
        if has_video and has_label:
            valid.append(row)
        else:
            skipped.append(row)
            missing = []
            if not has_video:
                missing.append("VIDEO")
            if not has_label:
                missing.append("LABEL")
            logger.warning(
                "Skipped row %d from %s: missing required field(s) %s",
                row.row_index,
                row.source_file,
                ", ".join(missing),
            )

    duplicates = _find_duplicate_videos(rows)
    for video, count in sorted(duplicates.items()):
        logger.warning("Duplicate VIDEO value %r appears %d times", video, count)

    logger.info(
        "QIPEDC validation summary: total=%d valid=%d skipped=%d duplicates=%d",
        len(rows),
        len(valid),
        len(skipped),
        len(duplicates),
    )

    # Flush and detach the file handler so the log file is fully written and the
    # handler does not leak across repeated calls.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    return valid, skipped

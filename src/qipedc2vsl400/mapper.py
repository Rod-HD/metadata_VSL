"""Map validated QIPEDC rows to the VSL400-referenced superset schema (Req 4).

This module turns each validated :class:`~qipedc2vsl400.qipedc_reader.QipedcRow`
into one :class:`OutputRecord` — a **superset** metadata object that keeps the
QIPEDC-native fields (``video_id`` from ``VIDEO`` unchanged, ``gloss`` renamed
from the already-normalized ``LABEL``, plus ``region``/``topic``/``id``)
and adds the VSL400-derived fields (``signer_id`` from the signer-extraction
step, and ``fps``/``resolution``/``num_frames``/``length_seconds`` measured from
the real video clip).

:func:`build_records` is dependency-injected: video properties come from a
``probe_fn`` (``probe(video_path) -> VideoProps | None`` from
:mod:`~qipedc2vsl400.video_probe`) and signer ids from a list of
:class:`~qipedc2vsl400.signer_extractor.SignerAssignment`. A clip whose video
cannot be found or read (``probe_fn`` returns ``None``) is skipped-and-logged
into the returned ``skipped_due_to_video`` list, per the documented default
``cfg.on_missing_video == "skip"`` policy (Requirement 4.4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Module logger; a file handler is attached on demand by ``_get_file_logger`` so
# that importing this module never creates files as a side effect.
_LOGGER_NAME = "qipedc2vsl400.mapper"


@dataclass
class OutputRecord:
    """One output metadata object (superset; VSL400 is a reference only).

    Field order matches the design's mapping table: the QIPEDC-native fields
    come first, then the VSL400-derived additions.

    Attributes:
        video_id: QIPEDC ``VIDEO`` kept unchanged (stem ``D0530`` by default,
            or with extension when ``cfg.video_id_keep_extension`` is true).
        gloss: QIPEDC ``LABEL`` (already normalized at read time), UTF-8
            Vietnamese text preserved.
        region: QIPEDC ``REGION`` copied as-is (may be ``None``).
        topic: QIPEDC ``TOPIC`` copied as-is (may be ``None``).
        id: QIPEDC ``ID`` copied as-is (may be ``None``).
        signer_id: From the matching ``SignerAssignment`` (3-digit or the
            ``unknown`` bucket label).
        fps: Frames per second, from ``VideoProps``.
        resolution: Vertical pixel height, from ``VideoProps``.
        num_frames: Total frame count, from ``VideoProps``.
        length_seconds: Clip duration in seconds, from ``VideoProps``.
    """

    # kept from QIPEDC
    video_id: str
    gloss: str
    region: str | None
    topic: str | None
    id: str | None
    # added (VSL400-derived)
    signer_id: str
    fps: float
    resolution: int
    num_frames: int
    length_seconds: float


def _get_file_logger(cfg: Any) -> logging.Logger:
    """Return a logger writing to a timestamped file under ``cfg.log_dir``.

    Mirrors the reader/extractor logging convention: the log directory is
    created if needed, a fresh :class:`~logging.FileHandler` is attached per
    call, and propagation is disabled to avoid duplicate console output.
    """
    log_dir = cfg.log_path
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"mapper_{timestamp}.log"

    logger = logging.getLogger(f"{_LOGGER_NAME}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _build_video_index(cfg: Any) -> dict[str, Path]:
    """Map each video *filename* to its absolute path under the search dirs.

    Search directories (``cfg.video_search_paths()``) are scanned in order; the
    first occurrence of a given filename wins, so the preferred directory
    (``processed_videos/resize_720p``) takes precedence over ``raw_videos``.
    Within a directory, paths are visited in sorted order for determinism. The
    scan is recursive because ``raw_videos`` nests each clip in its own
    subfolder.
    """
    index: dict[str, Path] = {}
    for base in cfg.video_search_paths():
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.mp4")):
            if path.is_file() and path.name not in index:
                index[path.name] = path
    return index


def _build_signer_lookup(signer_assignments: list[Any]) -> dict[str, str]:
    """Map each ``VIDEO`` filename to its assigned ``signer_id``.

    The first assignment seen for a given ``video`` wins (assignments are
    expected to be unique per clip).
    """
    lookup: dict[str, str] = {}
    for assignment in signer_assignments:
        video = getattr(assignment, "video", None)
        if video and video not in lookup:
            lookup[video] = assignment.signer_id
    return lookup


def _video_id_from(video: str, cfg: Any) -> str:
    """Derive ``video_id`` from a QIPEDC ``VIDEO`` filename.

    Returns the filename stem (extension stripped, e.g. ``D0530``) unless
    ``cfg.video_id_keep_extension`` is true, in which case the full filename is
    kept.
    """
    if getattr(cfg, "video_id_keep_extension", False):
        return video
    return Path(video).stem


def build_records(
    valid_rows: list[Any],
    probe_fn: Callable[[Path], Any],
    signer_assignments: list[Any],
    cfg: Any,
) -> tuple[list[OutputRecord], list[Any]]:
    """Build superset :class:`OutputRecord` objects from validated QIPEDC rows.

    For each row in *valid_rows*:

    * ``video_id`` = QIPEDC ``VIDEO`` stem (or kept with extension when
      ``cfg.video_id_keep_extension``);
    * ``gloss`` = the already-normalized ``LABEL`` (kept as-is, UTF-8 preserved);
    * ``region``/``topic``/``id`` are copied from the row as-is;
    * ``signer_id`` is looked up from *signer_assignments* by ``VIDEO`` filename
      (falling back to ``cfg.signer_unknown_label`` if absent);
    * ``fps``/``resolution``/``num_frames``/``length_seconds`` come from
      ``probe_fn(video_path)`` after resolving the clip under
      ``cfg.video_search_paths()``.

    When the video cannot be found or ``probe_fn`` returns ``None`` (unreadable /
    missing clip), the row is skipped-and-logged into ``skipped_due_to_video``
    per the default ``cfg.on_missing_video == "skip"`` policy (Requirement 4.4).

    Args:
        valid_rows: Validated ``QipedcRow`` objects (each has a non-empty
            ``video`` and ``label``).
        probe_fn: ``probe(video_path: Path) -> VideoProps | None`` used to read
            real clip properties; injectable for tests.
        signer_assignments: ``SignerAssignment`` objects providing ``signer_id``
            per ``VIDEO`` filename.
        cfg: A :class:`~qipedc2vsl400.config.Config`-like object.

    Returns:
        A tuple ``(records, skipped_due_to_video)`` where ``records`` is the list
        of built :class:`OutputRecord` objects (row order preserved) and
        ``skipped_due_to_video`` holds the rows whose video was missing or
        unreadable.
    """
    logger = _get_file_logger(cfg)
    try:
        index = _build_video_index(cfg)
        signer_lookup = _build_signer_lookup(signer_assignments)
        unknown_label = getattr(cfg, "signer_unknown_label", "unknown")

        records: list[OutputRecord] = []
        skipped_due_to_video: list[Any] = []

        for row in valid_rows:
            video = row.video or ""
            video_path = index.get(video)

            if video_path is None:
                logger.warning(
                    "No video file found for VIDEO %r (row %s of %s); skipping",
                    video,
                    getattr(row, "row_index", "?"),
                    getattr(row, "source_file", "?"),
                )
                skipped_due_to_video.append(row)
                continue

            props = probe_fn(video_path)
            if props is None:
                logger.warning(
                    "Unreadable video %s for VIDEO %r; skipping per on_missing_video=%r",
                    video_path,
                    video,
                    getattr(cfg, "on_missing_video", "skip"),
                )
                skipped_due_to_video.append(row)
                continue

            signer_id = signer_lookup.get(video, unknown_label)

            records.append(
                OutputRecord(
                    video_id=_video_id_from(video, cfg),
                    gloss=row.label,
                    region=row.region,
                    topic=row.topic,
                    id=row.id,
                    signer_id=signer_id,
                    fps=props.fps,
                    resolution=props.resolution,
                    num_frames=props.num_frames,
                    length_seconds=props.length_seconds,
                )
            )

        logger.info(
            "Mapper summary: valid_rows=%d records=%d skipped_due_to_video=%d",
            len(valid_rows),
            len(records),
            len(skipped_due_to_video),
        )

        return records, skipped_due_to_video
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

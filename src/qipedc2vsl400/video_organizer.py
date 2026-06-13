"""Organize QIPEDC clips into per-signer folders for manual review (Req 9).

After signer extraction (:mod:`~qipedc2vsl400.signer_extractor`) and mapping
(:mod:`~qipedc2vsl400.mapper`), :func:`organize_by_signer` materializes a
``Dataset/by_signer/`` tree on ``D:`` so the project owner can eyeball each
signer's clips and confirm the clustering is correct.

Layout produced::

    Dataset/by_signer/
    ├── signer_001/   signer_002/   …   signer_unknown/
    │       <original VIDEO filenames, e.g. D0530.mp4>
    └── _summary.txt          # signer -> clip-count, including unknown

Design guarantees (Requirement 9, Properties 10 & 11):

* The folder a clip lands in is named ``signer_<signer_id>`` where ``signer_id``
  is **exactly** the value emitted in the metadata for that clip — both come
  from the same ``signer_assignments`` (Property 11, AC5). For the unknown
  bucket the id is ``cfg.signer_unknown_label`` (``"unknown"``), giving the
  ``signer_unknown`` folder.
* Each clip keeps its **original ``VIDEO`` filename** in the target folder so
  the layout is directly comparable to the source during review (AC2).
* Source media is taken from ``cfg.foldering_source``
  (``processed_videos/resize_720p``); if a clip is missing there we fall back to
  the other configured search dirs, including ``raw_videos`` (AC3, OQ6).
* Placement uses a **hardlink** on the same ``D:`` volume, falling back to a
  **copy** when the hardlink fails (e.g. cross-device). ``foldering_copy_mode``
  set to ``"copy"`` forces a copy (AC3, OQ6).
* Originals under ``processed_videos`` / ``raw_videos`` are **never moved or
  deleted** — only read for linking/copying (AC3).
* A ``_summary.txt`` reports per-signer clip counts, including ``unknown``
  (AC4).
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_LOGGER_NAME = "qipedc2vsl400.video_organizer"


@dataclass
class OrganizeReport:
    """Result of :func:`organize_by_signer`.

    Attributes:
        by_signer_dir: Absolute path to the produced ``by_signer`` root.
        counts: Per-signer clip counts keyed by ``signer_id`` (3-digit, or the
            ``unknown`` bucket label). Sums to the number of clips placed.
        placed: Total number of clips successfully linked/copied.
        missing_source: ``VIDEO`` filenames whose source file could not be
            found under any search directory (logged, never fatal).
        summary_file: Absolute path to the written ``_summary.txt``.
    """

    by_signer_dir: Path
    counts: dict[str, int] = field(default_factory=dict)
    placed: int = 0
    missing_source: list[str] = field(default_factory=list)
    summary_file: Path | None = None


def _get_file_logger(cfg: Any) -> logging.Logger:
    """Return a logger writing to a timestamped file under ``cfg.log_dir``.

    Mirrors the convention used across the pipeline: the log directory is
    created if needed, a fresh :class:`~logging.FileHandler` is attached per
    call, and propagation is disabled to avoid duplicate console output.
    """
    log_dir = cfg.log_path
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"video_organizer_{timestamp}.log"

    logger = logging.getLogger(f"{_LOGGER_NAME}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _build_source_index(cfg: Any) -> dict[str, Path]:
    """Map each video *filename* to its absolute source path.

    ``cfg.foldering_source`` (the preferred, smaller ``resize_720p`` tree) is
    scanned first, then the remaining ``cfg.video_search_paths()`` (which
    include ``raw_videos``) as a fallback. The first occurrence of a filename
    wins, so the preferred source takes precedence (AC3). Within each directory
    paths are visited in sorted order for determinism; the scan is recursive
    because ``raw_videos`` nests each clip in its own subfolder.
    """
    bases: list[Path] = [cfg.foldering_source_path]
    for base in cfg.video_search_paths():
        if base not in bases:
            bases.append(base)

    index: dict[str, Path] = {}
    for base in bases:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.mp4")):
            if path.is_file() and path.name not in index:
                index[path.name] = path
    return index


def _video_id_from(video: str, cfg: Any) -> str:
    """Derive the metadata ``video_id`` from a ``VIDEO`` filename.

    Mirrors :func:`qipedc2vsl400.mapper._video_id_from` so a clip's record can
    be matched back to its ``signer_assignment`` by ``video_id``.
    """
    if getattr(cfg, "video_id_keep_extension", False):
        return video
    return Path(video).stem


def _place(src: Path, dst: Path, copy_mode: str, logger: logging.Logger) -> None:
    """Link or copy *src* to *dst*, never touching the original.

    Uses :func:`os.link` (hardlink) by default, falling back to
    :func:`shutil.copy2` when the link fails (e.g. cross-device ``OSError``).
    ``copy_mode == "copy"`` forces a copy. Idempotent: an existing *dst* is left
    as-is.
    """
    if dst.exists():
        return
    if copy_mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError as exc:
        logger.info("Hardlink failed for %s -> %s (%s); copying instead", src, dst, exc)
        shutil.copy2(src, dst)


def organize_by_signer(
    records: list[Any],
    signer_assignments: list[Any],
    cfg: Any,
) -> OrganizeReport:
    """Materialize ``Dataset/by_signer/signer_<id>/`` folders for manual review.

    The clip-to-signer mapping is built from *signer_assignments* (each carries
    the original ``VIDEO`` filename and its ``signer_id``). Iteration is driven
    by *records* so that every clip placed on disk corresponds to a clip emitted
    in the metadata, and the folder ``signer_id`` is exactly the metadata
    ``signer_id`` for that clip (Property 11). Each clip's source file is located
    by its original ``VIDEO`` filename under ``cfg.foldering_source`` with a
    fallback to the other search dirs (``raw_videos``), then linked/copied into
    ``signer_<signer_id>/`` keeping that original filename (AC2, AC3).

    Args:
        records: ``OutputRecord`` objects (each exposes ``video_id`` and
            ``signer_id``).
        signer_assignments: ``SignerAssignment`` objects supplying the original
            ``VIDEO`` filename per clip.
        cfg: A :class:`~qipedc2vsl400.config.Config`-like object.

    Returns:
        An :class:`OrganizeReport` with per-signer counts (including
        ``unknown``), the total placed, and any clips whose source was missing.
    """
    logger = _get_file_logger(cfg)
    try:
        by_signer_dir = cfg.by_signer_path
        by_signer_dir.mkdir(parents=True, exist_ok=True)

        unknown_label = getattr(cfg, "signer_unknown_label", "unknown")
        copy_mode = getattr(cfg, "foldering_copy_mode", "hardlink")

        # Always provide an explicit unknown bucket folder (AC1).
        (by_signer_dir / f"signer_{unknown_label}").mkdir(parents=True, exist_ok=True)

        source_index = _build_source_index(cfg)

        # Map metadata video_id -> original VIDEO filename, derived from the
        # signer assignments (which carry the real filename).
        filename_by_video_id: dict[str, str] = {}
        for assignment in signer_assignments:
            video = getattr(assignment, "video", None)
            if not video:
                continue
            vid = _video_id_from(video, cfg)
            filename_by_video_id.setdefault(vid, video)

        counts: dict[str, int] = {}
        missing_source: list[str] = []
        placed = 0

        for record in records:
            signer_id = record.signer_id
            # Recover the original VIDEO filename for this clip.
            video = filename_by_video_id.get(record.video_id)
            if video is None:
                # Fall back to reconstructing a filename from the video_id.
                video = (
                    record.video_id
                    if getattr(cfg, "video_id_keep_extension", False)
                    else f"{record.video_id}.mp4"
                )

            src = source_index.get(video)
            if src is None:
                logger.warning(
                    "No source file found for VIDEO %r (video_id=%s); skipping",
                    video,
                    record.video_id,
                )
                missing_source.append(video)
                continue

            target_dir = by_signer_dir / f"signer_{signer_id}"
            target_dir.mkdir(parents=True, exist_ok=True)
            dst = target_dir / video

            _place(src, dst, copy_mode, logger)
            counts[signer_id] = counts.get(signer_id, 0) + 1
            placed += 1

        summary_file = _write_summary(by_signer_dir, counts, missing_source, cfg)

        logger.info(
            "Organize summary: signers=%d placed=%d missing_source=%d",
            len(counts),
            placed,
            len(missing_source),
        )

        return OrganizeReport(
            by_signer_dir=by_signer_dir,
            counts=counts,
            placed=placed,
            missing_source=missing_source,
            summary_file=summary_file,
        )
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def _write_summary(
    by_signer_dir: Path,
    counts: dict[str, int],
    missing_source: list[str],
    cfg: Any,
) -> Path:
    """Write ``_summary.txt`` mapping each signer to its clip count (AC4).

    Real signer ids are listed first in ascending order, then the ``unknown``
    bucket, then a total line and any missing-source count.
    """
    unknown_label = getattr(cfg, "signer_unknown_label", "unknown")
    summary_file = by_signer_dir / "_summary.txt"

    real_ids = sorted(sid for sid in counts if sid != unknown_label)
    ordered_ids = real_ids + ([unknown_label] if unknown_label in counts else [])

    lines = [f"signer_{sid}: {counts[sid]}" for sid in ordered_ids]
    total = sum(counts.values())
    lines.append(f"total: {total}")
    if missing_source:
        lines.append(f"missing_source: {len(missing_source)}")

    summary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_file

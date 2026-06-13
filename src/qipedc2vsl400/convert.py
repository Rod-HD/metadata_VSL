"""CLI orchestrator for the QIPEDC -> VSL400 conversion pipeline (Task 11).

Runnable as ``python -m qipedc2vsl400.convert``. Wires the full pipeline in
order::

    fetch -> read -> probe -> extract-signers -> map -> write
          -> organize-by-signer -> verify

Every stage is dependency-light and the two heavy/external dependencies — video
probing (``probe_fn``) and face embedding (``embed_fn``) — are **injectable**, so
:func:`run_pipeline` can be exercised by the smoke test with deterministic stubs
and dummy files (no network, no real videos, no ONNX models).

Flags
-----
``--no-fetch``
    Skip the Zenodo label download and the ONNX face-model download.
``--on-missing-video {skip,placeholder}``
    Override ``Config.on_missing_video`` (default ``skip``).
``--skip-signer``
    Reuse an existing ``signers.csv`` (``cfg.signer_sidecar_path``) instead of
    re-embedding every clip.
``--copy-mode {hardlink,copy}``
    Override ``Config.foldering_copy_mode`` for the per-signer foldering.
``--signer-threshold FLOAT``
    Override ``Config.signer_cosine_threshold`` (the cosine "same-signer"
    threshold used by clustering).
``--project-root PATH``
    Override the project root (defaults to the repository root inferred from
    this file's location).

A verification failure (``VerifyReport.ok is False``) is mapped to a non-zero
process exit code so CI/automation can detect regressions.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .mapper import OutputRecord, build_records
from .qipedc_reader import read_rows, validate_rows
from .signer_extractor import SignerAssignment, embed_clip, extract_signers
from .verifier import VerifyReport, verify
from .video_organizer import organize_by_signer
from .video_probe import probe as default_probe
from .writer import write_outputs

_LOGGER_NAME = "qipedc2vsl400.convert"

# Repository root inferred from this file: src/qipedc2vsl400/convert.py -> root.
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass
class PipelineResult:
    """Outcome of a full pipeline run.

    Attributes:
        records: The emitted :class:`OutputRecord` objects.
        valid_rows: Validated QIPEDC rows fed to the mapper.
        skipped_rows: Rows dropped by the reader's validation (missing fields).
        skipped_videos: Rows dropped by the mapper (missing/unreadable video).
        signer_assignments: Per-clip :class:`SignerAssignment` list.
        verify_report: The :class:`VerifyReport` from the final stage.
        summary: The human-readable console summary string.
    """

    records: list[OutputRecord]
    valid_rows: list[Any]
    skipped_rows: list[Any]
    skipped_videos: list[Any]
    signer_assignments: list[SignerAssignment]
    verify_report: VerifyReport
    summary: str

    @property
    def exit_code(self) -> int:
        """Process exit code derived from the verifier (``0`` ok, ``1`` fail)."""
        return self.verify_report.exit_code


def _setup_logger(cfg: Config) -> tuple[logging.Logger, Path]:
    """Create a logger that writes to ``Dataset/logs/convert_<timestamp>.log``.

    A :class:`~logging.FileHandler` (full INFO-level structured log) and a
    console :class:`~logging.StreamHandler` (warnings and above, so routine
    progress stays quiet) are attached. Propagation is disabled to avoid
    duplicate output from the root logger.

    Returns:
        The configured logger and the absolute path of the log file.
    """
    log_dir = cfg.log_path
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"convert_{timestamp}.log"

    logger = logging.getLogger(f"{_LOGGER_NAME}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(file_handler)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(console)

    return logger, log_file


def _close_logger(logger: logging.Logger) -> None:
    """Flush and detach every handler so the log file is fully written."""
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _load_signer_assignments(cfg: Config) -> list[SignerAssignment]:
    """Load signer assignments from the existing ``signers.csv`` side-car.

    Used by ``--skip-signer`` to reuse a previous extraction instead of
    re-embedding every clip. Reads ``cfg.signer_sidecar_path`` and reconstructs
    one :class:`SignerAssignment` per row.

    Raises:
        FileNotFoundError: When the side-car does not exist (so the user knows
            they must run signer extraction at least once first).
    """
    sidecar = cfg.signer_sidecar_path
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"--skip-signer requested but no signers.csv found at {sidecar}. "
            "Run the pipeline once without --skip-signer to produce it."
        )

    assignments: list[SignerAssignment] = []
    with open(sidecar, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video = (row.get("video") or "").strip()
            if not video:
                continue
            signer_id = (row.get("signer_id") or "").strip()
            raw_distance = (row.get("distance") or "").strip()
            distance = float(raw_distance) if raw_distance else None
            raw_cluster = (row.get("cluster_index") or "").strip()
            try:
                cluster_index = int(raw_cluster)
            except ValueError:
                cluster_index = -1
            has_face = (row.get("has_face") or "").strip().lower() == "true"
            assignments.append(
                SignerAssignment(
                    video=video,
                    signer_id=signer_id,
                    cluster_index=cluster_index,
                    distance=distance,
                    has_face=has_face,
                )
            )
    return assignments


def _build_summary(
    records: list[OutputRecord],
    signer_assignments: list[SignerAssignment],
    skipped_rows: list[Any],
    skipped_videos: list[Any],
    cfg: Config,
) -> str:
    """Compose the console summary line block (Requirement 5.3).

    Reports distinct glosses, total records, signer count, unknown count and the
    total skipped count (reader-skips plus mapper video-skips).
    """
    unknown_label = getattr(cfg, "signer_unknown_label", "unknown")
    distinct_glosses = len({r.gloss for r in records})
    real_signers = {
        a.signer_id for a in signer_assignments if a.signer_id != unknown_label
    }
    unknown_count = sum(
        1 for a in signer_assignments if a.signer_id == unknown_label
    )
    skipped_total = len(skipped_rows) + len(skipped_videos)

    lines = [
        "Conversion summary:",
        f"  distinct glosses : {distinct_glosses}",
        f"  total records    : {len(records)}",
        f"  signers          : {len(real_signers)}",
        f"  unknown clips    : {unknown_count}",
        f"  skipped          : {skipped_total} "
        f"(missing-field={len(skipped_rows)}, missing-video={len(skipped_videos)})",
    ]
    return "\n".join(lines)


def run_pipeline(
    cfg: Config,
    *,
    probe_fn: Callable[[Path], Any] | None = None,
    embed_fn: Callable[[Path | None, Any], Any] | None = None,
    fetch: bool = True,
    skip_signer: bool = False,
    embeddings_cache: "Path | str | None" = None,
    session: Any | None = None,
) -> PipelineResult:
    """Run the end-to-end conversion pipeline.

    Stages, in order: fetch (optional) -> read + validate -> extract-signers (or
    load existing ``signers.csv`` when *skip_signer*) -> map (probing each clip
    via *probe_fn*) -> write ``front_view.json`` -> organize per-signer folders
    -> verify.

    Args:
        cfg: The (frozen) :class:`Config` for this run.
        probe_fn: ``probe(video_path) -> VideoProps | None``; defaults to
            :func:`qipedc2vsl400.video_probe.probe`. Injected by tests.
        embed_fn: ``embed(video_path, cfg) -> vector | None``; defaults to
            :func:`qipedc2vsl400.signer_extractor.embed_clip`. Injected by tests.
        fetch: When ``True`` (default) download the VSL400 labels and the ONNX
            face models; ``--no-fetch`` sets this ``False``.
        skip_signer: When ``True`` reuse an existing ``signers.csv`` instead of
            re-embedding and clustering.
        embeddings_cache: Optional path to a NumPy ``.npz`` embedding cache. When
            given (and not ``skip_signer``), only clips missing from the cache
            are re-embedded; clustering still runs over the full set, so the
            result matches a full re-run while skipping the expensive embedding
            of previously-seen clips.
        session: Optional ``requests.Session`` passed through to the fetch
            stage (injectable for testing).

    Returns:
        A :class:`PipelineResult`. ``result.exit_code`` is non-zero when the
        verifier reports a failure.
    """
    probe_fn = probe_fn or default_probe
    embed_fn = embed_fn or embed_clip

    logger, log_file = _setup_logger(cfg)
    try:
        logger.info("Pipeline start; project_root=%s log=%s", cfg.project_root, log_file)

        # 1) Fetch VSL400 labels + ONNX face models (skippable).
        if fetch:
            # Imported lazily so a --no-fetch run never imports ``requests``.
            from .vsl400_fetch import fetch_face_models, fetch_labels

            logger.info("Fetching VSL400 labels into %s", cfg.vsl400_path)
            fetch_labels(cfg, session=session)
            logger.info("Fetching ONNX face models into %s", cfg.models_path)
            fetch_face_models(cfg, session=session)
        else:
            logger.info("Skipping fetch (--no-fetch)")

        # 2) Read + validate QIPEDC labels.
        rows = read_rows(cfg)
        valid_rows, skipped_rows = validate_rows(cfg, rows)
        logger.info(
            "Read %d rows; valid=%d skipped(missing-field)=%d",
            len(rows),
            len(valid_rows),
            len(skipped_rows),
        )

        # 3) Signer assignments: extract from video, or reuse signers.csv.
        if skip_signer:
            signer_assignments = _load_signer_assignments(cfg)
            logger.info(
                "Loaded %d signer assignments from %s (--skip-signer)",
                len(signer_assignments),
                cfg.signer_sidecar_path,
            )
        else:
            signer_assignments = extract_signers(
                valid_rows, cfg, embed_fn=embed_fn, embeddings_cache=embeddings_cache
            )
            logger.info(
                "Extracted signers for %d clips%s",
                len(signer_assignments),
                " (embedding cache enabled)" if embeddings_cache else "",
            )

        # 4) Map rows -> superset records (probing each clip).
        records, skipped_videos = build_records(
            valid_rows, probe_fn, signer_assignments, cfg
        )
        logger.info(
            "Mapped %d records; skipped(missing-video)=%d",
            len(records),
            len(skipped_videos),
        )

        # 5) Write the front_view.json output.
        write_outputs(records, cfg)
        logger.info("Wrote metadata to %s", cfg.output_view_file)

        # 6) Organize clips into per-signer folders.
        organize_report = organize_by_signer(records, signer_assignments, cfg)
        logger.info(
            "Organized %d clips into %s (missing source=%d)",
            organize_report.placed,
            organize_report.by_signer_dir,
            len(organize_report.missing_source),
        )

        # 7) Verify the output against source + (optional) VSL400 reference.
        reference_schema = None
        try:
            from .vsl400_fetch import load_reference_schema

            reference_schema = load_reference_schema(cfg.vsl400_path)
        except Exception as exc:  # pragma: no cover - reference is best-effort
            logger.warning("Could not load VSL400 reference schema: %s", exc)

        report = verify(
            records,
            valid_rows,
            skipped_videos,
            signer_assignments,
            reference_schema,
            cfg,
        )

        summary = _build_summary(
            records, signer_assignments, skipped_rows, skipped_videos, cfg
        )
        logger.info("Verification ok=%s", report.ok)
        if not report.ok:
            for failure in report.failures:
                logger.error("Verification failure [%s]: %s", failure.name, failure.detail)

        return PipelineResult(
            records=records,
            valid_rows=valid_rows,
            skipped_rows=skipped_rows,
            skipped_videos=skipped_videos,
            signer_assignments=signer_assignments,
            verify_report=report,
            summary=summary,
        )
    finally:
        _close_logger(logger)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m qipedc2vsl400.convert",
        description="Convert QIPEDC labels into the VSL400 superset metadata format.",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip downloading VSL400 labels and the ONNX face models.",
    )
    parser.add_argument(
        "--on-missing-video",
        choices=["skip", "placeholder"],
        default=None,
        help="Policy when a referenced video is missing/unreadable (default: skip).",
    )
    parser.add_argument(
        "--skip-signer",
        action="store_true",
        help="Reuse an existing signers.csv instead of re-embedding clips.",
    )
    parser.add_argument(
        "--embeddings-cache",
        action="store_true",
        help=(
            "Cache per-clip face embeddings (under the output dir) and reuse them "
            "on re-runs, so only newly added clips are embedded."
        ),
    )
    parser.add_argument(
        "--copy-mode",
        choices=["hardlink", "copy"],
        default=None,
        help="How to place clips into per-signer folders (default: hardlink).",
    )
    parser.add_argument(
        "--signer-threshold",
        type=float,
        default=None,
        help="Cosine 'same-signer' distance threshold for clustering.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Override the project root (defaults to the repository root).",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    """Build a :class:`Config`, overriding frozen fields from CLI flags.

    ``Config`` is frozen, so :func:`dataclasses.replace` is used to apply any
    overrides supplied on the command line.
    """
    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else _DEFAULT_PROJECT_ROOT
    )
    cfg = Config(project_root=project_root)

    overrides: dict[str, Any] = {}
    if args.on_missing_video is not None:
        overrides["on_missing_video"] = args.on_missing_video
    if args.copy_mode is not None:
        overrides["foldering_copy_mode"] = args.copy_mode
    if args.signer_threshold is not None:
        overrides["signer_cosine_threshold"] = args.signer_threshold
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    Parses arguments, builds the :class:`Config` (applying flag overrides),
    runs the pipeline, prints the console summary, and returns the verifier's
    exit code (non-zero on verification failure).
    """
    args = _build_arg_parser().parse_args(argv)
    cfg = _config_from_args(args)

    result = run_pipeline(
        cfg,
        fetch=not args.no_fetch,
        skip_signer=args.skip_signer,
        embeddings_cache=(
            cfg.signer_embeddings_cache_path if args.embeddings_cache else None
        ),
    )

    print(result.summary)
    if not result.verify_report.ok:
        print("Verification FAILED:", file=sys.stderr)
        for failure in result.verify_report.failures:
            print(f"  - {failure.name}: {failure.detail}", file=sys.stderr)

    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

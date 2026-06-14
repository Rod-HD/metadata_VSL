"""Verify the converted output against the source and the VSL400 reference (Req 6).

:func:`verify` runs a battery of checks over the conversion result and returns a
:class:`VerifyReport`. A failing report (``ok is False``) is the mechanism the
CLI maps to a **non-zero exit code** so CI/automation can detect regressions
(design "Error Handling": *Verifier failure -> non-zero exit code*). Use
:pyattr:`VerifyReport.exit_code` (``0`` when ``ok``, ``1`` otherwise) or
:meth:`VerifyReport.raise_for_status` from the orchestrator.

Checks performed (all reference Requirement 6 unless noted):

* **Count preservation** (6.1): ``len(records) == len(valid_rows) - len(skipped)``.
* **Superset field / type conformance** (6.2): every record carries all superset
  fields with the right types; ``video_id`` equals the kept QIPEDC value (stem
  unless ``cfg.video_id_keep_extension``); ``signer_id`` matches ``^\\d{N}$``
  (``N = cfg.signer_id_width``, default 3) or equals ``cfg.signer_unknown_label``.
* **Gloss-set equality** (6.3): the distinct ``gloss`` set of the records equals
  the distinct ``label`` set of the *mapped* valid rows (``valid_rows`` minus
  ``skipped``).
* **Signer <-> folder <-> signers.csv consistency** (9.5): every record's
  ``signer_id`` equals the ``SignerAssignment`` for the same clip (matched by the
  original ``VIDEO`` filename), and — when the ``signers.csv`` side-car exists —
  the assignments agree with it.
* **Reference-only compatibility** (6.4): when a sample VSL400 ``*_view.json`` is
  available, the VSL400-derived fields (``signer_id``, ``fps``, ``resolution``,
  ``num_frames``, ``length_seconds``) are checked for **name/type** compatibility
  only — not strict equality; QIPEDC-only fields are expected as additions.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The full superset key set every output record must carry (design Property 2).
SUPERSET_FIELDS: tuple[str, ...] = (
    "video_id",
    "gloss",
    "region",
    "topic",
    "id",
    "signer_id",
    "fps",
    "resolution",
    "num_frames",
    "length_seconds",
)

# The QIPEDC-native optional fields (kept; may be ``None``).
_OPTIONAL_STR_FIELDS: tuple[str, ...] = ("region", "topic", "id")

# Fields added from the VSL400 reference; the only ones compared against a real
# VSL400 sample (reference check, not strict equality).
VSL400_DERIVED_FIELDS: tuple[str, ...] = (
    "signer_id",
    "fps",
    "resolution",
    "num_frames",
    "length_seconds",
)


@dataclass
class CheckResult:
    """Outcome of a single verification check.

    Attributes:
        name: Short identifier of the check (e.g. ``"count_preservation"``).
        passed: Whether the check succeeded.
        detail: Human-readable explanation, empty on success.
    """

    name: str
    passed: bool
    detail: str = ""


@dataclass
class VerifyReport:
    """Aggregate result of :func:`verify`.

    Attributes:
        ok: ``True`` only when every check passed.
        checks: The individual :class:`CheckResult` entries, in run order.
    """

    ok: bool
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        """The subset of :attr:`checks` that did not pass."""
        return [c for c in self.checks if not c.passed]

    @property
    def exit_code(self) -> int:
        """Process exit code: ``0`` when :attr:`ok`, ``1`` otherwise.

        The CLI orchestrator uses this so a verification failure surfaces as a
        non-zero exit status (design "Error Handling").
        """
        return 0 if self.ok else 1

    def raise_for_status(self) -> None:
        """Raise :class:`VerificationError` when verification failed.

        Convenience for callers that prefer exceptions over inspecting
        :attr:`ok` / :attr:`exit_code`.
        """
        if not self.ok:
            summary = "; ".join(f"{c.name}: {c.detail}" for c in self.failures)
            raise VerificationError(f"Verification failed: {summary}")


class VerificationError(RuntimeError):
    """Raised by :meth:`VerifyReport.raise_for_status` on a failing report."""


# --- helpers -----------------------------------------------------------------


def _type_category(value: Any) -> str:
    """Classify *value* into a coarse type category for compatibility checks.

    ``bool`` is treated distinctly from ``number`` (Python's ``bool`` is an
    ``int`` subclass, but the schema never expects booleans); ``int``/``float``
    collapse to ``"number"`` so e.g. ``25`` and ``25.0`` are compatible.
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return "none"
    return type(value).__name__


def _video_id_from(video: str, cfg: Any) -> str:
    """Derive ``video_id`` from a ``VIDEO`` filename (mirrors the mapper rule)."""
    if getattr(cfg, "video_id_keep_extension", False):
        return video
    return Path(video).stem


def _mapped_rows(valid_rows: list[Any], skipped: list[Any]) -> list[Any]:
    """Return the valid rows that were actually mapped (``valid_rows`` - ``skipped``).

    Matching is by object identity so duplicate field values never cause a row
    to be wrongly excluded.
    """
    skipped_ids = {id(row) for row in skipped}
    return [row for row in valid_rows if id(row) not in skipped_ids]


def _signer_pattern(cfg: Any) -> re.Pattern[str]:
    """Compile the ``^\\d{N}$`` signer-id pattern for ``N = cfg.signer_id_width``."""
    width = int(getattr(cfg, "signer_id_width", 3))
    return re.compile(rf"^\d{{{width}}}$")


# --- individual checks -------------------------------------------------------


def _check_count_preservation(
    records: list[Any], valid_rows: list[Any], skipped: list[Any]
) -> CheckResult:
    expected = len(valid_rows) - len(skipped)
    actual = len(records)
    if actual == expected:
        return CheckResult("count_preservation", True)
    return CheckResult(
        "count_preservation",
        False,
        f"expected {expected} records (valid={len(valid_rows)} - "
        f"skipped={len(skipped)}) but got {actual}",
    )


def _check_record_schema(
    records: list[Any], mapped_rows: list[Any], cfg: Any
) -> CheckResult:
    unknown_label = getattr(cfg, "signer_unknown_label", "unknown")
    pattern = _signer_pattern(cfg)
    expected_video_ids = {_video_id_from(r.video, cfg) for r in mapped_rows if r.video}

    problems: list[str] = []
    for index, record in enumerate(records):
        # All superset fields present.
        for fname in SUPERSET_FIELDS:
            if not hasattr(record, fname):
                problems.append(f"record[{index}] missing field {fname!r}")

        video_id = getattr(record, "video_id", None)
        if not isinstance(video_id, str) or not video_id:
            problems.append(f"record[{index}] video_id must be a non-empty str")
        elif expected_video_ids and video_id not in expected_video_ids:
            problems.append(
                f"record[{index}] video_id {video_id!r} does not match any kept "
                f"QIPEDC VIDEO value"
            )

        gloss = getattr(record, "gloss", None)
        if not isinstance(gloss, str) or not gloss:
            problems.append(f"record[{index}] gloss must be a non-empty str")

        for fname in _OPTIONAL_STR_FIELDS:
            value = getattr(record, fname, None)
            if value is not None and not isinstance(value, str):
                problems.append(f"record[{index}] {fname} must be str or None")

        signer_id = getattr(record, "signer_id", None)
        if not (
            isinstance(signer_id, str)
            and (pattern.match(signer_id) or signer_id == unknown_label)
        ):
            problems.append(
                f"record[{index}] signer_id {signer_id!r} is neither "
                f"{pattern.pattern} nor {unknown_label!r}"
            )

        fps = getattr(record, "fps", None)
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            problems.append(f"record[{index}] fps must be a number")

        resolution = getattr(record, "resolution", None)
        if isinstance(resolution, bool) or not isinstance(resolution, int):
            problems.append(f"record[{index}] resolution must be an int")

        num_frames = getattr(record, "num_frames", None)
        if isinstance(num_frames, bool) or not isinstance(num_frames, int):
            problems.append(f"record[{index}] num_frames must be an int")

        length_seconds = getattr(record, "length_seconds", None)
        if isinstance(length_seconds, bool) or not isinstance(
            length_seconds, (int, float)
        ):
            problems.append(f"record[{index}] length_seconds must be a number")

    if problems:
        return CheckResult("record_schema", False, "; ".join(problems))
    return CheckResult("record_schema", True)


def _check_gloss_set_equality(
    records: list[Any], mapped_rows: list[Any]
) -> CheckResult:
    record_glosses = {r.gloss for r in records}
    source_glosses = {row.label for row in mapped_rows}
    if record_glosses == source_glosses:
        return CheckResult("gloss_set_equality", True)
    missing = source_glosses - record_glosses
    extra = record_glosses - source_glosses
    return CheckResult(
        "gloss_set_equality",
        False,
        f"gloss sets differ: missing_from_output={sorted(map(str, missing))}, "
        f"unexpected_in_output={sorted(map(str, extra))}",
    )


def _read_sidecar(cfg: Any) -> dict[str, str] | None:
    """Return ``{video: signer_id}`` from the ``signers.csv`` side-car, if present."""
    try:
        sidecar_path = cfg.signer_sidecar_path
    except AttributeError:
        return None
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.is_file():
        return None
    mapping: dict[str, str] = {}
    with open(sidecar_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video = row.get("video")
            signer_id = row.get("signer_id")
            if video is not None and signer_id is not None:
                mapping[video] = signer_id
    return mapping


def _check_signer_consistency(
    records: list[Any], signer_assignments: list[Any], cfg: Any
) -> CheckResult:
    # Map each clip's metadata video_id -> assigned signer_id (via VIDEO filename).
    assignment_by_video: dict[str, str] = {}
    assignment_signer_by_video_id: dict[str, str] = {}
    for assignment in signer_assignments:
        video = getattr(assignment, "video", None)
        if not video:
            continue
        assignment_by_video[video] = assignment.signer_id
        assignment_signer_by_video_id.setdefault(
            _video_id_from(video, cfg), assignment.signer_id
        )

    problems: list[str] = []

    # Every record's signer_id must match its clip's assignment.
    for record in records:
        expected = assignment_signer_by_video_id.get(record.video_id)
        if expected is None:
            problems.append(
                f"no signer assignment for clip video_id={record.video_id!r}"
            )
        elif record.signer_id != expected:
            problems.append(
                f"signer mismatch for {record.video_id!r}: metadata="
                f"{record.signer_id!r} assignment={expected!r}"
            )

    # When the signers.csv side-car exists, assignments must agree with it.
    sidecar = _read_sidecar(cfg)
    if sidecar is not None:
        for video, signer_id in assignment_by_video.items():
            csv_signer = sidecar.get(video)
            if csv_signer is None:
                problems.append(f"{video!r} missing from signers.csv side-car")
            elif csv_signer != signer_id:
                problems.append(
                    f"signers.csv mismatch for {video!r}: csv={csv_signer!r} "
                    f"assignment={signer_id!r}"
                )

    if problems:
        return CheckResult("signer_consistency", False, "; ".join(problems))
    return CheckResult("signer_consistency", True)


def _check_reference_compatibility(
    records: list[Any], reference_schema: list[dict] | None
) -> CheckResult:
    if not reference_schema:
        return CheckResult(
            "reference_compatibility", True, "no VSL400 reference sample available"
        )
    ref = next((obj for obj in reference_schema if isinstance(obj, dict)), None)
    if ref is None:
        return CheckResult(
            "reference_compatibility", True, "reference sample had no objects"
        )
    if not records:
        return CheckResult(
            "reference_compatibility", True, "no records to compare against reference"
        )

    sample = records[0]
    problems: list[str] = []
    for fname in VSL400_DERIVED_FIELDS:
        if fname not in ref:
            problems.append(
                f"VSL400 reference object lacks derived field {fname!r}"
            )
            continue
        ref_cat = _type_category(ref[fname])
        our_cat = _type_category(getattr(sample, fname, None))
        if ref_cat != our_cat:
            problems.append(
                f"type mismatch for {fname!r}: reference={ref_cat}, output={our_cat}"
            )

    if problems:
        return CheckResult("reference_compatibility", False, "; ".join(problems))
    return CheckResult("reference_compatibility", True)


# --- public API --------------------------------------------------------------


def verify(
    records: list[Any],
    valid_rows: list[Any],
    skipped: list[Any],
    signer_assignments: list[Any],
    reference_schema: list[dict] | None,
    cfg: Any,
) -> VerifyReport:
    """Verify the conversion result; return a :class:`VerifyReport`.

    Args:
        records: The emitted :class:`~qipedc2vsl400.mapper.OutputRecord` objects.
        valid_rows: The validated QIPEDC rows that fed the mapper.
        skipped: Rows dropped by the mapper (e.g. missing/unreadable video).
        signer_assignments: The :class:`~qipedc2vsl400.signer_extractor.SignerAssignment`
            list produced by signer extraction.
        reference_schema: A sample VSL400 ``*_view.json`` list (or ``None`` when
            unavailable); used for name/type compatibility only.
        cfg: A :class:`~qipedc2vsl400.config.Config`-like object.

    Returns:
        A :class:`VerifyReport`; ``ok`` is ``True`` only when every check passes.
        Map ``ok is False`` to a non-zero exit code via
        :pyattr:`VerifyReport.exit_code` (or :meth:`VerifyReport.raise_for_status`).
    """
    mapped_rows = _mapped_rows(valid_rows, skipped)

    checks = [
        _check_count_preservation(records, valid_rows, skipped),
        _check_record_schema(records, mapped_rows, cfg),
        _check_gloss_set_equality(records, mapped_rows),
        _check_signer_consistency(records, signer_assignments, cfg),
        _check_reference_compatibility(records, reference_schema),
    ]

    return VerifyReport(ok=all(c.passed for c in checks), checks=checks)

"""Tests for ``qipedc2vsl400.verifier.verify`` (Task 10, Requirement 6 + 9.5).

These cover a fully-passing conversion plus one failing case per check:

* count mismatch (6.1)
* bad ``signer_id`` form (6.2)
* gloss-set mismatch (6.3)
* signer / folder inconsistency (9.5)
* reference-schema type mismatch (6.4)

A verification failure must yield a non-zero exit signal — asserted via
``VerifyReport.ok is False`` and ``VerifyReport.exit_code != 0``. All fixtures
are synthetic (no real videos, models, or network), and every test constructs
``Config(project_root=tmp_path)`` so nothing touches ``C:``.
"""

from __future__ import annotations

import pytest

from qipedc2vsl400.config import Config
from qipedc2vsl400.mapper import OutputRecord
from qipedc2vsl400.qipedc_reader import QipedcRow
from qipedc2vsl400.signer_extractor import SignerAssignment
from qipedc2vsl400.verifier import VerificationError, verify


# --- helpers ----------------------------------------------------------------


def _row(video: str, label: str, index: int) -> QipedcRow:
    return QipedcRow(
        stt=str(index),
        id=f"id{index}",
        video=video,
        label=label,
        region="Chung",
        topic="Số",
        signer=None,
        source_file="batch_1.xlsx",
        row_index=index,
    )


def _record(video_id: str, gloss: str, signer_id: str) -> OutputRecord:
    return OutputRecord(
        video_id=video_id,
        gloss=gloss,
        region="Chung",
        topic="Số",
        id="id1",
        signer_id=signer_id,
        fps=25.0,
        resolution=720,
        num_frames=50,
        length_seconds=2.0,
    )


def _assignment(video: str, signer_id: str) -> SignerAssignment:
    has_face = signer_id != "unknown"
    return SignerAssignment(
        video=video,
        signer_id=signer_id,
        cluster_index=0 if has_face else -1,
        distance=0.1 if has_face else None,
        has_face=has_face,
    )


def _reference_schema() -> list[dict]:
    """A minimal, schema-faithful VSL400 sample object."""
    return [
        {
            "video_id": "000000",
            "signer_id": "001",
            "fps": 25.0,
            "resolution": 1080,
            "gloss": "Anh",
            "num_frames": 65,
            "length_seconds": 2.6,
        }
    ]


def _passing_fixture():
    """Three valid rows -> three records, all signer-consistent."""
    rows = [
        _row("D0530.mp4", "Anh", 2),
        _row("D0531.mp4", "Em", 3),
        _row("D0532.mp4", "Ba", 4),
    ]
    records = [
        _record("D0530", "Anh", "001"),
        _record("D0531", "Em", "001"),
        _record("D0532", "Ba", "002"),
    ]
    assignments = [
        _assignment("D0530.mp4", "001"),
        _assignment("D0531.mp4", "001"),
        _assignment("D0532.mp4", "002"),
    ]
    skipped: list[QipedcRow] = []
    return rows, records, assignments, skipped


# --- tests ------------------------------------------------------------------


def test_passing_case(tmp_path):
    """A consistent conversion passes every check (ok, exit_code 0)."""
    cfg = Config(project_root=tmp_path)
    rows, records, assignments, skipped = _passing_fixture()

    report = verify(records, rows, skipped, assignments, _reference_schema(), cfg)

    assert report.ok is True, [c.detail for c in report.failures]
    assert report.exit_code == 0
    assert report.failures == []
    # raise_for_status is a no-op on success.
    report.raise_for_status()


def test_passing_without_reference(tmp_path):
    """Reference compatibility is skipped (passes) when no sample is available."""
    cfg = Config(project_root=tmp_path)
    rows, records, assignments, skipped = _passing_fixture()

    report = verify(records, rows, skipped, assignments, None, cfg)

    assert report.ok is True, [c.detail for c in report.failures]


def test_count_mismatch_fails(tmp_path):
    """len(records) != len(valid_rows) - len(skipped) -> failure (6.1)."""
    cfg = Config(project_root=tmp_path)
    rows, records, assignments, _skipped = _passing_fixture()
    # Mark one valid row as skipped while still emitting all three records.
    skipped = [rows[2]]

    report = verify(records, rows, skipped, assignments, _reference_schema(), cfg)

    assert report.ok is False
    assert report.exit_code == 1
    assert any(c.name == "count_preservation" for c in report.failures)
    with pytest.raises(VerificationError):
        report.raise_for_status()


def test_bad_signer_id_fails(tmp_path):
    """A signer_id that is neither 3-digit nor 'unknown' -> failure (6.2)."""
    cfg = Config(project_root=tmp_path)
    rows, records, assignments, skipped = _passing_fixture()
    # Corrupt the third record's signer_id to an invalid form.
    records[2] = _record("D0532", "Ba", "2")  # not zero-padded 3-digit
    assignments[2] = _assignment("D0532.mp4", "2")  # keep signer-consistent

    report = verify(records, rows, skipped, assignments, _reference_schema(), cfg)

    assert report.ok is False
    assert report.exit_code == 1
    assert any(c.name == "record_schema" for c in report.failures)


def test_unknown_signer_id_is_allowed(tmp_path):
    """The 'unknown' bucket label is a valid signer_id (6.2)."""
    cfg = Config(project_root=tmp_path)
    rows, records, assignments, skipped = _passing_fixture()
    records[2] = _record("D0532", "Ba", "unknown")
    assignments[2] = _assignment("D0532.mp4", "unknown")

    report = verify(records, rows, skipped, assignments, _reference_schema(), cfg)

    assert report.ok is True, [c.detail for c in report.failures]


def test_gloss_set_mismatch_fails(tmp_path):
    """Distinct gloss set differs from source label set -> failure (6.3)."""
    cfg = Config(project_root=tmp_path)
    rows, records, assignments, skipped = _passing_fixture()
    # Change a record gloss so the output gloss set no longer matches the source.
    records[1] = _record("D0531", "WRONG", "001")

    report = verify(records, rows, skipped, assignments, _reference_schema(), cfg)

    assert report.ok is False
    assert report.exit_code == 1
    assert any(c.name == "gloss_set_equality" for c in report.failures)


def test_signer_folder_inconsistency_fails(tmp_path):
    """Record signer_id disagrees with the clip's assignment -> failure (9.5)."""
    cfg = Config(project_root=tmp_path)
    rows, records, assignments, skipped = _passing_fixture()
    # Record says 002 but the assignment for the same clip says 001.
    records[2] = _record("D0532", "Ba", "002")
    assignments[2] = _assignment("D0532.mp4", "001")

    report = verify(records, rows, skipped, assignments, _reference_schema(), cfg)

    assert report.ok is False
    assert report.exit_code == 1
    assert any(c.name == "signer_consistency" for c in report.failures)


def test_signers_csv_consistency_passes(tmp_path):
    """When signers.csv agrees with the assignments, the check passes."""
    cfg = Config(project_root=tmp_path)
    rows, records, assignments, skipped = _passing_fixture()
    # Write a matching side-car at cfg.signer_sidecar_path.
    sidecar = cfg.signer_sidecar_path
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        "video,signer_id,cluster_index,distance,has_face\n"
        "D0530.mp4,001,0,0.1,True\n"
        "D0531.mp4,001,0,0.1,True\n"
        "D0532.mp4,002,1,0.1,True\n",
        encoding="utf-8",
    )

    report = verify(records, rows, skipped, assignments, _reference_schema(), cfg)

    assert report.ok is True, [c.detail for c in report.failures]


def test_signers_csv_inconsistency_fails(tmp_path):
    """A signers.csv that disagrees with the assignments -> failure (9.5)."""
    cfg = Config(project_root=tmp_path)
    rows, records, assignments, skipped = _passing_fixture()
    sidecar = cfg.signer_sidecar_path
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    # D0532 is 003 in the side-car but 002 in the assignments.
    sidecar.write_text(
        "video,signer_id,cluster_index,distance,has_face\n"
        "D0530.mp4,001,0,0.1,True\n"
        "D0531.mp4,001,0,0.1,True\n"
        "D0532.mp4,003,2,0.1,True\n",
        encoding="utf-8",
    )

    report = verify(records, rows, skipped, assignments, _reference_schema(), cfg)

    assert report.ok is False
    assert any(c.name == "signer_consistency" for c in report.failures)


def test_reference_type_mismatch_fails(tmp_path):
    """A reference object with an incompatible field type -> failure (6.4)."""
    cfg = Config(project_root=tmp_path)
    rows, records, assignments, skipped = _passing_fixture()
    bad_reference = _reference_schema()
    # fps as a string is type-incompatible with our numeric fps.
    bad_reference[0]["fps"] = "25.0"

    report = verify(records, rows, skipped, assignments, bad_reference, cfg)

    assert report.ok is False
    assert report.exit_code == 1
    assert any(c.name == "reference_compatibility" for c in report.failures)


def test_reference_missing_derived_field_fails(tmp_path):
    """A reference object missing a VSL400-derived field name -> failure (6.4)."""
    cfg = Config(project_root=tmp_path)
    rows, records, assignments, skipped = _passing_fixture()
    bad_reference = _reference_schema()
    del bad_reference[0]["num_frames"]

    report = verify(records, rows, skipped, assignments, bad_reference, cfg)

    assert report.ok is False
    assert any(c.name == "reference_compatibility" for c in report.failures)

"""Tests for ``qipedc2vsl400.video_organizer.organize_by_signer`` (Task 9).

These validate the design's correctness **Properties 10 & 11** against a
throw-away temp tree of dummy ``.mp4`` files (no real videos needed):

* **Property 10** - signer accounting & no-face handling: every clip is placed
  in exactly one folder; the unknown bucket is honored; the per-signer counts
  sum to the number of clips placed.
* **Property 11** - metadata <-> folder signer consistency: the ``signer_id`` of
  the folder a clip lands in equals the ``signer_id`` emitted in its metadata.

Plus the Requirement 9 behaviors: original ``VIDEO`` filename preserved (AC2),
hardlink with copy fallback / forced copy (AC3), originals never moved or
deleted (AC3), and a ``_summary.txt`` with per-signer counts incl. ``unknown``
(AC4).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from qipedc2vsl400.config import Config
from qipedc2vsl400.mapper import OutputRecord
from qipedc2vsl400.signer_extractor import SignerAssignment
from qipedc2vsl400.video_organizer import organize_by_signer


# --- helpers ----------------------------------------------------------------


def _record(video_id: str, signer_id: str) -> OutputRecord:
    """A minimal ``OutputRecord`` (only ``video_id``/``signer_id`` matter here)."""
    return OutputRecord(
        video_id=video_id,
        gloss="x",
        region=None,
        topic=None,
        id=None,
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


def _make_source_files(cfg: Config, filenames: list[str]) -> dict[str, str]:
    """Create dummy ``.mp4`` files under ``cfg.foldering_source`` with content."""
    src_dir = cfg.foldering_source_path
    src_dir.mkdir(parents=True, exist_ok=True)
    contents: dict[str, str] = {}
    for name in filenames:
        content = f"dummy-video-bytes-for-{name}"
        (src_dir / name).write_text(content, encoding="utf-8")
        contents[name] = content
    return contents


def _build_fixture(cfg: Config):
    """Three signers + one unknown clip; create matching source files."""
    spec = [
        ("D0530.mp4", "001"),
        ("D0531.mp4", "001"),
        ("D0532.mp4", "002"),
        ("D0533N.mp4", "003"),
        ("D0534.mp4", "unknown"),
    ]
    contents = _make_source_files(cfg, [name for name, _ in spec])
    records = [_record(name.rsplit(".", 1)[0], sid) for name, sid in spec]
    assignments = [_assignment(name, sid) for name, sid in spec]
    return spec, contents, records, assignments


# --- tests ------------------------------------------------------------------


def test_per_signer_placement_and_original_name_preserved(tmp_path):
    """Clips land in signer_<id> folders under their original VIDEO filename."""
    cfg = Config(project_root=tmp_path)
    spec, _contents, records, assignments = _build_fixture(cfg)

    organize_by_signer(records, assignments, cfg)

    base = cfg.by_signer_path
    for name, sid in spec:
        placed = base / f"signer_{sid}" / name
        assert placed.exists(), f"{name} should be placed in signer_{sid}"
        # Original VIDEO filename preserved exactly (AC2).
        assert placed.name == name


def test_folder_signer_id_matches_metadata(tmp_path):
    """Property 11: folder signer_id equals the metadata signer_id per clip."""
    cfg = Config(project_root=tmp_path)
    spec, _contents, records, assignments = _build_fixture(cfg)

    organize_by_signer(records, assignments, cfg)

    base = cfg.by_signer_path
    for record in records:
        # Find which signer_<id> folder contains this clip.
        matches = [
            p.parent.name
            for p in base.rglob("*.mp4")
            if p.stem == record.video_id
        ]
        assert matches == [f"signer_{record.signer_id}"]


def test_accounting_sums_to_placed_including_unknown(tmp_path):
    """Property 10: per-signer counts (incl. unknown) sum to clips placed."""
    cfg = Config(project_root=tmp_path)
    spec, _contents, records, assignments = _build_fixture(cfg)

    report = organize_by_signer(records, assignments, cfg)

    assert report.placed == len(spec)
    assert sum(report.counts.values()) == report.placed
    # The unknown clip is accounted for in its own bucket.
    assert report.counts["unknown"] == 1
    assert report.counts["001"] == 2


def test_originals_are_untouched(tmp_path):
    """Originals under the source tree are never moved or deleted (AC3)."""
    cfg = Config(project_root=tmp_path)
    spec, contents, records, assignments = _build_fixture(cfg)

    organize_by_signer(records, assignments, cfg)

    src_dir = cfg.foldering_source_path
    for name, _sid in spec:
        original = src_dir / name
        assert original.exists(), f"original {name} must remain"
        assert original.read_text(encoding="utf-8") == contents[name]


def test_summary_file_reports_counts(tmp_path):
    """_summary.txt lists per-signer counts including unknown (AC4)."""
    cfg = Config(project_root=tmp_path)
    spec, _contents, records, assignments = _build_fixture(cfg)

    report = organize_by_signer(records, assignments, cfg)

    summary = cfg.by_signer_path / "_summary.txt"
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "signer_001: 2" in text
    assert "signer_002: 1" in text
    assert "signer_003: 1" in text
    assert "signer_unknown: 1" in text
    assert "total: 5" in text


def test_hardlink_used_by_default(tmp_path, monkeypatch):
    """Default mode hardlinks the clip (os.link is invoked)."""
    cfg = Config(project_root=tmp_path)
    spec, _contents, records, assignments = _build_fixture(cfg)

    real_link = os.link
    calls: list[tuple[str, str]] = []

    def spy_link(src, dst, *args, **kwargs):
        calls.append((str(src), str(dst)))
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", spy_link)

    organize_by_signer(records, assignments, cfg)

    assert len(calls) == len(spec), "every clip should be hardlinked by default"


def test_copy_fallback_when_hardlink_fails(tmp_path, monkeypatch):
    """When os.link raises OSError, placement falls back to a copy (AC3)."""
    cfg = Config(project_root=tmp_path)
    spec, contents, records, assignments = _build_fixture(cfg)

    def failing_link(src, dst, *args, **kwargs):
        raise OSError("cross-device link not permitted")

    monkeypatch.setattr(os, "link", failing_link)

    report = organize_by_signer(records, assignments, cfg)

    assert report.placed == len(spec)
    base = cfg.by_signer_path
    for name, sid in spec:
        placed = base / f"signer_{sid}" / name
        assert placed.exists()
        # Content was copied faithfully.
        assert placed.read_text(encoding="utf-8") == contents[name]


def test_copy_mode_forces_copy(tmp_path, monkeypatch):
    """foldering_copy_mode='copy' never attempts a hardlink."""
    cfg = Config(project_root=tmp_path, foldering_copy_mode="copy")
    spec, contents, records, assignments = _build_fixture(cfg)

    def forbidden_link(src, dst, *args, **kwargs):
        raise AssertionError("os.link must not be called in copy mode")

    monkeypatch.setattr(os, "link", forbidden_link)

    report = organize_by_signer(records, assignments, cfg)

    assert report.placed == len(spec)
    base = cfg.by_signer_path
    for name, sid in spec:
        placed = base / f"signer_{sid}" / name
        assert placed.exists()
        assert placed.read_text(encoding="utf-8") == contents[name]


def test_missing_source_is_reported_not_fatal(tmp_path):
    """A clip with no source file is logged/reported, not fatal."""
    cfg = Config(project_root=tmp_path)
    # Only create a source for D0530; D0999 has no file.
    _make_source_files(cfg, ["D0530.mp4"])
    records = [_record("D0530", "001"), _record("D0999", "002")]
    assignments = [_assignment("D0530.mp4", "001"), _assignment("D0999.mp4", "002")]

    report = organize_by_signer(records, assignments, cfg)

    assert report.placed == 1
    assert report.missing_source == ["D0999.mp4"]
    assert (cfg.by_signer_path / "signer_001" / "D0530.mp4").exists()

"""Tests for ``qipedc2vsl400.writer.write_outputs`` (Task 8).

These validate the design's correctness **Property 4 (determinism / idempotency,
Requirement 5.4)**: converting the same input twice yields BYTE-IDENTICAL output,
independent of the order the records are supplied in (records are sorted by
``video_id`` before serialization). They also check that Vietnamese diacritics
are preserved in UTF-8 and NOT ASCII-escaped to ``\\uXXXX`` (Requirement 5.2).

No real videos or models are needed: :class:`OutputRecord` instances are built
directly and written under a throw-away ``Config(project_root=tmp_path)``.
"""

from __future__ import annotations

import json
import random

from qipedc2vsl400.config import Config
from qipedc2vsl400.mapper import OutputRecord
from qipedc2vsl400.writer import write_outputs


def _sample_records() -> list[OutputRecord]:
    """A small set of records including Vietnamese diacritics in the gloss."""
    return [
        OutputRecord(
            video_id="D0531",
            gloss="Tỉnh",
            region="Chung",
            topic="Số",
            id="id-2",
            signer_id="002",
            fps=25.0,
            resolution=720,
            num_frames=60,
            length_seconds=2.4,
        ),
        OutputRecord(
            video_id="D0530",
            gloss="Anh ơi",
            region="Nam",
            topic="Chữ cái",
            id="id-1",
            signer_id="001",
            fps=30.0,
            resolution=1080,
            num_frames=90,
            length_seconds=3.0,
        ),
        OutputRecord(
            video_id="D0529",
            gloss="Đặng",
            region=None,
            topic=None,
            id=None,
            signer_id="unknown",
            fps=25.0,
            resolution=720,
            num_frames=50,
            length_seconds=2.0,
        ),
    ]


def test_output_written_to_front_view_json(tmp_path):
    """The output lands at ``cfg.output_view_file`` and is a valid JSON array."""
    cfg = Config(project_root=tmp_path)
    records = _sample_records()

    write_outputs(records, cfg)

    out_file = cfg.output_view_file
    assert out_file.exists()
    assert out_file.name == "front_view.json"

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    assert len(loaded) == len(records)


def test_records_sorted_by_video_id(tmp_path):
    """Emitted records are ordered ascending by ``video_id`` (Property 4)."""
    cfg = Config(project_root=tmp_path)

    write_outputs(_sample_records(), cfg)

    loaded = json.loads(cfg.output_view_file.read_text(encoding="utf-8"))
    video_ids = [obj["video_id"] for obj in loaded]
    assert video_ids == sorted(video_ids)
    assert video_ids == ["D0529", "D0530", "D0531"]


def test_determinism_rerun_is_byte_identical(tmp_path):
    """Property 4: re-running on unchanged input yields byte-identical output."""
    cfg = Config(project_root=tmp_path)
    records = _sample_records()

    write_outputs(records, cfg)
    first_bytes = cfg.output_view_file.read_bytes()

    write_outputs(records, cfg)
    second_bytes = cfg.output_view_file.read_bytes()

    assert first_bytes == second_bytes


def test_shuffled_input_produces_identical_bytes(tmp_path):
    """Property 4: input order does not affect the serialized bytes.

    Two separate project roots are used so the two runs write independent files;
    the records are supplied in different (shuffled) orders but must serialize to
    exactly the same bytes because the writer sorts by ``video_id``.
    """
    records = _sample_records()

    ordered = sorted(records, key=lambda r: r.video_id)
    shuffled = list(records)
    rng = random.Random(1234)
    rng.shuffle(shuffled)
    # Ensure the shuffle actually changed the order for a meaningful test.
    assert [r.video_id for r in shuffled] != [r.video_id for r in ordered]

    root_a = tmp_path / "run_a"
    root_b = tmp_path / "run_b"
    cfg_a = Config(project_root=root_a)
    cfg_b = Config(project_root=root_b)

    write_outputs(ordered, cfg_a)
    write_outputs(shuffled, cfg_b)

    assert cfg_a.output_view_file.read_bytes() == cfg_b.output_view_file.read_bytes()


def test_vietnamese_diacritics_not_ascii_escaped(tmp_path):
    """Requirement 5.2: Vietnamese text is UTF-8, not ``\\uXXXX``-escaped."""
    cfg = Config(project_root=tmp_path)

    write_outputs(_sample_records(), cfg)

    raw = cfg.output_view_file.read_text(encoding="utf-8")
    # No ASCII unicode escapes anywhere in the file.
    assert "\\u" not in raw
    # The actual diacritic characters survive the round-trip.
    assert "Tỉnh" in raw
    assert "Đặng" in raw
    assert "Chữ cái" in raw


def test_creates_output_directory_if_missing(tmp_path):
    """The output directory is created on demand."""
    cfg = Config(project_root=tmp_path)
    assert not cfg.output_path.exists()

    write_outputs(_sample_records(), cfg)

    assert cfg.output_path.exists()
    assert cfg.output_view_file.exists()


def test_input_list_not_mutated(tmp_path):
    """Sorting for output must not reorder the caller's list in place."""
    cfg = Config(project_root=tmp_path)
    records = _sample_records()
    original_order = [r.video_id for r in records]

    write_outputs(records, cfg)

    assert [r.video_id for r in records] == original_order

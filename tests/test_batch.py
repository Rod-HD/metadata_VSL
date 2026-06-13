"""Integration tests for batch / incremental conversion (``convert.run_batch``).

Simulates processing a large video set in batches: process batch 1, delete its
videos, then process batch 2. Asserts that the accumulated clip store still
clusters and emits **all** clips correctly — including a batch-2 clip that shares
a signer with a batch-1 clip whose video was already deleted.

Everything uses an injected stub ``probe_fn``/``embed_fn`` and dummy ``.mp4``
files under ``Config(project_root=tmp_path)`` — no real videos, models, or
network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openpyxl

from qipedc2vsl400 import convert
from qipedc2vsl400.config import Config
from qipedc2vsl400.video_probe import VideoProps

# Synthetic embeddings by filename. D0001 and D0003 are the SAME person.
_EMB = {
    "D0001.mp4": np.array([1.0, 0.0, 0.0]),
    "D0002.mp4": np.array([0.0, 1.0, 0.0]),
    "D0003.mp4": np.array([1.0, 0.02, 0.0]),  # ~same direction as D0001
    "D0004.mp4": np.array([0.0, 0.0, 1.0]),
}


def _stub_probe(_path: Path) -> VideoProps:
    return VideoProps(fps=25.0, num_frames=50, length_seconds=2.0, resolution=720)


def _stub_embed(path, _cfg):
    return _EMB[Path(path).name]


def _write_labels(cfg: Config, rows: list[tuple[str, str]]) -> None:
    """(Re)write the QIPEDC label xlsx with the given (VIDEO, LABEL) rows."""
    labels_dir = cfg.resolve("Dataset/labels")
    labels_dir.mkdir(parents=True, exist_ok=True)
    # Remove any prior label file so only the current batch's rows are present.
    for old in labels_dir.glob("*.xlsx"):
        old.unlink()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["STT", "ID", "VIDEO", "LABEL", "REGION", "TOPIC", "signer"])
    for i, (video, label) in enumerate(rows, start=1):
        ws.append([i, f"id{i}", video, label, "Chung", "Số", None])
    wb.save(labels_dir / "batch.xlsx")


def _make_videos(cfg: Config, names: list[str]) -> None:
    src = cfg.foldering_source_path
    src.mkdir(parents=True, exist_ok=True)
    for name in names:
        (src / name).write_text(f"dummy-{name}", encoding="utf-8")


def _delete_videos(cfg: Config, names: list[str]) -> None:
    src = cfg.foldering_source_path
    for name in names:
        (src / name).unlink(missing_ok=True)


def test_two_batches_cluster_old_and_new_together(tmp_path):
    cfg = Config(project_root=tmp_path)

    # --- Batch 1: D0001 (person A) + D0002 (person B) ---
    _write_labels(cfg, [("D0001.mp4", "Anh"), ("D0002.mp4", "Em")])
    _make_videos(cfg, ["D0001.mp4", "D0002.mp4"])
    r1 = convert.run_batch(cfg, probe_fn=_stub_probe, embed_fn=_stub_embed)
    assert r1.verify_report.ok, [c.detail for c in r1.verify_report.failures]
    assert {rec.video_id for rec in r1.records} == {"D0001", "D0002"}

    # Delete batch-1 videos to free space (their data persists in the store).
    _delete_videos(cfg, ["D0001.mp4", "D0002.mp4"])

    # --- Batch 2: D0003 (person A again) + D0004 (person C) ---
    _write_labels(cfg, [("D0003.mp4", "Chào"), ("D0004.mp4", "Bà")])
    _make_videos(cfg, ["D0003.mp4", "D0004.mp4"])

    embedded: list[str] = []

    def spy_embed(path, _cfg):
        embedded.append(Path(path).name)
        return _EMB[Path(path).name]

    r2 = convert.run_batch(cfg, probe_fn=_stub_probe, embed_fn=spy_embed)
    assert r2.verify_report.ok, [c.detail for c in r2.verify_report.failures]

    # Only the NEW clips were embedded in batch 2 (old ones came from the store).
    assert sorted(embedded) == ["D0003.mp4", "D0004.mp4"]

    # Output covers ALL four clips, even the deleted batch-1 ones.
    data = json.loads(cfg.output_view_file.read_text(encoding="utf-8"))
    by_id = {rec["video_id"]: rec for rec in data}
    assert set(by_id) == {"D0001", "D0002", "D0003", "D0004"}

    # D0001 (deleted) and D0003 (new) are the same person -> SAME signer_id.
    assert by_id["D0001"]["signer_id"] == by_id["D0003"]["signer_id"]
    # The three distinct people get three distinct signer ids.
    assert len({by_id[v]["signer_id"] for v in ("D0001", "D0002", "D0004")}) == 3
    # Gloss of a deleted-but-stored clip is preserved.
    assert by_id["D0001"]["gloss"] == "Anh"


def test_batch_matches_single_full_run_grouping(tmp_path):
    """Two-batch processing groups clips the same way as one full run."""
    # Full run over all four at once.
    cfg_full = Config(project_root=tmp_path / "full")
    _write_labels(
        cfg_full,
        [("D0001.mp4", "Anh"), ("D0002.mp4", "Em"),
         ("D0003.mp4", "Chào"), ("D0004.mp4", "Bà")],
    )
    _make_videos(cfg_full, ["D0001.mp4", "D0002.mp4", "D0003.mp4", "D0004.mp4"])
    full = convert.run_batch(cfg_full, probe_fn=_stub_probe, embed_fn=_stub_embed)

    # Batched run.
    cfg_b = Config(project_root=tmp_path / "batched")
    _write_labels(cfg_b, [("D0001.mp4", "Anh"), ("D0002.mp4", "Em")])
    _make_videos(cfg_b, ["D0001.mp4", "D0002.mp4"])
    convert.run_batch(cfg_b, probe_fn=_stub_probe, embed_fn=_stub_embed)
    _delete_videos(cfg_b, ["D0001.mp4", "D0002.mp4"])
    _write_labels(cfg_b, [("D0003.mp4", "Chào"), ("D0004.mp4", "Bà")])
    _make_videos(cfg_b, ["D0003.mp4", "D0004.mp4"])
    batched = convert.run_batch(cfg_b, probe_fn=_stub_probe, embed_fn=_stub_embed)

    def grouping(result):
        # Partition video_ids by signer_id -> set of frozensets (label-agnostic).
        by_signer: dict[str, set[str]] = {}
        for a in result.signer_assignments:
            by_signer.setdefault(a.signer_id, set()).add(a.video.replace(".mp4", ""))
        return {frozenset(v) for v in by_signer.values()}

    assert grouping(full) == grouping(batched)


def test_prune_missing_drops_stale_clips(tmp_path):
    cfg = Config(project_root=tmp_path)
    _write_labels(cfg, [("D0001.mp4", "Anh"), ("D0002.mp4", "Em")])
    _make_videos(cfg, ["D0001.mp4", "D0002.mp4"])
    convert.run_batch(cfg, probe_fn=_stub_probe, embed_fn=_stub_embed)

    # Next batch lists only D0003; with prune_missing, D0001/D0002 are dropped.
    _delete_videos(cfg, ["D0001.mp4", "D0002.mp4"])
    _write_labels(cfg, [("D0003.mp4", "Chào")])
    _make_videos(cfg, ["D0003.mp4"])
    result = convert.run_batch(
        cfg, probe_fn=_stub_probe, embed_fn=_stub_embed, prune_missing=True
    )

    ids = {rec.video_id for rec in result.records}
    assert ids == {"D0003"}
    assert result.verify_report.ok, [c.detail for c in result.verify_report.failures]

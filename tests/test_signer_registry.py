"""Tests for stable signer ids across runs (``signer_registry`` + batch mode).

Verifies that, with ``--stable-signers``, a person matched to an existing signer
keeps that ``signer_id`` across runs (even after their earlier videos are
deleted), new people only append new numbers, and ``--recluster`` re-seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openpyxl

from qipedc2vsl400 import convert
from qipedc2vsl400.config import Config
from qipedc2vsl400.signer_registry import SignerRegistry
from qipedc2vsl400.video_probe import VideoProps


# --- unit tests for the registry itself -------------------------------------


class _Cfg:
    signer_id_width = 3
    signer_cosine_threshold = 0.363
    signer_unknown_label = "unknown"


def _u(axis: int, n: int = 3) -> np.ndarray:
    v = np.zeros(n, dtype=np.float64)
    v[axis] = 1.0
    return v


def test_registry_matches_existing_and_appends_new():
    reg = SignerRegistry()
    cfg = _Cfg()

    sid_a, dist_a = reg.assign(_u(0), cfg)
    assert sid_a == "001" and dist_a is None  # first person, freshly minted

    # Same direction as person A -> reuses 001.
    sid_a2, dist_a2 = reg.assign(_u(0) + np.array([0.0, 0.01, 0.0]), cfg)
    assert sid_a2 == "001" and dist_a2 is not None

    # A new, orthogonal person -> appends 002.
    sid_b, _ = reg.assign(_u(1), cfg)
    assert sid_b == "002"


def test_registry_round_trip(tmp_path):
    reg = SignerRegistry()
    cfg = _Cfg()
    reg.assign(_u(0), cfg)
    reg.assign(_u(1), cfg)
    path = tmp_path / "signer_registry.json"
    reg.save(path)

    loaded = SignerRegistry.load(path)
    assert set(loaded.centroids) == {"001", "002"}
    assert loaded.next_number == 3
    # A clip near 001 still resolves to 001 after reload.
    sid, _ = loaded.assign(_u(0), cfg)
    assert sid == "001"


# --- batch integration: stable numbers across runs --------------------------

_EMB = {
    "D0001.mp4": np.array([1.0, 0.0, 0.0]),
    "D0002.mp4": np.array([0.0, 1.0, 0.0]),
    "D0003.mp4": np.array([1.0, 0.02, 0.0]),  # same person as D0001
    "D0004.mp4": np.array([0.0, 0.0, 1.0]),  # new person
}


def _stub_probe(_p: Path) -> VideoProps:
    return VideoProps(fps=25.0, num_frames=50, length_seconds=2.0, resolution=720)


def _stub_embed(path, _cfg):
    return _EMB[Path(path).name]


def _labels(cfg: Config, rows):
    d = cfg.resolve("Dataset/labels")
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.xlsx"):
        old.unlink()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["STT", "ID", "VIDEO", "LABEL", "REGION", "TOPIC", "signer"])
    for i, (v, lab) in enumerate(rows, start=1):
        ws.append([i, f"id{i}", v, lab, "Chung", "Số", None])
    wb.save(d / "batch.xlsx")


def _videos(cfg: Config, names):
    s = cfg.foldering_source_path
    s.mkdir(parents=True, exist_ok=True)
    for n in names:
        (s / n).write_text("x", encoding="utf-8")


def _del(cfg: Config, names):
    for n in names:
        (cfg.foldering_source_path / n).unlink(missing_ok=True)


def test_stable_signer_id_preserved_across_batches(tmp_path):
    cfg = Config(project_root=tmp_path)

    # Batch 1: D0001 (person A) + D0002 (person B).
    _labels(cfg, [("D0001.mp4", "Anh"), ("D0002.mp4", "Em")])
    _videos(cfg, ["D0001.mp4", "D0002.mp4"])
    r1 = convert.run_batch(
        cfg, probe_fn=_stub_probe, embed_fn=_stub_embed, stable_signers=True
    )
    assert r1.verify_report.ok
    sid_a = {a.video: a.signer_id for a in r1.signer_assignments}["D0001.mp4"]
    sid_b = {a.video: a.signer_id for a in r1.signer_assignments}["D0002.mp4"]

    # Delete batch-1 videos; add batch-2: D0003 (== person A) + D0004 (new C).
    _del(cfg, ["D0001.mp4", "D0002.mp4"])
    _labels(cfg, [("D0003.mp4", "Chào"), ("D0004.mp4", "Bà")])
    _videos(cfg, ["D0003.mp4", "D0004.mp4"])
    r2 = convert.run_batch(
        cfg, probe_fn=_stub_probe, embed_fn=_stub_embed, stable_signers=True
    )
    assert r2.verify_report.ok
    by_id = {rec.video_id: rec.signer_id for rec in r2.records}

    # KEY: D0003 (person A in batch 2) keeps person A's ORIGINAL number from run 1.
    assert by_id["D0003"] == sid_a
    # Existing numbers unchanged; the new person C gets a fresh, distinct number.
    assert by_id["D0001"] == sid_a
    assert by_id["D0002"] == sid_b
    assert by_id["D0004"] not in {sid_a, sid_b}


def test_recluster_reseeds_registry(tmp_path):
    cfg = Config(project_root=tmp_path)
    _labels(cfg, [("D0001.mp4", "Anh"), ("D0002.mp4", "Em")])
    _videos(cfg, ["D0001.mp4", "D0002.mp4"])
    convert.run_batch(
        cfg, probe_fn=_stub_probe, embed_fn=_stub_embed, stable_signers=True
    )

    # Recluster should still verify and keep two distinct signers.
    result = convert.run_batch(
        cfg, probe_fn=_stub_probe, embed_fn=_stub_embed,
        stable_signers=True, recluster=True,
    )
    assert result.verify_report.ok
    signer_ids = {rec.signer_id for rec in result.records}
    assert len(signer_ids) == 2
    # Registry file exists and lists the signers.
    reg = json.loads(cfg.signer_registry_path.read_text(encoding="utf-8"))
    assert {s["signer_id"] for s in reg["signers"]} == signer_ids

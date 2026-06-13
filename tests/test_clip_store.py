"""Tests for the persistent :class:`qipedc2vsl400.clip_store.ClipStore`."""

from __future__ import annotations

import numpy as np

from qipedc2vsl400.clip_store import ClipEntry, ClipStore


def _entry(video_id: str, has_face: bool = True) -> ClipEntry:
    return ClipEntry(
        video=f"{video_id}.mp4",
        video_id=video_id,
        gloss=f"gloss-{video_id}",
        region="Chung",
        topic="Số",
        stt="1",
        id=f"id-{video_id}",
        fps=25.0,
        resolution=720,
        num_frames=50,
        length_seconds=2.0,
        has_face=has_face,
    )


def test_round_trip_entries_and_embeddings(tmp_path):
    json_path = tmp_path / "clip_store.json"
    npz_path = tmp_path / "clip_store_embeddings.npz"

    store = ClipStore()
    store.upsert(_entry("D0001"), np.array([1.0, 0.0, 0.0]))
    store.upsert(_entry("D0002"), np.array([0.0, 1.0, 0.0]))
    store.upsert(_entry("D0003", has_face=False), None)  # no-face clip
    store.save(json_path, npz_path)

    loaded = ClipStore.load(json_path, npz_path)
    assert set(loaded.entries) == {"D0001", "D0002", "D0003"}
    assert loaded.entries["D0001"].gloss == "gloss-D0001"
    assert loaded.entries["D0003"].has_face is False
    # Embeddings only for face clips.
    assert set(loaded.embeddings) == {"D0001", "D0002"}
    assert np.allclose(loaded.embeddings["D0001"], [1.0, 0.0, 0.0])
    # The no-face clip has no embedding.
    assert "D0003" not in loaded.embeddings


def test_json_is_human_readable_utf8(tmp_path):
    json_path = tmp_path / "clip_store.json"
    npz_path = tmp_path / "clip_store_embeddings.npz"
    store = ClipStore()
    entry = _entry("D0001")
    entry.gloss = "Đặng"  # Vietnamese diacritics
    store.upsert(entry, np.array([1.0, 0.0]))
    store.save(json_path, npz_path)

    raw = json_path.read_text(encoding="utf-8")
    assert "Đặng" in raw and "\\u" not in raw  # not ASCII-escaped


def test_load_missing_returns_empty(tmp_path):
    store = ClipStore.load(tmp_path / "nope.json", tmp_path / "nope.npz")
    assert len(store) == 0
    assert store.embeddings == {}


def test_upsert_replaces_and_clears_stale_embedding(tmp_path):
    store = ClipStore()
    store.upsert(_entry("D0001"), np.array([1.0, 0.0]))
    assert "D0001" in store.embeddings
    # Re-upsert the same clip as a no-face clip -> embedding must be dropped.
    store.upsert(_entry("D0001", has_face=False), None)
    assert "D0001" not in store.embeddings
    assert store.entries["D0001"].has_face is False


def test_prune_and_prune_missing(tmp_path):
    store = ClipStore()
    for vid in ("D0001", "D0002", "D0003"):
        store.upsert(_entry(vid), np.array([1.0, 0.0]))

    removed = store.prune(["D0002"])
    assert removed == 1
    assert set(store.entries) == {"D0001", "D0003"}
    assert "D0002" not in store.embeddings

    # Keep only D0001 -> D0003 is dropped.
    removed = store.prune_missing({"D0001"})
    assert removed == 1
    assert set(store.entries) == {"D0001"}


def test_save_is_deterministic(tmp_path):
    json_a = tmp_path / "a.json"
    json_b = tmp_path / "b.json"
    npz = tmp_path / "e.npz"

    store = ClipStore()
    # Insert out of order; save must sort by video_id.
    store.upsert(_entry("D0003"), np.array([0.0, 1.0]))
    store.upsert(_entry("D0001"), np.array([1.0, 0.0]))
    store.save(json_a, npz)
    store.save(json_b, npz)

    assert json_a.read_bytes() == json_b.read_bytes()
    ids_order = [c["video_id"] for c in __import__("json").loads(
        json_a.read_text(encoding="utf-8"))["clips"]]
    assert ids_order == sorted(ids_order)

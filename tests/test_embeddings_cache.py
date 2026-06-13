"""Tests for the incremental embedding cache in ``qipedc2vsl400.signer_extractor``.

The cache lets a re-run skip the expensive face-embedding step for clips it has
already seen, while still clustering over the full (cached + new) set so the
result matches a full re-run. These tests use an injected ``embed_fn`` with
synthetic vectors (no real videos or ONNX models) and write everything under
``Config(project_root=tmp_path)`` so nothing touches the real dataset tree.

Covers Requirements 8.1-8.7 (incremental signer extraction) and asserts the
cache is a pure performance optimization: clustering output is identical to the
non-cached path, and old + new clips of the same person share one ``signer_id``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qipedc2vsl400.config import Config
from qipedc2vsl400.signer_extractor import (
    extract_signers,
    load_embeddings_cache,
    save_embeddings_cache,
)


@dataclass
class _Row:
    """Minimal ``QipedcRow``-like stand-in (only ``video`` is read)."""

    video: str


def _embed_by_name(mapping):
    """Return an ``embed_fn`` that yields a vector per row, keyed by call order.

    The vectors are supplied as a list aligned with ``valid_rows`` order; the
    resolved ``video_path`` argument is ignored (it is ``None`` on disk here).
    """
    iterator = iter(mapping)

    def embed_fn(_video_path, _cfg):
        return next(iterator)

    return embed_fn


def _proto(axis: int) -> np.ndarray:
    v = np.zeros(6, dtype=np.float64)
    v[axis] = 1.0
    return v


# --- cache round-trip -------------------------------------------------------


def test_cache_round_trip_includes_noface(tmp_path):
    """save -> load preserves face vectors and the no-face set exactly."""
    cache = tmp_path / "embeddings.npz"
    vectors = {"D0001.mp4": _proto(0), "D0002.mp4": _proto(1)}
    noface = {"D0003.mp4"}

    save_embeddings_cache(cache, vectors, noface)
    loaded_vectors, loaded_noface = load_embeddings_cache(cache)

    assert set(loaded_vectors) == set(vectors)
    for name, vec in vectors.items():
        assert np.allclose(loaded_vectors[name], vec)
    assert loaded_noface == noface


def test_load_missing_cache_returns_empty(tmp_path):
    """Loading a non-existent cache yields empty containers (first run)."""
    vectors, noface = load_embeddings_cache(tmp_path / "nope.npz")
    assert vectors == {}
    assert noface == set()


def test_cache_round_trip_empty(tmp_path):
    """A cache with no face vectors and no no-face clips round-trips cleanly."""
    cache = tmp_path / "embeddings.npz"
    save_embeddings_cache(cache, {}, set())
    vectors, noface = load_embeddings_cache(cache)
    assert vectors == {}
    assert noface == set()


# --- incremental behavior ---------------------------------------------------


def test_first_run_writes_cache_then_reused(tmp_path):
    """First run embeds all clips and writes the cache; second run reuses it."""
    cfg = Config(project_root=tmp_path)
    cache = cfg.signer_embeddings_cache_path

    rows = [_Row("D0001.mp4"), _Row("D0002.mp4"), _Row("D0003.mp4")]
    vectors = [_proto(0), _proto(0), None]  # two same-person + one no-face

    first = extract_signers(
        rows, cfg, embed_fn=_embed_by_name(vectors), embeddings_cache=cache
    )
    assert cache.is_file()

    cached_vectors, cached_noface = load_embeddings_cache(cache)
    assert set(cached_vectors) == {"D0001.mp4", "D0002.mp4"}
    assert cached_noface == {"D0003.mp4"}

    # Second run with an embed_fn that explodes if called -> proves every clip
    # was served from the cache, and the assignments are unchanged.
    def exploding_embed(_path, _cfg):
        raise AssertionError("embed_fn must not be called when all clips cached")

    second = extract_signers(
        rows, cfg, embed_fn=exploding_embed, embeddings_cache=cache
    )
    assert {a.video: a.signer_id for a in first} == {
        a.video: a.signer_id for a in second
    }


def test_only_new_clips_are_embedded_on_re_run(tmp_path):
    """Adding a clip re-embeds ONLY the new clip, not the cached ones."""
    cfg = Config(project_root=tmp_path)
    cache = cfg.signer_embeddings_cache_path

    # First run over two clips (same person).
    rows1 = [_Row("D0001.mp4"), _Row("D0002.mp4")]
    extract_signers(
        rows1, cfg, embed_fn=_embed_by_name([_proto(0), _proto(0)]),
        embeddings_cache=cache,
    )

    # Second run: same two clips + a NEW third clip (a different person). The
    # spy embed_fn records which videos it actually computes.
    embedded: list[str] = []

    def spy_embed(video_path, _cfg):
        # Only the new clip should reach here; map it to a fresh vector.
        embedded.append("called")
        return _proto(1)

    rows2 = rows1 + [_Row("D0003.mp4")]
    result = extract_signers(
        rows2, cfg, embed_fn=spy_embed, embeddings_cache=cache
    )

    # embed_fn invoked exactly once (for the single new clip).
    assert len(embedded) == 1

    by_video = {a.video: a for a in result}
    # The two cached same-person clips still share a signer.
    assert by_video["D0001.mp4"].signer_id == by_video["D0002.mp4"].signer_id
    # The new, different person is a distinct signer.
    assert by_video["D0003.mp4"].signer_id != by_video["D0001.mp4"].signer_id
    # All three are real signers (3-digit ids).
    assert all(by_video[v].has_face for v in ("D0001.mp4", "D0002.mp4", "D0003.mp4"))


def test_cached_result_matches_uncached(tmp_path):
    """Clustering with the cache equals clustering without it (same dataset)."""
    rows = [_Row("D0001.mp4"), _Row("D0002.mp4"), _Row("D0003.mp4"), _Row("D0004.mp4")]
    vectors = [_proto(0), _proto(1), _proto(0), _proto(2)]

    cfg_no_cache = Config(project_root=tmp_path / "a")
    no_cache = extract_signers(
        rows, cfg_no_cache, embed_fn=_embed_by_name(vectors)
    )

    cfg_cache = Config(project_root=tmp_path / "b")
    with_cache = extract_signers(
        rows, cfg_cache, embed_fn=_embed_by_name(vectors),
        embeddings_cache=cfg_cache.signer_embeddings_cache_path,
    )

    assert {a.video: a.signer_id for a in no_cache} == {
        a.video: a.signer_id for a in with_cache
    }

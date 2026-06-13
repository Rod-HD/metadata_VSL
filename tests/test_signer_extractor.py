"""Property and unit tests for ``qipedc2vsl400.signer_extractor.extract_signers``
(Task 6.3).

These tests exercise the signer clustering / assignment half of the module with
an **injected** ``embed_fn`` that supplies synthetic, deterministic embedding
vectors (including ``None`` to simulate clips with no detectable face). No real
videos and no ONNX face models are required: the stub ignores the resolved video
path and returns the intended vector per row in order.

All filesystem side effects (the ``signers.csv`` side-car and per-run log files)
are redirected under ``tmp_path`` via ``Config(project_root=tmp_path)`` so the
tests never touch the real ``D:`` dataset tree.

Covers the design's correctness properties:

* Property 7  - signer_id format invariant (Requirements 8.4, 6.2)
* Property 9  - signer assignment determinism (Requirement 8.4)
* Property 10 - signer accounting & no-face handling (Requirements 8.5, 8.7)

plus unit tests for within-threshold vectors sharing a ``signer_id`` and
beyond-threshold vectors receiving distinct ``signer_id`` values
(Requirements 8.1, 8.2, 8.3, 8.6).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from qipedc2vsl400.config import Config
from qipedc2vsl400.signer_extractor import SignerAssignment, extract_signers


# --- test helpers -----------------------------------------------------------

# Dimensionality of the synthetic embedding vectors. Small but >1 so cosine
# clustering has something meaningful to work with.
DIM = 6

SIGNER_ID_RE = re.compile(r"^\d{3}$")


@dataclass
class _Row:
    """Minimal ``QipedcRow``-like stand-in.

    ``extract_signers`` only reads ``row.video``; everything else is irrelevant
    for clustering, so a one-field object keeps the tests focused.
    """

    video: str | None


def _make_embed_fn(
    vectors: list[np.ndarray | None],
) -> Callable[[object, object], np.ndarray | None]:
    """Return an ``embed_fn`` yielding *vectors* in call (row) order.

    The injected function ignores the resolved ``video_path`` argument (which is
    ``None`` for every row here because no real files exist on disk) and instead
    returns the pre-computed vector for the current row. A fresh closure is built
    per call so the same vector list can drive several independent
    ``extract_signers`` invocations (e.g. the determinism property).
    """
    iterator = iter(vectors)

    def embed_fn(video_path: object, cfg: object) -> np.ndarray | None:
        return next(iterator)

    return embed_fn


def _rows_for(vectors: list[np.ndarray | None]) -> list[_Row]:
    """Build one uniquely-named row per vector (``D0000.mp4``, ``D0001.mp4`` ...)."""
    return [_Row(video=f"D{i:04d}.mp4") for i in range(len(vectors))]


# --- Hypothesis strategies --------------------------------------------------


@st.composite
def _optional_vectors(draw: st.DrawFn) -> list[np.ndarray | None]:
    """Generate a list of optional embedding vectors.

    Each entry is either ``None`` (a no-face clip) or a ``DIM``-length vector of
    finite floats with a non-negligible norm (degenerate near-zero vectors are
    filtered so the cosine clustering stays well-defined).
    """
    size = draw(st.integers(min_value=0, max_value=12))
    floats = st.floats(
        min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
    )
    result: list[np.ndarray | None] = []
    for _ in range(size):
        if draw(st.booleans()):
            result.append(None)
            continue
        components = draw(st.lists(floats, min_size=DIM, max_size=DIM))
        vector = np.asarray(components, dtype=np.float64)
        assume(np.linalg.norm(vector) > 1e-2)
        result.append(vector)
    return result


# --- Property 9: signer assignment determinism ------------------------------


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(vectors=_optional_vectors())
def test_property9_assignment_is_deterministic(vectors, tmp_path):
    """Property 9: two runs over the same embeddings give identical assignments.

    The ``video -> signer_id`` mapping and the full per-row assignment (including
    cluster numbering, which is ordered by each cluster's minimum ``VIDEO``) must
    be byte-for-byte stable across repeated invocations.

    **Validates: Requirements 8.4**
    """
    cfg = Config(project_root=tmp_path)
    rows = _rows_for(vectors)

    first = extract_signers(rows, cfg, _make_embed_fn(vectors))
    second = extract_signers(rows, cfg, _make_embed_fn(vectors))

    # Full structural equality (covers signer_id, cluster_index, distance, etc.).
    assert first == second

    # And specifically the video -> signer_id mapping is identical.
    map_first = {a.video: a.signer_id for a in first}
    map_second = {a.video: a.signer_id for a in second}
    assert map_first == map_second

    # Cluster -> id numbering is stable: ids are ordered by each cluster's
    # minimum VIDEO, so the first time a new cluster_index appears (scanning rows
    # by VIDEO order) its signer_id must be monotonically increasing.
    real = sorted(
        (a for a in first if a.has_face), key=lambda a: a.video
    )
    seen: dict[int, str] = {}
    ordering: list[str] = []
    for a in real:
        if a.cluster_index not in seen:
            seen[a.cluster_index] = a.signer_id
            ordering.append(a.signer_id)
    assert ordering == sorted(ordering)


# --- Property 10: signer accounting & no-face handling ----------------------


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(vectors=_optional_vectors())
def test_property10_full_accounting_including_unknown(vectors, tmp_path):
    """Property 10: every clip is accounted for exactly once; None -> unknown.

    * One :class:`SignerAssignment` per valid row, no drops, no duplicates.
    * Clips whose embedding is ``None`` land in the ``unknown`` bucket and keep
      ``has_face is False`` (they are never silently discarded).
    * ``sum(per-signer counts incl. unknown) == len(valid_rows)``.

    **Validates: Requirements 8.5, 8.7**
    """
    cfg = Config(project_root=tmp_path)
    rows = _rows_for(vectors)

    assignments = extract_signers(rows, cfg, _make_embed_fn(vectors))

    # Exactly one assignment per row, each original VIDEO present once.
    assert len(assignments) == len(rows)
    assert {a.video for a in assignments} == {r.video for r in rows}
    assert len(assignments) == len({a.video for a in assignments})

    # None-embedding clips go to the unknown bucket and are never dropped.
    none_videos = {
        rows[i].video for i, vec in enumerate(vectors) if vec is None
    }
    for a in assignments:
        if a.video in none_videos:
            assert a.has_face is False
            assert a.signer_id == cfg.signer_unknown_label
            assert a.cluster_index == -1
            assert a.distance is None
        else:
            assert a.has_face is True

    # Per-signer counts (the unknown bucket included) sum to the row total.
    counts = Counter(a.signer_id for a in assignments)
    assert sum(counts.values()) == len(rows)


# --- Property 7: signer_id format invariant ---------------------------------


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(vectors=_optional_vectors())
def test_property7_signer_id_format_invariant(vectors, tmp_path):
    """Property 7: real clusters use ``^\\d{3}$``; no-face clips use ``unknown``.

    No other ``signer_id`` forms may occur.

    **Validates: Requirements 8.4, 6.2**
    """
    cfg = Config(project_root=tmp_path)
    rows = _rows_for(vectors)

    assignments = extract_signers(rows, cfg, _make_embed_fn(vectors))

    for a in assignments:
        if a.has_face:
            assert SIGNER_ID_RE.match(a.signer_id), (
                f"real-cluster signer_id {a.signer_id!r} is not 3-digit zero-padded"
            )
        else:
            assert a.signer_id == cfg.signer_unknown_label

    # Exhaustively: the only forms present are 3-digit ids and the unknown label.
    for signer_id in {a.signer_id for a in assignments}:
        assert (
            SIGNER_ID_RE.match(signer_id) or signer_id == cfg.signer_unknown_label
        )


# --- unit tests: threshold-based grouping -----------------------------------


def _prototype(index: int) -> np.ndarray:
    """Return a unit basis vector along axis *index* (well-separated prototypes)."""
    vector = np.zeros(DIM, dtype=np.float64)
    vector[index] = 1.0
    return vector


def test_within_threshold_vectors_share_signer_id(tmp_path):
    """Near-identical embeddings (cosine distance < threshold) cluster together.

    Two clips whose vectors point almost the same direction must receive the
    same ``signer_id``; a third clip pointing in an orthogonal direction must
    not join them.

    **Validates: Requirements 8.1, 8.2, 8.3**
    """
    cfg = Config(project_root=tmp_path)

    near_a1 = _prototype(0)
    near_a2 = _prototype(0) + np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0])
    far_b = _prototype(1)

    vectors = [near_a1, near_a2, far_b]
    rows = _rows_for(vectors)

    assignments = extract_signers(rows, cfg, _make_embed_fn(vectors))
    by_video = {a.video: a for a in assignments}

    # The two near vectors share a signer; the orthogonal one differs.
    assert by_video["D0000.mp4"].signer_id == by_video["D0001.mp4"].signer_id
    assert by_video["D0000.mp4"].signer_id != by_video["D0002.mp4"].signer_id

    # All three are real clusters with valid 3-digit ids.
    for video in ("D0000.mp4", "D0001.mp4", "D0002.mp4"):
        assert by_video[video].has_face is True
        assert SIGNER_ID_RE.match(by_video[video].signer_id)

    # Deterministic numbering ordered by minimum VIDEO: cluster A (min D0000) is
    # "001", cluster B (min D0002) is "002".
    assert by_video["D0000.mp4"].signer_id == "001"
    assert by_video["D0002.mp4"].signer_id == "002"


def test_beyond_threshold_vectors_get_different_signer_ids(tmp_path):
    """Mutually orthogonal embeddings each form their own signer cluster.

    Three orthogonal vectors (cosine distance 1.0, far beyond the ``0.363``
    same-identity threshold) must produce three distinct ``signer_id`` values.

    **Validates: Requirements 8.1, 8.2, 8.3**
    """
    cfg = Config(project_root=tmp_path)

    vectors = [_prototype(0), _prototype(1), _prototype(2)]
    rows = _rows_for(vectors)

    assignments = extract_signers(rows, cfg, _make_embed_fn(vectors))
    signer_ids = [a.signer_id for a in assignments]

    assert len(set(signer_ids)) == 3
    assert signer_ids == ["001", "002", "003"]


def test_none_embeddings_route_to_unknown_without_models(tmp_path):
    """No-face clips (``None`` embeddings) become ``unknown`` with no models.

    Confirms the injected-``embed_fn`` path needs neither real videos nor ONNX
    models on disk: a mix of real and ``None`` embeddings yields the expected
    unknown bucket alongside normal clusters.

    **Validates: Requirements 8.5, 8.6, 8.7**
    """
    cfg = Config(project_root=tmp_path)

    vectors = [_prototype(0), None, _prototype(0)]
    rows = _rows_for(vectors)

    assignments = extract_signers(rows, cfg, _make_embed_fn(vectors))
    by_video = {a.video: a for a in assignments}

    # The two real clips cluster together; the None clip is unknown.
    assert by_video["D0000.mp4"].signer_id == by_video["D0002.mp4"].signer_id
    assert SIGNER_ID_RE.match(by_video["D0000.mp4"].signer_id)

    unknown = by_video["D0001.mp4"]
    assert unknown.signer_id == cfg.signer_unknown_label
    assert unknown.has_face is False
    assert unknown.cluster_index == -1
    assert unknown.distance is None

    # Side-car CSV was written under tmp_path (not on the real D: tree).
    assert cfg.signer_sidecar_path.is_file()
    assert tmp_path in cfg.signer_sidecar_path.parents

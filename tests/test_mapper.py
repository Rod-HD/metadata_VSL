"""Property-based tests for ``qipedc2vsl400.mapper.build_records`` (Task 7.2).

These tests drive :func:`build_records` with **synthetic** ``QipedcRow`` lists, a
**stub** ``probe_fn`` and **stub** ``signer_assignments`` so no real videos or
face models are required. ``build_records`` resolves each row's clip from disk
via ``cfg.video_search_paths()`` and only probes files that actually exist, so
to make the tests deterministic we create tiny dummy ``.mp4`` files under a
throw-away project root (``Config(project_root=root)``); rows whose video file is
absent — or whose stub probe returns ``None`` — are expected to land in
``skipped_due_to_video``.

Each Hypothesis example uses a freshly created, uniquely-named project root under
``tmp_path`` so dummy files and per-run log files never leak between examples.

Covers the design's correctness properties:

* Property 1 - count preservation (Requirements 6.1, 5.3)
* Property 2 - schema conformance / superset (Requirements 4.1, 6.2)
* Property 3 - gloss-set equality (Requirements 6.3)
* Property 5 - video_id fidelity & uniqueness (Requirements 4.2, 5.4)
* Property 6 - UTF-8 round-trip (Requirements 4.4, 5.2)
* Property 8 - length consistency (Requirements 4.3)
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qipedc2vsl400.config import Config
from qipedc2vsl400.mapper import build_records
from qipedc2vsl400.qipedc_reader import QipedcRow
from qipedc2vsl400.signer_extractor import SignerAssignment
from qipedc2vsl400.video_probe import VideoProps

SIGNER_ID_RE = re.compile(r"^\d{3}$")

# The superset keys every emitted record must expose (design Property 2).
SUPERSET_KEYS = {
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
}

# Vietnamese letters (with diacritics) used to exercise the UTF-8 round-trip
# property; no whitespace so generated glosses are always non-empty strings.
_VN = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵđ"
_VIETNAMESE_ALPHABET = _VN + _VN.upper()

# A gloss is either generated Vietnamese text or a sampled value; the sampled
# pool includes float-formatted numbers and repeats so duplicate labels occur.
_GLOSS = st.one_of(
    st.text(alphabet=_VIETNAMESE_ALPHABET, min_size=1, max_size=8),
    st.sampled_from(["1", "2", "1.0", "Anh", "Chào", "Số", "Mẹ"]),
)

# Optional QIPEDC-native attributes (region / topic / stt / id) — may be None.
_OPTIONAL = st.one_of(st.none(), st.text(alphabet=_VIETNAMESE_ALPHABET, min_size=0, max_size=5))

# Signer id choice attached to a clip: None means "no assignment supplied" (the
# mapper falls back to the unknown bucket); otherwise a 3-digit id or "unknown".
_SIGNER_CHOICE = st.one_of(
    st.none(),
    st.integers(min_value=0, max_value=999).map(lambda n: f"{n:03d}"),
    st.just("unknown"),
)

# Disposition controls whether a row is mapped or skipped. Weighted toward
# "mapped" so most examples produce records to validate.
_DISPOSITION = st.sampled_from(
    ["mapped", "mapped", "mapped", "missing_file", "probe_none"]
)


@dataclass
class _Spec:
    """A single synthetic row plus the test's intent for it."""

    video: str  # e.g. "D00042.mp4" (unique within an example)
    gloss: str
    region: str | None
    topic: str | None
    stt: str | None
    id: str | None
    fps: float
    num_frames: int
    resolution: int
    signer_choice: str | None
    disposition: str  # "mapped" | "missing_file" | "probe_none"


@st.composite
def _specs(draw: st.DrawFn) -> list[_Spec]:
    """Generate a list of synthetic row specs with unique VIDEO filenames."""
    ids = draw(
        st.lists(st.integers(min_value=0, max_value=99999), max_size=8, unique=True)
    )
    specs: list[_Spec] = []
    for n in ids:
        specs.append(
            _Spec(
                video=f"D{n:05d}.mp4",
                gloss=draw(_GLOSS),
                region=draw(_OPTIONAL),
                topic=draw(_OPTIONAL),
                stt=draw(_OPTIONAL),
                id=draw(_OPTIONAL),
                fps=draw(
                    st.floats(
                        min_value=0.01,
                        max_value=240.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                ),
                num_frames=draw(st.integers(min_value=0, max_value=500000)),
                resolution=draw(st.integers(min_value=1, max_value=4320)),
                signer_choice=draw(_SIGNER_CHOICE),
                disposition=draw(_DISPOSITION),
            )
        )
    return specs


@dataclass
class _Built:
    """Bundle of everything needed to call and check ``build_records``."""

    cfg: Config
    valid_rows: list[QipedcRow]
    probe_fn: object
    signer_assignments: list[SignerAssignment]
    expected_props: dict  # video name -> VideoProps for mapped rows
    mapped_videos: set  # VIDEO filenames expected to produce a record


def _setup(specs: list[_Spec], tmp_path: Path) -> _Built:
    """Materialise dummy videos + rows for *specs* under a fresh project root."""
    root = tmp_path / uuid.uuid4().hex
    video_dir = root / "Dataset" / "processed_videos" / "resize_720p"
    video_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(project_root=root)

    valid_rows: list[QipedcRow] = []
    signer_assignments: list[SignerAssignment] = []
    props_by_name: dict[str, VideoProps | None] = {}
    expected_props: dict[str, VideoProps] = {}
    mapped_videos: set[str] = set()

    for i, spec in enumerate(specs, start=1):
        valid_rows.append(
            QipedcRow(
                stt=spec.stt,
                id=spec.id,
                video=spec.video,
                label=spec.gloss,
                region=spec.region,
                topic=spec.topic,
                signer=None,
                source_file="synthetic.xlsx",
                row_index=i,
            )
        )

        if spec.signer_choice is not None:
            signer_assignments.append(
                SignerAssignment(
                    video=spec.video,
                    signer_id=spec.signer_choice,
                    cluster_index=0,
                    distance=None,
                    has_face=True,
                )
            )

        if spec.disposition == "missing_file":
            # No file on disk -> resolver can't find it -> skipped.
            continue

        (video_dir / spec.video).write_bytes(b"\x00")
        if spec.disposition == "probe_none":
            props_by_name[spec.video] = None  # readable file, unreadable probe
            continue

        props = VideoProps(
            fps=spec.fps,
            num_frames=spec.num_frames,
            length_seconds=round(spec.num_frames / spec.fps, 2),
            resolution=spec.resolution,
        )
        props_by_name[spec.video] = props
        expected_props[spec.video] = props
        mapped_videos.add(spec.video)

    def probe_fn(video_path: Path) -> VideoProps | None:
        return props_by_name.get(video_path.name)

    return _Built(
        cfg=cfg,
        valid_rows=valid_rows,
        probe_fn=probe_fn,
        signer_assignments=signer_assignments,
        expected_props=expected_props,
        mapped_videos=mapped_videos,
    )


_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
    ],
)


# --- Property 1: count preservation -----------------------------------------


@_SETTINGS
@given(specs=_specs())
def test_property1_count_preservation(specs, tmp_path):
    """Property 1: ``len(records) == len(valid_rows) - len(skipped)``.

    Every valid row is accounted for in exactly one of the two returned lists —
    no rows are dropped or duplicated.

    **Validates: Requirements 6.1, 5.3**
    """
    b = _setup(specs, tmp_path)
    records, skipped = build_records(
        b.valid_rows, b.probe_fn, b.signer_assignments, b.cfg
    )

    assert len(records) == len(b.valid_rows) - len(skipped)
    assert len(records) + len(skipped) == len(b.valid_rows)
    assert len(records) == len(b.mapped_videos)


# --- Property 2: schema conformance (superset) ------------------------------


@_SETTINGS
@given(specs=_specs())
def test_property2_schema_conformance(specs, tmp_path):
    """Property 2: every record exposes the full superset with correct types.

    ``video_id`` is the kept QIPEDC value (stem here), ``signer_id`` matches
    ``^\\d{3}$`` or ``unknown``, numeric fields satisfy their bounds, ``gloss``
    is a non-empty string, and the QIPEDC-only fields are present (may be None).

    **Validates: Requirements 4.1, 6.2**
    """
    b = _setup(specs, tmp_path)
    records, _ = build_records(
        b.valid_rows, b.probe_fn, b.signer_assignments, b.cfg
    )

    for rec in records:
        keys = set(asdict(rec).keys())
        assert SUPERSET_KEYS <= keys

        assert isinstance(rec.video_id, str) and len(rec.video_id) > 0
        assert rec.video_id == Path(rec.video_id).stem  # extension stripped

        assert isinstance(rec.signer_id, str)
        assert SIGNER_ID_RE.match(rec.signer_id) or rec.signer_id == "unknown"

        assert isinstance(rec.fps, float) and rec.fps > 0
        assert isinstance(rec.resolution, int) and rec.resolution > 0
        assert isinstance(rec.num_frames, int) and rec.num_frames >= 0
        assert isinstance(rec.length_seconds, float) and rec.length_seconds >= 0

        assert isinstance(rec.gloss, str) and len(rec.gloss) > 0

        # QIPEDC-only fields are present (attributes exist); may be None.
        for field in ("region", "topic", "id"):
            assert hasattr(rec, field)
            value = getattr(rec, field)
            assert value is None or isinstance(value, str)


# --- Property 3: gloss-set equality -----------------------------------------


@_SETTINGS
@given(specs=_specs())
def test_property3_gloss_set_equality(specs, tmp_path):
    """Property 3: emitted gloss set equals the mapped rows' label set.

    **Validates: Requirements 6.3**
    """
    b = _setup(specs, tmp_path)
    records, _ = build_records(
        b.valid_rows, b.probe_fn, b.signer_assignments, b.cfg
    )

    emitted = {rec.gloss for rec in records}
    expected = {
        row.label for row in b.valid_rows if row.video in b.mapped_videos
    }
    assert emitted == expected


# --- Property 5: video_id fidelity & uniqueness -----------------------------


@_SETTINGS
@given(specs=_specs())
def test_property5_video_id_fidelity_and_uniqueness(specs, tmp_path):
    """Property 5: each ``video_id`` equals its source VIDEO stem and is unique.

    **Validates: Requirements 4.2, 5.4**
    """
    b = _setup(specs, tmp_path)
    records, _ = build_records(
        b.valid_rows, b.probe_fn, b.signer_assignments, b.cfg
    )

    video_ids = [rec.video_id for rec in records]
    # Uniqueness: one record per source clip, no collisions.
    assert len(video_ids) == len(set(video_ids))

    # Fidelity: every video_id is the stem of an expected mapped VIDEO.
    expected_ids = {Path(name).stem for name in b.mapped_videos}
    assert set(video_ids) == expected_ids


# --- Property 6: UTF-8 round-trip -------------------------------------------


@_SETTINGS
@given(specs=_specs())
def test_property6_utf8_round_trip(specs, tmp_path):
    """Property 6: JSON round-trip preserves Vietnamese gloss strings exactly.

    **Validates: Requirements 4.4, 5.2**
    """
    b = _setup(specs, tmp_path)
    records, _ = build_records(
        b.valid_rows, b.probe_fn, b.signer_assignments, b.cfg
    )

    for rec in records:
        payload = asdict(rec)
        round_tripped = json.loads(json.dumps(payload, ensure_ascii=False))
        assert round_tripped["gloss"] == rec.gloss
        # Diacritics survive without ASCII escaping.
        assert json.dumps(rec.gloss, ensure_ascii=False) == json.dumps(
            rec.gloss, ensure_ascii=False
        )


# --- Property 8: length consistency -----------------------------------------


@_SETTINGS
@given(specs=_specs())
def test_property8_length_consistency(specs, tmp_path):
    """Property 8: when ``fps > 0``, ``length_seconds == round(num_frames/fps, 2)``.

    **Validates: Requirements 4.3**
    """
    b = _setup(specs, tmp_path)
    records, _ = build_records(
        b.valid_rows, b.probe_fn, b.signer_assignments, b.cfg
    )

    for rec in records:
        if rec.fps > 0:
            assert rec.length_seconds == round(rec.num_frames / rec.fps, 2)

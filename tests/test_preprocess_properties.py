"""Property-based tests for the ``qipedc_video_preprocess`` pure logic layer.

Each correctness property from the design document (design.md, section
"Correctness Properties") is implemented by exactly one Hypothesis property test
and annotated with its property number and the requirements it validates.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from qipedc_video_preprocess.config import PreprocessConfig
from qipedc_video_preprocess.discovery import VideoEntry, select_entries

# A fixed, valid project root that lives on the required ``D:`` drive. Output
# path validity is evaluated relative to this root.
PROJECT_ROOT = Path("D:/projects/metadata_VSL")
_ROOT_RESOLVED = PROJECT_ROOT.resolve()
_ROOT_STR = "D:/projects/metadata_VSL"

# Path segments that are safe on Windows: no separators, no dots, no drive
# colons, never ".." — so a relative path built from them never escapes the
# project tree on its own.
_SAFE_SEGMENT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=8,
)
_SEGMENTS = st.lists(_SAFE_SEGMENT, min_size=1, max_size=3)


@st.composite
def output_path_value(draw) -> tuple[str, bool]:
    """Generate an output-path config value together with its expected validity.

    "Valid" means the value resolves to a location that is BOTH on drive ``D:``
    AND inside the ``PROJECT_ROOT`` tree (Req 1.3/1.4).
    """
    kind = draw(
        st.sampled_from(
            ["valid_rel", "valid_abs", "wrong_drive", "outside_abs", "escape_rel"]
        )
    )
    parts = "/".join(draw(_SEGMENTS))

    if kind == "valid_rel":
        # relative path inside the tree -> joined onto project_root, on D:
        return parts, True
    if kind == "valid_abs":
        # absolute path that is explicitly under the project root, on D:
        return f"{_ROOT_STR}/{parts}", True
    if kind == "wrong_drive":
        # absolute path on the forbidden C: drive
        return f"C:/{parts}", False
    if kind == "outside_abs":
        # absolute path on D: but outside the project tree
        return f"D:/outside_root_dir/{parts}", False
    # escape_rel: relative path that climbs above the project root
    return f"../escaped_out/{parts}", False


@st.composite
def roi_value(draw) -> tuple[tuple[float, float, float, float], bool]:
    """Generate an ROI tuple and its expected validity.

    Valid iff every coordinate is in [0, 1] AND x0 < x1 AND y0 < y1 (Req 3.6).
    """
    coord = st.floats(
        min_value=-0.5, max_value=1.5, allow_nan=False, allow_infinity=False
    )
    x0 = draw(coord)
    y0 = draw(coord)
    x1 = draw(coord)
    y1 = draw(coord)
    in_range = all(0.0 <= c <= 1.0 for c in (x0, y0, x1, y1))
    valid = in_range and (x0 < x1) and (y0 < y1)
    return (x0, y0, x1, y1), valid


@st.composite
def preprocess_config_with_validity(draw) -> tuple[PreprocessConfig, bool]:
    """Build a PreprocessConfig with randomized fields plus its oracle validity."""
    split_v, split_ok = draw(output_path_value())
    labels_v, labels_ok = draw(output_path_value())
    log_v, log_ok = draw(output_path_value())

    roi, roi_ok = draw(roi_value())

    conf = draw(
        st.floats(min_value=-0.5, max_value=1.5, allow_nan=False, allow_infinity=False)
    )
    conf_ok = 0.0 <= conf <= 1.0

    interval = draw(
        st.floats(min_value=0.0, max_value=6.0, allow_nan=False, allow_infinity=False)
    )
    interval_ok = 0.1 <= interval <= 5.0

    margin = draw(st.integers(min_value=-10, max_value=70))
    margin_ok = 0 <= margin <= 60

    cfg = PreprocessConfig(
        project_root=PROJECT_ROOT,
        split_output_dir=split_v,
        new_labels_path=labels_v,
        log_dir=log_v,
        roi_top_left=roi,
        ocr_confidence_threshold=conf,
        sample_interval_seconds=interval,
        safety_margin_frames=margin,
    )
    expected_valid = (
        split_ok
        and labels_ok
        and log_ok
        and roi_ok
        and conf_ok
        and interval_ok
        and margin_ok
    )
    return cfg, expected_valid


# Feature: multi-variant-video-splitting, Property 1: Config chỉ hợp lệ khi mọi
# đường ra trên D: trong cây dự án và tham số đúng miền — validate() trả về danh
# sách rỗng KHI VÀ CHỈ KHI mọi đường dẫn đầu ra (split_output_dir,
# new_labels_path, log_dir) resolve về trong cây project_root và trên ổ D:, VÀ
# mọi tham số nằm trong miền hợp lệ (ROI mỗi tọa độ ∈ [0,1] với x0<x1, y0<y1;
# ocr_confidence_threshold ∈ [0,1]; sample_interval_seconds ∈ [0.1,5.0];
# safety_margin_frames là số nguyên ∈ [0,60]); ngược lại trả về ít nhất một lỗi.
# Validates: Requirements 1.3, 1.4, 3.6, 5.5
@settings(max_examples=300)
@given(preprocess_config_with_validity())
def test_property_1_config_validation_iff_outputs_on_d_and_params_in_domain(
    cfg_and_expected: tuple[PreprocessConfig, bool],
) -> None:
    cfg, expected_valid = cfg_and_expected

    errors = cfg.validate()

    # IFF: validate() returns an empty list exactly when the config is valid.
    assert (errors == []) == expected_valid, (
        f"expected_valid={expected_valid} but validate() returned {errors!r} "
        f"for split={cfg.split_output_dir!r}, labels={cfg.new_labels_path!r}, "
        f"log={cfg.log_dir!r}, roi={cfg.roi_top_left!r}, "
        f"conf={cfg.ocr_confidence_threshold!r}, "
        f"interval={cfg.sample_interval_seconds!r}, "
        f"margin={cfg.safety_margin_frames!r}"
    )

    # When invalid, at least one error message must be reported.
    if not expected_valid:
        assert len(errors) >= 1


# ---------------------------------------------------------------------------
# Property 2 — discovery selection (select_entries pure logic)
# ---------------------------------------------------------------------------

# Filename pieces designed to exercise the .mp4 case-insensitive filter and the
# video_id (stem) dedup logic. Bases use a tiny alphabet so the same video_id
# collides across directories, forcing the priority dedup path to be tested.
_VID_BASE = st.text(alphabet="abAB012", min_size=1, max_size=4)

# A mix of extensions: .mp4 in assorted cases (all should match) plus non-.mp4
# extensions and tricky tails that must be rejected (e.g. ".mp4.bak" -> ".bak",
# "mp4" with no dot -> no suffix).
_EXTENSION = st.sampled_from(
    [
        ".mp4",
        ".MP4",
        ".Mp4",
        ".mP4",
        ".avi",
        ".MOV",
        ".txt",
        ".mp4.bak",
        ".mp",
        "mp4",
        "",
    ]
)


@st.composite
def _filename(draw) -> str:
    """A single directory entry name = base + extension."""
    return draw(_VID_BASE) + draw(_EXTENSION)


@st.composite
def _listing_by_dir(draw) -> "dict[str, list[str]]":
    """An ordered mapping ``source_dir -> [filenames]``.

    Directory keys are emitted in a fixed, distinct, priority order
    (``dir0`` highest priority, then ``dir1`` ...). Each directory gets an
    arbitrary list of filenames with mixed-case and assorted extensions.
    """
    n_dirs = draw(st.integers(min_value=0, max_value=4))
    listing: dict[str, list[str]] = {}
    for i in range(n_dirs):
        names = draw(st.lists(_filename(), min_size=0, max_size=6))
        listing[f"dir{i}"] = names
    return listing


def _expected_selection(
    listing_by_dir: "dict[str, list[str]]",
) -> "dict[str, str]":
    """Oracle: map each kept video_id -> source_dir it must come from.

    Independently expresses the spec: keep only names whose extension is
    ``.mp4`` (case-insensitive); the video_id is the name without its
    extension; the first occurrence in directory-priority order (then file
    order) wins (Req 2.1, 2.2).
    """
    expected: dict[str, str] = {}
    for source_dir, names in listing_by_dir.items():
        for name in names:
            if Path(name).suffix.lower() != ".mp4":
                continue
            video_id = Path(name).stem
            if video_id in expected:
                continue
            expected[video_id] = source_dir
    return expected


# Feature: multi-variant-video-splitting, Property 2: Duyệt video lọc đúng, khử
# trùng theo ưu tiên, và xác định — với bất kỳ tập listing thư mục nào (mỗi thư
# mục là danh sách tên file tùy ý, chữ hoa/thường và phần mở rộng khác nhau),
# select_entries chỉ trả về file đuôi .mp4 (không phân biệt hoa/thường) trực
# tiếp trong thư mục (không đệ quy); mỗi video_id xuất hiện ĐÚNG MỘT LẦN, lấy từ
# thư mục ưu tiên cao nhất chứa nó (thứ tự khóa mapping = ưu tiên); và kết quả
# luôn được sắp xếp tăng dần theo mã ký tự của video_id (xác định, tái lập được).
# Validates: Requirements 2.1, 2.2, 2.5
@settings(max_examples=300)
@given(_listing_by_dir())
def test_property_2_discovery_selection_filters_dedups_and_orders(
    listing_by_dir: "dict[str, list[str]]",
) -> None:
    result = select_entries(listing_by_dir)
    expected = _expected_selection(listing_by_dir)

    # Every returned entry is a VideoEntry.
    assert all(isinstance(entry, VideoEntry) for entry in result)

    # (Req 2.1) Only .mp4 files survive; the exact set of video_ids matches the
    # oracle, so non-.mp4 names are excluded and no spurious ids are added.
    assert {entry.video_id for entry in result} == set(expected)

    # (Req 2.2) Each video_id appears exactly once...
    result_ids = [entry.video_id for entry in result]
    assert len(result_ids) == len(set(result_ids))

    # ...and is taken from the highest-priority directory containing it, with a
    # path built as ``Path(source_dir) / "<video_id>.<orig-ext>"`` rooted in
    # that directory (directly, never a nested subdirectory).
    for entry in result:
        assert entry.source_dir == expected[entry.video_id]
        assert entry.path.parent == Path(entry.source_dir)
        assert entry.path.stem == entry.video_id

    # (Req 2.5) The result is sorted ascending by video_id code-point order.
    assert result_ids == sorted(result_ids)

    # Determinism: re-running on identical input yields an identical ordering.
    again = select_entries(listing_by_dir)
    assert [e.video_id for e in again] == result_ids
    assert [(e.video_id, e.source_dir, str(e.path)) for e in again] == [
        (e.video_id, e.source_dir, str(e.path)) for e in result
    ]


# ---------------------------------------------------------------------------
# Property 4 — OCR interpretation (interpret_ocr pure logic)
# ---------------------------------------------------------------------------

from qipedc_video_preprocess.number_detector import interpret_ocr

# Token text pieces chosen to exercise every branch of the accept/reject rule:
# single valid digits 1..9, the rejected digit "0", whitespace-padded digits
# (interpret_ocr strips both ends), multi-character strings, non-digit
# characters, and the empty string.
_TOKEN_TEXT = st.one_of(
    st.sampled_from(["1", "2", "3", "4", "5", "6", "7", "8", "9"]),  # valid digits
    st.sampled_from([" 1", "2 ", "  7  ", "\t9\n"]),  # padded valid digits
    st.sampled_from(["0", "10", "12", "99", "1.0", " ", "", "a", "x9", "1a"]),  # rejects
)

# Confidence values across (and slightly beyond) the unit interval so the
# threshold comparison gets exercised at, above and below the boundary.
_TOKEN_CONF = st.floats(
    min_value=-0.2, max_value=1.2, allow_nan=False, allow_infinity=False
)

_TOKEN = st.tuples(_TOKEN_TEXT, _TOKEN_CONF)
_TOKENS = st.lists(_TOKEN, min_size=0, max_size=6)

# threshold is constrained to its valid domain [0, 1] (Req 3.6).
_THRESHOLD = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

_VALID_DIGIT_CHARS = set("123456789")


def _qualifying_digits(
    tokens: "list[tuple[str, float]]", threshold: float
) -> "list[int]":
    """Oracle mirroring the spec rule, independently of the implementation.

    A token qualifies iff its text (stripped) is exactly one character that is a
    digit in {1..9} AND its confidence is >= threshold (Req 3.2/3.3).
    """
    out: list[int] = []
    for text, conf in tokens:
        stripped = text.strip()
        if len(stripped) == 1 and stripped in _VALID_DIGIT_CHARS and conf >= threshold:
            out.append(int(stripped))
    return out


# Feature: multi-variant-video-splitting, Property 4: Diễn giải OCR chỉ chấp
# nhận đúng một chữ số 1–9 đủ tin cậy — với bất kỳ danh sách token OCR nào (mỗi
# token gồm chuỗi nhận dạng + độ tin cậy) và threshold ∈ [0,1], interpret_ocr
# trả về một số nguyên n KHI VÀ CHỈ KHI có đúng một token mà chuỗi của nó là một
# ký tự chữ số trong {1..9} và độ tin cậy >= threshold; trong mọi trường hợp
# khác (không token hợp lệ, nhiều token chữ số, chữ số ngoài [1,9], hoặc tin cậy
# < threshold) nó trả về None ("không có số").
# Validates: Requirements 3.2, 3.3
@settings(max_examples=300)
@given(_TOKENS, _THRESHOLD)
def test_property_4_interpret_ocr_accepts_exactly_one_confident_digit(
    tokens: "list[tuple[str, float]]",
    threshold: float,
) -> None:
    result = interpret_ocr(tokens, threshold)
    qualifying = _qualifying_digits(tokens, threshold)

    if len(qualifying) == 1:
        # IFF direction 1: exactly one confident single-digit token -> that int.
        assert result == qualifying[0], (
            f"expected {qualifying[0]} but got {result!r} for tokens={tokens!r}, "
            f"threshold={threshold!r}"
        )
        # The accepted value is always a digit in the inclusive range [1, 9].
        assert 1 <= result <= 9
    else:
        # IFF direction 2: zero or multiple qualifying tokens -> "không có số".
        assert result is None, (
            f"expected None (qualifying={qualifying!r}) but got {result!r} for "
            f"tokens={tokens!r}, threshold={threshold!r}"
        )


# ---------------------------------------------------------------------------
# Property 5 — frame sampling indices (sample_frame_indices pure logic)
# ---------------------------------------------------------------------------

from qipedc_video_preprocess.segmenter import sample_frame_indices

# fps strictly positive (Req 4.1 samples frames of a real video). A broad range
# of realistic and extreme rates exercises the step computation.
_FPS = st.floats(
    min_value=0.01, max_value=240.0, allow_nan=False, allow_infinity=False
)

# num_frames >= 1 (a video has at least one frame).
_NUM_FRAMES = st.integers(min_value=1, max_value=200_000)

# sample_interval_seconds constrained to its valid domain [0.1, 5.0] (Req 4.1).
_SAMPLE_INTERVAL = st.floats(
    min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
)


# Feature: multi-variant-video-splitting, Property 5: Chỉ số frame lấy mẫu cách
# đều và nằm trong biên — với bất kỳ fps > 0, num_frames >= 1, và
# sample_interval_seconds ∈ [0.1,5.0], tập chỉ số frame mẫu do sample_frame_indices
# sinh ra đều nằm trong [0, num_frames), tăng nghiêm ngặt, và khoảng cách giữa
# hai chỉ số liên tiếp bằng max(1, round(fps · sample_interval_seconds)).
# Validates: Requirements 4.1
# deadline=None: with num_frames up to ~200k the sampler can build very large
# index lists, whose construction time is unrelated to correctness; disabling the
# per-example deadline prevents spurious DeadlineExceeded flakes on slow machines.
@settings(max_examples=300, deadline=None)
@given(_FPS, _NUM_FRAMES, _SAMPLE_INTERVAL)
def test_property_5_sample_frame_indices_evenly_spaced_within_bounds(
    fps: float,
    num_frames: int,
    sample_interval_seconds: float,
) -> None:
    indices = sample_frame_indices(fps, num_frames, sample_interval_seconds)

    # The expected step independently mirrors the spec rule (Req 4.1).
    expected_step = max(1, round(fps * sample_interval_seconds))

    # With num_frames >= 1 the sampler always yields at least the first frame.
    assert len(indices) >= 1
    assert indices[0] == 0

    # All indices lie within [0, num_frames).
    for idx in indices:
        assert 0 <= idx < num_frames

    # Strictly increasing, and every consecutive gap equals the expected step.
    for prev, cur in zip(indices, indices[1:]):
        assert cur > prev
        assert cur - prev == expected_step


# ---------------------------------------------------------------------------
# Property 6 — sequence classification (classify_sequence pure logic)
# ---------------------------------------------------------------------------

from qipedc_video_preprocess.segmenter import classify_sequence

# A config whose project_root lives on the required ``D:`` drive; only
# ``boundary_confirm_frames`` is consulted by classify_sequence. It is rebuilt
# per example so the confirmation threshold can be randomized.
def _cfg_with_confirm(boundary_confirm_frames: int) -> PreprocessConfig:
    return PreprocessConfig(
        project_root=PROJECT_ROOT,
        boundary_confirm_frames=boundary_confirm_frames,
    )


def _classify_oracle(
    samples: "list[tuple[int, int | None]]", confirm_raw: int
) -> "tuple[str, int, tuple[tuple[int, int, int], ...]]":
    """Independent oracle for classify_sequence (Req 4.2/4.3/4.5/4.6/4.7).

    Returns ``(kind, variant_count, spans)`` where each span is a plain
    ``(variant_index, start_frame, end_frame)`` triple. Re-expresses the spec
    rule independently of the implementation:

    * single (variant_count=1) when there is no valid number;
    * multi with N variants IFF the distinct values (consecutive duplicates
      collapsed, ``None`` ignored) form ``1, 2, …, N`` (N >= 2) AND every new
      value 2..N is confirmed on >= ``confirm`` consecutive samples from its
      first appearance, giving exactly N-1 boundaries and N contiguous,
      non-overlapping spans covering ``[lo, hi]``;
    * manual_review otherwise.
    """
    confirm = max(1, int(confirm_raw))

    if not samples:
        return ("single", 1, ())

    lo = samples[0][0]
    hi = samples[-1][0]

    # Distinct-value run sequence: skip None, collapse consecutive equal values.
    runs: list[tuple[int, int, int]] = []  # (value, first_frame, first_pos)
    last: int | None = None
    for pos, (frame_index, num) in enumerate(samples):
        if num is None:
            continue
        if num != last:
            runs.append((num, frame_index, pos))
            last = num

    # No valid number at all -> single covering the whole sampled range.
    if not runs:
        return ("single", 1, ((1, lo, hi),))

    values = [value for value, _, _ in runs]
    n = len(values)
    consecutive_from_one = values == list(range(1, n + 1))

    def _run_len(pos: int, value: int) -> int:
        count = 0
        for probe in range(pos, len(samples)):
            if samples[probe][1] == value:
                count += 1
            else:
                break
        return count

    # Each new value (2..N) must be confirmed on >= confirm consecutive samples.
    confirmed = all(_run_len(pos, value) >= confirm for value, _, pos in runs[1:])

    if n >= 2 and consecutive_from_one and confirmed:
        boundary_frames = [frame_index for _, frame_index, _ in runs]
        spans: list[tuple[int, int, int]] = []
        for k in range(n):
            start = lo if k == 0 else boundary_frames[k]
            end = hi if k == n - 1 else boundary_frames[k + 1] - 1
            spans.append((k + 1, start, end))
        return ("multi", n, tuple(spans))

    return ("manual_review", 0, ())


@st.composite
def classification_case(draw) -> "tuple[PreprocessConfig, list[tuple[int, int | None]]]":
    """Generate a (config, samples) pair across single / multi / anomalous cases.

    ``samples`` is a list of ``(frame_index, number | None)`` with strictly
    increasing frame indices. Three independent constructors guarantee coverage
    of every branch: a *valid-multi* builder (clean 1..N runs each new value
    confirmed), a *single* builder (all-None / empty), and an *arbitrary*
    builder (random numbers + None) that mostly lands on manual_review.
    """
    confirm_raw = draw(st.integers(min_value=0, max_value=4))
    confirm = max(1, confirm_raw)
    cfg = _cfg_with_confirm(confirm_raw)

    def _increasing_frames(count: int) -> list[int]:
        if count <= 0:
            return []
        frames = [draw(st.integers(min_value=0, max_value=50))]
        for _ in range(count - 1):
            frames.append(frames[-1] + draw(st.integers(min_value=1, max_value=10)))
        return frames

    category = draw(st.sampled_from(["valid_multi", "single", "arbitrary"]))

    if category == "valid_multi":
        n = draw(st.integers(min_value=2, max_value=5))
        # variant 1 needs >= 1 sample (its first value is not confirmation-checked);
        # variants 2..N need >= confirm consecutive samples to be confirmed.
        block_lengths = [draw(st.integers(min_value=1, max_value=4))]
        for _ in range(n - 1):
            block_lengths.append(
                draw(st.integers(min_value=confirm, max_value=confirm + 3))
            )
        values: list[int | None] = []
        for variant_value, length in enumerate(block_lengths, start=1):
            values.extend([variant_value] * length)
    elif category == "single":
        length = draw(st.integers(min_value=0, max_value=6))
        values = [None] * length
    else:  # arbitrary -> exercises manual_review and odd edge cases
        values = draw(
            st.lists(
                st.one_of(st.none(), st.integers(min_value=1, max_value=6)),
                min_size=0,
                max_size=12,
            )
        )

    frames = _increasing_frames(len(values))
    samples = list(zip(frames, values))
    return cfg, samples


# Feature: multi-variant-video-splitting, Property 6: Phân loại chuỗi số đúng
# theo single / multi / manual_review — với bất kỳ chuỗi quan sát
# (frame_index, number|None) nào, classify_sequence cho: (a) single
# (variant_count=1) khi không có số hợp lệ; (b) multi với N cách KHI VÀ CHỈ KHI
# các giá trị phân biệt tạo dãy tăng liền kề bắt đầu từ 1 (1..N) và mỗi giá trị
# mới được xác nhận trên >= boundary_confirm_frames frame mẫu liên tiếp, với
# đúng N-1 ranh giới và các spans liên tục, không chồng lấn; (c) manual_review
# trong mọi trường hợp còn lại; observed_numbers giữ nguyên chuỗi quan sát.
# Validates: Requirements 4.2, 4.3, 4.5, 4.6, 4.7
@settings(max_examples=300)
@given(classification_case())
def test_property_6_classify_sequence_single_multi_manual_review(
    cfg_and_samples: "tuple[PreprocessConfig, list[tuple[int, int | None]]]",
) -> None:
    cfg, samples = cfg_and_samples
    result = classify_sequence(samples, cfg, video_id="VID")

    expected_kind, expected_count, expected_spans = _classify_oracle(
        samples, cfg.boundary_confirm_frames
    )

    # Classification kind and variant_count match the independent oracle.
    assert result.kind == expected_kind, (
        f"expected kind={expected_kind!r} but got {result.kind!r} for "
        f"samples={samples!r}, confirm={cfg.boundary_confirm_frames!r}"
    )
    assert result.variant_count == expected_count

    # observed_numbers always preserves the observed sequence verbatim (incl None).
    assert result.observed_numbers == tuple(num for _, num in samples)

    # video_id is threaded through unchanged.
    assert result.video_id == "VID"

    # The spans match the oracle exactly (frame positions included).
    assert (
        tuple(
            (span.variant_index, span.start_frame, span.end_frame)
            for span in result.spans
        )
        == expected_spans
    )

    if result.kind == "single":
        assert result.variant_count == 1

    elif result.kind == "multi":
        # N >= 2 variants, exactly N spans and therefore N-1 internal boundaries.
        n = result.variant_count
        assert n >= 2
        assert len(result.spans) == n
        # variant_index runs 1..N in order.
        assert [s.variant_index for s in result.spans] == list(range(1, n + 1))
        # Spans are contiguous, non-overlapping and cover the full sampled range.
        lo = samples[0][0]
        hi = samples[-1][0]
        assert result.spans[0].start_frame == lo
        assert result.spans[-1].end_frame == hi
        for prev, cur in zip(result.spans, result.spans[1:]):
            assert cur.start_frame == prev.end_frame + 1

    else:
        # manual_review carries no variants and no spans.
        assert result.kind == "manual_review"
        assert result.variant_count == 0
        assert result.spans == ()


# ---------------------------------------------------------------------------
# Property 7 — boundary refinement (refine_boundary pure-ish logic)
# ---------------------------------------------------------------------------

from qipedc_video_preprocess.number_detector import DetectionResult
from qipedc_video_preprocess.segmenter import refine_boundary


class _StepDetector:
    """A deterministic detector with a single value transition at frame ``T``.

    The frame source used in this property is the identity callable
    ``frame_index -> frame_index`` (see :func:`_boundary_case`), so the value
    passed to :meth:`detect` *is* the frame index. The detector returns the OLD
    number for any frame strictly before ``T`` and the NEW number from ``T``
    onward. This makes the detector monotonic in time, exactly matching the
    coarse-boundary contract that :func:`refine_boundary` refines (Req 4.4).
    """

    def __init__(self, transition_frame: int, old_number: int, new_number: int) -> None:
        self._t = transition_frame
        self._old = old_number
        self._new = new_number

    def detect(self, frame) -> DetectionResult:
        number = self._old if int(frame) < self._t else self._new
        return DetectionResult(number=number, confidence=1.0)


@st.composite
def _boundary_case(draw) -> "tuple[_StepDetector, int, int, int, int]":
    """Generate ``(detector, coarse_lo, coarse_hi, target_number, T)``.

    Guarantees ``coarse_lo < T <= coarse_hi`` so that ``coarse_lo`` bears the
    OLD number and ``coarse_hi`` bears the NEW (target) number, mirroring two
    adjacent coarse sample frames straddling the true transition ``T``.
    """
    t = draw(st.integers(min_value=1, max_value=5000))
    coarse_lo = draw(st.integers(min_value=0, max_value=t - 1))
    coarse_hi = draw(st.integers(min_value=t, max_value=t + 5000))

    old_number = draw(st.integers(min_value=1, max_value=9))
    new_number = draw(st.integers(min_value=1, max_value=9).filter(lambda n: n != old_number))

    detector = _StepDetector(transition_frame=t, old_number=old_number, new_number=new_number)
    return detector, coarse_lo, coarse_hi, new_number, t


# A config whose project_root lives on the required ``D:`` drive. refine_boundary
# keeps cfg in its signature for interface symmetry but does not consult it.
_BOUNDARY_CFG = PreprocessConfig(project_root=PROJECT_ROOT)


# Feature: multi-variant-video-splitting, Property 7: Tinh chỉnh ranh giới chính
# xác tới ±1 frame — với bất kỳ detector tổng hợp đơn điệu nào trả về giá trị CŨ
# ở mọi frame trước điểm chuyển T và giá trị MỚI (target_number) từ frame T trở
# đi, và với một ranh giới thô (coarse_lo, coarse_hi] thỏa coarse_lo < T <=
# coarse_hi, refine_boundary trả về chỉ số r mang target_number với sai số
# |r - T| <= 1.
# Validates: Requirements 4.4
@settings(max_examples=200)
@given(_boundary_case())
def test_property_7_refine_boundary_within_one_frame(
    case: "tuple[_StepDetector, int, int, int, int]",
) -> None:
    detector, coarse_lo, coarse_hi, target_number, t = case

    # The frame source is the identity callable: reading frame i yields value i,
    # which the detector maps to old/new number relative to T.
    frame_source = lambda index: index

    r = refine_boundary(
        detector,
        frame_source,
        coarse_lo,
        coarse_hi,
        target_number,
        _BOUNDARY_CFG,
    )

    # The refined boundary is accurate to within +/- 1 frame of the true T.
    assert abs(r - t) <= 1, (
        f"|r - T| = |{r} - {t}| = {abs(r - t)} exceeds 1 for "
        f"coarse_lo={coarse_lo}, coarse_hi={coarse_hi}, target={target_number}"
    )

    # The returned index stays inside the coarse interval (coarse_lo, coarse_hi].
    assert coarse_lo < r <= coarse_hi

    # The frame at the returned index genuinely bears the target number.
    assert detector.detect(r).number == target_number


# ---------------------------------------------------------------------------
# Property 3 — ROI cropping (crop_roi pure logic)
# ---------------------------------------------------------------------------

import numpy as np

from qipedc_video_preprocess.number_detector import crop_roi

# Frame dimensions: at least 1x1, up to a realistic 720p-ish range so the
# proportion->pixel rounding is exercised across many sizes (Req 3.1).
_FRAME_W = st.integers(min_value=1, max_value=1280)
_FRAME_H = st.integers(min_value=1, max_value=720)


@st.composite
def _valid_roi(draw) -> tuple[float, float, float, float]:
    """Generate a ROI ``(x0, y0, x1, y1)`` with 0 <= x0 < x1 <= 1, 0 <= y0 < y1 <= 1.

    Only ROIs with a strictly positive proportional area are produced so that
    ``crop_roi`` is expected to return a non-empty region (the empty/degenerate
    ROI behaviour is covered by the detector unit tests, Req 3.4).
    """
    coord = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    x0 = draw(coord)
    x1 = draw(coord.filter(lambda c: c > x0))
    y0 = draw(coord)
    y1 = draw(coord.filter(lambda c: c > y0))
    return (x0, y0, x1, y1)


# Feature: multi-variant-video-splitting, Property 3: Crop ROI ánh xạ đúng tỉ lệ
# sang pixel trong khung hình — với bất kỳ kích thước khung hình (w, h) với
# w,h >= 1 và ROI tỉ lệ (x0,y0,x1,y1) với 0 <= x0 < x1 <= 1 và 0 <= y0 < y1 <= 1,
# vùng crop trả về có biên pixel bằng (round(x0·w), round(y0·h), round(x1·w),
# round(y1·h)), nằm hoàn toàn trong [0,w]×[0,h], và chỉ phủ phần bên trái khung
# hình khi x1 <= 0.5 (không chạm góc phải).
# Validates: Requirements 3.1, 3.5
@settings(max_examples=200)
@given(_FRAME_W, _FRAME_H, _valid_roi())
def test_property_3_crop_roi_maps_proportions_to_pixels(
    width: int,
    height: int,
    roi: tuple[float, float, float, float],
) -> None:
    x0, y0, x1, y1 = roi

    # Build a frame whose pixels encode their own (row, col) so we can verify the
    # crop is positioned exactly at the expected pixel window. Channel 0 = row,
    # channel 1 = col (mod 256 to fit uint8 — enough to localise the window).
    rows = np.arange(height, dtype=np.uint16).reshape(height, 1)
    cols = np.arange(width, dtype=np.uint16).reshape(1, width)
    frame = np.zeros((height, width, 3), dtype=np.uint16)
    frame[:, :, 0] = np.broadcast_to(rows, (height, width))
    frame[:, :, 1] = np.broadcast_to(cols, (height, width))

    # Independently re-derive the expected pixel bounds from the spec rule, then
    # clamp into [0,w]x[0,h] exactly as the implementation must.
    px0 = min(max(round(x0 * width), 0), width)
    px1 = min(max(round(x1 * width), 0), width)
    py0 = min(max(round(y0 * height), 0), height)
    py1 = min(max(round(y1 * height), 0), height)

    cropped = crop_roi(frame, roi)

    # A strictly positive proportional ROI may still round to a zero-area pixel
    # window on tiny frames; in that case crop_roi returns None (Req 3.4) and the
    # proportional-mapping property has nothing to assert about a region.
    if px1 <= px0 or py1 <= py0:
        assert cropped is None
        return

    assert cropped is not None
    # The crop has exactly the expected height/width in pixels.
    assert cropped.shape[0] == py1 - py0
    assert cropped.shape[1] == px1 - px0

    # The crop is positioned at the expected pixel window: its top-left pixel
    # encodes (py0, px0) and the bounds stay within [0,w]x[0,h].
    assert 0 <= px0 < px1 <= width
    assert 0 <= py0 < py1 <= height
    assert int(cropped[0, 0, 0]) == py0
    assert int(cropped[0, 0, 1]) == px0
    assert int(cropped[-1, -1, 0]) == py1 - 1
    assert int(cropped[-1, -1, 1]) == px1 - 1

    # When x1 <= 0.5 the crop covers only the left half — its right pixel bound
    # never crosses the horizontal midpoint, so it cannot touch the top-right
    # corner where the gloss text lives (Req 3.5).
    if x1 <= 0.5:
        assert px1 <= round(0.5 * width)


# ---------------------------------------------------------------------------
# Properties 8 & 9 — safety-margin trimming and valid clip generation
# (trimmed_spans pure logic)
# ---------------------------------------------------------------------------

from qipedc_video_preprocess.segmenter import VariantSpan
from qipedc_video_preprocess.splitter import trimmed_spans


@st.composite
def _adjacent_spans(draw) -> "tuple[list[VariantSpan], int, int]":
    """Generate ``(spans, num_frames, margin)`` for a single video.

    ``spans`` are the contiguous, non-overlapping :class:`VariantSpan` of a video
    BEFORE any safety-margin trimming, exactly as ``classify_sequence`` produces
    them: the first span starts at frame 0, each next span starts at the previous
    span's ``end_frame + 1`` (its start frame is the internal boundary ``b``), and
    the last span ends at ``num_frames - 1``.
    """
    n = draw(st.integers(min_value=1, max_value=5))
    # Per-variant length in frames; each variant has at least 1 frame pre-trim.
    lengths = [draw(st.integers(min_value=1, max_value=40)) for _ in range(n)]

    spans: list[VariantSpan] = []
    cursor = 0
    for index, length in enumerate(lengths, start=1):
        start = cursor
        end = cursor + length - 1
        spans.append(VariantSpan(variant_index=index, start_frame=start, end_frame=end))
        cursor = end + 1

    num_frames = cursor  # last span ends at num_frames - 1
    margin = draw(st.integers(min_value=0, max_value=60))
    return spans, num_frames, margin


# Feature: multi-variant-video-splitting, Property 8: Biên an toàn loại trừ trọn
# vẹn vùng quanh ranh giới — với bất kỳ danh sách VariantSpan liền kề của một
# video và margin >= 0, sau trimmed_spans: với mọi ranh giới nội bộ tại frame b
# (frame đầu của cách kế tiếp), không frame nào trong [b - margin, b + margin - 1]
# thuộc bất kỳ span kết quả nào; các span kết quả không chồng lấn; frame đầu của
# cách 1 là 0 và frame cuối của cách N là num_frames - 1 (không bị trừ thêm ở hai
# đầu video).
# Validates: Requirements 5.1, 5.2
@settings(max_examples=300)
@given(_adjacent_spans())
def test_property_8_safety_margin_excludes_region_around_boundary(
    case: "tuple[list[VariantSpan], int, int]",
) -> None:
    spans, num_frames, margin = case

    # Internal boundaries are the start frames of every span except the first.
    boundaries = [s.start_frame for s in spans[1:]]

    result = trimmed_spans(spans, num_frames, margin)

    # All frames retained by any result span, collected for exclusion checks.
    retained: set[int] = set()
    for span in result:
        assert span.start_frame <= span.end_frame  # only >= 1-frame spans survive
        retained.update(range(span.start_frame, span.end_frame + 1))

    # No frame inside [b - margin, b + margin - 1] survives for any boundary b.
    for b in boundaries:
        for f in range(b - margin, b + margin):
            assert f not in retained, (
                f"frame {f} in margin band around boundary {b} (margin={margin}) "
                f"leaked into a trimmed span"
            )

    # Result spans are non-overlapping and ordered.
    ordered = sorted(result, key=lambda s: s.start_frame)
    for prev, cur in zip(ordered, ordered[1:]):
        assert prev.end_frame < cur.start_frame

    # The video ends are never trimmed: the surviving variant-1 span (if any)
    # still starts at 0, and the surviving variant-N span (if any) still ends at
    # num_frames - 1. We locate them by their preserved variant_index.
    by_index = {s.variant_index: s for s in result}
    first_index = spans[0].variant_index
    last_index = spans[-1].variant_index
    if first_index in by_index:
        assert by_index[first_index].start_frame == 0
    if last_index in by_index:
        assert by_index[last_index].end_frame == num_frames - 1


# Feature: multi-variant-video-splitting, Property 9: Chỉ sinh clip cho cách còn
# khoảng frame hợp lệ — với bất kỳ danh sách span và margin, số span trả về bằng
# đúng số span còn lại >= 1 frame sau khi trừ margin ở hai phía; mỗi span phủ đúng
# một khoảng frame liên tục; mọi span bị thu còn < 1 frame không được giữ lại.
# Validates: Requirements 5.1, 5.8
@settings(max_examples=300)
@given(_adjacent_spans())
def test_property_9_only_valid_spans_kept(
    case: "tuple[list[VariantSpan], int, int]",
) -> None:
    spans, num_frames, margin = case
    n = len(spans)
    last_index = n - 1

    # Independently re-derive, per span, its trimmed bounds and whether it keeps
    # at least one frame (mirrors the spec rule without calling the impl).
    expected_kept_indices: list[int] = []
    for i, span in enumerate(spans):
        new_start = 0 if i == 0 else span.start_frame + margin
        new_end = (num_frames - 1) if i == last_index else span.end_frame - margin
        if new_end >= new_start:
            expected_kept_indices.append(span.variant_index)

    result = trimmed_spans(spans, num_frames, margin)

    # Exactly the spans that retain >= 1 frame are produced, in order.
    assert [s.variant_index for s in result] == expected_kept_indices

    # Each produced span covers a single contiguous, valid (>= 1) frame range.
    for span in result:
        assert span.start_frame <= span.end_frame
        assert span.start_frame >= 0
        assert span.end_frame <= num_frames - 1


# ---------------------------------------------------------------------------
# Property 10 — naming convention and conflict detection (plan_outputs)
# ---------------------------------------------------------------------------

from qipedc_video_preprocess.segmenter import SegmentationResult
from qipedc_video_preprocess.splitter import plan_outputs


def _single_result(video_id: str) -> SegmentationResult:
    return SegmentationResult(
        video_id=video_id,
        kind="single",
        variant_count=1,
        spans=(VariantSpan(variant_index=1, start_frame=0, end_frame=9),),
        observed_numbers=(None,),
    )


def _multi_result(video_id: str, n: int) -> SegmentationResult:
    spans = tuple(
        VariantSpan(variant_index=k, start_frame=(k - 1) * 10, end_frame=k * 10 - 1)
        for k in range(1, n + 1)
    )
    return SegmentationResult(
        video_id=video_id,
        kind="multi",
        variant_count=n,
        spans=spans,
        observed_numbers=tuple(range(1, n + 1)),
    )


@st.composite
def _seg_results_case(draw) -> "tuple[list[SegmentationResult], bool]":
    """Generate a list of SegmentationResult plus whether a name conflict is forced.

    Two modes:
    * *clean*: every video_id is distinct and uses a reserved namespace, so single
      ``<id>.mp4`` and multi ``<id>_c<k>.mp4`` names can never collide across
      sources -> no conflict expected.
    * *collision*: a single video named e.g. ``"W7_c1"`` and a 2-variant multi
      video named ``"W7"`` both lay claim to ``W7_c1.mp4`` from DIFFERENT
      sources -> exactly one conflict expected.
    """
    mode = draw(st.sampled_from(["clean", "collision"]))

    if mode == "collision":
        # multi "W7" -> {W7_c1.mp4, W7_c2.mp4}; single "W7_c1" -> W7_c1.mp4.
        results = [
            _multi_result("W7", 2),
            _single_result("W7_c1"),
        ]
        # Optionally pad with clean, distinct videos that never collide.
        for i in range(draw(st.integers(min_value=0, max_value=3))):
            results.append(_single_result(f"CLEAN{i}"))
        return results, True

    count = draw(st.integers(min_value=0, max_value=6))
    results = []
    for i in range(count):
        vid = f"V{i}"  # distinct, reserved namespace -> no cross-source collision
        if draw(st.booleans()):
            results.append(_single_result(vid))
        else:
            results.append(_multi_result(vid, draw(st.integers(min_value=2, max_value=4))))
    return results, False


# Feature: multi-variant-video-splitting, Property 10: Quy ước đặt tên nhất quán
# và không trùng tên — với tập SegmentationResult của các video_id phân biệt,
# plan_outputs cho: mỗi Video_Một_Cách đúng một tên <video_id>.mp4 (không hậu tố);
# mỗi Video_Nhiều_Cách N cách (N >= 2) tập tên đúng bằng {<id>_c1.mp4..<id>_cN.mp4}
# với k chạy 1..N không sót/lặp và clip thứ k ứng với cách thứ k; toàn bộ tên đầu
# ra trong một lần chạy là duy nhất; mọi cặp clip từ nguồn KHÁC NHAU trùng tên
# được báo cáo là Conflict + manual_review thay vì ghi đè lẫn nhau.
# Validates: Requirements 5.4, 6.1, 6.2, 6.3, 6.5
@settings(max_examples=200)
@given(_seg_results_case())
def test_property_10_naming_convention_and_conflicts(
    case: "tuple[list[SegmentationResult], bool]",
) -> None:
    results, conflict_expected = case
    clips, conflicts = plan_outputs(results)

    clip_names = [c.out_filename for c in clips]
    conflict_names = {c.out_filename for c in conflicts}

    # All kept output names are unique within the run.
    assert len(clip_names) == len(set(clip_names))

    # Group kept clips by their source video_id to check per-video naming.
    by_source: dict[str, list] = {}
    for clip in clips:
        by_source.setdefault(clip.video_id, []).append(clip)

    for result in results:
        produced = [c.out_filename for c in by_source.get(result.video_id, [])]
        # Names that this video WOULD claim, regardless of conflict outcome.
        if result.kind == "single":
            wanted = [f"{result.video_id}.mp4"]
        else:  # multi
            wanted = [f"{result.video_id}_c{k}.mp4" for k in range(1, result.variant_count + 1)]

        # Every produced name for this source is one it legitimately claims, and
        # carries no zero-padding (c1, c2, ... not c01).
        for name in produced:
            assert name in wanted

        # Names dropped from output are exactly those flagged as conflicts.
        for name in wanted:
            if name in conflict_names:
                assert name not in produced
            else:
                assert name in produced

    if conflict_expected:
        # A genuine cross-source collision is reported, not silently overwritten.
        assert conflicts
        for conflict in conflicts:
            assert len(conflict.video_ids) >= 2
            assert conflict.out_filename not in clip_names
    else:
        # Distinct video_ids in a reserved namespace never collide.
        assert conflicts == []


# ---------------------------------------------------------------------------
# Properties 11 & 12 — new label row construction & STT reassignment
# (build_new_rows pure logic)
# ---------------------------------------------------------------------------

from qipedc_video_preprocess.label_writer import (
    COLUMN_HEADERS,
    LabelRow,
    build_new_rows,
)
from qipedc_video_preprocess.splitter import OutputClip

# Text values for label cells: includes Vietnamese diacritics and the empty
# string so inheritance is exercised across realistic gloss values.
_LABEL_TEXT = st.one_of(
    st.none(),
    st.sampled_from(["bắt tay", "bút bi", "1", "1.0", "xin chào", "", "REGION-A"]),
)


@st.composite
def _label_build_case(draw):
    """Generate (source_rows, output_clips, discovered_ids) plus an oracle map.

    A pool of distinct video_ids is split into *discovered* and *undiscovered*.
    Source rows are built for a random subset; output clips reference random
    video_ids (some discovered, some not). The oracle is the per-video_id source
    row used for inheritance.
    """
    pool = [f"W{n:03d}" for n in range(draw(st.integers(min_value=1, max_value=8)))]
    discovered = set(
        draw(st.lists(st.sampled_from(pool), min_size=0, max_size=len(pool), unique=True))
    )

    # Build at most one source row per video_id (build_new_rows keeps the first).
    source_ids = draw(
        st.lists(st.sampled_from(pool), min_size=0, max_size=len(pool), unique=True)
    )
    source_rows = []
    source_by_id = {}
    for vid in source_ids:
        row = LabelRow(
            stt=draw(st.one_of(st.none(), st.text(max_size=4))),
            id=draw(st.one_of(st.none(), st.text(max_size=6))),
            video=f"{vid}.mp4",
            label=draw(_LABEL_TEXT),
            region=draw(_LABEL_TEXT),
            topic=draw(_LABEL_TEXT),
            signer=draw(st.one_of(st.none(), st.text(max_size=6))),
        )
        source_rows.append(row)
        source_by_id[vid] = row

    # Output clips reference random video_ids; both single and multi naming.
    clips = []
    n_clips = draw(st.integers(min_value=0, max_value=10))
    for i in range(n_clips):
        vid = draw(st.sampled_from(pool))
        is_multi = draw(st.booleans())
        if is_multi:
            k = draw(st.integers(min_value=1, max_value=3))
            clips.append(
                OutputClip(
                    video_id=vid,
                    variant_index=k,
                    out_filename=f"{vid}_c{k}.mp4",
                    start_frame=0,
                    end_frame=9,
                )
            )
        else:
            clips.append(
                OutputClip(
                    video_id=vid,
                    variant_index=None,
                    out_filename=f"{vid}.mp4",
                    start_frame=0,
                    end_frame=9,
                )
            )

    return source_rows, clips, discovered, source_by_id


class _NullLogger:
    """A logger stub that swallows warnings (build_new_rows logging is incidental)."""

    def warning(self, *args, **kwargs) -> None:
        pass


# Feature: multi-variant-video-splitting, Property 11: Dựng dòng nhãn kế thừa
# đúng và lọc video thiếu — với bất kỳ tập LabelRow nguồn, tập OutputClip, và tập
# video_id đã duyệt, build_new_rows sinh ĐÚNG MỘT dòng cho mỗi OutputClip có
# video_id thuộc tập đã duyệt, trong đó cột VIDEO bằng out_filename và các cột ID,
# LABEL, REGION, TOPIC, signer giữ nguyên giá trị của dòng nguồn cùng video_id;
# mọi OutputClip tham chiếu video_id không thuộc tập đã duyệt đều bị loại.
# Validates: Requirements 7.1, 7.2, 7.5
@settings(max_examples=300)
@given(_label_build_case())
def test_property_11_build_new_rows_inheritance_and_filtering(case) -> None:
    source_rows, clips, discovered, source_by_id = case

    result = build_new_rows(source_rows, clips, discovered, logger=_NullLogger())

    # Exactly the clips whose video_id is discovered produce a row, in order.
    kept_clips = [c for c in clips if c.video_id in discovered]
    assert len(result) == len(kept_clips)

    for row, clip in zip(result, kept_clips):
        # VIDEO column is the clip's output filename (Req 7.1/7.2).
        assert row.video == clip.out_filename

        # Inherited columns equal the source row for the same video_id, or None
        # when there is no source row for that video.
        src = source_by_id.get(clip.video_id)
        if src is None:
            assert row.id is None
            assert row.label is None
            assert row.region is None
            assert row.topic is None
            assert row.signer is None
        else:
            assert row.id == src.id
            assert row.label == src.label
            assert row.region == src.region
            assert row.topic == src.topic
            assert row.signer == src.signer


# Feature: multi-variant-video-splitting, Property 12: STT được đánh lại liên tục
# và duy nhất — với bất kỳ tập dòng nhãn đầu ra gồm M dòng, cột STT sau khi gán
# lại đúng bằng hoán vị của 1..M (tăng dần, duy nhất, không khoảng trống), và cấu
# trúc cột giữ nguyên thứ tự STT, ID, VIDEO, LABEL, REGION, TOPIC, signer.
# Validates: Requirements 7.3
@settings(max_examples=300)
@given(_label_build_case())
def test_property_12_stt_reassigned_unique_and_contiguous(case) -> None:
    source_rows, clips, discovered, _ = case

    result = build_new_rows(source_rows, clips, discovered, logger=_NullLogger())
    m = len(result)

    # STT values are exactly "1".."M" in order: contiguous, unique, gap-free.
    stt_values = [row.stt for row in result]
    assert stt_values == [str(i) for i in range(1, m + 1)]

    # Column structure is preserved (the canonical header order, Req 7.3).
    assert COLUMN_HEADERS == ("STT", "ID", "VIDEO", "LABEL", "REGION", "TOPIC", "signer")
    # Each LabelRow exposes the fields in that order.
    for row in result:
        assert list(row.__dataclass_fields__) == [
            "stt",
            "id",
            "video",
            "label",
            "region",
            "topic",
            "signer",
        ]


# ---------------------------------------------------------------------------
# Property 13 — label round-trip for single-variant videos
# (write_new_labels + read_source_labels)
# ---------------------------------------------------------------------------

import tempfile

from qipedc_video_preprocess.label_writer import read_source_labels, write_new_labels


@st.composite
def _single_variant_label_table(draw):
    """Generate a valid label table of ONLY single-variant videos (distinct ids).

    Each row has a non-empty VIDEO ``<id>.mp4`` and arbitrary other columns,
    including Vietnamese diacritics and float-like number glosses.
    """
    n = draw(st.integers(min_value=0, max_value=8))
    rows = []
    for i in range(n):
        vid = f"V{i:03d}"
        rows.append(
            LabelRow(
                stt=str(i + 1),
                id=draw(st.sampled_from(["D0530", "W00202", "", "ID-7"])),
                video=f"{vid}.mp4",
                label=draw(st.sampled_from(["bắt tay", "bút bi", "1", "xin chào"])),
                region=draw(st.sampled_from(["Bắc", "Nam", "", "Trung"])),
                topic=draw(st.sampled_from(["chào hỏi", "số đếm", ""])),
                signer=draw(st.sampled_from(["s1", "s2", ""])),
            )
        )
    return rows


def _norm(value):
    """Treat None and empty-string as equivalent for round-trip comparison.

    write/read collapses blank cells to None, so a source "" returns as None.
    """
    return None if (value is None or value == "") else value


# Feature: multi-variant-video-splitting, Property 13: Round-trip bảng nhãn cho
# video một cách — với bất kỳ bảng nhãn hợp lệ chỉ gồm Video_Một_Cách, việc ghi
# Bảng_Nhãn_Mới rồi đọc lại tạo ra tập dòng tương đương với nguồn trên các cột ID,
# VIDEO, LABEL, REGION, TOPIC, signer (bỏ qua khác biệt về thứ tự dòng và STT).
# Validates: Requirements 7.6
@settings(max_examples=60, deadline=None)
@given(_single_variant_label_table())
def test_property_13_label_round_trip_single_variant(rows) -> None:
    with tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()))) as tmp:
        # A config rooted at a temp dir; new_labels_path lives under it. (The
        # round-trip property is about read/write equivalence, not the D: drive
        # placement validated separately by Property 1.)
        cfg = PreprocessConfig(
            project_root=Path(tmp),
            new_labels_path="out/labels_split.xlsx",
            labels_glob="out/*.xlsx",
        )

        out_path = write_new_labels(rows, cfg, logger=_NullLogger())
        assert out_path.exists()

        read_back = read_source_labels(cfg)

    def _key(row):
        # Map each normalized field to a sortable string ("" for None) so tuples
        # of mixed None/str compare cleanly; equivalence semantics are unchanged
        # because _norm already collapsed "" and None together.
        return tuple(
            "" if v is None else v
            for v in (
                _norm(row.id),
                _norm(row.video),
                _norm(row.label),
                _norm(row.region),
                _norm(row.topic),
                _norm(row.signer),
            )
        )

    expected = sorted(_key(r) for r in rows)
    actual = sorted(_key(r) for r in read_back)
    assert actual == expected


# ---------------------------------------------------------------------------
# Property 14 — run report consistency (build_run_report pure logic)
# ---------------------------------------------------------------------------

from qipedc_video_preprocess.run_logger import build_run_report


@st.composite
def _seg_results_for_report(draw):
    """Generate (seg_results, skipped) covering single / multi / manual_review."""
    results = []
    n = draw(st.integers(min_value=0, max_value=12))
    for i in range(n):
        kind = draw(st.sampled_from(["single", "multi", "manual_review"]))
        vid = f"R{i:03d}"
        if kind == "single":
            results.append(_single_result(vid))
        elif kind == "multi":
            k = draw(st.integers(min_value=2, max_value=4))
            results.append(_multi_result(vid, k))
        else:
            results.append(
                SegmentationResult(
                    video_id=vid,
                    kind="manual_review",
                    variant_count=0,
                    spans=(),
                    observed_numbers=(1, 5),
                )
            )
    skipped = draw(st.integers(min_value=0, max_value=5))
    return results, skipped


# Feature: multi-variant-video-splitting, Property 14: Báo cáo tóm tắt nhất quán
# và không âm — với bất kỳ tập SegmentationResult của một lần chạy, RunReport có
# mọi số đếm là số nguyên >= 0, với total_discovered = single_variant +
# multi_variant + skipped, manual_review <= total_discovered, và
# sub_clips_generated bằng tổng số span hợp lệ trên toàn bộ video nhiều cách.
# Validates: Requirements 8.2
@settings(max_examples=300)
@given(_seg_results_for_report())
def test_property_14_run_report_consistent_and_nonnegative(case) -> None:
    results, skipped = case
    report = build_run_report(results, skipped=skipped)

    # Every count is a non-negative integer.
    for value in (
        report.total_discovered,
        report.single_variant,
        report.multi_variant,
        report.sub_clips_generated,
        report.skipped,
        report.manual_review,
    ):
        assert isinstance(value, int)
        assert value >= 0

    # total_discovered = single + multi + skipped (manual_review-kind folded into
    # skipped because such videos produce no sub-clips).
    assert report.total_discovered == (
        report.single_variant + report.multi_variant + report.skipped
    )

    # manual_review never exceeds the number of discovered videos.
    assert report.manual_review <= report.total_discovered

    # sub_clips_generated equals the total number of spans across multi videos.
    expected_sub_clips = sum(
        len(r.spans) for r in results if r.kind == "multi"
    )
    assert report.sub_clips_generated == expected_sub_clips

    # Independent re-count of single / multi classifications.
    assert report.single_variant == sum(1 for r in results if r.kind == "single")
    assert report.multi_variant == sum(1 for r in results if r.kind == "multi")


# ---------------------------------------------------------------------------
# Property 15 — deterministic, local output (idempotence)
# (plan_outputs + build_new_rows pure composition)
# ---------------------------------------------------------------------------


@st.composite
def _idempotence_case(draw):
    """Generate (source_rows, seg_results_S, seg_results_Sprime) with S ⊆ S'.

    A pool of distinct video_ids is partitioned: ``S`` is a subset of ``S'``.
    Each video_id maps to a fixed SegmentationResult (single or multi) so that
    processing the same id in S or S' yields identical planning input.
    """
    pool = [f"V{i:03d}" for i in range(draw(st.integers(min_value=1, max_value=8)))]

    # Fixed classification per video_id (deterministic function of the source).
    seg_for_id = {}
    source_rows = []
    for vid in pool:
        if draw(st.booleans()):
            seg_for_id[vid] = _single_result(vid)
        else:
            seg_for_id[vid] = _multi_result(vid, draw(st.integers(min_value=2, max_value=4)))
        source_rows.append(
            LabelRow(
                stt=None,
                id=draw(st.sampled_from(["D1", "D2", "", "ID-9"])),
                video=f"{vid}.mp4",
                label=draw(st.sampled_from(["bắt tay", "1", "xin chào"])),
                region=draw(st.sampled_from(["Bắc", "Nam", ""])),
                topic=draw(st.sampled_from(["chào", ""])),
                signer=draw(st.sampled_from(["s1", "s2"])),
            )
        )

    # S' = the whole pool; S = a subset.
    s_prime_ids = list(pool)
    s_ids = draw(st.lists(st.sampled_from(pool), min_size=0, max_size=len(pool), unique=True))

    seg_S = [seg_for_id[v] for v in s_ids]
    seg_Sprime = [seg_for_id[v] for v in s_prime_ids]
    return source_rows, seg_S, set(s_ids), seg_Sprime, set(s_prime_ids)


def _outputs(source_rows, seg_results, discovered_ids):
    """Run the pure planning composition and return (clip_names, label_tuples)."""
    clips, _conflicts = plan_outputs(seg_results)
    rows = build_new_rows(source_rows, clips, discovered_ids, logger=_NullLogger())
    clip_names = sorted(c.out_filename for c in clips)
    # Label tuples ignore STT and row order (Req 9.2).
    label_tuples = sorted(
        (
            "" if r.id is None else r.id,
            "" if r.video is None else r.video,
            "" if r.label is None else r.label,
            "" if r.region is None else r.region,
            "" if r.topic is None else r.topic,
            "" if r.signer is None else r.signer,
        )
        for r in rows
    )
    return clip_names, label_tuples


# Feature: multi-variant-video-splitting, Property 15: Đầu ra xác định và cục bộ
# (idempotence) — với bất kỳ tập video nguồn, chạy plan_outputs + build_new_rows
# hai lần cho cùng input tạo ra cùng tập tên file clip con và cùng tập dòng nhãn
# (trên ID, VIDEO, LABEL, REGION, TOPIC, signer, bỏ qua STT/thứ tự); và với mọi
# tập nguồn S ⊆ S', đầu ra sinh cho mỗi video_id trong S là GIỐNG HỆT khi xử lý S
# hay S' (thêm video mới không làm đổi hay nhân bản đầu ra của video cũ).
# Validates: Requirements 9.1, 9.2, 9.4
@settings(max_examples=200)
@given(_idempotence_case())
def test_property_15_deterministic_local_output(case) -> None:
    source_rows, seg_S, s_ids, seg_Sprime, sprime_ids = case

    # (a) Determinism: running twice on the same input gives identical output.
    names_1, labels_1 = _outputs(source_rows, seg_S, s_ids)
    names_2, labels_2 = _outputs(source_rows, seg_S, s_ids)
    assert names_1 == names_2
    assert labels_1 == labels_2

    # (b) Locality: for every video_id in S, its outputs are identical whether
    # processed within S or within the superset S'. We compare the per-id slice.
    _names_S, labels_S = _outputs(source_rows, seg_S, s_ids)
    _names_Sprime, labels_Sprime = _outputs(source_rows, seg_Sprime, sprime_ids)

    def _slice_for_ids(label_tuples, ids):
        # VIDEO column is "<id>.mp4" or "<id>_c<k>.mp4"; keep rows whose id is in S.
        kept = []
        for t in label_tuples:
            video = t[1]
            stem = video[:-4] if video.endswith(".mp4") else video
            base = stem.split("_c")[0]
            if base in ids:
                kept.append(t)
        return sorted(kept)

    assert _slice_for_ids(labels_S, s_ids) == _slice_for_ids(labels_Sprime, s_ids)

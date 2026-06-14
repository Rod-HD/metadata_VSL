"""Bộ_Phân_Đoạn — xác định số cách và ranh giới giữa các cách.

Module này định nghĩa các kiểu dữ liệu kết quả phân đoạn (:class:`VariantSpan`,
:class:`SegmentationResult`) và bộ sinh chỉ số frame mẫu thuần
(:func:`sample_frame_indices`).

Phần logic phân loại chuỗi số (``classify_sequence``), tinh chỉnh ranh giới
(``refine_boundary``) và điều phối (``segment_video``) sẽ được hiện thực ở các
task sau (7.3 / 7.5). Xem design.md — Requirement 4.

Quy ước frame:

* Frame được đánh chỉ số **0-based**.
* :class:`VariantSpan` dùng khoảng **đóng** ``[start_frame, end_frame]``
  (``end_frame`` inclusive).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - chỉ phục vụ type hint, tránh import vòng
    from .config import PreprocessConfig
    from .discovery import VideoEntry
    from .number_detector import NumberDetector
    from .video_probe import VideoProps


@dataclass(frozen=True)
class VariantSpan:
    """Khoảng frame của một cách (variant) trong video.

    Khoảng là **đóng** ``[start_frame, end_frame]`` (cả hai đầu inclusive), frame
    đánh chỉ số 0-based. ``start_frame``/``end_frame`` là vị trí ranh giới đã được
    tinh chỉnh và **chưa** trừ ``safety_margin`` (việc trừ biên an toàn do
    ``splitter.trimmed_spans`` đảm nhiệm).

    Attributes:
        variant_index: Số thứ tự cách, 1-based, theo thứ tự thời gian.
        start_frame: Frame bắt đầu (0-based, inclusive).
        end_frame: Frame kết thúc (0-based, inclusive).
    """

    variant_index: int
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class SegmentationResult:
    """Kết quả phân đoạn của một video.

    Attributes:
        video_id: ``video_id`` của video nguồn (stem), ví dụ ``"W00202"``.
        kind: Phân loại video — một trong ``"single"``, ``"multi"`` hoặc
            ``"manual_review"``.
        variant_count: Số cách — ``1`` nếu ``single``; ``N`` nếu ``multi``;
            ``0`` nếu ``manual_review``.
        spans: Tuple các :class:`VariantSpan` liên tục không chồng lấn theo thứ tự
            thời gian (rỗng khi ``manual_review``).
        observed_numbers: Chuỗi con số quan sát được trên các frame mẫu (mỗi phần
            tử là ``int`` hoặc ``None`` cho "không có số"), giữ lại để ghi log.
    """

    video_id: str
    kind: str
    variant_count: int
    spans: tuple[VariantSpan, ...]
    observed_numbers: tuple[int | None, ...]


def sample_frame_indices(
    fps: float, num_frames: int, sample_interval_seconds: float
) -> list[int]:
    """Sinh các chỉ số frame mẫu cách đều trong ``[0, num_frames)``.

    Đây là **logic thuần** (không chạm I/O) để phục vụ property-test. Hiện thực
    Req 4.1: lấy mẫu frame theo ``sample_interval_seconds``.

    Bước lấy mẫu được tính bằng::

        step = max(1, round(fps * sample_interval_seconds))

    Các chỉ số bắt đầu từ ``0`` và tăng dần theo ``step`` trong khi còn nhỏ hơn
    ``num_frames``. Vì ``step >= 1`` nên dãy chỉ số luôn **tăng nghiêm ngặt** và
    nằm trọn trong khoảng ``[0, num_frames)``.

    Args:
        fps: Tốc độ khung hình của video (khung/giây).
        num_frames: Tổng số frame của video.
        sample_interval_seconds: Khoảng thời gian lấy mẫu (giây).

    Returns:
        Danh sách chỉ số frame (0-based) tăng nghiêm ngặt. Trả về danh sách rỗng
        khi ``num_frames <= 0``.
    """
    if num_frames <= 0:
        return []

    step = max(1, round(fps * sample_interval_seconds))
    return list(range(0, num_frames, step))


# Loại đối tượng mẫu: cặp (chỉ_số_frame, số|None). ``None`` nghĩa là "không có số"
# trên frame mẫu đó (frame rỗng, OCR không đọc được, hoặc detector báo lỗi — Req 4.8).
Sample = tuple[int, "int | None"]


def classify_sequence(
    samples: list[Sample],
    cfg: "PreprocessConfig",
    video_id: str = "",
) -> SegmentationResult:
    """Phân loại một video thành single / multi / manual_review (LOGIC THUẦN).

    Hàm chỉ làm việc trên chuỗi quan sát ``samples`` — danh sách cặp
    ``(frame_index, number | None)`` theo thứ tự thời gian (frame_index tăng dần)
    — nên có thể property-test mà không cần video thật. Đây là kết quả **thô**:
    các ranh giới đặt tại frame mẫu nơi giá trị số mới xuất hiện lần đầu; bước
    tinh chỉnh ±1 frame do :func:`refine_boundary` (task 7.5) đảm nhiệm.

    Quy tắc phân loại (Req 4.2/4.3/4.5/4.6/4.7):

    * **single** (``variant_count = 1``): không có frame mẫu nào mang số hợp lệ
      (mọi giá trị là ``None``) — Req 4.5. ``spans`` gồm đúng một khoảng phủ toàn
      bộ phạm vi đã lấy mẫu.
    * **multi** (``variant_count = N``, ``N >= 2``): các giá trị số phân biệt theo
      thứ tự thời gian tạo thành dãy tăng-liền-kề bắt đầu từ 1 (``1, 2, …, N``),
      **và** mỗi giá trị mới (2..N) được xác nhận trên ``>= cfg.boundary_confirm_frames``
      frame mẫu **liên tiếp** kể từ lần xuất hiện đầu — Req 4.2/4.6. Kết quả có
      đúng ``N - 1`` ranh giới; ``spans`` là ``N`` khoảng liên tục, không chồng
      lấn, phủ kín phạm vi đã lấy mẫu. Ranh giới giữa cách ``k`` và ``k+1`` nằm
      tại frame mẫu **đầu tiên** mang giá trị ``k+1``: cách ``k`` kết thúc ngay
      trước frame đó, cách ``k+1`` bắt đầu tại frame đó.
    * **manual_review** (``variant_count = 0``, ``spans`` rỗng): mọi trường hợp còn
      lại — không bắt đầu từ 1, nhảy không liền kề (vd 2→5), đảo thứ tự, thiếu số,
      chỉ có đúng một giá trị (vd chỉ thấy "1" mà không có "2"), hoặc một bước
      chuyển không được xác nhận đủ số frame liên tiếp — Req 4.3/4.7.

    ``observed_numbers`` luôn giữ **nguyên** chuỗi giá trị quan sát (kể cả
    ``None``) theo đúng thứ tự để ghi log (Req 4.7/8.3).

    Args:
        samples: Danh sách cặp ``(frame_index, number | None)`` theo thứ tự thời
            gian. ``frame_index`` là chỉ số frame 0-based; ``number`` là số nguyên
            ``>= 1`` đã đạt Ngưỡng_Tin_Cậy, hoặc ``None`` cho "không có số".
        cfg: :class:`~qipedc_video_preprocess.config.PreprocessConfig`; chỉ dùng
            ``boundary_confirm_frames``.
        video_id: ``video_id`` của video nguồn (mặc định rỗng; ``segment_video``
            sẽ điền giá trị thật ở task 7.5).

    Returns:
        :class:`SegmentationResult` thô.
    """
    observed_numbers: tuple[int | None, ...] = tuple(num for _, num in samples)
    confirm = max(1, int(cfg.boundary_confirm_frames))

    # Không có frame mẫu nào → không có gì để phân loại; coi như single rỗng.
    if not samples:
        return SegmentationResult(
            video_id=video_id,
            kind="single",
            variant_count=1,
            spans=(),
            observed_numbers=observed_numbers,
        )

    lo = samples[0][0]
    hi = samples[-1][0]

    # Chuỗi các lần "đổi giá trị" theo thời gian, bỏ qua None (None = không quan
    # sát, không phá vỡ tính liên tục của một giá trị). Mỗi phần tử ghi lại
    # (giá_trị, vị_trí_trong_samples, frame_index) tại lần xuất hiện đầu của giá
    # trị đó trong một block liên tục.
    transitions: list[tuple[int, int, int]] = []
    last_value: int | None = None
    for pos, (frame_index, num) in enumerate(samples):
        if num is None:
            continue
        if num != last_value:
            transitions.append((num, pos, frame_index))
            last_value = num

    # Req 4.5: không có số hợp lệ nào → Video_Một_Cách.
    if not transitions:
        return SegmentationResult(
            video_id=video_id,
            kind="single",
            variant_count=1,
            spans=(VariantSpan(variant_index=1, start_frame=lo, end_frame=hi),),
            observed_numbers=observed_numbers,
        )

    value_sequence = [value for value, _, _ in transitions]
    n_variants = len(value_sequence)

    # Multi hợp lệ ⟺ dãy giá trị phân biệt đúng bằng [1, 2, …, N] với N >= 2.
    is_consecutive_from_one = value_sequence == list(range(1, n_variants + 1))

    def _confirmed(pos: int, value: int) -> bool:
        """Đếm số frame mẫu liên tiếp mang ``value`` kể từ vị trí ``pos``."""
        run = 0
        for probe in range(pos, len(samples)):
            if samples[probe][1] == value:
                run += 1
            else:
                break
        return run >= confirm

    # Mỗi giá trị mới (2..N) phải được xác nhận trên >= confirm frame liên tiếp.
    all_confirmed = all(
        _confirmed(pos, value) for value, pos, _ in transitions[1:]
    )

    if n_variants >= 2 and is_consecutive_from_one and all_confirmed:
        boundary_frames = [frame_index for _, _, frame_index in transitions]
        spans: list[VariantSpan] = []
        for k in range(n_variants):
            start = lo if k == 0 else boundary_frames[k]
            end = (hi if k == n_variants - 1 else boundary_frames[k + 1] - 1)
            spans.append(
                VariantSpan(variant_index=k + 1, start_frame=start, end_frame=end)
            )
        return SegmentationResult(
            video_id=video_id,
            kind="multi",
            variant_count=n_variants,
            spans=tuple(spans),
            observed_numbers=observed_numbers,
        )

    # Req 4.3/4.7: mọi trường hợp bất thường còn lại → cần rà soát thủ công.
    return SegmentationResult(
        video_id=video_id,
        kind="manual_review",
        variant_count=0,
        spans=(),
        observed_numbers=observed_numbers,
    )


# Một "nguồn frame" là callable nhận chỉ_số_frame (0-based) và trả về frame
# (mảng numpy) hoặc ``None`` nếu không đọc được frame đó.
FrameReader = Callable[[int], "object | None"]


def _make_frame_reader(video) -> "FrameReader":
    """Chuẩn hóa *video* thành một callable ``read(frame_index) -> frame | None``.

    Cho phép :func:`refine_boundary` đọc một frame bất kỳ theo chỉ số mà không
    phụ thuộc cứng vào OpenCV — thuận tiện cho property-test (Property 7) chỉ cần
    truyền một hàm thuần ánh xạ ``frame_index -> frame``.

    *video* có thể là:

    * một **callable** ``frame_index -> frame | None`` (dùng nguyên); hoặc
    * một đối tượng kiểu :class:`cv2.VideoCapture` (có ``set`` + ``read``): khi đó
      đọc frame bằng ``set(CAP_PROP_POS_FRAMES, index)`` rồi ``read()``.

    Args:
        video: Nguồn frame (callable hoặc capture giống ``cv2.VideoCapture``).

    Returns:
        Callable đọc frame theo chỉ số; trả ``None`` khi không đọc được.
    """
    if callable(video):
        return video

    # Giả định đối tượng kiểu cv2.VideoCapture.
    import cv2  # import trễ: chỉ cần khi thực sự đọc từ capture

    def _read(index: int):
        try:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            grabbed, frame = video.read()
        except Exception:  # noqa: BLE001 - capture lỗi -> coi như không đọc được
            return None
        if not grabbed or frame is None:
            return None
        return frame

    return _read


def refine_boundary(
    detector: "NumberDetector",
    video,
    coarse_lo: int,
    coarse_hi: int,
    target_number: int,
    cfg: "PreprocessConfig",
    logger: "logging.Logger | None" = None,
) -> int:
    """Tinh chỉnh một ranh giới thô về frame ĐẦU TIÊN mang ``target_number``.

    Hiện thực Req 4.4: cho một ranh giới thô nằm giữa hai frame mẫu
    ``coarse_lo`` (frame mẫu mang giá trị **cũ**) và ``coarse_hi`` (frame mẫu đầu
    tiên mang giá trị **mới** ``target_number``), tìm chỉ số frame ``r`` đầu tiên
    trong khoảng ``(coarse_lo, coarse_hi]`` mà :meth:`NumberDetector.detect` đọc
    được đúng ``target_number``. Vì detector tổng hợp là **đơn điệu** (giá trị cũ
    trước điểm chuyển ``T``, giá trị mới từ ``T`` trở đi), tìm-nhị-phân cho ra
    đúng ``T``; trong mọi trường hợp sai số ``|r − T| ≤ 1`` (Property 7).

    Xử lý lỗi (Req 4.8): nếu đọc frame thất bại, hoặc ``detector.detect`` ném lỗi
    / quá thời gian trên **một** frame, frame đó được **coi như không có số**
    (không mang ``target_number``), ghi log, rồi tiếp tục — không làm hỏng cả quá
    trình tinh chỉnh.

    Args:
        detector: Bộ phát hiện số (:class:`NumberDetector`).
        video: Nguồn frame — callable ``frame_index -> frame`` hoặc capture kiểu
            ``cv2.VideoCapture`` (xem :func:`_make_frame_reader`).
        coarse_lo: Chỉ số frame mẫu mang giá trị cũ (cận dưới, **loại trừ**).
        coarse_hi: Chỉ số frame mẫu đầu tiên mang ``target_number`` (cận trên,
            **bao gồm**) — cũng là giá trị mặc định khi không tìm thấy frame nào
            khác mang số mới.
        target_number: Giá trị số mới cần định vị điểm xuất hiện đầu tiên.
        cfg: :class:`~qipedc_video_preprocess.config.PreprocessConfig` (giữ trong
            chữ ký để đồng nhất interface; hiện chưa dùng tham số nào trực tiếp).
        logger: Logger tùy chọn cho lỗi frame (Req 4.8); mặc định logger module.

    Returns:
        Chỉ số frame ``r`` (0-based) đầu tiên mang ``target_number``, nằm trong
        ``(coarse_lo, coarse_hi]``; trả về ``coarse_hi`` nếu không frame nào trong
        khoảng được xác nhận mang số mới.
    """
    log = logger if logger is not None else globals()["logger"]
    read = _make_frame_reader(video)

    lo = int(coarse_lo)
    hi = int(coarse_hi)
    # Khoảng tìm kiếm rỗng/đảo -> không có gì để tinh chỉnh, trả cận trên.
    if hi <= lo:
        return hi

    def _bears_target(index: int) -> bool:
        """True nếu frame *index* được detector đọc ra đúng ``target_number``."""
        frame = read(index)
        if frame is None:
            log.warning(
                "refine_boundary: không đọc được frame %d -> coi như không có số "
                "(Req 4.8).",
                index,
            )
            return False
        try:
            result = detector.detect(frame)
        except Exception as exc:  # noqa: BLE001 - Req 4.8: lỗi/timeout 1 frame
            log.warning(
                "refine_boundary: detector lỗi tại frame %d (%s: %s) -> coi như "
                "không có số, tiếp tục (Req 4.8).",
                index,
                type(exc).__name__,
                exc,
            )
            return False
        return result.number == target_number

    # Tìm-nhị-phân cận-dưới (lower bound) chỉ số đầu tiên mang target_number trong
    # (coarse_lo, coarse_hi]. coarse_hi đã biết mang số mới (frame mẫu) nên luôn là
    # ứng viên hợp lệ mặc định.
    search_lo = lo + 1
    search_hi = hi
    answer = hi
    while search_lo <= search_hi:
        mid = (search_lo + search_hi) // 2
        if _bears_target(mid):
            answer = mid
            search_hi = mid - 1
        else:
            search_lo = mid + 1

    return answer


def segment_video(
    entry: "VideoEntry",
    props: "VideoProps",
    detector: "NumberDetector",
    cfg: "PreprocessConfig",
    logger: "logging.Logger | None" = None,
) -> SegmentationResult:
    """Phân đoạn một video: lấy mẫu → phát hiện → phân loại → tinh chỉnh ranh giới.

    Điều phối toàn bộ Requirement 4 cho một video:

    1. Sinh chỉ số frame mẫu cách đều bằng :func:`sample_frame_indices` theo
       ``props.fps`` / ``props.num_frames`` và ``cfg.sample_interval_seconds``
       (Req 4.1).
    2. Đọc từng frame mẫu từ ``entry.path`` bằng :class:`cv2.VideoCapture` và gọi
       ``detector.detect``; lỗi/timeout của detector trên một frame → coi frame đó
       **không có số**, ghi log, tiếp tục (Req 4.8).
    3. Phân loại chuỗi ``(frame_index, number|None)`` bằng :func:`classify_sequence`.
    4. Với video nhiều cách, tinh chỉnh từng ranh giới thô về sai số ≤ 1 frame
       bằng :func:`refine_boundary` (Req 4.4), rồi dựng lại các :class:`VariantSpan`
       liên tục không chồng lấn quanh các ranh giới đã tinh chỉnh.

    Args:
        entry: :class:`~qipedc_video_preprocess.discovery.VideoEntry` của video.
        props: :class:`~qipedc_video_preprocess.video_probe.VideoProps` (fps, số
            frame) đã probe được.
        detector: Bộ phát hiện số (:class:`NumberDetector`).
        cfg: :class:`~qipedc_video_preprocess.config.PreprocessConfig`.
        logger: Logger tùy chọn; mặc định logger của module.

    Returns:
        :class:`SegmentationResult` với ``video_id`` của ``entry`` và các span đã
        tinh chỉnh (với video nhiều cách).
    """
    log = logger if logger is not None else globals()["logger"]

    sample_indices = sample_frame_indices(
        props.fps, props.num_frames, cfg.sample_interval_seconds
    )

    import cv2  # import trễ: phần thuần (classify/refine với callable) không cần

    capture = cv2.VideoCapture(str(entry.path))
    try:
        if not capture.isOpened():
            log.warning(
                "segment_video: không mở được video %s (video_id=%s) -> coi như "
                "không có frame mẫu nào.",
                entry.path,
                entry.video_id,
            )

        samples: list[Sample] = []
        for index in sample_indices:
            number: int | None = None
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                grabbed, frame = capture.read()
            except Exception as exc:  # noqa: BLE001 - lỗi đọc frame
                log.warning(
                    "segment_video[%s]: lỗi đọc frame %d (%s: %s) -> coi như "
                    "không có số, tiếp tục (Req 4.8).",
                    entry.video_id,
                    index,
                    type(exc).__name__,
                    exc,
                )
                grabbed, frame = False, None

            if grabbed and frame is not None:
                try:
                    number = detector.detect(frame).number
                except Exception as exc:  # noqa: BLE001 - Req 4.8
                    log.warning(
                        "segment_video[%s]: detector lỗi tại frame %d (%s: %s) -> "
                        "coi như không có số, tiếp tục (Req 4.8).",
                        entry.video_id,
                        index,
                        type(exc).__name__,
                        exc,
                    )
                    number = None
            samples.append((index, number))

        result = classify_sequence(samples, cfg, video_id=entry.video_id)

        # Chỉ video nhiều cách mới có ranh giới nội bộ cần tinh chỉnh (Req 4.4).
        if result.kind != "multi" or len(result.spans) < 2:
            return result

        coarse_spans = result.spans
        sample_frames = [idx for idx, _ in samples]
        n_variants = len(coarse_spans)

        # Với mỗi ranh giới thô (giữa cách k và k+1), giá trị số mới là k+1; frame
        # mẫu đầu tiên mang giá trị mới là start_frame của span k (0-based theo
        # thứ tự). coarse_lo là frame mẫu liền trước trong chuỗi mẫu.
        refined: list[int] = []
        for k in range(1, n_variants):
            coarse_hi = coarse_spans[k].start_frame
            target_number = k + 1

            # frame mẫu liền trước frame mẫu mang số mới = cận dưới (loại trừ).
            try:
                pos = sample_frames.index(coarse_hi)
            except ValueError:
                pos = -1
            if pos > 0:
                coarse_lo = sample_frames[pos - 1]
            else:
                coarse_lo = max(0, coarse_hi - 1)

            r = refine_boundary(
                detector,
                capture,
                coarse_lo,
                coarse_hi,
                target_number,
                cfg,
                logger=log,
            )
            refined.append(r)

        # Dựng lại span liên tục, không chồng lấn quanh các ranh giới đã tinh chỉnh.
        new_spans: list[VariantSpan] = []
        for k in range(n_variants):
            start = coarse_spans[0].start_frame if k == 0 else refined[k - 1]
            end = (
                coarse_spans[-1].end_frame
                if k == n_variants - 1
                else refined[k] - 1
            )
            new_spans.append(
                VariantSpan(variant_index=k + 1, start_frame=start, end_frame=end)
            )

        return SegmentationResult(
            video_id=result.video_id,
            kind=result.kind,
            variant_count=result.variant_count,
            spans=tuple(new_spans),
            observed_numbers=result.observed_numbers,
        )
    finally:
        capture.release()

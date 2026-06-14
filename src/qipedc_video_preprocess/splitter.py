"""Bộ_Cắt_Video — cắt video nhiều cách và đặt tên file đầu ra.

Module này định nghĩa kiểu dữ liệu clip đầu ra (:class:`OutputClip`), logic thuần
tính khoảng frame sau khi trừ Biên_An_Toàn (:func:`trimmed_spans`), đặt tên +
phát hiện xung đột (:func:`variant_filename` / :func:`plan_outputs`), và hàm biên
ghi sub-clip bằng ``cv2.VideoWriter`` (:func:`write_clip`). Xem design.md —
Requirements 5 và 6.

Quy ước frame & ranh giới (xem design.md "Quy ước frame & ranh giới"):

* Frame đánh chỉ số **0-based**; :class:`~qipedc_video_preprocess.segmenter.VariantSpan`
  dùng khoảng **đóng** ``[start_frame, end_frame]`` (cả hai đầu inclusive).
* Ranh giới giữa cách ``k`` và ``k+1`` nằm tại ``b`` = frame đầu tiên mang số
  ``k+1`` (tức ``start_frame`` của cách ``k+1``). Sau khi trừ Biên_An_Toàn
  ``margin = m``: cách ``k`` kết thúc tại ``b - 1 - m`` và cách ``k+1`` bắt đầu
  tại ``b + m``. Nhờ đó **không frame nào trong ``[b - m, b + m - 1]``** còn xuất
  hiện ở bất kỳ clip nào (Req 5.2).
* Cách đầu tiên bắt đầu tại frame ``0`` (không trừ ở đầu video); cách cuối kết
  thúc tại ``num_frames - 1`` (không trừ ở cuối video). Chỉ các **ranh giới nội
  bộ** mới bị trừ margin hai phía.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .segmenter import SegmentationResult, VariantSpan

logger = logging.getLogger(__name__)

# Codec mp4v cho container .mp4 — không phụ thuộc binary ngoài trên C: (Req 1.6).
_FOURCC = "mp4v"


@dataclass(frozen=True)
class OutputClip:
    """Một file video con dự kiến được ghi ra.

    Attributes:
        video_id: ``video_id`` của video nguồn (stem), ví dụ ``"W00202"``.
        variant_index: Số thứ tự cách (1-based) với video nhiều cách; ``None`` với
            video một cách (giữ nguyên tên gốc).
        out_filename: Tên file đầu ra — ``"<id>.mp4"`` (single) hoặc
            ``"<id>_c<k>.mp4"`` (multi).
        start_frame: Frame bắt đầu (0-based, inclusive) **đã** trừ Biên_An_Toàn.
        end_frame: Frame kết thúc (0-based, inclusive) **đã** trừ Biên_An_Toàn.
    """

    video_id: str
    variant_index: int | None
    out_filename: str
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class Conflict:
    """Xung đột tên file giữa hai (hoặc nhiều) video nguồn KHÁC NHAU.

    Khi hai video con phát sinh từ các ``video_id`` phân biệt cùng quy về một tên
    file đầu ra (Req 6.3/6.5), ta không ghi đè lẫn nhau mà ghi nhận một
    :class:`Conflict` và đánh dấu các video liên quan cần rà soát thủ công.

    Attributes:
        out_filename: Tên file đầu ra bị trùng (ví dụ ``"W00202.mp4"``).
        video_ids: Tuple các ``video_id`` nguồn phân biệt cùng sinh ra tên này,
            sắp xếp tăng dần theo mã ký tự để tái lập được.
    """

    out_filename: str
    video_ids: tuple[str, ...]


def trimmed_spans(
    spans: "list[VariantSpan] | tuple[VariantSpan, ...]",
    num_frames: int,
    margin: int,
) -> list[VariantSpan]:
    """Trừ Biên_An_Toàn quanh MỖI ranh giới nội bộ giữa hai cách liền kề (THUẦN).

    Hiện thực Req 5.1/5.2/5.8. Với danh sách :class:`VariantSpan` liền kề của một
    video (theo thứ tự thời gian) và ``margin >= 0``:

    * **Đầu video / cuối video không bị trừ thêm:** cách đầu tiên luôn bắt đầu tại
      frame ``0``; cách cuối cùng luôn kết thúc tại ``num_frames - 1``.
    * **Mỗi ranh giới nội bộ** tại frame ``b`` (start của cách kế tiếp) bị trừ
      ``margin`` ở cả hai phía: cách trước kết thúc tại ``b - 1 - margin``, cách
      sau bắt đầu tại ``b + margin``. Do đó không frame nào trong
      ``[b - margin, b + margin - 1]`` còn thuộc bất kỳ span kết quả nào, và các
      span kết quả không chồng lấn.
    * **Span còn lại ``< 1`` frame bị loại bỏ** (Req 5.8); các span giữ lại bảo
      toàn ``variant_index`` gốc.

    Args:
        spans: Danh sách (hoặc tuple) :class:`VariantSpan` liền kề, không chồng
            lấn, theo thứ tự thời gian. Khoảng đóng ``[start_frame, end_frame]``.
        num_frames: Tổng số frame của video gốc; cách cuối kết thúc tại
            ``num_frames - 1``.
        margin: Số frame Biên_An_Toàn trừ ở mỗi phía của mỗi ranh giới nội bộ
            (``>= 0``).

    Returns:
        Danh sách :class:`VariantSpan` đã trừ Biên_An_Toàn, theo thứ tự thời gian,
        chỉ gồm các span còn ``>= 1`` frame.
    """
    if not spans:
        return []

    m = max(0, int(margin))
    # Sắp xếp theo thời gian để xác định chắc chắn cách đầu / cách cuối.
    ordered = sorted(spans, key=lambda s: s.start_frame)
    n = len(ordered)
    last_index = n - 1

    result: list[VariantSpan] = []
    for i, span in enumerate(ordered):
        # Đầu video (cách 1) giữ nguyên start = 0; ranh giới nội bộ bên trái trừ m.
        new_start = 0 if i == 0 else span.start_frame + m
        # Cuối video (cách N) giữ nguyên end = num_frames - 1; ranh giới nội bộ
        # bên phải trừ m.
        new_end = (num_frames - 1) if i == last_index else span.end_frame - m

        # Bỏ span còn < 1 frame sau khi trừ margin (Req 5.8).
        if new_end < new_start:
            continue

        result.append(
            VariantSpan(
                variant_index=span.variant_index,
                start_frame=new_start,
                end_frame=new_end,
            )
        )

    return result


def variant_filename(
    video_id: str,
    variant_index: int | None,
    total: int,
) -> str:
    """Sinh tên file đầu ra theo quy ước đặt tên (THUẦN — Req 6.1/6.2).

    * **Một cách** (``total <= 1`` hoặc ``variant_index is None``): trả về
      ``"<video_id>.mp4"`` — giữ nguyên ``video_id``, không thêm hậu tố
      ``_c<k>`` (Req 6.2).
    * **Nhiều cách** (``total >= 2`` và ``variant_index`` xác định): trả về
      ``"<video_id>_c<k>.mp4"`` với ``k = variant_index`` viết thập phân **không**
      có số 0 ở đầu, ``k`` chạy 1..N (Req 6.1).

    Args:
        video_id: ``video_id`` nguồn (stem), ví dụ ``"W00202"``.
        variant_index: Số thứ tự cách (1-based) với video nhiều cách; ``None`` với
            video một cách.
        total: Tổng số cách của video (``1`` nếu một cách; ``N >= 2`` nếu nhiều
            cách).

    Returns:
        Tên file đầu ra (``str``) theo đúng quy ước.
    """
    if variant_index is None or int(total) <= 1:
        return f"{video_id}.mp4"
    return f"{video_id}_c{int(variant_index)}.mp4"


def plan_outputs(
    seg_results: "Iterable[SegmentationResult]",
) -> tuple[list[OutputClip], list[Conflict]]:
    """Dựng toàn bộ tên file đầu ra + phát hiện xung đột tên (THUẦN — Req 5.4/6).

    Với mỗi :class:`~qipedc_video_preprocess.segmenter.SegmentationResult`:

    * ``kind == "single"``: sinh đúng **một** :class:`OutputClip` với
      ``variant_index = None`` và tên ``"<video_id>.mp4"`` (Req 5.4/6.2). Khoảng
      frame lấy từ span duy nhất nếu có (mặc định ``(0, 0)`` cho trường hợp suy
      biến không có span).
    * ``kind == "multi"``: sinh **một** :class:`OutputClip` cho mỗi span theo thứ
      tự thời gian, tên ``"<video_id>_c<k>.mp4"`` với ``k = span.variant_index``
      (Req 6.1). Khoảng frame mang trực tiếp ``span.start_frame`` /
      ``span.end_frame`` (việc trừ Biên_An_Toàn do :func:`trimmed_spans` đảm
      nhiệm ở bước khác).
    * ``kind == "manual_review"`` (hoặc ``spans`` rỗng với multi): **không** sinh
      clip nào.

    **Phát hiện xung đột (Req 6.3/6.5):** chỉ các tên file phát sinh từ **các
    ``video_id`` KHÁC NHAU** mới tính là xung đột. Khi một tên file được dùng bởi
    từ hai ``video_id`` phân biệt trở lên, mọi clip mang tên đó **bị loại khỏi**
    danh sách đầu ra (không ghi đè lẫn nhau) và ghi nhận một :class:`Conflict`.
    Các clip có tên trùng nhưng cùng một ``video_id`` không bị coi là xung đột.

    Args:
        seg_results: Iterable các :class:`SegmentationResult` (mỗi video một kết
            quả). Không yêu cầu các ``video_id`` phân biệt, nhưng xung đột chỉ
            được tính giữa các ``video_id`` khác nhau.

    Returns:
        Cặp ``(clips, conflicts)``:

        * ``clips``: danh sách :class:`OutputClip` đầu ra (đã loại các clip dính
          xung đột), giữ thứ tự xuất hiện theo ``seg_results``.
        * ``conflicts``: danh sách :class:`Conflict`, sắp xếp tăng dần theo
          ``out_filename`` để tái lập được.
    """
    clips: list[OutputClip] = []

    for result in seg_results:
        if result.kind == "single":
            if result.spans:
                span = result.spans[0]
                start_frame, end_frame = span.start_frame, span.end_frame
            else:
                start_frame, end_frame = 0, 0
            clips.append(
                OutputClip(
                    video_id=result.video_id,
                    variant_index=None,
                    out_filename=variant_filename(result.video_id, None, 1),
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )
        elif result.kind == "multi":
            total = result.variant_count
            for span in result.spans:
                clips.append(
                    OutputClip(
                        video_id=result.video_id,
                        variant_index=span.variant_index,
                        out_filename=variant_filename(
                            result.video_id, span.variant_index, total
                        ),
                        start_frame=span.start_frame,
                        end_frame=span.end_frame,
                    )
                )
        # manual_review (hoặc kind khác) → không sinh clip nào.

    # Gom các video_id phân biệt theo từng tên file để phát hiện xung đột.
    sources_by_name: dict[str, set[str]] = {}
    for clip in clips:
        sources_by_name.setdefault(clip.out_filename, set()).add(clip.video_id)

    conflicting_names = {
        name for name, sources in sources_by_name.items() if len(sources) > 1
    }

    conflicts = [
        Conflict(
            out_filename=name,
            video_ids=tuple(sorted(sources_by_name[name])),
        )
        for name in sorted(conflicting_names)
    ]

    # Loại mọi clip mang tên bị xung đột (không ghi đè lẫn nhau — Req 6.5).
    kept_clips = [clip for clip in clips if clip.out_filename not in conflicting_names]

    return kept_clips, conflicts


def write_clip(
    src_path: "Path | str",
    out_path: "Path | str",
    span: VariantSpan,
    props,
    logger=logger,
) -> bool:
    """Ghi sub-clip ``[span.start_frame, span.end_frame]`` ra ``out_path`` (BIÊN).

    Hiện thực Req 1.2/5.3/5.6/5.7/6.4/9.3:

    * **Giữ nguyên fps & (w, h):** :class:`cv2.VideoWriter` được mở với đúng
      ``props.fps`` và ``(props.width, props.height)`` của video gốc (Req 5.3).
    * **Ghi đè trùng tên:** ``cv2.VideoWriter`` mở cùng đường dẫn sẽ ghi đè hoàn
      toàn file cũ; không thêm hậu tố (Req 6.4/9.3).
    * **Lỗi ghi clip → log + tiếp tục:** mọi lỗi mở/đọc/ghi clip được ghi log kèm
      ``video_id`` (suy từ tên file) và **trả về ``False``**; caller tiếp tục với
      video khác (Req 5.6).
    * **Lỗi ghi-log → dừng:** nếu chính việc ghi log thất bại, lỗi được để lan ra
      (không nuốt) nên lần chạy dừng, bảo đảm không lỗi nào bị mất âm thầm
      (Req 5.7).

    Khoảng frame dùng quy ước **đóng** ``[start_frame, end_frame]`` (inclusive),
    0-based; hàm đọc đúng các frame trong khoảng này từ ``src_path``.

    Args:
        src_path: Đường dẫn video nguồn.
        out_path: Đường dẫn file đầu ra (trong ``split_variants/``).
        span: Khoảng frame của clip (đã trừ Biên_An_Toàn nếu có), inclusive.
        props: :class:`~qipedc_video_preprocess.video_probe.VideoProps` của video
            gốc — cung cấp ``fps`` và ``(width, height)`` để giữ nguyên (Req 5.3).
        logger: Logger để ghi lỗi; mặc định logger của module.

    Returns:
        ``True`` nếu ghi thành công; ``False`` nếu có lỗi (đã được ghi log).
    """
    # Import cv2 trong thân hàm để lớp logic thuần (đã được property-test) không
    # phụ thuộc OpenCV khi import module.
    import cv2

    src_path = Path(src_path)
    out_path = Path(out_path)
    video_id = out_path.stem

    capture = None
    writer = None
    try:
        capture = cv2.VideoCapture(str(src_path))
        if not capture.isOpened():
            logger.error(
                "write_clip: không mở được video nguồn %s cho clip %s",
                src_path,
                out_path.name,
            )
            return False

        fps = float(props.fps)
        width = int(props.width)
        height = int(props.height)
        if fps <= 0 or width <= 0 or height <= 0:
            logger.error(
                "write_clip: thuộc tính video không hợp lệ (fps=%s, w=%s, h=%s) "
                "cho clip %s",
                fps,
                width,
                height,
                out_path.name,
            )
            return False

        # Bảo đảm thư mục đầu ra tồn tại (Dataset/processed_videos/split_variants/).
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*_FOURCC)
        # Mở cùng đường dẫn → ghi đè hoàn toàn file cũ, không thêm hậu tố (Req 6.4/9.3).
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.error(
                "write_clip: không mở được VideoWriter cho clip %s (video_id=%s)",
                out_path.name,
                video_id,
            )
            return False

        start = max(0, int(span.start_frame))
        end = int(span.end_frame)  # inclusive

        # Tua tới frame bắt đầu nếu được hỗ trợ; nếu không, đọc tuần tự đến đó.
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames_written = 0
        current = start
        while current <= end:
            grabbed, frame = capture.read()
            if not grabbed:
                break
            writer.write(frame)
            frames_written += 1
            current += 1

        if frames_written == 0:
            logger.error(
                "write_clip: không ghi được frame nào cho clip %s (video_id=%s, "
                "span=[%s,%s])",
                out_path.name,
                video_id,
                span.start_frame,
                span.end_frame,
            )
            return False

        return True
    except Exception as exc:  # noqa: BLE001 — lỗi I/O cục bộ không làm hỏng cả lần chạy
        # Ghi log lỗi kèm video_id & số cách (Req 5.6). Nếu CHÍNH việc ghi log
        # thất bại, lỗi mới sẽ lan ra ngoài except → lần chạy dừng (Req 5.7).
        logger.error(
            "write_clip: lỗi khi ghi clip %s (video_id=%s, variant=%s): %s",
            out_path.name,
            video_id,
            span.variant_index,
            exc,
        )
        return False
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()

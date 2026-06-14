"""Bộ_Ghi_Log — ghi nhật ký và báo cáo tóm tắt cho mỗi lần chạy (Requirement 8).

Module này định nghĩa:

* :class:`RunReport` — các số đếm tóm tắt của một lần chạy (đều là số nguyên
  ``>= 0``).
* :func:`build_run_report` — LOGIC THUẦN kế toán từ tập
  :class:`~qipedc_video_preprocess.segmenter.SegmentationResult`, bảo đảm bất
  biến ``total_discovered = single_variant + multi_variant + skipped``,
  ``manual_review <= total_discovered``, và ``sub_clips_generated`` bằng tổng số
  span hợp lệ trên các video nhiều cách (Req 8.2).
* :func:`get_run_logger` — tạo ``Dataset/logs/`` nếu thiếu (Req 8.5) và gắn một
  :class:`logging.FileHandler` ghi tới ``preprocess_<YYYYmmdd_HHMMSS>.log`` với
  tên **duy nhất** không ghi đè log lần chạy trước (Req 8.1); lỗi mở/ghi file log
  → raise để dừng (Req 8.6).

Xem design.md, Requirement 8.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

_LOGGER_NAME = "qipedc_video_preprocess.preprocess"


@dataclass
class RunReport:
    """Số đếm tóm tắt của một lần chạy (Req 8.2). Mọi trường là số nguyên ``>= 0``.

    Attributes:
        total_discovered: Tổng video đã duyệt.
        single_variant: Số Video_Một_Cách.
        multi_variant: Số Video_Nhiều_Cách.
        sub_clips_generated: Tổng số video con đã sinh (span hợp lệ).
        skipped: Số video bị bỏ qua (không đọc được / xung đột tên...).
        manual_review: Số video cần rà soát thủ công.
    """

    total_discovered: int = 0
    single_variant: int = 0
    multi_variant: int = 0
    sub_clips_generated: int = 0
    skipped: int = 0
    manual_review: int = 0


def build_run_report(
    seg_results: "Iterable",
    skipped: int = 0,
    manual_review_ids: "Iterable[str] | None" = None,
) -> RunReport:
    """Kế toán :class:`RunReport` từ kết quả phân đoạn (THUẦN — Req 8.2).

    Phân loại mỗi :class:`~qipedc_video_preprocess.segmenter.SegmentationResult`:

    * ``kind == "single"`` → tăng ``single_variant``;
    * ``kind == "multi"`` → tăng ``multi_variant`` và cộng ``len(spans)`` vào
      ``sub_clips_generated`` (mỗi span là một video con hợp lệ);
    * ``kind == "manual_review"`` → đếm vào ``manual_review`` (cũng có thể được bổ
      sung qua ``manual_review_ids`` cho các video bị đánh dấu ở bước khác, ví dụ
      xung đột tên).

    Bất biến (Req 8.2): ``total_discovered = single_variant + multi_variant +
    skipped`` và ``manual_review <= total_discovered``. Video có ``kind ==
    "manual_review"`` **không** sinh được video con nên được tính vào ``skipped``
    (chúng đã duyệt nhưng bị bỏ qua việc cắt). ``manual_review`` đếm số video bị
    đánh dấu cần rà soát thủ công (từ segmenter và/hoặc các bước ngoài như xung
    đột tên), là một tập con của các video đã duyệt.

    Args:
        seg_results: Iterable các ``SegmentationResult`` của lần chạy.
        skipped: Số video bị bỏ qua từ các bước ngoài segmenter (không đọc được,
            bị loại ở discovery...). Video ``kind == "manual_review"`` được cộng
            thêm vào số này.
        manual_review_ids: Tập ``video_id`` bị đánh dấu manual_review từ các bước
            ngoài segmenter (vd xung đột tên đầu ra). Hợp nhất với các video có
            ``kind == "manual_review"`` để tránh đếm trùng.

    Returns:
        :class:`RunReport` đã kế toán với mọi bất biến thỏa mãn.
    """
    single = 0
    multi = 0
    sub_clips = 0
    manual_kind = 0
    manual_ids: set[str] = set(manual_review_ids or set())

    for result in seg_results:
        if result.kind == "single":
            single += 1
        elif result.kind == "multi":
            multi += 1
            sub_clips += len(result.spans)
        else:  # manual_review
            manual_kind += 1
            manual_ids.add(result.video_id)

    # Video manual_review-kind không sinh video con → tính như "bỏ qua" việc cắt,
    # nên gộp vào skipped để giữ bất biến total = single + multi + skipped.
    skipped = max(0, int(skipped)) + manual_kind
    total_discovered = single + multi + skipped

    manual_review = len(manual_ids)
    # Bất biến: manual_review <= total_discovered.
    if manual_review > total_discovered:
        manual_review = total_discovered

    return RunReport(
        total_discovered=total_discovered,
        single_variant=single,
        multi_variant=multi,
        sub_clips_generated=sub_clips,
        skipped=skipped,
        manual_review=manual_review,
    )


def get_run_logger(cfg) -> logging.Logger:
    """Trả về logger ghi tới file log có dấu thời gian dưới ``cfg.log_dir`` (Req 8).

    Tạo thư mục log (mặc định ``Dataset/logs/``) nếu thiếu (Req 8.5). Gắn một
    :class:`logging.FileHandler` mới tới ``preprocess_<YYYYmmdd_HHMMSS>.log`` —
    tên có dấu thời gian đến giây nên **không ghi đè** log lần chạy trước
    (Req 8.1). Nếu trùng tên (cùng giây), thêm hậu tố ``_N`` để vẫn duy nhất.

    Lỗi tạo thư mục / mở file log được để **lan ra** (raise) nên lần chạy dừng,
    bảo đảm không thông tin nào bị mất âm thầm (Req 8.6).

    Args:
        cfg: :class:`~qipedc_video_preprocess.config.PreprocessConfig`.

    Returns:
        Một :class:`logging.Logger` đã gắn FileHandler ghi UTF-8.
    """
    log_dir = cfg.resolve(cfg.log_dir)
    # Lỗi tạo thư mục → lan ra (Req 8.6); không nuốt.
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"preprocess_{timestamp}.log"
    # Bảo đảm tên duy nhất ngay cả khi hai lần chạy trong cùng một giây (Req 8.1).
    suffix = 1
    while log_file.exists():
        log_file = log_dir / f"preprocess_{timestamp}_{suffix}.log"
        suffix += 1

    logger = logging.getLogger(f"{_LOGGER_NAME}.{log_file.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Dọn handler cũ nếu logger cùng tên bị tái dùng.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    # Lỗi mở file log → lan ra (Req 8.6).
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger

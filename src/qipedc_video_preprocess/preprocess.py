"""Bộ_Tiền_Xử_Lý — CLI orchestrator điều phối toàn bộ công đoạn (Req 1, 9).

Nối các bước theo đúng thiết kế:

``cfg.validate()`` → nếu có lỗi: ghi lỗi cấu hình & **dừng trước khi xử lý bất kỳ
video nào, không tạo file đầu ra** (Req 1.4) → ``discover_videos`` → nếu rỗng:
cảnh báo & kết thúc (Req 2.6) → mỗi video: ``probe`` + ``segment_video`` →
``plan_outputs`` (xung đột tên → manual_review, Req 6.5) → ``write_clip`` từng
clip → ``build_new_rows`` + ``write_new_labels`` → ghi ``RunReport``.

Tính idempotent (Req 9) đạt được nhờ tên file & STT là hàm xác định của nguồn,
và luôn ghi đè cùng tên thay vì thêm hậu tố.

Công đoạn **không** tạo/sửa/xóa file thuộc ``src/qipedc2vsl400/`` hay
``Dataset/final_dataset/`` (Req 1.5): nó chỉ ghi vào ``split_output_dir``,
``new_labels_path`` và ``log_dir`` đã được ``validate()`` ràng buộc nằm trong cây
dự án trên ``D:``.

CLI::

    python -m qipedc_video_preprocess.preprocess
        [--project-root D:/...] [--sample-interval 1.0] [--safety-margin 3]
        [--ocr-threshold 0.5] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import PreprocessConfig
from .discovery import discover_videos
from .label_writer import build_new_rows, read_source_labels, write_new_labels
from .run_logger import RunReport, build_run_report, get_run_logger
from .segmenter import segment_video
from .splitter import plan_outputs, trimmed_spans, write_clip


def run_preprocess(
    cfg: PreprocessConfig,
    detector=None,
    *,
    dry_run: bool = False,
    logger=None,
) -> RunReport:
    """Chạy toàn bộ công đoạn tách video cho cấu hình *cfg* (Req 1, 9).

    Args:
        cfg: :class:`PreprocessConfig` đã cấu hình.
        detector: Bộ phát hiện số (:class:`NumberDetector`). Nếu ``None`` và không
            ``dry_run``, một :class:`EasyOcrNumberDetector` được tạo từ ``cfg``.
            Tham số này cho phép inject detector giả trong integration test.
        dry_run: Nếu ``True``, không ghi file clip/nhãn (chỉ duyệt + phân đoạn +
            kế toán) — hữu ích để kiểm tra phân loại mà không tốn I/O.
        logger: Logger của lần chạy. Nếu ``None``, tạo bằng :func:`get_run_logger`
            (tạo ``Dataset/logs/`` nếu thiếu — Req 8.5). Inject được để test.

    Returns:
        :class:`RunReport` tóm tắt lần chạy.
    """
    # --- Bước 1: validate cấu hình TRƯỚC khi tạo bất kỳ file đầu ra nào (Req 1.4) ---
    config_errors = cfg.validate()
    if config_errors:
        # Không tạo logger/file log ở đây để tuân thủ "không tạo file đầu ra" khi
        # cấu hình sai (Req 1.4); báo lỗi ra stderr.
        for err in config_errors:
            print(f"[config error] {err}", file=sys.stderr)
        # Trả report rỗng; không có video nào được xử lý.
        return RunReport()

    log = logger if logger is not None else get_run_logger(cfg)

    # --- Bước 2: duyệt video nguồn ---
    entries = discover_videos(cfg, log)
    if not entries:
        log.warning("Không có video .mp4 nào để xử lý — kết thúc, không tạo đầu ra.")
        return RunReport()

    # Detector: tạo EasyOCR mặc định nếu cần (chỉ khi không dry-run và chưa inject).
    if detector is None and not dry_run:
        from .number_detector import EasyOcrNumberDetector

        detector = EasyOcrNumberDetector(cfg=cfg, log=log)

    discovered_ids = {entry.video_id: entry for entry in entries}

    # --- Bước 3: phân đoạn từng video ---
    from . import video_probe

    seg_results = []
    props_by_id: dict[str, object] = {}
    skipped = 0
    for entry in entries:
        props = video_probe.probe(entry.path)
        if props is None:
            log.warning(
                "Bỏ qua %s — không probe được thuộc tính video (video_id=%s).",
                entry.path,
                entry.video_id,
            )
            skipped += 1
            continue
        props_by_id[entry.video_id] = props

        if dry_run and detector is None:
            # Không có detector trong dry-run thuần: coi như single (không phân đoạn).
            from .segmenter import SegmentationResult

            result = SegmentationResult(
                video_id=entry.video_id,
                kind="single",
                variant_count=1,
                spans=(),
                observed_numbers=(),
            )
        else:
            result = segment_video(entry, props, detector, cfg, log)
        seg_results.append(result)

    # --- Bước 4: lên kế hoạch đầu ra + phát hiện xung đột tên (Req 6.5) ---
    clips, conflicts = plan_outputs(seg_results)
    manual_review_ids: set[str] = set()
    for conflict in conflicts:
        log.error(
            "Xung đột tên đầu ra %s giữa các video_id %s — đánh dấu rà soát thủ công.",
            conflict.out_filename,
            ", ".join(conflict.video_ids),
        )
        manual_review_ids.update(conflict.video_ids)

    # --- Bước 5: ghi từng clip (trừ Biên_An_Toàn cho video nhiều cách) ---
    seg_by_id = {r.video_id: r for r in seg_results}
    sub_clips_written = 0
    if not dry_run:
        split_dir = cfg.split_output_path
        for clip in clips:
            entry = discovered_ids[clip.video_id]
            props = props_by_id[clip.video_id]
            seg = seg_by_id[clip.video_id]

            # Tính span thực tế đã trừ Biên_An_Toàn cho video nhiều cách.
            span = _clip_span(clip, seg, props, cfg)
            if span is None:
                log.warning(
                    "Cách %s của %s còn < 1 frame sau Biên_An_Toàn — bỏ qua + rà "
                    "soát thủ công.",
                    clip.variant_index,
                    clip.video_id,
                )
                manual_review_ids.add(clip.video_id)
                continue

            out_path = split_dir / clip.out_filename
            ok = write_clip(entry.path, out_path, span, props, log)
            if ok:
                sub_clips_written += 1
            else:
                manual_review_ids.add(clip.video_id)

    # --- Bước 5b: gom video cần rà soát vào folder riêng ---
    # Hợp nhất mọi nguồn manual_review: segmenter không phân loại được
    # (kind == "manual_review") + xung đột tên + clip < 1 frame (đã thêm ở trên).
    inferred_ids: set[str] = set()
    for seg in seg_results:
        if seg.kind == "manual_review":
            manual_review_ids.add(seg.video_id)
        elif seg.kind == "multi" and getattr(seg, "inferred", False):
            inferred_ids.add(seg.video_id)

    if not dry_run and manual_review_ids:
        copied = _copy_review_videos(
            manual_review_ids, discovered_ids, cfg.manual_review_path, "manual_review", log
        )
        log.info(
            "Đã copy %d/%d video manual_review vào %s",
            copied,
            len(manual_review_ids),
            cfg.manual_review_path,
        )

    # Video tách bằng SUY LUẬN: đã tách tự động (có clip) nhưng cần người kiểm lại
    # ranh giới → copy cả các clip đã cắt lẫn bản gốc vào inferred_review, để
    # người review thấy ngay kết quả dự đoán mà không phải tự seek video gốc.
    if not dry_run and inferred_ids:
        inferred_dir = cfg.inferred_review_path
        # 1) Copy các clip sub-variant đã cắt (D0105_c1.mp4, D0105_c2.mp4, ...).
        copied_clips = _copy_inferred_clips(inferred_ids, clips, split_dir, inferred_dir, log)
        # 2) Copy bản gốc (D0105.mp4) để người review so sánh ranh giới thực tế.
        copied_orig = _copy_review_videos(
            inferred_ids, discovered_ids, inferred_dir, "inferred_review", log
        )
        log.info(
            "inferred_review: copy %d clip đã cắt + %d bản gốc vào %s",
            copied_clips,
            copied_orig,
            inferred_dir,
        )

    # --- Bước 6: dựng + ghi bảng nhãn mới (Req 7) ---
    if not dry_run:
        source_rows = read_source_labels(cfg)
        new_rows = build_new_rows(source_rows, clips, discovered_ids.keys(), logger=log)
        write_new_labels(new_rows, cfg, logger=log)

    # --- Bước 7: kế toán + ghi RunReport (Req 8.2) ---
    report = build_run_report(
        seg_results, skipped=skipped, manual_review_ids=manual_review_ids
    )
    log.info(
        "RunReport: total=%d single=%d multi=%d sub_clips=%d skipped=%d manual_review=%d",
        report.total_discovered,
        report.single_variant,
        report.multi_variant,
        report.sub_clips_generated,
        report.skipped,
        report.manual_review,
    )
    return report


def _copy_review_videos(video_ids, discovered_ids, dest_dir, label, log) -> int:
    """Copy file gốc của mỗi ``video_id`` vào *dest_dir* để con người xem lại.

    Dùng chung cho cả ``manual_review`` (cắt tay) lẫn ``inferred_review`` (tách
    bằng suy luận, cần kiểm ranh giới). File được **copy** (không move/hardlink)
    để giữ nguyên video nguồn và để folder tự chứa, dễ mở xem lại. Thư mục được
    tạo nếu thiếu. Lỗi copy từng file được ghi log và bỏ qua (không làm hỏng cả
    lần chạy); chỉ video đã thực sự duyệt được (có trong ``discovered_ids``) mới
    copy được.

    Args:
        video_ids: Tập ``video_id`` cần copy.
        discovered_ids: Mapping ``video_id -> VideoEntry`` (để lấy đường dẫn gốc).
        dest_dir: Thư mục đích (đường dẫn tuyệt đối trên ``D:``).
        label: Nhãn để ghi log ("manual_review" / "inferred_review").
        log: Logger của lần chạy.

    Returns:
        Số file đã copy thành công.
    """
    import shutil

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("Không tạo được thư mục %s %s: %s", label, dest_dir, exc)
        return 0

    copied = 0
    for video_id in sorted(video_ids):
        entry = discovered_ids.get(video_id)
        if entry is None:
            # Video không nằm trong tập đã duyệt (vd bị loại từ discovery) → bỏ qua.
            log.warning(
                "%s: không tìm thấy video gốc cho video_id=%s — bỏ qua copy.",
                label,
                video_id,
            )
            continue
        dest = dest_dir / f"{video_id}.mp4"
        try:
            shutil.copy2(entry.path, dest)
            copied += 1
        except OSError as exc:
            log.error("%s: lỗi copy %s -> %s: %s", label, entry.path, dest, exc)
    return copied


def _copy_inferred_clips(inferred_ids, clips, split_dir, dest_dir, log) -> int:
    """Copy các clip đã cắt của video inferred vào *dest_dir* để người review xem ngay.

    Chỉ copy clip thuộc video_id có trong *inferred_ids* (tức multi+inferred).
    File nguồn lấy từ *split_dir* (nơi write_clip vừa ghi ra).

    Returns:
        Số clip đã copy thành công.
    """
    import shutil

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("Không tạo được thư mục inferred_review %s: %s", dest_dir, exc)
        return 0

    copied = 0
    for clip in clips:
        if clip.video_id not in inferred_ids:
            continue
        src = split_dir / clip.out_filename
        if not src.is_file():
            log.warning("inferred_review: clip %s không tìm thấy tại %s — bỏ qua.", clip.out_filename, src)
            continue
        dest = dest_dir / clip.out_filename
        try:
            shutil.copy2(src, dest)
            copied += 1
        except OSError as exc:
            log.error("inferred_review: lỗi copy %s -> %s: %s", src, dest, exc)
    return copied


def _clip_span(clip, seg, props, cfg):
    """Tính :class:`VariantSpan` đã trừ Biên_An_Toàn cho *clip*.

    Với video một cách (``variant_index is None``) trả về span phủ toàn bộ video.
    Với video nhiều cách, áp :func:`trimmed_spans` lên toàn bộ span của video rồi
    chọn span ứng với ``clip.variant_index``; trả ``None`` nếu span đó bị thu còn
    ``< 1`` frame (Req 5.8).
    """
    from .segmenter import VariantSpan

    num_frames = int(props.num_frames)

    if clip.variant_index is None:
        # Video một cách: giữ nguyên toàn bộ frame (không cắt — Req 5.4).
        return VariantSpan(variant_index=1, start_frame=0, end_frame=max(0, num_frames - 1))

    trimmed = trimmed_spans(seg.spans, num_frames, cfg.safety_margin_frames)
    for span in trimmed:
        if span.variant_index == clip.variant_index:
            return span
    return None


def _build_config_from_args(args) -> PreprocessConfig:
    """Dựng :class:`PreprocessConfig` từ các tham số CLI (chỉ ghi đè khi được cấp)."""
    project_root = Path(args.project_root) if args.project_root else Path.cwd()
    overrides = {}
    if args.sample_interval is not None:
        overrides["sample_interval_seconds"] = args.sample_interval
    if args.safety_margin is not None:
        overrides["safety_margin_frames"] = args.safety_margin
    if args.ocr_threshold is not None:
        overrides["ocr_confidence_threshold"] = args.ocr_threshold
    return PreprocessConfig(project_root=project_root, **overrides)


def main(argv=None) -> int:
    """Điểm vào CLI. Trả về mã thoát (0 = thành công)."""
    parser = argparse.ArgumentParser(
        prog="qipedc_video_preprocess.preprocess",
        description="Tách video QIPEDC nhiều cách thành các video con (một cách/clip).",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Thư mục gốc dự án trên ổ D: (mặc định: thư mục hiện tại).",
    )
    parser.add_argument("--sample-interval", type=float, default=None,
                        help="Khoảng lấy mẫu frame (giây, 0.1–5.0).")
    parser.add_argument("--safety-margin", type=int, default=None,
                        help="Biên an toàn quanh ranh giới (frame, 0–60).")
    parser.add_argument("--ocr-threshold", type=float, default=None,
                        help="Ngưỡng tin cậy OCR (0.0–1.0).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ duyệt + phân đoạn, không ghi clip/nhãn.")
    args = parser.parse_args(argv)

    cfg = _build_config_from_args(args)
    report = run_preprocess(cfg, dry_run=args.dry_run)

    # Mã thoát 0 luôn (lỗi cục bộ không làm cả lần chạy fail); báo tóm tắt ra stdout.
    print(
        f"Done. total={report.total_discovered} single={report.single_variant} "
        f"multi={report.multi_variant} sub_clips={report.sub_clips_generated} "
        f"skipped={report.skipped} manual_review={report.manual_review}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

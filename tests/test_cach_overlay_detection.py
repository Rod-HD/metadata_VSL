"""Tests cho phát hiện nhãn "CÁCH" lệch/đè-logo và cờ ``saw_cach``.

Bối cảnh: ba video QIPEDC ("con chim" W00738, "con khỉ" W00767, "con sông"
W00792) có overlay "Cách 1/2" mà logo lệch hoặc con số đè lên biểu tượng cuốn
sách + nét mảnh, khiến OCR ROI hẹp bỏ sót → trước đây bị phân loại nhầm là
**single** (cắt sót cách 2). Bản sửa:

* :func:`number_detector.find_number_near_cach` dùng ROI nửa-trên-trái + neo vào
  token "CÁCH" (kể cả biến thể OCR mất dấu "Cacl"/"Cack"/"Bach") + zoom/đa-ngưỡng/
  aspect cho số mảnh, và trả thêm cờ ``saw_cach``.
* :class:`number_detector.DetectionResult` mang thêm ``saw_cach``.
* :func:`segmenter.classify_sequence` nhận ``saw_cach_flags``: nếu thấy "CÁCH"
  nhưng không dựng được multi hợp lệ thì → **manual_review** (KHÔNG âm thầm coi là
  single) — tránh cắt sót.

Các test này thuần logic (không chạy EasyOCR/video thật) nên nhanh và ổn định.
"""

from __future__ import annotations

from pathlib import Path

from qipedc_video_preprocess.config import PreprocessConfig
from qipedc_video_preprocess.number_detector import DetectionResult, _is_cach_token
from qipedc_video_preprocess.segmenter import classify_sequence

# project_root giả lập trên ổ D: (Req 1.3); classify_sequence chỉ đọc
# boundary_confirm_frames.
PROJECT_ROOT = Path("D:/projects/metadata_VSL")


def _cfg(confirm: int = 2) -> PreprocessConfig:
    return PreprocessConfig(project_root=PROJECT_ROOT, boundary_confirm_frames=confirm)


# --------------------------------------------------------------------------- #
# DetectionResult.saw_cach
# --------------------------------------------------------------------------- #
def test_detection_result_saw_cach_defaults_false():
    """Các nơi dựng DetectionResult cũ (không truyền saw_cach) vẫn hợp lệ."""
    r = DetectionResult(number=3, confidence=0.9)
    assert r.saw_cach is False


def test_detection_result_saw_cach_settable():
    r = DetectionResult(number=None, confidence=0.0, saw_cach=True)
    assert r.saw_cach is True and r.number is None


# --------------------------------------------------------------------------- #
# _is_cach_token — nhận dạng nhãn "CÁCH" qua các biến thể OCR thực tế
# --------------------------------------------------------------------------- #
def test_cach_token_matches_observed_ocr_variants():
    """Biến thể quan sát thực tế trên QIPEDC: Cach/Cacl/Cack/Bach + 'Cach 2'."""
    for tok in ("Cach", "cach", "Cách", "Cacl", "Cack", "Bach", "Cach 2", "cacl 1"):
        assert _is_cach_token(tok), tok


def test_cach_token_rejects_non_cach():
    """Không khớp logo/nhãn khác để tránh dương tính giả."""
    for tok in ("QIPEDC", "Q1PEDC", "con chim", "2", "", "ca", "back"):
        assert not _is_cach_token(tok), tok


# --------------------------------------------------------------------------- #
# classify_sequence — saw_cach điều khiển single ↔ manual_review
# --------------------------------------------------------------------------- #
def test_no_cach_no_number_stays_single():
    """Không thấy CÁCH, không số nào → video một-cách thật (hành vi cũ giữ nguyên)."""
    samples = [(0, None), (30, None), (60, None)]
    res = classify_sequence(
        samples, _cfg(), video_id="VID", saw_cach_flags=[False, False, False]
    )
    assert res.kind == "single"
    assert res.variant_count == 1


def test_saw_cach_but_no_number_is_manual_review():
    """Có frame thấy 'CÁCH' nhưng không đọc nổi số nào → rà soát thủ công.

    Đây chính là ca "con chim/sông" trước đây bị cắt sót: overlay nhiều-cách tồn
    tại nhưng OCR số thất bại. Không được coi là single.
    """
    samples = [(0, None), (30, None), (60, None)]
    res = classify_sequence(
        samples, _cfg(), video_id="VID", saw_cach_flags=[True, True, False]
    )
    assert res.kind == "manual_review"
    assert res.variant_count == 0
    assert res.spans == ()


def test_saw_cach_single_number_only_is_manual_review():
    """Thấy CÁCH và chỉ đọc được một giá trị (chỉ '1', thiếu '2') → manual_review.

    (Quy tắc 'chỉ một giá trị' vốn đã là manual_review; ở đây xác nhận saw_cach
    không phá vỡ điều đó.)
    """
    samples = [(0, 1), (30, 1), (60, 1)]
    res = classify_sequence(
        samples, _cfg(), video_id="VID", saw_cach_flags=[True, True, True]
    )
    assert res.kind == "manual_review"


def test_saw_cach_valid_multi_still_splits():
    """Thấy CÁCH và đọc được 1→2 hợp lệ (đủ xác nhận) → vẫn tách multi bình thường.

    Mô phỏng "con khỉ"/"con sông" sau khi sửa: 1,1,1,2,2,2.
    """
    samples = [(0, 1), (30, 1), (60, 1), (90, 2), (120, 2), (150, 2)]
    res = classify_sequence(
        samples,
        _cfg(confirm=2),
        video_id="VID",
        saw_cach_flags=[True] * 6,
    )
    assert res.kind == "multi"
    assert res.variant_count == 2
    # ranh giới tại frame mẫu đầu tiên mang '2'.
    assert res.spans[0].start_frame == 0
    assert res.spans[1].start_frame == 90


def test_single_number_blip_not_confirmed_is_manual_review():
    """'2' chỉ thoáng 1 frame lẻ (không đủ boundary_confirm_frames) → manual_review.

    Đây là ca "con chim" sau khi sửa: 1,1,1,2,1,1,1 — có thấy '2' nhưng không xác
    nhận đủ 2 frame liên tiếp nên KHÔNG tách tự động, đưa vào rà soát thủ công.
    """
    samples = [(0, 1), (30, 1), (60, 1), (90, 2), (120, 1), (150, 1), (180, 1)]
    res = classify_sequence(
        samples, _cfg(confirm=2), video_id="VID", saw_cach_flags=[True] * 7
    )
    assert res.kind == "manual_review"


def test_backward_compatible_without_flags():
    """Không truyền saw_cach_flags (mặc định None) → hành vi cũ: toàn None → single."""
    samples = [(0, None), (30, None)]
    res = classify_sequence(samples, _cfg(), video_id="VID")
    assert res.kind == "single"

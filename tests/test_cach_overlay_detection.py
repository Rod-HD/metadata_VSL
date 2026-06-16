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
from qipedc_video_preprocess.number_detector import (
    DetectionResult,
    _is_cach_token,
    _is_cach_fragment,
    _QIPEDC_NOISE_RE,
    _DIACRITIC_MAP,
)
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
    """Biến thể quan sát thực tế trên QIPEDC: Cach/Cacl/Cack/Bach + 'Cach 2'.

    D0105 vi+en reader: logo che → OCR đọc 'P█GH 1'/'c█CH 2' (block char █ giữa token).
    D0105 en-only reader: bọc trong dấu nháy → "'SacH 1'" (C→S, quoted).
    Sau strip non-alpha + strip nháy đầu/cuối → match nhánh [cpbs]..ch/gh hoặc [cs]ac..
    """
    for tok in ("Cach", "cach", "Cách", "Cacl", "Cack", "Bach", "Cach 2", "cacl 1",
                "P█GH 1", "c█CH 2", "c█GH 2",
                "'SacH 1'", "'SacH 2'", "'SAcH 2'"):
        assert _is_cach_token(tok), tok


def test_cach_token_rejects_non_cach():
    """Không khớp logo/nhãn khác để tránh dương tính giả."""
    for tok in ("QIPEDC", "Q1PEDC", "con chim", "2", "", "ca", "back"):
        assert not _is_cach_token(tok), tok


# --------------------------------------------------------------------------- #
# _is_cach_fragment — nhận dạng mảnh nhỏ của "CÁCH"
# --------------------------------------------------------------------------- #
def test_cach_fragment_accepts_full_token():
    """Token đầy đủ → fragment cũng True (superset)."""
    for tok in ("Cach", "Cacl", "Bach", "Cách"):
        assert _is_cach_fragment(tok), tok


def test_cach_fragment_accepts_partial():
    """Mảnh nhỏ 'ca', 'ch', 'á', ký tự có dấu → True."""
    for tok in ("ca", "ch", "á", "ă", "Ca", "CH"):
        assert _is_cach_fragment(tok), tok


def test_cach_fragment_rejects_unrelated():
    """Ký tự đơn số, QIPEDC, 'back' tiếng Anh → False."""
    for tok in ("2", "1", "QIPEDC", "back", "", "Q"):
        assert not _is_cach_fragment(tok), tok


# --------------------------------------------------------------------------- #
# _QIPEDC_NOISE_RE — lọc token nhiễu cố định của logo/QIPEDC
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    return text.strip().lower().translate(_DIACRITIC_MAP)


def test_qipedc_noise_re_matches_logo_tokens():
    """Token cố định của logo QIPEDC (xuất hiện cả ở video single) → bị lọc."""
    for tok in ("QIPEDC", "qipedc", "Q1PEDC", "l", "I", "|", "1", "o", "O", "0", "-", "_"):
        assert _QIPEDC_NOISE_RE.match(_norm(tok)), f"expected noise: {tok!r}"


def test_qipedc_noise_re_passes_overlay_tokens():
    """Token của overlay CÁCH (KHÔNG phải logo) → KHÔNG bị lọc bởi noise RE."""
    for tok in ("Cach", "ca", "ch", "á", "2", "3", "Bach", "Cacl"):
        # Token số đơn "2"/"3" không khớp noise RE (noise RE chỉ lọc "l"/"I"/"1"/"o"/"0")
        # → Pass 3 sẽ thấy chúng và bật saw_cach.
        # Ngoại trừ: "1" là nhiễu logo, nhưng digit_tokens xử lý riêng qua allowlist.
        if tok in ("2", "3", "ca", "ch", "á", "Bach", "Cacl", "Cach"):
            assert not _QIPEDC_NOISE_RE.match(_norm(tok)), f"should not be noise: {tok!r}"


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


# --------------------------------------------------------------------------- #
# SUY LUẬN theo chuỗi: khôi phục dãy 1,2,… khi OCR đọc thiếu/nhiễu
# --------------------------------------------------------------------------- #
def test_infer_missing_first_variant():
    """Chỉ đọc được '2' (ổn định), '1' toàn None → KHÔNG suy ngược (D0105 thực tế).

    Một khối ổn định duy nhất không đủ tín hiệu để xác định ranh giới cắt:
    đoạn None phía trước có thể là overlay chưa rõ ràng, không phải cách riêng.
    → manual_review để người xem xét.
    """
    samples = [(0, None), (30, None), (60, None), (90, None), (120, 2), (150, 2), (180, 2)]
    res = classify_sequence(
        samples, _cfg(confirm=2), video_id="D0105", saw_cach_flags=[True] * 7
    )
    assert res.kind == "manual_review"
    assert res.inferred is False


def test_infer_two_to_three_recovers_full_sequence():
    """Hai khối khác giá trị (None rồi '3') → KHÔNG đủ vì None không phải khối số.

    Chỉ có 1 khối số ổn định ('3') → manual_review, không suy luận ranh giới.
    """
    samples = [(0, None), (30, None), (60, 3), (90, 3), (120, 3)]
    res = classify_sequence(
        samples, _cfg(confirm=2), video_id="VID", saw_cach_flags=[True] * 5
    )
    assert res.kind == "manual_review"
    assert res.inferred is False


def test_clean_sequence_is_multi_not_inferred():
    """OCR đọc trọn 1→2 hợp lệ → multi BÌNH THƯỜNG (inferred=False, không cần suy luận)."""
    samples = [(0, 1), (30, 1), (60, 1), (90, 2), (120, 2), (150, 2)]
    res = classify_sequence(
        samples, _cfg(confirm=2), video_id="VID", saw_cach_flags=[True] * 6
    )
    assert res.kind == "multi"
    assert res.inferred is False


def test_no_stable_transition_falls_to_manual_review():
    """Có CÁCH nhưng KHÔNG có khối chuyển ổn định ('2' lẻ giữa toàn '1') → manual_review.

    Ca "con chim" W00738: số '2' bị đọc nhầm thành '1' gần hết, chỉ lẻ 1 frame →
    không suy luận nổi → cắt tay.
    """
    samples = [(0, 1), (30, 1), (60, 1), (90, 2), (120, 1), (150, 1), (180, 1)]
    res = classify_sequence(
        samples, _cfg(confirm=2), video_id="W00738", saw_cach_flags=[True] * 7
    )
    assert res.kind == "manual_review"
    assert res.inferred is False


def test_inference_requires_cach():
    """Không có CÁCH thì KHÔNG suy luận (dù chuỗi giống) — tránh tách bừa video 1-cách.

    (Trên thực tế video một-cách không có overlay nên saw_cach toàn False.)
    """
    samples = [(0, None), (30, None), (60, 2), (90, 2), (120, 2)]
    res = classify_sequence(
        samples, _cfg(confirm=2), video_id="VID", saw_cach_flags=[False] * 5
    )
    # Không CÁCH + chuỗi chỉ '2' (thiếu 1) → manual_review theo luật cũ, KHÔNG infer.
    assert res.kind == "manual_review"
    assert res.inferred is False

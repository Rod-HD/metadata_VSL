"""Integration tests cho ``qipedc_video_preprocess.preprocess`` — Bộ_Tiền_Xử_Lý.

Chạy toàn bộ pipeline ``run_preprocess`` với **fake injectables** (không cần video
thật, OpenCV codec, hay trọng số OCR):

* Fake :class:`NumberDetector` đổi số tại frame ``T`` xác định → tạo video nhiều
  cách có ranh giới biết trước.
* Fake ``cv2.VideoCapture`` / ``cv2.VideoWriter`` (monkeypatch trong các module
  ``discovery``, ``segmenter``, ``splitter`` và ``video_probe``) ghi nhận đầu ra.
* Fake logger ném lỗi để kiểm tra hành vi **dừng** khi ghi log thất bại.

Phủ:

* **Req 1.2** — clip ghi vào ``Dataset/processed_videos/split_variants/``.
* **Req 1.5** — không tạo/sửa file trong ``src/qipedc2vsl400/`` hay
  ``Dataset/final_dataset/``.
* **Req 5.6** — lỗi ghi một clip → log + tiếp tục (lần chạy không hỏng).
* **Req 8.6** — logger ném lỗi → ``run_preprocess`` dừng (raise), không nuốt.

``project_root`` là một thư mục tạm; ``tempfile.gettempdir()`` trên máy này nằm
trên ổ ``D:`` nên ``cfg.validate()`` (Req 1.3) đạt.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import openpyxl
import pytest

from qipedc_video_preprocess import preprocess
from qipedc_video_preprocess.config import PreprocessConfig
from qipedc_video_preprocess.label_writer import COLUMN_HEADERS
from qipedc_video_preprocess.number_detector import DetectionResult


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]


def _logger() -> tuple[logging.Logger, CapturingHandler]:
    handler = CapturingHandler()
    logger = logging.getLogger(f"test_preprocess_integration.{id(handler)}")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, handler


class FakeStepDetector:
    """Detector trả số theo frame: số 1 trước frame T, số 2 từ T trở đi.

    "frame" truyền vào ``detect`` chính là chỉ số frame (do FakeCapture trả về),
    nên video có đúng một ranh giới tại frame ``T`` → 2 cách.
    """

    def __init__(self, transition_frame: int) -> None:
        self._t = transition_frame

    def detect(self, frame) -> DetectionResult:
        number = 1 if int(frame) < self._t else 2
        return DetectionResult(number=number, confidence=1.0)


class FakeCapture:
    """Fake ``cv2.VideoCapture``: read() trả (True, <frame_index>) tuần tự."""

    def __init__(self, path, total: int = 40) -> None:
        self.path = str(path)
        self._pos = 0
        self._total = total

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def get(self, prop) -> float:
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return 1.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._total)
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 320.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 240.0
        return 0.0

    def set(self, prop, value) -> bool:  # noqa: A003
        self._pos = int(value)
        return True

    def grab(self) -> bool:
        if self._pos >= self._total:
            return False
        self._pos += 1
        return True

    def read(self):
        if self._pos >= self._total:
            return False, None
        frame = self._pos
        self._pos += 1
        return True, frame

    def release(self) -> None:
        pass


class FakeWriter:
    instances: list["FakeWriter"] = []

    def __init__(self, path, fourcc, fps, size, fail: bool = False) -> None:
        self.path = str(path)
        self.fps = fps
        self.size = size
        self._fail = fail
        self.frames: list[int] = []
        FakeWriter.instances.append(self)

    def isOpened(self) -> bool:  # noqa: N802
        return not self._fail

    def write(self, frame) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        pass


def _install_cv2_fakes(monkeypatch, *, writer_fail: bool = False, total: int = 40):
    """Monkeypatch cv2 trong mọi module dùng nó cho pipeline."""
    import cv2

    from qipedc_video_preprocess import discovery, segmenter, video_probe

    FakeWriter.instances.clear()

    def _capture(path):
        return FakeCapture(path, total=total)

    def _writer(path, fourcc, fps, size):
        return FakeWriter(path, fourcc, fps, size, fail=writer_fail)

    # cv2.VideoCapture được dùng qua: discovery.is_readable, video_probe.probe,
    # segmenter.segment_video (import trễ -> patch trên module cv2), splitter.write_clip.
    monkeypatch.setattr(cv2, "VideoCapture", _capture)
    monkeypatch.setattr(cv2, "VideoWriter", _writer)
    monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *a: 0, raising=False)
    # discovery import cv2 ở cấp module:
    monkeypatch.setattr(discovery.cv2, "VideoCapture", _capture)


# --------------------------------------------------------------------------- #
# Dựng cây dự án tạm với video & bảng nhãn nguồn
# --------------------------------------------------------------------------- #
def _make_project(tmp: Path, video_ids: list[str]) -> PreprocessConfig:
    """Tạo cây dự án tối thiểu: raw_videos/*.mp4 (rỗng — nội dung do fake cung cấp)
    + một bảng nhãn nguồn cho từng video_id."""
    raw = tmp / "Dataset" / "raw_videos"
    raw.mkdir(parents=True, exist_ok=True)
    for vid in video_ids:
        (raw / f"{vid}.mp4").write_bytes(b"")  # placeholder; fake capture bỏ qua nội dung

    labels_dir = tmp / "Dataset" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(list(COLUMN_HEADERS))
    for i, vid in enumerate(video_ids, start=1):
        ws.append([str(i), f"ID{i}", f"{vid}.mp4", "bắt tay", "Bắc", "chào hỏi", "s1"])
    wb.save(str(labels_dir / "batch_1.xlsx"))
    wb.close()

    return PreprocessConfig(project_root=tmp)


# --------------------------------------------------------------------------- #
# Req 1.2 / 1.5 — ghi đúng split_variants/, không đụng các vùng cấm
# --------------------------------------------------------------------------- #
def test_run_preprocess_writes_to_split_variants(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        cfg = _make_project(tmp, ["W001"])
        _install_cv2_fakes(monkeypatch, total=40)
        logger, handler = _logger()

        # Detector đổi số tại frame 20 → 2 cách.
        report = preprocess.run_preprocess(
            cfg, detector=FakeStepDetector(20), logger=logger
        )

        split_dir = tmp / "Dataset" / "processed_videos" / "split_variants"

        # Clip được ghi (qua FakeWriter) đúng vào split_variants/ (Req 1.2).
        assert FakeWriter.instances, "không có clip nào được ghi"
        for w in FakeWriter.instances:
            assert "split_variants" in w.path.replace("\\", "/")

        # Bảng nhãn mới nằm trong split_variants/, không đụng Dataset/labels/.
        new_labels = split_dir / "labels_split.xlsx"
        assert new_labels.exists()

        # Req 1.5: không tạo qipedc2vsl400/ hay final_dataset/ dưới project tạm.
        assert not (tmp / "src" / "qipedc2vsl400").exists()
        assert not (tmp / "Dataset" / "final_dataset").exists()

        # Phân loại đúng: 1 video nhiều cách.
        assert report.multi_variant == 1
        assert report.total_discovered == 1


def test_run_preprocess_single_variant_video(monkeypatch):
    """Detector luôn trả 'không có số' → video một cách, giữ nguyên tên."""

    class _NoNumberDetector:
        def detect(self, frame):
            return DetectionResult(number=None, confidence=0.0)

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        cfg = _make_project(tmp, ["W050"])
        _install_cv2_fakes(monkeypatch, total=20)
        logger, _ = _logger()

        report = preprocess.run_preprocess(
            cfg, detector=_NoNumberDetector(), logger=logger
        )

        assert report.single_variant == 1
        assert report.multi_variant == 0
        # Tên file đầu ra giữ nguyên <id>.mp4 (không hậu tố _c).
        assert any(w.path.endswith("W050.mp4") for w in FakeWriter.instances)


# --------------------------------------------------------------------------- #
# Req 5.6 — lỗi ghi một clip → log + tiếp tục, lần chạy không hỏng
# --------------------------------------------------------------------------- #
def test_run_preprocess_clip_write_failure_continues(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        cfg = _make_project(tmp, ["W001"])
        _install_cv2_fakes(monkeypatch, writer_fail=True, total=40)
        logger, handler = _logger()

        # writer_fail=True → mọi write_clip trả False; lần chạy vẫn hoàn tất.
        report = preprocess.run_preprocess(
            cfg, detector=FakeStepDetector(20), logger=logger
        )

        # Không ném; report vẫn trả về với video bị đánh dấu rà soát thủ công.
        assert report.total_discovered == 1
        assert report.manual_review >= 1
        # Lỗi ghi clip được log (Req 5.6).
        assert any("clip" in m.lower() for m in handler.messages())


# --------------------------------------------------------------------------- #
# Req 8.6 — logger ném lỗi → dừng (raise), không nuốt
# --------------------------------------------------------------------------- #
def test_run_preprocess_logger_failure_propagates(monkeypatch):
    class ExplodingLogger:
        def warning(self, *a, **k):
            raise OSError("log write failed")

        def info(self, *a, **k):
            raise OSError("log write failed")

        def error(self, *a, **k):
            raise OSError("log write failed")

        def debug(self, *a, **k):
            raise OSError("log write failed")

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        # Không có video nào → run_preprocess gọi logger.warning("Không có video...")
        # → ExplodingLogger ném OSError, không nuốt (Req 8.6).
        cfg = _make_project(tmp, [])
        _install_cv2_fakes(monkeypatch, total=10)

        with pytest.raises(OSError, match="log write failed"):
            preprocess.run_preprocess(
                cfg, detector=FakeStepDetector(5), logger=ExplodingLogger()
            )


# --------------------------------------------------------------------------- #
# Req 1.4 — cấu hình sai → dừng trước khi xử lý, không tạo đầu ra
# --------------------------------------------------------------------------- #
def test_run_preprocess_invalid_config_aborts(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        cfg = _make_project(tmp, ["W001"])
        # Ép một đường dẫn đầu ra trỏ sang ổ C: → validate() lỗi (Req 1.4).
        bad_cfg = PreprocessConfig(
            project_root=tmp,
            split_output_dir="C:/forbidden/split",
        )
        _install_cv2_fakes(monkeypatch, total=40)

        report = preprocess.run_preprocess(
            bad_cfg, detector=FakeStepDetector(20), logger=None
        )

        # Không xử lý video nào; không clip nào được ghi.
        assert report.total_discovered == 0
        assert FakeWriter.instances == []
        # Lỗi cấu hình được báo ra stderr.
        captured = capsys.readouterr()
        assert "config error" in captured.err.lower()

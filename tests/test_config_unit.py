"""Unit tests for ``qipedc_video_preprocess.config`` — defaults & isolation.

Phủ hai khía cạnh của Requirement 1:

* **Req 1.6** — mọi đường dẫn đầu ra mặc định resolve về ổ ``D:`` (không bao giờ
  ``C:``) khi ``project_root`` nằm trên ổ ``D:``.
* **Req 1.1** — công đoạn tách video được cô lập trong package
  ``qipedc_video_preprocess`` và **không** import/tham chiếu hay đặt module bên
  trong package ``qipedc2vsl400``.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

from qipedc_video_preprocess.config import PreprocessConfig

# project_root giả lập nằm trên ổ D: (test thuần logic đường dẫn, không chạm đĩa).
PROJECT_ROOT = Path("D:/projects/metadata_VSL")

# Thư mục package nguồn của công đoạn và của pipeline metadata hiện có.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
_PREPROCESS_PKG_DIR = _SRC_DIR / "qipedc_video_preprocess"
_LEGACY_PKG_DIR = _SRC_DIR / "qipedc2vsl400"

# Tên các module thuộc công đoạn tách video (theo design.md / tasks.md).
_PREPROCESS_MODULE_NAMES = {
    "config",
    "discovery",
    "number_detector",
    "video_probe",
    "segmenter",
    "splitter",
    "label_writer",
    "run_logger",
    "preprocess",
}


def make_cfg() -> PreprocessConfig:
    return PreprocessConfig(project_root=PROJECT_ROOT)


# --------------------------------------------------------------------------- #
# Req 1.6 — mọi đường dẫn đầu ra mặc định resolve trên ổ D:
# --------------------------------------------------------------------------- #
def test_default_output_paths_resolve_on_d_drive():
    """Mọi đường dẫn đầu ra mặc định phải resolve lên ổ D:, không bao giờ C:."""
    cfg = make_cfg()
    output_paths = {
        "split_output_dir": cfg.resolve(cfg.split_output_dir),
        "new_labels_path": cfg.resolve(cfg.new_labels_path),
        "log_dir": cfg.resolve(cfg.log_dir),
        "ocr_models_dir": cfg.resolve(cfg.ocr_models_dir),
    }
    for name, path in output_paths.items():
        assert path.is_absolute(), f"{name} phải là đường dẫn tuyệt đối"
        assert path.drive.upper() == "D:", f"{name} phải nằm trên ổ D: (được {path.drive!r})"
        assert path.drive.upper() != "C:", f"{name} không được nằm trên ổ C:"


def test_default_output_paths_under_project_root():
    """Mọi đường dẫn đầu ra mặc định nằm bên trong cây project_root."""
    cfg = make_cfg()
    root = PROJECT_ROOT.resolve()
    for raw in (cfg.split_output_dir, cfg.new_labels_path, cfg.log_dir, cfg.ocr_models_dir):
        resolved = cfg.resolve(raw)
        # Không ném ValueError nghĩa là nằm trong cây project_root.
        resolved.relative_to(root)


def test_convenience_path_properties_on_d_drive():
    """Các property tiện ích cũng resolve lên ổ D: tại đúng vị trí mong đợi."""
    cfg = make_cfg()
    root = PROJECT_ROOT.resolve()
    assert cfg.split_output_path == (root / cfg.split_output_dir).resolve()
    assert cfg.new_labels_full_path == (root / cfg.new_labels_path).resolve()
    assert cfg.log_path == (root / cfg.log_dir).resolve()
    assert cfg.ocr_models_path == (root / cfg.ocr_models_dir).resolve()
    for path in (cfg.split_output_path, cfg.new_labels_full_path, cfg.log_path, cfg.ocr_models_path):
        assert path.drive.upper() == "D:"


def test_default_config_validates_clean_on_d_drive():
    """Cấu hình mặc định trên ổ D: không có lỗi (tất cả output trên D:, trong cây)."""
    cfg = make_cfg()
    assert cfg.validate() == []


def test_video_search_paths_resolve_on_d_drive():
    cfg = make_cfg()
    paths = cfg.video_search_paths()
    assert len(paths) == len(cfg.video_search_dirs)
    assert all(p.drive.upper() == "D:" for p in paths)


# --------------------------------------------------------------------------- #
# Req 1.1 — cô lập: không import/tham chiếu/đặt module trong qipedc2vsl400
# --------------------------------------------------------------------------- #
def _imported_module_names(source: str) -> set[str]:
    """Trích tên module gốc từ mọi câu lệnh import trong mã nguồn."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                names.add(node.module.split(".")[0])
    return names


def test_config_module_does_not_import_legacy_package():
    """config.py không import qipedc2vsl400 (phân tích AST)."""
    source = (_PREPROCESS_PKG_DIR / "config.py").read_text(encoding="utf-8")
    assert "qipedc2vsl400" not in _imported_module_names(source)


def test_config_module_source_has_no_legacy_reference():
    """config.py không nhắc tới qipedc2vsl400 trong bất kỳ câu lệnh import nào."""
    source = (_PREPROCESS_PKG_DIR / "config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            segment = ast.get_source_segment(source, node) or ""
            assert "qipedc2vsl400" not in segment


@pytest.mark.parametrize("module_name", sorted(_PREPROCESS_MODULE_NAMES))
def test_no_preprocess_module_imports_legacy_package(module_name):
    """Không module nào của công đoạn import qipedc2vsl400 (cô lập, Req 1.1)."""
    module_file = _PREPROCESS_PKG_DIR / f"{module_name}.py"
    if not module_file.exists():
        pytest.skip(f"{module_name}.py chưa được hiện thực")
    source = module_file.read_text(encoding="utf-8")
    assert "qipedc2vsl400" not in _imported_module_names(source), (
        f"{module_name}.py không được import qipedc2vsl400"
    )


def test_importing_config_does_not_load_legacy_package():
    """Import config (mới) không kéo theo qipedc2vsl400 vào sys.modules."""
    # Loại bỏ mọi tàn dư để phép kiểm tra phản ánh đúng quan hệ phụ thuộc.
    for name in list(sys.modules):
        if name == "qipedc2vsl400" or name.startswith("qipedc2vsl400."):
            del sys.modules[name]
    for name in list(sys.modules):
        if name == "qipedc_video_preprocess" or name.startswith("qipedc_video_preprocess."):
            del sys.modules[name]

    importlib.import_module("qipedc_video_preprocess.config")

    leaked = [n for n in sys.modules if n == "qipedc2vsl400" or n.startswith("qipedc2vsl400.")]
    assert leaked == [], f"Import config không được nạp qipedc2vsl400: {leaked}"


def test_preprocess_modules_live_in_dedicated_package():
    """Mọi module công đoạn được khai báo trong package qipedc_video_preprocess (Req 1.1)."""
    assert _PREPROCESS_PKG_DIR.is_dir(), "package qipedc_video_preprocess phải tồn tại"
    for module_name in _PREPROCESS_MODULE_NAMES:
        assert (_PREPROCESS_PKG_DIR / f"{module_name}.py").exists(), (
            f"{module_name}.py phải nằm trong qipedc_video_preprocess/"
        )


def test_preprocess_specific_modules_absent_from_legacy_package():
    """Các module đặc thù công đoạn (vd splitter, segmenter) không được đặt trong qipedc2vsl400/."""
    if not _LEGACY_PKG_DIR.is_dir():
        pytest.skip("qipedc2vsl400 không tồn tại")
    # 'config' và 'video_probe' là tên chung tồn tại hợp lệ và độc lập ở cả hai
    # package (mỗi package có bản hiện thực riêng); việc cô lập thực sự (không
    # import chéo) đã được các test quét AST ở trên bảo đảm. Ở đây chỉ kiểm các
    # module mang tính đặc thù của công đoạn tách video.
    preprocess_specific = _PREPROCESS_MODULE_NAMES - {"config", "video_probe"}
    for module_name in preprocess_specific:
        assert not (_LEGACY_PKG_DIR / f"{module_name}.py").exists(), (
            f"{module_name}.py không được khai báo bên trong qipedc2vsl400/"
        )


def test_legacy_package_does_not_reference_preprocess_package():
    """Package qipedc2vsl400 không tham chiếu công đoạn mới (cô lập hai chiều)."""
    if not _LEGACY_PKG_DIR.is_dir():
        pytest.skip("qipedc2vsl400 không tồn tại")
    for py_file in _LEGACY_PKG_DIR.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        assert "qipedc_video_preprocess" not in _imported_module_names(source), (
            f"{py_file.name} (legacy) không được import qipedc_video_preprocess"
        )

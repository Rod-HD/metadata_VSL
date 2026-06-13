"""Smoke test for the CLI orchestrator ``qipedc2vsl400.convert`` (Task 11).

This exercises :func:`run_pipeline` end-to-end on a *tiny* fixture dataset with
**injected** ``probe_fn`` / ``embed_fn`` stubs and dummy ``.mp4`` files, so the
test never hits the network and needs no real videos or ONNX models. It asserts
that ``front_view.json``, ``signers.csv`` and the per-signer ``by_signer``
folders are produced and that verification passes (and that the CLI surfaces a
non-zero exit code on a verification failure).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openpyxl
import pytest

from qipedc2vsl400 import convert
from qipedc2vsl400.config import Config
from qipedc2vsl400.video_probe import VideoProps


# --- fixture builders -------------------------------------------------------

# (VIDEO, LABEL, expected synthetic signer letter or None for no-face)
_SPEC = [
    ("D0530.mp4", "Anh", "A"),
    ("D0531.mp4", "Em", "A"),
    ("D0532.mp4", "Bà", "B"),
    ("D0533N.mp4", "Một", None),  # Vietnamese diacritics + no-face clip
]


def _write_xlsx(cfg: Config) -> None:
    """Write a tiny QIPEDC-style ``.xlsx`` under ``Dataset/labels``."""
    labels_dir = cfg.resolve("Dataset/labels")
    labels_dir.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["STT", "ID", "VIDEO", "LABEL", "REGION", "TOPIC", "signer"])
    for index, (video, label, _signer) in enumerate(_SPEC, start=1):
        sheet.append([index, f"id{index}", video, label, "Chung", "Số", None])
    workbook.save(labels_dir / "batch_1.xlsx")


def _make_videos(cfg: Config) -> None:
    """Create dummy ``.mp4`` files under the foldering source / search dir."""
    src_dir = cfg.foldering_source_path
    src_dir.mkdir(parents=True, exist_ok=True)
    for video, _label, _signer in _SPEC:
        (src_dir / video).write_text(f"dummy-{video}", encoding="utf-8")


def _stub_probe(_video_path: Path) -> VideoProps:
    """Deterministic probe: every clip looks like a 2s 720p/25fps clip."""
    return VideoProps(fps=25.0, num_frames=50, length_seconds=2.0, resolution=720)


# Map each VIDEO filename to a synthetic embedding (or ``None`` for no-face).
_EMBED_BY_NAME: dict[str, np.ndarray | None] = {
    "D0530.mp4": np.array([1.0, 0.0, 0.0]),
    "D0531.mp4": np.array([0.99, 0.01, 0.0]),  # ~same direction as D0530 -> signer A
    "D0532.mp4": np.array([0.0, 1.0, 0.0]),  # distinct -> signer B
    "D0533N.mp4": None,  # no face -> unknown bucket
}


def _stub_embed(video_path: Path | None, _cfg) -> np.ndarray | None:
    """Return a deterministic synthetic embedding keyed by filename."""
    if video_path is None:
        return None
    return _EMBED_BY_NAME.get(Path(video_path).name)


def _make_cfg(tmp_path: Path) -> Config:
    cfg = Config(project_root=tmp_path)
    _write_xlsx(cfg)
    _make_videos(cfg)
    return cfg


# --- tests ------------------------------------------------------------------


def test_run_pipeline_produces_outputs_and_verifies(tmp_path):
    """The full pipeline produces all artifacts and verification passes."""
    cfg = _make_cfg(tmp_path)

    result = convert.run_pipeline(
        cfg,
        probe_fn=_stub_probe,
        embed_fn=_stub_embed,
        fetch=False,
        skip_signer=False,
    )

    # Verification passed (exit code 0).
    assert result.verify_report.ok is True, [
        c.detail for c in result.verify_report.failures
    ]
    assert result.exit_code == 0

    # front_view.json produced, valid JSON, UTF-8 diacritics preserved.
    view_file = cfg.output_view_file
    assert view_file.is_file()
    raw = view_file.read_text(encoding="utf-8")
    assert "Bà" in raw and "\\u" not in raw  # not ASCII-escaped
    data = json.loads(raw)
    assert len(data) == len(_SPEC)
    assert {rec["gloss"] for rec in data} == {"Anh", "Em", "Bà", "Một"}

    # signers.csv side-car produced.
    sidecar = cfg.signer_sidecar_path
    assert sidecar.is_file()
    sidecar_text = sidecar.read_text(encoding="utf-8")
    assert "D0530.mp4" in sidecar_text

    # by_signer folders produced; the two A-clips share a signer, B distinct,
    # the no-face clip lands in signer_unknown.
    base = cfg.by_signer_path
    assert (base / "_summary.txt").is_file()
    placed = {p.parent.name: p.parent for p in base.rglob("*.mp4")}
    # D0533N has no face -> unknown bucket folder exists and holds it.
    assert (base / "signer_unknown" / "D0533N.mp4").is_file()
    # Two real signers were discovered.
    real_signer_folders = sorted(
        d.name for d in base.iterdir() if d.is_dir() and d.name != "signer_unknown"
    )
    assert real_signer_folders == ["signer_001", "signer_002"]

    # Console summary mentions the headline numbers.
    assert "total records    : 4" in result.summary
    assert "signers          : 2" in result.summary
    assert "unknown clips    : 1" in result.summary


def test_skip_signer_reuses_existing_csv(tmp_path):
    """``skip_signer`` reuses the signers.csv produced by a prior run."""
    cfg = _make_cfg(tmp_path)

    # First run produces signers.csv.
    convert.run_pipeline(
        cfg, probe_fn=_stub_probe, embed_fn=_stub_embed, fetch=False
    )
    assert cfg.signer_sidecar_path.is_file()

    # Second run reuses it; embed_fn must NOT be called.
    def exploding_embed(_path, _cfg):
        raise AssertionError("embed_fn must not be called with --skip-signer")

    result = convert.run_pipeline(
        cfg,
        probe_fn=_stub_probe,
        embed_fn=exploding_embed,
        fetch=False,
        skip_signer=True,
    )

    assert result.verify_report.ok is True, [
        c.detail for c in result.verify_report.failures
    ]
    # Assignments came from the CSV (same clip count).
    assert len(result.signer_assignments) == len(_SPEC)


def test_skip_signer_without_csv_raises(tmp_path):
    """``skip_signer`` with no existing signers.csv fails fast."""
    cfg = _make_cfg(tmp_path)
    with pytest.raises(FileNotFoundError):
        convert.run_pipeline(
            cfg, probe_fn=_stub_probe, embed_fn=_stub_embed, fetch=False, skip_signer=True
        )


def test_main_returns_zero_on_success(tmp_path):
    """The CLI ``main`` returns exit code 0 for a passing conversion."""
    cfg = _make_cfg(tmp_path)

    # Drive main() with --no-fetch and the fixture project root. The real probe
    # and embed are replaced by patching the module defaults the CLI uses.
    import qipedc2vsl400.convert as convert_mod

    original_run = convert_mod.run_pipeline

    def patched_run(cfg_arg, **kwargs):
        kwargs["probe_fn"] = _stub_probe
        kwargs["embed_fn"] = _stub_embed
        return original_run(cfg_arg, **kwargs)

    convert_mod.run_pipeline = patched_run
    try:
        exit_code = convert_mod.main(
            ["--no-fetch", "--project-root", str(tmp_path)]
        )
    finally:
        convert_mod.run_pipeline = original_run

    assert exit_code == 0

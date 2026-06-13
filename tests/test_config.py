"""Unit tests for ``qipedc2vsl400.config``."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from qipedc2vsl400.config import Config

PROJECT_ROOT = Path("D:/projects/metadata_VSL")


def make_cfg() -> Config:
    return Config(project_root=PROJECT_ROOT)


def test_default_values_match_design():
    cfg = make_cfg()
    assert cfg.qipedc_labels_glob == "Dataset/labels/*.xlsx"
    assert cfg.video_search_dirs == (
        "Dataset/processed_videos/resize_720p",
        "Dataset/raw_videos",
    )
    assert cfg.vsl400_dir == "Dataset/labels/vsl400"
    assert cfg.output_dir == "Dataset/final_dataset"
    assert cfg.output_view_name == "front_view"
    assert cfg.log_dir == "Dataset/logs"
    assert cfg.video_id_keep_extension is False
    assert cfg.signer_id_width == 3
    assert cfg.on_missing_video == "skip"
    assert cfg.zenodo_record_id == "17943574"
    assert cfg.models_dir == "Dataset/models"
    assert cfg.signer_frames_per_video == 5
    assert cfg.signer_cosine_threshold == pytest.approx(0.363)
    assert cfg.signer_unknown_label == "unknown"
    assert cfg.signer_sidecar == "signers.csv"
    assert cfg.by_signer_dir == "Dataset/by_signer"
    assert cfg.foldering_source == "Dataset/processed_videos/resize_720p"
    assert cfg.foldering_copy_mode == "hardlink"


def test_config_is_frozen():
    cfg = make_cfg()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.output_dir = "elsewhere"  # type: ignore[misc]


def test_resolve_joins_relative_paths_onto_project_root():
    cfg = make_cfg()
    resolved = cfg.resolve("Dataset/final_dataset")
    assert resolved == (PROJECT_ROOT / "Dataset/final_dataset").resolve()


def test_resolve_returns_absolute_paths_on_d_drive():
    cfg = make_cfg()
    for path in (
        cfg.resolve(cfg.output_dir),
        cfg.resolve(cfg.log_dir),
        cfg.resolve(cfg.models_dir),
        cfg.resolve(cfg.by_signer_dir),
        cfg.resolve(cfg.vsl400_dir),
    ):
        assert path.is_absolute()
        # Never resolve onto the C: drive (Requirement 1).
        assert path.drive.upper() != "C:"
        assert path.drive.upper() == "D:"


def test_resolve_keeps_absolute_input_unchanged():
    cfg = make_cfg()
    abs_in = Path("D:/some/other/place")
    assert cfg.resolve(str(abs_in)) == abs_in.resolve()


def test_output_view_file_uses_view_name():
    cfg = make_cfg()
    assert cfg.output_view_file == (PROJECT_ROOT / "Dataset/final_dataset/front_view.json").resolve()


def test_signer_sidecar_path_under_output_dir():
    cfg = make_cfg()
    assert cfg.signer_sidecar_path == (
        PROJECT_ROOT / "Dataset/final_dataset/signers.csv"
    ).resolve()


def test_video_search_paths_resolved():
    cfg = make_cfg()
    paths = cfg.video_search_paths()
    assert len(paths) == 2
    assert paths[0] == (PROJECT_ROOT / "Dataset/processed_videos/resize_720p").resolve()
    assert all(p.drive.upper() == "D:" for p in paths)


def test_convenience_path_properties():
    cfg = make_cfg()
    assert cfg.vsl400_path == (PROJECT_ROOT / "Dataset/labels/vsl400").resolve()
    assert cfg.output_path == (PROJECT_ROOT / "Dataset/final_dataset").resolve()
    assert cfg.log_path == (PROJECT_ROOT / "Dataset/logs").resolve()
    assert cfg.models_path == (PROJECT_ROOT / "Dataset/models").resolve()
    assert cfg.by_signer_path == (PROJECT_ROOT / "Dataset/by_signer").resolve()
    assert cfg.foldering_source_path == (
        PROJECT_ROOT / "Dataset/processed_videos/resize_720p"
    ).resolve()

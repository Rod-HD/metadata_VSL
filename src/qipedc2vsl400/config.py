"""Configuration for the QIPEDC -> VSL400 metadata conversion pipeline.

The :class:`Config` dataclass is the single source of truth for every path and
tunable option used by the pipeline. All paths are expressed *relative to the
project root* and resolved to absolute paths on ``D:`` via the
:meth:`Config.resolve` helper, keeping every artifact off the ``C:`` drive
(Requirement 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Immutable pipeline configuration.

    Attributes mirror the design document. ``project_root`` is the only required
    field; everything else has a sensible default so a bare
    ``Config(project_root=Path("D:/projects/metadata_VSL"))`` is fully
    usable.
    """

    project_root: Path

    # --- QIPEDC source labels (Req 3) ---
    qipedc_labels_glob: str = "Dataset/labels/*.xlsx"

    # --- video discovery (Req 4) ---
    video_search_dirs: tuple[str, ...] = (
        "Dataset/processed_videos/resize_720p",
        "Dataset/raw_videos",
    )

    # --- VSL400 reference labels (Req 2) ---
    vsl400_dir: str = "Dataset/labels/vsl400"

    # --- output (Req 5) ---
    output_dir: str = "Dataset/final_dataset"
    output_view_name: str = "front_view"  # OQ2
    log_dir: str = "Dataset/logs"

    # --- mapping options (Req 4) ---
    video_id_keep_extension: bool = False  # False -> stem "D0530" (OQ1)
    signer_id_width: int = 3
    on_missing_video: str = "skip"  # "skip" | "placeholder" (OQ3)
    zenodo_record_id: str = "17943574"

    # --- signer extraction (Req 8) ---
    models_dir: str = "Dataset/models"  # ONNX face models live here, on D:
    signer_frames_per_video: int = 5  # frames sampled per clip for face embedding
    signer_cosine_threshold: float = 0.363  # SFace "same identity" cosine threshold
    signer_unknown_label: str = "unknown"  # bucket when no face detected
    signer_sidecar: str = "signers.csv"  # per-clip VIDEO -> signer_id + distance
    signer_embeddings_cache: str = "embeddings.npz"  # cached per-clip face embeddings

    # --- per-signer foldering (Req 9) ---
    by_signer_dir: str = "Dataset/by_signer"
    foldering_source: str = "Dataset/processed_videos/resize_720p"  # OQ6
    foldering_copy_mode: str = "hardlink"  # "hardlink" | "copy" (OQ6)

    def resolve(self, relative_path: str) -> Path:
        """Resolve a project-relative path to an absolute path under ``project_root``.

        If ``relative_path`` is already absolute it is returned unchanged
        (resolved). Otherwise it is joined onto :attr:`project_root` so the
        result lives on the same drive as the project (``D:``).
        """
        candidate = Path(relative_path)
        if candidate.is_absolute():
            return candidate.resolve()
        return (Path(self.project_root) / candidate).resolve()

    # --- convenience accessors (absolute paths on D:) ---

    @property
    def vsl400_path(self) -> Path:
        return self.resolve(self.vsl400_dir)

    @property
    def output_path(self) -> Path:
        return self.resolve(self.output_dir)

    @property
    def log_path(self) -> Path:
        return self.resolve(self.log_dir)

    @property
    def models_path(self) -> Path:
        return self.resolve(self.models_dir)

    @property
    def by_signer_path(self) -> Path:
        return self.resolve(self.by_signer_dir)

    @property
    def foldering_source_path(self) -> Path:
        return self.resolve(self.foldering_source)

    @property
    def output_view_file(self) -> Path:
        """Absolute path to the emitted view JSON, e.g. ``.../front_view.json``."""
        return self.output_path / f"{self.output_view_name}.json"

    @property
    def signer_sidecar_path(self) -> Path:
        """Absolute path to the signer side-car CSV under the output dir."""
        return self.output_path / self.signer_sidecar

    @property
    def signer_embeddings_cache_path(self) -> Path:
        """Absolute path to the cached per-clip face embeddings (under output dir).

        Used by the incremental signer-extraction cache so that, on a re-run,
        only clips not already present in the cache are re-embedded.
        """
        return self.output_path / self.signer_embeddings_cache

    def video_search_paths(self) -> tuple[Path, ...]:
        """Absolute paths of every configured video search directory."""
        return tuple(self.resolve(d) for d in self.video_search_dirs)

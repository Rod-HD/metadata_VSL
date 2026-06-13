"""Signer-identity extraction from QIPEDC video content (Requirement 8).

QIPEDC has no signer information, so the signer of each clip must be inferred
from the video itself. This module performs the per-clip half of that work:

* :func:`embed_clip` samples a handful of frames from a clip, finds the largest
  face in each with OpenCV's YuNet detector (:class:`cv2.FaceDetectorYN`),
  computes a 128-D face embedding for each detected face with the SFace
  recognizer (:class:`cv2.FaceRecognizerSF`), and returns the mean,
  L2-normalized embedding for the clip (or ``None`` when no face is found).

The clustering / assignment half (``extract_signers``) is implemented in a
separate subtask and consumes embeddings via an injectable ``embed_fn`` so it
can be tested with synthetic vectors and needs neither real videos nor models.

Both ONNX model files live under ``cfg.models_path`` on ``D:`` (Requirement
8.6); nothing is written to or cached on ``C:``. The detector and recognizer
are loaded lazily and cached per model path so repeated calls across many clips
do not re-read the ONNX files.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

# Module logger; handlers are attached on demand by ``_get_file_logger`` so that
# importing this module never creates files as a side effect.
_LOGGER_NAME = "qipedc2vsl400.signer_extractor"

# OpenCV Zoo model filenames expected under ``cfg.models_path`` (on D:).
YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_FILENAME = "face_recognition_sface_2021dec.onnx"


@dataclass
class SignerAssignment:
    """Signer-extraction result for a single clip.

    Mirrors the design's ``SignerAssignment`` model. Produced by
    ``extract_signers`` and written to the ``signers.csv`` side-car.

    Attributes:
        video: Original QIPEDC ``VIDEO`` filename (e.g. ``"D0530.mp4"``).
        signer_id: Zero-padded 3-digit signer id, or the configured
            ``unknown`` bucket label when no face was obtained.
        cluster_index: Raw cluster id from clustering; ``-1`` means no face /
            unknown.
        distance: Cosine distance to the cluster representative, or ``None``
            when unknown / not applicable.
        has_face: Whether a usable face embedding was obtained for the clip.
    """

    video: str
    signer_id: str
    cluster_index: int
    distance: float | None
    has_face: bool


# --- lazy, cached model loading ------------------------------------------------

# Cache keyed by the absolute model-file path so we build each detector /
# recognizer once even when embedding hundreds of clips.
_DETECTOR_CACHE: dict[str, Any] = {}
_RECOGNIZER_CACHE: dict[str, Any] = {}


def _get_detector(models_path: Path) -> Any:
    """Return a cached :class:`cv2.FaceDetectorYN`, loading it on first use.

    The YuNet ONNX weights are read from ``models_path`` on ``D:``. The input
    size is a placeholder here and is set per-frame in :func:`embed_clip` via
    ``setInputSize`` once the frame dimensions are known.
    """
    model_file = Path(models_path) / YUNET_MODEL_FILENAME
    key = str(model_file)
    detector = _DETECTOR_CACHE.get(key)
    if detector is None:
        if not model_file.is_file():
            raise FileNotFoundError(
                f"YuNet face-detection model not found at {model_file}. "
                "Download it into cfg.models_path (on D:) first."
            )
        detector = cv2.FaceDetectorYN.create(
            model=str(model_file),
            config="",
            input_size=(320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
        )
        _DETECTOR_CACHE[key] = detector
    return detector


def _get_recognizer(models_path: Path) -> Any:
    """Return a cached :class:`cv2.FaceRecognizerSF`, loading it on first use.

    The SFace ONNX weights are read from ``models_path`` on ``D:``.
    """
    model_file = Path(models_path) / SFACE_MODEL_FILENAME
    key = str(model_file)
    recognizer = _RECOGNIZER_CACHE.get(key)
    if recognizer is None:
        if not model_file.is_file():
            raise FileNotFoundError(
                f"SFace face-recognition model not found at {model_file}. "
                "Download it into cfg.models_path (on D:) first."
            )
        recognizer = cv2.FaceRecognizerSF.create(
            model=str(model_file),
            config="",
        )
        _RECOGNIZER_CACHE[key] = recognizer
    return recognizer


# --- frame sampling ------------------------------------------------------------


def _sample_frame_indices(num_frames: int, samples: int) -> list[int]:
    """Pick *samples* frame indices spread evenly across ``[0, num_frames)``.

    Uses the midpoints of ``samples`` equal-width buckets so the chosen frames
    avoid the very first/last frame and are deterministic. Returns a sorted list
    of unique indices; an empty list when there are no frames or no samples.
    """
    if num_frames <= 0 or samples <= 0:
        return []
    if samples >= num_frames:
        return list(range(num_frames))
    indices = sorted(
        {int((i + 0.5) * num_frames / samples) for i in range(samples)}
    )
    # Clamp into range just in case of rounding at the upper edge.
    return [min(max(idx, 0), num_frames - 1) for idx in indices]


def _largest_face(faces: np.ndarray | None) -> np.ndarray | None:
    """Return the row of *faces* with the greatest bounding-box area.

    ``FaceDetectorYN.detect`` returns ``(retval, faces)`` where ``faces`` is an
    ``Nx15`` array whose first four columns are ``x, y, w, h``. Returns ``None``
    when there are no faces.
    """
    if faces is None or len(faces) == 0:
        return None
    # Columns 2 and 3 are width and height of the detection box.
    areas = faces[:, 2] * faces[:, 3]
    best = int(np.argmax(areas))
    return faces[best]


# --- public API ----------------------------------------------------------------


def embed_clip(video_path: Path, cfg: Any) -> np.ndarray | None:
    """Compute a single face embedding for a clip.

    Samples ``cfg.signer_frames_per_video`` frames evenly across the clip, finds
    the largest face in each sampled frame with YuNet, aligns/crops and embeds
    that face with SFace, and returns the mean of the per-frame embeddings,
    L2-normalized.

    Args:
        video_path: Path to the clip on ``D:``.
        cfg: A :class:`~qipedc2vsl400.config.Config`-like object exposing
            ``signer_frames_per_video`` and ``models_path``.

    Returns:
        A 1-D :class:`numpy.ndarray` (the L2-normalized mean embedding), or
        ``None`` when the clip cannot be opened or no face is found in any
        sampled frame.
    """
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None

        frame_count_raw = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        try:
            num_frames = int(frame_count_raw)
        except (TypeError, ValueError):
            num_frames = 0

        samples = int(getattr(cfg, "signer_frames_per_video", 5))
        indices = _sample_frame_indices(num_frames, samples)

        detector = _get_detector(cfg.models_path)
        recognizer = _get_recognizer(cfg.models_path)

        embeddings: list[np.ndarray] = []
        last_input_size: tuple[int, int] | None = None

        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            height, width = frame.shape[:2]
            input_size = (width, height)
            if input_size != last_input_size:
                detector.setInputSize(input_size)
                last_input_size = input_size

            _, faces = detector.detect(frame)
            face = _largest_face(faces)
            if face is None:
                continue

            aligned = recognizer.alignCrop(frame, face)
            feature = recognizer.feature(aligned)
            # ``feature`` is a 1xD array; flatten to 1-D.
            embeddings.append(np.asarray(feature, dtype=np.float64).reshape(-1))

        if not embeddings:
            return None

        mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm == 0:
            return None
        return mean_embedding / norm
    finally:
        capture.release()


# --- signer clustering / assignment (Requirement 8 AC2-AC7) --------------------


def _get_file_logger(cfg: Any) -> logging.Logger:
    """Return a logger writing to a timestamped file under ``cfg.log_dir``.

    Mirrors the reader's logging convention: the log directory is created if
    needed, a fresh :class:`~logging.FileHandler` is attached per call, and
    propagation is disabled to avoid duplicate console output.
    """
    log_dir = cfg.log_path
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"signer_extractor_{timestamp}.log"

    logger = logging.getLogger(f"{_LOGGER_NAME}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    return logger


def _build_video_index(cfg: Any) -> dict[str, Path]:
    """Map each video *filename* to its absolute path under the search dirs.

    Search directories (``cfg.video_search_paths()``) are scanned in order; the
    first occurrence of a given filename wins, so the preferred directory
    (``processed_videos/resize_720p``) takes precedence over ``raw_videos``.
    Within a directory, paths are visited in sorted order for determinism. The
    scan is recursive because ``raw_videos`` nests each clip in its own
    subfolder.
    """
    index: dict[str, Path] = {}
    for base in cfg.video_search_paths():
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.mp4")):
            if path.is_file() and path.name not in index:
                index[path.name] = path
    return index


def _resolve_video_path(video: str | None, index: dict[str, Path]) -> Path | None:
    """Return the resolved path for a ``VIDEO`` filename, or ``None`` if absent."""
    if not video:
        return None
    return index.get(video)


def _cosine_distance(vec: np.ndarray, ref: np.ndarray) -> float:
    """Return the cosine distance ``1 - cos_sim`` between *vec* and *ref*.

    Both vectors are treated as direction-only; a zero-norm vector yields a
    distance of ``0.0`` (no meaningful direction to compare).
    """
    vec_norm = float(np.linalg.norm(vec))
    ref_norm = float(np.linalg.norm(ref))
    if vec_norm == 0.0 or ref_norm == 0.0:
        return 0.0
    similarity = float(np.dot(vec, ref) / (vec_norm * ref_norm))
    # Guard against tiny floating-point excursions outside [-1, 1].
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def _cluster_labels(embeddings: list[np.ndarray], cfg: Any) -> list[int]:
    """Cluster *embeddings* with a cosine ``distance_threshold`` (no preset count).

    Returns one raw cluster label per input embedding. Handles the degenerate
    cases that :class:`~sklearn.cluster.AgglomerativeClustering` cannot (it
    requires at least two samples):

    * zero embeddings  -> ``[]``
    * one embedding    -> ``[0]``
    """
    count = len(embeddings)
    if count == 0:
        return []
    if count == 1:
        return [0]

    from sklearn.cluster import AgglomerativeClustering

    matrix = np.stack(embeddings, axis=0)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=cfg.signer_cosine_threshold,
    )
    labels = clustering.fit_predict(matrix)
    return [int(label) for label in labels]


def assign_signers(
    order_keys: list[str],
    embeddings: list[np.ndarray | None],
    cfg: Any,
) -> list[tuple[str, int, float | None, bool]]:
    """Cluster *embeddings* and assign deterministic signer ids.

    This is the shared clustering/numbering core used by both per-row signer
    extraction and the batch clip-store flow.

    Args:
        order_keys: One string per item, used purely to order clusters
            deterministically (clusters are numbered by the *minimum* key among
            their members). Typically the ``VIDEO`` filename or ``video_id``.
        embeddings: One embedding per item, aligned with *order_keys*; ``None``
            marks a clip with no detectable face (routed to the unknown bucket).
        cfg: A :class:`~qipedc2vsl400.config.Config`-like object.

    Returns:
        A list aligned with the inputs of
        ``(signer_id, cluster_index, distance, has_face)`` tuples. Items with a
        face get a zero-padded ``signer_id`` (width ``cfg.signer_id_width``) and
        the cosine ``distance`` to their cluster representative; items without a
        face get ``(cfg.signer_unknown_label, -1, None, False)``.
    """
    face_idx = [i for i, emb in enumerate(embeddings) if emb is not None]
    face_emb = [embeddings[i] for i in face_idx]
    raw_labels = _cluster_labels(face_emb, cfg)

    clusters: dict[int, list[int]] = defaultdict(list)
    for local_idx, raw_label in enumerate(raw_labels):
        clusters[raw_label].append(face_idx[local_idx])

    ordered_raw_labels = sorted(
        clusters.keys(),
        key=lambda lbl: min(order_keys[i] for i in clusters[lbl]),
    )

    width = cfg.signer_id_width
    result: list[tuple[str, int, float | None, bool]] = [
        (cfg.signer_unknown_label, -1, None, False) for _ in embeddings
    ]
    for ordered_index, raw_label in enumerate(ordered_raw_labels):
        members = clusters[raw_label]
        signer_id = str(ordered_index + 1).zfill(width)
        representative = np.mean(
            np.stack([embeddings[i] for i in members], axis=0), axis=0
        )
        for i in members:
            distance = _cosine_distance(embeddings[i], representative)
            result[i] = (signer_id, ordered_index, distance, True)
    return result


def load_embeddings_cache(
    cache_path: "Path | str",
) -> tuple[dict[str, np.ndarray], set[str]]:
    """Load cached per-clip face embeddings from *cache_path*.

    Returns ``(vectors_by_video, noface_videos)`` where ``vectors_by_video`` maps
    a ``VIDEO`` filename to its stored embedding vector and ``noface_videos`` is
    the set of clips previously found to have no detectable face. Returns empty
    containers when the cache file does not exist (first run).

    The cache is a NumPy ``.npz`` archive loaded with ``allow_pickle=False`` (no
    arbitrary-object deserialization).
    """
    path = Path(cache_path)
    if not path.is_file():
        return {}, set()
    with np.load(path, allow_pickle=False) as data:
        face_names = [str(name) for name in data["videos_face"]]
        emb = data["emb"]
        vectors = {
            name: np.asarray(emb[i], dtype=np.float64)
            for i, name in enumerate(face_names)
        }
        noface = {str(name) for name in data["videos_noface"]}
    return vectors, noface


def save_embeddings_cache(
    cache_path: "Path | str",
    vectors_by_video: dict[str, np.ndarray],
    noface_videos: set[str],
) -> Path:
    """Persist per-clip face embeddings to *cache_path* (NumPy ``.npz``).

    Stores the face embeddings (one row per clip), the parallel list of clip
    filenames, and the list of no-face clips. Filenames are stored as fixed-width
    unicode arrays so the archive needs no pickling. Returns the path written.
    """
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    face_names = list(vectors_by_video.keys())
    if face_names:
        emb = np.stack(
            [np.asarray(vectors_by_video[name], dtype=np.float64).reshape(-1)
             for name in face_names],
            axis=0,
        )
    else:
        emb = np.zeros((0, 0), dtype=np.float64)
    np.savez(
        path,
        videos_face=np.array(face_names, dtype=str),
        emb=emb,
        videos_noface=np.array(sorted(noface_videos), dtype=str),
    )
    return path


def extract_signers(
    valid_rows: list[Any],
    cfg: Any,
    embed_fn: Callable[[Path | None, Any], np.ndarray | None] = embed_clip,
    embeddings_cache: "Path | str | None" = None,
) -> list[SignerAssignment]:
    """Assign a signer identity to every valid QIPEDC clip (Requirement 8).

    For each row in *valid_rows* the clip's video is located under
    ``cfg.video_search_paths()`` and embedded via *embed_fn* (defaults to
    :func:`embed_clip`). Non-``None`` embeddings are clustered with a cosine
    ``distance_threshold`` (``cfg.signer_cosine_threshold``) using agglomerative
    clustering with **no** preset cluster count (AC3). Each resulting cluster is
    one signer (AC2).

    Clusters are numbered deterministically: ordered by the *minimum* ``VIDEO``
    filename within each cluster, then labelled ``001, 002, …`` zero-padded to
    ``cfg.signer_id_width`` (AC4). Clips with no detectable face (embedding
    ``None``) are placed in the constant ``cfg.signer_unknown_label`` bucket with
    ``cluster_index == -1`` and ``distance == None`` (AC5), and are never
    dropped — every valid clip yields exactly one :class:`SignerAssignment`
    (Property 10).

    A side-car CSV (``cfg.signer_sidecar_path``) recording
    ``video, signer_id, cluster_index, distance, has_face`` is written for
    inspection and reuse (AC7).

    Args:
        valid_rows: Validated ``QipedcRow`` objects (each has a non-empty
            ``video`` filename).
        cfg: A :class:`~qipedc2vsl400.config.Config`-like object.
        embed_fn: Injectable embedding function ``(video_path, cfg) -> vector |
            None``; defaults to :func:`embed_clip`. Tests pass a deterministic
            stub so no real videos or models are required.
        embeddings_cache: Optional path to a NumPy ``.npz`` embedding cache. When
            given, clips already present in the cache reuse their stored
            embedding (the expensive face detection/embedding is skipped) and
            only **new** clips are embedded via *embed_fn*; the cache is then
            updated with any newly computed embeddings. Clustering still runs
            over the full set (cached + new), so the result is identical to a
            full re-run — the cache is purely a performance optimization.

    Returns:
        A list of :class:`SignerAssignment`, one per row in *valid_rows*, in the
        same order.
    """
    logger = _get_file_logger(cfg)
    try:
        index = _build_video_index(cfg)

        # 0) Load any existing embedding cache (incremental re-runs).
        cache_vectors: dict[str, np.ndarray] = {}
        cache_noface: set[str] = set()
        if embeddings_cache is not None:
            cache_vectors, cache_noface = load_embeddings_cache(embeddings_cache)
            logger.info(
                "Loaded embedding cache: %d face, %d no-face from %s",
                len(cache_vectors),
                len(cache_noface),
                embeddings_cache,
            )

        # 1) Embed every clip, preserving row order. Cached clips reuse their
        #    stored embedding; only new clips are embedded via ``embed_fn``.
        #    Rows without a usable embedding are routed to the unknown bucket.
        videos: list[str] = []
        embeddings_per_row: list[np.ndarray | None] = []
        cache_hits = 0
        newly_computed = 0
        for row in valid_rows:
            video = row.video or ""
            videos.append(video)

            if embeddings_cache is not None and video in cache_vectors:
                embeddings_per_row.append(cache_vectors[video])
                cache_hits += 1
                continue
            if embeddings_cache is not None and video in cache_noface:
                embeddings_per_row.append(None)
                cache_hits += 1
                continue

            path = _resolve_video_path(row.video, index)
            if path is None:
                logger.warning("No video file found for VIDEO %r", video)
            # ``embed_fn`` is always invoked (with the resolved path, which may
            # be ``None``); ``embed_clip`` handles a missing/None path by
            # returning ``None``, and test stubs supply synthetic vectors
            # without needing real files on disk.
            embedding = embed_fn(path, cfg)
            newly_computed += 1
            if embedding is None:
                logger.warning(
                    "No face/embedding for VIDEO %r; routing to %r bucket",
                    video,
                    cfg.signer_unknown_label,
                )
                if embeddings_cache is not None:
                    cache_noface.add(video)
            elif embeddings_cache is not None:
                cache_vectors[video] = np.asarray(
                    embedding, dtype=np.float64
                ).reshape(-1)
            embeddings_per_row.append(embedding)

        # 1b) Persist the (possibly extended) cache for the next run.
        if embeddings_cache is not None:
            save_embeddings_cache(embeddings_cache, cache_vectors, cache_noface)
            logger.info(
                "Saved embedding cache: %d face, %d no-face "
                "(cache hits=%d, computed=%d)",
                len(cache_vectors),
                len(cache_noface),
                cache_hits,
                newly_computed,
            )

        # 2-6) Cluster the embeddings and assign deterministic signer ids,
        #      ordered by the minimum VIDEO filename per cluster (AC2-AC5).
        per_row = assign_signers(videos, embeddings_per_row, cfg)
        assignments = [
            SignerAssignment(
                video=videos[i],
                signer_id=signer_id,
                cluster_index=cluster_index,
                distance=distance,
                has_face=has_face,
            )
            for i, (signer_id, cluster_index, distance, has_face) in enumerate(
                per_row
            )
        ]

        signer_count = len({a.signer_id for a in assignments if a.has_face})
        unknown_count = sum(1 for a in assignments if not a.has_face)
        logger.info(
            "Signer extraction summary: clips=%d signers=%d unknown=%d",
            len(assignments),
            signer_count,
            unknown_count,
        )

        # 7) Write the side-car CSV (AC7).
        _write_signers_csv(assignments, cfg)

        return assignments
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def _write_signers_csv(assignments: list[SignerAssignment], cfg: Any) -> Path:
    """Write ``video -> signer_id, cluster_index, distance, has_face`` to CSV.

    The file is written to ``cfg.signer_sidecar_path`` (under the output dir),
    creating parent directories as needed. ``distance`` is rendered empty for
    unknown clips. Returns the path written.
    """
    sidecar_path = cfg.signer_sidecar_path
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["video", "signer_id", "cluster_index", "distance", "has_face"]
        )
        for assignment in assignments:
            distance = (
                "" if assignment.distance is None else f"{assignment.distance:.6f}"
            )
            writer.writerow(
                [
                    assignment.video,
                    assignment.signer_id,
                    assignment.cluster_index,
                    distance,
                    assignment.has_face,
                ]
            )
    return sidecar_path


def write_signers_csv(assignments: list[SignerAssignment], cfg: Any) -> Path:
    """Public wrapper around :func:`_write_signers_csv` for reuse by other modules
    (e.g. the batch clip-store flow). Writes the per-clip signer side-car CSV to
    ``cfg.signer_sidecar_path`` and returns the path written.
    """
    return _write_signers_csv(assignments, cfg)

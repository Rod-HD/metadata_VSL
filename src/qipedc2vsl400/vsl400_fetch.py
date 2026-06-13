"""Fetch the VSL400 *labels/metadata only* from Zenodo (Requirement 2).

This module downloads a small, explicit **allow-list** of artifacts from the
VSL400 Zenodo record (DOI ``10.5281/zenodo.17943574``) into ``Dataset/labels/
vsl400/`` on ``D:``. It deliberately fetches only the label/metadata/code
artifacts and **never** requests any ``Part_*.zip`` video archive (Req 2.1).

Workflow:

1. Fetch the record metadata JSON (``/api/records/<id>``) to obtain, per file,
   its download URL and its Zenodo MD5 checksum (Req 2.2).
2. For each *allow-listed* key, stream the file from the Zenodo files API
   (``/api/records/<id>/files/<key>/content``) into ``cfg.vsl400_dir``.
3. Verify each downloaded file against its Zenodo MD5; **fail fast** on a
   mismatch so a corrupt artifact is never trusted (Req 2.2).
4. Write ``provenance.json`` recording the DOI/URL, the file keys, their MD5s,
   and a retrieval timestamp (Req 2.4).

All target paths resolve under the project root on ``D:`` (Req 2.3). No video
media is downloaded (Out of Scope / Req 2.1).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from .config import Config

# --- explicit allow-list of Zenodo keys (labels/metadata/code only) ----------
# These are the ONLY keys that will ever be requested. Video archives
# (``Part_*.zip``) are intentionally absent and are additionally hard-excluded
# by :func:`_is_excluded` as defence-in-depth (Req 2.1).
ALLOWED_KEYS: tuple[str, ...] = (
    "README.txt",
    "VietnameseSignLanguageRecognition.zip",
    "merge_splits.py",
)

# Any key matching this pattern is a video archive and must NEVER be fetched.
_EXCLUDED_PATTERN = re.compile(r"^Part_.*\.zip$", re.IGNORECASE)

# Zenodo DOI for the VSL400 record (provenance only).
VSL400_DOI = "10.5281/zenodo.17943574"

_CHUNK_SIZE = 1 << 16  # 64 KiB streaming chunks
_REQUEST_TIMEOUT = 60  # seconds


def _is_excluded(key: str) -> bool:
    """Return ``True`` for keys that must never be downloaded (video archives)."""
    return bool(_EXCLUDED_PATTERN.match(key))


def _api_base(record_id: str) -> str:
    return f"https://zenodo.org/api/records/{record_id}"


def _file_content_url(record_id: str, key: str) -> str:
    """Zenodo files API URL that streams the raw content of *key*."""
    return f"{_api_base(record_id)}/files/{key}/content"


def _parse_md5(checksum: str | None) -> str | None:
    """Normalize a Zenodo checksum string (``"md5:<hex>"``) to the bare hex.

    Returns ``None`` when no usable md5 checksum is available.
    """
    if not checksum:
        return None
    value = checksum.strip()
    if value.lower().startswith("md5:"):
        value = value[4:]
    value = value.strip()
    return value.lower() or None


def _fetch_record_metadata(record_id: str, session: requests.Session) -> dict:
    """Fetch and return the Zenodo record metadata JSON."""
    resp = session.get(_api_base(record_id), timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _index_checksums(record_metadata: dict) -> dict[str, str | None]:
    """Map ``key -> md5 hex`` from a Zenodo record's ``files`` list.

    Zenodo file entries expose the key under ``"key"`` (newer API) or
    ``"filename"`` (older API) and the checksum under ``"checksum"``.
    """
    index: dict[str, str | None] = {}
    for entry in record_metadata.get("files", []) or []:
        key = entry.get("key") or entry.get("filename")
        if not key:
            continue
        index[key] = _parse_md5(entry.get("checksum"))
    return index


def _md5_of_file(path: Path) -> str:
    """Compute the lowercase hex MD5 digest of *path* streaming from disk."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, dest: Path, session: requests.Session) -> None:
    """Stream *url* to *dest*, writing in chunks to avoid loading into memory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=_REQUEST_TIMEOUT) as resp:
        resp.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)


def fetch_labels(cfg: Config, session: requests.Session | None = None) -> Path:
    """Download the VSL400 label/metadata artifacts into ``cfg.vsl400_dir``.

    Downloads ONLY the keys in :data:`ALLOWED_KEYS` from the Zenodo record
    ``cfg.zenodo_record_id``, verifies each against its Zenodo MD5 checksum, and
    writes ``provenance.json``. Any ``Part_*.zip`` (video) key is hard-excluded
    and never requested (Req 2.1).

    Args:
        cfg: Pipeline configuration; ``cfg.vsl400_path`` is the (D:) destination.
        session: Optional :class:`requests.Session` (injectable for testing). A
            new session is created when not supplied.

    Returns:
        The absolute path to the ``provenance.json`` written under
        ``cfg.vsl400_dir``.

    Raises:
        RuntimeError: If a downloaded file's MD5 does not match the Zenodo
            checksum (fail-fast on corruption).
        requests.HTTPError: If the Zenodo API or a file download fails.
    """
    record_id = cfg.zenodo_record_id
    dest_dir = cfg.vsl400_path
    dest_dir.mkdir(parents=True, exist_ok=True)

    owns_session = session is None
    session = session or requests.Session()

    try:
        record_metadata = _fetch_record_metadata(record_id, session)
        checksums = _index_checksums(record_metadata)

        downloaded: list[dict[str, str]] = []

        for key in ALLOWED_KEYS:
            # Defence-in-depth: never request an excluded (video) key even if it
            # somehow appears in the allow-list.
            if _is_excluded(key):
                continue

            url = _file_content_url(record_id, key)
            dest = dest_dir / key
            _download_file(url, dest, session)

            expected_md5 = checksums.get(key)
            actual_md5 = _md5_of_file(dest)
            if expected_md5 is not None and actual_md5 != expected_md5:
                raise RuntimeError(
                    f"MD5 mismatch for '{key}': expected {expected_md5}, "
                    f"got {actual_md5}. Refusing to trust a corrupt artifact."
                )

            downloaded.append(
                {
                    "key": key,
                    "url": url,
                    "md5": actual_md5,
                    "expected_md5": expected_md5 or "",
                }
            )

        provenance = {
            "doi": VSL400_DOI,
            "record_id": record_id,
            "record_url": f"https://zenodo.org/records/{record_id}",
            "api_url": _api_base(record_id),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "allowed_keys": list(ALLOWED_KEYS),
            "excluded": "Part_*.zip (video archives) — never downloaded",
            "files": downloaded,
        }

        provenance_path = dest_dir / "provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return provenance_path
    finally:
        if owns_session:
            session.close()


# --- OpenCV Zoo face models (YuNet detector + SFace recognizer) --------------
# Signer extraction (Req 8) uses OpenCV's bundled ``FaceDetectorYN`` (YuNet) and
# ``FaceRecognizerSF`` (SFace). Their ONNX weight files are downloaded once into
# ``cfg.models_dir`` (``Dataset/models/`` on ``D:``) so nothing is cached on
# ``C:`` (Req 8 AC6 / 8.6). Weights come from the official OpenCV Zoo GitHub
# repository's ``main`` branch via the raw-content host.


@dataclass(frozen=True)
class ModelSpec:
    """A single OpenCV Zoo ONNX model to download.

    Attributes:
        filename: Destination filename under ``cfg.models_dir``.
        url: Official OpenCV Zoo raw GitHub URL for the ``.onnx`` weights.
        min_size_bytes: Conservative lower bound used as a sanity check when the
            server does not advertise a ``Content-Length`` (guards against a
            truncated/HTML error page being saved as a model).
        kind: Human-readable role ("detector" / "recognizer") for provenance.
    """

    filename: str
    url: str
    min_size_bytes: int
    kind: str


# OpenCV Zoo raw URLs (``main`` branch). These point at the canonical ONNX
# weights published by the OpenCV team.
FACE_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        min_size_bytes=100_000,  # YuNet 2023mar is ~230 KiB
        kind="detector",
    ),
    ModelSpec(
        filename="face_recognition_sface_2021dec.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/"
            "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        min_size_bytes=1_000_000,  # SFace 2021dec is ~37 MiB
        kind="recognizer",
    ),
)


def _remote_size(url: str, session: requests.Session) -> int | None:
    """Best-effort fetch of a remote file's size via ``Content-Length``.

    Returns the advertised byte length, or ``None`` when the server does not
    provide one (e.g. chunked transfer). Never raises for a missing header.
    """
    try:
        resp = session.head(url, allow_redirects=True, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    length = resp.headers.get("Content-Length")
    if length is None:
        return None
    try:
        return int(length)
    except ValueError:
        return None


def _is_present_with_correct_size(
    path: Path, expected_size: int | None, min_size_bytes: int
) -> bool:
    """Return ``True`` if *path* already holds a valid copy (skip re-download).

    A file counts as valid when it exists and either matches the server's
    advertised ``expected_size`` exactly, or — when no size is advertised —
    is at least ``min_size_bytes`` (sanity bound).
    """
    if not path.exists():
        return False
    actual = path.stat().st_size
    if expected_size is not None:
        return actual == expected_size
    return actual >= min_size_bytes


def fetch_face_models(
    cfg: Config,
    session: requests.Session | None = None,
    models: tuple[ModelSpec, ...] = FACE_MODELS,
) -> Path:
    """Download the YuNet + SFace ONNX face models into ``cfg.models_path``.

    Streams each OpenCV Zoo model into ``Dataset/models/`` on ``D:`` (creating
    the directory if needed), skips any model already present with the correct
    size, validates the downloaded size, and writes ``provenance.json`` recording
    the source URLs and a retrieval timestamp (Req 8.6 — models live on ``D:``).

    Args:
        cfg: Pipeline configuration; ``cfg.models_path`` is the (D:) destination.
        session: Optional :class:`requests.Session` (injectable for testing). A
            new session is created when not supplied.
        models: The model specs to fetch (defaults to :data:`FACE_MODELS`).

    Returns:
        The absolute path to the ``provenance.json`` written under
        ``cfg.models_path``.

    Raises:
        RuntimeError: If a freshly downloaded file is smaller than expected
            (truncated/corrupt), so a bad model is never trusted.
        requests.HTTPError: If a model download request fails.
    """
    dest_dir = cfg.models_path
    dest_dir.mkdir(parents=True, exist_ok=True)

    owns_session = session is None
    session = session or requests.Session()

    try:
        results: list[dict[str, object]] = []

        for spec in models:
            dest = dest_dir / spec.filename
            expected_size = _remote_size(spec.url, session)

            if _is_present_with_correct_size(dest, expected_size, spec.min_size_bytes):
                results.append(
                    {
                        "filename": spec.filename,
                        "kind": spec.kind,
                        "url": spec.url,
                        "size_bytes": dest.stat().st_size,
                        "expected_size_bytes": expected_size,
                        "skipped": True,
                    }
                )
                continue

            _download_file(spec.url, dest, session)
            actual_size = dest.stat().st_size

            # Fail fast on a clearly bad download (e.g. an HTML error page).
            if expected_size is not None and actual_size != expected_size:
                raise RuntimeError(
                    f"Size mismatch for '{spec.filename}': expected "
                    f"{expected_size} bytes, got {actual_size}. Refusing to "
                    f"trust a corrupt model."
                )
            if expected_size is None and actual_size < spec.min_size_bytes:
                raise RuntimeError(
                    f"Downloaded '{spec.filename}' is only {actual_size} bytes "
                    f"(< {spec.min_size_bytes}); likely a truncated/error "
                    f"response. Refusing to trust a corrupt model."
                )

            results.append(
                {
                    "filename": spec.filename,
                    "kind": spec.kind,
                    "url": spec.url,
                    "size_bytes": actual_size,
                    "expected_size_bytes": expected_size,
                    "skipped": False,
                }
            )

        provenance = {
            "source": "OpenCV Zoo (https://github.com/opencv/opencv_zoo)",
            "branch": "main",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "models_dir": str(dest_dir),
            "models": results,
        }
        provenance_path = dest_dir / "provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return provenance_path
    finally:
        if owns_session:
            session.close()


def load_reference_schema(vsl400_dir: Path) -> list[dict] | None:
    """Load a sample VSL400 ``*_view.json`` for schema validation, if present.

    Searches *vsl400_dir* (recursively) for a file named like ``front_view.json``
    / ``*_view.json`` and returns its parsed JSON array. Returns ``None`` when no
    such sample is available (the reference check is then skipped — Req 2.5/6.4).

    Args:
        vsl400_dir: Directory containing the downloaded/extracted VSL400 labels.

    Returns:
        The parsed list of metadata dicts, or ``None`` if no sample is found or
        the file is not a JSON array of objects.
    """
    directory = Path(vsl400_dir)
    if not directory.exists():
        return None

    candidates = sorted(directory.rglob("*_view.json"))
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, list):
            return data
    return None

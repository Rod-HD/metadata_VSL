"""Unit tests for ``qipedc2vsl400.vsl400_fetch`` (Task 5.3).

All network access is **mocked**. A :class:`FakeSession` stands in for
``requests.Session`` and is injected via the ``session=`` parameter that
:func:`fetch_labels` and :func:`fetch_face_models` accept, so the suite runs
fully offline and deterministically.

Covers Requirements:
- 2.1 — only allow-listed Zenodo keys are fetched; ``Part_*.zip`` is never
  requested.
- 2.2 — MD5 mismatch fails fast (``RuntimeError``).
- 2.3 / 8.6 — all target paths resolve under ``D:`` (never ``C:``).
- 2.4 — provenance (DOI/URL + retrieval timestamp) is recorded.
- 2.5 — happy-path download writes the allow-listed artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import requests

from qipedc2vsl400.config import Config
from qipedc2vsl400 import vsl400_fetch
from qipedc2vsl400.vsl400_fetch import (
    ALLOWED_KEYS,
    FACE_MODELS,
    ModelSpec,
    _file_content_url,
    _is_excluded,
    fetch_face_models,
    fetch_labels,
)

# A project root that is genuinely on D: — used only for the *drive* assertions
# (no files are written there). File-writing tests use ``tmp_path`` instead.
D_PROJECT_ROOT = Path("D:/projects/metadata_VSL")


# --------------------------------------------------------------------------- #
# Fake network layer
# --------------------------------------------------------------------------- #
class FakeResponse:
    """A scripted stand-in for :class:`requests.Response`.

    Doubles as a context manager so it works for both the plain metadata
    ``GET`` and the streaming download ``with session.get(...) as resp``.
    """

    def __init__(
        self,
        *,
        json_data=None,
        content: bytes = b"",
        headers: dict | None = None,
        status_ok: bool = True,
    ):
        self._json = json_data
        self._content = content
        self.headers = headers or {}
        self._status_ok = status_ok

    def raise_for_status(self) -> None:
        if not self._status_ok:
            raise requests.HTTPError("simulated HTTP error")

    def json(self):
        return self._json

    def iter_content(self, chunk_size: int = 1):
        data = self._content
        for i in range(0, len(data), max(1, chunk_size)):
            yield data[i : i + chunk_size]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False


class FakeSession:
    """In-memory replacement for :class:`requests.Session`.

    Records every URL requested so tests can assert that no ``Part_*.zip`` URL
    is ever touched and that only allow-listed keys are downloaded.
    """

    def __init__(
        self,
        *,
        record_metadata: dict | None = None,
        file_contents: dict[str, bytes] | None = None,
    ):
        self.record_metadata = record_metadata or {}
        self.file_contents = file_contents or {}
        self.get_calls: list[str] = []
        self.head_calls: list[str] = []
        self.closed = False

    def get(self, url, *, stream: bool = False, timeout=None):
        self.get_calls.append(url)
        if stream:
            content = self.file_contents.get(url)
            if content is None:
                return FakeResponse(status_ok=False)
            return FakeResponse(content=content)
        return FakeResponse(json_data=self.record_metadata)

    def head(self, url, *, allow_redirects: bool = True, timeout=None):
        self.head_calls.append(url)
        content = self.file_contents.get(url, b"")
        return FakeResponse(headers={"Content-Length": str(len(content))})

    def close(self) -> None:
        self.closed = True


def _md5_checksum(data: bytes) -> str:
    return "md5:" + hashlib.md5(data).hexdigest()


def _build_record_metadata(contents: dict[str, bytes]) -> dict:
    """Build a Zenodo-style record metadata dict for the given key->content."""
    return {
        "files": [
            {"key": key, "checksum": _md5_checksum(data)}
            for key, data in contents.items()
        ]
    }


def _make_label_session(
    cfg: Config,
    *,
    extra_files: dict[str, bytes] | None = None,
    corrupt_key: str | None = None,
) -> FakeSession:
    """Wire up a FakeSession serving the allow-listed label artifacts.

    ``extra_files`` injects additional (e.g. ``Part_*.zip``) entries into the
    record metadata to prove they are never requested. ``corrupt_key`` makes the
    advertised checksum disagree with the served bytes to exercise the
    fail-fast MD5 check.
    """
    contents = {key: f"content-of-{key}".encode("utf-8") for key in ALLOWED_KEYS}
    record_id = cfg.zenodo_record_id

    metadata_contents = dict(contents)
    if extra_files:
        metadata_contents.update(extra_files)

    metadata = _build_record_metadata(metadata_contents)

    if corrupt_key is not None:
        # Advertise a checksum that does NOT match the served bytes.
        for entry in metadata["files"]:
            if entry["key"] == corrupt_key:
                entry["checksum"] = "md5:" + ("0" * 32)

    url_contents = {
        _file_content_url(record_id, key): data for key, data in contents.items()
    }
    return FakeSession(record_metadata=metadata, file_contents=url_contents)


# --------------------------------------------------------------------------- #
# Allow-list / exclusion (Req 2.1)
# --------------------------------------------------------------------------- #
def test_allowed_keys_never_include_part_zip():
    # The static allow-list must not contain any video archive.
    assert not any(_is_excluded(key) for key in ALLOWED_KEYS)
    assert all("Part_" not in key for key in ALLOWED_KEYS)
    assert set(ALLOWED_KEYS) == {
        "README.txt",
        "VietnameseSignLanguageRecognition.zip",
        "merge_splits.py",
    }


@pytest.mark.parametrize(
    "key",
    ["Part_1.zip", "Part_2.zip", "Part_10.zip", "part_3.ZIP", "PART_99.zip"],
)
def test_is_excluded_matches_part_archives(key):
    assert _is_excluded(key) is True


@pytest.mark.parametrize("key", list(ALLOWED_KEYS) + ["something_else.json"])
def test_is_excluded_allows_non_part_keys(key):
    assert _is_excluded(key) is False


def test_fetch_labels_only_requests_allowed_keys(tmp_path):
    cfg = Config(project_root=tmp_path)
    # Inject a Part_*.zip into the record metadata to prove it is never fetched.
    session = _make_label_session(
        cfg, extra_files={"Part_1.zip": b"VIDEO-ARCHIVE-DO-NOT-DOWNLOAD"}
    )

    fetch_labels(cfg, session=session)

    record_id = cfg.zenodo_record_id
    download_calls = [c for c in session.get_calls if "/files/" in c]
    allowed_urls = {_file_content_url(record_id, k) for k in ALLOWED_KEYS}

    # Only allow-listed content URLs were ever streamed.
    assert set(download_calls) == allowed_urls
    # No request anywhere touched a Part_*.zip key.
    assert all("Part_" not in c for c in session.get_calls)


# --------------------------------------------------------------------------- #
# MD5 mismatch fails fast (Req 2.2)
# --------------------------------------------------------------------------- #
def test_fetch_labels_md5_mismatch_raises(tmp_path):
    cfg = Config(project_root=tmp_path)
    session = _make_label_session(cfg, corrupt_key="README.txt")

    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        fetch_labels(cfg, session=session)


# --------------------------------------------------------------------------- #
# All target paths resolve under D: (Req 2.3 / 8.6)
# --------------------------------------------------------------------------- #
def test_target_paths_resolve_under_d_drive():
    cfg = Config(project_root=D_PROJECT_ROOT)
    assert cfg.vsl400_path.drive.upper() == "D:"
    assert cfg.models_path.drive.upper() == "D:"
    # Never C:.
    assert cfg.vsl400_path.drive.upper() != "C:"
    assert cfg.models_path.drive.upper() != "C:"


# --------------------------------------------------------------------------- #
# Happy path: files written + provenance recorded (Req 2.4 / 2.5)
# --------------------------------------------------------------------------- #
def test_fetch_labels_happy_path_writes_files_and_provenance(tmp_path):
    cfg = Config(project_root=tmp_path)
    session = _make_label_session(cfg)

    provenance_path = fetch_labels(cfg, session=session)

    # Every allow-listed artifact was written to disk under the project root.
    for key in ALLOWED_KEYS:
        written = cfg.vsl400_path / key
        assert written.exists()
        assert written.read_bytes() == f"content-of-{key}".encode("utf-8")

    # provenance.json exists, records the DOI/URL, timestamp, and file md5s.
    assert provenance_path == cfg.vsl400_path / "provenance.json"
    assert provenance_path.exists()
    prov = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert prov["doi"] == "10.5281/zenodo.17943574"
    assert prov["record_id"] == cfg.zenodo_record_id
    assert "retrieved_at" in prov and prov["retrieved_at"]
    assert sorted(f["key"] for f in prov["files"]) == sorted(ALLOWED_KEYS)
    for entry in prov["files"]:
        assert entry["md5"] == entry["expected_md5"]

    # The injected session was closed only if it owned it — here we own it, so
    # fetch_labels must NOT close our injected session.
    assert session.closed is False


# --------------------------------------------------------------------------- #
# Face models: written + provenance recorded (Req 8.6)
# --------------------------------------------------------------------------- #
def _make_model_session(models: tuple[ModelSpec, ...]) -> tuple[FakeSession, dict]:
    contents = {
        spec.url: (f"ONNX-MODEL-{spec.filename}".encode("utf-8") * 8) for spec in models
    }
    return FakeSession(file_contents=contents), contents


def test_fetch_face_models_writes_under_project_and_records_provenance(tmp_path):
    cfg = Config(project_root=tmp_path)
    models = (
        ModelSpec(
            filename="face_detection_yunet.onnx",
            url="https://example.test/yunet.onnx",
            min_size_bytes=1,
            kind="detector",
        ),
        ModelSpec(
            filename="face_recognition_sface.onnx",
            url="https://example.test/sface.onnx",
            min_size_bytes=1,
            kind="recognizer",
        ),
    )
    session, contents = _make_model_session(models)

    provenance_path = fetch_face_models(cfg, session=session, models=models)

    # Models written under the project root (same drive as project_root).
    for spec in models:
        written = cfg.models_path / spec.filename
        assert written.exists()
        assert written.read_bytes() == contents[spec.url]
        # Path stays under the configured project root (never escapes to C:).
        assert str(written).startswith(str(cfg.project_root))

    # provenance.json records the source + timestamp + each model.
    assert provenance_path == cfg.models_path / "provenance.json"
    prov = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert "opencv_zoo" in prov["source"]
    assert "retrieved_at" in prov and prov["retrieved_at"]
    recorded = {m["filename"] for m in prov["models"]}
    assert recorded == {spec.filename for spec in models}
    for m in prov["models"]:
        assert m["skipped"] is False


def test_fetch_face_models_default_specs_point_at_opencv_zoo():
    # The default FACE_MODELS are the YuNet detector + SFace recognizer ONNX
    # weights, hosted on the OpenCV Zoo (sanity check, no network).
    assert len(FACE_MODELS) == 2
    kinds = {spec.kind for spec in FACE_MODELS}
    assert kinds == {"detector", "recognizer"}
    assert all(spec.filename.endswith(".onnx") for spec in FACE_MODELS)
    assert all("opencv_zoo" in spec.url for spec in FACE_MODELS)

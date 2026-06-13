"""Persistent clip store for batched / incremental conversion.

When the full video set is too large to keep on disk at once, clips can be
processed in batches: add a batch's videos, run the conversion, then delete those
videos and bring in the next batch. To still cluster and emit **all** clips
correctly — including ones whose video files have since been deleted — everything
needed to produce a clip's output record and to cluster it is accumulated in this
store, keyed by ``video_id``:

* the QIPEDC-native fields (``gloss``, ``region``, ``topic``, ``stt``, ``id``),
* the probed video properties (``fps``, ``resolution``, ``num_frames``,
  ``length_seconds``), captured while the video was still present, and
* the face embedding (or a no-face marker).

The store persists as two files under the output dir:

* ``clip_store.json``           — the per-clip records + props (human-readable),
* ``clip_store_embeddings.npz`` — the per-clip embeddings (compact binary).

Clustering on each batch run is performed over **all** embeddings in the store
(old + new), so a new clip of a previously-seen person joins that person's
cluster, and the emitted metadata covers every clip ever processed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

_STORE_VERSION = 1


@dataclass
class ClipEntry:
    """Everything needed to emit one output record without the video present.

    ``has_face`` records whether a usable face embedding was obtained; the actual
    embedding vector is stored separately (in the ``.npz`` companion) to keep the
    JSON small and readable.
    """

    video: str  # original QIPEDC VIDEO filename, e.g. "D0530.mp4"
    video_id: str  # output id (VIDEO stem unless keep_extension)
    gloss: str
    region: str | None
    topic: str | None
    stt: str | None
    id: str | None
    fps: float
    resolution: int
    num_frames: int
    length_seconds: float
    has_face: bool


class ClipStore:
    """An accumulating, on-disk collection of :class:`ClipEntry` + embeddings.

    Keyed by ``video_id``. Use :meth:`load` to read an existing store (or get an
    empty one), :meth:`upsert` to add/replace a clip, :meth:`prune` to drop
    clips, and :meth:`save` to persist.
    """

    def __init__(
        self,
        entries: dict[str, ClipEntry] | None = None,
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.entries: dict[str, ClipEntry] = entries or {}
        # Only clips with ``has_face`` have an embedding here.
        self.embeddings: dict[str, np.ndarray] = embeddings or {}

    # --- persistence --------------------------------------------------------

    @classmethod
    def load(cls, json_path: "Path | str", npz_path: "Path | str") -> "ClipStore":
        """Load a store from its JSON + ``.npz`` files (empty if absent)."""
        json_path = Path(json_path)
        entries: dict[str, ClipEntry] = {}
        if json_path.is_file():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for rec in data.get("clips", []):
                entry = ClipEntry(**rec)
                entries[entry.video_id] = entry

        embeddings: dict[str, np.ndarray] = {}
        npz_path = Path(npz_path)
        if npz_path.is_file():
            with np.load(npz_path, allow_pickle=False) as bundle:
                ids = [str(v) for v in bundle["video_ids"]]
                emb = bundle["emb"]
                embeddings = {
                    vid: np.asarray(emb[i], dtype=np.float64)
                    for i, vid in enumerate(ids)
                }
        return cls(entries=entries, embeddings=embeddings)

    def save(self, json_path: "Path | str", npz_path: "Path | str") -> None:
        """Persist the store to its JSON + ``.npz`` files (deterministic order)."""
        json_path = Path(json_path)
        npz_path = Path(npz_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        ordered_ids = sorted(self.entries)
        payload = {
            "version": _STORE_VERSION,
            "clips": [asdict(self.entries[vid]) for vid in ordered_ids],
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        face_ids = [vid for vid in ordered_ids if vid in self.embeddings]
        if face_ids:
            emb = np.stack(
                [
                    np.asarray(self.embeddings[vid], dtype=np.float64).reshape(-1)
                    for vid in face_ids
                ],
                axis=0,
            )
        else:
            emb = np.zeros((0, 0), dtype=np.float64)
        np.savez(
            npz_path,
            video_ids=np.array(face_ids, dtype=str),
            emb=emb,
        )

    # --- mutation -----------------------------------------------------------

    def upsert(self, entry: ClipEntry, embedding: np.ndarray | None) -> None:
        """Add or replace a clip's record + embedding (keyed by ``video_id``)."""
        self.entries[entry.video_id] = entry
        if embedding is not None:
            self.embeddings[entry.video_id] = np.asarray(
                embedding, dtype=np.float64
            ).reshape(-1)
        else:
            # Ensure no stale embedding lingers if a clip flipped to no-face.
            self.embeddings.pop(entry.video_id, None)

    def prune(self, video_ids: "set[str] | list[str]") -> int:
        """Remove the given ``video_id`` clips from the store. Returns count removed."""
        removed = 0
        for vid in set(video_ids):
            if vid in self.entries:
                del self.entries[vid]
                removed += 1
            self.embeddings.pop(vid, None)
        return removed

    def prune_missing(self, keep_video_ids: "set[str] | list[str]") -> int:
        """Drop every clip whose ``video_id`` is not in *keep_video_ids*.

        Returns the number of clips removed.
        """
        keep = set(keep_video_ids)
        to_remove = [vid for vid in self.entries if vid not in keep]
        return self.prune(to_remove)

    # --- accessors ----------------------------------------------------------

    def __contains__(self, video_id: str) -> bool:
        return video_id in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def ordered_ids(self) -> list[str]:
        """Return all ``video_id``s in deterministic (sorted) order."""
        return sorted(self.entries)

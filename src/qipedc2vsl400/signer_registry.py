"""Persistent signer registry for *stable* signer ids across runs.

The default clustering (``signer_extractor.assign_signers``) re-clusters every
run and numbers clusters by their minimum ``VIDEO`` — correct grouping, but the
numeric ``signer_id`` of a given person may change between runs. When you process
clips in batches and want a person to keep the **same** ``signer_id`` forever,
this registry assigns ids incrementally against stored per-signer representatives
(centroids):

* For each clip embedding, find the nearest registered signer centroid. If it is
  within ``cfg.signer_cosine_threshold`` -> assign that existing ``signer_id``
  (and fold the embedding into its running-mean centroid). Otherwise -> mint the
  next new ``signer_id``.
* New people only ever *append* new numbers; existing numbers never change.

Trade-offs (documented for the caller):

* The assignment is order-dependent (online), not a global optimum.
* It never *merges* two already-registered signer ids even if later evidence
  shows they are the same person; use a full re-cluster (rebuild) for that.

The registry persists as a small human-readable JSON (`signer_registry.json`):
one entry per signer with its ``count`` and its centroid vector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_REGISTRY_VERSION = 1


def _cosine_distance(vec: np.ndarray, ref: np.ndarray) -> float:
    """Cosine distance ``1 - cos_sim``; zero-norm inputs yield ``0.0``."""
    vn = float(np.linalg.norm(vec))
    rn = float(np.linalg.norm(ref))
    if vn == 0.0 or rn == 0.0:
        return 0.0
    sim = float(np.dot(vec, ref) / (vn * rn))
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


class SignerRegistry:
    """A persistent map of stable ``signer_id`` -> (centroid, count)."""

    def __init__(
        self,
        centroids: dict[str, np.ndarray] | None = None,
        counts: dict[str, int] | None = None,
        next_number: int = 1,
    ) -> None:
        self.centroids: dict[str, np.ndarray] = centroids or {}
        self.counts: dict[str, int] = counts or {}
        self.next_number = next_number

    # --- persistence --------------------------------------------------------

    @classmethod
    def load(cls, path: "Path | str") -> "SignerRegistry":
        path = Path(path)
        if not path.is_file():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        centroids: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for entry in data.get("signers", []):
            sid = str(entry["signer_id"])
            centroids[sid] = np.asarray(entry["centroid"], dtype=np.float64)
            counts[sid] = int(entry["count"])
        next_number = int(data.get("next_number", len(centroids) + 1))
        return cls(centroids=centroids, counts=counts, next_number=next_number)

    def save(self, path: "Path | str") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _REGISTRY_VERSION,
            "next_number": self.next_number,
            "signers": [
                {
                    "signer_id": sid,
                    "count": self.counts[sid],
                    "centroid": [float(x) for x in self.centroids[sid]],
                }
                for sid in sorted(self.centroids)
            ],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --- helpers ------------------------------------------------------------

    def is_empty(self) -> bool:
        return not self.centroids

    def _mint(self, width: int) -> str:
        sid = str(self.next_number).zfill(width)
        self.next_number += 1
        return sid

    def _register(self, sid: str, embedding: np.ndarray) -> None:
        self.centroids[sid] = np.asarray(embedding, dtype=np.float64).reshape(-1)
        self.counts[sid] = 1

    def _update(self, sid: str, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float64).reshape(-1)
        n = self.counts[sid]
        self.centroids[sid] = (self.centroids[sid] * n + emb) / (n + 1)
        self.counts[sid] = n + 1

    # --- assignment ---------------------------------------------------------

    def assign(
        self, embedding: np.ndarray, cfg: Any
    ) -> tuple[str, float | None]:
        """Assign *embedding* to the nearest signer within threshold, else a new one.

        Returns ``(signer_id, distance)`` where ``distance`` is the cosine
        distance to the matched centroid (``None`` for a freshly minted signer).
        Updates the registry in place.
        """
        width = cfg.signer_id_width
        threshold = cfg.signer_cosine_threshold

        best_sid: str | None = None
        best_dist = float("inf")
        for sid, centroid in self.centroids.items():
            dist = _cosine_distance(embedding, centroid)
            if dist < best_dist or (
                dist == best_dist and (best_sid is None or sid < best_sid)
            ):
                best_dist = dist
                best_sid = sid

        if best_sid is not None and best_dist <= threshold:
            self._update(best_sid, embedding)
            return best_sid, best_dist

        sid = self._mint(width)
        self._register(sid, embedding)
        return sid, None

    def assign_all(
        self,
        order_keys: list[str],
        embeddings: list[np.ndarray | None],
        cfg: Any,
    ) -> list[tuple[str, int, float | None, bool]]:
        """Assign every embedding (in input order), reusing existing signer ids.

        Mirrors the tuple shape of ``signer_extractor.assign_signers``:
        ``(signer_id, cluster_index, distance, has_face)``. ``None`` embeddings
        go to the unknown bucket. Process *order_keys* pre-sorted for determinism.
        """
        result: list[tuple[str, int, float | None, bool]] = []
        for emb in embeddings:
            if emb is None:
                result.append((cfg.signer_unknown_label, -1, None, False))
                continue
            sid, dist = self.assign(emb, cfg)
            cluster_index = int(sid) - 1 if sid.isdigit() else -1
            result.append((sid, cluster_index, dist, True))
        return result

    def rebuild_from(
        self,
        order_keys: list[str],
        embeddings: list[np.ndarray | None],
        assignments: list[tuple[str, int, float | None, bool]],
        cfg: Any,
    ) -> None:
        """Reset the registry from a completed (global) clustering.

        Centroid of each ``signer_id`` becomes the mean of its members'
        embeddings; ``next_number`` continues past the largest numeric id. Used
        to seed the registry on the first run and to re-seed after ``--recluster``.
        """
        self.centroids = {}
        self.counts = {}
        members: dict[str, list[np.ndarray]] = {}
        for emb, (sid, _ci, _dist, has_face) in zip(embeddings, assignments):
            if not has_face or emb is None:
                continue
            members.setdefault(sid, []).append(
                np.asarray(emb, dtype=np.float64).reshape(-1)
            )
        max_num = 0
        for sid, vecs in members.items():
            self.centroids[sid] = np.mean(np.stack(vecs, axis=0), axis=0)
            self.counts[sid] = len(vecs)
            if sid.isdigit():
                max_num = max(max_num, int(sid))
        self.next_number = max_num + 1

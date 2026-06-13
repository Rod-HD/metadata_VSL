"""Emit the converted superset metadata as a VSL400-style view JSON (Req 5).

:func:`write_outputs` serializes a list of
:class:`~qipedc2vsl400.mapper.OutputRecord` objects to a JSON **array** at
``cfg.output_view_file`` (e.g. ``Dataset/final_dataset/front_view.json``).

The write is **deterministic** (Requirement 5.4 / design Property 4): records are
sorted ascending by ``video_id`` before serialization, so re-running the
converter on unchanged input yields byte-identical output. Vietnamese text is
preserved in UTF-8 with diacritics intact (``ensure_ascii=False``) rather than
ASCII-escaped to ``\\uXXXX`` (Requirement 5.2). The output directory is created
if it does not yet exist.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any


def write_outputs(records: list[Any], cfg: Any) -> None:
    """Write the superset metadata array to ``cfg.output_view_file``.

    Args:
        records: The :class:`~qipedc2vsl400.mapper.OutputRecord` objects to
            emit. They are sorted ascending by ``video_id`` for deterministic
            output; the input list is not mutated.
        cfg: A :class:`~qipedc2vsl400.config.Config`-like object exposing
            ``output_view_file`` (the absolute ``front_view.json`` path).

    Side effects:
        Creates the output directory if needed and writes a UTF-8 JSON file with
        ``indent=2`` and ``ensure_ascii=False`` (Vietnamese diacritics
        preserved). Re-running on unchanged ``records`` produces byte-identical
        output (Requirement 5.4).
    """
    output_file = cfg.output_view_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(records, key=lambda r: r.video_id)
    payload = [asdict(record) for record in ordered]

    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

#!/usr/bin/env python3
"""
merge_splits.py

Merge all split_* folders in vsl400_blurred_splits into a single merged
output directory, combining front/left/right videos and their JSON metadata.

Default behavior:
- Reads split_1 .. split_N under the current directory.
- Writes merged data to ./merged/ with subfolders front_view/, left_view/, right_view/.
- Merges JSON files per view into merged/front_view.json (and left/right).
- Copies files by default; can hardlink or symlink instead.

Usage:
    python merge_splits.py [--splits-root .] [--out merged] [--copy-mode copy|hardlink|symlink]
                           [--overwrite] [--stop-on-dup]

Examples:
    python merge_splits.py
    python merge_splits.py --out merged_all --copy-mode hardlink
    python merge_splits.py --splits-root . --out merged --stop-on-dup
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("merge_splits")

VIEWS = ("front_view", "left_view", "right_view")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge split_* folders into one dataset")
    p.add_argument("--splits-root", type=str, default=".", help="Path containing split_* folders")
    p.add_argument("--out", type=str, default="merged", help="Output directory for merged dataset")
    p.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="copy",
                   help="How to place video files in the merged directory")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files in output")
    p.add_argument("--stop-on-dup", action="store_true", help="Fail if duplicate video_ids are found across splits")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def find_splits(root: Path) -> List[Path]:
    splits = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("split_")]
    splits.sort(key=lambda p: int(re.sub(r"[^0-9]", "", p.name) or 0))
    return splits


def copy_file(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)
        except Exception:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def merge_json_lists(json_paths: List[Path], stop_on_dup: bool) -> List[dict]:
    merged: List[dict] = []
    seen_ids = set()

    def get_id(obj: dict) -> Optional[str]:
        for k in ("video_id", "id", "name"):
            if k in obj:
                return str(obj[k])
        return None

    for jp in json_paths:
        if not jp.exists():
            logger.warning("JSON missing, skipping: %s", jp)
            continue
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load JSON: %s", jp)
            continue

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                vid = get_id(item)
                if vid is not None:
                    if stop_on_dup and vid in seen_ids:
                        raise RuntimeError(f"Duplicate video_id {vid} found in {jp}")
                    if vid in seen_ids:
                        continue
                    seen_ids.add(vid)
                merged.append(item)
        elif isinstance(data, dict):
            for vid, item in data.items():
                if stop_on_dup and vid in seen_ids:
                    raise RuntimeError(f"Duplicate video_id {vid} found in {jp}")
                if vid in seen_ids:
                    continue
                seen_ids.add(vid)
                if isinstance(item, dict):
                    merged.append({"video_id": vid, **item})
                else:
                    merged.append({"video_id": vid, "value": item})
        else:
            logger.warning("Unexpected JSON root (ignored): %s", jp)

    return merged


def merge_dataset(root: Path, out_dir: Path, copy_mode: str, overwrite: bool, stop_on_dup: bool) -> None:
    splits = find_splits(root)
    if not splits:
        raise RuntimeError(f"No split_* directories found under {root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # merge JSONs per view
    for view in VIEWS:
        json_paths = [s / f"{view}.json" for s in splits]
        merged_list = merge_json_lists(json_paths, stop_on_dup=stop_on_dup)
        out_json = out_dir / f"{view}.json"
        out_json.write_text(json.dumps(merged_list, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Merged %s entries -> %s", view, out_json)

    # merge video files per view
    for view in VIEWS:
        out_view_dir = out_dir / view
        for s in splits:
            view_dir = s / view
            if not view_dir.exists():
                logger.warning("Missing view dir %s in split %s", view_dir, s)
                continue
            for src in view_dir.iterdir():
                if not src.is_file() or src.suffix.lower() not in VIDEO_EXTS:
                    continue
                dst = out_view_dir / src.name
                copy_file(src, dst, mode=copy_mode, overwrite=overwrite)
        logger.info("Merged videos for %s -> %s", view, out_view_dir)



def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(levelname)s: %(message)s')

    splits_root = Path(args.splits_root)
    out_dir = Path(args.out)

    try:
        merge_dataset(splits_root, out_dir, copy_mode=args.copy_mode,
                      overwrite=args.overwrite, stop_on_dup=args.stop_on_dup)
    except Exception as e:
        logger.exception("Merge failed: %s", e)
        return 1

    logger.info("Merge completed. Output: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

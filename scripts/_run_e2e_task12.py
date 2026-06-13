"""Temporary Task-12 E2E runner: run the full pipeline over a single label file.

Mirrors ``qipedc2vsl400.convert.main`` but overrides ``qipedc_labels_glob`` to
point at only ``batch_1.xlsx`` (the duplicate ``batch_1(1).xlsx`` is left on disk
untouched). Accepts an optional ``--signer-threshold`` for tuning.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from qipedc2vsl400.config import Config
from qipedc2vsl400.convert import run_pipeline

# Project root inferred from this file's location (scripts/ is one level down).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signer-threshold", type=float, default=None)
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--label-file", default="Dataset/labels/batch_1.xlsx")
    args = parser.parse_args()

    cfg = Config(project_root=PROJECT_ROOT, qipedc_labels_glob=args.label_file)
    if args.signer_threshold is not None:
        cfg = dataclasses.replace(cfg, signer_cosine_threshold=args.signer_threshold)

    print(f"Running E2E: labels={args.label_file} "
          f"threshold={cfg.signer_cosine_threshold} fetch={not args.no_fetch}",
          flush=True)

    result = run_pipeline(cfg, fetch=not args.no_fetch)

    print(result.summary, flush=True)
    if not result.verify_report.ok:
        print("Verification FAILED:", file=sys.stderr)
        for failure in result.verify_report.failures:
            print(f"  - {failure.name}: {failure.detail}", file=sys.stderr)
    else:
        print("Verification PASSED", flush=True)

    # Per-signer counts for manual review.
    from collections import Counter
    counts = Counter(a.signer_id for a in result.signer_assignments)
    print("\nPer-signer counts:", flush=True)
    for signer_id in sorted(counts, key=lambda s: (s == cfg.signer_unknown_label, s)):
        print(f"  {signer_id}: {counts[signer_id]}", flush=True)
    print(f"distinct real signers: "
          f"{len([s for s in counts if s != cfg.signer_unknown_label])}", flush=True)
    print(f"exit_code={result.exit_code}", flush=True)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

# QIPEDC → VSL400 Metadata Conversion

Convert our 1000-word **QIPEDC** sign-language subset into the **VSL400** metadata format
(Zenodo DOI `10.5281/zenodo.17943574`). The output is a *superset* of VSL400: it keeps QIPEDC's own
fields (`region`, `topic`, `stt`, `id`), renames `LABEL` → `gloss`, keeps `video_id` from QIPEDC's
`VIDEO` (extension stripped), and adds the VSL400-derived fields (`signer_id`, `fps`, `resolution`,
`num_frames`, `length_seconds`).

Since QIPEDC has no signer data, `signer_id` is **extracted from video content** (OpenCV YuNet face
detection + SFace embeddings, threshold-based clustering) and clips are organized into per-signer
folders for manual verification.

## Project layout

```
src/qipedc2vsl400/      # pipeline package (reader, probe, signer extractor, mapper, writer, ...)
tests/                  # unit + property-based (Hypothesis) tests
scripts/                # helper / verification scripts
Dataset/final_dataset/  # output: front_view.json, signers.csv (tracked)
Dataset/labels/         # source QIPEDC labels (*.xlsx, tracked)
```

Large media (`Dataset/raw_videos/`, `Dataset/processed_videos/`, `Dataset/by_signer/`),
downloaded models (`Dataset/models/`), the virtual environment (`.venv/`), and caches are
git-ignored. **Nothing is installed on `C:` — everything stays under the project on `D:`.**

## Setup (Windows / PowerShell)

Creates a project-local `.venv` on `D:`, redirects pip's cache to `D:`, and installs pinned deps:

```powershell
./setup_env.ps1
```

## End-to-end run (from scratch)

Follow these steps on a fresh machine to go from raw inputs all the way to the converted
metadata. Everything stays under the project folder on `D:` (no `C:` installs).

### 1. Get the QIPEDC data into place

The QIPEDC source data is **not** part of this repository (only the small label spreadsheet and the
final metadata are tracked). Put the inputs where the pipeline expects them:

```
Dataset/labels/<your-labels>.xlsx              # QIPEDC label files (one or more)
Dataset/processed_videos/resize_720p/*.mp4     # the clips (preferred 720p source)
Dataset/raw_videos/**/*.mp4                     # optional fallback source
```

- Label spreadsheets are read from `Dataset/labels/*.xlsx`. Each must have the header row
  `STT, ID, VIDEO, LABEL, REGION, TOPIC, signer` (matched case-insensitively); multiple files are
  combined automatically.
- Each row's `VIDEO` value (e.g. `D0530.mp4`) must resolve to a real file under
  `Dataset/processed_videos/resize_720p/` (searched first) or `Dataset/raw_videos/` (recursive
  fallback). Rows whose video is missing/unreadable are skipped and logged, not fatal.

> The VSL400 reference labels and the face-detection models are downloaded **automatically** by the
> run in step 3 — you do not fetch them by hand.

### 2. Create the environment

Creates a project-local `.venv` on `D:`, redirects pip's cache to `D:`, and installs pinned deps:

```powershell
./setup_env.ps1
```

### 3. Run the conversion

The package is run as a module. Since there is no editable install, add `src` to `PYTHONPATH` and
run from the project root using the venv interpreter:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert
```

This runs the full pipeline in order:

```
fetch → read → probe → extract-signers → map → write → organize-by-signer → verify
```

On the **first** run (without `--no-fetch`) it downloads:

- the VSL400 reference labels from Zenodo into `Dataset/labels/vsl400/`, and
- the YuNet + SFace ONNX face models from the OpenCV Zoo into `Dataset/models/` (on `D:`).

Signer extraction then runs face detection/embedding over every clip and clusters them, so the first
run takes a while (≈ several minutes for ~1000 clips). A non-zero exit code means verification
failed; check the run log.

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--no-fetch` | Skip the Zenodo label + ONNX model download (use after the first run). |
| `--skip-signer` | Reuse an existing `signers.csv` instead of re-embedding every clip. |
| `--embeddings-cache` | Cache per-clip face embeddings and reuse them on re-runs, so only newly added clips are embedded (see *Adding new clips* below). |
| `--batch` | Batch/incremental mode using a persistent clip store, for video sets too large to hold on disk at once (see *Batch mode* below). |
| `--prune-missing` | With `--batch`, drop stored clips not in the current label rows (use only when the label file lists the full dataset). |
| `--stable-signers` | With `--batch`, keep `signer_id` stable across runs via a persistent signer registry (see *Stable signer ids* below). |
| `--recluster` | With `--batch --stable-signers`, re-cluster the whole store and re-seed the registry (numbers may change). |
| `--on-missing-video {skip,placeholder}` | Policy when a referenced video is missing/unreadable (default: `skip`). |
| `--copy-mode {hardlink,copy}` | How clips are placed into per-signer folders (default: `hardlink`). |
| `--signer-threshold <float>` | Cosine "same-signer" distance threshold for clustering (default `0.363`). |
| `--project-root <path>` | Override the project root (defaults to the repository root). |

A fast re-run that reuses the previous signer extraction and skips downloads:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --no-fetch --skip-signer
```

### Adding new clips later (incremental embedding cache)

When you add more videos (with matching rows in the label spreadsheet) and re-run, signer
extraction normally re-embeds **every** clip — the costly step. Pass `--embeddings-cache` to keep a
cache of per-clip face embeddings (`Dataset/final_dataset/embeddings.npz`, git-ignored) so only the
**new** clips are embedded; the cached vectors are reused for everything seen before:

```powershell
$env:PYTHONPATH = "src"
# first run: builds the cache while converting
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --embeddings-cache
# after adding new clips + their label rows: only new clips get embedded
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --no-fetch --embeddings-cache
```

The cache is purely a speed optimization: clustering still runs over **all** clips (cached + new),
so the result is identical to a full re-run. New clips of an existing person join that person's
cluster (same `signer_id`); new people get new clusters. Within any single run, all clips of one
person always share one `signer_id`. (Note: the numeric label assigned to a given person — e.g.
`signer_001` — may differ between runs, since clusters are numbered by their smallest `VIDEO`
filename; within one output it is always consistent.)

> Do **not** combine `--embeddings-cache` with `--skip-signer`: `--skip-signer` bypasses extraction
> entirely (reusing `signers.csv`), so new clips would all fall into the `unknown` bucket.

### Batch mode for very large video sets (`--batch`)

If the full video set is too large to keep on disk at once, process it in batches: add a batch's
videos, run, delete them, then bring in the next batch. `--batch` accumulates everything needed to
emit and cluster each clip into a persistent **clip store** (`Dataset/final_dataset/clip_store.json`
+ `clip_store_embeddings.npz`, both git-ignored), so the output always covers **all** clips ever
processed — even ones whose video files have since been deleted.

```powershell
$env:PYTHONPATH = "src"
# Batch 1: put batch-1 videos + their label rows in place, then:
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --batch
# delete batch-1 videos, add batch-2 videos + label rows, then:
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --batch
# ...repeat for each batch
```

How it behaves each run:

- A clip already in the store is kept as-is (its video need not be present).
- A clip new to the store is probed + embedded from its on-disk video and added to the store.
- A clip in the labels but with no on-disk video and not yet in the store is skipped-and-logged.

Clustering runs over **all** stored embeddings (old + new), so a new clip of a previously-seen
person joins that person's cluster and shares its `signer_id` — even if that person's earlier videos
were deleted. (`signer_id` numbers may still be relabeled between runs, as noted above.)

Notes / limits:

- `--batch` assumes the ONNX face models are already downloaded (run once normally, or with
  `--embeddings-cache`, first). It does not fetch.
- `Dataset/by_signer/` can only contain clips whose videos are currently on disk; clips deleted in
  earlier batches won't appear there for manual review (their metadata is still emitted).
- `--prune-missing` (with `--batch`) drops stored clips whose `video_id` is not in the **current**
  label rows. Use it only when the label file lists the full intended dataset; otherwise it would
  drop earlier batches.

### Stable signer ids across runs (`--stable-signers`)

By default the `signer_id` *number* assigned to a given person can change between runs (clusters are
renumbered by their smallest `VIDEO`). To keep a person's number fixed forever, add
`--stable-signers` (with `--batch`). It maintains a persistent **signer registry**
(`Dataset/final_dataset/signer_registry.json`, git-ignored) holding one representative embedding
(centroid) per signer:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --batch --stable-signers
```

Each run, every clip is matched to the nearest registered signer centroid: within
`--signer-threshold` → it keeps that existing `signer_id` (and updates the centroid); otherwise a new
number is appended. So a new clip of a previously-seen person always lands on that person's existing
number — even if their earlier videos were deleted — and existing numbers never change.

How clustering compares people: each clip is reduced to a 128-D **face embedding** (OpenCV SFace);
two clips are "the same signer" when their embeddings are close in **cosine distance**. These
embeddings are exactly what the clip store / registry persist.

Trade-offs:

- Assignment is order-dependent (online), not a global optimum.
- It never *merges* two already-registered signer ids even if later evidence shows they are the same
  person. Run `--recluster` (with `--batch --stable-signers`) to re-cluster the whole store globally
  and re-seed the registry when you want to fix that (numbers may change on that run).

### 4. Outputs

```
Dataset/final_dataset/front_view.json   # the converted metadata (JSON array, sorted by video_id)
Dataset/final_dataset/signers.csv       # per-clip signer assignment side-car
Dataset/by_signer/signer_XXX/           # clips grouped per extracted signer (+ signer_unknown/)
Dataset/by_signer/_summary.txt          # clip counts per signer
Dataset/logs/convert_<timestamp>.log    # full run log
```

Open each `Dataset/by_signer/signer_XXX/` folder to manually verify that the extracted signers look
correct. If a real signer was split into two folders (or two signers merged into one), re-run with a
tuned `--signer-threshold` (raise it to merge more, lower it to split more).

## Test

```powershell
python -m pytest
```

## Requirements

See `requirements.txt` for pinned dependencies (`openpyxl`, `opencv-python-headless`, `numpy`,
`scikit-learn`, `requests`, `pytest`, `hypothesis`).

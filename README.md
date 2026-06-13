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
| `--on-missing-video {skip,placeholder}` | Policy when a referenced video is missing/unreadable (default: `skip`). |
| `--copy-mode {hardlink,copy}` | How clips are placed into per-signer folders (default: `hardlink`). |
| `--signer-threshold <float>` | Cosine "same-signer" distance threshold for clustering (default `0.363`). |
| `--project-root <path>` | Override the project root (defaults to the repository root). |

A fast re-run that reuses the previous signer extraction and skips downloads:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --no-fetch --skip-signer
```

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

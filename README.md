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

## Run the conversion

```powershell
# from the project root, with the venv active
python -m qipedc2vsl400.convert
```

Useful flags: `--no-fetch`, `--on-missing-video {skip|error}`, `--skip-signer`,
`--copy-mode {hardlink|copy}`, `--signer-threshold <float>`. Logs are written to `Dataset/logs/`.

## Test

```powershell
python -m pytest
```

## Requirements

See `requirements.txt` for pinned dependencies (`openpyxl`, `opencv-python-headless`, `numpy`,
`scikit-learn`, `requests`, `pytest`, `hypothesis`).

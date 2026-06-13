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

---

## Hướng dẫn theo từng trường hợp (Tiếng Việt)

Tất cả lệnh chạy **từ thư mục gốc của project** bằng trình Python trong `.venv`. Mở PowerShell,
`cd` vào thư mục project, rồi mỗi phiên làm việc đặt biến môi trường một lần:

```powershell
$env:PYTHONPATH = "src"
```

> Vì package nằm trong `src\` và không cài editable, cần `$env:PYTHONPATH = "src"` trước khi chạy
> `-m qipedc2vsl400.convert`. Có thể dùng trực tiếp `.\.venv\Scripts\python.exe` mà không cần
> "activate" venv.

### Cơ chế phân cụm signer (để hiểu các mode bên dưới)

Mỗi clip được rút thành một **vector embedding khuôn mặt 128 chiều** (OpenCV YuNet phát hiện mặt →
SFace sinh embedding → trung bình, chuẩn hóa L2). Hai clip được coi là **cùng một người** khi hai
vector gần nhau theo **cosine distance** (ngưỡng `--signer-threshold`, mặc định `0.363`). Các vector
này là thứ được lưu lại trong cache / clip store / registry.

Lưu ý chung về đánh số: trong **một lần chạy**, mọi clip của một người luôn cùng một `signer_id`.
Nhưng **con số** `signer_001/002/...` có thể đổi nhãn giữa các lần chạy (trừ khi dùng
`--stable-signers`).

---

### Trường hợp 1 — Chạy lần đầu từ đầu (máy mới)

1. Đặt dữ liệu đúng chỗ:
   - Label QIPEDC: `Dataset\labels\*.xlsx` (header: `STT, ID, VIDEO, LABEL, REGION, TOPIC, signer`).
   - Video: `Dataset\processed_videos\resize_720p\*.mp4` (và/hoặc `Dataset\raw_videos\`).
2. Tạo môi trường venv trên D::
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
   ```
3. Chạy chuyển đổi (lần đầu tự tải VSL400 labels + model khuôn mặt):
   ```powershell
   $env:PYTHONPATH = "src"
   .\.venv\Scripts\python.exe -m qipedc2vsl400.convert
   ```
4. Kết quả: `Dataset\final_dataset\front_view.json`, `signers.csv`, `Dataset\by_signer\...`,
   log trong `Dataset\logs\`. Mã thoát `0` nghĩa là verify đạt (`echo $LASTEXITCODE`).

### Trường hợp 2 — Đã chạy 1 lần, thêm video mới (KHÔNG xóa video cũ)

Vẫn giữ toàn bộ video cũ trên đĩa. Nhớ thêm **dòng label** cho video mới vào `*.xlsx`.

- Cách nhanh (chỉ embed clip mới, dùng lại embedding cũ qua cache):
  ```powershell
  $env:PYTHONPATH = "src"
  # lần đầu bật cache để dựng cache:
  .\.venv\Scripts\python.exe -m qipedc2vsl400.convert --embeddings-cache
  # sau khi thêm video mới + dòng label, chỉ clip mới được embed:
  .\.venv\Scripts\python.exe -m qipedc2vsl400.convert --no-fetch --embeddings-cache
  ```
- `--no-fetch`: bỏ tải mạng (model + VSL400 labels đã có).
- **Không** dùng `--skip-signer` ở đây (nó bỏ qua trích xuất, clip mới sẽ rơi vào `unknown`).
- Kết quả: phân cụm lại trên toàn bộ (cũ + mới) → đúng; clip mới trùng người với clip cũ vào chung
  cụm. Số `signer_id` có thể đổi nhãn so với lần trước.

### Trường hợp 3 — Video quá nặng, chạy theo từng đợt (xóa cũ, thêm mới)

Dùng chế độ batch: clip store tích lũy nên metadata cuối luôn đủ cả các đợt, kể cả video đã xóa.

```powershell
$env:PYTHONPATH = "src"
# Đợt 1: bỏ video đợt 1 + dòng label tương ứng, rồi:
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --batch
# Xóa video đợt 1, bỏ video đợt 2 + dòng label, rồi:
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --batch
# ... lặp lại cho từng đợt
```

- `--batch` giả định **model đã tải sẵn** (chạy Trường hợp 1 một lần trước, hoặc đã có
  `Dataset\models\`). Nó không tự fetch.
- Phân cụm chạy trên **toàn bộ kho** (cũ + mới) → clip mới trùng người với clip cũ (đã xóa video)
  vẫn vào **chung cụm**, và metadata vẫn xuất đủ clip cũ.
- `Dataset\by_signer\` chỉ chứa được video còn trên đĩa; clip của đợt đã xóa sẽ không có thư mục để
  xem lại (nhưng metadata vẫn đủ).

### Trường hợp 4 — Muốn `signer_id` CỐ ĐỊNH giữa các lần chạy

Thêm `--stable-signers` (kết hợp `--batch`). Một người được khớp với signer cũ sẽ **giữ nguyên số**;
người mới mới được cấp số mới; số cũ không bao giờ đổi.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --batch --stable-signers
# các đợt sau cũng thêm --stable-signers
```

- Khi muốn gom lại tối ưu toàn cục + đánh số lại (vd để gộp hai signer thật ra là một người):
  ```powershell
  .\.venv\Scripts\python.exe -m qipedc2vsl400.convert --batch --stable-signers --recluster
  ```
- Lưu ý: chế độ ổn định gán "trực tuyến" (phụ thuộc thứ tự) và **không tự gộp** hai `signer_id` đã
  đăng ký — dùng `--recluster` khi cần gộp/đặt lại số.

### Trường hợp 5 — Chạy lại nhanh trên dữ liệu không đổi

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m qipedc2vsl400.convert --no-fetch --skip-signer
```
`--skip-signer` dùng lại `signers.csv` đã có (không nhúng lại). Chỉ dùng khi **không** thêm clip mới.

### Trường hợp 6 — Đổi đường dẫn video / output / thư mục signer

Các đường dẫn nằm trong `Config` (`src\qipedc2vsl400\config.py`). CLI chỉ cho đổi vài thứ qua flag,
nên đổi đường dẫn thì:

- **Cách A** — sửa giá trị mặc định trong `config.py`. Các field hay đổi: `video_search_dirs`
  (nơi tìm video), `foldering_source` (nguồn gom signer), `by_signer_dir`, `output_dir`,
  `output_view_name`, `qipedc_labels_glob`, `models_dir`. Đường dẫn có thể tương đối (ghép vào
  `project_root`) hoặc tuyệt đối.
- **Cách B** — viết script ghi đè bằng `dataclasses.replace` mà không sửa code gốc:
  ```python
  from pathlib import Path
  import dataclasses
  from qipedc2vsl400.config import Config
  from qipedc2vsl400.convert import run_pipeline

  cfg = Config(project_root=Path.cwd())
  cfg = dataclasses.replace(
      cfg,
      video_search_dirs=("E:/my_videos",),
      foldering_source="E:/my_videos",
      output_dir="E:/output/metadata",
  )
  raise SystemExit(run_pipeline(cfg, fetch=False).exit_code)
  ```

### Trường hợp 7 — Xóa bớt clip khỏi dataset

- Khi muốn loại một clip: xóa **cả dòng label** của nó (và file video). Chỉ xóa file video mà giữ
  dòng label sẽ khiến clip bị skip-and-log (chế độ thường) hoặc bị kéo vào cụm "ma" nếu còn trong
  cache/kho.
- Chế độ batch: dùng `--prune-missing` (với `--batch`) để bỏ khỏi kho các clip **không** có trong
  dòng label hiện tại — chỉ dùng khi file label đang liệt kê **toàn bộ** dataset mong muốn (nếu
  không sẽ xóa nhầm các đợt cũ).

### Bảng chọn nhanh

| Tình huống | Lệnh |
| --- | --- |
| Lần đầu từ đầu | `... convert` |
| Thêm video mới (giữ video cũ) | `... convert --no-fetch --embeddings-cache` |
| Chạy lại nhanh, dữ liệu không đổi | `... convert --no-fetch --skip-signer` |
| Chạy theo đợt (xóa cũ thêm mới) | `... convert --batch` |
| Chạy theo đợt + giữ cố định số signer | `... convert --batch --stable-signers` |
| Gom lại + đánh số lại (chế độ ổn định) | `... convert --batch --stable-signers --recluster` |

(`...` = `.\.venv\Scripts\python.exe -m qipedc2vsl400`, sau khi đã `$env:PYTHONPATH = "src"`.)

### Kiểm tra kết quả nhanh

```powershell
echo $LASTEXITCODE                       # 0 = verify đạt
Get-Content Dataset\by_signer\_summary.txt
Get-ChildItem Dataset\final_dataset\
```

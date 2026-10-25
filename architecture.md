# Custom-Album Architecture

## Purpose

Custom-Album is a local-first desktop photo-library application. It scans folders, detects faces, groups face embeddings into people, and presents People, Timeline, and Album views. The UI and all persisted library data remain on the local machine; no web service or cloud database is required.

## System overview

```text
PySide6 desktop UI (py_app/main.py)
        |
        | direct in-process Python calls
        v
Face service (resources/face_service.py)
        |-------------------|-------------------|
        v                   v                   v
InsightFace + ONNX       FAISS               JSON/image files
detection/embeddings     similarity index    data/
```

`main.py` imports the service directly rather than calling an HTTP API. The service lazily imports the heavyweight native libraries, so the UI can open and display an existing library without starting the face model.

## Components

### Desktop UI — `py_app/main.py`

`AppWindow` owns the primary navigation and scan controls.

- **People** displays person clusters, supports filtering, sorting, inline naming, manual merge, automatic candidate review, and re-clustering.
- **Timeline** groups every discovered image by capture month, including images without a detected face.
- **Albums** groups detected faces by source folder.
- **Dialogs** provide face search, galleries, full-photo bounding-box view, duplicate review, and merge/re-cluster workflows.
- **Thumbnail cache** stores scaled UI previews at `py_app/.cache/thumbnails`. Cache keys include source file size and nanosecond modification time, so replacing a photo at the same path produces a new preview.

The scan itself runs on a daemon `threading.Thread`; `QTimer` polls persisted progress every 1.5 seconds. Actions that would mutate or read the shared in-memory service state are disabled while a scan runs.

### Face service — `resources/face_service.py`

This module is the domain and persistence layer. Its public functions are used by the UI and its `handle()` function exposes an optional JSON-stdin command interface.

Responsibilities include:

- configuration loading and validation;
- image discovery and incremental-scan decisions;
- EXIF/creation/modification-date resolution;
- lazy model initialization and CPU/CUDA provider selection;
- face detection, crop generation, embedding storage, and clustering;
- JSON and FAISS persistence;
- people editing, merging, splitting, duplicate detection, and face search;
- review-only cleanup suggestions for screenshots, filename-based likely forwards, and exact content duplicates;
- CSV library-report export that joins timeline, scan, face, and people records;
- view-model construction for the UI.

### ML and similarity dependencies

- **InsightFace `FaceAnalysis`** uses the `buffalo_s` model for face detection and 512-dimensional embeddings.
- **ONNX Runtime** uses CUDA when `CUDAExecutionProvider` is installed; otherwise CPU is used.
- **FAISS `IndexFlatIP`** stores L2-normalized embeddings. Inner product is therefore cosine similarity.
- **OpenCV** decodes images, scales detection inputs, and writes JPEG face crops.
- **Pillow** reads EXIF timestamps.

## Scan lifecycle

1. The user selects a source folder and chooses Incremental Scan or Full Scan.
2. The service recursively discovers supported images without following directory symlinks.
3. `timeline.json` is updated first, so photos with no detectable face still appear in Timeline.
4. An incremental scan compares `size` and `mtime_ns` against `processed.json`. Older records using second-level `mtime` remain supported.
5. Each changed or unseen image is decoded, optionally downscaled for detection, and sent to InsightFace.
6. Valid detections produce a cropped JPEG and a 512-value embedding. The source-image bounding box is retained alongside the detector-space box.
7. The closest normalized FAISS vector is compared with `similarity_threshold`. A match joins its existing person; otherwise a new person is created.
8. The scanner periodically checkpoints processed metadata, face/person JSON, and the FAISS index, then emits final progress.

### Full-scan semantics

A full scan removes records for the selected folder before rebuilding them. Other source folders remain intact. Face-thumbnail files that become orphaned are deliberately retained: deleting them during a folder-scoped operation risks deleting assets still referenced by another record.

## Persistence model

All runtime data belongs under `data/` and is excluded from Git.

| File | Role |
|---|---|
| `faces.json` | Source of truth for detected faces, embeddings, crop paths, source path, date, album, and person ID. |
| `people.json` | Saved names and face memberships. Membership is reconstructed from `faces.json` when loading, repairing stale or duplicate lists. |
| `processed.json` | Incremental-scan signatures and per-image status. |
| `timeline.json` | Every discovered image and its date, independently of face detection. |
| `faiss.index` | Search index aligned with face embeddings. Rebuilt from stored embeddings whenever possible. |
| `progress.json` | Current scan progress for the UI. |
| `errors.jsonl` | Append-only recovery diagnostics. |
| `cleanup_moves.jsonl` | Audit trail for review-approved no-face photo moves. |
| `faces/<id>.jpg` | Generated face crops. |

JSON writes use a uniquely named temporary sibling, flush and `fsync`, then `os.replace`. The FAISS index follows the same write-then-replace strategy. These operations avoid presenting a partially written file after an interrupted write.

## Data contracts

### Face record

```json
{
  "face_id": 42,
  "person_id": 7,
  "image": "D:/Photos/holiday/photo.jpg",
  "bbox": [12.3, 45.6, 180.1, 220.4],
  "bbox_original": [24.6, 91.2, 360.2, 440.8],
  "score": 0.97,
  "landmarks": [[0, 0]],
  "face_path": ".../data/faces/42.jpg",
  "date": "2026-03",
  "album": "D:/Photos/holiday",
  "embedding": ["512 float values"]
}
```

`bbox` is compatible with older detector-space records. `bbox_original` should be preferred for crop regeneration and overlays.

### Person record

```json
{"id": 7, "name": "Asha", "faces": [42, 55, 98]}
```

The face record's `person_id` is authoritative. Person membership is rebuilt from those IDs on service load; names are retained.

### Incremental record

```json
{"size": 1842312, "mtime": 1760000000, "mtime_ns": 1760000000123456789,
 "status": "ok", "faces": 2, "date": "2026-03"}
```

Statuses `ok` and `invalid_image` are skipped when their signature is unchanged. Decode/model failures are retried in a later scan because they may be transient.

## Clustering and search behaviour

New faces are compared with the nearest individual stored embedding, then assigned according to the configured similarity threshold. Re-clustering repeats this ordered, nearest-neighbour process using all valid stored embeddings. This is intentionally simple and incremental; it is not a global clustering algorithm, so scan order can influence cluster boundaries.

Auto-merge computes one centroid per person and offers the two nearest other clusters per person. It is a review tool, not an automatic background action. Batch merges mutate in memory and write once at the end, avoiding repeated full-database rewrites.

## Configuration

Optional `resources/config.json` values override defaults:

| Key | Default | Meaning |
|---|---:|---|
| `similarity_threshold` | 0.45 | Similarity required to join an existing person. |
| `face_threshold` | 0.55 | Minimum detector confidence. |
| `max_image_side` | 720 | Longest side used for detection. |
| `det_size` | 256 | Detection input resolution. |
| `cpu_threads` | 1 | OpenCV/BLAS worker count. |
| `save_every` | 50 | Checkpoint interval in images. |
| `progress_every` | 5 | Progress-update interval in images. |

Malformed configuration values are ignored and safe defaults are retained.

## Operational constraints

- The service has process-wide mutable state and is designed for one active operation per application process.
- There is no cancellation mechanism during a scan; closing the app ends the daemon worker without a final guaranteed checkpoint.
- Source paths are absolute. Moving a library drive makes records temporarily unavailable until it returns at the same path or is rescanned.
- The no-face cleanup feature only offers images that completed face detection with zero results. It moves only paths explicitly selected by the user, writes an audit record, and never deletes source files.
- Cleanup-suggestion analysis groups files by size before calculating SHA-256 exact-duplicate hashes, then caches verified hashes while the file size and modification time remain unchanged. It runs in a background UI worker but can take time for very large libraries or network drives. “Likely forward” and “screenshot” are explainable filename/folder signals, not content classification.
- JPEG/PNG/WebP/BMP/TIFF files are discovered; video, HEIC, RAW, and cloud-only placeholders are not currently supported.
- Full scans are destructive for the selected folder's database records. A manual merge or split also has no undo history.

## Extension points

The cleanest next boundaries are a repository layer for data files, a worker/job abstraction for long-running operations, and a clustering strategy interface. Keeping the present service API stable would let these be introduced without rewriting the UI.

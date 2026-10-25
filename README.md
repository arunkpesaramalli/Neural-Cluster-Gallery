# Custom-Album — AI Media

A desktop photo library application with AI-powered face recognition, person clustering, and timeline/album organisation.

## Architecture

```
Custom-Album/
├── py_app/
│   ├── main.py              # PySide6 (Qt6) desktop GUI — entry point
│   └── requirements.txt     # Python dependencies
├── resources/
│   ├── face_service.py      # Core AI backend: detection, clustering, persistence
│   └── config.json          # Optional tuning (thresholds, threads, etc.)
└── data/                    # Auto-created at runtime
    ├── faces.json           # Face records, source metadata and embeddings
    ├── people.json          # Person clusters + names
    ├── processed.json       # Incremental scan index
    ├── faiss.index          # FAISS similarity index
    ├── progress.json        # Live scan progress
    ├── errors.jsonl         # Per-image error log
    └── faces/               # Cropped face thumbnail images
```

## Requirements

- Python 3.9+
- See `py_app/requirements.txt`

```
PySide6>=6.5
numpy
opencv-python
insightface
onnxruntime      # or onnxruntime-gpu for CUDA
faiss-cpu        # or faiss-gpu
```

## Setup

```bash
cd py_app
pip install -r requirements.txt
```

## Running

```bash
cd py_app
python main.py
```

## Features

- **People tab** — Auto-clustered persons with face thumbnails; inline rename support
- **Timeline tab** — Every discovered photo grouped by YYYY-MM (EXIF date when available)
- **Albums tab** — Faces grouped by source folder
- **Incremental scanning** — Skips already-processed images (size + mtime signature)
- **Face gallery** — Navigate all face crops for a person or group; open original in OS
- **No-face cleanup** — Review and selectively move scanned zero-face photos; collision-safe moves are logged
- **Cleanup suggestions** — Review likely screenshots/social forwards and exact duplicate files before moving anything
- **Library report** — Export an Excel-friendly CSV inventory with timeline month, full path, scan status, faces, and people
- **Progress bar** — Live scan progress with file name, face count and error count
- **Thumbnail cache** — Persistent disk cache + LRU in-memory cache for fast browsing
- **CUDA auto-detect** — Uses GPU if available, falls back to CPU automatically
- **Crash-safe writes** — Atomic `.tmp` → rename for JSON and FAISS persistence

## Configuration (`resources/config.json`)

Create this file to override defaults without touching code:

```json
{
  "similarity_threshold": 0.45,
  "face_threshold": 0.55,
  "max_image_side": 720,
  "det_size": 256,
  "cpu_threads": 1,
  "save_every": 50,
  "progress_every": 5
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `similarity_threshold` | 0.45 | Cosine similarity cutoff for same-person match |
| `face_threshold` | 0.55 | Minimum InsightFace detection confidence |
| `max_image_side` | 720 | Downsample images larger than this before detection |
| `det_size` | 256 | InsightFace detection input resolution |
| `cpu_threads` | 1 | OpenCV/BLAS thread count |
| `save_every` | 50 | Save DB every N images |
| `progress_every` | 5 | Update progress file every N images |

## Documentation and checks

- [Architecture](architecture.md) describes components, data contracts, scan flow, and constraints.
- [Functionality guide](FUNCTIONALITY.md) describes every user-facing workflow and the feature roadmap.

Run the dependency-light regression suite from the project root:

```powershell
python -m unittest -v tests.test_face_service
```

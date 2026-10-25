# AI Media — Python GUI

This folder contains a minimal PySide6 GUI wrapper that imports the existing `face_service.py` (from `resources/`) and exposes folder selection, processing, and a simple view of `People`, `Timeline`, and `Albums`.

Quick start

1. Create a Python environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

2. Run the app:

```powershell
python main.py
```

Notes
- `face_service.py` depends on heavy native libraries (InsightFace, FAISS, OpenCV). Install platform-appropriate FAISS wheel if `pip install faiss-cpu` fails.
- The GUI polls `face_service.get_progress()` while processing; heavy compute still runs in Python and may utilize GPU if configured.

Thumbnail cache
- The GUI generates on-demand thumbnails and stores them under `py_app/.cache/thumbnails` to speed up subsequent loads.
- Thumbnails are generated using OpenCV when available; a Qt fallback is used otherwise.

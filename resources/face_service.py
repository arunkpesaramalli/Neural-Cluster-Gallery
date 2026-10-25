import contextlib
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

BASE = os.path.dirname(os.path.abspath(__file__))
# resources/ is now at the repository root; set ROOT to the project root (one level up)
ROOT = os.path.abspath(os.path.join(BASE, ".."))

DATA_DIR = os.path.join(ROOT, "data")
FACE_IMG_DIR = os.path.join(DATA_DIR, "faces")

FACES_DB = os.path.join(DATA_DIR, "faces.json")
PEOPLE_DB = os.path.join(DATA_DIR, "people.json")
PROCESSED_DB = os.path.join(DATA_DIR, "processed.json")
TIMELINE_DB = os.path.join(DATA_DIR, "timeline.json")
INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
PROGRESS_FILE = os.path.join(DATA_DIR, "progress.json")
ERROR_LOG = os.path.join(DATA_DIR, "errors.jsonl")
CLEANUP_LOG = os.path.join(DATA_DIR, "cleanup_moves.jsonl")
CONFIG_FILE = os.path.join(BASE, "config.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FACE_IMG_DIR, exist_ok=True)

DIM = 512
SIM_THRESHOLD = 0.45
FACE_SCORE_THRESHOLD = 0.55
GALLERY_LIMIT = 50
SAVE_EVERY = 50
PROGRESS_EVERY = 5
MAX_IMAGE_SIDE = 720
DET_SIZE = 256
CPU_THREADS = 1
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

cv2 = None
np = None
faiss = None
FaceAnalysis = None
onnxruntime = None
app = None

faces_db = []
people_db = []
processed_db = {}
timeline_db = {}
index = None
id_map = []
faces_by_id = {}
people_by_id = {}
next_face_id_value = 0
_next_person_id = 0             # OPT-2: O(1) person ID counter (replaces O(n) max scan)
_date_cache: dict = {}          # OPT-1: (path, mtime_int) -> "YYYY-MM"  avoids re-opening same JPEG per face
thumbnail_metadata_dirty = False


def normalize(path):
    return os.path.abspath(path).replace("\\", "/")


def read_json(path, default):
    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def atomic_write_json(path, payload):
    """Write JSON without exposing a partially-written destination file.

    A fixed ``.tmp`` name races when two backend requests save at once.  On
    Windows, a virus scanner or a just-closed reader can also hold the target
    briefly without delete sharing, so retry its atomic replacement.
    """
    # OPT-8: json.dumps() + one write() is faster than json.dump() which makes many small writes.
    tmp = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8", buffering=1 << 16) as f:
            f.write(json.dumps(payload, separators=(",", ":")))
            # A successful rename alone does not guarantee that buffered data
            # reached disk.  Flush the file before replacing the previous
            # version so an interrupted save cannot leave a zero-byte DB.
            f.flush()
            os.fsync(f.fileno())

        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        # ``os.replace`` removes tmp on success; this only cleans up failed
        # attempts without touching another writer's temporary file.
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)


def load_config():
    global SIM_THRESHOLD, FACE_SCORE_THRESHOLD, MAX_IMAGE_SIDE, DET_SIZE
    global CPU_THREADS, SAVE_EVERY, PROGRESS_EVERY

    config = read_json(CONFIG_FILE, {})
    if not isinstance(config, dict):
        return

    # A hand-edited config file must not prevent the library from opening.
    # Keep the previous/default value for an invalid setting.
    def configured(key, converter, current, minimum=None, maximum=None):
        try:
            value = converter(config.get(key, current))
            if isinstance(value, float) and not math.isfinite(value):
                return current
            if minimum is not None:
                value = max(minimum, value)
            if maximum is not None:
                value = min(maximum, value)
            return value
        except (TypeError, ValueError):
            return current

    SIM_THRESHOLD = configured("similarity_threshold", float, SIM_THRESHOLD, -1.0, 1.0)
    FACE_SCORE_THRESHOLD = configured("face_threshold", float, FACE_SCORE_THRESHOLD, 0.0, 1.0)
    MAX_IMAGE_SIDE = configured("max_image_side", int, MAX_IMAGE_SIDE, 320)
    DET_SIZE = configured("det_size", int, DET_SIZE, 160)
    CPU_THREADS = configured("cpu_threads", int, CPU_THREADS, 1, os.cpu_count() or 1)
    SAVE_EVERY = configured("save_every", int, SAVE_EVERY, 1)
    PROGRESS_EVERY = configured("progress_every", int, PROGRESS_EVERY, 1)


def load_db():
    global faces_db, people_db, processed_db, timeline_db, faces_by_id, people_by_id
    global next_face_id_value, _next_person_id, thumbnail_metadata_dirty

    load_config()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FACE_IMG_DIR, exist_ok=True)
    thumbnail_metadata_dirty = False

    faces_db = read_json(FACES_DB, [])
    people_db = read_json(PEOPLE_DB, [])
    processed_db = read_json(PROCESSED_DB, {})
    timeline_db = read_json(TIMELINE_DB, {})

    if not isinstance(faces_db, list):
        faces_db = []
    if not isinstance(people_db, list):
        people_db = []
    if not isinstance(processed_db, dict):
        processed_db = {}
    if not isinstance(timeline_db, dict):
        timeline_db = {}

    # OPT-6: only call os.path.abspath when a path still contains backslashes
    # (already-normalised paths — the common case — skip the syscall entirely)
    for record in faces_db:
        if not isinstance(record, dict):
            continue
        if isinstance(record.get("image"), str) and "\\" in record["image"]:
            record["image"] = normalize(record["image"])
        if isinstance(record.get("face_path"), str) and "\\" in record["face_path"]:
            record["face_path"] = normalize(record["face_path"])
        if isinstance(record.get("album"), str) and "\\" in record["album"]:
            record["album"] = normalize(record["album"])

    valid_faces = []
    faces_by_id = {}
    for face in faces_db:
        if not isinstance(face, dict):
            continue
        try:
            face_id = int(face["face_id"])
        except (KeyError, TypeError, ValueError):
            continue
        face["face_id"] = face_id
        valid_faces.append(face)
        faces_by_id[face_id] = face
    faces_db = valid_faces

    # Face records are the source of truth for membership.  Older versions
    # appended each newly detected face twice, so rebuild memberships on load
    # while retaining any saved person names.
    names_by_id = {}
    for person in people_db:
        if not isinstance(person, dict):
            continue
        try:
            person_id = int(person["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if person.get("name"):
            names_by_id[person_id] = person["name"]

    rebuilt_people = {}
    for face in faces_db:
        try:
            person_id = int(face["person_id"])
        except (KeyError, TypeError, ValueError):
            face["person_id"] = None
            continue
        face["person_id"] = person_id
        person = rebuilt_people.setdefault(person_id, {"id": person_id, "faces": []})
        person["faces"].append(face["face_id"])
    for person_id, name in names_by_id.items():
        if person_id in rebuilt_people:
            rebuilt_people[person_id]["name"] = name
    repaired_people = list(rebuilt_people.values())
    if repaired_people != people_db:
        # Persist the repair on the next normal library read/save. This fixes
        # duplicate or stale membership lists left by prior releases.
        thumbnail_metadata_dirty = True
    people_db = repaired_people
    people_by_id = rebuilt_people

    numeric_face_ids = list(faces_by_id)
    next_face_id_value = max(numeric_face_ids, default=-1) + 1

    # OPT-2: initialise O(1) person ID counter
    _next_person_id = (
        max(people_by_id, default=-1) + 1
    )


# OPT-4: separate save helpers so process_folder can choose per-file frequency
def save_faces():
    atomic_write_json(FACES_DB, faces_db)


def save_people():
    atomic_write_json(PEOPLE_DB, people_db)


def save_processed():
    atomic_write_json(PROCESSED_DB, processed_db)


def save_timeline():
    atomic_write_json(TIMELINE_DB, timeline_db)


def save_db():
    """Save all three databases. Prefer targeted helpers inside hot loops."""
    save_faces()
    save_people()
    save_processed()


def log_error(path, message, trace=None):
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "path": normalize(path) if path else None,
        "error": str(message),
    }
    if trace:
        payload["trace"] = trace

    # Logging must never turn a recoverable image/progress failure into a
    # failed scan (for example when antivirus temporarily locks this file).
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except OSError:
        pass


def update_progress(done, total, status="processing", current=None, added_faces=None, errors=None):
    # A zero total means the folder is still being enumerated, not that the
    # operation is complete. The UI presents this state as an indeterminate
    # progress bar.
    percent = 100 if total == 0 and status == "complete" else (0 if total == 0 else int((done / total) * 100))
    payload = {
        "done": done,
        "total": total,
        "percent": percent,
        "status": status,
        "current": current,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if added_faces is not None:
        payload["added_faces"] = added_faces
    if errors is not None:
        payload["errors"] = errors

    try:
        atomic_write_json(PROGRESS_FILE, payload)
    except OSError as exc:
        # Progress is informational. A file temporarily locked by Windows
        # Explorer, antivirus, or another reader must not cancel a scan.
        log_error(PROGRESS_FILE, f"Could not update scan progress: {exc}")


def get_progress():
    return read_json(PROGRESS_FILE, {"done": 0, "total": 0, "percent": 0, "status": "idle"})


def image_signature(path):
    try:
        stat = os.stat(path)
        # Second-level timestamps can miss quick edits.  Keep the legacy
        # value too, so existing scan indexes remain incremental on upgrade.
        return {
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
            "mtime_ns": stat.st_mtime_ns,
        }
    except OSError:
        return {"size": 0, "mtime": 0, "mtime_ns": 0}


def same_signature(existing, signature):
    if not isinstance(existing, dict):
        return False
    if existing.get("size") != signature["size"]:
        return False
    # Prefer nanosecond precision for new records, while accepting legacy
    # seconds-only entries produced by older versions.
    if "mtime_ns" in existing:
        return existing.get("mtime_ns") == signature["mtime_ns"]
    return existing.get("mtime") == signature["mtime"]


def get_image_date(path):
    """Return YYYY-MM date string. OPT-1: result cached by (path, mtime) to avoid
    re-opening the same JPEG for every face crop found in it."""
    try:
        mtime_key = os.stat(path).st_mtime_ns
    except OSError:
        mtime_key = 0
    cache_key = (path, mtime_key)
    cached = _date_cache.get(cache_key)
    if cached is not None:
        return cached
    result = _get_image_date_uncached(path)
    _date_cache[cache_key] = result
    return result


def _get_image_date_uncached(path):
    """Read date from EXIF / ctime / mtime — no caching."""
    # Priority (most accurate → least accurate):
    #   1. EXIF DateTimeOriginal (tag 36867) — exact shutter-press timestamp
    #   2. EXIF DateTime         (tag 306)   — camera write time
    #   3. File creation time   (ctime)      — correct for transfers on Windows
    #   4. File modification time (mtime)    — last resort

    # 1 & 2 — EXIF (most accurate: when the photo was actually taken)
    try:
        from PIL import Image
        with Image.open(path) as img:
            exif = img.getexif()        # public Pillow API, returns {} on non-EXIF files
            for tag_id in (36867, 306): # DateTimeOriginal, then DateTime
                raw = exif.get(tag_id)
                if raw:
                    # EXIF format: "YYYY:MM:DD HH:MM:SS"
                    dt = datetime.strptime(str(raw)[:19], "%Y:%m:%d %H:%M:%S")
                    return dt.strftime("%Y-%m")
    except Exception:
        pass

    # 3 — File creation time (on Windows this is the real creation timestamp)
    try:
        ts = os.path.getctime(path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m")
    except Exception:
        pass

    # 4 — Last resort: modification time
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m")
    except Exception:
        return "unknown"


def require_cv():
    global cv2, np

    if cv2 is not None and np is not None:
        return cv2, np

    try:
        import cv2 as cv_module
        import numpy as np_module

        cv_module.setNumThreads(CPU_THREADS)
    except Exception as exc:
        raise RuntimeError(
            "OpenCV/Numpy are not available. Install them in the Python used by the app."
        ) from exc

    cv2 = cv_module
    np = np_module
    return cv2, np


def require_faiss():
    global faiss

    if faiss is not None:
        return faiss

    try:
        import faiss as faiss_module
    except Exception as exc:
        raise RuntimeError("FAISS is not available in the Python used by the app.") from exc

    faiss = faiss_module
    return faiss


def require_model():
    global FaceAnalysis, onnxruntime

    if FaceAnalysis is not None:
        return FaceAnalysis, onnxruntime

    try:
        from insightface.app import FaceAnalysis as face_analysis
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError(
            "InsightFace/ONNXRuntime are not available. Run: python -m pip install insightface onnxruntime"
        ) from exc

    FaceAnalysis = face_analysis
    onnxruntime = ort
    return FaceAnalysis, onnxruntime


def get_model():
    global app

    if app is not None:
        return app

    face_analysis, ort = require_model()
    available = set(ort.get_available_providers())
    providers = ["CPUExecutionProvider"]
    ctx_id = -1

    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        model = face_analysis(name="buffalo_s", providers=providers)
        model.prepare(ctx_id=ctx_id, det_size=(DET_SIZE, DET_SIZE))

    app = model
    return app


def init_index():
    global index, id_map

    faiss_module = require_faiss()
    id_map = list(faces_by_id)

    # When every record stores its embedding, rebuilding is both reliable and
    # avoids relying on a separate index file whose row-to-face mapping cannot
    # otherwise be verified after a partial write or manual recovery.
    embeddings = [face.get("embedding") for face in faces_db]
    if faces_db and all(embedding is not None for embedding in embeddings):
        _, np_module = require_cv()
        try:
            vectors = np_module.asarray(embeddings, dtype="float32")
            if vectors.ndim != 2 or vectors.shape[1] != DIM:
                raise ValueError(f"expected embeddings with dimension {DIM}")
            faiss_module.normalize_L2(vectors)
            index = faiss_module.IndexFlatIP(DIM)
            index.add(vectors)
            return
        except Exception as exc:
            log_error(INDEX_FILE, f"Could not rebuild FAISS index from stored embeddings: {exc}")

    if os.path.exists(INDEX_FILE):
        try:
            index = faiss_module.read_index(INDEX_FILE)
        except Exception as exc:
            log_error(INDEX_FILE, f"Could not read FAISS index, starting a new one: {exc}")
            index = faiss_module.IndexFlatIP(DIM)
    else:
        index = faiss_module.IndexFlatIP(DIM)

    if index.ntotal != len(id_map):
        log_error(INDEX_FILE, "FAISS index and faces.json have different entry counts; starting a new index.")
        index = faiss_module.IndexFlatIP(DIM)
        id_map = []


def save_index():
    """Persist the FAISS index without exposing a half-written index file."""
    if index is None:
        return

    # ``faiss.write_index`` writes directly to its destination.  Write to a
    # unique sibling first so an interrupted checkpoint leaves the previous
    # index usable, just like the JSON stores.
    tmp = f"{INDEX_FILE}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        require_faiss().write_index(index, tmp)
        for attempt in range(10):
            try:
                os.replace(tmp, INDEX_FILE)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)


def add_embedding(embedding, face_id):
    faiss_module = require_faiss()
    vector = np.array([embedding], dtype="float32")
    faiss_module.normalize_L2(vector)
    index.add(vector)
    id_map.append(int(face_id))


def search_embedding(embedding):
    if index is None or index.ntotal == 0:
        return None

    faiss_module = require_faiss()
    vector = np.array([embedding], dtype="float32")
    faiss_module.normalize_L2(vector)

    distances, positions = index.search(vector, 1)
    if distances[0][0] >= SIM_THRESHOLD:
        pos = int(positions[0][0])
        if 0 <= pos < len(id_map):
            return id_map[pos]

    return None


def next_face_id():
    global next_face_id_value

    face_id = next_face_id_value
    next_face_id_value += 1
    return face_id


def face_path_for_id(face_id):
    return normalize(os.path.join(FACE_IMG_DIR, f"{int(face_id)}.jpg"))


def create_person(face_id):
    # OPT-2: O(1) ID via counter instead of O(n) max scan over people_db
    global _next_person_id
    pid = _next_person_id
    _next_person_id += 1

    person = {"id": pid, "faces": [int(face_id)]}
    people_db.append(person)
    people_by_id[pid] = person
    return pid


def attach_face_to_person(person_id, face_id):
    if person_id is None:
        return create_person(face_id)

    person = people_by_id.get(person_id)
    if not person:
        person = {"id": person_id, "faces": []}
        people_db.append(person)
        people_by_id[person_id] = person

    face_id = int(face_id)
    if face_id not in person["faces"]:
        person["faces"].append(face_id)
    return person_id


def person_for_match(match_face_id, new_face_id):
    matched = faces_by_id.get(int(match_face_id))
    if not matched:
        return create_person(new_face_id)

    person_id = matched.get("person_id")
    if person_id is None:
        person_id = create_person(match_face_id)
        matched["person_id"] = person_id

    return attach_face_to_person(person_id, new_face_id)


def resize_for_detection(img):
    height, width = img.shape[:2]
    longest = max(height, width)
    if longest <= MAX_IMAGE_SIDE:
        return img, 1.0

    scale = MAX_IMAGE_SIDE / float(longest)
    resized = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def crop_face(img, bbox):
    height, width = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None, [x1, y1, x2, y2]

    return img[y1:y2, x1:x2], [x1, y1, x2, y2]


def save_face_thumbnail(face_id, crop):
    cv_module, _ = require_cv()
    # OPT-7: FACE_IMG_DIR is created at import and in load_db — no need to check every face
    face_path = face_path_for_id(face_id)
    if not cv_module.imwrite(face_path, crop, [cv_module.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Could not write thumbnail: {face_path}")
    return face_path


def ensure_face_thumbnail(face):
    global thumbnail_metadata_dirty

    face_path = face.get("face_path")
    if face_path:
        normalized_face_path = normalize(face_path)
        if normalized_face_path != face_path:
            face["face_path"] = normalized_face_path
            thumbnail_metadata_dirty = True
        face_path = normalized_face_path
        if os.path.exists(face_path):
            return face_path

    image_path = face.get("image")
    # Newer records retain original-image coordinates.  Older records only
    # have detector-space coordinates, so preserve the old resize fallback.
    original_bbox = face.get("bbox_original")
    bbox = face.get("bbox")
    face_id = face.get("face_id")
    if image_path is None or (original_bbox is None and bbox is None) or face_id is None:
        return face_path

    # A disconnected external drive or moved photo collection is normal for a
    # library.  Do not ask OpenCV to decode a nonexistent source for every
    # face while populating the UI: it is slow and emits one warning per face.
    if not isinstance(image_path, str) or not os.path.exists(image_path):
        return face_path

    try:
        cv_module, _ = require_cv()
        img = cv_module.imread(normalize(image_path), cv_module.IMREAD_COLOR)
        if img is None:
            return face_path

        if original_bbox is not None:
            crop, _ = crop_face(img, original_bbox)
        else:
            model_img, _ = resize_for_detection(img)
            crop, _ = crop_face(model_img, bbox)
        if crop is None:
            return face_path

        regenerated_path = save_face_thumbnail(face_id, crop)
        if face.get("face_path") != regenerated_path:
            face["face_path"] = regenerated_path
            thumbnail_metadata_dirty = True
        return regenerated_path
    except Exception as exc:
        log_error(image_path, f"Could not repair thumbnail for face {face_id}: {exc}")
        return face_path


def thumbnail_for_face(face):
    face_path = ensure_face_thumbnail(face)
    if face_path and os.path.exists(face_path):
        return face_path
    return None


def process_image(path):
    try:
        cv_module, _ = require_cv()
        path = normalize(path)
        img = cv_module.imread(path, cv_module.IMREAD_COLOR)
        if img is None:
            return {"status": "invalid_image", "faces": 0}

        model_img, _ = resize_for_detection(img)
        original_height, original_width = img.shape[:2]
        model_height, model_width = model_img.shape[:2]
        # The detector operates on a downscaled image.  Keep an original-space
        # box too: it produces sharper regenerated thumbnails and lets the UI
        # draw an exact overlay even when MAX_IMAGE_SIDE changes later.
        scale_x = original_width / float(model_width)
        scale_y = original_height / float(model_height)
        detected_faces = get_model().get(model_img)
        added = 0

        for detected in detected_faces:
            score = float(detected.det_score)
            if score < FACE_SCORE_THRESHOLD:
                continue

            detection_bbox = [float(v) for v in detected.bbox]
            original_bbox = [
                detection_bbox[0] * scale_x,
                detection_bbox[1] * scale_y,
                detection_bbox[2] * scale_x,
                detection_bbox[3] * scale_y,
            ]
            crop, original_bbox = crop_face(img, original_bbox)
            if crop is None:
                continue

            face_id = next_face_id()
            try:
                face_path = save_face_thumbnail(face_id, crop)
            except Exception as exc:
                log_error(path, exc)
                continue

            embedding = detected.embedding
            match = search_embedding(embedding)
            person_id = person_for_match(match, face_id) if match is not None else create_person(face_id)

            record = {
                "face_id": face_id,
                "person_id": person_id,
                "image": path,
                # ``bbox`` remains detector-space for backwards compatibility.
                # New records additionally carry exact original-image bounds.
                "bbox": detection_bbox,
                "bbox_original": original_bbox,
                "score": score,
                "landmarks": detected.kps.tolist(),
                "face_path": face_path,
                "date": get_image_date(path),
                "album": normalize(os.path.dirname(path)),
                "embedding": embedding.tolist(),  # stored for re-clustering
            }

            faces_db.append(record)
            faces_by_id[face_id] = record
            add_embedding(embedding, face_id)
            added += 1

        return {"status": "ok", "faces": added}
    except Exception as exc:
        trace = traceback.format_exc()
        log_error(path, exc, trace)
        return {"status": "error", "faces": 0, "error": str(exc)}


def discover_images(folder):
    for root, _, files in os.walk(folder, followlinks=False):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                yield os.path.join(root, filename)


def index_timeline_folder(folder):
    """Record every image in *folder* for the Timeline without face analysis."""
    if not folder or not os.path.isdir(folder):
        return {"error": "folder_not_found", "path": folder}

    files = []
    update_progress(0, 0, status="discovering photos", current=folder)
    for path in discover_images(folder):
        files.append(normalize(path))
        if len(files) % 250 == 0:
            update_progress(
                len(files), 0, status="discovering photos", current=path
            )
    total = len(files)
    indexed = 0
    known = 0
    update_progress(0, total, status="indexing timeline")

    for done, path in enumerate(files, start=1):
        signature = image_signature(path)
        entry = timeline_db.get(path)
        if (
            isinstance(entry, dict)
            and same_signature(entry, signature)
            and isinstance(entry.get("date"), str)
        ):
            known += 1
        else:
            timeline_db[path] = {**signature, "date": get_image_date(path)}
            indexed += 1
        if done % PROGRESS_EVERY == 0 or done == total:
            update_progress(done, total, status="indexing timeline", current=path)

    save_timeline()
    # Returning the discovered paths allows process_folder to reuse this walk
    # instead of recursively enumerating large photo libraries twice.
    return {
        "photos_found": total,
        "indexed": indexed,
        "unchanged": known,
        "paths": files,
    }


def already_processed(path, signature, legacy_processed):
    normalized = normalize(path)
    existing = processed_db.get(normalized)
    # Failed reads/model errors are often transient (a copied file may still
    # be changing, or a removable drive may reconnect). Retry those files on
    # the next incremental scan, but avoid repeatedly decoding a file already
    # determined to be an invalid image.
    if same_signature(existing, signature) and existing.get("status", "ok") in ("ok", "invalid_image"):
        return True
    return existing is None and normalized in legacy_processed


def path_is_in_folder(path, folder):
    """Return whether path is inside folder, including the folder itself."""
    try:
        path = os.path.normcase(os.path.abspath(path))
        folder = os.path.normcase(os.path.abspath(folder))
        return os.path.commonpath([path, folder]) == folder
    except (TypeError, ValueError):
        return False


def full_scan_folder(folder):
    """Rebuild face and timeline records for one selected folder.

    Existing records from that folder are removed first, so a forced scan does
    not create duplicate faces. Records for all other folders are preserved.
    """
    global faces_db, processed_db, timeline_db

    if not folder or not os.path.isdir(folder):
        return {"error": "folder_not_found", "path": folder}

    try:
        load_db()
        update_progress(0, 0, status="preparing full scan")

        faces_db = [
            face for face in faces_db
            if not path_is_in_folder(face.get("image"), folder)
        ]
        processed_db = {
            path: record for path, record in processed_db.items()
            if not path_is_in_folder(path, folder)
        }
        timeline_db = {
            path: record for path, record in timeline_db.items()
            if not path_is_in_folder(path, folder)
        }

        # Persist the scoped cleanup before processing. Orphaned thumbnail
        # files are harmless and retained so no unrelated assets are deleted.
        save_faces()
        save_processed()
        save_timeline()
        load_db()  # rebuild people membership and ID maps from retained faces
        save_people()

        result = process_folder(folder)
        result["full_scan"] = True
        return result
    except Exception as exc:
        trace = traceback.format_exc()
        log_error(folder, f"Full scan failed: {exc}", trace)
        update_progress(0, 0, status="full scan failed", current=folder)
        return {"error": f"Full scan failed: {exc}", "trace": trace}


def process_folder(folder):
    if not folder or not os.path.isdir(folder):
        return {"error": "folder_not_found", "path": folder}

    started = time.time()
    load_db()
    timeline_result = index_timeline_folder(folder)
    if timeline_result.get("error"):
        return timeline_result
    init_index()
    update_progress(0, 0, status="scanning")

    legacy_processed = {normalize(f.get("image")) for f in faces_db if f.get("image")}
    files = []

    for file_path in timeline_result.get("paths", ()):
        signature = image_signature(file_path)
        if already_processed(file_path, signature, legacy_processed):
            continue
        files.append((normalize(file_path), signature))

    total = len(files)
    done = 0
    added_faces = 0
    errors = 0
    update_progress(done, total, status="processing", added_faces=added_faces, errors=errors)

    for file_path, signature in files:
        result = process_image(file_path)
        done += 1
        added_faces += int(result.get("faces", 0))
        if result.get("status") == "error":
            errors += 1

        processed_db[file_path] = {
            **signature,
            "status": result.get("status", "ok"),
            "faces": int(result.get("faces", 0)),
            # Store this for the Timeline so photos with no detected face are
            # still grouped without reopening their EXIF data on every launch.
            "date": get_image_date(file_path),
        }

        if done % PROGRESS_EVERY == 0 or done == total:
            update_progress(
                done,
                total,
                status="processing",
                current=file_path,
                added_faces=added_faces,
                errors=errors,
            )

        if done % SAVE_EVERY == 0:
            # OPT-4: only write the small, fast files on every checkpoint;
            # save the large faces/people DBs 5× less often to reduce I/O stalls.
            save_processed()
            save_index()

        if done % (SAVE_EVERY * 5) == 0:
            save_faces()
            save_people()

    save_db()
    save_index()
    update_progress(done, total, status="complete", added_faces=added_faces, errors=errors)

    return {
        "processed_images": total,
        "added_faces": added_faces,
        "faces": len(faces_db),
        "people": len(people_db),
        "errors": errors,
        "timeline_photos": timeline_result.get("photos_found", 0),
        "seconds": round(time.time() - started, 2),
    }


def compact_face(face):
    return {
        "face_id": face.get("face_id"),
        "person_id": face.get("person_id"),
        "image": face.get("image"),
        "bbox": face.get("bbox"),
        "bbox_original": face.get("bbox_original"),
        "score": face.get("score"),
        "face_path": thumbnail_for_face(face),
        "date": face.get("date", "unknown"),
        "album": face.get("album", "unknown"),
    }


def list_people(should_load=True):
    if should_load:
        load_db()

    grouped = {}
    for face in faces_db:
        pid = face.get("person_id")
        if pid is None:
            pid = "unassigned"
        grouped.setdefault(pid, []).append(face)

    people = []
    for pid, faces in grouped.items():
        faces = sorted(faces, key=lambda item: item.get("face_id", 0))
        thumbnail = None
        face_records = []  # rich records: {face_id, face_path, score, image}
        available_count = 0
        for face in faces:
            face_path = thumbnail_for_face(face)
            image_path = face.get("image")
            source_available = isinstance(image_path, str) and os.path.exists(image_path)
            if not face_path and not source_available:
                # Do not show a person that cannot display either a face crop
                # or an original photo. Keep the record on disk: it will be
                # shown automatically if a disconnected photo drive returns.
                continue

            available_count += 1
            if face_path:
                if thumbnail is None:
                    thumbnail = face_path
            if len(face_records) < GALLERY_LIMIT:
                face_records.append({
                    "face_id": face.get("face_id"),
                    "face_path": face_path,
                    "score": face.get("score"),
                    "image": face.get("image"),
                    "bbox": face.get("bbox"),   # for full-photo bounding-box viewer
                    "bbox_original": face.get("bbox_original"),
                })

        if available_count == 0:
            continue

        # Merge saved name from people_db
        person_record = people_by_id.get(pid, {})
        saved_name = person_record.get("name") if isinstance(person_record, dict) else None

        people.append(
            {
                "id": pid,
                "name": saved_name,
                "count": available_count,
                "thumbnail": thumbnail,
                "faces": face_records,
            }
        )

    people.sort(key=lambda p: p["count"], reverse=True)
    if thumbnail_metadata_dirty:
        save_db()

    return {"people": people, "total_faces": len(faces_db)}


def timeline(should_load=True):
    """Group every successfully scanned source photo by its capture month.

    The old timeline was built from face records, so photos containing no
    detected faces disappeared and a single group could show the same photo
    multiple times.  ``processed_db`` records every scan, including zero-face
    photos; face data is used only to provide an optional representative crop.
    """
    if should_load:
        load_db()

    photos = {}
    processed_dirty = False

    # Timeline indexing is intentionally separate from face processing: it
    # contains every photo discovered in the selected folder, including photos
    # that have never had a face detection run.
    for path, metadata in timeline_db.items():
        if not isinstance(path, str) or not isinstance(metadata, dict):
            continue
        date = metadata.get("date") if isinstance(metadata.get("date"), str) else "unknown"
        photos[normalize(path)] = {
            "image": normalize(path),
            "date": date or "unknown",
            "display_original": True,
        }

    for path, metadata in processed_db.items():
        if not isinstance(path, str) or not isinstance(metadata, dict):
            continue
        if metadata.get("status") not in (None, "ok"):
            continue
        normalized_path = normalize(path)
        date = metadata.get("date")
        if not isinstance(date, str) or not date:
            date = get_image_date(normalized_path) if os.path.exists(normalized_path) else "unknown"
            metadata["date"] = date
            processed_dirty = True
        photos[normalized_path] = {
            "image": normalized_path,
            "date": date,
            "display_original": True,
        }

    # Support libraries created before processed.json existed. A source photo
    # can have many face records, but must appear only once in the Timeline.
    for face in faces_db:
        image_path = face.get("image")
        if not isinstance(image_path, str) or not image_path:
            continue
        normalized_path = normalize(image_path)
        photo = photos.setdefault(
            normalized_path,
            {
                "image": normalized_path,
                "date": face.get("date") or (
                    get_image_date(normalized_path) if os.path.exists(normalized_path) else "unknown"
                ),
                "display_original": True,
            },
        )
        if photo.get("date") in (None, "", "unknown") and face.get("date"):
            photo["date"] = face["date"]
        # A crop is only a preview for the month list. Timeline galleries
        # deliberately open the original source image, never this crop.
        if not photo.get("preview_path"):
            face_path = face.get("face_path")
            if isinstance(face_path, str) and os.path.exists(face_path):
                photo.update({
                    "preview_path": face_path,
                    "face_id": face.get("face_id"),
                    "bbox": face.get("bbox"),
                    "bbox_original": face.get("bbox_original"),
                    "score": face.get("score"),
                })

    groups = {}
    for photo in photos.values():
        groups.setdefault(photo.get("date") or "unknown", []).append(photo)

    for photo_list in groups.values():
        photo_list.sort(key=lambda photo: photo.get("image", "").lower())

    if processed_dirty:
        save_processed()

    return {
        "timeline": {
            date: {"count": len(photo_list), "items": photo_list, "kind": "photos"}
            for date, photo_list in groups.items()
        },
        "total_photos": len(photos),
    }


def albums(should_load=True):
    if should_load:
        load_db()

    groups = {}
    for face in faces_db:
        groups.setdefault(face.get("album", "unknown"), []).append(face)

    return {
        "albums": {
            album: {"count": len(faces), "items": [compact_face(f) for f in faces[:GALLERY_LIMIT]]}
            for album, faces in groups.items()
        },
        "total_faces": len(faces_db),
    }


def no_face_photos(should_load=True):
    """Return accessible photos that completed scanning with zero faces.

    This deliberately excludes unscanned, failed, and previously-cleaned
    photos. A cleanup recommendation must be based on an actual completed
    detection result, rather than assuming a Timeline item has no face.
    """
    if should_load:
        load_db()

    paths_with_faces = {
        normalize(face["image"])
        for face in faces_db
        if isinstance(face, dict) and isinstance(face.get("image"), str)
    }
    candidates = []
    unavailable = 0

    for path, metadata in processed_db.items():
        if not isinstance(path, str) or not isinstance(metadata, dict):
            continue
        if metadata.get("status") != "ok" or metadata.get("cleanup_status") == "moved_no_face":
            continue
        try:
            face_count = int(metadata.get("faces"))
        except (TypeError, ValueError):
            continue
        source = normalize(path)
        if face_count != 0 or source in paths_with_faces:
            continue
        if not os.path.isfile(source):
            unavailable += 1
            continue
        try:
            size = os.path.getsize(source)
        except OSError:
            unavailable += 1
            continue
        candidates.append({
            "image": source,
            "date": metadata.get("date") or "unknown",
            "folder": normalize(os.path.dirname(source)),
            "size": size,
            "extension": os.path.splitext(source)[1].lower(),
        })

    candidates.sort(key=lambda item: (item["date"], item["image"].lower()), reverse=True)
    return {"items": candidates, "count": len(candidates), "unavailable": unavailable}


def _unique_cleanup_destination(folder, filename):
    """Return an unused path in *folder* without overwriting an existing file."""
    stem, extension = os.path.splitext(filename)
    candidate = os.path.join(folder, filename)
    number = 2
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{stem} ({number}){extension}")
        number += 1
    return candidate


def _log_cleanup_move(source, destination, action="move_no_face_photo"):
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "source": source,
        "destination": destination,
    }
    try:
        with open(CLEANUP_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except OSError as exc:
        # The moved file and database update are more important than optional
        # audit history; retain this diagnostic without treating it as a move
        # failure.
        log_error(CLEANUP_LOG, f"Could not record cleanup move: {exc}")


def move_no_face_photos(paths, destination):
    """Move reviewed zero-face photos to an existing folder safely.

    The caller must supply explicit paths selected from :func:`no_face_photos`.
    Existing destination files are never replaced; a numbered filename is used
    on collision. The processed/timeline records follow each moved file and an
    audit entry is appended for future undo support.
    """
    if not isinstance(paths, (list, tuple, set)):
        return {"error": "invalid_paths", "moved": [], "errors": []}
    if not isinstance(destination, str) or not os.path.isdir(destination):
        return {"error": "destination_not_found", "moved": [], "errors": []}

    load_db()
    destination = normalize(destination)
    available = {item["image"] for item in no_face_photos(should_load=False)["items"]}
    # Older libraries may have persisted Windows backslash keys. Keep the raw
    # key so a reviewed move updates that record instead of leaving a stale
    # duplicate behind.
    processed_keys = {
        normalize(path): path for path in processed_db if isinstance(path, str)
    }
    timeline_keys = {
        normalize(path): path for path in timeline_db if isinstance(path, str)
    }
    moved = []
    errors = []
    seen = set()

    for raw_path in paths:
        if not isinstance(raw_path, str):
            errors.append("invalid photo path")
            continue
        source = normalize(raw_path)
        if source in seen:
            continue
        seen.add(source)
        if source not in available:
            errors.append(f"not an eligible no-face photo: {source}")
            continue
        if os.path.normcase(os.path.dirname(source)) == os.path.normcase(destination):
            errors.append(f"destination is already the source folder: {source}")
            continue

        target = _unique_cleanup_destination(destination, os.path.basename(source))
        try:
            shutil.move(source, target)
        except OSError as exc:
            errors.append(f"could not move {source}: {exc}")
            continue

        target = normalize(target)
        old_metadata = processed_db.pop(processed_keys.get(source, source), {})
        new_metadata = dict(old_metadata) if isinstance(old_metadata, dict) else {}
        new_metadata.update(image_signature(target))
        new_metadata["cleanup_status"] = "moved_no_face"
        new_metadata["cleanup_moved_at"] = datetime.now().isoformat(timespec="seconds")
        processed_db[target] = new_metadata

        old_timeline = timeline_db.pop(timeline_keys.get(source, source), {})
        new_timeline = dict(old_timeline) if isinstance(old_timeline, dict) else {}
        new_timeline.update(image_signature(target))
        new_timeline.setdefault("date", new_metadata.get("date") or get_image_date(target))
        timeline_db[target] = new_timeline
        _log_cleanup_move(source, target)
        moved.append({"source": source, "destination": target})

    if moved:
        save_processed()
        save_timeline()
    return {"moved": moved, "errors": errors}


def _file_sha256(path):
    """Return a content hash without loading a whole image into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cleanup_suggestions(should_load=True):
    """Find reviewable screenshots, likely forwards, and exact duplicates.

    Screenshot/forward suggestions are explainable filename or folder signals,
    not claims about image contents. Duplicate suggestions use SHA-256 and are
    exact byte-for-byte matches. This function does not change any files.
    """
    if should_load:
        load_db()

    records = []
    unavailable = 0
    for path, metadata in processed_db.items():
        if not isinstance(path, str) or not isinstance(metadata, dict):
            continue
        if metadata.get("status") != "ok" or metadata.get("cleanup_status") == "moved_no_face":
            continue
        source = normalize(path)
        if not os.path.isfile(source):
            unavailable += 1
            continue
        try:
            size = os.path.getsize(source)
        except OSError:
            unavailable += 1
            continue
        signature = image_signature(source)
        records.append({
            "image": source,
            "date": metadata.get("date") or "unknown",
            "folder": normalize(os.path.dirname(source)),
            "size": size,
            "mtime_ns": signature.get("mtime_ns", 0),
            "metadata": metadata,
        })

    suggestions = {}

    def add_suggestion(record, category, reason):
        item = suggestions.setdefault(record["image"], {
            **record,
            "categories": [],
            "reasons": [],
        })
        if category not in item["categories"]:
            item["categories"].append(category)
        if reason not in item["reasons"]:
            item["reasons"].append(reason)

    screenshot_pattern = re.compile(r"(?:screenshot|screen[ _-]?shot|screen[ _-]?capture|screencap)", re.I)
    social_pattern = re.compile(
        r"(?:whatsapp|telegram|sharechat|messenger|forwarded|^img[-_].*[-_]wa\d*)",
        re.I,
    )
    for record in records:
        filename = os.path.basename(record["image"])
        folder = os.path.basename(record["folder"])
        if screenshot_pattern.search(filename) or screenshot_pattern.search(folder):
            add_suggestion(record, "screenshot", "Filename or folder indicates a screenshot")
        if social_pattern.search(filename):
            add_suggestion(record, "likely_forward", "Filename resembles a social-media export or forward")

    # Exact duplicates must have exactly the same byte size. Grouping before
    # hashing avoids opening the overwhelmingly common unique-size files.
    size_groups = {}
    for record in records:
        size_groups.setdefault(record["size"], []).append(record)

    hash_groups = {}
    hash_errors = 0
    photos_hashed = 0
    hash_cache_dirty = False
    for size_group in size_groups.values():
        if len(size_group) < 2:
            continue
        for record in size_group:
            metadata = record["metadata"]
            digest = metadata.get("content_hash")
            if not (
                isinstance(digest, str)
                and metadata.get("content_hash_size") == record["size"]
                and metadata.get("content_hash_mtime_ns") == record["mtime_ns"]
            ):
                try:
                    digest = _file_sha256(record["image"])
                except OSError:
                    hash_errors += 1
                    continue
                metadata["content_hash"] = digest
                metadata["content_hash_size"] = record["size"]
                metadata["content_hash_mtime_ns"] = record["mtime_ns"]
                hash_cache_dirty = True
                photos_hashed += 1
            hash_groups.setdefault(digest, []).append(record)

    duplicate_groups = 0
    for group in hash_groups.values():
        if len(group) < 2:
            continue
        duplicate_groups += 1
        for record in group:
            add_suggestion(
                record,
                "exact_duplicate",
                f"Exact duplicate: the same file content appears {len(group)} times",
            )

    category_order = {"exact_duplicate": 0, "screenshot": 1, "likely_forward": 2}
    items = list(suggestions.values())
    for item in items:
        # The metadata reference is useful while building suggestions but is
        # not part of the public view model returned to the UI/CLI.
        item.pop("metadata", None)
        item.pop("mtime_ns", None)
        item["categories"].sort(key=lambda category: category_order[category])
    items.sort(key=lambda item: (category_order[item["categories"][0]], item["image"].lower()))
    if hash_cache_dirty:
        save_processed()
    return {
        "items": items,
        "count": len(items),
        "photos_checked": len(records),
        "duplicate_groups": duplicate_groups,
        "unavailable": unavailable,
        "hash_errors": hash_errors,
        "photos_hashed": photos_hashed,
    }


def move_reviewed_photos(paths, destination):
    """Move explicitly reviewed scanned photos and repair library references."""
    if not isinstance(paths, (list, tuple, set)):
        return {"error": "invalid_paths", "moved": [], "errors": []}
    if not isinstance(destination, str) or not os.path.isdir(destination):
        return {"error": "destination_not_found", "moved": [], "errors": []}

    load_db()
    destination = normalize(destination)
    processed_keys = {
        normalize(path): path for path, metadata in processed_db.items()
        if isinstance(path, str) and isinstance(metadata, dict) and metadata.get("status") == "ok"
    }
    timeline_keys = {
        normalize(path): path for path in timeline_db if isinstance(path, str)
    }
    moved = []
    errors = []
    seen = set()
    faces_changed = False

    for raw_path in paths:
        if not isinstance(raw_path, str):
            errors.append("invalid photo path")
            continue
        source = normalize(raw_path)
        if source in seen:
            continue
        seen.add(source)
        if source not in processed_keys or not os.path.isfile(source):
            errors.append(f"not an accessible scanned photo: {source}")
            continue
        if os.path.normcase(os.path.dirname(source)) == os.path.normcase(destination):
            errors.append(f"destination is already the source folder: {source}")
            continue

        target = _unique_cleanup_destination(destination, os.path.basename(source))
        try:
            shutil.move(source, target)
        except OSError as exc:
            errors.append(f"could not move {source}: {exc}")
            continue

        target = normalize(target)
        old_metadata = processed_db.pop(processed_keys[source], {})
        new_metadata = dict(old_metadata)
        new_metadata.update(image_signature(target))
        processed_db[target] = new_metadata

        old_timeline = timeline_db.pop(timeline_keys.get(source, source), {})
        new_timeline = dict(old_timeline) if isinstance(old_timeline, dict) else {}
        new_timeline.update(image_signature(target))
        new_timeline.setdefault("date", new_metadata.get("date") or get_image_date(target))
        timeline_db[target] = new_timeline

        for face in faces_db:
            image_path = face.get("image") if isinstance(face, dict) else None
            if isinstance(image_path, str) and normalize(image_path) == source:
                face["image"] = target
                face["album"] = normalize(os.path.dirname(target))
                faces_changed = True

        _log_cleanup_move(source, target, action="move_cleanup_suggestion")
        moved.append({"source": source, "destination": target})

    if moved:
        save_processed()
        save_timeline()
        if faces_changed:
            save_faces()
    return {"moved": moved, "errors": errors}


def export_library_report(destination):
    """Export a spreadsheet-friendly report of every known library photo.

    The report is intentionally built from the Timeline, scan, and face
    records together. That includes zero-face and missing photos instead of
    silently limiting the report to photos that have a person assignment.
    """
    if not isinstance(destination, str) or not destination.strip():
        return {"error": "invalid_destination"}
    destination = os.path.abspath(destination)
    parent = os.path.dirname(destination)
    if not os.path.isdir(parent):
        return {"error": "destination_folder_not_found", "path": destination}

    load_db()
    faces_by_image = {}
    for face in faces_db:
        image_path = face.get("image") if isinstance(face, dict) else None
        if isinstance(image_path, str) and image_path:
            faces_by_image.setdefault(normalize(image_path), []).append(face)

    # Preserve compatibility with older databases that used Windows backslash
    # keys by normalising the lookup maps as well as the combined path list.
    processed_by_path = {
        normalize(path): metadata for path, metadata in processed_db.items()
        if isinstance(path, str)
    }
    timeline_by_path = {
        normalize(path): metadata for path, metadata in timeline_db.items()
        if isinstance(path, str)
    }
    all_paths = set(faces_by_image)
    all_paths.update(timeline_by_path)
    all_paths.update(processed_by_path)
    rows = []
    accessible = 0
    missing = 0
    scanned_ok = 0
    no_face = 0

    for path in sorted(all_paths, key=str.lower):
        processed = processed_by_path.get(path, {})
        timeline_entry = timeline_by_path.get(path, {})
        if not isinstance(processed, dict):
            processed = {}
        if not isinstance(timeline_entry, dict):
            timeline_entry = {}
        faces = faces_by_image.get(path, [])
        status = processed.get("status", "not_scanned")
        date = (
            timeline_entry.get("date")
            or processed.get("date")
            or (faces[0].get("date") if faces else None)
            or "unknown"
        )
        exists = os.path.isfile(path)
        if exists:
            accessible += 1
            try:
                size = os.path.getsize(path)
            except OSError:
                size = processed.get("size", "")
        else:
            missing += 1
            size = processed.get("size", timeline_entry.get("size", ""))

        if status == "ok":
            scanned_ok += 1
        face_count = len(faces)
        if face_count == 0:
            try:
                face_count = int(processed.get("faces", 0))
            except (TypeError, ValueError):
                face_count = 0
        is_no_face = status == "ok" and face_count == 0
        if is_no_face:
            no_face += 1

        people = []
        for face in faces:
            person_id = face.get("person_id")
            person = people_by_id.get(person_id, {})
            name = person.get("name") if isinstance(person, dict) else None
            label = name or (f"Person {person_id}" if person_id is not None else "Unassigned")
            if label not in people:
                people.append(label)
        flags = []
        if is_no_face:
            flags.append("no_face")
        if processed.get("cleanup_status"):
            flags.append(str(processed["cleanup_status"]))
        rows.append({
            "timeline_month": date,
            "file_name": os.path.basename(path),
            "full_path": path,
            "folder": normalize(os.path.dirname(path)),
            "scan_status": status,
            "face_count": face_count,
            "people": "; ".join(people),
            "file_exists": "yes" if exists else "no",
            "file_size_bytes": size,
            "flags": "; ".join(flags),
        })

    fieldnames = [
        "timeline_month", "file_name", "full_path", "folder", "scan_status",
        "face_count", "people", "file_exists", "file_size_bytes", "flags",
    ]
    tmp = f"{destination}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, destination)
    except OSError as exc:
        return {"error": "report_write_failed", "detail": str(exc), "path": destination}
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)

    return {
        "path": destination,
        "total_files": len(rows),
        "accessible_files": accessible,
        "missing_files": missing,
        "scanned_ok": scanned_ok,
        "no_face_files": no_face,
    }


def library():
    load_db()
    people_payload = list_people(should_load=False)
    timeline_payload = timeline(should_load=False)
    albums_payload = albums(should_load=False)

    return {
        "people": people_payload.get("people", []),
        "timeline": timeline_payload.get("timeline", {}),
        "albums": albums_payload.get("albums", {}),
        "total_faces": len(faces_db),
    }


def set_person_name(person_id, name):
    """Set a human-readable name for a person and persist DB."""
    global people_by_id

    # Only reload from disk if the DB hasn't been loaded yet (B8: avoids clobbering
    # in-progress state when called while process_folder is running in another thread)
    if not faces_db:
        load_db()

    # try to find the person
    person = people_by_id.get(person_id)
    if person is None:
        # attempt numeric lookup
        try:
            pid = int(person_id)
        except Exception:
            pid = None

        if pid is not None:
            person = people_by_id.get(pid)

    if person is None:
        return {"error": "person_not_found", "person_id": person_id}

    if name is None:
        # clear name
        person.pop("name", None)
    else:
        person["name"] = str(name)

    # sync people_db list
    for i, p in enumerate(people_db):
        if p.get("id") == person.get("id"):
            people_db[i] = person
            break

    save_db()
    return {"id": person.get("id"), "name": person.get("name")}


# ---------------------------------------------------------------------------
# Face & People extended features
# ---------------------------------------------------------------------------

def faces_in_image(image_path):
    """Return all face records detected in a specific source photo."""
    if not faces_db:
        load_db()
    normalized = normalize(image_path)
    results = []
    for face in faces_db:
        if normalize(face.get("image", "")) == normalized:
            pid = face.get("person_id")
            person_rec = people_by_id.get(pid, {})
            person_name = person_rec.get("name") if isinstance(person_rec, dict) else None
            results.append({
                "face_id": face.get("face_id"),
                "face_path": thumbnail_for_face(face),
                "person_id": pid,
                "person_name": person_name or (f"Person {pid}" if pid is not None else "Unassigned"),
                "score": face.get("score"),
                "bbox": face.get("bbox"),
                "bbox_original": face.get("bbox_original"),
            })
    return {"faces": results, "image": normalized, "count": len(results)}


def merge_people(keep_id, merge_id, persist=True):
    """Move all faces from merge_id into keep_id and delete merge_id."""
    if not faces_db:
        load_db()

    # Normalise IDs — may arrive as str from JSON
    try:
        keep_id = int(keep_id)
        merge_id = int(merge_id)
    except Exception:
        return {"error": "invalid_ids"}

    if keep_id == merge_id:
        return {"error": "same_person"}

    keep_person = people_by_id.get(keep_id)
    merge_person = people_by_id.get(merge_id)
    if not keep_person:
        return {"error": "keep_person_not_found", "id": keep_id}
    if not merge_person:
        return {"error": "merge_person_not_found", "id": merge_id}

    keep_faces_set = {int(x) for x in keep_person["faces"]}  # OPT-5: build once
    for fid in list(merge_person.get("faces", [])):
        face = faces_by_id.get(int(fid))
        if face:
            face["person_id"] = keep_id
        if int(fid) not in keep_faces_set:          # OPT-5: O(1) via pre-built set
            keep_person["faces"].append(int(fid))
            keep_faces_set.add(int(fid))

    # Remove merged person
    people_db[:] = [p for p in people_db if p.get("id") != merge_id]
    people_by_id.pop(merge_id, None)

    if persist:
        save_db()
    return {"merged": merge_id, "into": keep_id, "total_faces": len(keep_person["faces"])}


def find_merge_candidates(threshold=0.82):
    """Return pairs of people whose face-embedding centroids are highly similar.

    Algorithm:
      1. For each person, average all stored face embeddings → centroid.
      2. Build a temporary FAISS index of all centroids.
      3. Search for nearest-neighbour pairs with cosine similarity ≥ threshold.
      4. Return deduplicated pairs sorted by similarity (highest first).

    Each result entry: {person_a, name_a, count_a, thumb_a,
                        person_b, name_b, count_b, thumb_b, similarity}
    """
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("Merge threshold must be a number between -1 and 1.") from exc
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("Merge threshold must be a finite number between -1 and 1.")

    if not faces_db:
        load_db()

    _, np_module = require_cv()
    faiss_module = require_faiss()

    # Build centroid per person
    centroids = []   # [(pid, centroid_array)]
    for pid, person in people_by_id.items():
        face_ids = person.get("faces", [])
        embeddings = []
        for fid in face_ids:
            face = faces_by_id.get(int(fid))
            if face and face.get("embedding"):
                try:
                    vector = np_module.asarray(face["embedding"], dtype="float32").reshape(-1)
                    if vector.size == DIM and np_module.isfinite(vector).all():
                        embeddings.append(vector)
                except (TypeError, ValueError):
                    continue
        if not embeddings:
            continue
        centroid = np_module.mean(
            np_module.array(embeddings, dtype="float32"), axis=0
        )
        centroids.append((pid, centroid))

    if len(centroids) < 2:
        return []

    pids     = [c[0] for c in centroids]
    matrix   = np_module.array([c[1] for c in centroids], dtype="float32")
    faiss_module.normalize_L2(matrix)

    temp_idx = faiss_module.IndexFlatIP(DIM)
    temp_idx.add(matrix)

    # Search the two nearest *other* clusters (row 0 is always itself).
    # Never ask FAISS for more entries than are present; it otherwise returns
    # sentinel rows that are easy to mistake for candidates.
    neighbor_count = min(2, len(pids) - 1)
    D, I = temp_idx.search(matrix, neighbor_count + 1)

    seen     = set()
    results  = []

    for row_i, pid_a in enumerate(pids):
        for col_j in range(1, neighbor_count + 1):
            sim = float(D[row_i][col_j])
            if sim < threshold:
                break
            match_pos = int(I[row_i][col_j])
            if match_pos < 0 or match_pos >= len(pids):
                continue
            pid_b = pids[match_pos]
            pair  = (min(pid_a, pid_b), max(pid_a, pid_b))
            if pair in seen:
                continue
            seen.add(pair)

            person_a = people_by_id.get(pid_a, {})
            person_b = people_by_id.get(pid_b, {})

            # Best thumbnail for each person
            def _thumb(person):
                for fid in person.get("faces", []):
                    face = faces_by_id.get(int(fid))
                    if face:
                        t = thumbnail_for_face(face)
                        if t:
                            return t
                return None

            results.append({
                "person_a":  pid_a,
                "name_a":    person_a.get("name") or f"Person {pid_a}",
                "count_a":   len(person_a.get("faces", [])),
                "thumb_a":   _thumb(person_a),
                "person_b":  pid_b,
                "name_b":    person_b.get("name") or f"Person {pid_b}",
                "count_b":   len(person_b.get("faces", [])),
                "thumb_b":   _thumb(person_b),
                "similarity": round(sim, 4),
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


def auto_merge_all(pairs):
    """Batch-merge a list of (keep_id, merge_id) pairs.

    pairs — list of dicts with 'keep_id' and 'merge_id' keys.
    Returns {merged: int, errors: list}.
    """
    merged = 0
    errors = []
    # Candidate pairs can overlap (A-B and B-C). Track the surviving ID so
    # the latter pair still succeeds after B has been merged into A.
    survivor = {}

    def resolve(person_id):
        while person_id in survivor:
            person_id = survivor[person_id]
        return person_id

    for pair in pairs:
        try:
            keep_id  = int(pair["keep_id"])
            merge_id = int(pair["merge_id"])
        except (KeyError, ValueError, TypeError) as exc:
            errors.append(f"bad pair {pair!r}: {exc}")
            continue
        keep_id = resolve(keep_id)
        merge_id = resolve(merge_id)
        if keep_id == merge_id:
            continue
        # Checkpoint once after the batch instead of rewriting every database
        # file for every pair. This makes large reviewed batches responsive.
        res = merge_people(keep_id, merge_id, persist=False)
        if res.get("error"):
            errors.append(f"merge {merge_id}→{keep_id}: {res['error']}")
        else:
            merged += 1
            survivor[merge_id] = keep_id
    if merged:
        save_db()
    return {"merged": merged, "errors": errors}


def split_face_to_new_person(face_id):
    """Move a single face out of its current person into a brand-new person."""
    if not faces_db:
        load_db()

    try:
        fid = int(face_id)
    except Exception:
        return {"error": "invalid_face_id"}

    face = faces_by_id.get(fid)
    if not face:
        return {"error": "face_not_found", "face_id": fid}

    old_pid = face.get("person_id")
    old_person = people_by_id.get(old_pid)
    if old_person:
        old_person["faces"] = [x for x in old_person["faces"] if int(x) != fid]
        if not old_person["faces"]:
            # Remove empty person
            people_db[:] = [p for p in people_db if p.get("id") != old_pid]
            people_by_id.pop(old_pid, None)

    # OPT-2: use O(1) counter instead of O(n) max scan
    global _next_person_id
    new_pid = _next_person_id
    _next_person_id += 1
    new_person = {"id": new_pid, "faces": [fid]}
    people_db.append(new_person)
    people_by_id[new_pid] = new_person
    face["person_id"] = new_pid

    save_db()
    return {"new_person_id": new_pid, "face_id": fid, "old_person_id": old_pid}


def recluster(threshold=None):
    """Re-assign all faces to people using stored embeddings and a new threshold.

    Requires embeddings to have been stored during process_folder (i.e. the
    faces were processed after the embedding-storage fix was applied).
    """
    global people_db, people_by_id, index, id_map

    if not faces_db:
        load_db()

    try:
        thr = float(threshold) if threshold is not None else SIM_THRESHOLD
    except (TypeError, ValueError):
        return {"error": "invalid_threshold", "hint": "Use a number between -1 and 1."}
    if not math.isfinite(thr) or not -1.0 <= thr <= 1.0:
        return {"error": "invalid_threshold", "hint": "Use a finite number between -1 and 1."}

    # Ensure numpy is available, then exclude damaged or incompatible stored
    # vectors instead of abandoning an otherwise recoverable library.
    _, np_module = require_cv()
    faiss_module = require_faiss()
    faces_with_emb = []
    skipped_embeddings = 0
    for face in faces_db:
        if face.get("embedding") is None:
            continue
        try:
            vector = np_module.asarray(face["embedding"], dtype="float32").reshape(-1)
            if vector.size != DIM or not np_module.isfinite(vector).all():
                raise ValueError("invalid vector")
            faces_with_emb.append((face, vector))
        except (TypeError, ValueError):
            skipped_embeddings += 1

    if not faces_with_emb:
        return {
            "error": "no_stored_embeddings",
            "hint": "Re-process your photos so embeddings are stored, then try again.",
        }

    # Reset all person assignments; also reset the person ID counter
    global _next_person_id
    people_db.clear()
    people_by_id.clear()
    _next_person_id = 0   # OPT-2: safe to restart from 0 since all people are being rebuilt
    for face in faces_db:
        face["person_id"] = None

    new_index = faiss_module.IndexFlatIP(DIM)
    new_id_map = []
    # OPT-5: per-person face-id sets so attach duplicate check is O(1)
    person_faces_sets: dict = {}

    for face, embedding in faces_with_emb:
        fid = int(face["face_id"])
        vec = embedding.reshape(1, -1).copy()
        faiss_module.normalize_L2(vec)

        person_id = None
        if new_index.ntotal > 0:
            D, I = new_index.search(vec, 1)
            if D[0][0] >= thr:
                pos = int(I[0][0])
                if 0 <= pos < len(new_id_map):
                    matched_fid = new_id_map[pos]
                    matched_face = faces_by_id.get(matched_fid)
                    if matched_face:
                        person_id = matched_face.get("person_id")

        if person_id is None:
            # OPT-2: O(1) counter instead of O(n) max scan
            new_pid = _next_person_id
            _next_person_id += 1
            new_person = {"id": new_pid, "faces": [fid]}
            people_db.append(new_person)
            people_by_id[new_pid] = new_person
            person_faces_sets[new_pid] = {fid}
            face["person_id"] = new_pid
        else:
            # Attach to existing person with O(1) duplicate check
            face["person_id"] = person_id
            p = people_by_id[person_id]
            fs_set = person_faces_sets.setdefault(person_id, {int(x) for x in p["faces"]})
            if fid not in fs_set:
                p["faces"].append(fid)
                fs_set.add(fid)

        new_index.add(vec)
        new_id_map.append(fid)

    index = new_index
    id_map = new_id_map
    save_db()
    save_index()

    return {
        "people": len(people_db),
        "faces_clustered": len(faces_with_emb),
        "skipped_invalid_embeddings": skipped_embeddings,
        "threshold": thr,
    }


def get_duplicates():
    """Find photos where the same person appears more than once (possible misdetection)."""
    if not faces_db:
        load_db()

    image_person: dict = {}
    for face in faces_db:
        img = face.get("image")
        pid = face.get("person_id")
        if img and pid is not None:
            key = (normalize(img), int(pid))
            image_person.setdefault(key, []).append(face)

    duplicates = []
    for (image_path, pid), dup_faces in image_person.items():
        if len(dup_faces) > 1:
            person_rec = people_by_id.get(pid, {})
            name = (person_rec.get("name") if isinstance(person_rec, dict) else None) or f"Person {pid}"
            duplicates.append({
                "image": image_path,
                "person_id": pid,
                "person_name": name,
                "face_count": len(dup_faces),
                "face_paths": [fp for f in dup_faces if (fp := thumbnail_for_face(f))],
            })

    duplicates.sort(key=lambda d: d["face_count"], reverse=True)
    return {"duplicates": duplicates, "total": len(duplicates)}


def search_face(path):
    load_db()
    init_index()

    cv_module, _ = require_cv()
    img = cv_module.imread(path, cv_module.IMREAD_COLOR)
    if img is None:
        return {"error": "invalid_image"}

    model_img, _ = resize_for_detection(img)
    detected_faces = get_model().get(model_img)
    if not detected_faces:
        return {"error": "no_face"}

    match = search_embedding(detected_faces[0].embedding)
    if match is None:
        return {"result": "unknown"}

    face = faces_by_id.get(int(match))
    if not face:
        return {"result": "unknown"}

    return {"person_id": face.get("person_id"), "face": compact_face(face)}


def handle(req):
    cmd = req.get("cmd")

    if cmd == "process_folder":
        return process_folder(req.get("path"))
    if cmd == "full_scan_folder":
        return full_scan_folder(req.get("path"))
    if cmd == "people":
        return list_people()
    if cmd == "library":
        return library()
    if cmd == "search":
        return search_face(req.get("path"))
    if cmd == "timeline":
        return timeline()
    if cmd == "albums":
        return albums()
    if cmd == "no_face_photos":
        return no_face_photos()
    if cmd == "move_no_face_photos":
        return move_no_face_photos(req.get("paths", []), req.get("destination"))
    if cmd == "cleanup_suggestions":
        return cleanup_suggestions()
    if cmd == "move_reviewed_photos":
        return move_reviewed_photos(req.get("paths", []), req.get("destination"))
    if cmd == "export_library_report":
        return export_library_report(req.get("destination"))
    if cmd == "progress":
        return get_progress()
    if cmd == "faces_in_image":
        return faces_in_image(req.get("path", ""))
    if cmd == "merge_people":
        return merge_people(req.get("keep_id"), req.get("merge_id"))
    if cmd == "split_face":
        return split_face_to_new_person(req.get("face_id"))
    if cmd == "recluster":
        return recluster(req.get("threshold"))
    if cmd == "duplicates":
        return get_duplicates()
    if cmd == "set_name":
        return set_person_name(req.get("person_id"), req.get("name"))
    if cmd == "find_merge_candidates":
        return {"candidates": find_merge_candidates(float(req.get("threshold", 0.82)))}
    if cmd == "auto_merge":
        return auto_merge_all(req.get("pairs", []))

    return {"error": "unknown_command", "cmd": cmd}


if __name__ == "__main__":
    try:
        req = json.loads(sys.stdin.read())
        res = handle(req)
    except Exception as exc:
        res = {"error": str(exc), "trace": traceback.format_exc()}

    print(json.dumps(res, separators=(",", ":")), flush=True)

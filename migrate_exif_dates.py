"""
migrate_exif_dates.py
=====================
One-shot migration: update the 'date' field in data/faces.json from file mtime
to the real EXIF DateTimeOriginal (or creation time on Windows) for every record.

Usage
-----
Run from the project root OR from inside py_app/:

    python migrate_exif_dates.py          # dry-run preview
    python migrate_exif_dates.py --apply  # write changes

A backup (faces.json.bak) is created before any changes are written.
Re-running is safe - unchanged records are skipped.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# -- locate data/faces.json --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = None
for _c in [SCRIPT_DIR, SCRIPT_DIR.parent]:
    if (_c / "data" / "faces.json").exists():
        DATA_DIR = _c / "data"
        break

if DATA_DIR is None:
    print("ERROR: data/faces.json not found. Run from the project root or py_app/.")
    sys.exit(1)

FACES_DB  = DATA_DIR / "faces.json"
FACES_BAK = DATA_DIR / "faces.json.bak"


# -- EXIF date helper ---------------------------------------------------------

def get_best_date(path: str) -> str:
    """Return YYYY-MM: EXIF DateTimeOriginal > ctime > mtime."""
    # 1. EXIF (most accurate - when shutter was pressed)
    try:
        from PIL import Image
        with Image.open(path) as img:
            exif = img.getexif()
            for tag_id in (36867, 306):  # DateTimeOriginal, DateTime
                raw = exif.get(tag_id)
                if raw:
                    dt = datetime.strptime(str(raw)[:19], "%Y:%m:%d %H:%M:%S")
                    return dt.strftime("%Y-%m")
    except Exception:
        pass

    # 2. File creation time (Windows: actual creation timestamp)
    try:
        return datetime.fromtimestamp(os.path.getctime(path)).strftime("%Y-%m")
    except Exception:
        pass

    # 3. Last resort: modification time
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m")
    except Exception:
        return "unknown"


# -- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Migrate face date fields to EXIF dates.")
    parser.add_argument(
        "--apply", action="store_true",
        help="Write changes to disk. Without this flag the script runs as a dry-run preview."
    )
    args = parser.parse_args()
    dry_run = not args.apply

    print(f"{'[DRY RUN] ' if dry_run else ''}Loading {FACES_DB} ...")
    with open(FACES_DB, encoding="utf-8") as f:
        faces = json.load(f)

    print(f"  {len(faces)} face records found.\n")

    changed, skipped, unchanged = 0, 0, 0

    for i, face in enumerate(faces):
        if (i + 1) % 200 == 0 or (i + 1) == len(faces):
            print(f"  processed {i + 1}/{len(faces)} ...")

        image_path = face.get("image", "")
        if not image_path or not os.path.exists(image_path):
            skipped += 1
            continue

        old_date = face.get("date", "unknown")
        new_date = get_best_date(image_path)

        if old_date == new_date:
            unchanged += 1
        else:
            if dry_run and changed < 20:
                print(f"  WOULD change  {old_date!r:12s}  ->  {new_date!r}   ({os.path.basename(image_path)})")
            face["date"] = new_date
            changed += 1

    print()
    print("-" * 58)
    print(f"  Records to update : {changed}")
    print(f"  Already correct   : {unchanged}")
    print(f"  Skipped (no file) : {skipped}")
    print("-" * 58)

    if dry_run:
        print("\n[DRY RUN] No files written. Re-run with --apply to save.\n")
        return

    if changed == 0:
        print("\nNothing to update - faces.json is already up to date.\n")
        return

    shutil.copy2(FACES_DB, FACES_BAK)
    print(f"\nBackup -> {FACES_BAK}")

    tmp = str(FACES_DB) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(faces, f, separators=(",", ":"))
    os.replace(tmp, FACES_DB)

    print(f"Updated -> {FACES_DB}")
    print("\nDone! Restart the app to see the updated Timeline.\n")


if __name__ == "__main__":
    main()

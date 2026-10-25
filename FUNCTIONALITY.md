# Custom-Album Functionality Guide

## What the app does

Custom-Album organizes a local image collection around the people in it. It builds a private, on-device library from folders you select and offers people, chronological, and folder-based browsing. It does not upload photo contents or face embeddings.

## First run

1. Install dependencies from `py_app/requirements.txt` in the Python environment used to launch the app.
2. Run `python main.py` from `py_app`.
3. Click **Folder** and choose an image folder.
4. Use **Incremental Scan** for normal scans. The first scan processes every supported image.

The first face operation downloads or initializes the InsightFace model if the local runtime needs it. CUDA is selected automatically only when ONNX Runtime exposes a CUDA provider.

## Main workflows

### Scan a library

**Incremental Scan** discovers all supported images, records them in Timeline, then analyzes only new or changed files. A file is considered changed when its size or high-resolution modification time changes. Images that previously produced a transient processing error are retried; files confirmed as invalid are skipped until they change.

**Full Scan Folder** removes and regenerates all scan records for the selected folder only. Use it after changing detection settings, repairing a library, or replacing many photos. It can take a long time and does not remove records belonging to other selected folders.

The sidebar shows a progress bar and current filename. While scanning, actions that can conflict with the shared library state are disabled.

### Browse People

The People tab contains one item per clustered person.

- Filter by a saved name with the search field.
- Sort by face count or alphabetically.
- Rename a person by editing their list label or opening the person dialog.
- Double-click a person to open their face gallery.
- Use left/right arrow keys in a gallery to navigate.
- **View in Photo** opens the original source image with the selected face highlighted.
- **Who's in this photo?** lists other detected faces from the same source image.
- **Open File** opens the source image through the operating system.

If the original photo is disconnected but its face crop still exists, the person remains browsable. If neither is accessible, that display entry is hidden but retained in the database for a future drive reconnection.

### Correct people groups

Face grouping is probabilistic. Review it before treating names or suggested merges as definitive.

- Select two or more people and choose **Merge Selected**. The first selected person is retained.
- In a face gallery choose **Split to New Person** to move one incorrect face into a new cluster.
- **Auto-Merge Similar** computes close person-centroid pairs, lets you review them, and only merges after confirmation. High-confidence results are preselected, but they still require human review.
- **Re-cluster Faces** recreates all groups from saved embeddings at a chosen threshold. It preserves no manual merge/split boundaries or person names attached to newly reassigned IDs, so use it deliberately.
- **Find Duplicates** finds a person appearing more than once in a photo. This can indicate a mistaken cluster, but it may also be a real photo of two similar people.

Manual merge and split changes currently do not have an undo command. Back up `data/` before large cleanup sessions.

### Search by face

Choose any supported reference image with a visible face and click **Search**. The first detected face is compared against the local FAISS index. A positive result opens the matched person's gallery. Photos with multiple faces should be cropped first if you want to search for someone other than the first detected face.

### Browse Timeline and Albums

**Timeline** groups every discovered image by `YYYY-MM`, including zero-face images. Capture date precedence is EXIF `DateTimeOriginal`, then EXIF `DateTime`, file creation time, then modification time. Double-click a month to open its source-photo gallery.

**Albums** groups detected faces by their source folder. It is a folder view, not a user-curated album system; photos with no face are intentionally absent here.

### Review and move no-face photos

Choose **Review No-Face Photos** from the sidebar to see photos that completed scanning with zero detected faces. The list deliberately excludes unscanned, failed, missing, and already-cleaned files, so it is a review queue rather than a guess about every image in the library.

Tick the photos you want, choose an existing destination folder, and confirm. The app moves only those selections; it never deletes files or overwrites an existing filename. If a filename already exists in the destination, it adds a number such as `photo (2).jpg`. The move is recorded in `data/cleanup_moves.jsonl`, and Timeline/scan metadata follow the file to its new path. A future release can use this audit trail to offer one-click undo.

### Review cleanup suggestions

Choose **Review Cleanup Suggestions** to open a separate inbox for three types of suggestions:

- **Exact duplicate** means two or more files have identical SHA-256 content hashes. These are byte-for-byte copies, not merely similar-looking photos.
- **Screenshot** is suggested when a filename or its folder contains a clear screenshot signal such as `Screenshot` or `Screen Shot`.
- **Likely forward** is suggested only when the filename resembles a WhatsApp, Telegram, Messenger, ShareChat, or `...-WA...` export.

The app calculates duplicate hashes in the background and does not preselect, move, delete, or hide any result. Double-click an item to open it in your operating system, then tick only the files you want to move. The move confirmation is the same safe workflow used for no-face photos. If a moved photo has a detected face, the app updates its People, Album, Timeline, and scan references so the library remains connected to it.

### Export a library report

Choose **Export Library Report** to save a CSV file that opens in Excel. It includes every known photo from the Timeline, scan index, or face library—even if the file is now missing. Each row includes the timeline month, filename, full path, source folder, scan status, face count, people, current availability, size, and cleanup flags. The report is read-only: exporting it does not move, scan, or change any photo.

## Stored data and recovery

The application creates `data/` automatically. The important recovery unit is the entire folder, especially `faces.json`, `people.json`, `processed.json`, `timeline.json`, `faiss.index`, and `faces/`.

- Data is local and safe to copy while the app is closed.
- JSON and FAISS writes are crash-safe at the individual-file level.
- If the FAISS index is unavailable or inconsistent, the service rebuilds it from face embeddings when possible.
- `errors.jsonl` contains per-file diagnostics for files the scanner could not handle.
- `migrate_exif_dates.py` previews date corrections by default; pass `--apply` only after reviewing its output. It creates `faces.json.bak` before writing.

## Supported media and limitations

Supported extensions are JPEG, PNG, WebP, BMP, TIFF, and TIF. HEIC, RAW camera formats, video, live photos, and remote cloud placeholders are not supported yet. The app indexes absolute file paths, so moving source folders requires returning them to the same location or rescanning.

The detector processes downscaled copies for speed, while stored original-space boxes retain accurate overlays and crop regeneration. Very small, blurred, occluded, side-profile, or heavily edited faces may be missed or grouped incorrectly.

## Suggested next features

### Highest value

1. **Undo/history for people edits.** Record merge, split, and rename operations as reversible events. This makes automated suggestions safe to explore.
2. **Backup and restore controls.** Add a one-click snapshot of `data/`, plus restore and integrity-check screens before a full scan or re-cluster.
3. **Cancellable background jobs.** Move face search, re-clustering, thumbnail generation, and auto-merge analysis off the UI thread; provide cancel, pause, and resumable checkpoints.
4. **Folder registry and scan history.** Remember selected roots, last scan time, scan settings, error counts, and a “scan changed files” action for each collection.

### Media and organization

5. **HEIC/RAW/video support.** Add pluggable decoders and video key-frame extraction.
6. **User-curated albums, tags, favourites, and rating.** Separate user organization from the current source-folder album grouping.
7. **Date/location map tools.** Read EXIF GPS safely, show a map only on demand, and support a privacy setting that strips location from exports.
8. **Near-duplicate photo detection.** Add perceptual hashes to find visually similar burst shots, resized copies, and lightly edited versions alongside the current exact-duplicate check.

### Accuracy and performance

9. **Representative-face selection.** Choose the sharpest, most frontal high-confidence crop per person instead of the first stored face.
10. **Global/reviewable clustering.** Add centroid and quality-weighted cluster strategies, confidence explanations, and a review queue for uncertain matches.
11. **SQLite metadata store.** Keep face records, scan state, undo history, and indexes transactionally while retaining generated crop files on disk.
12. **Cache management.** Add UI controls to inspect and safely purge orphaned face crops and scaled thumbnail-cache entries.

### Sharing and accessibility

13. **Export selected photos or a person album.** Copy/link source files with date-based naming and optional face/location metadata removal.
14. **Keyboard shortcuts and accessibility labels.** Make frequent actions faster and ensure screen-reader-friendly status text.
15. **Tests and diagnostics panel.** Cover persistence migrations, scan decisions, and people edits; expose environment/model/provider details for troubleshooting.

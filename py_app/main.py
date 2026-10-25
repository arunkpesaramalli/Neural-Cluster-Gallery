import os
import sys
import threading
import traceback
import hashlib
from pathlib import Path
from functools import partial

from PySide6.QtCore import QTimer, Qt, QUrl, QSize
from PySide6.QtGui import QPixmap, QIcon, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QProgressBar,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QTabWidget,
    QMessageBox,
    QDialog,
    QLineEdit,
    QSlider,
    QComboBox,
    QCheckBox,
    QFrame,
    QScrollArea,
)
from PySide6.QtGui import QDesktopServices
from collections import OrderedDict


RESOURCES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
if RESOURCES_DIR not in sys.path:
    sys.path.insert(0, RESOURCES_DIR)

try:
    import face_service as fs
except Exception:
    fs = None


class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Media — Python GUI")
        self.resize(1000, 700)

        self.folder_path = ""
        self.process_thread = None
        self.process_result = None

        self._pix_cache = OrderedDict()
        self._pix_cache_limit = 300
        self._people_filter_timer = QTimer(self)
        self._people_filter_timer.setSingleShot(True)
        self._people_filter_timer.setInterval(150)
        self._people_filter_timer.timeout.connect(self._apply_people_filter)

        # persistent thumbnail cache directory (py_app/.cache/thumbnails)
        self.cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "thumbnails")
        os.makedirs(self.cache_dir, exist_ok=True)

        self._suppress_item_changed = False
        self._build_ui()

        if fs:
            try:
                fs.load_db()
            except Exception:
                # ignore; will show error when calling endpoints
                pass

        self.progress_timer = QTimer(self)
        self.progress_timer.setInterval(1500)
        self.progress_timer.timeout.connect(self._poll_progress)

        self.load_library()

    def _build_ui(self):
        root = QHBoxLayout(self)

        # Sidebar
        sidebar = QVBoxLayout()
        brand = QLabel("<h1>AI Media</h1>")
        sidebar.addWidget(brand)

        self.status_label = QLabel("Ready")
        sidebar.addWidget(self.status_label)

        self.btn_folder = QPushButton("Folder")
        self.btn_folder.clicked.connect(self.select_folder)
        sidebar.addWidget(self.btn_folder)

        self.btn_process = QPushButton("Incremental Scan")
        self.btn_process.clicked.connect(self.process_folder)
        self.btn_process.setEnabled(False)
        sidebar.addWidget(self.btn_process)

        self.btn_full_scan = QPushButton("Full Scan Folder")
        self.btn_full_scan.clicked.connect(self.full_scan_folder)
        self.btn_full_scan.setEnabled(False)
        sidebar.addWidget(self.btn_full_scan)

        sidebar.addSpacing(6)

        self.btn_search_face = QPushButton("🔍  Search by Face")
        self.btn_search_face.clicked.connect(self.search_by_face)
        sidebar.addWidget(self.btn_search_face)

        self.btn_merge = QPushButton("🔀  Merge Selected")
        self.btn_merge.clicked.connect(self.merge_selected_people)
        self.btn_merge.setEnabled(False)
        sidebar.addWidget(self.btn_merge)

        self.btn_auto_merge = QPushButton("🤝  Auto-Merge Similar")
        self.btn_auto_merge.clicked.connect(self.show_auto_merge_dialog)
        sidebar.addWidget(self.btn_auto_merge)

        self.btn_recluster = QPushButton("⚙️  Re-cluster Faces")
        self.btn_recluster.clicked.connect(self.show_recluster_dialog)
        sidebar.addWidget(self.btn_recluster)

        self.btn_duplicates = QPushButton("🔁  Find Duplicates")
        self.btn_duplicates.clicked.connect(self.show_duplicates)
        sidebar.addWidget(self.btn_duplicates)

        self.btn_no_face_cleanup = QPushButton("🧹  Review No-Face Photos")
        self.btn_no_face_cleanup.setToolTip(
            "Review successfully scanned photos with no detected face before moving them."
        )
        self.btn_no_face_cleanup.clicked.connect(self.show_no_face_cleanup)
        sidebar.addWidget(self.btn_no_face_cleanup)

        self.btn_cleanup_suggestions = QPushButton("🔎  Review Cleanup Suggestions")
        self.btn_cleanup_suggestions.setToolTip(
            "Review screenshots, likely forwards, and exact duplicate files before moving them."
        )
        self.btn_cleanup_suggestions.clicked.connect(self.show_cleanup_suggestions)
        sidebar.addWidget(self.btn_cleanup_suggestions)

        self.btn_export_report = QPushButton("📄  Export Library Report")
        self.btn_export_report.setToolTip(
            "Save a CSV report with each known photo's timeline month and full path."
        )
        self.btn_export_report.clicked.connect(self.export_library_report)
        sidebar.addWidget(self.btn_export_report)

        sidebar.addSpacing(6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        sidebar.addWidget(self.progress_bar)

        sidebar.addStretch(1)
        root.addLayout(sidebar, 1)

        # Content area
        content = QVBoxLayout()

        self.tabs = QTabWidget()

        # People tab — filter row + list in a container widget
        people_tab_widget = QWidget()
        people_tab_layout = QVBoxLayout(people_tab_widget)
        people_tab_layout.setContentsMargins(4, 4, 4, 0)
        people_tab_layout.setSpacing(4)

        filter_row = QHBoxLayout()
        self.people_filter_edit = QLineEdit()
        self.people_filter_edit.setPlaceholderText("🔍  Filter by name…")
        self.people_filter_edit.setClearButtonEnabled(True)
        self.people_filter_edit.textChanged.connect(self._schedule_people_filter)
        filter_row.addWidget(self.people_filter_edit, 3)

        self.people_sort_combo = QComboBox()
        self.people_sort_combo.addItems(["Most faces", "Fewest faces", "A → Z", "Z → A"])
        self.people_sort_combo.currentIndexChanged.connect(self._apply_people_filter)
        filter_row.addWidget(self.people_sort_combo, 2)
        people_tab_layout.addLayout(filter_row)

        self.people_list = QListWidget()
        self.people_list.setIconSize(QSize(96, 96))
        self.people_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.people_list.itemDoubleClicked.connect(self._on_people_item_activated)
        self.people_list.itemChanged.connect(self._on_people_item_changed)
        self.people_list.itemSelectionChanged.connect(self._on_people_selection_changed)
        people_tab_layout.addWidget(self.people_list)

        self.tabs.addTab(people_tab_widget, "People")

        # Timeline
        timeline_tab = QWidget()
        timeline_layout = QVBoxLayout(timeline_tab)
        timeline_layout.setContentsMargins(4, 4, 4, 0)
        timeline_layout.addWidget(QLabel(
            "All scanned photos, grouped by the month they were taken. "
            "Double-click a month to browse its photos."
        ))
        self.timeline_list = QListWidget()
        self.timeline_list.setIconSize(QSize(96, 96))
        self.timeline_list.itemDoubleClicked.connect(partial(self._on_group_item_activated, self.timeline_list))
        timeline_layout.addWidget(self.timeline_list)
        self.tabs.addTab(timeline_tab, "Timeline")

        # Albums
        self.albums_list = QListWidget()
        self.albums_list.setIconSize(QSize(96, 96))
        self.albums_list.itemDoubleClicked.connect(partial(self._on_group_item_activated, self.albums_list))
        self.tabs.addTab(self.albums_list, "Albums")

        content.addWidget(self.tabs)

        root.addLayout(content, 3)

    def select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if path:
            self.folder_path = path
            self.status_label.setText(f"Folder: {path}")
            self.btn_process.setEnabled(True)
            self.btn_full_scan.setEnabled(True)

    def process_folder(self):
        self._start_scan(full_scan=False)

    def full_scan_folder(self):
        if not self.folder_path or fs is None:
            return
        reply = QMessageBox.question(
            self,
            "Full scan selected folder",
            "Rebuild all face and Timeline records for the selected folder?\n\n"
            "Records from other folders are kept. This may take a long time.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._start_scan(full_scan=True)

    def _start_scan(self, full_scan=False):
        if not self.folder_path or fs is None:
            return

        # Capture the target before the worker starts.  The backend has
        # process-wide mutable state, so concurrent library actions must wait.
        folder_path = self.folder_path
        self._set_scan_controls(scanning=True)
        self.status_label.setText("Preparing full scan..." if full_scan else "Scanning folder...")
        self.progress_bar.setValue(0)

        def target():
            try:
                scanner = fs.full_scan_folder if full_scan else fs.process_folder
                self.process_result = scanner(folder_path)
            except Exception as exc:
                self.process_result = {"error": str(exc), "trace": traceback.format_exc()}

        self.process_thread = threading.Thread(target=target, daemon=True)
        self.process_thread.start()
        self.progress_timer.start()

    def _set_scan_controls(self, scanning: bool):
        """Prevent concurrent reads/writes while the scan worker owns the DB."""
        can_scan = bool(self.folder_path) and not scanning
        self.btn_folder.setEnabled(not scanning)
        self.btn_process.setEnabled(can_scan)
        self.btn_full_scan.setEnabled(can_scan)
        for widget in (
            self.btn_search_face,
            self.btn_auto_merge,
            self.btn_recluster,
            self.btn_duplicates,
            self.btn_no_face_cleanup,
            self.btn_cleanup_suggestions,
            self.btn_export_report,
            self.people_list,
            self.timeline_list,
            self.albums_list,
        ):
            widget.setEnabled(not scanning)
        self.btn_merge.setEnabled(
            not scanning and len(self.people_list.selectedItems()) >= 2
        )

    def _poll_progress(self):
        try:
            progress = fs.get_progress()
        except Exception:
            progress = None

        if progress:
            percent = int(progress.get("percent", 0))
            status = progress.get("status", "processing")
            total = int(progress.get("total", 0) or 0)
            if total == 0 and status not in ("complete", "idle", "full scan failed"):
                self.progress_bar.setRange(0, 0)
            else:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(percent)
            current = progress.get("current")
            meta = status.capitalize() if total == 0 else f"{percent}% {status}"
            if current:
                meta += f" — {os.path.basename(current)}"
            self.status_label.setText(meta)

        if self.process_thread and not self.process_thread.is_alive():
            self.progress_timer.stop()
            self._set_scan_controls(scanning=False)
            # reload library and show result
            if isinstance(self.process_result, dict) and self.process_result.get("error"):
                QMessageBox.critical(self, "Processing error", str(self.process_result.get("error")))
            else:
                result = self.process_result or {}
                self.status_label.setText(
                    "Scan complete: "
                    f"{result.get('timeline_photos', 0)} photos indexed, "
                    f"{result.get('processed_images', 0)} new photos processed"
                )
            self.load_library()

    def load_library(self):
        if fs is None:
            QMessageBox.critical(self, "Error", f"Could not import face_service from {RESOURCES_DIR}")
            return

        try:
            payload = fs.library()
        except Exception as exc:
            self.status_label.setText(f"Unable to load library: {exc}")
            return

        people = payload.get("people", [])
        timeline = payload.get("timeline", {})
        albums = payload.get("albums", {})

        self._populate_people(people)
        self._populate_groups(self.timeline_list, timeline, item_label="photos")
        self._populate_groups(self.albums_list, albums)

    def _populate_people(self, people):
        """Cache the full people list and apply current filter/sort."""
        self._all_people_data = list(people)
        self._apply_people_filter()

    def _schedule_people_filter(self):
        """Coalesce keystrokes so large people libraries do not re-render per key."""
        self._people_filter_timer.start()

    def _apply_people_filter(self):
        """Re-render the people list using the current filter text and sort order."""
        query    = self.people_filter_edit.text().strip().lower()
        sort_idx = self.people_sort_combo.currentIndex()

        filtered = list(getattr(self, "_all_people_data", []))

        # Filter by name substring
        if query:
            filtered = [
                p for p in filtered
                if query in (p.get("name") or f"Person {p.get('id')}").lower()
            ]

        # Sort
        def sort_key(p):
            return (p.get("name") or f"zzz_person_{p.get('id')}").lower()

        if sort_idx == 0:   # Most faces (default from backend)
            filtered = sorted(filtered, key=lambda p: p.get("count", 0), reverse=True)
        elif sort_idx == 1: # Fewest faces
            filtered = sorted(filtered, key=lambda p: p.get("count", 0))
        elif sort_idx == 2: # A → Z
            filtered = sorted(filtered, key=sort_key)
        elif sort_idx == 3: # Z → A
            filtered = sorted(filtered, key=sort_key, reverse=True)

        # Render
        self._suppress_item_changed = True
        self.people_list.setUpdatesEnabled(False)
        try:
            self.people_list.clear()
            for person in filtered:
                thumbnail    = person.get("thumbnail")
                count        = person.get("count", 0)
                pid          = person.get("id")
                display_name = person.get("name") or f"Person {pid}"
                item = QListWidgetItem(display_name)
                item.setData(Qt.UserRole, person)
                item.setToolTip(f"{count} face{'s' if count != 1 else ''}")
                item.setFlags(item.flags() | Qt.ItemIsEditable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                if thumbnail and os.path.exists(thumbnail):
                    try:
                        pix = self._load_pixmap(thumbnail, 96)
                        if pix:
                            item.setIcon(QIcon(pix))
                    except Exception:
                        pass
                self.people_list.addItem(item)
        finally:
            self.people_list.setUpdatesEnabled(True)
            self._suppress_item_changed = False

    def _populate_groups(self, widget: QListWidget, groups: dict, item_label="faces"):
        widget.setUpdatesEnabled(False)
        try:
            widget.clear()
            # Newest months first; put undated photos at the end instead of above
            # real dates.
            for name, group in sorted(
                groups.items(), key=lambda entry: (entry[0] != "unknown", entry[0]), reverse=True
            ):
                count = group.get("count", 0)
                text = f"{name} — {count} faces"
                item = QListWidgetItem(text)
                if item_label != "faces":
                    item.setText(f"{name} - {count} {item_label}")
                item.setData(Qt.UserRole, group)
                item.setToolTip(f"Double-click to browse {count} {item_label}")
                # show first thumbnail if available
                items = group.get("items", [])
                if items:
                    fp = (
                        items[0].get("preview_path")
                        or items[0].get("face_path")
                        or items[0].get("image")
                    )
                    if fp and os.path.exists(fp):
                        try:
                            pix = self._load_pixmap(fp, 96)
                            if pix:
                                item.setIcon(QIcon(pix))
                        except Exception:
                            pass
                widget.addItem(item)
        finally:
            widget.setUpdatesEnabled(True)

    def _load_pixmap(self, path: str, size: int = 96):
        # LRU cache for scaled pixmaps
        key = (path, size)
        if key in self._pix_cache:
            # move to end
            pix = self._pix_cache.pop(key)
            self._pix_cache[key] = pix
            return pix
        if not os.path.exists(path):
            return None

        # Try to use a persistent cached thumbnail on disk (generate on demand)
        try:
            cache_path = self._ensure_cached_thumbnail(path, size)
            if cache_path and os.path.exists(cache_path):
                pix = QPixmap(cache_path)
                if pix.isNull():
                    return None
                self._pix_cache[key] = pix
                # evict if over limit
                if len(self._pix_cache) > self._pix_cache_limit:
                    self._pix_cache.popitem(last=False)
                return pix
        except Exception:
            # fall back to in-memory scaling
            try:
                pix = QPixmap(path)
                if pix.isNull():
                    return None
                pix = pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self._pix_cache[key] = pix
                if len(self._pix_cache) > self._pix_cache_limit:
                    self._pix_cache.popitem(last=False)
                return pix
            except Exception:
                return None

    def _on_people_item_activated(self, item: QListWidgetItem):
        person = item.data(Qt.UserRole)
        if not person:
            return
        # open a dialog that shows name and gallery
        dialog = PersonDialog(person, parent=self)
        if dialog.exec() == QDialog.Accepted:
            # refresh library after potential edits
            self.load_library()

    # ------------------------------------------------------------------
    # New feature methods
    # ------------------------------------------------------------------

    def _on_people_selection_changed(self):
        """Enable/disable Merge button based on selection count."""
        count = len(self.people_list.selectedItems())
        self.btn_merge.setEnabled(count >= 2)

    def search_by_face(self):
        """Open a face-search dialog: pick a reference image, find matching person."""
        if fs is None:
            QMessageBox.critical(self, "Error", "face_service not loaded.")
            return
        dlg = FaceSearchDialog(parent=self)
        dlg.exec()

    def merge_selected_people(self):
        """Merge all selected people into the topmost selected person."""
        if fs is None:
            return
        items = self.people_list.selectedItems()
        if len(items) < 2:
            return

        # First selected item = keep; the rest are merged into it
        keep_item = items[0]
        merge_items = items[1:]
        keep_person = keep_item.data(Qt.UserRole)
        if not keep_person:
            return

        keep_name = keep_person.get("name") or f"Person {keep_person.get('id')}"
        merge_names = ", ".join(
            (p.data(Qt.UserRole).get("name") or f"Person {p.data(Qt.UserRole).get('id')}")
            for p in merge_items
            if p.data(Qt.UserRole)
        )

        reply = QMessageBox.question(
            self,
            "Merge People",
            f"Merge:\n  {merge_names}\ninto:\n  {keep_name}?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        pairs = []
        for item in merge_items:
            person = item.data(Qt.UserRole)
            if person:
                pairs.append({"keep_id": keep_person.get("id"), "merge_id": person.get("id")})
        try:
            result = fs.auto_merge_all(pairs)
            errors = result.get("errors", [])
        except Exception as exc:
            errors = [str(exc)]

        if errors:
            QMessageBox.warning(self, "Merge Errors", "\n".join(errors))
        else:
            self.status_label.setText(f"Merged into {keep_name}")

        self.load_library()

    def show_recluster_dialog(self):
        if fs is None:
            return
        dlg = ReclusterDialog(parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.load_library()

    def show_duplicates(self):
        if fs is None:
            return
        try:
            result = fs.get_duplicates()
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return
        dlg = DuplicatesDialog(result.get("duplicates", []), parent=self)
        dlg.exec()

    def show_no_face_cleanup(self):
        """Review scanned zero-face photos before allowing any file moves."""
        if fs is None:
            return
        try:
            result = fs.no_face_photos()
        except Exception as exc:
            QMessageBox.critical(self, "No-face cleanup", str(exc))
            return
        dlg = NoFaceCleanupDialog(result, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.load_library()

    def show_cleanup_suggestions(self):
        """Open the review-first inbox for screenshots, forwards, and duplicates."""
        if fs is None:
            return
        dlg = CleanupSuggestionsDialog(parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.load_library()

    def export_library_report(self):
        """Export an Excel-friendly inventory without changing library data."""
        if fs is None:
            return
        default_name = "custom_album_library_report.csv"
        if self.folder_path:
            folder_name = os.path.basename(os.path.normpath(self.folder_path)) or "library"
            default_name = f"{folder_name}_report.csv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Library Report",
            default_name,
            "CSV files (*.csv)",
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            result = fs.export_library_report(path)
        except Exception as exc:
            QMessageBox.critical(self, "Report export failed", str(exc))
            return
        if result.get("error"):
            QMessageBox.critical(self, "Report export failed", result.get("detail") or result["error"])
            return
        QMessageBox.information(
            self,
            "Library report exported",
            f"Saved {result.get('total_files', 0)} file(s) to:\n{result.get('path')}\n\n"
            f"Accessible: {result.get('accessible_files', 0)}\n"
            f"Missing: {result.get('missing_files', 0)}\n"
            f"No-face: {result.get('no_face_files', 0)}",
        )

    def show_auto_merge_dialog(self):
        if fs is None:
            return
        dlg = AutoMergeDialog(parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.load_library()

    # ------------------------------------------------------------------
    def _on_people_item_changed(self, item: QListWidgetItem):
        """Inline rename handler — fires when user edits a list item directly."""
        if self._suppress_item_changed:
            return

        if fs is None:
            return

        person = item.data(Qt.UserRole)
        if not person:
            return

        new_name = item.text().strip()
        # treat empty string as clearing the name
        try:
            res = fs.set_person_name(person.get("id"), new_name or None)
            if res.get("error"):
                QMessageBox.critical(self, "Error", f"Failed to save name: {res.get('error')}")
                # revert
                self._suppress_item_changed = True
                display_name = person.get("name") or f"Person {person.get('id')}"
                item.setText(display_name)
                self._suppress_item_changed = False
            else:
                # update local person record
                person["name"] = res.get("name")
                # update item display text to reflect new name
                display_name = person.get("name") or f"Person {person.get('id')}"
                self._suppress_item_changed = True
                item.setText(display_name)
                self._suppress_item_changed = False
                item.setData(Qt.UserRole, person)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save name: {exc}")
            # revert
            self._suppress_item_changed = True
            display_name = person.get("name") or f"Person {person.get('id')}"
            item.setText(display_name)
            self._suppress_item_changed = False

    def _on_group_item_activated(self, widget: QListWidget, item: QListWidgetItem):
        group = item.data(Qt.UserRole)
        if not group:
            return
        items = group.get("items", [])
        records = []
        for it in items:
            fp = it.get("face_path")
            img = it.get("image")
            if (fp and os.path.exists(fp)) or (img and os.path.exists(img)):
                records.append(dict(it))

        if not records:
            QMessageBox.information(self, "No photos", "No accessible photos are available for this group.")
            return
        title = "Photo Timeline" if group.get("kind") == "photos" else "Gallery"
        viewer = ImageViewer(records, title=title, parent=self)
        viewer.exec()

    def _cached_thumb_path(self, src_path: str, size: int) -> str:
        # Include the file signature: a path-only key serves a stale preview
        # after a photo is edited or replaced in place.
        try:
            stat = os.stat(src_path)
            signature = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            signature = "missing"
        h = hashlib.sha1(f"{src_path}\0{size}\0{signature}".encode("utf-8")).hexdigest()
        fn = f"{h}_{size}.jpg"
        return os.path.join(self.cache_dir, fn)

    def _ensure_cached_thumbnail(self, src_path: str, size: int) -> str:
        """Return path to cached thumbnail; generate it if missing."""
        if not os.path.exists(src_path):
            return None

        cache_path = self._cached_thumb_path(src_path, size)
        if os.path.exists(cache_path):
            return cache_path

        # Try OpenCV first for fast resizing
        try:
            import cv2

            img = cv2.imread(src_path)
            if img is not None:
                h, w = img.shape[:2]
                longest = max(h, w)
                if longest > size:
                    scale = size / float(longest)
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                if cv2.imwrite(cache_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
                    return cache_path
        except Exception:
            pass

        # Fallback: use QPixmap to scale and save
        try:
            pix = QPixmap(src_path)
            if not pix.isNull():
                scaled = pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                if scaled.save(cache_path, "JPEG", quality=85):
                    return cache_path
        except Exception:
            pass

        return None


class PersonDialog(QDialog):
    def __init__(self, person, parent=None):
        super().__init__(parent)
        self.person = person
        self.setWindowTitle(f"Person {person.get('id')}")
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        # name editor
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Optional name for this person")
        self.name_edit.setText(str(person.get("name", "")) if person.get("name") is not None else "")
        layout.addWidget(QLabel("Name:"))
        layout.addWidget(self.name_edit)

        # This also shows source photos for records whose cached face crop is
        # no longer available.
        btn_view = QPushButton("View Faces / Original Photos")
        btn_view.clicked.connect(self._view_faces)
        layout.addWidget(btn_view)

        # save/close
        controls = QHBoxLayout()
        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self._save)
        controls.addWidget(btn_save)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        controls.addWidget(btn_close)
        layout.addLayout(controls)

    def _view_faces(self):
        faces = self.person.get("faces", [])
        records = []
        for f in faces:
            # New format: dict with face_id, face_path, score, image
            if isinstance(f, dict):
                fp = f.get("face_path")
                image = f.get("image")
                if (fp and os.path.exists(fp)) or (image and os.path.exists(image)):
                    records.append(f)  # ImageViewer falls back to the original photo
            elif isinstance(f, str) and os.path.exists(f):
                records.append({"face_path": f, "image": None, "score": None})
            else:
                # legacy: numeric face ID
                try:
                    fid = int(f)
                    rec = fs.faces_by_id.get(fid)
                    if rec:
                        fp = rec.get("face_path")
                        if fp and os.path.exists(fp):
                            records.append({
                                "face_id": fid,
                                "face_path": fp,
                                "image": rec.get("image"),
                                "score": rec.get("score"),
                                "bbox": rec.get("bbox"),
                                "bbox_original": rec.get("bbox_original"),
                            })
                except Exception:
                    continue

        if not records:
            QMessageBox.information(
                self,
                "No photos available",
                "Neither face thumbnails nor the original photos are available for this person.",
            )
            return

        viewer = ImageViewer(records, title=f"Faces — {self.person.get('name') or 'Person '+str(self.person.get('id'))}", parent=self)
        viewer.exec()

    def _save(self):
        name = self.name_edit.text().strip()
        try:
            res = fs.set_person_name(self.person.get("id"), name or None)
            if res.get("error"):
                QMessageBox.critical(self, "Error", str(res.get("error")))
                return
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save name: {exc}")
            return

        self.accept()

    # end of PersonDialog


class ImageViewer(QDialog):
    def __init__(self, image_paths, title="Gallery", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(860, 640)
        self.paths = image_paths
        self.index = 0

        layout = QVBoxLayout(self)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label, 1)

        # Info bar: quality score + counter
        info_row = QHBoxLayout()
        self.score_label = QLabel()
        self.score_label.setAlignment(Qt.AlignLeft)
        self.score_label.setStyleSheet("color: #aaa; font-size: 12px;")
        info_row.addWidget(self.score_label)
        self.counter_label = QLabel()
        self.counter_label.setAlignment(Qt.AlignRight)
        self.counter_label.setStyleSheet("color: #aaa; font-size: 12px;")
        info_row.addWidget(self.counter_label)
        layout.addLayout(info_row)

        controls = QHBoxLayout()
        btn_prev = QPushButton("◄ Prev")
        btn_prev.clicked.connect(self.prev)
        controls.addWidget(btn_prev)

        btn_who = QPushButton("👥 Who's in this photo?")
        btn_who.clicked.connect(self.show_cooccurrence)
        controls.addWidget(btn_who)

        btn_split = QPushButton("✂ Split to New Person")
        btn_split.clicked.connect(self.split_current_face)
        controls.addWidget(btn_split)

        btn_photo = QPushButton("🖼 View in Photo")
        btn_photo.clicked.connect(self.view_in_photo)
        controls.addWidget(btn_photo)

        btn_open = QPushButton("📂 Open File")
        btn_open.clicked.connect(self.open_file)
        controls.addWidget(btn_open)

        btn_next = QPushButton("Next ►")
        btn_next.clicked.connect(self.next)
        controls.addWidget(btn_next)
        layout.addLayout(controls)

        self._show_current()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev()
        elif event.key() == Qt.Key_Right:
            self.next()
        else:
            super().keyPressEvent(event)

    def _current_record(self):
        if not self.paths:
            return {}
        p = self.paths[self.index]
        return p if isinstance(p, dict) else {"face_path": p}

    def _show_current(self):
        if not self.paths:
            return
        rec = self._current_record()
        thumb = rec.get("image") if rec.get("display_original") else (rec.get("face_path") or rec.get("image"))

        if thumb and os.path.exists(thumb):
            pix = QPixmap(thumb)
            if not pix.isNull():
                label_size = self.image_label.size()
                if label_size.width() < 10 or label_size.height() < 10:
                    label_size = self.image_label.sizeHint()
                if label_size.width() < 10 or label_size.height() < 10:
                    label_size = self.size() * 0.85
                scaled = pix.scaled(label_size * 0.95, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled)
            else:
                self.image_label.setText("Unable to load image")
        else:
            self.image_label.setText("Unable to load image")

        # Update info labels
        score = rec.get("score")
        person_name = rec.get("person_name", "")
        score_text = ""
        if not rec.get("face_path") or not os.path.exists(rec.get("face_path")):
            score_text = "Original photo (face thumbnail unavailable)"
        if score is not None:
            pct = int(round(float(score) * 100))
            bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
            quality_text = f"Quality: {bar} {pct}%"
            score_text = f"{score_text}  |  {quality_text}" if score_text else quality_text
        if person_name:
            score_text = f"{person_name}  |  {score_text}" if score_text else person_name
        self.score_label.setText(score_text)
        self.counter_label.setText(f"{self.index + 1} / {len(self.paths)}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._show_current()

    def prev(self):
        if not self.paths:
            return
        self.index = (self.index - 1) % len(self.paths)
        self._show_current()

    def next(self):
        if not self.paths:
            return
        self.index = (self.index + 1) % len(self.paths)
        self._show_current()

    def open_file(self):
        rec = self._current_record()
        target = rec.get("image") or rec.get("face_path")
        if target and os.path.exists(target):
            QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    def show_cooccurrence(self):
        """Show all faces detected in the same source photo as this face crop."""
        if fs is None:
            return
        rec = self._current_record()
        image_path = rec.get("image")
        if not image_path:
            QMessageBox.information(self, "No source", "Source image path not available for this face.")
            return
        try:
            result = fs.faces_in_image(image_path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return
        dlg = PhotoCoOccurrenceDialog(result, parent=self)
        dlg.exec()

    def view_in_photo(self):
        """Open the original full-size photo with this face's bounding box drawn on it."""
        rec = self._current_record()
        image_path = rec.get("image")
        if not image_path or not os.path.exists(image_path):
            QMessageBox.information(self, "No source",
                                    "Original photo path not available for this face crop.")
            return
        original_bbox = rec.get("bbox_original")
        bbox = original_bbox or rec.get("bbox")
        person_name = rec.get("person_name", "")
        dlg = FullPhotoViewer(
            image_path,
            bbox=bbox,
            bbox_is_original=bool(original_bbox),
            person_name=person_name,
            parent=self,
        )
        dlg.exec()

    def split_current_face(self):
        """Split the currently viewed face into its own new person."""
        if fs is None:
            return
        rec = self._current_record()
        face_id = rec.get("face_id")
        if face_id is None:
            QMessageBox.information(self, "Cannot split", "Face ID not available for this record.")
            return
        reply = QMessageBox.question(
            self, "Split Face",
            f"Move face #{face_id} to a new, separate person?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            res = fs.split_face_to_new_person(face_id)
            if res.get("error"):
                QMessageBox.critical(self, "Error", str(res["error"]))
            else:
                QMessageBox.information(self, "Done", f"Face moved to new Person {res['new_person_id']}.")
                self.accept()  # Close gallery so caller can refresh
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))





# ---------------------------------------------------------------------------
# FullPhotoViewer — original photo with bounding-box overlay
# ---------------------------------------------------------------------------

class FullPhotoViewer(QDialog):
    """Show the full-size source photo with the face bounding box drawn on it.

    The bbox stored in faces.json is in the coordinate space of the *resized*
    detection image (capped at MAX_IMAGE_SIDE px on the longest edge).
    We reverse that scale to paint on the original full-resolution photo.
    """

    def __init__(self, image_path: str, bbox=None, bbox_is_original: bool = False, person_name: str = "",
                 face_id=None, parent=None):
        super().__init__(parent)
        self.image_path   = image_path
        self.bbox         = bbox
        self.bbox_is_original = bbox_is_original
        self.person_name  = person_name or ""
        self._annotated   = None   # QPixmap with box drawn

        name_part = f" — {person_name}" if person_name else ""
        self.setWindowTitle(f"Photo{name_part}  |  {os.path.basename(image_path)}")
        self.resize(1020, 760)

        layout = QVBoxLayout(self)

        # Info bar
        info = QLabel(
            f"Person: {person_name or '(unknown)'}     "
            f"File: {image_path}"
        )
        info.setStyleSheet("color: #9da5b1; font-size: 12px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(400, 300)
        layout.addWidget(self.img_label, 1)

        controls = QHBoxLayout()
        btn_open = QPushButton("📂 Open Original in OS")
        btn_open.clicked.connect(self._open_original)
        controls.addWidget(btn_open)
        controls.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        controls.addWidget(btn_close)
        layout.addLayout(controls)

        self._build_annotated()
        self._render()

    # ------------------------------------------------------------------
    def _build_annotated(self):
        """Load original pixmap and paint the bounding-box overlay."""
        if not self.image_path or not os.path.exists(self.image_path):
            return

        pix = QPixmap(self.image_path)
        if pix.isNull():
            return

        if not self.bbox or len(self.bbox) < 4:
            self._annotated = pix
            return

        # ── convert bbox from model-space → original-image-space ──────────
        if self.bbox_is_original:
            x1, y1, x2, y2 = self.bbox[:4]
        else:
            # Legacy records only have detector-space coordinates.
            orig_w, orig_h = pix.width(), pix.height()
            longest = max(orig_w, orig_h)
            max_side = getattr(fs, "MAX_IMAGE_SIDE", 720) if fs else 720
            det_scale = (max_side / longest) if longest > max_side else 1.0
            x1 = self.bbox[0] / det_scale
            y1 = self.bbox[1] / det_scale
            x2 = self.bbox[2] / det_scale
            y2 = self.bbox[3] / det_scale
        bw = max(1, int(x2 - x1))
        bh = max(1, int(y2 - y1))

        # ── draw on a copy ─────────────────────────────────────────────────
        annotated = pix.copy()
        painter   = QPainter(annotated)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Dark shadow border (makes box visible on any background)
        painter.setPen(QPen(QColor(0, 0, 0, 160), 6))
        painter.drawRect(int(x1), int(y1), bw, bh)

        # Bright green highlight border
        painter.setPen(QPen(QColor(0, 220, 110), 3))
        painter.drawRect(int(x1), int(y1), bw, bh)

        # Corner accents (small filled squares at each corner for emphasis)
        corner = 12
        accent = QColor(0, 220, 110)
        for cx, cy in [(int(x1), int(y1)), (int(x2) - corner, int(y1)),
                       (int(x1), int(y2) - corner), (int(x2) - corner, int(y2) - corner)]:
            painter.fillRect(cx, cy, corner, corner, accent)

        # Name label badge above the box
        label_text = self.person_name or "Face"
        font = QFont("Segoe UI", 13, QFont.Bold)
        painter.setFont(font)
        fm     = painter.fontMetrics()
        pad_x, pad_y = 10, 5
        txt_w  = fm.horizontalAdvance(label_text) + pad_x * 2
        txt_h  = fm.height() + pad_y * 2
        lx     = int(x1)
        ly     = max(0, int(y1) - txt_h - 2)

        # Badge background
        painter.fillRect(lx, ly, txt_w, txt_h, QColor(0, 180, 90, 220))
        # Badge text
        painter.setPen(QPen(Qt.white))
        painter.drawText(lx + pad_x, ly + pad_y + fm.ascent(), label_text)

        painter.end()
        self._annotated = annotated

    # ------------------------------------------------------------------
    def _render(self):
        if self._annotated is None or self._annotated.isNull():
            self.img_label.setText("⚠  Could not load image.")
            return
        size = self.img_label.size()
        if size.width() < 20 or size.height() < 20:
            size = self.size() * 0.88
        scaled = self._annotated.scaled(
            size * 0.97, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.img_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def _open_original(self):
        if self.image_path and os.path.exists(self.image_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.image_path))



# ---------------------------------------------------------------------------
# AutoMergeDialog — find + review + merge similar person clusters in bulk
# ---------------------------------------------------------------------------

class AutoMergeDialog(QDialog):
    """Find person clusters that are likely the same real person and merge them."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤝 Auto-Merge Similar People")
        self.resize(700, 540)
        self._candidates = []          # latest results from find_merge_candidates
        self._row_widgets = []         # (checkbox, pair_dict) per row

        layout = QVBoxLayout(self)

        # ── threshold row ──────────────────────────────────────────────
        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Similarity threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(60, 95)   # maps to 0.60 – 0.95
        self.threshold_slider.setValue(82)        # default 0.82
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setFixedWidth(220)
        thr_row.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("0.82")
        self.threshold_label.setFixedWidth(34)
        thr_row.addWidget(self.threshold_label)
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(f"{v/100:.2f}")
        )
        btn_preview = QPushButton("🔍  Preview")
        btn_preview.setDefault(False)
        btn_preview.clicked.connect(self._run_preview)
        thr_row.addWidget(btn_preview)
        thr_row.addStretch()
        layout.addLayout(thr_row)

        # ── status label ───────────────────────────────────────────────
        self.status_lbl = QLabel("Click Preview to find candidate pairs.")
        self.status_lbl.setStyleSheet("color: #888; padding: 4px;")
        layout.addWidget(self.status_lbl)

        # ── scrollable pair list ───────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.pair_container = QWidget()
        self.pair_layout = QVBoxLayout(self.pair_container)
        self.pair_layout.setAlignment(Qt.AlignTop)
        self.pair_layout.setSpacing(4)
        scroll.setWidget(self.pair_container)
        layout.addWidget(scroll, stretch=1)

        # ── bottom buttons ─────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_merge_all = QPushButton("✅  Merge All")
        self.btn_merge_all.clicked.connect(self._merge_all)
        self.btn_merge_all.setEnabled(False)
        btn_row.addWidget(self.btn_merge_all)

        self.btn_merge_checked = QPushButton("✅  Merge Checked")
        self.btn_merge_checked.clicked.connect(self._merge_checked)
        self.btn_merge_checked.setEnabled(False)
        btn_row.addWidget(self.btn_merge_checked)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        # Run a preview automatically on open
        self._run_preview()

    # ------------------------------------------------------------------
    def _run_preview(self):
        if fs is None:
            return
        thr = self.threshold_slider.value() / 100.0
        self.status_lbl.setText("⏳ Computing…")
        self.status_lbl.repaint()
        try:
            result = fs.find_merge_candidates(thr)
        except Exception as exc:
            self.status_lbl.setText(f"⚠  Error: {exc}")
            return

        self._candidates = result
        self._rebuild_list()

    def _rebuild_list(self):
        """Clear and repopulate the pair list from self._candidates."""
        # Remove old rows
        while self.pair_layout.count():
            item = self.pair_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._row_widgets.clear()

        if not self._candidates:
            self.status_lbl.setText("No candidates found. Try lowering the threshold.")
            self.btn_merge_all.setEnabled(False)
            self.btn_merge_checked.setEnabled(False)
            return

        self.status_lbl.setText(
            f"Found {len(self._candidates)} candidate pair(s). "
            "High-confidence pairs (≥ 0.90) are pre-checked."
        )

        for pair in self._candidates:
            row = self._make_pair_row(pair)
            self.pair_layout.addWidget(row)

        self.btn_merge_all.setEnabled(True)
        self.btn_merge_checked.setEnabled(True)

    def _make_pair_row(self, pair):
        """Build one row widget for a candidate pair."""
        sim = pair["similarity"]
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        hl = QHBoxLayout(frame)
        hl.setContentsMargins(6, 4, 6, 4)

        # Checkbox (pre-checked for high-confidence matches)
        cb = QCheckBox()
        cb.setChecked(sim >= 0.90)
        hl.addWidget(cb)
        self._row_widgets.append((cb, pair))

        # Thumbnail A
        thumb_a = self._load_thumb(pair.get("thumb_a"))
        hl.addWidget(thumb_a)

        # Person A info
        lbl_a = QLabel(f"<b>{pair['name_a']}</b><br><small>{pair['count_a']} face(s)</small>")
        lbl_a.setAlignment(Qt.AlignCenter)
        hl.addWidget(lbl_a, stretch=1)

        # Similarity badge
        pct = int(sim * 100)
        badge_color = "#27ae60" if sim >= 0.90 else "#e67e22" if sim >= 0.80 else "#c0392b"
        badge = QLabel(f"<b style='color:{badge_color};'>{pct}%<br>similar</b>")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedWidth(62)
        hl.addWidget(badge)

        # Person B info
        lbl_b = QLabel(f"<b>{pair['name_b']}</b><br><small>{pair['count_b']} face(s)</small>")
        lbl_b.setAlignment(Qt.AlignCenter)
        hl.addWidget(lbl_b, stretch=1)

        # Thumbnail B
        thumb_b = self._load_thumb(pair.get("thumb_b"))
        hl.addWidget(thumb_b)

        # Which person to keep note
        keep_note = QLabel("<small style='color:#888;'>A kept</small>")
        keep_note.setFixedWidth(46)
        hl.addWidget(keep_note)

        return frame

    @staticmethod
    def _load_thumb(path):
        lbl = QLabel()
        lbl.setFixedSize(52, 52)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("background:#222; border-radius:4px;")
        if path and os.path.exists(path):
            pix = QPixmap(path).scaled(52, 52, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            # centre-crop to 52×52
            x = (pix.width() - 52) // 2
            y = (pix.height() - 52) // 2
            lbl.setPixmap(pix.copy(x, y, 52, 52))
        else:
            lbl.setText("?")
        return lbl

    # ------------------------------------------------------------------
    def _collect_pairs(self, checked_only):
        """Return list of {keep_id, merge_id} to execute."""
        pairs = []
        for cb, pair in self._row_widgets:
            if checked_only and not cb.isChecked():
                continue
            # person_a is always kept (higher face count — tie-break by id)
            keep_id  = pair["person_a"]
            merge_id = pair["person_b"]
            if pair["count_b"] > pair["count_a"]:
                keep_id, merge_id = merge_id, keep_id
            pairs.append({"keep_id": keep_id, "merge_id": merge_id})
        return pairs

    def _execute_merges(self, pairs):
        if not pairs:
            QMessageBox.information(self, "Nothing to merge", "No pairs selected.")
            return

        n = len(pairs)
        reply = QMessageBox.question(
            self, "Confirm merge",
            f"Merge {n} pair(s)? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            result = fs.auto_merge_all(pairs)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        merged = result.get("merged", 0)
        errors = result.get("errors", [])
        msg = f"Merged {merged} pair(s) successfully."
        if errors:
            msg += f"\n\n{len(errors)} error(s):\n" + "\n".join(errors[:5])
            QMessageBox.warning(self, "Merge complete", msg)
        else:
            QMessageBox.information(self, "Merge complete", msg)

        self.accept()   # triggers load_library() in AppWindow

    def _merge_all(self):
        self._execute_merges(self._collect_pairs(checked_only=False))

    def _merge_checked(self):
        self._execute_merges(self._collect_pairs(checked_only=True))


# ---------------------------------------------------------------------------
# FaceSearchDialog — pick any image, find the matching person in the library
# ---------------------------------------------------------------------------

class FaceSearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Search by Face")
        self.resize(500, 300)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select a reference photo containing a face to search:"))

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Image path…")
        path_row.addWidget(self.path_edit)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        path_row.addWidget(btn_browse)
        layout.addLayout(path_row)

        btn_search = QPushButton("🔍  Search")
        btn_search.clicked.connect(self._search)
        layout.addWidget(btn_search)

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        self.btn_view = QPushButton("View Person's Gallery")
        self.btn_view.setVisible(False)
        self.btn_view.clicked.connect(self._view_result)
        layout.addWidget(self.btn_view)

        self._result = None

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)"
        )
        if path:
            self.path_edit.setText(path)

    def _search(self):
        path = self.path_edit.text().strip()
        if not path or not os.path.exists(path):
            self.result_label.setText("❌  Please select a valid image file.")
            return
        self.result_label.setText("Searching…")
        QApplication.processEvents()
        try:
            res = fs.search_face(path)
        except Exception as exc:
            self.result_label.setText(f"❌  Error: {exc}")
            return

        if res.get("error") == "invalid_image":
            self.result_label.setText("❌  Could not read the image.")
        elif res.get("error") == "no_face":
            self.result_label.setText("❌  No face detected in the selected photo.")
        elif res.get("result") == "unknown":
            self.result_label.setText("🤷  No matching person found in the library.")
            self.btn_view.setVisible(False)
        else:
            pid = res.get("person_id")
            person_rec = fs.people_by_id.get(pid, {})
            name = person_rec.get("name") or f"Person {pid}"
            face = res.get("face", {})
            score = face.get("score")
            score_str = f"  (score {score:.2f})" if score else ""
            self.result_label.setText(f"✅  Match: {name}{score_str}")
            self._result = res
            self.btn_view.setVisible(True)

    def _view_result(self):
        if not self._result:
            return
        pid = self._result.get("person_id")
        person_rec = fs.people_by_id.get(pid, {})
        name = person_rec.get("name") or f"Person {pid}"
        # Build face records for this person from faces_by_id
        records = []
        for fid in person_rec.get("faces", []):
            face = fs.faces_by_id.get(int(fid))
            if face:
                fp = face.get("face_path")
                if fp and os.path.exists(fp):
                    records.append({
                        "face_id": fid,
                        "face_path": fp,
                        "image": face.get("image"),
                        "score": face.get("score"),
                    })
        if not records:
            QMessageBox.information(self, "No faces", "No thumbnails found for this person.")
            return
        viewer = ImageViewer(records, title=f"Gallery — {name}", parent=self)
        viewer.exec()


# ---------------------------------------------------------------------------
# ReclusterDialog — adjust threshold and re-run clustering in background
# ---------------------------------------------------------------------------

class ReclusterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Re-cluster Faces")
        self.resize(420, 220)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Adjust the similarity threshold and re-assign all faces to people.\n"
            "Lower = more clusters (stricter), Higher = fewer clusters (looser).\n"
            "⚠ Requires embeddings stored during processing (re-process if needed)."
        ))

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Threshold:"))
        self.thr_slider = QSlider(Qt.Horizontal)
        self.thr_slider.setRange(20, 90)       # 0.20 – 0.90
        self.thr_slider.setValue(45)            # default 0.45
        self.thr_slider.setTickInterval(5)
        self.thr_slider.setTickPosition(QSlider.TicksBelow)
        self.thr_slider.valueChanged.connect(self._update_label)
        thr_row.addWidget(self.thr_slider)
        self.thr_label = QLabel("0.45")
        self.thr_label.setMinimumWidth(36)
        thr_row.addWidget(self.thr_label)
        layout.addLayout(thr_row)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        btns = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run Re-cluster")
        self.btn_run.clicked.connect(self._run)
        btns.addWidget(self.btn_run)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

    def _update_label(self, value):
        self.thr_label.setText(f"{value / 100:.2f}")

    def _run(self):
        thr = self.thr_slider.value() / 100.0
        self.btn_run.setEnabled(False)
        self.status_label.setText("Re-clustering… please wait.")
        QApplication.processEvents()
        try:
            res = fs.recluster(thr)
        except Exception as exc:
            self.status_label.setText(f"❌  {exc}")
            self.btn_run.setEnabled(True)
            return
        if res.get("error"):
            self.status_label.setText(f"❌  {res['error']}\n{res.get('hint', '')}")
            self.btn_run.setEnabled(True)
            return
        n_people = res.get("people", "?")
        n_faces = res.get("faces_clustered", "?")
        self.status_label.setText(
            f"✅  Done — {n_faces} faces → {n_people} people  (threshold {thr:.2f})"
        )
        self.accept()


# ---------------------------------------------------------------------------
# DuplicatesDialog — list photos where the same person appears multiple times
# ---------------------------------------------------------------------------

class DuplicatesDialog(QDialog):
    def __init__(self, duplicates, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Duplicate Face Detections")
        self.resize(700, 480)
        self.duplicates = duplicates

        layout = QVBoxLayout(self)
        if not duplicates:
            layout.addWidget(QLabel("✅  No duplicates found — every person appears at most once per photo."))
            layout.addWidget(self._close_btn())
            return

        layout.addWidget(QLabel(
            f"Found {len(duplicates)} photo(s) where the same person appears more than once.\n"
            "This may indicate mis-clustering. Use 'Split to New Person' in the gallery to fix."
        ))

        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(80, 80))
        self.list_widget.itemDoubleClicked.connect(self._on_item_activated)
        layout.addWidget(self.list_widget)

        for dup in duplicates:
            name = dup.get("person_name", "Unknown")
            img = dup.get("image", "")
            count = dup.get("face_count", 0)
            text = f"{name}  ×{count}  |  {os.path.basename(img)}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, dup)
            item.setToolTip(img)
            # Show first face thumbnail
            fps = dup.get("face_paths", [])
            if fps and os.path.exists(fps[0]):
                pix = QPixmap(fps[0]).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                item.setIcon(QIcon(pix))
            self.list_widget.addItem(item)

        layout.addWidget(QLabel("Double-click a row to view the face crops."))
        layout.addWidget(self._close_btn())

    def _close_btn(self):
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        return btn

    def _on_item_activated(self, item):
        dup = item.data(Qt.UserRole)
        if not dup:
            return
        fps = dup.get("face_paths", [])
        name = dup.get("person_name", "")
        records = [{"face_path": fp, "image": dup.get("image"), "person_name": name}
                   for fp in fps if fp and os.path.exists(fp)]
        if not records:
            QMessageBox.information(self, "No faces", "No face thumbnails found.")
            return
        viewer = ImageViewer(records, title=f"Duplicates — {name}", parent=self)
        viewer.exec()


# ---------------------------------------------------------------------------
# PhotoCoOccurrenceDialog — all faces in the same source photo
# ---------------------------------------------------------------------------

class PhotoCoOccurrenceDialog(QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        faces = result.get("faces", [])
        image_path = result.get("image", "")
        self.setWindowTitle(f"Who's in: {os.path.basename(image_path)}")
        self.resize(700, 440)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"Photo: {image_path}\n"
            f"{len(faces)} face(s) detected."
        ))

        if not faces:
            layout.addWidget(QLabel("No faces found for this photo."))
        else:
            grid_widget = QListWidget()
            grid_widget.setViewMode(QListWidget.IconMode)
            grid_widget.setIconSize(QSize(110, 110))
            grid_widget.setResizeMode(QListWidget.Adjust)
            grid_widget.setMovement(QListWidget.Static)
            for face in faces:
                name = face.get("person_name", "Unknown")
                score = face.get("score")
                score_str = f" ({score:.0%})" if score else ""
                item = QListWidgetItem(f"{name}{score_str}")
                fp = face.get("face_path")
                if fp and os.path.exists(fp):
                    pix = QPixmap(fp).scaled(110, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    item.setIcon(QIcon(pix))
                item.setData(Qt.UserRole, face)
                grid_widget.addItem(item)
            layout.addWidget(grid_widget)

        btn_open = QPushButton("📂 Open Original Photo")
        btn_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(image_path))
                                 if image_path and os.path.exists(image_path) else None)
        layout.addWidget(btn_open)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)


# ---------------------------------------------------------------------------
# NoFaceCleanupDialog — review-first, non-destructive photo organization
# ---------------------------------------------------------------------------

class NoFaceCleanupDialog(QDialog):
    """Let the user explicitly review zero-face scan results before moving them."""

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review No-Face Photos")
        self.resize(820, 560)
        self._items = result.get("items", [])

        layout = QVBoxLayout(self)
        count = len(self._items)
        unavailable = int(result.get("unavailable", 0) or 0)
        intro = (
            f"Found {count} successfully scanned photo(s) with no detected face. "
            "Choose exactly what to move; nothing is deleted."
        )
        if unavailable:
            intro += f" {unavailable} missing file(s) were not included."
        label = QLabel(intro)
        label.setWordWrap(True)
        layout.addWidget(label)

        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(72, 72))
        self.list_widget.setUniformItemSizes(True)
        layout.addWidget(self.list_widget, 1)

        self.list_widget.setUpdatesEnabled(False)
        try:
            for record in self._items:
                path = record.get("image", "")
                size_mb = record.get("size", 0) / (1024 * 1024)
                text = (
                    f"{os.path.basename(path)}\n"
                    f"{record.get('date', 'unknown')}  •  {size_mb:.1f} MB\n"
                    f"{record.get('folder', '')}"
                )
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, record)
                item.setCheckState(Qt.Unchecked)
                item.setToolTip(path)
                if path and os.path.isfile(path):
                    pix = QPixmap(path)
                    if not pix.isNull():
                        item.setIcon(QIcon(pix.scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
                self.list_widget.addItem(item)
        finally:
            self.list_widget.setUpdatesEnabled(True)

        button_row = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self._select_all)
        self.btn_select_all.setEnabled(bool(self._items))
        button_row.addWidget(self.btn_select_all)

        self.btn_move = QPushButton("Move Selected…")
        self.btn_move.clicked.connect(self._move_selected)
        self.btn_move.setEnabled(bool(self._items))
        button_row.addWidget(self.btn_move)
        button_row.addStretch()

        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        button_row.addWidget(close)
        layout.addLayout(button_row)

    def _select_all(self):
        for index in range(self.list_widget.count()):
            self.list_widget.item(index).setCheckState(Qt.Checked)

    def _move_selected(self):
        selected = []
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.checkState() == Qt.Checked:
                record = item.data(Qt.UserRole) or {}
                if record.get("image"):
                    selected.append(record["image"])
        if not selected:
            QMessageBox.information(self, "No photos selected", "Tick one or more photos to move.")
            return

        destination = QFileDialog.getExistingDirectory(
            self, "Choose a destination folder for the selected photos"
        )
        if not destination:
            return
        reply = QMessageBox.question(
            self,
            "Confirm move",
            f"Move {len(selected)} selected photo(s) to:\n{destination}\n\n"
            "Files are not deleted. Existing filenames will be preserved by adding a number.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            result = fs.move_no_face_photos(selected, destination)
        except Exception as exc:
            QMessageBox.critical(self, "Move failed", str(exc))
            return

        moved = result.get("moved", [])
        errors = result.get("errors", [])
        if errors:
            QMessageBox.warning(
                self,
                "Move completed with warnings",
                f"Moved {len(moved)} photo(s).\n\n" + "\n".join(errors[:8]),
            )
        else:
            QMessageBox.information(self, "Move complete", f"Moved {len(moved)} photo(s).")
        if moved:
            self.accept()


class CleanupSuggestionsDialog(QDialog):
    """Background analysis plus explicit review for non-face cleanup signals."""

    CATEGORY_LABELS = {
        "exact_duplicate": "Exact duplicate",
        "screenshot": "Screenshot",
        "likely_forward": "Likely forward",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review Cleanup Suggestions")
        self.resize(860, 580)
        self._result = None
        self._worker_result = None
        self._worker = None

        layout = QVBoxLayout(self)
        self.status_label = QLabel(
            "Analysing scanned photos for exact duplicates, screenshots, and likely social-media forwards…"
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(72, 72))
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.itemDoubleClicked.connect(self._open_selected)
        layout.addWidget(self.list_widget, 1)

        buttons = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self._select_all)
        self.btn_select_all.setEnabled(False)
        buttons.addWidget(self.btn_select_all)

        self.btn_move = QPushButton("Move Selected…")
        self.btn_move.clicked.connect(self._move_selected)
        self.btn_move.setEnabled(False)
        buttons.addWidget(self.btn_move)
        buttons.addStretch()

        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self._start_analysis()

    def _start_analysis(self):
        def worker():
            try:
                self._worker_result = {"result": fs.cleanup_suggestions()}
            except Exception as exc:
                self._worker_result = {"error": str(exc)}

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._poll_analysis)
        self._timer.start()

    def _poll_analysis(self):
        if self._worker and self._worker.is_alive():
            return
        self._timer.stop()
        outcome = self._worker_result or {"error": "Analysis did not return a result."}
        if outcome.get("error"):
            self.status_label.setText(f"Could not analyse cleanup suggestions: {outcome['error']}")
            return

        self._result = outcome["result"]
        items = self._result.get("items", [])
        if not items:
            self.status_label.setText(
                f"No suggestions found after checking {self._result.get('photos_checked', 0)} scanned photo(s)."
            )
            return

        self.status_label.setText(
            f"Found {len(items)} suggestion(s): "
            f"{self._result.get('duplicate_groups', 0)} exact duplicate group(s). "
            "Nothing is selected or moved automatically; double-click a row to open it."
        )
        self.list_widget.setUpdatesEnabled(False)
        try:
            for record in items:
                path = record.get("image", "")
                categories = ", ".join(
                    self.CATEGORY_LABELS.get(category, category)
                    for category in record.get("categories", [])
                )
                reasons = " • ".join(record.get("reasons", []))
                size_mb = record.get("size", 0) / (1024 * 1024)
                text = (
                    f"{categories}: {os.path.basename(path)}\n"
                    f"{reasons}\n"
                    f"{record.get('date', 'unknown')}  •  {size_mb:.1f} MB  •  {record.get('folder', '')}"
                )
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, record)
                item.setToolTip(path)
                item.setCheckState(Qt.Unchecked)
                if path and os.path.isfile(path):
                    pix = QPixmap(path)
                    if not pix.isNull():
                        item.setIcon(QIcon(pix.scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
                self.list_widget.addItem(item)
        finally:
            self.list_widget.setUpdatesEnabled(True)

        self.btn_select_all.setEnabled(True)
        self.btn_move.setEnabled(True)

    def _select_all(self):
        for index in range(self.list_widget.count()):
            self.list_widget.item(index).setCheckState(Qt.Checked)

    def _open_selected(self, item):
        record = item.data(Qt.UserRole) or {}
        path = record.get("image")
        if path and os.path.isfile(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _move_selected(self):
        selected = []
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.checkState() == Qt.Checked:
                record = item.data(Qt.UserRole) or {}
                if record.get("image"):
                    selected.append(record["image"])
        if not selected:
            QMessageBox.information(self, "No photos selected", "Tick one or more reviewed photos to move.")
            return

        destination = QFileDialog.getExistingDirectory(
            self, "Choose a destination folder for the reviewed photos"
        )
        if not destination:
            return
        reply = QMessageBox.question(
            self,
            "Confirm move",
            f"Move {len(selected)} reviewed photo(s) to:\n{destination}\n\n"
            "Files are not deleted. The app will preserve library links and never overwrite a filename.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            result = fs.move_reviewed_photos(selected, destination)
        except Exception as exc:
            QMessageBox.critical(self, "Move failed", str(exc))
            return

        moved = result.get("moved", [])
        errors = result.get("errors", [])
        if errors:
            QMessageBox.warning(
                self,
                "Move completed with warnings",
                f"Moved {len(moved)} photo(s).\n\n" + "\n".join(errors[:8]),
            )
        else:
            QMessageBox.information(self, "Move complete", f"Moved {len(moved)} photo(s).")
        if moved:
            self.accept()


def main():
    app = QApplication([])
    window = AppWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

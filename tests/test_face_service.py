import csv
import json
import os
import tempfile
import unittest

from resources import face_service as fs


class FaceServiceRegressionTests(unittest.TestCase):
    def setUp(self):
        # The desktop sandbox does not grant the test interpreter access to
        # Windows' global temp directory. Keep ephemeral fixtures inside the
        # repository, where both the app and its test runner can write.
        self.temp_dir = tempfile.TemporaryDirectory(dir=os.path.dirname(__file__))
        self.original_paths = (
            fs.FACES_DB,
            fs.PEOPLE_DB,
            fs.PROCESSED_DB,
            fs.TIMELINE_DB,
            fs.CLEANUP_LOG,
        )
        self.original_state = (
            fs.faces_db,
            fs.people_db,
            fs.processed_db,
            fs.timeline_db,
            fs.faces_by_id,
            fs.people_by_id,
        )
        fs.FACES_DB = os.path.join(self.temp_dir.name, "faces.json")
        fs.PEOPLE_DB = os.path.join(self.temp_dir.name, "people.json")
        fs.PROCESSED_DB = os.path.join(self.temp_dir.name, "processed.json")
        fs.TIMELINE_DB = os.path.join(self.temp_dir.name, "timeline.json")
        fs.CLEANUP_LOG = os.path.join(self.temp_dir.name, "cleanup_moves.jsonl")

    def tearDown(self):
        (
            fs.FACES_DB,
            fs.PEOPLE_DB,
            fs.PROCESSED_DB,
            fs.TIMELINE_DB,
            fs.CLEANUP_LOG,
        ) = self.original_paths
        (
            fs.faces_db,
            fs.people_db,
            fs.processed_db,
            fs.timeline_db,
            fs.faces_by_id,
            fs.people_by_id,
        ) = self.original_state
        self.temp_dir.cleanup()

    def test_signature_prefers_nanosecond_precision_and_supports_legacy_records(self):
        current = {"size": 10, "mtime": 100, "mtime_ns": 100_000_000_123}
        self.assertTrue(fs.same_signature({"size": 10, "mtime": 100}, current))
        self.assertTrue(fs.same_signature({"size": 10, "mtime_ns": 100_000_000_123}, current))
        self.assertFalse(fs.same_signature({"size": 10, "mtime_ns": 100_000_000_122}, current))

    def test_invalid_similarity_thresholds_are_rejected_before_ml_initialization(self):
        original_faces = fs.faces_db
        try:
            fs.faces_db = [{"face_id": 0}]
            self.assertEqual(fs.recluster(float("nan"))["error"], "invalid_threshold")
            with self.assertRaises(ValueError):
                fs.find_merge_candidates(float("inf"))
        finally:
            fs.faces_db = original_faces

    def test_batch_merge_writes_final_state_once_and_updates_face_membership(self):
        fs.faces_db = [
            {"face_id": 1, "person_id": 1},
            {"face_id": 2, "person_id": 2},
            {"face_id": 3, "person_id": 3},
        ]
        fs.people_db = [
            {"id": 1, "faces": [1]},
            {"id": 2, "faces": [2]},
            {"id": 3, "faces": [3]},
        ]
        fs.processed_db = {}
        fs.faces_by_id = {face["face_id"]: face for face in fs.faces_db}
        fs.people_by_id = {person["id"]: person for person in fs.people_db}

        result = fs.auto_merge_all([
            {"keep_id": 1, "merge_id": 2},
            {"keep_id": 2, "merge_id": 3},
        ])

        self.assertEqual(result, {"merged": 2, "errors": []})
        self.assertEqual(fs.people_by_id[1]["faces"], [1, 2, 3])
        self.assertEqual(fs.faces_by_id[2]["person_id"], 1)
        self.assertEqual(fs.faces_by_id[3]["person_id"], 1)
        with open(fs.FACES_DB, encoding="utf-8") as handle:
            self.assertEqual(len(json.load(handle)), 3)

    def test_no_face_cleanup_moves_only_reviewable_photos_and_updates_metadata(self):
        source_folder = os.path.join(self.temp_dir.name, "source")
        destination_folder = os.path.join(self.temp_dir.name, "review")
        os.makedirs(source_folder)
        os.makedirs(destination_folder)
        source = os.path.join(source_folder, "no-face.jpg")
        with open(source, "wb") as handle:
            handle.write(b"fixture image")
        source = fs.normalize(source)

        fs.faces_db = []
        fs.people_db = []
        fs.processed_db = {
            source: {**fs.image_signature(source), "status": "ok", "faces": 0, "date": "2026-10"}
        }
        fs.timeline_db = {source: {**fs.image_signature(source), "date": "2026-10"}}
        fs.faces_by_id = {}
        fs.people_by_id = {}
        fs.save_faces()
        fs.save_people()
        fs.save_processed()
        fs.save_timeline()

        review = fs.no_face_photos()
        self.assertEqual(review["count"], 1)
        result = fs.move_no_face_photos([source], destination_folder)

        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["moved"]), 1)
        target = result["moved"][0]["destination"]
        self.assertFalse(os.path.exists(source))
        self.assertTrue(os.path.isfile(target))
        self.assertEqual(fs.processed_db[target]["cleanup_status"], "moved_no_face")
        self.assertEqual(fs.no_face_photos()["count"], 0)
        self.assertTrue(os.path.isfile(fs.CLEANUP_LOG))

    def test_cleanup_inbox_finds_explainable_signals_and_preserves_face_links_on_move(self):
        source_folder = os.path.join(self.temp_dir.name, "source")
        destination_folder = os.path.join(self.temp_dir.name, "review")
        os.makedirs(source_folder)
        os.makedirs(destination_folder)
        screenshot = fs.normalize(os.path.join(source_folder, "Screenshot_2026.jpg"))
        forwarded = fs.normalize(os.path.join(source_folder, "WhatsApp Image 2026.jpg"))
        normal = fs.normalize(os.path.join(source_folder, "camera.jpg"))
        for path, content in ((screenshot, b"same-image"), (forwarded, b"same-image"), (normal, b"other-image")):
            with open(path, "wb") as handle:
                handle.write(content)

        fs.faces_db = [{"face_id": 1, "person_id": 1, "image": forwarded, "album": source_folder}]
        fs.people_db = [{"id": 1, "faces": [1]}]
        fs.processed_db = {
            path: {**fs.image_signature(path), "status": "ok", "faces": 1 if path == forwarded else 0, "date": "2026-10"}
            for path in (screenshot, forwarded, normal)
        }
        fs.timeline_db = {
            path: {**fs.image_signature(path), "date": "2026-10"}
            for path in (screenshot, forwarded, normal)
        }
        fs.faces_by_id = {1: fs.faces_db[0]}
        fs.people_by_id = {1: fs.people_db[0]}
        fs.save_faces()
        fs.save_people()
        fs.save_processed()
        fs.save_timeline()

        inbox = fs.cleanup_suggestions()
        by_path = {item["image"]: item for item in inbox["items"]}
        self.assertEqual(inbox["photos_hashed"], 2)
        self.assertIn("screenshot", by_path[screenshot]["categories"])
        self.assertIn("likely_forward", by_path[forwarded]["categories"])
        self.assertIn("exact_duplicate", by_path[screenshot]["categories"])
        self.assertIn("exact_duplicate", by_path[forwarded]["categories"])
        self.assertEqual(fs.cleanup_suggestions()["photos_hashed"], 0)

        result = fs.move_reviewed_photos([forwarded], destination_folder)
        self.assertEqual(result["errors"], [])
        target = result["moved"][0]["destination"]
        self.assertTrue(os.path.isfile(target))
        self.assertEqual(fs.faces_db[0]["image"], target)
        self.assertIn(target, fs.processed_db)
        self.assertIn(target, fs.timeline_db)

    def test_library_report_includes_full_paths_timeline_and_missing_files(self):
        source = fs.normalize(os.path.join(self.temp_dir.name, "photo.jpg"))
        missing = fs.normalize(os.path.join(self.temp_dir.name, "missing.jpg"))
        with open(source, "wb") as handle:
            handle.write(b"photo")

        fs.faces_db = [{"face_id": 1, "person_id": 1, "image": source, "date": "2026-10"}]
        fs.people_db = [{"id": 1, "name": "Asha", "faces": [1]}]
        fs.processed_db = {
            source: {**fs.image_signature(source), "status": "ok", "faces": 1, "date": "2026-10"},
            missing: {"size": 10, "mtime": 1, "mtime_ns": 1, "status": "ok", "faces": 0, "date": "2026-09"},
        }
        fs.timeline_db = {
            source: {**fs.image_signature(source), "date": "2026-10"},
            missing: {"size": 10, "mtime": 1, "mtime_ns": 1, "date": "2026-09"},
        }
        fs.faces_by_id = {1: fs.faces_db[0]}
        fs.people_by_id = {1: fs.people_db[0]}
        fs.save_faces()
        fs.save_people()
        fs.save_processed()
        fs.save_timeline()

        report_path = os.path.join(self.temp_dir.name, "library.csv")
        result = fs.export_library_report(report_path)

        self.assertEqual(result["total_files"], 2)
        self.assertEqual(result["missing_files"], 1)
        with open(report_path, encoding="utf-8-sig", newline="") as handle:
            rows = {row["full_path"]: row for row in csv.DictReader(handle)}
        self.assertEqual(rows[source]["timeline_month"], "2026-10")
        self.assertEqual(rows[source]["people"], "Asha")
        self.assertEqual(rows[missing]["file_exists"], "no")
        self.assertEqual(rows[missing]["flags"], "no_face")


if __name__ == "__main__":
    unittest.main()

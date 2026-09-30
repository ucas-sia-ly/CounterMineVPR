"""Pillow/CPU-only checks for candidate visualization and metadata alignment."""

import csv
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools/06_visualize_rgb_candidates.py"
SPEC = importlib.util.spec_from_file_location("rgb_visualizer", SCRIPT_PATH)
visualizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(visualizer)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TestRGBVisualization(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "dataset"
        self.dataset_root.mkdir()
        self.manifest_path = self.root / "manifest.csv"
        self.candidates_path = self.root / "candidates.csv"
        self.output = self.root / "output" / "pairs.png"
        self.manifest = []
        for index, color in enumerate(("red", "green", "blue")):
            path = self.dataset_root / f"image-{index}.jpg"
            with Image.new("RGB", (18 + index, 12), color) as image:
                image.save(path)
            self.manifest.append({
                "row_index": index, "image_id": f"id-{index}",
                "relative_path": path.name, "place_uid": f"Boston:{index}",
            })
        write_csv(self.manifest_path, self.manifest)
        self.candidates = [self.candidate(2, 0, 0.8), self.candidate(0, 2, 0.9), self.candidate(0, 1, 0.9)]
        write_csv(self.candidates_path, self.candidates)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def candidate(query, negative, similarity):
        return {
            "query_row_index": query, "negative_row_index": negative,
            "query_image_id": f"id-{query}", "negative_image_id": f"id-{negative}",
            "query_place_uid": f"Boston:{query}", "negative_place_uid": f"Boston:{negative}",
            "rank": 1, "similarity": similarity, "geo_distance_m": 125.5,
        }

    def test_streamed_selection_uses_deterministic_ties(self):
        selected = visualizer.select_top_candidates(self.candidates_path, top_n=2)
        self.assertEqual([(row["query_row_index"], row["negative_row_index"]) for row in selected], [(0, 1), (0, 2)])

    def test_contact_sheet_dimensions_and_sources_unchanged(self):
        checksums = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.dataset_root.iterdir()}
        output = visualizer.create_contact_sheet(
            self.manifest_path, self.candidates_path, self.dataset_root,
            self.output, top_n=2, image_width=180, image_height=100,
        )
        self.assertEqual(output, self.output)
        with Image.open(output) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (408, 396))
        self.assertEqual(checksums, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in checksums})

    def test_metadata_mismatch_fails_before_output(self):
        self.candidates[1]["query_image_id"] = "misaligned"
        write_csv(self.candidates_path, self.candidates)
        with self.assertRaisesRegex(ValueError, "image_id does not match manifest"):
            visualizer.create_contact_sheet(self.manifest_path, self.candidates_path, self.dataset_root, self.output)
        self.assertFalse(self.output.exists())

    def test_manifest_reordering_is_rejected(self):
        write_csv(self.manifest_path, list(reversed(self.manifest)))
        with self.assertRaisesRegex(ValueError, "row_index must equal range"):
            visualizer.load_manifest(self.manifest_path)

    def test_missing_source_image_has_clear_error(self):
        (self.dataset_root / "image-0.jpg").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "source image is missing"):
            visualizer.create_contact_sheet(self.manifest_path, self.candidates_path, self.dataset_root, self.output)
        self.assertFalse(self.output.exists())

    def test_output_cannot_overwrite_unselected_source(self):
        source = self.dataset_root / "image-2.jpg"
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "must not overwrite a source image"):
            visualizer.create_contact_sheet(
                self.manifest_path, self.candidates_path, self.dataset_root,
                source, top_n=1,
            )
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), checksum)


if __name__ == "__main__":
    unittest.main()

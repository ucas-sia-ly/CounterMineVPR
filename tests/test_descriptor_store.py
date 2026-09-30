"""CPU-only checks for ordered manifest paths and streaming descriptor storage."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from countermine.mining.descriptor_store import (
    load_manifest_paths,
    verify_descriptor_memmap,
    write_descriptor_memmap,
)


class DescriptorStoreTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write_manifest(self, rows):
        path = self.root / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=("row_index", "relative_path"))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_manifest_paths_follow_row_index_order(self):
        manifest = self.write_manifest(
            [
                {"row_index": 2, "relative_path": "Images/Boston/third.jpg"},
                {"row_index": 0, "relative_path": "Images/London/first.jpg"},
                {"row_index": 1, "relative_path": "Images/London/second.jpg"},
            ]
        )
        self.assertEqual(
            load_manifest_paths(manifest),
            [
                "Images/London/first.jpg",
                "Images/London/second.jpg",
                "Images/Boston/third.jpg",
            ],
        )

    def test_manifest_rejects_noncontiguous_indices_and_unsafe_paths(self):
        for indices in ((0, 2), (0, 0)):
            with self.subTest(indices=indices):
                manifest = self.write_manifest(
                    [
                        {"row_index": index, "relative_path": f"Images/London/{i}.jpg"}
                        for i, index in enumerate(indices)
                    ]
                )
                with self.assertRaisesRegex(ValueError, "row_index"):
                    load_manifest_paths(manifest)
        for path in ("/Images/London/a.jpg", "Images/London/../a.jpg", "Images\\London\\a.jpg"):
            with self.subTest(path=path):
                manifest = self.write_manifest([{"row_index": 0, "relative_path": path}])
                with self.assertRaisesRegex(ValueError, "relative_path"):
                    load_manifest_paths(manifest)

    def test_streamed_memmap_has_float16_rows_in_order(self):
        path = self.root / "descriptors_fp16.npy"
        progress = []
        shape = write_descriptor_memmap(
            [(0, np.eye(3, dtype=np.float32)[:2]), (2, np.eye(3, dtype=np.float32)[2:])],
            path,
            row_count=3,
            flush_every=1,
            progress_callback=progress.append,
        )
        self.assertEqual(shape, (3, 3))
        self.assertEqual(progress, [2, 3])
        stored = np.load(path, mmap_mode="r")
        self.assertEqual(stored.dtype, np.float16)
        np.testing.assert_array_equal(stored, np.eye(3, dtype=np.float16))
        self.assertEqual(verify_descriptor_memmap(path, expected_rows=3), (3, 3))

    def test_incomplete_or_unordered_write_does_not_publish_output(self):
        path = self.root / "descriptors_fp16.npy"
        first = np.array([[1.0, 0.0]], dtype=np.float32)
        for batches in ([(0, first)], [(1, first)], [(0, first), (2, first)]):
            with self.subTest(batches=len(batches)):
                with self.assertRaises(ValueError):
                    write_descriptor_memmap(batches, path, row_count=2)
                self.assertFalse(path.exists())

    def test_nonfinite_or_nonunit_descriptors_are_rejected(self):
        path = self.root / "descriptors_fp16.npy"
        with self.assertRaisesRegex(ValueError, "non-finite"):
            write_descriptor_memmap(
                [(0, np.array([[np.nan, 0.0]], dtype=np.float32))],
                path,
                row_count=1,
            )
        self.assertFalse(path.exists())
        write_descriptor_memmap(
            [(0, np.array([[2.0, 0.0]], dtype=np.float32))],
            path,
            row_count=1,
        )
        with self.assertRaisesRegex(ValueError, "norms"):
            verify_descriptor_memmap(path, expected_rows=1)


if __name__ == "__main__":
    unittest.main()

"""CPU-only checks for deterministic generator-audit preprocessing."""

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools/07_build_generator_audit_set.py"
SPEC = importlib.util.spec_from_file_location("generator_audit_builder", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

AUDIT_COLUMNS = [
    "audit_index", "group", "row_index", "image_id", "relative_path",
    "place_uid", "city_id", "source_512_path", "original_width",
    "original_height", "crop_left", "crop_top", "crop_size",
]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GeneratorAuditFixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest_path = self.root / "manifest.csv"
        self.candidates_path = self.root / "rgb_candidates_raw.csv"
        self.manifest = [
            {
                "row_index": index,
                "image_id": f"id-{index}",
                "relative_path": f"Images/Boston/image-{index}.jpg",
                "place_uid": f"Boston:{index // 2:07d}",
                "city_id": "Boston",
            }
            for index in range(140)
        ]
        self.random_indices = random.Random(42).sample(list(range(140)), 50)
        self.remaining_indices = [
            index for index in range(140) if index not in set(self.random_indices)
        ]
        self.candidates = [
            self.candidate(query, negative, 1.0 - pair_index / 1000.0)
            for pair_index, (query, negative) in enumerate(
                zip(self.remaining_indices[::2], self.remaining_indices[1::2])
            )
        ]
        write_csv(self.manifest_path, self.manifest)
        write_csv(self.candidates_path, self.candidates)

    def candidate(self, query, negative, similarity):
        return {
            "query_row_index": query,
            "negative_row_index": negative,
            "query_image_id": self.manifest[query]["image_id"],
            "negative_image_id": self.manifest[negative]["image_id"],
            "query_place_uid": self.manifest[query]["place_uid"],
            "negative_place_uid": self.manifest[negative]["place_uid"],
            "query_city_id": "Boston",
            "negative_city_id": "Boston",
            "rank": 1,
            "similarity": similarity,
            "geo_distance_m": 500,
            "same_city": True,
            "pair_uid": f"pair-{query}-{negative}",
        }

    def select(self, manifest=None):
        if manifest is None:
            manifest = builder.load_manifest(self.manifest_path)
        return builder.select_audit_sources(manifest, self.candidates_path, seed=42)


class GeneratorAuditSelectionTests(GeneratorAuditFixture, unittest.TestCase):
    def test_selection_is_seeded_deterministic_unique_and_uses_both_pair_sides(self):
        original = [dict(row) for row in self.manifest]
        selected = self.select()
        self.assertEqual(selected, self.select())
        self.assertEqual(len(selected), 100)
        self.assertEqual([row["group"] for row in selected],
                         ["random"] * 50 + ["hard_candidate"] * 50)
        self.assertEqual([row["row_index"] for row in selected[:50]], self.random_indices)
        self.assertEqual([row["row_index"] for row in selected[50:]], self.remaining_indices[:50])
        self.assertEqual(len({row["row_index"] for row in selected}), 100)
        self.assertEqual(len({row["image_id"] for row in selected}), 100)
        self.assertEqual(self.manifest, original)

    def test_random_selection_is_independent_of_manifest_row_order(self):
        self.assertEqual(self.select(self.manifest), self.select(list(reversed(self.manifest))))

    def test_similarity_order_ties_and_overlapping_endpoints_are_deterministic(self):
        # CSV order breaks similarity ties; each pair is visited query then negative.
        high_pairs = list(reversed(self.candidates[:25]))
        for pair in high_pairs:
            pair["similarity"] = 0.99
        random_index = self.random_indices[0]
        hard_index = int(high_pairs[0]["query_row_index"])
        overlapping_pair = self.candidate(random_index, hard_index, 1.0)
        rows = [self.candidates[30], overlapping_pair, overlapping_pair, *high_pairs]
        write_csv(self.candidates_path, rows)
        hard = [row["row_index"] for row in self.select()[50:]]
        expected = [hard_index]
        for pair in high_pairs:
            for column in ("query_row_index", "negative_row_index"):
                index = int(pair[column])
                if index not in expected:
                    expected.append(index)
        self.assertEqual(hard, expected)
        self.assertEqual(len(hard), 50)

    def test_insufficient_distinct_hard_candidates_fails_clearly(self):
        # Repeated pairs and random-group endpoints cannot make up the deficit.
        rows = self.candidates[:24] + [self.candidates[0]] * 5
        rows.append(self.candidate(self.random_indices[0], self.random_indices[1], 1.0))
        write_csv(self.candidates_path, rows)
        with self.assertRaisesRegex(ValueError, r"(?i)(50|hard.candidate|unique)"):
            self.select()

    def test_insufficient_manifest_fails(self):
        write_csv(self.manifest_path, self.manifest[:49])
        with self.assertRaises(ValueError):
            self.select()

    def test_duplicate_manifest_row_indices_or_image_ids_are_rejected(self):
        for column in ("row_index", "image_id"):
            with self.subTest(column=column):
                rows = [dict(row) for row in self.manifest]
                rows[-1][column] = rows[0][column]
                write_csv(self.manifest_path, rows)
                with self.assertRaises(ValueError):
                    builder.load_manifest(self.manifest_path)

    def test_invalid_candidate_endpoints_are_rejected(self):
        corruptions = (
            ("query_image_id", "wrong-image"),
            ("negative_image_id", "wrong-image"),
            ("query_row_index", 140),
            ("negative_row_index", -1),
        )
        for column, value in corruptions:
            with self.subTest(column=column):
                rows = [dict(row) for row in self.candidates]
                rows[0][column] = value
                write_csv(self.candidates_path, rows)
                with self.assertRaises(ValueError):
                    self.select()


class GeneratorAuditArtifactTests(GeneratorAuditFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset_root = self.root / "dataset"
        self.output_dir = self.root / "generator_audit"
        self.dimensions = ((18, 12), (12, 18), (12, 12))
        for row in self.manifest:
            index = row["row_index"]
            path = self.dataset_root / row["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with Image.new("RGB", self.dimensions[index % 3],
                           (index % 256, (index * 3) % 256, (index * 7) % 256)) as image:
                image.save(path)

    def build(self):
        return builder.build_audit_set(
            self.manifest_path, self.candidates_path, self.dataset_root,
            self.output_dir, seed=42,
        )

    def output_checksums(self):
        return {path.relative_to(self.output_dir).as_posix(): sha256(path)
                for path in self.output_dir.rglob("*") if path.is_file()}

    def test_build_writes_canonical_images_crop_metadata_and_reference_hashes(self):
        sources_before = {path: sha256(path) for path in self.dataset_root.rglob("*.jpg")}
        returned_summary = self.build()
        with (self.output_dir / "audit_manifest.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, AUDIT_COLUMNS)
            rows = list(reader)
        self.assertEqual(len(rows), 100)
        self.assertEqual([int(row["audit_index"]) for row in rows], list(range(100)))
        self.assertEqual(len({row["row_index"] for row in rows}), 100)
        self.assertEqual(len({row["image_id"] for row in rows}), 100)
        self.assertEqual({path.name for path in (self.output_dir / "source_512").iterdir()},
                         {f'{int(row["row_index"]):08d}.jpg' for row in rows})
        for row in rows:
            index = int(row["row_index"])
            original = self.manifest[index]
            for column in ("image_id", "relative_path", "place_uid", "city_id"):
                self.assertEqual(row[column], original[column])
            width, height = self.dimensions[index % 3]
            crop_size = min(width, height)
            self.assertEqual(int(row["original_width"]), width)
            self.assertEqual(int(row["original_height"]), height)
            self.assertEqual(int(row["crop_left"]), (width - crop_size) // 2)
            self.assertEqual(int(row["crop_top"]), (height - crop_size) // 2)
            self.assertEqual(int(row["crop_size"]), crop_size)
            output_path = self.output_dir / "source_512" / f"{index:08d}.jpg"
            self.assertFalse(Path(row["relative_path"]).is_absolute())
            self.assertFalse(Path(row["source_512_path"]).is_absolute())
            self.assertEqual(row["source_512_path"], os.path.relpath(output_path, Path.cwd()))
            with Image.open(output_path) as image:
                self.assertEqual(image.size, (512, 512))
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.format, "JPEG")
        summary = json.loads((self.output_dir / "audit_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(returned_summary, summary)
        self.assertEqual(summary["number_of_images"], 100)
        self.assertEqual(summary["count_random"], 50)
        self.assertEqual(summary["count_hard_candidate"], 50)
        self.assertEqual(summary["seed"], 42)
        self.assertEqual(summary["canonical_resolution"], [512, 512])
        self.assertEqual(summary["manifest_sha256"], sha256(self.manifest_path))
        self.assertEqual(summary["rgb_candidate_sha256"], sha256(self.candidates_path))
        self.assertEqual(summary["manifest_reference"], os.path.relpath(self.manifest_path, Path.cwd()))
        self.assertEqual(summary["rgb_candidate_reference"], os.path.relpath(self.candidates_path, Path.cwd()))
        self.assertEqual(summary["config"]["seed"], 42)
        self.assertFalse(Path(summary["config"]["dataset_root"]).is_absolute())
        self.assertEqual(summary["config"]["dataset_root"],
                         os.path.relpath(self.dataset_root, Path.cwd()))
        self.assertFalse(Path(summary["config"]["output_dir"]).is_absolute())
        self.assertEqual(sources_before, {path: sha256(path) for path in sources_before})

    def test_repeated_build_produces_identical_artifact_bytes(self):
        first_summary = self.build()
        checksums = self.output_checksums()
        self.assertEqual(first_summary, self.build())
        self.assertEqual(checksums, self.output_checksums())

    def test_failed_image_preserves_already_published_set(self):
        self.build()
        checksums = self.output_checksums()
        last_selected = self.select()[-1]
        source_path = self.dataset_root / last_selected["relative_path"]
        original_bytes = source_path.read_bytes()
        for failure, exception in (("missing", FileNotFoundError), ("corrupt", ValueError)):
            with self.subTest(failure=failure):
                if failure == "missing":
                    source_path.unlink()
                else:
                    source_path.write_bytes(b"this is not a decodable image\n")
                try:
                    with self.assertRaises(exception):
                        self.build()
                    self.assertEqual(checksums, self.output_checksums())
                    self.assertEqual({path.name for path in self.output_dir.iterdir()},
                                     {"source_512", "audit_manifest.csv", "audit_summary.json"})
                finally:
                    source_path.write_bytes(original_bytes)

    def test_dataset_inside_managed_image_directory_is_rejected_before_sources_change(self):
        self.output_dir = self.root / "nested"
        for suffix in ("source_512", "source_512/child"):
            with self.subTest(dataset_suffix=suffix):
                # Relocate the complete fixture, including unselected sources.
                staging_dataset = self.root / "relocating_dataset"
                self.dataset_root.rename(staging_dataset)
                self.dataset_root = self.output_dir / suffix
                self.dataset_root.parent.mkdir(parents=True, exist_ok=True)
                staging_dataset.rename(self.dataset_root)
                sources_before = {
                    path.relative_to(self.dataset_root).as_posix(): sha256(path)
                    for path in self.dataset_root.rglob("*.jpg")
                }
                with self.assertRaises(ValueError):
                    self.build()
                self.assertEqual(
                    sources_before,
                    {path.relative_to(self.dataset_root).as_posix(): sha256(path)
                     for path in self.dataset_root.rglob("*.jpg")},
                )
                self.assertFalse((self.output_dir / "audit_manifest.csv").exists())
                self.assertFalse((self.output_dir / "audit_summary.json").exists())


if __name__ == "__main__":
    unittest.main()

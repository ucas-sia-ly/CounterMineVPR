"""CPU-only identity/geometry checks; no sampling, models, or CUDA."""

import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from countermine.probe.canonical import canonicalize_probe
from countermine.probe.geometry_audit import reference, sha256, validate_png
from countermine.probe.native100_audit import (
    load_frozen_population, load_native100_manifest, validate_manifest_metadata,
)

SPEC = importlib.util.spec_from_file_location("native100_build", Path(__file__).resolve().parents[1] / "tools/19_build_native100_audit.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Native100BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.dataset = cls.root / "dataset"
        cls.dataset.mkdir()
        historical = cls.root / "historical"
        historical.mkdir()
        population, mini = [], []
        for index in range(100):
            name = f"image_{index:03d}.jpg"
            original = cls.dataset / name
            with Image.new("RGB", (640, 480), (index, index * 2, 255 - index)) as image:
                image.save(original, format="JPEG")
            old_png = historical / f"{index}.png"
            with Image.open(original) as image:
                old, _ = canonicalize_probe(image)
                old.save(old_png, format="PNG")
                old.close()
            identity = {"row_index": str(index + 1000), "image_id": name, "relative_path": name,
                        "place_uid": f"city:{index}", "city_id": "city"}
            mini.append(identity)
            population.append({"audit_index": str(index), "group": "random" if index < 50 else "hard_candidate",
                               **identity, "source_512_path": reference(old_png),
                               "original_width": "640", "original_height": "480",
                               "selection_note": f"frozen-{index}"})
        cls.population = population
        cls.mini_path = cls.root / "mini.csv"
        cls.manifest = cls.root / "audit_manifest.csv"
        cls.summary_path = cls.root / "audit_summary.json"
        write_csv(cls.mini_path, mini)
        write_csv(cls.manifest, population)
        cls.summary = {"number_of_images": 100, "count_random": 50, "count_hard_candidate": 50,
                       "seed": 42, "manifest_reference": reference(cls.mini_path),
                       "manifest_sha256": sha256(cls.mini_path), "rgb_candidate_reference": "old_candidates.csv",
                       "rgb_candidate_sha256": "a" * 64, "config": {"seed": 42, "dataset_root": reference(cls.dataset)}}
        cls.summary_path.write_text(json.dumps(cls.summary), encoding="utf-8")
        cls.output = cls.root / "native"
        with patch("random.Random", side_effect=AssertionError("no resampling")):
            cls.build_summary = BUILD.build_native100_audit(cls.manifest, cls.summary_path, cls.dataset, cls.output)
        cls.rows = load_native100_manifest(cls.output / "native100_manifest.csv", cls.manifest, cls.summary_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_frozen_population_and_provenance_without_resampling(self):
        old, _ = load_frozen_population(self.manifest, self.summary_path)
        self.assertEqual(len(self.rows), 100)
        for previous, current in zip(old, self.rows):
            for name in ("audit_index", "row_index", "image_id", "group", "selection_note"):
                self.assertEqual(previous[name], current[name])
        self.assertEqual(self.build_summary["source_population_provenance"]["audit_manifest_sha256"], sha256(self.manifest))

    def test_native_geometry_and_decoded_pixels(self):
        for row in self.rows:
            self.assertEqual((row["canonical_width"], row["canonical_height"]), (640, 480))
            for key in ("scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction"):
                self.assertEqual(row[key], 1.0)
            with Image.open(row["original_path"]) as original, Image.open(row["source_path"]) as source:
                self.assertEqual(source.format, "PNG")
                self.assertEqual(source.mode, "RGB")
                self.assertEqual(original.convert("RGB").tobytes(), source.tobytes())

    def test_population_count_and_each_identity_must_be_unique(self):
        for key in ("audit_index", "row_index", "image_id"):
            rows = copy.deepcopy(self.rows)
            rows[1][key] = rows[0][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest_metadata(rows)
        for rows in (self.rows[:99], [*self.rows, self.rows[0]]):
            with self.assertRaises(ValueError):
                validate_manifest_metadata(rows)

    def test_geometry_scale_and_fov_reject_unexpected_values(self):
        for key, bad in (("canonical_width", 512), ("original_height", 481), ("scale_x", .8),
                         ("scale_y", float("nan")), ("retained_area_fraction", .75),
                         ("retained_long_axis_fraction", .9)):
            rows = copy.deepcopy(self.rows)
            rows[0][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest_metadata(rows)

    def test_source_and_relit_png_dimension_validation(self):
        for name, size, mode in (("resize", (512, 384), "RGB"), ("grayscale", (640, 480), "L")):
            path = self.root / f"{name}.png"
            with Image.new(mode, size) as image:
                image.save(path)
            with self.assertRaises(ValueError):
                validate_png(path, (640, 480))

    def test_original_identity_resolution_and_provenance_cannot_change(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["selection_note"] = "resampled"
        old, _ = load_frozen_population(self.manifest, self.summary_path)
        with self.assertRaises(ValueError):
            validate_manifest_metadata(rows, old)
        bad_summary = dict(self.summary, manifest_sha256="b" * 64)
        with self.assertRaises(ValueError):
            BUILD._originals(old, bad_summary, self.dataset)

    def test_original_stored_dimensions_fail_without_resize(self):
        original_path = self.dataset / "image_000.jpg"
        contents = original_path.read_bytes()
        try:
            with Image.new("RGB", (512, 384)) as bad:
                bad.save(original_path)
            old, _ = load_frozen_population(self.manifest, self.summary_path)
            with self.assertRaises(ValueError):
                BUILD._originals(old, self.summary, self.dataset)
        finally:
            original_path.write_bytes(contents)

    def test_changed_original_is_rejected_against_historical_pixels(self):
        with Image.new("RGB", (640, 480), "red") as changed:
            with self.assertRaises(ValueError):
                BUILD._verify_historical_pixels(changed, self.population[0])


if __name__ == "__main__":
    unittest.main()

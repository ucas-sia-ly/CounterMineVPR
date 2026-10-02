"""CPU-only Step 2A pixel, metric, cache, and matcher-boundary tests."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from countermine.mining.structural_matcher import (
    BoundedFeatureCache, StructuralMatcher, decode_original_rgb,
    deterministic_match_indices, normalized_grid_cells, rgb_pixel_sha256,
    structural_metrics,
)


class StructuralMetricTests(unittest.TestCase):
    def test_match_ratio_and_independent_spatial_metrics(self):
        a = np.array([[0, 0], [80, 60], [639, 479]], dtype=float)
        b = np.array([[100, 100], [101, 101], [102, 102]], dtype=float)
        metrics = structural_metrics(10, 6, a, b)
        self.assertEqual(metrics["local_match_ratio"], .5)
        self.assertEqual(metrics["num_matches"], 3)
        self.assertEqual(metrics["matched_source_cell_coverage"], 3 / 64)
        self.assertEqual(metrics["matched_target_cell_coverage"], 1 / 64)
        self.assertEqual(metrics["symmetric_match_coverage"], 1 / 64)
        self.assertEqual(metrics["source_max_cell_match_fraction"], 1 / 3)
        self.assertEqual(metrics["target_max_cell_match_fraction"], 1)
        self.assertAlmostEqual(metrics["source_match_entropy"], np.log(3) / np.log(64))
        self.assertEqual(metrics["target_match_entropy"], 0)
        self.assertEqual(metrics["symmetric_match_entropy"], 0)

    def test_zero_keypoints_zero_matches(self):
        metrics = structural_metrics(0, 12, [], [])
        self.assertEqual(metrics["local_match_ratio"], 0)
        self.assertEqual(metrics["num_matches"], 0)
        for name in metrics:
            if "coverage" in name or "entropy" in name or "fraction" in name:
                self.assertEqual(metrics[name], 0)

    def test_grid_mapping_and_roundoff_only_clamping(self):
        points = [[0, 0], [79.99, 59.99], [80, 60], [639.99, 479.99], [640, 480], [-1e-8, -1e-8]]
        np.testing.assert_array_equal(normalized_grid_cells(points), [0, 0, 9, 63, 63, 0])
        for bad in ([[641, 1]], [[-0.1, 1]], [[0, 481]], [[np.nan, 0]], [[1, 2, 3]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalized_grid_cells(bad)

    def test_uniform_grid_has_full_coverage_and_entropy(self):
        points = [[x * 80 + 40, y * 60 + 30] for y in range(8) for x in range(8)]
        metrics = structural_metrics(64, 64, points, points)
        self.assertEqual(metrics["matched_source_cell_coverage"], 1)
        self.assertEqual(metrics["source_max_cell_match_fraction"], 1 / 64)
        self.assertAlmostEqual(metrics["source_match_entropy"], 1)

    def test_invalid_counts_and_matches_rejected(self):
        for counts in ((-1, 5), (1.5, 5), (True, 5), (0, 5)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                structural_metrics(*counts, [[1, 1]], [[2, 2]])
        with self.assertRaises(ValueError):
            structural_metrics(3, 3, [[1, 1]], [])

    def test_confidence_selection_is_deterministic_and_limited(self):
        np.testing.assert_array_equal(deterministic_match_indices([.1, .9, .9, .2], 4), [1, 2, 3, 0])
        np.testing.assert_array_equal(deterministic_match_indices(None, 4, 2), [0, 1])
        self.assertEqual(len(deterministic_match_indices(None, 150)), 100)
        with self.assertRaises(ValueError):
            deterministic_match_indices(None, 3, 101)

    def test_cache_has_bounded_lru_and_zero_capacity(self):
        cache = BoundedFeatureCache(2)
        cache.put("a", "feature-a")
        cache.put("b", "feature-b")
        self.assertEqual(cache.get("a"), "feature-a")
        cache.put("c", "feature-c")
        self.assertIsNone(cache.get("b"))
        self.assertEqual(len(cache), 2)
        disabled = BoundedFeatureCache(0)
        disabled.put("a", "a")
        self.assertEqual(len(disabled), 0)
        with self.assertRaises(ValueError):
            BoundedFeatureCache(33)


class OriginalRGBTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_hash_is_exact_decoded_rgb_bytes_across_file_metadata(self):
        pixels = np.full((480, 640, 3), 40, dtype=np.uint8)
        a, b = self.root / "a.png", self.root / "b.bmp"
        Image.fromarray(pixels).save(a)
        Image.fromarray(pixels).save(b)
        self.assertEqual(rgb_pixel_sha256(a), hashlib.sha256(pixels.tobytes()).hexdigest())
        decoded_a, fingerprint_a = decode_original_rgb(a)
        _, fingerprint_b = decode_original_rgb(b)
        np.testing.assert_array_equal(decoded_a, pixels)
        self.assertEqual(fingerprint_a["rgb_pixel_sha256"], fingerprint_b["rgb_pixel_sha256"])
        self.assertNotEqual(fingerprint_a["image_file_sha256"], fingerprint_b["image_file_sha256"])

    def test_rgb_conversion_and_original_orientation(self):
        path = self.root / "gray.png"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("L", (640, 480), 55).save(path, exif=exif)
        pixels, _ = decode_original_rgb(path)
        self.assertEqual(pixels.shape, (480, 640, 3))
        self.assertTrue((pixels == 55).all())

    def test_non_native_geometry_and_invalid_decode_fail(self):
        path = self.root / "wrong.png"
        Image.new("RGB", (480, 640)).save(path)
        with self.assertRaisesRegex(ValueError, "640x480"):
            decode_original_rgb(path)
        path.write_bytes(b"invalid-image")
        with self.assertRaises(ValueError):
            decode_original_rgb(path)

    def test_extractor_receives_rgb_native_tensor_resize_none_and_size_is_checked(self):
        import torch
        path = self.root / "native.png"
        Image.new("RGB", (640, 480), (10, 20, 30)).save(path)
        matcher = StructuralMatcher.__new__(StructuralMatcher)
        matcher.torch, matcher.device = torch, torch.device("cpu")
        matcher.cache, matcher.fingerprints = BoundedFeatureCache(1), {}

        class Extractor:
            wrong = False

            def extract(self, image, **kwargs):
                self.image, self.arguments = image, kwargs
                self.inference = torch.is_inference_mode_enabled()
                return {"image_size": torch.tensor([[322, 322] if self.wrong else [640, 480]])}

        matcher.extractor = Extractor()
        matcher._features("a", path)
        self.assertEqual(matcher.extractor.arguments, {"resize": None})
        self.assertEqual(tuple(matcher.extractor.image.shape), (3, 480, 640))
        self.assertTrue(matcher.extractor.inference)
        self.assertAlmostEqual(float(matcher.extractor.image[0, 0, 0]), 10 / 255, places=7)
        matcher.extractor.wrong = True
        with self.assertRaisesRegex(ValueError, "image_size"):
            matcher._features("b", path)


if __name__ == "__main__":
    unittest.main()

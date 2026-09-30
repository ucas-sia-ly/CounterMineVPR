"""CPU-only fidelity metrics and mocked adapters; never load model weights."""

import dataclasses
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from countermine.probe.local_fidelity import (
    EPSILONS, METRIC_COLUMNS, LocalFidelityConfig, LocalFidelityMatcher,
    MatchResult, _import_vendored_models, compute_fidelity_metrics,
)


class FidelityMetricsTest(unittest.TestCase):
    def metrics(self, source, relit=None, scores=None, n_source=None, n_relit=None):
        source = np.asarray(source, dtype=float).reshape(-1, 2)
        relit = source.copy() if relit is None else np.asarray(relit, dtype=float).reshape(-1, 2)
        scores = np.full(len(source), 0.75) if scores is None else scores
        n_source = len(source) if n_source is None else n_source
        n_relit = len(relit) if n_relit is None else n_relit
        return compute_fidelity_metrics(source, relit, scores, n_source, n_relit)

    def test_zero_displacement_and_counts(self):
        result = self.metrics([[10, 20], [100, 80]], n_source=4, n_relit=3)
        self.assertEqual(tuple(result), METRIC_COLUMNS)
        self.assertEqual(EPSILONS, (2, 4, 8, 16))
        self.assertEqual(result["num_matches"], 2)
        self.assertEqual(result["match_ratio_min"], 2 / 3)
        for eps in EPSILONS:
            self.assertEqual(result[f"good_matches_{eps}px"], 2)
            self.assertEqual(result[f"repeatability_min_{eps}px"], 2 / 3)
            self.assertEqual(result[f"precision_matches_{eps}px"], 1.0)
        for name in ("mean", "median", "q75", "q90", "q95", "max"):
            self.assertEqual(result[f"displacement_{name}"], 0.0)
        self.assertEqual(result["match_score_mean"], 0.75)
        json.dumps(result, allow_nan=False)

    def test_known_displacement_is_euclidean_without_registration(self):
        result = self.metrics([[10, 20], [100, 80]], [[13, 24], [103, 84]])
        self.assertEqual(result["displacement_mean"], 5.0)
        self.assertEqual(result["displacement_max"], 5.0)
        self.assertEqual(result["good_matches_4px"], 0)
        self.assertEqual(result["good_matches_8px"], 2)
        self.assertEqual(result["grid_coverage_4px"], 0.0)
        self.assertEqual(result["grid_coverage_8px"], 2 / 64)

    def test_epsilon_boundaries_are_inclusive(self):
        displacements = np.array([0, 2, 2.001, 4, 4.001, 8, 8.001, 16, 16.001])
        source = np.full((len(displacements), 2), 100.0)
        relit = source.copy()
        relit[:, 0] += displacements
        result = self.metrics(source, relit)
        for eps, count in ((2, 2), (4, 4), (8, 6), (16, 8)):
            self.assertEqual(result[f"good_matches_{eps}px"], count)
            self.assertEqual(result[f"precision_matches_{eps}px"], count / 9)

    def test_denominator_uses_minimum_keypoint_count(self):
        result = self.metrics([[10, 20], [30, 40]], n_source=100, n_relit=4)
        self.assertEqual(result["match_ratio_min"], 0.5)
        self.assertEqual(result["repeatability_min_8px"], 0.5)
        self.assertEqual(result["precision_matches_8px"], 1.0)
        swapped = self.metrics([[10, 20], [30, 40]], n_source=4, n_relit=100)
        self.assertEqual(swapped["match_ratio_min"], result["match_ratio_min"])

    def test_empty_matches_and_zero_denominators_are_explicit(self):
        for n_source, n_relit in ((0, 0), (0, 12), (12, 0), (5, 12)):
            with self.subTest(counts=(n_source, n_relit)):
                result = self.metrics([], n_source=n_source, n_relit=n_relit)
                self.assertEqual(result["num_matches"], 0)
                self.assertEqual(result["match_ratio_min"], 0.0)
                for eps in EPSILONS:
                    self.assertEqual(result[f"good_matches_{eps}px"], 0)
                    self.assertEqual(result[f"repeatability_min_{eps}px"], 0.0)
                    self.assertEqual(result[f"precision_matches_{eps}px"], 0.0)
                for key, value in result.items():
                    if key.startswith("displacement_") or key.startswith("match_score_"):
                        self.assertIsNone(value)
                self.assertEqual(result["grid_coverage_4px"], 0.0)
                self.assertEqual(result["occupied_grid_cells_8px"], 0)
                json.dumps(result, allow_nan=False)

    def test_source_grid_covers_8_by_8_cells(self):
        source = np.array([[x * 64 + 32, y * 64 + 32] for y in range(8) for x in range(8)])
        result = self.metrics(source)
        for eps in (4, 8):
            self.assertEqual(result[f"occupied_grid_cells_{eps}px"], 64)
            self.assertEqual(result[f"grid_coverage_{eps}px"], 1.0)

    def test_grid_boundaries_duplicates_and_source_side(self):
        source = [[63.5, 1], [64, 1], [64.5, 1], [511.9, 511.9]]
        relit = [[65.5, 1], [65, 1], [66, 1], [511.9, 511.9]]
        result = self.metrics(source, relit)
        self.assertEqual(result["occupied_grid_cells_4px"], 3)
        self.assertEqual(result["grid_coverage_4px"], 3 / 64)

    def test_grid_coverage_filters_good_matches_before_counting(self):
        source = np.array([[1, 1], [65, 1], [129, 1], [193, 1]], dtype=float)
        relit = source + np.array([[0, 0], [4, 0], [8, 0], [16, 0]])
        result = self.metrics(source, relit)
        self.assertEqual(result["occupied_grid_cells_4px"], 2)
        self.assertEqual(result["occupied_grid_cells_8px"], 3)

    def test_deterministic_linear_quantiles_and_confidences(self):
        source = np.full((5, 2), 50.0)
        relit = source.copy()
        relit[:, 0] += [0, 2, 4, 8, 16]
        result = self.metrics(source, relit, [0.1, 0.2, 0.5, 0.8, 0.9])
        expected = {
            "displacement_mean": 6.0, "displacement_median": 4.0,
            "displacement_q75": 8.0, "displacement_q90": 12.8,
            "displacement_q95": 14.4, "displacement_max": 16.0,
            "match_score_mean": 0.5, "match_score_median": 0.5,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(result[key], value)
        reversed_result = self.metrics(source[::-1], relit[::-1], [0.9, 0.8, 0.5, 0.2, 0.1])
        for key, value in result.items():
            self.assertAlmostEqual(value, reversed_result[key])

    def test_invalid_points_scores_and_counts_are_rejected(self):
        valid = np.array([[10.0, 20.0]])
        cases = (
            ([], [], [], 0, 0),  # Empty coordinates must explicitly be (0,2).
            ([[float("nan"), 20]], valid, [0.5], 1, 1),
            ([[-0.1, 20]], valid, [0.5], 1, 1),
            ([[512, 20]], valid, [0.5], 1, 1),
            (valid, [[10, float("inf")]], [0.5], 1, 1),
            (valid, np.empty((0, 2)), [0.5], 1, 1),
            (valid, valid, [], 1, 1),
            (valid, valid, [float("nan")], 1, 1),
            (valid, valid, [0.5], 0, 1),
            (valid, valid, [0.5], -1, 1),
            (valid, valid, [0.5], 1.5, 1),
            (valid, valid, [0.5], True, 1),
        )
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    compute_fidelity_metrics(*args)

    def test_inputs_are_not_mutated(self):
        source = np.array([[10, 20], [40, 60]], dtype=np.float32)
        relit = source + 2
        scores = np.array([0.5, 0.75])
        original = [array.copy() for array in (source, relit, scores)]
        self.metrics(source, relit, scores)
        for actual, expected in zip((source, relit, scores), original):
            np.testing.assert_array_equal(actual, expected)

    def test_match_result_cdf_displacements_share_metric_precision(self):
        source = np.array([[0.01, 0.02]], dtype=np.float32)
        relit = np.array([[10.001, 10.123]], dtype=np.float32)
        matches = MatchResult(1, 1, source, relit, np.array([0.8], dtype=np.float32))
        metrics = self.metrics(source, relit, matches.match_scores)
        self.assertEqual(matches.displacements.dtype, np.float64)
        self.assertEqual(float(matches.displacements[0]), metrics["displacement_mean"])

    def test_square_default_is_exactly_equal_to_explicit_geometry(self):
        source = np.array([[63.5, 63.5], [64, 64], [511.9, 511.9]])
        relit = source - 2
        scores = np.full(len(source), 0.75)
        default = compute_fidelity_metrics(source, relit, scores, 5, 3)
        explicit = compute_fidelity_metrics(
            source, relit, scores, 5, 3, canonical_width=512, canonical_height=512,
        )
        self.assertEqual(default, explicit)
        self.assertEqual(default["canonical_width"], 512)
        self.assertEqual(default["canonical_height"], 512)

    def test_rectangular_coordinates_use_separate_width_and_height(self):
        source = np.array([[511.9, 383.9]])
        result = compute_fidelity_metrics(
            source, source, [0.75], 1, 1, canonical_width=512, canonical_height=384,
        )
        self.assertEqual(result["canonical_width"], 512)
        self.assertEqual(result["canonical_height"], 384)
        self.assertEqual(result["displacement_q95"], 0.0)
        for points in ([[512, 10]], [[10, 384]], [[-0.01, 0]], [[0, -0.01]]):
            for invalid_side in ("source", "relit"):
                with self.subTest(points=points, invalid_side=invalid_side):
                    with self.assertRaisesRegex(ValueError, "512x384"):
                        compute_fidelity_metrics(
                            points if invalid_side == "source" else [[10, 10]],
                            points if invalid_side == "relit" else [[10, 10]],
                            [0.75], 1, 1, canonical_width=512, canonical_height=384,
                        )

    def test_normalized_rectangular_grid_covers_all_64_cells(self):
        source = np.array([[x * 64 + 32, y * 48 + 24] for y in range(8) for x in range(8)])
        result = compute_fidelity_metrics(
            source, source, np.ones(64), 64, 64, canonical_width=512, canonical_height=384,
        )
        for eps in (4, 8):
            self.assertEqual(result[f"occupied_grid_cells_{eps}px"], 64)
            self.assertEqual(result[f"grid_coverage_{eps}px"], 1.0)

    def test_rectangular_grid_boundaries_use_48_pixel_height_cells(self):
        source = np.array([[1, 47.9], [1, 48], [1, 48.1], [511.9, 383.9]])
        result = compute_fidelity_metrics(
            source, source, np.ones(4), 4, 4, canonical_width=512, canonical_height=384,
        )
        self.assertEqual(result["occupied_grid_cells_8px"], 3)
        self.assertEqual(result["grid_coverage_8px"], 3 / 64)

    def test_geometry_does_not_change_pixel_displacement_thresholds(self):
        source = np.array([[32, 24], [100, 100]])
        relit = source + np.array([[3, 4], [0, 8]])
        args = (source, relit, [0.5, 0.75], 2, 2)
        square = compute_fidelity_metrics(*args)
        rectangular = compute_fidelity_metrics(*args, canonical_width=512, canonical_height=384)
        for key in square:
            if key.startswith(("displacement_", "repeatability_", "precision_", "good_matches_")):
                self.assertEqual(square[key], rectangular[key])
        self.assertEqual(rectangular["displacement_median"], 6.5)

    def test_invalid_dimensions_are_rejected_even_with_empty_matches(self):
        for dimension in (0, -1, 384.0, True, float("nan"), float("inf"), "384"):
            for keyword in ("canonical_width", "canonical_height"):
                with self.subTest(dimension=dimension, keyword=keyword):
                    with self.assertRaises(ValueError):
                        compute_fidelity_metrics(
                            np.empty((0, 2)), np.empty((0, 2)), [], 0, 0, **{keyword: dimension},
                        )


class _FakeALIKED:
    checkpoint_url = "https://example.invalid/{}.pth"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.conf = SimpleNamespace(model_name="aliked-n16", nms_radius=2, **kwargs)
        self.eval_calls = 0
        self.device = None
        self.extraction_calls = []

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.device = device
        return self

    def extract(self, image, **kwargs):
        assert torch.is_inference_mode_enabled()
        self.extraction_calls.append((image.detach().clone(), kwargs))
        return {
            "keypoints": torch.tensor([[[10.0, 20.0], [100.0, 120.0]]]),
            "descriptors": torch.zeros((1, 2, 128)),
            "image_size": torch.tensor([[float(image.shape[2]), float(image.shape[1])]]),
        }


class _FakeLightGlue:
    version = "v0.1_arxiv"
    url = "https://example.invalid/{}/{}.pth"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.conf = SimpleNamespace(weights="aliked_lightglue", **kwargs)
        self.eval_calls = 0
        self.device = None
        self.calls = []
        self.prediction = {
            "matches": [torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)],
            "scores": [torch.tensor([0.8, 0.6])],
        }

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, data):
        assert torch.is_inference_mode_enabled()
        self.calls.append(data)
        return self.prediction


class LocalFidelityAdapterTest(unittest.TestCase):
    def fake_adapter(self, cache):
        adapter = LocalFidelityMatcher(LocalFidelityConfig(device="cpu"))
        with mock.patch("countermine.probe.local_fidelity._import_vendored_models", return_value=(_FakeALIKED, _FakeLightGlue)), \
             mock.patch.object(torch.hub, "get_dir", return_value=cache), \
             mock.patch.object(torch.hub, "load_state_dict_from_url", side_effect=AssertionError("Weights must not load in tests")):
            adapter._ensure_models()
        return adapter

    def test_config_is_fixed_frozen_and_validated(self):
        config = LocalFidelityConfig()
        self.assertEqual(config.max_keypoints, 2048)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.device, "auto")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.seed = 4
        for overrides in ({"device": "bad"}, {"device": "cuda:-1"}, {"max_keypoints": 0}, {"max_keypoints": True}, {"seed": -1}, {"seed": True}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    LocalFidelityConfig(**overrides)

    def test_import_constructor_and_metadata_do_not_import_torch_or_models(self):
        script = """
import builtins, json
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'kornia', 'lightglue', '_countermine_vendored_lightglue'}:
        raise AssertionError('Eager model import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from countermine.probe.local_fidelity import LocalFidelityMatcher
adapter = LocalFidelityMatcher()
metadata = adapter.runtime_metadata()
assert metadata['resolved_device'] is None
json.dumps(metadata, allow_nan=False)
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_model_settings_eval_device_and_lazy_loading(self):
        with TemporaryDirectory() as temporary:
            adapter = self.fake_adapter(temporary)
            self.assertEqual(adapter.extractor.kwargs, {"max_num_keypoints": 2048, "detection_threshold": 0.2})
            self.assertEqual(adapter.matcher.kwargs, {
                "features": "aliked", "depth_confidence": -1,
                "width_confidence": -1, "filter_threshold": 0.1, "mp": False,
            })
            self.assertEqual(adapter.extractor.eval_calls, 1)
            self.assertEqual(adapter.matcher.eval_calls, 1)
            self.assertEqual(str(adapter.extractor.device), "cpu")
            with mock.patch("countermine.probe.local_fidelity._import_vendored_models") as imports:
                adapter._ensure_models()
            imports.assert_not_called()
            metadata = adapter.runtime_metadata()
            self.assertEqual(metadata["resolved_device"], "cpu")
            self.assertIsNone(metadata["extract_resize"])
            self.assertFalse(metadata["compiled"])
            self.assertEqual(metadata["models"]["lightglue_weights_version"], "v0.1_arxiv")
            self.assertIn("aliked", metadata["models"]["checkpoints"])
            json.dumps(metadata, allow_nan=False)

    def test_png_extract_is_rgb_float_canonical_and_resize_none_every_call(self):
        with TemporaryDirectory() as temporary:
            adapter = self.fake_adapter(temporary)
            source = Path(temporary) / "source.png"
            Image.new("RGB", (512, 512), (0, 127, 255)).save(source)
            for _ in range(2):
                adapter.extract(source)
            for image, kwargs in adapter.extractor.extraction_calls:
                self.assertEqual(kwargs, {"resize": None})
                self.assertEqual(tuple(image.shape), (3, 512, 512))
                self.assertEqual(image.dtype, torch.float32)
                torch.testing.assert_close(image[:, 0, 0], torch.tensor([0, 127 / 255, 1]))
                self.assertGreaterEqual(float(image.min()), 0)
                self.assertLessEqual(float(image.max()), 1)

    def test_invalid_png_geometry_or_mode_fails_before_model_load(self):
        with TemporaryDirectory() as temporary:
            cases = (
                ("legacy.jpg", "RGB", (512, 512), "JPEG"),
                ("hidden_jpeg.png", "RGB", (512, 512), "JPEG"),
                ("gray.png", "L", (512, 512), "PNG"),
                ("alpha.png", "RGBA", (512, 512), "PNG"),
            )
            adapter = LocalFidelityMatcher()
            with mock.patch.object(adapter, "_ensure_models") as load:
                for filename, mode, size, image_format in cases:
                    with self.subTest(filename=filename):
                        path = Path(temporary) / filename
                        Image.new(mode, size).save(path, format=image_format)
                        with self.assertRaises(ValueError):
                            adapter.extract(path)
            load.assert_not_called()

    def test_extract_rejects_noncanonical_feature_metadata(self):
        with TemporaryDirectory() as temporary:
            adapter = self.fake_adapter(temporary)
            source = Path(temporary) / "source.png"
            Image.new("RGB", (512, 512)).save(source)
            for features in (
                {"keypoints": torch.tensor([[[10, 20]]]), "image_size": torch.tensor([[1024, 1024]])},
                {"keypoints": torch.tensor([[10, 20]]), "image_size": torch.tensor([[512, 512]])},
                {"keypoints": torch.tensor([[[512, 20]]]), "image_size": torch.tensor([[512, 512]])},
            ):
                with self.subTest(features=features):
                    with mock.patch.object(adapter.extractor, "extract", return_value=features):
                        with self.assertRaises(ValueError):
                            adapter.extract(source)

    def test_rectangular_png_extract_preserves_geometry_and_resize_none(self):
        with TemporaryDirectory() as temporary:
            adapter = self.fake_adapter(temporary)
            source = Path(temporary) / "source.png"
            Image.new("RGB", (512, 384), (0, 127, 255)).save(source)
            features = adapter.extract(source)
            image, kwargs = adapter.extractor.extraction_calls[-1]
            self.assertEqual(tuple(image.shape), (3, 384, 512))
            self.assertEqual(kwargs, {"resize": None})
            torch.testing.assert_close(features["image_size"], torch.tensor([[512.0, 384.0]]))
            result = adapter.match(features, features)
            self.assertEqual((result.canonical_width, result.canonical_height), (512, 384))

    def test_rectangular_extract_rejects_square_metadata_and_outside_keypoints(self):
        with TemporaryDirectory() as temporary:
            adapter = self.fake_adapter(temporary)
            source = Path(temporary) / "source.png"
            Image.new("RGB", (512, 384)).save(source)
            for features in (
                {"keypoints": torch.tensor([[[10, 20]]]), "image_size": torch.tensor([[512, 512]])},
                {"keypoints": torch.tensor([[[10, 384]]]), "image_size": torch.tensor([[512, 384]])},
                {"keypoints": torch.tensor([[[512, 20]]]), "image_size": torch.tensor([[512, 384]])},
            ):
                with self.subTest(features=features):
                    with mock.patch.object(adapter.extractor, "extract", return_value=features):
                        with self.assertRaises(ValueError):
                            adapter.extract(source)

    def test_pair_geometry_mismatch_fails_before_model_loading(self):
        adapter = LocalFidelityMatcher()
        source = {"keypoints": torch.empty((1, 0, 2)), "image_size": torch.tensor([[512, 512]])}
        relit = {"keypoints": torch.empty((1, 0, 2)), "image_size": torch.tensor([[512, 384]])}
        with mock.patch.object(adapter, "_ensure_models") as load:
            with self.assertRaisesRegex(ValueError, "dimensions must be identical"):
                adapter.match(source, relit)
        load.assert_not_called()

    def test_missing_or_invalid_pair_geometry_fails_before_loading(self):
        adapter = LocalFidelityMatcher()
        for image_size in (None, torch.tensor([512, 384]), torch.tensor([[512.5, 384]]), torch.tensor([[512, float("inf")]]), torch.tensor([[512, 0]])):
            features = {"keypoints": torch.empty((1, 0, 2))}
            if image_size is not None:
                features["image_size"] = image_size
            with self.subTest(image_size=image_size), mock.patch.object(adapter, "_ensure_models") as load:
                with self.assertRaises(ValueError):
                    adapter.match(features, features)
            load.assert_not_called()

    def test_empty_rectangular_extraction_keeps_pair_geometry(self):
        with TemporaryDirectory() as temporary:
            adapter = self.fake_adapter(temporary)
            features = {"keypoints": torch.empty((1, 0, 2)), "image_size": torch.tensor([[512, 384]])}
            result = adapter.match(features, features)
            self.assertEqual((result.canonical_width, result.canonical_height), (512, 384))
            self.assertEqual(adapter.matcher.calls, [])

    def test_match_extracts_matched_indices_scores_into_numpy(self):
        with TemporaryDirectory() as temporary:
            adapter = self.fake_adapter(temporary)
            source = {"keypoints": torch.tensor([[[10.0, 20.0], [100.0, 120.0]]]), "image_size": torch.tensor([[512, 512]])}
            relit = {"keypoints": torch.tensor([[[103.0, 120.0], [10.0, 24.0]]]), "image_size": torch.tensor([[512, 512]])}
            result = adapter.match(source, relit)
            self.assertIsInstance(result, MatchResult)
            self.assertEqual(result.num_keypoints_source, 2)
            self.assertEqual(result.num_keypoints_relit, 2)
            np.testing.assert_array_equal(result.points_source, [[10, 20], [100, 120]])
            np.testing.assert_array_equal(result.points_relit, [[10, 24], [103, 120]])
            np.testing.assert_allclose(result.match_scores, [0.8, 0.6])
            np.testing.assert_array_equal(result.displacements, [4, 3])
            for array in (result.points_source, result.points_relit, result.match_scores):
                self.assertIsInstance(array, np.ndarray)
            self.assertIs(adapter.matcher.calls[0]["image0"], source)
            self.assertIs(adapter.matcher.calls[0]["image1"], relit)
            source["keypoints"].fill_(0)
            np.testing.assert_array_equal(result.points_source, [[10, 20], [100, 120]])

    def test_empty_extraction_skips_matcher_and_returns_shaped_arrays(self):
        with TemporaryDirectory() as temporary:
            adapter = self.fake_adapter(temporary)
            empty = {"keypoints": torch.empty((1, 0, 2)), "image_size": torch.tensor([[512, 512]])}
            other = {"keypoints": torch.tensor([[[10.0, 20.0]]]), "image_size": torch.tensor([[512, 512]])}
            result = adapter.match(empty, other)
            self.assertEqual(result.points_source.shape, (0, 2))
            self.assertEqual(result.points_relit.shape, (0, 2))
            self.assertEqual(result.match_scores.shape, (0,))
            self.assertEqual(adapter.matcher.calls, [])
            self.assertEqual(result.num_keypoints_relit, 1)

    def test_empty_match_prediction_is_supported(self):
        with TemporaryDirectory() as temporary:
            adapter = self.fake_adapter(temporary)
            adapter.matcher.prediction = {"matches": [torch.empty((0, 2), dtype=torch.int64)], "scores": [torch.empty(0)]}
            features = {"keypoints": torch.tensor([[[10.0, 20.0]]]), "image_size": torch.tensor([[512, 512]])}
            result = adapter.match(features, features)
            self.assertEqual(result.points_source.shape, (0, 2))
            self.assertEqual(result.num_keypoints_source, 1)

    def test_invalid_match_indices_fail_instead_of_misreporting_coordinates(self):
        with TemporaryDirectory() as temporary:
            adapter = self.fake_adapter(temporary)
            features = {"keypoints": torch.tensor([[[10.0, 20.0]]]), "image_size": torch.tensor([[512, 512]])}
            for matches in (torch.tensor([[-1, 0]]), torch.tensor([[0, 1]]), torch.tensor([[0.0, 0.0]])):
                adapter.matcher.prediction = {"matches": [matches], "scores": [torch.tensor([0.5])]}
                with self.subTest(matches=matches):
                    with self.assertRaises(ValueError):
                        adapter.match(features, features)

    def test_cache_inside_protected_trees_is_rejected_before_import_or_download(self):
        root = Path(__file__).resolve().parents[1]
        adapter = LocalFidelityMatcher(LocalFidelityConfig(device="cpu"))
        for name in ("third_party", "salad"):
            with self.subTest(root=name), \
                 mock.patch.object(torch.hub, "get_dir", return_value=str(root / name / "cache")), \
                 mock.patch("countermine.probe.local_fidelity._import_vendored_models") as imports:
                with self.assertRaisesRegex(ValueError, "outside read-only"):
                    adapter._ensure_models()
                imports.assert_not_called()

    def test_vendored_import_uses_required_source_and_disables_bytecode(self):
        root = Path(__file__).resolve().parents[1] / "third_party" / "LightGlue" / "lightglue"
        aliked = SimpleNamespace(__file__=str(root / "aliked.py"), ALIKED=_FakeALIKED)
        lightglue = SimpleNamespace(__file__=str(root / "lightglue.py"), LightGlue=_FakeLightGlue)
        original = sys.dont_write_bytecode
        calls = []

        def fake_import(name):
            self.assertTrue(sys.dont_write_bytecode)
            calls.append(name)
            return aliked if name.endswith(".aliked") else lightglue

        with mock.patch.dict(sys.modules), mock.patch("countermine.probe.local_fidelity.importlib.import_module", side_effect=fake_import):
            self.assertEqual(_import_vendored_models(), (_FakeALIKED, _FakeLightGlue))
        self.assertEqual(calls, ["_countermine_vendored_lightglue.aliked", "_countermine_vendored_lightglue.lightglue"])
        self.assertEqual(sys.dont_write_bytecode, original)


if __name__ == "__main__":
    unittest.main()

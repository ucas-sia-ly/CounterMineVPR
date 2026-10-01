"""CPU-only cross-place support and visual audit checks; no model weights."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from countermine.probe.local_fidelity import LocalFidelityConfig, MatchResult
from countermine.probe.step3a_local import (
    Step3ALocalMatcher,
    compute_cross_place_metrics,
    cross_place_metrics_from_match,
    normalized_matched_cell_coverage,
)
from countermine.probe.step3a_visuals import (
    FIGURE_NAMES,
    MONTAGE_NAME,
    create_step3a_visual_outputs,
    create_top_margin_collapse_montage,
    select_top_margin_collapse,
)

try:
    import torch
except ImportError:
    torch = None


class TestCrossPlaceSupport(unittest.TestCase):
    def test_ratio_and_independent_normalized_cell_coverage(self):
        result = compute_cross_place_metrics(
            [[0, 0], [79, 59], [80, 0]],
            [[639, 479], [0, 0], [320, 240]],
            6, 10,
        )
        self.assertEqual(result["num_matches"], 3)
        self.assertEqual(result["cross_match_ratio"], 0.5)
        self.assertEqual(result["matched_source_cell_coverage"], 2 / 64)
        self.assertEqual(result["matched_target_cell_coverage"], 3 / 64)
        self.assertEqual(set(result), {
            "num_keypoints_source", "num_keypoints_target", "num_matches",
            "cross_match_ratio", "matched_source_cell_coverage", "matched_target_cell_coverage",
        })

    def test_coverage_uses_each_image_dimensions_and_half_open_boundaries(self):
        points = [[0, 0], [39.999, 29.999], [40, 30], [319.999, 239.999]]
        self.assertEqual(normalized_matched_cell_coverage(points, width=320, height=240), 3 / 64)
        scaled = np.asarray(points) * [2, 2]
        self.assertEqual(normalized_matched_cell_coverage(scaled), 3 / 64)
        centers = [[80 * x + 40, 60 * y + 30] for x in range(8) for y in range(8)]
        self.assertEqual(normalized_matched_cell_coverage(centers), 1.0)

    def test_match_result_does_not_read_displacements_or_scores(self):
        result = MatchResult(
            2, 5, np.array([[1, 1], [81, 61]]), np.array([[639, 479], [0, 0]]),
            np.array([0.7, 0.9]), 640, 480,
        )
        with mock.patch.object(MatchResult, "displacements", new_callable=mock.PropertyMock,
                               side_effect=AssertionError("Cross-place displacement is forbidden")):
            measured = cross_place_metrics_from_match(result)
        self.assertEqual(measured["cross_match_ratio"], 1.0)
        self.assertEqual(measured["matched_target_cell_coverage"], 2 / 64)

    def test_empty_support_has_zero_ratio_and_coverages(self):
        for counts in ((0, 10), (2, 0), (0, 0), (10, 10)):
            with self.subTest(counts=counts):
                result = compute_cross_place_metrics(np.empty((0, 2)), np.empty((0, 2)), *counts)
                self.assertEqual(result["num_matches"], 0)
                self.assertEqual(result["cross_match_ratio"], 0.0)
                self.assertEqual(result["matched_source_cell_coverage"], 0.0)
                self.assertEqual(result["matched_target_cell_coverage"], 0.0)

    def test_invalid_counts_shapes_bounds_and_nonfinite_coordinates_rejected(self):
        for count in (-1, True, 1.2):
            with self.subTest(count=count), self.assertRaises(ValueError):
                compute_cross_place_metrics([[0, 0]], [[0, 0]], count, 2)
        for points in ([[640, 1]], [[1, 480]], [[-1, 0]], [[float("nan"), 0]],
                       [[0, float("inf")]], [0, 0], []):
            with self.subTest(points=points), self.assertRaises(ValueError):
                compute_cross_place_metrics(points, [[0, 0]], 2, 2)
        with self.assertRaises(ValueError):
            compute_cross_place_metrics([[0, 0], [1, 1]], [[2, 2], [3, 3]], 1, 2)
        with self.assertRaises(ValueError):
            compute_cross_place_metrics([[0, 0]], [[0, 0], [1, 1]], 2, 2)
        with self.assertRaises(ValueError):
            normalized_matched_cell_coverage([[0, 0]], width=0)

    def test_step3a_constructor_is_lazy_and_freezes_keypoint_count(self):
        matcher = Step3ALocalMatcher(LocalFidelityConfig(device="cpu"))
        self.assertIsNone(matcher.extractor)
        self.assertIsNone(matcher.matcher)
        with self.assertRaisesRegex(ValueError, "2048"):
            Step3ALocalMatcher(LocalFidelityConfig(max_keypoints=1024))


@unittest.skipIf(torch is None, "Torch is unavailable; pure NumPy checks still run")
class TestCrossPlaceMatcherAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = Step3ALocalMatcher(LocalFidelityConfig(device="cpu"))
        self.adapter._torch = torch
        self.adapter._ensure_models = mock.Mock()
        self.adapter.matcher = mock.Mock(return_value={
            "matches": [torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)],
            "scores": [torch.tensor([0.8, 0.6])],
        })
        self.source = {"keypoints": torch.tensor([[[1.0, 1.0], [80.0, 60.0]]]),
                       "image_size": torch.tensor([[640.0, 480.0]])}
        self.target = {"keypoints": torch.tensor([[[639.0, 479.0], [320.0, 240.0]]]),
                       "image_size": torch.tensor([[640.0, 480.0]])}

    def test_cross_place_matching_bypasses_same_coordinate_fidelity(self):
        with mock.patch("countermine.probe.local_fidelity.compute_fidelity_metrics",
                        side_effect=AssertionError("No cross-place fidelity")), \
             mock.patch.object(MatchResult, "displacements", new_callable=mock.PropertyMock,
                               side_effect=AssertionError("No cross-place displacement")):
            result = self.adapter.match_cross_place(self.source, self.target)
        np.testing.assert_array_equal(result.points_source, [[1, 1], [80, 60]])
        np.testing.assert_array_equal(result.points_relit, [[320, 240], [639, 479]])
        self.assertEqual(cross_place_metrics_from_match(result)["cross_match_ratio"], 1.0)
        self.assertIs(self.adapter.matcher.call_args.args[0]["image0"], self.source)
        self.assertIs(self.adapter.matcher.call_args.args[0]["image1"], self.target)

    def test_native_geometry_required_before_model_loading(self):
        self.target["image_size"] = torch.tensor([[512.0, 384.0]])
        with self.assertRaisesRegex(ValueError, "640x480"):
            self.adapter.match_cross_place(self.source, self.target)
        self.adapter._ensure_models.assert_not_called()
        self.adapter.matcher.assert_not_called()

    def test_empty_keypoints_skip_matcher(self):
        self.source["keypoints"] = torch.empty((1, 0, 2))
        result = self.adapter.match_cross_place(self.source, self.target)
        self.assertEqual(result.points_source.shape, (0, 2))
        self.assertEqual(result.num_keypoints_relit, 2)
        self.adapter.matcher.assert_not_called()

    def test_bad_match_indices_and_nonfinite_scores_rejected(self):
        for indices, scores in (
            (torch.tensor([[-1, 0]]), torch.tensor([0.5])),
            (torch.tensor([[0, 2]]), torch.tensor([0.5])),
            (torch.tensor([[0.0, 0.0]]), torch.tensor([0.5])),
            (torch.tensor([[0, 0]]), torch.tensor([float("nan")])),
        ):
            self.adapter.matcher.return_value = {"matches": [indices], "scores": [scores]}
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.adapter.match_cross_place(self.source, self.target)


class TestStep3AVisualAudit(unittest.TestCase):
    @staticmethod
    def row(query="q", delta=-0.1, passed=True):
        return {
            "q_image_id": query, "p_image_id": "p", "n_hard_image_id": "hard",
            "n_random_image_id": "random", "all_images_R8_pass": passed,
            "s_qp": 0.8, "s_qhard": 0.6, "sz_qp": 0.7, "sz_qhard": 0.6,
            "margin_hard_rgb": 0.2, "margin_hard_z": 0.1,
            "delta_margin_hard": delta, "delta_margin_random": 0.01,
            "delta_cross_match_ratio_hard": 0.12,
            "q_source_relit_R8": 0.8, "p_source_relit_R8": 0.9,
            "n_hard_source_relit_R8": 0.7,
        }

    def test_margin_only_selection_pass_filter_and_deterministic_ties(self):
        rows = [self.row("z", -0.2), self.row("fail", -0.9, False),
                self.row("b", -0.3), self.row("a", -0.3)]
        selected = select_top_margin_collapse(rows, top_n=2)
        self.assertEqual([row["q_image_id"] for row in selected], ["a", "b"])
        self.assertEqual(select_top_margin_collapse([self.row(passed=False)]), [])
        with self.assertRaises(ValueError):
            select_top_margin_collapse([self.row(delta=float("nan"))])

    def test_all_four_artifacts_native_ratio_and_source_preservation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            lookup = {}
            for image_id, color in (("q", "red"), ("p", "green"), ("hard", "blue"), ("random", "yellow")):
                paths = {}
                for variant in ("source", "relit"):
                    path = root / f"{image_id}_{variant}.png"
                    Image.new("RGB", (640, 480), color).save(path)
                    paths[f"{variant}_path"] = path
                lookup[image_id] = paths
            checksums = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                         for paths in lookup.values() for path in paths.values()}
            outputs = create_step3a_visual_outputs([self.row()], lookup, root / "output")
            self.assertEqual(set(outputs), {*FIGURE_NAMES, MONTAGE_NAME})
            for path in outputs.values():
                with Image.open(path) as image:
                    self.assertGreater(image.width, 0)
                    self.assertGreater(image.height, 0)
            with Image.open(outputs[MONTAGE_NAME]) as image:
                self.assertEqual(image.size, (2004, 463))
                # Row panels are in RGB q/p/hard then relit q/p/hard order.
                expected = [(255, 0, 0), (0, 128, 0), (0, 0, 255)] * 2
                for column, color in enumerate(expected):
                    pixel = image.getpixel((12 + column * 332 + 160, 84 + 120))
                    self.assertTrue(all(abs(a - b) <= 3 for a, b in zip(pixel, color)))
            self.assertEqual(checksums, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in checksums})
            with self.assertRaisesRegex(ValueError, "overwrite"):
                create_top_margin_collapse_montage([self.row()], lookup, lookup["random"]["source_path"])

    def test_montage_rejects_legacy_square_geometry(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "square.png"
            Image.new("RGB", (512, 512)).save(path)
            lookup = {key: {"source_path": path, "relit_path": path} for key in ("q", "p", "hard")}
            with self.assertRaisesRegex(ValueError, "640x480"):
                create_top_margin_collapse_montage([self.row()], lookup, root / MONTAGE_NAME)
            self.assertFalse((root / MONTAGE_NAME).exists())


if __name__ == "__main__":
    unittest.main()

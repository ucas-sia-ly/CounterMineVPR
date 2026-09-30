"""CPU-only arithmetic for scale-fair geometry fidelity; no model weights."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np

from countermine.probe.geometry_audit import (
    ORIGINAL_METRIC_COLUMNS, compute_original_pixel_metrics, plot_fidelity,
)


SQUARE_SCALE = 512 / 480
FULL_SCALE = 512 / 640


class OriginalPixelFidelityTest(unittest.TestCase):
    def metrics(self, source, relit, scale, n_source=None, n_relit=None):
        source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
        relit = np.asarray(relit, dtype=np.float64).reshape(-1, 2)
        return compute_original_pixel_metrics(
            source, relit, len(source) if n_source is None else n_source,
            len(relit) if n_relit is None else n_relit, scale_x=scale, scale_y=scale,
        )

    def test_eight_canonical_pixels_are_7_5_original_pixels_in_square_policy(self):
        result = self.metrics([[128, 128]], [[136, 128]], SQUARE_SCALE)
        self.assertEqual(tuple(result), ORIGINAL_METRIC_COLUMNS)
        self.assertEqual(result["displacement_original_mean"], 7.5)
        self.assertEqual(result["displacement_original_q95"], 7.5)
        self.assertEqual(result["repeatability_original_8px"], 1.0)
        self.assertEqual(result["precision_original_8px"], 1.0)

    def test_eight_canonical_pixels_are_10_original_pixels_in_full_fov_policy(self):
        result = self.metrics([[128, 128]], [[136, 128]], FULL_SCALE)
        self.assertEqual(result["displacement_original_mean"], 10.0)
        self.assertEqual(result["displacement_original_q95"], 10.0)
        self.assertEqual(result["repeatability_original_8px"], 0.0)
        self.assertEqual(result["precision_original_8px"], 0.0)
        self.assertEqual(result["repeatability_original_16px"], 1.0)

    def test_exact_eight_original_pixels_have_identical_inclusive_classification(self):
        # The decimal coordinates expose rounding after division by 0.8: a
        # mathematically exact 8-pixel shift can evaluate as 8.000000000000057.
        # The intended inclusive boundary must be stable under both policies.
        for original_source, original_shift in (
            ([0.0, 0.0], [8.0, 0.0]),
            ([32.0, 24.0], [4.8, 6.4]),
            ([450.1, 400.7], [0.0, 8.0]),
            ([470.1, 460.7], [8.0, 0.0]),
        ):
            for scale in (SQUARE_SCALE, FULL_SCALE):
                with self.subTest(source=original_source, shift=original_shift, scale=scale):
                    source = np.array([original_source]) * scale
                    relit = (np.array([original_source]) + original_shift) * scale
                    result = self.metrics(source, relit, scale)
                    self.assertAlmostEqual(result["displacement_original_q95"], 8.0, places=12)
                    self.assertEqual(result["repeatability_original_4px"], 0.0)
                    self.assertEqual(result["repeatability_original_8px"], 1.0)
                    self.assertEqual(result["precision_original_8px"], 1.0)
                    self.assertEqual(result["repeatability_original_16px"], 1.0)

    def test_displacement_meaningfully_above_threshold_is_not_rounded_into_pass(self):
        original_source = np.array([[450.1, 400.7]])
        for scale in (SQUARE_SCALE, FULL_SCALE):
            with self.subTest(scale=scale):
                result = self.metrics(original_source * scale,
                                      (original_source + [0, 8 + 1e-8]) * scale, scale)
                self.assertGreater(result["displacement_original_q95"], 8.0)
                self.assertEqual(result["repeatability_original_8px"], 0.0)
                self.assertEqual(result["precision_original_8px"], 0.0)

    def test_denominator_uses_minimum_keypoint_count_and_precision_uses_matches(self):
        original_source = np.array([[20, 30], [50, 70]], dtype=float)
        original_relit = original_source + [[2, 0], [4, 0]]
        for scale in (SQUARE_SCALE, FULL_SCALE):
            with self.subTest(scale=scale):
                result = self.metrics(original_source * scale, original_relit * scale,
                                      scale, n_source=4, n_relit=6)
                self.assertEqual(result["repeatability_original_2px"], .25)
                self.assertEqual(result["precision_original_2px"], .5)
                for epsilon in (4, 8, 16):
                    self.assertEqual(result[f"repeatability_original_{epsilon}px"], .5)
                    self.assertEqual(result[f"precision_original_{epsilon}px"], 1.0)
                swapped = self.metrics(original_source * scale, original_relit * scale,
                                       scale, n_source=6, n_relit=4)
                self.assertEqual(result, swapped)

    def test_empty_matches_use_null_displacement_statistics_and_zero_ratios(self):
        for counts in ((0, 0), (0, 5), (5, 0), (5, 9)):
            with self.subTest(counts=counts):
                result = self.metrics([], [], FULL_SCALE, *counts)
                for key, value in result.items():
                    if key.startswith("displacement_original_"):
                        self.assertIsNone(value)
                    else:
                        self.assertEqual(value, 0.0)
                json.dumps(result, allow_nan=False)

    def test_original_displacement_quantiles_use_linear_interpolation(self):
        original_source = np.full((5, 2), 50.0)
        original_relit = original_source.copy()
        original_relit[:, 0] += [0, 2, 4, 8, 16]
        expected = {"mean": 6.0, "median": 4.0, "q75": 8.0,
                    "q90": 12.8, "q95": 14.4, "max": 16.0}
        for scale in (SQUARE_SCALE, FULL_SCALE):
            with self.subTest(scale=scale):
                result = self.metrics(original_source * scale, original_relit * scale, scale)
                for suffix, value in expected.items():
                    self.assertAlmostEqual(result[f"displacement_original_{suffix}"], value)

    def test_shared_crop_translation_cancels_without_registration(self):
        source = np.array([[80, 64], [320, 256]], dtype=float)
        relit = source + [[3, 4], [0, 8]]
        baseline = self.metrics(source, relit, SQUARE_SCALE)
        translated = self.metrics(source + [73.25, 27.5], relit + [73.25, 27.5], SQUARE_SCALE)
        for key, value in baseline.items():
            self.assertAlmostEqual(translated[key], value)

    def test_anisotropic_nonfinite_and_nonpositive_scales_are_rejected(self):
        valid = np.array([[10.0, 20.0]])
        for scale_x, scale_y in ((1.0, 1.01), (SQUARE_SCALE, FULL_SCALE),
                                 (0, 0), (-1, -1), (float("nan"), 1),
                                 (1, float("inf")), (True, True)):
            with self.subTest(scale_x=scale_x, scale_y=scale_y):
                with self.assertRaises(ValueError):
                    compute_original_pixel_metrics(valid, valid, 1, 1,
                                                   scale_x=scale_x, scale_y=scale_y)

    def test_numerically_isotropic_scale_within_tolerance_is_accepted(self):
        result = compute_original_pixel_metrics([[10, 20]], [[10, 20]], 1, 1,
                                                scale_x=1.0, scale_y=1.0 + 5e-13)
        self.assertEqual(result["displacement_original_q95"], 0.0)

    def test_invalid_points_and_counts_are_rejected(self):
        for source, relit, n_source, n_relit in (
            ([[10, float("nan")]], [[10, 20]], 1, 1),
            ([[10, 20]], [[float("inf"), 20]], 1, 1),
            ([[10, 20]], [], 1, 1),
            ([[10, 20]], [[10, 20]], 0, 1),
            ([[10, 20]], [[10, 20]], 1.5, 1),
            ([[10, 20]], [[10, 20]], True, 1),
        ):
            with self.subTest(source=source, relit=relit, counts=(n_source, n_relit)):
                with self.assertRaises(ValueError):
                    self.metrics(source, relit, FULL_SCALE, n_source, n_relit)

    def test_coordinate_arrays_and_keypoint_counts_are_not_modified(self):
        source = np.array([[20, 30], [70, 80]], dtype=np.float64)
        relit = source + [[3, 4], [0, 8]]
        before_source, before_relit = source.copy(), relit.copy()
        self.metrics(source, relit, FULL_SCALE, n_source=100, n_relit=200)
        np.testing.assert_array_equal(source, before_source)
        np.testing.assert_array_equal(relit, before_relit)


class OriginalPixelPlotTest(unittest.TestCase):
    def test_main_plot_reads_only_fair_metrics_and_labels_original_image_pixels(self):
        from matplotlib.axes import Axes

        rows = []
        for index in range(10):
            for policy in ("square_crop_512", "full_fov_512"):
                rows.append({"audit_index": index, "row_index": 100 + index, "policy": policy,
                             "repeatability_original_4px": .2,
                             "repeatability_original_8px": .7,
                             "displacement_original_q95": 3.5,
                             # Deliberately different legacy values expose
                             # accidental use of output-pixel plot inputs.
                             "repeatability_min_4px": .99, "repeatability_min_8px": .98,
                             "displacement_q95": 99.0})
        plotted, labels = [], []
        original_plot = Axes.plot
        original_title = Axes.set_title
        original_ylabel = Axes.set_ylabel

        def record_plot(axis, x, y, *args, **kwargs):
            plotted.append(list(y))
            return original_plot(axis, x, y, *args, **kwargs)

        def record_title(axis, label, *args, **kwargs):
            labels.append(label)
            return original_title(axis, label, *args, **kwargs)

        def record_ylabel(axis, label, *args, **kwargs):
            labels.append(label)
            return original_ylabel(axis, label, *args, **kwargs)

        with TemporaryDirectory() as temporary, \
             mock.patch.object(Axes, "plot", record_plot), \
             mock.patch.object(Axes, "set_title", record_title), \
             mock.patch.object(Axes, "set_ylabel", record_ylabel):
            destination = Path(temporary) / "fair_plot.png"
            plot_fidelity(rows, destination)
            self.assertGreater(destination.stat().st_size, 100)
        self.assertEqual(plotted, [[.2] * 10, [.2] * 10, [.7] * 10, [.7] * 10,
                                   [3.5] * 10, [3.5] * 10])
        self.assertGreaterEqual(sum("original-image pixels" in label for label in labels), 3)


if __name__ == "__main__":
    unittest.main()

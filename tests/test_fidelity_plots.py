"""CPU-only synthetic checks for bounded canonical-coordinate diagnostics."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from countermine.probe import fidelity_plots as plots


class FidelityPlotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def assert_readable(self, path, expected_format):
        with Image.open(path) as image:
            image.load()
            self.assertEqual(image.format, expected_format)
            self.assertGreater(image.width, 0)
            self.assertGreater(image.height, 0)

    def test_overlay_categories_use_closed_upper_boundaries(self):
        np.testing.assert_array_equal(
            plots.displacement_categories([0, 4, 4.001, 8, 8.001, 16, 16.001, 700]),
            [0, 0, 1, 1, 2, 2, -1, -1],
        )

    def test_match_selection_is_capped_deterministic_and_rejects_large_failures(self):
        source = np.column_stack((np.arange(200, dtype=float), np.ones(200)))
        relit = source + [4, 0]
        relit[-3:] += [17, 0]
        scores = np.ones(200)
        first = plots.select_visualization_matches(source, relit, scores)
        second = plots.select_visualization_matches(source, relit, scores)
        np.testing.assert_array_equal(first, np.arange(80))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(np.linalg.norm(source[first] - relit[first], axis=1) <= 16))
        self.assertEqual(len(plots.select_visualization_matches(source, relit, scores, 5)), 5)
        self.assertEqual(len(plots.select_visualization_matches(source, relit, scores, 0)), 0)

    def test_match_selection_ranks_score_then_displacement_then_coordinates(self):
        source = np.array([[20, 10], [10, 10], [5, 10], [1, 1], [1, 1]], dtype=float)
        relit = source + np.array([[8, 0], [8, 0], [16, 0], [4, 0], [4, 0]])
        scores = np.array([0.8, 0.8, 0.9, 0.8, 0.8])
        np.testing.assert_array_equal(
            plots.select_visualization_matches(source, relit, scores), [2, 3, 4, 1, 0])

    def test_empty_matches_and_bad_inputs(self):
        self.assertEqual(len(plots.select_visualization_matches(
            np.empty((0, 2)), np.empty((0, 2)), np.empty(0))), 0)
        with self.assertRaises(ValueError):
            plots.select_visualization_matches(np.zeros((1, 2)), np.zeros((1, 2)), [1], -1)
        with self.assertRaises(ValueError):
            plots.select_visualization_matches(np.zeros((1, 2)), np.zeros((2, 2)), [1])
        with self.assertRaises(ValueError):
            plots.displacement_categories([np.nan])

    def test_repeatability_plot_uses_only_both_relit_modes(self):
        records = []
        for mode in (*plots.RELIGHT_MODES, "identity_control"):
            for source in range(3):
                records.append({"mode": mode, **{
                    f"repeatability_min_{epsilon}px": (source + 1) / 4
                    for epsilon in plots.EPSILONS}})
        # Identity is intentionally invalid; touching its values would fail.
        records[-1]["repeatability_min_2px"] = float("nan")
        pyplot = plots._pyplot()
        real_subplots = pyplot.subplots
        captured = []

        def capture(*args, **kwargs):
            figure, axis = real_subplots(*args, **kwargs)
            captured.append(axis)
            return figure, axis

        path = self.root / "plots/repeatability.png"
        with mock.patch.object(pyplot, "subplots", side_effect=capture):
            plots.plot_repeatability(records, path)
        self.assert_readable(path, "PNG")
        labels = captured[0].get_legend_handles_labels()[1]
        self.assertEqual(labels, ["Official RMBG (n=3 sources)", "Full scene (n=3 sources)"])

    def test_cdf_is_per_match_retains_large_displacements_and_excludes_identity(self):
        pyplot = plots._pyplot()
        real_subplots = pyplot.subplots
        captured = []

        def capture(*args, **kwargs):
            figure, axis = real_subplots(*args, **kwargs)
            captured.append(axis)
            return figure, axis

        path = self.root / "cdf.png"
        samples = {"official_rmbg": np.array([700, 4, 0, 4]),
                   "full_scene": np.array([1, 16]),
                   "identity_control": np.array([np.nan])}
        with mock.patch.object(pyplot, "subplots", side_effect=capture):
            plots.plot_displacement_cdf(samples, path)
        self.assert_readable(path, "PNG")
        axis = captured[0]
        self.assertEqual(len(axis.lines), 2)
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), [0, 0, 4, 4, 700])
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), [0, 0.25, 0.5, 0.75, 1])
        self.assertGreater(axis.get_xlim()[1], 700)

    def test_worst_examples_render_source_relit_overlay_with_bounded_matches(self):
        source_path, relit_path = self.root / "source.png", self.root / "relit.png"
        Image.new("RGB", (512, 512), (20, 50, 80)).save(source_path)
        Image.new("RGB", (512, 512), (60, 90, 100)).save(relit_path)
        source = np.array([[50, 50], [100, 100], [200, 200], [300, 300]], dtype=float)
        relit = source + [[4, 0], [8, 0], [16, 0], [25, 0]]
        example = {"record": {"audit_index": 1, "row_index": 8,
                              "repeatability_min_8px": 0.2},
                   "source_path": source_path, "relit_path": relit_path,
                   "points_source": source, "points_relit": relit,
                   "match_scores": np.ones(4)}
        path = self.root / "worst.jpg"
        with mock.patch.object(plots, "select_visualization_matches",
                               wraps=plots.select_visualization_matches) as select:
            plots.plot_worst_examples({mode: [example] for mode in plots.RELIGHT_MODES}, path)
        self.assert_readable(path, "JPEG")
        self.assertEqual(select.call_count, 2)
        with Image.open(path) as image:
            self.assertGreater(image.height, 1024)
            self.assertGreater(image.width, 1536)
        with self.assertRaisesRegex(ValueError, "at most five"):
            plots.plot_worst_examples({"official_rmbg": [example] * 6}, path)

    def test_empty_modes_still_generate_readable_diagnostics(self):
        for function, data, filename, expected in (
            (plots.plot_repeatability, [], "empty_r.png", "PNG"),
            (plots.plot_displacement_cdf, {}, "empty_c.png", "PNG"),
            (plots.plot_worst_examples, {}, "empty_w.jpg", "JPEG"),
        ):
            with self.subTest(filename=filename):
                path = self.root / filename
                function(data, path)
                self.assert_readable(path, expected)


if __name__ == "__main__":
    unittest.main()

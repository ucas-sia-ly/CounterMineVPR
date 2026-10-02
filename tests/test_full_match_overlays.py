"""Optional visualization selection/score ordering stays CPU-testable."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from test_full_graph_analysis import full_fixture
from countermine.mining.full_graph_analysis import analyze_full_graphs


def overlay_module():
    path = Path(__file__).resolve().parents[1] / "tools/25_visualize_full_structural_matches.py"
    spec = importlib.util.spec_from_file_location("full_overlay_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OverlayTests(unittest.TestCase):
    def test_selection_is_unique_and_contains_top20_plus_q99_hub_representatives(self):
        tables = full_fixture()
        _, _, places, _ = analyze_full_graphs(*tables)
        selected = overlay_module().select_overlay_edges(tables[1], places)
        self.assertEqual(selected.pair_uid.tolist(), ["pair-3", "pair-1", "pair-0", "pair-2"])
        self.assertFalse(selected.pair_uid.duplicated().any())

    def test_overlay_displays_at_most100_in_descending_score_and_stable_tie_order(self):
        class Reader:
            def read(self, identity, width):
                return Image.new("RGB", (width, width * 3 // 4), "#eeeeee")
        row = full_fixture()[1].iloc[0].to_dict()
        scores = np.arange(120, dtype=float)
        scores[-2:] = 119
        points = np.column_stack((np.arange(120), np.full(120, 20)))
        match = SimpleNamespace(scores=scores, points_a=points, points_b=points)
        captured = []
        original = ImageDraw.ImageDraw.line
        def capture(self, xy, *args, **kwargs):
            captured.append(xy)
            return original(self, xy, *args, **kwargs)
        with patch.object(ImageDraw.ImageDraw, "line", capture):
            panel = overlay_module().overlay_panel(Reader(), row, match)
        panel.close()
        self.assertEqual(len(captured), 100)
        self.assertEqual(captured[0][0][0], 118)
        self.assertEqual(captured[1][0][0], 119)
        self.assertEqual(captured[-1][0][0], 20)

    def test_score_ordered_overlay_rejects_missing_matcher_scores(self):
        row = full_fixture()[1].iloc[0].to_dict()
        class Reader:
            def read(self, identity, width):
                return Image.new("RGB", (width, width * 3 // 4), "#eeeeee")
        with self.assertRaisesRegex(ValueError, "confidence scores"):
            overlay_module().overlay_panel(Reader(), row, SimpleNamespace(scores=None))


if __name__ == "__main__":
    unittest.main()

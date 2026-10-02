"""CPU-only complete graph aggregation, diagnostic flags and source binding."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from test_countermine_graph import fixture
from countermine.mining.full_countermine_graph import (
    CORE_SLICES, GRAPH_SOURCE_FILES, SLICES, build_full_image_graph, build_full_place_graph,
    load_full_graph, validate_full_calibrated, write_full_graph_artifacts,
)


class FullGraphTests(unittest.TestCase):
    def test_every_edge_retained_and_extra_slices_predeclared(self):
        candidates, manifest = fixture()
        nodes, edges = build_full_image_graph(candidates, manifest)
        self.assertEqual(len(edges), len(candidates))
        self.assertEqual(nodes.image_id.tolist(), ["a", "b", "c", "d"])
        self.assertEqual(edges.core_q95.tolist(), [True, True, False, True])
        self.assertEqual(edges.core_q975.tolist(), [False, True, False, True])
        self.assertEqual(edges.core_q99.tolist(), [False, True, False, True])
        self.assertEqual(edges.core_q995.tolist(), [False, False, False, True])
        self.assertEqual(edges.full_geo500.tolist(), [False, True, True, True])
        self.assertEqual(len(SLICES), 10)

    def test_zero_evidence_and_duplicate_edges_survive(self):
        candidates, manifest = fixture()
        candidates["ratio_null_percentile"] = 0
        candidates["match_count_null_percentile"] = 0
        candidates["structural_bottleneck"] = 0
        candidates["structural_geomean"] = 0
        candidates["exact_pixel_duplicate"] = True
        _, edges = build_full_image_graph(candidates, manifest)
        self.assertEqual(len(edges), 4)
        for flag in CORE_SLICES:
            self.assertFalse(edges[flag].any())

    def test_q975_q995_include_exact_boundary_and_exclude_duplicates(self):
        candidates, manifest = fixture()
        candidates.loc[0, ["ratio_null_percentile", "match_count_null_percentile", "structural_bottleneck", "structural_geomean"]] = .975
        candidates.loc[1, ["ratio_null_percentile", "match_count_null_percentile", "structural_bottleneck", "structural_geomean"]] = .995
        candidates.loc[3, "exact_pixel_duplicate"] = True
        _, edges = build_full_image_graph(candidates, manifest)
        self.assertTrue(edges.iloc[0].core_q975)
        self.assertTrue(edges.iloc[1].core_q995)
        self.assertFalse(edges.iloc[3].core_q995)

    def test_unordered_place_support_aligns_images_independent_of_image_order(self):
        candidates, manifest = fixture()
        images = build_full_image_graph(candidates, manifest)
        places, edges = build_full_place_graph(*images)
        self.assertEqual(places.place_uid.tolist(), ["alpha", "z"])
        row = edges.iloc[0]
        self.assertEqual((row.place_uid_a, row.place_uid_b), ("alpha", "z"))
        self.assertEqual(row.num_supporting_image_edges, 4)
        self.assertEqual(row.num_core_q95_image_edges, 3)
        self.assertEqual(row.num_core_q95_unique_images_a, 2)
        self.assertEqual(row.num_core_q95_unique_images_b, 2)
        self.assertTrue(row.core_q95_independent_support_3)
        self.assertEqual(row.num_core_q95_geo500_unique_images_a, 1)
        self.assertEqual(row.num_core_q95_geo500_unique_images_b, 2)
        self.assertFalse(row.core_q95_geo500_independent_support_2)
        self.assertEqual(row.num_matches_median, 35)
        self.assertAlmostEqual(row.local_match_ratio_median, .175)
        self.assertEqual(row.min_geo_distance_m, 300)
        self.assertEqual(row.num_full_geo500_image_edges, 3)

    def test_weak_views_do_not_count_as_core_independent_support(self):
        candidates, manifest = fixture()
        places, edges = build_full_place_graph(*build_full_image_graph(candidates.iloc[:3], manifest))
        self.assertEqual(edges.iloc[0].independent_view_support, 2)
        self.assertEqual(edges.iloc[0].core_q95_independent_view_support, 1)
        self.assertFalse(edges.iloc[0].core_q95_independent_support_2)

    def test_same_place_formula_nonfinite_relation_and_flags_fail_closed(self):
        candidates, manifest = fixture()
        changes = [("place_uid_a", candidates.iloc[0].place_uid_b, "same-place"),
                   ("structural_bottleneck", .3, "formula"),
                   ("ratio_null_percentile", np.inf, "finite"),
                   ("ratio_null_source", "global", "relation-specific"),
                   ("core_q975", True, "diagnostic slice"),
                   ("min_num_keypoints", 12, "endpoint counts")]
        for column, value, message in changes:
            with self.subTest(column=column):
                altered = candidates.copy()
                if column == "core_q975":
                    altered[column] = False
                if column == "min_num_keypoints":
                    altered[column] = 200
                altered.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, message):
                    build_full_image_graph(altered, manifest)

    def test_roundtrip_hashes_and_source_destination_guards(self):
        candidates, manifest = fixture()
        tables = (*build_full_image_graph(candidates, manifest),)
        tables = (*tables, *build_full_place_graph(*tables))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "cache/countermine_rgb/step2d"
            runtime.mkdir(parents=True)
            for source in GRAPH_SOURCE_FILES:
                path = root / source
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("project source", encoding="utf-8")
            for name in ("full_calibration_summary.json", "full_calibrated_candidates.csv"):
                (runtime / name).write_text("calibrated source", encoding="utf-8")
            summary = write_full_graph_artifacts(runtime, *tables, {}, repo_root=root)
            loaded = load_full_graph(runtime, repo_root=root, source_validation=False)
            self.assertEqual(loaded[1].core_q995.tolist(), tables[1].core_q995.tolist())
            self.assertEqual(loaded[3].iloc[0].num_matches_median, 35)
            self.assertEqual(summary["artifacts"]["image_edges.csv"]["count"], 4)
            path = runtime / "image_edges.csv"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_full_graph(runtime, repo_root=root, source_validation=False)
            with self.assertRaisesRegex(ValueError, "Step 2D writes"):
                write_full_graph_artifacts(root / "cache/countermine_rgb/step2b", *tables, {}, repo_root=root)


if __name__ == "__main__":
    unittest.main()

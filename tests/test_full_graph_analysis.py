"""CPU topology, hubs, denominator audits and portable full snapshot checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_countermine_graph import fixture
from countermine.mining.full_countermine_graph import build_full_image_graph, build_full_place_graph
from countermine.mining.full_graph_analysis import (
    ANALYSIS_SOURCE_FILES, analyze_full_graphs, build_full_snapshot, full_topology,
    select_full_top_edges, validate_full_snapshot,
)


def full_fixture():
    candidates, manifest = fixture()
    images = build_full_image_graph(candidates, manifest)
    return (*images, *build_full_place_graph(*images))


class FullAnalysisTests(unittest.TestCase):
    def test_ten_slices_topology_q99_degree_and_giant_fraction(self):
        images, edges, places, place_edges = full_fixture()
        report, image_audit, place_audit, components = analyze_full_graphs(images, edges, places, place_edges)
        self.assertEqual(len(report["image_graph"]), 10)
        self.assertEqual(report["image_graph"]["full"]["edge_count"], 4)
        self.assertEqual(report["place_graph"]["full"]["edge_count"], 1)
        self.assertEqual(report["image_graph"]["core_q995"]["active_node_count"], 2)
        self.assertEqual(report["image_graph"]["core_q995"]["isolated_node_count"], 2)
        self.assertEqual(report["image_graph"]["core_q995"]["largest_component_size"], 2)
        self.assertEqual(report["image_graph"]["core_q995"]["largest_component_fraction_of_active_nodes"], 1)
        self.assertIn("q99", report["place_graph"]["full"]["degree"])
        self.assertEqual(place_audit.structural_edge_fraction_core_q95.tolist(), [1., 1.])
        self.assertEqual(components["core_q995"][0]["size"], 2)

    def test_empty_core_has_fixed_isolates_and_no_active_giant_fraction(self):
        images, edges, places, place_edges = full_fixture()
        summary, degrees, components = full_topology(images, edges.iloc[:0], "image_id", ("image_id_a", "image_id_b"))
        self.assertEqual(summary["isolated_node_count"], 4)
        self.assertEqual(summary["largest_component_size"], 1)
        self.assertIsNone(summary["largest_component_fraction_of_active_nodes"])
        self.assertEqual(summary["degree"]["q99"], 0)
        self.assertEqual(len(components), 4)

    def test_hub_fraction_ranking_requires_five_full_place_neighbors(self):
        nodes = pd.DataFrame({"place_uid": ["center", *[f"neighbor-{i}" for i in range(6)], "low"], "city_id": "city"})
        edges = pd.DataFrame({"place_uid_a": "center", "place_uid_b": [f"neighbor-{i}" for i in range(6)]})
        summary, degrees, _ = full_topology(nodes, edges, "place_uid", ("place_uid_a", "place_uid_b"))
        self.assertEqual(degrees["center"], 6)
        self.assertEqual(summary["degree"]["max"], 6)
        tables = full_fixture()
        report, _, _, _ = analyze_full_graphs(*tables)
        self.assertEqual(report["hub_analysis"]["fraction_min_full_degree"], 5)
        self.assertEqual(report["hub_analysis"]["core_q95"]["top50_structural_edge_fraction"], [])

    def test_star_hub_degree_and_fraction_use_simple_place_neighbors(self):
        candidates, manifest = fixture()
        base = candidates.iloc[0].to_dict()
        rows, nodes = [], [{"image_id": "center", "row_index": 0, "place_uid": "center", "city_id": "city"}]
        for number in range(6):
            image = f"neighbor-{number}"
            nodes.append({"image_id": image, "row_index": number + 1, "place_uid": image, "city_id": "city"})
            row = base | {"pair_uid": f"pair-{number}", "image_id_a": "center", "image_id_b": image,
                          "place_uid_a": "center", "place_uid_b": image, "row_index_a": 0, "row_index_b": number + 1}
            if number >= 3:
                row.update(ratio_null_percentile=.5, match_count_null_percentile=.5,
                           structural_bottleneck=.5, structural_geomean=.5)
            rows.append(row)
        images = build_full_image_graph(pd.DataFrame(rows), pd.DataFrame(nodes))
        report, _, audit, _ = analyze_full_graphs(*images, *build_full_place_graph(*images))
        center = audit.loc[audit.place_uid == "center"].iloc[0]
        self.assertEqual(center.candidate_degree_full, 6)
        self.assertEqual(center.degree_core_q95, 3)
        self.assertEqual(center.structural_edge_fraction_core_q95, .5)
        ranked = report["hub_analysis"]["core_q95"]["top50_structural_edge_fraction"]
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["place_uid"], "center")

    def test_denominator_bins_rates_and_full_correlations(self):
        report, _, _, _ = analyze_full_graphs(*full_fixture())
        row = report["denominator_audit"]["by_bin"]["lt256"]
        self.assertEqual(row["pair_count"], 4)
        self.assertEqual(row["core_q95_fraction"], .75)
        self.assertEqual(row["core_q99_fraction"], .5)
        self.assertIn("q99", row["local_match_ratio"])
        self.assertEqual(report["denominator_audit"]["top100_raw_ratio"]["count_below_256"], 4)
        self.assertIn("pearson", report["candidate_structural_correlations"]["max_salad_similarity_vs_structural_bottleneck"])
        self.assertIsNone(report["candidate_structural_correlations"]["best_rgb_rank_vs_structural_bottleneck"]["pearson"])
        self.assertEqual(report["rank_analysis"]["rank_1"]["core_q975_count"], 2)
        self.assertIsNone(report["rank_analysis"]["rank_2_5"]["core_q99_fraction"])

    def test_top_edges_use_salad_similarity_before_uid_tie(self):
        _, edges, _, _ = full_fixture()
        selected = edges.iloc[[0, 1]].copy()
        selected[["structural_bottleneck", "structural_geomean", "num_matches"]] = [1., 1., 20]
        selected["max_salad_similarity"] = [.1, .9]
        self.assertEqual(select_full_top_edges(selected).pair_uid.tolist(), ["pair-1", "pair-0"])

    def test_frozen_random_rates_remain_fixed_and_zero_denominator_is_null(self):
        tables = full_fixture()
        null = {"q95_fraction": .1, "q99_fraction": 0., "count": 100}
        step2c = {"joint_null": {"rates": {name: {"random": null} for name in ("all", "same_city", "cross_city")}}}
        report, _, _, _ = analyze_full_graphs(*tables, step2c_snapshot=step2c)
        comparison = report["candidate_enrichment_consistency"]["all"]
        self.assertEqual(comparison["q95_rate_ratio"], 7.5)
        self.assertIsNone(comparison["q99_rate_ratio"])
        self.assertEqual(comparison["frozen_random"], null)

    def test_snapshot_rejects_nonfinite_paths_pixels_features_and_wrong_flags(self):
        valid = {"provenance": {"real_rgb_only": True, "synthetic_images_used": False}}
        validate_full_snapshot(valid)
        for child in ({"bad": np.nan}, {"bad": "/tmp/source"}, {"descriptors": []}, {"keypoints": []}, {"points_a": []}):
            with self.subTest(child=child), self.assertRaises(ValueError):
                validate_full_snapshot(valid | child)
        with self.assertRaisesRegex(ValueError, "real-RGB"):
            validate_full_snapshot({"provenance": {"real_rgb_only": True, "synthetic_images_used": True}})

    def test_snapshot_contains_required_hashes_counts_and_frozen_calibration(self):
        report, _, _, _ = analyze_full_graphs(*full_fixture())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "cache/countermine_rgb/step2d"
            sources = ["cache/gsv_mini/manifest.csv", "cache/gsv_mini/rgb_candidates_raw.csv",
                       "cache/countermine_rgb/step2d/full_population.csv", "cache/countermine_rgb/step2d/full_candidate_structural_metrics.csv",
                       "cache/countermine_rgb/step2d/full_calibration_summary.json", *ANALYSIS_SOURCE_FILES]
            for name in sources:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("frozen source", encoding="utf-8")
            bank = runtime / "aliked_bank/summary.json"
            bank.parent.mkdir(parents=True)
            bank.write_text('{"image_count":4}')
            graph = {"graph_code_hashes": {"graph.py": "hash"}, "artifacts": {
                name: {"sha256": "hash", "count": count} for name, count in (
                    ("image_nodes.csv", 4), ("image_edges.csv", 4), ("place_nodes.csv", 2), ("place_edges.csv", 1))}}
            calibration = {"provenance": {"code_sha256": {"calibration.py": "hash"}},
                "pilot_reproduction": {"passed": True, "pair_count": 5000},
                "relation_null_counts": {"same_city": 3900, "cross_city": 1099},
                "frozen_q95_q99_thresholds": {"same_city": {"ratio_q95_threshold": .05}}}
            with patch("countermine.mining.full_graph_analysis.frozen_provenance", return_value={"step2a_snapshot_sha256": "hash", "step2b_snapshot_sha256": "hash", "step2c_snapshot_sha256": "hash"}):
                snapshot = build_full_snapshot(report, {"population": {"raw_directed_candidate_rows": 100}},
                    {"shard_count": 1, "completed_shard_count": 1}, calibration, graph,
                    runtime_dir=runtime, repo_root=root, git_commit="commit", visual_metadata={})
            self.assertEqual(snapshot["population"]["canonical_edge_count"], 4)
            self.assertEqual(snapshot["measurement"]["feature_bank_image_count"], 4)
            self.assertEqual(snapshot["calibration"]["relation_counts"]["cross_city"], 1099)
            self.assertEqual(snapshot["provenance"]["calibration_code_hashes"], {"calibration.py": "hash"})
            self.assertNotIn(str(root), json.dumps(snapshot, allow_nan=False))


if __name__ == "__main__":
    unittest.main()

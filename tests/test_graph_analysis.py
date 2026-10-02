"""Small CPU graph fixtures for topology and scientific diagnostic boundaries."""

import copy
import unittest

import numpy as np
import pandas as pd

from countermine.mining.graph_analysis import (
    KEYPOINT_BINS, SLICES, UnionFind, analyze_graphs, analyze_topology,
    keypoint_bin, select_repeated_place_pairs, select_top_edges, validate_graph_snapshot,
)


def graph_fixture():
    image_nodes = pd.DataFrame({
        "image_id": list("abcde"), "row_index": range(5),
        "place_uid": ["P", "Q", "R", "Q", "S"], "city_id": ["X"] * 5,
    })
    image_edges = pd.DataFrame({
        "pair_uid": ["ab", "ac", "ad", "bc", "ce"],
        "image_id_a": ["a", "a", "a", "b", "c"], "image_id_b": ["b", "c", "d", "c", "e"],
        "place_uid_a": ["P", "P", "P", "Q", "R"], "place_uid_b": ["Q", "R", "Q", "R", "S"],
        "core_q95": [True, True, False, False, False], "core_q99": [True, False, False, False, False],
        "core_q95_geo500": [False, True, False, False, False], "core_q99_geo500": [False] * 5,
        "rank_bin": ["rank_1", "rank_2_5", "rank_6_10", "rank_11_20", "rank_21_50"],
        "best_rgb_rank": [1, 2, 6, 11, 21], "max_salad_similarity": [.8, .7, .6, .5, .4],
        "num_keypoints_a": [255, 256, 512, 1024, 1536], "num_keypoints_b": [2048] * 5,
        "num_matches": [100, 99, 50, 40, 30], "local_match_ratio": [.4, .3, .2, .1, .05],
        "structural_bottleneck": [.999, .96, .6, .5, .4], "structural_geomean": [.999, .97, .7, .6, .5],
        "symmetric_match_coverage": [.1, .3, .4, .5, .6], "symmetric_match_entropy": [.2, .4, .5, .6, .7],
        "source_max_cell_match_fraction": [.6, .2, .2, .2, .2],
        "target_max_cell_match_fraction": [.7, .3, .2, .2, .2],
        "geo_distance_m": [300., 1000., 1500., 2000., 3000.], "exact_pixel_duplicate": [False] * 5,
    })
    place_nodes = pd.DataFrame({"place_uid": list("PQRS"), "city_id": ["X"] * 4})
    place_edges = pd.DataFrame({
        "place_uid_a": ["P", "P", "Q", "R"], "place_uid_b": ["Q", "R", "R", "S"],
        "num_core_q95_image_edges": [1, 1, 0, 0], "num_core_q99_image_edges": [1, 0, 0, 0],
        "num_core_q95_geo500_image_edges": [0, 1, 0, 0], "num_core_q99_geo500_image_edges": [0] * 4,
        "structural_bottleneck_max": [.999, .96, .5, .4],
    })
    for name in ("repeated_support_2", "repeated_support_3", "independent_support_2", "independent_support_3",
                 "core_independent_support_2", "core_independent_support_3"):
        place_edges[name] = False
        place_edges[f"{name}_geo500"] = False
    return image_nodes, image_edges, place_nodes, place_edges


class TopologyTests(unittest.TestCase):
    def test_union_find_components_and_deterministic_order(self):
        union = UnionFind(list("fedcba"))
        for a, b in [("a", "b"), ("b", "c"), ("a", "c"), ("e", "d")]:
            union.union(a, b)
        self.assertEqual(union.components(), [list("abc"), list("de"), ["f"]])
        self.assertEqual(union.find("a"), union.find("c"))
        with self.assertRaises(ValueError):
            union.find("unknown")
        with self.assertRaises(ValueError):
            UnionFind(["x", "x"])

    def test_degrees_components_and_singleton_population(self):
        nodes = pd.DataFrame({"node": list("abcdef")})
        edges = pd.DataFrame({"a": list("abe"), "b": list("bcd")})
        summary, degree, components = analyze_topology(nodes, edges, "node", ("a", "b"))
        self.assertEqual(degree, {"a": 1, "b": 2, "c": 1, "d": 1, "e": 1, "f": 0})
        self.assertEqual(summary["node_count"], 6)
        self.assertEqual(summary["active_node_count"], 5)
        self.assertEqual(summary["connected_component_count"], 3)
        self.assertEqual(summary["non_singleton_component_count"], 2)
        self.assertAlmostEqual(summary["degree"]["mean"], 1.)
        self.assertEqual([group["size"] for group in components], [3, 2, 1])
        self.assertEqual([group["edge_count"] for group in components], [2, 1, 0])
        empty = edges.iloc[:0]
        report, degree, _ = analyze_topology(nodes, empty, "node", ("a", "b"))
        self.assertEqual(report["connected_component_count"], 6)
        self.assertEqual(report["active_node_count"], 0)
        self.assertEqual(report["component_sizes"]["max"], 1)
        self.assertEqual(sum(degree.values()), 0)

    def test_rejects_unknown_self_or_reverse_duplicate_edges(self):
        nodes = pd.DataFrame({"node": ["a", "b"]})
        for pairs in ([('a', 'a')], [('a', 'missing')], [('a', 'b'), ('b', 'a')]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                analyze_topology(nodes, pd.DataFrame(pairs, columns=["a", "b"]), "node", ("a", "b"))


class DiagnosticTests(unittest.TestCase):
    def test_frozen_keypoint_bins(self):
        values = [0, 255, 256, 511, 512, 1023, 1024, 1535, 1536, 2048]
        self.assertEqual([keypoint_bin(x) for x in values], [name for name in KEYPOINT_BINS for _ in range(2)])
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                keypoint_bin(value)

    def test_top_lists_preserve_ratio_ties_and_bottleneck_ties(self):
        frame = pd.DataFrame({
            "pair_uid": ["d", "c", "b", "a", "duplicate"],
            "local_match_ratio": [.5] * 5, "num_matches": [20, 20, 20, 30, 100],
            "max_salad_similarity": [.8, .8, .9, .1, 1.],
            "structural_bottleneck": [.9, .9, .9, .8, 1.], "structural_geomean": [.95] * 5,
            "exact_pixel_duplicate": [False] * 4 + [True],
        })
        self.assertEqual(list(select_top_edges(frame, ranking="ratio").pair_uid), ["a", "b", "c", "d"])
        self.assertEqual(list(select_top_edges(frame).pair_uid), ["b", "c", "d", "a"])
        self.assertEqual(list(select_top_edges(frame.sample(frac=1, random_state=42)).pair_uid), ["b", "c", "d", "a"])

    def test_support_selection_priority_before_max_evidence(self):
        frame = pd.DataFrame({
            "place_uid_a": ["P"] * 5, "place_uid_b": list("ABCDE"),
            "independent_support_3": [False, True, False, False, False],
            "independent_support_2": [False, True, False, True, False],
            "repeated_support_3": [True, True, False, False, False],
            "repeated_support_2": [True, True, True, True, False],
            "structural_bottleneck_max": [.99, .95, 1., .96, 1.],
            "num_core_q95_image_edges": [3, 3, 2, 2, 0],
        })
        self.assertEqual(list(select_repeated_place_pairs(frame).place_uid_b), list("BDAC"))

    def test_graph_slices_rank_denominator_spatial_and_hubs(self):
        images, edges, places, pairs = graph_fixture()
        report, image_audit, place_audit = analyze_graphs(images, edges, places, pairs)
        self.assertEqual(set(report["image_graph"]), set(SLICES))
        self.assertEqual(report["image_graph"]["full"]["edge_count"], 5)
        self.assertEqual(report["image_graph"]["core_q95"]["node_count"], 5)
        self.assertEqual(report["image_graph"]["core_q95"]["active_node_count"], 3)
        self.assertEqual(report["image_graph"]["core_q95_geo500"]["edge_count"], 1)
        self.assertEqual(report["place_graph"]["core_q95"]["edge_count"], 2)
        self.assertEqual(report["place_core_q95_components"][0]["place_uids"], list("PQR"))
        self.assertEqual(report["rank_analysis"]["rank_1"]["core_q95_fraction"], 1.)
        for name in KEYPOINT_BINS:
            self.assertEqual(report["small_denominator_audit"]["by_bin"][name]["pair_count"], 1)
        spatial = report["spatial_support_audit"]
        self.assertAlmostEqual(spatial["candidate_q10_symmetric_match_coverage"], .18)
        self.assertEqual(spatial["core_q95_low_spatial_support_pair_uids"], ["ab"])
        self.assertEqual(report["hub_analysis"]["image_core_q95_top20"][0]["image_id"], "a")
        normalized = report["hub_analysis"]["degree_normalized_image_top20"]
        self.assertEqual(normalized[0]["image_id"], "a")
        self.assertAlmostEqual(normalized[0]["structural_edge_fraction"], 2 / 3)
        self.assertTrue(all(row["candidate_degree_full"] >= 3 for row in normalized))
        self.assertEqual(image_audit.set_index("image_id").loc["e", "structural_degree_q95"], 0)
        self.assertEqual(place_audit.set_index("place_uid").loc["S", "structural_degree_q95"], 0)

    def test_snapshot_finite_portable_and_no_new_matching(self):
        snapshot = {"provenance": {"real_rgb_only": True, "synthetic_images_used": False,
                                   "no_new_local_matching": True}, "number": .5, "missing": None}
        validate_graph_snapshot(snapshot)
        for key, value in (("number", float('nan')), ("number", float('inf')),
                           ("path", "/tmp/private"), ("path", "C:\\private"),
                           ("descriptors", [1, 2])):
            bad = copy.deepcopy(snapshot)
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_graph_snapshot(bad)
        for key, value in (("real_rgb_only", False), ("synthetic_images_used", True), ("no_new_local_matching", False)):
            bad = copy.deepcopy(snapshot)
            bad["provenance"][key] = value
            with self.assertRaises(ValueError):
                validate_graph_snapshot(bad)


if __name__ == "__main__":
    unittest.main()

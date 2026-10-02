"""CPU-only graph identity, pilot-slice, aggregation and publication checks."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from countermine.mining.countermine_graph import (
    build_image_graph, build_place_graph, load_graph_artifacts, place_pair_uid,
    validate_graph_source_geometry, write_graph_artifacts,
)


def fixture():
    manifest = pd.DataFrame([
        {"image_id": "a", "row_index": 0, "place_uid": "z", "city_id": "city"},
        {"image_id": "b", "row_index": 1, "place_uid": "alpha", "city_id": "city"},
        {"image_id": "c", "row_index": 2, "place_uid": "alpha", "city_id": "city"},
        {"image_id": "d", "row_index": 3, "place_uid": "z", "city_id": "city"},
        {"image_id": "unmeasured", "row_index": 4, "place_uid": "other", "city_id": "other"},
    ])
    lookup = manifest.set_index("image_id")
    rows = []
    for index, (image_a, image_b, ratio, count, matches, distance) in enumerate([
        ("a", "b", .95, .95, 20, 300),
        ("a", "c", .99, .99, 30, 1000),
        ("b", "d", 1, .5, 40, 1100),
        ("c", "d", 1, 1, 50, 1200),
    ]):
        row = {
            "pair_uid": f"pair-{index}", "image_id_a": image_a, "image_id_b": image_b,
            "geo_distance_m": distance, "same_city": True, "best_rgb_rank": 1,
            "rank_bin": "rank_1", "num_candidate_directions": 2,
            "max_salad_similarity": .9 - index * .1, "mean_salad_similarity": .8 - index * .1,
            "num_keypoints_a": 200, "num_keypoints_b": 400, "num_matches": matches,
            "local_match_ratio": matches / 200, "matched_source_cell_coverage": .1 + index * .1,
            "matched_target_cell_coverage": .2 + index * .1, "symmetric_match_coverage": .1 + index * .1,
            "source_max_cell_match_fraction": .3, "target_max_cell_match_fraction": .4,
            "source_match_entropy": .3, "target_match_entropy": .4, "symmetric_match_entropy": .3,
            "exact_pixel_duplicate": False, "ratio_null_percentile": ratio,
            "match_count_null_percentile": count, "coverage_null_percentile": .8,
            "entropy_null_percentile": .7, "structural_bottleneck": min(ratio, count),
            "structural_geomean": np.sqrt(ratio * count), "spatial_support_percentile": .8,
            "salad_similarity_percentile": (4 - index) / 4,
        }
        for side, identity in (("a", image_a), ("b", image_b)):
            for name in ("row_index", "place_uid", "city_id"):
                row[f"{name}_{side}"] = lookup.loc[identity, name]
        for metric in ("ratio", "match_count", "coverage", "entropy"):
            row[f"{metric}_null_source"] = "same_city"
            row[f"global_{metric}_null_percentile"] = row[f"{metric}_null_percentile"]
            row[f"relation_{metric}_null_percentile"] = row[f"{metric}_null_percentile"]
        rows.append(row)
    return pd.DataFrame(rows), manifest


class ImageGraphTests(unittest.TestCase):
    def test_nodes_and_all_edges_preserved_in_fixed_pilot_universe(self):
        calibrated, manifest = fixture()
        nodes, edges = build_image_graph(calibrated, manifest)
        self.assertEqual(nodes["image_id"].tolist(), ["a", "b", "c", "d"])
        self.assertEqual(nodes.columns.tolist(), ["image_id", "row_index", "place_uid", "city_id"])
        self.assertEqual(len(edges), 4)
        self.assertEqual(edges["pair_uid"].tolist(), calibrated["pair_uid"].tolist())
        for column in calibrated:
            self.assertIn(column, edges)
        self.assertEqual(edges["mean_salad_similarity"].tolist(), calibrated["mean_salad_similarity"].tolist())
        self.assertEqual(edges["num_candidate_directions"].tolist(), [2] * 4)

    def test_all_5000_edges_retained_without_evidence_filter(self):
        calibrated, _ = fixture()
        row = calibrated.iloc[0].to_dict()
        rows, nodes = [], []
        for index in range(5000):
            current = dict(row, pair_uid=f"pair-{index:05}", structural_bottleneck=0,
                           structural_geomean=0, ratio_null_percentile=0, match_count_null_percentile=0)
            for side, row_index in (("a", index * 2), ("b", index * 2 + 1)):
                image = f"image-{row_index:05}"
                place = f"place-{row_index:05}"
                current.update({f"image_id_{side}": image, f"place_uid_{side}": place, f"row_index_{side}": row_index})
                nodes.append({"image_id": image, "place_uid": place, "row_index": row_index, "city_id": "city"})
            rows.append(current)
        image_nodes, edges = build_image_graph(pd.DataFrame(rows), pd.DataFrame(nodes))
        self.assertEqual(len(edges), 5000)
        self.assertEqual(len(image_nodes), 10000)
        self.assertEqual(int(edges["core_q95"].sum()), 0)

    def test_undirected_uniqueness_and_realign_endpoint_diagnostics(self):
        calibrated, manifest = fixture()
        row = calibrated.iloc[0].to_dict()
        for name in ("image_id", "row_index", "place_uid", "city_id", "num_keypoints"):
            row[f"{name}_a"], row[f"{name}_b"] = row[f"{name}_b"], row[f"{name}_a"]
        for left, right in (("matched_source_cell_coverage", "matched_target_cell_coverage"),
                            ("source_max_cell_match_fraction", "target_max_cell_match_fraction"),
                            ("source_match_entropy", "target_match_entropy")):
            row[left], row[right] = row[right], row[left]
        _, edges = build_image_graph(pd.DataFrame([row]), manifest)
        self.assertEqual(edges.iloc[0]["image_id_a"], "a")
        self.assertEqual(edges.iloc[0]["num_keypoints_a"], 200)
        self.assertEqual(edges.iloc[0]["matched_source_cell_coverage"], .1)
        self.assertEqual(edges.iloc[0]["source_max_cell_match_fraction"], .3)
        row["pair_uid"] = "another-pair"
        with self.assertRaisesRegex(ValueError, "undirected"):
            build_image_graph(pd.concat([calibrated.iloc[[0]], pd.DataFrame([row])]), manifest)

    def test_frozen_q95_q99_duplicates_and_geo500(self):
        calibrated, manifest = fixture()
        _, edges = build_image_graph(calibrated, manifest)
        self.assertEqual(edges["core_q95"].tolist(), [True, True, False, True])
        self.assertEqual(edges["core_q99"].tolist(), [False, True, False, True])
        self.assertEqual(edges["core_q95_geo500"].tolist(), [False, True, False, True])
        self.assertEqual(edges["near_geo_500m"].tolist(), [True, False, False, False])
        self.assertTrue(edges.iloc[0]["core_q95_low_spatial_support"])
        calibrated.loc[3, "exact_pixel_duplicate"] = True
        _, edges = build_image_graph(calibrated, manifest)
        self.assertFalse(edges.iloc[3]["core_q95"])
        self.assertFalse(edges.iloc[3]["core_q99"])
        self.assertEqual(len(edges), 4)

    def test_inconsistent_manifest_same_place_and_nonfinite_rejected(self):
        calibrated, manifest = fixture()
        for column, value, message in (("place_uid_a", "bad", "manifest"),
                                       ("ratio_null_percentile", np.inf, "finite"),
                                       ("structural_bottleneck", .1, "disagrees")):
            changed = calibrated.copy()
            changed.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, message):
                build_image_graph(changed, manifest)
        calibrated.loc[0, "place_uid_a"] = calibrated.loc[0, "place_uid_b"]
        with self.assertRaisesRegex(ValueError, "same-place"):
            build_image_graph(calibrated, manifest)

    def test_source_geometry_and_canonical_uids_checked_without_model_imports(self):
        calibrated, manifest = fixture()
        calibrated = calibrated.iloc[[0]].copy()
        manifest["lat"] = 0
        manifest["lon"] = [0, 1, 2, 3, 4]
        calibrated["pair_uid"] = place_pair_uid("a", "b")
        calibrated["geo_distance_m"] = 6_371_008.8 * np.pi / 180
        result = validate_graph_source_geometry(calibrated, manifest)
        self.assertTrue(result["manifest_geographic_distances"])
        calibrated.loc[0, "geo_distance_m"] += .01
        with self.assertRaisesRegex(ValueError, "haversine"):
            validate_graph_source_geometry(calibrated, manifest)
        calibrated.loc[0, "pair_uid"] = "changed"
        with self.assertRaisesRegex(ValueError, "pair_uid"):
            validate_graph_source_geometry(calibrated, manifest)


class PlaceGraphTests(unittest.TestCase):
    def graph(self):
        calibrated, manifest = fixture()
        nodes, edges = build_image_graph(calibrated, manifest)
        return build_place_graph(nodes, edges)

    def test_place_pair_identity_and_endpoint_realignment(self):
        nodes, edges = self.graph()
        self.assertEqual(nodes["place_uid"].tolist(), ["alpha", "z"])
        self.assertEqual(len(edges), 1)
        row = edges.iloc[0]
        self.assertEqual((row["place_uid_a"], row["place_uid_b"]), ("alpha", "z"))
        self.assertEqual(row["place_pair_uid"], place_pair_uid("z", "alpha"))
        self.assertEqual(row["num_unique_images_a"], 2)
        self.assertEqual(row["num_unique_images_b"], 2)
        self.assertEqual(row["independent_view_support"], 2)
        self.assertEqual(row["num_core_q95_image_edges"], 3)
        self.assertEqual(row["num_core_q99_image_edges"], 2)
        self.assertEqual(row["num_core_q95_geo500_image_edges"], 2)
        self.assertTrue(row["independent_support_3"])
        self.assertTrue(row["core_independent_support_3"])
        self.assertTrue(row["independent_support_2_geo500"])
        self.assertFalse(row["core_geo500_independent_support_2"])
        self.assertEqual(row["num_core_q95_geo500_unique_images_a"], 1)
        self.assertEqual(row["num_core_q95_geo500_unique_images_b"], 2)

    def test_prescribed_independence_uses_all_support_and_strict_uses_core(self):
        calibrated, manifest = fixture()
        nodes, edges = build_image_graph(calibrated.iloc[:3], manifest)
        _, places = build_place_graph(nodes, edges)
        row = places.iloc[0]
        self.assertTrue(row["repeated_support_2"])
        self.assertFalse(row["repeated_support_3"])
        self.assertTrue(row["independent_support_2"])
        self.assertFalse(row["core_independent_support_2"])
        self.assertEqual(row["num_core_q95_unique_images_a"], 2)
        self.assertEqual(row["num_core_q95_unique_images_b"], 1)

    def test_raw_calibrated_spatial_and_geo_max_median(self):
        _, edges = self.graph()
        row = edges.iloc[0]
        self.assertEqual(row["num_supporting_image_edges"], 4)
        self.assertEqual(row["num_matches_max"], 50)
        self.assertEqual(row["num_matches_median"], 35)
        self.assertAlmostEqual(row["local_match_ratio_median"], .175)
        self.assertEqual(row["structural_bottleneck_max"], 1)
        self.assertAlmostEqual(row["structural_bottleneck_median"], .97)
        self.assertAlmostEqual(row["max_salad_similarity_median"], .75)
        self.assertEqual(row["min_geo_distance_m"], 300)
        self.assertEqual(row["median_geo_distance_m"], 1050)
        self.assertAlmostEqual(row["symmetric_match_coverage_median"], .25)
        self.assertEqual(row["symmetric_match_entropy_max"], .3)


class GraphArtifactTests(unittest.TestCase):
    def test_writer_preserves_canonical_step2a_and_rejects_source_aliases(self):
        calibrated, manifest = fixture()
        images = build_image_graph(calibrated, manifest)
        places = build_place_graph(*images)
        calibration = {"stage": "step2b_structural_calibration", "provenance": {},
                       "input_files": {"runtime_dir": "custom/step2a"}, "calibration": {}}
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            for relative in ("cache/countermine_rgb/step2a", "custom/step2a"):
                protected = repository / relative / "forbidden"
                with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "Step 2A runtime"):
                    write_graph_artifacts(protected, *images, *places, calibration_summary=calibration, repo_root=repository)
                self.assertFalse(protected.exists())
            source = repository / "docs/audits/step2a_original.jpg"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"frozen-curated-evidence")
            output = repository / "step2b"
            output.mkdir()
            for filename in ("image_edges.csv", "graph_summary.json"):
                alias = output / filename
                alias.symlink_to(source)
                with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "alias a source"):
                    write_graph_artifacts(output, *images, *places, calibration_summary=calibration, repo_root=repository)
                self.assertEqual(source.read_bytes(), b"frozen-curated-evidence")
                self.assertTrue(alias.is_symlink())
                alias.unlink()

    def test_fractional_match_count_medians_survive_csv_roundtrip(self):
        calibrated, manifest = fixture()
        calibrated = calibrated.iloc[:2].copy()
        calibrated.loc[0, "num_matches"] = 21
        calibrated.loc[0, "local_match_ratio"] = 21 / 200
        images = build_image_graph(calibrated, manifest)
        places = build_place_graph(*images)
        calibration = {"stage": "step2b_structural_calibration", "provenance": {}, "calibration": {}}
        with tempfile.TemporaryDirectory() as directory:
            write_graph_artifacts(directory, *images, *places, calibration_summary=calibration)
            loaded = load_graph_artifacts(directory, source_validation=False)
            self.assertEqual(loaded[3].iloc[0]["num_matches_median"], 25.5)

    def test_hash_checked_roundtrip_finite_summary_and_full_population_q10(self):
        calibrated, manifest = fixture()
        images = build_image_graph(calibrated, manifest)
        places = build_place_graph(*images)
        calibration = {"stage": "2B real-RGB structural evidence calibration", "provenance": {}, "calibration": {}}
        with tempfile.TemporaryDirectory() as directory:
            summary = write_graph_artifacts(directory, *images, *places, calibration_summary=calibration)
            self.assertAlmostEqual(summary["candidate_q10_symmetric_match_coverage"], .13)
            self.assertTrue(summary["provenance"]["real_rgb_only"])
            self.assertFalse(summary["provenance"]["synthetic_images_used"])
            self.assertTrue(summary["provenance"]["no_new_local_matching"])
            payload = json.dumps(summary, allow_nan=False)
            self.assertNotIn(str(Path.cwd()), payload)
            loaded = load_graph_artifacts(directory, source_validation=False)
            self.assertEqual(len(loaded), 5)
            self.assertEqual(loaded[1]["core_q95"].tolist(), images[1]["core_q95"].tolist())
            self.assertEqual(loaded[3]["independent_support_3"].tolist(), [True])
            path = Path(directory) / "image_edges.csv"
            path.write_text(path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_graph_artifacts(directory, source_validation=False)

    def test_graph_import_does_not_load_models_or_tensor_runtime(self):
        code = ("import sys; import countermine.mining.countermine_graph; "
                "assert 'torch' not in sys.modules; "
                "assert not any(k.startswith(('lightglue','salad','aliked')) for k in sys.modules)")
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()

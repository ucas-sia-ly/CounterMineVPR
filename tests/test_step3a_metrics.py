"""Pure scalar Step 3A analysis tests; never import or load model weights."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from countermine.probe import step3a_metrics as metrics


def triplet(index=0, cosines=(.8, .7, .1, .7, .8, .2), *, fail_random=False):
    row = {"triplet_index": index}
    for role in metrics.ROLES:
        row[f"{role}_image_id"] = f"Images/{role}-{index}.jpg"
        row[f"{role}_row_index"] = index * 4 + metrics.ROLES.index(role)
        row[f"{role}_place_uid"] = f"place-{index}" if role in ("q", "p") else f"{role}-place-{index}"
        row[f"{role}_source_relit_R8"] = .4
        row[f"{role}_rgb_mae_normalized"] = .1 + index * .02
        row[f"{role}_abs_luma_mean_delta"] = 4 + index
    if fail_random:
        row["n_random_source_relit_R8"] = .399
    row.update(dict(zip(("s_qp", "s_qhard", "s_qrandom", "sz_qp", "sz_qhard", "sz_qrandom"), cosines)))
    for kind in metrics.NEGATIVE_KINDS:
        for name in metrics.LOCAL_METRICS:
            row[f"{name}_{kind}_rgb"] = .1
            row[f"{name}_{kind}_z"] = .15 if kind == "hard" else .11
    return row


def population():
    # Binary-exact values make the requested strict zero/sign counts unambiguous.
    return [triplet(0, (.75, .625, .125, .625, .75, .25)),
            triplet(1, (.625, .75, .125, .875, .5, .375)),
            triplet(2, (.75, .5, .375, .625, .75, .5), fail_random=True)]


def snapshot(rows=None):
    return metrics.build_snapshot(population() if rows is None else rows,
        provenance={"manifest_reference": "cache/gsv_manifest.csv", "seed": 42},
        population_construction={"query_limit": 500, "same_query_positive_pair": True},
        salad_config={"frozen": True, "preprocessing_policy": "candidate_miner"},
        iclight_config={"width": 640, "height": 480, "mode": "full_scene", "seed_per_image": True},
        aliked_config={"max_num_keypoints": 2048, "detection_threshold": .2, "resize": None},
        lightglue_config={"features": "aliked", "depth_confidence": -1, "width_confidence": -1,
                          "filter_threshold": .1, "mp": False})


class Step3AMarginTests(unittest.TestCase):
    def test_exact_margins_relative_gain_and_paired_difference(self):
        result = metrics.compute_margin_metrics(.8, .7, .1, .7, .8, .2)
        self.assertAlmostEqual(result["margin_hard_rgb"], .1)
        self.assertAlmostEqual(result["margin_hard_z"], -.1)
        self.assertAlmostEqual(result["delta_margin_hard"], -.2)
        self.assertAlmostEqual(result["relative_gain_hard"], .2)
        self.assertAlmostEqual(result["margin_random_rgb"], .7)
        self.assertAlmostEqual(result["margin_random_z"], .5)
        self.assertAlmostEqual(result["paired_delta_margin_difference"], 0)
        for kind in metrics.NEGATIVE_KINDS:
            self.assertAlmostEqual(result[f"relative_gain_{kind}"], -result[f"delta_margin_{kind}"])
        row = metrics.enrich_triplet_record(triplet(3, (.7, .8, .1, .9, .6, .4)))
        self.assertAlmostEqual(row["paired_delta_margin_difference"], .5)

    def test_ties_are_wrong_and_flip_eligibility_is_rgb_state(self):
        result = metrics.compute_margin_metrics(.8, .7, .8, .7, .7, .5)
        self.assertTrue(result["hard_correct_to_wrong"])
        self.assertFalse(result["correct_vs_hard_z"])
        self.assertFalse(result["correct_vs_random_rgb"])
        self.assertTrue(result["random_wrong_to_correct"])
        summary = metrics.summarize_triplets(population())["flip_counts"]["all_triplets"]
        self.assertEqual(summary["hard"]["correct_to_wrong"],
                         {"count": 2, "eligible_rgb_correct_count": 2, "fraction": 1.0})
        self.assertEqual(summary["hard"]["wrong_to_correct"],
                         {"count": 1, "eligible_rgb_wrong_or_tied_count": 1, "fraction": 1.0})
        self.assertEqual(summary["random"]["correct_to_wrong"]["fraction"], 0)
        self.assertIsNone(summary["random"]["wrong_to_correct"]["fraction"])

    def test_common_subset_requires_all_four_R8_inclusive(self):
        rows = metrics.enrich_triplet_records(population())
        self.assertTrue(rows[0]["all_images_R8_pass"])
        self.assertFalse(rows[2]["all_images_R8_pass"])
        self.assertTrue(rows[2]["hard_images_R8_pass"])
        summary = metrics.summarize_triplets(rows)
        self.assertEqual(summary["hard_negative_summary"]["all_images_R8_pass"]["triplet_count"], 2)
        self.assertEqual(summary["random_negative_summary"]["all_images_R8_pass"]["triplet_count"], 2)
        self.assertEqual(summary["paired_hard_vs_random"]["all_triplets"]["num_negative"], 1)
        self.assertAlmostEqual(summary["paired_hard_vs_random"]["all_triplets"]["fraction_hard_delta_less_than_random_delta"], 1 / 3)

    def test_null_ddof1_weak_ecdf_and_frozen_row_calibration(self):
        null = metrics.random_null_metrics([0, 1, 3])
        self.assertAlmostEqual(null["mu_random"], 4 / 3)
        self.assertAlmostEqual(null["sigma_random"], np.std([0, 1, 3], ddof=1))
        self.assertEqual(metrics.empirical_percentile(1, [0, 1, 1, 3]), 75)
        self.assertEqual(metrics.empirical_percentile(-1, [0, 1, 3]), 0)
        self.assertEqual(metrics.empirical_percentile(4, [0, 1, 3]), 100)
        records = metrics.enrich_triplet_records(population())
        gains = [row["relative_gain_random"] for row in records]
        row = records[0]
        self.assertAlmostEqual(row["centered_hard_gain"], row["relative_gain_hard"] - np.mean(gains))
        self.assertAlmostEqual(row["z_hard_gain"], row["centered_hard_gain"] / (np.std(gains, ddof=1) + 1e-12))
        self.assertNotAlmostEqual(row["centered_hard_gain"], row["centered_hard_gain_R8_pass_null"])
        self.assertIsNone(records[2]["centered_hard_gain_R8_pass_null"])
        both = metrics.summarize_triplets(records)["random_null"]
        self.assertEqual(both["all_triplets"]["count"], 3)
        self.assertEqual(both["all_images_R8_pass"]["count"], 2)

    def test_singleton_zero_variance_and_empty_controls_are_finite_json(self):
        self.assertIsNone(metrics.random_null_metrics([])["mu_random"])
        self.assertIsNone(metrics.random_null_metrics([1])["sigma_random"])
        self.assertIsNone(metrics.empirical_percentile(1, []))
        single = metrics.enrich_triplet_records([triplet()])[0]
        self.assertIsNone(single["z_hard_gain"])
        zero_variance = metrics.enrich_triplet_records([triplet(0), triplet(1)])[0]
        self.assertTrue(np.isfinite(zero_variance["z_hard_gain"]))
        json.dumps(snapshot([]), allow_nan=False)
        json.dumps(snapshot([triplet()]), allow_nan=False)

    def test_positive_control_local_changes_strength_aggregate_and_correlations(self):
        record = metrics.enrich_triplet_record(triplet())
        self.assertAlmostEqual(record["delta_positive_similarity"], -.1)
        self.assertAlmostEqual(record["delta_hard_negative_similarity"], .1)
        self.assertAlmostEqual(record["delta_random_negative_similarity"], .1)
        self.assertAlmostEqual(record["delta_cross_match_ratio_hard"], .05)
        self.assertAlmostEqual(record["delta_matched_source_cell_coverage_random"], .01)
        self.assertAlmostEqual(record["mean_rgb_mae_triplet"], .1)
        self.assertAlmostEqual(record["mean_abs_luma_delta_triplet"], 4)
        corr = metrics.compute_correlations([1, 1, 3], [1, 2, 3])
        self.assertAlmostEqual(corr["spearman"], np.sqrt(3) / 2)
        self.assertAlmostEqual(corr["pearson"], np.sqrt(3) / 2)
        self.assertIsNone(metrics.compute_correlations([1, 1], [1, 2])["pearson"])
        self.assertIsNone(metrics.compute_correlations([], [])["spearman"])
        summary = metrics.summarize_triplets(population())
        self.assertEqual(summary["positive_control"]["all_triplets"]["original_positive_similarity"]["count"], 3)
        self.assertEqual(summary["intervention_strength_correlations"]["all_triplets"]
                         ["mean_rgb_mae_triplet"]["relative_gain_hard"]["count"], 3)

    def test_identity_place_constraints_and_input_immutability(self):
        source = triplet()
        before = copy.deepcopy(source)
        metrics.enrich_triplet_record(source)
        self.assertEqual(source, before)
        for role, value in (("p_image_id", source["q_image_id"]),
                            ("n_hard_image_id", source["p_image_id"]),
                            ("p_place_uid", "different"), ("n_random_place_uid", source["q_place_uid"])):
            invalid = dict(source)
            invalid[role] = value
            with self.subTest(role=role), self.assertRaises(ValueError):
                metrics.enrich_triplet_record(invalid)
        data = snapshot()
        self.assertEqual(data["per_triplet_compact"][0]["q_place_uid"], source["q_place_uid"])
        self.assertEqual(data["per_triplet_compact"][0]["p_place_uid"], source["p_place_uid"])

    def test_distribution_quantiles_sample_std_and_invalid_metrics(self):
        result = metrics.distribution_summary([0, 1, 2, 3, 4])
        self.assertEqual(result["median"], 2)
        self.assertAlmostEqual(result["q05"], .2)
        self.assertAlmostEqual(result["std"], np.std([0, 1, 2, 3, 4], ddof=1))
        for name, value in (("s_qp", float("nan")), ("q_source_relit_R8", 1.01),
                            ("cross_match_ratio_hard_rgb", -.1), ("q_rgb_mae_normalized", float("inf"))):
            bad = triplet()
            bad[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                metrics.enrich_triplet_record(bad)


class Step3ASnapshotTests(unittest.TestCase):
    def test_snapshot_compact_fields_top50_common_subset_and_counts(self):
        data = snapshot()
        self.assertEqual(data["counts"], {"query_count": 3, "unique_image_count": 12,
                                         "generated_image_count": 12, "R8_pass_triplet_count": 2})
        self.assertEqual(len(data["per_triplet_compact"]), 3)
        self.assertEqual(len(data["top_margin_collapse"]), 2)
        self.assertEqual(data["top_margin_collapse"][0]["q_image_id"], "Images/q-0.jpg")
        self.assertTrue(all(row["all_images_R8_pass"] for row in data["top_margin_collapse"]))
        self.assertEqual(data["fidelity_subset"]["required_images"], list(metrics.ROLES))
        self.assertIn("big-endian", data["seed_derivation_rule"]["algorithm"])
        self.assertIn("weak ECDF", data["random_null"]["all_triplets"]["empirical_percentile_definition"])
        self.assertEqual(metrics.strict_json_dumps(data), metrics.strict_json_dumps(snapshot()))

    def test_export_rejects_nan_inf_absolute_paths_and_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "snapshot.json"
            output.write_text("previous snapshot\n", encoding="utf-8")
            forbidden = [float("nan"), float("inf"), -float("inf"), "/private/models/checkpoint", 
                         "C:\\models\\checkpoint", "\\\\server\\share\\weights", "file:///private/model", "~/model"]
            for value in forbidden:
                invalid = snapshot()
                invalid["provenance"]["nested"] = {"value": value}
                with self.subTest(value=value), self.assertRaises(ValueError):
                    metrics.export_snapshot(invalid, output)
                self.assertEqual(output.read_text(), "previous snapshot\n")
            expected = snapshot()
            metrics.export_snapshot(expected, output)
            self.assertEqual(json.loads(output.read_text()), expected)
            self.assertEqual(list(output.parent.glob("*.tmp")), [])

    def test_top50_order_uses_only_margin_and_common_fidelity_subset(self):
        rows = [triplet(index, (.8, .7, .1, .7, .75 + index * .002, .2), fail_random=index >= 55)
                for index in range(60)]
        for row in rows:
            for role in metrics.ROLES:
                row[f"{role}_rgb_mae_normalized"] = .1
        data = snapshot(rows)
        selected = data["top_margin_collapse"]
        self.assertEqual(len(selected), 50)
        self.assertEqual(selected[0]["q_image_id"], "Images/q-54.jpg")
        self.assertEqual(selected[-1]["q_image_id"], "Images/q-5.jpg")
        self.assertTrue(all(row["all_images_R8_pass"] for row in selected))
        self.assertEqual([row["delta_margin_hard"] for row in selected],
                         sorted(row["delta_margin_hard"] for row in selected))

    def test_export_rejects_model_objects_arrays_and_duplicate_queries(self):
        for value in (object(), np.zeros((2, 2))):
            with self.assertRaises(ValueError):
                metrics.strict_json_dumps({"model_or_coordinates": value})
        with self.assertRaisesRegex(ValueError, "unique query"):
            snapshot([triplet(), triplet()])
        invalid = snapshot()
        invalid["per_triplet_compact"][0]["q_image_id"] = "/private/image.jpg"
        with self.assertRaisesRegex(ValueError, "absolute paths"):
            metrics.strict_json_dumps(invalid)


if __name__ == "__main__":
    unittest.main()

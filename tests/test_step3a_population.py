"""CPU-only population/seed contract tests; never load model weights."""

import copy
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from countermine.probe.step3a_population import (
    derive_image_seed, haversine_distance_m, iter_candidates, load_manifest, select_population,
    validate_triplet, validate_triplets,
)


class Step3APopulationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = [
            {"row_index": index, "image_id": f"Images/city/image_{index}.jpg",
             "relative_path": f"Images/city/image_{index}.jpg", "place_uid": f"city:{index // 2}",
             "city_id": "city", "lat": index // 2 * 0.02, "lon": 0.0}
            for index in range(8)
        ]
        self.candidates = []
        for q in self.manifest:
            others = [row for row in self.manifest if row["place_uid"] != q["place_uid"]]
            for rank, n in enumerate(others, 1):
                self.candidates.append(self.candidate(q, n, rank))

    @staticmethod
    def candidate(q, n, rank):
        return {"query_row_index": q["row_index"], "negative_row_index": n["row_index"],
                "query_image_id": q["image_id"], "negative_image_id": n["image_id"],
                "query_place_uid": q["place_uid"], "negative_place_uid": n["place_uid"],
                "rank": rank, "similarity": 1.0 / (rank + 1)}

    def select(self, **kwargs):
        return select_population(self.manifest, self.candidates, min_geo_distance_m=250, **kwargs)

    def test_fewer_than_500_returns_exact_valid_queries_without_duplicates(self):
        triplets, accounting = self.select()
        self.assertEqual(len(triplets), 8)
        self.assertEqual(len({row["q_image_id"] for row in triplets}), 8)
        self.assertEqual(accounting["query_count"], 8)
        self.assertEqual(accounting["eligible_query_count"], 8)
        self.assertTrue(accounting["fewer_than_requested"])
        self.assertEqual(accounting["excluded_query_count"], 0)
        self.assertEqual(accounting["eligible_queries_not_selected"], 0)

    def test_q_p_share_place_negatives_differ_and_matched_triplets_share_q_p(self):
        triplets, _ = self.select()
        for row in triplets:
            self.assertTrue(validate_triplet(row, self.manifest, min_geo_distance_m=250))
            self.assertEqual(row["q_place_uid"], row["p_place_uid"])
            self.assertNotEqual(row["q_image_id"], row["p_image_id"])
            for kind in ("hard", "random"):
                self.assertNotEqual(row["q_place_uid"], row[f"n_{kind}_place_uid"])
                self.assertNotEqual(row["q_image_id"], row[f"n_{kind}_image_id"])
                self.assertNotEqual(row["p_image_id"], row[f"n_{kind}_image_id"])
            hard = (row["q_image_id"], row["p_image_id"], row["n_hard_image_id"])
            random = (row["q_image_id"], row["p_image_id"], row["n_random_image_id"])
            self.assertEqual(hard[:2], random[:2])

    def test_all_choices_are_repeatable_and_independent_of_input_row_order(self):
        first, accounting = self.select(selection_seed=91, query_count=4)
        second, accounting2 = select_population(self.manifest[::-1], self.candidates[::-1],
                                                min_geo_distance_m=250, selection_seed=91, query_count=4)
        self.assertEqual(first, second)
        self.assertEqual(accounting, accounting2)
        self.assertEqual(accounting["eligible_queries_not_selected"], 4)
        self.assertEqual(first, self.select(selection_seed=91, query_count=4)[0])

    def test_random_selection_does_not_depend_on_hard_candidate_order_or_identity(self):
        original, _ = self.select()
        reranked = copy.deepcopy(self.candidates)
        for row in reranked:
            row["rank"] = 7 - row["rank"]
        changed, _ = select_population(self.manifest, reranked, min_geo_distance_m=250)
        a, b = ({row["q_image_id"]: row for row in values} for values in (original, changed))
        self.assertTrue(any(a[key]["n_hard_image_id"] != b[key]["n_hard_image_id"] for key in a))
        for key in a:
            self.assertEqual(a[key]["p_image_id"], b[key]["p_image_id"])
            self.assertEqual(a[key]["n_random_image_id"], b[key]["n_random_image_id"])

    def test_hard_selection_is_highest_rank_eligible_existing_candidate(self):
        q = self.manifest[0]
        invalid_same_place = self.candidate(q, self.manifest[1], 1)
        too_close = dict(self.manifest[2], lat=0.00001)
        manifest = [dict(row) if row["row_index"] != 2 else too_close for row in self.manifest]
        candidates = [invalid_same_place, self.candidate(q, too_close, 2),
                      self.candidate(q, manifest[4], 4), self.candidate(q, manifest[5], 3)]
        triplets, accounting = select_population(manifest, candidates, min_geo_distance_m=250)
        self.assertEqual(len(triplets), 1)
        self.assertEqual(triplets[0]["n_hard_image_id"], manifest[5]["image_id"])
        self.assertEqual(triplets[0]["hard_rgb_rank"], 3)
        self.assertEqual(accounting["candidate_accounting"]["candidate_rejected_same_place"], 1)
        self.assertEqual(accounting["candidate_accounting"]["candidate_rejected_geography"], 1)
        self.assertEqual(accounting["query_exclusion_counts"]["missing_eligible_rgb_hard_negative"], 7)
        self.assertEqual(len(accounting["query_exclusions"]), 7)
        self.assertGreaterEqual(triplets[0]["random_geo_distance_m"], 250)

    def test_explicit_source_availability_reports_every_omission(self):
        available = {row["image_id"] for row in self.manifest[1:]}
        triplets, accounting = self.select(available_image_ids=available)
        self.assertEqual(len(triplets), 6)
        self.assertEqual(accounting["unavailable_image_count"], 1)
        self.assertEqual(accounting["unavailable_image_ids"], [self.manifest[0]["image_id"]])
        self.assertEqual(accounting["query_exclusion_counts"]["unavailable_query_source"], 1)
        self.assertEqual(accounting["query_exclusion_counts"]["missing_same_place_positive"], 1)
        selected_ids = {row[f"{role}_image_id"] for row in triplets for role in ("q", "p", "n_hard", "n_random")}
        self.assertNotIn(self.manifest[0]["image_id"], selected_ids)
        self.assertEqual(accounting["unique_image_count"], len(selected_ids))

    def test_no_candidates_does_not_replace_missing_hard_examples(self):
        triplets, accounting = select_population(self.manifest, [], min_geo_distance_m=250)
        self.assertEqual(triplets, [])
        self.assertEqual(accounting["query_count"], 0)
        self.assertEqual(accounting["query_exclusion_counts"]["missing_eligible_rgb_hard_negative"], 8)
        self.assertTrue(accounting["fewer_than_requested"])

    def test_all_missing_sources_is_explicit_not_replacement(self):
        triplets, accounting = self.select(available_image_ids=[])
        self.assertEqual(triplets, [])
        self.assertEqual(accounting["unavailable_image_count"], 8)
        self.assertEqual(accounting["query_exclusion_counts"]["unavailable_query_source"], 8)

    def test_image_seed_exact_independent_unsigned_big_endian_rule(self):
        ids = [row["image_id"] for row in self.manifest]
        seeds = [derive_image_seed(identifier) for identifier in ids]
        self.assertEqual(seeds, [derive_image_seed(identifier) for identifier in ids])
        self.assertEqual(len(set(seeds)), len(ids))
        for identifier, seed in zip(ids, seeds):
            expected = int.from_bytes(hashlib.sha256(("CounterMineVPR-Step3A|" + identifier).encode("utf-8")).digest()[:8], "big") % (2**63)
            self.assertEqual(seed, expected)
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 2**63)
        self.assertEqual(derive_image_seed("é.jpg"), derive_image_seed("é.jpg"))

    def test_invalid_seed_identifiers_are_rejected(self):
        for value in ("", "/absolute.jpg", "../escape.jpg", "C:\\secret\\image.jpg", "..\\escape.jpg", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                derive_image_seed(value)

    def test_geography_formula_matches_frozen_miner_known_distances_and_broadcasting(self):
        distances = haversine_distance_m(np.array([0.0, 45.0])[:, None],
                                        np.array([0.0, 12.0])[:, None],
                                        np.array([0.0, 45.0])[None, :],
                                        np.array([1.0, 12.0])[None, :])
        self.assertEqual(distances.shape, (2, 2))
        self.assertAlmostEqual(float(distances[0, 0]), 111_195.0802335329, places=7)
        self.assertEqual(distances[1, 1], 0.0)
        # Saved existing candidate/manifest coordinates, not a model inference.
        distance = haversine_distance_m(42.32195294686369, -71.1349747990554,
                                       42.32782397861875, -71.13229757613539)
        self.assertAlmostEqual(float(distance), 688.93349969373435, places=7)

    def test_candidate_manifest_identity_and_invalid_numbers_fail_loudly(self):
        for key, value in (("query_row_index", 999), ("negative_row_index", 999),
                           ("query_image_id", "other.jpg"), ("negative_place_uid", "other-place"),
                           ("rank", 0), ("rank", 1.5), ("similarity", float("nan")),
                           ("geo_distance_m", 1.0)):
            candidates = copy.deepcopy(self.candidates)
            candidates[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                select_population(self.manifest, candidates, min_geo_distance_m=250)

    def test_invalid_manifest_and_parameters_fail_loudly(self):
        for key, value in (("image_id", self.manifest[0]["image_id"]), ("row_index", 0),
                           ("lat", float("inf")), ("lon", 181), ("relative_path", "../bad.jpg")):
            rows = copy.deepcopy(self.manifest)
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                select_population(rows, self.candidates, min_geo_distance_m=250)
        for kwargs in ({"query_count": 0}, {"selection_seed": -1}, {"available_image_ids": ["other.jpg"]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.select(**kwargs)
        for threshold in (-1, float("inf"), float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                select_population(self.manifest, self.candidates, min_geo_distance_m=threshold)

    def test_validate_triplet_rejects_broken_pair_identity_and_geography(self):
        triplets, _ = self.select()
        original = triplets[0]
        for role, replacement in (("p", "q"), ("n_hard", "p"), ("n_random", "q"), ("p", "n_hard")):
            row = dict(original)
            for key in ("row_index", "image_id", "place_uid", "relative_path"):
                row[f"{role}_{key}"] = row[f"{replacement}_{key}"]
            with self.subTest(role=role, replacement=replacement), self.assertRaises(ValueError):
                validate_triplet(row, self.manifest, min_geo_distance_m=250)
        with self.assertRaises(ValueError):
            validate_triplet(original, self.manifest, min_geo_distance_m=100_000)

    def test_batch_triplet_validation_preserves_checks_and_rejects_duplicate_queries(self):
        triplets, _ = self.select()
        self.assertTrue(validate_triplets(triplets, self.manifest, min_geo_distance_m=250))
        with self.assertRaisesRegex(ValueError, "duplicate queries"):
            validate_triplets([*triplets, triplets[0]], self.manifest, min_geo_distance_m=250)
        altered = copy.deepcopy(triplets)
        altered[0]["p_image_id"] = altered[0]["q_image_id"]
        with self.assertRaises(ValueError):
            validate_triplets(altered, self.manifest, min_geo_distance_m=250)

    def test_csv_loader_and_streamed_candidate_inputs_match_in_memory_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, rows in (("manifest.csv", self.manifest), ("candidates.csv", self.candidates)):
                with (root / name).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            loaded = load_manifest(root / "manifest.csv")
            streamed, accounting = select_population(loaded, iter_candidates(root / "candidates.csv"),
                                                      min_geo_distance_m=250)
            expected, expected_accounting = self.select()
            self.assertEqual(streamed, expected)
            self.assertEqual(accounting, expected_accounting)
            self.assertEqual(select_population(loaded, root / "candidates.csv", min_geo_distance_m=250)[0], expected)


if __name__ == "__main__":
    unittest.main()

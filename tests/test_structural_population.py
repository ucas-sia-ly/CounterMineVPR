"""CPU fixtures for frozen metadata selection; no model weights or CUDA."""

import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import pandas as pd

from countermine.mining.candidate_miner import CANDIDATE_COLUMNS, haversine_distance_m, stable_pair_uid
from countermine.mining.structural_population import (
    GEO_DISTANCE_ATOL_M,
    MIN_GEO_DISTANCE_M,
    RANK_BINS,
    build_random_controls,
    canonicalize_candidates,
    geo_distance_bin,
    rank_bin,
    raw_candidate_statistics,
    sample_candidate_population,
    sha256_file,
    validate_raw_candidates,
    write_population_artifacts,
)


def make_manifest():
    longitudes = [0, .004, .0038, .0042, .008, .009, .015, .019, .0041, 10, 10.004, 10.008]
    return pd.DataFrame({
        "row_index": np.arange(len(longitudes)),
        "image_id": ["z", "a"] + [f"image-{index:02}" for index in range(2, len(longitudes))],
        "place_uid": [f"X:{index}" for index in range(9)] + [f"Y:{index}" for index in range(3)],
        "city_id": ["X"] * 9 + ["Y"] * 3,
        "lat": [0.0] * len(longitudes), "lon": longitudes,
    })


def directed(manifest, query, negative, rank=1, similarity=.8, distance=None):
    q = manifest.iloc[query]
    n = manifest.iloc[negative]
    return {
        "query_row_index": query, "negative_row_index": negative,
        "query_image_id": q.image_id, "negative_image_id": n.image_id,
        "query_place_uid": q.place_uid, "negative_place_uid": n.place_uid,
        "query_city_id": q.city_id, "negative_city_id": n.city_id,
        "rank": rank, "similarity": similarity,
        "geo_distance_m": float(haversine_distance_m(q.lat, q.lon, n.lat, n.lon)) if distance is None else distance,
        "same_city": q.city_id == n.city_id,
        "pair_uid": stable_pair_uid(q.image_id, n.image_id),
    }


def raw_fixture():
    manifest = make_manifest().iloc[:6].copy()
    records = []
    for query in range(len(manifest)):
        targets = [index for index in range(len(manifest)) if index != query][:2]
        records.extend(directed(manifest, query, negative, rank, .9 - rank * .1)
                       for rank, negative in enumerate(targets, 1))
    raw = pd.DataFrame(records, columns=CANDIDATE_COLUMNS)
    summary = raw_candidate_statistics(raw)
    summary.update(top_k=2, min_geo_distance_m=0.0, stage="1C raw RGB diagnostic candidates")
    return manifest, raw, summary


class RawValidationTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.raw, self.summary = raw_fixture()

    def validate(self, raw=None, manifest=None, summary=None):
        validate_raw_candidates(self.manifest if manifest is None else manifest,
                                self.raw if raw is None else raw,
                                self.summary if summary is None else summary, expected_top_k=2)

    def test_valid_raw_schema_and_frozen_default(self):
        self.validate()
        with self.assertRaisesRegex(ValueError, "top_k=50"):
            validate_raw_candidates(self.manifest, self.raw, self.summary)
        for bad in (self.raw.drop(columns="similarity"), self.raw.assign(extra=1),
                    self.raw.loc[:, CANDIDATE_COLUMNS[::-1]]):
            with self.subTest(columns=bad.columns.tolist()), self.assertRaisesRegex(ValueError, "schema"):
                self.validate(raw=bad)

    def test_same_place_rejected(self):
        manifest = self.manifest.copy()
        manifest.loc[1, "place_uid"] = manifest.loc[0, "place_uid"]
        raw = self.raw.copy()
        for prefix in ("query", "negative"):
            raw[prefix + "_place_uid"] = manifest.place_uid.to_numpy()[raw[prefix + "_row_index"]]
        with self.assertRaisesRegex(ValueError, "same-place"):
            self.validate(raw, manifest)
        with self.assertRaisesRegex(ValueError, "same-place"):
            canonicalize_candidates(raw)

    def test_haversine_tolerance_and_pair_uid(self):
        raw = self.raw.copy()
        raw.loc[0, "geo_distance_m"] += GEO_DISTANCE_ATOL_M * 2
        with self.assertRaisesRegex(ValueError, "haversine"):
            self.validate(raw=raw)
        raw = self.raw.copy()
        raw.loc[0, "pair_uid"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "stable_pair_uid"):
            self.validate(raw=raw)
        with self.assertRaisesRegex(ValueError, "stable_pair_uid"):
            canonicalize_candidates(raw)

    def test_summary_counts_statistics_and_raw_exclusion_checked(self):
        for key, replacement in (
            ("total_candidate_pairs", len(self.raw) + 1),
            ("number_of_queries", 100), ("unique_pair_uids", 100),
            ("reverse_duplicate_pairs", 100), ("fraction_same_city", .5),
            ("similarity_statistics", {"mean": .9}),
            ("rank_distribution", {"1": 1}),
            ("geo_distance_counts_below_m", {"250": 0}),
            ("top_candidates", []), ("min_geo_distance_m", 250.0),
        ):
            summary = copy.deepcopy(self.summary)
            summary[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(summary=summary)

    def test_manifest_hash_and_metadata_alignment(self):
        summary = dict(self.summary, manifest_sha256="abc")
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            validate_raw_candidates(self.manifest, self.raw, summary, expected_top_k=2,
                                    manifest_sha256="def")
        raw = self.raw.copy()
        raw.loc[0, "negative_city_id"] = "bad"
        with self.assertRaisesRegex(ValueError, "negative_city_id"):
            self.validate(raw=raw)

    def test_complete_top_k_numeric_boolean_order_and_uniqueness(self):
        corruptions = []
        corruptions.append(self.raw.iloc[:-1])
        corruptions.append(self.raw.assign(rank=1))
        corruptions.append(self.raw.assign(same_city="True"))
        corruptions.append(self.raw.assign(similarity=np.nan))
        corruptions.append(self.raw.assign(geo_distance_m=-1.0))
        corruptions.append(self.raw.assign(negative_row_index=100))
        duplicate = self.raw.copy()
        duplicate.loc[1, "negative_row_index"] = duplicate.loc[0, "negative_row_index"]
        corruptions.append(duplicate)
        out_of_order = self.raw.copy()
        out_of_order.loc[0, "similarity"] = .1
        corruptions.append(out_of_order)
        for raw in corruptions:
            with self.subTest(rows=len(raw)), self.assertRaises(ValueError):
                self.validate(raw=raw)


class CanonicalPopulationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = make_manifest()

    def table(self, records):
        return pd.DataFrame(records, columns=CANDIDATE_COLUMNS)

    def test_250m_boundary_and_lexicographic_undirected_endpoints(self):
        raw = self.table([directed(self.manifest, 0, 1, distance=250.0),
                          directed(self.manifest, 0, 2, distance=249.999),
                          directed(self.manifest, 1, 0, rank=2, similarity=.6, distance=250.0)])
        population = canonicalize_candidates(raw)
        self.assertEqual(len(population), 1)
        row = population.iloc[0]
        self.assertEqual((row.image_id_a, row.image_id_b), ("a", "z"))
        self.assertEqual((row.row_index_a, row.row_index_b), (1, 0))
        self.assertEqual(row.num_candidate_directions, 2)
        self.assertEqual(row.geo_distance_m, MIN_GEO_DISTANCE_M)

    def test_directional_aggregation_and_missing_direction(self):
        raw = self.table([directed(self.manifest, 0, 1, rank=4, similarity=.6),
                          directed(self.manifest, 1, 0, rank=2, similarity=.8),
                          directed(self.manifest, 0, 4, rank=8, similarity=.5)])
        population = canonicalize_candidates(raw).set_index("pair_uid")
        row = population.loc[stable_pair_uid("z", "a")]
        self.assertEqual(row.best_rgb_rank, 2)
        self.assertEqual(row.max_salad_similarity, .8)
        self.assertAlmostEqual(row.mean_salad_similarity, .7)
        self.assertEqual(row.a_to_b_rank, 2)
        self.assertEqual(row.b_to_a_rank, 4)
        self.assertEqual(row.a_to_b_similarity, .8)
        self.assertEqual(row.b_to_a_similarity, .6)
        one_way = population.loc[stable_pair_uid("z", "image-04")]
        self.assertTrue(pd.isna(one_way.a_to_b_rank))
        self.assertEqual(one_way.b_to_a_rank, 8)

    def test_anchor_lower_rank_higher_similarity_then_row(self):
        for rank_a, rank_b, score_a, score_b, expected in (
            (1, 2, .5, .9, 0), (2, 2, .5, .9, 1), (2, 2, .8, .8, 0),
        ):
            raw = self.table([directed(self.manifest, 0, 1, rank_a, score_a),
                              directed(self.manifest, 1, 0, rank_b, score_b)])
            row = canonicalize_candidates(raw).iloc[0]
            self.assertEqual(row.anchor_query_row_index, expected)
            self.assertEqual(row.anchor_query_image_id, self.manifest.iloc[expected].image_id)

    def test_sha256_sampling_exact_repeatable_and_independent_of_local_results(self):
        canonical = canonicalize_candidates(self.table([
            directed(self.manifest, 0, index, index, .9 - index * .01)
            for index in (1, 4, 5, 6, 7, 9, 10, 11)]))
        expected = sorted(canonical.pair_uid, key=lambda uid: (
            hashlib.sha256(("CounterMineVPR-Step2A|42|" + uid).encode()).hexdigest(), uid))[:3]
        sampled = sample_candidate_population(canonical, max_pairs=3, seed=42)
        self.assertEqual(sampled.pair_uid.tolist(), expected)
        self.assertEqual(sample_candidate_population(canonical.sample(frac=1, random_state=9), 3, 42)
                         .pair_uid.tolist(), expected)
        modified = canonical.assign(local_match_ratio=np.arange(len(canonical)))
        self.assertEqual(sample_candidate_population(modified, 3, 42).pair_uid.tolist(), expected)
        self.assertEqual(len(sample_candidate_population(canonical, 0, 42)), len(canonical))
        self.assertEqual(len(sample_candidate_population(canonical, 5000, 42)), len(canonical))

    def test_frozen_rank_and_geo_bin_boundaries(self):
        for lower, upper, expected in ((1, 1, RANK_BINS[0]), (2, 5, RANK_BINS[1]),
                                      (6, 10, RANK_BINS[2]), (11, 20, RANK_BINS[3]),
                                      (21, 50, RANK_BINS[4])):
            self.assertEqual(rank_bin(lower), expected)
            self.assertEqual(rank_bin(upper), expected)
        for invalid in (0, 51, True, 1.5):
            with self.assertRaises(ValueError):
                rank_bin(invalid)
        for distance, label in ((250, "[250,500)"), (499.999, "[250,500)"),
                                (500, "[500,1000)"), (1000, "[1000,2000)"),
                                (2000, "[2000,5000)"), (5000, "[5000,+inf)")):
            self.assertEqual(geo_distance_bin(distance), label)


class MatchedControlTests(unittest.TestCase):
    def setUp(self):
        self.manifest = make_manifest()
        self.manifest.loc[8, "place_uid"] = self.manifest.loc[0, "place_uid"]

    def make(self, candidate_target=1, extra_targets=(2,)):
        raw = pd.DataFrame([directed(self.manifest, 0, candidate_target)] + [
            directed(self.manifest, 0, index, rank + 2, .7 - rank * .01)
            for rank, index in enumerate(extra_targets)], columns=CANDIDATE_COLUMNS)
        population = canonicalize_candidates(raw)
        population = population.loc[population.anchor_negative_row_index.eq(candidate_target)].reset_index(drop=True)
        return population, raw

    def test_same_anchor_different_place_outside_entire_retrieval_and_distance_stratum(self):
        population, raw = self.make()
        controls = build_random_controls(population, self.manifest, raw)
        row = controls.iloc[0]
        self.assertEqual(row.anchor_query_image_id, "z")
        self.assertEqual(row.anchor_query_row_index, 0)
        self.assertEqual(row.random_negative_row_index, 3)
        self.assertNotEqual(row.random_negative_image_id, row.candidate_negative_image_id)
        self.assertNotEqual(row.anchor_query_place_uid, row.random_negative_place_uid)
        self.assertNotIn(row.random_negative_row_index, raw.negative_row_index.tolist())
        self.assertGreaterEqual(row.random_geo_distance_m, 250.0)
        self.assertTrue(row.candidate_same_city)
        self.assertTrue(row.random_same_city)
        self.assertEqual(row.random_negative_city_id, row.anchor_query_city_id)
        self.assertEqual(geo_distance_bin(row.random_geo_distance_m), row.geo_distance_bin)
        self.assertEqual(row.geo_distance_bin, "[250,500)")

    def test_cross_city_preserves_candidate_target_city_and_exact_sha_draw(self):
        population, raw = self.make(candidate_target=9, extra_targets=(1,))
        row = build_random_controls(population, self.manifest, raw).iloc[0]
        uid = population.iloc[0].pair_uid
        draw = int(hashlib.sha256(("CounterMineVPR-Step2A-random|42|" + uid).encode()).hexdigest(), 16)
        expected = [10, 11][random.Random(draw).randrange(2)]
        self.assertEqual(row.random_negative_row_index, expected)
        self.assertEqual(row.random_negative_city_id, row.candidate_negative_city_id)
        self.assertFalse(row.same_city)
        self.assertFalse(row.random_same_city)
        self.assertIsNone(row.geo_distance_bin)
        repeat = build_random_controls(population, self.manifest, raw.sample(frac=1, random_state=2))
        self.assertEqual(repeat.iloc[0].random_negative_row_index, expected)
        self.assertEqual(row.random_pair_uid, stable_pair_uid("z", self.manifest.iloc[expected].image_id))

    def test_unavailable_explicitly_retained_and_default_95_percent_requirement(self):
        population, raw = self.make(extra_targets=(2, 3))
        controls = build_random_controls(population, self.manifest, raw, min_available_fraction=0)
        self.assertEqual(len(controls), 1)
        self.assertFalse(controls.iloc[0].random_control_available)
        self.assertIsNone(controls.iloc[0].random_negative_image_id)
        self.assertIsNone(controls.iloc[0].random_pair_uid)
        self.assertEqual(controls.iloc[0].candidate_pair_uid, population.iloc[0].pair_uid)
        with self.assertRaisesRegex(ValueError, "require >=95%"):
            build_random_controls(population, self.manifest, raw)

    def test_saved_top_list_and_population_integrity_not_silently_replaced(self):
        population, raw = self.make()
        bad = population.copy()
        bad.loc[0, "anchor_query_image_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identities"):
            build_random_controls(bad, self.manifest, raw)
        with self.assertRaisesRegex(ValueError, "Top-K"):
            build_random_controls(population, self.manifest, raw.iloc[1:])
        bad = population.copy()
        bad.loc[0, "geo_distance_m"] += .1
        with self.assertRaisesRegex(ValueError, "distance"):
            build_random_controls(bad, self.manifest, raw)


class PopulationArtifactTests(unittest.TestCase):
    def test_atomic_finite_hash_outputs_and_provenance_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [root / name for name in ("manifest.csv", "raw.csv", "raw.json")]
            for index, path in enumerate(inputs):
                path.write_text(str(index), encoding="utf-8")
            kwargs = dict(manifest_path=inputs[0], candidate_path=inputs[1], candidate_summary_path=inputs[2])
            population = pd.DataFrame({"pair_uid": ["abc"], "rank": [1]})
            controls = pd.DataFrame({"pair_uid": ["abc"], "random_control_available": [True]})
            summary = {"seed": 42, "max_pairs": 5000, "min_geo_distance_m": 250.0,
                       "sampling_definition": "fixed", "random_sampling_definition": "fixed"}
            out = root / "out"
            saved = write_population_artifacts(population, controls, summary, out, **kwargs)
            self.assertEqual(saved["manifest_sha256"], sha256_file(inputs[0]))
            self.assertEqual(saved["population_sha256"], sha256_file(out / "population.csv"))
            self.assertEqual(saved["random_controls_sha256"], sha256_file(out / "random_controls.csv"))
            self.assertEqual(json.loads((out / "population_summary.json").read_text()), saved)
            before = (out / "population_summary.json").read_bytes()
            write_population_artifacts(population, controls, summary, out, **kwargs)
            self.assertEqual((out / "population_summary.json").read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "provenance"):
                write_population_artifacts(population, controls, dict(summary, seed=43), out, **kwargs)
            with self.assertRaisesRegex(ValueError, "different population_sha256"):
                write_population_artifacts(population.assign(rank=2), controls, summary, out, **kwargs)
            self.assertEqual(list(out.glob("*.partial")), [])
            with self.assertRaises(ValueError):
                write_population_artifacts(population, controls, dict(summary, bad=np.nan), root / "bad", **kwargs)
            self.assertFalse((root / "bad" / "population.csv").exists())


if __name__ == "__main__":
    unittest.main()

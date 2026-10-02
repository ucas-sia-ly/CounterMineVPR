"""CPU-only sampler, binary graph, and structural-exposure audits."""

from collections import Counter
import csv
from dataclasses import FrozenInstanceError
import hashlib
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest

from countermine.training.countermine_batch_sampler import (
    CounterMineBatchSampler, EdgeBundle, load_edge_bundle,
)
from countermine.training.gsv_place_mapping import PlaceRecord


def records_for(cities):
    counts = Counter()
    records = []
    for index, city in enumerate(cities):
        counts[city] += 1
        records.append(PlaceRecord(index, index + 10000, city, f"{city}:{counts[city]:07d}"))
    return tuple(records)


def bundle_for(records, edges, q95=None, q99=None):
    def uid_edges(items):
        return frozenset(tuple(sorted((records[a].canonical_place_uid,
                                       records[b].canonical_place_uid))) for a, b in items)
    geo = uid_edges(edges)
    q99 = geo if q99 is None else uid_edges(q99)
    q95 = q99 if q95 is None else uid_edges(q95)
    return EdgeBundle("a" * 64, "b" * 64,
                      frozenset(record.canonical_place_uid for record in records), q95, q99, geo)


class BatchSamplerTests(unittest.TestCase):
    def mixed_fixture(self):
        cities = (["Boston", "Paris", "Boston", "London", "Paris", "London"] * 4
                  + ["Boston", "Boston", "London", "London", "Paris"])
        records = records_for(cities)
        edges = [(a, b) for a, b in combinations(range(len(records)), 2)
                 if records[a].city_id == records[b].city_id and records[a].city_id in ("Boston", "London")]
        return records, bundle_for(records, edges)

    def test_baseline_is_exact_sequential_grouping_and_keeps_last_batch(self):
        records = records_for(["Boston"] * 8)
        sampler = CounterMineBatchSampler(records, batch_size=3)
        self.assertEqual(list(sampler), [[0, 1, 2], [3, 4, 5], [6, 7]])
        self.assertEqual(len(sampler), 3)
        self.assertFalse(sampler.drop_last)
        self.assertEqual(sampler.plan.summary["batch_sizes"], [3, 3, 2])
        self.assertEqual(sampler.plan.summary["max_guided_pairs_per_place"], 0)

    def test_baseline_graph_evidence_does_not_alter_order(self):
        records, bundle = self.mixed_fixture()
        plain = CounterMineBatchSampler(records, batch_size=6)
        audited = CounterMineBatchSampler(records, bundle, batch_size=6)
        self.assertEqual(plain.plan.batches, audited.plan.batches)
        audited.set_epoch(3)
        self.assertEqual(plain.plan.batches, audited.plan.batches)

    def test_all_marginals_city_slots_and_non_target_slots_preserved(self):
        records, bundle = self.mixed_fixture()
        baseline = CounterMineBatchSampler(records, bundle, batch_size=6)
        treatment = CounterMineBatchSampler(records, bundle, mode="countermine_q99_geo500",
                                            batch_size=6, require_exposure_increase=False)
        self.assertEqual(sorted(index for batch in treatment for index in batch), list(range(len(records))))
        self.assertEqual([len(batch) for batch in treatment], [len(batch) for batch in baseline])
        for before, after in zip(baseline, treatment):
            self.assertEqual(Counter(records[i].city_id for i in before), Counter(records[i].city_id for i in after))
            for original_index, treatment_index in zip(before, after):
                if records[original_index].city_id == "Paris":
                    self.assertEqual(original_index, treatment_index)
        self.assertEqual(baseline.plan.summary["marginal_exposure"], treatment.plan.summary["marginal_exposure"])
        for key in ("duplicate_place_count", "missing_place_count", "accidentally_split_pair_count",
                    "per_batch_city_composition_mismatch_count"):
            self.assertEqual(treatment.plan.summary[key], 0)

    def test_guided_pairs_cooccur_and_each_vertex_used_at_most_once(self):
        records, bundle = self.mixed_fixture()
        sampler = CounterMineBatchSampler(records, bundle, mode="countermine_q99_geo500",
                                          batch_size=6, require_exposure_increase=False)
        used = []
        for pair in sampler.plan.guided_pairs:
            self.assertIn(pair.dataset_index_a, sampler.plan.batches[pair.batch_index])
            self.assertIn(pair.dataset_index_b, sampler.plan.batches[pair.batch_index])
            self.assertEqual(records[pair.dataset_index_a].city_id, records[pair.dataset_index_b].city_id)
            self.assertIn(tuple(sorted((pair.canonical_place_uid_a, pair.canonical_place_uid_b))), bundle.q99_geo500)
            used.extend((pair.dataset_index_a, pair.dataset_index_b))
        self.assertEqual(len(used), len(set(used)))
        self.assertEqual(sampler.plan.summary["max_guided_pairs_per_place"], 1)
        self.assertEqual(sampler.plan.summary["unique_guided_place_count"], len(used))

    def test_seed_and_epoch_plans_are_deterministic(self):
        records, bundle = self.mixed_fixture()
        args = dict(mode="countermine_q99_geo500", batch_size=6, seed=42, require_exposure_increase=False)
        first = CounterMineBatchSampler(records, bundle, **args)
        repeat = CounterMineBatchSampler(records, bundle, **args)
        self.assertEqual(first.plan, repeat.plan)
        zero_plan = first.plan
        first.set_epoch(1)
        repeat.set_epoch(1)
        self.assertEqual(first.plan, repeat.plan)
        self.assertNotEqual(zero_plan.guided_pairs, first.plan.guided_pairs)
        self.assertNotEqual(zero_plan.summary["treatment_sequence_sha256"], first.plan.summary["treatment_sequence_sha256"])

    def test_cross_city_and_other_city_edges_are_never_guided(self):
        records = records_for(["Boston", "London", "Paris", "Boston", "London", "Paris"])
        bundle = bundle_for(records, [(0, 1), (2, 5), (0, 3), (1, 4)])
        sampler = CounterMineBatchSampler(records, bundle, mode="countermine_q99_geo500",
                                          batch_size=6, require_exposure_increase=False)
        self.assertEqual(sampler.plan.summary["eligible_q99_geo500_same_city_edge_count"], 2)
        self.assertEqual({pair.city_id for pair in sampler.plan.guided_pairs}, {"Boston", "London"})
        self.assertEqual(sampler.plan.summary["exposure"]["core_q99_geo500"], 4)

    def test_city_pair_capacity_is_respected_without_relaxation(self):
        records = records_for(["Boston", "London"] * 6)
        bundle = bundle_for(records, [(0, 2), (4, 6), (1, 3), (5, 7)])
        with self.assertRaisesRegex(ValueError, "pair-slot capacity"):
            CounterMineBatchSampler(records, bundle, mode="countermine_q99_geo500",
                                   batch_size=2, require_exposure_increase=False)
        with self.assertRaisesRegex(ValueError, "pair-slot capacity"):
            CounterMineBatchSampler(records, bundle, mode="countermine_q99_geo500", batch_size=2)

    def test_baseline_and_treatment_exposure_counts(self):
        records = records_for(["Boston"] * 8)
        geo = [(0, 4), (1, 5), (2, 6), (3, 7)]
        bundle = bundle_for(records, geo, q95=geo + [(0, 1), (4, 5)], q99=geo + [(0, 1)])
        baseline = CounterMineBatchSampler(records, bundle, batch_size=4)
        self.assertEqual(baseline.plan.summary["exposure"]["core_q95"], 2)
        self.assertEqual(baseline.plan.summary["exposure"]["core_q99"], 1)
        self.assertEqual(baseline.plan.summary["exposure"]["core_q99_geo500"], 0)
        treatment = CounterMineBatchSampler(records, bundle, mode="countermine_q99_geo500", batch_size=4)
        self.assertEqual(treatment.plan.summary["exposure"]["core_q99_geo500"], 4)
        self.assertEqual(treatment.plan.summary["exposure"]["unique_structural_places_exposed"], 8)
        self.assertEqual(treatment.plan.summary["greedy_matched_pair_count"], 4)
        self.assertEqual(treatment.plan.summary["baseline_exposure"], baseline.plan.summary["exposure"])
        self.assertIsNone(treatment.plan.summary["structural_exposure_ratio"])

    def test_accidental_exposure_is_distinct_from_guided_pairs(self):
        records = records_for(["Boston"] * 8)
        bundle = bundle_for(records, list(combinations(range(8), 2)))
        sampler = CounterMineBatchSampler(records, bundle, mode="countermine_q99_geo500",
                                          batch_size=4, require_exposure_increase=False)
        self.assertEqual(sampler.plan.summary["greedy_matched_pair_count"], 4)
        self.assertEqual(sampler.plan.summary["exposure"]["core_q99_geo500"], 12)
        self.assertEqual(sampler.plan.summary["accidental_q99_geo500_exposure_count"], 8)
        with self.assertRaisesRegex(ValueError, "does not increase"):
            CounterMineBatchSampler(records, bundle, mode="countermine_q99_geo500", batch_size=4)

    def test_empty_dataset(self):
        sampler = CounterMineBatchSampler(())
        self.assertEqual(list(sampler), [])
        self.assertEqual(len(sampler), 0)
        self.assertEqual(sampler.plan.summary["missing_place_count"], 0)

    def test_saved_plans_are_finite_and_csv_hashes_match(self):
        records = records_for(["Boston"] * 8)
        bundle = bundle_for(records, [(0, 4), (1, 5), (2, 6), (3, 7)])
        with tempfile.TemporaryDirectory() as directory:
            plan_dir = Path(directory) / "cache/countermine_rgb/step3a/run/batch_plans"
            sampler = CounterMineBatchSampler(records, bundle, mode="countermine_q99_geo500",
                                              batch_size=4, plan_dir=plan_dir, repo_root=directory)
            summary_path, csv_path = sampler.save_plan()
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["guided_pairs_csv_sha256"], hashlib.sha256(csv_path.read_bytes()).hexdigest())
            order_path = plan_dir / "epoch_00_place_order.csv"
            self.assertEqual(summary["place_order_csv_sha256"], hashlib.sha256(order_path.read_bytes()).hexdigest())
            with order_path.open(newline="") as stream:
                order = list(csv.DictReader(stream))
            self.assertEqual([row["canonical_place_uid"] for row in order],
                             [record.canonical_place_uid for record in records])
            with csv_path.open(newline="") as stream:
                pairs = list(csv.DictReader(stream))
            self.assertEqual(len(pairs), 4)
            self.assertNotIn(directory, summary_path.read_text())
            sampler.set_epoch(1)
            self.assertTrue((plan_dir / "epoch_01_summary.json").exists())

    def test_prior_namespaces_and_resolved_symlinks_rejected_before_writes(self):
        records = records_for(["Boston"] * 2)
        sampler = CounterMineBatchSampler(records)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = root / "cache/countermine_rgb/step2d"
            prior.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "output namespace"):
                sampler.save_plan(prior, repo_root=root)
            runtime = root / "cache/countermine_rgb/step3a"
            runtime.mkdir()
            symlink = runtime / "batch_plans"
            symlink.symlink_to(prior, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "output namespace"):
                sampler.save_plan(symlink, repo_root=root)
            symlink.unlink()
            symlink.mkdir()
            (symlink / "epoch_00_summary.json").symlink_to(prior / "frozen.json")
            with self.assertRaisesRegex(ValueError, "output namespace"):
                sampler.save_plan(symlink, repo_root=root)
            self.assertEqual(list(prior.iterdir()), [])
            self.assertFalse((symlink / "epoch_00_guided_pairs.csv").exists())

    def test_empty_eligible_treatment_cannot_silently_run(self):
        records = records_for(["Boston"] * 2)
        bundle = bundle_for(records, [])
        with self.assertRaisesRegex(ValueError, "No eligible"):
            CounterMineBatchSampler(records, bundle, mode="countermine_q99_geo500")

    def test_unmapped_graph_places_and_invalid_inputs_rejected(self):
        records = records_for(["Boston"] * 2)
        bundle = bundle_for(records, [(0, 1)])
        with self.assertRaisesRegex(ValueError, "Every frozen"):
            CounterMineBatchSampler(records[:1], bundle)
        with self.assertRaises(ValueError):
            CounterMineBatchSampler(records, mode="weighted")
        with self.assertRaises(ValueError):
            CounterMineBatchSampler(records, mode="countermine_q99_geo500")
        with self.assertRaises(ValueError):
            CounterMineBatchSampler(records, batch_size=0)
        with self.assertRaises(ValueError):
            CounterMineBatchSampler(records, epoch=-1)


class FrozenGraphTests(unittest.TestCase):
    def test_bundle_is_frozen_and_owns_immutable_binary_edge_sets(self):
        records = records_for(["Boston"] * 2)
        edges = {(records[0].canonical_place_uid, records[1].canonical_place_uid)}
        bundle = EdgeBundle("a" * 64, "b" * 64, {r.canonical_place_uid for r in records}, edges, edges, edges)
        edges.clear()
        self.assertEqual(len(bundle.q99_geo500), 1)
        with self.assertRaises(FrozenInstanceError):
            bundle.q99_geo500 = frozenset()
        with self.assertRaises(AttributeError):
            bundle.q99_geo500.add(("Boston:0000003", "Boston:0000004"))

    def write_graph(self, directory):
        path = Path(directory) / "place_edges.csv"
        columns = ("place_uid_a", "place_uid_b", "city_id_a", "city_id_b",
                   "num_core_q95_image_edges", "num_core_q99_image_edges", "num_core_q99_geo500_image_edges")
        rows = [
            ("Boston:0000001", "Boston:0000002", "Boston", "Boston", 3, 2, 1),
            ("London:0000001", "London:0000002", "London", "London", 1, 0, 0),
            ("Boston:0000001", "London:0000001", "Boston", "London", 1, 1, 1),
        ]
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        snapshot = Path(directory) / "metrics.json"
        snapshot.write_text(json.dumps({"complete": True, "graph_artifact_hashes": {
            "place_edges.csv": hashlib.sha256(path.read_bytes()).hexdigest()}}))
        return path, snapshot

    def test_load_frozen_graph_hash_and_binary_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            path, snapshot = self.write_graph(directory)
            bundle = load_edge_bundle(path, snapshot, expected_graph_places=4)
            self.assertEqual(len(bundle.graph_place_uids), 4)
            self.assertEqual(len(bundle.q95), 3)
            self.assertEqual(len(bundle.q99), 2)
            self.assertEqual(len(bundle.q99_geo500), 2)
            self.assertEqual(len(bundle.eligible_edges("Boston")), 1)
            self.assertEqual(bundle.eligible_edges("London"), ())
            self.assertEqual(bundle.eligible_edges("Paris"), ())
            self.assertEqual(bundle.snapshot_sha256, hashlib.sha256(snapshot.read_bytes()).hexdigest())

    def test_tampered_graph_and_wrong_place_count_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, snapshot = self.write_graph(directory)
            with self.assertRaisesRegex(ValueError, "Expected 2000"):
                load_edge_bundle(path, snapshot)
            with path.open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_edge_bundle(path, snapshot, expected_graph_places=4)

    def test_incomplete_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, snapshot = self.write_graph(directory)
            snapshot.write_text(snapshot.read_text().replace("true", "false"))
            with self.assertRaisesRegex(ValueError, "not complete"):
                load_edge_bundle(path, snapshot, expected_graph_places=4)

    def test_non_nested_binary_slices_and_same_place_edges_rejected(self):
        uids = frozenset(("Boston:0000001", "Boston:0000002"))
        pair = frozenset((("Boston:0000001", "Boston:0000002"),))
        with self.assertRaisesRegex(ValueError, "not nested"):
            EdgeBundle("a", "b", uids, frozenset(), pair, pair)
        with self.assertRaisesRegex(ValueError, "distinct"):
            EdgeBundle("a", "b", uids, {(uids.__iter__().__next__(),) * 2}, frozenset(), frozenset())


if __name__ == "__main__":
    unittest.main()

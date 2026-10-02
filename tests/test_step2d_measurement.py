"""CPU scalar fixtures for deterministic shards, resume and atomic finalization."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from countermine.mining.full_population_metadata import POPULATION_COLUMNS
from countermine.mining.full_structural_io import read_json, write_csv, write_json
from countermine.mining.full_structural_measurement import (
    MEASUREMENT_COLUMNS, compare_metric_frames, finalize_full_metrics,
    load_full_metrics, load_valid_shard, run_full_measurement,
    sequence_sha256, shard_boundaries, shard_paths,
)
from countermine.mining.structural_matcher import LocalMatchResult, structural_metrics


def fixture_population():
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
    rows = []
    for index, (a, b) in enumerate(pairs):
        rows.append({
            "pair_uid": f"pair{index}", "image_id_a": f"image{a}", "image_id_b": f"image{b}",
            "row_index_a": a, "row_index_b": b, "place_uid_a": f"place{a}", "place_uid_b": f"place{b}",
            "city_id_a": "city", "city_id_b": "city", "geo_distance_m": 600.0,
            "same_city": True, "num_candidate_directions": 1,
            "best_rgb_rank": 1, "max_salad_similarity": .2, "mean_salad_similarity": .2,
            "a_to_b_rank": 1, "a_to_b_similarity": .2, "b_to_a_rank": "", "b_to_a_similarity": "",
            "anchor_query_image_id": f"image{a}", "anchor_query_row_index": a,
            "anchor_negative_image_id": f"image{b}", "anchor_negative_row_index": b,
            "anchor_rank": 1, "anchor_similarity": .2, "rank_bin": "rank_1",
        })
    return pd.DataFrame(rows, columns=POPULATION_COLUMNS)


def fixture_metrics(row):
    points = np.asarray([[10, 20], [300, 300]], dtype=np.float32)
    values = structural_metrics(10, 10, points, points)
    values.update(min_num_keypoints=10, exact_pixel_duplicate=False)
    return values


class ReplayAndShardHelpers(unittest.TestCase):
    def test_deterministic_boundaries_cover_population_without_overlap(self):
        self.assertEqual(shard_boundaries(5, 2), [(0, 0, 2), (1, 2, 4), (2, 4, 5)])
        covered = [row for _, start, end in shard_boundaries(260502) for row in range(start, end)]
        self.assertEqual(covered, list(range(260502)))
        self.assertEqual(len(shard_boundaries(260502)), 131)
        for count, size in ((0, 2), (3, 0), (3, True), (3, 1.5)):
            with self.assertRaises(ValueError):
                shard_boundaries(count, size)
        self.assertNotEqual(sequence_sha256(["a", "b"]), sequence_sha256(["b", "a"]))

    def test_pilot_replay_fails_closed_for_every_historical_metric(self):
        population = fixture_population().iloc[:2]
        historical = pd.DataFrame([{**row, **fixture_metrics(row)} for row in population.to_dict("records")])
        self.assertTrue(compare_metric_frames(historical, historical, expected_count=2)["passed"])
        for key in ("num_keypoints_a", "num_keypoints_b", "num_matches", "exact_pixel_duplicate",
                    "local_match_ratio", "matched_source_cell_coverage", "matched_target_cell_coverage",
                    "symmetric_match_coverage", "source_max_cell_match_fraction", "target_max_cell_match_fraction",
                    "source_match_entropy", "target_match_entropy", "symmetric_match_entropy"):
            measured = historical.copy()
            if key == "exact_pixel_duplicate":
                measured.loc[0, key] = True
            else:
                measured.loc[0, key] += 1 if key.startswith("num_") else 1e-8
            with self.subTest(metric=key), self.assertRaisesRegex(ValueError, "reproduction.*drift"):
                compare_metric_frames(measured, historical)
        with self.assertRaisesRegex(ValueError, "missing"):
            compare_metric_frames(historical.iloc[:1], historical)
        with self.assertRaisesRegex(ValueError, "count differs"):
            compare_metric_frames(historical, historical, expected_count=3)

    def test_replay_requires_counts_exact_but_bounded_float_difference(self):
        row = fixture_population().iloc[0].to_dict()
        historical = pd.DataFrame([{**row, **fixture_metrics(row)}])
        measured = historical.copy()
        measured["local_match_ratio"] += 1e-13
        report = compare_metric_frames(measured, historical)
        self.assertLessEqual(report["maximum_metric_differences"]["local_match_ratio"], 1e-12)


class ShardedMeasurementIntegration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "cache/countermine_rgb/step2d"
        self.bank = self.runtime / "aliked_bank"
        self.population = fixture_population()
        self.population_summary = {"complete": True, "candidate_count": len(self.population)}
        write_csv(self.runtime / "full_population.csv", self.population, repo_root=self.root)
        write_json(self.runtime / "full_population_summary.json", self.population_summary, repo_root=self.root)
        self.bank_summary = {"configuration": {"seed": 42, "manifest_file": "cache/gsv_mini/manifest.csv"}}
        write_json(self.bank / "summary.json", self.bank_summary, repo_root=self.root)
        self.validation = {"complete": True, "passed": True}
        write_json(self.bank / "validation.json", self.validation, repo_root=self.root)
        self.bank_index = pd.DataFrame({"row_index": range(4), "image_id": [f"image{i}" for i in range(4)],
                                       "num_keypoints": [10] * 4, "rgb_pixel_sha256": [str(i) * 64 for i in range(4)]})
        self.calls = []
        calls = self.calls
        class FakeMatcher:
            provenance = {"lightglue_weights_sha256": "fake_frozen_weights", "feature_cache_size": 128}
            def __init__(self, **kwargs):
                pass
            def match(self, a, b):
                calls.append((a, b))
                metrics = fixture_metrics({})
                return LocalMatchResult(metrics, np.empty((0, 2)), np.empty((0, 2)), None)
        self.factory = FakeMatcher
        self.args = SimpleNamespace(runtime_dir=self.runtime, bank_dir=self.bank,
                                    dataset_root=self.root / "dataset", device="cpu", seed=42,
                                    feature_cache_size=128, shard_size=2, start_shard=0, max_shards=1)
        self.addCleanup(patch.stopall)
        patch("countermine.mining.full_structural_population.load_full_population", side_effect=lambda *a, **k: (self.population.copy(), self.population_summary)).start()
        patch("countermine.mining.full_structural_measurement.load_feature_bank", side_effect=lambda *a, **k: (self.bank_index.copy(), self.bank_summary)).start()
        patch("countermine.mining.full_structural_measurement.require_bank_validation", return_value=self.validation).start()
        patch("countermine.mining.full_structural_measurement.load_bank_manifest", return_value=(pd.DataFrame(), {}, {})).start()
        patch("countermine.mining.full_structural_measurement.verify_source_images").start()
        patch("countermine.mining.full_structural_measurement.frozen_provenance", return_value={"frozen": "unchanged"}).start()
        patch("countermine.mining.full_structural_measurement.code_hashes", return_value={"implementation": "same"}).start()

    def _run(self):
        return run_full_measurement(self.args, repo_root=self.root, matcher_factory=self.factory)

    def _all_shards(self):
        self.args.max_shards = None
        return self._run()

    def test_one_shard_benchmark_resume_then_atomic_finalization(self):
        result = self._run()
        self.assertEqual(result["new_complete_shards"], 1)
        self.assertEqual(len(self.calls), 2)
        csv_path, summary_path = shard_paths(self.runtime, 0)
        summary = read_json(summary_path)
        self.assertEqual(summary["benchmark"]["pairs_processed"], 2)
        self.assertGreater(summary["benchmark"]["pairs_per_second"], 0)
        self.assertIsNone(summary["benchmark"]["peak_cuda_allocated_bytes"])
        self.assertEqual(pd.read_csv(csv_path)["pair_uid"].tolist(), ["pair0", "pair1"])
        self.assertEqual(self._run()["validated_skipped_shards"], 1)
        self.assertEqual(len(self.calls), 2)
        self.args.start_shard = 1
        self.args.max_shards = None
        self.assertEqual(self._run()["new_complete_shards"], 2)
        self.assertEqual(len(self.calls), 5)
        summary = finalize_full_metrics(self.runtime, repo_root=self.root)
        self.assertEqual(summary["candidate_count"], 5)
        self.assertEqual(summary["completed_shard_count"], 3)
        frame, loaded = load_full_metrics(self.runtime, repo_root=self.root)
        self.assertEqual(tuple(frame.columns), MEASUREMENT_COLUMNS)
        self.assertEqual(frame["pair_uid"].tolist(), self.population["pair_uid"].tolist())
        self.assertEqual(loaded, summary)
        self.assertEqual(finalize_full_metrics(self.runtime, repo_root=self.root), summary)
        self.assertFalse(list(self.runtime.rglob("*.tmp")))

    def test_execution_order_is_cache_friendly_but_output_stays_uid_order(self):
        self.population.loc[0, ["image_id_a", "image_id_b", "row_index_a", "row_index_b", "place_uid_a", "place_uid_b", "anchor_query_image_id", "anchor_negative_image_id", "anchor_query_row_index", "anchor_negative_row_index"]] = ["image2", "image3", 2, 3, "place2", "place3", "image2", "image3", 2, 3]
        self._run()
        self.assertEqual(self.calls, [("image0", "image2"), ("image2", "image3")])
        self.assertEqual(pd.read_csv(shard_paths(self.runtime, 0)[0])["pair_uid"].tolist(), ["pair0", "pair1"])

    def test_wrong_pair_sequence_bank_and_config_hashes_reject_resume(self):
        self._run()
        _, summary_path = shard_paths(self.runtime, 0)
        original = read_json(summary_path)
        for key in ("pair_uid_sequence_sha256", "feature_bank_summary_sha256", "matcher_configuration_sha256"):
            write_json(summary_path, {**original, key: "changed"}, repo_root=self.root)
            with self.subTest(binding=key), self.assertRaisesRegex(ValueError, "provenance differs"):
                self._run()
        write_json(summary_path, original, repo_root=self.root)
        self.args.shard_size = 3
        with self.assertRaisesRegex(ValueError, "configuration/provenance changed"):
            self._run()

    def test_incomplete_shard_recomputed_corrupt_complete_shard_refused(self):
        self._run()
        csv_path, summary_path = shard_paths(self.runtime, 0)
        original = read_json(summary_path)
        write_json(summary_path, {**original, "complete": False}, repo_root=self.root)
        self.assertEqual(self._run()["new_complete_shards"], 1)
        self.assertEqual(len(self.calls), 4)
        csv_path.write_text("corrupt published shard")
        with self.assertRaisesRegex(ValueError, "missing or corrupt"):
            self._run()

    def test_missing_unknown_shard_and_changed_final_csv_fail_closed(self):
        self._run()
        with self.assertRaisesRegex(ValueError, "incomplete or missing"):
            finalize_full_metrics(self.runtime, repo_root=self.root)
        self._all_shards()
        unexpected = self.runtime / "shards/shard_99999.csv"
        unexpected.write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "unknown shard"):
            finalize_full_metrics(self.runtime, repo_root=self.root)
        unexpected.unlink()
        finalize_full_metrics(self.runtime, repo_root=self.root)
        (self.runtime / "full_candidate_structural_metrics.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "CSV hash differs"):
            load_full_metrics(self.runtime, repo_root=self.root)

    def test_malformed_shard_benchmark_and_aggregate_summary_are_refused(self):
        self._run()
        _, summary_path = shard_paths(self.runtime, 0)
        original = read_json(summary_path)
        changes = (None, {}, {**original["benchmark"], "elapsed_seconds": 0},
                   {**original["benchmark"], "pairs_per_second": -1},
                   {**original["benchmark"], "pairs_processed": 500},
                   {**original["benchmark"], "cpu_peak_rss_bytes": "bad"},
                   {**original["benchmark"], "peak_cuda_allocated_bytes": 10, "peak_cuda_reserved_bytes": 5})
        for benchmark in changes:
            write_json(summary_path, {**original, "benchmark": benchmark}, repo_root=self.root)
            with self.subTest(benchmark=benchmark), self.assertRaisesRegex(ValueError, "benchmark|memory peak"):
                self._run()
        write_json(summary_path, original, repo_root=self.root)
        self._all_shards()
        finalize_full_metrics(self.runtime, repo_root=self.root)
        final_path = self.runtime / "full_measurement_summary.json"
        final = read_json(final_path)
        final["runtime_throughput_summary"]["pairs_per_second"] += 1
        write_json(final_path, final, repo_root=self.root)
        with self.assertRaisesRegex(ValueError, "aggregate throughput"):
            load_full_metrics(self.runtime, repo_root=self.root)


if __name__ == "__main__":
    unittest.main()

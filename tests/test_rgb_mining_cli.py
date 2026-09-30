"""CPU-only integration and publication checks for the Step 1C RGB CLI."""

import contextlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from countermine.mining.candidate_miner import CANDIDATE_COLUMNS, ChunkedCandidateMiner


REPO_ROOT = Path(__file__).resolve().parents[1]


class RGBMiningCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = importlib.import_module("tools.04_mine_rgb_candidates")

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output_dir = self.root / "output"
        self.manifest_path = self.root / "manifest.csv"
        self.descriptor_path = self.root / "descriptors_fp16.npy"
        self.manifest = pd.DataFrame(
            {
                "row_index": np.arange(6, dtype=np.int64),
                "image_id": [f"real-image-{i}" for i in range(6)],
                "place_uid": ["p0", "p0", "p1", "p2", "p3", "p4"],
                "city_id": ["Boston", "Boston", "London", "Boston", "London", "London"],
                "lat": [0.0, 0.0, 0.0, 0.002, 0.004, 0.02],
                "lon": np.zeros(6),
            }
        )
        self.manifest.to_csv(self.manifest_path, index=False)
        descriptors = np.array(
            [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0.6, 0.8, 0], [0, 0, 1], [-0.6, 0.8, 0]],
            dtype=np.float32,
        )
        np.save(self.descriptor_path, descriptors.astype(np.float16))
        self.top_k = 4

    def miner(self, query_chunk_size=2):
        return ChunkedCandidateMiner(
            self.descriptor_path,
            self.manifest,
            top_k=self.top_k,
            query_chunk_size=query_chunk_size,
            reference_chunk_size=3,
            device="cpu",
        )

    def arguments(self):
        return [
            "--manifest", str(self.manifest_path),
            "--descriptors", str(self.descriptor_path),
            "--output-dir", str(self.output_dir),
            "--top-k", str(self.top_k),
            "--query-chunk-size", "2",
            "--reference-chunk-size", "3",
            "--device", "cpu",
            "--seed", "42",
        ]

    def validate(self, candidates, expected_query_start=0):
        return self.cli.validate_candidate_chunk(
            candidates, self.manifest, self.top_k, expected_query_start
        )

    def test_cpu_cli_writes_candidates_and_exact_summary_statistics(self):
        command = [sys.executable, str(REPO_ROOT / "tools/04_mine_rgb_candidates.py"),
                   *self.arguments()]
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True,
                                timeout=60, check=False)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        csv_path = self.output_dir / "rgb_candidates_raw.csv"
        summary_path = self.output_dir / "rgb_candidates_raw_summary.json"
        actual = pd.read_csv(csv_path, float_precision="round_trip")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(actual.columns.tolist(), list(CANDIDATE_COLUMNS))
        self.assertEqual(self.validate(actual), len(self.manifest))
        self.assertEqual(len(actual), len(self.manifest) * self.top_k)
        self.assertFalse((actual["query_place_uid"] == actual["negative_place_uid"]).any())
        self.assertFalse((actual["query_row_index"] == actual["negative_row_index"]).any())
        # Geographic exclusion defaults to zero; co-located different places remain eligible.
        colocated = actual[(actual["query_row_index"] == 0) & (actual["negative_row_index"] == 2)]
        self.assertEqual(len(colocated), 1)
        self.assertEqual(colocated.iloc[0]["geo_distance_m"], 0.0)
        self.assertEqual(summary["total_candidate_pairs"], len(actual))
        self.assertEqual(summary["number_of_queries"], len(self.manifest))
        self.assertEqual(summary["top_k"], self.top_k)
        self.assertEqual(summary["descriptor_shape"], [6, 3])
        self.assertEqual(summary["descriptor_dtype"], "float16")
        self.assertEqual(summary["device"], "cpu")
        self.assertEqual(summary["min_geo_distance_m"], 0.0)
        saved_seed = summary.get("config", {}).get("seed", summary.get("seed"))
        self.assertEqual(saved_seed, 42)
        similarities = actual["similarity"].to_numpy(dtype=np.float64)
        distances = actual["geo_distance_m"].to_numpy()
        for key, values in (("similarity_statistics", similarities),
                            ("geo_distance_statistics", distances)):
            statistics = summary[key]
            self.assertAlmostEqual(statistics["min"], float(values.min()), places=9)
            self.assertAlmostEqual(statistics["max"], float(values.max()), places=9)
            self.assertAlmostEqual(statistics["median"], float(np.median(values)), places=9)
            for name, quantile in (("q05", 0.05), ("q25", 0.25), ("q75", 0.75), ("q95", 0.95)):
                self.assertAlmostEqual(statistics[name], float(np.quantile(values, quantile)), places=9)
        self.assertAlmostEqual(summary["similarity_statistics"]["mean"],
                               float(similarities.mean()), places=9)
        self.assertAlmostEqual(summary["fraction_same_city"], float(actual["same_city"].mean()),
                               places=9)
        for threshold in (150, 200, 250, 500, 1000):
            self.assertEqual(summary["geo_distance_counts_below_m"][str(threshold)],
                             int((distances < threshold).sum()))
        self.assertEqual({path.name for path in self.output_dir.iterdir()},
                         {csv_path.name, summary_path.name})

    def test_validator_accepts_complete_ordered_query_chunks(self):
        next_query = 0
        for chunk in self.miner().iter_candidates():
            query_count = self.validate(chunk, expected_query_start=next_query)
            self.assertEqual(query_count, 2)
            next_query += query_count
        self.assertEqual(next_query, len(self.manifest))

    def test_validator_rejects_bad_schema_and_incomplete_or_unordered_queries(self):
        valid = self.miner().mine()
        cases = {
            "missing_column": valid.drop(columns="pair_uid"),
            "extra_column": valid.assign(extra=1),
            "reordered_columns": valid.loc[:, list(reversed(CANDIDATE_COLUMNS))],
            "empty_chunk": valid.iloc[:0].copy(),
            "incomplete_query": valid.iloc[:-1].copy(),
            "query_gap": valid[valid["query_row_index"] != 2].reset_index(drop=True),
            "reversed_queries": valid.iloc[::-1].reset_index(drop=True),
        }
        for name, candidates in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                self.validate(candidates)
        with self.assertRaises(ValueError):
            self.validate(valid, expected_query_start=1)

    def test_validator_rejects_corrupt_ranks_similarities_uids_and_row_indices(self):
        valid = next(self.miner().iter_candidates())
        corruptions = (
            ("rank", 0), ("rank", 2), ("rank", 1.5),
            ("similarity", np.nan), ("similarity", np.inf), ("similarity", -np.inf),
            ("pair_uid", ""), ("pair_uid", "  "), ("pair_uid", None), ("pair_uid", 123),
            ("query_row_index", -1), ("query_row_index", 6), ("query_row_index", 0.5),
            ("negative_row_index", -1), ("negative_row_index", 6), ("negative_row_index", 1.5),
        )
        for column, value in corruptions:
            with self.subTest(column=column, value=value):
                candidates = valid.copy()
                # Use object storage to avoid pandas assignment warnings for nonintegers.
                if value is None or isinstance(value, str) or column == "pair_uid":
                    candidates[column] = candidates[column].astype(object)
                elif column in ("rank", "query_row_index", "negative_row_index") and isinstance(value, float):
                    candidates[column] = candidates[column].astype(float)
                candidates.loc[candidates.index[0], column] = value
                with self.assertRaises(ValueError):
                    self.validate(candidates)

    def test_validator_rejects_duplicate_pairs_same_places_and_metadata_misalignment(self):
        valid = next(self.miner().iter_candidates())
        duplicate = valid.copy()
        duplicate.iloc[1] = duplicate.iloc[0]
        duplicate.loc[duplicate.index[1], "rank"] = 2
        with self.assertRaises(ValueError):
            self.validate(duplicate)

        same_place = valid.copy()
        same_place.loc[same_place.index[0], "negative_row_index"] = 1
        for column in ("image_id", "place_uid", "city_id"):
            same_place.loc[same_place.index[0], f"negative_{column}"] = self.manifest.iloc[1][column]
        same_place.loc[same_place.index[0], "same_city"] = True
        with self.assertRaises(ValueError):
            self.validate(same_place)
        disguised_same_place = same_place.copy()
        disguised_same_place.loc[disguised_same_place.index[0], "negative_place_uid"] = "claimed-other-place"
        with self.assertRaises(ValueError):
            self.validate(disguised_same_place)

        for column in ("query_image_id", "negative_image_id", "query_place_uid", "negative_place_uid",
                       "query_city_id", "negative_city_id", "same_city"):
            with self.subTest(column=column):
                candidates = valid.copy()
                candidates.loc[candidates.index[0], column] = (
                    not candidates.iloc[0][column] if column == "same_city" else "wrong-metadata"
                )
                with self.assertRaises(ValueError):
                    self.validate(candidates)
        candidates = valid.copy()
        candidates["same_city"] = candidates["same_city"].astype(np.int64)
        with self.assertRaises(ValueError):
            self.validate(candidates)

    def test_late_invalid_chunk_preserves_published_outputs_and_cleans_partials(self):
        self.output_dir.mkdir()
        csv_path = self.output_dir / "rgb_candidates_raw.csv"
        summary_path = self.output_dir / "rgb_candidates_raw_summary.json"
        original_csv = b"existing-published-candidates\n"
        original_summary = b'{"existing": true}\n'
        csv_path.write_bytes(original_csv)
        summary_path.write_bytes(original_summary)
        miner = self.miner()
        chunks = list(miner.iter_candidates())
        chunks[-1] = chunks[-1].copy()
        chunks[-1].loc[chunks[-1].index[0], "similarity"] = np.nan
        with mock.patch.object(self.cli, "ChunkedCandidateMiner", return_value=miner), \
                mock.patch.object(miner, "iter_candidates", return_value=iter(chunks)), \
                mock.patch.object(sys, "argv", ["04_mine_rgb_candidates.py", *self.arguments()]), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            self.cli.main()
        self.assertEqual(csv_path.read_bytes(), original_csv)
        self.assertEqual(summary_path.read_bytes(), original_summary)
        self.assertEqual({path.name for path in self.output_dir.iterdir()},
                         {csv_path.name, summary_path.name})


if __name__ == "__main__":
    unittest.main()

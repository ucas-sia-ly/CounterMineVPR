"""CPU-only checks for bounded RGB candidate diagnostics."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from countermine.mining.candidate_diagnostics import CandidateDiagnostics


class CandidateDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def table():
        query_rows = [0, 0, 1, 1, 2, 3]
        negative_rows = [1, 2, 0, 3, 3, 2]
        return pd.DataFrame({
            "query_row_index": query_rows,
            "negative_row_index": negative_rows,
            "query_image_id": [f"image-{row}" for row in query_rows],
            "negative_image_id": [f"image-{row}" for row in negative_rows],
            "query_place_uid": [f"place-{row}" for row in query_rows],
            "negative_place_uid": [f"place-{row}" for row in negative_rows],
            "query_city_id": ["a"] * 6,
            "negative_city_id": ["a", "b", "a", "a", "b", "a"],
            "rank": [1, 2, 1, 2, 1, 1],
            "similarity": [0.9, 0.9, 0.9, 0.3, -0.2, 0.0],
            "geo_distance_m": [100.0, 150.0, 200.0, 250.0, 500.0, 1000.0],
            "same_city": [True, False, True, True, False, True],
            "pair_uid": ["0-1", "0-2", "0-1", "1-3", "2-3", "2-3"],
        })

    def test_exact_statistics_reverse_pairs_and_strict_thresholds(self):
        table = self.table()
        with CandidateDiagnostics() as diagnostics:
            diagnostics.add(table.iloc[:3])
            diagnostics.add(table.iloc[3:])
            summary = diagnostics.summary()
            self.assertIs(summary, diagnostics.summary())
            json.dumps(summary, allow_nan=False)
        self.assertEqual(summary["number_of_queries"], 4)
        self.assertEqual(summary["total_candidate_pairs"], 6)
        self.assertEqual(summary["unique_pair_uids"], 4)
        self.assertEqual(summary["reverse_duplicate_pairs"], 2)
        self.assertEqual(summary["reverse_duplicated_candidate_rows"], 4)
        self.assertAlmostEqual(summary["reverse_duplicate_fraction"], 4 / 6)
        self.assertAlmostEqual(summary["fraction_same_city"], 4 / 6)
        self.assertEqual(summary["rank_distribution"], {"1": 4, "2": 2})
        self.assertEqual(summary["geo_distance_counts_below_m"], {
            "150": 1, "200": 2, "250": 3, "500": 4, "1000": 5,
        })
        for source, key in (("similarity", "similarity_statistics"), ("geo_distance_m", "geo_distance_statistics")):
            expected = np.quantile(table[source], [0.05, 0.25, 0.50, 0.75, 0.95])
            actual = [summary[key][name] for name in ("q05", "q25", "median", "q75", "q95")]
            np.testing.assert_allclose(actual, expected)
            self.assertEqual(summary[key]["min"], table[source].min())
            self.assertEqual(summary[key]["max"], table[source].max())
        self.assertAlmostEqual(summary["similarity_statistics"]["mean"], table["similarity"].mean())
        self.assertEqual(
            [(row["query_row_index"], row["negative_row_index"]) for row in summary["top_candidates"][:3]],
            [(0, 1), (0, 2), (1, 0)],
        )

    def test_chunk_boundaries_do_not_change_statistics_or_top_rows(self):
        table = self.table()
        summaries = []
        for chunk_size in (1, 2, 6):
            with CandidateDiagnostics() as diagnostics:
                for start in range(0, len(table), chunk_size):
                    diagnostics.add(table.iloc[start:start + chunk_size])
                summaries.append(diagnostics.summary())
        for summary in summaries[1:]:
            self.assertEqual(summary["top_candidates"], summaries[0]["top_candidates"])
            self.assertEqual(summary["geo_distance_statistics"], summaries[0]["geo_distance_statistics"])
            self.assertEqual(summary["rank_distribution"], summaries[0]["rank_distribution"])
            self.assertAlmostEqual(summary["similarity_statistics"]["mean"], summaries[0]["similarity_statistics"]["mean"])

    def test_duplicate_detection_is_global_and_rejection_is_atomic(self):
        table = self.table()
        with CandidateDiagnostics() as diagnostics:
            diagnostics.add(table.iloc[:2])
            rejected = pd.concat([table.iloc[2:3], table.iloc[:1]])
            with self.assertRaisesRegex(ValueError, "duplicated"):
                diagnostics.add(rejected)
            diagnostics.add(table.iloc[2:])
            self.assertEqual(diagnostics.summary()["total_candidate_pairs"], len(table))

    def test_invalid_rows_and_finalization_are_rejected(self):
        examples = (
            ("similarity", np.inf, "finite"),
            ("geo_distance_m", -1.0, "nonnegative"),
            ("pair_uid", " ", "nonempty"),
            ("same_city", "False", "boolean"),
            ("rank", 0, "integers"),
            ("query_row_index", 0.5, "integers"),
            ("negative_row_index", -1, "integers"),
        )
        for column, invalid, message in examples:
            with self.subTest(column=column), CandidateDiagnostics() as diagnostics:
                table = self.table()
                # object dtype avoids silently truncating intentionally invalid values.
                table[column] = table[column].astype(object)
                table.loc[0, column] = invalid
                with self.assertRaisesRegex(ValueError, message):
                    diagnostics.add(table)
        with CandidateDiagnostics() as diagnostics:
            with self.assertRaisesRegex(ValueError, "no rows"):
                diagnostics.summary()
            diagnostics.add(self.table())
            diagnostics.summary()
            with self.assertRaisesRegex(RuntimeError, "finalized"):
                diagnostics.add(self.table())

    def test_temp_files_are_cleaned_on_context_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            with CandidateDiagnostics(directory) as diagnostics:
                diagnostics.add(self.table())
                diagnostics.summary()
                self.assertEqual(len(list(Path(directory).iterdir())), 1)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()

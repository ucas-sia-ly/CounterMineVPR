"""Raw shard serialization must preserve frozen ECDF ties without rounding."""

from pathlib import Path
import tempfile
import unittest

from countermine.mining.full_structural_calibration import calibrate_full_evidence
from countermine.mining.full_structural_io import write_csv
from countermine.mining.full_structural_measurement import _atomic_merge
from countermine.mining.structural_analysis import read_metadata_csv, validate_metrics
from test_structural_calibration import measured_tables


class FullShardCSVTieTests(unittest.TestCase):
    def test_numeric_match_outputs_and_streamed_shards_preserve_frozen_tie(self):
        candidates, controls, _ = measured_tables()
        # A valid match fraction whose shortest decimal and .17g strings
        # parse differently under the frozen pandas numeric convention.
        for table in (candidates, controls):
            table.loc[0, "num_keypoints_a"] = 101
            table.loc[0, "num_keypoints_b"] = 120
            table.loc[0, "num_matches"] = 52
            table.loc[0, "local_match_ratio"] = 52 / 101
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "cache/countermine_rgb/step2d"
            # Emulate the already frozen Step 2A control serialization.
            runtime.mkdir(parents=True)
            control_path = runtime / "fixture_frozen_controls.csv"
            controls.to_csv(control_path, index=False, lineterminator="\n")
            null = validate_metrics(read_metadata_csv(control_path), candidate=False)
            parts = []
            for index, frame in enumerate((candidates.iloc[:2], candidates.iloc[2:])):
                path = runtime / f"fixture_shard_{index}.csv"
                write_csv(path, validate_metrics(frame, candidate=True), repo_root=root)
                parts.append(path)
            destination = runtime / "fixture_full_metrics.csv"
            _atomic_merge(parts, destination, repo_root=root)
            expected_bytes = parts[0].read_bytes() + parts[1].read_bytes().split(b"\n", 1)[1]
            self.assertEqual(destination.read_bytes(), expected_bytes)
            actual = validate_metrics(read_metadata_csv(destination), candidate=True)
            self.assertEqual(actual.loc[0, "local_match_ratio"], null.loc[0, "local_match_ratio"])
            calibrated = calibrate_full_evidence(actual, null)
            self.assertEqual(calibrated.loc[0, "ratio_null_percentile"], 1)
            self.assertEqual(calibrated.loc[0, "structural_bottleneck"], 1)
            self.assertTrue(calibrated.loc[0, "core_q95"])
            self.assertTrue(calibrated.loc[0, "core_q99"])

            # The fixture exposes the precise former failure: fixed 17-digit
            # formatting breaks the tie and changes empirical core labels.
            wrong = runtime / "fixture_rounded_metrics.csv"
            candidates.to_csv(wrong, index=False, float_format="%.17g", lineterminator="\n")
            rounded = validate_metrics(read_metadata_csv(wrong), candidate=True)
            self.assertLess(rounded.loc[0, "local_match_ratio"], null.loc[0, "local_match_ratio"])
            broken = calibrate_full_evidence(rounded, null)
            self.assertEqual(broken.loc[0, "ratio_null_percentile"], 2 / 3)
            self.assertFalse(broken.loc[0, "core_q95"])
            self.assertFalse(broken.loc[0, "core_q99"])


if __name__ == "__main__":
    unittest.main()

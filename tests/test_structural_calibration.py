"""CPU-only frozen-input, matched-null and structural-evidence tests."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from countermine.mining.structural_analysis import (
    ALIKED_CONFIG, LIGHTGLUE_CONFIG, METRIC_COLUMNS, RANK_BINS,
    analyze_tables, build_snapshot, sha256_file,
)
from countermine.mining.structural_calibration import (
    calibrate_structural_evidence, load_calibrated_artifacts, load_step2a_evidence,
    run_calibration, stable_pair_uid, validate_calibrated_candidates, weak_ecdf,
)


def measured_tables(count=5, same_city_count=3):
    candidates, controls, manifest = [], [], []
    for index in range(count):
        city_a, city_b = "a", "a" if index < same_city_count else "b"
        image_a, image_b, random_b = f"image-{index:04d}-a", f"image-{index:04d}-b", f"random-{index:04d}"
        row_a, row_b, row_random = 3 * index, 3 * index + 1, 3 * index + 2
        places = (f"{city_a}:place-{index:04d}-a", f"{city_b}:place-{index:04d}-b", f"{city_b}:random-{index:04d}")
        for row_index, image_id, place, city in ((row_a, image_a, places[0], city_a), (row_b, image_b, places[1], city_b), (row_random, random_b, places[2], city_b)):
            manifest.append({"row_index": row_index, "image_id": image_id, "place_uid": place, "city_id": city})
        rank = (1, 2, 6, 11, 21)[index % 5]
        matches, null_matches = (25, 20, 10, 40, 0)[index % 5], (10, 20, 30, 0, 10)[index % 5]
        pair = {
            "pair_uid": stable_pair_uid(image_a, image_b), "image_id_a": image_a, "image_id_b": image_b,
            "row_index_a": row_a, "row_index_b": row_b,
            "place_uid_a": places[0], "place_uid_b": places[1], "city_id_a": city_a, "city_id_b": city_b,
            "geo_distance_m": 300.0 if index == 0 else 1000.0, "same_city": city_a == city_b,
            "num_candidate_directions": 1, "rank_bin": RANK_BINS[index % 5], "best_rgb_rank": rank,
            "max_salad_similarity": .5 - (index % 5) * .05, "mean_salad_similarity": .4 - (index % 5) * .05,
            "num_keypoints_a": 100, "num_keypoints_b": 100, "num_matches": matches, "local_match_ratio": matches / 100,
            "matched_source_cell_coverage": .5, "matched_target_cell_coverage": .25, "symmetric_match_coverage": .25,
            "source_max_cell_match_fraction": .5, "target_max_cell_match_fraction": .6,
            "source_match_entropy": .4, "target_match_entropy": .3, "symmetric_match_entropy": .3,
            "exact_pixel_duplicate": False,
        }
        control = dict(pair)
        control.update(
            candidate_pair_uid=pair["pair_uid"], random_pair_uid=stable_pair_uid(image_a, random_b),
            anchor_query_image_id=image_a, anchor_query_row_index=row_a,
            random_negative_image_id=random_b, random_negative_row_index=row_random,
            image_id_b=random_b, place_uid_b=places[2], random_control_available=True,
            num_matches=null_matches, local_match_ratio=null_matches / 100,
        )
        candidates.append(pair)
        controls.append(control)
    return pd.DataFrame(candidates), pd.DataFrame(controls), pd.DataFrame(manifest)


def completed_fixture(root):
    candidates, controls, manifest = measured_tables()
    manifest.to_csv(root / "manifest.csv", index=False)
    for name in ("raw.csv", "raw_summary.json"):
        (root / name).write_text("frozen original input\n")
    fingerprints = {
        image_id: {"image_file_sha256": str(index).zfill(64), "rgb_pixel_sha256": str(index).zfill(64)}
        for index, image_id in enumerate(sorted(manifest["image_id"]))
    }
    for table in (candidates, controls):
        for endpoint in ("a", "b"):
            table[f"rgb_pixel_sha_{endpoint}"] = table[f"image_id_{endpoint}"].map(lambda uid: fingerprints[uid]["rgb_pixel_sha256"])
            table[f"image_file_sha_{endpoint}"] = table[f"image_id_{endpoint}"].map(lambda uid: fingerprints[uid]["image_file_sha256"])
    population = candidates.drop(columns=list(METRIC_COLUMNS))
    random_inputs = controls.drop(columns=list(METRIC_COLUMNS))
    population.to_csv(root / "population.csv", index=False)
    random_inputs.to_csv(root / "random_controls.csv", index=False)
    candidates.to_csv(root / "candidate_structural_metrics.csv", index=False)
    controls.to_csv(root / "random_structural_metrics.csv", index=False)
    (root / "image_fingerprints.json").write_text(json.dumps(fingerprints))
    hashes = {
        "manifest_sha256": sha256_file(root / "manifest.csv"), "candidate_csv_sha256": sha256_file(root / "raw.csv"),
        "candidate_summary_sha256": sha256_file(root / "raw_summary.json"),
        "population_sha256": sha256_file(root / "population.csv"), "random_controls_sha256": sha256_file(root / "random_controls.csv"),
    }
    population_summary = {
        **hashes, "seed": 42, "min_geo_distance_m": 250, "max_pairs": 5000,
        "sampled_pair_count": 5, "matched_random_count": 5,
        "raw_directed_candidate_rows": 5, "eligible_geo_directed_rows": 5, "eligible_unique_canonical_pairs": 5,
        "reverse_candidate_count": 0, "same_city_count": 3, "cross_city_count": 2,
        "rank_bin_counts": dict.fromkeys(RANK_BINS, 1),
    }
    (root / "population_summary.json").write_text(json.dumps(population_summary))
    hashes["population_summary_sha256"] = sha256_file(root / "population_summary.json")
    config = {
        "schema_version": 1, "seed": 42, "input_hashes": hashes,
        "image_policy": {"real_rgb_only": True, "synthetic_images_used": False, "geometry": [640, 480], "local_resize": None, "registration": None},
        "aliked_config": ALIKED_CONFIG, "lightglue_config": LIGHTGLUE_CONFIG,
        "input_image_index_sha256": sha256_file(root / "image_fingerprints.json"),
        "source_sha256": {}, "vendor_provenance": {}, "matcher_provenance": {},
    }
    (root / "measurement_config.json").write_text(json.dumps(config))
    completion = {
        "schema_version": 1, "complete": True, "input_hashes": hashes,
        "expected_candidate_count": 5, "expected_random_count": 5, "candidate_count": 5, "random_count": 5,
        "config_sha256": sha256_file(root / "measurement_config.json"),
        "candidate_metrics_sha256": sha256_file(root / "candidate_structural_metrics.csv"),
        "random_metrics_sha256": sha256_file(root / "random_structural_metrics.csv"),
    }
    (root / "measurement_summary.json").write_text(json.dumps(completion))
    _, report = analyze_tables(candidates, controls)
    snapshot = build_snapshot(report, population_summary, config, "d" * 40)
    snapshot["provenance"]["measurement_config_sha256"] = sha256_file(root / "measurement_config.json")
    (root / "snapshot.json").write_text(json.dumps(snapshot))
    return candidates, controls, snapshot, completion


class NullCalibrationTests(unittest.TestCase):
    def test_weak_ecdf_ties_and_nonfinite_inputs(self):
        np.testing.assert_array_equal(weak_ecdf([.2, .1, .2, .4], [.2, 0, .4, .3]), [.75, 0, 1, .75])
        for null, queries in (([], [.1]), ([np.nan], [.1]), ([.1], [np.inf])):
            with self.assertRaises(ValueError):
                weak_ecdf(null, queries)

    def test_global_fallback_and_salad_population_percentile(self):
        candidates, controls, _ = measured_tables()
        calibrated, metadata = calibrate_structural_evidence(candidates, controls)
        self.assertEqual(metadata["relation_null_counts"], {"same_city": 3, "cross_city": 2})
        self.assertEqual(metadata["relation_percentile_unavailable_count"], 5)
        for short in ("ratio", "match_count", "coverage", "entropy"):
            self.assertTrue(calibrated[f"relation_{short}_null_percentile"].isna().all())
            self.assertEqual(set(calibrated[f"{short}_null_source"]), {"global"})
            np.testing.assert_array_equal(calibrated[f"{short}_null_percentile"], calibrated[f"global_{short}_null_percentile"])
            self.assertTrue(np.isfinite(calibrated[f"{short}_null_percentile"]).all())
        np.testing.assert_allclose(calibrated["salad_similarity_percentile"], [1, .8, .6, .4, .2])

    def test_relation_null_requires_100_and_records_primary_source(self):
        candidates, controls, _ = measured_tables(199, 100)
        controls.loc[controls["same_city"], ["num_matches", "local_match_ratio"]] = [10, .1]
        controls.loc[~controls["same_city"], ["num_matches", "local_match_ratio"]] = [30, .3]
        calibrated, metadata = calibrate_structural_evidence(candidates, controls)
        self.assertEqual(metadata["relation_null_available"], {"same_city": True, "cross_city": False})
        self.assertEqual(calibrated.loc[0, "ratio_null_source"], "same_city")
        self.assertEqual(calibrated.loc[0, "ratio_null_percentile"], 1)
        self.assertAlmostEqual(calibrated.loc[0, "global_ratio_null_percentile"], 100 / 199)
        self.assertEqual(calibrated.loc[100, "ratio_null_source"], "global")
        self.assertTrue(pd.isna(calibrated.loc[100, "relation_ratio_null_percentile"]))
        candidates, controls, _ = measured_tables(200, 100)
        calibrated, metadata = calibrate_structural_evidence(candidates, controls)
        self.assertTrue(metadata["relation_null_available"]["cross_city"])
        self.assertEqual(calibrated.loc[100, "ratio_null_source"], "cross_city")

    def test_bottleneck_geomean_and_small_denominator_without_coverage_gate(self):
        candidates, controls, _ = measured_tables()
        candidates.loc[0, ["num_keypoints_a", "num_keypoints_b", "num_matches", "local_match_ratio"]] = [10, 10, 5, .5]
        candidates.loc[1, ["num_matches", "local_match_ratio"]] = [40, .4]
        calibrated, _ = calibrate_structural_evidence(candidates, controls)
        self.assertEqual(calibrated.loc[0, "ratio_null_percentile"], 1)
        self.assertLess(calibrated.loc[0, "structural_bottleneck"], calibrated.loc[1, "structural_bottleneck"])
        np.testing.assert_array_equal(calibrated["structural_bottleneck"], np.minimum(calibrated["ratio_null_percentile"], calibrated["match_count_null_percentile"]))
        np.testing.assert_allclose(calibrated["structural_geomean"], np.sqrt(calibrated["ratio_null_percentile"] * calibrated["match_count_null_percentile"]))
        candidates[["matched_source_cell_coverage", "matched_target_cell_coverage", "symmetric_match_coverage"]] = 0
        low_coverage, _ = calibrate_structural_evidence(candidates, controls)
        np.testing.assert_array_equal(calibrated["structural_bottleneck"], low_coverage["structural_bottleneck"])
        np.testing.assert_array_equal(calibrated["structural_geomean"], low_coverage["structural_geomean"])
        self.assertTrue((low_coverage["coverage_null_percentile"] == 0).all())

    def test_invalid_percentile_formula_source_and_random_identity_rejected(self):
        candidates, controls, _ = measured_tables()
        calibrated, _ = calibrate_structural_evidence(candidates, controls)
        for column, value in (("ratio_null_percentile", np.inf), ("structural_bottleneck", .999), ("ratio_null_source", "cross_city")):
            corrupted = calibrated.copy()
            corrupted.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_calibrated_candidates(corrupted)
        controls.loc[1, "pair_uid"] = controls.loc[0, "pair_uid"]
        with self.assertRaisesRegex(ValueError, "unique"):
            calibrate_structural_evidence(candidates, controls)


class FrozenEvidenceTests(unittest.TestCase):
    def arguments(self, root):
        return (root, root / "manifest.csv", root / "raw.csv", root / "raw_summary.json", root / "snapshot.json")

    def test_completed_snapshot_evidence_and_fixed_production_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed_fixture(root)
            candidates, controls, _, _, _, provenance = load_step2a_evidence(*self.arguments(root), expected_candidate_count=5, expected_random_count=5)
            self.assertEqual((len(candidates), len(controls)), (5, 5))
            self.assertTrue(provenance["no_new_local_matching"])
            self.assertTrue(provenance["validation"]["all_snapshot_report_fields_reproduced"])
            self.assertFalse(provenance["validation"]["measurement_csv_hashes_present_in_historical_snapshot"])
            with self.assertRaisesRegex(ValueError, "5000 candidates and 4999"):
                load_step2a_evidence(*self.arguments(root))

    def test_every_snapshot_report_field_and_top_identity_are_bound(self):
        for field in ("paired_candidate_vs_random", "candidate_summary", "top50_structural_candidates"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, _, snapshot, _ = completed_fixture(root)
                if field == "paired_candidate_vs_random":
                    snapshot[field]["fraction_candidate_gt_random"] = .123
                elif field == "candidate_summary":
                    snapshot[field]["mean"] += .01
                else:
                    snapshot[field][0]["image_id_a"] = "wrong-image"
                (root / "snapshot.json").write_text(json.dumps(snapshot))
                with self.assertRaisesRegex(ValueError, "frozen Step 2A snapshot"):
                    load_step2a_evidence(*self.arguments(root), expected_candidate_count=5, expected_random_count=5)

    def test_population_and_matcher_configuration_hashes_are_bound(self):
        for field in ("measurement_config_sha256", "input_hashes"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, _, snapshot, _ = completed_fixture(root)
                if field == "input_hashes":
                    snapshot["provenance"][field]["population_sha256"] = "f" * 64
                else:
                    snapshot["provenance"][field] = "f" * 64
                (root / "snapshot.json").write_text(json.dumps(snapshot))
                with self.assertRaisesRegex(ValueError, "frozen Step 2A snapshot"):
                    load_step2a_evidence(*self.arguments(root), expected_candidate_count=5, expected_random_count=5)

    def test_canonical_pair_identity_validation_has_no_model_dependency(self):
        candidates, controls, manifest = measured_tables()
        from countermine.mining.structural_calibration import _validate_endpoint_identities
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            manifest.to_csv(path, index=False)
            _validate_endpoint_identities(candidates, controls, path)
            candidates.loc[0, "pair_uid"] = "wrong-pair"
            with self.assertRaisesRegex(ValueError, "canonical endpoint"):
                _validate_endpoint_identities(candidates, controls, path)
            candidates, controls, manifest = measured_tables()
            controls.loc[0, "random_pair_uid"] = "wrong-random-pair"
            with self.assertRaisesRegex(ValueError, "canonical endpoint"):
                _validate_endpoint_identities(candidates, controls, path)

    def test_reproducible_atomic_calibration_resume_and_mixing_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "step2a"
            runtime.mkdir()
            completed_fixture(runtime)
            output = root / "step2b"
            module = "countermine.mining.structural_calibration"
            real_loader = load_step2a_evidence
            def fixture_loader(*args, **kwargs):
                return real_loader(*args, expected_candidate_count=5, expected_random_count=5)
            with patch(f"{module}.load_step2a_evidence", side_effect=fixture_loader), patch(f"{module}._source_paths", return_value={"calibration.py": "d" * 64}):
                summary = run_calibration(*self.arguments(runtime), output, repo_root=root)
                artifact = output / "calibrated_candidates.csv"
                initial_hash = sha256_file(artifact)
                initial_mtime = artifact.stat().st_mtime_ns
                self.assertEqual(run_calibration(*self.arguments(runtime), output, repo_root=root), summary)
                self.assertEqual(sha256_file(artifact), initial_hash)
                self.assertEqual(artifact.stat().st_mtime_ns, initial_mtime)
                frame, loaded = load_calibrated_artifacts(output, source_validation=False)
                self.assertEqual(len(frame), 5)
                self.assertEqual(loaded, summary)
                self.assertNotIn(str(root), json.dumps(loaded, allow_nan=False))
                artifact.write_text(artifact.read_text() + "changed")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    load_calibrated_artifacts(output, source_validation=False)
                with self.assertRaisesRegex(ValueError, "refusing to mix"):
                    run_calibration(*self.arguments(runtime), output, repo_root=root)

    def test_cpu_import_and_cli_with_all_model_runtime_imports_blocked(self):
        repository = Path(__file__).resolve().parents[1]
        code = """
import importlib.abc, runpy, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0].lower() in {'torch', 'torchvision', 'lightglue', 'aliked', 'salad', 'diffusers', 'iclight'}:
            raise RuntimeError('forbidden model import: ' + fullname)
sys.meta_path.insert(0, NoModels())
import countermine.mining.structural_calibration
sys.argv = ['11_calibrate_structural_evidence.py', '--help']
runpy.run_path('tools/11_calibrate_structural_evidence.py', run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=repository, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--snapshot", result.stdout)

    def test_destination_aliases_preserve_step2a_original_inputs_and_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "copied_step2a"
            runtime.mkdir()
            completed_fixture(runtime)
            canonical = root / "cache/countermine_rgb/step2a"
            canonical.mkdir(parents=True)
            audits = root / "docs/audits"
            audits.mkdir(parents=True)
            figure = audits / "step2a_top50_structural_candidates.jpg"
            figure.write_bytes(b"frozen curated figure")
            arguments = self.arguments(runtime)
            canonical_alias = root / "canonical_alias"
            canonical_alias.symlink_to(canonical, target_is_directory=True)
            for output in (runtime, runtime / "nested", canonical, canonical / "nested", canonical_alias):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "must not modify"):
                    run_calibration(*arguments, output, repo_root=root)
            for target, name in ((runtime / "raw.csv", "calibrated_candidates.csv"), (figure, "calibration_summary.json")):
                output = root / f"alias_{name}"
                output.mkdir()
                (output / name).symlink_to(target)
                before = target.read_bytes()
                with self.subTest(target=target), self.assertRaisesRegex(ValueError, "must not modify"):
                    run_calibration(*arguments, output, repo_root=root)
                self.assertEqual(target.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

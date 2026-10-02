"""CPU-only descriptive null, paired audit, and compact export checks."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from countermine.mining.structural_analysis import (
    ALIKED_CONFIG, LIGHTGLUE_CONFIG, METRIC_COLUMNS, RANK_BINS, add_null_percentiles,
    analyze_tables, build_snapshot, create_figures, create_top_montage, curate_audit_artifacts,
    distribution, load_completed_metrics, select_top_structural_candidates, sha256_file,
    validate_original_image_fingerprints, validate_snapshot, weak_ecdf, write_snapshot,
)


def tables():
    candidates = []
    random = []
    for index, (rank, matches, random_matches) in enumerate(((1, 3, 1), (2, 2, 2), (6, 1, 3), (11, 4, 0), (21, 0, 1))):
        city_a, city_b = "a", "a" if index % 2 == 0 else "b"
        row = {
            "pair_uid": f"pair-{index}", "image_id_a": f"image-{index}-a", "image_id_b": f"image-{index}-b",
            "place_uid_a": f"place-{index}-a", "place_uid_b": f"place-{index}-b", "city_id_a": city_a,
            "city_id_b": city_b, "geo_distance_m": 300.0 if index == 0 else 1000.0,
            "same_city": city_a == city_b, "rank_bin": RANK_BINS[index], "best_rgb_rank": rank,
            "max_salad_similarity": .5 - index * .05, "mean_salad_similarity": .4 - index * .05,
            "num_keypoints_a": 10, "num_keypoints_b": 10, "num_matches": matches,
            "local_match_ratio": matches / 10, "matched_source_cell_coverage": .0625,
            "matched_target_cell_coverage": .03125, "symmetric_match_coverage": .03125,
            "source_max_cell_match_fraction": .5, "target_max_cell_match_fraction": .5,
            "source_match_entropy": .3, "target_match_entropy": .2, "symmetric_match_entropy": .2,
            "exact_pixel_duplicate": False,
        }
        candidates.append(row)
        control = dict(row)
        control.update(
            candidate_pair_uid=row["pair_uid"], image_id_b=f"random-{index}", place_uid_b=f"random-place-{index}",
            anchor_query_image_id=row["image_id_a"], random_negative_image_id=f"random-{index}",
            num_matches=random_matches, local_match_ratio=random_matches / 10,
        )
        random.append(control)
    return pd.DataFrame(candidates), pd.DataFrame(random)


class StructuralAnalysisTests(unittest.TestCase):
    def test_weak_ecdf_includes_ties_and_preserves_query_order(self):
        actual = weak_ecdf([.2, .1, .2, .4], [.2, 0, .4, .3])
        np.testing.assert_array_equal(actual, [.75, 0, 1, .75])
        with self.assertRaisesRegex(ValueError, "at least one"):
            weak_ecdf([], [.1])
        with self.assertRaisesRegex(ValueError, "finite"):
            weak_ecdf([np.inf], [.1])

    def test_relation_null_uses_exactly_100_controls_and_global_all_controls(self):
        candidates, random = tables()
        same = pd.concat([random.iloc[[0]]] * 100, ignore_index=True)
        cross = pd.concat([random.iloc[[1]]] * 99, ignore_index=True)
        same["local_match_ratio"] = .1
        cross["local_match_ratio"] = .2
        enriched = add_null_percentiles(candidates, pd.concat([same, cross], ignore_index=True))
        self.assertEqual(enriched.loc[0, "relation_local_null_percentile"], 1)
        self.assertEqual(enriched.loc[2, "relation_local_null_percentile"], 1)
        self.assertEqual(enriched.loc[4, "relation_local_null_percentile"], 0)
        self.assertTrue(pd.isna(enriched.loc[1, "relation_local_null_percentile"]))
        self.assertAlmostEqual(enriched.loc[2, "global_local_null_percentile"], 100 / 199)

    def test_paired_delta_signs_rank_and_city_groups(self):
        candidates, random = tables()
        enriched, report = analyze_tables(candidates, random)
        paired = report["paired_candidate_vs_random"]
        self.assertEqual(paired["candidate_gt_random"], 2)
        self.assertEqual(paired["candidate_eq_random"], 1)
        self.assertEqual(paired["candidate_lt_random"], 2)
        self.assertEqual(paired["fraction_candidate_gt_random"], .4)
        self.assertAlmostEqual(paired["delta_local_match_ratio"]["mean"], .06)
        self.assertAlmostEqual(paired["delta_local_match_ratio"]["std"], np.std([.2, 0, -.2, .4, -.1], ddof=1))
        self.assertEqual(list(report["rank_conditioned"]), list(RANK_BINS))
        for rank in RANK_BINS:
            self.assertEqual(report["rank_conditioned"][rank]["pair_count"], 1)
        self.assertEqual(report["city_relation_conditioned"]["same_city"]["candidate_summary"]["count"], 3)
        self.assertEqual(report["city_relation_conditioned"]["cross_city"]["candidate_summary"]["count"], 2)
        self.assertEqual(report["integrity_audit"]["near_geo_500m_count"], 1)
        self.assertTrue(enriched.loc[0, "near_geo_500m"])
        self.assertNotIn("p_value", json.dumps(report))

    def test_statistics_are_sample_std_and_empty_or_singleton_are_finite(self):
        stats = distribution([1, 2, 3])
        self.assertEqual(stats["std"], 1)
        self.assertEqual(stats["median"], 2)
        self.assertEqual(stats["q01"], 1.02)
        self.assertIsNone(distribution([1])["std"])
        self.assertIsNone(distribution([])["mean"])
        json.dumps(distribution([]), allow_nan=False)

    def test_top_selection_all_tiebreaks_and_duplicates_excluded(self):
        candidates, _ = tables()
        base = candidates.iloc[[0]].copy()
        entries = []
        for uid, matches, similarity, duplicate in (("z", 2, .9, False), ("b", 3, .8, False), ("a", 3, .8, False), ("c", 3, .9, False), ("d", 10, 1, True)):
            row = base.iloc[0].to_dict()
            row.update(pair_uid=uid, local_match_ratio=.5, num_matches=matches, max_salad_similarity=similarity, exact_pixel_duplicate=duplicate)
            entries.append(row)
        selected = select_top_structural_candidates(pd.DataFrame(entries))
        self.assertEqual(selected["pair_uid"].tolist(), ["c", "a", "b", "z"])
        self.assertEqual(select_top_structural_candidates(pd.DataFrame(entries).sample(frac=1, random_state=42))["pair_uid"].tolist(), ["c", "a", "b", "z"])

    def test_invalid_ratio_rank_bin_and_relation_rejected(self):
        for column, value, message in (("local_match_ratio", .99, "disagrees"), ("rank_bin", "rank_21_50", "boundaries"), ("geo_distance_m", 249, ">=250"), ("max_salad_similarity", np.nan, "finite")):
            candidates, random = tables()
            candidates.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, message):
                analyze_tables(candidates, random)
        candidates, random = tables()
        random.loc[0, "rank_bin"] = "rank_21_50"
        with self.assertRaisesRegex(ValueError, "relation/rank"):
            analyze_tables(candidates, random)

    def test_correlations_are_descriptive_tie_rank_and_constant_safe(self):
        candidates, random = tables()
        _, report = analyze_tables(candidates, random)
        expected = np.corrcoef(candidates["best_rgb_rank"].rank(), candidates["local_match_ratio"].rank())[0, 1]
        self.assertAlmostEqual(report["similarity_local_correlations"]["best_rgb_rank_vs_local_match_ratio"]["spearman"], expected)
        candidates["max_salad_similarity"] = .5
        _, report = analyze_tables(candidates, random)
        self.assertIsNone(report["similarity_local_correlations"]["max_salad_similarity_vs_local_match_ratio"]["pearson"])


class StructuralExportTests(unittest.TestCase):
    def snapshot(self):
        candidates, random = tables()
        _, report = analyze_tables(candidates, random)
        population = {
            "manifest_sha256": "a" * 64, "candidate_csv_sha256": "b" * 64, "candidate_summary_sha256": "c" * 64,
            "min_geo_distance_m": 250, "max_pairs": 5000, "seed": 42, "raw_directed_candidate_rows": 10,
            "eligible_geo_directed_rows": 10, "eligible_unique_canonical_pairs": 5, "sampled_pair_count": 5,
            "matched_random_count": 5, "reverse_candidate_count": 5, "same_city_count": 3, "cross_city_count": 2,
            "rank_bin_counts": dict.fromkeys(RANK_BINS, 1), "dataset_root": "/private/dataset",
        }
        config = {"aliked_config": ALIKED_CONFIG, "lightglue_config": LIGHTGLUE_CONFIG, "private_path": "/private/weights"}
        return build_snapshot(report, population, config, "d" * 40)

    def test_snapshot_finite_path_free_real_rgb_flags_and_null(self):
        snapshot = self.snapshot()
        self.assertTrue(snapshot["provenance"]["real_rgb_only"])
        self.assertFalse(snapshot["provenance"]["synthetic_images_used"])
        payload = json.dumps(snapshot, allow_nan=False)
        self.assertNotIn("/private", payload)
        self.assertIsNone(snapshot["top50_structural_candidates"][0]["relation_local_null_percentile"])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "snapshot.json"
            write_snapshot(target, snapshot)
            self.assertEqual(json.loads(target.read_text()), snapshot)
        snapshot["candidate_summary"]["mean"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_snapshot(snapshot)

    def test_absolute_paths_and_non_rgb_flags_rejected(self):
        for path in ("/home/image.jpg", "C:\\images\\a.jpg", "file:///secret"):
            snapshot = self.snapshot()
            snapshot["top50_structural_candidates"][0]["image_id_a"] = path
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "absolute"):
                validate_snapshot(snapshot)
        snapshot = self.snapshot()
        snapshot["provenance"]["synthetic_images_used"] = True
        with self.assertRaisesRegex(ValueError, "real RGB"):
            validate_snapshot(snapshot)

    def test_cpu_module_and_cli_import_with_model_tensor_imports_blocked(self):
        root = Path(__file__).resolve().parents[1]
        code = """
import importlib.abc, runpy, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'lightglue', 'salad', 'diffusers'}:
            raise RuntimeError('forbidden import: ' + fullname)
sys.meta_path.insert(0, NoModels())
import countermine.mining.structural_analysis
sys.argv = ['09_analyze_structural_audit.py', '--help']
runpy.run_path('tools/09_analyze_structural_audit.py', run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--runtime-dir", result.stdout)

    def test_required_figures_and_original_rgb_montage_and_curated_copies(self):
        candidates, random = tables()
        candidates, _ = analyze_tables(candidates, random)
        top = select_top_structural_candidates(candidates, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs"
            manifest = []
            for endpoint, color in (("a", (255, 0, 0)), ("b", (0, 0, 255))):
                row = top.iloc[0]
                relative = f"image-{endpoint}.png"
                Image.new("RGB", (640, 480), color).save(root / relative)
                manifest.append({"image_id": row[f"image_id_{endpoint}"], "relative_path": relative, "place_uid": row[f"place_uid_{endpoint}"], "city_id": row[f"city_id_{endpoint}"]})
            pd.DataFrame(manifest).to_csv(root / "manifest.csv", index=False)
            create_figures(candidates, random, output)
            create_top_montage(top, root / "manifest.csv", root, output / "top50_structural_candidates.jpg")
            with Image.open(output / "top50_structural_candidates.jpg") as montage:
                self.assertEqual(montage.size, (1316, 650))
                self.assertGreater(montage.getpixel((12 + 320, 52 + 240))[0], 240)
                self.assertGreater(montage.getpixel((12 + 640 + 12 + 320, 52 + 240))[2], 240)
            copied = curate_audit_artifacts(output, root / "audits")
            self.assertEqual(len(copied), 4)
            for target in copied:
                self.assertGreater(target.stat().st_size, 100)
            with self.assertRaisesRegex(FileNotFoundError, "top20"):
                curate_audit_artifacts(output, root / "audits", require_overlays=True)

    def _completed_fixture(self, root):
        candidates, random = tables()
        used = set(candidates["image_id_a"]) | set(candidates["image_id_b"]) | set(random["image_id_b"])
        fingerprints = {
            image_id: {"image_file_sha256": str(index).zfill(64), "rgb_pixel_sha256": str(index).zfill(64)}
            for index, image_id in enumerate(sorted(used))
        }
        for table in (candidates, random):
            for endpoint in ("a", "b"):
                table[f"rgb_pixel_sha_{endpoint}"] = table[f"image_id_{endpoint}"].map(lambda image_id: fingerprints[image_id]["rgb_pixel_sha256"])
                table[f"image_file_sha_{endpoint}"] = table[f"image_id_{endpoint}"].map(lambda image_id: fingerprints[image_id]["image_file_sha256"])
        population = candidates.drop(columns=list(METRIC_COLUMNS)).copy()
        controls = random.drop(columns=list(METRIC_COLUMNS)).copy()
        controls["random_control_available"] = True
        random["random_control_available"] = True
        population.to_csv(root / "population.csv", index=False)
        controls.to_csv(root / "random_controls.csv", index=False)
        candidates.to_csv(root / "candidate_structural_metrics.csv", index=False)
        random.to_csv(root / "random_structural_metrics.csv", index=False)
        for name in ("manifest.csv", "raw.csv", "raw_summary.json"):
            (root / name).write_text("original input\n")
        (root / "image_fingerprints.json").write_text(json.dumps(fingerprints))
        hashes = {
            "manifest_sha256": sha256_file(root / "manifest.csv"), "candidate_csv_sha256": sha256_file(root / "raw.csv"),
            "candidate_summary_sha256": sha256_file(root / "raw_summary.json"),
            "population_sha256": sha256_file(root / "population.csv"), "random_controls_sha256": sha256_file(root / "random_controls.csv"),
        }
        population_summary = {**hashes, "seed": 42, "min_geo_distance_m": 250, "sampled_pair_count": 5, "matched_random_count": 5}
        (root / "population_summary.json").write_text(json.dumps(population_summary))
        hashes["population_summary_sha256"] = sha256_file(root / "population_summary.json")
        config = {
            "schema_version": 1, "seed": 42, "input_hashes": hashes,
            "image_policy": {"real_rgb_only": True, "synthetic_images_used": False, "geometry": [640, 480], "local_resize": None, "registration": None},
            "aliked_config": ALIKED_CONFIG, "lightglue_config": LIGHTGLUE_CONFIG,
            "input_image_index_sha256": sha256_file(root / "image_fingerprints.json"),
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
        return completion

    def test_completion_provenance_checksums_and_identity_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completion = self._completed_fixture(root)
            arguments = (root, root / "manifest.csv", root / "raw.csv", root / "raw_summary.json")
            candidates, random, _, _ = load_completed_metrics(*arguments)
            self.assertEqual(len(candidates), 5)
            self.assertEqual(len(random), 5)
            completion["complete"] = False
            (root / "measurement_summary.json").write_text(json.dumps(completion))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_completed_metrics(*arguments)
            self._completed_fixture(root)
            (root / "raw.csv").write_text("changed input")
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_completed_metrics(*arguments)
            completion = self._completed_fixture(root)
            measured = pd.read_csv(root / "candidate_structural_metrics.csv")
            measured.loc[0, "image_id_b"] = "wrong-original-image"
            measured.to_csv(root / "candidate_structural_metrics.csv", index=False)
            completion["candidate_metrics_sha256"] = sha256_file(root / "candidate_structural_metrics.csv")
            (root / "measurement_summary.json").write_text(json.dumps(completion))
            with self.assertRaisesRegex(ValueError, "metadata image_id_b"):
                load_completed_metrics(*arguments)

    def test_original_image_file_bytes_must_match_inference_fingerprints(self):
        candidates, random = tables()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            used = set(candidates["image_id_a"]) | set(candidates["image_id_b"]) | set(random["image_id_b"])
            manifest, fingerprints = [], {}
            for image_id in sorted(used):
                target = root / f"{image_id}.png"
                Image.new("RGB", (640, 480), (100, 150, 200)).save(target)
                manifest.append({"image_id": image_id, "relative_path": target.name})
                fingerprints[image_id] = {"image_file_sha256": sha256_file(target)}
            pd.DataFrame(manifest).to_csv(root / "manifest.csv", index=False)
            (root / "image_fingerprints.json").write_text(json.dumps(fingerprints))
            validate_original_image_fingerprints(root, candidates, random, root / "manifest.csv", root)
            (root / f"{sorted(used)[0]}.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed after inference"):
                validate_original_image_fingerprints(root, candidates, random, root / "manifest.csv", root)


if __name__ == "__main__":
    unittest.main()

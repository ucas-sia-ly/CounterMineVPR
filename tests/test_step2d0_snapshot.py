"""Small fake scalar artifacts exercise the stdlib-only Step 2D0 exporter."""

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools/14_export_step2d0_snapshot.py"
SPEC = importlib.util.spec_from_file_location("step2d0_snapshot_exporter", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_summary(values, delta=False):
    available = [value for value in values if value is not None]
    if delta:
        result = {"median": statistics.median(available) if available else None,
                  "min": min(available) if available else None, "max": max(available) if available else None}
    else:
        quantiles = statistics.quantiles(available, n=20, method="inclusive") if available else [None] * 19
        result = {"median": statistics.median(available) if available else None,
                  "q05": quantiles[0], "q25": quantiles[4], "q75": quantiles[14], "q95": quantiles[18]}
    return {**result, "valid_count": len(available), "missing_count": len(values) - len(available)}


class Step2D0SnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest_path = self.root / "geometry_manifest.csv"
        self.generation_path = self.root / "geometry_generation_summary.json"
        self.fidelity_path = self.root / "geometry_fidelity.csv"
        self.paired_path = self.root / "geometry_paired.csv"
        self.summary_path = self.root / "geometry_summary.json"
        self.output = self.root / "docs/audits/step2d0_metrics.json"
        self.manifest, self.records = [], []
        for index in range(10):
            for policy in exporter.POLICIES:
                square = policy == "square_crop_512"
                scale = 512 / (480 if square else 640)
                width, height = (512, 512) if square else (512, 384)
                self.manifest.append({
                    "audit_index": index, "row_index": 100 + index, "image_id": f"image-{index}", "policy": policy,
                    "source_path": f"cache/geometry_audit/{policy}/source/{100 + index:08d}.png",
                    "original_path": f"data/fake/Images/{index}.jpg",
                    "original_width": 640, "original_height": 480, "canonical_width": width,
                    "canonical_height": height, "scale_x": scale, "scale_y": scale,
                    "retained_area_fraction": .75 if square else 1.0,
                    "retained_long_axis_fraction": .75 if square else 1.0,
                })
                displacement = 2 + index * .1 if square else 1 + index * .05
                row = {
                    "audit_index": index, "row_index": 100 + index, "image_id": f"image-{index}",
                    "policy": policy, "mode": "full_scene", "canonical_width": width,
                    "canonical_height": height, "scale_x": scale, "scale_y": scale,
                    "num_keypoints_source": 200, "num_keypoints_relit": 200,
                    "num_matches": 100, "match_ratio_min": .5,
                    "grid_coverage_4px": .75 if square else .875,
                    "grid_coverage_8px": .75 if square else .875,
                    "displacement_median": displacement, "displacement_q95": displacement + .3,
                }
                for epsilon, legacy, original in zip((2, 4, 8, 16),
                        ((.35, .42, .47, .5) if square else (.40, .45, .5, .5)),
                        ((.35, .40 + .001 * index, .45 + .001 * index, .49) if square
                         else (.34, .39 + .002 * index, .43 + .0015 * index, .50))):
                    row[f"repeatability_min_{epsilon}px"] = legacy
                    row[f"precision_matches_{epsilon}px"] = legacy / .5
                    row[f"repeatability_original_{epsilon}px"] = original
                    row[f"precision_original_{epsilon}px"] = original / .5
                for name, offset in (("mean", 0), ("median", 0), ("q75", .1), ("q90", .2), ("q95", .3), ("max", .4)):
                    row[f"displacement_original_{name}"] = (displacement + offset) / scale
                self.records.append(row)
        self.write_inputs()

    def write_inputs(self):
        write_csv(self.manifest_path, self.manifest)
        write_csv(self.fidelity_path, self.records)
        paired, legacy_deltas = [], []
        for index in range(10):
            square, full = [row for row in self.records if row["audit_index"] == index]
            def deltas(fields):
                return {name: None if square[metric] is None or full[metric] is None else full[metric] - square[metric]
                        for name, metric in fields.items()}
            paired.append({"audit_index": index, "row_index": 100 + index, **deltas(exporter.DELTAS)})
            legacy_deltas.append(deltas(exporter.LEGACY_DELTAS))
        write_csv(self.paired_path, paired)
        generation = {
            "number_of_sources": 10, "count_outputs": 20, "manifest_sha256": sha256(self.manifest_path),
            "snapshot_sha256": "a" * 64,
            "config": {"seed": 12345, "mode": "full_scene", "policies": list(exporter.POLICIES),
                       "configs_by_geometry": {f"512x{height}": {
                           "seed": 12345, "width": 512, "height": height,
                           "prompt": "soft diffuse overcast daylight, uniform outdoor illumination, natural lighting",
                       } for height in (512, 384)}},
            "runs": [{name: row[name] for name in
                      ("audit_index", "row_index", "image_id", "policy", "canonical_width", "canonical_height")}
                     | {"mode": "full_scene", "seed": 12345, "source_sha256": "b" * 64,
                        "output_sha256": "c" * 64}
                     for row in self.manifest],
        }
        self.generation_path.write_text(json.dumps(generation), encoding="utf-8")
        def policies(names):
            return {policy: {"count_pairs": 10, "metrics": {
                name: metric_summary([row[name] for row in self.records if row["policy"] == policy])
                for name in names
            }} for policy in exporter.POLICIES}
        gate_policies = {}
        for policy in exporter.POLICIES:
            rows = [row for row in self.records if row["policy"] == policy]
            passed = sum(row["repeatability_min_8px"] >= .4 and row["grid_coverage_8px"] >= .6
                         and row["displacement_q95"] is not None and row["displacement_q95"] <= 5 for row in rows)
            gate_policies[policy] = {"pass_count": passed, "total_count": 10, "acceptance_fraction": passed / 10}
        summary = {
            "number_of_sources": 10, "count_pairs": 20, "manifest_sha256": sha256(self.manifest_path),
            "snapshot_sha256": "a" * 64, "generation_summary_sha256": sha256(self.generation_path),
            "config": {"seed": 42, "mode": "full_scene", "extract_resize": None, "registration": None},
            "matcher": {"compiled": False, "extract_resize": None, "registration": None,
                        "extractor_settings": {"max_num_keypoints": 2048, "detection_threshold": .2},
                        "matcher_settings": {"features": "aliked", "depth_confidence": -1, "width_confidence": -1,
                                             "filter_threshold": .1, "mp": False},
                        "models": {"extractor_config": {"model_name": "aliked-n16", "nms_radius": 2},
                                   "matcher_config": {"weights": "aliked_lightglue", "flash": True, "n_layers": 9},
                                   "lightglue_weights_version": "v0.1_arxiv",
                                   "checkpoints": {"unused_image_hash": "d" * 64, "cache_path": "/private/models"}}},
            "fair_original_pixel_metrics": policies(exporter.FAIR_METRICS),
            "paired_original_pixel_deltas": {name: metric_summary([row[name] for row in paired], True)
                                              for name in exporter.DELTAS},
            "output_pixel_metrics": {"cross_policy_comparable": False, "reason": exporter.REASON,
                                     "policies": policies(exporter.LEGACY_METRICS)},
            "legacy_output_pixel_deltas": {"cross_policy_comparable": False, "reason": exporter.REASON,
                                           "metrics": {name: metric_summary([row[name] for row in legacy_deltas], True)
                                                       for name in exporter.LEGACY_DELTAS}},
            "legacy_output_pixel_gate": {"cross_policy_comparable": False, "reason": exporter.REASON,
                                         "thresholds": {"repeatability_min_8px_min": .4, "grid_coverage_8px_min": .6,
                                                        "displacement_q95_max": 5.0, "diagnostic_only": True,
                                                        "filters_data": False}, "policies": gate_policies},
        }
        self.summary_path.write_text(json.dumps(summary), encoding="utf-8")

    def export(self, output=None):
        return exporter.export_snapshot(self.manifest_path, self.generation_path, self.fidelity_path,
                                        self.paired_path, self.summary_path, output or self.output,
                                        git_commit="1234567890abcdef", repo_root=self.root)

    def edit_summary(self, edit):
        data = json.loads(self.summary_path.read_text())
        edit(data)
        self.summary_path.write_text(json.dumps(data), encoding="utf-8")

    def assert_rejected_preserves_output(self, pattern=None):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text("previous completed snapshot\n")
        with self.assertRaisesRegex(ValueError, pattern or ".+"):
            self.export()
        self.assertEqual(self.output.read_text(), "previous completed snapshot\n")
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])

    def test_snapshot_copies_exact_saved_statistics_and_preserves_six_sections(self):
        expected = json.loads(self.summary_path.read_text())
        result = self.export()
        self.assertEqual(set(result), {"provenance", "geometry", "fair_original_pixel_metrics",
                                     "paired_original_pixel_deltas", "per_source", "legacy_output_pixel_metrics"})
        self.assertEqual(result["fair_original_pixel_metrics"], expected["fair_original_pixel_metrics"])
        self.assertEqual(result["paired_original_pixel_deltas"], expected["paired_original_pixel_deltas"])
        legacy = result["legacy_output_pixel_metrics"]
        self.assertEqual(legacy["policies"], expected["output_pixel_metrics"]["policies"])
        self.assertEqual(legacy["legacy_output_pixel_deltas"], expected["legacy_output_pixel_deltas"])
        self.assertEqual(legacy["legacy_output_pixel_gate"], expected["legacy_output_pixel_gate"])
        self.assertIs(legacy["cross_policy_comparable"], False)
        self.assertEqual(legacy["coverage"]["grid_shape"], [8, 8])
        self.assertIs(legacy["coverage"]["qualification_mask_changed"], False)
        provenance = result["provenance"]
        self.assertEqual(provenance["frozen_step2c_snapshot_sha256"], "a" * 64)
        self.assertEqual((provenance["iclight_seed"], provenance["fidelity_seed"]), (12345, 42))
        self.assertEqual(provenance["aliked_config"]["model_name"], "aliked-n16")
        self.assertEqual(provenance["lightglue_config"]["weights_version"], "v0.1_arxiv")
        self.assertIsNone(provenance["registration"])
        self.assertIsNone(provenance["resize"])
        self.assertEqual(len(result["per_source"]), 10)
        for row in result["per_source"]:
            self.assertEqual(set(row["policies"]), set(exporter.POLICIES))
            self.assertEqual(set(row["paired_original_pixel_deltas"]), set(exporter.DELTAS))
        square, full = (result["geometry"][policy] for policy in exporter.POLICIES)
        self.assertEqual(square["scale_x"], 512 / 480)
        self.assertEqual(full["scale_x"], 512 / 640)
        self.assertEqual((full["canonical_width"], full["canonical_height"]), (512, 384))
        serialized = self.output.read_text()
        self.assertNotIn("NaN", serialized)
        self.assertNotIn("Infinity", serialized)
        self.assertNotIn("/private/models", serialized)
        self.assertNotIn("b" * 64, serialized)
        self.assertNotIn("d" * 64, serialized)
        self.assertEqual(json.loads(serialized), result)
        first = self.output.read_bytes()
        self.export()
        self.assertEqual(first, self.output.read_bytes())

    def test_genuinely_undefined_displacement_and_paired_delta_use_null(self):
        row = self.records[0]
        for name in exporter.COUNTS:
            row[name] = 0
        for name in set(exporter.LEGACY_METRICS + exporter.ORIGINAL_METRICS) - set(exporter.COUNTS):
            row[name] = None if name.startswith("displacement_") else 0.0
        self.write_inputs()
        result = self.export()
        self.assertIsNone(result["per_source"][0]["policies"][exporter.POLICIES[0]]["displacement_original_q95"])
        self.assertIsNone(result["per_source"][0]["paired_original_pixel_deltas"]["delta_displacement_original_q95"])
        self.assertEqual(result["paired_original_pixel_deltas"]["delta_displacement_original_q95"]["missing_count"], 1)

    def test_nonfinite_csv_and_saved_statistics_are_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad=bad):
                self.write_inputs()
                rows = copy.deepcopy(self.records)
                rows[0]["repeatability_original_8px"] = bad
                write_csv(self.fidelity_path, rows)
                self.assert_rejected_preserves_output("finite")
                self.write_inputs()
                self.edit_summary(lambda data: data["fair_original_pixel_metrics"][exporter.POLICIES[0]]["metrics"]
                                  ["repeatability_original_8px"].update(median=bad))
                self.assert_rejected_preserves_output("finite")

    def test_required_summaries_counts_and_statistics_must_agree(self):
        edits = (
            lambda data: data.pop("fair_original_pixel_metrics"),
            lambda data: data["fair_original_pixel_metrics"].pop(exporter.POLICIES[1]),
            lambda data: data["fair_original_pixel_metrics"][exporter.POLICIES[0]]["metrics"].pop("num_matches"),
            lambda data: data["fair_original_pixel_metrics"][exporter.POLICIES[0]]["metrics"]["num_matches"].pop("q95"),
            lambda data: data["fair_original_pixel_metrics"][exporter.POLICIES[0]]["metrics"]["num_matches"].update(valid_count=9),
            lambda data: data["fair_original_pixel_metrics"][exporter.POLICIES[0]]["metrics"]["num_matches"].update(median=99),
            lambda data: data.update(number_of_sources=9),
            lambda data: data.update(count_pairs=19),
            lambda data: data["legacy_output_pixel_gate"]["policies"][exporter.POLICIES[0]].update(pass_count=0),
            lambda data: data["output_pixel_metrics"].update(cross_policy_comparable=True),
        )
        for index, edit in enumerate(edits):
            with self.subTest(index=index):
                self.write_inputs()
                self.edit_summary(edit)
                self.assert_rejected_preserves_output()

    def test_wrong_pair_counts_identity_or_delta_are_rejected(self):
        self.records.pop()
        write_csv(self.fidelity_path, self.records)
        self.assert_rejected_preserves_output("20")
        self.setUp()
        rows = copy.deepcopy(self.records)
        rows[-1] = dict(rows[0])
        write_csv(self.fidelity_path, rows)
        self.assert_rejected_preserves_output("duplicate")
        self.write_inputs()
        with self.paired_path.open(newline="") as handle:
            paired = list(csv.DictReader(handle))
        paired[0]["delta_R8_original"] = 0.25
        write_csv(self.paired_path, paired)
        self.assert_rejected_preserves_output("inconsistent")

    def test_anisotropic_or_varying_per_policy_geometry_is_rejected(self):
        self.manifest[0]["scale_y"] = 1.1
        self.write_inputs()
        self.assert_rejected_preserves_output("anisotropic")
        self.manifest[0]["scale_y"] = 512 / 480
        # Each variant is individually valid, but one original has double resolution.
        for index in (0, 1):
            self.manifest[index]["original_width"] = 1280
            self.manifest[index]["original_height"] = 960
            self.manifest[index]["scale_x"] /= 2
            self.manifest[index]["scale_y"] /= 2
        self.write_inputs()
        self.assert_rejected_preserves_output("geometry varies")

    def test_undefined_metrics_are_only_allowed_when_they_are_genuine(self):
        self.records[0]["displacement_original_q95"] = None
        write_csv(self.fidelity_path, self.records)
        self.assert_rejected_preserves_output("finite")
        self.records[0]["displacement_original_q95"] = 2.0
        self.write_inputs()
        self.edit_summary(lambda data: data["fair_original_pixel_metrics"][exporter.POLICIES[0]]["metrics"]
                          ["repeatability_original_8px"].update(median=None))
        self.assert_rejected_preserves_output("undefined")

    def test_repeatability_precision_counts_and_zero_match_ratios_must_agree(self):
        rows = copy.deepcopy(self.records)
        rows[0]["precision_original_8px"] = .5
        write_csv(self.fidelity_path, rows)
        self.assert_rejected_preserves_output("counts")
        row = self.records[0]
        row["num_matches"] = 0
        for name in set(exporter.LEGACY_METRICS + exporter.ORIGINAL_METRICS) - set(exporter.COUNTS):
            row[name] = None if name.startswith("displacement_") else 0.0
        row["precision_original_8px"] = .1
        self.write_inputs()
        self.assert_rejected_preserves_output("zero")

    def test_absolute_isotropy_tolerance_has_no_relative_expansion(self):
        self.manifest[0]["scale_y"] += 1.04e-12
        self.write_inputs()
        self.assert_rejected_preserves_output("anisotropic")

    def test_absolute_paths_in_written_provenance_are_rejected(self):
        for value in ("/private/source.jpg", "C:\\private\\source.jpg", "\\\\server\\share\\source.jpg",
                      "file:///private/source.jpg", "note: /private/source.jpg"):
            with self.subTest(value=value):
                self.write_inputs()
                self.edit_summary(lambda data: data["matcher"]["models"]["extractor_config"].update(model_name=value))
                self.assert_rejected_preserves_output("absolute filesystem path")

    def test_inconsistent_generation_provenance_and_seeds_are_rejected(self):
        for edit in (
            lambda data: data.update(snapshot_sha256="f" * 64),
            lambda data: data.update(generation_summary_sha256="f" * 64),
            lambda data: data["config"].update(seed=7),
            lambda data: data["config"].update(registration="homography"),
            lambda data: data["matcher"]["models"]["extractor_config"].update(max_num_keypoints=1024),
        ):
            self.write_inputs()
            self.edit_summary(edit)
            self.assert_rejected_preserves_output()

    def test_output_cannot_overwrite_inputs_or_protected_artifacts(self):
        before = self.summary_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "input artifacts"):
            self.export(self.summary_path)
        self.assertEqual(before, self.summary_path.read_bytes())
        for suffix in ("third_party/new.json", "salad/new.json", "cache/generator_audit/new.json",
                       "outputs/step2/new.json", "docs/audits/step2c_metrics.json"):
            with self.subTest(suffix=suffix):
                output = self.root / suffix
                with self.assertRaisesRegex(ValueError, "(protected|frozen)"):
                    self.export(output)
                self.assertFalse(output.exists())

    def test_atomic_write_failure_preserves_previous_snapshot(self):
        self.export()
        before = self.output.read_bytes()
        with patch.object(exporter.os, "replace", side_effect=OSError("publication failed")):
            with self.assertRaisesRegex(OSError, "publication failed"):
                self.export()
        self.assertEqual(before, self.output.read_bytes())
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])

    def test_import_and_help_require_only_standard_library(self):
        code = """import builtins, importlib.util, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'PIL', 'diffusers', 'transformers', 'countermine'}:
        raise AssertionError('non-stdlib dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('isolated_exporter', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main(['--help'])
"""
        result = subprocess.run([sys.executable, "-c", code, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--generation-summary", result.stdout)


if __name__ == "__main__":
    unittest.main()

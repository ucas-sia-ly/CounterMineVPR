"""Stdlib-only fake scalar artifacts; no images, weights, models or CUDA."""

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
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "tools/18_export_native_fov_snapshot.py"
SPEC = importlib.util.spec_from_file_location("native_snapshot", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values, paired=False):
    available = sorted(value for value in values if value is not None)
    def q(probability):
        if not available:
            return None
        if probability == .5:
            return statistics.median(available)
        if probability == 0:
            return min(available)
        if probability == 1:
            return max(available)
        return statistics.quantiles(available, n=20, method="inclusive")[round(probability * 20) - 1]
    probabilities = exporter.PAIRED_QUANTILES if paired else exporter.NATIVE_QUANTILES
    result = {key: q(probability) for key, probability in probabilities.items()}
    if paired:
        result.update(num_positive=sum(value > 0 for value in available),
                      num_zero=sum(value == 0 for value in available),
                      num_negative=sum(value < 0 for value in available))
    result.update(valid_count=len(available), missing_count=len(values) - len(available))
    return result


class NativeSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest_path = self.root / "native_manifest.csv"
        self.generation_path = self.root / "native_generation_summary.json"
        self.fidelity_path = self.root / "native_fidelity.csv"
        self.paired_path = self.root / "native_paired.csv"
        self.summary_path = self.root / "native_summary.json"
        self.history_path = self.root / "step2d0_metrics.json"
        self.step2c_path = self.root / "step2c_metrics.json"
        self.output = self.root / "docs/audits/step2d1_native_fov_metrics.json"
        self.aliked = {"max_num_keypoints": 2048, "detection_threshold": .2, "model_name": "aliked-n16"}
        self.lightglue = {"features": "aliked", "depth_confidence": -1, "width_confidence": -1,
                          "filter_threshold": .1, "mp": False, "compiled": False}
        config = {"seed": 12345, "cfg": 2.0, "steps": 25, "highres_scale": 1.0,
                  "highres_denoise": .5, "prompt": "soft diffuse overcast daylight",
                  "width": 512, "height": 512, "device": "cuda",
                  "base_model": "fake-base", "offset_filename": "iclight_sd15_fc.safetensors",
                  "scheduler_algorithm_type": "sde-dpmsolver++"}
        self.step2c = {"provenance": {"iclight_config": config, "aliked_config": self.aliked,
                                      "lightglue_config": self.lightglue}}
        self.history = {"provenance": {"aliked_config": self.aliked, "lightglue_config": self.lightglue},
                        "per_source": []}
        self.manifest, self.records = [], []
        for index in range(10):
            source = {"audit_index": index, "row_index": index + 100, "image_id": f"Images/fake/{index}.jpg"}
            policies = {policy: {"num_matches": 200, "repeatability_original_4px": .45 if policy == exporter.BASELINES[0] else .40,
                                "repeatability_original_8px": .48 if policy == exporter.BASELINES[0] else .44,
                                "displacement_original_q95": 3.4 if policy == exporter.BASELINES[0] else 5.0,
                                "grid_coverage_8px": .765625} for policy in exporter.BASELINES}
            self.history["per_source"].append({**source, "policies": policies})
            geometry = {"policy": exporter.POLICY, "canonical_width": 640, "canonical_height": 480,
                        "scale_x": 1.0, "scale_y": 1.0}
            self.manifest.append({**source, **geometry, "original_width": 640, "original_height": 480,
                                  "retained_area_fraction": 1.0, "retained_long_axis_fraction": 1.0})
            offset = (index % 3 - 1) * .01
            self.records.append({**source, **geometry, "mode": "full_scene",
                "num_keypoints_source": 400, "num_keypoints_relit": 400, "num_matches": 200,
                "match_ratio_min": .5, "repeatability_2px": .3, "repeatability_4px": .45 + offset,
                "repeatability_8px": .47 + offset, "repeatability_16px": .5,
                "displacement_median": 1.5, "displacement_q95": 3.0 + index * .1,
                "grid_coverage_4px": .75, "grid_coverage_8px": .75})
        self.write_inputs()

    def write_inputs(self):
        self.step2c_path.write_text(json.dumps(self.step2c))
        self.history["provenance"]["frozen_step2c_snapshot_sha256"] = digest(self.step2c_path)
        self.history_path.write_text(json.dumps(self.history))
        write_csv(self.manifest_path, self.manifest)
        write_csv(self.fidelity_path, self.records)
        config = {**self.step2c["provenance"]["iclight_config"], "width": 640, "height": 480}
        hashes = {"manifest_sha256": digest(self.manifest_path), "historical_snapshot_sha256": digest(self.history_path),
                  "step2c_snapshot_sha256": digest(self.step2c_path)}
        runs = [{key: row[key] for key in ("audit_index", "row_index", "image_id", "policy", "mode",
                                         "canonical_width", "canonical_height")}
                | {"seed": 12345, "elapsed_seconds": .25, "peak_cuda_memory_allocated_bytes": 1000,
                   "peak_cuda_memory_reserved_bytes": 2000, "source_sha256": "b" * 64,
                   "output_sha256": "c" * 64} for row in self.records]
        generation = {**hashes, "number_of_sources": 10, "count_outputs": 10, "elapsed_seconds": 4.0,
                      "config": {"seed": 12345, "mode": "full_scene", "policy": exporter.POLICY,
                                 "iclight_config": config}, "runs": runs}
        self.generation_path.write_text(json.dumps(generation))
        paired = []
        for row, frozen in zip(self.records, self.history["per_source"]):
            pair = {key: row[key] for key in ("audit_index", "row_index")}
            for policy in exporter.BASELINES:
                for name, (native, original) in exporter.DELTA_FIELDS.items():
                    if policy == exporter.BASELINES[1] and name == "delta_coverage8":
                        continue
                    values = (row[native], frozen["policies"][policy][original])
                    pair[f"{name}_vs_{policy}"] = None if None in values else values[0] - values[1]
            paired.append(pair)
        write_csv(self.paired_path, paired)
        self.saved = {**hashes, "number_of_sources": 10, "count_pairs": 10,
            "generation_summary_sha256": digest(self.generation_path),
            "config": {"seed": 42, "mode": "full_scene", "policy": exporter.POLICY,
                       "grid_shape": [8, 8], "extract_resize": None, "registration": None},
            "matcher": {"extractor_settings": {"max_num_keypoints": 2048, "detection_threshold": .2},
                        "matcher_settings": {key: value for key, value in self.lightglue.items() if key != "compiled"},
                        "models": {"extractor_config": {"model_name": "aliked-n16"}, "matcher_config": {},
                                   "checkpoints": {"irrelevant_path": "/private/models"}}, "compiled": False},
            "native_metrics": {name: summary([row[name] for row in self.records]) for name in exporter.METRICS},
            "paired_comparisons": {policy: {name: summary([row[f"{name}_vs_{policy}"] for row in paired], True)
                for name in exporter.DELTA_FIELDS if policy == exporter.BASELINES[0] or name != "delta_coverage8"}
                for policy in exporter.BASELINES}}
        self.summary_path.write_text(json.dumps(self.saved))

    def export(self, output=None):
        return exporter.export_snapshot(self.manifest_path, self.generation_path, self.fidelity_path,
            self.paired_path, self.summary_path, self.history_path, self.step2c_path, output or self.output,
            git_commit="a" * 40, repo_root=self.root)

    def edit_summary(self, edit):
        data = json.loads(self.summary_path.read_text())
        edit(data)
        self.summary_path.write_text(json.dumps(data))

    def reject(self, pattern=".+"):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text("previous result\n")
        with self.assertRaisesRegex((ValueError, KeyError), pattern):
            self.export()
        self.assertEqual(self.output.read_text(), "previous result\n")

    def test_exact_summaries_geometry_runtime_and_compact_records(self):
        result = self.export()
        self.assertEqual(result["summary_native_metrics"], self.saved["native_metrics"])
        self.assertEqual(result["native_geometry"]["output_height"], 480)
        self.assertEqual(result["native_geometry"]["scale_x"], 1.0)
        self.assertEqual(len(result["per_source_native_metrics"]), 10)
        self.assertEqual(result["generation_runtime"]["peak_cuda_memory_reserved_bytes"], 2000)
        for policy in exporter.BASELINES:
            comparison = result[f"paired_comparison_vs_{policy}"]
            self.assertEqual(comparison["summary"], self.saved["paired_comparisons"][policy])
            self.assertEqual(len(comparison["per_source"]), 10)
        text = self.output.read_text()
        for excluded in ("NaN", "Infinity", "/private/models", "b" * 64, "c" * 64):
            self.assertNotIn(excluded, text)
        json.dumps(result, allow_nan=False)
        previous = self.output.read_bytes()
        self.export()
        self.assertEqual(previous, self.output.read_bytes())

    def test_nonfinite_csv_and_json_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            self.write_inputs()
            rows = copy.deepcopy(self.records)
            rows[0]["repeatability_8px"] = value
            write_csv(self.fidelity_path, rows)
            self.reject("finite")
            self.write_inputs()
            self.edit_summary(lambda data: data["native_metrics"]["repeatability_8px"].update(median=value))
            self.reject("finite")

    def test_absolute_paths_are_rejected(self):
        for value in ("/private/fake.jpg", "C:\\private\\fake.jpg", "\\\\host\\share\\fake.jpg",
                      "file:///private/fake.jpg", "note: /private/fake.jpg"):
            self.write_inputs()
            # A path in an image identifier passes identity joins, then must fail
            # output validation; unexported model cache paths remain omitted.
            self.records[0]["image_id"] = value
            self.manifest[0]["image_id"] = value
            self.history["per_source"][0]["image_id"] = value
            self.write_inputs()
            self.reject("absolute filesystem path")

    def test_zero_matches_use_genuine_null_displacements(self):
        self.records[0].update(num_matches=0, match_ratio_min=0.0, displacement_median=None,
                               displacement_q95=None, grid_coverage_4px=0.0, grid_coverage_8px=0.0)
        for epsilon in (2, 4, 8, 16):
            self.records[0][f"repeatability_{epsilon}px"] = 0.0
        self.write_inputs()
        result = self.export()
        self.assertIsNone(result["per_source_native_metrics"][0]["displacement_q95"])
        self.assertIsNone(result["paired_comparison_vs_full_fov_512"]["per_source"][0]["delta_D95"])
        self.assertEqual(result["summary_native_metrics"]["displacement_q95"]["missing_count"], 1)

    def test_defined_metrics_cannot_be_null(self):
        rows = copy.deepcopy(self.records)
        rows[0]["displacement_q95"] = None
        write_csv(self.fidelity_path, rows)
        self.reject("finite")

    def test_source_counts_geometry_and_identities_are_checked(self):
        write_csv(self.fidelity_path, self.records[:-1])
        self.reject("ten")
        self.write_inputs()
        self.manifest[0]["scale_x"] = .8
        self.write_inputs()
        self.reject("1.0")
        self.manifest[0]["scale_x"] = 1.0
        self.manifest[0]["canonical_height"] = 512
        self.write_inputs()
        self.reject("640x480")

    def test_paired_deltas_sign_counts_and_summary_values_are_checked(self):
        with self.paired_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["delta_R4_vs_square_crop_512"] = 1.0
        write_csv(self.paired_path, rows)
        self.reject("inconsistent")
        self.write_inputs()
        self.edit_summary(lambda data: data["paired_comparisons"]["square_crop_512"]["delta_R4"].update(num_positive=9))
        self.reject("sign counts")

    def test_frozen_provenance_and_settings_are_checked(self):
        for edit in (lambda data: data.update(historical_snapshot_sha256="f" * 64),
                     lambda data: data["config"].update(seed=0),
                     lambda data: data["config"].update(registration="homography"),
                     lambda data: data["matcher"]["extractor_settings"].update(max_num_keypoints=4096)):
            self.write_inputs()
            self.edit_summary(edit)
            self.reject()

    def test_required_saved_metrics_cannot_be_omitted(self):
        self.edit_summary(lambda data: data["native_metrics"].pop("num_matches"))
        self.reject()

    def test_protected_and_input_outputs_are_rejected(self):
        before = self.summary_path.read_bytes()
        with self.assertRaises(ValueError):
            self.export(self.summary_path)
        self.assertEqual(before, self.summary_path.read_bytes())
        for path in ("third_party/result.json", "salad/result.json", "cache/geometry_audit/result.json",
                     "docs/audits/step2d0_metrics.json", "docs/audits/step2c_metrics.json"):
            with self.assertRaises(ValueError):
                self.export(self.root / path)

    def test_failed_atomic_publication_preserves_completed_snapshot(self):
        self.export()
        before = self.output.read_bytes()
        with mock.patch.object(exporter.os, "replace", side_effect=OSError("failed publication")):
            with self.assertRaises(OSError):
                self.export()
        self.assertEqual(before, self.output.read_bytes())
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])

    def test_import_and_help_never_import_ml_dependencies(self):
        code = """import builtins,importlib.util,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','numpy','PIL','countermine','diffusers','transformers','salad'}:
        raise AssertionError('ML dependency imported: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
spec=importlib.util.spec_from_file_location('isolated',sys.argv[1])
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main(['--help'])
"""
        result = subprocess.run([sys.executable, "-c", code, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--step2c-snapshot", result.stdout)


if __name__ == "__main__":
    unittest.main()

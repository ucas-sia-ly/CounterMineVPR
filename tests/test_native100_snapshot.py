"""Portable exporter tests with 100 fake scalars; no images, weights, or CUDA."""

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools/22_export_native100_snapshot.py"
SPEC = importlib.util.spec_from_file_location("native100_snapshot", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constant_summary(value):
    return {**{name: value for name in exporter.QUANTILES},
            "valid_count": 100, "missing_count": 0, "total_count": 100}


class Native100SnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest_path = self.root / "native100_manifest.csv"
        self.generation_path = self.root / "native100_generation_summary.json"
        self.fidelity_path = self.root / "native100_fidelity.csv"
        self.summary_path = self.root / "native100_fidelity_summary.json"
        self.snapshot_paths = []
        for name in ("step2c_metrics.json", "step2d0_metrics.json", "step2d1_native_fov_metrics.json"):
            path = self.root / name
            path.write_bytes((REPO_ROOT / "docs/audits" / name).read_bytes())
            self.snapshot_paths.append(path)
        self.output = self.root / "docs/audits/step2d2_native100_metrics.json"
        self.manifest, self.records = [], []
        for index in range(100):
            source = {"audit_index": index, "row_index": 1000 + index, "image_id": f"Images/fake/{index}.jpg",
                      "group": "random" if index < 50 else "hard_candidate", "place_uid": f"fake:{index}",
                      "city_id": "fake", "selection_rank": str(index)}
            geometry = {"policy": "native_full_fov", "canonical_width": 640, "canonical_height": 480,
                        "scale_x": 1.0, "scale_y": 1.0}
            self.manifest.append({**source, **geometry, "original_width": 640, "original_height": 480,
                                  "retained_area_fraction": 1.0, "retained_long_axis_fraction": 1.0,
                                  "original_path": f"/machine/local/{index}.jpg", "source_path": f"/machine/local/{index}.png",
                                  "source_512_path": f"cache/generator_audit/source_512/{index}.png",
                                  "source_sha256": "b" * 64, "crop_left": 80, "relative_path": source["image_id"]})
            self.records.append({**source, **geometry, "mode": "full_scene", "num_keypoints_source": 400,
                "num_keypoints_relit": 400, "num_matches": 200, "match_ratio_min": .5,
                "repeatability_2px": .25, "repeatability_4px": .4, "repeatability_8px": .45,
                "repeatability_16px": .5, "precision_2px": .5, "precision_4px": .8,
                "precision_8px": .9, "precision_16px": 1.0, "displacement_mean": 1.5,
                "displacement_median": 1.0, "displacement_q75": 1.8, "displacement_q90": 2.0,
                "displacement_q95": 3.0, "displacement_max": 8.0, "grid_coverage_4px": .65,
                "grid_coverage_8px": .75, "rgb_mae_normalized": .1, "luma_source_mean": 100.0,
                "luma_relit_mean": 105.0, "luma_mean_delta": 5.0, "luma_mae_normalized": .09,
                "pass_R8": True, "pass_coverage8": True, "pass_D95": True, "inherited_pilot_gate_pass": True})
        self.write_inputs()

    def write_inputs(self):
        write_csv(self.manifest_path, self.manifest)
        write_csv(self.fidelity_path, self.records)
        frozen = json.loads(self.snapshot_paths[2].read_text())["provenance"]
        self.hashes = {"manifest_sha256": digest(self.manifest_path),
                       **{f"step2{name}_snapshot_sha256": digest(path)
                          for name, path in zip(("c", "d0", "d1"), self.snapshot_paths)}}
        runs = [{**{name: row[name] for name in ("audit_index", "row_index", "image_id", "policy", "mode",
                                                 "canonical_width", "canonical_height")},
                 "seed": 12345, "elapsed_seconds": .25, "peak_cuda_memory_allocated_bytes": 1000,
                 "peak_cuda_memory_reserved_bytes": 2000, "source_sha256": "b" * 64,
                 "output_sha256": "c" * 64} for row in self.records]
        config = {**frozen["iclight_config"], "device": "cuda", "checkpoint_path": "/private/models/fc.safetensors",
                  "cache_dir": "/private/models", "rmbg_model": "unused"}
        self.generation = {**self.hashes, "number_of_sources": 100, "count_outputs": 100,
            "config": {"seed": 12345, "mode": "full_scene", "policy": "native_full_fov", "iclight_config": config},
            "source_population_provenance": {"number_of_sources": 100, "audit_manifest_sha256": "d" * 64,
                                             "audit_summary_sha256": "e" * 64, "seed": 42,
                                             "count_random": 50, "count_hard_candidate": 50},
            "rmbg_used": False, "two_stage_inference": True,
            "model_load_elapsed_seconds": 1.0, "generation_elapsed_seconds": 25.0, "elapsed_seconds": 27.0,
            "generation_runtime": {"elapsed_seconds": {name: .25 for name in ("median", "q05", "q25", "q75", "q95")},
                                   "peak_cuda_memory_allocated_bytes": 1000, "peak_cuda_memory_reserved_bytes": 2000,
                                   "timing_scope": "per-image inference after explicit model preload; model loading excluded"},
            "runs": runs}
        self.generation_path.write_text(json.dumps(self.generation))
        aliked, lightglue = frozen["aliked_config"], frozen["lightglue_config"]
        extractor_fields = {name: aliked[name] for name in ("max_num_keypoints", "detection_threshold")}
        matcher_fields = {name: lightglue[name] for name in ("features", "depth_confidence", "width_confidence", "filter_threshold", "mp")}
        self.saved = {**self.hashes, "number_of_sources": 100, "count_pairs": 100,
            "generation_summary_sha256": digest(self.generation_path),
            "source_population_provenance": self.generation["source_population_provenance"],
            "config": {"seed": 42, "mode": "full_scene", "policy": "native_full_fov", "grid_shape": [8, 8],
                       "extract_resize": None, "registration": None,
                       "luminance_definition": exporter.LUMA_DEFINITION},
            "matcher": {"extractor_settings": extractor_fields, "matcher_settings": matcher_fields,
                        "models": {"extractor_config": {key: value for key, value in aliked.items() if key not in extractor_fields},
                                   "matcher_config": {key: value for key, value in lightglue.items()
                                                      if key not in matcher_fields and key not in ("compiled", "weights_version")},
                                   "lightglue_weights_version": lightglue["weights_version"],
                                   "checkpoints": {"irrelevant_path": "/private/models"}}, "compiled": False},
            "fidelity_summary": {alias: constant_summary(self.records[0][name]) for alias, name in exporter.SUMMARY_METRICS.items()},
            "intervention_strength": {name: constant_summary(self.records[0][name]) for name in exporter.PIXEL_METRICS},
            "gate": {"name": "inherited_pilot_gate_pass", "thresholds": dict(exporter.FROZEN_GATE),
                     "pass_count": 100, "fail_count": 0, "acceptance_fraction": 1.0,
                     "wilson_95": {"lower": .9630065017930143, "upper": 1.0, "confidence_level": .95,
                                   "z": 1.959963984540054}},
            "bottom_tail": {name: [dict(row) for row in self.records[:20]] for name in
                            ("lowest_R8", "highest_D95", "lowest_coverage8", "deduplicated_manual_audit_set")}}
        self.saved["bottom_tail"]["gate_failures"] = []
        self.summary_path.write_text(json.dumps(self.saved))

    def export(self, output=None):
        return exporter.export_snapshot(self.manifest_path, self.generation_path, self.fidelity_path,
            self.summary_path, *self.snapshot_paths, output or self.output, git_commit="a" * 40, repo_root=self.root)

    def edit_summary(self, edit):
        data = json.loads(self.summary_path.read_text())
        edit(data)
        self.summary_path.write_text(json.dumps(data))

    def reject(self, pattern=".+"):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text("previous completed audit\n")
        with self.assertRaisesRegex((ValueError, KeyError), pattern):
            self.export()
        self.assertEqual(self.output.read_text(), "previous completed audit\n")

    def test_exact_100_records_quantiles_frozen_thresholds_and_portability(self):
        result = self.export()
        self.assertEqual(len(result["per_source_compact"]), 100)
        self.assertEqual(result["gate"]["thresholds"], exporter.FROZEN_GATE)
        self.assertEqual(result["fidelity_summary"], self.saved["fidelity_summary"])
        self.assertEqual(result["intervention_strength"], self.saved["intervention_strength"])
        self.assertAlmostEqual(result["gate"]["wilson_95"]["lower"], .9630065017930143)
        self.assertEqual(result["generation_runtime"]["model_load_elapsed_seconds"], 1.0)
        self.assertEqual(result["generation_runtime"]["total_elapsed_seconds"], 27.0)
        self.assertEqual(result["generation_runtime"]["elapsed_seconds"],
                         {name: .25 for name in ("median", "q05", "q25", "q75", "q95")})
        self.assertEqual(result["intervention_strength_definition"], exporter.LUMA_DEFINITION)
        self.assertEqual(result["per_source_compact"][0]["selection_rank"], "0")
        self.assertEqual(result["provenance"]["frozen_step2d1_snapshot_sha256"], digest(self.snapshot_paths[2]))
        self.assertEqual(result["provenance"]["native_geometry"]["canonical_height"], 480)
        serialized = self.output.read_text()
        for excluded in ("NaN", "Infinity", "/private/models", "/machine/local", "source_512", "b" * 64, "c" * 64):
            self.assertNotIn(excluded, serialized)
        json.dumps(result, allow_nan=False)
        before = self.output.read_bytes()
        self.export()
        self.assertEqual(before, self.output.read_bytes())

    def test_rejects_count_identity_and_geometry_changes(self):
        write_csv(self.fidelity_path, self.records[:-1])
        self.reject("100 distinct")
        self.write_inputs()
        self.manifest[0]["row_index"] = self.manifest[1]["row_index"]
        self.write_inputs()
        self.reject("100 distinct")
        self.manifest[0]["row_index"] = 1000
        self.manifest[0]["scale_x"] = .8
        self.write_inputs()
        self.reject("1.0")
        self.manifest[0]["scale_x"] = 1.0
        self.manifest[0]["canonical_height"] = 512
        self.write_inputs()
        self.reject("640x480")

    def test_rejects_nonfinite_csv_and_json_even_unexported_metadata(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            self.write_inputs()
            rows = copy.deepcopy(self.records)
            rows[0]["rgb_mae_normalized"] = value
            write_csv(self.fidelity_path, rows)
            self.reject("finite")
            self.write_inputs()
            self.edit_summary(lambda data: data.update(unexported_value=value))
            self.reject("nonfinite")

    def test_rejects_absolute_identifiers_and_selection_provenance(self):
        for value in ("/machine/private.jpg", "C:\\private\\image.jpg", "\\\\host\\share\\image.jpg", "file:///private/image.jpg"):
            self.write_inputs()
            self.manifest[0]["selection_rank"] = value
            self.records[0]["selection_rank"] = value
            self.write_inputs()
            self.reject("absolute filesystem path")
        self.manifest[0]["selection_rank"] = "0"
        self.records[0]["selection_rank"] = "0"

    def test_rejects_tuned_gate_and_inconsistent_components(self):
        self.edit_summary(lambda data: data["gate"]["thresholds"].update(displacement_q95_max=6.0))
        self.reject("frozen")
        self.write_inputs()
        rows = copy.deepcopy(self.records)
        rows[0]["pass_R8"] = False
        write_csv(self.fidelity_path, rows)
        self.reject("component flags")

    def test_rejects_inconsistent_distribution_and_tail_order(self):
        self.edit_summary(lambda data: data["fidelity_summary"]["R8"].update(q01=.5))
        self.reject("inconsistent")
        self.write_inputs()
        self.edit_summary(lambda data: data["bottom_tail"]["lowest_R8"].reverse())
        self.reject("deterministic order")

    def test_rejects_changed_frozen_config_and_artifact_hashes(self):
        for edit in (lambda data: data.update(step2d1_snapshot_sha256="0" * 64),
                     lambda data: data["config"].update(registration="homography"),
                     lambda data: data["matcher"]["extractor_settings"].update(max_num_keypoints=1024)):
            self.write_inputs()
            self.edit_summary(edit)
            self.reject()
        self.write_inputs()
        self.generation["config"]["iclight_config"]["highres_denoise"] = .6
        self.generation_path.write_text(json.dumps(self.generation))
        self.edit_summary(lambda data: data.update(generation_summary_sha256=digest(self.generation_path)))
        self.reject("IC-Light configuration")

    def test_cpu_metadata_uses_null_cuda_peaks_without_loading_cuda(self):
        self.generation["config"]["iclight_config"]["device"] = "cpu"
        for row in self.generation["runs"]:
            row["peak_cuda_memory_allocated_bytes"] = None
            row["peak_cuda_memory_reserved_bytes"] = None
        self.generation["generation_runtime"]["peak_cuda_memory_allocated_bytes"] = None
        self.generation["generation_runtime"]["peak_cuda_memory_reserved_bytes"] = None
        self.generation_path.write_text(json.dumps(self.generation))
        self.edit_summary(lambda data: data.update(generation_summary_sha256=digest(self.generation_path)))
        result = self.export()
        self.assertIsNone(result["generation_runtime"]["peak_cuda_memory_allocated_bytes"])
        self.assertIsNone(result["generation_runtime"]["peak_cuda_memory_reserved_bytes"])

    def test_rejects_changed_luminance_or_source_selection_provenance(self):
        self.edit_summary(lambda data: data["config"].update(luminance_definition="BT.601"))
        self.reject("luminance definition")
        self.write_inputs()
        rows = copy.deepcopy(self.records)
        rows[0]["group"] = "hard_candidate"
        write_csv(self.fidelity_path, rows)
        self.reject("selection provenance")

    def test_zero_match_nulls_are_retained_and_ranked_as_structural_failures(self):
        record = copy.deepcopy(self.records[0])
        record.update(num_matches=0, match_ratio_min=0.0, grid_coverage_4px=0.0, grid_coverage_8px=0.0,
                      pass_R8=False, pass_coverage8=False, pass_D95=False, inherited_pilot_gate_pass=False)
        for epsilon in (2, 4, 8, 16):
            record[f"repeatability_{epsilon}px"] = 0.0
            record[f"precision_{epsilon}px"] = 0.0
        for name in exporter.DISPLACEMENTS:
            record[name] = None
        tails = exporter.bottom_tail_tables([*self.records[1:], record])
        self.assertEqual(tails["highest_D95"][0]["audit_index"], 0)
        self.assertIsNone(tails["highest_D95"][0]["displacement_q95"])
        self.assertEqual(tails["gate_failures"][0]["audit_index"], 0)
        self.assertEqual(tails["deduplicated_manual_audit_set"][0]["audit_index"], 0)
        write_csv(self.fidelity_path, [record, *self.records[1:]])
        parsed = exporter.load_records(self.manifest_path, self.fidelity_path)
        self.assertEqual(len(parsed), 100)
        self.assertIsNone(parsed[(0, 1000)]["displacement_q95"])
        self.assertFalse(parsed[(0, 1000)]["inherited_pilot_gate_pass"])

    def test_protected_outputs_and_failed_atomic_write_preserve_evidence(self):
        for path in (self.summary_path, self.root / "third_party/result.json",
                     self.root / "docs/audits/step2c_metrics.json", self.root / "cache/native_fov_audit/new.json"):
            with self.assertRaises(ValueError):
                self.export(path)
        self.export()
        before = self.output.read_bytes()
        with mock.patch.object(exporter.os, "replace", side_effect=OSError("publication failed")):
            with self.assertRaises(OSError):
                self.export()
        self.assertEqual(before, self.output.read_bytes())
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])

    def test_import_and_help_require_only_standard_library(self):
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
        self.assertIn("--step2d1-snapshot", result.stdout)


if __name__ == "__main__":
    unittest.main()

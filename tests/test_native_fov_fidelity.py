"""CPU-only native fidelity and frozen comparisons with fake local features."""

import csv
from contextlib import redirect_stdout
from dataclasses import asdict
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
import weakref

import numpy as np
from PIL import Image

from countermine.probe.geometry_audit import reference, sha256, write_csv
from countermine.probe.iclight_adapter import ICLightConfig
from countermine.probe.local_fidelity import LocalFidelityConfig, MatchResult
from countermine.probe.native_fov_audit import MANIFEST_COLUMNS
from countermine.probe.native_fov_fidelity import (
    COMPARISON_METRICS, FROZEN_METRICS, HISTORICAL_POLICIES, NATIVE_METRICS,
    compute_native_metrics, load_historical_metrics, paired_native_comparisons,
    plot_native_fidelity, summarize_native, summarize_values,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("_test_native_measurement", REPO_ROOT / "tools/17_measure_native_fov_fidelity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _native_record(index, *, displacement=3.0, matches=64):
    return {
        "audit_index": index, "row_index": 100 + index * 7, "image_id": f"Images/source_{index}.jpg",
        "num_keypoints_source": 64, "num_keypoints_relit": 64, "num_matches": matches,
        "match_ratio_min": matches / 64, "repeatability_2px": 0.0,
        "repeatability_4px": matches / 64, "repeatability_8px": matches / 64,
        "repeatability_16px": matches / 64, "displacement_median": displacement if matches else None,
        "displacement_q95": displacement if matches else None,
        "grid_coverage_4px": matches / 64, "grid_coverage_8px": matches / 64,
    }


class NativeFidelityArithmeticTest(unittest.TestCase):
    def test_native_grid_uses_80_by_60_cells_and_all_64_are_covered(self):
        points = np.array([[(x + .5) * 80, (y + .5) * 60] for y in range(8) for x in range(8)])
        result = compute_native_metrics(MatchResult(64, 64, points, points + [3, 0], np.ones(64), 640, 480))
        self.assertEqual(tuple(result), NATIVE_METRICS)
        self.assertEqual(result["repeatability_2px"], 0.0)
        for epsilon in (4, 8, 16):
            self.assertEqual(result[f"repeatability_{epsilon}px"], 1.0)
        self.assertEqual(result["displacement_median"], 3.0)
        self.assertEqual(result["grid_coverage_4px"], 1.0)
        self.assertEqual(result["grid_coverage_8px"], 1.0)

    def test_native_grid_boundaries_and_coordinate_bounds(self):
        points = np.array([[79.9, 59.9], [80, 60], [80.1, 60.1], [639.9, 479.9]])
        result = compute_native_metrics(MatchResult(4, 4, points, points, np.ones(4), 640, 480))
        self.assertEqual(result["grid_coverage_8px"], 3 / 64)
        for points in (np.array([[640, 10]]), np.array([[10, 480]])):
            with self.subTest(points=points), self.assertRaises(ValueError):
                compute_native_metrics(MatchResult(1, 1, points, points, np.ones(1), 640, 480))

    def test_native_empty_matches_are_zero_ratios_and_undefined_displacements(self):
        result = compute_native_metrics(MatchResult(0, 64, np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 640, 480))
        for key, value in result.items():
            if key.startswith("displacement_"):
                self.assertIsNone(value)
            elif key.startswith(("repeatability_", "grid_coverage_", "match_ratio_")):
                self.assertEqual(value, 0.0)
        json.dumps(result, allow_nan=False)

    def test_native_rejects_matcher_geometry_change(self):
        for dimensions in ((512, 512), (640, 512), (512, 384)):
            with self.subTest(dimensions=dimensions), self.assertRaisesRegex(ValueError, "640x480"):
                compute_native_metrics(MatchResult(0, 0, np.empty((0, 2)), np.empty((0, 2)), np.empty(0), *dimensions))

    def test_paired_deltas_signs_and_undefined_values(self):
        records = [_native_record(index) for index in range(10)]
        historical = {}
        for record in records:
            identity = record["audit_index"], record["row_index"]
            historical[identity] = {"image_id": record["image_id"], "policies": {}}
            for policy in HISTORICAL_POLICIES:
                frozen = {name: record[name] for name in NATIVE_METRICS}
                frozen["repeatability_4px"] = .5
                frozen["repeatability_8px"] = .75
                frozen["displacement_q95"] = 5.0
                frozen["grid_coverage_8px"] = .75
                historical[identity]["policies"][policy] = frozen
        paired = paired_native_comparisons(list(reversed(records)), historical)
        self.assertEqual(paired[0]["audit_index"], 0)
        for policy in HISTORICAL_POLICIES:
            self.assertEqual(paired[0][f"delta_R4_vs_{policy}"], .5)
            self.assertEqual(paired[0][f"delta_R8_vs_{policy}"], .25)
            self.assertEqual(paired[0][f"delta_D95_vs_{policy}"], -2.0)
        self.assertEqual(paired[0]["delta_coverage8_vs_square_crop_512"], .25)
        summary = summarize_native(records, paired)
        self.assertEqual(summary["paired_comparisons"]["square_crop_512"]["delta_D95"]["num_negative"], 10)
        records[0] = _native_record(0, matches=0)
        paired = paired_native_comparisons(records, historical)
        self.assertIsNone(paired[0]["delta_D95_vs_square_crop_512"])
        summary = summarize_native(records, paired)
        data = summary["paired_comparisons"]["full_fov_512"]["delta_D95"]
        self.assertEqual((data["valid_count"], data["missing_count"], data["num_negative"]), (9, 1, 9))
        json.dumps(summary, allow_nan=False)

    def test_summary_quantiles_and_sign_counts_exclude_only_undefined(self):
        summary = summarize_values([-4, -2, 0, 2, 4, None], signs=True)
        self.assertEqual(summary, {"median": 0.0, "q25": -2.0, "q75": 2.0,
                                   "min": -4, "max": 4, "valid_count": 5, "missing_count": 1,
                                   "num_positive": 2, "num_zero": 1, "num_negative": 2})
        empty = summarize_values([None] * 10, signs=True)
        self.assertIsNone(empty["median"])
        self.assertEqual(empty["num_positive"], 0)
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize_values([value])

    def test_plot_uses_all_three_policies_in_original_image_pixel_units(self):
        from matplotlib.axes import Axes

        records = [_native_record(index) for index in range(10)]
        historical = {}
        for record in records:
            identity = record["audit_index"], record["row_index"]
            historical[identity] = {"image_id": record["image_id"], "policies": {}}
            for policy in HISTORICAL_POLICIES:
                historical[identity]["policies"][policy] = dict(record, repeatability_4px=.5,
                                                               repeatability_8px=.75, displacement_q95=5.0)
        values, labels = [], []
        original_plot, original_title = Axes.plot, Axes.set_title
        def capture_plot(axis, x, y, *args, **kwargs):
            values.append(list(y))
            return original_plot(axis, x, y, *args, **kwargs)
        def capture_title(axis, label, *args, **kwargs):
            labels.append(label)
            return original_title(axis, label, *args, **kwargs)
        with TemporaryDirectory() as temporary, mock.patch.object(Axes, "plot", capture_plot), \
                mock.patch.object(Axes, "set_title", capture_title):
            plot_native_fidelity(records, historical, Path(temporary) / "native.png")
        self.assertEqual(values, [[.5] * 10, [.5] * 10, [1.] * 10,
                                  [.75] * 10, [.75] * 10, [1.] * 10,
                                  [5.] * 10, [5.] * 10, [3.] * 10])
        self.assertTrue(all("original-image pixels" in label for label in labels))


class _Feature:
    def __init__(self, path):
        self.path = Path(path)


class _FakeMatcher:
    def __init__(self, config, *, fail_at=None, empty_at=None, threshold=.2):
        self.config = config
        self.fail_at, self.empty_at, self.threshold = fail_at, empty_at, threshold
        self.extractions, self.matches = [], []
        self.live, self.peak_live = 0, 0

    def _release(self):
        self.live -= 1

    def extract(self, path):
        with Image.open(path) as image:
            assert image.format == "PNG" and image.mode == "RGB" and image.size == (640, 480)
        feature = _Feature(path)
        self.live += 1
        self.peak_live = max(self.peak_live, self.live)
        weakref.finalize(feature, self._release)
        self.extractions.append(Path(path))
        return feature

    def match(self, source, relit):
        assert source.path != relit.path
        self.matches.append((source.path, relit.path))
        if len(self.matches) == self.fail_at:
            raise RuntimeError("late native fake failure")
        if len(self.matches) == self.empty_at:
            return MatchResult(64, 64, np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 640, 480)
        points = np.array([[(x + .5) * 80, (y + .5) * 60] for y in range(8) for x in range(8)])
        return MatchResult(64, 64, points, points + [3, 0], np.ones(64), 640, 480)

    def runtime_metadata(self):
        extractor = {"max_num_keypoints": 2048, "detection_threshold": self.threshold,
                     "model_name": "aliked-n16", "nms_radius": 2}
        matcher = {"features": "aliked", "depth_confidence": -1, "width_confidence": -1,
                   "filter_threshold": .1, "mp": False}
        return {"config": asdict(self.config), "extractor_settings": extractor,
                "matcher_settings": matcher, "models": {"extractor_config": extractor, "matcher_config": matcher},
                "compiled": False, "extract_resize": None, "registration": None}


class NativeFidelityPipelineTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.output = self.root / "measurement"
        self.plots = self.root / "plots"
        self.manifest = self.inputs / "native_manifest.csv"
        self.generation_csv = self.inputs / "native_generation.csv"
        self.generation_summary = self.inputs / "native_generation_summary.json"
        self.snapshot = self.inputs / "historical.json"
        self.step2c_snapshot = self.inputs / "step2c.json"
        self.historical_csv = self.inputs / "historical_generation.csv"
        self.tool = _tool()
        metadata = _FakeMatcher(LocalFidelityConfig(device="cpu")).runtime_metadata()
        self.provenance = {"fidelity_seed": 42, "aliked_config": metadata["extractor_settings"],
                           "lightglue_config": metadata["matcher_settings"], "iclight_config": asdict(ICLightConfig())}
        self.step2c_snapshot.write_text(json.dumps({"provenance": self.provenance}), encoding="utf-8")
        frozen = []
        for index in range(10):
            record = _native_record(index, displacement=5.0, matches=48)
            policies = {policy: {old: record[name] for name, old in FROZEN_METRICS.items()}
                        for policy in HISTORICAL_POLICIES}
            frozen.append({"audit_index": record["audit_index"], "row_index": record["row_index"],
                           "image_id": record["image_id"], "policies": policies})
        historic_provenance = dict(self.provenance)
        historic_provenance["frozen_step2c_snapshot_sha256"] = sha256(self.step2c_snapshot)
        self.snapshot.write_text(json.dumps({"provenance": historic_provenance, "per_source": frozen}), encoding="utf-8")
        self.manifest_rows, self.generation_rows, self.historical_rows = [], [], []
        for index in range(10):
            identity = index, 100 + index * 7
            original = self.inputs / f"original_{index}.jpg"
            source = self.inputs / f"source_{index}.png"
            relit = self.inputs / f"relit_{index}.png"
            Image.new("RGB", (640, 480), (index * 10, 100, 160)).save(original, "JPEG")
            with Image.open(original) as image:
                image.convert("RGB").save(source, "PNG")
            Image.new("RGB", (640, 480), (100, index * 10, 160)).save(relit, "PNG")
            self.manifest_rows.append({
                "audit_index": identity[0], "row_index": identity[1], "image_id": f"Images/source_{index}.jpg",
                "policy": "native_full_fov", "source_path": reference(source), "original_path": reference(original),
                "original_width": 640, "original_height": 480, "canonical_width": 640, "canonical_height": 480,
                "scale_x": 1.0, "scale_y": 1.0, "retained_area_fraction": 1.0, "retained_long_axis_fraction": 1.0,
                "original_format": "JPEG", "source_format": "PNG", "original_sha256": sha256(original),
                "source_sha256": sha256(source),
            })
            self.generation_rows.append({
                "audit_index": identity[0], "row_index": identity[1], "image_id": f"Images/source_{index}.jpg",
                "policy": "native_full_fov", "mode": "full_scene", "source_path": reference(source),
                "output_path": reference(relit), "canonical_width": 640, "canonical_height": 480,
                "seed": 12345, "source_sha256": sha256(source), "output_sha256": sha256(relit),
            })
            for policy, size in zip(HISTORICAL_POLICIES, ((512, 512), (512, 384))):
                path = self.inputs / f"historical_{policy}_{index}.png"
                Image.new("RGB", size, (index * 10, 50, 90)).save(path, "PNG")
                self.historical_rows.append({"audit_index": identity[0], "row_index": identity[1], "policy": policy,
                                             "mode": "full_scene", "output_path": reference(path),
                                             "canonical_width": size[0], "canonical_height": size[1], "output_sha256": sha256(path)})
        write_csv(self.manifest, MANIFEST_COLUMNS, self.manifest_rows)
        write_csv(self.generation_csv, self.tool.GENERATION_FIELDS, self.generation_rows)
        write_csv(self.historical_csv, self.historical_rows[0].keys(), self.historical_rows)
        config = asdict(ICLightConfig(width=640, height=480, device="cpu"))
        self.generated = {"number_of_sources": 10, "count_outputs": 10,
                          "manifest_sha256": sha256(self.manifest), "historical_snapshot_sha256": sha256(self.snapshot),
                          "step2c_snapshot_sha256": sha256(self.step2c_snapshot),
                          "config": {"seed": 12345, "mode": "full_scene", "policy": "native_full_fov", "iclight_config": config},
                          "runs": self.generation_rows}
        self._save_generation()

    def _save_generation(self):
        self.generation_summary.write_text(json.dumps(self.generated), encoding="utf-8")

    def _run(self, **fake_options):
        matchers = []
        def factory(config):
            matcher = _FakeMatcher(config, **fake_options)
            matchers.append(matcher)
            return matcher
        with redirect_stdout(io.StringIO()):
            summary = self.tool.run_native_fidelity(
                self.manifest, self.generation_csv, self.generation_summary, self.snapshot,
                self.step2c_snapshot, self.historical_csv, self.output, self.plots,
                device="cpu", matcher_factory=factory,
            )
        return summary, matchers

    def test_streams_only_ten_native_pairs_and_preserves_historical_files(self):
        frozen_before = _bytes(self.inputs)
        summary, matchers = self._run()
        matcher = matchers[0]
        self.assertEqual(len(matcher.extractions), 20)
        self.assertEqual(len(matcher.matches), 10)
        self.assertEqual(matcher.peak_live, 2)
        self.assertEqual(matcher.live, 0)
        self.assertEqual(summary["config"]["seed"], 42)
        self.assertIsNone(summary["config"]["registration"])
        self.assertIsNone(summary["config"]["extract_resize"])
        self.assertEqual(summary["native_metrics"]["repeatability_4px"]["median"], 1.0)
        self.assertEqual(summary["paired_comparisons"]["square_crop_512"]["delta_R4"]["median"], .25)
        self.assertEqual(summary["paired_comparisons"]["full_fov_512"]["delta_D95"]["median"], -2.0)
        self.assertEqual(_bytes(self.inputs), frozen_before)
        with (self.output / "native_fidelity.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 10)
        self.assertEqual({row["canonical_height"] for row in rows}, {"480"})
        self.assertNotIn("gate", json.dumps(summary))
        for name in self.tool.PLOT_NAMES:
            self.assertGreater((self.plots / name).stat().st_size, 100)

    def test_empty_native_pair_remains_in_results_and_comparison_sign_counts(self):
        summary, _ = self._run(empty_at=3)
        self.assertEqual(summary["count_pairs"], 10)
        stats = summary["native_metrics"]["displacement_q95"]
        self.assertEqual((stats["valid_count"], stats["missing_count"]), (9, 1))
        stats = summary["paired_comparisons"]["square_crop_512"]["delta_D95"]
        self.assertEqual((stats["valid_count"], stats["missing_count"], stats["num_negative"]), (9, 1, 9))
        with (self.output / "native_fidelity.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[2]["displacement_q95"], "")

    def test_late_matching_or_plot_failure_preserves_published_results(self):
        self._run()
        before = _bytes(self.output), _bytes(self.plots)
        with self.assertRaisesRegex(RuntimeError, "late native"):
            self._run(fail_at=9)
        self.assertEqual((_bytes(self.output), _bytes(self.plots)), before)
        with mock.patch.object(self.tool, "plot_native_fidelity", side_effect=RuntimeError("fake plot failure")):
            with self.assertRaisesRegex(RuntimeError, "fake plot"):
                self._run()
        self.assertEqual((_bytes(self.output), _bytes(self.plots)), before)

    def test_changed_native_png_hash_fails_before_matcher_construction(self):
        Image.new("RGB", (640, 480), "black").save(Path(self.generation_rows[0]["output_path"]))
        with mock.patch.object(self.tool, "LocalFidelityMatcher") as model, self.assertRaisesRegex(ValueError, "changed after generation"):
            self.tool.run_native_fidelity(self.manifest, self.generation_csv, self.generation_summary,
                                          self.snapshot, self.step2c_snapshot, self.historical_csv,
                                          self.output, self.plots, matcher_factory=model)
        model.assert_not_called()

    def test_generation_setting_change_fails_before_matcher_construction(self):
        self.generated["config"]["iclight_config"]["cfg"] = 3.0
        self._save_generation()
        with mock.patch.object(self.tool, "LocalFidelityMatcher") as model, self.assertRaisesRegex(ValueError, "science settings"):
            self.tool.run_native_fidelity(self.manifest, self.generation_csv, self.generation_summary,
                                          self.snapshot, self.step2c_snapshot, self.historical_csv,
                                          self.output, self.plots, matcher_factory=model)
        model.assert_not_called()

    def test_actual_matching_setting_change_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ALIKED configuration differs"):
            self._run(threshold=.1)
        self.assertFalse((self.output / "native_fidelity.csv").exists())

    def test_historical_source_or_policy_missing_is_rejected(self):
        snapshot = json.loads(self.snapshot.read_text())
        for edit in (lambda item: item["per_source"].pop(),
                     lambda item: item["per_source"][0]["policies"].pop("full_fov_512")):
            invalid = json.loads(json.dumps(snapshot))
            edit(invalid)
            self.snapshot.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                load_historical_metrics(self.snapshot)


class NativeCLIImportTest(unittest.TestCase):
    def test_help_is_available_without_torch_or_model_imports(self):
        script = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'diffusers', 'transformers', 'kornia', 'lightglue'}:
        raise AssertionError('unexpected model import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['17_measure_native_fov_fidelity.py', '--help']
runpy.run_path('tools/17_measure_native_fov_fidelity.py', run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--historical-generation-csv", result.stdout)


if __name__ == "__main__":
    unittest.main()

"""CPU-only Step 2D0 pipeline checks with fake inference and local features."""

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

from countermine.probe.geometry_audit import (
    DELTA_METRICS, MANIFEST_COLUMNS, POLICIES, paired_deltas, pilot_gate_pass, sha256, summarize,
)
from countermine.probe.iclight_adapter import ICLightConfig
from countermine.probe.local_fidelity import MatchResult, compute_fidelity_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]


def _tool(filename):
    name = "_test_" + Path(filename).stem
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_rows(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _file_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


class _FakeAdapter:
    def __init__(self, config, calls, *, fail_at=None):
        self.config = config
        self.calls = calls
        self.fail_at = fail_at
        self.last_run_stats = {}

    def relight(self, image, mode):
        if mode != "full_scene":
            raise AssertionError("The geometry experiment must never use RMBG")
        if image.mode != "RGB" or image.size != (self.config.width, self.config.height):
            raise AssertionError("Generation must preserve the configured geometry")
        self.calls.append((self.config, mode, image.size))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("late fake generation failure")
        self.last_run_stats = {
            "elapsed_seconds": 0.001,
            "peak_cuda_memory_allocated_bytes": None,
            "peak_cuda_memory_reserved_bytes": None,
        }
        return image.copy()


class _Feature:
    def __init__(self, path, size):
        self.path = Path(path)
        self.size = size


class _FakeMatcher:
    def __init__(self, config, *, fail_at=None, empty_at=None, detection_threshold=.2):
        self.config = config
        self.fail_at = fail_at
        self.empty_at = empty_at
        self.detection_threshold = detection_threshold
        self.live_features = 0
        self.max_live_features = 0
        self.extractions = []
        self.matches = []

    def _released(self):
        self.live_features -= 1

    def extract(self, path):
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise AssertionError("The matcher receives lossless RGB PNGs")
            feature = _Feature(path, image.size)
        self.live_features += 1
        self.max_live_features = max(self.max_live_features, self.live_features)
        weakref.finalize(feature, self._released)
        self.extractions.append((str(path), feature.size))
        return feature

    def match(self, source, relit):
        if source.size != relit.size or source.path == relit.path:
            raise AssertionError("Only source-relight pairs with identical geometry are allowed")
        width, height = source.size
        self.matches.append((str(source.path), str(relit.path), source.size))
        if len(self.matches) == self.fail_at:
            raise RuntimeError("late fake matching failure")
        if len(self.matches) == self.empty_at:
            return MatchResult(64, 64, np.empty((0, 2)), np.empty((0, 2)), np.empty(0), width, height)
        points = np.array([[(x + .5) * width / 8, (y + .5) * height / 8]
                           for y in range(8) for x in range(8)])
        shift = 2.0 if height == 512 else 1.0
        return MatchResult(64, 64, points, points + [shift, 0], np.ones(64), width, height)

    def runtime_metadata(self):
        extractor = {"max_num_keypoints": self.config.max_keypoints,
                     "detection_threshold": self.detection_threshold}
        return {
            "config": asdict(self.config), "resolved_device": "cpu",
            "extractor_settings": extractor,
            "matcher_settings": {"features": "aliked", "depth_confidence": -1,
                                 "width_confidence": -1, "filter_threshold": .1, "mp": False},
            "models": {"extractor_config": {"model_name": "aliked-n16", "nms_radius": 2,
                                             **extractor}, "matcher_config": {}},
            "compiled": False, "extract_resize": None, "registration": None,
            "grid_shape": [8, 8],
        }


class GeometryAuditPipelineTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.generated = self.root / "generated"
        self.fidelity = self.root / "fidelity"
        self.plots = self.root / "plots"
        self.manifest = self.inputs / "geometry_manifest.csv"
        self.snapshot = self.inputs / "step2c_metrics.json"
        self.rows = []
        for audit_index in range(10):
            row_index = audit_index * 7 + 100
            original = self.inputs / f"original_{audit_index:02d}.png"
            Image.new("RGB", (640, 480), (audit_index, 80, 160)).save(original)
            for policy, size in zip(POLICIES, ((512, 512), (512, 384))):
                source = self.inputs / policy / f"source_{audit_index:02d}.png"
                source.parent.mkdir(exist_ok=True)
                Image.new("RGB", size, (audit_index, 80, 160)).save(source)
                retained = .75 if policy == POLICIES[0] else 1.0
                scale = 512 / 480 if policy == POLICIES[0] else .8
                self.rows.append({
                    "audit_index": audit_index, "row_index": row_index,
                    "image_id": f"source_{row_index}", "policy": policy,
                    "source_path": Path(os.path.relpath(source)).as_posix(),
                    "original_path": Path(os.path.relpath(original)).as_posix(),
                    "original_width": 640, "original_height": 480,
                    "canonical_width": size[0], "canonical_height": size[1],
                    "scale_x": scale, "scale_y": scale,
                    "retained_area_fraction": retained, "retained_long_axis_fraction": retained,
                })
        self._save_manifest()
        config = asdict(ICLightConfig())
        config.pop("iclight_root")
        config.pop("cache_dir")
        self.frozen_snapshot = {
            "per_source_compact": [
                {"audit_index": index, "row_index": index * 7 + 100, "mode": mode}
                for index in range(10) for mode in ("official_rmbg", "full_scene")
            ],
            "provenance": {
                "git_commit": "a" * 40, "iclight_config": config,
                "iclight_seed": 12345, "fidelity_seed": 42,
                "prompt": config["prompt"], "extract_resize": None, "registration": None,
                "aliked_config": {"max_num_keypoints": 2048, "detection_threshold": .2,
                                  "model_name": "aliked-n16", "nms_radius": 2},
                "lightglue_config": {"features": "aliked", "depth_confidence": -1,
                                     "width_confidence": -1, "filter_threshold": .1, "mp": False},
            },
        }
        self.snapshot.write_text(json.dumps(self.frozen_snapshot), encoding="utf-8")
        self.generation_tool = _tool("12_generate_geometry_audit.py")
        self.fidelity_tool = _tool("13_measure_geometry_fidelity.py")

    def _save_manifest(self):
        _write_rows(self.manifest, MANIFEST_COLUMNS, self.rows)

    def _generate(self, *, fail_at=None):
        calls = []
        configs = []

        def factory(config):
            configs.append(config)
            return _FakeAdapter(config, calls, fail_at=fail_at)

        output = io.StringIO()
        with redirect_stdout(output):
            summary = self.generation_tool.run_generation(
                self.manifest, self.snapshot, self.generated, device="cpu", adapter_factory=factory,
            )
        self.last_generation_stdout = output.getvalue()
        return summary, calls, configs

    def _measure(self, *, fail_at=None, empty_at=None):
        matchers = []

        def factory(config):
            matcher = _FakeMatcher(config, fail_at=fail_at, empty_at=empty_at)
            matchers.append(matcher)
            return matcher

        with redirect_stdout(io.StringIO()):
            summary = self.fidelity_tool.run_geometry_fidelity(
                self.manifest, self.generated / "geometry_generation.csv",
                self.generated / "geometry_generation_summary.json", self.snapshot,
                self.fidelity, self.plots, device="cpu", matcher_factory=factory,
            )
        return summary, matchers

    def _assert_measure_rejected_without_models(self):
        factory = mock.Mock(side_effect=AssertionError("Models must remain unloaded"))
        with self.assertRaises(ValueError):
            self.fidelity_tool.run_geometry_fidelity(
                self.manifest, self.generated / "geometry_generation.csv",
                self.generated / "geometry_generation_summary.json", self.snapshot,
                self.fidelity, self.plots, device="cpu", matcher_factory=factory,
            )
        factory.assert_not_called()

    def test_exactly_20_full_scene_generations_keep_frozen_settings_and_geometry(self):
        frozen_bytes = self.snapshot.read_bytes()
        summary, calls, configs = self._generate()
        self.assertEqual(len(calls), 20)
        self.assertEqual(len(configs), 1)
        self.assertEqual({mode for _, mode, _ in calls}, {"full_scene"})
        self.assertEqual(sum(size == (512, 512) for _, _, size in calls), 10)
        self.assertEqual(sum(size == (512, 384) for _, _, size in calls), 10)
        frozen = self.frozen_snapshot["provenance"]["iclight_config"]
        for config, _, size in calls:
            actual = asdict(config)
            self.assertEqual((config.width, config.height), size)
            for key, value in frozen.items():
                if key not in ("width", "height", "device"):
                    self.assertEqual(actual[key], value, key)
            self.assertEqual(actual["device"], "cpu")
        self.assertEqual(self.last_generation_stdout.count("elapsed="), 20)
        self.assertEqual(self.last_generation_stdout.count("peak CUDA allocated="), 20)
        records = _read_rows(self.generated / "geometry_generation.csv")
        self.assertEqual(len(records), 20)
        pngs = list(self.generated.rglob("*.png"))
        self.assertEqual(len(pngs), 20)
        sizes = []
        for path in pngs:
            with Image.open(path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.mode, "RGB")
                sizes.append(image.size)
                image.load()
        self.assertEqual(sizes.count((512, 512)), 10)
        self.assertEqual(sizes.count((512, 384)), 10)
        saved = json.loads((self.generated / "geometry_generation_summary.json").read_text())
        self.assertEqual(saved, summary)
        self.assertEqual(saved["config"]["mode"], "full_scene")
        self.assertEqual(saved["config"]["seed"], 12345)
        self.assertEqual(set(saved["config"]["configs_by_geometry"]), {"512x512", "512x384"})
        for geometry, config in saved["config"]["configs_by_geometry"].items():
            self.assertEqual(geometry, f"{config['width']}x{config['height']}")
            for key, value in frozen.items():
                if key not in ("width", "height", "device"):
                    self.assertEqual(config[key], value, key)
        self.assertEqual(self.snapshot.read_bytes(), frozen_bytes)

    def test_20_fidelity_pairs_and_10_paired_rows_stream_features_and_save_plots(self):
        self._generate()
        summary, matchers = self._measure()
        self.assertEqual(len(matchers), 1)
        matcher = matchers[0]
        self.assertEqual(len(matcher.extractions), 40)
        self.assertEqual(len(matcher.matches), 20)
        self.assertLessEqual(matcher.max_live_features, 2)
        self.assertEqual(matcher.live_features, 0)
        self.assertEqual(matcher.config.seed, 42)
        self.assertEqual(matcher.config.max_keypoints, 2048)
        rows = _read_rows(self.fidelity / "geometry_fidelity.csv")
        paired = _read_rows(self.fidelity / "geometry_paired.csv")
        self.assertEqual(len(rows), 20)
        self.assertEqual(len(paired), 10)
        self.assertEqual({row["policy"] for row in rows}, set(POLICIES))
        for row in rows:
            expected_height = 512 if row["policy"] == POLICIES[0] else 384
            self.assertEqual(int(row["canonical_width"]), 512)
            self.assertEqual(int(row["canonical_height"]), expected_height)
            self.assertEqual(int(row["num_matches"]), 64)
            self.assertEqual(float(row["grid_coverage_8px"]), 1.0)
        for row in paired:
            self.assertEqual(float(row["delta_R4"]), 0.0)
            self.assertEqual(float(row["delta_R8"]), 0.0)
            self.assertEqual(float(row["delta_displacement_q95"]), -1.0)
            self.assertEqual(float(row["delta_grid_coverage_8"]), 0.0)
        self.assertEqual(summary["count_pairs"], 20)
        self.assertEqual(summary["number_of_sources"], 10)
        self.assertEqual(summary["config"]["seed"], 42)
        self.assertIsNone(summary["config"]["extract_resize"])
        self.assertIsNone(summary["config"]["registration"])
        self.assertEqual(summary["config"]["grid_shape"], [8, 8])
        for policy, displacement in zip(POLICIES, (2.0, 1.0)):
            statistics = summary["policies"][policy]["metrics"]["displacement_q95"]
            for quantile in ("median", "q05", "q25", "q75", "q95"):
                self.assertEqual(statistics[quantile], displacement)
            self.assertEqual(statistics["valid_count"], 10)
            self.assertEqual(statistics["missing_count"], 0)
        self.assertEqual(summary["paired_deltas"]["delta_displacement_q95"]["median"], -1.0)
        self.assertEqual(json.loads((self.fidelity / "geometry_summary.json").read_text()), summary)
        for filename, format_name in (("geometry_compare_10.jpg", "JPEG"),
                                      ("geometry_fidelity_compare.png", "PNG")):
            with Image.open(self.plots / filename) as image:
                self.assertEqual(image.format, format_name)
                self.assertGreater(image.width, 100)
                self.assertGreater(image.height, 100)
                image.load()

    def test_pilot_gate_is_reported_without_filtering_failed_pairs(self):
        self._generate()
        summary, _ = self._measure(empty_at=1)
        self.assertEqual(len(_read_rows(self.fidelity / "geometry_fidelity.csv")), 20)
        self.assertEqual(len(_read_rows(self.fidelity / "geometry_paired.csv")), 10)
        self.assertEqual(summary["policies"][POLICIES[0]]["pilot_gate"]["pass_count"], 9)
        self.assertEqual(summary["policies"][POLICIES[1]]["pilot_gate"]["pass_count"], 10)
        self.assertEqual(summary["paired_deltas"]["delta_displacement_q95"]["missing_count"], 1)
        self.assertNotIn("NaN", (self.fidelity / "geometry_summary.json").read_text())
        self.assertNotIn("Infinity", (self.fidelity / "geometry_summary.json").read_text())

    def test_missing_policy_is_rejected_before_adapter_construction(self):
        self.rows.pop()
        self._save_manifest()
        factory = mock.Mock(side_effect=AssertionError("Models must remain unloaded"))
        with self.assertRaises(ValueError):
            self.generation_tool.run_generation(
                self.manifest, self.snapshot, self.generated, device="cpu", adapter_factory=factory,
            )
        factory.assert_not_called()

    def test_invalid_source_png_is_rejected_before_adapter_construction(self):
        invalid = Path(self.rows[-1]["source_path"])
        Image.new("RGB", (512, 384)).save(invalid, format="JPEG")
        factory = mock.Mock(side_effect=AssertionError("Models must remain unloaded"))
        with self.assertRaises(ValueError):
            self.generation_tool.run_generation(
                self.manifest, self.snapshot, self.generated, device="cpu", adapter_factory=factory,
            )
        factory.assert_not_called()

    def test_invalid_relit_png_is_rejected_before_matcher_construction(self):
        self._generate()
        invalid = list(self.generated.rglob("*.png"))[-1]
        with Image.open(invalid) as image:
            size = image.size
        Image.new("RGB", size).save(invalid, format="JPEG")
        factory = mock.Mock(side_effect=AssertionError("Models must remain unloaded"))
        with self.assertRaises(ValueError):
            self.fidelity_tool.run_geometry_fidelity(
                self.manifest, self.generated / "geometry_generation.csv",
                self.generated / "geometry_generation_summary.json", self.snapshot,
                self.fidelity, self.plots, device="cpu", matcher_factory=factory,
            )
        factory.assert_not_called()

    def test_csv_output_aliases_to_inputs_are_rejected_before_matcher_construction(self):
        self._generate()
        csv_path = self.generated / "geometry_generation.csv"
        summary_path = self.generated / "geometry_generation_summary.json"
        original_csv = csv_path.read_bytes()
        original_summary = summary_path.read_bytes()
        for input_row, field in ((0, "source_path"), (2, "source_path"), (0, "original_path")):
            with self.subTest(input_row=input_row, field=field):
                csv_path.write_bytes(original_csv)
                records = _read_rows(csv_path)
                alias = self.rows[input_row][field]
                records[0]["output_path"] = alias
                records[0]["output_sha256"] = sha256(Path(alias))
                _write_rows(csv_path, records[0].keys(), records)
                self._assert_measure_rejected_without_models()
                self.assertEqual(summary_path.read_bytes(), original_summary)

    def test_even_consistent_csv_and_summary_cannot_alias_output_to_source(self):
        self._generate()
        csv_path = self.generated / "geometry_generation.csv"
        summary_path = self.generated / "geometry_generation_summary.json"
        records = _read_rows(csv_path)
        metadata = json.loads(summary_path.read_text())
        for row in (records[0], metadata["runs"][0]):
            row["output_path"] = row["source_path"]
            row["output_sha256"] = row["source_sha256"]
        _write_rows(csv_path, records[0].keys(), records)
        summary_path.write_text(json.dumps(metadata), encoding="utf-8")
        self._assert_measure_rejected_without_models()

    def test_changed_generation_cfg_is_rejected_against_frozen_snapshot(self):
        self._generate()
        summary_path = self.generated / "geometry_generation_summary.json"
        original_summary = summary_path.read_bytes()
        csv_path = self.generated / "geometry_generation.csv"
        original_csv = csv_path.read_bytes()
        for geometry in ("512x512", "512x384"):
            with self.subTest(geometry=geometry):
                metadata = json.loads(original_summary)
                metadata["config"]["configs_by_geometry"][geometry]["cfg"] = 2.5
                summary_path.write_text(json.dumps(metadata), encoding="utf-8")
                self._assert_measure_rejected_without_models()
                self.assertEqual(csv_path.read_bytes(), original_csv)

    def test_missing_or_duplicate_summary_runs_are_rejected_before_model_loading(self):
        self._generate()
        summary_path = self.generated / "geometry_generation_summary.json"
        original_summary = summary_path.read_bytes()
        for mutation in ("missing_key", "missing_row", "duplicate"):
            with self.subTest(mutation=mutation):
                metadata = json.loads(original_summary)
                if mutation == "missing_key":
                    metadata.pop("runs")
                elif mutation == "missing_row":
                    metadata["runs"].pop()
                else:
                    metadata["runs"][-1] = dict(metadata["runs"][0])
                    self.assertEqual(len(metadata["runs"]), 20)
                summary_path.write_text(json.dumps(metadata), encoding="utf-8")
                self._assert_measure_rejected_without_models()

    def test_changed_matcher_settings_preserve_prior_results_and_figures(self):
        self._generate()
        self.fidelity.mkdir()
        self.plots.mkdir()
        (self.fidelity / "geometry_summary.json").write_bytes(b"previous completed result\n")
        (self.plots / "geometry_fidelity_compare.png").write_bytes(b"previous completed figure\n")
        before = _file_bytes(self.fidelity), _file_bytes(self.plots)
        matchers = []

        def factory(config):
            matcher = _FakeMatcher(config, detection_threshold=.3)
            matchers.append(matcher)
            return matcher

        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            self.fidelity_tool.run_geometry_fidelity(
                self.manifest, self.generated / "geometry_generation.csv",
                self.generated / "geometry_generation_summary.json", self.snapshot,
                self.fidelity, self.plots, device="cpu", matcher_factory=factory,
            )
        self.assertEqual(len(matchers), 1)
        self.assertGreaterEqual(len(matchers[0].extractions), 1)
        self.assertLessEqual(matchers[0].max_live_features, 2)
        self.assertEqual((_file_bytes(self.fidelity), _file_bytes(self.plots)), before)

    def test_late_generation_failure_preserves_completed_artifacts(self):
        self._generate()
        before = _file_bytes(self.generated)
        with self.assertRaisesRegex(RuntimeError, "late fake generation failure"):
            self._generate(fail_at=20)
        self.assertEqual(_file_bytes(self.generated), before)

    def test_late_matching_failure_preserves_existing_results_and_figures(self):
        self._generate()
        self.fidelity.mkdir()
        self.plots.mkdir()
        for filename in ("geometry_fidelity.csv", "geometry_paired.csv", "geometry_summary.json"):
            (self.fidelity / filename).write_bytes(b"previous completed result\n")
        for filename in ("geometry_compare_10.jpg", "geometry_fidelity_compare.png"):
            (self.plots / filename).write_bytes(b"previous completed figure\n")
        before = _file_bytes(self.fidelity), _file_bytes(self.plots)
        with self.assertRaisesRegex(RuntimeError, "late fake matching failure"):
            self._measure(fail_at=20)
        self.assertEqual((_file_bytes(self.fidelity), _file_bytes(self.plots)), before)


class GeometryPairedStatisticsTest(unittest.TestCase):
    def _records(self):
        records = []
        for index in range(10):
            for policy in POLICIES:
                metrics = compute_fidelity_metrics([[32, 24]], [[32, 24]], [1.0], 1, 1)
                record = {"audit_index": index, "row_index": index + 100,
                          "policy": policy, **metrics}
                full = policy == POLICIES[1]
                record.update(repeatability_min_4px=.3 + (.2 if full else 0),
                              repeatability_min_8px=.5 + (.1 if full else 0),
                              displacement_q95=5.0 - (1.5 if full else 0),
                              grid_coverage_8px=.625 + (.125 if full else 0))
                records.append(record)
        return records

    def test_paired_deltas_are_full_minus_square_with_known_math(self):
        records = self._records()
        paired = paired_deltas(list(reversed(records)))
        self.assertEqual(len(paired), 10)
        self.assertEqual([row["audit_index"] for row in paired], list(range(10)))
        for row in paired:
            self.assertAlmostEqual(row["delta_R4"], .2)
            self.assertAlmostEqual(row["delta_R8"], .1)
            self.assertEqual(row["delta_displacement_q95"], -1.5)
            self.assertEqual(row["delta_grid_coverage_8"], .125)
        summary = summarize(records, paired)
        for key, expected in zip(DELTA_METRICS, (.2, .1, -1.5, .125)):
            for statistic in ("median", "min", "max"):
                self.assertAlmostEqual(summary["paired_deltas"][key][statistic], expected)

    def test_genuinely_undefined_delta_remains_null_and_is_counted(self):
        records = self._records()
        records[0]["displacement_q95"] = None
        paired = paired_deltas(records)
        self.assertIsNone(paired[0]["delta_displacement_q95"])
        summary = summarize(records, paired)
        delta = summary["paired_deltas"]["delta_displacement_q95"]
        self.assertEqual(delta["valid_count"], 9)
        self.assertEqual(delta["missing_count"], 1)
        self.assertEqual(delta["median"], -1.5)
        json.dumps(summary, allow_nan=False)

    def test_paired_analysis_rejects_missing_duplicate_and_nonfinite_records(self):
        for mutation in ("missing", "duplicate", "nonfinite"):
            with self.subTest(mutation=mutation):
                records = self._records()
                if mutation == "missing":
                    records.pop()
                elif mutation == "duplicate":
                    records.append(dict(records[0]))
                else:
                    records[0]["repeatability_min_8px"] = float("nan")
                with self.assertRaises(ValueError):
                    paired_deltas(records)

    def test_provisional_gate_uses_inclusive_boundaries_and_rejects_undefined(self):
        passing = {"repeatability_min_8px": .4, "grid_coverage_8px": .6, "displacement_q95": 5.0}
        self.assertTrue(pilot_gate_pass(passing))
        for key, bad_value in (("repeatability_min_8px", .3999), ("grid_coverage_8px", .5999),
                               ("displacement_q95", 5.0001), ("displacement_q95", None)):
            self.assertFalse(pilot_gate_pass(dict(passing, **{key: bad_value})))
        with self.assertRaises(ValueError):
            pilot_gate_pass(dict(passing, displacement_q95=float("inf")))


class GeometryCLIHelpTest(unittest.TestCase):
    def test_help_does_not_import_ml_or_cuda_dependencies(self):
        for filename in ("12_generate_geometry_audit.py", "13_measure_geometry_fidelity.py"):
            with self.subTest(filename=filename):
                script = """
import builtins, runpy, sys
original_import = builtins.__import__
forbidden = {'torch', 'diffusers', 'transformers', 'safetensors', 'gradio',
             'salad', '_countermine_vendored_lightglue'}
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in forbidden:
        raise AssertionError('Unexpected model dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
path = sys.argv[1]
sys.argv = [path, '--help']
runpy.run_path(path, run_name='__main__')
"""
                result = subprocess.run(
                    [sys.executable, "-c", script, str(REPO_ROOT / "tools" / filename)],
                    cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()

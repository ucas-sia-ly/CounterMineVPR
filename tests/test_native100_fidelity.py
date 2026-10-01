"""CPU-only native100 audit checks; fake matchers never load model weights."""

from contextlib import redirect_stdout
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
import weakref

import numpy as np
from PIL import Image

from countermine.probe.local_fidelity import LocalFidelityConfig, MatchResult
from countermine.probe.native100_fidelity import (
    FIDELITY_METRICS, FROZEN_GATE, GATE_COLUMNS, PIXEL_METRICS, QUANTILES,
    bottom_tail_tables, compute_intervention_strength, compute_native100_metrics,
    inherited_gate, summarize_distribution, summarize_native100,
    validate_fidelity_record, validate_population_records, wilson_interval,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _points():
    return np.array([[(x + .5) * 80, (y + .5) * 60] for y in range(8) for x in range(8)])


def _record(index, *, displacement=3.0, empty=False):
    points = np.empty((0, 2)) if empty else _points()
    result = {"audit_index": index, "row_index": 1000 + index * 7, "image_id": f"Images/{index}.jpg",
              "group": "random" if index < 50 else "hard_candidate"}
    result.update(compute_native100_metrics(MatchResult(
        64, 64, points, points + [displacement, 0], np.ones(len(points)), 640, 480,
    )))
    result.update(rgb_mae_normalized=.1, luma_source_mean=100., luma_relit_mean=105.,
                  luma_mean_delta=5., luma_mae_normalized=.02)
    result.update(inherited_gate(result))
    return result


def _tool():
    spec = importlib.util.spec_from_file_location("native100_fidelity_tool_tests", REPO_ROOT / "tools/21_measure_native100_fidelity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Native100MetricsTest(unittest.TestCase):
    def test_native_metrics_include_precisions_displacements_and_inclusive_thresholds(self):
        points = np.array([[40., 30.], [120., 90.], [200., 150.], [280., 210.]])
        offsets = np.array([[2., 0], [4., 0], [8., 0], [16., 0]])
        metrics = compute_native100_metrics(MatchResult(8, 10, points, points + offsets, np.ones(4), 640, 480))
        self.assertEqual(tuple(metrics), FIDELITY_METRICS)
        self.assertEqual(metrics["match_ratio_min"], .5)
        for index, epsilon in enumerate((2, 4, 8, 16), 1):
            self.assertEqual(metrics[f"repeatability_{epsilon}px"], index / 8)
            self.assertEqual(metrics[f"precision_{epsilon}px"], index / 4)
        self.assertEqual(metrics["grid_coverage_4px"], 2 / 64)
        self.assertEqual(metrics["grid_coverage_8px"], 3 / 64)
        self.assertEqual(metrics["displacement_mean"], 7.5)
        self.assertEqual(metrics["displacement_median"], 6.)
        self.assertEqual(metrics["displacement_max"], 16.)
        self.assertAlmostEqual(metrics["displacement_q95"], 14.8)

    def test_empty_matches_preserved_as_null_displacements_and_gate_failure(self):
        record = _record(0, empty=True)
        validate_fidelity_record(record)
        for key, value in record.items():
            if key.startswith("displacement_"):
                self.assertIsNone(value)
            elif key.startswith(("repeatability_", "precision_", "grid_coverage_")):
                self.assertEqual(value, 0)
        self.assertEqual([record[name] for name in GATE_COLUMNS], [False] * 4)
        records = [_record(index, empty=index == 0) for index in range(100)]
        summary = summarize_native100(records)
        self.assertEqual(summary["gate"]["pass_count"], 99)
        self.assertEqual(summary["fidelity_summary"]["displacement_q95"]["missing_count"], 1)
        self.assertEqual(summary["bottom_tail"]["highest_D95"][0]["audit_index"], 0)
        json.dumps(summary, allow_nan=False)

    def test_geometry_change_is_rejected_without_any_models(self):
        for dimensions in ((512, 512), (640, 512), (512, 384)):
            with self.subTest(dimensions=dimensions), self.assertRaisesRegex(ValueError, "640x480"):
                compute_native100_metrics(MatchResult(0, 0, np.empty((0, 2)), np.empty((0, 2)), np.empty(0), *dimensions))

    def test_gate_components_frozen_inclusive_and_independent(self):
        baseline = {"repeatability_8px": .40, "grid_coverage_8px": .60, "displacement_q95": 5.0}
        self.assertTrue(all(inherited_gate(baseline).values()))
        for field, value, component in (
            ("repeatability_8px", np.nextafter(.4, 0), "pass_R8"),
            ("grid_coverage_8px", np.nextafter(.6, 0), "pass_coverage8"),
            ("displacement_q95", np.nextafter(5., 6.), "pass_D95"),
            ("displacement_q95", None, "pass_D95"),
        ):
            flags = inherited_gate(dict(baseline, **{field: value}))
            self.assertFalse(flags[component])
            self.assertFalse(flags["inherited_pilot_gate_pass"])
            self.assertEqual(sum(flags[name] for name in GATE_COLUMNS[:3]), 2)
        self.assertEqual(FROZEN_GATE["repeatability_8px_min"], .4)
        self.assertEqual(FROZEN_GATE["grid_coverage_8px_min"], .6)
        self.assertEqual(FROZEN_GATE["displacement_q95_max"], 5.)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                inherited_gate(dict(baseline, displacement_q95=value))

    def test_wilson_interval_known_values_and_boundary_cases(self):
        interval = wilson_interval(50, 100)
        self.assertAlmostEqual(interval["lower"], .4038315303659956)
        self.assertAlmostEqual(interval["upper"], .5961684696340044)
        self.assertEqual(wilson_interval(0, 100)["lower"], 0.)
        self.assertAlmostEqual(wilson_interval(0, 100)["upper"], .03699349820698568)
        self.assertAlmostEqual(wilson_interval(100, 100)["lower"], .9630065017930143)
        self.assertEqual(wilson_interval(100, 100)["upper"], 1.)
        for passed, total in ((0, 0), (-1, 100), (101, 100), (True, 100)):
            with self.subTest(passed=passed, total=total), self.assertRaises(ValueError):
                wilson_interval(passed, total)

    def test_full_quantiles_equal_source_weight_and_nonfinite_rejection(self):
        summary = summarize_distribution(list(range(100)))
        for name, probability in QUANTILES:
            self.assertAlmostEqual(summary[name], 99 * probability)
        self.assertEqual(summary["valid_count"], 100)
        self.assertEqual(summary["missing_count"], 0)
        missing = summarize_distribution([None, 1., 3.])
        self.assertEqual(missing["median"], 2.)
        self.assertEqual(missing["missing_count"], 1)
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                summarize_distribution([value])

    def test_exactly100_distinct_frozen_identities_no_record_filtering(self):
        records = [_record(index) for index in range(100)]
        validate_population_records(records)
        for bad in (records[:-1], [*records, _record(101)]):
            with self.assertRaisesRegex(ValueError, "exactly 100"):
                validate_population_records(bad)
        for name in ("audit_index", "row_index", "image_id"):
            bad = [dict(record) for record in records]
            bad[1][name] = bad[0][name]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "distinct"):
                validate_population_records(bad)
        bad = [dict(record) for record in records]
        bad[0]["pass_D95"] = False
        with self.assertRaisesRegex(ValueError, "frozen thresholds"):
            validate_population_records(bad)

    def test_deterministic_ties_raw_overlap_and_priority_deduplication(self):
        records = [_record(index, displacement=6. if index in (25, 70) else 3.) for index in range(100)]
        reference = bottom_tail_tables(records)
        random.Random(42).shuffle(records)
        self.assertEqual(reference, bottom_tail_tables(records))
        self.assertEqual([row["audit_index"] for row in reference["lowest_R8"]], list(range(20)))
        self.assertEqual([row["audit_index"] for row in reference["highest_D95"][:2]], [25, 70])
        self.assertEqual([row["audit_index"] for row in reference["gate_failures"]], [25, 70])
        union = reference["deduplicated_manual_audit_set"]
        self.assertEqual([row["audit_index"] for row in union[:2]], [25, 70])
        self.assertEqual(len(union), len({row["image_id"] for row in union}))
        self.assertIn(0, [row["audit_index"] for row in reference["lowest_R8"]])
        self.assertIn(0, [row["audit_index"] for row in reference["lowest_coverage8"]])

    def test_pixel_diagnostics_fixed_luma_and_native_image_validation(self):
        with TemporaryDirectory() as temporary:
            source, relit = Path(temporary) / "source.png", Path(temporary) / "relit.png"
            Image.new("RGB", (640, 480), (255, 0, 0)).save(source)
            Image.new("RGB", (640, 480), (0, 255, 0)).save(relit)
            result = compute_intervention_strength(source, relit)
            self.assertEqual(tuple(result), PIXEL_METRICS)
            self.assertAlmostEqual(result["rgb_mae_normalized"], 2 / 3)
            self.assertAlmostEqual(result["luma_source_mean"], .2126 * 255)
            self.assertAlmostEqual(result["luma_relit_mean"], .7152 * 255)
            self.assertAlmostEqual(result["luma_mean_delta"], (.7152 - .2126) * 255)
            self.assertAlmostEqual(result["luma_mae_normalized"], .7152 - .2126)
            no_op = compute_intervention_strength(source, source)
            for name in ("rgb_mae_normalized", "luma_mean_delta", "luma_mae_normalized"):
                self.assertEqual(no_op[name], 0.)
            for size, mode, file_format in (((512, 384), "RGB", "PNG"), ((640, 480), "RGBA", "PNG"),
                                            ((640, 480), "RGB", "JPEG")):
                Image.new(mode, size).save(relit, file_format)
                with self.subTest(size=size, mode=mode, file_format=file_format), self.assertRaisesRegex(ValueError, "640x480 RGB PNG"):
                    compute_intervention_strength(source, relit)
                with self.assertRaises(ValueError):
                    compute_intervention_strength(relit, source)


class _Feature:
    def __init__(self, path):
        self.path = Path(path)


class _FakeMatcher:
    def __init__(self, config, provenance, *, fail_at=None, empty_at=3):
        self.config, self.provenance = config, provenance
        self.fail_at, self.empty_at = fail_at, empty_at
        self.live = self.peak_live = self.extractions = self.matches = 0

    def _release(self):
        self.live -= 1

    def extract(self, path):
        feature = _Feature(path)
        self.live += 1
        self.peak_live = max(self.peak_live, self.live)
        self.extractions += 1
        weakref.finalize(feature, self._release)
        return feature

    def match(self, source, relit):
        self.matches += 1
        if self.matches == self.fail_at:
            raise RuntimeError("late fake matching failure")
        points = np.empty((0, 2)) if self.matches == self.empty_at else _points()
        return MatchResult(64, 64, points, points + [3., 0], np.ones(len(points)), 640, 480)

    def runtime_metadata(self):
        aliked, lightglue = self.provenance["aliked_config"], self.provenance["lightglue_config"]
        return {"config": asdict(self.config), "extractor_settings": aliked,
                "matcher_settings": lightglue, "models": {"extractor_config": aliked, "matcher_config": lightglue,
                                                           "lightglue_weights_version": lightglue["weights_version"]},
                "compiled": False, "extract_resize": None, "registration": None}


class Native100FidelityPipelineTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tool = _tool()
        self.output, self.plots, self.curated = self.root / "reports", self.root / "plots", self.root / "curated"
        self.manifest = self.root / "manifest.csv"
        self.generation_csv = self.root / "generation.csv"
        self.generation_summary = self.root / "generation.json"
        self.population_manifest, self.population_summary = self.root / "population.csv", self.root / "population.json"
        for path in (self.manifest, self.generation_csv, self.generation_summary, self.population_manifest, self.population_summary):
            path.write_text("{}")
        self.snapshots = [REPO_ROOT / "docs/audits" / name for name in
                          ("step2c_metrics.json", "step2d0_metrics.json", "step2d1_native_fov_metrics.json")]
        self.provenance = json.loads(self.snapshots[-1].read_text())["provenance"]
        self.rows, self.outputs = [], {}
        for index in range(100):
            source, relit = self.root / f"source_{index}.png", self.root / f"relit_{index}.png"
            Image.new("RGB", (640, 480), (20, 30, 40)).save(source)
            Image.new("RGB", (640, 480), (30, 40, 50)).save(relit)
            row = {"audit_index": index, "row_index": 1000 + index * 7, "image_id": f"Images/{index}.jpg",
                   "group": "random" if index < 50 else "hard_candidate", "place_uid": f"place:{index}",
                   "city_id": "test", "policy": "native_full_fov", "source_path": source,
                   "original_path": self.root / f"original_{index}.jpg", "canonical_width": 640,
                   "canonical_height": 480, "scale_x": 1., "scale_y": 1.,
                   "retained_area_fraction": 1., "retained_long_axis_fraction": 1.}
            self.rows.append(row)
            self.outputs[index, row["row_index"]] = relit
        self.generation = {"source_population_provenance": {"selection": "frozen; no resampling"},
                           "config": {"policy": "native_full_fov", "mode": "full_scene", "seed": 12345}}

    def _run(self, **matcher_options):
        matchers = []
        def factory(config):
            matcher = _FakeMatcher(config, self.provenance, **matcher_options)
            matchers.append(matcher)
            return matcher
        with mock.patch.object(self.tool, "load_native100_manifest", return_value=self.rows), \
                mock.patch.object(self.tool, "_generation_loader", return_value=lambda *args: (self.outputs, self.generation)), \
                redirect_stdout(io.StringIO()):
            result = self.tool.run_native100_fidelity(
                self.manifest, self.generation_csv, self.generation_summary, *self.snapshots,
                self.output, self.plots, device="cpu", curated_dir=self.curated,
                population_manifest_path=self.population_manifest, population_summary_path=self.population_summary,
                matcher_factory=factory,
            )
        return result, matchers

    def test_all100_streamed_without_filtering_and_curated_figures_preserved(self):
        import csv

        summary, matchers = self._run()
        matcher = matchers[0]
        self.assertEqual((matcher.extractions, matcher.matches, matcher.peak_live, matcher.live), (200, 100, 2, 0))
        self.assertEqual(summary["count_pairs"], 100)
        self.assertEqual(summary["gate"]["pass_count"], 99)
        self.assertEqual(summary["gate"]["fail_count"], 1)
        with (self.output / "native100_fidelity.csv").open(newline="") as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual(len(records), 100)
        self.assertEqual(records[2]["displacement_q95"], "")
        self.assertEqual(records[2]["inherited_pilot_gate_pass"], "False")
        self.assertEqual({row["group"] for row in records}, {"random", "hard_candidate"})
        for runtime, curated in zip(self.tool.PLOT_NAMES, self.tool.CURATED_NAMES):
            self.assertEqual((self.plots / runtime).read_bytes(), (self.curated / curated).read_bytes())
        with Image.open(self.plots / "native100_worst20.jpg") as sheet:
            self.assertEqual(sheet.size, (1348, 3008))
        # A failure on a later source cannot partially replace published evidence.
        before = {path: path.read_bytes() for directory in (self.output, self.plots, self.curated) for path in directory.iterdir()}
        with self.assertRaisesRegex(RuntimeError, "late fake"):
            self._run(fail_at=2)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_bad_native_relit_fails_before_any_matcher_construction(self):
        Image.new("RGB", (512, 384)).save(self.outputs[0, 1000])
        matcher = mock.Mock()
        with mock.patch.object(self.tool, "load_native100_manifest", return_value=self.rows), \
                mock.patch.object(self.tool, "_generation_loader", return_value=lambda *args: (self.outputs, self.generation)), \
                self.assertRaisesRegex(ValueError, "RGB PNG"):
            self.tool.run_native100_fidelity(
                self.manifest, self.generation_csv, self.generation_summary, *self.snapshots,
                self.output, self.plots, curated_dir=self.curated, matcher_factory=matcher,
                population_manifest_path=self.population_manifest, population_summary_path=self.population_summary,
            )
        matcher.assert_not_called()


class Native100CLIImportTest(unittest.TestCase):
    def test_help_without_torch_or_any_inference_library(self):
        script = """
import builtins,runpy,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','torchvision','diffusers','transformers','kornia','lightglue'}:
        raise AssertionError('unexpected model import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=['21_measure_native100_fidelity.py','--help']
runpy.run_path('tools/21_measure_native100_fidelity.py',run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--curated-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()

"""CPU-only integration checks with fake local features and no model weights."""

from contextlib import redirect_stdout
import csv
from dataclasses import asdict
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import weakref

import numpy as np
from PIL import Image

from countermine.probe.local_fidelity import MatchResult


AUDIT_COLUMNS = ("audit_index", "row_index", "image_id", "source_512_path")
SMOKE_COLUMNS = ("audit_index", "row_index", "mode", "source_512_path", "output_path")
RELIGHT_MODES = ("official_rmbg", "full_scene")


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class FeatureBatch:
    def __init__(self, path):
        self.path = path


class FakeMatcher:
    """Track lifetimes without retaining features in the fake's history."""

    def __init__(self, config, *, fail_on_match=None):
        self.config = config
        self.fail_on_match = fail_on_match
        self.extract_calls = []
        self.match_calls = []
        self.references = []
        self.maximum_live_features = 0

    def extract(self, path):
        features = FeatureBatch(Path(path))
        self.references.append(weakref.ref(features))
        live = sum(reference() is not None for reference in self.references)
        self.maximum_live_features = max(live, self.maximum_live_features)
        self.extract_calls.append(features.path)
        return features

    def match(self, source, relit):
        self.match_calls.append((source.path, relit.path, source is relit))
        if len(self.match_calls) == self.fail_on_match:
            raise RuntimeError("synthetic matcher failure")
        points = np.array([[10, 10], [74, 74], [138, 138], [202, 202]], dtype=float)
        shifts = np.zeros(4)
        if source.path != relit.path:
            shifts = np.array([2, 4, 8, 16], dtype=float)
            if relit.path.parent.name == "full_scene":
                shifts += 1
        return MatchResult(
            num_keypoints_source=8, num_keypoints_relit=10,
            points_source=points, points_relit=points + np.column_stack((shifts, np.zeros(4))),
            match_scores=np.array([0.9, 0.8, 0.7, 0.6]),
        )

    def runtime_metadata(self):
        return {"resolved_device": "cpu", "test_adapter": True}


class ICLightFidelityCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = importlib.import_module("tools.09_measure_iclight_fidelity")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audit_dir = self.root / "generator_audit"
        self.audit_dir.mkdir()
        self.manifest = self.audit_dir / "audit_manifest.csv"
        self.smoke_csv = self.audit_dir / "iclight_smoke.csv"
        self.output_dir = self.audit_dir / "fidelity"
        self.plot_dir = self.root / "outputs/step2"
        self.audit_rows, self.smoke_rows = [], []
        for audit_index in range(6):
            row_index = 100 + audit_index * 7
            source_path = self.audit_dir / "source_512" / f"{row_index:08d}.png"
            source_path.parent.mkdir(exist_ok=True)
            with Image.new("RGB", (512, 512), (10 + audit_index, 20, 30)) as image:
                image.save(source_path)
            source_reference = os.path.relpath(source_path, Path.cwd())
            self.audit_rows.append({
                "audit_index": audit_index, "row_index": row_index,
                "image_id": f"image-{row_index}", "source_512_path": source_reference,
            })
            if audit_index not in (1, 4):
                continue
            for mode in RELIGHT_MODES:
                relit_path = self.audit_dir / "relit" / mode / source_path.name
                relit_path.parent.mkdir(parents=True, exist_ok=True)
                with Image.new("RGB", (512, 512), (60, 70, 80)) as image:
                    image.save(relit_path)
                self.smoke_rows.append({
                    "audit_index": audit_index, "row_index": row_index,
                    "mode": mode, "source_512_path": source_reference,
                    "output_path": os.path.relpath(relit_path, Path.cwd()),
                })
        write_csv(self.manifest, AUDIT_COLUMNS, self.audit_rows)
        # CSV order should not affect deterministic source/mode processing order.
        write_csv(self.smoke_csv, SMOKE_COLUMNS, self.smoke_rows[::-1])
        self.matchers = []

    def factory(self, config):
        matcher = FakeMatcher(config)
        self.matchers.append(matcher)
        return matcher

    def run_fidelity(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return self.runner.run_fidelity(
                self.manifest, self.smoke_csv, self.output_dir,
                device="cpu", plot_dir=self.plot_dir,
                matcher_factory=kwargs.pop("matcher_factory", self.factory), **kwargs,
            )

    def artifact_hashes(self):
        return {path: hashlib.sha256(path.read_bytes()).hexdigest()
                for directory in (self.output_dir, self.plot_dir)
                for path in directory.rglob("*") if path.is_file()}

    def test_cli_help_never_imports_vendored_models_or_downloads_weights(self):
        code = """
import importlib
import sys
from unittest import mock
import torch
with mock.patch.object(torch.hub, 'load_state_dict_from_url', side_effect=AssertionError('download')):
    module = importlib.import_module('tools.09_measure_iclight_fidelity')
    assert not any(name == 'lightglue' or name.startswith('lightglue.') for name in sys.modules)
    module.build_parser().parse_args(['--help'])
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--audit-manifest", result.stdout)
        self.assertIn("--max-keypoints", result.stdout)

    def test_runner_uses_only_smoke_sources_and_discards_features_between_sources(self):
        summary = self.run_fidelity()
        self.assertEqual(summary["number_of_sources"], 2)
        self.assertEqual(summary["count_pairs"], 6)
        matcher = self.matchers[0]
        expected_extractions = []
        for audit_index in (1, 4):
            source = Path(self.audit_rows[audit_index]["source_512_path"]).resolve()
            expected_extractions.extend([
                source, source, self.audit_dir / "relit/official_rmbg" / source.name,
                self.audit_dir / "relit/full_scene" / source.name,
            ])
        self.assertEqual(matcher.extract_calls, expected_extractions)
        self.assertEqual(matcher.maximum_live_features, 2)
        self.assertTrue(all(reference() is None for reference in matcher.references))
        self.assertEqual(len(matcher.match_calls), 6)
        self.assertTrue(all(not reused for _, _, reused in matcher.match_calls))
        self.assertEqual(matcher.match_calls[0][0], matcher.match_calls[0][1])
        self.assertEqual(summary["config"]["seed"], 42)
        self.assertIsNone(summary["config"]["extract_resize"])
        self.assertIsNone(summary["config"]["registration"])
        self.assertFalse(summary["config"]["matcher_compiled"])
        for name, value in asdict(matcher.config).items():
            self.assertEqual(summary["config"][name], value)
        with (self.output_dir / "fidelity_10.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, list(self.runner.CSV_COLUMNS))
            records = list(reader)
        self.assertEqual([int(record["audit_index"]) for record in records], [1, 1, 1, 4, 4, 4])
        self.assertEqual([record["mode"] for record in records], list(self.runner.ALL_MODES) * 2)
        self.assertIn("occupied_grid_cells_4px", records[0])
        self.assertIn("precision_matches_16px", records[0])
        self.assertEqual(set(summary["modes"]), set(self.runner.ALL_MODES))
        for mode in self.runner.ALL_MODES:
            self.assertEqual(summary["modes"][mode]["count_pairs"], 2)
            self.assertEqual(set(summary["modes"][mode]["metrics"]), set(self.runner.SUMMARY_METRICS))
        self.assertEqual(len(summary["probe_sha256"]), 6)
        self.assertEqual(json.loads((self.output_dir / "fidelity_summary.json").read_text()), summary)
        for name in self.runner.PLOT_NAMES:
            with Image.open(self.plot_dir / name) as image:
                image.verify()

    def test_join_rejects_missing_duplicate_unknown_or_misaligned_smoke_rows(self):
        invalid_sets = [
            self.smoke_rows[:-1],
            self.smoke_rows + [self.smoke_rows[0]],
            [{**self.smoke_rows[0], "row_index": 99999}, *self.smoke_rows[1:]],
            [{**self.smoke_rows[0], "mode": "identity_control"}, *self.smoke_rows[1:]],
            [{**self.smoke_rows[0], "source_512_path": self.audit_rows[4]["source_512_path"]}, *self.smoke_rows[1:]],
            [{**self.smoke_rows[0], "output_path": self.smoke_rows[1]["output_path"]}, *self.smoke_rows[1:]],
            [{**self.smoke_rows[0], "output_path": self.audit_rows[5]["source_512_path"]}, *self.smoke_rows[1:]],
            [],
        ]
        for rows in invalid_sets:
            with self.subTest(rows=rows):
                write_csv(self.smoke_csv, SMOKE_COLUMNS, rows)
                with self.assertRaises(ValueError):
                    self.run_fidelity()
                self.assertEqual(self.matchers, [])

    def test_duplicate_canonical_paths_with_distinct_identities_are_rejected(self):
        duplicate = [dict(row) for row in self.audit_rows]
        duplicate[5]["source_512_path"] = duplicate[0]["source_512_path"]
        write_csv(self.manifest, AUDIT_COLUMNS, duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate canonical source paths"):
            self.run_fidelity()
        self.assertEqual(self.matchers, [])

    def test_destinations_cannot_replace_directory_containing_an_input(self):
        self.output_dir.mkdir()
        dangerous_destination = self.output_dir / "fidelity_10.csv"
        dangerous_destination.mkdir()
        relocated_manifest = dangerous_destination / "input.csv"
        relocated_manifest.write_bytes(self.manifest.read_bytes())
        with self.assertRaisesRegex(ValueError, "must not overwrite input"):
            self.runner.run_fidelity(
                relocated_manifest, self.smoke_csv, self.output_dir,
                device="cpu", plot_dir=self.plot_dir, matcher_factory=self.factory,
            )
        self.assertTrue(relocated_manifest.is_file())
        self.assertEqual(self.matchers, [])

    def test_outputs_cannot_write_to_read_only_vendored_or_salad_trees(self):
        for name in ("third_party", "salad"):
            with self.subTest(root=name), self.assertRaisesRegex(ValueError, "read-only"):
                self.runner.run_fidelity(
                    self.manifest, self.smoke_csv, self.runner.REPO_ROOT / name / "fidelity",
                    device="cpu", plot_dir=self.plot_dir, matcher_factory=self.factory,
                )
        self.assertEqual(self.matchers, [])

    def test_legacy_and_mixed_png_sources_are_rejected_before_model_loading(self):
        # A JPEG reference outside the measured subset must still be rejected.
        legacy = [dict(row) for row in self.audit_rows]
        legacy[5]["source_512_path"] = str(Path(legacy[5]["source_512_path"]).with_suffix(".jpg"))
        write_csv(self.manifest, AUDIT_COLUMNS, legacy)
        with self.assertRaisesRegex(ValueError, "lossless .png"):
            self.run_fidelity()
        write_csv(self.manifest, AUDIT_COLUMNS, self.audit_rows)
        jpeg = self.audit_dir / "relit/full_scene/legacy.jpg"
        with Image.new("RGB", (512, 512)) as image:
            image.save(jpeg)
        with self.assertRaisesRegex(ValueError, "legacy JPEG probes"):
            self.run_fidelity()
        self.assertEqual(self.matchers, [])

    def test_bad_images_are_rejected_before_model_loading(self):
        output = Path(self.smoke_rows[-1]["output_path"])
        original = output.read_bytes()
        for format_, size, mode in (
            ("JPEG", (512, 512), "RGB"), ("PNG", (513, 512), "RGB"),
            ("PNG", (512, 512), "L"),
        ):
            with self.subTest(format=format_, size=size, mode=mode):
                with Image.new(mode, size) as image:
                    image.save(output, format=format_)
                with self.assertRaises(ValueError):
                    self.run_fidelity()
                output.write_bytes(original)
        self.assertEqual(self.matchers, [])

    def test_summary_quantiles_skip_null_displacements_without_zero_imputation(self):
        records = []
        for value in (0.0, 2.0, 10.0, 20.0, None):
            row = {name: value for name in self.runner.SUMMARY_METRICS}
            row["mode"] = "official_rmbg"
            records.append(row)
        metrics = self.runner.summarize_modes(records)["official_rmbg"]["metrics"]
        for name in self.runner.SUMMARY_METRICS:
            for key, expected in {"median": 6.0, "q05": 0.3, "q25": 1.5,
                                  "q75": 12.5, "q95": 18.5}.items():
                self.assertAlmostEqual(metrics[name][key], expected)
            self.assertEqual(metrics[name]["valid_count"], 4)
            self.assertEqual(metrics[name]["missing_count"], 1)
        empty = self.runner.summarize_modes([])["full_scene"]["metrics"]["displacement_median"]
        self.assertIsNone(empty["median"])
        self.assertEqual(empty["valid_count"], 0)

    def test_empty_matches_publish_blank_csv_statistics_and_null_summary_statistics(self):
        class EmptyMatcher(FakeMatcher):
            def match(self, source, relit):
                return MatchResult(8, 0, np.empty((0, 2)), np.empty((0, 2)), np.empty(0))

        summary = self.run_fidelity(matcher_factory=EmptyMatcher)
        with (self.output_dir / "fidelity_10.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                self.assertEqual(row["displacement_median"], "")
                self.assertEqual(row["match_score_mean"], "")
                self.assertEqual(float(row["repeatability_min_8px"]), 0.0)
                self.assertEqual(float(row["grid_coverage_8px"]), 0.0)
        for mode in self.runner.ALL_MODES:
            metric = summary["modes"][mode]["metrics"]["displacement_median"]
            self.assertIsNone(metric["median"])
            self.assertEqual(metric["valid_count"], 0)
            self.assertEqual(metric["missing_count"], 2)

    def test_worst_selection_caps_five_per_mode_and_resolves_ties_by_identity(self):
        examples = []
        for index in (8, 2, 7, 1, 5, 6, 4, 3, 0):
            self.runner._retain_worst(examples, {"record": {
                "repeatability_min_8px": 0.1 if index != 0 else 0.2,
                "audit_index": index, "row_index": index + 10,
            }})
        self.assertEqual([item["record"]["audit_index"] for item in examples], [1, 2, 3, 4, 5])

    def test_late_inference_or_plot_failure_preserves_all_published_artifacts(self):
        self.run_fidelity()
        original = self.artifact_hashes()
        factory = lambda config: FakeMatcher(config, fail_on_match=4)
        with self.assertRaisesRegex(RuntimeError, "synthetic matcher failure"):
            self.run_fidelity(matcher_factory=factory)
        self.assertEqual(self.artifact_hashes(), original)
        with mock.patch.object(self.runner, "plot_displacement_cdf", side_effect=ValueError("plot failure")):
            with self.assertRaisesRegex(ValueError, "plot failure"):
                self.run_fidelity()
        self.assertEqual(self.artifact_hashes(), original)


if __name__ == "__main__":
    unittest.main()

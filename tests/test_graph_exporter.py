"""Exporter rejects changed visual sources and protected write destinations."""

import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from countermine.mining.structural_analysis import sha256_file


SPEC = importlib.util.spec_from_file_location(
    "step2b_exporter", Path(__file__).resolve().parents[1] / "tools/13_analyze_countermine_graph.py")
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class ExporterProtectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        runtime = self.root / "cache/countermine_rgb/step2a"
        runtime.mkdir(parents=True)
        (runtime / "image_fingerprints.json").write_text("{}")
        manifest = self.root / "manifest.csv"
        manifest.write_text("image_id\noriginal\n")
        snapshot = self.root / "docs/audits/step2a_rgb_structural_metrics.json"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text("{}")
        self.args = argparse.Namespace(
            runtime_dir=self.root / "cache/countermine_rgb/step2b", step2a_dir=runtime,
            manifest=manifest, dataset_root=self.root / "data/gsv-cities",
            output_dir=self.root / "outputs/step2b",
            snapshot=snapshot.with_name("step2b_countermine_graph_metrics.json"),
        )
        self.calibration = {
            "input_files": {"runtime_dir": "cache/countermine_rgb/step2a", "manifest": "manifest.csv",
                            "candidates": "candidates.csv", "candidate_summary": "population.json",
                            "snapshot": "docs/audits/step2a_rgb_structural_metrics.json"},
            "provenance": {"step2a_input_hashes": {"manifest_sha256": sha256_file(manifest)},
                           "step2a_image_fingerprints_sha256": sha256_file(runtime / "image_fingerprints.json")},
        }
        self.addCleanup(patch.stopall)
        patch.object(EXPORTER, "ROOT", self.root).start()

    def validate(self):
        EXPORTER.validate_export_inputs(self.args, self.calibration)

    def test_valid_preflight_is_read_only(self):
        self.validate()
        self.assertFalse(self.args.output_dir.exists())
        self.assertFalse(self.args.runtime_dir.exists())
        self.assertFalse(self.args.snapshot.exists())

    def test_rejects_changed_manifest_and_fingerprints(self):
        self.args.manifest.write_text("changed")
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            self.validate()
        self.args.manifest.write_text("image_id\noriginal\n")
        (self.args.step2a_dir / "image_fingerprints.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            self.validate()

    def test_rejects_alternate_runtime_and_step2a_snapshot(self):
        original = self.args.step2a_dir
        self.args.step2a_dir = self.root / "copy_step2a"
        with self.assertRaisesRegex(ValueError, "frozen Step 2A runtime"):
            self.validate()
        self.args.step2a_dir = original
        self.args.snapshot = self.root / self.calibration["input_files"]["snapshot"]
        with self.assertRaisesRegex(ValueError, "step2b_"):
            self.validate()

    def test_rejects_protected_output_trees_and_symlink_aliases(self):
        original = self.args.output_dir
        for directory in (self.args.step2a_dir, self.args.dataset_root, self.root / "third_party"):
            self.args.output_dir = directory
            with self.assertRaisesRegex(ValueError, "overwrite Step 2A"):
                self.validate()
        self.args.output_dir = original
        self.args.snapshot.symlink_to(self.root / self.calibration["input_files"]["snapshot"])
        with self.assertRaisesRegex(ValueError, "overwrite Step 2A"):
            self.validate()
        self.assertEqual((self.root / self.calibration["input_files"]["snapshot"]).read_text(), "{}")


if __name__ == "__main__":
    unittest.main()

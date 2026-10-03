"""CPU checks for the exact monitoring transition, without relaxing science."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from countermine.training import step3a_config as config

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "countermine/training/step3a_monitoring_compatibility.json"


class SourceCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / MANIFEST).read_text())
        self.reference = deepcopy(self.manifest["prepared_code_hashes"])

    def copy_release(self, root):
        for name in [*self.manifest["compatible_code_hashes"], MANIFEST]:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)

    def test_manifest_binds_complete_reviewed_release_and_only_observational_changes(self):
        current = config.code_hashes(ROOT)
        self.assertEqual(current, self.manifest["compatible_code_hashes"])
        changed = {name for name in self.reference.keys() | current.keys()
                   if self.reference.get(name) != current.get(name)}
        self.assertEqual(changed, {
            "countermine/training/cached_dinov2.py",
            "countermine/training/runtime_preflight.py",
            "countermine/training/step3a_config.py", "countermine/training/step3a_audit.py",
            "countermine/training/step3a_monitoring.py",
            "tools/31_train_step3a.py", "tools/32_evaluate_step3a.py",
        })
        for name in ("countermine/training/countermine_batch_sampler.py",
                     "countermine/training/salad_datamodule.py",
                     "countermine/training/gsv_place_mapping.py",
                     "countermine/training/step3a_runtime.py", "tools/30_prepare_step3a_training.py"):
            self.assertEqual(self.reference[name], current[name])

    def test_approved_transition_preserves_reference_and_records_real_executed_hashes(self):
        before = deepcopy(self.reference)
        report = config.validate_source_hashes(self.reference, ROOT)
        self.assertEqual(self.reference, before)
        self.assertEqual(report["policy"], "pinned_monitoring_only_transition")
        self.assertEqual(report["prepared_code_hashes"], self.reference)
        self.assertEqual(report["executed_code_hashes"], config.code_hashes(ROOT))
        self.assertEqual(report["compatibility_manifest_sha256"], config.sha256_file(ROOT / MANIFEST))
        self.assertNotEqual(report["prepared_code_hashes"], report["executed_code_hashes"])
        report["prepared_code_hashes"].clear()
        self.assertEqual(self.reference, before)

    def test_new_preparation_with_exact_current_sources_requires_no_exception(self):
        current = config.code_hashes(ROOT)
        report = config.validate_source_hashes(current, ROOT)
        self.assertEqual(report["policy"], "exact_source_identity")
        self.assertNotIn("compatibility_manifest", report)

    def test_every_further_change_to_guard_monitoring_or_scientific_source_is_rejected(self):
        for name in self.manifest["compatible_code_hashes"]:
            with self.subTest(source=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.copy_release(root)
                config.validate_source_hashes(self.reference, root)
                with (root / name).open("a") as stream:
                    stream.write("\n# unreviewed source change\n")
                with self.assertRaisesRegex(ValueError, "not the exact approved monitoring release"):
                    config.validate_source_hashes(self.reference, root)

    def test_added_removed_sources_and_unknown_preparation_are_rejected(self):
        for change in ("added", "removed", "unknown", "missing_manifest", "wrong_manifest"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.copy_release(root)
                reference = deepcopy(self.reference)
                if change == "added":
                    (root / "countermine/training/unreviewed.py").write_text("# extra source\n")
                elif change == "removed":
                    (root / "countermine/training/step3a_monitoring.py").unlink()
                elif change == "unknown":
                    reference["tools/31_train_step3a.py"] = "f" * 64
                elif change == "missing_manifest":
                    (root / MANIFEST).unlink()
                else:
                    manifest = deepcopy(self.manifest)
                    manifest["schema_version"] = 99
                    (root / MANIFEST).write_text(json.dumps(manifest))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    config.validate_source_hashes(reference, root)

    def test_archived_partial_source_checks_reject_drift_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.py"
            source.write_text("# archived fixture\n")
            reference = {"fixture.py": config.sha256_file(source)}
            report = config.validate_source_hashes(reference, root, check_inventory=False)
            self.assertEqual(report["executed_code_hashes"], reference)
            source.write_text("# drift\n")
            with self.assertRaises(ValueError):
                config.validate_source_hashes(reference, root, check_inventory=False)
            for name in ("../fixture.py", "/fixture.py", "x\\fixture.py"):
                with self.assertRaisesRegex(ValueError, "safe relative paths"):
                    config.validate_source_hashes({name: "a" * 64}, root, check_inventory=False)

    def test_existing_preparation_is_returned_unchanged_and_frozen_input_guards_remain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_release(root)
            paths = config.resolved_paths(root)
            for path in (paths["snapshot"], paths["graph"], paths["shared"] / "initial_state.pt"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"frozen fixture")
            identity = {"salad_submodule_commit": "a" * 40, "salad_source_hashes": {}}
            provenance = {"seed": 42, "step3a_code_hashes": self.reference, **identity,
                          "step2d_snapshot_sha256": config.sha256_file(paths["snapshot"]),
                          "step2d_place_edges_sha256": config.sha256_file(paths["graph"]),
                          "initial_state_sha256": config.sha256_file(paths["shared"] / "initial_state.pt")}
            preparation = {"complete": True, "provenance": provenance}
            path = paths["shared"] / "preparation_summary.json"
            path.write_text(json.dumps(preparation))
            before = path.read_bytes()
            with patch.object(config, "salad_identity", return_value=identity):
                self.assertEqual(config.check_preparation(paths), preparation)
                for key, target in (("snapshot", paths["snapshot"]), ("graph", paths["graph"]),
                                    ("initialization", paths["shared"] / "initial_state.pt")):
                    with self.subTest(input=key):
                        target.write_bytes(b"changed fixture")
                        with self.assertRaises(ValueError):
                            config.check_preparation(paths)
                        target.write_bytes(b"frozen fixture")
                with patch.object(config, "salad_identity", return_value={**identity, "salad_submodule_commit": "b" * 40}):
                    with self.assertRaisesRegex(ValueError, "SALAD source identity changed"):
                        config.check_preparation(paths)
            self.assertEqual(path.read_bytes(), before)

    def test_evaluation_reaches_completed_run_gate_after_passing_compatible_source_check(self):
        spec = importlib.util.spec_from_file_location("compatible_eval_cli", ROOT / "tools/32_evaluate_step3a.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        preparation = {"dataset_metadata": {}, "provenance": {
            "step3a_code_hashes": self.reference, "runtime": {},
            "salad_submodule_commit": "a" * 40, "salad_source_hashes": {},
        }}
        identity = {name: preparation["provenance"][name] for name in ("salad_submodule_commit", "salad_source_hashes")}
        with patch.object(cli, "run_preflight", return_value={"runtime": {}}), \
                patch.object(cli, "validate_dataset_paths", return_value={}), \
                patch.object(cli, "salad_identity", return_value=identity), \
                patch.object(cli, "read_json", side_effect=[preparation, {"complete": False}]) as read, \
                patch.object(cli, "create_model") as model:
            with self.assertRaisesRegex(ValueError, "Only completed full scientific"):
                cli.main(["--mode", "baseline"])
            self.assertEqual(read.call_count, 2)
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()

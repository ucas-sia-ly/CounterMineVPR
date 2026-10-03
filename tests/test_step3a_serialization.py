"""CPU regressions for the PyTorch 2.1 initialization transaction."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from countermine.training import step3a_config as config


class SharedDirectoryTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.paths = config.resolved_paths(self.root)
        self.shared = self.paths["shared"]
        self.destination = self.shared / "initial_state.pt"
        self.summary = self.shared / "initial_state_summary.json"


class AtomicTorchSaveTests(SharedDirectoryTestCase):
    def test_real_cpu_tensor_round_trip(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable; file-object regressions still run")
        payload = {"x": torch.tensor([1.0, -2.0, 3.5], device="cpu")}
        result = config.atomic_torch_save(payload, self.destination, root=self.root)
        self.assertEqual(result, self.destination)
        self.assertTrue(self.destination.is_file())
        self.assertGreater(self.destination.stat().st_size, 0)
        restored = torch.load(self.destination, map_location="cpu", weights_only=True)
        self.assertTrue(torch.equal(restored["x"], payload["x"]))
        self.assertEqual(list(self.shared.glob("initial_state_tmp_*.pt")), [])

    def test_file_object_visible_pt_temporary_and_flush_before_publish(self):
        events, names = [], []
        original_mkstemp = config.tempfile.mkstemp
        original_fsync, original_replace = config.os.fsync, config.os.replace

        def temporary(*args, **kwargs):
            descriptor, name = original_mkstemp(*args, **kwargs)
            names.append(Path(name))
            return descriptor, name

        def save(payload, target):
            self.assertNotIsInstance(target, (str, Path))
            self.assertTrue(callable(target.write))
            self.assertFalse(target.closed)
            self.assertEqual(target.mode, "wb")
            events.append("save")
            target.write(b"frozen tensor bytes")

        def fsync(descriptor):
            events.append("fsync")
            original_fsync(descriptor)

        def replace(source, target):
            events.append("replace")
            self.assertEqual(Path(source).read_bytes(), b"frozen tensor bytes")
            original_replace(source, target)

        with patch.object(config.tempfile, "mkstemp", side_effect=temporary), \
                patch.object(config.os, "fsync", side_effect=fsync), \
                patch.object(config.os, "replace", side_effect=replace):
            config.atomic_torch_save({}, self.destination, torch_module=SimpleNamespace(save=save), root=self.root)
        self.assertEqual(events, ["save", "fsync", "replace"])
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0].parent, self.destination.parent)
        self.assertTrue(names[0].name.startswith("initial_state_tmp_"))
        self.assertFalse(names[0].name.startswith("."))
        self.assertEqual(names[0].suffix, ".pt")
        self.assertFalse(names[0].exists())

    def test_save_failure_cleans_partial_temporary_without_publishing(self):
        def fail(payload, target):
            target.write(b"incomplete")
            raise RuntimeError("intentional tensor writer failure")

        with self.assertRaisesRegex(RuntimeError, "intentional tensor writer failure"):
            config.atomic_torch_save({}, self.destination, torch_module=SimpleNamespace(save=fail), root=self.root)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.shared.glob("initial_state_tmp_*.pt")), [])

    def test_existing_destination_is_never_silently_overwritten(self):
        self.shared.mkdir(parents=True)
        self.destination.write_bytes(b"frozen original")
        writer = Mock()
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            config.atomic_torch_save({}, self.destination, torch_module=SimpleNamespace(save=writer), root=self.root)
        writer.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), b"frozen original")
        self.assertEqual(list(self.shared.glob("initial_state_tmp_*.pt")), [])

    def test_destination_appearing_during_write_is_preserved(self):
        def save(payload, target):
            target.write(b"our state")
            self.destination.write_bytes(b"independently frozen state")

        with self.assertRaisesRegex(FileExistsError, "appeared during serialization"):
            config.atomic_torch_save({}, self.destination, torch_module=SimpleNamespace(save=save), root=self.root)
        self.assertEqual(self.destination.read_bytes(), b"independently frozen state")
        self.assertEqual(list(self.shared.glob("initial_state_tmp_*.pt")), [])

    def test_guard_rejects_destinations_in_frozen_prior_steps(self):
        writer = Mock()
        forbidden = self.root / "cache/countermine_rgb/step2d/initial_state.pt"
        with self.assertRaisesRegex(ValueError, "project-owned output namespace"):
            config.atomic_torch_save({}, forbidden, torch_module=SimpleNamespace(save=writer), root=self.root)
        writer.assert_not_called()
        self.assertFalse(forbidden.parent.exists())


class InitializationTransactionTests(SharedDirectoryTestCase):
    def test_each_incomplete_final_pair_requires_manual_inspection(self):
        self.shared.mkdir(parents=True)
        for present in (self.destination, self.summary):
            with self.subTest(present=present.name):
                present.write_bytes(b"preserve me")
                with self.assertRaisesRegex(ValueError, "Incomplete shared initialization transaction.*manually after inspection"):
                    config.check_shared_initialization_transaction(self.shared, root=self.root)
                self.assertEqual(present.read_bytes(), b"preserve me")
                present.unlink()

    def test_cleanup_is_narrow_and_preserves_final_artifacts(self):
        self.shared.mkdir(parents=True)
        preserved = [self.destination, self.summary, self.shared / "preparation_summary.json",
                     self.shared / ".initial_state_old", self.shared / "initial_state_tmp_other.txt",
                     self.shared / "batch_plans/initial_state_tmp_nested.pt",
                     self.root / "cache/countermine_rgb/step2d/initial_state_tmp_previous.pt"]
        for artifact in preserved:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"preserve")
        stale = self.shared / "initial_state_tmp_abcd.pt"
        stale.write_bytes(b"interrupted write")
        directory = self.shared / "initial_state_tmp_directory.pt"
        directory.mkdir()
        self.assertTrue(config.check_shared_initialization_transaction(self.shared, root=self.root))
        self.assertFalse(stale.exists())
        self.assertTrue(directory.is_dir())
        for artifact in preserved:
            self.assertEqual(artifact.read_bytes(), b"preserve")

    def test_absent_shared_pair_is_ready_for_creation(self):
        self.assertFalse(config.check_shared_initialization_transaction(self.shared, root=self.root))
        self.assertFalse(self.shared.exists())

    def test_cleanup_refuses_a_different_namespace(self):
        with self.assertRaisesRegex(ValueError, "exact Step 3A shared directory"):
            config.check_shared_initialization_transaction(self.root / "cache/countermine_rgb/step2d", root=self.root)


class Stage30InitializationTests(SharedDirectoryTestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "tools/30_prepare_step3a_training.py"
        spec = importlib.util.spec_from_file_location("step3a_prepare_serialization_test", source)
        cls.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.launcher)

    def setUp(self):
        super().setUp()
        self.identity = {"salad_submodule_commit": "a" * 40,
                         "salad_source_hashes": {"salad/vpr_model.py": "b" * 64}}
        self.state = {"backbone.weight": object()}
        self.model = SimpleNamespace(
            state_dict=Mock(return_value=self.state),
            parameters=Mock(return_value=[SimpleNamespace(numel=lambda: 4, requires_grad=True)]),
            load_state_dict=Mock(return_value=SimpleNamespace(missing_keys=[], unexpected_keys=[])),
        )
        self.torch = SimpleNamespace(
            __version__="2.1.0+cpu", save=Mock(side_effect=lambda payload, target: target.write(b"verified state")),
            load=Mock(return_value=self.state),
        )

    def prepare(self):
        return self.launcher.prepare_shared_initialization(
            self.model, self.paths, self.identity, torch_module=self.torch,
        )

    def test_round_trip_precedes_summary_publication(self):
        events = []

        def strict_load(state, *, strict):
            self.assertTrue(strict)
            self.assertIs(state, self.state)
            self.assertFalse(self.summary.exists())
            self.assertGreater(self.destination.stat().st_size, 0)
            events.append("strict-load")
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def write_summary(path, payload, **kwargs):
            events.append("summary")
            config.write_json(path, payload, **kwargs)

        self.model.load_state_dict.side_effect = strict_load
        with patch.object(self.launcher, "write_json", side_effect=write_summary):
            summary = self.prepare()
        self.assertEqual(events, ["strict-load", "summary"])
        self.torch.load.assert_called_once_with(self.destination, map_location="cpu", weights_only=True)
        self.assertEqual(summary["initial_state_sha256"], config.sha256_file(self.destination))
        self.assertEqual(summary["model_configuration"], config.MODEL_CONFIG)
        self.assertEqual(summary["salad_source_hashes"], self.identity["salad_source_hashes"])
        self.assertEqual(summary["parameter_count"], 4)
        self.assertEqual(summary["trainable_parameter_count"], 4)
        self.assertEqual(summary["seed"], 42)
        self.assertEqual(config.read_json(self.summary), summary)

    def test_failed_round_trip_leaves_no_certifying_summary(self):
        self.model.load_state_dict.side_effect = RuntimeError("size mismatch for backbone.weight")
        with self.assertRaisesRegex(RuntimeError, "size mismatch for backbone.weight"):
            self.prepare()
        self.assertTrue(self.destination.exists())
        self.assertFalse(self.summary.exists())
        with self.assertRaisesRegex(ValueError, "Incomplete shared initialization transaction"):
            self.prepare()
        self.assertEqual(self.torch.save.call_count, 1)

    def test_existing_pair_is_strict_loaded_and_never_saved_again(self):
        original = self.prepare()
        self.torch.save.reset_mock()
        self.torch.load.reset_mock()
        self.model.state_dict.reset_mock()
        self.assertEqual(self.prepare(), original)
        self.torch.save.assert_not_called()
        self.model.state_dict.assert_not_called()
        self.torch.load.assert_called_once_with(self.destination, map_location="cpu", weights_only=True)

    def test_same_commit_with_changed_source_hashes_cannot_be_reused(self):
        self.prepare()
        self.torch.load.reset_mock()
        self.identity["salad_source_hashes"]["salad/vpr_model.py"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "different SALAD source"):
            self.prepare()
        self.torch.load.assert_not_called()

    def test_empty_serialized_file_never_gets_a_summary(self):
        self.torch.save.side_effect = lambda payload, target: None
        with self.assertRaisesRegex(ValueError, "tensor artifact is empty"):
            self.prepare()
        self.assertFalse(self.summary.exists())
        self.torch.load.assert_not_called()

    def test_startup_stops_partial_pairs_before_preflight_or_dataset_work(self):
        self.shared.mkdir(parents=True)
        for present in (self.destination, self.summary):
            with self.subTest(present=present.name):
                present.write_bytes(b"preserve scientific provenance")
                with patch.object(self.launcher, "ROOT", self.root), \
                        patch.object(self.launcher, "run_preflight") as preflight, \
                        patch.object(self.launcher, "enter_salad") as enter:
                    with self.assertRaisesRegex(ValueError, "Incomplete shared initialization transaction"):
                        self.launcher.main(["--seed", "42"])
                preflight.assert_not_called()
                enter.assert_not_called()
                present.unlink()


if __name__ == "__main__":
    unittest.main()

"""CPU engineering checks for callbacks, AMP skips and complete epoch evidence."""

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from countermine.training import step3a_runtime as runtime
from countermine.training.step3a_config import MODES


class FakeTensor:
    """Only the tensor inspection protocol used by the audit callbacks."""
    def __init__(self, values):
        self.values = np.asarray(values)
        self.grad = None

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return self.values.item()

    def clone(self):
        return FakeTensor(self.values.copy())

    def tolist(self):
        return self.values.tolist()

    def expand_as(self, other):
        return FakeTensor(np.broadcast_to(self.values, other.shape))

    def __getitem__(self, key):
        return FakeTensor(self.values[key])

    def __float__(self):
        return float(self.values)


class FakeSampler:
    def __init__(self, batches=((0, 1), (2, 3))):
        self.plan = SimpleNamespace(batches=batches)

    def __len__(self):
        return len(self.plan.batches)


def fake_runtimes():
    lightning, torch = ModuleType("pytorch_lightning"), ModuleType("torch")
    lightning.Callback = type("Callback", (), {})
    torch.equal = Mock(side_effect=lambda a, b: np.array_equal(a.values, b.values))
    torch.unique = Mock(side_effect=lambda value: np.unique(value.values))
    torch.isfinite = Mock(side_effect=lambda value: np.isfinite(value.values))
    torch.count_nonzero = Mock(side_effect=lambda value: np.count_nonzero(value.values))
    return {"pytorch_lightning": lightning, "torch": torch}


class Step3ARuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "cache/countermine_rgb/step3a/baseline/seed42"
        self.provenance = {"seed": 42, "initial_state_sha256": "a" * 64, "graph_sha256": "b" * 64}
        self.runtimes = fake_runtimes()

    def callback(self, *, smoke=True):
        with patch.dict(sys.modules, self.runtimes):
            return runtime.create_training_audit_callback(
                run_dir=self.run, mode="baseline", smoke=smoke,
                provenance=self.provenance, root=self.root,
            )

    def trainer_and_model(self, batches=((0, 1), (2, 3))):
        parameter = FakeTensor([1.0, 2.0])
        parameter.grad = FakeTensor([0.25, 0.5])
        model = SimpleNamespace(parameters=lambda: iter([parameter]), batch_acc=[0.25])
        data = SimpleNamespace(
            step3a_sampler=FakeSampler(batches),
            train_dataset=SimpleNamespace(places_ids=np.asarray([101, 102, 103, 104])),
            smoke_validation_audit=[],
        )
        trainer = SimpleNamespace(
            current_epoch=0, datamodule=data, global_step=0,
            callback_metrics={key: FakeTensor(0.7) for key in runtime.RECALL_KEYS},
        )
        return trainer, model, parameter

    def batch(self, *internal_ids):
        size = len(internal_ids)
        places = FakeTensor(np.broadcast_to(np.asarray(0.0), (size, 4, 3, 224, 224)))
        labels = FakeTensor(np.repeat(np.asarray(internal_ids)[:, None], 4, axis=1))
        return places, labels

    def test_finite_metrics_handles_scalars_and_tensor_protocol_without_torch(self):
        actual = runtime.finite_metrics({"loss": FakeTensor(1.25), "recall": np.float64(0.8), "count": 3})
        self.assertEqual(actual, {"loss": 1.25, "recall": 0.8, "count": 3.0})
        for value in (float("nan"), float("inf"), float("-inf"), FakeTensor(float("nan"))):
            with self.subTest(value=type(value).__name__):
                with self.assertRaisesRegex(ValueError, "Nonfinite Step 3A metric"):
                    runtime.finite_metrics({"loss": value})

    def test_critical_provenance_strips_only_exact_runtime_without_mutation(self):
        provenance = {**self.provenance, "runtime": {"gpu_name": "Test GPU"},
                      "Runtime": {"gpu_name": "Still critical"},
                      "unknown_future_invariant": {"runtime": "nested value stays critical"}}
        observed = runtime.critical_provenance(provenance)
        self.assertNotIn("runtime", observed)
        self.assertEqual(observed["Runtime"], {"gpu_name": "Still critical"})
        self.assertEqual(observed["unknown_future_invariant"],
                         {"runtime": "nested value stays critical"})
        observed["unknown_future_invariant"]["runtime"] = "updated copy"
        self.assertEqual(provenance["unknown_future_invariant"]["runtime"],
                         "nested value stays critical")
        self.assertIn("runtime", provenance)

    def test_smoke_gate_accepts_observed_runtime_changes(self):
        self.provenance["runtime"] = {"gpu_name": "Current GPU", "torch_version": "2.1.0+cu121"}
        paths, _ = self.smoke_files(change=lambda summary: summary["provenance"].update(
            runtime={"gpu_name": "Prior GPU", "torch_version": "2.1.0+local"}))
        runtime.require_smokes(paths, self.provenance)

    def test_smoke_gate_preserves_seed_code_and_unknown_critical_identity(self):
        paths, files = self.smoke_files()
        for field, value in (("seed", 7), ("code_hashes", {"training.py": "d" * 64}),
                             ("unknown_invariant", "changed"), ("Runtime", {"gpu_name": "GPU"})):
            with self.subTest(field=field):
                path, summary = files[1]
                original = summary["provenance"]
                summary["provenance"] = {**original, field: value}
                path.write_text(json.dumps(summary))
                with self.assertRaisesRegex(ValueError, "provenance differs"):
                    runtime.require_smokes(paths, self.provenance)
                summary["provenance"] = original
                path.write_text(json.dumps(summary))

    def test_training_and_evaluation_gate_runtime_before_reading_or_constructing(self):
        root = Path(__file__).resolve().parents[1]
        for tool, stage in (("31_train_step3a", "train"), ("32_evaluate_step3a", "evaluate")):
            with self.subTest(stage=stage):
                spec = importlib.util.spec_from_file_location(tool, root / "tools" / (tool + ".py"))
                cli = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cli)
                with patch.object(cli, "run_preflight", side_effect=RuntimeError("runtime unavailable")) as preflight, \
                        patch.object(cli, "validate_dataset_paths") as data_check, \
                        patch.object(cli, "read_json") as artifact_read, \
                        patch.object(cli, "create_model") as model:
                    with self.assertRaisesRegex(RuntimeError, "runtime unavailable"):
                        cli.main(["--mode", "baseline"])
                    preflight.assert_called_once_with(stage=stage, root=root,
                                                       save_report=False, progress=False)
                    data_check.assert_not_called()
                    artifact_read.assert_not_called()
                    model.assert_not_called()

    def smoke_files(self, *, change=None):
        paths = {"runtime": self.root / "cache/countermine_rgb/step3a"}
        check_names = (
            "shared_initialization_strict_load", "batch_shape_and_labels", "finite_loss",
            "finite_gradients", "optimizer_step", "validation", "sampler_invariants",
        )
        summaries = []
        for mode, path in zip(MODES, runtime.smoke_summary_paths(paths)):
            summary = {
                "complete": True, "smoke": True, "mode": mode,
                "provenance": dict(self.provenance),
                "engineering_checks": {name: True for name in check_names},
            }
            if change:
                change(summary)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(summary))
            summaries.append((path, summary))
        return paths, summaries

    def test_both_smoke_conditions_must_pass_with_current_shared_provenance(self):
        paths, files = self.smoke_files()
        runtime.require_smokes(paths, self.provenance)
        path, summary = files[1]
        summary["provenance"]["initial_state_sha256"] = "c" * 64
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            runtime.require_smokes(paths, self.provenance)

    def test_smoke_gate_rejects_incomplete_checks_and_missing_condition(self):
        paths, files = self.smoke_files()
        path, summary = files[0]
        del summary["engineering_checks"]["optimizer_step"]
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "optimizer_step"):
            runtime.require_smokes(paths, self.provenance)
        summary["engineering_checks"]["optimizer_step"] = True
        summary["smoke"] = False
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "Both independent smoke conditions"):
            runtime.require_smokes(paths, self.provenance)
        path.write_text(json.dumps({**summary, "smoke": True}))
        files[1][0].unlink()
        with self.assertRaises(FileNotFoundError):
            runtime.require_smokes(paths, self.provenance)

    def test_amp_overflow_is_left_to_the_original_scaler_and_never_counted_as_update(self):
        callback = self.callback(smoke=True)
        trainer, model, parameter = self.trainer_and_model()
        callback.on_train_epoch_start(trainer, model)
        callback.on_train_batch_start(trainer, model, self.batch(101, 102), 0)
        parameter.grad = FakeTensor([float("inf"), 0.5])
        callback.on_after_backward(trainer, model)
        # on_after_backward sees scaled gradients. It must not interrupt AMP.
        self.runtimes["torch"].isfinite.assert_not_called()
        callback.on_before_optimizer_step(trainer, model, object())
        callback.on_train_batch_end(trainer, model, {"loss": FakeTensor(1.0)}, None, 0)
        self.assertEqual(callback.backward_checks, 1)
        self.assertEqual(callback.amp_overflow_steps, 1)
        self.assertEqual(callback.finite_gradient_steps, 0)
        self.assertEqual(callback.optimizer_updates, 0)
        self.assertIsNone(callback.pending_step_parameter)
        self.assertEqual(parameter.values.tolist(), [1.0, 2.0])

    def test_smoke_counts_an_optimizer_update_only_after_weights_actually_change(self):
        callback = self.callback(smoke=True)
        trainer, model, parameter = self.trainer_and_model()
        callback.on_train_epoch_start(trainer, model)
        batch = self.batch(101, 102)
        callback.on_train_batch_start(trainer, model, batch, 0)
        callback.on_after_backward(trainer, model)
        callback.on_before_optimizer_step(trainer, model, object())
        self.assertEqual(callback.optimizer_updates, 0)
        parameter.values -= 0.1  # Simulate the original optimizer completing its update.
        callback.on_train_batch_end(trainer, model, {"loss": FakeTensor(1.0)}, batch, 0)
        self.assertEqual(callback.finite_gradient_steps, 1)
        self.assertEqual(callback.optimizer_updates, 1)
        self.assertIsNone(callback.pending_step_parameter)
        callback.on_validation_end(trainer, model)
        callback.on_train_epoch_end(trainer, model)
        self.assertEqual(callback.epochs[0]["train_batch_count"], 1)
        exported = json.loads((self.run / "epoch_metrics.json").read_text())
        self.assertEqual(exported["epochs"][0]["metrics"]["pitts30k_val/R1"], 0.7)

    def test_smoke_rejects_a_finite_gradient_step_that_does_not_update_weights(self):
        callback = self.callback(smoke=True)
        trainer, model, _ = self.trainer_and_model()
        callback.on_train_epoch_start(trainer, model)
        callback.on_before_optimizer_step(trainer, model, object())
        with self.assertRaisesRegex(ValueError, "did not update model weights"):
            callback.on_train_batch_end(trainer, model, {"loss": FakeTensor(1.0)}, None, 0)

    def test_backward_requires_gradients_and_smoke_requires_nonzero_gradient(self):
        callback = self.callback(smoke=True)
        trainer, model, parameter = self.trainer_and_model()
        parameter.grad = None
        with self.assertRaisesRegex(ValueError, "Backward produced no gradients"):
            callback.on_after_backward(trainer, model)
        parameter.grad = FakeTensor([0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "nonzero gradient"):
            callback.on_before_optimizer_step(trainer, model, object())

    def test_full_epoch_rejects_missing_or_extra_batches_and_preserves_amp_behavior(self):
        for consumed_batches in (1, 3):
            with self.subTest(consumed_batches=consumed_batches):
                callback = self.callback(smoke=False)
                trainer, model, parameter = self.trainer_and_model()
                callback.on_train_epoch_start(trainer, model)
                # Full runs do not replace the original GradScaler's overflow policy.
                parameter.grad = FakeTensor([float("inf"), 1.0])
                callback.on_after_backward(trainer, model)
                callback.on_before_optimizer_step(trainer, model, object())
                self.runtimes["torch"].isfinite.assert_not_called()
                for index in range(consumed_batches):
                    callback.on_train_batch_end(trainer, model, {"loss": FakeTensor(1.0)}, None, index)
                callback.on_validation_end(trainer, model)
                with self.assertRaisesRegex(ValueError, "every frozen place batch exactly once"):
                    callback.on_train_epoch_end(trainer, model)
                self.assertEqual(callback.epochs, [])

    def test_full_epoch_records_metrics_after_every_planned_batch_and_validation(self):
        callback = self.callback(smoke=False)
        trainer, model, _ = self.trainer_and_model()
        callback.on_train_epoch_start(trainer, model)
        for batch_index, internal_ids in enumerate(((101, 102), (103, 104))):
            batch = self.batch(*internal_ids)
            callback.on_train_batch_start(trainer, model, batch, batch_index)
            callback.on_after_backward(trainer, model)
            callback.on_train_batch_end(trainer, model, {"loss": FakeTensor(1.0 + batch_index)}, batch, batch_index)
        callback.on_validation_end(trainer, model)
        callback.on_train_epoch_end(trainer, model)
        self.assertEqual(callback.batch_checks, 2)
        self.assertEqual(callback.epochs[0]["metrics"]["loss"], 1.5)
        self.assertEqual(callback.epochs[0]["metrics"]["b_acc"], 0.25)
        self.assertEqual(callback.epochs[0]["train_batch_count"], len(trainer.datamodule.step3a_sampler))

    def test_loss_or_incomplete_validation_cannot_be_exported_as_complete_epoch(self):
        callback = self.callback(smoke=True)
        trainer, model, _ = self.trainer_and_model()
        callback.on_train_epoch_start(trainer, model)
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            callback.on_train_batch_end(trainer, model, {"loss": FakeTensor(float("nan"))}, None, 0)
        callback.on_train_batch_end(trainer, model, {"loss": FakeTensor(1.0)}, None, 0)
        with self.assertRaisesRegex(ValueError, "all upstream validation sets"):
            callback.on_train_epoch_end(trainer, model)
        trainer.callback_metrics["msls_val/R1"] = FakeTensor(float("inf"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            callback.on_validation_end(trainer, model)
        self.assertFalse((self.run / "epoch_metrics.json").exists())

    def test_training_summary_rejects_smoke_without_successful_optimizer_update(self):
        callback = self.callback(smoke=True)
        trainer, model, _ = self.trainer_and_model()
        callback.on_train_epoch_start(trainer, model)
        callback.on_train_batch_start(trainer, model, self.batch(101, 102), 0)
        callback.on_after_backward(trainer, model)
        callback.on_train_batch_end(trainer, model, {"loss": FakeTensor(1.0)}, None, 0)
        callback.on_validation_end(trainer, model)
        callback.on_train_epoch_end(trainer, model)
        self.run.mkdir(parents=True, exist_ok=True)
        best, last = self.run / "best.ckpt", self.run / "last.ckpt"
        best.write_bytes(b"temporary test checkpoint")
        last.write_bytes(b"temporary test checkpoint")
        checkpoint = SimpleNamespace(best_model_path=str(best), last_model_path=str(last))
        with self.assertRaisesRegex(ValueError, "finite successful optimizer update"):
            runtime.training_summary(
                trainer=trainer, callback=callback, checkpoint=checkpoint,
                run_dir=self.run, mode="baseline", smoke=True,
                provenance=self.provenance, root=self.root,
            )
        self.assertFalse((self.run / "training_summary.json").exists())


if __name__ == "__main__":
    unittest.main()

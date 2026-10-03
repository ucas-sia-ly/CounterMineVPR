"""CPU-only monitoring checks; no SALAD training or scientific artifacts."""

import ast
from contextlib import ExitStack, nullcontext, redirect_stdout
from copy import deepcopy
import importlib.abc
import importlib.util
import io
import random
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MappingProxyType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from countermine.training import step3a_monitoring as monitoring
from countermine.training.countermine_batch_sampler import CounterMineBatchSampler, EdgeBundle
from countermine.training.gsv_place_mapping import PlaceRecord
from countermine.training.step3a_config import MODES, TRAINER_CONFIG

ROOT = Path(__file__).resolve().parents[1]


def fake_lightning(*, wandb_available=True):
    lightning = ModuleType("pytorch_lightning")
    lightning.Callback = type("Callback", (), {})
    callbacks = ModuleType("pytorch_lightning.callbacks")
    callbacks.LearningRateMonitor = Mock(name="LearningRateMonitor")
    callbacks.ModelCheckpoint = Mock(name="ModelCheckpoint")
    lightning.callbacks = callbacks
    loggers = ModuleType("pytorch_lightning.loggers")
    loggers.CSVLogger = Mock(name="CSVLogger")
    loggers.TensorBoardLogger = Mock(name="TensorBoardLogger")
    loggers.wandb_import_attempts = []

    def optional_logger(name):
        if name == "WandbLogger":
            loggers.wandb_import_attempts.append(name)
            if not wandb_available:
                raise ImportError("wandb is not installed")
            return wandb_logger
        raise AttributeError(name)

    wandb_logger = Mock(name="WandbLogger")
    loggers.__getattr__ = optional_logger
    lightning.Trainer = Mock(name="Trainer")
    modules = {"pytorch_lightning": lightning,
               "pytorch_lightning.callbacks": callbacks,
               "pytorch_lightning.loggers": loggers, "wandb": None}
    return modules, lightning, loggers, wandb_logger


def sampler_fixture(mode, *, baseline_zero=False, epoch=0):
    records = tuple(PlaceRecord(i, 100 + i, "Boston", f"Boston:{i + 1:07d}") for i in range(12))
    edges = frozenset((records[a].canonical_place_uid, records[b].canonical_place_uid)
                      for a, b in ((0, 7), (2, 9), (4, 11)))
    if not baseline_zero:
        edges |= frozenset([(records[0].canonical_place_uid, records[1].canonical_place_uid)])
    bundle = EdgeBundle("a" * 64, "b" * 64,
                        frozenset(record.canonical_place_uid for record in records),
                        edges, edges, edges)
    return CounterMineBatchSampler(records, bundle, mode=mode, batch_size=4,
                                  epoch=epoch, require_exposure_increase=False)


class Step3AMonitoringTests(unittest.TestCase):
    def configuration(self, mode="baseline", **kwargs):
        return monitoring.monitoring_configuration(mode=mode, seed=42, **kwargs)

    def callback(self):
        modules, _, _, _ = fake_lightning()
        with patch.dict(sys.modules, modules):
            lr, callback = monitoring.create_monitoring_callbacks()
        modules["pytorch_lightning.callbacks"].LearningRateMonitor.assert_called_once_with(
            logging_interval="step")
        return callback

    def trainer(self, sampler, *, epoch=0, global_zero=True):
        return SimpleNamespace(
            current_epoch=epoch, global_step=epoch * 20, is_global_zero=global_zero,
            datamodule=SimpleNamespace(step3a_sampler=sampler),
            loggers=[Mock(), Mock()], callback_metrics={"pitts30k_val/R1": 70},
        )

    def test_csv_and_tensorboard_paths_are_identical_for_both_conditions_and_smokes(self):
        for mode in MODES:
            for smoke in (False, True):
                with self.subTest(mode=mode, smoke=smoke):
                    modules, _, loggers, _ = fake_lightning(wandb_available=False)
                    run = ROOT / "cache/countermine_rgb/step3a" / ("smoke" if smoke else "") / mode / "seed42"
                    with patch.dict(sys.modules, modules):
                        actual = monitoring.create_training_loggers(run, self.configuration(mode, smoke=smoke))
                    self.assertEqual(actual, [loggers.CSVLogger.return_value,
                                              loggers.TensorBoardLogger.return_value])
                    loggers.CSVLogger.assert_called_once_with(str(run), name="logs", version=0)
                    loggers.TensorBoardLogger.assert_called_once_with(
                        str(run), name="tensorboard", version="", log_graph=False,
                        default_hp_metric=False, flush_secs=20)
                    self.assertEqual(loggers.wandb_import_attempts, [])

    def test_wandb_is_lazy_and_missing_dependency_has_installation_instructions(self):
        modules, _, loggers, _ = fake_lightning(wandb_available=False)
        with patch.dict(sys.modules, modules):
            monitoring.create_training_loggers("run", self.configuration())
            self.assertEqual(loggers.wandb_import_attempts, [])
            with self.assertRaisesRegex(RuntimeError, r"python -m pip install wandb"):
                monitoring.create_training_loggers("run", self.configuration(wandb=True))
        self.assertTrue(loggers.wandb_import_attempts)

    def test_wandb_configuration_names_tags_offline_and_no_checkpoint_upload(self):
        for mode in MODES:
            for smoke in (False, True):
                for offline in (False, True):
                    with self.subTest(mode=mode, smoke=smoke, offline=offline):
                        modules, _, _, wandb = fake_lightning()
                        config = self.configuration(mode, smoke=smoke, wandb=True,
                                                    wandb_entity="team", wandb_offline=offline)
                        with patch.dict(sys.modules, modules):
                            actual = monitoring.create_training_loggers("run", config)
                        name = "step3a-" + ("smoke-" if smoke else "") + mode.replace("_", "-") + "-seed42"
                        expected_tags = ["step3a", "seed42", *mode.split("_", 1)] + (["smoke"] if smoke else [])
                        wandb.assert_called_once_with(
                            save_dir="run", name=name, project="CounterMineVPR", entity="team",
                            offline=offline, tags=expected_tags, log_model=False)
                        self.assertIs(actual[2], wandb.return_value)
                        self.assertEqual(config["run_name"], name)
        default = self.configuration(wandb_offline=True)
        self.assertEqual(default, {"csv": True, "tensorboard": True, "wandb": False,
                                   "wandb_offline": False})
        custom = self.configuration(wandb=True, wandb_project="custom")
        self.assertEqual(custom["project"], "custom")
        self.assertNotIn("entity", custom)

    def test_callback_logs_real_frozen_summary_once_each_epoch_without_mutation(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                sampler = sampler_fixture(mode)
                callback = self.callback()
                trainer = self.trainer(sampler)
                model = Mock()
                for epoch in range(2):
                    # The training datamodule creates its plan; monitoring only observes it.
                    sampler.set_epoch(epoch)
                    trainer.current_epoch, trainer.global_step = epoch, epoch * 20
                    plan, before_batches = sampler.plan, list(sampler)
                    summary = deepcopy(plan.summary)
                    random_state, numpy_state = random.getstate(), np.random.get_state()
                    callback.on_train_epoch_start(trainer, model)
                    callback.on_train_epoch_start(trainer, model)
                    self.assertIs(sampler.plan, plan)
                    self.assertEqual(plan.summary, summary)
                    self.assertEqual(list(sampler), before_batches)
                    self.assertEqual(random.getstate(), random_state)
                    after_numpy = np.random.get_state()
                    self.assertEqual(after_numpy[0], numpy_state[0])
                    np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
                    self.assertEqual(after_numpy[2:], numpy_state[2:])
                    expected = {
                        "countermine/q95_cobatched_edges": summary["exposure"]["core_q95"],
                        "countermine/q99_cobatched_edges": summary["exposure"]["core_q99"],
                        "countermine/q99_geo500_cobatched_edges": summary["exposure"]["core_q99_geo500"],
                        "countermine/guided_pairs": summary["greedy_matched_pair_count"],
                        "countermine/unique_guided_places": summary["unique_guided_place_count"],
                        "countermine/q99_geo500_exposure_gain": summary["structural_exposure_gain"],
                        "countermine/q99_geo500_exposure_ratio": summary["structural_exposure_ratio"],
                        "epoch": epoch,
                    }
                    for logger in trainer.loggers:
                        self.assertEqual(logger.log_metrics.call_count, epoch + 1)
                        logger.log_metrics.assert_called_with(expected, step=epoch * 20)
                        self.assertEqual(logger.save.call_count, epoch + 1)
                    if mode == "baseline":
                        self.assertEqual(expected["countermine/guided_pairs"], 0)
                        self.assertEqual(expected["countermine/unique_guided_places"], 0)
                    self.assertEqual(trainer.callback_metrics, {"pitts30k_val/R1": 70})
                self.assertEqual(model.mock_calls, [])

    def test_callback_uses_only_readonly_summary_and_omits_zero_denominator_ratio(self):
        for mode in MODES:
            sampler = sampler_fixture(mode, baseline_zero=True)
            summary = sampler.plan.summary
            readonly = MappingProxyType({key: MappingProxyType(value) if isinstance(value, dict)
                                         else value for key, value in summary.items()})

            class SummaryOnlyPlan:
                def __init__(self):
                    self.summary = readonly

                def __getattr__(self, name):
                    raise AssertionError(f"Monitoring read plan data beyond the summary: {name}")

            trainer = self.trainer(SimpleNamespace(plan=SummaryOnlyPlan()))
            self.callback().on_train_epoch_start(trainer, object())
            metrics = trainer.loggers[0].log_metrics.call_args.args[0]
            self.assertNotIn("countermine/q99_geo500_exposure_ratio", metrics)
            self.assertTrue(all(isinstance(value, (int, float)) for value in metrics.values()))
            self.assertEqual(metrics["countermine/q99_geo500_exposure_gain"], summary["structural_exposure_gain"])

    def test_nonzero_rank_does_not_log(self):
        trainer = self.trainer(object(), global_zero=False)
        self.callback().on_train_epoch_start(trainer, object())
        for logger in trainer.loggers:
            logger.log_metrics.assert_not_called()

    def test_import_and_cli_help_need_no_monitoring_or_model_dependencies(self):
        code = '''
import importlib.abc, runpy, sys
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'pytorch_lightning', 'wandb', 'tensorboard'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, RejectRuntime())
import countermine.training.step3a_monitoring
sys.argv = ['31_train_step3a', '--help']
runpy.run_path('tools/31_train_step3a.py', run_name='__main__')
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--wandb", "--wandb-project", "--wandb-entity", "--wandb-offline"):
            self.assertIn(flag, result.stdout)

    def test_no_watch_graph_capture_large_object_logging_or_training_overrides(self):
        paths = list((ROOT / "countermine/training").glob("*.py")) + [ROOT / "tools/31_train_step3a.py"]
        forbidden_calls = {"watch", "add_graph", "log_graph", "log_image", "add_image",
                           "add_images", "add_histogram", "log_table"}
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                    self.assertNotIn(name, forbidden_calls, str(path))
        tree = ast.parse(Path(monitoring.__file__).read_text())
        functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertEqual(functions, {"monitoring_configuration", "create_training_loggers",
                                     "create_monitoring_callbacks", "__init__", "on_train_epoch_start"})

    def test_launcher_keeps_scientific_settings_and_exports_with_wandb_absent(self):
        spec = importlib.util.spec_from_file_location("step3a_monitoring_cli", ROOT / "tools/31_train_step3a.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        for mode in MODES:
            for smoke in (False, True):
                with self.subTest(mode=mode, smoke=smoke), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    paths = {key: root / key for key in ("shared", "graph", "snapshot")}
                    paths["runtime"] = root / "cache/countermine_rgb/step3a"
                    prep = {"dataset_metadata": {}, "provenance": {"seed": 42, "runtime": {},
                            "step3a_code_hashes": {"fixture.py": "a" * 64}},
                            "mapping_audit": {}, "epoch0_plans": {"baseline": {"dataset_place_order_sha256": "a" * 64}},
                            "configuration": {"frozen": True}}
                    provenance = deepcopy(prep["provenance"])
                    modules, lightning, loggers, _ = fake_lightning(wandb_available=False)
                    torch = ModuleType("torch")
                    torch.cuda = SimpleNamespace(is_available=lambda: True)
                    torch.backends = SimpleNamespace(cudnn=SimpleNamespace(),
                                                     cuda=SimpleNamespace(matmul=SimpleNamespace()))
                    modules["torch"] = torch
                    model, dm, audit = object(), object(), object()
                    # All model/data/training operations are stubs; only CLI wiring executes.
                    replacements = {
                        "ROOT": root, "resolved_paths": Mock(return_value=paths),
                        "run_preflight": Mock(return_value={"runtime": {}}),
                        "validate_dataset_paths": Mock(return_value={}),
                        "check_preparation": Mock(return_value=prep), "require_smokes": Mock(),
                        "validate_source_hashes": Mock(return_value={"policy": "fixture"}),
                        "guard_step3a": lambda path: path,
                        "load_edge_bundle": Mock(), "enter_salad": Mock(), "seed_runtime": Mock(),
                        "create_model": Mock(return_value=model), "strict_load_initial_state": Mock(),
                        "cached_dinov2_runtime": Mock(return_value=nullcontext({"mode": "fixture"})),
                        "read_json": Mock(return_value={"complete": True, "smoke": False, "provenance": provenance}),
                        "create_training_audit_callback": Mock(return_value=audit),
                        "write_json": Mock(), "training_summary": Mock(return_value={"best_epoch": 0}),
                    }
                    from countermine.training import salad_datamodule
                    with ExitStack() as stack:
                        stack.enter_context(patch.dict(sys.modules, modules))
                        for name, value in replacements.items():
                            stack.enter_context(patch.object(cli, name, value))
                        build_dm = stack.enter_context(patch.object(salad_datamodule, "build_datamodule", return_value=dm))
                        stack.enter_context(redirect_stdout(io.StringIO()))
                        args = ["--mode", mode]
                        if smoke:
                            args += ["--smoke", "--max-epochs", "1", "--limit-train-batches", "3", "--limit-val-batches", "2"]
                        cli.main(args)
                    trainer_args = lightning.Trainer.call_args.kwargs
                    self.assertEqual(trainer_args["logger"], [loggers.CSVLogger.return_value,
                                                             loggers.TensorBoardLogger.return_value])
                    self.assertEqual(trainer_args["callbacks"][:2], [audit, lightning.callbacks.ModelCheckpoint.return_value])
                    self.assertEqual(len(trainer_args["callbacks"]), 4)
                    lightning.callbacks.LearningRateMonitor.assert_called_once_with(logging_interval="step")
                    self.assertEqual(type(trainer_args["callbacks"][-1]).__name__, "Step3AMonitoringCallback")
                    for name, value in TRAINER_CONFIG.items():
                        self.assertEqual(trainer_args[name], (1 if smoke else 4) if name == "max_epochs" else value)
                    self.assertEqual(trainer_args["log_every_n_steps"], 20)
                    checkpoint_args = lightning.callbacks.ModelCheckpoint.call_args.kwargs
                    self.assertEqual(checkpoint_args["monitor"], "pitts30k_val/R1")
                    self.assertEqual(checkpoint_args["mode"], "max")
                    self.assertEqual(checkpoint_args["save_top_k"], 1)
                    self.assertTrue(checkpoint_args["save_last"])
                    replacements["seed_runtime"].assert_called_once_with(42)
                    self.assertEqual(build_dm.call_args.kwargs["mode"], mode)
                    lightning.Trainer.return_value.fit.assert_called_once_with(model=model, datamodule=dm)
                    payload = replacements["write_json"].call_args.args[1]
                    self.assertEqual(payload["monitoring"], self.configuration(mode, smoke=smoke))
                    self.assertEqual(payload["configuration"], prep["configuration"])
                    self.assertEqual(payload["provenance"], provenance)
                    summary_args = replacements["training_summary"].call_args.kwargs
                    self.assertIs(summary_args["callback"], audit)
                    self.assertEqual(summary_args["provenance"], provenance)
                    self.assertNotIn("monitoring", summary_args)
                    self.assertEqual(loggers.wandb_import_attempts, [])


if __name__ == "__main__":
    unittest.main()

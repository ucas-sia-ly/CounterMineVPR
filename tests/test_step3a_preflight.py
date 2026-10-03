"""CPU-only regression checks for read-only, stage-specific runtime preflight."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from countermine.training import runtime_preflight as preflight
from countermine.training.step3a_audit import validate_portable_json
from countermine.training.step3a_config import TRAIN_CITIES, resolved_paths


ROOT = Path(__file__).resolve().parents[1]


def fake_torch(*, available=False, devices=0, finite=True):
    result = SimpleNamespace(all=Mock(return_value=SimpleNamespace(item=Mock(return_value=finite))))
    tensor = Mock()
    tensor.__matmul__ = Mock(return_value=object())
    return SimpleNamespace(
        version=SimpleNamespace(cuda="12.1"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=Mock(return_value=8902))),
        cuda=SimpleNamespace(
            is_available=Mock(return_value=available), device_count=Mock(return_value=devices),
            get_device_properties=Mock(return_value=SimpleNamespace(name="Example CUDA GPU", total_memory=24 * 1024 ** 3)),
            get_device_capability=Mock(return_value=(8, 9)), synchronize=Mock(),
        ),
        ones=Mock(return_value=tensor), isfinite=Mock(return_value=result),
    )


class RuntimePreflightTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.torch = fake_torch()
        self.graph = {"sha256": "1" * 64, "snapshot_sha256": "2" * 64,
                      "place_count": 2000, "eligible_same_city_edges": {"Boston": 1083, "London": 1003}}
        self.dataset = {"dataset_root": "data/GSVCities", "root_is_symlink": False,
                        "resolved_dataset_root_sha256": "3" * 64,
                        "metadata_sha256": {f"data/GSVCities/Dataframes/{city}.csv": "4" * 64
                                            for city in TRAIN_CITIES}}
        self.salad = {"salad_submodule_commit": "a" * 40, "salad_source_hashes": {"salad/vpr_model.py": "5" * 64}}
        self.versions = {"python": "3.10.18", **{name: "1.2.3+local" for name in preflight.PACKAGE_DISTRIBUTIONS}}
        self.imported = []

    def successful_checks(self, *, importer=None):
        def default_importer(name):
            self.imported.append(name)
            return self.torch if name == "torch" else SimpleNamespace()
        stack = ExitStack()
        stack.enter_context(patch.object(preflight, "_package_versions", return_value=self.versions))
        stack.enter_context(patch.object(preflight, "_validate_graph", return_value=deepcopy(self.graph)))
        stack.enter_context(patch.object(preflight, "validate_dataset_paths", return_value=deepcopy(self.dataset)))
        stack.enter_context(patch.object(preflight, "_validate_salad", return_value=deepcopy(self.salad)))
        stack.enter_context(patch.object(preflight.importlib, "import_module", side_effect=importer or default_importer))
        stack.enter_context(patch.object(preflight.shutil, "disk_usage", return_value=SimpleNamespace(free=182 * 1024 ** 3)))
        return stack

    def test_prepare_permits_cpu_and_preserves_authoritative_metadata(self):
        with self.successful_checks():
            report = preflight.run_preflight("prepare", root=self.root)
        self.assertIs(report["complete"], True)
        self.assertIs(report["gpu"]["cuda_available"], False)
        self.torch.ones.assert_not_called()
        self.assertNotIn("cuda_smoke", report["checks"])
        self.assertEqual(report["dataset_metadata"], self.dataset)
        self.assertEqual(report["dataset_status"]["training_city_csv_count"], len(TRAIN_CITIES))
        self.assertIn("xformers.ops", self.imported)
        self.assertNotIn("faiss", self.imported)
        self.assertNotIn("matplotlib", self.imported)
        self.assertIn("+local", report["runtime"]["torch_version"])
        self.assertFalse((self.root / "cache").exists())

    def test_training_requires_cuda_and_smokes_only_tiny_tensor(self):
        self.torch = fake_torch(available=True, devices=1)
        with self.successful_checks():
            report = preflight.run_preflight("train", root=self.root)
        self.torch.ones.assert_called_once_with((2, 2), device="cuda")
        self.torch.cuda.synchronize.assert_called_once_with()
        self.assertTrue(report["checks"]["cuda_smoke"]["passed"])
        self.assertEqual(report["gpu"]["gpu_name"], "Example CUDA GPU")
        self.assertEqual(report["runtime"]["gpu_capability"], [8, 9])
        self.assertEqual(report["disk"]["minimum_free_bytes"], 5 * 1024 ** 3)
        self.assertIn("faiss.contrib.torch_utils", self.imported)

    def test_cublas_workspace_is_configured_before_first_runtime_import_and_cuda_operation(self):
        self.torch = fake_torch(available=True, devices=1)
        tensor = self.torch.ones.return_value

        def importer(name):
            self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")
            return self.torch if name == "torch" else SimpleNamespace()

        def ones(*args, **kwargs):
            self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")
            return tensor

        self.torch.ones.side_effect = ones
        with patch.dict(os.environ, {}), self.successful_checks(importer=importer):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            report = preflight.run_preflight("train", root=self.root)
        self.assertEqual(report["cublas_workspace_config"], ":4096:8")
        self.assertTrue(report["checks"]["cublas_workspace"]["passed"])
        self.torch.ones.assert_called_once_with((2, 2), device="cuda")

    def test_invalid_workspace_stops_before_runtime_imports_or_cuda(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ""}), self.successful_checks():
            with self.assertRaisesRegex(preflight.RuntimePreflightError, "Invalid CUBLAS_WORKSPACE_CONFIG"):
                preflight.run_preflight("train", root=self.root)
        self.assertEqual(self.imported, [])
        self.torch.ones.assert_not_called()
        self.torch.cuda.is_available.assert_not_called()

    def test_cpu_export_does_not_change_cublas_environment_or_require_valid_config(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ""}), self.successful_checks():
            report = preflight.run_preflight("export", root=self.root)
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], "")
        self.assertNotIn("cublas_workspace_config", report)
        self.assertNotIn("cublas_workspace", report["checks"])
        self.torch.ones.assert_not_called()

    def test_prepare_cuda_metadata_failure_is_diagnostic_and_cpu_still_allowed(self):
        self.torch.cuda.device_count.side_effect = RuntimeError("NVIDIA driver query failed")
        with self.successful_checks():
            report = preflight.run_preflight("prepare", root=self.root)
        self.assertTrue(report["complete"])
        self.assertFalse(report["checks"]["runtime_metadata"]["cuda_observation_available"])
        self.assertIn("NVIDIA driver query failed", report["checks"]["runtime_metadata"]["diagnostic"])
        self.assertIsNone(report["gpu"]["cuda_available"])
        self.assertEqual(report["runtime"]["cuda_version"], "12.1")
        self.torch.ones.assert_not_called()

    def test_cuda_unavailable_blocks_training_and_evaluation(self):
        for stage in ("train", "evaluate", "all"):
            with self.subTest(stage=stage), self.successful_checks():
                with self.assertRaisesRegex(preflight.RuntimePreflightError, "CUDA is unavailable") as caught:
                    preflight.run_preflight(stage, root=self.root, save_report=True)
            self.assertFalse(caught.exception.report["complete"])
            self.assertFalse((self.root / "cache").exists())
        self.torch.ones.assert_not_called()

    def test_zero_devices_is_not_a_valid_cuda_workstation(self):
        self.torch = fake_torch(available=True, devices=0)
        with self.successful_checks(), self.assertRaisesRegex(preflight.RuntimePreflightError, "CUDA is unavailable"):
            preflight.run_preflight("evaluate", root=self.root)
        self.torch.ones.assert_not_called()

    def test_nonfinite_cuda_smoke_is_rejected(self):
        self.torch = fake_torch(available=True, devices=1, finite=False)
        with self.successful_checks(), self.assertRaisesRegex(preflight.RuntimePreflightError, "nonfinite output"):
            preflight.run_preflight("train", root=self.root)

    def test_broken_import_keeps_original_reason_and_action(self):
        def importer(name):
            if name == "torchvision":
                raise RuntimeError("operator torchvision::nms does not exist")
            return self.torch if name == "torch" else SimpleNamespace()
        with self.successful_checks(importer=importer):
            with self.assertRaises(preflight.RuntimePreflightError) as caught:
                preflight.run_preflight("prepare", root=self.root)
        self.assertIn("operator torchvision::nms does not exist", str(caught.exception))
        self.assertIn("Install or repair torchvision", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_missing_matplotlib_blocks_export_without_model_imports(self):
        def importer(name):
            self.imported.append(name)
            if name == "matplotlib":
                raise ModuleNotFoundError("No module named 'matplotlib'")
            return SimpleNamespace()
        with self.successful_checks(importer=importer), self.assertRaisesRegex(preflight.RuntimePreflightError, "repair matplotlib"):
            preflight.run_preflight("export", root=self.root)
        self.assertFalse(set(self.imported) & {"torch", "torchvision", "pytorch_lightning", "faiss", "xformers"})

    def test_successful_export_is_cpu_only_even_with_loaded_torch(self):
        with self.successful_checks(), patch.dict(sys.modules, {"torch": self.torch}):
            report = preflight.run_preflight("export", root=self.root)
        self.torch.cuda.is_available.assert_not_called()
        self.torch.cuda.device_count.assert_not_called()
        self.torch.ones.assert_not_called()
        self.assertTrue(report["complete"])
        self.assertIsNone(report["runtime"]["gpu_name"])
        self.assertIsNone(report["runtime"]["cuda_version"])
        self.assertNotIn("checkpoint_disk", report["checks"])

    def test_low_disk_blocks_training_but_never_deletes_caches(self):
        self.torch = fake_torch(available=True, devices=1)
        artifact = self.root / "cache/countermine_rgb/step3a/important.pt"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"frozen")
        with self.successful_checks(), patch.object(preflight.shutil, "disk_usage", return_value=SimpleNamespace(free=4 * 1024 ** 3)):
            with self.assertRaisesRegex(preflight.RuntimePreflightError, "at least 5 GiB"):
                preflight.run_preflight("train", root=self.root)
        self.assertEqual(artifact.read_bytes(), b"frozen")

    def test_disk_threshold_is_inclusive_and_not_required_for_evaluation(self):
        self.torch = fake_torch(available=True, devices=1)
        with self.successful_checks(), patch.object(preflight.shutil, "disk_usage", return_value=SimpleNamespace(free=5 * 1024 ** 3)):
            self.assertTrue(preflight.run_preflight("train", root=self.root)["complete"])
        with self.successful_checks(), patch.object(preflight.shutil, "disk_usage", side_effect=AssertionError("evaluation must not check disk")):
            self.assertTrue(preflight.run_preflight("evaluate", root=self.root)["complete"])

    def test_dataset_failure_preserves_exact_root_message(self):
        with self.successful_checks(), patch.object(preflight, "validate_dataset_paths", side_effect=FileNotFoundError(
                "Required upstream dataset paths are missing: data/GSVCities/Dataframes/Boston.csv")):
            with self.assertRaisesRegex(preflight.RuntimePreflightError, "data/GSVCities/Dataframes/Boston.csv"):
                preflight.run_preflight("prepare", root=self.root)

    def test_successful_report_is_portable_finite_utc_and_saved_only_on_request(self):
        with self.successful_checks():
            report = preflight.run_preflight("prepare", root=self.root, save_report=True)
        destination = self.root / "cache/countermine_rgb/step3a/runtime_preflight.json"
        self.assertEqual(json.loads(destination.read_text()), report)
        validate_portable_json(report)
        serialized = json.dumps(report, allow_nan=False)
        for forbidden in (str(self.root), "hostname", "username", "environment_variables", "serial_number"):
            self.assertNotIn(forbidden, serialized)
        self.assertTrue(report["timestamp_utc"].endswith("+00:00"))
        self.assertEqual(report["graph"], self.graph)
        self.assertEqual(set(report["runtime"]), {
            "python_version", "torch_version", "torchvision_version", "lightning_version",
            "pytorch_metric_learning_version", "faiss_version", "xformers_version", "numpy_version",
            "pandas_version", "cuda_version", "cudnn_version", "gpu_name", "gpu_capability",
        })
        self.assertEqual(set(report["package_versions"]), {"python", *preflight.PACKAGE_DISTRIBUTIONS})

    def test_prepare_progress_labels_precede_model_construction(self):
        output = io.StringIO()
        with self.successful_checks(), redirect_stdout(output):
            preflight.run_preflight("prepare", root=self.root, progress=True)
        self.assertEqual(output.getvalue().splitlines(), [
            "[1/6] validating frozen Step 2D inputs", "[2/6] validating dataset roots",
            "[3/6] validating SALAD source",
        ])

    def test_unknown_stage_fails_before_checks(self):
        with self.assertRaisesRegex(ValueError, "Unknown Step 3A"):
            preflight.run_preflight("training", root=self.root)

    def test_compact_runtime_helper_does_not_import_or_query_torch_implicitly(self):
        with patch.object(preflight, "_package_versions", return_value=self.versions), \
                patch.object(preflight.importlib, "import_module", side_effect=AssertionError("unexpected import")), \
                patch.dict(sys.modules, {"torch": self.torch}):
            report = preflight.collect_runtime_metadata()
        self.assertEqual(report["torch_version"], "1.2.3+local")
        self.assertIsNone(report["gpu_name"])
        self.torch.cuda.is_available.assert_not_called()

    def test_installed_faiss_cpu_version_is_supported_without_gpu_resources(self):
        def version(name):
            if name == "faiss-gpu":
                raise preflight.metadata.PackageNotFoundError(name)
            return "1.7.4+cpu" if name == "faiss-cpu" else "2.1.0+cu121"
        with patch.object(preflight.metadata, "version", side_effect=version):
            self.assertEqual(preflight._package_versions()["faiss"], "1.7.4+cpu")


class FrozenSourcePreflightTests(unittest.TestCase):
    def test_graph_loader_requires_exact_population_and_same_city_edge_counts(self):
        from countermine.training import countermine_batch_sampler as sampler
        bundle = SimpleNamespace(
            graph_sha256="a" * 64, snapshot_sha256="b" * 64, graph_place_uids=frozenset(range(2000)),
            eligible_edges=lambda city: [None] * (1083 if city == "Boston" else 1003),
        )
        paths = resolved_paths(ROOT)
        with patch.object(sampler, "load_edge_bundle", return_value=bundle) as load:
            graph = preflight._validate_graph(paths)
        load.assert_called_once_with(paths["graph"], paths["snapshot"], expected_graph_places=2000)
        self.assertEqual(graph["eligible_same_city_edges"], {"Boston": 1083, "London": 1003})
        for wrong_city in ("Boston", "London"):
            bundle.eligible_edges = lambda city, wrong_city=wrong_city: [None] * (
                (1083 if city == "Boston" else 1003) + (city == wrong_city))
            with self.subTest(city=wrong_city), patch.object(sampler, "load_edge_bundle", return_value=bundle):
                with self.assertRaisesRegex(ValueError, "Frozen same-city q99_geo500 edge counts differ"):
                    preflight._validate_graph(paths)

    def test_salad_untracked_changes_are_not_accepted_as_clean(self):
        with patch.object(preflight, "salad_identity", return_value={"salad_submodule_commit": "abc"}), \
                patch.object(preflight.subprocess, "check_output", return_value="?? changed_model.py\n"):
            with self.assertRaisesRegex(ValueError, "SALAD submodule is not clean"):
                preflight._validate_salad(ROOT)

    def test_error_messages_redact_host_paths(self):
        error = ImportError("failed to load /home/example/private/build/extension.so from /tmp/build")
        portable = preflight._portable_error(error, ROOT)
        self.assertNotIn("/home/", portable)
        self.assertNotIn("/tmp/", portable)
        self.assertIn("failed to load", portable)


class RuntimePreflightCliTests(unittest.TestCase):
    def load_cli(self):
        spec = importlib.util.spec_from_file_location("step3a_preflight_cli", ROOT / "tools/29_check_step3a_runtime.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        return cli

    def test_cli_help_requires_no_neural_runtime(self):
        code = '''
import importlib.abc, runpy, sys
class RejectModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'pytorch_lightning', 'faiss', 'xformers'}:
            raise AssertionError('help imported runtime: ' + fullname)
sys.meta_path.insert(0, RejectModels())
sys.argv = ['29_check_step3a_runtime', '--help']
runpy.run_path('tools/29_check_step3a_runtime.py', run_name='__main__')
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--save-report", result.stdout)
        self.assertIn("--all", result.stdout)
        self.assertIn("--stage", result.stdout)

    def test_cli_nonzero_failure_is_actionable_and_report_is_optional(self):
        from contextlib import redirect_stderr
        cli = self.load_cli()
        error = preflight.RuntimePreflightError("CUDA unavailable: repair driver", {"complete": False})
        output = io.StringIO()
        with patch.object(preflight, "run_preflight", side_effect=error) as run, redirect_stderr(output):
            self.assertEqual(cli.main(["--stage", "train"]), 1)
        run.assert_called_once_with("train", root=ROOT, save_report=False)
        self.assertIn("repair driver", output.getvalue())

    def test_cli_all_and_stage_are_mutually_exclusive(self):
        from contextlib import redirect_stderr
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            self.load_cli().main(["--all", "--stage", "prepare"])
        self.assertEqual(caught.exception.code, 2)


class CublasStartupTests(unittest.TestCase):
    def test_default_valid_overrides_and_invalid_values(self):
        with patch.dict(os.environ, {}):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            self.assertEqual(preflight.configure_cublas_workspace(), ":4096:8")
            for valid in (":4096:8", ":16:8"):
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = valid
                self.assertEqual(preflight.configure_cublas_workspace(), valid)
                self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], valid)
            for invalid in ("", "4096:8", ":4096:8 ", ":1:1"):
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = invalid
                with self.assertRaisesRegex(ValueError, "fresh process"):
                    preflight.configure_cublas_workspace()
                self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], invalid)

    def test_training_and_evaluation_configure_workspace_before_even_mocked_preflight(self):
        for name in ("31_train_step3a", "32_evaluate_step3a"):
            with self.subTest(cli=name), patch.dict(os.environ, {}):
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
                spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
                cli = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cli)

                def gate(**kwargs):
                    self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
                    raise RuntimeError("stopped before runtime import")

                with patch.object(cli, "run_preflight", side_effect=gate), \
                        patch.object(cli, "validate_dataset_paths") as dataset, \
                        patch.object(cli, "create_model") as model:
                    with self.assertRaisesRegex(RuntimeError, "stopped before runtime import"):
                        cli.main(["--mode", "baseline"])
                    dataset.assert_not_called()
                    model.assert_not_called()
if __name__ == "__main__":
    unittest.main()

"""CPU-only boundaries for the sampling-only Step 3A SALAD integration."""

import ast
from contextlib import redirect_stderr
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "countermine/training"
CLI_NAMES = (
    "29_check_step3a_runtime",
    "30_prepare_step3a_training", "31_train_step3a",
    "32_evaluate_step3a", "33_compare_step3a",
)
FORBIDDEN_IMPORT_PARTS = {
    "lightglue", "aliked", "diffusers", "iclight", "ic_light",
    "torch_geometric", "dgl", "networkx",
}


def project_sources():
    return sorted(TRAINING.glob("*.py")) + [
        ROOT / "tools" / (name + ".py") for name in CLI_NAMES
    ]


class Step3ABoundaryTests(unittest.TestCase):
    def test_original_salad_model_loss_and_miner_are_unchanged(self):
        salad = ROOT / "salad"
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=salad, capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout, "", "The SALAD submodule is read-only")
        expected = subprocess.run(
            ["git", "ls-files", "--stage", "salad"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.split()[1]
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=salad,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.assertEqual(actual, expected, "Do not change the SALAD submodule revision")
        for relative in (
            "main.py", "vpr_model.py", "utils/losses.py",
            "utils/validation.py", "dataloaders/GSVCitiesDataloader.py",
            "dataloaders/GSVCitiesDataset.py",
        ):
            with self.subTest(upstream_file=relative):
                original = subprocess.run(
                    ["git", "show", "HEAD:" + relative], cwd=salad,
                    capture_output=True, check=True,
                ).stdout
                self.assertEqual((salad / relative).read_bytes(), original)

    def test_training_has_no_local_matching_generation_or_graph_network_imports(self):
        for path in project_sources():
            with self.subTest(file=path.name):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        modules = [node.module or ""]
                    else:
                        continue
                    for module in modules:
                        self.assertFalse(
                            set(module.lower().split(".")) & FORBIDDEN_IMPORT_PARTS,
                            module,
                        )
                        self.assertNotIn("structural_matcher", module)
                        self.assertNotIn("banked_structural_matcher", module)
                        self.assertNotIn("local_feature_bank", module)

    def test_model_is_imported_without_training_or_loss_override(self):
        model_imports = []
        forbidden_definitions = {
            "training_step", "loss_function", "configure_optimizers",
            "optimizer_step", "compute_loss", "get_loss", "get_miner",
        }
        for path in project_sources():
            with self.subTest(file=path.name):
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.assertNotIn(node.name, forbidden_definitions)
                    if isinstance(node, ast.ClassDef):
                        for base in node.bases:
                            name = (base.id if isinstance(base, ast.Name) else
                                    base.attr if isinstance(base, ast.Attribute) else "")
                            self.assertNotEqual(name, "VPRModel")
                    if isinstance(node, ast.ImportFrom) and any(
                        item.name == "VPRModel" for item in node.names
                    ):
                        model_imports.append(node.module)
                    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        targets = (node.targets if isinstance(node, ast.Assign)
                                   else [node.target])
                        for target in targets:
                            for child in ast.walk(target):
                                if isinstance(child, ast.Attribute):
                                    self.assertNotIn(child.attr, {"loss_fn", "miner", "miner_outputs"})
        self.assertTrue(model_imports, "Instantiate the original upstream VPRModel")
        self.assertTrue(all(name == "vpr_model" for name in model_imports), model_imports)

    def test_transforms_and_validation_are_not_reimplemented(self):
        tree = ast.parse((TRAINING / "salad_datamodule.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called = (node.func.attr if isinstance(node.func, ast.Attribute)
                          else node.func.id if isinstance(node.func, ast.Name) else "")
                self.assertNotIn(called, {"Resize", "RandAugment", "ToTensor", "Normalize"})
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.assertNotIn(node.name, {"validation_step", "on_validation_epoch_end"})

    def test_all_cli_help_is_available_without_neural_runtime(self):
        # Running help in a fresh interpreter proves dependency imports happen after
        # argparse exits, and never instantiates a model or accesses the datasets.
        for name in CLI_NAMES:
            with self.subTest(cli=name):
                code = f'''
import importlib.abc, runpy, sys
class RejectModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0].lower() in {{
            'torch', 'torchvision', 'pytorch_lightning', 'lightning',
            'faiss', 'timm', 'vpr_model', 'dataloaders', 'models',
            'lightglue', 'aliked', 'diffusers', 'iclight', 'ic_light'
        }}:
            raise AssertionError('CLI help imported a model runtime: ' + fullname)
sys.meta_path.insert(0, RejectModels())
sys.argv = [{name!r}, '--help']
runpy.run_path('tools/{name}.py', run_name='__main__')
'''
                result = subprocess.run(
                    [sys.executable, "-c", code], cwd=ROOT,
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout.lower())
                if name == "29_check_step3a_runtime":
                    self.assertIn("--all", result.stdout)
                    self.assertIn("--stage", result.stdout)
                else:
                    self.assertIn("--seed", result.stdout)
                if name == "31_train_step3a":
                    for flag in ("--mode", "--max-epochs", "--limit-train-batches",
                                 "--limit-val-batches", "--smoke"):
                        self.assertIn(flag, result.stdout)

    def test_scientific_cli_rejects_integer_one_batch_limits(self):
        spec = importlib.util.spec_from_file_location(
            "step3a_training_cli_limits", ROOT / "tools/31_train_step3a.py",
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        for mode in ("baseline", "countermine_q99_geo500"):
            for flag in ("--limit-train-batches", "--limit-val-batches"):
                with self.subTest(mode=mode, limit=flag):
                    dataset_check = Mock(side_effect=RuntimeError("dataset check reached"))
                    with patch.object(cli, "validate_dataset_paths", dataset_check), redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as caught:
                            cli.main(["--mode", mode, flag, "1"])
                    self.assertEqual(caught.exception.code, 2)
                    dataset_check.assert_not_called()
        # Lightning interprets1 as a batch count and1.0 as the full fraction.
        # The supported full-fraction argument must pass the scientific CLI gate.
        for arguments in ([], ["--limit-train-batches", "1.0", "--limit-val-batches", "1.0"]):
            dataset_check = Mock(side_effect=RuntimeError("dataset check reached"))
            with patch.object(cli, "validate_dataset_paths", dataset_check), patch.object(
                    cli, "run_preflight", return_value={"runtime": {}}):
                with self.assertRaisesRegex(RuntimeError, "dataset check reached"):
                    cli.main(["--mode", "baseline", *arguments])
            dataset_check.assert_called_once()

    def test_cpu_configuration_import_does_not_load_model_runtimes(self):
        code = '''
import importlib.abc, sys
class RejectModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0].lower() in {
            'torch', 'torchvision', 'pytorch_lightning', 'lightning',
            'faiss', 'timm', 'vpr_model', 'dataloaders', 'models',
        }:
            raise AssertionError('Configuration imported a model runtime: ' + fullname)
sys.meta_path.insert(0, RejectModels())
import countermine.training
import countermine.training.step3a_config
'''
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT,
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

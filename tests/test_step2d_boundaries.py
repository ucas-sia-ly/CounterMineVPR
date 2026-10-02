"""Step 2D permits inference-bank work, but never training or generation."""

import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "full_structural_io", "full_population_metadata", "full_structural_population", "local_feature_bank",
    "banked_structural_matcher", "full_structural_measurement", "full_structural_calibration",
    "full_countermine_graph", "full_graph_analysis", "full_graph_visuals",
)
CLIS = (
    "17_build_full_structural_population", "18_build_aliked_feature_bank",
    "19_validate_aliked_bank", "20_measure_full_structural_pairs",
    "21_finalize_full_structural_metrics", "22_calibrate_full_structural_evidence",
    "23_build_full_countermine_graph", "24_analyze_full_countermine_graph",
    "25_visualize_full_structural_matches",
)
CPU_MODULES = (
    "full_structural_io", "full_population_metadata", "full_structural_population",
    "full_structural_measurement", "full_structural_calibration",
    "full_countermine_graph", "full_graph_analysis", "full_graph_visuals",
)
CPU_CLIS = (
    "17_build_full_structural_population", "21_finalize_full_structural_metrics",
    "22_calibrate_full_structural_evidence", "23_build_full_countermine_graph",
    "24_analyze_full_countermine_graph",
)


class FullMiningBoundaryTests(unittest.TestCase):
    def test_cpu_population_adapter_preserves_frozen_algorithms(self):
        def declarations(path):
            result = {}
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[node.name] = node
                elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    result[node.targets[0].id] = node
            return result
        originals = declarations(ROOT / "countermine/mining/candidate_miner.py")
        originals.update(declarations(ROOT / "countermine/mining/structural_population.py"))
        adapter = declarations(ROOT / "countermine/mining/full_population_metadata.py")
        self.assertIn("canonicalize_candidates", adapter)
        self.assertIn("load_population_inputs", adapter)
        for name, node in adapter.items():
            with self.subTest(declaration=name):
                self.assertIn(name, originals)
                self.assertEqual(ast.dump(node, include_attributes=False),
                                 ast.dump(originals[name], include_attributes=False))

    def test_historical_and_symlink_destinations_are_rejected(self):
        from countermine.mining.full_structural_io import guard_step2d
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "cache/countermine_rgb/step2d"
            runtime.mkdir(parents=True)
            audits = root / "docs/audits"
            audits.mkdir(parents=True)
            guard_step2d([
                runtime / "full_population.csv",
                root / "outputs/step2d/full_bottleneck_distribution.png",
                audits / "step2d_full_countermine_metrics.json",
            ], root)
            historical = root / "cache/countermine_rgb/step2a"
            historical.mkdir()
            (runtime / "old_source").symlink_to(historical, target_is_directory=True)
            (runtime / "old_file.csv").symlink_to(historical / "candidate_structural_metrics.csv")
            paths = [
                root / f"cache/countermine_rgb/step2{stage}/overwrite.csv"
                for stage in "abc"
            ]
            paths += [audits / f"step2{stage}_snapshot.json" for stage in "abc"]
            paths += [runtime / "old_source/overwrite.csv", runtime / "old_file.csv",
                      audits / "step2d_nested/metrics.json", runtime / "../step2a/escape.csv"]
            for path in paths:
                with self.subTest(path=str(path)):
                    with self.assertRaises(ValueError):
                        guard_step2d([path], root)

    def test_portable_metadata_rejects_nonfinite_and_absolute_paths(self):
        from countermine.mining.full_structural_io import validate_portable_json
        validate_portable_json({"feature_file": "features/000000.pt", "real_rgb_only": True,
                                "synthetic_images_used": False, "undefined": None})
        for payload in (
            {"nested": [float("nan")]}, {"nested": {"value": float("inf")}},
            {"feature_file": "/tmp/000000.pt"}, {"feature_file": "C:\\bank\\000000.pt"},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    validate_portable_json(payload)

    def test_cpu_analysis_imports_without_neural_or_generative_runtimes(self):
        source = f'''
import importlib, importlib.abc, importlib.util, sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0].lower()
        if root in {{'torch', 'torchvision', 'salad', 'lightglue', 'aliked', 'diffusers', 'transformers', 'iclight', 'ic_light'}} or fullname.startswith('countermine.training'):
            raise AssertionError('Forbidden CPU Step 2D import: ' + fullname)
sys.meta_path.insert(0, BlockModels())
for name in {CPU_MODULES!r}:
    importlib.import_module('countermine.mining.' + name)
for name in {CPU_CLIS!r}:
    spec = importlib.util.spec_from_file_location('full_cli_' + name, 'tools/' + name + '.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
assert 'torch' not in sys.modules
'''
        result = subprocess.run(
            [sys.executable, "-c", source], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_no_generation_training_or_registration_implementation(self):
        forbidden_roots = {
            "diffusers", "transformers", "iclight", "ic_light", "salad",
            "adaptvpr", "accelerate",
        }
        forbidden_calls = {
            "backward", "training_step", "compute_loss", "optimizer", "zero_grad",
            "findHomography", "findFundamentalMat", "findEssentialMat",
            "estimateAffine2D", "estimateAffinePartial2D", "warpPerspective",
        }
        paths = [ROOT / "countermine/mining" / (name + ".py") for name in MODULES]
        paths.extend(ROOT / "tools" / (name + ".py") for name in CLIS)
        for path in paths:
            with self.subTest(path=path.name):
                source = path.read_text()
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        modules = [node.module or ""]
                    else:
                        modules = []
                    for module in modules:
                        self.assertNotIn(module.split(".")[0].lower(), forbidden_roots)
                        self.assertFalse(
                            set(module.lower().split(".")) & {"training", "optim", "losses", "sampler", "samplers"},
                            module,
                        )
                        self.assertNotEqual(module, "countermine.mining.salad_encoder")
                    if isinstance(node, ast.Call):
                        called = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                        self.assertNotIn(called, forbidden_calls)
                        self.assertFalse(called.lower().endswith("loss"), called)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.assertNotIn(node.name, forbidden_calls)
                        self.assertFalse(node.name.lower().endswith("loss"), node.name)


if __name__ == "__main__":
    unittest.main()

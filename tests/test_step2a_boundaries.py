"""Static research boundaries; no neural models or GPU are loaded."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MODULES = (
    "countermine/mining/structural_population.py",
    "countermine/mining/structural_matcher.py",
    "countermine/mining/structural_analysis.py",
    "tools/07_build_structural_population.py",
    "tools/08_measure_structural_pairs.py",
    "tools/09_analyze_structural_audit.py",
    "tools/10_visualize_structural_matches.py",
)


class Step2ABoundaryTests(unittest.TestCase):
    def test_active_code_has_no_generator_training_or_registration_calls(self):
        forbidden_modules = {"diffusers", "transformers", "ic_light", "iclight"}
        forbidden_calls = {
            "backward", "optimizer", "training_step", "compute_loss",
            "findHomography", "findFundamentalMat", "estimateAffine2D",
            "estimateAffinePartial2D", "recoverPose",
        }
        forbidden_names = {"R2", "R4", "R8", "R16", "r2", "r4", "r8", "r16"}
        for relative in ACTIVE_MODULES:
            with self.subTest(file=relative):
                source = (ROOT / relative).read_text()
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules = [item.name.lower() for item in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        modules = [(node.module or "").lower()]
                    else:
                        modules = []
                    for module in modules:
                        parts = set(module.split("."))
                        self.assertFalse(parts & forbidden_modules, module)
                        self.assertFalse(parts & {"training", "optim", "losses"}, module)
                    if isinstance(node, ast.Call):
                        name = (node.func.attr if isinstance(node.func, ast.Attribute)
                                else node.func.id if isinstance(node.func, ast.Name) else "")
                        self.assertNotIn(name, forbidden_calls)
                        self.assertFalse(name.lower().endswith("loss"), name)
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        self.assertNotIn(node.name, forbidden_names)
                    if isinstance(node, ast.Name):
                        self.assertNotIn(node.id, forbidden_names)
                self.assertNotIn("third_party/IC-Light", source)

    def test_cpu_analysis_imports_no_neural_runtime(self):
        code = """
import importlib.util, sys
from countermine.mining import structural_analysis
spec = importlib.util.spec_from_file_location('audit_cli', 'tools/09_analyze_structural_audit.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for prefix in ('torch', 'lightglue', 'diffusers', 'salad'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_environment_is_mining_only(self):
        source = (ROOT / "environment_mining.yml").read_text().lower()
        for required in ("name: countermine-mining", "python=3.10", "pytorch=2.1.0",
                         "torchvision=0.16.0", "pytorch-cuda=12.1", "numpy=1.26.4"):
            self.assertIn(required, source)
        for forbidden in ("diffusers", "transformers", "ic-light", "lightning",
                          "metric-learning"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()

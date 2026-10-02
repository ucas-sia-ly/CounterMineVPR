"""Import and static boundaries for the CPU-only graph pilot."""

import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = (
    "countermine/mining/structural_calibration.py",
    "countermine/mining/countermine_graph.py",
    "countermine/mining/graph_analysis.py",
    "countermine/mining/graph_visuals.py",
    "tools/11_calibrate_structural_evidence.py",
    "tools/12_build_countermine_graph.py",
    "tools/13_analyze_countermine_graph.py",
)


class GraphBoundaryTests(unittest.TestCase):
    def test_cpu_modules_and_clis_import_with_neural_runtimes_blocked(self):
        source = '''
import importlib, importlib.abc, importlib.util, sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0].lower()
        if root in {'torch','torchvision','salad','lightglue','aliked','diffusers','transformers','iclight','ic_light','networkx'} or fullname in {'countermine.mining.salad_encoder','countermine.mining.structural_matcher','countermine.mining.candidate_miner'} or fullname.startswith('countermine.training'):
            raise AssertionError('Forbidden graph runtime import: ' + fullname)
sys.meta_path.insert(0, BlockModels())
for name in ('structural_calibration','countermine_graph','graph_analysis','graph_visuals'):
    importlib.import_module('countermine.mining.'+name)
for path in ('tools/11_calibrate_structural_evidence.py','tools/12_build_countermine_graph.py','tools/13_analyze_countermine_graph.py'):
    spec=importlib.util.spec_from_file_location('graph_cli',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", source], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_no_model_training_registration_or_matching_api(self):
        forbidden_roots = {"torch", "torchvision", "salad", "lightglue", "aliked", "diffusers", "transformers", "iclight", "ic_light", "networkx"}
        forbidden_calls = {"backward", "training_step", "compute_loss", "optimizer", "match_pair", "findHomography", "findFundamentalMat"}
        for name in ACTIVE:
            with self.subTest(path=name):
                source = (ROOT / name).read_text()
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        imports = [node.module or ""]
                    else:
                        imports = []
                    for module in imports:
                        self.assertNotIn(module.split('.')[0].lower(), forbidden_roots)
                        self.assertFalse(set(module.lower().split('.')) & {"training", "optim", "losses"})
                        self.assertNotIn(module, {"countermine.mining.structural_matcher", "countermine.mining.salad_encoder", "countermine.mining.candidate_miner"})
                    if isinstance(node, ast.Call):
                        call = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                        self.assertNotIn(call, forbidden_calls)
                        self.assertFalse(call.lower().endswith("loss"))
                self.assertNotIn("08_measure_structural_pairs", source)
                self.assertNotIn("10_visualize_structural_matches", source)


if __name__ == "__main__":
    unittest.main()

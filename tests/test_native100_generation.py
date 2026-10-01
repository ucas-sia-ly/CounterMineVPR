"""CPU-only native100 generation checks using a preloaded fake IC-Light adapter."""

from contextlib import redirect_stdout
import csv
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from PIL import Image

from countermine.probe.geometry_audit import reference, sha256
from countermine.probe.native100_audit import MANIFEST_COLUMNS, NATIVE_SIZE, POLICY


ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("_test_native100_generator", ROOT / "tools/20_generate_native100_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path, rows):
    columns = tuple(dict.fromkeys((*rows[0].keys(), *MANIFEST_COLUMNS)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class _FakeAdapter:
    def __init__(self, config, calls, *, invalid_size=None, fail_at=None, bad_runtime=False):
        self.config = config
        self.calls = calls
        self.loaded = False
        self.rmbg = None
        self.device = None
        self.last_run_stats = {}
        self.invalid_size = invalid_size
        self.fail_at = fail_at
        self.bad_runtime = bad_runtime

    def _ensure_models(self):
        self.loaded = True
        self.calls.append("preload")

    def relight(self, source, mode):
        if not self.loaded or mode != "full_scene" or source.size != NATIVE_SIZE or source.mode != "RGB":
            raise AssertionError("Only preloaded full_scene at native RGB geometry is authorized")
        count = len(self.calls)
        if self.fail_at == count:
            raise RuntimeError("fake late image failure")
        self.calls.append((source.size, mode, asdict(self.config)))
        self.last_run_stats = {
            "elapsed_seconds": float("nan") if self.bad_runtime else count / 1000,
            "peak_cuda_memory_allocated_bytes": None,
            "peak_cuda_memory_reserved_bytes": None,
        }
        return Image.new("RGB", self.invalid_size) if self.invalid_size else source.copy()


class Native100GenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.fixture = Path(cls.temporary.name)
        cls.source = cls.fixture / "source"
        cls.original = cls.fixture / "dataset"
        cls.source.mkdir()
        cls.original.mkdir()
        cls.population = cls.fixture / "audit_manifest.csv"
        cls.population_summary = cls.fixture / "audit_summary.json"
        cls.frozen = cls.fixture / "step2d1.json"
        cls.frozen.write_bytes((ROOT / "docs/audits/step2d1_native_fov_metrics.json").read_bytes())
        cls.population_rows, cls.rows = [], []
        for index in range(100):
            image_id = f"Images/place/source_{index}.jpg"
            original = cls.original / image_id
            original.parent.mkdir(parents=True, exist_ok=True)
            source = cls.source / f"{index:08d}.png"
            image = Image.new("RGB", NATIVE_SIZE, (index, 100, 180))
            image.save(original, format="JPEG")
            image.close()
            with Image.open(original) as decoded:
                decoded.convert("RGB").save(source, format="PNG")
            old = {"audit_index": index, "row_index": index, "image_id": image_id,
                   "group": "random" if index < 50 else "hard_candidate", "relative_path": image_id,
                   "source_512_path": f"cache/old/source_{index}.png", "original_width": "640", "original_height": "480"}
            cls.population_rows.append(old)
            cls.rows.append({**old, "policy": POLICY, "source_path": reference(source),
                             "original_path": reference(original), "canonical_width": 640, "canonical_height": 480,
                             "scale_x": 1.0, "scale_y": 1.0, "retained_area_fraction": 1.0,
                             "retained_long_axis_fraction": 1.0, "original_format": "JPEG", "source_format": "PNG",
                             "original_sha256": sha256(original), "source_sha256": sha256(source)})
        with cls.population.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=cls.population_rows[0])
            writer.writeheader()
            writer.writerows(cls.population_rows)
        cls.population_summary.write_text(json.dumps({"number_of_images": 100, "count_random": 50,
            "count_hard_candidate": 50, "seed": 42, "config": {"seed": 42},
            "manifest_reference": "cache/old/mini.csv", "manifest_sha256": "a" * 64,
            "rgb_candidate_reference": "cache/old/candidates.csv", "rgb_candidate_sha256": "b" * 64}))
        cls.tool = _tool()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "native"
        self.output.mkdir()
        self.manifest = self.output / "native100_manifest.csv"
        _write_csv(self.manifest, self.rows)
        self.build_summary = self.output / "native100_build_summary.json"
        self.build_summary.write_text(json.dumps({"number_of_sources": 100, "manifest_sha256": sha256(self.manifest),
            "source_population_provenance": {"number_of_sources": 100,
                "selection": "exact existing Step 2 generator-audit population; no resampling",
                "audit_manifest_sha256": sha256(self.population), "audit_summary_sha256": sha256(self.population_summary)}}))

    def _generate(self, **kwargs):
        calls = []
        with redirect_stdout(io.StringIO()):
            result = self.tool.run_generation(self.manifest, self.frozen, self.output, device="cpu",
                population_manifest_path=self.population, population_summary_path=self.population_summary,
                adapter_factory=lambda config: _FakeAdapter(config, calls, **kwargs))
        return result, calls

    def test_all_100_frozen_sources_are_generated_after_model_preload(self):
        result, calls = self._generate()
        self.assertEqual(calls[0], "preload")
        self.assertEqual(len(calls), 101)
        frozen = json.loads(self.frozen.read_text())["provenance"]["iclight_config"]
        for size, mode, actual in calls[1:]:
            self.assertEqual(size, NATIVE_SIZE)
            self.assertEqual(mode, "full_scene")
            self.assertEqual(actual["device"], "cpu")
            for name, value in frozen.items():
                self.assertEqual(actual[name], value, name)
        self.assertEqual(result["number_of_sources"], 100)
        self.assertEqual(result["count_outputs"], 100)
        self.assertFalse(result["rmbg_used"])
        self.assertTrue(result["two_stage_inference"])
        self.assertGreaterEqual(result["model_load_elapsed_seconds"], 0)
        self.assertAlmostEqual(result["generation_elapsed_seconds"], 5.05)
        expected = {"median": .0505, "q05": .00595, "q25": .02575, "q75": .07525, "q95": .09505}
        for name, value in expected.items():
            self.assertAlmostEqual(result["generation_runtime"]["elapsed_seconds"][name], value)
        self.assertIsNone(result["generation_runtime"]["peak_cuda_memory_allocated_bytes"])
        self.assertEqual(len(list((self.output / "relit").glob("*.png"))), 100)
        self.assertEqual([run["audit_index"] for run in result["runs"]], list(range(100)))
        from countermine.probe.native100_audit import load_native100_manifest
        rows = load_native100_manifest(self.manifest, self.population, self.population_summary)
        outputs, metadata = self.tool.load_native100_generated_pairs(rows,
            self.output / "native100_generation.csv", self.output / "native100_generation_summary.json",
            self.manifest, self.frozen)
        self.assertEqual(len(outputs), 100)
        self.assertEqual(metadata, result)

    def test_incomplete_or_non_native_manifest_is_rejected_before_model_construction(self):
        for mutation in ("missing", "scale", "dimensions"):
            with self.subTest(mutation=mutation):
                rows = [dict(row) for row in self.rows]
                if mutation == "missing":
                    rows.pop()
                elif mutation == "scale":
                    rows[0]["scale_x"] = .9
                else:
                    rows[0]["canonical_height"] = 512
                _write_csv(self.manifest, rows)
                factory = mock.Mock(side_effect=AssertionError("no model construction permitted"))
                with self.assertRaises(ValueError):
                    self.tool.run_generation(self.manifest, self.frozen, self.output, device="cpu",
                        population_manifest_path=self.population, population_summary_path=self.population_summary,
                        adapter_factory=factory)
                factory.assert_not_called()

    def test_incorrect_step2d1_settings_are_rejected_without_inference(self):
        path = self.output / "mutated_step2d1.json"
        for name, value in (("cfg", 3.0), ("steps", 24), ("width", 512)):
            with self.subTest(name=name):
                data = json.loads(self.frozen.read_text())
                data["provenance"]["iclight_config"][name] = value
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    self.tool.read_frozen_generation_settings(path)
        data = json.loads(self.frozen.read_text())
        data["provenance"]["two_stage_inference"] = False
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.tool.read_frozen_generation_settings(path)

    def test_late_failure_wrong_geometry_and_nan_preserve_previous_outputs(self):
        self._generate()
        before = {str(path.relative_to(self.output)): path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
        for options in ({"fail_at": 100}, {"invalid_size": (640, 512)}, {"bad_runtime": True}):
            with self.subTest(options=options):
                with self.assertRaises((RuntimeError, ValueError)):
                    self._generate(**options)
                after = {str(path.relative_to(self.output)): path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
                self.assertEqual(after, before)

    def test_import_and_help_load_no_torch_diffusers_or_models(self):
        script = ("import importlib.util,sys; s=importlib.util.spec_from_file_location('gen'," +
            repr(str(ROOT / "tools/20_generate_native100_audit.py")) +
            "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert not any(n in sys.modules for n in ('torch','diffusers','transformers')); "
            "m.read_frozen_generation_settings(" + repr(str(self.frozen)) + "); "
            "assert not any(n in sys.modules for n in ('torch','diffusers','transformers'))")
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

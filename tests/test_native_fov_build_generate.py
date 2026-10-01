"""CPU-only native source/provenance and fake-generation pipeline checks."""

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

from PIL import Image, ImageDraw

from countermine.probe.geometry_audit import sha256
from countermine.probe.iclight_adapter import ICLightConfig
from countermine.probe.native_fov_audit import MANIFEST_COLUMNS, NATIVE_SIZE, POLICY


REPO_ROOT = Path(__file__).resolve().parents[1]


def _tool(filename):
    spec = importlib.util.spec_from_file_location("_native_test_" + Path(filename).stem,
                                                 REPO_ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _files(path):
    return {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()}


class _NativeFakeAdapter:
    def __init__(self, config, calls, *, fail_at=None, invalid_size=None, elapsed=.001):
        self.config = config
        self.calls = calls
        self.fail_at = fail_at
        self.invalid_size = invalid_size
        self.elapsed = elapsed
        self.last_run_stats = {}

    def relight(self, source, mode):
        if mode != "full_scene" or source.size != NATIVE_SIZE or source.mode != "RGB":
            raise AssertionError("Only exact native full-scene generation is authorized")
        self.calls.append((source.size, mode, asdict(self.config)))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("late fake native inference failure")
        self.last_run_stats = {"elapsed_seconds": self.elapsed,
                               "peak_cuda_memory_allocated_bytes": None,
                               "peak_cuda_memory_reserved_bytes": None}
        return Image.new("RGB", self.invalid_size) if self.invalid_size else source.copy()


class NativeBuildGenerateTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        self.output = self.root / "native"
        self.audit = self.root / "audit_manifest.csv"
        self.step2c = self.root / "step2c_metrics.json"
        self.historical = self.root / "step2d0_metrics.json"
        self.originals, identities, audit_rows = [], [], []
        for index in range(10):
            row_index = 100 + 7 * index
            image_id = f"Images/place/source_{index:02d}.jpg"
            path = self.dataset / image_id
            path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", NATIVE_SIZE, (40 + index, 100, 180))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 63, 479), fill="red")
            draw.rectangle((576, 0, 639, 479), fill="green")
            exif = image.getexif()
            exif[274] = 6
            image.save(path, format="JPEG", quality=95, exif=exif)
            image.close()
            self.originals.append(path)
            identities.append({"audit_index": index, "row_index": row_index, "image_id": image_id,
                               "policies": {"square_crop_512": {}, "full_fov_512": {}}})
            audit_rows.append({"audit_index": index, "row_index": row_index,
                               "image_id": image_id, "relative_path": image_id})
        audit_rows.append({"audit_index": 99, "row_index": 99999,
                           "image_id": "not_selected.jpg", "relative_path": "not_selected.jpg"})
        _write_csv(self.audit, ("audit_index", "row_index", "image_id", "relative_path"),
                   list(reversed(audit_rows)))
        config = asdict(ICLightConfig())
        self.step2c.write_text(json.dumps({"provenance": {"iclight_seed": 12345,
                                                          "prompt": config["prompt"],
                                                          "iclight_config": config}}))
        self.historical.write_text(json.dumps({"provenance": {
            "iclight_seed": 12345, "prompt": config["prompt"],
            "frozen_step2c_snapshot_sha256": sha256(self.step2c)}, "per_source": identities}))
        self.build_tool = _tool("15_build_native_fov_audit.py")
        self.generate_tool = _tool("16_generate_native_fov_audit.py")

    def _build(self, output=None):
        return self.build_tool.build_native_fov_audit(self.historical, self.audit, self.dataset,
                                                     output or self.output)

    def _generate(self, *, fail_at=None, invalid_size=None, elapsed=.001):
        calls, configs = [], []

        def factory(config):
            configs.append(config)
            return _NativeFakeAdapter(config, calls, fail_at=fail_at,
                                      invalid_size=invalid_size, elapsed=elapsed)

        printed = io.StringIO()
        with redirect_stdout(printed):
            summary = self.generate_tool.run_generation(
                self.output / "native_manifest.csv", self.step2c, self.historical,
                self.output, device="cpu", adapter_factory=factory,
            )
        return summary, calls, configs, printed.getvalue()

    def _assert_preflight_failure(self):
        factory = mock.Mock(side_effect=AssertionError("No models may be constructed"))
        with self.assertRaises(ValueError):
            self.generate_tool.run_generation(self.output / "native_manifest.csv", self.step2c,
                                               self.historical, self.output, device="cpu", adapter_factory=factory)
        factory.assert_not_called()

    def test_builder_uses_exact_frozen_jpeg_pixels_geometry_and_identities(self):
        historical_bytes, step2c_bytes = self.historical.read_bytes(), self.step2c.read_bytes()
        summary = self._build()
        rows = _rows(self.output / "native_manifest.csv")
        self.assertEqual(len(rows), 10)
        self.assertEqual([int(row["row_index"]) for row in rows], [100 + 7 * i for i in range(10)])
        self.assertEqual({row["policy"] for row in rows}, {POLICY})
        for row, original_path in zip(rows, self.originals):
            self.assertEqual((int(row["original_width"]), int(row["original_height"])), NATIVE_SIZE)
            self.assertEqual((int(row["canonical_width"]), int(row["canonical_height"])), NATIVE_SIZE)
            for name in ("scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction"):
                self.assertEqual(float(row[name]), 1.0)
            self.assertEqual(row["original_format"], "JPEG")
            self.assertEqual(row["source_format"], "PNG")
            with Image.open(original_path) as original, Image.open(row["source_path"]) as source:
                self.assertEqual(source.size, NATIVE_SIZE)
                self.assertEqual(source.format, "PNG")
                self.assertEqual(source.mode, "RGB")
                self.assertEqual(source.tobytes(), original.convert("RGB").tobytes())
            self.assertEqual(row["original_sha256"], sha256(original_path))
            self.assertEqual(row["source_sha256"], sha256(Path(row["source_path"])))
        self.assertEqual(summary["number_of_sources"], 10)
        self.assertEqual(summary["historical_snapshot_sha256"], sha256(self.historical))
        self.assertEqual(summary["config"]["seed"], 42)
        self.assertEqual(summary["manifest_sha256"], sha256(self.output / "native_manifest.csv"))
        self.assertEqual(self.historical.read_bytes(), historical_bytes)
        self.assertEqual(self.step2c.read_bytes(), step2c_bytes)
        self.assertEqual(json.loads((self.output / "native_build_summary.json").read_text()), summary)

    def test_unexpected_original_geometry_preserves_previous_sources(self):
        self._build()
        before = _files(self.output)
        Image.new("RGB", (640, 481)).save(self.originals[-1], format="JPEG")
        with self.assertRaisesRegex(ValueError, "640x480"):
            self._build()
        self.assertEqual(_files(self.output), before)

    def test_missing_duplicate_or_wrong_frozen_identity_is_rejected(self):
        original = self.historical.read_bytes()
        for mutation in ("missing", "duplicate", "wrong_image_id"):
            with self.subTest(mutation=mutation):
                snapshot = json.loads(original)
                if mutation == "missing":
                    snapshot["per_source"].pop()
                elif mutation == "duplicate":
                    snapshot["per_source"][-1] = dict(snapshot["per_source"][0])
                else:
                    snapshot["per_source"][0]["image_id"] = "wrong.jpg"
                self.historical.write_text(json.dumps(snapshot))
                with self.assertRaises(ValueError):
                    self._build()
                self.assertFalse(self.output.exists())

    def test_builder_rejects_frozen_or_dataset_destinations(self):
        for output in (REPO_ROOT / "docs/audits", REPO_ROOT / "cache/geometry_audit",
                       self.dataset / "new_native_outputs"):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    self._build(output)

    def test_generator_runs_only_ten_native_full_scene_pairs_with_frozen_settings(self):
        self._build()
        summary, calls, configs, printed = self._generate()
        self.assertEqual(len(calls), 10)
        self.assertEqual(len(configs), 1)
        frozen = json.loads(self.step2c.read_text())["provenance"]["iclight_config"]
        for size, mode, config in calls:
            self.assertEqual(size, NATIVE_SIZE)
            self.assertEqual(mode, "full_scene")
            self.assertEqual((config["width"], config["height"]), NATIVE_SIZE)
            self.assertEqual(config["device"], "cpu")
            for key, value in frozen.items():
                if key not in ("width", "height", "device"):
                    self.assertEqual(config[key], value, key)
        self.assertEqual(printed.count("elapsed="), 10)
        self.assertEqual(printed.count("peak CUDA allocated="), 10)
        records = _rows(self.output / "native_generation.csv")
        self.assertEqual(len(records), 10)
        for row in records:
            self.assertEqual(row["mode"], "full_scene")
            self.assertEqual(row["policy"], POLICY)
            self.assertEqual(int(row["seed"]), 12345)
            with Image.open(row["output_path"]) as output:
                self.assertEqual(output.size, NATIVE_SIZE)
                self.assertEqual(output.format, "PNG")
                self.assertEqual(output.mode, "RGB")
        self.assertEqual(summary["count_outputs"], 10)
        self.assertEqual(summary["step2c_snapshot_sha256"], sha256(self.step2c))
        self.assertEqual(summary["historical_snapshot_sha256"], sha256(self.historical))
        self.assertEqual(summary["config"]["iclight_config"], asdict(configs[0]))
        self.assertEqual(json.loads((self.output / "native_generation_summary.json").read_text()), summary)

    def test_missing_or_non_native_manifest_is_rejected_before_model_loading(self):
        self._build()
        path = self.output / "native_manifest.csv"
        original = _rows(path)
        for mutation in ("missing", "scale", "geometry"):
            with self.subTest(mutation=mutation):
                rows = [dict(row) for row in original]
                if mutation == "missing":
                    rows.pop()
                elif mutation == "scale":
                    rows[0]["scale_x"] = ".8"
                else:
                    rows[0]["canonical_height"] = "512"
                _write_csv(path, MANIFEST_COLUMNS, rows)
                self._assert_preflight_failure()

    def test_changed_png_pixels_are_rejected_even_if_manifest_hash_is_updated(self):
        self._build()
        path = self.output / "native_manifest.csv"
        rows = _rows(path)
        source = Path(rows[0]["source_path"])
        Image.new("RGB", NATIVE_SIZE, "purple").save(source, format="PNG")
        rows[0]["source_sha256"] = sha256(source)
        _write_csv(path, MANIFEST_COLUMNS, rows)
        self._assert_preflight_failure()

    def test_invalid_png_encoding_is_rejected_before_model_loading(self):
        self._build()
        path = self.output / "native_manifest.csv"
        rows = _rows(path)
        source = Path(rows[0]["source_path"])
        Image.new("RGB", NATIVE_SIZE).save(source, format="JPEG")
        rows[0]["source_sha256"] = sha256(source)
        _write_csv(path, MANIFEST_COLUMNS, rows)
        self._assert_preflight_failure()

    def test_missing_or_changed_fixed_generation_config_is_rejected_before_models(self):
        self._build()
        step2c_original, historical_original = self.step2c.read_bytes(), self.historical.read_bytes()
        for mutation in ("missing_steps", "cfg", "unlinked_history"):
            with self.subTest(mutation=mutation):
                self.step2c.write_bytes(step2c_original)
                historical = json.loads(historical_original)
                data = json.loads(step2c_original)
                if mutation == "missing_steps":
                    data["provenance"]["iclight_config"].pop("steps")
                elif mutation == "cfg":
                    data["provenance"]["iclight_config"]["cfg"] = 3.0
                self.step2c.write_text(json.dumps(data))
                historical["provenance"]["frozen_step2c_snapshot_sha256"] = (
                    "0" * 64 if mutation == "unlinked_history" else sha256(self.step2c))
                self.historical.write_text(json.dumps(historical))
                self._assert_preflight_failure()

    def test_late_generation_failure_preserves_completed_outputs(self):
        self._build()
        self._generate()
        before = _files(self.output)
        with self.assertRaisesRegex(RuntimeError, "late fake native inference failure"):
            self._generate(fail_at=10)
        self.assertEqual(_files(self.output), before)

    def test_wrong_final_geometry_and_nonfinite_runtime_preserve_prior_outputs(self):
        self._build()
        self._generate()
        before = _files(self.output)
        for kwargs in ({"invalid_size": (640, 512)}, {"elapsed": float("nan")}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self._generate(**kwargs)
                self.assertEqual(_files(self.output), before)


class NativeCLIHelpTest(unittest.TestCase):
    def test_help_does_not_import_ml_or_cuda_dependencies(self):
        script = """
import builtins, runpy, sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'diffusers', 'transformers', 'safetensors',
                             'gradio', 'salad', '_countermine_vendored_lightglue'}:
        raise AssertionError('Unexpected ML dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
path = sys.argv[1]
sys.argv = [path, '--help']
runpy.run_path(path, run_name='__main__')
"""
        for filename in ("15_build_native_fov_audit.py", "16_generate_native_fov_audit.py"):
            with self.subTest(filename=filename):
                result = subprocess.run([sys.executable, "-c", script, str(REPO_ROOT / "tools" / filename)],
                                        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()

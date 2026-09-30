"""CPU-only checks for the ten-source IC-Light audit runner.

Every inference call uses a fake adapter. These tests never load model weights,
download models, or initialize CUDA.
"""

import csv
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from countermine.probe.iclight_adapter import ICLightConfig


SMOKE_COLUMNS = [
    "audit_index", "row_index", "mode", "source_512_path", "output_path",
    "seed", "prompt", "width", "height", "steps", "cfg", "highres_scale",
    "elapsed_seconds",
]
AUDIT_COLUMNS = [
    "audit_index", "group", "row_index", "image_id", "relative_path",
    "place_uid", "city_id", "source_512_path", "original_width",
    "original_height", "crop_left", "crop_top", "crop_size",
]
MODES = ("official_rmbg", "full_scene")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeAdapter:
    """A CPU replacement exposing only the runner's adapter interface."""

    def __init__(self, config, calls, fail_on_call=None, bad_output=None):
        self.config = config
        self.calls = calls
        self.fail_on_call = fail_on_call
        self.bad_output = bad_output
        self.last_run_stats = {}

    def relight(self, image, mode):
        self.calls.append({
            "mode": mode,
            "size": image.size,
            "image_mode": image.mode,
            "source_bytes": image.tobytes(),
        })
        self.last_run_stats = {"peak_cuda_memory_bytes": None}
        if len(self.calls) == self.fail_on_call:
            return self.bad_output
        # Distinct colors make accidental reuse of one mode's output detectable.
        tint = (130, 145, 160) if mode == "official_rmbg" else (150, 160, 170)
        return Image.new("RGB", (512, 512), tint)


class ICLightSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = importlib.import_module("tools.08_iclight_audit_smoke")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output_dir = self.root / "generator_audit"
        self.sources_dir = self.output_dir / "source_512"
        self.sources_dir.mkdir(parents=True)
        self.manifest_path = self.output_dir / "audit_manifest.csv"
        self.contact_sheet_path = self.root / "outputs/step2/iclight_smoke_10.jpg"
        self.rows = []
        for audit_index in range(12):
            row_index = 1200 + audit_index * 7
            source_path = self.sources_dir / f"{row_index:08d}.jpg"
            with Image.new("RGB", (512, 512),
                           (audit_index * 10, audit_index * 7, audit_index * 3)) as image:
                image.save(source_path)
            self.rows.append({
                "audit_index": audit_index,
                "group": "random" if audit_index < 6 else "hard_candidate",
                "row_index": row_index,
                "image_id": f"image-{row_index}",
                "relative_path": f"Images/TestCity/{row_index}.jpg",
                "place_uid": f"TestCity:{audit_index:07d}",
                "city_id": "TestCity",
                "source_512_path": os.path.relpath(source_path, Path.cwd()),
                "original_width": 800,
                "original_height": 600,
                "crop_left": 100,
                "crop_top": 0,
                "crop_size": 600,
            })
        self.write_manifest()
        self.calls = []
        self.factory_configs = []

    def write_manifest(self, rows=None):
        with self.manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows if rows is None else rows)

    def adapter_factory(self, config):
        self.factory_configs.append(config)
        return FakeAdapter(config, self.calls)

    def run_smoke(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return self.runner.run_smoke(
                self.manifest_path, self.output_dir, self.contact_sheet_path,
                adapter_factory=kwargs.pop("adapter_factory", self.adapter_factory),
                **kwargs,
            )

    def published_hashes(self):
        paths = list((self.output_dir / "relit").rglob("*"))
        paths.extend([
            self.output_dir / "iclight_smoke.csv",
            self.output_dir / "iclight_smoke_summary.json",
            self.contact_sheet_path,
        ])
        return {path: sha256(path) for path in paths if path.is_file()}

    def test_selection_is_exactly_first_ten_and_deterministic(self):
        selected = self.runner.load_smoke_sources(self.manifest_path)
        self.assertEqual(selected,
                         self.runner.load_smoke_sources(self.manifest_path))
        self.assertEqual(len(selected), 10)
        self.assertEqual([int(row["audit_index"]) for row in selected], list(range(10)))
        self.assertEqual([int(row["row_index"]) for row in selected],
                         [row["row_index"] for row in self.rows[:10]])
        self.assertEqual(len({int(row["row_index"]) for row in selected}), 10)
        self.assertEqual(len({row["image_id"] for row in selected}), 10)
        self.assertEqual(
            [int(row["audit_index"]) for row in
             self.runner.load_smoke_sources(self.manifest_path, count=3)],
            [0, 1, 2],
        )

    def test_smoke_writes_both_modes_exact_csv_config_and_readable_contact_sheet(self):
        source_hashes = {path: sha256(path) for path in self.sources_dir.glob("*.jpg")}
        config = ICLightConfig()
        report = self.run_smoke(config=config)
        self.assertEqual(report["number_of_sources"], 10)
        self.assertEqual(report["count_outputs"], 20)
        # JSON-normalize to allow dataclass Path fields while checking all settings.
        self.assertEqual(report["config"],
                         json.loads(json.dumps(asdict(config), default=str)))
        self.assertEqual(len(self.factory_configs), 1)
        self.assertEqual(self.factory_configs[0], config)
        self.assertEqual(len(self.calls), 20)
        self.assertEqual([call["mode"] for call in self.calls], list(MODES) * 10)
        for call in self.calls:
            self.assertEqual(call["size"], (512, 512))
            self.assertEqual(call["image_mode"], "RGB")
        for index in range(10):
            self.assertEqual(self.calls[index * 2]["source_bytes"],
                             self.calls[index * 2 + 1]["source_bytes"])

        csv_path = self.output_dir / "iclight_smoke.csv"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, SMOKE_COLUMNS)
            actual = list(reader)
        self.assertEqual(len(actual), 20)
        self.assertEqual([int(row["audit_index"]) for row in actual],
                         [index for index in range(10) for _ in MODES])
        self.assertEqual([row["mode"] for row in actual], list(MODES) * 10)
        for row in actual:
            audit_index = int(row["audit_index"])
            source_row = self.rows[audit_index]
            self.assertEqual(int(row["row_index"]), source_row["row_index"])
            self.assertEqual(row["source_512_path"], source_row["source_512_path"])
            output_path = self.output_dir / "relit" / row["mode"] / (
                f'{source_row["row_index"]:08d}.jpg'
            )
            self.assertFalse(Path(row["output_path"]).is_absolute())
            self.assertEqual(row["output_path"], os.path.relpath(output_path, Path.cwd()))
            self.assertEqual(int(row["seed"]), 12345)
            self.assertEqual(row["prompt"], config.prompt)
            self.assertEqual(int(row["width"]), 512)
            self.assertEqual(int(row["height"]), 512)
            self.assertEqual(int(row["steps"]), 25)
            self.assertEqual(float(row["cfg"]), 2.0)
            self.assertEqual(float(row["highres_scale"]), 1.0)
            self.assertGreaterEqual(float(row["elapsed_seconds"]), 0)
            with Image.open(output_path) as image:
                image.verify()
            with Image.open(output_path) as image:
                image.load()
                self.assertEqual(image.size, (512, 512))
                self.assertEqual(image.mode, "RGB")
                expected = ((130, 145, 160) if row["mode"] == "official_rmbg"
                            else (150, 160, 170))
                self.assertTrue(all(abs(actual - reference) <= 2
                                    for actual, reference in
                                    zip(image.getpixel((256, 256)), expected)))

        self.assertEqual(source_hashes, {path: sha256(path) for path in source_hashes})
        with Image.open(self.contact_sheet_path) as contact_sheet:
            contact_sheet.verify()
        with Image.open(self.contact_sheet_path) as contact_sheet:
            contact_sheet.load()
            self.assertEqual(contact_sheet.mode, "RGB")
            self.assertGreaterEqual(contact_sheet.width, 3 * 256)
            self.assertGreaterEqual(contact_sheet.height, 10 * 256)
        saved_report = json.loads(
            (self.output_dir / "iclight_smoke_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved_report, report)

    def test_invalid_counts_are_rejected_before_adapter_creation(self):
        for count in (0, -1, 11, 100):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.run_smoke(count=count)
        self.assertEqual(self.factory_configs, [])
        self.assertFalse((self.output_dir / "relit").exists())

    def test_duplicate_manifest_identities_are_rejected_before_adapter_creation(self):
        for column in ("audit_index", "row_index", "image_id"):
            with self.subTest(column=column):
                corrupt = [dict(row) for row in self.rows]
                corrupt[2][column] = corrupt[0][column]
                self.write_manifest(corrupt)
                with self.assertRaises(ValueError):
                    self.run_smoke()
                self.assertEqual(self.factory_configs, [])
        self.assertFalse((self.output_dir / "relit").exists())

    def test_insufficient_manifest_rows_fail_before_adapter_creation(self):
        self.write_manifest(self.rows[:9])
        with self.assertRaises(ValueError):
            self.run_smoke()
        self.assertEqual(self.factory_configs, [])
        self.assertFalse((self.output_dir / "relit").exists())

    def test_bad_source_is_rejected_before_adapter_creation(self):
        source_path = Path(self.rows[8]["source_512_path"])
        original = source_path.read_bytes()
        invalid_sources = (
            ("wide", Image.new("RGB", (513, 512))),
            ("grayscale", Image.new("L", (512, 512))),
            ("corrupt", None),
        )
        for label, invalid_source in invalid_sources:
            with self.subTest(source=label):
                if invalid_source is None:
                    source_path.write_bytes(b"this is not a readable image\n")
                else:
                    invalid_source.save(source_path)
                    invalid_source.close()
                try:
                    with self.assertRaises((ValueError, OSError)):
                        self.run_smoke()
                    self.assertEqual(self.factory_configs, [])
                finally:
                    source_path.write_bytes(original)
        source_path.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.run_smoke()
        self.assertEqual(self.factory_configs, [])

    def test_late_bad_output_preserves_all_previously_published_artifacts(self):
        self.run_smoke()
        previous_hashes = self.published_hashes()
        previous_names = {
            path.relative_to(self.output_dir).as_posix()
            for path in self.output_dir.rglob("*") if path.is_file()
        }
        for name, invalid_output in (
            ("wrong_size", Image.new("RGB", (511, 512))),
            ("wrong_mode", Image.new("L", (512, 512))),
            ("not_image", None),
        ):
            with self.subTest(output=name):
                calls = []
                factory = lambda config: FakeAdapter(
                    config, calls, fail_on_call=7, bad_output=invalid_output
                )
                with self.assertRaises((ValueError, TypeError)):
                    self.run_smoke(adapter_factory=factory)
                self.assertEqual(previous_hashes, self.published_hashes())
                self.assertEqual(previous_names, {
                    path.relative_to(self.output_dir).as_posix()
                    for path in self.output_dir.rglob("*") if path.is_file()
                })
                self.assertEqual(len(calls), 7)
            if invalid_output is not None:
                invalid_output.close()

    def test_corrupt_saved_output_is_rejected_and_previous_outputs_preserved(self):
        self.run_smoke()
        previous_hashes = self.published_hashes()
        original_save = Image.Image.save

        def corrupt_relit_save(image, path, *args, **kwargs):
            if isinstance(path, (str, os.PathLike)) and "relit" in Path(path).parts:
                Path(path).write_bytes(b"corrupt generated output\n")
            else:
                return original_save(image, path, *args, **kwargs)

        with mock.patch.object(Image.Image, "save", new=corrupt_relit_save):
            with self.assertRaises((ValueError, OSError)):
                self.run_smoke()
        self.assertEqual(previous_hashes, self.published_hashes())


if __name__ == "__main__":
    unittest.main()

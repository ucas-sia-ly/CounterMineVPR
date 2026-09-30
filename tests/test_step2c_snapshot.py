"""CPU-only Step 2C snapshot checks using small, entirely fake artifacts."""

import copy
import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


MODES = ("identity_control", "official_rmbg", "full_scene")
RELIGHT_MODES = MODES[1:]
SUMMARY_METRICS = (
    "num_matches", "match_ratio_min", "repeatability_min_2px",
    "repeatability_min_4px", "repeatability_min_8px", "repeatability_min_16px",
    "displacement_median", "displacement_q95", "grid_coverage_4px",
    "grid_coverage_8px",
)
COMPACT_FIELDS = (
    "audit_index", "row_index", "mode", "num_matches", "repeatability_min_4px",
    "repeatability_min_8px", "repeatability_min_16px", "displacement_median",
    "displacement_q95", "grid_coverage_4px", "grid_coverage_8px",
)
ALPHA_SERIES = {
    "alpha_mean": (0.1, 0.04),
    "alpha_q05": (0.0, 0.01),
    "alpha_q50": (0.0, 0.05),
    "alpha_q95": (0.5, 0.05),
    "alpha_fraction_lt_0_5": (0.9, -0.05),
    "alpha_fraction_gt_0_9": (0.0, 0.05),
}
QUANTILES = ("median", "q05", "q25", "q75", "q95")
CSV_FIELDS = ("audit_index", "row_index", "mode", *SUMMARY_METRICS,
              "num_keypoints_source", "num_keypoints_relit", "displacement_mean")
FAKE_COMMIT = "1234567890abcdef1234567890abcdef12345678"


def quantile(values, probability):
    """Independent stdlib equivalent of inclusive linear quantiles."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower])


def summarize(values):
    defined = [value for value in values if value is not None]
    probabilities = {"median": 0.5, "q05": 0.05, "q25": 0.25, "q75": 0.75, "q95": 0.95}
    result = {name: quantile(defined, probability) if defined else None
              for name, probability in probabilities.items()}
    return {**result, "valid_count": len(defined), "missing_count": len(values) - len(defined)}


def fake_inputs():
    crop = {"min": 0.5, "median": 0.75, "q05": 0.525,
            "q25": 0.625, "q75": 0.875, "q95": 0.975}
    audit = {
        "number_of_images": 100, "canonical_resolution": [512, 512],
        "seed": 42, "crop_long_axis_fraction": dict(crop),
        "crop_area_fraction": {key: value * value for key, value in crop.items()},
        "manifest_sha256": "a" * 64,
    }
    smoke = {
        "number_of_sources": 10, "count_outputs": 20,
        "modes": list(RELIGHT_MODES), "probe_image_format": "PNG",
        "config": {
            "seed": 12345, "prompt": "soft diffuse overcast daylight",
            "width": 512, "height": 512, "steps": 25, "cfg": 2.0,
            "num_samples": 1, "background": None, "added_prompt": "",
            "negative_prompt": "lowres, cropped", "highres_scale": 1.0,
            "highres_denoise": 0.5, "lowres_denoise": 0.9,
            "base_model": "test/base", "base_revision": None,
            "offset_model": "test/offset", "offset_revision": None,
            "offset_filename": "iclight_sd15_fc.safetensors",
            "rmbg_model": "test/rmbg", "rmbg_revision": None,
            "rmbg_sigma": 0.0, "rmbg_dtype": "float32",
            "text_encoder_dtype": "float16", "unet_dtype": "float16",
            "vae_dtype": "bfloat16", "device": "cuda",
            "scheduler_algorithm_type": "sde-dpmsolver++",
            "scheduler_beta_start": 0.00085, "scheduler_beta_end": 0.012,
            "scheduler_num_train_timesteps": 1000, "scheduler_steps_offset": 1,
            "scheduler_use_karras_sigmas": True,
            "iclight_root": "third_party/IC-Light", "cache_dir": None,
            "checkpoint_path": None, "local_files_only": False,
        },
        "source_sha256": {"/ignored/cache/source.png": "b" * 64},
        "runs": [],
    }
    rows = []
    for index in range(10):
        row_index = 100 + (9 - index) * 7
        for mode_index, mode in enumerate(MODES):
            r8 = (index // 2 + mode_index + 1) / 20
            rows.append({
                "audit_index": index, "row_index": row_index, "mode": mode,
                "num_matches": 30 + index + mode_index,
                "match_ratio_min": r8 + 0.1,
                "repeatability_min_2px": r8 / 4,
                "repeatability_min_4px": r8 / 2,
                "repeatability_min_8px": r8,
                "repeatability_min_16px": r8 + 0.05,
                "displacement_median": index / 7 + mode_index,
                "displacement_q95": index / 3 + mode_index + 2,
                "grid_coverage_4px": (index + 1) / 20,
                "grid_coverage_8px": (index + 2) / 20,
                "num_keypoints_source": 100, "num_keypoints_relit": 90,
                "displacement_mean": index / 5 + mode_index,
            })
            if mode in RELIGHT_MODES:
                stats = ({name: base + step * index
                          for name, (base, step) in ALPHA_SERIES.items()}
                         if mode == "official_rmbg" else {})
                smoke["runs"].append({
                    "audit_index": index, "row_index": row_index, "mode": mode,
                    "output_path": f"cache/relit/{mode}/{row_index:08d}.png",
                    "output_sha256": "c" * 64,
                    "adapter_stats": {**stats, "elapsed_seconds": 3.0},
                })
    fidelity = {
        "number_of_sources": 10, "count_pairs": 30,
        "config": {
            "seed": 42, "canonical_resolution": [512, 512], "device": "cuda",
            "max_keypoints": 2048, "extract_resize": None, "registration": None,
            "matcher_compiled": False,
        },
        "matcher": {
            "canonical_size": [512, 512], "extract_resize": None, "registration": None,
            "extractor_settings": {"max_num_keypoints": 2048, "detection_threshold": 0.2},
            "matcher_settings": {"features": "aliked", "depth_confidence": -1,
                                 "width_confidence": -1, "filter_threshold": 0.1, "mp": False},
            "models": {
                "extractor_config": {"model_name": "aliked-n16", "nms_radius": 2,
                                     "max_num_keypoints": 2048, "detection_threshold": 0.2},
                "matcher_config": {"name": "lightglue", "weights": "aliked_lightglue",
                                   "descriptor_dim": 256, "input_dim": 128, "flash": True,
                                   "depth_confidence": -1, "width_confidence": -1,
                                   "filter_threshold": 0.1, "mp": False},
                "checkpoints": {"aliked": {"cache_path": "/ignored/checkpoint.pth", "sha256": "d" * 64}},
            },
            "vendored_root": "/ignored/vendor/LightGlue",
        },
        "probe_sha256": {"/ignored/probe.png": "e" * 64},
        "per_match_coordinates": [[1.0, 2.0]],
        "modes": {},
    }
    for mode in MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        fidelity["modes"][mode] = {
            "count_pairs": 10,
            "metrics": {metric: summarize([row[metric] for row in mode_rows])
                        for metric in SUMMARY_METRICS},
        }
    # The snapshot must copy logged quantiles, rather than recalculate from CSV.
    fidelity["modes"]["full_scene"]["metrics"]["displacement_q95"]["median"] = 4.123456789012345
    return audit, smoke, fidelity, rows


class Step2CSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exporter = importlib.import_module("tools.10_export_step2c_snapshot")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = [self.root / name for name in (
            "audit_summary.json", "iclight_smoke_summary.json",
            "fidelity_summary.json", "fidelity_10.csv",
        )]
        self.output = self.root / "step2c_metrics.json"
        self.reset_inputs()

    def reset_inputs(self):
        self.audit, self.smoke, self.fidelity, self.rows = fake_inputs()
        self.write_inputs()

    def write_inputs(self):
        for path, content in zip(self.paths, (self.audit, self.smoke, self.fidelity)):
            path.write_text(json.dumps(content), encoding="utf-8")
        with self.paths[3].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def export(self):
        return self.exporter.export_snapshot(*self.paths, self.output, git_commit=FAKE_COMMIT)

    def assert_rejected_without_overwrite(self):
        previous = b'{"previous": "preserved"}\n'
        self.output.write_bytes(previous)
        self.write_inputs()
        with self.assertRaises(ValueError):
            self.export()
        self.assertEqual(self.output.read_bytes(), previous)

    def test_preserves_exact_statistics_and_compact_machine_readable_results(self):
        snapshot = self.export()
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), snapshot)
        self.assertEqual(set(snapshot), {"provenance", "canonical_crop", "rmbg_alpha",
                                        "fidelity", "per_source_compact", "worst_by_r8"})
        self.assertEqual(snapshot["fidelity"], self.fidelity["modes"])
        self.assertEqual(snapshot["canonical_crop"], {
            name: self.audit[name] for name in ("crop_long_axis_fraction", "crop_area_fraction")
        })
        provenance = snapshot["provenance"]
        for key, expected in {
            "git_commit": FAKE_COMMIT, "number_of_audit_sources": 10,
            "number_of_canonical_audit_images": 100,
            "number_of_fidelity_sources": 10, "canonical_resolution": [512, 512],
            "iclight_seed": 12345, "fidelity_seed": 42,
            "prompt": self.smoke["config"]["prompt"], "extract_resize": None,
            "registration": None,
        }.items():
            self.assertEqual(provenance[key], expected)
        for key in ("steps", "cfg", "highres_scale", "highres_denoise", "lowres_denoise",
                    "width", "height", "negative_prompt", "scheduler_algorithm_type"):
            self.assertEqual(provenance["iclight_config"][key], self.smoke["config"][key])
        for key, value in self.fidelity["matcher"]["models"]["extractor_config"].items():
            self.assertEqual(provenance["aliked_config"][key], value)
        for key, value in self.fidelity["matcher"]["models"]["matcher_config"].items():
            self.assertEqual(provenance["lightglue_config"][key], value)
        records = snapshot["per_source_compact"]
        self.assertEqual(len(records), 20)
        expected = [{key: row[key] for key in COMPACT_FIELDS}
                    for row in self.rows if row["mode"] in RELIGHT_MODES]
        key = lambda row: (row["audit_index"], row["row_index"], row["mode"])
        self.assertEqual(sorted(records, key=key), sorted(expected, key=key))
        self.assertTrue(all(set(record) == set(COMPACT_FIELDS) for record in records))
        serialized = self.output.read_text(encoding="utf-8")
        for excluded in ("sha256", "per_match_coordinates", "/ignored/"):
            self.assertNotIn(excluded, serialized)
        self.assertLess(len(serialized), 40000)
        self.assertNotIn("NaN", serialized)
        self.assertNotIn("Infinity", serialized)

    def test_alpha_aggregates_only_official_rmbg_with_linear_quantiles(self):
        snapshot = self.export()
        alpha = snapshot["rmbg_alpha"]
        probabilities = {"min": 0, "median": 0.5, "q05": 0.05,
                         "q25": 0.25, "q75": 0.75, "q95": 0.95, "max": 1}
        for name, (base, step) in ALPHA_SERIES.items():
            self.assertEqual(set(alpha[name]), set(probabilities))
            minimum, maximum = sorted((base, base + step * 9))
            for statistic, probability in probabilities.items():
                self.assertAlmostEqual(alpha[name][statistic], minimum + probability * (maximum - minimum))
        self.assertNotIn("full_scene", alpha)

    def test_worst_rows_are_deterministic_with_ties_and_shuffled_inputs(self):
        self.rows.reverse()
        self.smoke["runs"].reverse()
        self.write_inputs()
        first = self.export()
        first_bytes = self.output.read_bytes()
        for mode in RELIGHT_MODES:
            expected = sorted((row for row in self.rows if row["mode"] == mode),
                              key=lambda row: (row["repeatability_min_8px"],
                                               row["audit_index"], row["row_index"]))[:5]
            actual = first["worst_by_r8"][mode]
            self.assertEqual(len(actual), 5)
            self.assertEqual([(row["audit_index"], row["row_index"], row["repeatability_min_8px"])
                              for row in actual],
                             [(row["audit_index"], row["row_index"], row["repeatability_min_8px"])
                              for row in expected])
        self.rows.reverse()
        self.smoke["runs"].reverse()
        self.write_inputs()
        self.assertEqual(self.export(), first)
        self.assertEqual(self.output.read_bytes(), first_bytes)

    def test_rejects_incorrect_source_counts_and_duplicate_missing_or_misaligned_rows(self):
        mutations = (
            lambda: self.fidelity.update(number_of_sources=9),
            lambda: self.fidelity.update(number_of_sources=11),
            lambda: self.fidelity.update(count_pairs=29),
            lambda: self.rows.__setitem__(slice(None), [row for row in self.rows if row["audit_index"] != 9]),
            lambda: self.rows.append(copy.deepcopy(self.rows[1])),
            lambda: self.rows.pop(1),
            lambda: self.rows[1].update(row_index=99999),
            lambda: self.smoke.update(number_of_sources=9),
            lambda: self.smoke.update(count_outputs=19),
            lambda: self.smoke["runs"].pop(),
            lambda: self.smoke["runs"].append(copy.deepcopy(self.smoke["runs"][0])),
            lambda: self.smoke["runs"][0].update(row_index=99999),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                self.reset_inputs()
                mutate()
                self.assert_rejected_without_overwrite()

    def test_rejects_missing_modes_required_summaries_and_alpha_statistics(self):
        mutations = (
            lambda: self.fidelity["modes"].pop("identity_control"),
            lambda: self.fidelity["modes"].pop("official_rmbg"),
            lambda: self.fidelity["modes"].pop("full_scene"),
            lambda: self.rows.__setitem__(slice(None), [row for row in self.rows if row["mode"] != "full_scene"]),
            lambda: self.smoke.update(modes=["official_rmbg"]),
            lambda: self.fidelity["modes"]["identity_control"]["metrics"].pop("num_matches"),
            lambda: self.fidelity["modes"]["official_rmbg"]["metrics"]["grid_coverage_8px"].pop("q95"),
            lambda: self.fidelity["modes"]["full_scene"]["metrics"]["displacement_median"].pop("valid_count"),
            lambda: self.fidelity["modes"]["full_scene"].update(count_pairs=9),
            lambda: self.fidelity["modes"]["full_scene"]["metrics"]["displacement_median"].update(missing_count=1),
            lambda: self.audit["crop_area_fraction"].pop("q05"),
            lambda: self.smoke["runs"][0]["adapter_stats"].pop("alpha_mean"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                self.reset_inputs()
                mutate()
                self.assert_rejected_without_overwrite()

    def test_rejects_nonfinite_json_and_csv_values(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for target in ("summary", "crop", "alpha", "generation", "csv", "identity_csv"):
                with self.subTest(value=value, target=target):
                    self.reset_inputs()
                    if target == "summary":
                        self.fidelity["modes"]["official_rmbg"]["metrics"]["num_matches"]["median"] = value
                    elif target == "crop":
                        self.audit["crop_long_axis_fraction"]["q95"] = value
                    elif target == "alpha":
                        self.smoke["runs"][0]["adapter_stats"]["alpha_q95"] = value
                    elif target == "generation":
                        self.smoke["config"]["cfg"] = value
                    else:
                        self.rows[0 if target == "identity_csv" else 1]["repeatability_min_8px"] = value
                    self.assert_rejected_without_overwrite()

    def test_rejects_absolute_paths_in_exported_provenance(self):
        for path in ("/home/private/model", r"C:\private\model", r"\\server\share\model",
                     "file:///home/private/model", "C://private/model",
                     "model located at '//server/share/model'"):
            for config in ("iclight", "aliked"):
                with self.subTest(path=path, config=config):
                    self.reset_inputs()
                    if config == "iclight":
                        self.smoke["config"]["base_model"] = path
                    else:
                        self.fidelity["matcher"]["models"]["extractor_config"]["model_name"] = path
                    self.assert_rejected_without_overwrite()

    def test_null_displacements_are_preserved_only_when_genuinely_undefined(self):
        for row in self.rows:
            if row["mode"] == "official_rmbg":
                for name in SUMMARY_METRICS:
                    row[name] = 0
                row["displacement_median"] = None
                row["displacement_q95"] = None
                row["displacement_mean"] = None
        metrics = self.fidelity["modes"]["official_rmbg"]["metrics"]
        for name in SUMMARY_METRICS:
            metrics[name] = summarize([row[name] for row in self.rows if row["mode"] == "official_rmbg"])
        self.write_inputs()
        snapshot = self.export()
        self.assertEqual(snapshot["fidelity"]["official_rmbg"]["metrics"]["displacement_median"],
                         {**{key: None for key in QUANTILES}, "valid_count": 0, "missing_count": 10})
        self.assertTrue(all(row["displacement_median"] is None and row["displacement_q95"] is None
                            for row in snapshot["per_source_compact"] if row["mode"] == "official_rmbg"))
        self.reset_inputs()
        self.rows[1]["displacement_median"] = None
        self.assert_rejected_without_overwrite()
        self.reset_inputs()
        self.fidelity["modes"]["official_rmbg"]["metrics"]["displacement_median"]["median"] = None
        self.assert_rejected_without_overwrite()
        self.reset_inputs()
        self.rows[1]["repeatability_min_8px"] = None
        self.assert_rejected_without_overwrite()

    def test_zero_matches_reject_defined_displacements(self):
        for name in ("displacement_median", "displacement_q95"):
            with self.subTest(metric=name):
                self.reset_inputs()
                self.rows[1].update(num_matches=0, displacement_median=None, displacement_q95=None)
                self.rows[1][name] = 0.0
                mode_rows = [row for row in self.rows if row["mode"] == "official_rmbg"]
                self.fidelity["modes"]["official_rmbg"]["metrics"] = {
                    metric: summarize([row[metric] for row in mode_rows])
                    for metric in SUMMARY_METRICS
                }
                self.assert_rejected_without_overwrite()

    def test_output_cannot_alias_an_input_or_replace_its_contents(self):
        originals = [path.read_bytes() for path in self.paths]
        for path in self.paths:
            with self.subTest(input=path.name):
                with self.assertRaises(ValueError):
                    self.exporter.export_snapshot(*self.paths, path, git_commit=FAKE_COMMIT)
                self.assertEqual([source.read_bytes() for source in self.paths], originals)
        alias = self.root / "input_alias.json"
        alias.symlink_to(self.paths[0])
        with self.assertRaises(ValueError):
            self.exporter.export_snapshot(*self.paths, alias, git_commit=FAKE_COMMIT)
        self.assertTrue(alias.is_symlink())
        self.assertEqual([source.read_bytes() for source in self.paths], originals)

    def test_cli_exports_fake_artifacts_without_importing_models_or_cuda(self):
        script = Path(self.exporter.__file__).resolve()
        flags = ("--audit-summary", "--smoke-summary", "--fidelity-summary", "--fidelity-csv")
        arguments = [str(script)]
        for flag, path in zip(flags, self.paths):
            arguments.extend((flag, str(path)))
        arguments.extend(("--output", str(self.output)))
        guard = """
import builtins
import runpy
import sys
forbidden = {'torch', 'torchvision', 'cuda', 'salad', 'aliked', 'lightglue',
             'iclight', 'ic_light', '_countermine_vendored_lightglue',
             'diffusers', 'transformers', 'numpy', 'countermine'}
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0].lower() in forbidden:
        raise AssertionError('Forbidden model/CUDA import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", guard, *arguments],
                                cwd=self.exporter.REPO_ROOT, capture_output=True, text=True,
                                timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        snapshot = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(len(snapshot["per_source_compact"]), 20)
        self.assertEqual(snapshot["fidelity"], self.fidelity["modes"])


if __name__ == "__main__":
    unittest.main()

"""Frozen Step 3A inference stage integration with CPU-only fake models.

Actual PNG, NPY, and JSON artifacts are exercised. No model weights are loaded,
and torch/CUDA imports in the inference pipeline are blocked during each test.
"""

import builtins
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
import weakref

import numpy as np
from PIL import Image

from countermine.probe.local_fidelity import MatchResult
from countermine.probe import step3a_pipeline as pipeline
from countermine.probe.step3a_population import derive_image_seed


REPO_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def diagnostic_import_guard():
    """Fail if a diagnostic stage reaches training or loads a real tensor stack."""
    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("Fake-stage tests must not import torch or CUDA")
        if name == "countermine.training" or name.startswith("countermine.training."):
            raise AssertionError("Synthetic diagnostic probes reached a training API")
        if name == "countermine" and "training" in fromlist:
            raise AssertionError("Synthetic diagnostic probes reached a training API")
        return original(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=guarded):
        yield


class _Feature:
    def __init__(self, path):
        self.path = Path(path)


class InferenceFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache" / "step3a_test"
        self.cache.mkdir(parents=True)
        historical = json.loads((REPO_ROOT / "docs/audits/step2d2_native100_metrics.json").read_text())
        iclight = dict(historical["provenance"]["iclight_config"])
        iclight.pop("seed")
        iclight["seed_derivation_rule"] = pipeline.SEED_RULE
        self.population = {
            "configs": {
                "iclight": iclight,
                "salad": {"model_name": "dinov2_salad", "image_size": [322, 322], "seed": 42,
                          "salad_git_commit": "frozen-salad-commit", "diagnostic_only": True},
                "aliked": historical["provenance"]["aliked_config"],
                "lightglue": historical["provenance"]["lightglue_config"],
            },
            "images": [],
            "triplets": [
                {"q_image_id": "q", "p_image_id": "p", "n_hard_image_id": "hard", "n_random_image_id": "random"},
                {"q_image_id": "p", "p_image_id": "q", "n_hard_image_id": "hard", "n_random_image_id": "random"},
            ],
        }
        for image_id, color in (("q", (100, 20, 10)), ("p", (20, 100, 10)),
                                ("hard", (10, 20, 100)), ("random", (80, 70, 10))):
            original = self.root / "data" / f"{image_id}.jpg"
            source = self.cache / "source" / f"{image_id}.png"
            original.parent.mkdir(exist_ok=True)
            source.parent.mkdir(exist_ok=True)
            Image.new("RGB", (640, 480), color).save(original)
            with Image.open(original) as decoded:
                decoded.convert("RGB").save(source)
            self.population["images"].append({
                "image_id": image_id,
                "original_path": original.relative_to(self.root).as_posix(),
                "source_path": source.relative_to(self.root).as_posix(),
                "relit_path": (self.cache / "relit" / f"{image_id}.png").relative_to(self.root).as_posix(),
                "source_sha256": pipeline.sha256(source), "original_sha256": pipeline.sha256(original),
                "seed": derive_image_seed(image_id),
            })
        pipeline.save_json(self.cache / "population.json", self.population)
        self.seed_mock = self._patch("_seed_inference")
        self.release_mock = self._patch("_release_models")
        self._patch("ROOT", self.root)
        self._patch("load_population", return_value=self.population)
        self._patch("git_head", return_value="frozen-salad-commit")
        self._patch("print", create=True, side_effect=lambda *args, **kwargs: None)

    def _patch(self, name, *args, **kwargs):
        patcher = mock.patch.object(pipeline, name, *args, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def encoder_factory(self, *, invalid=None):
        instances = []

        class Encoder:
            def __init__(self, image_size, batch_size):
                self.image_size = image_size
                self.batch_size = batch_size
                self.paths = []
                instances.append(self)

            def iter_encode(self, paths):
                paths = list(paths)
                self.paths = paths
                vectors = {
                    "q": [1., 0., 0.], "p": [.8, .6, 0.],
                    "hard": [.6, 0., .8], "random": [0., 1., 0.],
                }
                if paths[0].parent.name == "relit":
                    vectors["p"] = [.7, np.sqrt(.51), 0.]
                    vectors["hard"] = [.75, 0., np.sqrt(.4375)]
                block = np.asarray([vectors[path.stem] for path in paths], dtype=np.float32)
                if invalid == "nan":
                    block[-1, 0] = np.nan
                elif invalid == "zero":
                    block[-1] = 0
                elif invalid == "incomplete":
                    block = block[:-1]
                for start in range(0, len(block), self.batch_size):
                    yield start, block[start:start + self.batch_size]

        return Encoder, instances

    def adapter_factory(self, *, fail_at=None, bad_size=False):
        instances, successful = [], []

        class Adapter:
            def __init__(self, config):
                self.config = config
                self.rmbg = None
                self.load_count = 0
                self.calls = []
                self.last_run_stats = {"elapsed_seconds": .25,
                                       "peak_cuda_memory_allocated_bytes": None,
                                       "peak_cuda_memory_reserved_bytes": None}
                instances.append(self)

            def _ensure_models(self):
                self.load_count += 1

            def relight(self, source, mode):
                image_id = Path(source.filename).stem
                self.calls.append((image_id, self.config.seed, mode))
                if fail_at is not None and len(self.calls) == fail_at:
                    raise RuntimeError("Simulated interruption")
                successful.append((image_id, self.config.seed, mode))
                if bad_size:
                    return Image.new("RGB", (512, 512))
                pixels = np.asarray(source).astype(np.uint16)
                return Image.fromarray(np.minimum(pixels + 1, 255).astype(np.uint8))

        return Adapter, instances, successful

    def matcher_factory(self, *, fail_cross_at=None, config_mismatch=False):
        instances = []
        config = copy.deepcopy(self.population["configs"])

        class Matcher:
            def __init__(self, settings):
                self.config = settings
                self.source_calls = []
                self.cross_calls = []
                self.features = weakref.WeakSet()
                self.max_live_features = 0
                self.model_loaded = False
                instances.append(self)

            def _ensure_models(self):
                self.model_loaded = True

            def extract(self, path):
                feature = _Feature(path)
                self.features.add(feature)
                self.max_live_features = max(self.max_live_features, len(self.features))
                return feature

            def match(self, source, relit):
                if source.path.stem != relit.path.stem:
                    raise AssertionError("Cross-place pair entered source-relit fidelity")
                self.source_calls.append(source.path.stem)
                num_good = {"q": 2, "p": 4, "hard": 1, "random": 5}[source.path.stem]
                points = np.array([[10. * i, 0.] for i in range(5)])
                target = points.copy()
                target[:num_good, 1] += 8  # Inclusive threshold must count these.
                target[num_good:, 1] += 20
                return MatchResult(5, 5, points, target, np.ones(5), 640, 480)

            def match_cross_place(self, source, target):
                if source.path.stem == target.path.stem:
                    raise AssertionError("Same image entered cross-place confusion")
                phase = "z" if source.path.parent.name == "relit" else "rgb"
                self.cross_calls.append((source.path.stem, target.path.stem, phase))
                if fail_cross_at is not None and len(self.cross_calls) == fail_cross_at:
                    raise RuntimeError("Simulated local interruption")
                count = 3 if phase == "z" and target.path.stem == "hard" else 2
                source_points = np.array([[0., 0.], [320., 240.], [80., 60.]])[:count]
                target_points = np.array([[639., 479.], [80., 60.], [320., 240.]])[:count]
                return MatchResult(5, 5, source_points, target_points, np.ones(count), 640, 480)

            def runtime_metadata(self):
                if not self.model_loaded:
                    raise AssertionError("Complete resumed journals still require model provenance")
                matcher_config = dict(config["lightglue"])
                features = matcher_config.pop("features")  # Actual LightGlue consumes this constructor argument.
                matcher_config.pop("compiled")
                version = matcher_config.pop("weights_version")
                if config_mismatch:
                    matcher_config["filter_threshold"] = .2
                return {"models": {"extractor_config": config["aliked"],
                                   "matcher_config": matcher_config,
                                   "lightglue_weights_version": version,
                                   "checkpoints": {}},
                        "matcher_settings": {"features": features},
                        "compiled": False, "versions": {}, "vendored_source_sha256": {},
                        "extract_resize": None, "resolved_device": "cpu"}

        return Matcher, instances

    def freeze_rgb(self):
        factory, instances = self.encoder_factory()
        with diagnostic_import_guard():
            result = pipeline.freeze_descriptors(self.cache, phase="rgb", batch_size=2, encoder_factory=factory)
        return result, instances

    def generate(self):
        self.freeze_rgb()
        factory, instances, calls = self.adapter_factory()
        with diagnostic_import_guard():
            result = pipeline.generate_probes(self.cache, device="cpu", adapter_factory=factory)
        return result, instances, calls


class TestStep3ADescriptorInference(InferenceFixture):
    def test_rgb_original_paths_frozen_resize_seed_and_cosines(self):
        summary, instances = self.freeze_rgb()
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].image_size, 322)
        self.assertEqual(instances[0].batch_size, 2)
        self.assertEqual(instances[0].paths,
                         [self.root / image["original_path"] for image in self.population["images"]])
        self.seed_mock.assert_called_once_with(42)
        self.assertAlmostEqual(summary["measurements"][0]["s_qp"], .8, places=6)
        self.assertAlmostEqual(summary["measurements"][0]["s_qhard"], .6, places=6)
        self.assertEqual(summary["measurements"][0]["s_qrandom"], 0.)
        bank = np.load(self.cache / "rgb_descriptors.npy", mmap_mode="r", allow_pickle=False)
        self.assertIsInstance(bank, np.memmap)
        self.assertEqual(bank.shape, (4, 3))

    def test_rgb_cache_reuse_and_descriptor_tampering_rejected(self):
        expected, _ = self.freeze_rgb()
        factory = mock.Mock(side_effect=AssertionError("Must reuse frozen descriptors"))
        with diagnostic_import_guard():
            self.assertEqual(pipeline.freeze_descriptors(self.cache, phase="rgb", batch_size=2,
                                                       encoder_factory=factory), expected)
        factory.assert_not_called()
        with (self.cache / "rgb_descriptors.npy").open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "cache differs"):
            pipeline.freeze_descriptors(self.cache, phase="rgb", batch_size=2, encoder_factory=factory)

    def test_relit_paths_are_diagnostic_inference_and_same_encoder_policy(self):
        self.generate()
        factory, instances = self.encoder_factory()
        with diagnostic_import_guard():
            result = pipeline.freeze_descriptors(self.cache, phase="relit", batch_size=2, encoder_factory=factory)
        self.assertEqual(instances[0].paths,
                         [self.root / image["relit_path"] for image in self.population["images"]])
        self.assertEqual(instances[0].image_size, 322)
        self.assertEqual(result["synthetic_role"], "diagnostic inference only")
        self.assertAlmostEqual(result["measurements"][0]["sz_qp"], .7, places=6)
        self.assertAlmostEqual(result["measurements"][0]["sz_qhard"], .75, places=6)

    def test_relit_requires_completed_generation_before_constructing_encoder(self):
        factory = mock.Mock()
        with diagnostic_import_guard(), self.assertRaises(FileNotFoundError):
            pipeline.freeze_descriptors(self.cache, phase="relit", encoder_factory=factory)
        factory.assert_not_called()

    def test_invalid_or_incomplete_descriptors_cannot_be_frozen(self):
        for failure in ("nan", "incomplete", "zero"):
            with self.subTest(failure=failure):
                factory, _ = self.encoder_factory(invalid=failure)
                with diagnostic_import_guard(), self.assertRaises(ValueError):
                    pipeline.freeze_descriptors(self.cache, phase="rgb", batch_size=2, encoder_factory=factory)
                self.assertFalse((self.cache / "rgb_salad.json").exists())


class TestStep3AGenerationInference(InferenceFixture):
    def test_unique_images_independent_seeds_frozen_condition_and_cache_reuse(self):
        summary, instances, calls = self.generate()
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].load_count, 1)
        self.assertEqual(calls, [(image["image_id"], image["seed"], "full_scene")
                                 for image in self.population["images"]])
        self.assertEqual(len({seed for _, seed, _ in calls}), 4)
        self.assertEqual(summary["generated_image_count"], 4)
        self.assertEqual(summary["generation_elapsed_seconds"], 1.)
        self.assertEqual(instances[0].config.prompt, pipeline.PROMPT)
        self.assertEqual((instances[0].config.width, instances[0].config.height), (640, 480))
        self.assertEqual(instances[0].config.steps, 25)
        self.assertEqual(instances[0].config.cfg, 2.)
        forbidden = mock.Mock(side_effect=AssertionError("Must reuse generated probes"))
        with diagnostic_import_guard():
            reused = pipeline.generate_probes(self.cache, device="cpu", adapter_factory=forbidden)
        self.assertEqual(reused, summary)
        forbidden.assert_not_called()

    def test_rgb_measurements_required_before_generation(self):
        factory = mock.Mock()
        with diagnostic_import_guard(), self.assertRaises(FileNotFoundError):
            pipeline.generate_probes(self.cache, device="cpu", adapter_factory=factory)
        factory.assert_not_called()

    def test_interrupted_generation_resumes_without_duplicate_probes(self):
        self.freeze_rgb()
        first, _, first_calls = self.adapter_factory(fail_at=3)
        with diagnostic_import_guard(), self.assertRaisesRegex(RuntimeError, "interruption"):
            pipeline.generate_probes(self.cache, device="cpu", adapter_factory=first)
        self.assertEqual(len(first_calls), 2)
        self.assertFalse((self.cache / "generation.json").exists())
        second, _, second_calls = self.adapter_factory()
        with diagnostic_import_guard():
            summary = pipeline.generate_probes(self.cache, device="cpu", adapter_factory=second)
        self.assertEqual([image_id for image_id, _, _ in first_calls + second_calls], ["q", "p", "hard", "random"])
        self.assertEqual(summary["generated_image_count"], 4)

    def test_unlogged_probe_and_non_native_output_are_rejected(self):
        self.freeze_rgb()
        target = self.root / self.population["images"][0]["relit_path"]
        target.parent.mkdir(exist_ok=True)
        Image.new("RGB", (640, 480)).save(target)
        factory = mock.Mock()
        with diagnostic_import_guard(), self.assertRaisesRegex(ValueError, "Unlogged"):
            pipeline.generate_probes(self.cache, device="cpu", adapter_factory=factory)
        factory.assert_not_called()
        target.unlink()
        factory, _, _ = self.adapter_factory(bad_size=True)
        with diagnostic_import_guard(), self.assertRaisesRegex(ValueError, "native RGB"):
            pipeline.generate_probes(self.cache, device="cpu", adapter_factory=factory)
        self.assertFalse(target.exists())

    def test_completed_probe_pixel_tampering_is_rejected(self):
        self.generate()
        target = self.root / self.population["images"][0]["relit_path"]
        Image.new("RGB", (640, 480), "white").save(target)
        with diagnostic_import_guard(), self.assertRaisesRegex(ValueError, "pixels changed"):
            pipeline.generate_probes(self.cache, device="cpu", adapter_factory=mock.Mock())


class TestStep3ALocalInference(InferenceFixture):
    def test_r8_inclusive_failed_images_retained_cross_support_and_bounded_features(self):
        self.generate()
        factory, instances = self.matcher_factory()
        with diagnostic_import_guard():
            saved = pipeline.measure_local(self.cache, device="cpu", matcher_factory=factory)
        self.assertEqual(instances[0].source_calls, ["q", "p", "hard", "random"])
        self.assertEqual(len(saved["per_image"]), 4)
        fidelity = {row["image_id"]: row for row in saved["per_image"]}
        self.assertEqual(fidelity["q"]["source_relit_R8"], .4)
        self.assertTrue(fidelity["q"]["fidelity_R8_pass"])
        self.assertFalse(fidelity["hard"]["fidelity_R8_pass"])
        self.assertEqual(len(saved["cross_place"]), 8)
        hard = [row for row in saved["cross_place"] if row["negative_image_id"] == "hard"]
        self.assertEqual({row["cross_match_ratio"] for row in hard}, {.4, .6})
        self.assertTrue(all("displacement_mean" not in row and "repeatability_min_8px" not in row
                            for row in saved["cross_place"]))
        self.assertEqual(instances[0].max_live_features, 2)
        self.assertEqual(len(instances[0].features), 0)
        self.assertEqual(instances[0].config.max_keypoints, 2048)
        self.assertEqual(instances[0].config.seed, 42)
        forbidden = mock.Mock(side_effect=AssertionError("Must reuse completed local observations"))
        with diagnostic_import_guard():
            self.assertEqual(pipeline.measure_local(self.cache, device="cpu", matcher_factory=forbidden), saved)
        forbidden.assert_not_called()

    def test_local_resume_reuses_completed_fidelity_and_paired_query_measurements(self):
        self.generate()
        first, _ = self.matcher_factory(fail_cross_at=5)
        with diagnostic_import_guard(), self.assertRaisesRegex(RuntimeError, "interruption"):
            pipeline.measure_local(self.cache, device="cpu", matcher_factory=first)
        progress = pipeline.read_json(self.cache / "local_progress.json")
        self.assertEqual(len(progress["per_image"]), 4)
        self.assertEqual(len(progress["cross_place"]), 4)
        second, instances = self.matcher_factory()
        with diagnostic_import_guard():
            result = pipeline.measure_local(self.cache, device="cpu", matcher_factory=second)
        self.assertEqual(instances[0].source_calls, [])
        self.assertEqual(len(instances[0].cross_calls), 4)
        self.assertEqual(len(result["cross_place"]), 8)

    def test_local_runtime_configuration_must_match_frozen_settings(self):
        self.generate()
        factory, _ = self.matcher_factory(config_mismatch=True)
        with diagnostic_import_guard(), self.assertRaisesRegex(ValueError, "Runtime local model"):
            pipeline.measure_local(self.cache, device="cpu", matcher_factory=factory)
        self.assertFalse((self.cache / "local.json").exists())

    def test_fully_measured_journal_resumes_without_matching_and_loads_provenance(self):
        self.generate()
        first, _ = self.matcher_factory()
        with diagnostic_import_guard():
            expected = pipeline.measure_local(self.cache, device="cpu", matcher_factory=first)
        (self.cache / "local.json").unlink()
        second, instances = self.matcher_factory()
        with diagnostic_import_guard():
            result = pipeline.measure_local(self.cache, device="cpu", matcher_factory=second)
        self.assertTrue(instances[0].model_loaded)
        self.assertEqual(instances[0].source_calls, [])
        self.assertEqual(instances[0].cross_calls, [])
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

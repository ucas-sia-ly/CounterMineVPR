"""Step 3A cache-to-snapshot integration with fake CPU scalar observations.

No inference or model constructors are called. The image and descriptor files
are tiny temporary fixtures, and figures are stubbed to inspect the complete
paired data passed to the separate visual-audit layer.
"""

import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from countermine.probe import step3a_pipeline as pipeline
from countermine.probe.geometry_audit import sha256
from countermine.probe.step3a_population import derive_image_seed


REPO_ROOT = Path(__file__).resolve().parents[1]


class Step3APipelineAnalysisTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache/step3a"
        self.cache.mkdir(parents=True)
        self.plot_dir, self.curated_dir = self.root / "outputs/step3a", self.root / "docs/audits"
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(pipeline, "ROOT", self.root).start()
        mock.patch.object(pipeline, "git_head", return_value="a" * 40).start()
        mock.patch.object(pipeline, "_seed_inference", side_effect=AssertionError("No inference in analysis")).start()
        mock.patch.object(pipeline, "_release_models", side_effect=AssertionError("No CUDA/models in analysis")).start()
        self.population = self.make_population()
        pipeline.save_json(self.cache / "population.json", self.population)
        mock.patch.object(pipeline, "load_population", return_value=self.population).start()
        self.generation = self.make_generation()
        pipeline.save_json(self.cache / "generation.json", self.generation)
        self.rgb, self.relit = self.make_salad_measurements()
        self.local = self.make_local()
        self.write_measurements()
        self.visuals = mock.patch("countermine.probe.step3a_visuals.create_step3a_visual_outputs",
                                  side_effect=self.fake_figures).start()

    def make_population(self):
        triplets, images = [], []
        for query_index in range(2):
            triplet = {"triplet_index": query_index}
            for role_index, role in enumerate(pipeline.ROLES):
                identifier = f"Images/{role}-{query_index}.jpg"
                place = f"q-place-{query_index}" if role in ("q", "p") else f"{role}-place-{query_index}"
                source_path = f"cache/step3a/source/{role}-{query_index}.png"
                relit_path = f"cache/step3a/relit/{role}-{query_index}.png"
                for name in (source_path, relit_path):
                    path = self.root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (640, 480), (role_index * 20, 10, query_index * 10)).save(path)
                triplet.update({f"{role}_image_id": identifier, f"{role}_place_uid": place,
                                f"{role}_row_index": query_index * 4 + role_index,
                                f"{role}_relative_path": identifier})
                images.append({"image_id": identifier, "place_uid": place, "row_index": query_index * 4 + role_index,
                               "source_path": source_path, "original_path": source_path, "relit_path": relit_path,
                               "source_sha256": sha256(self.root / source_path),
                               "original_sha256": sha256(self.root / source_path), "seed": derive_image_seed(identifier)})
            triplets.append(triplet)
        salad = {"model_name": "dinov2_salad", "image_size": [322, 322], "seed": 42,
                 "salad_git_commit": "b" * 40, "diagnostic_only": True}
        return {"triplets": triplets, "images": images, "inputs": {},
                "population_construction": {"requested_query_count": 500, "selected_query_count": 2,
                                            "selection_seed": 42, "same_query_positive_pair": True},
                "configs": {"salad": salad, "iclight": {"width": 640, "height": 480, "prompt": pipeline.PROMPT,
                    "cfg": 2, "steps": 25, "highres_scale": 1, "highres_denoise": .5,
                    "seed_derivation_rule": pipeline.SEED_RULE},
                    "aliked": {"max_num_keypoints": 2048, "detection_threshold": .2},
                    "lightglue": {"features": "aliked", "depth_confidence": -1, "width_confidence": -1,
                                  "filter_threshold": .1, "mp": False},
                    "local_seed": 42, "extract_resize": None,
                    "geometry": {"policy": "native_full_fov", "canonical_width": 640, "canonical_height": 480}},
                "query_count": 2, "unique_image_count": 8, "min_geo_distance_m": 25}

    def make_generation(self):
        runs = [{"image_id": image["image_id"], "seed": image["seed"],
                 "source_sha256": image["source_sha256"], "relit_path": image["relit_path"],
                 "output_sha256": sha256(self.root / image["relit_path"]), "elapsed_seconds": .1}
                for image in self.population["images"]]
        return {**pipeline.generation_identity(self.cache, self.population), "runs": runs,
                "generated_image_count": len(runs), "generation_elapsed_seconds": .8}

    def make_salad_measurements(self):
        rgb_cosines = {"q": 1, "p": .8, "n_hard": .6, "n_random": .1}
        relit_cosines = {"q": 1, "p": .6, "n_hard": .8, "n_random": .2}
        summaries = []
        for phase, cosines, prefix in (("rgb", rgb_cosines, "s"), ("relit", relit_cosines, "sz")):
            descriptors = np.asarray([[cosines[role], np.sqrt(max(0, 1 - cosines[role] ** 2))]
                                      for _query in range(2) for role in pipeline.ROLES], dtype=np.float32)
            np.save(self.cache / f"{phase}_descriptors.npy", descriptors)
            measurements = [{"q_image_id": triplet["q_image_id"],
                             **{f"{prefix}_q{kind}": pipeline.cosine(descriptors[index * 4], descriptors[index * 4 + offset])
                                for kind, offset in (("p", 1), ("hard", 2), ("random", 3))}}
                            for index, triplet in enumerate(self.population["triplets"])]
            summaries.append({"population_sha256": sha256(self.cache / "population.json"),
                "generation_sha256": sha256(self.cache / "generation.json") if phase == "relit" else None,
                "phase": phase, "config": {**self.population["configs"]["salad"], "batch_size": 8},
                "checkpoint_sha256": "c" * 64,
                "descriptor_sha256": sha256(self.cache / f"{phase}_descriptors.npy"), "measurements": measurements})
        return summaries

    def make_local(self):
        per_image = []
        for image in self.population["images"]:
            r8 = .3 if image["image_id"] == "Images/n_random-1.jpg" else .5
            per_image.append({"image_id": image["image_id"], "source_relit_R8": r8,
                              "fidelity_R8_pass": r8 >= .4, "rgb_mae_normalized": .1,
                              "abs_luma_mean_delta": 4})
        cross = []
        for triplet in self.population["triplets"]:
            for role in ("n_hard", "n_random"):
                for phase in ("rgb", "z"):
                    ratio = .2 if role == "n_hard" and phase == "z" else .1
                    cross.append({"q_image_id": triplet["q_image_id"], "negative_image_id": triplet[f"{role}_image_id"],
                                  "phase": phase, "num_matches": int(ratio * 100),
                                  "num_keypoints_source": 100, "num_keypoints_target": 100,
                                  "cross_match_ratio": ratio, "matched_source_cell_coverage": .25,
                                  "matched_target_cell_coverage": .25})
        return {"population_sha256": sha256(self.cache / "population.json"),
                "generation_sha256": sha256(self.cache / "generation.json"),
                "aliked_config": self.population["configs"]["aliked"],
                "lightglue_config": self.population["configs"]["lightglue"], "seed": 42,
                "resize": None, "R8_threshold": .4, "R8_coordinate_dtype": "float64",
                "cross_place_geometry": "unregistered; no displacement/R4/R8/homography/F-matrix",
                "per_image": per_image, "cross_place": cross,
                "runtime_provenance": {"resolved_device": "cpu", "extract_resize": None,
                                       "versions": {}, "vendored_source_sha256": {}, "checkpoint_sha256": {}}}

    def write_measurements(self):
        for name, data in (("rgb_salad.json", self.rgb), ("relit_salad.json", self.relit), ("local.json", self.local)):
            pipeline.save_json(self.cache / name, data)

    def fake_figures(self, records, image_lookup, output_dir):
        for name in pipeline.PLOT_NAMES:
            (Path(output_dir) / name).write_bytes(b"stub descriptive figure\n")

    def analyze(self):
        return pipeline.analyze_and_export(self.cache, plot_dir=self.plot_dir, curated_dir=self.curated_dir)

    def test_analysis_exports_all_paired_records_common_subset_and_portable_artifacts(self):
        result = self.analyze()
        self.assertEqual(result["counts"]["query_count"], 2)
        self.assertEqual(result["counts"]["unique_image_count"], 8)
        self.assertEqual(result["counts"]["R8_pass_triplet_count"], 1)
        self.assertEqual(len(result["per_triplet_compact"]), 2)
        self.assertEqual(len(result["top_margin_collapse"]), 1)
        first, second = result["per_triplet_compact"]
        self.assertAlmostEqual(first["delta_margin_hard"], -.4, places=6)
        self.assertAlmostEqual(first["delta_cross_match_ratio_hard"], .1)
        self.assertTrue(first["hard_correct_to_wrong"])
        for role in pipeline.ROLES:
            self.assertEqual(first[f"{role}_seed"], derive_image_seed(first[f"{role}_image_id"]))
        self.assertFalse(second["all_images_R8_pass"])
        self.assertTrue(second["hard_images_R8_pass"])
        self.assertEqual(result["hard_negative_summary"]["all_triplets"]["triplet_count"], 2)
        self.assertEqual(result["random_negative_summary"]["all_images_R8_pass"]["triplet_count"], 1)
        passed_records, lookup, destination = self.visuals.call_args.args
        self.assertEqual(len(passed_records), 2)
        self.assertEqual(set(lookup), {image["image_id"] for image in self.population["images"]})
        self.assertEqual(destination, self.plot_dir)
        self.assertEqual(json.loads((self.curated_dir / "step3a_counterfactual_margin_metrics.json").read_text()), result)
        self.assertNotIn(str(self.root), json.dumps(result, allow_nan=False))
        for name in pipeline.PLOT_NAMES:
            self.assertEqual((self.plot_dir / name).read_bytes(), (self.curated_dir / f"step3a_{name}").read_bytes())
        self.assertEqual(result["provenance"]["runtime_artifacts"]["local.json"], sha256(self.cache / "local.json"))

    def test_rejects_population_generation_descriptor_and_model_drift(self):
        edits = [
            ("rgb", lambda data: data.update(population_sha256="x" * 64)),
            ("rgb", lambda data: data.update(phase="relit")),
            ("relit", lambda data: data.update(generation_sha256="x" * 64)),
            ("local", lambda data: data.update(generation_sha256="x" * 64)),
            ("rgb", lambda data: data.update(descriptor_sha256="x" * 64)),
            ("relit", lambda data: data.update(checkpoint_sha256="x" * 64)),
        ]
        originals = {name: copy.deepcopy(getattr(self, name)) for name in ("rgb", "relit", "local")}
        for name, edit in edits:
            for key, value in originals.items():
                setattr(self, key, copy.deepcopy(value))
            edit(getattr(self, name))
            self.write_measurements()
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.analyze()
        self.assertFalse(self.visuals.called)

    def test_rejects_duplicate_extra_or_missing_measurement_identities(self):
        originals = {name: copy.deepcopy(getattr(self, name)) for name in ("rgb", "relit", "local")}
        cases = (("rgb", "measurements", "q_image_id"), ("relit", "measurements", "q_image_id"),
                 ("local", "per_image", "image_id"), ("local", "cross_place", "q_image_id"))
        for owner, field, identity in cases:
            for mutation in ("duplicate", "extra", "missing"):
                for key, value in originals.items():
                    setattr(self, key, copy.deepcopy(value))
                rows = getattr(self, owner)[field]
                if mutation == "missing":
                    rows.pop()
                else:
                    extra = dict(rows[0])
                    if mutation == "extra":
                        extra[identity] = "Images/not-in-pilot.jpg"
                    rows.append(extra)
                self.write_measurements()
                with self.subTest(owner=owner, field=field, mutation=mutation), self.assertRaises(ValueError):
                    self.analyze()
        self.assertFalse(self.visuals.called)

    def test_rejects_equal_but_unfrozen_salad_and_local_configuration(self):
        self.rgb["config"]["image_size"] = [512, 512]
        self.relit["config"]["image_size"] = [512, 512]
        self.write_measurements()
        with self.assertRaisesRegex(ValueError, "frozen|SALAD|configuration"):
            self.analyze()
        self.rgb, self.relit = self.make_salad_measurements()
        self.local["resize"] = 512
        self.write_measurements()
        with self.assertRaisesRegex(ValueError, "frozen|local|configuration"):
            self.analyze()

    def test_rejects_cached_cosine_values_inconsistent_with_descriptor_bank(self):
        self.rgb["measurements"][0]["s_qp"] -= .1
        self.write_measurements()
        with self.assertRaisesRegex(ValueError, "cosine|descriptor|measurements"):
            self.analyze()

    def test_generation_count_seed_and_output_hash_agree_with_unique_images(self):
        original = copy.deepcopy(self.generation)
        for mutate in (lambda data: data.update(generated_image_count=7),
                       lambda data: data["runs"][0].update(seed=12345),
                       lambda data: data["runs"][0].update(output_sha256="x" * 64)):
            changed = copy.deepcopy(original)
            mutate(changed)
            pipeline.save_json(self.cache / "generation.json", changed)
            with self.assertRaises(ValueError):
                pipeline.validate_generation(self.cache, self.population)

    def test_managed_cache_rejects_historical_directory_outside_cache_and_symlink(self):
        for name in ("cache", "cache/native100_audit/subdirectory", "outputs/new", "cache/gsv_mini/probe"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pipeline.managed_cache(self.root / name)
        target = self.root / "cache/target"
        target.mkdir()
        linked = self.root / "cache/linked"
        linked.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            pipeline.managed_cache(linked / "subdirectory")


class Step3ANoTrainingAPITests(unittest.TestCase):
    def test_step3a_transitive_local_sources_have_no_training_imports_or_calls(self):
        roots = list((REPO_ROOT / "countermine/probe").glob("step3a_*.py"))
        roots.extend((REPO_ROOT / "tools").glob("2[3-7]_*step3a*.py"))
        self.assertGreaterEqual(len(roots), 10)
        pending, seen = list(roots), set()
        forbidden_modules = ("countermine.training", "countermine.losses", "countermine.optimization", "torch.optim")
        forbidden_calls = {"backward", "fit", "train", "training_step", "configure_optimizers", "optimizer_step"}
        violations = []
        while pending:
            path = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if any(name == forbidden or name.startswith(forbidden + ".") for forbidden in forbidden_modules):
                        violations.append(f"{path.name}:{node.lineno} imports {name}")
                    if name.startswith("countermine."):
                        dependency = REPO_ROOT / (name.replace(".", "/") + ".py")
                        if dependency.is_file():
                            pending.append(dependency)
                if isinstance(node, ast.Call):
                    function = node.func
                    call = function.attr if isinstance(function, ast.Attribute) else function.id if isinstance(function, ast.Name) else None
                    if call in forbidden_calls:
                        violations.append(f"{path.name}:{node.lineno} calls {call}")
        self.assertIn(REPO_ROOT / "countermine/mining/salad_encoder.py", seen)
        self.assertIn(REPO_ROOT / "countermine/probe/iclight_adapter.py", seen)
        self.assertIn(REPO_ROOT / "countermine/probe/local_fidelity.py", seen)
        self.assertEqual(violations, [], "Synthetic diagnostic inference must not reach any training API")


if __name__ == "__main__":
    unittest.main()

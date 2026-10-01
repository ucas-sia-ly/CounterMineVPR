"""CPU population/cache integrity checks using tiny native RGB JPEG fixtures."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from countermine.probe.geometry_audit import sha256
from countermine.probe import step3a_pipeline as pipeline
from countermine.probe.step3a_population import ROLES


REPOSITORY = Path(__file__).resolve().parents[1]


class Step3APipelinePopulationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cache = self.root / "cache/pilot"
        self.dataset = self.root / "data/gsv-cities"
        self.metadata = self.root / "metadata"
        self.metadata.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(pipeline, "ROOT", self.root).start()
        patch.object(pipeline, "git_head", return_value="b" * 40).start()
        self.manifest = []
        for index in range(12):
            name = f"Images/city/image_{index:02d}.jpg"
            row = {"row_index": index, "image_id": name, "relative_path": name,
                   "place_uid": f"city:{index // 3}", "city_id": "city",
                   "lat": index // 3 * .02, "lon": 0.0}
            self.manifest.append(row)
            original = self.dataset / name
            original.parent.mkdir(parents=True, exist_ok=True)
            with Image.new("RGB", (640, 480), (index * 13, index * 19, 255 - index * 7)) as image:
                image.save(original, format="JPEG")
        self.candidates = []
        for q in self.manifest:
            negatives = [row for row in self.manifest if row["place_uid"] != q["place_uid"]]
            for rank, negative in enumerate(negatives, 1):
                self.candidates.append({"query_row_index": q["row_index"], "negative_row_index": negative["row_index"],
                    "query_image_id": q["image_id"], "negative_image_id": negative["image_id"],
                    "query_place_uid": q["place_uid"], "negative_place_uid": negative["place_uid"],
                    "rank": rank, "similarity": 1.0 / (rank + 1)})
        self.paths = {name: self.metadata / f"{name}.{'csv' if name in ('manifest', 'candidates') else 'json'}"
                      for name in ("manifest", "candidates", "mining", "index", "snapshot")}
        self._write_csv(self.paths["manifest"], self.manifest)
        self._write_csv(self.paths["candidates"], self.candidates)
        self.index = {"manifest_sha256": sha256(self.paths["manifest"]),
                      "salad_model_name": "dinov2_salad", "salad_image_size": [322, 322],
                      "salad_git_commit": "b" * 40, "seed": 42,
                      "number_of_images": 12, "descriptor_shape": [12, 8448],
                      "device": "cuda", "dtype": "float16", "batch_size": 32}
        self.mining = {"manifest_sha256": sha256(self.paths["manifest"]), "min_geo_distance_m": 250.0,
                       "number_of_queries": 12, "total_candidate_pairs": 108, "top_k": 9,
                       "descriptor_shape": [12, 8448], "descriptor_dtype": "float16", "seed": 42}
        self.snapshot = json.loads((REPOSITORY / "docs/audits/step2d2_native100_metrics.json").read_text())
        for name, value in (("index", self.index), ("mining", self.mining), ("snapshot", self.snapshot)):
            pipeline.save_json(self.paths[name], value)

    @staticmethod
    def _write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _build(self, **kwargs):
        return pipeline.build_population(self.paths["manifest"], self.paths["candidates"],
            self.paths["mining"], self.paths["index"], self.paths["snapshot"], self.dataset, self.cache, **kwargs)

    def _rewrite_population(self, value):
        pipeline.save_json(self.cache / "population.json", value)

    @staticmethod
    def _replace_role(triplet, role, metadata):
        for key in ("row_index", "image_id", "place_uid", "relative_path"):
            triplet[f"{role}_{key}"] = metadata[key]

    def test_build_and_load_retain_all_valid_unique_queries_and_exact_unique_images(self):
        built = self._build()
        loaded = pipeline.load_population(self.cache)
        self.assertEqual(built, loaded)
        self.assertEqual(loaded["query_count"], 12)
        self.assertEqual(loaded["unique_image_count"], 12)
        self.assertTrue(loaded["population_construction"]["fewer_than_requested"])
        for image in loaded["images"]:
            with Image.open(self.root / image["original_path"]) as original, Image.open(self.root / image["source_path"]) as source:
                self.assertEqual(original.convert("RGB").tobytes(), source.tobytes())
                self.assertEqual(source.size, (640, 480))
        self.assertEqual(len({row["q_image_id"] for row in loaded["triplets"]}), 12)

    def test_build_accounts_for_missing_real_rgb_files_before_freezing_selection(self):
        (self.dataset / self.manifest[0]["relative_path"]).unlink()
        population = self._build()
        accounting = population["population_construction"]
        self.assertEqual(accounting["unavailable_image_count"], 1)
        self.assertEqual(accounting["query_count"], 11)
        self.assertEqual(accounting["query_exclusion_counts"]["unavailable_query_source"], 1)
        self.assertEqual(population, pipeline.load_population(self.cache))

    def test_load_rejects_an_alternate_valid_same_place_positive(self):
        population = self._build()
        triplet = population["triplets"][0]
        alternative = next(row for row in self.manifest if row["place_uid"] == triplet["q_place_uid"]
                           and row["image_id"] not in (triplet["q_image_id"], triplet["p_image_id"]))
        self._replace_role(triplet, "p", alternative)
        self._rewrite_population(population)
        with self.assertRaises(ValueError):
            pipeline.load_population(self.cache)

    def test_load_rejects_a_valid_lower_ranked_rgb_hard_negative(self):
        population = self._build()
        triplet = population["triplets"][0]
        alternate = next(row for row in self.candidates if row["query_image_id"] == triplet["q_image_id"] and row["rank"] == 2)
        negative = self.manifest[alternate["negative_row_index"]]
        self._replace_role(triplet, "n_hard", negative)
        triplet["hard_rgb_rank"] = alternate["rank"]
        triplet["hard_candidate_similarity"] = alternate["similarity"]
        self._rewrite_population(population)
        with self.assertRaises(ValueError):
            pipeline.load_population(self.cache)

    def test_load_rejects_an_alternate_eligible_random_negative(self):
        population = self._build()
        triplet = population["triplets"][0]
        alternate = next(row for row in self.manifest if row["place_uid"] != triplet["q_place_uid"]
                         and row["image_id"] != triplet["n_random_image_id"])
        self._replace_role(triplet, "n_random", alternate)
        self._rewrite_population(population)
        with self.assertRaises(ValueError):
            pipeline.load_population(self.cache)

    def test_load_rejects_a_reduced_valid_population_instead_of_silently_accepting_it(self):
        population = self._build()
        population["triplets"] = population["triplets"][:1]
        used = {row[f"{role}_image_id"] for row in population["triplets"] for role in ROLES}
        population["images"] = [image for image in population["images"] if image["image_id"] in used]
        population["query_count"] = 1
        population["unique_image_count"] = len(used)
        population["population_construction"]["query_count"] = 1
        self._rewrite_population(population)
        with self.assertRaises(ValueError):
            pipeline.load_population(self.cache)

    def test_load_rejects_mutated_frozen_model_configs_geo_rule_and_counts(self):
        pristine = self._build()
        mutations = (
            lambda value: value["configs"]["iclight"].update(cfg=3.0),
            lambda value: value["configs"]["salad"].update(image_size=[512, 512]),
            lambda value: value["configs"]["lightglue"].update(filter_threshold=.3),
            lambda value: value.update(min_geo_distance_m=0.0),
            lambda value: value.update(query_count=999),
            lambda value: value.update(unique_image_count=999),
            lambda value: value["population_construction"].update(selection_seed=77),
        )
        for mutation in mutations:
            changed = copy.deepcopy(pristine)
            mutation(changed)
            self._rewrite_population(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                pipeline.load_population(self.cache)

    def test_load_rejects_image_record_original_source_swaps_even_with_matching_hashes(self):
        population = self._build()
        first, second = population["images"][:2]
        for key in ("original_path", "original_sha256", "source_path", "source_sha256"):
            first[key], second[key] = second[key], first[key]
        self._rewrite_population(population)
        with self.assertRaises(ValueError):
            pipeline.load_population(self.cache)

    def test_load_rejects_wrong_image_manifest_metadata(self):
        population = self._build()
        population["images"][0]["row_index"] = 999
        self._rewrite_population(population)
        with self.assertRaises(ValueError):
            pipeline.load_population(self.cache)

    def test_load_rejects_absolute_source_paths_even_if_the_pixels_match(self):
        population = self._build()
        image = population["images"][0]
        image["source_path"] = str(self.root / image["source_path"])
        self._rewrite_population(population)
        with self.assertRaises(ValueError):
            pipeline.load_population(self.cache)

    def test_failed_rebuild_never_changes_a_frozen_source_cache(self):
        population = self._build()
        image = population["images"][0]
        source = self.root / image["source_path"]
        previous = source.read_bytes()
        with Image.new("RGB", (640, 480), "magenta") as modified:
            modified.save(self.root / image["original_path"], format="JPEG")
        with self.assertRaises(ValueError):
            self._build()
        self.assertEqual(source.read_bytes(), previous)

    def test_load_rejects_mutated_frozen_input_or_seed(self):
        population = self._build()
        pristine = copy.deepcopy(population)
        population["images"][0]["seed"] += 1
        self._rewrite_population(population)
        with self.assertRaises(ValueError):
            pipeline.load_population(self.cache)
        self._rewrite_population(pristine)
        with self.paths["candidates"].open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaises(ValueError):
            pipeline.load_population(self.cache)

    def test_managed_cache_preserves_historical_directories_and_rejects_symlinks(self):
        for name in ("gsv_mini", "generator_audit", "geometry_audit", "native_fov_audit", "native100_audit"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pipeline.managed_cache(self.root / "cache" / name / "child")
        real = self.root / "cache/real"
        real.mkdir(parents=True)
        linked = self.root / "cache/linked"
        linked.symlink_to(real, target_is_directory=True)
        with self.assertRaises(ValueError):
            pipeline.managed_cache(linked)

    def test_build_rejects_symlinked_source_child_without_writing_to_historical_cache(self):
        historical = self.root / "cache/native100_audit/source"
        historical.mkdir(parents=True)
        self.cache.mkdir(parents=True)
        (self.cache / "source").symlink_to(historical, target_is_directory=True)
        with self.assertRaises(ValueError):
            self._build()
        self.assertEqual(list(historical.iterdir()), [])

    def test_load_rejects_symlinked_relit_child_before_generation_can_write_elsewhere(self):
        self._build()
        historical = self.root / "cache/native100_audit/relit"
        historical.mkdir(parents=True)
        (self.cache / "relit").symlink_to(historical, target_is_directory=True)
        with self.assertRaises(ValueError):
            pipeline.load_population(self.cache)


if __name__ == "__main__":
    unittest.main()

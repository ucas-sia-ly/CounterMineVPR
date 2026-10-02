"""Frozen upstream configuration and shared initialization, without model imports."""

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from collections import namedtuple
import unittest
from unittest.mock import Mock, patch

from countermine.training import step3a_config as config


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL = {
    "backbone_arch": "dinov2_vitb14",
    "backbone_config": {"num_trainable_blocks": 4, "return_token": True, "norm_layer": True},
    "agg_arch": "SALAD",
    "agg_config": {"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256},
    "lr": 6e-5,
    "optimizer": "adamw",
    "weight_decay": 9.5e-9,
    "momentum": 0.9,
    "lr_sched": "linear",
    "lr_sched_args": {"start_factor": 1, "end_factor": 0.2, "total_iters": 4000},
    "loss_name": "MultiSimilarityLoss",
    "miner_name": "MultiSimilarityMiner",
    "miner_margin": 0.1,
    "faiss_gpu": False,
}
EXPECTED_DATA = {
    "batch_size": 60,
    "img_per_place": 4,
    "min_img_per_place": 4,
    "shuffle_all": False,
    "random_sample_from_each_place": True,
    "image_size": (224, 224),
    "num_workers": 10,
    "show_data_stats": True,
    "val_set_names": ["pitts30k_val", "pitts30k_test", "msls_val"],
}
EXPECTED_TRAINER = {
    "accelerator": "gpu", "devices": 1, "num_nodes": 1,
    "precision": "16-mixed", "max_epochs": 4, "num_sanity_val_steps": 0,
    "check_val_every_n_epoch": 1, "reload_dataloaders_every_n_epochs": 1,
    "log_every_n_steps": 20,
}


def upstream_call_keywords(name):
    """Read frozen arguments without importing SALAD or constructing any model."""
    tree = ast.parse((ROOT / "salad/main.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = (node.func.id if isinstance(node.func, ast.Name)
                      else node.func.attr if isinstance(node.func, ast.Attribute) else "")
            if called == name:
                values = {}
                for keyword in node.keywords:
                    try:
                        values[keyword.arg] = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError):
                        pass
                return values
    raise AssertionError("No upstream constructor for " + name)


class Step3AConfigurationTests(unittest.TestCase):
    def test_model_optimizer_scheduler_loss_and_miner_match_frozen_upstream(self):
        self.assertEqual(config.MODEL_CONFIG, EXPECTED_MODEL)
        self.assertEqual(config.MODEL_CONFIG, upstream_call_keywords("VPRModel"))

    def test_data_and_trainer_configuration_match_original_salad(self):
        self.assertEqual(config.DATA_CONFIG, EXPECTED_DATA)
        self.assertEqual(config.DATA_CONFIG, upstream_call_keywords("GSVCitiesDataModule"))
        self.assertEqual(config.TRAINER_CONFIG, EXPECTED_TRAINER)
        original = upstream_call_keywords("Trainer")
        for key, value in EXPECTED_TRAINER.items():
            self.assertEqual(value, original[key], key)
        self.assertEqual(config.MODES, ("baseline", "countermine_q99_geo500"))
        self.assertEqual(config.SEED, 42)
        cities_tree = ast.parse((ROOT / "salad/dataloaders/GSVCitiesDataloader.py").read_text())
        upstream_cities = next(
            ast.literal_eval(node.value) for node in cities_tree.body
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "TRAIN_CITIES"
                for target in node.targets
            )
        )
        self.assertEqual(tuple(config.TRAIN_CITIES), tuple(upstream_cities))

    def test_treatment_remains_binary_same_city_sampling_only(self):
        treatment = config.TREATMENT_DEFINITION
        self.assertEqual(treatment["slice"], "core_q99_geo500")
        self.assertEqual(treatment["cities"], ["Boston", "London"])
        for key in ("same_city_only", "binary_edge_use", "vertex_disjoint_guided_matching"):
            self.assertIs(treatment[key], True)
        for key in ("edge_weighting", "hub_weighting", "loss_modification"):
            self.assertIs(treatment[key], False)

    def test_frozen_configuration_is_a_deep_copy_with_original_batch_dimensions(self):
        frozen = config.frozen_configuration()
        self.assertEqual(frozen["model"], EXPECTED_MODEL)
        self.assertEqual(frozen["data"], EXPECTED_DATA)
        self.assertEqual(frozen["trainer"], EXPECTED_TRAINER)
        self.assertEqual(frozen["batch_dimensions"], [60, 4, 3, 224, 224])
        self.assertIs(frozen["drop_last"], False)
        self.assertIs(frozen["seed_workers"], True)
        frozen["model"]["backbone_config"]["num_trainable_blocks"] = 100
        frozen["data"]["val_set_names"].clear()
        self.assertEqual(config.MODEL_CONFIG, EXPECTED_MODEL)
        self.assertEqual(config.DATA_CONFIG, EXPECTED_DATA)

    def test_create_model_passes_only_frozen_kwargs_to_original_class(self):
        upstream = ModuleType("vpr_model")
        upstream.VPRModel = Mock(side_effect=lambda **kwargs: SimpleNamespace(configuration=kwargs))
        with patch.dict(sys.modules, {"vpr_model": upstream}):
            baseline = config.create_model()
            treatment = config.create_model()
        self.assertEqual(upstream.VPRModel.call_count, 2)
        self.assertEqual(baseline.configuration, EXPECTED_MODEL)
        self.assertEqual(treatment.configuration, EXPECTED_MODEL)
        baseline.configuration["agg_config"]["num_clusters"] = -1
        self.assertEqual(treatment.configuration, EXPECTED_MODEL)
        self.assertEqual(config.MODEL_CONFIG, EXPECTED_MODEL)
        self.assertFalse(set(EXPECTED_MODEL) & {"graph", "structural_bottleneck", "weight", "margin"})

    def test_seed_is_applied_with_workers_before_runtime_work(self):
        lightning = ModuleType("pytorch_lightning")
        lightning.seed_everything = Mock()
        with patch.dict(sys.modules, {"pytorch_lightning": lightning}):
            config.seed_runtime(42)
            lightning.seed_everything.assert_called_once_with(42, workers=True)
            with self.assertRaises(ValueError):
                config.seed_runtime(43)
        lightning.seed_everything.assert_called_once()

    def test_paths_are_absolute_before_entering_upstream_relative_path_context(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = config.resolved_paths(Path(directory))
            self.assertTrue(all(path.is_absolute() for path in paths.values()))
            self.assertEqual(paths["dataset"], Path(directory) / "data/GSVCities")
            self.assertEqual(paths["salad"], Path(directory) / "salad")
            self.assertEqual(paths["graph"], Path(directory) / "cache/countermine_rgb/step2d/place_edges.csv")

    def test_missing_dataset_fails_clearly_before_model_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = config.resolved_paths(directory)
            with self.assertRaises(FileNotFoundError) as caught:
                config.validate_dataset_paths(paths)
            self.assertIn("Required upstream dataset paths are missing", str(caught.exception))
            self.assertIn("data/GSVCities/Dataframes/Boston.csv", str(caught.exception))
            self.assertIn("data/GSVCities/Dataframes/London.csv", str(caught.exception))
            self.assertIn("data/Pittsburgh/datasets/pitts30k_val.mat", str(caught.exception))

    def test_output_guard_preserves_all_prior_scientific_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = (
                "cache/countermine_rgb/step3a/shared/initial_state.pt",
                "outputs/step3a/recall_curves.png",
                "docs/audits/step3a_edge_cobatching_metrics.json",
            )
            for relative in allowed:
                self.assertEqual(config.guard_step3a(root / relative, root), root / relative)
            forbidden = [f"cache/countermine_rgb/step2{stage}/overwrite.csv" for stage in "abcd"]
            forbidden += [f"docs/audits/step2{stage}_snapshot.json" for stage in "abcd"]
            forbidden += ["outputs/step2d/overwrite.png", "salad/vpr_model.py", "data/GSVCities/new.csv",
                          "docs/audits/step3a_nested/metrics.json",
                          "cache/countermine_rgb/step3a/../step2d/overwrite.csv"]
            for relative in forbidden:
                with self.subTest(path=relative):
                    with self.assertRaises(ValueError):
                        config.guard_step3a(root / relative, root)
            runtime = root / "cache/countermine_rgb/step3a"
            runtime.mkdir(parents=True)
            previous = root / "cache/countermine_rgb/step2d"
            previous.mkdir()
            (runtime / "old_inputs").symlink_to(previous, target_is_directory=True)
            with self.assertRaises(ValueError):
                config.guard_step3a(runtime / "old_inputs/overwrite.csv", root)


class Step3ASharedInitializationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state_path = Path(self.directory.name) / "initial_state.pt"
        # An opaque artifact is enough: the supplied fake loader never imports torch.
        self.state_path.write_bytes(b"one shared frozen SALAD state dictionary")
        self.sha = hashlib.sha256(self.state_path.read_bytes()).hexdigest()
        self.summary = {"initial_state_sha256": self.sha, "model_configuration": deepcopy(EXPECTED_MODEL)}
        self.state = {"backbone.weight": object(), "aggregator.weight": object()}
        self.torch = SimpleNamespace(load=Mock(return_value=self.state))

    def fake_model(self, *, missing=(), unexpected=()):
        return SimpleNamespace(load_state_dict=Mock(return_value=SimpleNamespace(
            missing_keys=list(missing), unexpected_keys=list(unexpected),
        )))

    def test_shared_initialization_hash_is_stable(self):
        self.assertEqual(config.sha256_file(self.state_path), self.sha)
        self.assertEqual(config.sha256_file(self.state_path), self.sha)

    def test_both_conditions_strict_load_exactly_the_same_state(self):
        baseline, treatment = self.fake_model(), self.fake_model()
        for model in (baseline, treatment):
            loaded_sha = config.strict_load_initial_state(
                model, self.state_path, self.summary, torch_module=self.torch,
            )
            self.assertEqual(loaded_sha, self.sha)
            model.load_state_dict.assert_called_once_with(self.state, strict=True)
        self.assertEqual(self.torch.load.call_count, 2)
        for call in self.torch.load.call_args_list:
            self.assertEqual(call.args, (self.state_path,))
            self.assertEqual(call.kwargs, {"map_location": "cpu", "weights_only": True})

    def test_changed_artifact_is_rejected_before_loading(self):
        self.state_path.write_bytes(b"different optimized condition-specific state")
        model = self.fake_model()
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            config.strict_load_initial_state(model, self.state_path, self.summary, torch_module=self.torch)
        self.torch.load.assert_not_called()
        model.load_state_dict.assert_not_called()

    def test_architecture_mismatch_is_rejected_before_loading(self):
        self.summary["model_configuration"]["agg_config"]["num_clusters"] = 63
        with self.assertRaisesRegex(ValueError, "model configuration differs"):
            config.strict_load_initial_state(self.fake_model(), self.state_path, self.summary, torch_module=self.torch)
        self.torch.load.assert_not_called()

    def test_missing_or_unexpected_keys_cannot_be_silently_accepted(self):
        for options in ({"missing": ["backbone.weight"]}, {"unexpected": ["structural_loss.weight"]}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "strict load failed"):
                    config.strict_load_initial_state(
                        self.fake_model(**options), self.state_path, self.summary,
                        torch_module=self.torch,
                    )

    def test_upstream_strict_load_error_propagates(self):
        model = self.fake_model()
        model.load_state_dict.side_effect = RuntimeError("size mismatch for backbone.weight")
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            config.strict_load_initial_state(model, self.state_path, self.summary, torch_module=self.torch)


class Step3ADataModuleTests(unittest.TestCase):
    def fake_torch_modules(self, *, subset=None, loader=None):
        modules = {name: ModuleType(name) for name in ("torch", "torch.utils", "torch.utils.data")}
        if subset is not None:
            modules["torch.utils.data"].Subset = subset
        if loader is not None:
            modules["torch.utils.data"].DataLoader = loader
        return modules

    def test_scientific_wrapper_inherits_validation_and_reuses_train_loader_options(self):
        from countermine.training import salad_datamodule as integration
        upstream = ModuleType("dataloaders.GSVCitiesDataloader")

        class OriginalDataModule:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                for key, value in kwargs.items():
                    setattr(self, key, value)
                self.train_loader_config = {
                    "batch_size": 60, "num_workers": 10, "drop_last": False,
                    "pin_memory": True, "shuffle": False,
                }
                self.train_dataset = object()
                self.trainer = SimpleNamespace(current_epoch=2)
                self.reload = Mock()

            def val_dataloader(self):
                return "original validation loaders"

        upstream.GSVCitiesDataModule = OriginalDataModule
        loader = Mock(side_effect=lambda **kwargs: SimpleNamespace(options=kwargs))
        sampler = SimpleNamespace(save_plan=Mock())
        sampler_factory = Mock(return_value=sampler)
        mapping = Mock(return_value=(["mapping records"], {"mapped_graph_places": 2000}))
        modules = self.fake_torch_modules(loader=loader)
        modules.update({"dataloaders": ModuleType("dataloaders"),
                        "dataloaders.GSVCitiesDataloader": upstream})
        graph = SimpleNamespace(graph_place_uids=("Boston:0005994", "London:0002184"))
        with patch.dict(sys.modules, modules), patch.object(integration, "build_place_mapping", mapping), \
                patch.object(integration, "CounterMineBatchSampler", sampler_factory):
            data = integration.build_datamodule(
                mode="baseline", edge_bundle=graph, plan_dir="plans", seed=42,
            )
            output = data.train_dataloader()
        self.assertEqual(data.kwargs, EXPECTED_DATA)
        self.assertIs(type(data).val_dataloader, OriginalDataModule.val_dataloader)
        self.assertEqual(data.val_dataloader(), "original validation loaders")
        data.reload.assert_called_once_with()
        mapping.assert_called_once_with(data.train_dataset, graph.graph_place_uids)
        sampler_factory.assert_called_once_with(
            ["mapping records"], graph, mode="baseline", batch_size=60,
            seed=42, epoch=2, plan_dir="plans",
        )
        # The sampler constructor persists its plan; the wrapper must not rewrite it.
        sampler.save_plan.assert_not_called()
        self.assertEqual(output.options, {
            "dataset": data.train_dataset, "batch_sampler": sampler,
            "num_workers": 10, "pin_memory": True,
        })
        # Upstream's loader config is retained for every later dataset reload.
        self.assertEqual(data.train_loader_config["drop_last"], False)
        self.assertEqual(data.train_loader_config["batch_size"], 60)
        self.assertEqual(data.train_loader_config["shuffle"], False)

    def test_dataset_identity_order_must_match_preparation_and_the_baseline_epoch(self):
        from countermine.training import salad_datamodule as integration
        upstream = ModuleType("dataloaders.GSVCitiesDataloader")

        class OriginalDataModule:
            def __init__(self, **kwargs):
                self.batch_size = kwargs["batch_size"]
                self.train_loader_config = {"batch_size": 60, "shuffle": False, "drop_last": False,
                                            "num_workers": 10, "pin_memory": True}
                self.train_dataset = object()
                self.trainer = SimpleNamespace(current_epoch=0)
                self.reload = Mock()

        upstream.GSVCitiesDataModule = OriginalDataModule
        current = {
            "baseline_sequence_sha256": "same_index_range_hash",
            "dataset_place_order_sha256": "actual_canonical_uid_order_hash",
            "dataset_place_count": 4, "batch_count": 1, "batch_sizes": [4],
            "marginal_exposure": {"city_place_count": 4},
        }
        sampler = SimpleNamespace(plan=SimpleNamespace(summary=current))
        graph = SimpleNamespace(graph_place_uids=())
        loader = Mock()
        modules = self.fake_torch_modules(loader=loader)
        modules.update({"dataloaders": ModuleType("dataloaders"),
                        "dataloaders.GSVCitiesDataloader": upstream})
        with patch.dict(sys.modules, modules), \
                patch.object(integration, "build_place_mapping", return_value=([], {"mapped": 0})), \
                patch.object(integration, "CounterMineBatchSampler", return_value=sampler):
            prepared = integration.build_datamodule(
                mode="baseline", edge_bundle=graph, plan_dir="plans",
                expected_epoch0_place_order="different_canonical_uid_order_hash",
            )
            with self.assertRaisesRegex(ValueError, "Initial dataset place ordering differs"):
                prepared.train_dataloader()
            # Identical range(0,N) hashes and equal city marginal counts do not
            # prove that the baseline dataset assigned the same places to slots.
            reference = {**current, "dataset_place_order_sha256": "different_canonical_uid_order_hash"}
            with patch.object(integration, "read_json", return_value=reference):
                treatment = integration.build_datamodule(
                    mode="countermine_q99_geo500", edge_bundle=graph,
                    plan_dir="plans", baseline_plan_dir="baseline_plans",
                )
                with self.assertRaisesRegex(ValueError, "dataset_place_order_sha256"):
                    treatment.train_dataloader()
            with patch.object(integration, "read_json", return_value=dict(current)):
                matching = integration.build_datamodule(
                    mode="countermine_q99_geo500", edge_bundle=graph, plan_dir="plans",
                    expected_mapping={"mapped": 0}, expected_epoch0_place_order=current["dataset_place_order_sha256"],
                    baseline_plan_dir="baseline_plans",
                )
                matching.train_dataloader()
        loader.assert_called_once()

    def test_smoke_subsets_contain_queries_and_retain_all_remapped_positives(self):
        import numpy as np
        from countermine.training.salad_datamodule import smoke_validation_subset

        class OriginalSubset:
            def __init__(self, dataset, indices):
                self.dataset, self.indices = dataset, indices

            def __len__(self):
                return len(self.indices)

        structure = namedtuple("Structure", ["numDb", "numQ"])
        positives = [np.asarray([40 + query, 100 + query]) for query in range(30)]
        images = ["original_real_image_" + str(index) for index in range(170)]
        for name in ("pitts30k_val", "pitts30k_test", "msls_val"):
            with self.subTest(validation_set=name):
                dataset = SimpleNamespace(
                    images=list(images), dbStruct=structure(140, 30),
                    num_references=140, pIdx=positives,
                    getPositives=Mock(return_value=positives),
                )
                with patch.dict(sys.modules, self.fake_torch_modules(subset=OriginalSubset)):
                    subset, audit = smoke_validation_subset(dataset, name)
                self.assertEqual(audit["reference_count"], 100)
                self.assertEqual(audit["query_count"], 20)
                self.assertTrue(audit["all_selected_query_positives_retained"])
                self.assertEqual(len(subset), 120)
                self.assertEqual(subset.indices[100:], list(range(140, 160)))
                selected_refs = subset.indices[:100]
                actual_positives = subset.getPositives() if "pitts" in name else subset.pIdx
                for query, mapped in enumerate(actual_positives):
                    self.assertEqual([selected_refs[int(index)] for index in mapped], positives[query].tolist())
                self.assertEqual(subset.images, [images[index] for index in subset.indices])
                if "pitts" in name:
                    self.assertEqual(subset.dbStruct.numDb, 100)
                    self.assertEqual(subset.dbStruct.numQ, 20)
                else:
                    self.assertEqual(subset.num_references, 100)
                # The scientific dataset is never cropped or rewritten.
                self.assertEqual(dataset.images, images)
                self.assertEqual(dataset.dbStruct, structure(140, 30))
                self.assertEqual(dataset.num_references, 140)
                self.assertIs(dataset.pIdx, positives)

    def test_smoke_refuses_validation_without_queries_or_sufficient_references(self):
        from countermine.training.salad_datamodule import smoke_validation_subset
        class OriginalSubset:
            pass
        modules = self.fake_torch_modules(subset=OriginalSubset)
        with patch.dict(sys.modules, modules):
            insufficient = SimpleNamespace(num_references=99, pIdx=[[0]])
            with self.assertRaisesRegex(ValueError, "at least 100"):
                smoke_validation_subset(insufficient, "msls_val")
            no_queries = SimpleNamespace(num_references=140, pIdx=[[], [140], [-1]])
            with self.assertRaisesRegex(ValueError, "without dropping query positives"):
                smoke_validation_subset(no_queries, "msls_val")


if __name__ == "__main__":
    unittest.main()

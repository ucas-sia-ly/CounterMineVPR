"""CPU-only scalar artifact checks for the single-seed Step 3A exporter."""
from copy import deepcopy
import csv
import hashlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from countermine.training.step3a_audit import (
    MODES, RECALL_METRICS, build_comparison, export_comparison,
    guard_step3a, sha256_file, validate_portable_json, write_finite_json,
)
from countermine.training.step3a_config import frozen_configuration, TREATMENT_DEFINITION

ROOT = Path(__file__).resolve().parents[1]


def artifacts():
    digest = "a" * 64
    provenance = {"step2d_snapshot_sha256": digest, "step2d_place_edges_sha256": digest,
                  "initial_state_sha256": digest, "salad_submodule_commit": "b" * 40,
                  "step3a_code_hashes": {"countermine/training/fixture.py": digest},
                  "seed": 42, "real_rgb_only": True, "synthetic_images_used": False}
    mapping = {"total_training_places": 120, "boston_training_places": 60,
               "london_training_places": 60, "per_city_training_places": {"Boston": 60, "London": 60},
               "countermine_graph_places": 6, "mapped_graph_places": 6, "unmapped_graph_places": 0,
               "one_to_one_mapping": True}
    shared = {"complete": True, "provenance": provenance,
              "configuration": json.loads(json.dumps(frozen_configuration())),
              "treatment_definition": deepcopy(TREATMENT_DEFINITION), "mapping_audit": mapping,
              "dataset_metadata": {"dataset_root": "data/GSVCities", "metadata_sha256": {"data/GSVCities/Dataframes/Boston.csv": digest},
                                   "resolved_dataset_root_sha256": digest}}
    runs, evaluations, plans = {}, {}, {}
    for mode in MODES:
        treatment = mode != "baseline"
        epochs = []
        for epoch in range(4):
            metrics = {name: 70 + epoch + index % 3 + .2 * treatment
                       for index, name in enumerate(RECALL_METRICS)}
            metrics.update(loss=1.5 - epoch * .1 - .05 * treatment, b_acc=.3 + epoch * .01)
            epochs.append({"epoch": epoch, "metrics": metrics})
        runs[mode] = {"complete": True, "smoke": False, "mode": mode, "seed": 42,
                      "provenance": deepcopy(provenance), "epochs": epochs, "best_epoch": 3,
                      "best_checkpoint": "checkpoints/best.ckpt", "best_checkpoint_sha256": digest,
                      "final_train_loss": epochs[-1]["metrics"]["loss"], "average_b_acc": .315}
        evaluations[mode] = {"complete": True, "mode": mode, "seed": 42,
                             "checkpoint_selection": "best_pitts30k_val_R1", "best_epoch": 3,
                             "checkpoint_sha256": digest, "provenance": deepcopy(provenance),
                             "metrics": {name: epochs[-1]["metrics"][name] for name in RECALL_METRICS}}
        plans[mode] = []
        for epoch in range(4):
            guided = 2 if treatment else 0
            plans[mode].append({
                "dataset_place_count": 120, "batch_count": 2, "batch_sizes": [60, 60],
                "baseline_sequence_sha256": digest, "dataset_place_order_sha256": digest,
                "treatment_sequence_sha256": "c" * 64 if treatment else digest,
                "graph_sha256": digest, "seed": 42, "epoch": epoch,
                "eligible_q99_geo500_same_city_edge_count": 4,
                "greedy_matched_pair_count": guided, "unique_guided_place_count": guided * 2,
                "boston_guided_pair_count": guided // 2, "london_guided_pair_count": guided // 2,
                "intentionally_guided_pairs_cooccur_count": guided, "accidentally_split_pair_count": 0,
                "duplicate_place_count": 0, "missing_place_count": 0,
                "per_batch_city_composition_mismatch_count": 0, "max_guided_pairs_per_place": int(treatment),
                "exposure": {"core_q95": 8 if treatment else 4, "core_q99": 5 if treatment else 3,
                             "core_q99_geo500": 4 if treatment else 2,
                             "unique_q99_geo500_graph_edges_exposed": 4 if treatment else 2,
                             "unique_structural_places_exposed": 6 if treatment else 4},
                "marginal_exposure": {"per_city_place_appearances": {"Boston": 60, "London": 60},
                                      "per_city_sorted_place_uids_sha256": {"Boston": digest, "London": digest},
                                      "place_multiset_sha256": digest},
            })
    return shared, runs, evaluations, plans


class Step3AAuditTests(unittest.TestCase):
    def test_descriptive_snapshot_preserves_full_epoch_marginals_and_exposure(self):
        snapshot = build_comparison(*artifacts())
        self.assertTrue(snapshot["complete"])
        self.assertTrue(snapshot["scope"]["single_seed_pilot"])
        self.assertFalse(snapshot["scope"]["statistical_significance_claimed"])
        self.assertFalse(snapshot["provenance"]["synthetic_images_used"])
        self.assertEqual(len(snapshot["training"]["baseline"]["epochs"]), 4)
        for exposure in snapshot["training_exposure_comparison"]:
            self.assertEqual(exposure["countermine_to_baseline_ratio"], 2)
            self.assertTrue(exposure["identical_marginal_place_exposure"])
            self.assertEqual(exposure["unique_guided_places"], 4)
            self.assertEqual(exposure["intentionally_guided_pairs"], 2)
        for delta in snapshot["evaluation"]["descriptive_countermine_minus_baseline"].values():
            self.assertAlmostEqual(delta, .2)
        text = json.dumps(snapshot, allow_nan=False)
        self.assertNotIn("checkpoints/best.ckpt", text)
        self.assertNotIn("winner", text)
        self.assertNotIn("confidence_interval", text)
        self.assertNotIn("p_value", text)

    def test_zero_baseline_denominator_is_explicit_null_without_infinite_json(self):
        fixture = artifacts()
        for plan in fixture[3]["baseline"]:
            plan["exposure"]["core_q99_geo500"] = 0
            plan["exposure"]["unique_q99_geo500_graph_edges_exposed"] = 0
        result = build_comparison(*fixture)
        self.assertIsNone(result["training_exposure_comparison"][0]["countermine_to_baseline_ratio"])
        self.assertTrue(result["training_exposure_comparison"][0]["baseline_zero"])
        json.dumps(result, allow_nan=False)

    def test_incomplete_smoke_and_wrong_selection_cannot_be_scientific_results(self):
        changes = (
            lambda f: f[1]["baseline"].update(smoke=True),
            lambda f: f[1]["baseline"].update(complete=False),
            lambda f: f[1]["baseline"]["epochs"].pop(),
            lambda f: f[1]["baseline"].update(best_epoch=2),
            lambda f: f[2]["baseline"].update(checkpoint_selection="last"),
            lambda f: f[2]["baseline"].update(checkpoint_sha256="f" * 64),
            lambda f: f[2]["baseline"]["metrics"].update({"msls_val/R1": float("nan")}),
            lambda f: f[1]["baseline"]["provenance"].update(initial_state_sha256="f" * 64),
            lambda f: f[0]["configuration"]["model"].update(miner_margin=.2),
            lambda f: f[0]["treatment_definition"].update(hub_weighting=True),
            lambda f: f[0]["mapping_audit"].update(unmapped_graph_places=1),
        )
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                fixture = artifacts()
                change(fixture)
                with self.assertRaises(ValueError):
                    build_comparison(*fixture)

    def test_marginal_and_batch_invariant_violations_stop_comparison(self):
        mutations = (
            {"duplicate_place_count": 1}, {"missing_place_count": 1},
            {"per_batch_city_composition_mismatch_count": 1},
            {"accidentally_split_pair_count": 1}, {"max_guided_pairs_per_place": 2},
            {"unique_guided_place_count": 3}, {"baseline_sequence_sha256": "f" * 64},
            {"dataset_place_order_sha256": "f" * 64},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                fixture = artifacts()
                fixture[3][MODES[1]][0].update(mutation)
                with self.assertRaises(ValueError):
                    build_comparison(*fixture)
        for field in ("place_multiset_sha256", "per_city_sorted_place_uids_sha256"):
            fixture = artifacts()
            marginal = fixture[3][MODES[1]][0]["marginal_exposure"]
            marginal[field] = "f" * 64 if field == "place_multiset_sha256" else {"Boston": "f" * 64, "London": "a" * 64}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "marginal place exposure"):
                build_comparison(*fixture)
        fixture = artifacts()
        fixture[3][MODES[1]][0]["exposure"]["core_q99_geo500"] = 2
        with self.assertRaisesRegex(ValueError, "did not increase"):
            build_comparison(*fixture)

    def test_finite_portable_json_has_no_tensor_or_absolute_path_payloads(self):
        validate_portable_json({"checkpoint_sha256": "a" * 64, "relative": "batch_plans/epoch_00_summary.json"})
        for payload in ({"path": "/tmp/file"}, {"path": "C:\\cache\\state.pt"},
                        {"a": float("inf")}, {"a": [float("nan")]}, {"state_dict": {}},
                        {"descriptors": []}, {"image_pixels": []}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_portable_json(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "finite.json"
            write_finite_json(path, {"finite": .2, "undefined_ratio": None})
            self.assertEqual(json.loads(path.read_text()), {"finite": .2, "undefined_ratio": None})

    def test_namespace_guard_rejects_historical_and_symlink_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "cache/countermine_rgb/step3a"
            runtime.mkdir(parents=True)
            old = root / "cache/countermine_rgb/step2d"
            old.mkdir()
            (runtime / "alias").symlink_to(old, target_is_directory=True)
            guard_step3a([runtime / "shared/summary.json", root / "docs/audits/step3a_metrics.json"], root)
            for path in (old / "place_edges.csv", runtime / "alias/overwrite.json",
                         root / "docs/audits/step2d_full_countermine_metrics.json",
                         root / "docs/audits/step3a_nested/data.json"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    guard_step3a([path], root)

    def prepare_export_fixture(self, root):
        fixture = artifacts()
        shared, runs, evaluations, plans = fixture
        runtime = root / "cache/countermine_rgb/step3a"
        frozen_path = root / "docs/audits/step2d_full_countermine_metrics.json"
        graph = root / "cache/countermine_rgb/step2d/place_edges.csv"
        graph.parent.mkdir(parents=True)
        with graph.open("w", newline="") as stream:
            columns = ["place_uid_a", "place_uid_b", "city_id_a", "city_id_b",
                       "num_core_q95_image_edges", "num_core_q99_image_edges", "num_core_q99_geo500_image_edges"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for city in ("Boston", "London"):
                for end in (40, 41):
                    writer.writerow(dict(zip(columns, [f"{city}:0000000", f"{city}:{end:07d}", city, city, 1, 1, 1])))
        graph_hash = sha256_file(graph)
        frozen_path.parent.mkdir(parents=True)
        frozen_path.write_text(json.dumps({"complete": True, "population": {"unique_places": 6},
                                          "graph_artifact_hashes": {"place_edges.csv": graph_hash}}))
        state = runtime / "shared/initial_state.pt"
        state.parent.mkdir(parents=True)
        state.write_bytes(b"shared init hash fixture; not a torch checkpoint")
        source = root / "countermine/training/fixture.py"
        source.parent.mkdir(parents=True)
        source.write_text("# source identity fixture\n")
        salad_source = root / "salad/vpr_model.py"
        salad_source.parent.mkdir()
        salad_source.write_text("# untouched upstream identity fixture\n")
        provenance = shared["provenance"]
        provenance.update(step2d_snapshot_sha256=sha256_file(frozen_path), step2d_place_edges_sha256=graph_hash,
                          initial_state_sha256=sha256_file(state),
                          step3a_code_hashes={"countermine/training/fixture.py": sha256_file(source)},
                          salad_source_hashes={"salad/vpr_model.py": sha256_file(salad_source)})
        metadata_path = root / "data/GSVCities/Dataframes/Boston.csv"
        metadata_path.parent.mkdir(parents=True)
        metadata_path.write_text("dataset metadata identity fixture")
        shared["dataset_metadata"] = {"dataset_root": "data/GSVCities", "root_is_symlink": False,
                                      "resolved_dataset_root_sha256": hashlib.sha256(str((root / "data/GSVCities").resolve()).encode()).hexdigest(),
                                      "metadata_sha256": {"data/GSVCities/Dataframes/Boston.csv": sha256_file(metadata_path)}}
        write_finite_json(runtime / "shared/preparation_summary.json", shared)
        from countermine.training.gsv_place_mapping import PlaceRecord
        from countermine.training.countermine_batch_sampler import load_edge_bundle, CounterMineBatchSampler
        records = tuple(PlaceRecord(index, index, "Boston" if index % 2 == 0 else "London",
                                    f"{'Boston' if index % 2 == 0 else 'London'}:{index // 2:07d}") for index in range(120))
        bundle = load_edge_bundle(graph, frozen_path, expected_graph_places=6)
        for mode in MODES:
            runs[mode]["provenance"] = deepcopy(provenance)
            evaluations[mode]["provenance"] = deepcopy(provenance)
            run_dir = runtime / mode / "seed42"
            checkpoint = run_dir / "checkpoints/best.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"best checkpoint file hash fixture")
            runs[mode]["best_checkpoint_sha256"] = sha256_file(checkpoint)
            evaluations[mode]["checkpoint_sha256"] = sha256_file(checkpoint)
            write_finite_json(run_dir / "training_summary.json", runs[mode])
            write_finite_json(run_dir / "evaluation_best.json", evaluations[mode])
            for epoch in range(4):
                sampler = CounterMineBatchSampler(records, bundle, mode=mode, seed=42, epoch=epoch, repo_root=root)
                sampler.save_plan(run_dir / "batch_plans", repo_root=root)
        return frozen_path, graph

    def test_real_cpu_export_writes_three_figures_and_leaves_inputs_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.prepare_export_fixture(root)
            before = {path: sha256_file(path) for path in inputs}
            from countermine.training.step3a_config import read_json
            provenance = read_json(root / "cache/countermine_rgb/step3a/shared/preparation_summary.json")["provenance"]
            with patch("countermine.training.step3a_config.salad_identity", return_value={key: provenance[key] for key in ("salad_submodule_commit", "salad_source_hashes")}):
                snapshot = export_comparison(root)
            target = root / "docs/audits/step3a_edge_cobatching_metrics.json"
            self.assertEqual(snapshot, json.loads(target.read_text()))
            self.assertEqual(len(snapshot["figure_sha256"]), 3)
            for name, digest in snapshot["figure_sha256"].items():
                self.assertEqual(sha256_file(root / "docs/audits" / name), digest)
                self.assertEqual((root / "docs/audits" / name).read_bytes(),
                                 (root / "outputs/step3a" / name.removeprefix("step3a_")).read_bytes())
            self.assertEqual(before, {path: sha256_file(path) for path in inputs})
            self.assertNotIn(str(root), json.dumps(snapshot, allow_nan=False))

    def test_missing_results_or_graph_drift_stops_before_plotting(self):
        for graph_drift in (False, True):
            with self.subTest(graph_drift=graph_drift), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, graph = self.prepare_export_fixture(root)
                if graph_drift:
                    graph.write_text("changed frozen graph")
                else:
                    (root / "cache/countermine_rgb/step3a/baseline/seed42/evaluation_best.json").unlink()
                from countermine.training.step3a_config import read_json
                provenance = read_json(root / "cache/countermine_rgb/step3a/shared/preparation_summary.json")["provenance"]
                with patch("countermine.training.step3a_config.salad_identity", return_value={key: provenance[key] for key in ("salad_submodule_commit", "salad_source_hashes")}), patch("countermine.training.step3a_audit.create_plots") as plotting:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        export_comparison(root)
                    plotting.assert_not_called()
                self.assertFalse((root / "docs/audits/step3a_edge_cobatching_metrics.json").exists())
                self.assertFalse((root / "outputs/step3a").exists())

    def test_checkpoint_and_guided_file_drift_or_traversal_stops_before_publication(self):
        targets = ("checkpoint", "guided_csv", "place_order_csv", "checkpoint_traversal", "salad_traversal")
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.prepare_export_fixture(root)
                runtime = root / "cache/countermine_rgb/step3a"
                run = runtime / "baseline/seed42"
                if target == "checkpoint":
                    (run / "checkpoints/best.ckpt").write_bytes(b"changed checkpoint bytes")
                elif target == "guided_csv":
                    (run / "batch_plans/epoch_00_guided_pairs.csv").write_text("changed guided bytes")
                elif target == "place_order_csv":
                    (run / "batch_plans/epoch_00_place_order.csv").write_text("changed place order bytes")
                elif target == "checkpoint_traversal":
                    summary = json.loads((run / "training_summary.json").read_text())
                    summary["best_checkpoint"] = "../../shared/initial_state.pt"
                    write_finite_json(run / "training_summary.json", summary)
                else:
                    preparation = json.loads((runtime / "shared/preparation_summary.json").read_text())
                    preparation["provenance"]["salad_source_hashes"] = {"salad/../escaped.py": "a" * 64}
                    write_finite_json(runtime / "shared/preparation_summary.json", preparation)
                provenance = json.loads((runtime / "shared/preparation_summary.json").read_text())["provenance"]
                with patch("countermine.training.step3a_config.salad_identity", return_value={key: provenance[key] for key in ("salad_submodule_commit", "salad_source_hashes")}), patch("countermine.training.step3a_audit.create_plots") as plotting:
                    with self.assertRaises(ValueError):
                        export_comparison(root)
                    plotting.assert_not_called()
                self.assertFalse((root / "docs/audits/step3a_edge_cobatching_metrics.json").exists())

    def test_audit_and_comparison_cli_import_without_model_runtimes(self):
        source = '''
import importlib, importlib.abc, importlib.util, sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0].lower() in {'torch', 'torchvision', 'salad', 'lightglue', 'aliked', 'diffusers', 'transformers'}:
            raise AssertionError('Forbidden Step 3A audit import: ' + fullname)
sys.meta_path.insert(0, BlockModels())
importlib.import_module('countermine.training.step3a_audit')
spec = importlib.util.spec_from_file_location('comparison_cli', 'tools/33_compare_step3a.py')
spec.loader.exec_module(importlib.util.module_from_spec(spec))
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", source], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

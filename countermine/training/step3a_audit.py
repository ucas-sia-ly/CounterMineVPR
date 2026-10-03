"""CPU-only, strict comparison of the two completed Step 3A pilot conditions.

This module consumes scalar run artifacts. It never loads checkpoints or models.
Scientific publication requires all four complete epochs and best-checkpoint
retrieval evaluation for both conditions; smoke artifacts cannot be substituted.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import tempfile
from typing import Any, Mapping, Sequence

from countermine.training.step3a_runtime import (
    RUNTIME_FIELDS, critical_provenance, runtime_differences,
)

MODES = ("baseline", "countermine_q99_geo500")
RECALL_METRICS = tuple(f"{dataset}/R{k}" for dataset in
                       ("pitts30k_val", "pitts30k_test", "msls_val") for k in (1, 5, 10))
FIGURE_NAMES = ("recall_curves.png", "training_loss.png", "structural_exposure.png")
PROVENANCE_FIELDS = (
    "step2d_snapshot_sha256", "step2d_place_edges_sha256", "salad_submodule_commit",
    "step3a_code_hashes", "initial_state_sha256", "seed", "real_rgb_only",
    "synthetic_images_used", "runtime",
)
ZERO_INVARIANTS = (
    "accidentally_split_pair_count", "duplicate_place_count", "missing_place_count",
    "per_batch_city_composition_mismatch_count",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_portable_json(value: Any, location: str = "snapshot") -> None:
    """Reject non-finite numbers, absolute paths and embedded model/image data."""
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite number")
        return
    if isinstance(value, str):
        if value.startswith(("/", "\\")) or PureWindowsPath(value).is_absolute():
            raise ValueError(f"{location} contains an absolute path")
        return
    if isinstance(value, list) or isinstance(value, tuple):
        for index, item in enumerate(value):
            validate_portable_json(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        forbidden = {"state_dict", "checkpoint_tensors", "descriptors", "image_pixels", "image_tensors"}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} has a non-string key")
            if key.lower() in forbidden:
                raise ValueError(f"{location} embeds forbidden checkpoint/image data")
            validate_portable_json(key, f"{location} key")
            validate_portable_json(item, f"{location}.{key}")
        return
    raise ValueError(f"{location} contains an unsupported JSON value")


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")))
    if not isinstance(data, dict):
        raise ValueError(f"JSON object required: {Path(path).name}")
    validate_portable_json(data)
    return data


def atomic_bytes(path: str | Path, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="." + destination.name,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_finite_json(path: str | Path, payload: dict) -> None:
    validate_portable_json(payload)
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def guard_step3a(paths: Sequence[str | Path], repo_root: str | Path) -> None:
    """Allow only Step 3A namespaces, including after symlink resolution."""
    root = Path(repo_root).resolve()
    runtime = root / "cache/countermine_rgb/step3a"
    output = root / "outputs/step3a"
    audits = root / "docs/audits"
    for item in paths:
        path = Path(item).resolve()
        runtime_or_output = any(path.is_relative_to(base) for base in (runtime, output))
        curated = path.parent == audits and path.name.startswith("step3a_")
        if not (runtime_or_output or curated):
            raise ValueError("Step 3A destinations must stay in step3a cache/output or docs/audits/step3a_*")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA256 digest")
    return value


def _validate_provenance(provenance: Mapping, reference: Mapping | None = None) -> None:
    if not isinstance(provenance, dict) or any(key not in provenance for key in PROVENANCE_FIELDS):
        raise ValueError("incomplete Step 3A provenance")
    for key in ("step2d_snapshot_sha256", "step2d_place_edges_sha256", "initial_state_sha256"):
        _digest(provenance[key], key)
    if provenance["seed"] != 42 or provenance["real_rgb_only"] is not True or provenance["synthetic_images_used"] is not False:
        raise ValueError("Step 3A requires seed 42 and original real RGB only")
    if not isinstance(provenance["salad_submodule_commit"], str) or not provenance["salad_submodule_commit"]:
        raise ValueError("SALAD submodule identity is required")
    runtime = provenance["runtime"]
    if not isinstance(runtime, dict) or any(key not in runtime for key in RUNTIME_FIELDS):
        raise ValueError("complete observed Step 3A runtime metadata is required")
    code_hashes = provenance["step3a_code_hashes"]
    if not isinstance(code_hashes, dict) or not code_hashes:
        raise ValueError("Step 3A code hashes are required")
    for name, digest in code_hashes.items():
        if not isinstance(name, str) or Path(name).is_absolute() or PureWindowsPath(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise ValueError("Step 3A code names must be relative")
        _digest(digest, name)
    for name, digest in provenance.get("salad_source_hashes", {}).items():
        if (not isinstance(name, str) or not name.startswith("salad/") or Path(name).is_absolute() or
                PureWindowsPath(name).is_absolute() or ".." in Path(name).parts or "\\" in name):
            raise ValueError("SALAD source names must be safe relative paths inside salad/")
        _digest(digest, name)
    if reference is not None and critical_provenance(provenance) != critical_provenance(reference):
        raise ValueError("run provenance differs from shared preparation")


def _recalls(metrics: Mapping) -> dict[str, float]:
    if not isinstance(metrics, dict):
        raise ValueError("retrieval metrics must be a dictionary")
    result = {name: _number(metrics[name], name) for name in RECALL_METRICS}
    for dataset in ("pitts30k_val", "pitts30k_test", "msls_val"):
        values = [result[f"{dataset}/R{k}"] for k in (1, 5, 10)]
        if not (0 <= values[0] <= values[1] <= values[2] <= 100):
            raise ValueError("recalls must be ordered values between 0 and 100")
    return result


def _validate_plan(plan: dict, mode: str, epoch: int, provenance: Mapping) -> None:
    if plan["seed"] != 42 or plan["epoch"] != epoch:
        raise ValueError("batch-plan seed/epoch differs")
    if plan["graph_sha256"] != provenance["step2d_place_edges_sha256"]:
        raise ValueError("batch-plan graph hash differs")
    count = _integer(plan["dataset_place_count"], "dataset_place_count", 1)
    batch_sizes = plan["batch_sizes"]
    if not isinstance(batch_sizes, list) or not batch_sizes or any(
            isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 60 for size in batch_sizes):
        raise ValueError("batch sizes must implement 60 places with drop_last=False")
    if batch_sizes[:-1] != [60] * (len(batch_sizes) - 1) or sum(batch_sizes) != count:
        raise ValueError("batch sizes do not preserve all place slots")
    if plan["batch_count"] != len(batch_sizes):
        raise ValueError("batch count differs from batch sizes")
    for key in ZERO_INVARIANTS:
        if _integer(plan[key], key) != 0:
            raise ValueError(f"sampler invariant failed: {key}")
    for key in ("baseline_sequence_sha256", "treatment_sequence_sha256", "dataset_place_order_sha256"):
        _digest(plan[key], key)
    guided = _integer(plan["greedy_matched_pair_count"], "guided pairs")
    if _integer(plan["unique_guided_place_count"], "unique guided places") != 2 * guided:
        raise ValueError("guided matching must be vertex-disjoint")
    if plan["intentionally_guided_pairs_cooccur_count"] != guided:
        raise ValueError("guided endpoints did not all co-occur")
    city_guided = [_integer(plan[key], key) for key in ("boston_guided_pair_count", "london_guided_pair_count")]
    if sum(city_guided) != guided:
        raise ValueError("guided pairs must be Boston/London same-city only")
    if plan["max_guided_pairs_per_place"] != (1 if guided else 0):
        raise ValueError("each place may occur in at most one guided pair")
    if mode == "baseline" and (guided or plan["baseline_sequence_sha256"] != plan["treatment_sequence_sha256"]):
        raise ValueError("baseline must reproduce sequential place grouping")
    if mode != "baseline" and not guided:
        raise ValueError("treatment did not intentionally guide any edges")
    eligible = _integer(plan["eligible_q99_geo500_same_city_edge_count"], "eligible guided edges")
    if guided > eligible:
        raise ValueError("guided count exceeds eligible edges")
    exposure = plan["exposure"]
    for key in ("core_q95", "core_q99", "core_q99_geo500",
                "unique_q99_geo500_graph_edges_exposed", "unique_structural_places_exposed"):
        _integer(exposure[key], key)
    if not exposure["core_q99_geo500"] <= exposure["core_q99"] <= exposure["core_q95"]:
        raise ValueError("nested structural exposures differ")
    if exposure["core_q99_geo500"] < guided:
        raise ValueError("exposure audit omits guided edges")
    marginal = plan["marginal_exposure"]
    appearances = marginal["per_city_place_appearances"]
    if not isinstance(appearances, dict) or not appearances:
        raise ValueError("per-city marginal place appearances are required")
    for city, value in appearances.items():
        _integer(value, city)
    if sum(appearances.values()) != count:
        raise ValueError("per-city marginal counts differ from dataset place count")
    _digest(marginal["place_multiset_sha256"], "place_multiset_sha256")
    for city in ("Boston", "London"):
        _digest(marginal["per_city_sorted_place_uids_sha256"][city], city + " place ID set")


def build_comparison(shared: dict, runs: Mapping[str, dict], evaluations: Mapping[str, dict],
                     plans: Mapping[str, Sequence[dict]]) -> dict:
    """Build a descriptive snapshot, stopping on incomplete or mixed run evidence."""
    for payload in (shared, runs, evaluations, plans):
        validate_portable_json(dict(payload))
    provenance = shared["provenance"]
    _validate_provenance(provenance)
    if shared.get("complete") is not True:
        raise ValueError("shared preparation must be complete")
    configuration = shared["configuration"]
    from countermine.training.step3a_config import frozen_configuration, TREATMENT_DEFINITION
    frozen_configuration_json = json.loads(json.dumps(frozen_configuration()))
    if configuration != frozen_configuration_json:
        raise ValueError("training configuration differs from the frozen original SALAD pilot")
    if shared["treatment_definition"] != TREATMENT_DEFINITION:
        raise ValueError("treatment must use binary same-city q99_geo500 edges only")
    mapping = shared["mapping_audit"]
    if (mapping["unmapped_graph_places"] != 0 or mapping["one_to_one_mapping"] is not True or
            mapping["mapped_graph_places"] != mapping["countermine_graph_places"]):
        raise ValueError("CounterMine graph places must map one-to-one into the training dataset")
    if set(runs) != set(MODES) or set(evaluations) != set(MODES) or set(plans) != set(MODES):
        raise ValueError("exactly baseline and countermine_q99_geo500 are required")
    training, evaluation, sampler = {}, {}, {}
    for mode in MODES:
        run, evaluated = runs[mode], evaluations[mode]
        if run.get("complete") is not True or run.get("smoke") is not False:
            raise ValueError("comparison requires completed full runs; smoke metrics are excluded")
        if run.get("mode") != mode or run.get("seed") != 42:
            raise ValueError("training mode/seed differs")
        _validate_provenance(run["provenance"], provenance)
        epochs = run["epochs"]
        if not isinstance(epochs, list) or [entry["epoch"] for entry in epochs] != list(range(4)):
            raise ValueError("comparison requires exactly four complete epochs per condition")
        clean_epochs = []
        for entry in epochs:
            metrics = _recalls(entry["metrics"])
            metrics["loss"] = _number(entry["metrics"]["loss"], "training loss")
            metrics["b_acc"] = _number(entry["metrics"]["b_acc"], "b_acc")
            clean_epochs.append({"epoch": entry["epoch"], "metrics": metrics})
        best = max(range(4), key=lambda index: clean_epochs[index]["metrics"]["pitts30k_val/R1"])
        if run["best_epoch"] != best:
            raise ValueError("best checkpoint must be selected only by Pitts30k val R1")
        final_loss = _number(run["final_train_loss"], "final_train_loss")
        average_b_acc = _number(run["average_b_acc"], "average_b_acc")
        if not math.isclose(final_loss, clean_epochs[-1]["metrics"]["loss"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("final train loss differs from final full epoch")
        if not math.isclose(average_b_acc, sum(e["metrics"]["b_acc"] for e in clean_epochs) / 4,
                            rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("average b_acc differs from full epoch metrics")
        training[mode] = {"epochs": clean_epochs, "best_epoch": best,
                          "final_train_loss": final_loss, "average_b_acc": average_b_acc,
                          "runtime": deepcopy(run["provenance"]["runtime"])}
        if evaluated.get("complete") is not True or evaluated.get("mode") != mode or evaluated.get("seed") != 42:
            raise ValueError("completed evaluation of both full conditions is required")
        if evaluated.get("checkpoint_selection") != "best_pitts30k_val_R1" or evaluated.get("best_epoch") != best:
            raise ValueError("primary evaluation must use best Pitts30k val R1 checkpoint")
        _validate_provenance(evaluated["provenance"], provenance)
        digest = _digest(evaluated["checkpoint_sha256"], "evaluation checkpoint")
        if digest != run["best_checkpoint_sha256"]:
            raise ValueError("evaluation checkpoint differs from the selected training checkpoint")
        evaluation[mode] = {"checkpoint_selection": "best_pitts30k_val_R1", "best_epoch": best,
                            "checkpoint_sha256": digest, "metrics": _recalls(evaluated["metrics"]),
                            "runtime": deepcopy(evaluated["provenance"]["runtime"])}
        if len(plans[mode]) != 4:
            raise ValueError("every full epoch requires a batch-plan audit")
        sampler[mode] = []
        for epoch, plan in enumerate(plans[mode]):
            _validate_plan(plan, mode, epoch, provenance)
            if (plan["dataset_place_count"] != mapping["total_training_places"] or
                    plan["marginal_exposure"]["per_city_place_appearances"] != mapping["per_city_training_places"]):
                raise ValueError("epoch place exposures differ from the shared mapping audit")
            sampler[mode].append(deepcopy(plan))
    exposure_comparison = []
    for epoch in range(4):
        baseline, treatment = sampler[MODES[0]][epoch], sampler[MODES[1]][epoch]
        if (baseline["baseline_sequence_sha256"] != treatment["baseline_sequence_sha256"] or
                baseline["dataset_place_order_sha256"] != treatment["dataset_place_order_sha256"]):
            raise ValueError("baseline dataset ordering differs between conditions")
        for key in ("dataset_place_count", "batch_count", "batch_sizes", "eligible_q99_geo500_same_city_edge_count"):
            if baseline[key] != treatment[key]:
                raise ValueError(f"marginal/batch audit differs between conditions: {key}")
        for key in ("per_city_place_appearances", "per_city_sorted_place_uids_sha256", "place_multiset_sha256"):
            if baseline["marginal_exposure"][key] != treatment["marginal_exposure"][key]:
                raise ValueError(f"marginal place exposure differs between conditions: {key}")
        base = baseline["exposure"]["core_q99_geo500"]
        treated = treatment["exposure"]["core_q99_geo500"]
        if treated <= base:
            raise ValueError("sampler error: treatment did not increase q99_geo500 structural exposure")
        exposure_comparison.append({"epoch": epoch, "baseline_q99_geo500_co_batched_edges": base,
                                    "countermine_q99_geo500_co_batched_edges": treated,
                                    "countermine_to_baseline_ratio": treated / base if base else None,
                                    "baseline_zero": base == 0, "exposure_gain": treated - base,
                                    "intentionally_guided_pairs": treatment["greedy_matched_pair_count"],
                                    "unique_guided_places": treatment["unique_guided_place_count"],
                                    "identical_marginal_place_exposure": True})
    deltas = {metric: evaluation[MODES[1]]["metrics"][metric] - evaluation[MODES[0]]["metrics"][metric]
              for metric in RECALL_METRICS}
    snapshot = {"schema_version": 1, "stage": "step3a_edge_cobatching", "complete": True,
                "provenance": deepcopy(provenance), "training_configuration": deepcopy(configuration),
                "treatment_definition": deepcopy(shared["treatment_definition"]),
                "mapping_audit": deepcopy(shared["mapping_audit"]),
                "dataset_metadata": deepcopy(shared["dataset_metadata"]), "sampler_audit": sampler,
                "training_exposure_comparison": exposure_comparison, "training": training,
                "evaluation": {**evaluation, "descriptive_countermine_minus_baseline": deltas},
                "runtime_comparison": {
                    "differences_are_descriptive": True,
                    "training_from_preparation": {
                        mode: runtime_differences(provenance["runtime"], training[mode]["runtime"])
                        for mode in MODES},
                    "evaluation_from_training": {
                        mode: runtime_differences(training[mode]["runtime"], evaluation[mode]["runtime"])
                        for mode in MODES},
                    "between_training_conditions": runtime_differences(
                        training[MODES[0]]["runtime"], training[MODES[1]]["runtime"]),
                    "between_evaluation_conditions": runtime_differences(
                        evaluation[MODES[0]]["runtime"], evaluation[MODES[1]]["runtime"]),
                },
                "scope": {"single_seed_pilot": True, "statistical_significance_claimed": False,
                          "description": "single-seed Step 3A pilot; descriptive retrieval deltas only",
                          "intervention": "place co-occurrence only; original SALAD architecture, loss and miner"}}
    validate_portable_json(snapshot)
    return snapshot


def create_plots(snapshot: dict, output_dir: str | Path, *, repo_root: str | Path) -> dict:
    """Draw three separate figures from already validated complete epoch metrics."""
    output_dir = Path(output_dir)
    guard_step3a([output_dir / name for name in FIGURE_NAMES], repo_root)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import io

    plot_specs = (
        (FIGURE_NAMES[0], "pitts30k_val/R1", "Pitts30k val R@1", "Pitts30k validation recall"),
        (FIGURE_NAMES[1], "loss", "Training loss", "Original SALAD training loss"),
        (FIGURE_NAMES[2], None, "q99_geo500 co-batched place edges", "Structural place co-exposure"),
    )
    for name, metric, ylabel, title in plot_specs:
        figure = plt.figure(figsize=(7, 4.5))
        axis = figure.add_subplot(111)
        for mode, label, color in ((MODES[0], "Baseline", "#315b88"),
                                   (MODES[1], "CounterMine", "#b65d3a")):
            entries = snapshot["training"][mode]["epochs"]
            values = ([entry["metrics"][metric] for entry in entries] if metric is not None else
                      [entry["exposure"]["core_q99_geo500"] for entry in snapshot["sampler_audit"][mode]])
            axis.plot([entry["epoch"] + 1 for entry in entries], values, marker="o", color=color, label=label)
        axis.set(xlabel="Epoch", ylabel=ylabel, title=title + " — single-seed Step 3A pilot", xticks=[1, 2, 3, 4])
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=.2)
        axis.legend(frameon=False)
        figure.tight_layout()
        with io.BytesIO() as buffer:
            figure.savefig(buffer, format="png", dpi=160)
            guard_step3a([output_dir / name], repo_root)
            atomic_bytes(output_dir / name, buffer.getvalue())
        plt.close(figure)
    return {"step3a_" + name: sha256_file(output_dir / name) for name in FIGURE_NAMES}


def _relative_file(base: Path, name: str, label: str) -> Path:
    if (not isinstance(name, str) or not name or Path(name).is_absolute() or
            PureWindowsPath(name).is_absolute() or ".." in Path(name).parts or "\\" in name):
        raise ValueError(f"{label} must be a safe relative artifact path")
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"{label} escapes its artifact directory")
    return path


def _validate_guided_csv(path: Path, plan: dict) -> None:
    expected_columns = ["dataset_index_a", "dataset_index_b", "canonical_place_uid_a",
                        "canonical_place_uid_b", "city_id", "batch_index"]
    if sha256_file(path) != _digest(plan["guided_pairs_csv_sha256"], "guided pair CSV"):
        raise ValueError("guided pair CSV SHA256 differs from its epoch summary")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_columns:
            raise ValueError("guided pair CSV schema differs")
        rows = list(reader)
    if len(rows) != plan["greedy_matched_pair_count"]:
        raise ValueError("guided CSV pair count differs from its epoch summary")
    used_indices, used_uids = set(), set()
    city_counts = {"Boston": 0, "London": 0}
    for row in rows:
        city = row["city_id"]
        if city not in city_counts:
            raise ValueError("guided CSV contains a non-target city")
        city_counts[city] += 1
        batch = int(row["batch_index"])
        if not 0 <= batch < plan["batch_count"]:
            raise ValueError("guided CSV has an invalid batch index")
        for endpoint in ("a", "b"):
            index = int(row["dataset_index_" + endpoint])
            uid = row["canonical_place_uid_" + endpoint]
            if not 0 <= index < plan["dataset_place_count"] or index in used_indices or uid in used_uids:
                raise ValueError("guided CSV matching duplicates a place or has an invalid place index")
            if not uid.startswith(city + ":") or len(uid.split(":")[-1]) != 7 or not uid.split(":")[-1].isdigit():
                raise ValueError("guided CSV endpoints must be canonical same-city place IDs")
            used_indices.add(index)
            used_uids.add(uid)
    if city_counts["Boston"] != plan["boston_guided_pair_count"] or city_counts["London"] != plan["london_guided_pair_count"]:
        raise ValueError("guided CSV per-city counts differ from its epoch summary")
    return rows


def _reproduce_plan(order_path: Path, plan: dict, mode: str, bundle, guided_rows: list) -> None:
    """Reconstruct scalar plan evidence from the saved epoch dataset identities."""
    if sha256_file(order_path) != _digest(plan["place_order_csv_sha256"], "epoch place order"):
        raise ValueError("epoch place-order CSV SHA256 differs from its summary")
    from countermine.training.gsv_place_mapping import PlaceRecord
    from countermine.training.countermine_batch_sampler import CounterMineBatchSampler
    with order_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["dataset_index", "internal_place_id", "city_id", "canonical_place_uid"]:
            raise ValueError("epoch place-order CSV schema differs")
        records = tuple(PlaceRecord(int(row["dataset_index"]), int(row["internal_place_id"]),
                                    row["city_id"], row["canonical_place_uid"]) for row in reader)
    reproduced = CounterMineBatchSampler(records, bundle, mode=mode, seed=42, epoch=plan["epoch"]).plan
    for key, value in reproduced.summary.items():
        if plan.get(key) != value:
            raise ValueError(f"epoch batch-plan reproduction differs: {key}")
    expected_guided = [{key: str(value) for key, value in asdict(pair).items()}
                       for pair in reproduced.guided_pairs]
    if guided_rows != expected_guided:
        raise ValueError("guided pair CSV differs from deterministic epoch plan")


def export_comparison(repo_root: str | Path, *, seed: int = 42, runtime_dir: str | Path | None = None,
                      output_dir: str | Path | None = None, snapshot_path: str | Path | None = None) -> dict:
    """Validate frozen inputs and two full runs before publishing any result."""
    root = Path(repo_root).resolve()
    if seed != 42:
        raise ValueError("primary Step 3A seed must be 42")
    runtime = Path(runtime_dir) if runtime_dir is not None else root / "cache/countermine_rgb/step3a"
    output = Path(output_dir) if output_dir is not None else root / "outputs/step3a"
    target = Path(snapshot_path) if snapshot_path is not None else root / "docs/audits/step3a_edge_cobatching_metrics.json"
    guard_step3a([runtime / "edge_cobatching_metrics.json", output / "recall_curves.png", target], root)
    if target.parent.resolve() != (root / "docs/audits").resolve() or not target.name.startswith("step3a_") or target.suffix != ".json":
        raise ValueError("curated comparison must be docs/audits/step3a_*.json")
    shared_path = runtime / "shared/preparation_summary.json"
    shared = read_json(shared_path)
    provenance = shared["provenance"]
    _validate_provenance(provenance)
    frozen_snapshot = root / "docs/audits/step2d_full_countermine_metrics.json"
    graph = root / "cache/countermine_rgb/step2d/place_edges.csv"
    frozen = read_json(frozen_snapshot)
    if frozen.get("complete") is not True:
        raise ValueError("Step 2D scientific snapshot must be complete")
    if shared["mapping_audit"]["countermine_graph_places"] != frozen["population"]["unique_places"]:
        raise ValueError("mapping does not cover every frozen Step 2D graph place")
    graph_digest = frozen["graph_artifact_hashes"]["place_edges.csv"]
    if isinstance(graph_digest, dict):
        graph_digest = graph_digest["sha256"]
    if sha256_file(graph) != graph_digest or graph_digest != provenance["step2d_place_edges_sha256"]:
        raise ValueError("frozen Step 2D place graph hash does not match")
    checks = {frozen_snapshot: provenance["step2d_snapshot_sha256"], graph: graph_digest,
              runtime / "shared/initial_state.pt": provenance["initial_state_sha256"]}
    from countermine.training.step3a_config import validate_source_hashes
    source_compatibility = validate_source_hashes(
        provenance["step3a_code_hashes"], root, check_inventory=False)
    checks.update({_relative_file(root, name, "Step 3A source"): digest
                   for name, digest in source_compatibility["executed_code_hashes"].items()})
    if "compatibility_manifest" in source_compatibility:
        checks[_relative_file(root, source_compatibility["compatibility_manifest"], "compatibility manifest")] = source_compatibility["compatibility_manifest_sha256"]
    salad_hashes = provenance.get("salad_source_hashes")
    if not isinstance(salad_hashes, dict) or not salad_hashes:
        raise ValueError("SALAD source hashes are required before scientific export")
    for name, digest in salad_hashes.items():
        if not isinstance(name, str) or not name.startswith("salad/"):
            raise ValueError("SALAD source hash names must be inside salad/")
        checks[_relative_file(root, name, "SALAD source")] = _digest(digest, "SALAD source")
    from countermine.training.step3a_config import salad_identity
    identity = salad_identity(root)
    if identity["salad_submodule_commit"] != provenance["salad_submodule_commit"] or identity["salad_source_hashes"] != salad_hashes:
        raise ValueError("current SALAD identity differs from shared preparation")
    dataset_metadata = shared["dataset_metadata"]
    if dataset_metadata["dataset_root"] != "data/GSVCities":
        raise ValueError("dataset metadata must bind the exact upstream GSVCities root")
    root_digest = hashlib.sha256(str((root / "data/GSVCities").resolve()).encode()).hexdigest()
    if dataset_metadata["resolved_dataset_root_sha256"] != root_digest:
        raise ValueError("resolved GSVCities root identity differs from shared preparation")
    metadata_hashes = dataset_metadata["metadata_sha256"]
    if not isinstance(metadata_hashes, dict) or not metadata_hashes:
        raise ValueError("dataset metadata file hashes are required")
    for name, digest in metadata_hashes.items():
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise ValueError("dataset metadata hash names must be safe relative paths")
        checks[root / name] = _digest(digest, "dataset metadata")
    for path, expected in checks.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Step 3A provenance mismatch: {path.name}")
    from countermine.training.countermine_batch_sampler import load_edge_bundle
    bundle = load_edge_bundle(graph, frozen_snapshot, expected_graph_places=frozen["population"]["unique_places"])
    runs, evaluations, plans = {}, {}, {}
    sources = [shared_path]
    for mode in MODES:
        run_dir = runtime / mode / f"seed{seed}"
        train_path, eval_path = run_dir / "training_summary.json", run_dir / "evaluation_best.json"
        sources.extend((train_path, eval_path))
        runs[mode], evaluations[mode] = read_json(train_path), read_json(eval_path)
        checkpoint_path = _relative_file(run_dir, runs[mode]["best_checkpoint"], "best checkpoint")
        checkpoint_digest = _digest(runs[mode]["best_checkpoint_sha256"], "best checkpoint")
        if sha256_file(checkpoint_path) != checkpoint_digest:
            raise ValueError("current best checkpoint SHA256 differs from its training summary")
        checks[checkpoint_path] = checkpoint_digest
        plans[mode] = []
        for epoch in range(4):
            plan_path = run_dir / "batch_plans" / f"epoch_{epoch:02d}_summary.json"
            sources.append(plan_path)
            plan = read_json(plan_path)
            pairs_path = run_dir / "batch_plans" / f"epoch_{epoch:02d}_guided_pairs.csv"
            guided_rows = _validate_guided_csv(pairs_path, plan)
            order_path = run_dir / "batch_plans" / f"epoch_{epoch:02d}_place_order.csv"
            _reproduce_plan(order_path, plan, mode, bundle, guided_rows)
            sources.extend((pairs_path, order_path))
            plans[mode].append(plan)
    source_hashes = {path: sha256_file(path) for path in sources}
    result = build_comparison(shared, runs, evaluations, plans)
    result["runtime_artifact_sha256"] = {path.relative_to(runtime).as_posix(): digest
                                        for path, digest in source_hashes.items()}
    result["figure_sha256"] = create_plots(result, output, repo_root=root)
    # Do not publish artifacts from inputs that changed during plotting.
    for path, expected in {**checks, **source_hashes}.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Step 3A input changed during export: {path.name}")
    destinations = [target, runtime / "edge_cobatching_metrics.json", output / "source_compatibility.json"]
    destinations.extend(root / "docs/audits" / ("step3a_" + name) for name in FIGURE_NAMES)
    guard_step3a(destinations, root)
    for name in FIGURE_NAMES:
        atomic_bytes(root / "docs/audits" / ("step3a_" + name), (output / name).read_bytes())
    write_finite_json(runtime / "edge_cobatching_metrics.json", result)
    write_finite_json(target, result)
    write_finite_json(output / "source_compatibility.json", source_compatibility)
    return result

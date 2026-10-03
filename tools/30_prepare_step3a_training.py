#!/usr/bin/env python3
"""Validate the frozen training relation and create one shared initialization."""
import argparse
from dataclasses import asdict
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from countermine.training.step3a_config import (
    SEED, MODES, MODEL_CONFIG, TREATMENT_DEFINITION, resolved_paths,
    enter_salad, seed_runtime, create_model, strict_load_initial_state, frozen_configuration,
    code_hashes, sha256_file, read_json, write_json, guard_step3a,
    atomic_torch_save, check_shared_initialization_transaction,
)
from countermine.training.gsv_place_mapping import build_place_mapping
from countermine.training.countermine_batch_sampler import CounterMineBatchSampler, load_edge_bundle
from countermine.training.runtime_preflight import run_preflight
from countermine.training.step3a_runtime import critical_provenance


def prepare_shared_initialization(model, paths, identity, *, seed=SEED, torch_module=None):
    """Verify and reuse a complete pair, or publish and verify a new tensor first."""
    if seed != SEED:
        raise ValueError("The primary Step 3A pilot is frozen at seed 42")
    existing = check_shared_initialization_transaction(paths["shared"], root=paths["root"])
    state_path = guard_step3a(paths["shared"] / "initial_state.pt", paths["root"])
    summary_path = paths["shared"] / "initial_state_summary.json"
    if existing:
        state_summary = read_json(summary_path)
        if any(state_summary.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing initialization was produced with different SALAD source")
        if state_summary.get("seed") != seed:
            raise ValueError("Existing initialization was produced with a different seed")
        strict_load_initial_state(model, state_path, state_summary, torch_module=torch_module)
        return state_summary
    if torch_module is None:
        import torch as torch_module
    atomic_torch_save(model.state_dict(), state_path, torch_module=torch_module, root=paths["root"])
    state_summary = {
        "model_configuration": MODEL_CONFIG, **identity,
        "initial_state_sha256": sha256_file(state_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "torch_version": str(torch_module.__version__), "seed": seed,
    }
    # A failed round trip intentionally leaves an incomplete final pair for
    # inspection. No JSON can certify an unverified serialized initialization.
    strict_load_initial_state(model, state_path, state_summary, torch_module=torch_module)
    write_json(summary_path, state_summary, root=paths["root"])
    return state_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED, choices=[SEED])
    args = parser.parse_args(argv)
    paths = resolved_paths(ROOT)
    check_shared_initialization_transaction(paths["shared"], root=ROOT)
    report = run_preflight(stage="prepare", root=ROOT, save_report=False, progress=True)
    edges = load_edge_bundle(paths["graph"], paths["snapshot"])
    metadata = report["dataset_metadata"]
    identity = report["salad_identity"]
    print("[4/6] building place mapping", flush=True)
    enter_salad(paths)
    seed_runtime(args.seed)
    import torch
    from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
    from copy import deepcopy
    from countermine.training.step3a_config import DATA_CONFIG
    dm = GSVCitiesDataModule(**deepcopy(DATA_CONFIG))
    dm.setup("fit")
    # Exactly the upstream setup + first train_dataloader reload sequence.
    dm.reload()
    records, mapping = build_place_mapping(dm.train_dataset, edges.graph_place_uids)
    baseline = CounterMineBatchSampler(records, edges, mode=MODES[0], seed=args.seed)
    treatment = CounterMineBatchSampler(records, edges, mode=MODES[1], seed=args.seed)
    provenance = {
        "step2d_snapshot_sha256": edges.snapshot_sha256,
        "step2d_place_edges_sha256": edges.graph_sha256,
        **identity, "step3a_code_hashes": code_hashes(ROOT), "seed": args.seed,
        "real_rgb_only": True, "synthetic_images_used": False,
        "runtime": report["runtime"],
    }
    existing = paths["shared"] / "preparation_summary.json"
    if existing.exists():
        old = read_json(existing)
        critical = critical_provenance(provenance)
        if {key: critical_provenance(old["provenance"]).get(key) for key in critical} != critical:
            raise ValueError("Existing shared preparation has different provenance; do not overwrite it")
        if old["dataset_metadata"] != metadata or old["mapping_audit"] != mapping:
            raise ValueError("Existing shared dataset mapping/metadata differs")
    print("[5/6] creating/verifying shared initialization", flush=True)
    seed_runtime(args.seed)
    model = create_model()
    state_summary = prepare_shared_initialization(model, paths, identity, seed=args.seed, torch_module=torch)
    provenance["initial_state_sha256"] = state_summary["initial_state_sha256"]
    if existing.exists() and critical_provenance(old["provenance"]) != critical_provenance(provenance):
        raise ValueError("Existing shared preparation has different initialization provenance; do not overwrite it")
    print("[6/6] freezing epoch-0 plans", flush=True)
    plans = {}
    for mode, sampler in zip(MODES, (baseline, treatment)):
        sampler.save_plan(paths["shared"] / "batch_plans" / mode)
        plans[mode] = sampler.plan.summary
    destination = guard_step3a(paths["shared"] / "place_mapping.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    preparation = {
        "complete": True, "provenance": provenance, "configuration": frozen_configuration(),
        "treatment_definition": TREATMENT_DEFINITION, "mapping_audit": mapping,
        "dataset_metadata": metadata, "epoch0_plans": plans,
        "initial_state_summary": state_summary,
    }
    write_json(existing, preparation)
    print("Training place count:", len(records))
    print("Mapped CounterMine place count:", len(edges.graph_place_uids))
    for city in ("Boston", "London"):
        print(f"Eligible {city} q99_geo500 edge count:", len(edges.eligible_edges(city)))
    for mode, sampler in zip(MODES, (baseline, treatment)):
        print(f"Epoch-0 {mode} q99_geo500 exposure:", sampler.plan.summary["exposure"]["core_q99_geo500"])
    print("Shared initialization SHA256:", state_summary["initial_state_sha256"])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        raise

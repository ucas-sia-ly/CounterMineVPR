#!/usr/bin/env python3
"""Evaluate the selected checkpoint with standard upstream global retrieval."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from countermine.training.step3a_config import (
    MODES, SEED, DATA_CONFIG, resolved_paths, validate_dataset_paths, enter_salad,
    seed_runtime, create_model, validate_source_hashes, salad_identity, sha256_file, read_json,
    write_json, guard_step3a,
)
from countermine.training.runtime_preflight import run_preflight, configure_cublas_workspace
from countermine.training.cached_dinov2 import cached_dinov2_runtime
from countermine.training.step3a_runtime import (
    RECALL_KEYS, finite_metrics, critical_provenance, runtime_differences,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--seed", type=int, default=SEED, choices=[SEED])
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    args = parser.parse_args(argv)
    cublas_workspace_config = configure_cublas_workspace()
    paths = resolved_paths(ROOT)
    preflight = run_preflight(stage="evaluate", root=ROOT, save_report=False, progress=False)
    metadata = validate_dataset_paths(paths)
    # The runtime gate validates the frozen source; retrieval itself uses no graph.
    preparation = read_json(paths["shared"] / "preparation_summary.json")
    provenance = deepcopy(preparation["provenance"])
    provenance["runtime"] = deepcopy(preflight["runtime"])
    source_compatibility = validate_source_hashes(provenance["step3a_code_hashes"], ROOT)
    if any(provenance[key] != value for key, value in salad_identity(ROOT).items()):
        raise ValueError("Evaluation SALAD source differs from training")
    if preparation["dataset_metadata"] != metadata:
        raise ValueError("Evaluation dataset metadata differs from training")
    run = guard_step3a(paths["runtime"] / args.mode / "seed42")
    training = read_json(run / "training_summary.json")
    if (not training["complete"] or training["smoke"] or
            critical_provenance(training["provenance"]) != critical_provenance(provenance)):
        raise ValueError("Only completed full scientific training conditions may be evaluated")
    checkpoint_path = guard_step3a(run / training[args.checkpoint + "_checkpoint"])
    if not checkpoint_path.is_relative_to(run) or sha256_file(checkpoint_path) != training[args.checkpoint + "_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint path/hash mismatch")
    write_json(run / f"evaluation_{args.checkpoint}_source_compatibility.json", source_compatibility)
    enter_salad(paths)
    seed_runtime(args.seed)
    import torch
    import pytorch_lightning as pl
    from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
    from countermine.training.salad_datamodule import verify_validation_images
    if not torch.cuda.is_available():
        raise RuntimeError("Standard SALAD evaluation requires one available CUDA GPU")
    dm = GSVCitiesDataModule(**deepcopy(DATA_CONFIG))
    dm.setup("fit")
    verify_validation_images(dm)
    with cached_dinov2_runtime(torch_module=torch) as model_loading:
        model = create_model()
    model_loading["cublas_workspace_config"] = cublas_workspace_config
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loaded = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("Evaluation checkpoint strict load failed")
    write_json(run / f"evaluation_{args.checkpoint}_model_loading.json", model_loading)
    trainer = pl.Trainer(accelerator="gpu", devices=1, num_nodes=1, precision="16-mixed",
                         deterministic=True, logger=False, enable_checkpointing=False,
                         default_root_dir=str(run / "evaluation"), num_sanity_val_steps=0)
    results = trainer.validate(model=model, datamodule=dm, verbose=True)
    merged = {key.split("/dataloader_idx_")[0]: value for result in results for key, value in result.items()}
    metrics = finite_metrics({key: merged[key] for key in RECALL_KEYS})
    report = {
        "complete": True, "mode": args.mode, "seed": args.seed, "provenance": provenance,
        "checkpoint_selection": "best_pitts30k_val_R1" if args.checkpoint == "best" else "last",
        "best_epoch": training["best_epoch"], "checkpoint_sha256": sha256_file(checkpoint_path),
        "metrics": metrics, "global_retrieval_only": True,
        "runtime_differences_from_training": runtime_differences(
            training["provenance"]["runtime"], provenance["runtime"]),
    }
    write_json(run / f"evaluation_{args.checkpoint}.json", report)
    for key, value in metrics.items():
        print(f"{args.mode} {key}: {value:.6f}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        raise SystemExit(f"error: {error}") from None

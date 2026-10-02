#!/usr/bin/env python3
"""Train the frozen SALAD model with place-composition-only intervention."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from countermine.training.step3a_config import (
    MODES, SEED, TRAINER_CONFIG, resolved_paths, validate_dataset_paths, enter_salad,
    seed_runtime, create_model, strict_load_initial_state, check_preparation, read_json,
    guard_step3a, write_json,
)
from countermine.training.countermine_batch_sampler import load_edge_bundle
from countermine.training.step3a_runtime import require_smokes, create_training_audit_callback, training_summary


def batch_limit(value):
    if value.isdigit():
        return int(value)
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("Use a positive batch count or fraction in (0,1]")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--seed", type=int, default=SEED, choices=[SEED])
    parser.add_argument("--max-epochs", type=int, default=4)
    parser.add_argument("--limit-train-batches", type=batch_limit, default=1.0)
    parser.add_argument("--limit-val-batches", type=batch_limit, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        if (args.max_epochs, args.limit_train_batches, args.limit_val_batches) != (1, 3, 2):
            parser.error("Smoke checks require --max-epochs 1 --limit-train-batches 3 --limit-val-batches 2")
    elif (args.max_epochs != 4 or not isinstance(args.limit_train_batches, float)
          or not isinstance(args.limit_val_batches, float)
          or args.limit_train_batches != 1.0 or args.limit_val_batches != 1.0):
        parser.error("Scientific runs require 4 full epochs; limits belong only to the separate smoke checks")
    paths = resolved_paths(ROOT)
    metadata = validate_dataset_paths(paths)
    preparation = check_preparation(paths)
    if preparation["dataset_metadata"] != metadata:
        raise ValueError("Dataset metadata changed after shared preparation")
    provenance = preparation["provenance"]
    if not args.smoke:
        require_smokes(paths, provenance)
        if args.mode == MODES[1]:
            baseline = read_json(paths["runtime"] / MODES[0] / "seed42/training_summary.json")
            if not baseline.get("complete") or baseline.get("smoke") or baseline["provenance"] != provenance:
                raise ValueError("Complete the independent full baseline before the CounterMine run")
    run = paths["runtime"] / ("smoke" if args.smoke else "") / args.mode / "seed42"
    run = guard_step3a(run)
    if run.exists() and any(run.iterdir()):
        raise ValueError("Run directory already contains artifacts; Step 3A does not resume or overwrite conditions")
    edges = load_edge_bundle(paths["graph"], paths["snapshot"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    enter_salad(paths)
    seed_runtime(args.seed)
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger
    from countermine.training.salad_datamodule import build_datamodule
    if not torch.cuda.is_available():
        raise RuntimeError("Step 3A smoke and full training require one available CUDA GPU")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dm = build_datamodule(mode=args.mode, edge_bundle=edges, plan_dir=run / "batch_plans",
                          seed=args.seed, smoke=args.smoke,
                          expected_mapping=preparation["mapping_audit"],
                          expected_epoch0_place_order=preparation["epoch0_plans"][MODES[0]]["dataset_place_order_sha256"],
                          baseline_plan_dir=(paths["runtime"] / MODES[0] / "seed42/batch_plans"
                                             if args.mode != MODES[0] and not args.smoke else None))
    model = create_model()
    initial = read_json(paths["shared"] / "initial_state_summary.json")
    strict_load_initial_state(model, paths["shared"] / "initial_state.pt", initial)
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=str(run / "checkpoints"), monitor="pitts30k_val/R1",
        filename="best-pitts-val-epoch{epoch:02d}", auto_insert_metric_name=False,
        save_top_k=1, save_last=True, save_weights_only=True, mode="max",
    )
    audit = create_training_audit_callback(run_dir=run, mode=args.mode, smoke=args.smoke,
                                         provenance=provenance, root=ROOT)
    config = deepcopy(TRAINER_CONFIG)
    config["max_epochs"] = args.max_epochs
    trainer = pl.Trainer(
        **config, deterministic=True, default_root_dir=str(run),
        callbacks=[audit, checkpoint], logger=CSVLogger(str(run), name="logs", version=0),
        limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
    )
    write_json(run / "run_configuration.json", {
        "mode": args.mode, "smoke": args.smoke, "seed": args.seed,
        "provenance": provenance, "configuration": preparation["configuration"],
        "actual_max_epochs": args.max_epochs, "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "deterministic_algorithms": True, "tf32": False, "cudnn_benchmark": False,
    })
    trainer.fit(model=model, datamodule=dm)
    summary = training_summary(trainer=trainer, callback=audit, checkpoint=checkpoint,
        run_dir=run, mode=args.mode, smoke=args.smoke, provenance=provenance, root=ROOT)
    print(f"Completed {'engineering smoke' if args.smoke else 'single-seed pilot'}: {args.mode}")
    print("Best Pitts30k-validation epoch:", summary["best_epoch"])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        raise SystemExit(f"error: {error}") from None

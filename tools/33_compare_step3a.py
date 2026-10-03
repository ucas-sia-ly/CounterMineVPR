#!/usr/bin/env python3
"""Export a descriptive single-seed Step 3A comparison after both full evaluations."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from countermine.training.step3a_audit import RECALL_METRICS, export_comparison
from countermine.training.runtime_preflight import run_preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "cache/countermine_rgb/step3a")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/step3a")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "docs/audits/step3a_edge_cobatching_metrics.json")
    args = parser.parse_args()
    try:
        run_preflight(stage="export", root=ROOT, save_report=False)
        snapshot = export_comparison(ROOT, seed=args.seed, runtime_dir=args.runtime_dir.resolve(),
                                     output_dir=args.output_dir.resolve(), snapshot_path=args.snapshot.resolve())
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")
    evaluation = snapshot["evaluation"]
    print("single-seed Step 3A pilot — descriptive comparison")
    print("metric,baseline,CounterMine,CounterMine-baseline")
    for name in RECALL_METRICS:
        print(f"{name},{evaluation['baseline']['metrics'][name]:.6g},"
              f"{evaluation['countermine_q99_geo500']['metrics'][name]:.6g},"
              f"{evaluation['descriptive_countermine_minus_baseline'][name]:+.6g}")
    for mode, result in snapshot["training"].items():
        print(f"{mode}: best_epoch={result['best_epoch']}, final_train_loss={result['final_train_loss']:.6g}, "
              f"average_b_acc={result['average_b_acc']:.6g}")
    print(f"Exported {args.snapshot.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only package, CUDA, dataset and frozen-source checks for Step 3A."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="validate all four stages (requires CUDA)")
    selection.add_argument("--stage", choices=("prepare", "train", "evaluate", "export"),
                           help="validate one stage; preparation and export permit CPU execution")
    parser.add_argument("--save-report", action="store_true", help="save successful portable engineering report")
    args = parser.parse_args(argv)
    from countermine.training.runtime_preflight import RuntimePreflightError, run_preflight
    stage = args.stage or "all"
    try:
        report = run_preflight(stage, root=ROOT, save_report=args.save_report)
    except RuntimePreflightError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

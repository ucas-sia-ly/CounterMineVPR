#!/usr/bin/env python3
"""Measure direct canonical SOURCE-to-RELIGHT local fidelity when invoked.

Model loading is lazy. No registration, cross-place matching, generation, or
training is performed. Only sources present in the paired smoke CSV are used.
"""

import argparse
from contextlib import ExitStack
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.probe.local_fidelity import (  # noqa: E402
    EPSILONS, METRIC_COLUMNS, LocalFidelityConfig, LocalFidelityMatcher,
    compute_fidelity_metrics,
)
from countermine.probe.fidelity_plots import (  # noqa: E402
    plot_displacement_cdf, plot_repeatability, plot_worst_examples,
)


RELIGHT_MODES = ("official_rmbg", "full_scene")
ALL_MODES = ("identity_control", *RELIGHT_MODES)
CSV_COLUMNS = ("audit_index", "row_index", "mode", *METRIC_COLUMNS)
SUMMARY_METRICS = (
    "num_matches", "match_ratio_min",
    *(f"repeatability_min_{epsilon}px" for epsilon in EPSILONS),
    "displacement_median", "displacement_q95",
    "grid_coverage_4px", "grid_coverage_8px",
)
SUMMARY_QUANTILES = (0.5, 0.05, 0.25, 0.75, 0.95)
SUMMARY_KEYS = ("median", "q05", "q25", "q75", "q95")
PLOT_NAMES = (
    "fidelity_repeatability.png", "fidelity_displacement_cdf.png",
    "fidelity_worst5.jpg",
)


def _read_csv(path: Path, required: tuple[str, ...]) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError(f"{path.name} contains duplicate columns")
        missing = set(required).difference(columns)
        if missing:
            raise ValueError(f"{path.name} lacks required columns: {', '.join(sorted(missing))}")
        rows = []
        for line, row in enumerate(reader, start=2):
            if any(not row[column] or not row[column].strip() for column in required):
                raise ValueError(f"{path.name}, CSV line {line}: required metadata is missing")
            rows.append(row)
    return rows


def _identity(row: dict) -> tuple[int, int]:
    try:
        audit_index, row_index = int(row["audit_index"]), int(row["row_index"])
    except (ValueError, TypeError) as error:
        raise ValueError("audit_index and row_index must be integers") from error
    if audit_index < 0 or row_index < 0:
        raise ValueError("audit_index and row_index must be nonnegative")
    return audit_index, row_index


def _png_reference(reference: str) -> Path:
    path = Path(reference)
    if path.is_absolute():
        raise ValueError("probe image references must be relative to the invocation directory")
    if path.suffix != ".png":
        raise ValueError(
            "fidelity requires lossless .png probes; rebuild sources with "
            "tools/07_build_generator_audit_set.py and explicitly rerun "
            "tools/08_iclight_audit_smoke.py before measurement"
        )
    return path.resolve()


def load_fidelity_sources(audit_manifest: str | Path, smoke_csv: str | Path) -> list[dict]:
    """Join by both indices and require one output per relighting mode/source."""
    audit_rows = _read_csv(
        Path(audit_manifest), ("audit_index", "row_index", "image_id", "source_512_path"),
    )
    audit_by_identity = {}
    indices, row_indices, image_ids, source_paths = set(), set(), set(), set()
    for row in audit_rows:
        identity = _identity(row)
        audit_index, row_index = identity
        if audit_index in indices or row_index in row_indices or row["image_id"] in image_ids:
            raise ValueError("audit manifest contains duplicate source identities")
        indices.add(audit_index)
        row_indices.add(row_index)
        image_ids.add(row["image_id"])
        # Reject a mixed-format manifest even when the legacy rows are unselected.
        source_path = _png_reference(row["source_512_path"])
        if source_path in source_paths:
            raise ValueError("audit manifest contains duplicate canonical source paths")
        source_paths.add(source_path)
        audit_by_identity[identity] = {
            "audit_index": audit_index, "row_index": row_index,
            "source_path": source_path,
            "source_512_path": row["source_512_path"], "outputs": {},
        }
    smoke_rows = _read_csv(
        Path(smoke_csv), ("audit_index", "row_index", "mode", "source_512_path", "output_path"),
    )
    selected = {}
    output_paths = set()
    for row in smoke_rows:
        identity = _identity(row)
        if identity not in audit_by_identity:
            raise ValueError("smoke source identity does not match the audit manifest")
        mode = row["mode"]
        if mode not in RELIGHT_MODES:
            raise ValueError(f"unsupported smoke mode: {mode!r}")
        source = audit_by_identity[identity]
        if _png_reference(row["source_512_path"]) != source["source_path"]:
            raise ValueError("smoke source path does not match the audit manifest")
        if mode in source["outputs"]:
            raise ValueError("duplicate source/mode in smoke CSV")
        output_path = _png_reference(row["output_path"])
        if output_path in source_paths or output_path in output_paths:
            raise ValueError("smoke outputs must be distinct from sources and each other")
        output_paths.add(output_path)
        source["outputs"][mode] = output_path
        selected[identity] = source
    if not selected:
        raise ValueError("smoke CSV contains no paired sources")
    for source in selected.values():
        if set(source["outputs"]) != set(RELIGHT_MODES):
            raise ValueError("each smoke source requires official_rmbg and full_scene outputs")
    return [selected[identity] for identity in sorted(selected)]


def _validate_probe(path: Path) -> None:
    """Validate pixels without resizing, recropping, or EXIF transposition."""
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (512, 512):
            raise ValueError(f"probe must be an actual 512x512 RGB PNG: {path.name}")
        image.verify()
    with Image.open(path) as image:
        image.load()


def _reference(path: Path) -> str:
    return Path(os.path.relpath(path, Path.cwd())).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_modes(records: list[dict]) -> dict:
    """Equal weight per source; omit undefined statistics and count missingness."""
    summary = {}
    for mode in ALL_MODES:
        rows = [row for row in records if row["mode"] == mode]
        metrics = {}
        for name in SUMMARY_METRICS:
            values = [row[name] for row in rows if row[name] is not None]
            if values and not np.isfinite(values).all():
                raise ValueError(f"non-finite summary metric: {name}")
            quantiles = np.quantile(values, SUMMARY_QUANTILES, method="linear") if values else [None] * 5
            metrics[name] = {
                **{key: None if value is None else float(value)
                   for key, value in zip(SUMMARY_KEYS, quantiles)},
                "valid_count": len(values), "missing_count": len(rows) - len(values),
            }
        summary[mode] = {"count_pairs": len(rows), "metrics": metrics}
    return summary


def _retain_worst(examples: list[dict], example: dict) -> None:
    examples.append(example)
    examples.sort(key=lambda item: (
        item["record"]["repeatability_min_8px"],
        item["record"]["audit_index"], item["record"]["row_index"],
    ))
    del examples[5:]


def _publish(artifacts: list[tuple[Path, Path]]) -> None:
    """Replace all completed reports/plots with rollback on publication failure."""
    with ExitStack() as stack:
        saved, published = [], []
        try:
            for staged, destination in artifacts:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() or destination.is_symlink():
                    temporary = stack.enter_context(tempfile.TemporaryDirectory(
                        prefix=".fidelity-previous-", dir=destination.parent,
                    ))
                    backup = Path(temporary) / destination.name
                    os.replace(destination, backup)
                    saved.append((backup, destination))
                os.replace(staged, destination)
                published.append(destination)
        except OSError:
            for destination in reversed(published):
                if destination.is_dir() and not destination.is_symlink():
                    shutil.rmtree(destination)
                else:
                    destination.unlink(missing_ok=True)
            for backup, destination in reversed(saved):
                os.replace(backup, destination)
            raise


def run_fidelity(
    audit_manifest: str | Path,
    smoke_csv: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    max_keypoints: int = 2048,
    *,
    seed: int = 42,
    plot_dir: str | Path = "outputs/step2",
    matcher_factory=LocalFidelityMatcher,
) -> dict:
    """Reuse only the current source's features; retain scalar/coordinate diagnostics."""
    config = LocalFidelityConfig(device=device, max_keypoints=max_keypoints, seed=seed)
    audit_manifest = Path(audit_manifest).expanduser().resolve()
    smoke_csv = Path(smoke_csv).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    plot_dir = Path(plot_dir).expanduser().resolve()
    sources = load_fidelity_sources(audit_manifest, smoke_csv)
    destinations = [output_dir / name for name in ("fidelity_10.csv", "fidelity_summary.json")]
    destinations += [plot_dir / name for name in PLOT_NAMES]
    inputs = [audit_manifest, smoke_csv]
    inputs += [path for source in sources for path in (source["source_path"], *source["outputs"].values())]
    protected_roots = [(REPO_ROOT / name).resolve() for name in ("third_party", "salad")]
    if any(destination.is_relative_to(root) for destination in destinations for root in protected_roots):
        raise ValueError("fidelity outputs must be outside read-only third_party/ and salad/")
    if len(set(destinations)) != len(destinations) or any(
        path == destination or path.is_relative_to(destination)
        for path in inputs for destination in destinations
    ):
        raise ValueError("fidelity reports must not overwrite input metadata or probe images")
    # Validate all selected files before constructing any model-bearing adapter.
    for directory in sorted({path.parent for path in inputs[2:]}):
        if any(path.suffix.lower() in (".jpg", ".jpeg") for path in directory.iterdir()):
            raise ValueError("legacy JPEG probes found alongside PNGs; rebuild/rerun the audit before fidelity")
    for path in inputs[2:]:
        _validate_probe(path)

    records = []
    displacements = {mode: [] for mode in RELIGHT_MODES}
    worst = {mode: [] for mode in RELIGHT_MODES}
    image_hashes = {}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fidelity-", dir=output_dir.parent) as temporary, \
            tempfile.TemporaryDirectory(prefix=".fidelity-plots-", dir=plot_dir.parent) as plot_temporary:
        staging, plot_staging = Path(temporary), Path(plot_temporary)
        matcher = matcher_factory(config)
        with (staging / "fidelity_10.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for source in sources:
                source_path = source["source_path"]
                image_hashes[_reference(source_path)] = _sha256(source_path)
                source_features = matcher.extract(source_path)
                try:
                    for mode in ALL_MODES:
                        comparison_path = source_path if mode == "identity_control" else source["outputs"][mode]
                        # Identity also uses an independent feature extraction and a real matcher call.
                        comparison_features = matcher.extract(comparison_path)
                        try:
                            matches = matcher.match(source_features, comparison_features)
                        finally:
                            del comparison_features
                        metrics = compute_fidelity_metrics(
                            matches.points_source, matches.points_relit, matches.match_scores,
                            matches.num_keypoints_source, matches.num_keypoints_relit,
                        )
                        record = {
                            "audit_index": source["audit_index"], "row_index": source["row_index"],
                            "mode": mode, **metrics,
                        }
                        writer.writerow(record)
                        records.append(record)
                        if mode in RELIGHT_MODES:
                            image_hashes[_reference(comparison_path)] = _sha256(comparison_path)
                            displacements[mode].append(matches.displacements.copy())
                            _retain_worst(worst[mode], {
                                "record": record, "source_path": source_path,
                                "relit_path": comparison_path,
                                "points_source": matches.points_source,
                                "points_relit": matches.points_relit,
                                "match_scores": matches.match_scores,
                            })
                        print(
                            f"audit {source['audit_index']} | row {source['row_index']} | {mode} | "
                            f"matches={metrics['num_matches']} | R8={metrics['repeatability_min_8px']:.4f}",
                            flush=True,
                        )
                        del matches
                finally:
                    del source_features
        plot_repeatability(records, plot_staging / PLOT_NAMES[0])
        plot_displacement_cdf({
            mode: np.concatenate(values) if values else np.empty(0)
            for mode, values in displacements.items()
        }, plot_staging / PLOT_NAMES[1])
        plot_worst_examples(worst, plot_staging / PLOT_NAMES[2])
        summary = {
            "number_of_sources": len(sources), "count_pairs": len(records),
            "config": {
                **asdict(config), "canonical_resolution": [512, 512],
                "probe_image_format": "PNG", "extract_resize": None,
                "epsilons_pixels": list(EPSILONS), "epsilon_comparison": "<=",
                "grid_shape": [8, 8], "grid_coordinates": "source keypoints",
                "zero_denominator_ratios": 0.0, "empty_displacement_and_score_stats": None,
                "summary_quantiles": "numpy linear interpolation; equal weight per source",
                "displacement_cdf_weighting": "each matched correspondence has equal weight",
                "registration": None, "matcher_compiled": False,
                "identity_control": "independent extraction of the same source; LightGlue matching",
                "feature_retention": "current source plus one comparison; discard before next source",
                "worst_examples_per_mode": 5, "visualization_max_matches": 80,
                "worst_order": "repeatability_min_8px, audit_index, row_index ascending",
            },
            "matcher": matcher.runtime_metadata(),
            "audit_manifest_reference": _reference(audit_manifest),
            "audit_manifest_sha256": _sha256(audit_manifest),
            "smoke_csv_reference": _reference(smoke_csv), "smoke_csv_sha256": _sha256(smoke_csv),
            "probe_sha256": image_hashes,
            "modes": summarize_modes(records),
            "plots": {name: _reference(plot_dir / name) for name in PLOT_NAMES},
            "worst_examples": {
                mode: [{
                    "audit_index": item["record"]["audit_index"],
                    "row_index": item["record"]["row_index"],
                    "repeatability_min_8px": item["record"]["repeatability_min_8px"],
                } for item in examples]
                for mode, examples in worst.items()
            },
        }
        (staging / "fidelity_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
        )
        artifacts = [(staging / name, output_dir / name) for name in ("fidelity_10.csv", "fidelity_summary.json")]
        artifacts += [(plot_staging / name, plot_dir / name) for name in PLOT_NAMES]
        _publish(artifacts)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", type=Path, default=Path("cache/generator_audit/audit_manifest.csv"))
    parser.add_argument("--smoke-csv", type=Path, default=Path("cache/generator_audit/iclight_smoke.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/generator_audit/fidelity"))
    parser.add_argument("--device", default="auto", help="auto selects CUDA when available, otherwise CPU")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-dir", type=Path, default=Path("outputs/step2"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run_fidelity(
            args.audit_manifest, args.smoke_csv, args.output_dir,
            args.device, args.max_keypoints, seed=args.seed, plot_dir=args.plot_dir,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Measured {summary['count_pairs']} pairs for {summary['number_of_sources']} sources")
    print(f"CSV: {args.output_dir / 'fidelity_10.csv'}")
    print(f"Summary: {args.output_dir / 'fidelity_summary.json'}")
    print(f"Plots: {args.plot_dir}")


if __name__ == "__main__":
    main()

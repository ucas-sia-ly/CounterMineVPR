"""Integrity checks for the existing 100-source population; never sample sources.

Historical manifest columns are retained verbatim except native original sizes
are typed as integers. References follow existing tools: relative to the cwd.
"""

from collections import Counter
import json
import math
from pathlib import Path

from countermine.probe.geometry_audit import REPO_ROOT, read_csv, sha256
from countermine.probe.native_fov_audit import (
    integer, relative_path, validate_native_destinations, validate_native_source,
)


SOURCE_COUNT = 100
NATIVE_SIZE = (640, 480)
POLICY = "native_full_fov"
POPULATION_MANIFEST = Path("cache/generator_audit/audit_manifest.csv")
POPULATION_SUMMARY = Path("cache/generator_audit/audit_summary.json")
IDENTITY_COLUMNS = ("audit_index", "row_index", "image_id")
MANIFEST_COLUMNS = (
    *IDENTITY_COLUMNS, "policy", "original_path", "source_path",
    "original_width", "original_height", "canonical_width", "canonical_height",
    "scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction",
    "original_format", "source_format", "original_sha256", "source_sha256",
)


def _unique_identities(rows):
    if len(rows) != SOURCE_COUNT:
        raise ValueError("the frozen audit requires exactly 100 sources; no resampling")
    seen = {name: set() for name in IDENTITY_COLUMNS}
    for row in rows:
        for name in ("audit_index", "row_index"):
            row[name] = integer(row[name], name)
        relative_path(row["image_id"], "image_id")
        for name in IDENTITY_COLUMNS:
            if row[name] in seen[name]:
                raise ValueError(f"frozen population contains duplicate {name}")
            seen[name].add(row[name])
    return rows


def load_frozen_population(manifest_path=POPULATION_MANIFEST, summary_path=POPULATION_SUMMARY):
    """Read all historical records and their provenance, without reselection."""
    rows = read_csv(manifest_path, (*IDENTITY_COLUMNS, "group", "relative_path",
                                    "source_512_path", "original_width", "original_height"))
    _unique_identities(rows)
    for row in rows:
        relative_path(row["relative_path"], "relative_path")
        relative_path(row["source_512_path"], "source_512_path", allow_parent=True)
        if row["image_id"] != row["relative_path"]:
            raise ValueError("image identity must resolve to its historical original relative_path")
        if tuple(integer(row[name], name) for name in ("original_width", "original_height")) != NATIVE_SIZE:
            raise ValueError("every frozen original must have historical dimensions 640x480")
    groups = dict(Counter(row["group"] for row in rows))
    if groups != {"random": 50, "hard_candidate": 50}:
        raise ValueError("reuse exactly 50 random and 50 hard_candidate historical sources")
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if (summary.get("number_of_images") != SOURCE_COUNT or summary.get("count_random") != 50
            or summary.get("count_hard_candidate") != 50):
        raise ValueError("source population and historical selection provenance disagree")
    if summary.get("seed") != 42 or summary.get("config", {}).get("seed") != 42:
        raise ValueError("the frozen source population must retain selection seed 42")
    for name in ("manifest_reference", "rgb_candidate_reference"):
        relative_path(summary[name], name, allow_parent=True)
    for name in ("manifest_sha256", "rgb_candidate_sha256"):
        if (not isinstance(summary.get(name), str) or len(summary[name]) != 64
                or any(c not in "0123456789abcdef" for c in summary[name])):
            raise ValueError(f"historical provenance is missing valid {name}")
    return sorted(rows, key=lambda row: (row["audit_index"], row["row_index"])), summary


def validate_manifest_metadata(rows, frozen_rows=None):
    """Validate 100 identities and native geometry without opening any images."""
    rows = [dict(row) for row in rows]
    _unique_identities(rows)
    frozen = {row["row_index"]: row for row in frozen_rows} if frozen_rows is not None else None
    paths = {"original_path": set(), "source_path": set()}
    for row in rows:
        if frozen is not None:
            old = frozen.get(row["row_index"])
            if old is None:
                raise ValueError("native manifest contains a non-frozen source")
            for name, value in old.items():
                if name in ("original_width", "original_height"):
                    continue
                if row.get(name) != value:
                    raise ValueError(f"native manifest changed frozen source/provenance field {name}")
        if row.get("policy") != POLICY:
            raise ValueError("native manifest requires native_full_fov")
        for prefix in ("original", "canonical"):
            for axis in ("width", "height"):
                name = f"{prefix}_{axis}"
                row[name] = integer(row[name], name)
            if (row[f"{prefix}_width"], row[f"{prefix}_height"]) != NATIVE_SIZE:
                raise ValueError("native geometry must be exactly 640x480")
        for name in ("scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction"):
            row[name] = float(row[name])
            if not math.isfinite(row[name]) or row[name] != 1.0:
                raise ValueError(f"native {name} must be exactly 1.0")
        if row.get("original_format") != "JPEG" or row.get("source_format") != "PNG":
            raise ValueError("frozen sources require JPEG originals and lossless RGB PNG copies")
        for name in paths:
            path = relative_path(row[name], name, allow_parent=True).resolve()
            if path in paths[name]:
                raise ValueError(f"duplicate resolved {name}")
            paths[name].add(path)
    if paths["source_path"].intersection(paths["original_path"]):
        raise ValueError("native sources must not overwrite originals")
    return sorted(rows, key=lambda row: (row["audit_index"], row["row_index"]))


def load_native100_manifest(path, population_manifest_path=POPULATION_MANIFEST,
                            population_summary_path=POPULATION_SUMMARY):
    frozen, _ = load_frozen_population(population_manifest_path, population_summary_path)
    rows = validate_manifest_metadata(read_csv(path, MANIFEST_COLUMNS), frozen)
    for row in rows:
        for name in ("original_path", "source_path"):
            row[name] = relative_path(row[name], name, allow_parent=True).resolve()
        validate_native_source(row)
    return rows


def validate_native100_destinations(destinations, inputs, *, extra_protected=()):
    validate_native_destinations(destinations, inputs, extra_protected=(
        REPO_ROOT / "cache/generator_audit", REPO_ROOT / "cache/native_fov_audit",
        REPO_ROOT / "outputs/step2d1", *extra_protected,
    ))


def population_provenance(manifest_path, summary_path, summary):
    from countermine.probe.geometry_audit import reference
    return {
        "selection": "exact existing Step 2 generator-audit population; no resampling",
        "number_of_sources": SOURCE_COUNT, "group_counts": {"random": 50, "hard_candidate": 50},
        "audit_manifest_reference": reference(manifest_path), "audit_manifest_sha256": sha256(manifest_path),
        "audit_summary_reference": reference(summary_path), "audit_summary_sha256": sha256(summary_path),
        "mini_manifest_reference": summary["manifest_reference"], "mini_manifest_sha256": summary["manifest_sha256"],
        "rgb_candidate_reference": summary["rgb_candidate_reference"], "rgb_candidate_sha256": summary["rgb_candidate_sha256"],
        "config": dict(summary["config"]),
    }

"""CPU-only graphs from frozen, calibrated real-RGB pair measurements.

Every candidate remains an image edge. The q95/q99 flags describe frozen
pilot analysis slices; they have no training interpretation. Place support
is aggregated after independently aligning endpoints by place identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from countermine.mining.structural_analysis import (
    read_metadata_csv, sha256_file, validate_metrics,
)


IMAGE_NODE_COLUMNS = ("image_id", "row_index", "place_uid", "city_id")
PLACE_NODE_COLUMNS = ("place_uid", "city_id")
GRAPH_STAGE = "2B CounterMine structural confusion graph pilot"
GRAPH_FILENAMES = ("image_nodes.csv", "image_edges.csv", "place_nodes.csv", "place_edges.csv")
SLICE_DEFINITIONS = {
    "full": "all frozen measured candidate image pairs, including duplicates",
    "core_q95": "ratio_null_percentile >= 0.95 and match_count_null_percentile >= 0.95 and not exact_pixel_duplicate",
    "core_q99": "structural_bottleneck >= 0.99 and not exact_pixel_duplicate",
    "core_q95_geo500": "core_q95 and geo_distance_m >= 500",
    "core_q99_geo500": "core_q99 and geo_distance_m >= 500",
    "core_q95_low_spatial_support": "core_q95 and symmetric_match_coverage <= q10 of all frozen candidate edges; inspection only",
}
CALIBRATED_COLUMNS = (
    "ratio_null_percentile", "match_count_null_percentile", "coverage_null_percentile",
    "entropy_null_percentile", "structural_bottleneck", "structural_geomean",
    "spatial_support_percentile", "salad_similarity_percentile",
)


def place_pair_uid(place_uid_a: str, place_uid_b: str) -> str:
    """Hash an unordered place identity with explicit unambiguous encoding."""
    payload = json.dumps(sorted((place_uid_a, place_uid_b)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_graph_source_geometry(calibrated: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    """Check canonical image identities and frozen haversine distances on CPU.

    This uses the existing 6,371,008.8 m mean-Earth radius and absolute 1e-6 m
    tolerance, without importing the model-owning Step 1C miner module.
    """
    _validate_nodes(manifest)
    expected = [place_pair_uid(a, b) for a, b in zip(calibrated["image_id_a"], calibrated["image_id_b"])]
    if calibrated["pair_uid"].tolist() != expected:
        raise ValueError("candidate pair_uid disagrees with canonical image endpoint identity")
    coordinates = np.column_stack([_numeric(manifest, column) for column in ("lat", "lon")])
    if (np.abs(coordinates[:, 0]) > 90).any() or (np.abs(coordinates[:, 1]) > 180).any():
        raise ValueError("manifest geographic coordinates are outside valid ranges")
    lookup = pd.DataFrame(coordinates, index=manifest["image_id"], columns=["lat", "lon"])
    try:
        a = np.deg2rad(lookup.loc[calibrated["image_id_a"]].to_numpy())
        b = np.deg2rad(lookup.loc[calibrated["image_id_b"]].to_numpy())
    except KeyError as error:
        raise ValueError("candidate geographic endpoint is missing from manifest") from error
    h = np.sin((b[:, 0] - a[:, 0]) / 2) ** 2 + np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin((b[:, 1] - a[:, 1]) / 2) ** 2
    distances = 2 * 6_371_008.8 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    if not np.allclose(_numeric(calibrated, "geo_distance_m"), distances, atol=1e-6, rtol=0):
        raise ValueError("candidate geo_distance_m disagrees with frozen manifest haversine")
    return {"canonical_image_pair_uids": True, "manifest_geographic_distances": True,
            "geo_distance_absolute_tolerance_m": 1e-6, "geo_distance_relative_tolerance": 0}


def _boolean(value, column: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"{column} must contain booleans")


def _numeric(frame: pd.DataFrame, column: str, *, integer: bool = False,
             optional: bool = False) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[column].replace("", np.nan), errors="raise").to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{column} must contain finite numeric values") from error
    if np.isinf(values).any() or (not optional and np.isnan(values).any()):
        raise ValueError(f"{column} must contain finite numeric values")
    if integer and ((values < 0).any() or (values != np.floor(values)).any()):
        raise ValueError(f"{column} must contain nonnegative integers")
    return values.astype(np.int64) if integer else values


def _validate_calibrated(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("a graph requires at least one measured candidate edge")
    if not frame.columns.is_unique:
        raise ValueError("calibrated column names must be unique")
    result = validate_metrics(frame, candidate=True)
    missing = set(CALIBRATED_COLUMNS).difference(result.columns)
    if missing:
        raise ValueError(f"calibrated candidates lack {sorted(missing)}")
    for column in CALIBRATED_COLUMNS:
        values = _numeric(result, column)
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column} must be in [0,1]")
        result[column] = values
    expected = np.minimum(result["ratio_null_percentile"], result["match_count_null_percentile"])
    if not np.allclose(result["structural_bottleneck"], expected, rtol=0, atol=1e-12):
        raise ValueError("structural_bottleneck disagrees with min(ratio,count percentiles)")
    expected = np.sqrt(result["ratio_null_percentile"] * result["match_count_null_percentile"])
    if not np.allclose(result["structural_geomean"], expected, rtol=0, atol=1e-12):
        raise ValueError("structural_geomean disagrees with sqrt(ratio*count percentiles)")
    if not np.allclose(result["spatial_support_percentile"], result["coverage_null_percentile"], rtol=0, atol=1e-12):
        raise ValueError("spatial_support_percentile must equal coverage_null_percentile")
    if "num_candidate_directions" not in result:
        raise ValueError("calibrated candidates lack num_candidate_directions")
    result["num_candidate_directions"] = _numeric(result, "num_candidate_directions", integer=True)
    if not result["num_candidate_directions"].isin([1, 2]).all():
        raise ValueError("num_candidate_directions must be 1 or 2")
    for metric in ("ratio", "match_count", "coverage", "entropy"):
        source = f"{metric}_null_source"
        if source not in result or not result[source].isin(["same_city", "cross_city", "global"]).all():
            raise ValueError(f"{source} must preserve explicit null provenance")
        relation = np.where(result["same_city"], "same_city", "cross_city")
        if not ((result[source] == "global") | (result[source] == relation)).all():
            raise ValueError(f"{source} disagrees with endpoint city relation")
    return result


def _validate_nodes(manifest: pd.DataFrame, columns=IMAGE_NODE_COLUMNS) -> pd.DataFrame:
    if not manifest.columns.is_unique or not set(columns).issubset(manifest.columns):
        raise ValueError(f"node metadata must have unique columns including {columns}")
    result = manifest.loc[:, columns].copy()
    for column in set(columns).difference({"row_index"}):
        if not result[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"node {column} must contain nonempty string identities")
    if result[columns[0]].duplicated().any():
        raise ValueError(f"node {columns[0]} must be unique")
    if "row_index" in columns:
        result["row_index"] = _numeric(result, "row_index", integer=True)
        if result["row_index"].duplicated().any():
            raise ValueError("node row_index must be unique")
    if result.groupby("place_uid")["city_id"].nunique().gt(1).any():
        raise ValueError("a place_uid cannot identify multiple cities")
    return result


def _canonicalize_image_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    """Align endpoint diagnostics as well as IDs; directional fields swap too."""
    result = frame.copy()
    swap = result["image_id_a"] > result["image_id_b"]
    columns = set(result.columns)
    pairs = [(column, column[:-2] + "_b") for column in result
             if column.endswith("_a") and column[:-2] + "_b" in columns]
    pairs += [("matched_source_cell_coverage", "matched_target_cell_coverage"),
              ("source_max_cell_match_fraction", "target_max_cell_match_fraction"),
              ("source_match_entropy", "target_match_entropy")]
    pairs += [(column, column.replace("a_to_b_", "b_to_a_", 1)) for column in result
              if column.startswith("a_to_b_") and column.replace("a_to_b_", "b_to_a_", 1) in columns]
    for left, right in pairs:
        old_left, old_right = result[left].copy(), result[right].copy()
        result.loc[swap, left] = old_right.loc[swap]
        result.loc[swap, right] = old_left.loc[swap]
    return result


def build_image_graph(calibrated: pd.DataFrame, manifest: pd.DataFrame):
    """Return endpoint nodes and ALL measured undirected candidate edges.

    The node universe is the union of measured candidate endpoints, rather
    than unmeasured images from the full manifest. Isolated nodes in a core
    slice stay in this fixed pilot node universe.
    """
    nodes = _validate_nodes(manifest)
    edges = _canonicalize_image_endpoints(_validate_calibrated(calibrated))
    if edges["image_id_a"].eq(edges["image_id_b"]).any():
        raise ValueError("self image edges are forbidden")
    if edges.duplicated(["image_id_a", "image_id_b"]).any():
        raise ValueError("undirected image edges must be unique")
    lookup = nodes.set_index("image_id")
    for side in ("a", "b"):
        identities = edges[f"image_id_{side}"]
        if not identities.isin(lookup.index).all():
            raise ValueError("candidate endpoint is missing from manifest")
        for column in IMAGE_NODE_COLUMNS[1:]:
            edge_column = f"{column}_{side}"
            expected = lookup.loc[identities, column].to_numpy()
            if edge_column in edges:
                actual = (_numeric(edges, edge_column, integer=True) if column == "row_index"
                          else edges[edge_column].to_numpy())
                if not np.array_equal(actual, expected):
                    raise ValueError(f"candidate {edge_column} disagrees with manifest")
            edges[edge_column] = expected
    edges["near_geo_500m"] = edges["geo_distance_m"] < 500
    nonduplicate = ~edges["exact_pixel_duplicate"]
    edges["core_q95"] = ((edges["ratio_null_percentile"] >= .95)
                         & (edges["match_count_null_percentile"] >= .95) & nonduplicate)
    edges["core_q99"] = (edges["structural_bottleneck"] >= .99) & nonduplicate
    for name in ("core_q95", "core_q99"):
        edges[f"{name}_geo500"] = edges[name] & (edges["geo_distance_m"] >= 500)
    q10 = float(np.quantile(edges["symmetric_match_coverage"], .1))
    edges["core_q95_low_spatial_support"] = edges["core_q95"] & (edges["symmetric_match_coverage"] <= q10)
    identities = set(edges["image_id_a"]) | set(edges["image_id_b"])
    nodes = nodes[nodes["image_id"].isin(identities)].sort_values("image_id", kind="stable").reset_index(drop=True)
    edges = edges.sort_values("pair_uid", kind="stable").reset_index(drop=True)
    return nodes, edges


def build_place_graph(image_nodes: pd.DataFrame, image_edges: pd.DataFrame):
    """Aggregate unordered place pairs, preserving support on each place side.

    Requested independent_support flags use unique images across ALL support.
    Additional core_independent_support flags require both endpoint image sets
    themselves to recur within core edges, exposing that scientific distinction.
    """
    nodes = _validate_nodes(image_nodes)
    lookup = nodes.set_index("image_id")
    edges = image_edges.copy()
    if edges["pair_uid"].duplicated().any() or edges.duplicated(["image_id_a", "image_id_b"]).any():
        raise ValueError("place aggregation requires unique image edges")
    for side in ("a", "b"):
        identities = edges[f"image_id_{side}"]
        if not identities.isin(lookup.index).all():
            raise ValueError("place aggregation endpoint is missing from image nodes")
        for name in ("place_uid", "city_id"):
            if not np.array_equal(edges[f"{name}_{side}"], lookup.loc[identities, name].to_numpy()):
                raise ValueError("place aggregation endpoint metadata disagrees with image nodes")
    if edges["place_uid_a"].eq(edges["place_uid_b"]).any():
        raise ValueError("same-place graph edges are forbidden")
    swap = edges["place_uid_a"] > edges["place_uid_b"]
    for name in ("image_id", "place_uid", "city_id"):
        a, b = edges[f"{name}_a"].copy(), edges[f"{name}_b"].copy()
        edges.loc[swap, f"{name}_a"] = b.loc[swap]
        edges.loc[swap, f"{name}_b"] = a.loc[swap]
    place_nodes = nodes.loc[:, PLACE_NODE_COLUMNS].drop_duplicates().sort_values("place_uid", kind="stable").reset_index(drop=True)
    rows = []
    for (place_a, place_b), support in edges.groupby(["place_uid_a", "place_uid_b"], sort=True):
        row = {"place_pair_uid": place_pair_uid(place_a, place_b),
               "place_uid_a": place_a, "place_uid_b": place_b,
               "city_id_a": support["city_id_a"].iloc[0], "city_id_b": support["city_id_b"].iloc[0],
               "same_city": bool(support["city_id_a"].iloc[0] == support["city_id_b"].iloc[0]),
               "num_supporting_image_edges": len(support),
               "num_unique_images_a": int(support["image_id_a"].nunique()),
               "num_unique_images_b": int(support["image_id_b"].nunique()),
               "best_rgb_rank_min": int(support["best_rgb_rank"].min()),
               "min_geo_distance_m": float(support["geo_distance_m"].min()),
               "median_geo_distance_m": float(support["geo_distance_m"].median()),
               "num_near_geo_500m_image_edges": int((support["geo_distance_m"] < 500).sum()),
               "num_geo500_image_edges": int((support["geo_distance_m"] >= 500).sum())}
        row["independent_view_support"] = min(row["num_unique_images_a"], row["num_unique_images_b"])
        row["near_geo_500m"] = row["num_near_geo_500m_image_edges"] > 0
        geo_support = support[support["geo_distance_m"] >= 500]
        row["num_geo500_unique_images_a"] = int(geo_support["image_id_a"].nunique())
        row["num_geo500_unique_images_b"] = int(geo_support["image_id_b"].nunique())
        for column in ("max_salad_similarity", "num_matches", "local_match_ratio",
                       "structural_bottleneck", "structural_geomean", "symmetric_match_coverage",
                       "symmetric_match_entropy"):
            row[f"{column}_max"] = float(support[column].max())
            row[f"{column}_median"] = float(support[column].median())
        for flag in ("core_q95", "core_q99", "core_q95_geo500", "core_q99_geo500"):
            selected = support[support[flag].map(lambda value: _boolean(value, flag))]
            row[f"num_{flag}_image_edges"] = len(selected)
            row[f"num_{flag}_unique_images_a"] = int(selected["image_id_a"].nunique())
            row[f"num_{flag}_unique_images_b"] = int(selected["image_id_b"].nunique())
            row[flag] = bool(len(selected))
        for minimum in (2, 3):
            row[f"repeated_support_{minimum}"] = row["num_core_q95_image_edges"] >= minimum
            row[f"independent_support_{minimum}"] = (row[f"repeated_support_{minimum}"]
                and row["num_unique_images_a"] >= 2 and row["num_unique_images_b"] >= 2)
            row[f"core_independent_support_{minimum}"] = (row[f"repeated_support_{minimum}"]
                and row["num_core_q95_unique_images_a"] >= 2 and row["num_core_q95_unique_images_b"] >= 2)
            row[f"repeated_support_{minimum}_geo500"] = row["num_core_q95_geo500_image_edges"] >= minimum
            row[f"independent_support_{minimum}_geo500"] = (row[f"repeated_support_{minimum}_geo500"]
                and row["num_geo500_unique_images_a"] >= 2 and row["num_geo500_unique_images_b"] >= 2)
            row[f"core_geo500_independent_support_{minimum}"] = (row[f"repeated_support_{minimum}_geo500"]
                and row["num_core_q95_geo500_unique_images_a"] >= 2 and row["num_core_q95_geo500_unique_images_b"] >= 2)
            row[f"core_independent_support_{minimum}_geo500"] = row[f"core_geo500_independent_support_{minimum}"]
        rows.append(row)
    return place_nodes, pd.DataFrame(rows)


def write_graph_artifacts(output_dir, image_nodes, image_edges, place_nodes, place_edges,
                          *, calibration_summary: dict, repo_root=None, source_validation=None) -> dict:
    """Publish four deterministic CSVs, then an atomic hash-bearing summary."""
    output_dir = Path(output_dir)
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
    input_files = calibration_summary.get("input_files", {})
    protected_runtime_dirs = {repo_root / "cache/countermine_rgb/step2a"}
    if "runtime_dir" in input_files:
        protected_runtime_dirs.add(repo_root / input_files["runtime_dir"])
    for runtime_dir in protected_runtime_dirs:
        if output_dir.resolve() == runtime_dir.resolve() or output_dir.resolve().is_relative_to(runtime_dir.resolve()):
            raise ValueError("Step 2B graph output must not modify the Step 2A runtime directory")
    protected_sources = {(repo_root / relative).resolve() for name, relative in input_files.items() if name != "runtime_dir"}
    for name in (*GRAPH_FILENAMES, "graph_summary.json"):
        destination = output_dir / name
        # Reject fixed-file aliases even though atomic replacement would
        # replace a symlink itself. Artifact locations must identify their own
        # Step 2B files, never a historical/source file through an alias.
        if destination.is_symlink() or destination.resolve() in protected_sources:
            raise ValueError(f"graph output must not alias a source file: {name}")
    if calibration_summary.get("stage") not in (
        "step2b_structural_calibration", "2B real-RGB structural evidence calibration",
    ):
        raise ValueError("graph construction requires a Step 2B calibration summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    q10 = float(np.quantile(image_edges["symmetric_match_coverage"], .1))
    summary = {
        "schema_version": 1, "stage": GRAPH_STAGE, "complete": True,
        "configuration": {"seed": 42, "image_node_universe": "union of frozen candidate endpoints",
                          "place_node_universe": "places represented by image graph nodes",
                          "min_geo_distance_m": 250.0, "analysis_slices": SLICE_DEFINITIONS,
                          "candidate_q10_symmetric_match_coverage": q10},
        "provenance": dict(calibration_summary.get("provenance", {})),
        "calibration": calibration_summary.get("calibration", {}),
        "calibration_configuration": calibration_summary.get("configuration", {}),
        "graph_input_validation": dict(source_validation or {}),
        "image_node_count": len(image_nodes), "image_edge_count": len(image_edges),
        "place_node_count": len(place_nodes), "place_edge_count": len(place_edges),
        "candidate_q10_symmetric_match_coverage": q10,
        "core_q95_low_spatial_support_count": int(image_edges["core_q95_low_spatial_support"].sum()),
        "independent_support_definitions": {
            "requested": "core_q95 edge count >= 2 or 3 AND ALL-support unique images >= 2 on both place endpoints",
            "core_only": "core_q95 edge count >= 2 or 3 AND core_q95 unique images >= 2 on both place endpoints",
            "geo500": "corresponding counts computed after retaining only geo_distance_m >= 500 support",
        },
    }
    summary["provenance"].update(real_rgb_only=True, synthetic_images_used=False, no_new_local_matching=True)
    summary["graph_code_hashes"] = {
        str(path.relative_to(repo_root)): sha256_file(path)
        for path in (Path(__file__), repo_root / "tools/12_build_countermine_graph.py") if path.exists()
    }
    calibration_path = output_dir / "calibration_summary.json"
    if calibration_path.exists():
        summary["calibration_summary_sha256"] = sha256_file(calibration_path)
    calibrated_path = output_dir / "calibrated_candidates.csv"
    if calibrated_path.exists():
        summary["calibrated_candidates_sha256"] = sha256_file(calibrated_path)
    temporary = []
    try:
        artifacts = {}
        for name, table in zip(GRAPH_FILENAMES, (image_nodes, image_edges, place_nodes, place_edges)):
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_dir,
                                             prefix=name + ".", suffix=".partial", delete=False) as destination:
                path = Path(destination.name)
                temporary.append((path, output_dir / name))
                table.to_csv(destination, index=False, float_format="%.17g")
            artifacts[name] = {"sha256": sha256_file(path), "count": len(table), "columns": list(table.columns)}
        summary["artifacts"] = artifacts
        summary["output_hashes"] = {name: metadata["sha256"] for name, metadata in artifacts.items()}
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir,
                                         prefix="graph_summary.json.", suffix=".partial", delete=False) as destination:
            path = Path(destination.name)
            temporary.append((path, output_dir / "graph_summary.json"))
            destination.write(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n")
        for temporary_path, final_path in temporary:
            temporary_path.replace(final_path)
    finally:
        for path, _ in temporary:
            path.unlink(missing_ok=True)
    return summary


def _load_typed_edges(path: Path, *, image: bool):
    frame = read_metadata_csv(path)
    if image:
        result = _validate_calibrated(frame)
        for column in result:
            if column.startswith(("global_", "relation_")) and column.endswith("null_percentile"):
                result[column] = _numeric(result, column, optional=column.startswith("relation_"))
            elif column in ("row_index_a", "row_index_b"):
                result[column] = _numeric(result, column, integer=True)
            elif column in ("near_geo_500m", "core_q95", "core_q99", "core_q95_geo500",
                            "core_q99_geo500", "core_q95_low_spatial_support"):
                result[column] = result[column].map(lambda value: _boolean(value, column))
        return result
    for column in frame:
        # A median of integer correspondence counts can be fractional. Parse
        # aggregation suffixes before the integer support-count prefix.
        if column.endswith(("_max", "_median", "_distance_m")):
            frame[column] = _numeric(frame, column)
        elif column.startswith(("num_", "best_rgb_rank_min", "independent_view_support")):
            frame[column] = _numeric(frame, column, integer=True)
        elif column in ("same_city", "near_geo_500m") or column.startswith(
            ("core_q", "repeated_support_", "independent_support_", "core_independent_support_", "core_geo500_independent_support_")):
            frame[column] = frame[column].map(lambda value: _boolean(value, column))
    return frame


def load_graph_artifacts(output_dir, *, source_validation: bool = True, repo_root=None):
    """Load only published, hash-consistent graph artifacts and their summary."""
    output_dir = Path(output_dir)
    with (output_dir / "graph_summary.json").open(encoding="utf-8") as source:
        summary = json.load(source, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON {value}")))
    if summary.get("stage") != GRAPH_STAGE or summary.get("schema_version") != 1 or summary.get("complete") is not True:
        raise ValueError("graph summary has the wrong stage")
    for name in GRAPH_FILENAMES:
        metadata = summary.get("artifacts", {}).get(name, {})
        if metadata.get("sha256") != sha256_file(output_dir / name):
            raise ValueError(f"graph artifact hash mismatch: {name}")
    if source_validation:
        from countermine.mining.structural_calibration import load_calibrated_artifacts
        _, calibration = load_calibrated_artifacts(output_dir, source_validation=True, repo_root=repo_root)
        repository = Path(repo_root or Path(__file__).resolve().parents[2])
        for name, digest in summary.get("graph_code_hashes", {}).items():
            if Path(name).is_absolute() or ".." in Path(name).parts or sha256_file(repository / name) != digest:
                raise ValueError("graph code checksum differs from published graph stage")
        if summary.get("calibration_summary_sha256") != sha256_file(output_dir / "calibration_summary.json"):
            raise ValueError("graph calibration summary hash mismatch")
        if summary.get("provenance") != calibration.get("provenance"):
            raise ValueError("graph source provenance differs from calibration")
        if summary.get("calibrated_candidates_sha256") != sha256_file(output_dir / "calibrated_candidates.csv"):
            raise ValueError("graph calibrated candidate hash mismatch")
    image_nodes = _validate_nodes(read_metadata_csv(output_dir / "image_nodes.csv"))
    image_edges = _load_typed_edges(output_dir / "image_edges.csv", image=True)
    place_nodes = _validate_nodes(read_metadata_csv(output_dir / "place_nodes.csv"), PLACE_NODE_COLUMNS)
    place_edges = _load_typed_edges(output_dir / "place_edges.csv", image=False)
    tables = (image_nodes, image_edges, place_nodes, place_edges)
    for name, table in zip(GRAPH_FILENAMES, tables):
        metadata = summary["artifacts"][name]
        if len(table) != metadata.get("count") or list(table.columns) != metadata.get("columns"):
            raise ValueError(f"graph artifact schema/count mismatch: {name}")
    q10 = float(np.quantile(image_edges["symmetric_match_coverage"], .1))
    if summary.get("candidate_q10_symmetric_match_coverage") != q10:
        raise ValueError("graph low-spatial-support q10 differs from complete candidate population")
    return (*tables, summary)

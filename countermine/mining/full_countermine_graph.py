"""CPU-only full real-RGB image/place graphs; every canonical edge survives.

The extra q975/q995 slices are predeclared diagnostics, never selected from
the full results. Scalar tables are aggregated with pandas, without images,
feature tensors, dense adjacency matrices, or a training interpretation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from countermine.mining.countermine_graph import (
    IMAGE_NODE_COLUMNS, PLACE_NODE_COLUMNS, _canonicalize_image_endpoints,
    _numeric, _validate_nodes, place_pair_uid, validate_graph_source_geometry,
)
from countermine.mining.graph_analysis import boolean_values
from countermine.mining.structural_analysis import read_metadata_csv, sha256_file, validate_metrics
from countermine.mining.full_structural_io import (
    ROOT, DEFAULT_DIR, code_hashes, guard_step2d, read_json, write_csv, write_json,
)
from countermine.mining.full_structural_provenance import PROVENANCE_SOURCE


QUANTILES = {"q95": .95, "q975": .975, "q99": .99, "q995": .995}
CORE_SLICES = tuple("core_" + name for name in QUANTILES)
SLICES = ("full", *CORE_SLICES, "full_geo500", *(name + "_geo500" for name in CORE_SLICES))
GRAPH_FILENAMES = ("image_nodes.csv", "image_edges.csv", "place_nodes.csv", "place_edges.csv")
GRAPH_SOURCE_FILES = ("countermine/mining/full_countermine_graph.py", "tools/23_build_full_countermine_graph.py",
                      PROVENANCE_SOURCE)
GEOGRAPHIC_COLUMNS = ("geo_distance_m", "min_geo_distance_m", "median_geo_distance_m")
CALIBRATED_COLUMNS = (
    "ratio_null_percentile", "match_count_null_percentile", "coverage_null_percentile",
    "entropy_null_percentile", "structural_bottleneck", "structural_geomean", "spatial_support_percentile",
)
SLICE_DEFINITIONS = {"full": "all eligible canonical original-RGB candidate edges",
                     "full_geo500": "full and geo_distance_m >= 500; geographic sensitivity only"}
for _name, _value in QUANTILES.items():
    SLICE_DEFINITIONS["core_" + _name] = f"structural_bottleneck >= {_value} and not exact_pixel_duplicate"
    SLICE_DEFINITIONS["core_" + _name + "_geo500"] = f"core_{_name} and geo_distance_m >= 500"


def validate_full_calibrated(frame):
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("full graph requires a nonempty table with unique columns")
    result = validate_metrics(frame, candidate=True)
    for column in CALIBRATED_COLUMNS:
        result[column] = _numeric(result, column)
        if not result[column].between(0, 1).all():
            raise ValueError(f"{column} must lie in [0,1]")
    for column, expected in (
        ("structural_bottleneck", np.minimum(result.ratio_null_percentile, result.match_count_null_percentile)),
        ("structural_geomean", np.sqrt(result.ratio_null_percentile * result.match_count_null_percentile)),
        ("spatial_support_percentile", result.coverage_null_percentile),
    ):
        if not np.allclose(result[column], expected, atol=1e-12, rtol=0):
            raise ValueError(f"{column} disagrees with frozen formula")
    relation = np.where(result.same_city, "same_city", "cross_city")
    for name in ("ratio", "match_count", "coverage", "entropy"):
        column = name + "_null_source"
        if column not in result or not np.array_equal(result[column], relation):
            raise ValueError(f"{column} must identify the frozen relation-specific null")
    result["num_candidate_directions"] = _numeric(result, "num_candidate_directions", integer=True)
    if not result.num_candidate_directions.isin((1, 2)).all():
        raise ValueError("num_candidate_directions must be one or two")
    denominator = np.minimum(result.num_keypoints_a, result.num_keypoints_b)
    if "min_num_keypoints" in result and not np.array_equal(_numeric(result, "min_num_keypoints", integer=True), denominator):
        raise ValueError("min_num_keypoints disagrees with endpoint counts")
    result["min_num_keypoints"] = denominator.astype(np.int64)
    for name, threshold in QUANTILES.items():
        flag = "core_" + name
        expected = (result.structural_bottleneck >= threshold) & ~result.exact_pixel_duplicate
        for column, actual in ((flag, expected), (flag + "_geo500", expected & (result.geo_distance_m >= 500))):
            if column in result and not np.array_equal(boolean_values(result[column]), actual):
                raise ValueError(f"{column} disagrees with predeclared diagnostic slice")
            result[column] = actual
    result["full_geo500"] = result.geo_distance_m >= 500
    return result


def build_full_image_graph(calibrated, manifest):
    nodes = _validate_nodes(manifest)
    edges = _canonicalize_image_endpoints(validate_full_calibrated(calibrated))
    if edges.image_id_a.eq(edges.image_id_b).any() or edges.duplicated(["image_id_a", "image_id_b"]).any():
        raise ValueError("image edges must be unique undirected nonself pairs")
    lookup = nodes.set_index("image_id")
    for side in ("a", "b"):
        ids = edges["image_id_" + side]
        if not ids.isin(lookup.index).all():
            raise ValueError("image endpoint is absent from frozen manifest")
        for column in IMAGE_NODE_COLUMNS[1:]:
            key = column + "_" + side
            expected = lookup.loc[ids, column].to_numpy()
            actual = _numeric(edges, key, integer=True) if column == "row_index" else edges[key].to_numpy()
            if not np.array_equal(actual, expected):
                raise ValueError(f"{key} disagrees with frozen manifest")
            edges[key] = expected
    used = set(edges.image_id_a) | set(edges.image_id_b)
    nodes = nodes.loc[nodes.image_id.isin(used)].sort_values("image_id", kind="stable").reset_index(drop=True)
    return nodes, edges.sort_values("pair_uid", kind="stable").reset_index(drop=True)


def build_full_place_graph(image_nodes, image_edges):
    nodes = _validate_nodes(image_nodes)
    edges = image_edges.copy()
    if edges.pair_uid.duplicated().any() or edges.duplicated(["image_id_a", "image_id_b"]).any():
        raise ValueError("place aggregation requires unique image edges")
    lookup = nodes.set_index("image_id")
    for side in ("a", "b"):
        ids = edges["image_id_" + side]
        if not ids.isin(lookup.index).all():
            raise ValueError("place aggregation image is absent from node table")
        for column in ("place_uid", "city_id"):
            if not np.array_equal(edges[column + "_" + side], lookup.loc[ids, column].to_numpy()):
                raise ValueError("place aggregation metadata differs from image nodes")
    if edges.place_uid_a.eq(edges.place_uid_b).any():
        raise ValueError("same-place edges are forbidden")
    swap = edges.place_uid_a > edges.place_uid_b
    for name in ("image_id", "place_uid", "city_id"):
        left, right = edges[name + "_a"].copy(), edges[name + "_b"].copy()
        edges.loc[swap, name + "_a"] = right.loc[swap]
        edges.loc[swap, name + "_b"] = left.loc[swap]
    keys = ["place_uid_a", "place_uid_b"]
    aggregate = {
        "city_id_a": ("city_id_a", "first"), "city_id_b": ("city_id_b", "first"),
        "num_supporting_image_edges": ("pair_uid", "size"),
        "num_unique_images_a": ("image_id_a", "nunique"), "num_unique_images_b": ("image_id_b", "nunique"),
        "best_rgb_rank_min": ("best_rgb_rank", "min"),
        "min_geo_distance_m": ("geo_distance_m", "min"), "median_geo_distance_m": ("geo_distance_m", "median"),
    }
    for column in ("max_salad_similarity", "num_matches", "local_match_ratio", "structural_bottleneck",
                   "structural_geomean", "symmetric_match_coverage", "symmetric_match_entropy"):
        for operation in ("max", "median"):
            aggregate[column + "_" + operation] = (column, operation)
    place_edges = edges.groupby(keys, sort=True).agg(**aggregate)
    for flag in SLICES[1:]:
        selected = edges.loc[boolean_values(edges[flag])]
        support = selected.groupby(keys, sort=False).agg(
            count=("pair_uid", "size"), views_a=("image_id_a", "nunique"), views_b=("image_id_b", "nunique"))
        for suffix, source in (("image_edges", "count"), ("unique_images_a", "views_a"), ("unique_images_b", "views_b")):
            place_edges[f"num_{flag}_{suffix}"] = support[source].reindex(place_edges.index, fill_value=0).astype(np.int64)
        count = place_edges[f"num_{flag}_image_edges"]
        a, b = place_edges[f"num_{flag}_unique_images_a"], place_edges[f"num_{flag}_unique_images_b"]
        place_edges[flag] = count > 0
        place_edges[f"{flag}_independent_view_support"] = np.minimum(a, b)
        for minimum in (2, 3):
            place_edges[f"{flag}_repeated_support_{minimum}"] = count >= minimum
            place_edges[f"{flag}_independent_support_{minimum}"] = (count >= minimum) & (a >= 2) & (b >= 2)
    place_edges = place_edges.reset_index()
    place_edges.insert(0, "place_pair_uid", [place_pair_uid(a, b) for a, b in place_edges[keys].itertuples(index=False, name=None)])
    place_edges["same_city"] = place_edges.city_id_a == place_edges.city_id_b
    place_edges["independent_view_support"] = np.minimum(place_edges.num_unique_images_a, place_edges.num_unique_images_b)
    place_nodes = nodes.loc[:, PLACE_NODE_COLUMNS].drop_duplicates().sort_values("place_uid", kind="stable").reset_index(drop=True)
    return place_nodes, place_edges


def write_full_graph_artifacts(runtime_dir, image_nodes, image_edges, place_nodes, place_edges,
                               calibration_summary, *, repo_root=ROOT, source_validation=None):
    root, runtime = Path(repo_root), Path(runtime_dir)
    if not runtime.is_absolute():
        runtime = root / runtime
    paths = [runtime / name for name in (*GRAPH_FILENAMES, "full_graph_summary.json")]
    guard_step2d(paths, repo_root=root)
    tables = (image_nodes, image_edges, place_nodes, place_edges)
    summary = {"schema_version": 1, "stage": "step2d_full_countermine_graph", "complete": True,
               "configuration": {"seed": 42, "slices": SLICE_DEFINITIONS,
                                 "node_universe": "all full-population image endpoints and their places"},
               "provenance": dict(calibration_summary.get("provenance", {})),
               "calibration_summary_sha256": sha256_file(runtime / "full_calibration_summary.json"),
               "calibrated_metrics_sha256": sha256_file(runtime / "full_calibrated_candidates.csv"),
               "graph_code_hashes": code_hashes(GRAPH_SOURCE_FILES, repo_root=root),
               "source_validation": dict(source_validation or {}), "artifacts": {}}
    summary["provenance"].update(real_rgb_only=True, synthetic_images_used=False)
    for name, table in zip(GRAPH_FILENAMES, tables):
        write_csv(runtime / name, table, repo_root=root)
        summary["artifacts"][name] = {"sha256": sha256_file(runtime / name), "count": len(table), "columns": list(table.columns)}
    write_json(runtime / "full_graph_summary.json", summary, repo_root=root)
    return summary


def run_full_graph(runtime_dir=DEFAULT_DIR, *, repo_root=ROOT):
    from countermine.mining.full_structural_calibration import load_full_calibrated, validate_pilot_reproduction
    from countermine.mining.full_structural_measurement import load_full_metrics
    root, runtime = Path(repo_root), Path(runtime_dir)
    if not runtime.is_absolute():
        runtime = root / runtime
    guard_step2d([runtime / name for name in (*GRAPH_FILENAMES, "full_graph_summary.json")], repo_root=root)
    metrics, _ = load_full_metrics(runtime, repo_root=root)
    replay = validate_pilot_reproduction(metrics, repo_root=root)
    calibrated, calibration = load_full_calibrated(runtime_dir=runtime, repo_root=root)
    if replay != calibration.get("pilot_reproduction"):
        raise ValueError("full graph pilot reproduction differs from calibration")
    manifest = read_metadata_csv(root / "cache/gsv_mini/manifest.csv")
    geometry = validate_graph_source_geometry(calibrated, manifest)
    images = build_full_image_graph(calibrated, manifest)
    places = build_full_place_graph(*images)
    return write_full_graph_artifacts(runtime, *images, *places, calibration, repo_root=root, source_validation=geometry)


def load_full_graph(runtime_dir=DEFAULT_DIR, *, repo_root=ROOT, source_validation=True):
    root, runtime = Path(repo_root), Path(runtime_dir)
    if not runtime.is_absolute():
        runtime = root / runtime
    summary = read_json(runtime / "full_graph_summary.json")
    if summary.get("schema_version") != 1 or summary.get("stage") != "step2d_full_countermine_graph" or summary.get("complete") is not True:
        raise ValueError("full graph publication is incomplete")
    tables = []
    for name in GRAPH_FILENAMES:
        metadata = summary.get("artifacts", {}).get(name, {})
        if metadata.get("sha256") != sha256_file(runtime / name):
            raise ValueError(f"full graph artifact checksum mismatch: {name}")
        table = read_metadata_csv(runtime / name)
        # Derived graph distances are written with float64's shortest round-trip
        # representation. pandas.to_numeric's string parser can lose an ULP at
        # million-metre magnitudes, exceeding the strict reproduction tolerance.
        # Restore these values with the round-trip float parser before validation;
        # keep identity columns as strings and all comparison tolerances intact.
        for column in GEOGRAPHIC_COLUMNS:
            if column in table:
                table[column] = table[column].astype(np.float64)
        if len(table) != metadata.get("count") or list(table.columns) != metadata.get("columns"):
            raise ValueError("full graph artifact schema or count mismatch")
        tables.append(table)
    image_nodes = _validate_nodes(tables[0])
    image_edges = validate_full_calibrated(tables[1])
    place_nodes = _validate_nodes(tables[2], PLACE_NODE_COLUMNS)
    place_edges = tables[3]
    for column in place_edges:
        if column in ("same_city", *SLICES[1:]) or "_support_" in column:
            place_edges[column] = boolean_values(place_edges[column])
        elif column.startswith("num_") or column.endswith(("_max", "_median", "_distance_m", "_view_support")) or column == "best_rgb_rank_min":
            place_edges[column] = _numeric(place_edges, column)
    if source_validation:
        from countermine.mining.full_structural_calibration import load_full_calibrated, validate_pilot_reproduction
        from countermine.mining.full_structural_measurement import load_full_metrics
        metrics, _ = load_full_metrics(runtime, repo_root=root)
        replay = validate_pilot_reproduction(metrics, repo_root=root)
        calibrated, calibration = load_full_calibrated(runtime_dir=runtime, repo_root=root)
        if replay != calibration.get("pilot_reproduction"):
            raise ValueError("full graph source pilot reproduction differs from calibration")
        if summary.get("calibration_summary_sha256") != sha256_file(runtime / "full_calibration_summary.json") or summary.get("calibrated_metrics_sha256") != sha256_file(runtime / "full_calibrated_candidates.csv"):
            raise ValueError("full graph calibration checksum differs")
        if summary.get("graph_code_hashes") != code_hashes(GRAPH_SOURCE_FILES, repo_root=root):
            raise ValueError("full graph code provenance differs")
        if summary.get("provenance") != calibration.get("provenance"):
            raise ValueError("full graph source provenance differs from calibration")
        manifest = read_metadata_csv(root / "cache/gsv_mini/manifest.csv")
        expected_images = build_full_image_graph(calibrated, manifest)
        expected_places = build_full_place_graph(*expected_images)
        for actual, expected in zip((image_nodes, image_edges, place_nodes, place_edges), (*expected_images, *expected_places)):
            if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
                raise ValueError("full graph does not reproduce calibrated full population")
            for column in expected:
                if pd.api.types.is_numeric_dtype(expected[column]) and not pd.api.types.is_bool_dtype(expected[column]):
                    valid = np.allclose(pd.to_numeric(actual[column]), expected[column], atol=1e-12, rtol=0)
                else:
                    valid = np.array_equal(actual[column], expected[column])
                if not valid:
                    raise ValueError(f"full graph reproduction differs in {column}")
    return image_nodes, image_edges, place_nodes, place_edges, summary

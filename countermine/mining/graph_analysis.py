"""CPU-only descriptive topology of the frozen CounterMine graph pilot.

Each slice keeps the full endpoint-node universe, including isolated nodes.
Evidence values describe a graph; they have no training interpretation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from countermine.mining.structural_analysis import distribution, validate_snapshot


SLICES = ("full", "core_q95", "core_q99", "core_q95_geo500", "core_q99_geo500")
RANK_BINS = ("rank_1", "rank_2_5", "rank_6_10", "rank_11_20", "rank_21_50")
KEYPOINT_BINS = ("lt256", "256_511", "512_1023", "1024_1535", "ge1536")
GRAPH_SOURCE_FILES = (
    "countermine/mining/structural_calibration.py",
    "countermine/mining/countermine_graph.py",
    "countermine/mining/graph_analysis.py",
    "countermine/mining/graph_visuals.py",
    "tools/11_calibrate_structural_evidence.py",
    "tools/12_build_countermine_graph.py",
    "tools/13_analyze_countermine_graph.py",
)


def boolean_values(values) -> np.ndarray:
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            result.append(bool(value))
        elif isinstance(value, str) and value.lower() in ("true", "false"):
            result.append(value.lower() == "true")
        else:
            raise ValueError("graph flags must contain explicit booleans")
    return np.asarray(result, dtype=bool)


class UnionFind:
    """Union by size and path compression, with deterministic root ties."""

    def __init__(self, nodes):
        nodes = list(nodes)
        if len(nodes) != len(set(nodes)):
            raise ValueError("node identities must be unique")
        self.parent = {node: node for node in nodes}
        self.size = {node: 1 for node in nodes}

    def find(self, node):
        if node not in self.parent:
            raise ValueError(f"unknown graph node: {node}")
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            previous = self.parent[node]
            self.parent[node] = root
            node = previous
        return root

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b] or (self.size[a] == self.size[b] and b < a):
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]

    def components(self):
        groups = {}
        for node in sorted(self.parent):
            groups.setdefault(self.find(node), []).append(node)
        return sorted(groups.values(), key=lambda nodes: (-len(nodes), tuple(nodes)))


def _topology_distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mean": None, "median": None, "q90": None, "q95": None, "max": None}
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "q90": float(np.quantile(values, .9)), "q95": float(np.quantile(values, .95)),
        "max": int(values.max()),
    }


def analyze_topology(nodes, edges, node_key, endpoints):
    """Return topology, degrees and ordered components, retaining isolates."""
    identifiers = list(nodes[node_key])
    union_find = UnionFind(identifiers)
    degrees = {node: 0 for node in identifiers}
    seen = set()
    for a, b in edges.loc[:, list(endpoints)].itertuples(index=False, name=None):
        if a == b:
            raise ValueError("graph slices must not contain self edges")
        key = tuple(sorted((a, b)))
        if key in seen:
            raise ValueError("graph slices must contain unique undirected edges")
        if a not in degrees or b not in degrees:
            raise ValueError("graph edge endpoint is absent from the node table")
        seen.add(key)
        degrees[a] += 1
        degrees[b] += 1
        union_find.union(a, b)
    groups = union_find.components()
    edge_counts = {union_find.find(node): 0 for node in identifiers}
    for a, _ in edges.loc[:, list(endpoints)].itertuples(index=False, name=None):
        root = union_find.find(a)
        edge_counts[root] += 1
    components = [
        {"component_id": f"component_{index:05d}", "node_ids": members,
         "size": len(members), "edge_count": edge_counts[union_find.find(members[0])]}
        for index, members in enumerate(groups, 1)
    ]
    active = [value for value in degrees.values() if value > 0]
    summary = {
        "node_count": len(identifiers), "edge_count": len(seen),
        "active_node_count": len(active), "isolated_node_count": len(identifiers) - len(active),
        "degree_population": "all full-pilot endpoint nodes, including zero-degree isolates",
        "degree": _topology_distribution(list(degrees.values())),
        "active_degree": _topology_distribution(active),
        "connected_component_count": len(groups),
        "non_singleton_component_count": sum(len(group) > 1 for group in groups),
        "component_size_population": "all components, including singleton isolates",
        "component_sizes": _topology_distribution([len(group) for group in groups]),
        "non_singleton_component_sizes": _topology_distribution([len(group) for group in groups if len(group) > 1]),
    }
    return summary, degrees, components


def keypoint_bin(count: int) -> str:
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("keypoint count must be a nonnegative integer")
    return KEYPOINT_BINS[0 if count < 256 else 1 if count < 512 else 2 if count < 1024 else 3 if count < 1536 else 4]


def select_top_edges(edges, top_n=50, ranking="bottleneck"):
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    selected = edges.loc[~boolean_values(edges["exact_pixel_duplicate"])].copy()
    if ranking == "bottleneck":
        keys = ["structural_bottleneck", "structural_geomean", "num_matches", "pair_uid"]
        ascending = [False, False, False, True]
    elif ranking == "ratio":
        keys = ["local_match_ratio", "num_matches", "max_salad_similarity", "pair_uid"]
        ascending = [False, False, False, True]
    else:
        raise ValueError("ranking must be ratio or bottleneck")
    return selected.sort_values(keys, ascending=ascending, kind="stable").head(top_n).copy()


def _quantiles(values, names=("q25", "median", "q75", "q95", "q99")):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("analysis requires finite scalar evidence")
    levels = {"q25": .25, "median": .5, "q75": .75, "q95": .95, "q99": .99}
    return {name: float(np.quantile(values, levels[name])) if len(values) else None for name in names}


def _spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("correlation inputs must be finite equal-length arrays")
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(pd.Series(a).rank(method="average"), pd.Series(b).rank(method="average"))[0, 1])


def _counts_of_flags(frame, names):
    return {name: int(boolean_values(frame[name]).sum()) for name in names}


def _json_value(value):
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("graph snapshot must not contain NaN or Infinity")
    return value


def _compact_image_edges(edges):
    fields = (
        "pair_uid", "image_id_a", "image_id_b", "place_uid_a", "place_uid_b", "city_id_a", "city_id_b",
        "geo_distance_m", "near_geo_500m", "best_rgb_rank", "rank_bin", "max_salad_similarity",
        "mean_salad_similarity", "num_candidate_directions", "num_keypoints_a", "num_keypoints_b",
        "min_num_keypoints", "num_matches", "local_match_ratio", "ratio_null_percentile",
        "match_count_null_percentile", "coverage_null_percentile", "entropy_null_percentile",
        "structural_bottleneck", "structural_geomean", "symmetric_match_coverage",
        "symmetric_match_entropy", "core_q95", "core_q99", "core_q95_geo500", "core_q99_geo500",
    )
    return _json_value(edges.loc[:, [key for key in fields if key in edges]].to_dict("records"))


def select_repeated_place_pairs(place_edges, max_pairs=None):
    """Preserve the four requested support priority groups and lexical ties."""
    frame = place_edges.copy()
    frame["_support_priority"] = 4
    for priority, name in reversed(list(enumerate(("independent_support_3", "independent_support_2", "repeated_support_3", "repeated_support_2")))):
        frame.loc[boolean_values(frame[name]), "_support_priority"] = priority
    frame = frame.loc[frame["_support_priority"] < 4].sort_values(
        ["_support_priority", "structural_bottleneck_max", "num_core_q95_image_edges", "place_uid_a", "place_uid_b"],
        ascending=[True, False, False, True, True], kind="stable",
    )
    if max_pairs is not None:
        frame = frame.head(max_pairs)
    return frame.drop(columns="_support_priority")


def analyze_graphs(image_nodes, image_edges, place_nodes, place_edges):
    """Produce pilot-only topology and audits plus per-node degree tables."""
    edges = image_edges.copy()
    edges["min_num_keypoints"] = np.minimum(edges["num_keypoints_a"], edges["num_keypoints_b"]).astype(np.int64)
    for name in SLICES[1:]:
        edges[name] = boolean_values(edges[name])
    image_reports, place_reports = {}, {}
    image_degrees, place_degrees = {}, {}
    place_components = []
    for name in SLICES:
        image_slice = edges if name == "full" else edges.loc[edges[name]]
        place_slice = place_edges if name == "full" else place_edges.loc[place_edges[f"num_{name}_image_edges"] > 0]
        image_reports[name], image_degrees[name], _ = analyze_topology(image_nodes, image_slice, "image_id", ("image_id_a", "image_id_b"))
        place_reports[name], place_degrees[name], components = analyze_topology(place_nodes, place_slice, "place_uid", ("place_uid_a", "place_uid_b"))
        if name == "core_q95":
            place_components = [
                {"component_id": component["component_id"], "place_uids": component["node_ids"],
                 "size": component["size"], "edge_count": component["edge_count"]}
                for component in components if component["size"] > 1
            ]
    image_audit = image_nodes.copy()
    place_audit = place_nodes.copy()
    for frame, node_key, degrees in ((image_audit, "image_id", image_degrees), (place_audit, "place_uid", place_degrees)):
        frame["candidate_degree_full"] = frame[node_key].map(degrees["full"]).astype(np.int64)
        for name in SLICES[1:]:
            frame[f"degree_{name}"] = frame[node_key].map(degrees[name]).astype(np.int64)
        frame["structural_degree_q95"] = frame["degree_core_q95"]
        denominator = frame["candidate_degree_full"].to_numpy()
        frame["structural_edge_fraction"] = np.divide(frame["structural_degree_q95"], denominator,
                                                        out=np.zeros(len(frame)), where=denominator != 0)
    hubs = {}
    for name, frame, node_key in (("image", image_audit, "image_id"), ("place", place_audit, "place_uid")):
        selected = frame.loc[frame["structural_degree_q95"] > 0].sort_values(
            ["structural_degree_q95", node_key], ascending=[False, True], kind="stable",
        ).head(20)
        columns = [column for column in (node_key, "place_uid", "city_id", "structural_degree_q95", "candidate_degree_full", "structural_edge_fraction") if column in selected]
        columns = list(dict.fromkeys(columns))
        hubs[name] = _json_value(selected.loc[:, columns].to_dict("records"))
    rate_hubs = image_audit.loc[image_audit["candidate_degree_full"] >= 3].sort_values(
        ["structural_edge_fraction", "structural_degree_q95", "candidate_degree_full", "image_id"],
        ascending=[False, False, False, True], kind="stable",
    ).head(20)
    rank_analysis = {}
    for name in RANK_BINS:
        selected = edges.loc[edges["rank_bin"] == name]
        rank_analysis[name] = {
            "count": len(selected), "structural_bottleneck": _quantiles(selected["structural_bottleneck"]),
            **{f"{flag}_count": int(selected[flag].sum()) for flag in ("core_q95", "core_q99")},
            **{f"{flag}_fraction": float(selected[flag].mean()) if len(selected) else None for flag in ("core_q95", "core_q99")},
        }
    top_ratio, top_bottleneck = select_top_edges(edges, ranking="ratio"), select_top_edges(edges)
    denominator_audit = {}
    bins = edges["min_num_keypoints"].map(keypoint_bin)
    for name in KEYPOINT_BINS:
        selected = edges.loc[bins == name]
        denominator_audit[name] = {
            "pair_count": len(selected),
            **{column: _quantiles(selected[column], ("median", "q95")) for column in ("local_match_ratio", "num_matches", "structural_bottleneck")},
            "core_q95_fraction": float(selected["core_q95"].mean()) if len(selected) else None,
        }
    def top_denominators(frame):
        counts = frame["min_num_keypoints"].map(keypoint_bin)
        return {"count": len(frame), "min_num_keypoints": distribution(frame["min_num_keypoints"]),
                "keypoint_bin_counts": {name: int((counts == name).sum()) for name in KEYPOINT_BINS},
                "pair_uids": list(frame["pair_uid"])}
    core = edges.loc[edges["core_q95"]]
    q10 = float(np.quantile(edges["symmetric_match_coverage"], .1)) if len(edges) else None
    low_spatial = core.loc[core["symmetric_match_coverage"] <= q10] if q10 is not None else core
    concentrations = np.maximum(core["source_max_cell_match_fraction"], core["target_max_cell_match_fraction"])
    repeated_flags = ("repeated_support_2", "repeated_support_3", "independent_support_2", "independent_support_3")
    strict_flags = ("core_independent_support_2", "core_independent_support_3")
    repeated = _counts_of_flags(place_edges, repeated_flags)
    repeated["unique_image_count_scope"] = "all image edges supporting the place pair (requested definitions)"
    repeated["strict_core_only"] = {
        **_counts_of_flags(place_edges, strict_flags),
        "unique_image_count_scope": "core_q95 image edges only; both endpoints must have >=2 distinct views",
    }
    geo500_repeated = _counts_of_flags(place_edges, tuple(f"{name}_geo500" for name in repeated_flags + strict_flags))
    geo_sensitivity = {
        "eligibility_min_geo_distance_m": 250.0, "sensitivity_min_geo_distance_m": 500.0,
        "candidate_edges_250_500m": int((edges["geo_distance_m"] < 500).sum()),
        "candidate_edges_ge500m": int((edges["geo_distance_m"] >= 500).sum()),
        "core_q95_edges_250_500m": int((edges["core_q95"] & (edges["geo_distance_m"] < 500)).sum()),
        "core_q99_edges_250_500m": int((edges["core_q99"] & (edges["geo_distance_m"] < 500)).sum()),
        "repeated_support_geo500": geo500_repeated,
        "requested_independence_scope": "geo500 core support edge count and unique images among all >=500m supporting edges",
        "strict_independence_scope": "geo500 core edges and geo500 core-only unique image counts",
    }
    report = {
        "image_graph": image_reports, "place_graph": place_reports,
        "repeated_support": repeated, "rank_analysis": rank_analysis,
        "candidate_structural_correlations": {
            "count": len(edges),
            "max_salad_similarity_vs_structural_bottleneck_spearman": _spearman(edges["max_salad_similarity"], edges["structural_bottleneck"]),
            "best_rgb_rank_vs_structural_bottleneck_spearman": _spearman(edges["best_rgb_rank"], edges["structural_bottleneck"]),
        },
        "small_denominator_audit": {"frozen_bins": list(KEYPOINT_BINS), "by_bin": denominator_audit,
                                    "top50_ratio": top_denominators(top_ratio), "top50_bottleneck": top_denominators(top_bottleneck),
                                    "top50_overlap_count": len(set(top_ratio["pair_uid"]) & set(top_bottleneck["pair_uid"]))},
        "spatial_support_audit": {
            "candidate_q10_symmetric_match_coverage": q10, "core_q95_count": len(core),
            "symmetric_match_coverage": distribution(core["symmetric_match_coverage"]),
            "symmetric_match_entropy": distribution(core["symmetric_match_entropy"]),
            "source_max_cell_match_fraction": distribution(core["source_max_cell_match_fraction"]),
            "target_max_cell_match_fraction": distribution(core["target_max_cell_match_fraction"]),
            "symmetric_max_cell_match_fraction": distribution(concentrations),
            "concentration_definition": "max(source_max_cell_match_fraction, target_max_cell_match_fraction)",
            "core_q95_low_spatial_support_count": len(low_spatial),
            "core_q95_low_spatial_support_pair_uids": list(low_spatial.sort_values("pair_uid")["pair_uid"]),
            "inspection_only": True,
        },
        "geo_sensitivity": geo_sensitivity,
        "top_image_edges": _compact_image_edges(top_bottleneck),
        "top_place_pairs": _json_value(select_repeated_place_pairs(place_edges, 20).to_dict("records")),
        "hub_analysis": {
            "image_core_q95_top20": hubs["image"], "place_core_q95_top20": hubs["place"],
            "selection": "positive core_q95 degree, descending degree then lexical node identity; at most 20",
            "degree_normalized_image_top20": _json_value(rate_hubs.to_dict("records")),
            "degree_normalization_selection": "full degree >=3; descending fraction, core degree, full degree, then image ID",
        },
        "place_core_q95_components": place_components,
    }
    return _json_value(report), image_audit, place_audit


def validate_graph_snapshot(snapshot):
    validate_snapshot(snapshot)
    if snapshot.get("provenance", {}).get("no_new_local_matching") is not True:
        raise ValueError("graph export requires no_new_local_matching=true")
    forbidden = {"descriptors", "matched_coordinates", "image_pixels", "model_weights"}
    def check(value):
        if isinstance(value, dict):
            if forbidden & set(value):
                raise ValueError("graph snapshot contains forbidden bulk data")
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(snapshot)
    json.dumps(snapshot, allow_nan=False)


def build_graph_snapshot(report, calibration_summary, graph_summary, *, repo_root, git_commit, visual_metadata):
    """Copy allowlisted scalar provenance; paths and runtime arrays stay local."""
    from countermine.mining.structural_analysis import sha256_file
    sources = {name: sha256_file(Path(repo_root) / name) for name in GRAPH_SOURCE_FILES}
    provenance = calibration_summary["provenance"]
    snapshot = {
        "schema_version": 1,
        "provenance": {
            "current_git_commit": git_commit,
            **{key: provenance[key] for key in (
                "step2a_snapshot_sha256", "step2a_candidate_metrics_sha256", "step2a_random_metrics_sha256",
                "step2a_measurement_summary_sha256", "step2a_measurement_config_sha256",
                "step2a_image_fingerprints_sha256", "step2a_input_hashes", "validation",
            )},
            "graph_code_sha256": sources,
            "software": provenance["software"],
            "real_rgb_only": True, "synthetic_images_used": False, "no_new_local_matching": True,
            "seed": 42,
            "node_universe": "real RGB image endpoints in the measured candidate pilot; slices preserve this universe",
        },
        "calibration": calibration_summary["calibration"],
        "structural_evidence": {
            "ratio_percentile": "weak ECDF over measured random local_match_ratio; relation-specific if >=100 controls, else global",
            "count_percentile": "weak ECDF over measured random num_matches; relation-specific if >=100 controls, else global",
            "structural_bottleneck": "min(ratio_null_percentile, match_count_null_percentile)",
            "structural_geomean": "sqrt(ratio_null_percentile * match_count_null_percentile)",
            "spatial_support_percentile": "coverage_null_percentile (diagnostic only)",
            "interpretation": "graph-analysis evidence only; future training mapping remains unselected",
            "pilot_slices": {"core_q95": "structural_bottleneck >=0.95 AND not exact_pixel_duplicate",
                             "core_q99": "structural_bottleneck >=0.99 AND not exact_pixel_duplicate",
                             "geo500": "corresponding core slice AND geo_distance_m >=500; sensitivity analysis only"},
        },
        "graph_artifact_hashes": graph_summary.get("output_hashes", {}),
        **report,
        "visualization": visual_metadata,
    }
    snapshot = _json_value(snapshot)
    validate_graph_snapshot(snapshot)
    return snapshot

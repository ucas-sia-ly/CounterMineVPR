"""CPU scalar/topology analysis of the complete fixed CounterMine population.

No threshold is selected from results. Communities/hubs are descriptive,
and repeated place-pair view support has no special scientific status.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from countermine.mining.full_countermine_graph import CORE_SLICES, QUANTILES, SLICES
from countermine.mining.full_structural_io import (
    ROOT, DEFAULT_DIR, code_hashes, frozen_provenance, git_commit as current_git_commit, guard_step2d, read_json,
    validate_portable_json, write_csv, write_json,
)
from countermine.mining.graph_analysis import (
    KEYPOINT_BINS, RANK_BINS, _compact_image_edges, _json_value,
    _spearman, analyze_topology, boolean_values, keypoint_bin,
)
from countermine.mining.structural_analysis import distribution, sha256_file


ANALYSIS_SOURCE_FILES = (
    "countermine/mining/full_graph_analysis.py", "countermine/mining/full_graph_visuals.py",
    "tools/24_analyze_full_countermine_graph.py",
)


def recorded_bank_dir(measurement, runtime_dir, *, repo_root=ROOT):
    relative = measurement.get("configuration", {}).get("feature_bank_directory")
    if relative is None:
        directory = Path(runtime_dir) / "aliked_bank"
    else:
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("recorded feature-bank directory must be a safe repository-relative path")
        directory = Path(repo_root) / relative
    guard_step2d([directory / "summary.json"], repo_root=repo_root)
    return directory


def select_full_top_edges(edges, count=100, *, ranking="bottleneck"):
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("top-edge count must be a positive integer")
    selected = edges.loc[~boolean_values(edges.exact_pixel_duplicate)]
    if ranking == "bottleneck":
        keys = ["structural_bottleneck", "structural_geomean", "num_matches", "max_salad_similarity", "pair_uid"]
    elif ranking == "ratio":
        keys = ["local_match_ratio", "num_matches", "max_salad_similarity", "pair_uid"]
    else:
        raise ValueError("ranking must be ratio or bottleneck")
    return selected.sort_values(keys, ascending=[False] * (len(keys) - 1) + [True], kind="stable").head(count).copy()


def _topology_values(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {key: None for key in ("mean", "median", "q90", "q95", "q99", "max")}
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "q90": float(np.quantile(values, .9)), "q95": float(np.quantile(values, .95)),
            "q99": float(np.quantile(values, .99)), "max": int(values.max())}


def full_topology(nodes, edges, node_key, endpoints):
    summary, degrees, components = analyze_topology(nodes, edges, node_key, endpoints)
    active = summary["active_node_count"]
    largest = max((row["size"] for row in components if row["size"] > 1), default=0)
    summary.update(
        degree_population="all full candidate endpoint nodes, including fixed-universe isolates",
        degree=_topology_values(list(degrees.values())),
        active_degree=_topology_values([value for value in degrees.values() if value]),
        component_sizes=_topology_values([row["size"] for row in components]),
        non_singleton_component_sizes=_topology_values([row["size"] for row in components if row["size"] > 1]),
        largest_component_size=max((row["size"] for row in components), default=0),
        largest_component_fraction_of_active_nodes=largest / active if active else None,
    )
    return summary, degrees, components


def core_rate_summary(frame):
    result = {"count": len(frame), "structural_bottleneck": distribution(frame.structural_bottleneck)}
    for flag in CORE_SLICES:
        selected = boolean_values(frame[flag])
        result[flag + "_count"] = int(selected.sum())
        result[flag + "_fraction"] = float(selected.mean()) if len(frame) else None
    return result


def _pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("correlation requires finite scalars")
    return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 2 and np.ptp(a) and np.ptp(b) else None


def _frozen_rate_comparison(groups, step2c_snapshot):
    frozen = step2c_snapshot["joint_null"]["rates"]
    report = {}
    for name, candidate in groups.items():
        random = frozen[name]["random"]
        row = {"candidate_count": candidate["count"], "frozen_random": dict(random)}
        for quantile in ("q95", "q99"):
            rate, null = candidate["core_" + quantile + "_fraction"], random[quantile + "_fraction"]
            row[quantile + "_absolute_difference"] = rate - null if rate is not None else None
            row[quantile + "_rate_ratio"] = rate / null if rate is not None and null else None
        report[name] = row
    return report


def analyze_full_graphs(image_nodes, image_edges, place_nodes, place_edges, *, step2c_snapshot=None):
    edges = image_edges.copy()
    for flag in SLICES[1:]:
        edges[flag] = boolean_values(edges[flag])
    edges["min_num_keypoints"] = np.minimum(edges.num_keypoints_a, edges.num_keypoints_b).astype(np.int64)
    image_reports, place_reports, image_degrees, place_degrees, component_records = {}, {}, {}, {}, {}
    for name in SLICES:
        selected_images = edges if name == "full" else edges.loc[edges[name]]
        selected_places = place_edges if name == "full" else place_edges.loc[pd.to_numeric(place_edges[f"num_{name}_image_edges"]) > 0]
        image_reports[name], image_degrees[name], _ = full_topology(image_nodes, selected_images, "image_id", ("image_id_a", "image_id_b"))
        place_reports[name], place_degrees[name], components = full_topology(place_nodes, selected_places, "place_uid", ("place_uid_a", "place_uid_b"))
        component_records[name] = [{"component_id": row["component_id"], "place_uids": row["node_ids"],
                                    "size": row["size"], "edge_count": row["edge_count"]}
                                   for row in components if row["size"] > 1]
    audits = []
    for nodes, key, degrees in ((image_nodes, "image_id", image_degrees), (place_nodes, "place_uid", place_degrees)):
        audit = nodes.copy()
        audit["candidate_degree_full"] = audit[key].map(degrees["full"]).astype(np.int64)
        for flag in SLICES[1:]:
            audit["degree_" + flag] = audit[key].map(degrees[flag]).astype(np.int64)
            audit["structural_edge_fraction_" + flag] = np.divide(
                audit["degree_" + flag], audit.candidate_degree_full,
                out=np.zeros(len(audit)), where=audit.candidate_degree_full.to_numpy() != 0)
        audits.append(audit)
    image_audit, place_audit = audits
    hubs = {"fraction_min_full_degree": 5, "interpretation": "descriptive structural-confusion hubs; no usefulness classification"}
    for flag in SLICES[1:]:
        if flag == "full_geo500":
            continue
        degree, fraction = "degree_" + flag, "structural_edge_fraction_" + flag
        columns = ["place_uid", "city_id", "candidate_degree_full", degree, fraction]
        degree_hubs = place_audit.loc[place_audit[degree] > 0].sort_values([degree, "place_uid"], ascending=[False, True], kind="stable").head(50)
        fraction_hubs = place_audit.loc[place_audit.candidate_degree_full >= 5].sort_values(
            [fraction, degree, "candidate_degree_full", "place_uid"], ascending=[False, False, False, True], kind="stable").head(50)
        hubs[flag] = {"top50_core_degree": degree_hubs[columns].to_dict("records"),
                      "top50_structural_edge_fraction": fraction_hubs[columns].to_dict("records")}
    rank_analysis = {name: core_rate_summary(edges.loc[edges.rank_bin == name]) for name in RANK_BINS}
    relations = {"same_city": core_rate_summary(edges.loc[boolean_values(edges.same_city)]),
                 "cross_city": core_rate_summary(edges.loc[~boolean_values(edges.same_city)])}
    groups = {"all": core_rate_summary(edges), **relations}
    denominator = {}
    bins = edges.min_num_keypoints.map(keypoint_bin)
    for name in KEYPOINT_BINS:
        selected = edges.loc[bins == name]
        denominator[name] = {"pair_count": len(selected),
            **{column: {q: distribution(selected[column])[q] for q in ("median", "q95", "q99")}
               for column in ("local_match_ratio", "num_matches", "structural_bottleneck")},
            **{flag + "_fraction": float(selected[flag].mean()) if len(selected) else None for flag in ("core_q95", "core_q99")}}
    ratio_top, bottleneck_top = select_full_top_edges(edges, ranking="ratio"), select_full_top_edges(edges)
    def top_denominators(selected):
        return {"count": len(selected), "median_minimum_keypoints": float(selected.min_num_keypoints.median()) if len(selected) else None,
                "minimum_minimum_keypoints": int(selected.min_num_keypoints.min()) if len(selected) else None,
                "median_match_count": float(selected.num_matches.median()) if len(selected) else None,
                "count_below_256": int((selected.min_num_keypoints < 256).sum()),
                "count_below_512": int((selected.min_num_keypoints < 512).sum())}
    correlations = {"count": len(edges)}
    for column in ("max_salad_similarity", "best_rgb_rank"):
        correlations[column + "_vs_structural_bottleneck"] = {
            "pearson": _pearson(edges[column], edges.structural_bottleneck),
            "spearman": _spearman(edges[column], edges.structural_bottleneck)}
    repeated = {"interpretation": "descriptive only; strict independent multi-view place-pair support was not unusual under the pilot fixed-graph null"}
    for flag in SLICES[1:]:
        if flag == "full_geo500":
            continue
        repeated[flag] = {"place_pairs_with_2_core_edges": int(boolean_values(place_edges[flag + "_repeated_support_2"]).sum()),
                          "place_pairs_with_3_core_edges": int(boolean_values(place_edges[flag + "_repeated_support_3"]).sum()),
                          "core_independent_support_2": int(boolean_values(place_edges[flag + "_independent_support_2"]).sum()),
                          "core_independent_support_3": int(boolean_values(place_edges[flag + "_independent_support_3"]).sum()),
                          "core_independent_view_support": distribution(pd.to_numeric(place_edges[flag + "_independent_view_support"]))}
    top_metadata = _compact_image_edges(bottleneck_top)
    for row, (_, selected) in zip(top_metadata, bottleneck_top.iterrows()):
        row.update({flag: bool(selected[flag]) for flag in SLICES[1:]})
    report = {"full_candidate_statistics": {"all": groups["all"],
              "exact_pixel_duplicate_count": int(boolean_values(edges.exact_pixel_duplicate).sum()), "raw_match_metrics": {
                  column: distribution(edges[column]) for column in (
                      "num_keypoints_a", "num_keypoints_b", "min_num_keypoints", "num_matches", "local_match_ratio",
                      "symmetric_match_coverage", "symmetric_match_entropy", "source_max_cell_match_fraction", "target_max_cell_match_fraction")}},
              "rank_analysis": rank_analysis, "relation_analysis": relations,
              "candidate_structural_correlations": correlations,
              "denominator_audit": {"frozen_bins": list(KEYPOINT_BINS), "by_bin": denominator,
                  "top100_raw_ratio": top_denominators(ratio_top), "top100_structural_bottleneck": top_denominators(bottleneck_top),
                  "top100_overlap_count": len(set(ratio_top.pair_uid) & set(bottleneck_top.pair_uid))},
              "image_graph": image_reports, "place_graph": place_reports, "hub_analysis": hubs,
              "repeated_support_descriptive": repeated,
              "geo500_sensitivity": {"eligibility_minimum_m": 250., "sensitivity_minimum_m": 500.,
                  "candidate_edges_250_500m": int((edges.geo_distance_m < 500).sum()),
                  "candidate_edges_ge500m": int((edges.geo_distance_m >= 500).sum()),
                  "core_counts_ge500m": {name: int(edges[name + "_geo500"].sum()) for name in CORE_SLICES},
                  "original_eligibility_preserved": True},
              "top_edges": top_metadata}
    if step2c_snapshot is not None:
        report["candidate_enrichment_consistency"] = _frozen_rate_comparison(groups, step2c_snapshot)
    return _json_value(report), image_audit, place_audit, component_records


def build_full_snapshot(report, population, measurement, calibration, graph, *, runtime_dir,
                        repo_root=ROOT, git_commit, visual_metadata):
    root, runtime = Path(repo_root), Path(runtime_dir)
    provenance = dict(frozen_provenance(repo_root=root))
    bank_dir = recorded_bank_dir(measurement, runtime, repo_root=root)
    files = {
        "manifest_sha256": root / "cache/gsv_mini/manifest.csv",
        "raw_candidate_csv_sha256": root / "cache/gsv_mini/rgb_candidates_raw.csv",
        "feature_bank_summary_sha256": bank_dir / "summary.json",
        "full_population_sha256": runtime / "full_population.csv",
        "full_metrics_sha256": runtime / "full_candidate_structural_metrics.csv",
        "full_calibration_summary_sha256": runtime / "full_calibration_summary.json",
    }
    provenance.update({name: sha256_file(path) for name, path in files.items()})
    provenance.update(git_commit_at_export=git_commit, real_rgb_only=True, synthetic_images_used=False,
                      seed=42, graph_code_hashes=graph["graph_code_hashes"],
                      analysis_code_hashes=code_hashes(ANALYSIS_SOURCE_FILES, repo_root=root),
                      calibration_code_hashes=calibration.get("provenance", {}).get("code_sha256", {}),
                      calibration_configuration=calibration.get("configuration", {}),
                      calibration_configuration_sha256=hashlib.sha256(json.dumps(
                          calibration.get("configuration", {}), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
    pilot = calibration.get("pilot_reproduction")
    if not isinstance(pilot, dict) or pilot.get("passed") is not True or pilot.get("pair_count") != 5000:
        raise ValueError("full scientific export requires the fail-closed pilot reproduction report")
    bank = read_json(bank_dir / "summary.json")
    snapshot = {"schema_version": 1, "stage": "step2d_full_countermine", "complete": True,
        "provenance": provenance,
        "population": {"canonical_edge_count": graph["artifacts"]["image_edges.csv"]["count"],
            "unique_images": graph["artifacts"]["image_nodes.csv"]["count"],
            "unique_places": graph["artifacts"]["place_nodes.csv"]["count"],
            "reconstruction": population.get("population", population.get("counts", {})),
            "same_city_count": report["relation_analysis"]["same_city"]["count"],
            "cross_city_count": report["relation_analysis"]["cross_city"]["count"]},
        "measurement": {"shard_count": measurement.get("shard_count", measurement.get("expected_shard_count")),
            "completed_shard_count": measurement.get("completed_shard_count", measurement.get("shard_count")),
            "feature_bank_image_count": bank.get("image_count", bank.get("manifest_image_count")),
            "runtime_throughput_summary": measurement.get("runtime_throughput_summary", measurement.get("throughput", measurement.get("benchmark", {}))),
            "duplicate_count": report["full_candidate_statistics"]["exact_pixel_duplicate_count"]},
        "pilot_reproduction": pilot,
        "calibration": {"frozen_random_null_definition": "same/cross-city weak ECDF count(random <= x)/N_relation",
                        "relation_counts": calibration.get("relation_null_counts", {}),
                        "q95_q99_thresholds": calibration.get("frozen_q95_q99_thresholds", {}),
                        "frozen_random_joint_rates": calibration.get("frozen_random_joint_rates", {}),
                        "frozen_null_preflight": calibration.get("frozen_null_preflight", {})},
        "configuration": {"seed": 42, "predeclared_core_thresholds": QUANTILES,
            "node_universe": "full candidate endpoints; every slice preserves isolates",
            "structural_bottleneck": "min(ratio_null_percentile, match_count_null_percentile)",
            "structural_geomean": "sqrt(ratio_null_percentile * match_count_null_percentile)",
            "interpretation": "descriptive structural-confusion communities/hubs; no training mapping"},
        "graph_artifact_hashes": {name: item["sha256"] for name, item in graph["artifacts"].items()},
        **report, "visualization": visual_metadata,
    }
    validate_full_snapshot(snapshot)
    return snapshot


def validate_full_snapshot(snapshot):
    validate_portable_json(snapshot)
    provenance = snapshot.get("provenance", {})
    if provenance.get("real_rgb_only") is not True or provenance.get("synthetic_images_used") is not False:
        raise ValueError("full graph export requires original real-RGB provenance")
    forbidden = {"descriptors", "keypoints", "matched_coordinates", "points_a", "points_b", "image_pixels", "model_weights"}
    def inspect(value):
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise ValueError("full graph snapshot contains forbidden bulk data")
            for child in value.values():
                inspect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                inspect(child)
    inspect(snapshot)


def export_full_audit(runtime_dir=DEFAULT_DIR, *, output_dir="outputs/step2d",
                      snapshot="docs/audits/step2d_full_countermine_metrics.json",
                      dataset_root="data/gsv-cities", repo_root=ROOT):
    from countermine.mining.full_countermine_graph import load_full_graph
    from countermine.mining.full_graph_visuals import ARTIFACT_NAMES, create_full_graph_visuals, curate_full_graph_artifacts
    from countermine.mining.full_structural_calibration import load_full_calibrated, validate_pilot_reproduction
    from countermine.mining.full_structural_population import load_full_population
    from countermine.mining.full_structural_measurement import load_full_metrics
    root = Path(repo_root)
    def absolute(path):
        path = Path(path)
        return path if path.is_absolute() else root / path
    runtime, output, destination, dataset = map(absolute, (runtime_dir, output_dir, snapshot, dataset_root))
    if destination.parent.resolve() != (root / "docs/audits").resolve() or not destination.name.startswith("step2d_") or destination.suffix != ".json":
        raise ValueError("full audit snapshot must be docs/audits/step2d_*.json")
    targets = [destination, runtime / "full_graph_analysis_summary.json", runtime / "image_node_audit.csv",
               runtime / "place_node_audit.csv", runtime / "full_place_components.json"]
    targets += [output / name for name in ARTIFACT_NAMES]
    targets += [destination.parent / ("step2d_" + name) for name in ARTIFACT_NAMES]
    guard_step2d(targets, repo_root=root)
    _, population = load_full_population(output_dir=runtime, repo_root=root)
    metrics, measurement = load_full_metrics(runtime, repo_root=root)
    replay = validate_pilot_reproduction(metrics, repo_root=root)
    _, calibration = load_full_calibrated(runtime_dir=runtime, repo_root=root)
    if replay != calibration.get("pilot_reproduction"):
        raise ValueError("full graph analysis pilot reproduction differs from saved calibration audit")
    images, edges, places, place_edges, graph = load_full_graph(runtime, repo_root=root)
    report, image_audit, place_audit, components = analyze_full_graphs(
        images, edges, places, place_edges, step2c_snapshot=read_json(root / "docs/audits/step2c_null_topology_metrics.json"))
    bank_dir = recorded_bank_dir(measurement, runtime, repo_root=root)
    source_hashes = {runtime / name: sha256_file(runtime / name) for name in (
        "full_population_summary.json", "full_measurement_summary.json", "full_calibration_summary.json",
        "full_graph_summary.json", *graph["artifacts"])}
    source_hashes[bank_dir / "summary.json"] = sha256_file(bank_dir / "summary.json")
    visual = create_full_graph_visuals(images, edges, places, place_edges, report, place_audit, components,
                                      root / "cache/gsv_mini/manifest.csv", dataset,
                                      bank_dir, output, repo_root=root)
    visual["figure_sha256"] = {"step2d_" + name: sha256_file(output / name) for name in ARTIFACT_NAMES}
    commit = current_git_commit(root)
    scientific = build_full_snapshot(report, population, measurement, calibration, graph,
        runtime_dir=runtime, repo_root=root, git_commit=commit, visual_metadata=visual)
    if any(sha256_file(path) != digest for path, digest in source_hashes.items()):
        raise ValueError("Step 2D inputs changed during analysis/figure export")
    # Revalidate the immutable historical provenance immediately before publish.
    if scientific["provenance"] != dict(scientific["provenance"], **frozen_provenance(repo_root=root)):
        raise ValueError("frozen historical provenance changed during Step 2D export")
    guard_step2d(targets, repo_root=root)
    write_csv(runtime / "image_node_audit.csv", image_audit, repo_root=root)
    write_csv(runtime / "place_node_audit.csv", place_audit, repo_root=root)
    write_json(runtime / "full_place_components.json", _json_value(components), repo_root=root)
    curate_full_graph_artifacts(output, destination.parent, repo_root=root)
    write_json(runtime / "full_graph_analysis_summary.json", scientific, repo_root=root)
    write_json(destination, scientific, repo_root=root)
    return scientific

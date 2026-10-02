"""Deterministic, fail-closed Step 2D pair shards and CPU finalization.

Only compact scalar pair metadata is collected. Feature tensors remain in the
matcher's bounded CPU LRU and the current pair's device allocation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import math
import os
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import pandas as pd

from countermine.mining.full_structural_io import (
    ROOT, DEFAULT_DIR, code_hashes, frozen_provenance, guard_step2d,
    read_json, relative_path, write_csv, write_json,
)
from countermine.mining.local_feature_bank import (
    BANK_DIR, load_bank_manifest, load_feature_bank, positive_integer,
    require_bank_validation, verify_source_images,
)
from countermine.mining.structural_analysis import METRIC_COLUMNS, read_metadata_csv, validate_metrics
from countermine.mining.structural_matcher import LIGHTGLUE_CONFIG, canonical_json_bytes, sha256_file
from countermine.mining.full_population_metadata import POPULATION_COLUMNS


MEASUREMENT_COLUMNS = (*POPULATION_COLUMNS, *METRIC_COLUMNS, "min_num_keypoints")
MEASUREMENT_CODE = (
    "countermine/mining/full_structural_measurement.py",
    "countermine/mining/banked_structural_matcher.py",
    "countermine/mining/local_feature_bank.py",
    "countermine/mining/full_structural_io.py",
    "countermine/mining/structural_matcher.py", "tools/20_measure_full_structural_pairs.py",
)
EXACT_REPLAY_METRICS = ("num_keypoints_a", "num_keypoints_b", "num_matches", "exact_pixel_duplicate")
FLOAT_REPLAY_METRICS = tuple(key for key in METRIC_COLUMNS if key not in EXACT_REPLAY_METRICS)


def sequence_sha256(pair_uids):
    return hashlib.sha256(canonical_json_bytes(list(pair_uids))).hexdigest()


def shard_boundaries(pair_count, shard_size=2000):
    pair_count = positive_integer(pair_count, "pair_count", 1)
    shard_size = positive_integer(shard_size, "shard_size", 1)
    return [(index, start, min(start + shard_size, pair_count))
            for index, start in enumerate(range(0, pair_count, shard_size))]


def shard_paths(runtime_dir, index):
    stem = f"shard_{positive_integer(index, 'shard_index'):05d}"
    directory = Path(runtime_dir) / "shards"
    return directory / f"{stem}.csv", directory / f"{stem}_summary.json"


def compare_metric_frames(measured, historical, *, expected_count=None):
    """Exact counts/identity and <=1e-12 diagnostics; report every max drift."""
    if expected_count is not None and len(historical) != expected_count:
        raise ValueError("pilot reproduction historical pair count differs")
    if historical.empty or not historical["pair_uid"].is_unique or not measured["pair_uid"].is_unique:
        raise ValueError("pilot reproduction requires unique nonempty pair identities")
    measured = measured.set_index("pair_uid")
    historical = historical.set_index("pair_uid")
    if not historical.index.isin(measured.index).all():
        raise ValueError("pilot reproduction has missing historical pair identities")
    measured = measured.loc[historical.index]
    for key in ("image_id_a", "image_id_b"):
        if measured[key].tolist() != historical[key].tolist():
            raise ValueError(f"pilot reproduction endpoint identity/orientation drift: {key}")
    differences = {}
    for key in EXACT_REPLAY_METRICS:
        if key == "exact_pixel_duplicate":
            parse = lambda values: values.map(lambda value: str(value).lower())
            equal = np.array_equal(parse(measured[key]), parse(historical[key]))
            differences[key] = 0 if equal else 1
        else:
            a = pd.to_numeric(measured[key], errors="raise").to_numpy(dtype=np.float64)
            b = pd.to_numeric(historical[key], errors="raise").to_numpy(dtype=np.float64)
            equal = np.isfinite(a).all() and np.isfinite(b).all() and np.array_equal(a, b)
            differences[key] = float(np.max(np.abs(a - b)))
        if not equal:
            raise ValueError(f"pilot reproduction exact metric drift: {key}; max_difference={differences[key]}; STOP")
    for key in FLOAT_REPLAY_METRICS:
        a = pd.to_numeric(measured[key], errors="raise").to_numpy(dtype=np.float64)
        b = pd.to_numeric(historical[key], errors="raise").to_numpy(dtype=np.float64)
        difference = float(np.max(np.abs(a - b)))
        differences[key] = difference
        if not np.isfinite(a).all() or not np.isfinite(b).all() or difference > 1e-12:
            raise ValueError(f"pilot reproduction metric drift: {key}; max_absolute_difference={difference:.17g}; tolerance=1e-12; STOP")
    return {"passed": True, "pair_count": len(historical), "absolute_tolerance": 1e-12,
            "maximum_metric_differences": differences}


def validate_full_metric_frame(frame, population, bank_index):
    if tuple(frame.columns) != MEASUREMENT_COLUMNS:
        raise ValueError("full measurement schema differs from frozen Step 2D columns")
    frame = validate_metrics(frame, candidate=True)
    if frame["pair_uid"].tolist() != population["pair_uid"].tolist():
        raise ValueError("shard/full measurement pair identity sequence differs from population")
    identities = ("pair_uid", "image_id_a", "image_id_b", "place_uid_a", "place_uid_b", "city_id_a", "city_id_b", "rank_bin",
                  "anchor_query_image_id", "anchor_negative_image_id")
    for column in identities:
        if frame[column].tolist() != population[column].tolist():
            raise ValueError(f"full measurement metadata differs from population: {column}")
    integer_columns = ("row_index_a", "row_index_b", "num_candidate_directions", "best_rgb_rank",
                       "anchor_query_row_index", "anchor_negative_row_index", "anchor_rank")
    for column in integer_columns:
        a, b = pd.to_numeric(frame[column]), pd.to_numeric(population[column])
        if not np.array_equal(a, b):
            raise ValueError(f"full measurement integer metadata differs: {column}")
    for column in ("geo_distance_m", "max_salad_similarity", "mean_salad_similarity", "anchor_similarity",
                   "a_to_b_rank", "a_to_b_similarity", "b_to_a_rank", "b_to_a_similarity"):
        a = pd.to_numeric(frame[column].replace("", np.nan), errors="raise").to_numpy(dtype=np.float64)
        b = pd.to_numeric(population[column].replace("", np.nan), errors="raise").to_numpy(dtype=np.float64)
        tolerance = 1e-6 if column == "geo_distance_m" else 1e-12
        if not np.allclose(a, b, rtol=0, atol=tolerance, equal_nan=True):
            raise ValueError(f"full measurement numeric metadata differs: {column}")
    minimum = np.minimum(frame["num_keypoints_a"], frame["num_keypoints_b"])
    if not np.array_equal(pd.to_numeric(frame["min_num_keypoints"], errors="raise"), minimum):
        raise ValueError("min_num_keypoints differs from frozen endpoint counts")
    frame["min_num_keypoints"] = minimum.astype(np.int64)
    records = bank_index.set_index("image_id")
    for endpoint in ("a", "b"):
        identities = frame[f"image_id_{endpoint}"]
        if not identities.isin(records.index).all():
            raise ValueError("measurement source image is missing from feature bank")
        if not np.array_equal(frame[f"num_keypoints_{endpoint}"].to_numpy(), records.loc[identities, "num_keypoints"].to_numpy()):
            raise ValueError("measurement keypoint count differs from feature bank")
        if not np.array_equal(pd.to_numeric(frame[f"row_index_{endpoint}"]), records.loc[identities, "row_index"]):
            raise ValueError("measurement source row index differs from feature bank")
    duplicate = records.loc[frame["image_id_a"], "rgb_pixel_sha256"].to_numpy() == records.loc[frame["image_id_b"], "rgb_pixel_sha256"].to_numpy()
    if not np.array_equal(frame["exact_pixel_duplicate"], duplicate):
        raise ValueError("measurement exact-pixel duplicate flag differs from feature bank")
    return frame


def shard_binding(population, configuration, index, start, end):
    return {
        "schema_version": 1, "shard_index": index, "start_row": start, "end_row": end,
        "pair_count": end - start,
        "pair_uid_sequence_sha256": sequence_sha256(population.iloc[start:end]["pair_uid"]),
        "population_sha256": configuration["population_sha256"],
        "feature_bank_summary_sha256": configuration["feature_bank_summary_sha256"],
        "matcher_configuration_sha256": configuration["matcher_configuration_sha256"],
        "code_sha256": configuration["code_sha256"],
        "configuration_sha256": hashlib.sha256(canonical_json_bytes(configuration)).hexdigest(),
    }


def load_valid_shard(runtime_dir, population, bank_index, configuration, index, start, end):
    csv_path, summary_path = shard_paths(runtime_dir, index)
    if not summary_path.exists():
        return None
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError("resume refused: shard summary is not an object")
    expected = shard_binding(population, configuration, index, start, end)
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError(f"resume refused: shard {index} pair/configuration/bank/code provenance differs")
    if summary.get("complete") is not True:
        return None
    if not csv_path.is_file() or summary.get("metrics_sha256") != sha256_file(csv_path):
        raise ValueError(f"resume refused: completed shard {index} is missing or corrupt")
    if not isinstance(summary.get("start_timestamp"), str) or not isinstance(summary.get("end_timestamp"), str):
        raise ValueError("completed shard has missing timestamps")
    validate_benchmark(summary.get("benchmark"), end - start, len(population))
    frame = validate_full_metric_frame(read_metadata_csv(csv_path), population.iloc[start:end], bank_index)
    return frame, summary


def validate_benchmark(benchmark, pair_count, full_pair_count):
    if not isinstance(benchmark, dict):
        raise ValueError("completed shard has missing or malformed benchmark metadata")
    required = {"pairs_processed", "elapsed_seconds", "pairs_per_second", "estimated_full_matching_seconds",
                "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "cpu_peak_rss_bytes"}
    if not required.issubset(benchmark):
        raise ValueError("completed shard benchmark metadata lacks required fields")
    if benchmark.get("pairs_processed") != pair_count or isinstance(benchmark.get("pairs_processed"), bool):
        raise ValueError("completed shard benchmark pair count differs")
    for key in ("elapsed_seconds", "pairs_per_second", "estimated_full_matching_seconds"):
        value = benchmark.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"completed shard benchmark has invalid {key}")
    if not math.isclose(benchmark["pairs_per_second"], pair_count / benchmark["elapsed_seconds"], rel_tol=1e-12, abs_tol=0):
        raise ValueError("completed shard benchmark throughput differs from elapsed time/count")
    if not math.isclose(benchmark["estimated_full_matching_seconds"], full_pair_count / benchmark["pairs_per_second"], rel_tol=1e-12, abs_tol=0):
        raise ValueError("completed shard benchmark full-run estimate differs")
    for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "cpu_peak_rss_bytes"):
        value = benchmark.get(key)
        if value is None and key.startswith("peak_cuda_"):
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"completed shard benchmark has invalid {key}")
    allocated, reserved = benchmark["peak_cuda_allocated_bytes"], benchmark["peak_cuda_reserved_bytes"]
    if (allocated is None) != (reserved is None) or (allocated is not None and allocated > reserved):
        raise ValueError("completed shard CUDA memory peak metadata is inconsistent")
    return benchmark


def aggregate_throughput(artifacts, pair_count):
    elapsed = sum(item["benchmark"]["elapsed_seconds"] for item in artifacts)
    return {
        "pairs_processed": pair_count, "elapsed_matching_seconds": elapsed,
        "pairs_per_second": pair_count / elapsed,
        "peak_cuda_allocated_bytes": max((item["benchmark"]["peak_cuda_allocated_bytes"] or 0) for item in artifacts),
        "peak_cuda_reserved_bytes": max((item["benchmark"]["peak_cuda_reserved_bytes"] or 0) for item in artifacts),
        "cpu_peak_rss_bytes": max(item["benchmark"]["cpu_peak_rss_bytes"] for item in artifacts),
    }


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _load_configuration(runtime_dir, population_summary, bank_dir, bank_summary, *, repo_root=ROOT):
    configuration = read_json(Path(runtime_dir) / "full_measurement_config.json")
    if configuration.get("schema_version") != 1 or configuration.get("seed") != 42:
        raise ValueError("full measurement configuration schema/seed differs")
    if configuration["population_sha256"] != sha256_file(Path(runtime_dir) / "full_population.csv"):
        raise ValueError("full measurement population hash differs")
    if configuration["population_summary_sha256"] != sha256_file(Path(runtime_dir) / "full_population_summary.json"):
        raise ValueError("full measurement population summary hash differs")
    if configuration["feature_bank_summary_sha256"] != sha256_file(Path(bank_dir) / "summary.json"):
        raise ValueError("full measurement feature-bank hash differs")
    if configuration["code_sha256"] != code_hashes(MEASUREMENT_CODE, repo_root=repo_root):
        raise ValueError("full measurement code provenance changed")
    if configuration["frozen_provenance"] != frozen_provenance(repo_root=repo_root):
        raise ValueError("frozen scientific input provenance changed")
    if configuration["bank_validation_sha256"] != sha256_file(Path(bank_dir) / "validation.json"):
        raise ValueError("bank replay marker differs from measurement provenance")
    require_bank_validation(bank_dir, bank_summary, repo_root=repo_root)
    if configuration["lightglue_config"] != LIGHTGLUE_CONFIG:
        raise ValueError("full measurement does not use frozen LightGlue settings")
    if configuration["matcher_configuration_sha256"] != hashlib.sha256(canonical_json_bytes(configuration["matcher_configuration"])).hexdigest():
        raise ValueError("full measurement matcher configuration hash differs")
    positive_integer(configuration["shard_size"], "shard_size", 1)
    return configuration


def run_full_measurement(args, *, repo_root=ROOT, matcher_factory=None):
    from countermine.mining.full_structural_population import load_full_population
    if matcher_factory is None:
        from countermine.mining.banked_structural_matcher import BankedStructuralMatcher
        matcher_factory = BankedStructuralMatcher
    runtime = Path(args.runtime_dir)
    root = Path(repo_root)
    shard_size = positive_integer(args.shard_size, "shard_size", 1)
    start_shard = positive_integer(args.start_shard, "start_shard")
    maximum = None if args.max_shards is None else positive_integer(args.max_shards, "max_shards", 1)
    guard_step2d([runtime / "full_measurement_config.json", runtime / ".matching.lock"], repo_root=root)
    population, population_summary = load_full_population(runtime, repo_root=root)
    index, bank_summary = load_feature_bank(args.bank_dir, repo_root=root)
    validation = require_bank_validation(args.bank_dir, bank_summary, repo_root=root)
    manifest, _, _ = load_bank_manifest(root / bank_summary["configuration"]["manifest_file"], repo_root=root)
    verify_source_images(index, manifest, args.dataset_root)
    matcher = matcher_factory(bank_dir=args.bank_dir, device=args.device, seed=args.seed,
                              feature_cache_size=args.feature_cache_size, repo_root=root)
    matcher_config = {"lightglue_config": LIGHTGLUE_CONFIG, "seed": args.seed,
                      "matcher_provenance": matcher.provenance}
    configuration = {
        "schema_version": 1, "seed": args.seed, "shard_size": shard_size,
        "population_sha256": sha256_file(runtime / "full_population.csv"),
        "population_summary_sha256": sha256_file(runtime / "full_population_summary.json"),
        "feature_bank_directory": relative_path(args.bank_dir, repo_root=root),
        "feature_bank_summary_sha256": sha256_file(Path(args.bank_dir) / "summary.json"),
        "bank_validation_sha256": sha256_file(Path(args.bank_dir) / "validation.json"),
        "lightglue_config": LIGHTGLUE_CONFIG, "matcher_configuration": matcher_config,
        "matcher_configuration_sha256": hashlib.sha256(canonical_json_bytes(matcher_config)).hexdigest(),
        "code_sha256": code_hashes(MEASUREMENT_CODE, repo_root=root),
        "frozen_provenance": frozen_provenance(repo_root=root),
        "real_rgb_only": True, "synthetic_images_used": False,
    }
    if args.seed != bank_summary["configuration"]["seed"] or args.seed != 42:
        raise ValueError("measurement seed differs from frozen bank/pilot seed")
    bounds = shard_boundaries(len(population), shard_size)
    if start_shard >= len(bounds):
        raise ValueError("start_shard is outside the deterministic shard range")
    with (runtime / ".matching.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another full-matching process owns this runtime directory") from error
        config_path = runtime / "full_measurement_config.json"
        if config_path.exists():
            if read_json(config_path) != configuration:
                raise ValueError("resume refused: full matching configuration/provenance changed")
        else:
            if any((runtime / "shards").glob("shard_*")):
                raise ValueError("resume refused: pair shards exist without configuration provenance")
            write_json(config_path, configuration, repo_root=root)
        selected = bounds[start_shard:] if maximum is None else bounds[start_shard:start_shard + maximum]
        processed = skipped = 0
        for shard_index, start, end in selected:
            existing = load_valid_shard(runtime, population, index, configuration, shard_index, start, end)
            if existing is not None:
                skipped += 1
                print(f"Shard {shard_index}: validated complete, skipped ({end-start} pairs)", flush=True)
                continue
            csv_path, summary_path = shard_paths(runtime, shard_index)
            guard_step2d([csv_path, summary_path], repo_root=root)
            binding = shard_binding(population, configuration, shard_index, start, end)
            subset = population.iloc[start:end]
            execution = subset.sort_values(["row_index_a", "row_index_b", "pair_uid"], kind="stable")
            started = _timestamp()
            gpu = hasattr(matcher, "torch") and getattr(getattr(matcher, "device", None), "type", None) == "cuda"
            if gpu:
                matcher.torch.cuda.synchronize(matcher.device)
                matcher.torch.cuda.reset_peak_memory_stats(matcher.device)
            clock_start = time.perf_counter()
            rows = {}
            for offset, row in enumerate(execution.to_dict("records"), 1):
                result = matcher.match(row["image_id_a"], row["image_id_b"])
                rows[row["pair_uid"]] = {**row, **result.metrics}
                del result
                if offset % 100 == 0:
                    print(f"Shard {shard_index}: {offset}/{end-start}", flush=True)
            if gpu:
                matcher.torch.cuda.synchronize(matcher.device)
            elapsed = time.perf_counter() - clock_start
            frame = pd.DataFrame([rows[uid] for uid in subset["pair_uid"]], columns=MEASUREMENT_COLUMNS)
            frame = validate_full_metric_frame(frame, subset, index)
            # Validate upstream hashes again before publishing a completed shard.
            _load_configuration(runtime, population_summary, args.bank_dir, bank_summary, repo_root=root)
            verify_source_images(index, manifest, args.dataset_root,
                                 image_ids=set(subset["image_id_a"]) | set(subset["image_id_b"]))
            write_csv(csv_path, frame, repo_root=root)
            elapsed = time.perf_counter() - clock_start
            throughput = (end - start) / elapsed if elapsed > 0 else None
            benchmark = {
                "pairs_processed": end - start, "elapsed_seconds": elapsed,
                "pairs_per_second": throughput,
                "peak_cuda_allocated_bytes": int(matcher.torch.cuda.max_memory_allocated(matcher.device)) if gpu else None,
                "peak_cuda_reserved_bytes": int(matcher.torch.cuda.max_memory_reserved(matcher.device)) if gpu else None,
                "cpu_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                "estimated_full_matching_seconds": len(population) / throughput if throughput else None,
                "estimate_scope": "per-shard matching, bank reads, provenance checks and CSV publication; excludes one-time bank extraction and startup",
            }
            summary = {**binding, "start_timestamp": started, "end_timestamp": _timestamp(),
                       "complete": True, "metrics_sha256": sha256_file(csv_path), "benchmark": benchmark}
            write_json(summary_path, summary, repo_root=root)
            processed += 1
            print(f"Shard {shard_index}: {end-start} pairs in {elapsed:.2f}s; {throughput:.3f} pairs/s; "
                  f"estimated full matching {benchmark['estimated_full_matching_seconds']/3600:.2f}h; "
                  f"CUDA allocated/reserved {benchmark['peak_cuda_allocated_bytes']}/{benchmark['peak_cuda_reserved_bytes']} bytes", flush=True)
        return {"new_complete_shards": processed, "validated_skipped_shards": skipped, "total_shards": len(bounds)}


def _atomic_merge(csv_paths, destination, *, repo_root=ROOT):
    """Stream complete ordered shards into one atomically published CSV."""
    guard_step2d([destination], repo_root=repo_root)
    destination = Path(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            expected_header = None
            for path in csv_paths:
                with Path(path).open("rb") as source:
                    header = source.readline()
                    if expected_header is None:
                        expected_header = header
                        target.write(header)
                    elif header != expected_header:
                        raise ValueError("shard headers differ during finalization")
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validated_shards(runtime, population, bank_index, configuration):
    expected_paths = set()
    artifacts = []
    bounds = shard_boundaries(len(population), configuration["shard_size"])
    for shard_index, start, end in bounds:
        result = load_valid_shard(runtime, population, bank_index, configuration, shard_index, start, end)
        if result is None:
            raise ValueError(f"full finalization refused: shard {shard_index} is incomplete or missing")
        _, summary = result
        csv_path, summary_path = shard_paths(runtime, shard_index)
        expected_paths.update((csv_path.name, summary_path.name))
        artifacts.append({"shard_index": shard_index, "pair_count": end-start,
                          "metrics_file": f"shards/{csv_path.name}", "metrics_sha256": sha256_file(csv_path),
                          "summary_file": f"shards/{summary_path.name}", "summary_sha256": sha256_file(summary_path),
                          "benchmark": summary["benchmark"]})
    if {path.name for path in (Path(runtime) / "shards").glob("shard_*")} != expected_paths:
        raise ValueError("full finalization refused: unknown shard artifacts are present")
    return artifacts


def finalize_full_metrics(runtime_dir=DEFAULT_DIR, *, repo_root=ROOT):
    from countermine.mining.full_structural_population import load_full_population
    runtime = Path(runtime_dir)
    guard_step2d([runtime / "full_candidate_structural_metrics.csv", runtime / "full_measurement_summary.json"], repo_root=repo_root)
    population, population_summary = load_full_population(runtime, repo_root=repo_root)
    preliminary = read_json(runtime / "full_measurement_config.json")
    bank_dir = Path(repo_root) / preliminary["feature_bank_directory"]
    index, bank_summary = load_feature_bank(bank_dir, repo_root=repo_root)
    config = _load_configuration(runtime, population_summary, bank_dir, bank_summary, repo_root=repo_root)
    artifacts = _validated_shards(runtime, population, index, config)
    csv_path, summary_path = runtime / "full_candidate_structural_metrics.csv", runtime / "full_measurement_summary.json"
    if summary_path.exists():
        frame, summary = load_full_metrics(runtime, repo_root=repo_root)
        return summary
    _atomic_merge([runtime / artifact["metrics_file"] for artifact in artifacts], csv_path, repo_root=repo_root)
    frame = validate_full_metric_frame(read_metadata_csv(csv_path), population, index)
    summary = {
        "schema_version": 1, "complete": True, "candidate_count": len(frame),
        "population_sha256": config["population_sha256"],
        "feature_bank_summary_sha256": config["feature_bank_summary_sha256"],
        "configuration_sha256": sha256_file(runtime / "full_measurement_config.json"),
        "full_metrics_sha256": sha256_file(csv_path), "shard_count": len(artifacts),
        "completed_shard_count": len(artifacts), "feature_bank_image_count": len(index),
        "exact_pixel_duplicate_count": int(frame["exact_pixel_duplicate"].sum()),
        "runtime_throughput_summary": aggregate_throughput(artifacts, len(frame)),
        "configuration": config, "shards": artifacts,
        "finalization_code_sha256": code_hashes(["tools/21_finalize_full_structural_metrics.py", "countermine/mining/full_structural_measurement.py"], repo_root=repo_root),
        "real_rgb_only": True, "synthetic_images_used": False,
    }
    # Files may not change between shard validation and final completion marker.
    _load_configuration(runtime, population_summary, bank_dir, bank_summary, repo_root=repo_root)
    if artifacts != _validated_shards(runtime, population, index, config):
        raise ValueError("shards changed during atomic metric finalization")
    write_json(summary_path, summary, repo_root=repo_root)
    return summary


def load_full_metrics(runtime_dir=DEFAULT_DIR, *, repo_root=ROOT):
    from countermine.mining.full_structural_population import load_full_population
    runtime = Path(runtime_dir)
    summary = read_json(runtime / "full_measurement_summary.json")
    if not isinstance(summary, dict) or summary.get("schema_version") != 1 or summary.get("complete") is not True:
        raise ValueError("full structural measurements have no valid atomic completion marker")
    population, population_summary = load_full_population(runtime, repo_root=repo_root)
    bank_dir = Path(repo_root) / summary["configuration"]["feature_bank_directory"]
    index, bank_summary = load_feature_bank(bank_dir, repo_root=repo_root)
    config = _load_configuration(runtime, population_summary, bank_dir, bank_summary, repo_root=repo_root)
    if summary["configuration"] != config or summary["configuration_sha256"] != sha256_file(runtime / "full_measurement_config.json"):
        raise ValueError("full measurement completion configuration differs")
    if summary["full_metrics_sha256"] != sha256_file(runtime / "full_candidate_structural_metrics.csv"):
        raise ValueError("full measurement final CSV hash differs")
    artifacts = _validated_shards(runtime, population, index, config)
    if artifacts != summary["shards"]:
        raise ValueError("full measurement shard provenance differs")
    if summary.get("runtime_throughput_summary") != aggregate_throughput(artifacts, len(population)):
        raise ValueError("full measurement aggregate throughput/memory metadata differs from validated shards")
    if summary["finalization_code_sha256"] != code_hashes(["tools/21_finalize_full_structural_metrics.py", "countermine/mining/full_structural_measurement.py"], repo_root=repo_root):
        raise ValueError("full measurement finalization code provenance differs")
    frame = validate_full_metric_frame(read_metadata_csv(runtime / "full_candidate_structural_metrics.csv"), population, index)
    expected = {"candidate_count": len(frame), "shard_count": len(artifacts), "completed_shard_count": len(artifacts),
                "feature_bank_image_count": len(index), "exact_pixel_duplicate_count": int(frame["exact_pixel_duplicate"].sum()),
                "population_sha256": config["population_sha256"], "feature_bank_summary_sha256": config["feature_bank_summary_sha256"]}
    expected.update(real_rgb_only=True, synthetic_images_used=False)
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("full measurement summary totals/provenance differ")
    return frame, summary

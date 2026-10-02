#!/usr/bin/env python3
"""Measure the frozen original-RGB Step 2A pilot with atomic resumability."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from countermine.mining.candidate_miner import haversine_distance_m, stable_pair_uid
from countermine.mining.structural_matcher import (
    ALIKED_CONFIG, LIGHTGLUE_CONFIG, METRIC_COLUMNS, StructuralMatcher,
    canonical_json_bytes, decode_original_rgb, sha256_file, source_provenance,
)


INPUT_FILENAMES = {
    "population_sha256": "population.csv",
    "random_controls_sha256": "random_controls.csv",
    "population_summary_sha256": "population_summary.json",
}


def atomic_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(Path(path), canonical_json_bytes(value))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"invalid CSV header: {path}")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f"malformed CSV rows: {path}")
        return reader.fieldnames, rows


def bool_value(value):
    if isinstance(value, bool):
        return value
    if value in ("True", "true"):
        return True
    if value in ("False", "false"):
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def image_path(root: Path, metadata: dict) -> Path:
    relative = Path(metadata["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("manifest image path must be a portable relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"manifest image is unavailable or outside dataset root: {relative}")
    return path


def _distance(a, b):
    return float(haversine_distance_m(float(a["lat"]), float(a["lon"]),
                                     float(b["lat"]), float(b["lon"])))


def prepare_tasks(population, controls, manifest, raw):
    """Validate endpoint identities and retain unavailable controls explicitly."""
    images = {row["image_id"]: row for row in manifest}
    if len(images) != len(manifest):
        raise ValueError("manifest image_id must be unique")
    controls_by_uid = {row["pair_uid"]: row for row in controls}
    if len(controls_by_uid) != len(controls) or set(controls_by_uid) != {row["pair_uid"] for row in population}:
        raise ValueError("controls must contain exactly one row per unique candidate")
    top50 = {}
    for row in raw:
        top50.setdefault(row["query_image_id"], set()).add(row["negative_image_id"])
    tasks = []
    seen = set()
    for candidate in population:
        uid = candidate["pair_uid"]
        if uid in seen or uid != stable_pair_uid(candidate["image_id_a"], candidate["image_id_b"]):
            raise ValueError("candidate pair_uid is duplicated or inconsistent")
        seen.add(uid)
        a, b = images[candidate["image_id_a"]], images[candidate["image_id_b"]]
        for role, endpoint in (("a", a), ("b", b)):
            for key in ("row_index", "place_uid", "city_id"):
                if str(candidate[f"{key}_{role}"]) != str(endpoint[key]):
                    raise ValueError(f"candidate {role} {key} disagrees with manifest")
        if a["place_uid"] == b["place_uid"] or _distance(a, b) < 250.0:
            raise ValueError("candidate violates frozen different-place / 250m exclusion")
        if abs(float(candidate["geo_distance_m"]) - _distance(a, b)) > 1e-6:
            raise ValueError("candidate geographic distance disagrees with manifest")
        if bool_value(candidate["same_city"]) != (a["city_id"] == b["city_id"]):
            raise ValueError("candidate city relation disagrees with manifest")
        query = images[candidate["anchor_query_image_id"]]
        negative = images[candidate["anchor_negative_image_id"]]
        if {query["image_id"], negative["image_id"]} != {a["image_id"], b["image_id"]}:
            raise ValueError("candidate anchor direction does not represent this pair")
        tasks.append(("candidate", uid, dict(candidate), a["image_id"], b["image_id"]))
        control = controls_by_uid[uid]
        if control["anchor_query_image_id"] != query["image_id"]:
            raise ValueError("random control changed the candidate query anchor")
        if control["candidate_negative_image_id"] != negative["image_id"]:
            raise ValueError("random control candidate negative identity is inconsistent")
        for role, endpoint in (("anchor_query", query), ("candidate_negative", negative)):
            for key in ("row_index", "place_uid", "city_id"):
                if str(control[f"{role}_{key}"]) != str(endpoint[key]):
                    raise ValueError(f"control {role} {key} disagrees with manifest")
        if control.get("candidate_pair_uid") != uid or control["rank_bin"] != candidate["rank_bin"]:
            raise ValueError("control candidate identity/rank bin disagrees with population")
        if abs(float(control["candidate_geo_distance_m"]) - float(candidate["geo_distance_m"])) > 1e-6:
            raise ValueError("control candidate geographic distance disagrees with population")
        for key in ("candidate_same_city", "same_city"):
            if bool_value(control[key]) != bool_value(candidate["same_city"]):
                raise ValueError("control candidate city relation disagrees with population")
        if not bool_value(control["random_control_available"]):
            for key in ("random_pair_uid", "random_negative_image_id", "random_negative_row_index",
                        "random_negative_place_uid", "random_negative_city_id", "random_geo_distance_m",
                        "random_same_city"):
                if control.get(key):
                    raise ValueError("unavailable random control must not contain random-target metadata")
            continue
        random = images[control["random_negative_image_id"]]
        for key in ("row_index", "place_uid", "city_id"):
            if str(control[f"random_negative_{key}"]) != str(random[key]):
                raise ValueError(f"control random-negative {key} disagrees with manifest")
        if control["random_pair_uid"] != stable_pair_uid(query["image_id"], random["image_id"]):
            raise ValueError("random pair_uid disagrees with its endpoints")
        if random["image_id"] in {query["image_id"], negative["image_id"]} or random["image_id"] in top50.get(query["image_id"], set()):
            raise ValueError("random target is an anchor, candidate, or saved Top-50 negative")
        distance = _distance(query, random)
        if random["place_uid"] == query["place_uid"] or distance < 250.0:
            raise ValueError("random target violates frozen place/geographic exclusion")
        if abs(float(control["random_geo_distance_m"]) - distance) > 1e-6:
            raise ValueError("random geographic distance disagrees with manifest")
        same_city = query["city_id"] == negative["city_id"]
        if random["city_id"] != negative["city_id"]:
            raise ValueError("random control changed the candidate target city")
        if bool_value(control["random_same_city"]) != same_city:
            raise ValueError("control random city relation disagrees with manifest")
        if same_city:
            bins = (250.0, 500.0, 1000.0, 2000.0, 5000.0, float("inf"))
            distance_bin = lambda value: next(i for i in range(5) if bins[i] <= value < bins[i + 1])
            if distance_bin(distance) != distance_bin(float(candidate["geo_distance_m"])):
                raise ValueError("random control changed the same-city geographic bin")
        metadata = dict(control)
        metadata.update({
            "candidate_pair_uid": uid, "image_id_a": query["image_id"],
            "image_id_b": random["image_id"], "place_uid_a": query["place_uid"],
            "place_uid_b": random["place_uid"], "city_id_a": query["city_id"],
            "city_id_b": random["city_id"], "geo_distance_m": control["random_geo_distance_m"],
            "same_city": candidate["same_city"], "rank_bin": candidate["rank_bin"],
        })
        tasks.append(("random", uid, metadata, query["image_id"], random["image_id"]))
    if not population:
        raise ValueError("population contains no candidate pairs")
    # Group small local neighborhoods by representative anchor for bounded reuse.
    return sorted(tasks, key=lambda task: (task[2]["anchor_query_image_id"], task[1], task[0])), images


def validate_metrics(row, fingerprints, a, b):
    absent = set(METRIC_COLUMNS) - set(row)
    if absent:
        raise ValueError(f"measurement lacks metrics: {sorted(absent)}")
    counts = []
    for key in ("num_keypoints_a", "num_keypoints_b", "num_matches"):
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("measurement counts must be nonnegative integers")
        counts.append(value)
    if max(counts[:2]) > ALIKED_CONFIG["max_num_keypoints"] or counts[2] > min(counts[:2]):
        raise ValueError("measurement counts violate the frozen keypoint budget")
    ratio = counts[2] / min(counts[:2]) if min(counts[:2]) else 0.0
    if not math.isclose(float(row["local_match_ratio"]), ratio, rel_tol=0, abs_tol=1e-12):
        raise ValueError("local_match_ratio disagrees with the match/keypoint counts")
    for key in METRIC_COLUMNS[3:12]:
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1e-12 <= value <= 1 + 1e-12:
            raise ValueError(f"invalid bounded descriptive metric: {key}")
    for symmetric, left, right in (
        ("symmetric_match_coverage", "matched_source_cell_coverage", "matched_target_cell_coverage"),
        ("symmetric_match_entropy", "source_match_entropy", "target_match_entropy"),
    ):
        if not math.isclose(row[symmetric], min(row[left], row[right]), abs_tol=1e-12):
            raise ValueError(f"inconsistent symmetric diagnostic: {symmetric}")
    for endpoint, image_id in (("a", a), ("b", b)):
        fingerprint = fingerprints[image_id]
        if row[f"rgb_pixel_sha_{endpoint}"] != fingerprint["rgb_pixel_sha256"] or row[f"image_file_sha_{endpoint}"] != fingerprint["image_file_sha256"]:
            raise ValueError("measurement input image hash mismatch")
    expected_duplicate = fingerprints[a]["rgb_pixel_sha256"] == fingerprints[b]["rgb_pixel_sha256"]
    if not isinstance(row["exact_pixel_duplicate"], bool) or row["exact_pixel_duplicate"] != expected_duplicate:
        raise ValueError("exact decoded-pixel duplicate diagnostic is inconsistent")


class MeasurementJournal:
    """Durable per-record checkpoints; CSV files are atomic published views."""

    def __init__(self, runtime_dir, configuration, tasks, fingerprints):
        self.root = Path(runtime_dir)
        self.configuration = configuration
        self.config_hash = hashlib.sha256(canonical_json_bytes(configuration)).hexdigest()
        self.tasks = {(kind, uid): (metadata, a, b) for kind, uid, metadata, a, b in tasks}
        if len(self.tasks) != len(tasks):
            raise ValueError("duplicate measurement tasks")
        self.fingerprints = fingerprints
        self.directory = self.root / ".measurements"
        config_path = self.root / "measurement_config.json"
        if config_path.exists():
            old = json.loads(config_path.read_text())
            if old != configuration or sha256_file(config_path) != self.config_hash:
                raise ValueError("resume refused: measurement configuration/provenance changed")
        else:
            if self.directory.exists() and any(self.directory.glob("*.json")):
                raise ValueError("resume refused: measurement journal has no provenance")
            if any((self.root / name).exists() for name in (
                "candidate_structural_metrics.csv", "random_structural_metrics.csv", "measurement_summary.json",
            )):
                raise ValueError("resume refused: measurements exist without provenance")
            atomic_json(config_path, configuration)
        self.directory.mkdir(exist_ok=True)
        self.completed = {}
        for path in sorted(self.directory.glob("*.json")):
            record = json.loads(path.read_text())
            key = record["pair_type"], record["pair_uid"]
            if key not in self.tasks or key in self.completed or record["config_sha256"] != self.config_hash:
                raise ValueError("resume refused: unknown, duplicate, or incompatible checkpoint")
            if path.name != self._filename(*key) or record.get("schema_version") != 1:
                raise ValueError("resume refused: malformed checkpoint identity")
            self._validate(key, record["row"])
            self.completed[key] = record["row"]
        marker = self.root / "measurement_summary.json"
        if marker.exists():
            summary = json.loads(marker.read_text())
            if summary.get("config_sha256") != self.config_hash:
                raise ValueError("resume refused: published summary configuration changed")
            # An interrupted multi-file publication can leave older partial
            # views. They are rebuilt from durable, individually validated
            # checkpoints. Completed scientific outputs must be untouched.
            if summary.get("complete") is True:
                if len(self.completed) != len(self.tasks):
                    raise ValueError("resume refused: completed publication lacks checkpoints")
                for kind in ("candidate", "random"):
                    path = self.root / f"{kind}_structural_metrics.csv"
                    if not path.exists() or summary.get(f"{kind}_metrics_sha256") != sha256_file(path):
                        raise ValueError("resume refused: published metrics CSV changed")

    @staticmethod
    def _filename(kind, uid):
        return f"{kind}-{hashlib.sha256(uid.encode()).hexdigest()}.json"

    def _validate(self, key, row):
        metadata, a, b = self.tasks[key]
        if set(row) != set(metadata) | set(METRIC_COLUMNS):
            raise ValueError("measurement checkpoint schema changed")
        if any(row.get(name) != value for name, value in metadata.items()):
            raise ValueError("measurement checkpoint pair metadata changed")
        validate_metrics(row, self.fingerprints, a, b)

    def record(self, kind, uid, metrics):
        key = kind, uid
        if key in self.completed:
            raise ValueError("attempted to duplicate a completed measurement")
        row = {**self.tasks[key][0], **metrics}
        self._validate(key, row)
        atomic_json(self.directory / self._filename(kind, uid), {
            "schema_version": 1, "pair_type": kind, "pair_uid": uid,
            "config_sha256": self.config_hash, "row": row,
        })
        self.completed[key] = row

    def publish(self, force_incomplete=False):
        import io
        summary = {
            "schema_version": 1, "complete": not force_incomplete and len(self.completed) == len(self.tasks),
            "config_sha256": self.config_hash,
            "input_hashes": self.configuration["input_hashes"],
        }
        for kind in ("candidate", "random"):
            expected = [key for key in self.tasks if key[0] == kind]
            rows = [self.completed[key] for key in sorted(expected) if key in self.completed]
            metadata_keys = list(self.tasks[expected[0]][0]) if expected else ["pair_uid", "candidate_pair_uid", "anchor_query_image_id", "random_negative_image_id", "same_city", "rank_bin"]
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=metadata_keys + list(METRIC_COLUMNS), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            path = self.root / f"{kind}_structural_metrics.csv"
            atomic_bytes(path, buffer.getvalue().encode("utf-8"))
            summary[f"expected_{kind}_count"] = len(expected)
            summary[f"{kind}_count"] = len(rows)
            summary[f"{kind}_metrics_sha256"] = sha256_file(path)
        atomic_json(self.root / "measurement_summary.json", summary)
        return summary


def run_measurement(args, matcher_factory=StructuralMatcher, input_loader=None):
    if input_loader is None:
        from countermine.mining.structural_population import load_population_inputs
        input_loader = load_population_inputs
    runtime = args.runtime_dir.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    # flock is released automatically after a crash, unlike a persistent PID file.
    with (runtime / ".measurement.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another measurement process owns this runtime directory") from error
        manifest, raw, _ = input_loader(args.manifest, args.candidates, args.candidate_summary)
        manifest = manifest.to_dict("records") if hasattr(manifest, "to_dict") else manifest
        raw = raw.to_dict("records") if hasattr(raw, "to_dict") else raw
        _, population = read_csv(runtime / "population.csv")
        _, controls = read_csv(runtime / "random_controls.csv")
        population_summary = json.loads((runtime / "population_summary.json").read_text())
        if population_summary.get("seed") != args.seed:
            raise ValueError("measurement seed differs from the frozen pilot seed")
        input_hashes = {
            "manifest_sha256": sha256_file(args.manifest),
            "candidate_csv_sha256": sha256_file(args.candidates),
            "candidate_summary_sha256": sha256_file(args.candidate_summary),
            **{key: sha256_file(runtime / name) for key, name in INPUT_FILENAMES.items()},
        }
        for key in ("manifest_sha256", "candidate_csv_sha256", "candidate_summary_sha256", "population_sha256", "random_controls_sha256"):
            if population_summary.get(key) != input_hashes[key]:
                raise ValueError(f"frozen population provenance mismatch: {key}")
        tasks, images = prepare_tasks(population, controls, manifest, raw)
        paths = {image_id: image_path(args.dataset_root.resolve(), images[image_id])
                 for image_id in sorted({item for task in tasks for item in task[3:]})}
        fingerprints = {}
        for index, (image_id, path) in enumerate(paths.items(), 1):
            _, fingerprints[image_id] = decode_original_rgb(path)
            if index % 500 == 0:
                print(f"Validated original RGB inputs: {index}/{len(paths)}", flush=True)
        fingerprint_content = canonical_json_bytes(fingerprints)
        fingerprint_hash = hashlib.sha256(fingerprint_content).hexdigest()
        config_static = {
            "schema_version": 1, "seed": args.seed,
            "image_policy": {"real_rgb_only": True, "synthetic_images_used": False,
                             "geometry": [640, 480], "local_resize": None, "registration": None},
            "aliked_config": ALIKED_CONFIG, "lightglue_config": LIGHTGLUE_CONFIG,
            "input_hashes": input_hashes, "input_image_index_sha256": fingerprint_hash,
            "source_sha256": {path: sha256_file(ROOT / path) for path in (
                "countermine/mining/structural_matcher.py", "tools/08_measure_structural_pairs.py",
                "countermine/mining/structural_population.py",
            )},
            "vendor_provenance": source_provenance(ROOT),
        }
        config_path = runtime / "measurement_config.json"
        if config_path.exists():
            saved = json.loads(config_path.read_text())
            if any(saved.get(key) != value for key, value in config_static.items()):
                raise ValueError("resume refused: inputs, original image bytes, or implementation changed")
            fingerprint_path = runtime / "image_fingerprints.json"
            if not fingerprint_path.exists() or sha256_file(fingerprint_path) != fingerprint_hash:
                raise ValueError("resume refused: original-image fingerprint index changed")
        else:
            atomic_bytes(runtime / "image_fingerprints.json", fingerprint_content)
        matcher = matcher_factory(device=args.device, seed=args.seed,
                                  feature_cache_size=args.feature_cache_size, repo_root=ROOT)
        matcher.set_image_fingerprints(fingerprints)
        config = {**config_static, "matcher_provenance": matcher.provenance}
        journal = MeasurementJournal(runtime, config, tasks, fingerprints)
        new = 0
        try:
            for kind, uid, _, a, b in tasks:
                if (kind, uid) in journal.completed:
                    continue
                result = matcher.match(a, paths[a], b, paths[b])
                journal.record(kind, uid, result.metrics)
                new += 1
                if new % args.checkpoint_every == 0:
                    journal.publish(force_incomplete=True)
                    print(f"Completed structural pairs: {len(journal.completed)}/{len(tasks)}", flush=True)
            # Recheck encoded file bytes before marking this run complete.
            for image_id, path in paths.items():
                if sha256_file(path) != fingerprints[image_id]["image_file_sha256"]:
                    raise ValueError(f"original image bytes changed during inference: {image_id}")
        except BaseException:
            journal.publish(force_incomplete=True)
            raise
        summary = journal.publish()
        print(f"Complete: {summary['candidate_count']} candidate and {summary['random_count']} random pairs; new measurements {new}", flush=True)
        return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/gsv_mini/manifest.csv"))
    parser.add_argument("--candidates", type=Path, default=Path("cache/gsv_mini/rgb_candidates_raw.csv"))
    parser.add_argument("--candidate-summary", type=Path, default=Path("cache/gsv_mini/rgb_candidates_raw_summary.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/gsv-cities"))
    parser.add_argument("--runtime-dir", type=Path, default=Path("cache/countermine_rgb/step2a"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-cache-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.checkpoint_every <= 0 or not 0 <= args.feature_cache_size <= 32:
        parser.error("checkpoint interval must be positive; feature cache must be in [0,32]")
    try:
        run_measurement(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()

"""CPU-only, binary-edge place co-batching with exact marginal exposure.

Edges are immutable relations; neither scores nor support counts become
sampling weights.  The sampler can be passed to DataLoader as batch_sampler
without inheriting from a torch class.
"""

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .gsv_place_mapping import PlaceRecord, validate_place_uid
from .step3a_config import ROOT, guard_step3a, write_json


TARGET_CITY_IDS = {"Boston": 1, "London": 2}
MODES = ("baseline", "countermine_q99_geo500")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_values(values):
    payload = json.dumps(list(values), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_csv(destination, columns, rows):
    handle, temporary = tempfile.mkstemp(prefix=".step3a_csv_", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


@dataclass(frozen=True)
class EdgeBundle:
    graph_sha256: str
    snapshot_sha256: str
    graph_place_uids: frozenset[str]
    q95: frozenset[tuple[str, str]]
    q99: frozenset[tuple[str, str]]
    q99_geo500: frozenset[tuple[str, str]]

    def __post_init__(self):
        # Frozen dataclasses alone do not freeze mutable constructor arguments.
        for field in ("graph_place_uids", "q95", "q99", "q99_geo500"):
            object.__setattr__(self, field, frozenset(getattr(self, field)))
        for uid in self.graph_place_uids:
            validate_place_uid(uid)
        for name in ("q95", "q99", "q99_geo500"):
            canonical = frozenset(tuple(sorted(edge)) for edge in getattr(self, name))
            for edge in canonical:
                if len(edge) != 2 or edge[0] == edge[1]:
                    raise ValueError("Graph relation must have two distinct place endpoints")
                if not set(edge) <= self.graph_place_uids:
                    raise ValueError("Structural edge endpoint outside graph place universe")
            object.__setattr__(self, name, canonical)
        if not self.q99_geo500 <= self.q99 <= self.q95:
            raise ValueError("Frozen core edge slices are not nested")

    def eligible_edges(self, city):
        if city not in TARGET_CITY_IDS:
            return ()
        return tuple(sorted(edge for edge in self.q99_geo500
                            if all(validate_place_uid(uid)[0] == city for uid in edge)))


def load_edge_bundle(place_edges_path, snapshot_path, *, expected_graph_places=2000):
    """Validate the frozen graph hash, then read only binary edge slices."""
    with Path(snapshot_path).open(encoding="utf-8") as stream:
        snapshot = json.load(stream)
    if snapshot.get("complete") is not True:
        raise ValueError("Step 2D scientific snapshot is not complete")
    expected_sha = snapshot.get("graph_artifact_hashes", {}).get("place_edges.csv")
    actual_sha = _sha256_file(place_edges_path)
    if not expected_sha or actual_sha != expected_sha:
        raise ValueError("Frozen Step 2D place_edges.csv SHA256 mismatch")
    graph_places = set()
    sets = {name: set() for name in ("q95", "q99", "q99_geo500")}
    seen_edges = set()
    with Path(place_edges_path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"place_uid_a", "place_uid_b", "city_id_a", "city_id_b"}
        required.update(f"num_core_{name}_image_edges" for name in sets)
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("Frozen place graph is missing required identity/support columns")
        for row in reader:
            a, b = row["place_uid_a"], row["place_uid_b"]
            if validate_place_uid(a)[0] != row["city_id_a"] or validate_place_uid(b)[0] != row["city_id_b"]:
                raise ValueError("Graph endpoint city disagrees with canonical UID")
            edge = tuple(sorted((a, b)))
            if a == b or edge in seen_edges:
                raise ValueError("Graph has same-place or duplicate canonical edge")
            seen_edges.add(edge)
            graph_places.update(edge)
            for name, edge_set in sets.items():
                value = row[f"num_core_{name}_image_edges"]
                try:
                    support = int(value)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Invalid frozen core support count: {value!r}") from error
                if support < 0:
                    raise ValueError("Frozen core support count cannot be negative")
                if support:
                    edge_set.add(edge)
    if expected_graph_places is not None and len(graph_places) != expected_graph_places:
        raise ValueError(f"Expected {expected_graph_places} graph places, found {len(graph_places)}")
    return EdgeBundle(actual_sha, _sha256_file(snapshot_path), frozenset(graph_places),
                      frozenset(sets["q95"]), frozenset(sets["q99"]), frozenset(sets["q99_geo500"]))


@dataclass(frozen=True)
class GuidedPair:
    dataset_index_a: int
    dataset_index_b: int
    canonical_place_uid_a: str
    canonical_place_uid_b: str
    city_id: str
    batch_index: int


@dataclass(frozen=True)
class BatchPlan:
    batches: tuple[tuple[int, ...], ...]
    guided_pairs: tuple[GuidedPair, ...]
    summary: dict


def _exposure(batches, records, bundle):
    if bundle is None:
        return {"core_q95": 0, "core_q99": 0, "core_q99_geo500": 0,
                "unique_q99_geo500_graph_edges_exposed": 0, "unique_structural_places_exposed": 0}
    batch_by_uid = {records[index].canonical_place_uid: batch_index
                    for batch_index, batch in enumerate(batches) for index in batch}
    result = {}
    geo_edges = set()
    for name in ("q95", "q99", "q99_geo500"):
        exposed = {edge for edge in getattr(bundle, name)
                   if batch_by_uid.get(edge[0]) is not None
                   and batch_by_uid.get(edge[0]) == batch_by_uid.get(edge[1])}
        result[f"core_{name}"] = len(exposed)
        if name == "q99_geo500":
            geo_edges = exposed
    result["unique_q99_geo500_graph_edges_exposed"] = len(geo_edges)
    result["unique_structural_places_exposed"] = len({uid for edge in geo_edges for uid in edge})
    return result


def _marginal(sequence, records):
    city_uids = {}
    for index in sequence:
        record = records[index]
        city_uids.setdefault(record.city_id, []).append(record.canonical_place_uid)
    return {
        "per_city_place_appearances": {city: len(uids) for city, uids in sorted(city_uids.items())},
        "per_city_sorted_place_uids_sha256": {city: _hash_values(sorted(uids))
                                             for city, uids in sorted(city_uids.items())},
        "place_multiset_sha256": _hash_values(sorted(records[index].canonical_place_uid for index in sequence)),
    }


class CounterMineBatchSampler:
    """Freeze one complete place plan per epoch, preserving every city slot.

    ``require_exposure_increase=False`` is intended only for tiny synthetic
    unit fixtures that may already expose every edge in the baseline.  The
    real training launcher leaves the stop condition enabled.
    """

    def __init__(self, records, edge_bundle=None, *, mode="baseline", batch_size=60,
                 seed=42, epoch=0, plan_dir=None, require_exposure_increase=True,
                 repo_root=ROOT):
        if mode not in MODES:
            raise ValueError(f"Unknown sampler mode: {mode}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.records = tuple(records)
        if any(not isinstance(record, PlaceRecord) or record.dataset_index != i
               for i, record in enumerate(self.records)):
            raise ValueError("Place mapping must be ordered by contiguous dataset_index")
        uids = [record.canonical_place_uid for record in self.records]
        if len(set(uids)) != len(uids):
            raise ValueError("Duplicate canonical place mapping")
        if len({record.internal_place_id for record in self.records}) != len(self.records):
            raise ValueError("Duplicate internal place mapping would corrupt training labels")
        for record in self.records:
            if validate_place_uid(record.canonical_place_uid)[0] != record.city_id:
                raise ValueError("Place record city disagrees with canonical UID")
        if edge_bundle is not None and not edge_bundle.graph_place_uids <= set(uids):
            raise ValueError("Every frozen CounterMine graph place must map to the training dataset")
        if mode != "baseline" and edge_bundle is None:
            raise ValueError("CounterMine treatment requires a frozen edge bundle")
        self.edge_bundle = edge_bundle
        self.mode = mode
        self.batch_size = batch_size
        self.drop_last = False
        self.seed = seed
        self.plan_dir = Path(plan_dir) if plan_dir is not None else None
        self.repo_root = Path(repo_root).resolve()
        self.require_exposure_increase = require_exposure_increase
        self.set_epoch(epoch)

    def __len__(self):
        return len(self.plan.batches)

    def __iter__(self):
        return (list(batch) for batch in self.plan.batches)

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch
        self.plan = self._build_plan()
        if self.plan_dir is not None:
            self.save_plan()

    def _build_plan(self):
        baseline = tuple(tuple(range(start, min(start + self.batch_size, len(self.records))))
                         for start in range(0, len(self.records), self.batch_size))
        batches = [list(batch) for batch in baseline]
        by_uid = {record.canonical_place_uid: record.dataset_index for record in self.records}
        guided = []
        eligible_counts = {city: len(self.edge_bundle.eligible_edges(city)) if self.edge_bundle else 0
                           for city in TARGET_CITY_IDS}
        capacities = {}
        for city, city_identifier in TARGET_CITY_IDS.items():
            slots_by_batch = [[position for position, index in enumerate(batch)
                               if self.records[index].city_id == city] for batch in baseline]
            pair_slots = [(batch_index, slots[offset], slots[offset + 1])
                          for batch_index, slots in enumerate(slots_by_batch)
                          for offset in range(0, len(slots) - 1, 2)]
            capacities[city] = len(pair_slots)
            if self.mode == "baseline":
                continue
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                [self.seed, self.epoch, city_identifier])))
            edges = self.edge_bundle.eligible_edges(city)
            used = set()
            pairs = []
            for offset in rng.permutation(len(edges)):
                a, b = edges[int(offset)]
                ia, ib = by_uid[a], by_uid[b]
                if ia not in used and ib not in used:
                    used.update((ia, ib))
                    pairs.append((ia, ib))
            if len(pairs) > len(pair_slots):
                raise ValueError(f"Greedy {city} matching exceeds unchanged city pair-slot capacity")
            # Uniformly disperse the disjoint pairs over available pair slots.
            shuffled_slots = [pair_slots[int(offset)] for offset in rng.permutation(len(pair_slots))]
            assigned_positions = set()
            for (ia, ib), (batch_index, pa, pb) in zip(pairs, shuffled_slots):
                batches[batch_index][pa], batches[batch_index][pb] = ia, ib
                assigned_positions.update(((batch_index, pa), (batch_index, pb)))
                guided.append(GuidedPair(ia, ib, self.records[ia].canonical_place_uid,
                                         self.records[ib].canonical_place_uid, city, batch_index))
            unused = iter(record.dataset_index for record in self.records
                          if record.city_id == city and record.dataset_index not in used)
            for batch_index, slots in enumerate(slots_by_batch):
                for position in slots:
                    if (batch_index, position) not in assigned_positions:
                        batches[batch_index][position] = next(unused)

        batches = tuple(tuple(batch) for batch in batches)
        sequence = tuple(index for batch in batches for index in batch)
        baseline_sequence = tuple(index for batch in baseline for index in batch)
        appearances = Counter(sequence)
        duplicates = sum(max(0, count - 1) for count in appearances.values())
        missing = len(set(baseline_sequence) - set(sequence))
        mismatches = sum(Counter(self.records[i].city_id for i in a) !=
                         Counter(self.records[i].city_id for i in b) for a, b in zip(baseline, batches))
        batch_by_index = {index: batch_index for batch_index, batch in enumerate(batches) for index in batch}
        split = sum(batch_by_index[pair.dataset_index_a] != batch_by_index[pair.dataset_index_b]
                    for pair in guided)
        guided_counts = Counter(index for pair in guided for index in (pair.dataset_index_a, pair.dataset_index_b))
        max_guided = max(guided_counts.values(), default=0)
        if (duplicates or missing or mismatches or split or max_guided > 1
                or tuple(map(len, baseline)) != tuple(map(len, batches))
                or Counter(sequence) != Counter(baseline_sequence)):
            raise ValueError("Step 3A sampler invariant failure")
        for before, after in zip(baseline_sequence, sequence):
            if self.records[before].city_id not in TARGET_CITY_IDS and before != after:
                raise ValueError("Non-target-city training slot changed")
        marginal = _marginal(sequence, self.records)
        baseline_marginal = _marginal(baseline_sequence, self.records)
        if marginal != baseline_marginal:
            raise ValueError("Treatment and baseline marginal place exposure differ")
        baseline_exposure = _exposure(baseline, self.records, self.edge_bundle)
        exposure = _exposure(batches, self.records, self.edge_bundle)
        baseline_geo = baseline_exposure["core_q99_geo500"]
        treatment_geo = exposure["core_q99_geo500"]
        if self.mode != "baseline" and self.require_exposure_increase:
            if not sum(eligible_counts.values()):
                raise ValueError("No eligible same-city q99_geo500 edges for the training treatment")
            if treatment_geo <= baseline_geo:
                raise ValueError("Sampler implementation error: treatment does not increase q99_geo500 co-exposure")
        summary = {
            "sampler_mode": self.mode,
            "dataset_place_count": len(self.records),
            "batch_count": len(batches),
            "batch_sizes": list(map(len, batches)),
            "baseline_sequence_sha256": _hash_values(baseline_sequence),
            "treatment_sequence_sha256": _hash_values(sequence),
            "dataset_place_order_sha256": _hash_values(record.canonical_place_uid for record in self.records),
            "graph_sha256": self.edge_bundle.graph_sha256 if self.edge_bundle else None,
            "step2d_snapshot_sha256": self.edge_bundle.snapshot_sha256 if self.edge_bundle else None,
            "seed": self.seed,
            "epoch": self.epoch,
            "rng": "NumPy PCG64 / SeedSequence([seed, epoch, city_identifier])",
            "stable_city_identifiers": dict(TARGET_CITY_IDS),
            "eligible_q99_geo500_same_city_edge_count": sum(eligible_counts.values()),
            "eligible_q99_geo500_same_city_edges_by_city": eligible_counts,
            "city_pair_slot_capacity": capacities,
            "greedy_matched_pair_count": len(guided),
            "unique_guided_place_count": len(guided_counts),
            "boston_guided_pair_count": sum(pair.city_id == "Boston" for pair in guided),
            "london_guided_pair_count": sum(pair.city_id == "London" for pair in guided),
            "intentionally_guided_pairs_cooccur_count": len(guided) - split,
            "accidentally_split_pair_count": split,
            "duplicate_place_count": duplicates,
            "missing_place_count": missing,
            "per_batch_city_composition_mismatch_count": mismatches,
            "max_guided_pairs_per_place": max_guided,
            "marginal_exposure": marginal,
            "baseline_marginal_exposure": baseline_marginal,
            "identical_place_multiset": True,
            "non_target_city_slots_unchanged": True,
            "drop_last": False,
            "exposure": exposure,
            "baseline_exposure": baseline_exposure,
            "structural_exposure_ratio": treatment_geo / baseline_geo if baseline_geo else None,
            "structural_exposure_gain": treatment_geo - baseline_geo,
            "baseline_q99_geo500_exposure_zero": baseline_geo == 0,
            "accidental_q99_geo500_exposure_count": treatment_geo - len(guided),
            "binary_edge_use": True,
            "edge_weighting": False,
            "hub_weighting": False,
        }
        return BatchPlan(batches, tuple(guided), summary)

    def save_plan(self, directory=None, *, repo_root=None):
        directory = Path(directory) if directory is not None else self.plan_dir
        if directory is None:
            raise ValueError("A batch plan output directory is required")
        repo_root = self.repo_root if repo_root is None else Path(repo_root).resolve()
        directory = guard_step3a(directory, repo_root)
        # Resolve both output files before creating anything: symlinks into
        # earlier scientific namespaces must never be followed for a write.
        stem = f"epoch_{self.epoch:02d}"
        csv_path = guard_step3a(directory / f"{stem}_guided_pairs.csv", repo_root)
        summary_path = guard_step3a(directory / f"{stem}_summary.json", repo_root)
        place_order_path = guard_step3a(directory / f"{stem}_place_order.csv", repo_root)
        directory.mkdir(parents=True, exist_ok=True)
        columns = ("dataset_index_a", "dataset_index_b", "canonical_place_uid_a",
                   "canonical_place_uid_b", "city_id", "batch_index")
        _atomic_csv(csv_path, columns, ({column: getattr(pair, column) for column in columns}
                                       for pair in self.plan.guided_pairs))
        place_columns = ("dataset_index", "internal_place_id", "city_id", "canonical_place_uid")
        _atomic_csv(place_order_path, place_columns,
                    ({column: getattr(record, column) for column in place_columns} for record in self.records))
        summary = dict(self.plan.summary)
        summary["guided_pairs_csv_sha256"] = _sha256_file(csv_path)
        summary["place_order_csv_sha256"] = _sha256_file(place_order_path)
        write_json(summary_path, summary, root=repo_root)
        return summary_path, csv_path

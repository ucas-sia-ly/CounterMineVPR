"""Read canonical place identities from an instantiated upstream dataset.

This module deliberately does not import SALAD or a model runtime.  In
particular, the upstream numeric city prefix is not reconstructed here: the
dataset's own image-name function supplies the original place identity.
"""

from collections import Counter
from dataclasses import dataclass
from numbers import Integral
from pathlib import PurePath
import re


@dataclass(frozen=True)
class PlaceRecord:
    dataset_index: int
    internal_place_id: int
    city_id: str
    canonical_place_uid: str


def validate_place_uid(value: str) -> tuple[str, str]:
    """Require the exact canonical city:seven-digit identity."""
    if not isinstance(value, str):
        raise ValueError("Canonical place UID must be a string")
    match = re.fullmatch(r"([^:\s]+):([0-9]{7})", value)
    if match is None:
        raise ValueError(f"Invalid canonical place UID: {value!r}")
    return match.group(1), match.group(2)


def build_place_mapping(dataset, graph_place_uids=()):
    """Return immutable records and a JSON-ready one-to-one mapping audit.

    Every metadata row is checked, including rows not sampled in training.
    ``dataset.get_img_name(row)`` is the authority for the original identifier.
    No pixels are loaded, and no assumption about numeric city prefixes is made.
    """
    if not hasattr(dataset, "places_ids") or not hasattr(dataset, "dataframe"):
        raise ValueError("Dataset must expose places_ids and dataframe metadata")
    if not callable(getattr(dataset, "get_img_name", None)):
        raise ValueError("Dataset must expose the upstream get_img_name function")
    places = tuple(dataset.places_ids)
    if len(places) != len(dataset):
        raise ValueError("Dataset length disagrees with places_ids")
    # A repeated .loc[label] on SALAD's shuffled, nonunique dataframe index
    # scans the metadata for every place. Group row positions once instead.
    grouped_positions = dataset.dataframe.groupby(level=0, sort=False).indices
    records = []
    seen_internal = set()
    seen_canonical = set()
    for dataset_index, internal_id in enumerate(places):
        if isinstance(internal_id, bool) or not isinstance(internal_id, Integral):
            raise ValueError(f"Non-integer internal place ID: {internal_id!r}")
        internal_id = int(internal_id)
        if internal_id in seen_internal:
            raise ValueError(f"Duplicate internal place ID: {internal_id}")
        seen_internal.add(internal_id)
        if internal_id not in grouped_positions:
            raise ValueError(f"Internal place {internal_id} has no metadata rows")
        rows = dataset.dataframe.iloc[grouped_positions[internal_id]]
        row_items = rows.iterrows()
        identities = set()
        row_count = 0
        for _, row in row_items:
            row_count += 1
            city = row["city_id"]
            if not isinstance(city, str) or not city or ":" in city or any(c.isspace() for c in city):
                raise ValueError(f"Invalid city metadata for internal place {internal_id}")
            image_name = PurePath(str(dataset.get_img_name(row))).name
            prefix = city + "_"
            if not image_name.startswith(prefix):
                raise ValueError(f"Image-name city disagrees with metadata for place {internal_id}")
            original_id = image_name[len(prefix):].split("_", 1)[0]
            canonical = f"{city}:{original_id}"
            validate_place_uid(canonical)
            identities.add((city, canonical))
        if row_count == 0 or len(identities) != 1:
            raise ValueError(f"Rows disagree on city or original place identity for {internal_id}")
        city, canonical = next(iter(identities))
        if canonical in seen_canonical:
            raise ValueError(f"Duplicate canonical place mapping: {canonical}")
        seen_canonical.add(canonical)
        records.append(PlaceRecord(dataset_index, internal_id, city, canonical))

    graph_places = frozenset(graph_place_uids)
    for uid in graph_places:
        validate_place_uid(uid)
    unmapped = sorted(graph_places - seen_canonical)
    if unmapped:
        raise ValueError(f"Unmapped CounterMine graph places ({len(unmapped)}): {', '.join(unmapped[:10])}")
    counts = dict(sorted(Counter(record.city_id for record in records).items()))
    audit = {
        "total_training_places": len(records),
        "boston_training_places": counts.get("Boston", 0),
        "london_training_places": counts.get("London", 0),
        "per_city_training_places": counts,
        "countermine_graph_places": len(graph_places),
        "mapped_graph_places": len(graph_places),
        "unmapped_graph_places": 0,
        "one_to_one_mapping": True,
        "identity_source": "dataset.get_img_name on every metadata row",
    }
    return tuple(records), audit

"""CPU-only descriptive analysis for the Step 3A diagnostic pilot.

This module accepts scalar inference results; it has no encoder, generator,
training loss, sampler, or mining API. The four-image R8 subset is shared by
the paired hard and random controls. A null computed on all triplets remains
the primary calibration for each compact row; subset recalibrations are named
explicitly. All standard deviations use ddof=1.
"""

import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np


R8_THRESHOLD = 0.40
ROLES = ("q", "p", "n_hard", "n_random")
NEGATIVE_KINDS = ("hard", "random")
LOCAL_METRICS = (
    "cross_match_ratio", "matched_source_cell_coverage", "matched_target_cell_coverage",
)
PERCENTILE_DEFINITION = "100 * count(random relative gain <= hard relative gain) / random count (weak ECDF)"
NULL_DEFINITION = {
    "control": "paired unrelated random negatives",
    "std_ddof": 1,
    "z_denominator_epsilon": 1e-12,
    "empirical_percentile_definition": PERCENTILE_DEFINITION,
    "singleton_policy": "sigma and z are null when fewer than two controls are available",
    "purpose": "diagnostic evidence only; no mining threshold or final CounterMine score",
}


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not a Boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _array(values, name="values"):
    return np.asarray([_finite(value, name) for value in values], dtype=np.float64)


def distribution_summary(values):
    """Linear empirical quantiles and sample standard deviation, without tests."""
    values = _array(values)
    result = {"count": int(values.size), "std_ddof": 1}
    names = ("mean", "std", "min", "q05", "q25", "q50", "median", "q75", "q95", "max")
    if not values.size:
        return {**result, **dict.fromkeys(names)}
    result.update({
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else None,
        "min": float(values.min()), "max": float(values.max()),
        **{f"q{q:02d}": float(np.quantile(values, q / 100)) for q in (5, 25, 50, 75, 95)},
        "median": float(np.median(values)),
    })
    return result


def compute_margin_metrics(s_qp, s_qhard, s_qrandom, sz_qp, sz_qhard, sz_qrandom):
    """Compute the two paired retrieval comparisons; a tie is never correct."""
    similarities = {
        name: _finite(value, name)
        for name, value in zip(
            ("s_qp", "s_qhard", "s_qrandom", "sz_qp", "sz_qhard", "sz_qrandom"),
            (s_qp, s_qhard, s_qrandom, sz_qp, sz_qhard, sz_qrandom),
        )
    }
    for name, value in similarities.items():
        if not -1.000001 <= value <= 1.000001:
            raise ValueError(f"{name} must be a cosine similarity in [-1, 1]")
    result = {"delta_positive_similarity": similarities["sz_qp"] - similarities["s_qp"]}
    for kind in NEGATIVE_KINDS:
        rgb = similarities["s_qp"] - similarities[f"s_q{kind}"]
        relit = similarities["sz_qp"] - similarities[f"sz_q{kind}"]
        shift = relit - rgb
        negative_shift = similarities[f"sz_q{kind}"] - similarities[f"s_q{kind}"]
        gain = negative_shift - result["delta_positive_similarity"]
        if not math.isclose(gain, -shift, abs_tol=1e-12, rel_tol=1e-10):
            raise ValueError("relative gain must equal negative delta margin")
        result.update({
            f"margin_{kind}_rgb": rgb, f"margin_{kind}_z": relit,
            f"delta_margin_{kind}": shift, f"relative_gain_{kind}": gain,
            f"delta_{kind}_negative_similarity": negative_shift,
            f"correct_vs_{kind}_rgb": rgb > 0,
            f"correct_vs_{kind}_z": relit > 0,
            f"{kind}_correct_to_wrong": rgb > 0 and relit <= 0,
            f"{kind}_wrong_to_correct": rgb <= 0 and relit > 0,
        })
    result["paired_delta_margin_difference"] = result["delta_margin_hard"] - result["delta_margin_random"]
    return result


def empirical_percentile(value, null_values):
    """Weak empirical CDF in percent; all ties count as <= the observation."""
    value = _finite(value, "hard gain")
    null = _array(null_values, "random gain")
    return float(100 * np.count_nonzero(null <= value) / len(null)) if len(null) else None


def random_null_metrics(random_gains):
    values = _array(random_gains, "random gain")
    return {
        "count": int(values.size),
        "mu_random": float(values.mean()) if values.size else None,
        "sigma_random": float(values.std(ddof=1)) if values.size > 1 else None,
        **NULL_DEFINITION,
        "relative_gain_random": distribution_summary(values),
    }


def _null_diagnostics(hard_gain, null, random_gains):
    mu, sigma = null["mu_random"], null["sigma_random"]
    centered = hard_gain - mu if mu is not None else None
    return {
        "centered_hard_gain": centered,
        "z_hard_gain": centered / (sigma + 1e-12) if sigma is not None else None,
        "empirical_percentile_hard_gain": empirical_percentile(hard_gain, random_gains),
    }


def enrich_triplet_record(row):
    """Return a fresh scalar row with margin, local, fidelity and strength fields."""
    result = dict(row)
    for role in ROLES:
        identity = result.get(f"{role}_image_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"{role}_image_id must be a nonempty string")
        _validate_string(identity, f"{role}_image_id")
    if (result["q_image_id"] == result["p_image_id"] or
            any(result[f"n_{kind}_image_id"] in (result["q_image_id"], result["p_image_id"])
                for kind in NEGATIVE_KINDS)):
        raise ValueError("q, p, and n must have distinct image identities within each triplet")
    cosine_names = ("s_qp", "s_qhard", "s_qrandom", "sz_qp", "sz_qhard", "sz_qrandom")
    for name in cosine_names:
        result[name] = _finite(result[name], name)
    result.update(compute_margin_metrics(**{name: result[name] for name in cosine_names}))
    place_fields = [f"{role}_place_uid" for role in ROLES]
    if any(name in result for name in place_fields):
        if not all(isinstance(result.get(name), str) and result[name] for name in place_fields):
            raise ValueError("all four place_uid fields must be present and nonempty")
        if result["q_place_uid"] != result["p_place_uid"]:
            raise ValueError("q and p must share place_uid")
        if any(result[f"n_{kind}_place_uid"] == result["q_place_uid"] for kind in NEGATIVE_KINDS):
            raise ValueError("q and n must differ in place_uid")
    for role in ROLES:
        name = f"{role}_source_relit_R8"
        result[name] = _finite(result[name], name)
        if not 0 <= result[name] <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
        result[f"{role}_fidelity_R8_pass"] = result[name] >= R8_THRESHOLD
        for suffix in ("rgb_mae_normalized", "abs_luma_mean_delta"):
            name = f"{role}_{suffix}"
            result[name] = _finite(result[name], name)
            if result[name] < 0 or (suffix == "rgb_mae_normalized" and result[name] > 1):
                raise ValueError(f"{name} must be a nonnegative valid intervention strength")
    result["all_images_R8_pass"] = all(result[f"{role}_fidelity_R8_pass"] for role in ROLES)
    result["hard_images_R8_pass"] = all(result[f"{role}_fidelity_R8_pass"] for role in ("q", "p", "n_hard"))
    result["mean_rgb_mae_triplet"] = sum(result[f"{role}_rgb_mae_normalized"] for role in ("q", "p", "n_hard")) / 3
    result["mean_abs_luma_delta_triplet"] = sum(result[f"{role}_abs_luma_mean_delta"] for role in ("q", "p", "n_hard")) / 3
    for kind in NEGATIVE_KINDS:
        for metric in LOCAL_METRICS:
            for condition in ("rgb", "z"):
                name = f"{metric}_{kind}_{condition}"
                result[name] = _finite(result[name], name)
                if not 0 <= result[name] <= 1:
                    raise ValueError(f"{name} must be in [0, 1]")
            result[f"delta_{metric}_{kind}"] = result[f"{metric}_{kind}_z"] - result[f"{metric}_{kind}_rgb"]
    return result


def enrich_triplet_records(rows):
    """Freeze per-row null diagnostics from all rows, with explicit subset extras."""
    records = [enrich_triplet_record(row) for row in rows]
    gains = [row["relative_gain_random"] for row in records]
    null = random_null_metrics(gains)
    selected = [row for row in records if row["all_images_R8_pass"]]
    subset_gains = [row["relative_gain_random"] for row in selected]
    subset_null = random_null_metrics(subset_gains)
    for row in records:
        row.update(_null_diagnostics(row["relative_gain_hard"], null, gains))
        if row["all_images_R8_pass"]:
            row.update({f"{name}_R8_pass_null": value for name, value in
                        _null_diagnostics(row["relative_gain_hard"], subset_null, subset_gains).items()})
        else:
            row.update({f"{name}_R8_pass_null": None for name in
                        ("centered_hard_gain", "z_hard_gain", "empirical_percentile_hard_gain")})
    return records


def _average_ranks(values):
    """One-based ranks with exact-value ties assigned their mean rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[start]] == values[order[stop]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    return ranks


def _pearson(a, b):
    if len(a) < 2:
        return None
    centered_a, centered_b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(centered_a) * np.linalg.norm(centered_b)
    return float(np.clip(np.dot(centered_a, centered_b) / denominator, -1, 1)) if denominator else None


def compute_correlations(x, y):
    """Pearson and Spearman coefficients only; no p-values or causal inference."""
    x, y = _array(x, "correlation x"), _array(y, "correlation y")
    if len(x) != len(y):
        raise ValueError("correlation vectors must have equal lengths")
    return {"count": int(len(x)), "pearson": _pearson(x, y),
            "spearman": _pearson(_average_ranks(x), _average_ranks(y)),
            "undefined_policy": "null for fewer than two samples or a constant vector"}


def _flip_summary(rows, kind):
    correct = sum(row[f"correct_vs_{kind}_rgb"] for row in rows)
    wrong = len(rows) - correct
    to_wrong = sum(row[f"{kind}_correct_to_wrong"] for row in rows)
    to_correct = sum(row[f"{kind}_wrong_to_correct"] for row in rows)
    return {
        "rgb_correct_count": int(correct), "rgb_wrong_or_tied_count": int(wrong),
        "correct_to_wrong": {"count": int(to_wrong), "eligible_rgb_correct_count": int(correct),
                             "fraction": float(to_wrong / correct) if correct else None},
        "wrong_to_correct": {"count": int(to_correct), "eligible_rgb_wrong_or_tied_count": int(wrong),
                             "fraction": float(to_correct / wrong) if wrong else None},
    }


def _subset_summary(rows):
    metric = lambda name: distribution_summary(row[name] for row in rows)
    negative = {
        kind: {"triplet_count": len(rows),
               "rgb_margin": metric(f"margin_{kind}_rgb"), "relit_margin": metric(f"margin_{kind}_z"),
               "delta_margin": metric(f"delta_margin_{kind}"), "relative_gain": metric(f"relative_gain_{kind}"),
               "flips": _flip_summary(rows, kind)}
        for kind in NEGATIVE_KINDS
    }
    differences = [row["paired_delta_margin_difference"] for row in rows]
    paired = distribution_summary(differences)
    paired.update({"num_negative": sum(value < 0 for value in differences),
                   "num_zero": sum(value == 0 for value in differences),
                   "num_positive": sum(value > 0 for value in differences),
                   "fraction_hard_delta_less_than_random_delta":
                       sum(value < 0 for value in differences) / len(rows) if rows else None,
                   "definition": "delta_margin_hard - delta_margin_random"})
    local = {kind: {name: {"rgb": metric(f"{name}_{kind}_rgb"),
                                   "relit": metric(f"{name}_{kind}_z"),
                                   "delta": metric(f"delta_{name}_{kind}")}
                    for name in LOCAL_METRICS} for kind in NEGATIVE_KINDS}
    strengths = ("mean_rgb_mae_triplet", "mean_abs_luma_delta_triplet")
    targets = ("delta_margin_hard", "relative_gain_hard", "delta_cross_match_ratio_hard")
    correlations = {strength: {target: compute_correlations(
        [row[strength] for row in rows], [row[target] for row in rows]) for target in targets}
        for strength in strengths}
    correlations["interpretation"] = "descriptive intervention-strength associations; no causal inference"
    joint = sum(row["delta_margin_hard"] < 0 and row["delta_cross_match_ratio_hard"] > 0 for row in rows)
    return {
        "positive_control": {"triplet_count": len(rows), "original_positive_similarity": metric("s_qp"),
                             "relit_positive_similarity": metric("sz_qp"),
                             "delta_positive_similarity": metric("delta_positive_similarity"),
                             "delta_hard_negative_similarity": metric("delta_hard_negative_similarity"),
                             "delta_random_negative_similarity": metric("delta_random_negative_similarity")},
        "hard_negative_summary": negative["hard"], "random_negative_summary": negative["random"],
        "paired_hard_vs_random": paired,
        "random_null": random_null_metrics(row["relative_gain_random"] for row in rows),
        "local_confusion_summary": local, "intervention_strength_correlations": correlations,
        "flip_counts": {kind: _flip_summary(rows, kind) for kind in NEGATIVE_KINDS},
        "joint_diagnostic": {"margin_collapse_and_increased_local_confusion_count": int(joint),
                             "fraction": float(joint / len(rows)) if rows else None,
                             "definition": "delta_margin_hard < 0 and delta_cross_match_ratio_hard > 0"},
    }


def summarize_triplets(rows):
    records = enrich_triplet_records(rows)
    all_summary = _subset_summary(records)
    pass_summary = _subset_summary([row for row in records if row["all_images_R8_pass"]])
    return {name: {"all_triplets": all_summary[name], "all_images_R8_pass": pass_summary[name]}
            for name in all_summary}


def _validate_string(value, location):
    if (value.startswith(("/", "\\", "~/", "~\\")) or
            re.match(r"^[A-Za-z]:[\\/]", value) or value.lower().startswith("file:")):
        raise ValueError(f"absolute paths are forbidden in compact snapshots: {location}")


def validate_portable_data(value, location="snapshot"):
    """Reject nonfinite values, absolute paths, arrays and non-JSON model objects."""
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        _validate_string(value, location)
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite value in {location}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"JSON keys must be strings in {location}")
            _validate_string(key, location)
            validate_portable_data(item, f"{location}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            validate_portable_data(item, f"{location}[{index}]")
        return
    raise ValueError(f"non-JSON object forbidden in compact snapshot: {location}")


def _compact_row(row):
    """Only scalar identities and metrics; never coordinates, paths or models."""
    result = {}
    for name, value in row.items():
        if "path" in name.lower() or "coordinate" in name.lower() or "checkpoint" in name.lower():
            continue
        if value is None or isinstance(value, (str, bool, int, float)):
            result[name] = value
        elif isinstance(value, np.generic):
            result[name] = value.item()
    validate_portable_data(result)
    return result


def build_snapshot(rows, *, provenance, population_construction, salad_config, iclight_config,
                   aliked_config, lightglue_config, counts=None):
    """Build the compact, portable snapshot from complete scalar diagnostic rows."""
    records = enrich_triplet_records(rows)
    query_ids = [row["q_image_id"] for row in records]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("each paired pilot record must have a unique query")
    image_ids = {row[f"{role}_image_id"] for row in records for role in ROLES}
    computed_counts = {"query_count": len(records), "unique_image_count": len(image_ids),
                       "generated_image_count": len(image_ids),
                       "R8_pass_triplet_count": sum(row["all_images_R8_pass"] for row in records)}
    if counts is not None:
        for name, value in counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"count {name} must be a nonnegative integer")
            if name in computed_counts and name != "generated_image_count" and value != computed_counts[name]:
                raise ValueError(f"supplied {name} disagrees with compact records")
            computed_counts[name] = value
        if computed_counts["generated_image_count"] > computed_counts["unique_image_count"]:
            raise ValueError("generated image count cannot exceed unique image count")
    compact = [_compact_row(row) for row in records]
    eligible = [row for row in compact if row["all_images_R8_pass"]]
    top = sorted(eligible, key=lambda row: (row["delta_margin_hard"], row["q_image_id"]))[:50]
    result = {
        "experiment": "Step 3A: Counterfactual Retrieval-Margin Pilot",
        "diagnostic_only": True,
        "provenance": provenance, "population_construction": population_construction,
        "salad_config": salad_config, "iclight_config": iclight_config,
        "seed_derivation_rule": {
            "namespace": "CounterMineVPR-Step3A|",
            "algorithm": "first 8 SHA256 bytes interpreted as unsigned big-endian integer modulo 2**63",
            "noise_realization": "independent deterministic per image_id",
        },
        "aliked_config": aliked_config, "lightglue_config": lightglue_config,
        "counts": computed_counts,
        "fidelity_subset": {"metric": "source_relit_R8", "threshold": R8_THRESHOLD,
                            "comparison": ">=", "required_images": list(ROLES),
                            "paired_subset": "common all-four-image R8-pass records"},
        "per_triplet_null_calibration": "all-triplet random-negative null; *_R8_pass_null fields explicitly recalibrate within the common subset",
        **summarize_triplets(records),
        "top_margin_collapse": top, "per_triplet_compact": compact,
        "analysis_scope": "descriptive only; no significance tests, final mining score, training objective, or claimed performance improvement",
    }
    validate_portable_data(result)
    # Validate the exact serializer requested by the experiment before writing.
    json.dumps(result, allow_nan=False)
    return result


def strict_json_dumps(snapshot):
    validate_portable_data(snapshot)
    return json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n"


def export_snapshot(snapshot, output_path):
    """Validate before touching a destination, then replace it atomically."""
    serialized = strict_json_dumps(snapshot)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return snapshot

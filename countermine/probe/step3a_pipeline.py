"""Frozen, resumable Step 3A diagnostic inference; no training entry points.

Only pilot global descriptors are memory mapped. Local features live for one
image pair at a time. Model imports are lazy, and each stage releases models
before the next stage. Failed R8 probes remain in the observations.
"""

import csv
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from countermine.probe.geometry_audit import sha256, validate_png
from countermine.probe.native100_fidelity import compute_intervention_strength, LUMA_DEFINITION
from countermine.probe.step3a_population import (
    derive_image_seed, load_manifest, iter_candidates, select_population, validate_triplets,
)


ROOT = Path(__file__).resolve().parents[2]
ROLES = ("q", "p", "n_hard", "n_random")
SEED_RULE = "first 8 bytes (big-endian unsigned) of SHA256('CounterMineVPR-Step3A|' + image_id), modulo 2**63"
PLOT_NAMES = ("top_margin_collapse_50.jpg", "hard_vs_random_margin_shift.png",
              "margin_vs_local_confusion.png", "rgb_vs_relit_margin.png")
PROMPT = "soft diffuse overcast daylight, uniform outdoor illumination, natural lighting"


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    json.dumps(value, allow_nan=False)
    return value


def save_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def save_csv(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def reference(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def git_head(directory=ROOT):
    return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()


def managed_cache(path):
    raw = Path(path).absolute()
    if any(parent.is_symlink() for parent in (raw, *raw.parents)):
        raise ValueError("Step3A managed cache must not use symlinks")
    path = raw.resolve()
    if not path.is_relative_to(ROOT / "cache") or path == ROOT / "cache":
        raise ValueError("Step3A runtime files must use a dedicated directory under cache/")
    if path.parts[len(ROOT.parts) + 1] in {"gsv_mini", "generator_audit", "geometry_audit", "native_fov_audit", "native100_audit"}:
        raise ValueError("Step3A must preserve historical caches")
    path.mkdir(parents=True, exist_ok=True)
    if any(child.is_symlink() for child in path.rglob("*")):
        raise ValueError("Step3A managed artifacts must not use child symlinks")
    return path


def image_token(image_id):
    return hashlib.sha256(image_id.encode("utf-8")).hexdigest()


def frozen_configs(snapshot_path, index_path, manifest_path, *, device="cuda"):
    """Freeze Step 2D2 settings and the candidate miner's SALAD policy."""
    saved = read_json(snapshot_path)
    provenance = saved["provenance"]
    iclight = dict(provenance["iclight_config"])
    fixed = {"width": 640, "height": 480, "prompt": PROMPT, "cfg": 2.0, "steps": 25,
             "highres_scale": 1.0, "highres_denoise": .5, "num_samples": 1,
             "added_prompt": "", "background": None}
    if any(iclight.get(key) != value for key, value in fixed.items()):
        raise ValueError("Step3A requires the frozen native Step2D2 IC-Light configuration")
    if provenance["rmbg_used"] is not False or provenance["two_stage_inference"] is not True:
        raise ValueError("Step3A requires full_scene two-stage inference without RMBG")
    geometry = saved["native_geometry"]
    if geometry["policy"] != "native_full_fov" or any(geometry[key] != value for key, value in
            {"canonical_width": 640, "canonical_height": 480, "scale_x": 1.0,
             "scale_y": 1.0, "retained_area_fraction": 1.0}.items()):
        raise ValueError("Step3A geometry must remain native full FOV")
    aliked = dict(provenance["aliked_config"])
    lightglue = dict(provenance["lightglue_config"])
    if aliked["max_num_keypoints"] != 2048 or aliked["detection_threshold"] != .2:
        raise ValueError("Step3A ALIKED settings differ from the frozen probe")
    if any(lightglue.get(key) != value for key, value in {
            "features": "aliked", "depth_confidence": -1, "width_confidence": -1,
            "filter_threshold": .1, "mp": False}.items()):
        raise ValueError("Step3A LightGlue settings differ from the frozen probe")
    index = read_json(index_path)
    if index["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("SALAD index does not belong to the selected manifest")
    if index["salad_model_name"] != "dinov2_salad" or index["salad_image_size"] != [322, 322]:
        raise ValueError("Expected the existing frozen 322x322 SALAD miner policy")
    if index["salad_git_commit"] != git_head(ROOT / "salad"):
        raise ValueError("SALAD checkout differs from the candidate index")
    iclight.pop("seed")
    iclight["seed_derivation_rule"] = SEED_RULE
    salad = {"model_name": index["salad_model_name"], "image_size": index["salad_image_size"],
             "salad_git_commit": index["salad_git_commit"], "seed": index["seed"],
             "preprocessing": "countermine.mining.salad_encoder.preprocess_image: RGB, torchvision bilinear Resize((322,322)), ToTensor, ImageNet normalization",
             "mean": [.485, .456, .406], "std": [.229, .224, .225],
             "output": "float32 L2-normalized descriptors; cosine computed after float64 normalization",
             "autocast": "float16 on CUDA, same as candidate encoder", "diagnostic_only": True}
    return {"iclight": iclight, "salad": salad, "aliked": aliked, "lightglue": lightglue,
            "local_seed": 42, "extract_resize": None, "geometry": geometry}


def build_population(manifest_path, candidates_path, mining_summary_path, index_path,
                     snapshot_path, dataset_root, cache_dir, *, selection_seed=42):
    cache_dir = managed_cache(cache_dir)
    manifest = load_manifest(manifest_path)
    mining = read_json(mining_summary_path)
    if mining["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("RGB candidate summary does not match manifest")
    configs = frozen_configs(snapshot_path, index_path, manifest_path)
    distance = float(mining["min_geo_distance_m"])
    inputs = {key: {"reference": reference(path), "sha256": sha256(path)} for key, path in {
        "manifest": manifest_path, "candidates": candidates_path, "mining_summary": mining_summary_path,
        "index_summary": index_path, "step2d2_snapshot": snapshot_path}.items()}
    destination = cache_dir / "population.json"
    if destination.exists():
        existing = read_json(destination)
        if (existing["inputs"] != inputs or existing["configs"] != configs
                or existing["selection_seed"] != selection_seed
                or existing["dataset_root"] != reference(dataset_root)):
            raise ValueError("Existing Step3A population differs; choose a new dedicated cache directory")
        return load_population(cache_dir)
    # Query eligibility is explicit; selection never changes after image failure.
    available = {row["image_id"] for row in manifest if
                 (Path(dataset_root) / row["relative_path"]).is_file()}
    triplets, accounting = select_population(manifest, iter_candidates(candidates_path),
        query_count=500, selection_seed=selection_seed, min_geo_distance_m=distance,
        available_image_ids=available)
    if not triplets:
        raise ValueError("No valid paired triplets; no experiment can run")
    used = {row[f"{role}_image_id"] for row in triplets for role in ROLES}
    images = []
    for row in manifest:
        if row["image_id"] not in used:
            continue
        token = image_token(row["image_id"])
        original = Path(dataset_root) / row["relative_path"]
        source = cache_dir / "source" / (token + ".png")
        source.parent.mkdir(exist_ok=True)
        with Image.open(original) as raw:
            pixels = raw.convert("RGB")
            if pixels.size != (640, 480):
                raise ValueError(f"Unexpected original geometry for {row['image_id']}: {pixels.size}; pilot not replaced")
            temporary = source.with_suffix(".partial")
            pixels.save(temporary, format="PNG")
            temporary.replace(source)
        images.append({"image_id": row["image_id"], "row_index": int(row["row_index"]),
                       "place_uid": row["place_uid"], "original_path": reference(original),
                       "source_path": reference(source),
                       "relit_path": reference(cache_dir / "relit" / (token + ".png")),
                       "original_sha256": sha256(original), "source_sha256": sha256(source),
                       "seed": derive_image_seed(row["image_id"])})
    population = {"triplets": triplets, "images": images, "population_construction": accounting,
                  "inputs": inputs, "configs": configs, "selection_seed": selection_seed,
                  "min_geo_distance_m": distance, "dataset_root": reference(dataset_root),
                  "query_count": len(triplets), "unique_image_count": len(images),
                  "probe_role": "diagnostic only; no optimization loss or training API",
                  "build_git_commit": git_head()}
    save_json(destination, population)
    save_csv(cache_dir / "triplets.csv", triplets)
    save_csv(cache_dir / "images.csv", images)
    print(f"Population: {len(triplets)} unique queries, {len(images)} unique images; geography >= {distance:g}m", flush=True)
    return population


def load_population(cache_dir):
    cache_dir = managed_cache(cache_dir)
    population = read_json(cache_dir / "population.json")
    from countermine.probe.step3a_metrics import validate_portable_data
    validate_portable_data(population)
    for entry in population["inputs"].values():
        if sha256(ROOT / entry["reference"]) != entry["sha256"]:
            raise ValueError("Frozen pilot input changed since population construction")
    manifest = load_manifest(ROOT / population["inputs"]["manifest"]["reference"])
    entries = population["inputs"]
    configs = frozen_configs(ROOT / entries["step2d2_snapshot"]["reference"],
        ROOT / entries["index_summary"]["reference"], ROOT / entries["manifest"]["reference"])
    if configs != population["configs"]:
        raise ValueError("Saved inference settings differ from frozen source snapshots")
    mining = read_json(ROOT / entries["mining_summary"]["reference"])
    if float(mining["min_geo_distance_m"]) != population["min_geo_distance_m"]:
        raise ValueError("Saved geographic rule differs from RGB candidate mining")
    available = {row["image_id"] for row in manifest if
                 (ROOT / population["dataset_root"] / row["relative_path"]).is_file()}
    selected, accounting = select_population(manifest, iter_candidates(ROOT / entries["candidates"]["reference"]),
        query_count=500, selection_seed=population["selection_seed"],
        min_geo_distance_m=population["min_geo_distance_m"], available_image_ids=available)
    if selected != population["triplets"] or accounting != population["population_construction"]:
        raise ValueError("Pilot no longer matches deterministic frozen paired selection")
    validate_triplets(population["triplets"], manifest, min_geo_distance_m=population["min_geo_distance_m"])
    image_ids = [image["image_id"] for image in population["images"]]
    expected = {row[f"{role}_image_id"] for row in population["triplets"] for role in ROLES}
    if len(image_ids) != len(set(image_ids)) or set(image_ids) != expected:
        raise ValueError("Pilot image cache must be exactly the unique triplet images")
    if population["query_count"] != len(selected) or population["unique_image_count"] != len(image_ids):
        raise ValueError("Saved pilot counts disagree with selected identities")
    manifest_lookup = {row["image_id"]: row for row in manifest}
    for image in population["images"]:
        original_row = manifest_lookup[image["image_id"]]
        token = image_token(image["image_id"])
        expected_record = {"row_index": int(original_row["row_index"]), "place_uid": original_row["place_uid"],
                           "original_path": reference(ROOT / population["dataset_root"] / original_row["relative_path"]),
                           "source_path": reference(cache_dir / "source" / (token + ".png")),
                           "relit_path": reference(cache_dir / "relit" / (token + ".png"))}
        if any(image.get(key) != value for key, value in expected_record.items()):
            raise ValueError("Pilot image record differs from manifest identity or image_id cache key")
        if image["seed"] != derive_image_seed(image["image_id"]):
            raise ValueError("Probe seed differs from independent image seed rule")
        validate_png(ROOT / image["source_path"], (640, 480))
        for kind in ("source", "original"):
            if sha256(ROOT / image[f"{kind}_path"]) != image[f"{kind}_sha256"]:
                raise ValueError("Frozen pilot RGB pixels changed")
    return population


def _seed_inference(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _release_models():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or denominator <= 0:
        raise ValueError("Diagnostic descriptors must be finite and nonzero")
    return float(np.dot(a, b) / denominator)


def freeze_descriptors(cache_dir, *, phase="rgb", batch_size=8, encoder_factory=None):
    if phase not in ("rgb", "relit") or batch_size < 1:
        raise ValueError("phase must be rgb/relit and batch_size positive")
    cache_dir = managed_cache(cache_dir)
    population = load_population(cache_dir)
    images = population["images"]
    generation = validate_generation(cache_dir, population) if phase == "relit" else None
    config = {**population["configs"]["salad"], "batch_size": batch_size}
    identity = {"population_sha256": sha256(cache_dir / "population.json"), "config": config,
                "phase": phase, "generation_sha256": sha256(cache_dir / "generation.json") if generation else None}
    summary_path = cache_dir / f"{phase}_salad.json"
    descriptor_path = cache_dir / f"{phase}_descriptors.npy"
    if summary_path.exists():
        saved = read_json(summary_path)
        if any(saved.get(key) != value for key, value in identity.items()) or sha256(descriptor_path) != saved["descriptor_sha256"]:
            raise ValueError("Frozen descriptor cache differs from requested inference")
        print(f"Using frozen {phase} SALAD descriptors", flush=True)
        return saved
    default_encoder = encoder_factory is None
    if default_encoder:
        from countermine.mining.salad_encoder import SaladEncoder
        if phase == "relit":
            import torch
            baseline = read_json(cache_dir / "rgb_salad.json")
            checkpoint = Path(torch.hub.get_dir()) / "checkpoints/dino_salad.ckpt"
            if not checkpoint.is_file() or sha256(checkpoint) != baseline["checkpoint_sha256"]:
                raise ValueError("SALAD checkpoint changed after frozen original RGB inference")
        encoder_factory = SaladEncoder
    _seed_inference(config["seed"])
    started = time.perf_counter()
    encoder = encoder_factory(image_size=config["image_size"][0], batch_size=batch_size)
    load_seconds = time.perf_counter() - started
    file = descriptor_path.with_suffix(".partial")
    bank = None
    paths = (ROOT / image["original_path" if phase == "rgb" else "relit_path"] for image in images)
    completed = 0
    for start, descriptors in encoder.iter_encode(paths):
        block = descriptors.numpy() if hasattr(descriptors, "numpy") else np.asarray(descriptors)
        if not np.isfinite(block).all() or block.ndim != 2 or start != completed:
            raise ValueError("Invalid streamed SALAD descriptors")
        if bank is None:
            bank = np.lib.format.open_memmap(file, mode="w+", dtype=np.float32,
                                            shape=(len(images), block.shape[1]))
        bank[start:start + len(block)] = block
        completed += len(block)
        if completed % (batch_size * 10) == 0 or completed == len(images):
            print(f"{phase} SALAD {completed}/{len(images)}", flush=True)
    if completed != len(images) or bank is None:
        raise ValueError("SALAD extraction did not produce exactly all pilot images")
    bank.flush()
    del bank, encoder
    _release_models()
    file.replace(descriptor_path)
    bank = np.load(descriptor_path, mmap_mode="r", allow_pickle=False)
    lookup = {image["image_id"]: i for i, image in enumerate(images)}
    measurements = []
    for row in population["triplets"]:
        descriptor = {role: bank[lookup[row[f"{role}_image_id"]]] for role in ROLES}
        prefix = "s" if phase == "rgb" else "sz"
        measured = {"q_image_id": row["q_image_id"],
                    **{f"{prefix}_q{name}": cosine(descriptor["q"], descriptor[role]) for name, role in
                       (("p", "p"), ("hard", "n_hard"), ("random", "n_random"))}}
        if phase == "rgb":
            for kind in ("hard", "random"):
                margin = measured["s_qp"] - measured[f"s_q{kind}"]
                measured[f"margin_{kind}_rgb"] = margin
                measured[f"correct_vs_{kind}_rgb"] = margin > 0
        measurements.append(measured)
    del bank
    checkpoint = None
    if default_encoder:
        import torch
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints/dino_salad.ckpt"
    summary = {**identity, "descriptor_sha256": sha256(descriptor_path),
               "checkpoint_sha256": sha256(checkpoint) if checkpoint is not None and checkpoint.is_file() else None,
               "model_load_elapsed_seconds": load_seconds,
               "elapsed_seconds": time.perf_counter() - started, "measurements": measurements,
               "synthetic_role": "diagnostic inference only" if phase == "relit" else "original RGB baseline"}
    save_json(summary_path, summary)
    save_csv(cache_dir / f"{phase}_similarities.csv", measurements)
    return summary


def generation_identity(cache_dir, population):
    return {"population_sha256": sha256(cache_dir / "population.json"),
            "iclight_config": population["configs"]["iclight"], "mode": "full_scene",
            "seed_derivation_rule": SEED_RULE, "rmbg_used": False, "two_stage_inference": True}


def record_iclight_assets(cache_dir, population):
    """Resolve cached model revisions and hash weights without a network request."""
    from huggingface_hub import snapshot_download, hf_hub_download
    config = population["configs"]["iclight"]
    base = Path(snapshot_download(config["base_model"], revision=config["base_revision"], local_files_only=True))
    offset = Path(hf_hub_download(config["offset_model"], config["offset_filename"],
                                 revision=config["offset_revision"], local_files_only=True))
    weights = {path.relative_to(base).as_posix(): sha256(path) for path in base.rglob("*")
               if path.is_file() and path.suffix in (".bin", ".safetensors")}
    result = {"base_model": config["base_model"], "resolved_base_revision": base.name,
              "base_weight_sha256": weights, "offset_model": config["offset_model"],
              "offset_filename": config["offset_filename"], "resolved_offset_revision": offset.parent.name,
              "offset_sha256": sha256(offset), "local_files_only": True,
              "historical_comparison_scope": "Step2D2 model IDs/settings preserved; historical snapshot did not record resolved checkpoint hashes"}
    destination = Path(cache_dir) / "iclight_assets.json"
    if destination.exists() and read_json(destination) != result:
        raise ValueError("Cached IC-Light assets changed from recorded pilot weights")
    save_json(destination, result)
    return result


def validate_generation(cache_dir, population):
    saved = read_json(cache_dir / "generation.json")
    if any(saved.get(key) != value for key, value in generation_identity(cache_dir, population).items()):
        raise ValueError("Generated cache differs from the frozen Step3A intervention")
    runs = {run["image_id"]: run for run in saved["runs"]}
    if len(runs) != len(saved["runs"]) or set(runs) != {image["image_id"] for image in population["images"]}:
        raise ValueError("Generation must retain every unique pilot image, including fidelity failures")
    if saved["generated_image_count"] != len(runs):
        raise ValueError("Generated image count must equal complete unique image runs")
    for image in population["images"]:
        run = runs[image["image_id"]]
        if (run["seed"] != image["seed"] or run["source_sha256"] != image["source_sha256"]
                or run["relit_path"] != image["relit_path"]):
            raise ValueError("Generated probe identity changed")
        path = ROOT / image["relit_path"]
        validate_png(path, (640, 480))
        if sha256(path) != run["output_sha256"]:
            raise ValueError("Generated probe pixels changed")
    return saved


def generate_probes(cache_dir, *, device="cuda", adapter_factory=None):
    cache_dir = managed_cache(cache_dir)
    population = load_population(cache_dir)
    # RGB measurements must be frozen before any synthetic inference.
    rgb = read_json(cache_dir / "rgb_salad.json")
    if rgb["population_sha256"] != sha256(cache_dir / "population.json"):
        raise ValueError("Original RGB measurements are not frozen for this pilot")
    if sha256(cache_dir / "rgb_descriptors.npy") != rgb["descriptor_sha256"] or rgb["phase"] != "rgb":
        raise ValueError("Original RGB descriptor measurements changed before generation")
    if (cache_dir / "generation.json").exists():
        return validate_generation(cache_dir, population)
    from countermine.probe.iclight_adapter import ICLightConfig, ICLightAdapter
    if adapter_factory is None:
        record_iclight_assets(cache_dir, population)
    config_values = dict(population["configs"]["iclight"])
    config_values.pop("seed_derivation_rule")
    config = ICLightConfig(**config_values, device=device,
                           seed=population["images"][0]["seed"], local_files_only=True)
    adapter_factory = adapter_factory or ICLightAdapter
    identity = generation_identity(cache_dir, population)
    journal_path = cache_dir / "generation_progress.json"
    journal = read_json(journal_path) if journal_path.exists() else {**identity, "runs": []}
    if any(journal.get(key) != value for key, value in identity.items()):
        raise ValueError("Generation progress differs from frozen configuration")
    runs = {run["image_id"]: run for run in journal["runs"]}
    if len(runs) != len(journal["runs"]) or not set(runs).issubset({i["image_id"] for i in population["images"]}):
        raise ValueError("Invalid generation progress identities")
    started = time.perf_counter()
    adapter = None
    load_seconds = 0.0
    (cache_dir / "relit").mkdir(exist_ok=True)
    for i, image in enumerate(population["images"]):
        destination = ROOT / image["relit_path"]
        if image["image_id"] in runs:
            run = runs[image["image_id"]]
            if run["seed"] != image["seed"] or run["source_sha256"] != image["source_sha256"] or sha256(destination) != run["output_sha256"]:
                raise ValueError("Resumed probe identity/pixels changed")
            continue
        if destination.exists():
            raise ValueError("Unlogged generated image exists; never silently replace it")
        if adapter is None:
            adapter = adapter_factory(config)
            load_started = time.perf_counter()
            adapter._ensure_models()
            load_seconds = time.perf_counter() - load_started
            print(f"IC-Light model loaded in {load_seconds:.2f}s", flush=True)
        # Only seed changes; the existing pipelines/checkpoints are retained.
        adapter.config = replace(config, seed=image["seed"])
        with Image.open(ROOT / image["source_path"]) as source:
            output = adapter.relight(source, "full_scene")
            if output.size != (640, 480) or output.mode != "RGB" or getattr(adapter, "rmbg", None) is not None:
                raise ValueError("Step3A output must be native RGB full_scene without RMBG")
            temporary = destination.with_suffix(".partial")
            output.save(temporary, format="PNG")
            output.close()
            temporary.replace(destination)
        run = {"image_id": image["image_id"], "seed": image["seed"],
               "source_sha256": image["source_sha256"], "output_sha256": sha256(destination),
               "relit_path": image["relit_path"],
               **{key: adapter.last_run_stats[key] for key in ("elapsed_seconds",
                  "peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes")}}
        runs[image["image_id"]] = run
        journal["runs"] = list(runs.values())
        journal.setdefault("model_load_elapsed_seconds_by_session", []).append(load_seconds) if load_seconds else None
        load_seconds = 0.0
        save_json(journal_path, journal)
        if (i + 1) % 10 == 0 or i + 1 == len(population["images"]):
            print(f"IC-Light {i + 1}/{len(population['images'])} (unique image seeds)", flush=True)
    del adapter
    _release_models()
    summary = {**journal, "generated_image_count": len(runs),
               "generation_elapsed_seconds": sum(r["elapsed_seconds"] for r in runs.values()),
               "latest_session_elapsed_seconds": time.perf_counter() - started}
    save_json(cache_dir / "generation.json", summary)
    save_csv(cache_dir / "generation.csv", list(runs.values()))
    return validate_generation(cache_dir, population)


def measure_local(cache_dir, *, device="cuda", matcher_factory=None):
    from countermine.probe.local_fidelity import LocalFidelityConfig
    from countermine.probe.step3a_local import Step3ALocalMatcher, cross_place_metrics_from_match
    cache_dir = managed_cache(cache_dir)
    population = load_population(cache_dir)
    validate_generation(cache_dir, population)
    identity = {"population_sha256": sha256(cache_dir / "population.json"),
                "generation_sha256": sha256(cache_dir / "generation.json"),
                "aliked_config": population["configs"]["aliked"],
                "lightglue_config": population["configs"]["lightglue"],
                "seed": 42, "resize": None, "R8_threshold": .40, "R8_coordinate_dtype": "float64",
                "cross_place_geometry": "unregistered; no displacement/R4/R8/homography/F-matrix"}
    destination = cache_dir / "local.json"
    if destination.exists():
        saved = read_json(destination)
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("Local cache differs from the frozen diagnostic configuration")
        validate_local_records(saved, population, complete=True)
        return saved
    progress = cache_dir / "local_progress.json"
    saved = read_json(progress) if progress.exists() else {**identity, "per_image": [], "cross_place": []}
    if any(saved.get(key) != value for key, value in identity.items()):
        raise ValueError("Local progress differs from the frozen diagnostic configuration")
    validate_local_records(saved, population, complete=False)
    matcher = (matcher_factory or Step3ALocalMatcher)(LocalFidelityConfig(device=device, max_keypoints=2048, seed=42))
    # A fully measured resumed journal still needs model/checkpoint provenance.
    if callable(getattr(matcher, "_ensure_models", None)):
        matcher._ensure_models()
    images = {image["image_id"]: image for image in population["images"]}
    measured = {row["image_id"] for row in saved["per_image"]}
    for i, image in enumerate(population["images"]):
        if image["image_id"] in measured:
            continue
        source = matcher.extract(ROOT / image["source_path"])
        relit = matcher.extract(ROOT / image["relit_path"])
        matches = matcher.match(source, relit)
        r8 = source_relit_R8(matches)
        strength = compute_intervention_strength(ROOT / image["source_path"], ROOT / image["relit_path"])
        saved["per_image"].append({"image_id": image["image_id"], "source_relit_R8": r8,
                                   "fidelity_R8_pass": r8 >= .40,
                                   "rgb_mae_normalized": strength["rgb_mae_normalized"],
                                   "abs_luma_mean_delta": abs(strength["luma_mean_delta"])})
        del source, relit, matches
        save_json(progress, saved)
        if (i + 1) % 25 == 0:
            print(f"Source-relit R8 {i + 1}/{len(images)}", flush=True)
    seen = {(row["q_image_id"], row["negative_image_id"], row["phase"]) for row in saved["cross_place"]}
    # Each pair extracts two images, matches, and releases its features immediately.
    for i, triplet in enumerate(population["triplets"]):
        for role in ("n_hard", "n_random"):
            q_id, n_id = triplet["q_image_id"], triplet[f"{role}_image_id"]
            for phase, key in (("rgb", "source_path"), ("z", "relit_path")):
                pair = (q_id, n_id, phase)
                if pair in seen:
                    continue
                a, b = matcher.extract(ROOT / images[q_id][key]), matcher.extract(ROOT / images[n_id][key])
                matches = matcher.match_cross_place(a, b)
                metrics = cross_place_metrics_from_match(matches)
                saved["cross_place"].append({"q_image_id": q_id, "negative_image_id": n_id,
                                             "phase": phase, **metrics})
                seen.add(pair)
                del a, b, matches
        save_json(progress, saved)
        if (i + 1) % 25 == 0 or i + 1 == len(population["triplets"]):
            print(f"Cross-place local triplets {i + 1}/{len(population['triplets'])}", flush=True)
    metadata = matcher.runtime_metadata()
    models = metadata.get("models") or {}
    for actual, frozen in (({**metadata.get("extractor_settings", {}), **models.get("extractor_config", {})}, population["configs"]["aliked"]),
                           ({**metadata.get("matcher_settings", {}), **models.get("matcher_config", {}), "compiled": metadata["compiled"],
                             "weights_version": models.get("lightglue_weights_version")}, population["configs"]["lightglue"])):
        if any(actual.get(key) != value for key, value in frozen.items()):
            raise ValueError("Runtime local model settings differ from Step2D2")
    saved["runtime_provenance"] = {"versions": metadata["versions"],
        "vendored_source_sha256": metadata["vendored_source_sha256"],
        "checkpoint_sha256": {key: value["sha256"] for key, value in models.get("checkpoints", {}).items()},
        "extract_resize": metadata["extract_resize"], "resolved_device": metadata["resolved_device"]}
    del matcher
    _release_models()
    save_json(destination, saved)
    save_csv(cache_dir / "fidelity.csv", saved["per_image"])
    save_csv(cache_dir / "cross_place.csv", saved["cross_place"])
    return saved


def source_relit_R8(matches):
    """The frozen inclusive 8-pixel repeatability in float64 native coordinates."""
    source = np.asarray(matches.points_source, dtype=np.float64)
    relit = np.asarray(matches.points_relit, dtype=np.float64)
    denominator = min(matches.num_keypoints_source, matches.num_keypoints_relit)
    displacement = np.linalg.norm(source - relit, axis=1)
    return float(np.count_nonzero(displacement <= 8.0) / denominator) if denominator else 0.0


def validate_local_records(saved, population, *, complete):
    """Validate resumable scalar identities and support definitions before use."""
    expected_images = {image["image_id"] for image in population["images"]}
    images = [row["image_id"] for row in saved["per_image"]]
    if len(images) != len(set(images)) or not set(images).issubset(expected_images):
        raise ValueError("Fidelity cache contains duplicate or unknown image identities")
    expected_pairs = {(row["q_image_id"], row[f"{role}_image_id"], phase) for row in population["triplets"]
                      for role in ("n_hard", "n_random") for phase in ("rgb", "z")}
    pairs = [(row["q_image_id"], row["negative_image_id"], row["phase"]) for row in saved["cross_place"]]
    if len(pairs) != len(set(pairs)) or not set(pairs).issubset(expected_pairs):
        raise ValueError("Cross-place cache contains duplicate or unknown pair identities")
    if complete and (set(images) != expected_images or set(pairs) != expected_pairs):
        raise ValueError("Completed local cache must cover every unique image and paired negative")
    for row in saved["per_image"]:
        r8 = row["source_relit_R8"]
        if not isinstance(r8, (int, float)) or not np.isfinite(r8) or not 0 <= r8 <= 1 or row["fidelity_R8_pass"] != (r8 >= .40):
            raise ValueError("Saved fidelity R8/flag differs from frozen definition")
        for key in ("rgb_mae_normalized", "abs_luma_mean_delta"):
            if not np.isfinite(row[key]) or row[key] < 0:
                raise ValueError("Saved intervention strength must be finite and nonnegative")
    for row in saved["cross_place"]:
        counts = [row[key] for key in ("num_matches", "num_keypoints_source", "num_keypoints_target")]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("Cross-place support counts must be nonnegative integers")
        matches, a, b = counts
        if matches > min(a, b) or row["cross_match_ratio"] != (matches / min(a, b) if min(a, b) else 0.0):
            raise ValueError("Cross-place match ratio differs from matched/keypoint counts")
        for key in ("matched_source_cell_coverage", "matched_target_cell_coverage"):
            value = row[key]
            if not np.isfinite(value) or not 0 <= value <= 1 or value * 64 != int(value * 64):
                raise ValueError("Cross-place coverage must use normalized occupied 8x8 cells")


def analyze_and_export(cache_dir, *, plot_dir=ROOT / "outputs/step3a", curated_dir=ROOT / "docs/audits"):
    from countermine.probe.step3a_metrics import build_snapshot, export_snapshot
    from countermine.probe.step3a_visuals import create_step3a_visual_outputs
    plot_dir, curated_dir = Path(plot_dir).resolve(), Path(curated_dir).resolve()
    if plot_dir != ROOT / "outputs/step3a" or curated_dir != ROOT / "docs/audits":
        raise ValueError("Use the dedicated Step3A plot and curated output directories")
    cache_dir = managed_cache(cache_dir)
    population = load_population(cache_dir)
    generation = validate_generation(cache_dir, population)
    rgb, relit, local = (read_json(cache_dir / name) for name in ("rgb_salad.json", "relit_salad.json", "local.json"))
    for data in (rgb, relit, local):
        if data["population_sha256"] != sha256(cache_dir / "population.json"):
            raise ValueError("Measurement belongs to a different pilot population")
    if rgb["config"] != relit["config"] or rgb["checkpoint_sha256"] != relit["checkpoint_sha256"]:
        raise ValueError("RGB/relit SALAD models and inference settings must be identical")
    expected_salad = population["configs"]["salad"]
    for phase, data in (("rgb", rgb), ("relit", relit)):
        if data["phase"] != phase or any(data["config"].get(key) != value for key, value in expected_salad.items()):
            raise ValueError("Saved SALAD phase/configuration differs from frozen pilot")
    for key, value in {"aliked_config": population["configs"]["aliked"],
                       "lightglue_config": population["configs"]["lightglue"],
                       "seed": 42, "resize": None, "R8_threshold": .40, "R8_coordinate_dtype": "float64",
                       "cross_place_geometry": "unregistered; no displacement/R4/R8/homography/F-matrix"}.items():
        if key not in local or local[key] != value:
            raise ValueError("Saved local configuration differs from frozen pilot")
    validate_local_records(local, population, complete=True)
    if relit["generation_sha256"] != sha256(cache_dir / "generation.json") or local["generation_sha256"] != sha256(cache_dir / "generation.json"):
        raise ValueError("Measurements do not belong to the completed generated probes")
    for phase, data in (("rgb", rgb), ("relit", relit)):
        if sha256(cache_dir / f"{phase}_descriptors.npy") != data["descriptor_sha256"]:
            raise ValueError("Frozen SALAD descriptor bank changed")
    baseline, intervention = ({row["q_image_id"]: row for row in data["measurements"]} for data in (rgb, relit))
    per_image = {row["image_id"]: row for row in local["per_image"]}
    cross = {(row["q_image_id"], row["negative_image_id"], row["phase"]): row for row in local["cross_place"]}
    queries = {row["q_image_id"] for row in population["triplets"]}
    for data, lookup in ((rgb, baseline), (relit, intervention)):
        if len(data["measurements"]) != len(lookup) or set(lookup) != queries:
            raise ValueError("SALAD measurements require exactly one row per paired query")
    expected_pairs = {(row["q_image_id"], row[f"{role}_image_id"], phase) for row in population["triplets"]
                      for role in ("n_hard", "n_random") for phase in ("rgb", "z")}
    if len(cross) != len(local["cross_place"]) or set(cross) != expected_pairs:
        raise ValueError("Cross-place measurements must retain exactly all paired negative probes")
    if len(per_image) != len(local["per_image"]):
        raise ValueError("Fidelity measurements contain duplicate image records")
    if set(per_image) != {image["image_id"] for image in population["images"]}:
        raise ValueError("Fidelity measurements must cover every pilot image")
    # Recompute the saved scalar cosine inputs from the frozen bank before export.
    descriptor_indices = {image["image_id"]: index for index, image in enumerate(population["images"])}
    for phase, lookup in (("rgb", baseline), ("relit", intervention)):
        bank = np.load(cache_dir / f"{phase}_descriptors.npy", mmap_mode="r", allow_pickle=False)
        if bank.ndim != 2 or len(bank) != len(descriptor_indices):
            raise ValueError("Pilot descriptor bank has inconsistent image identities")
        prefix = "s" if phase == "rgb" else "sz"
        for row in population["triplets"]:
            q = bank[descriptor_indices[row["q_image_id"]]]
            for name, role in (("p", "p"), ("hard", "n_hard"), ("random", "n_random")):
                expected_cosine = cosine(q, bank[descriptor_indices[row[f"{role}_image_id"]]])
                if lookup[row["q_image_id"]][f"{prefix}_q{name}"] != expected_cosine:
                    raise ValueError("Saved scalar cosine does not match frozen SALAD descriptors")
        del bank
    rows = []
    image_records = {image["image_id"]: image for image in population["images"]}
    for triplet in population["triplets"]:
        row = {key: value for key, value in triplet.items() if not key.endswith("relative_path")}
        row.update(baseline[row["q_image_id"]])
        row.update(intervention[row["q_image_id"]])
        for role in ROLES:
            row[f"{role}_seed"] = image_records[row[f"{role}_image_id"]]["seed"]
            metrics = per_image[row[f"{role}_image_id"]]
            for key in ("source_relit_R8", "fidelity_R8_pass", "rgb_mae_normalized", "abs_luma_mean_delta"):
                row[f"{role}_{key}"] = metrics[key]
        for kind in ("hard", "random"):
            for phase in ("rgb", "z"):
                metrics = cross[(row["q_image_id"], row[f"n_{kind}_image_id"], phase)]
                for key in ("num_matches", "num_keypoints_source", "num_keypoints_target", "cross_match_ratio",
                            "matched_source_cell_coverage", "matched_target_cell_coverage"):
                    row[f"{key}_{kind}_{phase}"] = metrics[key]
        rows.append(row)
    configs = population["configs"]
    provenance = {"git_commit": git_head(), "git_commit_scope": "repository HEAD at export; working changes included in source digests",
                  "source_sha256": {reference(path): sha256(path) for path in sorted(
                      list((ROOT / "countermine/probe").glob("step3a_*.py")) +
                      list((ROOT / "tools").glob("*step3a*.py")) +
                      [path for path in (ROOT / "countermine/mining/salad_encoder.py",
                       ROOT / "countermine/probe/iclight_adapter.py", ROOT / "countermine/probe/local_fidelity.py",
                       ROOT / "countermine/probe/canonical.py") if path.is_file()])},
                  "inputs": population["inputs"], "runtime_artifacts": {name: sha256(cache_dir / name) for name in
                    ("population.json", "rgb_salad.json", "generation.json", "relit_salad.json", "local.json")},
                  "salad_checkpoint_sha256": rgb["checkpoint_sha256"],
                  "iclight_assets": read_json(cache_dir / "iclight_assets.json") if (cache_dir / "iclight_assets.json").exists() else None,
                  "local_runtime": local["runtime_provenance"], "generation_elapsed_seconds": generation["generation_elapsed_seconds"],
                  "seed_derivation_rule": SEED_RULE, "extract_resize": None, "local_seed": 42,
                  "rmbg_used": False, "two_stage_inference": True, "native_geometry": configs["geometry"],
                  "intervention_strength_definition": LUMA_DEFINITION,
                  "fidelity_gate": {"source_relit_R8_min": .40, "filters_saved_triplets": False,
                                    "coordinate_dtype": "float64",
                                    "paired_subset": "all four q/p/hard/random images pass inclusive R8 >= 0.40"},
                  "interpretation": "descriptive pilot only; no significance test, causal claim, mining score or training objective"}
    snapshot = build_snapshot(rows, provenance=provenance,
        population_construction=population["population_construction"],
        salad_config=rgb["config"], iclight_config=configs["iclight"],
        aliked_config=configs["aliked"], lightglue_config=configs["lightglue"],
        counts={"query_count": len(rows), "unique_image_count": len(population["images"]),
                "generated_image_count": generation["generated_image_count"]})
    records = snapshot["per_triplet_compact"]
    save_json(cache_dir / "analysis.json", snapshot)
    save_csv(cache_dir / "per_triplet.csv", records)
    plot_dir.mkdir(parents=True, exist_ok=True)
    image_lookup = {image["image_id"]: {key: ROOT / image[key] for key in ("source_path", "relit_path")}
                    for image in population["images"]}
    create_step3a_visual_outputs(records, image_lookup, plot_dir)
    curated_dir.mkdir(parents=True, exist_ok=True)
    export_snapshot(snapshot, curated_dir / "step3a_counterfactual_margin_metrics.json")
    for name in PLOT_NAMES:
        shutil.copyfile(plot_dir / name, curated_dir / ("step3a_" + name))
    print(f"Exported Step3A: {len(rows)} paired queries and four curated figures", flush=True)
    return snapshot

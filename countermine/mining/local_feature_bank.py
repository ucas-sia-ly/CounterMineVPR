"""One-time, exact original-RGB ALIKED caches for Step 2D.

Model imports are lazy. Index/provenance validation uses only CPU metadata;
loading tensors is restricted to project-owned, hash-verified cache files.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import pandas as pd

from countermine.mining.full_structural_io import (
    ROOT, DEFAULT_DIR, code_hashes, frozen_provenance,
    guard_step2d, read_json, relative_path, write_csv, write_json,
)
from countermine.mining.structural_matcher import (
    ALIKED_CONFIG, IMAGE_SIZE, _state_sha256, canonical_json_bytes,
    decode_original_rgb, sha256_file, source_provenance,
)
from countermine.mining.full_population_metadata import validate_manifest


BANK_DIR = DEFAULT_DIR / "aliked_bank"
FEATURE_KEYS = ("descriptors", "image_size", "keypoint_scores", "keypoints")
INDEX_COLUMNS = (
    "row_index", "image_id", "image_file_sha256", "rgb_pixel_sha256",
    "num_keypoints", "feature_file", "feature_file_sha256",
)
BANK_CODE = (
    "countermine/mining/local_feature_bank.py",
    "countermine/mining/structural_matcher.py",
    "countermine/mining/full_structural_io.py", "tools/18_build_aliked_feature_bank.py",
    "countermine/mining/full_population_metadata.py",
)
VALIDATION_CODE = (
    "countermine/mining/local_feature_bank.py",
    "countermine/mining/banked_structural_matcher.py",
    "countermine/mining/structural_matcher.py",
    "countermine/mining/full_structural_io.py", "tools/19_validate_aliked_bank.py",
    "countermine/mining/full_structural_measurement.py",
)


def positive_integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def image_path(dataset_root, metadata):
    root = Path(dataset_root).resolve()
    relative = Path(metadata["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("manifest image path must be portable and relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"original manifest image is unavailable: {relative}")
    return path


def verify_source_images(index, manifest, dataset_root, *, image_ids=None):
    """Check encoded originals without re-extraction or retaining image pixels."""
    records = index.set_index("image_id")
    selected = set(index["image_id"] if image_ids is None else image_ids)
    if not selected.issubset(records.index):
        raise ValueError("source-byte validation includes images absent from the bank")
    for metadata in manifest.to_dict("records"):
        image_id = metadata["image_id"]
        if image_id in selected and sha256_file(image_path(dataset_root, metadata)) != records.loc[image_id, "image_file_sha256"]:
            raise ValueError(f"original RGB source bytes differ from the frozen bank: {image_id}")


def _array(value):
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


def feature_tensor_schema(features, *, tensor_type=None):
    """Check the actual four ALIKED output fields without converting precision.

    ``tensor_type`` supports NumPy CPU fixtures; production always requires
    torch.Tensor. No descriptors or coordinates enter JSON metadata.
    """
    if tensor_type is None:
        import torch
        tensor_type = torch.Tensor
    if not isinstance(features, dict) or tuple(sorted(features)) != FEATURE_KEYS:
        raise ValueError("ALIKED feature tensor keys differ from the frozen extractor output")
    schema = {}
    for key, value in features.items():
        if not isinstance(value, tensor_type):
            raise ValueError(f"feature {key} is not a tensor")
        array = _array(value)
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"feature {key} must preserve finite original float32 values")
        schema[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    keypoints = _array(features["keypoints"])
    if keypoints.ndim != 3 or keypoints.shape[0] != 1 or keypoints.shape[2] != 2:
        raise ValueError("ALIKED keypoints must have shape [1,N,2]")
    count = keypoints.shape[1]
    if count > ALIKED_CONFIG["max_num_keypoints"]:
        raise ValueError("ALIKED feature count exceeds the frozen keypoint budget")
    if tuple(features["descriptors"].shape) != (1, count, 128):
        raise ValueError("ALIKED descriptors must have shape [1,N,128]")
    if tuple(features["keypoint_scores"].shape) != (1, count):
        raise ValueError("ALIKED keypoint_scores must have shape [1,N]")
    size = _array(features["image_size"])
    if size.shape != (1, 2) or not np.array_equal(size[0], IMAGE_SIZE):
        raise ValueError("bank features must retain original 640x480 image_size")
    if (keypoints < -1e-6).any() or (keypoints > np.asarray(IMAGE_SIZE) + 1e-4).any():
        raise ValueError("bank keypoints are outside original image coordinates")
    return schema


def atomic_feature_file(path, features, *, repo_root=ROOT, torch_module=None):
    """Serialize only project-owned CPU tensors, with durable atomic rename."""
    guard_step2d([path], repo_root=repo_root)
    if torch_module is None:
        import torch as torch_module
    cpu = {key: tensor.detach().cpu() for key, tensor in features.items()}
    schema = feature_tensor_schema(cpu, tensor_type=torch_module.Tensor)
    if any(tensor.device.type != "cpu" for tensor in cpu.values()):
        raise ValueError("feature bank serialization requires CPU tensors")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch_module.save(cpu, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return schema


def feature_path(bank_dir, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("feature file path must be portable and bank-relative")
    bank_dir = Path(bank_dir).resolve()
    path = bank_dir / relative
    if path.is_symlink() or not path.resolve().is_relative_to(bank_dir):
        raise ValueError("feature file escapes the bank directory")
    return path


def load_tensor_features(bank_dir, record, *, torch_module=None):
    """Verify bytes before weights-only deserialization of our own tensor cache."""
    path = feature_path(bank_dir, record["feature_file"])
    if not path.is_file() or sha256_file(path) != record["feature_file_sha256"]:
        raise ValueError(f"missing or corrupt feature cache for {record['image_id']}")
    if torch_module is None:
        import torch as torch_module
    features = torch_module.load(path, map_location="cpu", weights_only=True)
    feature_tensor_schema(features, tensor_type=torch_module.Tensor)
    if int(features["keypoints"].shape[1]) != int(record["num_keypoints"]):
        raise ValueError("feature tensor count differs from its bank index")
    if any(value.device.type != "cpu" for value in features.values()):
        raise ValueError("feature file did not deserialize as CPU tensors")
    return features


def setup_torch(device="cuda", seed=42, repo_root=ROOT):
    """Mirror frozen Step 2A deterministic settings before constructing models."""
    seed = positive_integer(seed, "seed")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vendor = (Path(repo_root) / "third_party/LightGlue").resolve()
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    import lightglue
    if not Path(lightglue.__file__).resolve().is_relative_to(vendor):
        raise RuntimeError("LightGlue import did not resolve to the frozen read-only vendor")
    selected = torch.device(device)
    provenance = {
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "device": str(selected),
        "device_name": torch.cuda.get_device_name(selected) if selected.type == "cuda" else "cpu",
        "deterministic_algorithms": True, "cudnn_benchmark": False,
        "cudnn_deterministic": True, "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "package_versions": {name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "numpy", "Pillow", "kornia", "opencv-python",
        )},
    }
    return torch, lightglue, selected, provenance


def bind_frozen_runtime(provenance, frozen_config, model):
    """A changed model/environment requires a new scientific task, not a tolerance."""
    frozen = frozen_config["matcher_provenance"]
    keys = ("torch_version", "cuda_version", "cudnn_version", "device_name",
            "cublas_workspace_config", "deterministic_algorithms",
            f"{model}_weights_sha256", f"{model}_effective_config")
    for key in keys:
        if provenance.get(key) != frozen.get(key):
            raise ValueError(f"frozen Step 2A {model} runtime mismatch: {key}")


class AlikedBankExtractor:
    def __init__(self, device="cuda", seed=42, repo_root=ROOT):
        self.torch, vendor, self.device, self.provenance = setup_torch(device, seed, repo_root)
        self.extractor = vendor.ALIKED(**ALIKED_CONFIG).eval().to(self.device)
        self.provenance.update({
            "aliked_weights_sha256": _state_sha256(self.extractor),
            "aliked_effective_config": vars(self.extractor.conf),
        })

    def extract_pixels(self, pixels):
        image = self.torch.from_numpy(pixels.transpose(2, 0, 1).copy()).to(
            device=self.device, dtype=self.torch.float32,
        ) / 255.0
        with self.torch.inference_mode():
            features = self.extractor.extract(image, resize=None)
        cpu = {key: value.detach().cpu() for key, value in features.items()}
        feature_tensor_schema(cpu, tensor_type=self.torch.Tensor)
        del image, features
        return cpu


def load_bank_manifest(manifest_path, *, repo_root=ROOT):
    """Bind every manifest image to frozen population metadata and hashes."""
    root = Path(repo_root)
    config = read_json(root / "cache/countermine_rgb/step2a/measurement_config.json")
    frozen_population = read_json(root / "cache/countermine_rgb/step2a/population_summary.json")
    raw_summary = read_json(root / "cache/gsv_mini/rgb_candidates_raw_summary.json")
    digest = sha256_file(manifest_path)
    if digest != config["input_hashes"]["manifest_sha256"] or digest != frozen_population["manifest_sha256"]:
        raise ValueError("manifest differs from frozen Step 2A provenance")
    manifest = pd.read_csv(manifest_path, float_precision="round_trip", dtype={
        "image_id": str, "place_uid": str, "city_id": str,
    })
    validate_manifest(manifest)
    if len(manifest) != raw_summary["number_of_queries"] or len(manifest) * raw_summary["top_k"] != frozen_population["raw_directed_candidate_rows"]:
        raise ValueError("manifest population size differs from frozen retrieval provenance")
    if config["aliked_config"] != ALIKED_CONFIG or config["seed"] != 42:
        raise ValueError("frozen ALIKED configuration or seed is incompatible")
    if config["vendor_provenance"]["source_sha256"] != source_provenance(root)["source_sha256"]:
        raise ValueError("vendored LightGlue source differs from frozen Step 2A")
    fingerprints = read_json(root / "cache/countermine_rgb/step2a/image_fingerprints.json")
    if sha256_file(root / "cache/countermine_rgb/step2a/image_fingerprints.json") != config["input_image_index_sha256"]:
        raise ValueError("historical original-image fingerprints changed")
    return manifest, config, fingerprints


def validate_bank_index(index, manifest):
    if tuple(index.columns) != INDEX_COLUMNS or len(index) != len(manifest):
        raise ValueError("feature-bank index schema/count differs from the full manifest")
    if index.isna().any().any() or not index["image_id"].is_unique:
        raise ValueError("feature-bank index contains missing or duplicate entries")
    rows = pd.to_numeric(index["row_index"], errors="raise").to_numpy()
    if not np.array_equal(rows, manifest["row_index"].to_numpy()) or index["image_id"].tolist() != manifest["image_id"].tolist():
        raise ValueError("feature-bank image identity/order differs from frozen manifest")
    count = pd.to_numeric(index["num_keypoints"], errors="raise").to_numpy()
    if not np.isfinite(count).all() or (count < 0).any() or (count > 2048).any() or (count != np.floor(count)).any():
        raise ValueError("invalid feature-bank keypoint counts")
    for key in ("image_file_sha256", "rgb_pixel_sha256", "feature_file_sha256"):
        if not index[key].map(lambda value: isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)).all():
            raise ValueError(f"invalid feature-bank hash: {key}")
    expected = [f"features/{int(row):06d}.pt" for row in rows]
    if index["feature_file"].tolist() != expected:
        raise ValueError("feature files must use deterministic frozen row-index names")
    return index.assign(row_index=rows.astype(np.int64), num_keypoints=count.astype(np.int64))


def _check_image_record(record, row, configuration_sha256, historical=None):
    if record.get("schema_version") != 1 or record.get("complete") is not True or record.get("configuration_sha256") != configuration_sha256:
        raise ValueError("resume refused: image feature completion/configuration mismatch")
    if record.get("row_index") != int(row["row_index"]) or record.get("image_id") != row["image_id"]:
        raise ValueError("resume refused: image feature identity mismatch")
    if historical is not None and any(record.get(key) != value for key, value in historical.items()):
        raise ValueError("source image differs from historical Step 2A fingerprint")
    expected = f"features/{int(row['row_index']):06d}.pt"
    if record.get("feature_file") != expected or set(record.get("tensor_schema", {})) != set(FEATURE_KEYS):
        raise ValueError("feature completion record has invalid layout or tensor keys")


def load_feature_bank(bank_dir=BANK_DIR, *, repo_root=ROOT, verify_features=True):
    bank_dir = Path(bank_dir)
    guard_step2d([bank_dir / "summary.json", bank_dir / "index.csv"], repo_root=repo_root)
    summary = read_json(bank_dir / "summary.json")
    if not isinstance(summary, dict) or summary.get("schema_version") != 1 or summary.get("complete") is not True:
        raise ValueError("ALIKED bank has no valid atomic completion marker")
    config = read_json(bank_dir / "configuration.json")
    config_sha = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
    if summary.get("configuration") != config or summary.get("configuration_sha256") != config_sha:
        raise ValueError("feature-bank configuration binding differs")
    if config["code_sha256"] != code_hashes(BANK_CODE, repo_root=repo_root):
        raise ValueError("feature-bank extraction code provenance differs")
    manifest_path = Path(repo_root) / config["manifest_file"]
    manifest, historical_config, historical = load_bank_manifest(manifest_path, repo_root=repo_root)
    if config["manifest_sha256"] != sha256_file(manifest_path) or config["frozen_provenance"] != frozen_provenance(repo_root=repo_root):
        raise ValueError("feature-bank input scientific provenance differs")
    bind_frozen_runtime(config["extractor_provenance"], historical_config, "aliked")
    if summary.get("index_sha256") != sha256_file(bank_dir / "index.csv"):
        raise ValueError("feature-bank index hash differs")
    index = validate_bank_index(pd.read_csv(bank_dir / "index.csv", keep_default_na=False), manifest)
    if summary.get("image_count") != len(index):
        raise ValueError("feature-bank summary image count differs")
    for row in index.to_dict("records"):
        record = read_json(bank_dir / "records" / f"{row['row_index']:06d}.json")
        _check_image_record(record, row, config_sha, historical.get(row["image_id"]))
        if any(record.get(key) != row[key] for key in INDEX_COLUMNS):
            raise ValueError("feature-bank completion record differs from index")
        path = feature_path(bank_dir, row["feature_file"])
        if not path.is_file() or (verify_features and sha256_file(path) != row["feature_file_sha256"]):
            raise ValueError(f"missing or corrupt feature cache: {row['image_id']}")
    return index, summary


def require_bank_validation(bank_dir, summary, *, repo_root=ROOT):
    marker_path = Path(bank_dir) / "validation.json"
    if not marker_path.exists():
        raise ValueError("full matching requires successful tools/19_validate_aliked_bank.py replay first")
    marker = read_json(marker_path)
    if not isinstance(marker, dict) or marker.get("schema_version") != 1 or marker.get("complete") is not True or marker.get("passed") is not True or marker.get("bank_summary_sha256") != sha256_file(Path(bank_dir) / "summary.json"):
        raise ValueError("bank replay completion marker does not validate this feature bank")
    if marker.get("code_sha256") != code_hashes(VALIDATION_CODE, repo_root=repo_root):
        raise ValueError("bank replay validation code changed; rerun validation")
    if marker.get("direct_feature_replay", {}).get("image_count") != 32 or marker.get("pair_replay", {}).get("pair_count") != 512:
        raise ValueError("bank replay marker lacks required 32-image / 512-pair validation")
    direct, paired = marker["direct_feature_replay"], marker["pair_replay"]
    if direct.get("passed") is not True or direct.get("exact_tensor_equality") is not True or paired.get("passed") is not True:
        raise ValueError("bank replay marker records failed feature or historical pair validation")
    if direct.get("maximum_absolute_difference") != {key: 0.0 for key in FEATURE_KEYS}:
        raise ValueError("bank replay marker does not certify exact feature tensor equality")
    if marker.get("configuration_sha256") != summary["configuration_sha256"]:
        raise ValueError("bank replay marker configuration differs")
    return marker


def build_feature_bank(args, *, repo_root=ROOT, extractor_factory=AlikedBankExtractor):
    root = Path(repo_root)
    bank_dir = Path(args.bank_dir)
    guard_step2d([bank_dir / "configuration.json", bank_dir / "index.csv", bank_dir / "summary.json", bank_dir / ".bank.lock"], repo_root=root)
    manifest, frozen_config, historical = load_bank_manifest(args.manifest, repo_root=root)
    if args.seed != frozen_config["seed"]:
        raise ValueError("feature bank seed differs from frozen Step 2A seed")
    scientific = frozen_provenance(repo_root=root)
    extractor = extractor_factory(device=args.device, seed=args.seed, repo_root=root)
    bind_frozen_runtime(extractor.provenance, frozen_config, "aliked")
    config = {
        "schema_version": 1, "seed": args.seed,
        "manifest_file": relative_path(args.manifest, repo_root=root),
        "manifest_sha256": sha256_file(args.manifest), "manifest_image_count": len(manifest),
        "aliked_config": ALIKED_CONFIG, "extractor_provenance": extractor.provenance,
        "vendor_provenance": {
            "source_sha256": source_provenance(root)["source_sha256"],
            "historical_reported_git_commit": frozen_config["vendor_provenance"].get("git_commit"),
            "containing_repository_git_commit": source_provenance(root).get("git_commit"),
        }, "code_sha256": code_hashes(BANK_CODE, repo_root=root),
        "image_policy": frozen_config["image_policy"], "frozen_provenance": scientific,
    }
    config_sha = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
    bank_dir.mkdir(parents=True, exist_ok=True)
    with (bank_dir / ".bank.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another process owns this feature bank") from error
        config_path = bank_dir / "configuration.json"
        if config_path.exists():
            if read_json(config_path) != config:
                raise ValueError("resume refused: source/configuration/weights/code/environment changed")
        else:
            if any((bank_dir / name).exists() for name in ("index.csv", "summary.json")) or any((bank_dir / "records").glob("*.json")):
                raise ValueError("resume refused: feature caches exist without configuration provenance")
            write_json(config_path, config, repo_root=root)
        rows = []
        for offset, metadata in enumerate(manifest.to_dict("records"), 1):
            path = image_path(args.dataset_root, metadata)
            pixels, fingerprints = decode_original_rgb(path)
            if metadata["image_id"] in historical and historical[metadata["image_id"]] != fingerprints:
                raise ValueError(f"original image differs from frozen Step 2A: {metadata['image_id']}")
            number = int(metadata["row_index"])
            record_path = bank_dir / "records" / f"{number:06d}.json"
            destination = bank_dir / "features" / f"{number:06d}.pt"
            guard_step2d([record_path, destination], repo_root=root)
            if record_path.exists():
                record = read_json(record_path)
                _check_image_record(record, metadata, config_sha, fingerprints)
                if not destination.is_file() or sha256_file(destination) != record["feature_file_sha256"]:
                    raise ValueError("resume refused: completed feature file is missing or corrupt")
            else:
                features = extractor.extract_pixels(pixels)
                schema = atomic_feature_file(destination, features, repo_root=root, torch_module=extractor.torch)
                record = {
                    "schema_version": 1, "complete": True, "configuration_sha256": config_sha,
                    "row_index": number, "image_id": metadata["image_id"], **fingerprints,
                    "num_keypoints": int(features["keypoints"].shape[1]),
                    "feature_file": f"features/{number:06d}.pt", "feature_file_sha256": sha256_file(destination),
                    "tensor_schema": schema,
                }
                write_json(record_path, record, repo_root=root)
                del features
            rows.append({key: record[key] for key in INDEX_COLUMNS})
            del pixels
            if offset % 100 == 0 or offset == len(manifest):
                print(f"ALIKED bank: {offset}/{len(manifest)} images", flush=True)
        index = validate_bank_index(pd.DataFrame(rows, columns=INDEX_COLUMNS), manifest)
        # Recheck source bytes before publishing completion, including resumed files.
        for metadata, row in zip(manifest.to_dict("records"), rows):
            if sha256_file(image_path(args.dataset_root, metadata)) != row["image_file_sha256"]:
                raise ValueError("original image bytes changed during bank construction")
        if frozen_provenance(repo_root=root) != scientific or code_hashes(BANK_CODE, repo_root=root) != config["code_sha256"]:
            raise ValueError("inputs/code changed during feature bank construction")
        if (bank_dir / "summary.json").exists():
            old_index, old_summary = load_feature_bank(bank_dir, repo_root=root)
            if old_index.to_dict("records") != index.to_dict("records"):
                raise ValueError("completed feature bank differs; publication refused")
            return old_summary
        write_csv(bank_dir / "index.csv", index, repo_root=root)
        summary = {
            "schema_version": 1, "complete": True, "image_count": len(index),
            "configuration": config, "configuration_sha256": config_sha,
            "index_sha256": sha256_file(bank_dir / "index.csv"),
            "real_rgb_only": True, "synthetic_images_used": False,
        }
        write_json(bank_dir / "summary.json", summary, repo_root=root)
        return summary


def deterministic_replay_ids(identities, prefix, count, seed=42):
    if not isinstance(identities, (list, tuple)):
        identities = list(identities)
    if len(set(identities)) != len(identities) or len(identities) < count:
        raise ValueError("replay population is duplicated or smaller than the required subset")
    return sorted(identities, key=lambda identity: (
        hashlib.sha256(f"{prefix}|{seed}|{identity}".encode()).hexdigest(), identity,
    ))[:count]


def compare_feature_replay(direct, cached, *, tensor_type=None):
    direct_schema = feature_tensor_schema(direct, tensor_type=tensor_type)
    cached_schema = feature_tensor_schema(cached, tensor_type=tensor_type)
    if direct_schema != cached_schema:
        raise ValueError("direct feature replay tensor keys/shapes/dtypes differ")
    maximum = {}
    for key in FEATURE_KEYS:
        a, b = _array(direct[key]), _array(cached[key])
        difference = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
        maximum[key] = difference
        if not np.array_equal(a, b):
            raise ValueError(f"direct ALIKED feature replay is not exact: {key}; max_absolute_difference={difference:.17g}; STOP")
    return maximum


def validate_feature_bank(args, *, repo_root=ROOT, extractor_factory=AlikedBankExtractor,
                          matcher_factory=None):
    """Publish success only after exact 32-image and frozen 512-pair replay."""
    from countermine.mining.full_structural_measurement import compare_metric_frames
    from countermine.mining.structural_analysis import read_metadata_csv, validate_metrics
    if matcher_factory is None:
        from countermine.mining.banked_structural_matcher import BankedStructuralMatcher
        matcher_factory = BankedStructuralMatcher
    root, bank_dir = Path(repo_root), Path(args.bank_dir)
    marker_path = bank_dir / "validation.json"
    guard_step2d([marker_path, bank_dir / ".validation.lock"], repo_root=root)
    index, summary = load_feature_bank(bank_dir, repo_root=root)
    manifest, frozen_config, historical = load_bank_manifest(args.manifest, repo_root=root)
    if args.seed != 42 or args.seed != summary["configuration"]["seed"]:
        raise ValueError("bank replay seed differs from the frozen experiment")
    provenance = frozen_provenance(repo_root=root)
    validation_code = code_hashes(VALIDATION_CODE, repo_root=root)
    base = {"schema_version": 1, "seed": args.seed,
            "bank_summary_sha256": sha256_file(bank_dir / "summary.json"),
            "configuration_sha256": summary["configuration_sha256"],
            "code_sha256": validation_code, "frozen_provenance": provenance,
            "real_rgb_only": True, "synthetic_images_used": False}
    with (bank_dir / ".validation.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another process owns the bank replay validation") from error
        # An interrupted or failed fresh replay cannot leave an earlier success
        # marker authorizing matching in the newly checked environment.
        write_json(marker_path, {**base, "complete": False, "passed": False}, repo_root=root)
        extractor = extractor_factory(device=args.device, seed=args.seed, repo_root=root)
        bind_frozen_runtime(extractor.provenance, frozen_config, "aliked")
        if extractor.provenance != summary["configuration"]["extractor_provenance"]:
            raise ValueError("bank replay extractor environment differs from bank construction")
        selected = deterministic_replay_ids(manifest["image_id"], "CounterMineVPR-Step2D-feature-replay", 32, args.seed)
        sources, records = manifest.set_index("image_id"), index.set_index("image_id")
        maximum = {key: 0.0 for key in FEATURE_KEYS}
        for image_id in selected:
            pixels, fingerprints = decode_original_rgb(image_path(args.dataset_root, sources.loc[image_id].to_dict()))
            record = records.loc[image_id].to_dict()
            record["image_id"] = image_id
            if any(record[key] != value for key, value in fingerprints.items()):
                raise ValueError(f"direct replay original image differs from feature-bank hash: {image_id}")
            direct = extractor.extract_pixels(pixels)
            cached = load_tensor_features(bank_dir, record, torch_module=extractor.torch)
            difference = compare_feature_replay(direct, cached, tensor_type=extractor.torch.Tensor)
            maximum = {key: max(maximum[key], difference[key]) for key in FEATURE_KEYS}
            del direct, cached, pixels
        direct_report = {"passed": True, "image_count": 32, "exact_tensor_equality": True,
                         "image_id_sequence_sha256": hashlib.sha256(canonical_json_bytes(selected)).hexdigest(),
                         "maximum_absolute_difference": maximum,
                         "selection": "SHA256(CounterMineVPR-Step2D-feature-replay|42| + image_id), ascending digest then image_id"}
        del extractor
        pilot_path = root / "cache/countermine_rgb/step2a/candidate_structural_metrics.csv"
        historical_pairs = validate_metrics(read_metadata_csv(pilot_path), candidate=True)
        if len(historical_pairs) != 5000:
            raise ValueError("Step 2A pair replay requires the frozen 5,000-pair pilot")
        selected_pairs = deterministic_replay_ids(historical_pairs["pair_uid"], "CounterMineVPR-Step2D-pair-replay", 512, args.seed)
        subset = historical_pairs.set_index("pair_uid").loc[selected_pairs].reset_index()
        matcher = matcher_factory(bank_dir=bank_dir, device=args.device, seed=args.seed,
                                  feature_cache_size=args.feature_cache_size, repo_root=root,
                                  require_validation=False)
        measured = []
        for offset, row in enumerate(subset.to_dict("records"), 1):
            result = matcher.match(row["image_id_a"], row["image_id_b"])
            measured.append({"pair_uid": row["pair_uid"], "image_id_a": row["image_id_a"],
                             "image_id_b": row["image_id_b"], **result.metrics})
            del result
            if offset % 64 == 0:
                print(f"Bank-backed Step 2A replay: {offset}/512 pairs", flush=True)
        pair_report = compare_metric_frames(pd.DataFrame(measured), subset, expected_count=512)
        pair_report.update({"pair_uid_sequence_sha256": hashlib.sha256(canonical_json_bytes(selected_pairs)).hexdigest(),
                            "historical_candidate_metrics_sha256": sha256_file(pilot_path),
                            "feature_source": "disk_bank_only"})
        # All manifest images, including those outside direct replay, retain
        # the source-byte binding that bank construction proved.
        for metadata in manifest.to_dict("records"):
            if sha256_file(image_path(args.dataset_root, metadata)) != records.loc[metadata["image_id"], "image_file_sha256"]:
                raise ValueError("original image bytes changed before bank replay publication")
        if sha256_file(bank_dir / "summary.json") != base["bank_summary_sha256"] or frozen_provenance(repo_root=root) != provenance:
            raise ValueError("bank/frozen evidence changed during replay validation")
        if code_hashes(VALIDATION_CODE, repo_root=root) != validation_code:
            raise ValueError("bank replay code changed during validation")
        result = {**base, "complete": True, "passed": True,
                  "direct_feature_replay": direct_report, "pair_replay": pair_report,
                  "matcher_provenance": matcher.provenance}
        write_json(marker_path, result, repo_root=root)
        return result

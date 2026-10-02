"""Frozen Step 3A configuration and portable, project-owned artifact helpers.

Importing this module does not import any model runtime or change directories.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
MODES = ("baseline", "countermine_q99_geo500")
TRAIN_CITIES = (
    "Bangkok", "BuenosAires", "LosAngeles", "MexicoCity", "OSL", "Rome",
    "Barcelona", "Chicago", "Madrid", "Miami", "Phoenix", "TRT", "Boston",
    "Lisbon", "Medellin", "Minneapolis", "PRG", "WashingtonDC", "Brussels",
    "London", "Melbourne", "Osaka", "PRS",
)
MODEL_CONFIG = {
    "backbone_arch": "dinov2_vitb14",
    "backbone_config": {"num_trainable_blocks": 4, "return_token": True, "norm_layer": True},
    "agg_arch": "SALAD",
    "agg_config": {"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256},
    "lr": 6e-5, "optimizer": "adamw", "weight_decay": 9.5e-9, "momentum": 0.9,
    "lr_sched": "linear",
    "lr_sched_args": {"start_factor": 1, "end_factor": 0.2, "total_iters": 4000},
    "loss_name": "MultiSimilarityLoss", "miner_name": "MultiSimilarityMiner",
    "miner_margin": 0.1, "faiss_gpu": False,
}
DATA_CONFIG = {
    "batch_size": 60, "img_per_place": 4, "min_img_per_place": 4,
    "shuffle_all": False, "random_sample_from_each_place": True,
    "image_size": (224, 224), "num_workers": 10, "show_data_stats": True,
    "val_set_names": ["pitts30k_val", "pitts30k_test", "msls_val"],
}
TRAINER_CONFIG = {
    "accelerator": "gpu", "devices": 1, "num_nodes": 1, "precision": "16-mixed",
    "max_epochs": 4, "num_sanity_val_steps": 0, "check_val_every_n_epoch": 1,
    "reload_dataloaders_every_n_epochs": 1, "log_every_n_steps": 20,
}
TREATMENT_DEFINITION = {
    "slice": "core_q99_geo500", "same_city_only": True, "cities": ["Boston", "London"],
    "binary_edge_use": True, "edge_weighting": False, "hub_weighting": False,
    "loss_modification": False, "vertex_disjoint_guided_matching": True,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def guard_step3a(path, root=ROOT):
    root, path = Path(root).resolve(), Path(path).resolve()
    allowed = [root / "cache/countermine_rgb/step3a", root / "outputs/step3a"]
    if any(path.is_relative_to(base) for base in allowed):
        return path
    if path.parent == root / "docs/audits" and path.name.startswith("step3a_"):
        return path
    raise ValueError("Step 3A destination must remain inside its project-owned output namespace")


def write_json(path, payload, *, root=ROOT):
    destination = guard_step3a(path, root)
    # Also protects runtime provenance from accidentally exporting host paths.
    from countermine.training.step3a_audit import validate_portable_json
    validate_portable_json(payload)
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".step3a_", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def frozen_configuration():
    return deepcopy({
        "model": MODEL_CONFIG, "data": DATA_CONFIG, "trainer": TRAINER_CONFIG,
        "drop_last": False, "seed": SEED, "seed_workers": True,
        "augmentation": {
            "source": "salad/dataloaders/GSVCitiesDataloader.py",
            "train": ["Resize(224,224; BILINEAR)", "RandAugment(num_ops=3; BILINEAR)",
                      "ToTensor", "Normalize(ImageNet mean/std)"],
            "validation": ["Resize(224,224; BILINEAR)", "ToTensor", "Normalize(ImageNet mean/std)"],
        }, "batch_dimensions": [60, 4, 3, 224, 224], "epochs": 4,
    })


def code_hashes(root=ROOT):
    root = Path(root)
    files = sorted((root / "countermine/training").glob("*.py"))
    files += [root / "tools" / name for name in (
        "30_prepare_step3a_training.py", "31_train_step3a.py",
        "32_evaluate_step3a.py", "33_compare_step3a.py",
    )]
    return {str(path.relative_to(root)): sha256_file(path) for path in files}


def salad_identity(root=ROOT):
    root = Path(root)
    directory = root / "salad"
    commit = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise ValueError("SALAD submodule has modified tracked files; Step 3A requires untouched upstream")
    files = [directory / "main.py", directory / "vpr_model.py"]
    files += sorted(path for folder in ("dataloaders", "models", "utils")
                    for path in (directory / folder).rglob("*.py"))
    return {"salad_submodule_commit": commit,
            "salad_source_hashes": {str(p.relative_to(root)): sha256_file(p) for p in files}}


def resolved_paths(root=ROOT):
    root = Path(root).resolve()
    return {
        "root": root, "salad": root / "salad", "dataset": root / "data/GSVCities",
        "graph": root / "cache/countermine_rgb/step2d/place_edges.csv",
        "snapshot": root / "docs/audits/step2d_full_countermine_metrics.json",
        "runtime": root / "cache/countermine_rgb/step3a",
        "shared": root / "cache/countermine_rgb/step3a/shared",
    }


def validate_dataset_paths(paths):
    """Check only the exact roots consumed by upstream's relative-path loaders."""
    root, dataset, salad = paths["root"], paths["dataset"], paths["salad"]
    required = [salad / "vpr_model.py", dataset / "Images"]
    required += [dataset / "Dataframes" / (city + ".csv") for city in TRAIN_CITIES]
    required += [root / "data/Pittsburgh/datasets" / (name + ".mat")
                 for name in ("pitts30k_val", "pitts30k_test")]
    required += [root / "data/Pittsburgh/queries_real", root / "data/mapillary/train_val"]
    required += [salad / "datasets/msls_val" / ("msls_val_" + name + ".npy")
                 for name in ("dbImages", "qImages", "qIdx", "pIdx")]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required upstream dataset paths are missing: " + ", ".join(missing))
    # Fingerprint metadata and resolved-root identity without exporting host paths.
    frames = {str(path.relative_to(root)): sha256_file(path) for path in required if path.is_file()}
    return {"dataset_root": "data/GSVCities", "root_is_symlink": dataset.is_symlink(),
            "resolved_dataset_root_sha256": hashlib.sha256(str(dataset.resolve()).encode()).hexdigest(),
            "metadata_sha256": frames}


def enter_salad(paths):
    """All launcher paths must already be absolute before entering this context."""
    os.chdir(paths["salad"])
    sys.path.insert(0, str(paths["salad"]))
    from dataloaders.GSVCitiesDataloader import TRAIN_CITIES as upstream_cities
    if tuple(upstream_cities) != TRAIN_CITIES:
        raise ValueError("Upstream city order changed from the frozen Step 3A configuration")


def seed_runtime(seed=SEED):
    if seed != SEED:
        raise ValueError("The primary Step 3A pilot is frozen at seed 42")
    import pytorch_lightning as pl
    pl.seed_everything(seed, workers=True)


def create_model():
    from vpr_model import VPRModel
    return VPRModel(**deepcopy(MODEL_CONFIG))


def strict_load_initial_state(model, state_path, summary, *, torch_module=None):
    if sha256_file(state_path) != summary["initial_state_sha256"]:
        raise ValueError("Shared initialization SHA256 mismatch")
    if summary["model_configuration"] != MODEL_CONFIG:
        raise ValueError("Shared initialization model configuration differs")
    if torch_module is None:
        import torch as torch_module
    state = torch_module.load(state_path, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("Shared initialization strict load failed")
    return summary["initial_state_sha256"]


def check_preparation(paths):
    preparation = read_json(paths["shared"] / "preparation_summary.json")
    provenance = preparation["provenance"]
    if not preparation.get("complete") or provenance["seed"] != SEED:
        raise ValueError("Stage 30 preparation must be complete for seed 42")
    if provenance["step3a_code_hashes"] != code_hashes(paths["root"]):
        raise ValueError("Step 3A source hashes changed since shared preparation")
    if provenance["step2d_snapshot_sha256"] != sha256_file(paths["snapshot"]):
        raise ValueError("Frozen Step 2D snapshot changed")
    if provenance["step2d_place_edges_sha256"] != sha256_file(paths["graph"]):
        raise ValueError("Frozen Step 2D graph changed")
    identity = salad_identity(paths["root"])
    if any(provenance[key] != value for key, value in identity.items()):
        raise ValueError("SALAD source identity changed since shared preparation")
    if provenance["initial_state_sha256"] != sha256_file(paths["shared"] / "initial_state.pt"):
        raise ValueError("Shared initialization changed since preparation")
    return preparation

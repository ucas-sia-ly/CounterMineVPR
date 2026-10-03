"""Read-only workstation checks for the frozen Step 3A experiment.

Importing this module does not import torch or construct an upstream model.
Package metadata and runtime observations are descriptive provenance, never
machine identity constraints. The optional report is an engineering artifact.
"""
from datetime import datetime, timezone
import importlib
from importlib import metadata
from pathlib import Path
import platform
import re
import shutil
import subprocess
import os

from .step3a_config import (
    ROOT, TRAIN_CITIES, resolved_paths, salad_identity, validate_dataset_paths,
    write_json,
)


STAGES = ("prepare", "train", "evaluate", "export", "all")
MIN_TRAIN_FREE_BYTES = 5 * 1024 ** 3
EXPECTED_ELIGIBLE_EDGES = {"Boston": 1083, "London": 1003}
PACKAGE_DISTRIBUTIONS = {
    "torch": ("torch",), "torchvision": ("torchvision",),
    "pytorch_lightning": ("pytorch-lightning",),
    "pytorch_metric_learning": ("pytorch-metric-learning",),
    "numpy": ("numpy",), "pandas": ("pandas",),
    "faiss": ("faiss-gpu", "faiss-cpu", "faiss"),
    "xformers": ("xformers",), "matplotlib": ("matplotlib",),
}

# FAISS is a lazy import in frozen SALAD's validation function. Preparation
# does not invoke validation; training and evaluation must import its bridge.
PREPARE_IMPORTS = (
    "torch", "torchvision", "pytorch_lightning", "pytorch_metric_learning",
    "numpy", "pandas", "xformers", "xformers.ops",
    "torchvision.transforms", "pytorch_metric_learning.losses",
    "pytorch_metric_learning.miners", "pytorch_metric_learning.distances",
    "scipy.io", "sklearn.neighbors", "PIL.Image", "prettytable",
)
VALIDATION_IMPORTS = ("faiss", "faiss.contrib.torch_utils")
EXPORT_IMPORTS = ("numpy", "matplotlib", "matplotlib.pyplot")


class RuntimePreflightError(RuntimeError):
    """A failed check, with an incomplete portable report for inspection."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def configure_cublas_workspace():
    """Configure deterministic CuBLAS before any runtime import/CUDA probe."""
    name = "CUBLAS_WORKSPACE_CONFIG"
    value = os.environ.setdefault(name, ":4096:8")
    if value not in (":4096:8", ":16:8"):
        raise ValueError(
            "Invalid CUBLAS_WORKSPACE_CONFIG. Start a fresh process with "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8); "
            "deterministic training remains enabled."
        )
    return value


def _package_versions():
    versions = {"python": platform.python_version()}
    for package, distributions in PACKAGE_DISTRIBUTIONS.items():
        versions[package] = None
        for distribution in distributions:
            try:
                versions[package] = metadata.version(distribution)
                break
            except metadata.PackageNotFoundError:
                continue
    return versions


def _gpu_metadata(torch_module):
    observed = {"cuda_available": False, "device_count": 0, "gpu_name": None,
                "gpu_capability": None, "total_vram_bytes": None,
                "cuda_version": None, "cudnn_version": None}
    if torch_module is None:
        return observed
    observed["cuda_version"] = torch_module.version.cuda
    observed["cudnn_version"] = torch_module.backends.cudnn.version()
    observed["cuda_available"] = bool(torch_module.cuda.is_available())
    observed["device_count"] = int(torch_module.cuda.device_count())
    if observed["cuda_available"] and observed["device_count"]:
        properties = torch_module.cuda.get_device_properties(0)
        observed["gpu_name"] = str(properties.name)
        observed["gpu_capability"] = list(torch_module.cuda.get_device_capability(0))
        observed["total_vram_bytes"] = int(properties.total_memory)
    return observed


def collect_runtime_metadata(*, torch_module=None):
    """Collect the compact scientific runtime block without implicit imports.

    Supply the stage's already-imported torch module to include CUDA observations.
    With no argument, CUDA fields remain unknown. This keeps CPU export and CLI
    help independent of model runtime imports and CUDA initialization.
"""
    versions = _package_versions()
    gpu = _gpu_metadata(torch_module)
    return _runtime_block(versions, gpu)


def _runtime_block(versions, gpu):
    return {
        "python_version": versions["python"], "torch_version": versions["torch"],
        "torchvision_version": versions["torchvision"],
        "lightning_version": versions["pytorch_lightning"],
        "pytorch_metric_learning_version": versions["pytorch_metric_learning"],
        "faiss_version": versions["faiss"], "xformers_version": versions["xformers"],
        "numpy_version": versions["numpy"], "pandas_version": versions["pandas"],
        **{name: gpu[name] for name in ("cuda_version", "cudnn_version", "gpu_name", "gpu_capability")},
    }


def _portable_error(error, root):
    message = str(error).replace(str(root), "<repository>")
    # Third-party import errors may contain a site-package or build directory.
    # Keep the exception reason, but never serialize host paths in the report.
    message = re.sub(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/)[^\s,'\";)]+", "<path>", message)
    return f"{type(error).__name__}: {message}"


def _validate_graph(paths):
    from .countermine_batch_sampler import load_edge_bundle
    bundle = load_edge_bundle(paths["graph"], paths["snapshot"], expected_graph_places=2000)
    counts = {city: len(bundle.eligible_edges(city)) for city in EXPECTED_ELIGIBLE_EDGES}
    if counts != EXPECTED_ELIGIBLE_EDGES:
        raise ValueError("Frozen same-city q99_geo500 edge counts differ: "
                         f"expected {EXPECTED_ELIGIBLE_EDGES}, found {counts}. "
                         "Inspect frozen Step 2D inputs; do not rewrite them.")
    return {"sha256": bundle.graph_sha256, "snapshot_sha256": bundle.snapshot_sha256,
            "place_count": len(bundle.graph_place_uids), "eligible_same_city_edges": counts}


def _validate_salad(root):
    identity = salad_identity(root)
    dirty = subprocess.check_output([
        "git", "-C", str(root / "salad"), "status", "--porcelain", "--untracked-files=all",
    ], text=True)
    if dirty:
        raise ValueError("SALAD submodule is not clean; inspect local changes and restore "
                         "the frozen upstream source before Step 3A.")
    return identity


def _dataset_status(paths, dataset_metadata):
    metadata_files = dataset_metadata["metadata_sha256"]
    return {
        "training_city_csv_count": sum(
            f"data/GSVCities/Dataframes/{city}.csv" in metadata_files for city in TRAIN_CITIES),
        "required_training_city_csv_count": len(TRAIN_CITIES),
        "metadata_file_count": len(metadata_files),
        "availability": {name: (paths["root"] / name).exists() for name in (
            "data/GSVCities/Images", "data/Pittsburgh/datasets",
            "data/Pittsburgh/queries_real", "data/mapillary/train_val", "salad/datasets/msls_val",
        )},
    }


def _required_imports(stage):
    if stage == "export":
        return EXPORT_IMPORTS
    modules = PREPARE_IMPORTS
    if stage in ("train", "evaluate", "all"):
        modules += VALIDATION_IMPORTS
    if stage == "all":
        modules += EXPORT_IMPORTS
    return tuple(dict.fromkeys(modules))


def _check_cuda(torch_module):
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable to PyTorch. Use the CUDA workstation and "
                           "verify its NVIDIA driver and the pinned PyTorch CUDA installation.")
    x = torch_module.ones((2, 2), device="cuda")
    try:
        y = x @ x
        torch_module.cuda.synchronize()
        if not bool(torch_module.isfinite(y).all().item()):
            raise RuntimeError("Tiny CUDA matrix multiplication produced nonfinite output")
    finally:
        del x


def _check_disk(runtime):
    # Do not create directories or delete caches during this read-only check.
    directory = runtime
    while not directory.exists():
        directory = directory.parent
    free = shutil.disk_usage(directory).free
    if free < MIN_TRAIN_FREE_BYTES:
        raise RuntimeError(f"Insufficient checkpoint disk space: {free / 1024 ** 3:.2f} GiB free; "
                           "at least 5 GiB is required before training. Make space manually; "
                           "preflight never deletes caches.")
    return {"directory": "cache/countermine_rgb/step3a", "free_bytes": int(free),
            "minimum_free_bytes": MIN_TRAIN_FREE_BYTES}


def run_preflight(stage="prepare", *, root=ROOT, save_report=False, progress=False):
    """Validate the exact frozen inputs and imports, without scientific writes.

Preparation is permitted on CPU. Training/evaluation perform only a tiny CUDA
operation; training additionally requires a conservative 5 GiB free disk space.
Only successful checks may publish the optional engineering report.
"""
    if stage not in STAGES:
        raise ValueError(f"Unknown Step 3A preflight stage: {stage!r}")
    root = Path(root).resolve()
    paths = resolved_paths(root)
    report = {"complete": False, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "stage": stage, "checks": {}, "package_versions": _package_versions()}
    failures, causes = [], []

    def check(name, function, action):
        try:
            value = function()
        except Exception as error:
            reason = _portable_error(error, root)
            report["checks"][name] = {"passed": False, "error": reason, "action": action}
            failures.append(f"{name}: {reason} {action}")
            causes.append(error)
            return None
        report["checks"][name] = {"passed": True}
        return value

    if stage != "export":
        report["cublas_workspace_config"] = check(
            "cublas_workspace", configure_cublas_workspace,
            "Export CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting a fresh Python process.")
        if report["cublas_workspace_config"] is None:
            raise RuntimePreflightError("Step 3A runtime preflight failed:\n" + "\n".join(failures), report) from causes[0]

    if progress:
        print("[1/6] validating frozen Step 2D inputs", flush=True)
    report["graph"] = check("frozen_step2d", lambda: _validate_graph(paths),
                            "Inspect the original complete Step 2D graph and snapshot.")
    if progress:
        print("[2/6] validating dataset roots", flush=True)
    report["dataset_metadata"] = check("dataset_roots", lambda: validate_dataset_paths(paths),
                                       "Install the scientific datasets at the exact documented relative roots.")
    if report["dataset_metadata"] is not None:
        report["dataset_status"] = _dataset_status(paths, report["dataset_metadata"])
    if progress:
        print("[3/6] validating SALAD source", flush=True)
    report["salad_identity"] = check("salad_source", lambda: _validate_salad(root),
                                     "Use the clean frozen SALAD submodule; do not change model code.")

    imported = {}
    for name in _required_imports(stage):
        package = name.split(".")[0]
        imported[name] = check("import:" + name, lambda name=name: importlib.import_module(name),
                               f"Install or repair {package} in countermine-vpr using environment_vpr.yml "
                               "and the documented SALAD-compatible package sources.")
    torch_module = imported.get("torch")
    # Export does not import torch or query CUDA, even in a long-lived process.
    if stage == "export":
        report["gpu"] = _gpu_metadata(None)
        report["runtime"] = _runtime_block(report["package_versions"], report["gpu"])
    elif stage == "prepare":
        # CPU preparation does not depend on a working CUDA driver. Record an
        # observation failure without turning it into a GPU requirement.
        try:
            gpu = _gpu_metadata(torch_module)
            report["checks"]["runtime_metadata"] = {"passed": True}
        except Exception as error:
            gpu = _gpu_metadata(None)
            gpu.update(cuda_available=None, device_count=None)
            if torch_module is not None:
                gpu["cuda_version"] = torch_module.version.cuda
            report["checks"]["runtime_metadata"] = {
                "passed": True, "cuda_observation_available": False,
                "diagnostic": _portable_error(error, root),
            }
        report["gpu"] = gpu
        report["runtime"] = _runtime_block(report["package_versions"], gpu)
    else:
        gpu = check("runtime_metadata", lambda: _gpu_metadata(torch_module),
                    "Inspect the pinned PyTorch CUDA installation and NVIDIA driver.")
        report["gpu"] = gpu
        report["runtime"] = _runtime_block(report["package_versions"], gpu) if gpu is not None else None
    if stage in ("train", "evaluate", "all"):
        if torch_module is None:
            report["checks"]["cuda_smoke"] = {"passed": False, "error": "torch import failed"}
        else:
            check("cuda_smoke", lambda: _check_cuda(torch_module),
                  "CUDA is mandatory for smoke/full training and scientific evaluation.")
    if stage in ("train", "all"):
        report["disk"] = check("checkpoint_disk", lambda: _check_disk(paths["runtime"]),
                               "Training requires at least 5 GiB free; make space manually without changing frozen inputs.")
    if failures:
        raise RuntimePreflightError("Step 3A runtime preflight failed:\n" + "\n".join(failures), report) from causes[0]
    report["complete"] = True
    if save_report:
        write_json(paths["runtime"] / "runtime_preflight.json", report, root=root)
    return report

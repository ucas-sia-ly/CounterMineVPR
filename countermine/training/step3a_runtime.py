"""Engineering callbacks and runtime-independent Step 3A identity checks.

Importing this module does not import torch or any model runtime.
"""
from copy import deepcopy
import math
from pathlib import Path
from collections.abc import Mapping

from countermine.training.step3a_config import (
    DATA_CONFIG, TRAINER_CONFIG, MODES, SEED, read_json, write_json, sha256_file,
)

RECALL_KEYS = tuple(f"{name}/R{k}" for name in DATA_CONFIG["val_set_names"] for k in (1, 5, 10))
RUNTIME_FIELDS = (
    "python_version", "torch_version", "torchvision_version", "lightning_version",
    "pytorch_metric_learning_version", "faiss_version", "xformers_version", "numpy_version",
    "pandas_version", "cuda_version", "cudnn_version", "gpu_name", "gpu_capability",
)


def critical_provenance(provenance):
    """Retain every scientific invariant, excluding only observed runtime metadata.

    Unknown present and future provenance fields remain identity-critical.
    Runtime versions and GPU names are recorded evidence, not a machine lock.
    """
    if not isinstance(provenance, Mapping):
        raise ValueError("Step 3A provenance must be a mapping")
    return deepcopy({key: value for key, value in provenance.items() if key != "runtime"})


def runtime_differences(reference, observed):
    """Describe changed runtime fields without accepting/rejecting a condition."""
    if not isinstance(reference, Mapping) or not isinstance(observed, Mapping):
        raise ValueError("Step 3A runtime metadata must be a mapping")
    return {
        key: {"reference": deepcopy(reference.get(key)), "observed": deepcopy(observed.get(key))}
        for key in sorted(set(reference) | set(observed))
        if key not in reference or key not in observed or reference[key] != observed[key]
    }


def finite_metrics(values):
    result = {}
    for key, value in values.items():
        scalar = float(value.detach().cpu().item() if hasattr(value, "detach") else value)
        if not math.isfinite(scalar):
            raise ValueError(f"Nonfinite Step 3A metric: {key}")
        result[key] = scalar
    return result


def smoke_summary_paths(paths):
    return [paths["runtime"] / "smoke" / mode / "seed42/training_summary.json" for mode in MODES]


def require_smokes(paths, provenance):
    for mode, path in zip(MODES, smoke_summary_paths(paths)):
        summary = read_json(path)
        if not summary.get("complete") or not summary.get("smoke") or summary["mode"] != mode:
            raise ValueError("Both independent smoke conditions must pass before full training")
        if critical_provenance(summary["provenance"]) != critical_provenance(provenance):
            raise ValueError("Smoke provenance differs from current shared initialization/source")
        checks = summary["engineering_checks"]
        for name in ("shared_initialization_strict_load", "batch_shape_and_labels", "finite_loss",
                     "finite_gradients", "optimizer_step", "validation", "sampler_invariants"):
            if checks.get(name) is not True:
                raise ValueError(f"Smoke check did not pass: {name}")


def create_training_audit_callback(*, run_dir, mode, smoke, provenance, root):
    import pytorch_lightning as pl
    import torch

    class TrainingAudit(pl.Callback):
        def __init__(self):
            self.epochs = []
            self.losses = []
            self.accuracies = []
            self.epoch_accuracy_values = []
            self.pending_step_parameter = None
            self.optimizer_updates = 0
            self.backward_checks = 0
            self.finite_gradient_steps = 0
            self.amp_overflow_steps = 0
            self.batch_checks = 0

        def on_train_epoch_start(self, trainer, module):
            self.losses = []
            self.epoch_accuracy_values = []
            self.epoch_recall = {}

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            places, labels = batch
            if places.ndim != 5 or tuple(places.shape[1:]) != (4, 3, 224, 224):
                raise ValueError("Training batch must retain [BS,4,3,224,224] shape")
            if tuple(labels.shape) != tuple(places.shape[:2]):
                raise ValueError("Training labels must retain [BS,4] shape")
            if not torch.equal(labels, labels[:, :1].expand_as(labels)):
                raise ValueError("Each place must contain four repeated upstream labels")
            if len(torch.unique(labels[:, 0])) != places.shape[0]:
                raise ValueError("Different dataset places have colliding training labels")
            indices = trainer.datamodule.step3a_sampler.plan.batches[batch_idx]
            expected = [int(trainer.datamodule.train_dataset.places_ids[index]) for index in indices]
            if labels[:, 0].detach().cpu().tolist() != expected:
                raise ValueError("Labels disagree with the frozen batch plan")
            if self.pending_step_parameter is not None:
                parameter, before = self.pending_step_parameter
                if torch.equal(parameter.detach(), before):
                    raise ValueError("Smoke optimizer step did not update the observed model parameter")
                self.optimizer_updates += 1
                self.pending_step_parameter = None
            self.batch_checks += 1

        def on_after_backward(self, trainer, module):
            if not any(parameter.grad is not None for parameter in module.parameters()):
                raise ValueError("Backward produced no gradients")
            self.backward_checks += 1

        def on_before_optimizer_step(self, trainer, module, optimizer):
            if smoke:
                # AMP unscales before this hook. Let upstream GradScaler handle
                # overflow/skip; requiring all scaled gradients finite would
                # change the original mixed-precision training behavior.
                gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
                if any(not torch.isfinite(gradient).all() for gradient in gradients):
                    self.amp_overflow_steps += 1
                    return
                self.finite_gradient_steps += 1
                for parameter in module.parameters():
                    if parameter.grad is not None and torch.count_nonzero(parameter.grad):
                        self.pending_step_parameter = (parameter, parameter.detach().clone())
                        break
                if self.pending_step_parameter is None:
                    raise ValueError("Smoke step requires a nonzero gradient")

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = float(outputs["loss"].detach().cpu())
            if not math.isfinite(loss):
                raise ValueError("Step 3A smoke/training loss is NaN or Inf")
            self.losses.append(loss)
            accuracy = float(module.batch_acc[-1])
            self.epoch_accuracy_values.append(accuracy)
            self.accuracies.append(accuracy)
            if self.pending_step_parameter is not None:
                parameter, before = self.pending_step_parameter
                if torch.equal(parameter.detach(), before):
                    raise ValueError("Smoke optimizer step did not update model weights")
                self.optimizer_updates += 1
                self.pending_step_parameter = None

        def on_validation_end(self, trainer, module):
            self.epoch_recall = finite_metrics({key: trainer.callback_metrics[key] for key in RECALL_KEYS})

        def on_train_epoch_end(self, trainer, module):
            if not self.losses or len(self.epoch_recall) != len(RECALL_KEYS):
                raise ValueError("Each epoch must complete training and all upstream validation sets")
            if not smoke and len(self.losses) != len(trainer.datamodule.step3a_sampler):
                raise ValueError("Full epoch did not consume every frozen place batch exactly once")
            epoch = int(trainer.current_epoch)
            metrics = {"loss": sum(self.losses) / len(self.losses),
                       "b_acc": sum(self.epoch_accuracy_values) / len(self.epoch_accuracy_values),
                       **self.epoch_recall}
            self.epochs.append({"epoch": epoch, "metrics": finite_metrics(metrics),
                                "train_batch_count": len(self.losses)})
            write_json(Path(run_dir) / "epoch_metrics.json", {"epochs": self.epochs}, root=root)

    return TrainingAudit()


def training_summary(*, trainer, callback, checkpoint, run_dir, mode, smoke, provenance, root):
    epochs = callback.epochs
    best = max(epochs, key=lambda item: item["metrics"]["pitts30k_val/R1"])
    best_path = Path(checkpoint.best_model_path)
    if not best_path.is_file() or not Path(checkpoint.last_model_path).is_file():
        raise ValueError("Both best-by-Pitts-validation and last checkpoints are required")
    summary = {
        "complete": True, "smoke": smoke, "mode": mode, "seed": SEED,
        "provenance": provenance, "epochs": epochs, "best_epoch": best["epoch"],
        "best_checkpoint": str(best_path.relative_to(run_dir)),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(Path(checkpoint.last_model_path).relative_to(run_dir)),
        "last_checkpoint_sha256": sha256_file(checkpoint.last_model_path),
        "checkpoint_selection": "best_pitts30k_val_R1",
        "final_train_loss": epochs[-1]["metrics"]["loss"],
        "average_b_acc": sum(callback.accuracies) / len(callback.accuracies),
        "engineering_checks": {
            "shared_initialization_strict_load": True, "batch_shape_and_labels": callback.batch_checks > 0,
            "finite_loss": True, "finite_gradients": callback.finite_gradient_steps > 0 if smoke else None,
            "optimizer_step": callback.optimizer_updates > 0 if smoke else trainer.global_step > 0,
            "validation": True, "sampler_invariants": True,
        },
        "smoke_validation_subsets": trainer.datamodule.smoke_validation_audit,
        "completed_optimizer_steps": int(trainer.global_step),
        "smoke_amp_overflow_steps": callback.amp_overflow_steps,
    }
    if smoke and not all(summary["engineering_checks"].values()):
        raise ValueError("Smoke requires a finite successful optimizer update and all engineering checks")
    if not smoke and [item["epoch"] for item in epochs] != list(range(TRAINER_CONFIG["max_epochs"])):
        raise ValueError("Scientific Step 3A runs require all four complete epochs")
    write_json(Path(run_dir) / "training_summary.json", summary, root=root)
    return summary

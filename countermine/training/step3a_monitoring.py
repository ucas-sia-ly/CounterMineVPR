"""Scalar-only Step 3A monitoring, independent of scientific exports.

Logger and callback runtimes are imported only when training requests them.
The sampler summary is the sole source of structural exposure diagnostics.
"""

from pathlib import Path


def monitoring_configuration(*, mode, seed, smoke=False, wandb=False,
                             wandb_project="CounterMineVPR", wandb_entity=None,
                             wandb_offline=False):
    configuration = {
        "csv": True, "tensorboard": True, "wandb": wandb,
        "wandb_offline": bool(wandb and wandb_offline),
    }
    if wandb:
        configuration.update({
            "project": wandb_project,
            "run_name": f"step3a-{'smoke-' if smoke else ''}{mode.replace('_', '-')}-seed{seed}",
            "tags": ["step3a", f"seed{seed}", *mode.split("_", 1)],
        })
        if smoke:
            configuration["tags"].append("smoke")
        if wandb_entity is not None:
            configuration["entity"] = wandb_entity
    return configuration


def create_training_loggers(run_dir, configuration):
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

    run_dir = Path(run_dir)
    loggers = [
        CSVLogger(str(run_dir), name="logs", version=0),
        TensorBoardLogger(str(run_dir), name="tensorboard", version="",
                          log_graph=False, default_hp_metric=False, flush_secs=20),
    ]
    if configuration["wandb"]:
        try:
            from pytorch_lightning.loggers import WandbLogger

            loggers.append(WandbLogger(
                save_dir=str(run_dir), name=configuration["run_name"],
                project=configuration["project"], entity=configuration.get("entity"),
                offline=configuration["wandb_offline"], tags=configuration["tags"],
                log_model=False,
            ))
        except ImportError as error:
            raise RuntimeError(
                "--wandb requires Weights & Biases; install it with "
                "`python -m pip install wandb` in the training environment, "
                "or omit --wandb."
            ) from error
    return loggers


def create_monitoring_callbacks():
    from pytorch_lightning import Callback
    from pytorch_lightning.callbacks import LearningRateMonitor

    class Step3AMonitoringCallback(Callback):
        """Publish each frozen full-epoch plan once, including smoke plans.

        Direct logger calls keep these diagnostics out of model reductions,
        checkpoint monitoring and the existing scientific JSON metrics.
        """

        def __init__(self):
            self.logged_epochs = set()

        def on_train_epoch_start(self, trainer, module):
            epoch = int(trainer.current_epoch)
            if not trainer.is_global_zero or epoch in self.logged_epochs:
                return
            summary = trainer.datamodule.step3a_sampler.plan.summary
            metrics = {
                "countermine/q95_cobatched_edges": summary["exposure"]["core_q95"],
                "countermine/q99_cobatched_edges": summary["exposure"]["core_q99"],
                "countermine/q99_geo500_cobatched_edges": summary["exposure"]["core_q99_geo500"],
                "countermine/guided_pairs": summary["greedy_matched_pair_count"],
                "countermine/unique_guided_places": summary["unique_guided_place_count"],
                "countermine/q99_geo500_exposure_gain": summary["structural_exposure_gain"],
            }
            if summary["baseline_exposure"]["core_q99_geo500"] != 0:
                metrics["countermine/q99_geo500_exposure_ratio"] = summary["structural_exposure_ratio"]
            for logger in trainer.loggers:
                logger.log_metrics({**metrics, "epoch": epoch}, step=trainer.global_step)
                logger.save()
            self.logged_epochs.add(epoch)

    return [LearningRateMonitor(logging_interval="step"), Step3AMonitoringCallback()]

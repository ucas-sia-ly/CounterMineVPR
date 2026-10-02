"""Project-owned place batching with untouched upstream transforms and model.

Model dependencies are imported only by the launcher after entering salad/.
"""
from copy import deepcopy
from pathlib import Path

from countermine.training.step3a_config import DATA_CONFIG, read_json
from countermine.training.gsv_place_mapping import build_place_mapping
from countermine.training.countermine_batch_sampler import CounterMineBatchSampler


def verify_validation_images(datamodule):
    for name, dataset in zip(datamodule.val_set_names, datamodule.val_datasets):
        for image in dataset.images:
            path = Path(image) if "pitts" in name else Path("../data/mapillary") / str(image)
            if not path.is_file():
                raise FileNotFoundError(f"Required {name} validation image is missing: {path}")


def smoke_validation_subset(dataset, name, *, max_images=120, reference_count=100):
    """Keep real images and remap positives for a two-batch engineering check.

    Upstream recall requests k=100. A simple prefix would have no queries.
    All selected query positives are retained; scientific runs never use this.
    """
    import numpy as np
    from torch.utils.data import Subset

    if "pitts" in name:
        number_refs = int(dataset.dbStruct.numDb)
        positives = dataset.getPositives()
    else:
        number_refs = int(dataset.num_references)
        positives = dataset.pIdx
    if number_refs < reference_count:
        raise ValueError("Smoke validation needs at least 100 upstream reference images")
    query_ids, required_refs = [], set()
    for query_id, query_positive in enumerate(positives):
        ids = set(int(index) for index in query_positive)
        if not ids or any(index < 0 or index >= number_refs for index in ids):
            continue
        if len(ids | required_refs) <= reference_count:
            query_ids.append(query_id)
            required_refs.update(ids)
        if len(query_ids) == max_images - reference_count:
            break
    if not query_ids:
        raise ValueError("Cannot build smoke reference/query subset without dropping query positives")
    references = sorted(required_refs)
    for index in range(number_refs):
        if len(references) == reference_count:
            break
        if index not in required_refs:
            references.append(index)
    references.sort()
    remap = {old: new for new, old in enumerate(references)}
    remapped = [np.asarray([remap[int(index)] for index in positives[query]], dtype=np.int64)
                for query in query_ids]
    indices = references + [number_refs + query for query in query_ids]

    class RecallSubset(Subset):
        def getPositives(self):
            return self.positives

    subset = RecallSubset(dataset, indices)
    subset.images = [dataset.images[index] for index in indices]
    subset.positives = remapped
    if "pitts" in name:
        subset.dbStruct = dataset.dbStruct._replace(numDb=reference_count, numQ=len(query_ids))
    else:
        subset.num_references = reference_count
        subset.pIdx = remapped
    return subset, {"validation_set": name, "reference_count": reference_count,
                    "query_count": len(query_ids), "real_image_indices": indices,
                    "all_selected_query_positives_retained": True}


def build_datamodule(*, mode, edge_bundle, plan_dir, seed=42, smoke=False,
                     expected_mapping=None, baseline_plan_dir=None,
                     expected_epoch0_place_order=None):
    from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
    from torch.utils.data import DataLoader

    class Step3ADataModule(GSVCitiesDataModule):
        def setup(self, stage):
            super().setup(stage)
            if stage == "fit":
                verify_validation_images(self)
                self.smoke_validation_audit = []
                if smoke:
                    converted = [smoke_validation_subset(dataset, name)
                                 for name, dataset in zip(self.val_set_names, self.val_datasets)]
                    self.val_datasets = [item[0] for item in converted]
                    self.smoke_validation_audit = [item[1] for item in converted]

        def train_dataloader(self):
            # Match upstream's reload, including setup's initial reload.
            self.reload()
            records, self.mapping_audit = build_place_mapping(self.train_dataset, edge_bundle.graph_place_uids)
            epoch = int(self.trainer.current_epoch) if self.trainer else 0
            self.step3a_sampler = CounterMineBatchSampler(
                records, edge_bundle, mode=mode, batch_size=self.batch_size,
                seed=seed, epoch=epoch, plan_dir=plan_dir,
            )
            if expected_mapping is not None and self.mapping_audit != expected_mapping:
                raise ValueError("Dataset place mapping changed since shared preparation")
            if (epoch == 0 and expected_epoch0_place_order is not None and
                    self.step3a_sampler.plan.summary["dataset_place_order_sha256"] != expected_epoch0_place_order):
                raise ValueError("Initial dataset place ordering differs from the prepared epoch-zero plan")
            if baseline_plan_dir is not None:
                reference = read_json(Path(baseline_plan_dir) / f"epoch_{epoch:02d}_summary.json")
                current = self.step3a_sampler.plan.summary
                for key in ("baseline_sequence_sha256", "dataset_place_order_sha256", "dataset_place_count", "batch_count",
                            "batch_sizes", "marginal_exposure"):
                    if current[key] != reference[key]:
                        raise ValueError(f"Baseline/treatment epoch place organization differs: {key}")
            config = dict(self.train_loader_config)
            for key in ("batch_size", "shuffle", "drop_last"):
                config.pop(key)
            return DataLoader(dataset=self.train_dataset,
                              batch_sampler=self.step3a_sampler, **config)

    return Step3ADataModule(**deepcopy(DATA_CONFIG))

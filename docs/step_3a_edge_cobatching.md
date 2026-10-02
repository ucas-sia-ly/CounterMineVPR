# Step 3A — CounterMine edge-aware place co-batching pilot

## Status and question

The project-owned training integration is implemented. The current workspace lacks the authoritative GSVCities Dataframes/images and required Pitts30k/MSLS validation data. CUDA is unavailable in this session, and `countermine-vpr` lacks the upstream `faiss` dependency. Stage 30 stops on the missing dataset paths before model creation. No SALAD smoke check, four-epoch training run, evaluation or Step 3A scientific snapshot has been produced. Scientific results require two completed four-epoch runs and their best-checkpoint evaluations; the comparison exporter fails without those artifacts and does not create a placeholder scientific snapshot.

The single-seed Step 3A pilot asks whether increased co-exposure to structurally confusing different-place pairs changes standard SALAD retrieval when architecture, loss, online miner and marginal place frequency are held fixed. This is a batch-composition experiment. No statistical significance, training objective, broader ablation or recognition improvement is asserted before measurement.

[Step 2D is completed](step_2d_full_countermine.md): the frozen population contains 260,502 measured canonical pairs and reproduces the 5,000-pair Step 2A pilot. All Step 2A–2D caches, scientific snapshots and figures remain immutable. The original Step 2D `git_commit_at_export` records the producer commit before the later result-containing commit; its code/input hashes pin source identity.

## Frozen intervention and SALAD settings

The primary treatment is `countermine_q99_geo500`. It intentionally co-batches only Boston–Boston and London–London place pairs with positive `core_q99_geo500` support in the frozen Step 2D place graph. Cross-city edges are excluded. Edges form a binary relation and are shuffled uniformly using explicit NumPy PCG64 / SeedSequence inputs for seed 42, epoch and city. A greedy vertex-disjoint matching allows each place to participate in at most one intentionally guided pair per epoch. Scores, support counts and hub degrees never become sampling or loss weights.

The baseline is the instantiated dataset’s exact sequential place-index sequence, grouped into 60-place batches with `drop_last=False`. Treatment rearranges Boston/London places within their original city slots. Every batch preserves its baseline number of places from every city; all other cities’ slots remain unchanged. Every place appears exactly once per full epoch in both conditions. One place exposure means one dataset place index, not its four sampled images.

Both conditions use the same project-owned DataModule wrapper and untouched upstream `VPRModel`. Training batches retain `[BS, 4, 3, 224, 224]` images and the repeated labels expected by SALAD. The existing online miner decides which pairs contribute to the original loss. Guided edges are not forced into miner outputs.

| Setting | Frozen value |
| --- | --- |
| Places / images per place / minimum images | 60 / 4 / 4 |
| Dataset ordering / view sampling | `shuffle_all=False` / random sampled views |
| Training transforms | Upstream Resize, RandAugment, ToTensor, Normalize |
| Backbone | DINOv2 ViT-B/14; 4 trainable blocks; token and norm enabled |
| Aggregator | SALAD: 768 channels, 64 clusters, 128 cluster dimension, 256 token dimension |
| Optimizer | AdamW, learning rate `6e-5`, weight decay `9.5e-9`, momentum `0.9` |
| LR schedule | Linear; start 1, end 0.2, 4,000 iterations |
| Loss / miner | MultiSimilarityLoss / MultiSimilarityMiner, margin 0.1 |
| Precision / devices / epochs | `16-mixed` / one GPU / four |
| Validation | Pitts30k val, Pitts30k test, MSLS val, each epoch |
| Checkpoint selection | Pitts30k val R@1 only |
| Seed | 42, with `pl.seed_everything(42, workers=True)` |

No SALAD submodule file is modified. The launcher resolves repository/cache paths before entering `salad/`, preserving upstream `../data/GSVCities/` behavior. It requires the exact GSVCities Dataframes and validation data consumed by upstream; it fails on missing data rather than selecting another copy. Dataset metadata supplies a one-to-one canonical place mapping; all 2,000 graph places must map.

## Run in order

Use a SALAD training environment with its upstream dependencies and the exact repository datasets available. Commands below start from the repository root. Preparation validates the frozen graph SHA256 against the completed Step 2D snapshot, maps training places, audits epoch-zero plans and creates one shared `model.state_dict()` artifact. Both conditions strictly load that identical artifact before optimization.

```bash
# A. Shared mapping, frozen treatment and exact initialization.
conda run -n <SALAD_TRAIN_ENV> \
  python tools/30_prepare_step3a_training.py --seed 42

# B. Engineering smoke checks, in separate smoke directories.
conda run -n <SALAD_TRAIN_ENV> \
  python tools/31_train_step3a.py --mode baseline --seed 42 \
  --max-epochs 1 --limit-train-batches 3 --limit-val-batches 2 --smoke
conda run -n <SALAD_TRAIN_ENV> \
  python tools/31_train_step3a.py --mode countermine_q99_geo500 --seed 42 \
  --max-epochs 1 --limit-train-batches 3 --limit-val-batches 2 --smoke

# C. Two scientific conditions, baseline followed by treatment.
conda run -n <SALAD_TRAIN_ENV> \
  python tools/31_train_step3a.py --mode baseline --seed 42 --max-epochs 4
conda run -n <SALAD_TRAIN_ENV> \
  python tools/31_train_step3a.py --mode countermine_q99_geo500 --seed 42 --max-epochs 4

# D. Standard global retrieval, using each best Pitts30k-val checkpoint.
conda run -n <SALAD_TRAIN_ENV> \
  python tools/32_evaluate_step3a.py --mode baseline --seed 42
conda run -n <SALAD_TRAIN_ENV> \
  python tools/32_evaluate_step3a.py --mode countermine_q99_geo500 --seed 42

# E. CPU-only descriptive comparison and three separate figures.
python tools/33_compare_step3a.py --seed 42
```

Both smoke runs must pass before full training. Smoke checks require strict initialization loading, the original batch shape/labels, finite loss, backward/optimizer execution, validation and sampler invariants. To preserve upstream k=100 retrieval with only two smoke validation batches, each smoke validation set keeps 100 real reference images and at most 20 real queries, retaining and remapping every selected query positive. This subset is an engineering check. Both scientific conditions use the complete unchanged upstream validation sets and recall implementation. Smoke metrics never enter scientific comparison. Conditions never resume from or warm-start from each other.

## Artifacts and interpretation

Runtime files belong in `cache/countermine_rgb/step3a/`: shared initialization/preparation under `shared/`, full conditions under `baseline/seed42/` and `countermine_q99_geo500/seed42/`, with separate smoke directories. Per-epoch `batch_plans/epoch_00_summary.json`, guided-pair CSVs and dataset place-order CSVs record graph/order hashes, disjoint matching, duplicate/missing counts, split-pair counts, batch sizes, per-batch city composition and per-city marginal exposure hashes. Dataset place-order hashes bind the actual canonical identities, beyond the sequential numeric index range. Export reconstructs every plan from its saved dataset identities and the frozen binary graph, checks the guided-pair CSV against that deterministic reconstruction, and hashes the actual selected checkpoint bytes without loading them.

Exposure is measured after freezing each plan. Audits count total q95, q99 and q99_geo500 graph relations co-batched, alongside intentionally guided pairs, unique guided places and accidental structural co-occurrences. A guided matching count is distinct from total structural exposure. The comparison verifies exact marginal place multiset and Boston/London place-ID set equality for each corresponding epoch, and requires increased q99_geo500 exposure. A zero baseline denominator yields a JSON `null` ratio with an explicit `baseline_zero` flag.

The curated `docs/audits/step3a_edge_cobatching_metrics.json` contains frozen configuration, mapping/sampler audits, four full epoch metrics, best-checkpoint retrieval recalls, descriptive CounterMine-minus-baseline deltas, exposure ratios and provenance hashes. It declares `single_seed_pilot=true` and `statistical_significance_claimed=false`. It contains finite portable JSON, without checkpoint tensors, image pixels, descriptors or absolute paths. Checkpoints and the shared initialization remain uncommitted cache files.

Three separate figures are produced in `outputs/step3a/` and copied with the `step3a_` prefix to `docs/audits/`: `recall_curves.png` (Pitts30k val R@1 by epoch), `training_loss.png` and `structural_exposure.png`. The exporter validates complete run evidence before plotting or publishing a scientific snapshot.

Stop on a frozen-input/source/init hash mismatch, missing graph mapping, duplicate/omitted place, changed city composition, split guided pair, differing marginal place multiset, non-finite smoke loss, altered validation path or required upstream/loss/miner modification. Do not relax an invariant to repair a scientific mismatch.

Only this sampling experiment is authorized. Similarity-only mining, other thresholds, hub weighting, cross-city sampling, communities, multiple seeds, margins and structural losses remain deferred. A positive, neutral or negative measured retrieval change should be reviewed with the exposure audit before choosing later experiments.

## Implementation verification

Compilation and all **318 CPU unit tests** passed, including 75 Step 3A tests. An actual small CPU Lightning fixture verified callback ordering, two optimizer updates, three validation loaders, nine metrics and best/last checkpoint publication. It used a tiny fixture model and provides no SALAD retrieval result. The additional `countermine-vpr` test run passed 74 tests; its plotting test requires Matplotlib, which is available in the default CPU environment but absent from that environment.

The frozen graph SHA256 passes, with 2,000 graph places and 1,083 Boston / 1,003 London eligible same-city edges. A before/after inventory found no size or modification-time drift among 26,560 historical cache, curated and vendored-source files. Stage 30's missing-data stop and the scientific comparison's missing-results stop were exercised. The compact [implementation validation audit](audits/step3a_implementation_validation.json) records these checks and runtime blockers; it is separate from the unproduced scientific snapshot. No checkpoint was committed, and no commit or push was made.

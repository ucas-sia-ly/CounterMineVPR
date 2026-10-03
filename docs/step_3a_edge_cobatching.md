# Step 3A — CounterMine edge-aware place co-batching pilot

## Status and question

The project-owned training integration has an engineering runtime-hardening pass. Preparation preflight and Stage 30 completed in `countermine-vpr` on CPU, producing the verified shared initialization, mapping and epoch-zero plans. The authoritative dataset roots and required package installations are now present. CUDA remains unavailable in this Codex session, so training preflight stops clearly. Smoke training, full training and scientific evaluation require CUDA. No four-epoch scientific training run, retrieval evaluation or Step 3A scientific snapshot has been produced. Scientific results require two completed four-epoch runs and their best-checkpoint evaluations; the comparison exporter fails without those artifacts and does not create a placeholder scientific snapshot.

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
# Check the actual environment before preparation. A successful report is optional.
conda run -n countermine-vpr \
  python tools/29_check_step3a_runtime.py --stage prepare --save-report

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

Training and evaluation configure `CUBLAS_WORKSPACE_CONFIG=:4096:8` before runtime preflight imports or CUDA probes. Standalone preparation/train/evaluation preflight also configures it before importing model runtimes. This order matters because the preflight's tiny matrix multiplication initializes CuBLAS before Lightning enables deterministic algorithms; setting the variable after that operation is too late. Existing valid `:16:8` or `:4096:8` settings are preserved; invalid or empty values stop before CUDA is initialized. CPU export does not set or require this variable. Effective values are recorded in the runtime preflight report, the training run configuration and evaluation model-loading sidecar. No determinism, seed, TF32, precision, optimizer or scheduler setting is relaxed. The runtime fix is included in the exact compatibility manifest.

For explicit configuration from the shell, run `export CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching a fresh training or evaluation process. An already initialized CuBLAS handle cannot be repaired by a late assignment. Preserve any failed run's directory under the Step 3A `incomplete_attempts/` namespace before a fresh restart; the launcher never resumes or overwrites an existing condition.

## Live scalar monitoring

Every smoke and scientific condition retains CSV logs at `<run>/logs/version_0/` and adds TensorBoard events at `<run>/tensorboard/`. Use the updated `environment_vpr.yml`, or install `tensorboard` in the existing training environment, then launch the dashboard from the repository root:

```bash
tensorboard --logdir cache/countermine_rgb/step3a
```

The original training and validation metric names remain intact, including checkpoint monitor `pitts30k_val/R1`. `LearningRateMonitor(logging_interval="step")` observes the optimizer learning rate. Global logging remains every 20 training steps, and validation recall retains its upstream epoch cadence.

At each training epoch start, a project-owned callback sends only the current frozen sampler summary scalars to every logger: `countermine/q95_cobatched_edges`, `countermine/q99_cobatched_edges`, `countermine/q99_geo500_cobatched_edges`, `countermine/guided_pairs`, `countermine/unique_guided_places`, and `countermine/q99_geo500_exposure_gain`. It adds `countermine/q99_geo500_exposure_ratio` only when baseline exposure is nonzero. Baseline guided pairs and places are zero. These values describe the full frozen epoch plan; smoke runs consume only their limited batches. The callback reads no images, descriptors, graph edges or batch contents and does not modify the plan. These dashboard scalars do not enter `epoch_metrics.json`, `training_summary.json` or the scientific comparison export.

Weights & Biases is optional and disabled by default. Enable it by appending `--wandb` to any training command; `--wandb-project` defaults to `CounterMineVPR`, `--wandb-entity` selects an optional entity, and `--wandb-offline` enables local offline recording when `--wandb` is present. The repository environment includes pip `wandb`, but default training does not require or import it. An enabled run without the package stops with installation instructions. Run names and tags identify Step 3A, condition, seed and smoke status. Checkpoint uploads are disabled with `log_model=False`; no model watching or graph capture is added.

`run_configuration.json` records CSV/TensorBoard enablement, W&B enablement and effective offline mode, plus project, optional entity, run name and tags when enabled. It contains no API keys. Scientific configuration, sampler behavior, seeds, model, loss, optimizer, scheduler and checkpoint selection retain their existing behavior.

Existing preparation, baseline and smoke artifacts remain usable through the exact monitoring-only transition pinned in `countermine/training/step3a_monitoring_compatibility.json`. The guard compares the entire prepared source inventory and entire executed source inventory to that reviewed release; it does not ignore monitoring files or accept arbitrary source changes. Any further edit to scientific code, monitoring code or compatibility guards, any added/removed source, or an unknown preparation still fails. Frozen data, graph, SALAD, initialization, seed, smoke and checkpoint checks remain active. No Step 30 rerun or historical JSON hash rewrite is needed to continue the existing pilot.

Historical `provenance.step3a_code_hashes` remains the original preparation identity in scientific artifacts. Actual execution identity is recorded separately in the new training run's `run_configuration.json` under `source_compatibility`, in `<run>/evaluation_best_source_compatibility.json` (or `evaluation_last_source_compatibility.json`) before evaluation starts, and in `outputs/step3a/source_compatibility.json` after a successful comparison export. These records include both hash inventories and the compatibility manifest SHA256. Scientific metric exports retain their existing schema and semantics. Existing preparation, run directories and scientific inputs are preserved.

Training and evaluation initialize the unchanged upstream `VPRModel` using the existing `facebookresearch_dinov2_main` source cache and `checkpoints/dinov2_vitb14_pretrain.pth` under `torch.hub.get_dir()`. A scoped project adapter directs the existing DINOv2 hub request to that local source, preserving the model entrypoint, pretrained defaults and RNG seeds. It avoids PyTorch 2.1's implicit GitHub default-branch query, which can fail with `Remote end closed connection without response` even when all assets are cached. Source and pretrained assets must already exist; missing assets produce explicit instructions, and downloads are blocked during model construction. The adapter restores all hub functions before training or validation and never changes SALAD files.

Cached source and pretrained checkpoint fingerprints are recorded in `run_configuration.json` under `model_loading`, and in `<run>/evaluation_best_model_loading.json` (or the corresponding `last` sidecar). Training still strictly loads the original shared initialization, and evaluation still strictly loads the selected verified checkpoint. The exact compatibility manifest also pins this cached-loading adapter. If initialization fails before a run directory is created, retry the same training command after resolving the error; no cleanup or preparation rerun is needed.

## Runtime preflight and shared initialization

`tools/29_check_step3a_runtime.py --all` checks the workstation before the full workflow. `--stage prepare`, `train`, `evaluate` and `export` check the dependencies for each stage. Preflight verifies the authoritative dataset metadata, untouched SALAD source and frozen Step 2D graph, including all 2,000 places and exactly 1,083 Boston / 1,003 London eligible edges. It reports installed package versions without rejecting local build suffixes. Training and evaluation additionally require an available CUDA device and a finite tiny CUDA matrix multiplication. Training requires at least **5 GiB** free on the Step 3A cache filesystem to avoid an obviously inadequate checkpoint destination. Preflight never trains a model or deletes cache data.

The optional successful report is `cache/countermine_rgb/step3a/runtime_preflight.json`. It records a UTC timestamp, package/CUDA/GPU details, dataset status and frozen hashes without hostnames, usernames, environment variables or absolute paths. Runtime versions and GPU details also enter preparation, training and evaluation provenance and the eventual scientific comparison. Runtime differences are descriptive; source, data, initialization and seed invariants remain exact. A changed GPU name alone cannot reject a run.

Stage 30 serializes the shared state through an open binary file object into a visible `initial_state_tmp_*.pt` file in the destination directory. It flushes and fsyncs before atomic publication. The published file must be nonempty and pass a CPU `weights_only=True` round trip and strict load before its summary is written. Existing final artifacts are reused after verification, never silently overwritten. Startup cleans only matching temporary files. If just the tensor or just its summary exists, Stage 30 stops with instructions to inspect and manually remove the incomplete shared initialization transaction; it preserves the final tensor and summaries.

`environment_vpr.yml` retains PyTorch 2.1.0, torchvision 0.16.0, Lightning 2.1.2 and pytorch-metric-learning 2.3.0. It adds pip `xformers==0.0.22.post7`, whose dependency metadata requires exactly PyTorch 2.1.0, and the upstream SALAD pip dependency `faiss-gpu==1.7.2`, plus Matplotlib for export. The existing Conda channels and CUDA 12.1 pin remain unchanged. The current workstation has `faiss-cpu==1.7.4`; the frozen `faiss_gpu=False` path uses its CPU retrieval implementation, and preflight records the actual installed version. No local environment installation or upgrade is performed by this hardening pass.

## Artifacts and interpretation

Runtime files belong in `cache/countermine_rgb/step3a/`: shared initialization/preparation under `shared/`, full conditions under `baseline/seed42/` and `countermine_q99_geo500/seed42/`, with separate smoke directories. Per-epoch `batch_plans/epoch_00_summary.json`, guided-pair CSVs and dataset place-order CSVs record graph/order hashes, disjoint matching, duplicate/missing counts, split-pair counts, batch sizes, per-batch city composition and per-city marginal exposure hashes. Dataset place-order hashes bind the actual canonical identities, beyond the sequential numeric index range. Export reconstructs every plan from its saved dataset identities and the frozen binary graph, checks the guided-pair CSV against that deterministic reconstruction, and hashes the actual selected checkpoint bytes without loading them.

Exposure is measured after freezing each plan. Audits count total q95, q99 and q99_geo500 graph relations co-batched, alongside intentionally guided pairs, unique guided places and accidental structural co-occurrences. A guided matching count is distinct from total structural exposure. The comparison verifies exact marginal place multiset and Boston/London place-ID set equality for each corresponding epoch, and requires increased q99_geo500 exposure. A zero baseline denominator yields a JSON `null` ratio with an explicit `baseline_zero` flag.

The curated `docs/audits/step3a_edge_cobatching_metrics.json` contains frozen configuration, mapping/sampler audits, four full epoch metrics, best-checkpoint retrieval recalls, descriptive CounterMine-minus-baseline deltas, exposure ratios and provenance hashes. It declares `single_seed_pilot=true` and `statistical_significance_claimed=false`. It contains finite portable JSON, without checkpoint tensors, image pixels, descriptors or absolute paths. Checkpoints and the shared initialization remain uncommitted cache files.

Three separate figures are produced in `outputs/step3a/` and copied with the `step3a_` prefix to `docs/audits/`: `recall_curves.png` (Pitts30k val R@1 by epoch), `training_loss.png` and `structural_exposure.png`. The exporter validates complete run evidence before plotting or publishing a scientific snapshot.

Stop on a frozen-input/source/init hash mismatch, missing graph mapping, duplicate/omitted place, changed city composition, split guided pair, differing marginal place multiset, non-finite smoke loss, altered validation path or required upstream/loss/miner modification. Do not relax an invariant to repair a scientific mismatch.

Only this sampling experiment is authorized. Similarity-only mining, other thresholds, hub weighting, cross-city sampling, communities, multiple seeds, margins and structural losses remain deferred. A positive, neutral or negative measured retrieval change should be reviewed with the exposure audit before choosing later experiments.

## Initial implementation verification

Compilation and all **318 CPU unit tests** passed, including 75 Step 3A tests. An actual small CPU Lightning fixture verified callback ordering, two optimizer updates, three validation loaders, nine metrics and best/last checkpoint publication. It used a tiny fixture model and provides no SALAD retrieval result. The additional `countermine-vpr` test run passed 74 tests; its plotting test requires Matplotlib, which is available in the default CPU environment but absent from that environment.

The frozen graph SHA256 passes, with 2,000 graph places and 1,083 Boston / 1,003 London eligible same-city edges. A before/after inventory found no size or modification-time drift among 26,560 historical cache, curated and vendored-source files. Stage 30's missing-data stop and the scientific comparison's missing-results stop were exercised. The compact [implementation validation audit](audits/step3a_implementation_validation.json) records these checks and runtime blockers; it is separate from the unproduced scientific snapshot. No checkpoint was committed, and no commit or push was made.

## Runtime-hardening validation

Compilation and all **364 CPU repository tests** passed. All **121 Step 3A CPU tests** also passed in the actual `countermine-vpr` environment, including the PyTorch serialization round trip. The preparation preflight passed and saved its portable runtime report. Stage 30 completed with cached DINOv2 source/weights, without CUDA or a download, and produced all required shared artifacts and epoch-zero plans. The final initialization is 352,035,834 bytes; its SHA256 matches the summary and its strict reload passed:

```text
888790bb6f61d69c64cfc25c0f450a74e00cb248b4a41909a2e0b0486e7fce1f
```

The real dataset contains 62,514 training places, including 3,283 Boston and 5,052 London places. All 2,000 CounterMine places map exactly once. The frozen eligible edge counts remain Boston 1,083 and London 1,003. Epoch-zero q99_geo500 exposure is **29 baseline / 436 treatment**, an increase of **407 edges**, or **15.0345×**. The treatment guides 406 disjoint pairs involving 812 unique places. Duplicate, missing, split-pair and per-batch city-composition mismatch counts are zero. These are preparation/exposure audits, not retrieval results.

The training preflight stops because CUDA is unavailable in this session; package, dataset, source and conservative disk checks pass. No training was launched. The [runtime-hardening validation audit](audits/step3a_runtime_hardening_validation.json) records runtime versions, tests, initialization identity, exposure and the remaining blocker. The 26,560-file historical inventory remains unchanged. SALAD and Step 2A–2D artifacts were not modified, and no commit or push was made.

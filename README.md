# CounterMineVPR

**CounterMine: Mining Structural Counterexamples for Visual Place Recognition**

CounterMineVPR investigates whether different-place negatives retrieved by a frozen SALAD model contain more local structural similarity than matched random different-place controls. The active method uses original real RGB images:

```text
Real RGB
  -> SALAD global candidate retrieval
  -> ALIKED + LightGlue real-RGB structural analysis
  -> matched-random calibration of structural evidence
  -> CounterMine structural confusion graph pilot
  -> joint-null and fixed-graph topology-null validation
  -> full-scale real-RGB structural mining and graph construction
  -> edge-aware place co-batching training pilot
```

## Research status

| Stage | Status |
| --- | --- |
| Step 1: RGB candidate retrieval (1A–1C) | Completed |
| Step 2A: real-RGB structural-confusion audit | Completed |
| Step 2B: CounterMine structural confusion graph pilot | Completed |
| Step 2C: joint-null and topology-null validation | Completed |
| Step 2D: full-scale structural mining and graph construction | Completed |
| Step 3A: edge-aware place co-batching training pilot | Implemented; scientific runs blocked by missing training data and GPU runtime |

Step 2A measures LightGlue matches divided by the smaller endpoint keypoint count. Candidate pairs must have different place IDs and be at least 250 m apart. A deterministic 5,000-pair pilot is sampled from the existing Top-50 retrieval pool after undirected canonicalization. Each candidate is compared with a random negative sharing its query anchor, city relation, and, for same-city pairs, geographic distance bin. Controls exclude the anchor's entire saved Top-50 list.

Coverage, match concentration, entropy, and empirical null percentiles are descriptive diagnostics. Different places are not assumed to share a global geometric transform. No final CounterMine score, graph threshold, or training objective has been defined. The completed Step 2B/2C pilot supports the explicitly authorized Step 2D full-population measurement and graph audit. Step 3A now tests place co-batching with the original SALAD model, loss and online miner. No recognition performance improvement is claimed.

## Running Step 2B

[The graph pilot protocol](docs/step_2b_countermine_graph.md) uses completed Step 2A scalar measurements on CPU. Weak empirical null percentiles calibrate match ratio, absolute match count, coverage, and entropy separately. The primary structural evidence is the minimum of ratio and match-count percentiles; SALAD retrieval attributes stay separate. All 5,000 image edges are retained and aggregated into place pairs.

The q95/q99 slices and >=500 m sensitivity variants are pilot analyses. Structural evidence has no loss, margin, sampling, or training interpretation. No VPR performance improvement is claimed.

```bash
python tools/11_calibrate_structural_evidence.py
python tools/12_build_countermine_graph.py
python tools/13_analyze_countermine_graph.py
```

The frozen [Step 2A snapshot](docs/audits/step2a_rgb_structural_metrics.json) and curated figures remain unchanged. These CPU commands validate the existing measurements and never rerun ALIKED or LightGlue.

The completed pilot retains 5,000 image edges. Its q95/q99 slices contain 352/77 edges, or 340/72 at >=500 m. Three place pairs repeat with two q95 supports; one has distinct core views on both sides, surviving the geographic sensitivity. See the [measured graph snapshot](docs/audits/step2b_countermine_graph_metrics.json) and [repeated-support montage](docs/audits/step2b_repeated_place_confusions.jpg). Step 2C tests those repeated supports against the fixed-graph null; they do not establish persistent pairwise aliasing.

## Running Step 2C

[The Step 2C protocol](docs/step_2c_null_topology.md) validates joint ratio/count tail rates against the existing matched controls, then permutes complete structural-evidence bundles within city-relation × SALAD-rank × geographic strata of the unchanged 5,000-edge graph. It runs entirely on CPU and preserves the frozen Step 2A/2B artifacts.

```bash
python tools/14_analyze_joint_null.py
python tools/15_analyze_topology_null.py --permutations 1000 --seed 42
python tools/16_export_step2c_audit.py
```

The completed [Step 2C snapshot](docs/audits/step2c_null_topology_metrics.json) and four curated figures are frozen. Candidate joint q95/q99 rates are **7.04% / 1.54%**, compared with **2.20044% / 0.380076%** for matched-random controls, giving descriptive enrichment ratios **3.19936 / 4.05182**. Step 2B Spearman correlations of bottleneck with SALAD similarity/rank are **0.10648 / -0.03483**, supporting complementary structural evidence.

With 1,000 stratified permutations (seed 42), q95/q99 largest place components are **29 / 6**, with empirical exceedance fractions **0.000999 / 0.006993**. Strong evidence is concentrated on the place graph. The three q95 repeated place pairs have exceedance **0.20879**, and the one strict independent multi-view pair has **0.51449**; independent repeated support itself is not unusual. q99 repeats have a smaller descriptive exceedance but no strict two-sided independent support. The active interpretation is **structural-confusion communities and hubs**, with no claim of persistent pairwise multi-view aliasing or VPR improvement. See [the measured Step 2C results and caveats](docs/step_2c_null_topology.md#measured-results).

Historical Step 2A/2B/2C outputs remain immutable. The Step 2C commands document the completed protocol; Step 2D reads their frozen artifacts.

## Running Step 2D

[The full-scale protocol](docs/step_2d_full_countermine.md) reconstructs every eligible canonical candidate, extracts ALIKED once per original manifest image, validates exact feature/pilot replay, measures deterministic resumable LightGlue shards, and applies the frozen 4,999-control weak-ECDF calibration. It retains weak edges and predeclares q95/q975/q99/q995 diagnostic views and their >=500 m sensitivity variants.

Run tools **17–24 in order**, with a one-shard benchmark between validation and full matching. Tool 25 optionally creates selected correspondence overlays from the bank. The complete commands, stop conditions, immutable inputs and output paths are in [the workflow](docs/step_2d_full_countermine.md#run-in-order). Do not continue after a replay or provenance mismatch. Step 2D writes only its own cache, output and curated artifact namespace; no sampler, SALAD change or training objective is implemented.

**Step 2D — Completed.** The [frozen completed snapshot](docs/audits/step2d_full_countermine_metrics.json) records 260,502 measured canonical pairs over 8,000 images and 2,000 places, and reproduces all 5,000 Step 2A pilot pairs. Candidate q95/q99 rates are 6.61492% / 1.58079%, with descriptive matched-control enrichment ratios 3.00618 / 4.15915. The scientific run was produced before the git commit containing its results; source identity is pinned by the snapshot’s code and input hashes. Its original `git_commit_at_export` remains unchanged. No VPR improvement is claimed.

## Running Step 3A

[The edge-aware co-batching protocol](docs/step_3a_edge_cobatching.md) preserves the original SALAD model, MultiSimilarityLoss, online miner, optimizer, schedule and image transforms. The two seed-42 conditions differ only in which Boston/London places share the original per-batch city slots. Every training place appears exactly once per epoch. The frozen same-city q99_geo500 relation guides a disjoint matching; scores and hub degrees supply no weights.

Run tools **30–33 in order**: shared initialization and mapping, two separate smoke checks, baseline then CounterMine for four full epochs, best-Pitts30k-val checkpoint evaluation, and CPU comparison/export. The current workspace lacks the authoritative GSVCities/validation datasets and a usable CUDA/Torch runtime, so preparation stops before model creation. No smoke, training, evaluation or Step 3A scientific result has been produced. The complete commands and stop conditions are in [the workflow](docs/step_3a_edge_cobatching.md#run-in-order).

## Completed Step 2A workflow

See [the frozen experiment protocol](docs/step_2a_rgb_structural_audit.md) for validation, provenance, resumption, artifacts, and manual review criteria.

```bash
conda env create -f environment_mining.yml
python tools/07_build_structural_population.py --max-pairs 5000 --seed 42
conda run -n countermine-mining python tools/08_measure_structural_pairs.py --device cuda
python tools/09_analyze_structural_audit.py
conda run -n countermine-mining python tools/10_visualize_structural_matches.py --device cuda
```

The existing Step 1 caches remain the input; descriptor extraction and candidate retrieval need not be repeated. [Step 1C](docs/step_1c_candidates.md) documents those inputs. Runtime caches and transient outputs are ignored; compact measured summaries and curated figures belong in `docs/audits/`.

## Repository

`countermine/mining/` contains project-owned retrieval and structural-audit code. `tools/` contains experiment CLIs, `tests/` contains CPU unit tests, and `docs/` contains protocols and scientific snapshots. The project-owned `training/` namespace contains the Step 3A place co-batching integration; `probe/` remains a placeholder.

The active upstream methods are [SALAD](salad/README.md) and [LightGlue / ALIKED](third_party/LightGlue/README.md). Vendored directories are read-only. IC-Light and AdaptVPR remain historical vendor directories outside the active method.

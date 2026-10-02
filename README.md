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
  -> future pair-aware training
```

## Research status

| Stage | Status |
| --- | --- |
| Step 1: RGB candidate retrieval (1A–1C) | Completed |
| Step 2A: real-RGB structural-confusion audit | Completed |
| Step 2B: CounterMine structural confusion graph pilot | Completed; awaiting scientific review |
| Step 2C: joint-null and topology-null validation | Implemented; experiment not yet run |
| Step 3: training | Not yet implemented |

Step 2A measures LightGlue matches divided by the smaller endpoint keypoint count. Candidate pairs must have different place IDs and be at least 250 m apart. A deterministic 5,000-pair pilot is sampled from the existing Top-50 retrieval pool after undirected canonicalization. Each candidate is compared with a random negative sharing its query anchor, city relation, and, for same-city pairs, geographic distance bin. Controls exclude the anchor's entire saved Top-50 list.

Coverage, match concentration, entropy, and empirical null percentiles are descriptive diagnostics. Different places are not assumed to share a global geometric transform. No final CounterMine score, graph threshold, or training objective has been defined. The Step 2B pilot studies dataset-level topology using the completed 5,000-pair measurements. Scientific review must establish the structural-confusion hypothesis before scaling local measurement or changing training. No recognition performance improvement is claimed.

## Running Step 2B

[The graph pilot protocol](docs/step_2b_countermine_graph.md) uses completed Step 2A scalar measurements on CPU. Weak empirical null percentiles calibrate match ratio, absolute match count, coverage, and entropy separately. The primary structural evidence is the minimum of ratio and match-count percentiles; SALAD retrieval attributes stay separate. All 5,000 image edges are retained and aggregated into place pairs.

The q95/q99 slices and >=500 m sensitivity variants are pilot analyses. Structural evidence has no loss, margin, sampling, or training interpretation. No VPR performance improvement is claimed.

```bash
python tools/11_calibrate_structural_evidence.py
python tools/12_build_countermine_graph.py
python tools/13_analyze_countermine_graph.py
```

The frozen [Step 2A snapshot](docs/audits/step2a_rgb_structural_metrics.json) and curated figures remain unchanged. These CPU commands validate the existing measurements and never rerun ALIKED or LightGlue.

The completed pilot retains 5,000 image edges. Its q95/q99 slices contain 352/77 edges, or 340/72 at >=500 m. Three place pairs repeat with two q95 supports; one has distinct core views on both sides, surviving the geographic sensitivity. See the [measured graph snapshot](docs/audits/step2b_countermine_graph_metrics.json) and [repeated-support montage](docs/audits/step2b_repeated_place_confusions.jpg). Semantic interpretation and any scaling decision remain for scientific review.

## Running Step 2C

[The Step 2C protocol](docs/step_2c_null_topology.md) validates joint ratio/count tail rates against the existing matched controls, then permutes complete structural-evidence bundles within city-relation × SALAD-rank × geographic strata of the unchanged 5,000-edge graph. It runs entirely on CPU and preserves the frozen Step 2A/2B artifacts.

```bash
python tools/14_analyze_joint_null.py
python tools/15_analyze_topology_null.py --permutations 1000 --seed 42
python tools/16_export_step2c_audit.py
```

These commands write runtime tables to `cache/countermine_rgb/step2c/`, four plots to `outputs/step2c/`, and a curated `docs/audits/step2c_null_topology_metrics.json` snapshot plus plots. The combined Step 2C audit has not been published yet. If stage outputs already exist from before the raw-ratio roundtrip fix, follow the [fresh-directory rerun instructions](docs/step_2c_null_topology.md#run) to preserve them and avoid mixing code hashes. Joint-null and topology-null evidence must be reviewed before spending GPU time on scaling; training remains unimplemented.

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

`countermine/mining/` contains project-owned retrieval and structural-audit code. `tools/` contains experiment CLIs, `tests/` contains CPU unit tests, and `docs/` contains protocols and scientific snapshots. The `probe/` and `training/` namespaces are placeholders; they do not implement the active pipeline.

The active upstream methods are [SALAD](salad/README.md) and [LightGlue / ALIKED](third_party/LightGlue/README.md). Vendored directories are read-only. IC-Light and AdaptVPR remain historical vendor directories outside the active method.

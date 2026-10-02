# Step 2D — Completed

Full-scale CounterMine structural mining

## Status and scientific scope

The [frozen scientific snapshot](audits/step2d_full_countermine_metrics.json) records `complete = true`. The validated full run measured **260,502 canonical pairs**, covering **8,000 images** and **2,000 places**, and exactly reproduced all **5,000 Step 2A pilot pairs**. Full-population q95/q99 rates are **6.61492% / 1.58079%**, compared with frozen matched-random rates **2.20044% / 0.380076%**, giving descriptive enrichment ratios **3.00618 / 4.15915**. These measurements establish structural evidence; they do not establish VPR improvement.

The scientific run was produced before the git commit containing the results. Exact source identity is pinned by code hashes and input hashes inside the snapshot. The snapshot’s original `git_commit_at_export` is historical provenance and must not be rewritten to the later result-containing commit.

The reference branch is `feat/rgb-structural-countermine`, with expected starting commit `cbe93cbc64ca7454a40774e247b60338a424167b`. Step 2A, Step 2B and the completed [Step 2C audit](step_2c_null_topology.md#measured-results) are frozen evidence. Candidate q95/q99 joint enrichment and place-graph concentration motivate measuring the entire eligible SALAD population. Step 2C did not establish unusual strict independent pairwise multi-view support. The scientific emphasis is structural-confusion concentration, communities and hubs.

The active pipeline is original real RGB → frozen SALAD Top-50 candidates → different-place, >=250 m canonical population → one-time ALIKED feature bank → full LightGlue measurement → frozen matched-random calibration → complete image/place graph. Step 2D implements no training, sampler, SALAD modification, loss, margin or pair-specific weight. There is no final CounterMine scalar score or final graph threshold. The authorized [Step 3A pilot](step_3a_edge_cobatching.md) preserves the original SALAD loss and varies place co-occurrence only. Step 2D inputs and outputs remain immutable during that work.

## Immutable inputs and output locations

Read the existing `cache/gsv_mini/manifest.csv`, `rgb_candidates_raw.csv` and `rgb_candidates_raw_summary.json`. Treat every file in `cache/countermine_rgb/step2a/`, `step2b/`, `step2c/`, and `docs/audits/step2a_*`, `step2b_*`, `step2c_*` as immutable. Vendored `third_party/` code is read-only.

All new runtime artifacts belong in `cache/countermine_rgb/step2d/`; transient figures belong in `outputs/step2d/`; curated artifacts use the immediate `docs/audits/step2d_*` namespace. Destination guards must reject historical paths and source aliases, including resolved symlinks. Do not delete, edit or relabel earlier evidence to make a new provenance check pass. Cache and transient outputs remain gitignored; no automatic commit or push occurs.

Expected population counts and manifest size come from frozen provenance. A read-only CPU reconstruction validates **400,000 raw directed rows**, **390,115 >=250 m eligible directed rows**, **260,502 canonical pairs**, **129,613 reverse pairs**, **8,000 manifest images** and **2,000 endpoint places**. Canonical relations are 201,804 same-city / 58,698 cross-city pairs; the five rank bins contain 6,331 / 22,104 / 26,810 / 52,432 / 152,825 pairs. These reproduce the frozen retrieval/population metadata; they are not new local matching or full-graph results. Runtime code derives and validates these counts from provenance rather than using unchecked literals.

## Frozen measurements and calibration

Original images use native **640×480 RGB**, without EXIF transpose, crop, resize, padding, stretching or SALAD preprocessing. ALIKED uses `max_num_keypoints=2048`, `detection_threshold=0.2`, `eval()`, `torch.inference_mode()` and `extract(image, resize=None)`. Preserve the Step 2A deterministic Torch/CUDA settings: deterministic algorithms on, cuDNN benchmark off, cuDNN deterministic on, TF32 off and the existing `CUBLAS_WORKSPACE_CONFIG`.

LightGlue uses `features="aliked"`, `depth_confidence=-1`, `width_confidence=-1`, `filter_threshold=0.1`, `mp=False`, `eval()` and inference mode. There is no adaptive pruning, homography, fundamental matrix, coordinate-displacement metric, global registration or additional verifier.

The primary raw metric is:

```text
local_match_ratio = num_matches / min(num_keypoints_a, num_keypoints_b)
```

It is zero for a zero denominator. Coverage, entropy and concentration follow the unchanged Step 2A spatial formulas and remain descriptive diagnostics, without hard gates or keypoint-count exclusions.

Calibrate against the **same 4,999 frozen Step 2A matched-random controls**, using separate same-city and cross-city populations. For ratio, match count, symmetric coverage and symmetric entropy, use the relation-specific weak empirical CDF, preserving ties:

```text
F_null(x) = count(random_value <= x) / N_relation
structural_bottleneck = min(ratio_null_percentile, match_count_null_percentile)
structural_geomean = sqrt(ratio_null_percentile * match_count_null_percentile)
spatial_support_percentile = coverage_null_percentile
```

The null is fixed before inspecting full candidates. Do not generate new random controls, derive thresholds from candidate distributions, multiply bottleneck by SALAD similarity or add coverage to its formula. Reproduce the frozen Step 2C null thresholds and pilot q95/q99 rates before full calibration. Candidate counts compare descriptively with the frozen random rates; they are not VPR accuracy.

## Predeclared graph slices

For each threshold below, a core edge requires `structural_bottleneck >= threshold` and `not exact_pixel_duplicate`:

| Slice | Threshold | Role |
| --- | --- | --- |
| `core_q95` | 0.95 | Frozen Step 2B/2C diagnostic view |
| `core_q975` | 0.975 | Additional full-scale diagnostic view, declared before the run |
| `core_q99` | 0.99 | Frozen Step 2B/2C diagnostic view |
| `core_q995` | 0.995 | Additional full-scale diagnostic view, declared before the run |

Each also has a `*_geo500` variant requiring distance >=500 m. These are diagnostic topology slices, not training or final mining thresholds. They allow inspection of concentration if q95 becomes highly connected at full scale, without choosing a threshold after seeing the result. All [250,500) m eligible edges and all weak edges remain in the full graph. The full topology also has a >=500 m sensitivity view.

## Run in order

From the repository root, keep the same validated Python/CUDA environment for bank extraction, replay and matching. Use `--help` for optional path and engineering settings.

```bash
# A. Reproduce the complete eligible canonical population, on CPU.
python tools/17_build_full_structural_population.py

# B. Extract once per image in the complete frozen manifest.
conda run -n countermine-mining \
  python tools/18_build_aliked_feature_bank.py --device cuda

# C. Direct feature replay and bank-backed frozen Step 2A pair replay.
conda run -n countermine-mining \
  python tools/19_validate_aliked_bank.py --device cuda

# D. Benchmark exactly one 2,000-pair shard after validation passes.
conda run -n countermine-mining \
  python tools/20_measure_full_structural_pairs.py \
  --device cuda --shard-size 2000 --max-shards 1

# E. Run/resume all shards; validated complete shards are skipped.
conda run -n countermine-mining \
  python tools/20_measure_full_structural_pairs.py \
  --device cuda --shard-size 2000

# F. Require complete coverage and finalize the measured table.
python tools/21_finalize_full_structural_metrics.py

# G. Apply the frozen empirical null and reproduce the exact pilot.
python tools/22_calibrate_full_structural_evidence.py

# H. Retain every eligible candidate in the full image/place graph.
python tools/23_build_full_countermine_graph.py

# I. Analyze, visualize and export the validated full-scale audit.
python tools/24_analyze_full_countermine_graph.py

# J. Optional small-set LightGlue overlays using existing bank features.
conda run -n countermine-mining \
  python tools/25_visualize_full_structural_matches.py --device cuda
```

**Do not continue after stage C fails.** A deterministic or historical mismatch is a stop condition, not a reason to loosen tolerances. The benchmark must record pair count, elapsed seconds, pairs/second, CUDA allocated/reserved peaks, CPU RSS when available and an estimated full matching time. It cannot change scientific parameters. Fixing a benchmark implementation failure requires rerunning bank/pair validation before full matching.

The matching CLI supports `--start-shard` and `--max-shards`. Its default CPU feature cache is bounded to **128 images**, adjustable with `--feature-cache-size`. The validated current population gives 131 default shards, with 502 pairs in the final shard. A one-shard command on an already completed shard may skip it; the recorded benchmark belongs to the shard's actual measurement run. Use a fresh authorized Step 2D run directory for a deliberately different engineering configuration, rather than mixing summaries or deleting frozen evidence.

## Population, bank and resumability

Stage 17 reuses Step 2A's different-place >=250 m eligibility, canonical undirected image identity and stable pair UID. Reverse directed occurrences are aggregated. It performs no sampling, rank filtering or matcher-based selection. `full_population.csv` is sorted by pair UID and saves both directional retrieval ranks/similarities alongside canonical identity, place/city relation, distance, best rank, maximum/mean similarity and rank bin. Its summary reproduces the frozen eligible counts.

Stage 18 processes **every manifest image**, retaining original float32 feature tensors. No descriptor downcast, quantization or coordinate alteration is allowed. One project-owned atomic file per row is saved under `aliked_bank/features/000000.pt`, etc.; tensors move to CPU with exact keys, shape, dtype and values preserved. LightGlue-required fields are determined from the actual vendored implementation. The bank index binds row/image identity, encoded-image SHA256, RGB-pixel SHA256, keypoint count, relative feature filename and feature-file SHA256.

Bank provenance includes manifest hash, extractor config, ALIKED weight hash, vendored LightGlue source identity, Torch/CUDA versions, GPU, deterministic settings and code hashes, without absolute paths. The historical fingerprint index covers 6,781 of the 8,000 images; all bank entries additionally bind current encoded/RGB hashes at extraction, and available historical hashes must agree. Vendored source-byte hashes establish code identity when a recorded `git -C third_party/LightGlue rev-parse HEAD` resolves to the containing project repository rather than a separate vendor checkout. Historical metadata remains untouched.

Atomic writes use a temporary file, fsync/close and rename. A complete feature entry is reusable only when source hashes, extractor config, weight and code provenance agree. Missing/corrupt entries and provenance drift fail clearly; interruption does not permit partially written data to masquerade as completed features. Never preload the entire feature bank into RAM or GPU memory.

Stage 19 selects 32 direct-replay images by ascending SHA256 of `CounterMineVPR-Step2D-feature-replay|42|` plus image ID. Re-extracted bank tensor keys, shapes, dtypes and deterministic float values must be exact; a mismatch reports maximum absolute difference and stops. A deterministic 512-pair subset of the frozen 5,000-candidate pilot is matched from bank features only. Endpoint counts, matches and duplicate flags must be exact; ratio, coverage, entropy and concentration differences must be <=`1e-12`.

Stage 20 divides pair-UID order into deterministic 2,000-pair shards: `[0,2000)`, `[2000,4000)`, etc., with a shorter final shard. It checks all current original image files against bank fingerprints before matching, and checks the current shard's source files again before publication. Internal execution may use endpoint-row order for cache efficiency, but scientific CSVs return to population order. Each `shards/shard_00000.csv` and corresponding summary is atomic; the summary is the completion marker. It binds shard index/bounds, count, exact pair-UID sequence hash, population hash, feature-bank summary hash, matcher-config hash, code hashes and start/end timestamps. A valid complete shard is skipped; an incomplete shard is recomputed; mismatched provenance fails. It does not append to a giant running CSV.

Stage 21 requires every population pair exactly once, no duplicates/unknowns/missing pairs, finite metrics, `num_matches <= min_num_keypoints`, valid bank entries and consistent shard provenance. Only then does it publish the sorted full metric table and completion summary. Per-edge output holds the frozen structural scalars and compact population metadata; long image hashes live only in the bank index.

## Pilot reproduction and stop conditions

The exact 5,000 pilot UIDs must be present in the full table. Before graph publication, compare Step 2D against frozen Step 2A endpoint keypoints, matches and exact duplicates with exact equality; compare ratio, coverage, entropy and concentration at absolute tolerance <=`1e-12`. Recalibration must reproduce frozen Step 2B bottleneck/geomean and q95/q99 labels and frozen Step 2C core counts. A failed comparison invalidates the full run and blocks scientific analysis/export.

Stop and report any population-count mismatch, source-image drift against available historical hashes, nondeterministic feature replay, historical pair mismatch, incompatible Step 2C calibration, shard identity gap/duplication, changed metric formula, or required mutation of Step 2A/B/C artifacts. Do not round away drift, relax tolerances, rebuild the frozen null or modify old evidence as a repair.

## Graph analysis and inspection

Image nodes are all endpoints of the full candidate population; image edges retain every eligible canonical pair. Place pairs aggregate unordered endpoints with all-support count, per-core supporting edge count, unique selected images on each canonical side and strict core-only independent two-sided view support. Weak edges cannot inflate core-only view counts. SALAD, raw matching, structural, spatial and distance aggregations remain distinct; no training weight is created.

For image and simple place graphs, report full/q95/q975/q99/q995 and all >=500 m variants: edges, active/isolated nodes, mean/median/q90/q95/q99/max degree, total/non-singleton components, largest component size and fraction of active nodes, and median/q90/q95/q99 component size. Place hub lists include the top 50 by core degree and top 50 by `core_degree / full_candidate_degree`, requiring full degree >=5 for the fraction ranking. Repeated >=2/>=3 edges and strict independent support are descriptive, alongside the active community/hub emphasis.

The denominator audit uses `<256`, `256–511`, `512–1023`, `1024–1535`, `>=1536` bins and reports count, raw ratio/match-count/bottleneck median/q95/q99 and q95/q99 fractions. Compare the top 100 raw-ratio and bottleneck pairs for overlap, minimum-keypoint median/minimum, match-count median and counts below 256/512. There is no denominator hard gate.

Report relation and the five frozen rank-bin bottleneck distributions, all four core fractions, and Pearson/Spearman correlations with SALAD similarity and rank. All scientific statistics use every row. Optional display subsampling is deterministic and affects figures only; no p-values or accuracy claim is required.

The required seven plots show bottleneck distribution with the four reference thresholds, bottleneck by rank, SALAD similarity versus bottleneck, q95/q99 fractions by rank, place component sizes, place degree and denominator behavior. Original-RGB contact sheets inspect the top 100 nonduplicate edges, q95/q99 hubs and largest components at each core threshold. Top edges sort by descending bottleneck, geomean, matches, SALAD similarity, then ascending UID. Hub sheets show a representative image and up to four strongest connections, with full/core degrees and structural-edge fraction. Component sheets favor compact strongest-edge/place examples over unreadable giant node-link diagrams. No pair is automatically labelled good/bad or classified as architectural structure.

Optional stage 25 loads the bank and reruns LightGlue only for top 20 bottleneck edges and strongest examples from top five q99 hubs. It does not re-extract ALIKED or overwrite scientific metrics. At most 100 matches per pair are displayed in deterministic score order.

### Recovering from a geographic CSV reproduction error

If an earlier version reports `full graph reproduction differs in median_geo_distance_m`, its string-to-number parser may have lost one float64 ULP when reading large geographic distances. Graph loading now restores distance columns from their shortest round-trip representation, keeping the existing `1e-12` comparison bound. After updating the code, rebuild the CPU graph to refresh its code provenance, then rerun analysis:

```bash
python tools/23_build_full_countermine_graph.py
python tools/24_analyze_full_countermine_graph.py
```

The feature bank, GPU matching shards and calibrated evidence can be reused. This repair does not alter distances, eligibility, metrics or graph-slice thresholds. Genuine source differences still stop reproduction.

If stage 23 instead reports `feature-bank input scientific provenance differs` after a Git commit, compare the recorded and current input hashes. A changed repository commit alone is descriptive producer metadata, not a changed scientific input. The repaired bank and measurement readers compare all remaining scientific fields and preserve the original producer commit. They accept only the explicitly fingerprinted legacy versions of this reader repair; arbitrary extraction/matching code edits still fail. No bank configuration, feature record, replay marker, shard or calibrated evidence needs to be rewritten. With this repair installed, rerun stages 23 and 24 as above.

## Runtime and curated artifacts

| Stage | Files under `cache/countermine_rgb/step2d/` |
| --- | --- |
| Population | `full_population.csv`, `full_population_summary.json` |
| Bank | `aliked_bank/features/*.pt`, `aliked_bank/records/*.json`, `aliked_bank/configuration.json`, `aliked_bank/index.csv`, `aliked_bank/summary.json`, `aliked_bank/validation.json` |
| Matching | `full_measurement_config.json`, `shards/shard_*.csv`, `shards/shard_*_summary.json` |
| Finalization | `full_candidate_structural_metrics.csv`, `full_measurement_summary.json` |
| Calibration | `full_calibrated_candidates.csv`, `full_calibration_summary.json` |
| Graph | `image_nodes.csv`, `image_edges.csv`, `place_nodes.csv`, `place_edges.csv`, `full_graph_summary.json` |
| Analysis | `image_node_audit.csv`, `place_node_audit.csv`, `full_place_components.json`, `full_graph_analysis_summary.json` |
| Optional overlays | `full_match_overlay_summary.json` |

The completed scientific export is `docs/audits/step2d_full_countermine_metrics.json`. It records original snapshot/source hashes, full-population and measurement hashes, bank summary, configs/code and export commit, measured runtime, pilot reproduction/max differences, fixed-null thresholds, candidate/rank/relation/denominator statistics, all topology slices, hub metadata, descriptive repeated support, geographic sensitivity and compact top-100 edge metadata. JSON is finite (`allow_nan=False`) and contains no absolute paths, descriptors, keypoints, match-coordinate arrays, image pixels or model weights. It records `real_rgb_only=true`, `synthetic_images_used=false`.

Required transient figures under `outputs/step2d/` are:

```text
full_bottleneck_distribution.png
full_bottleneck_by_rank.png
full_salad_vs_bottleneck.png
full_core_fraction_by_rank.png
full_place_component_sizes.png
full_place_degree_distribution.png
full_keypoints_vs_ratio.png
top100_full_structural_edges.jpg
top_place_hubs.jpg
largest_place_components.jpg
top_full_match_overlays.jpg              # optional stage 25 only
```

Curated copies use the same names with a `step2d_` prefix in `docs/audits/`. A generated figure is not evidence that full scientific finalization succeeded; the validated final snapshot supplies completion and provenance.

## Verification and later decision

```bash
python -m compileall countermine tools
python -m unittest discover -s tests -v
```

Unit tests remain CPU-only and exercise population identities, frozen calibration/ties, bank metadata/tensor integrity, resumable shard boundaries/provenance, exact pilot drift rejection, graph aggregation/topology, output boundaries, finite exports and forbidden generative/training behavior. A read-only calibration preflight reconstructs the 4,999 frozen controls and pilot 352/77 core counts, with maximum calibration difference `1.11e-16`; this is a check of existing evidence, not new full-scale matching. ALIKED/LightGlue replay and actual throughput require the prescribed environment and GPU run and are reported separately.

Implementation validation passed: compilation and all **239 tests** in the default environment, plus all **65 Step 2D tests** in `countermine-mining`. All nine CLI help commands passed. Stage 17 produced and reloaded all 260,502 canonical pairs in a disposable Step 2D directory, which was then removed. A before/after inventory confirmed that all 10,101 frozen-input and vendored-source files were unchanged. No formal Step 2D cache, GPU feature bank, replay, measurement, throughput result or scientific export was produced during implementation validation.

The completed run records sustained enrichment against the frozen controls and a q95 largest component spanning 1,874 of 2,000 places. Predeclared stricter slices, hub visualizations and >=500 m sensitivity are available in the frozen snapshot for review. The explicitly authorized Step 3A pilot now tests same-city q99_geo500 place co-batching while preserving the original SALAD loss and marginal place exposure. Hub/community weighting and loss modifications remain deferred. No VPR improvement, better retrieval, architectural-only structure, causal explanation or final training score is claimed by Step 2D.

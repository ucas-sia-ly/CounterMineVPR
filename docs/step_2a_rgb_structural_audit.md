# Step 2A — Real-RGB Structural Confusion Audit

## Scope and baseline

CounterMine: Mining Structural Counterexamples for Visual Place Recognition begins with frozen SALAD RGB retrieval, followed by ALIKED + LightGlue local structural analysis and a matched-random structural null. Step 2A asks whether retrieved different-place candidates contain more local structural similarity than matched random different-place controls. Graph construction (Step 2B) and pair-aware training (Step 3) remain future work. No final score, graph threshold, loss, sampler, margin, or weight is defined here.

The implementation began on `feat/rgb-structural-countermine`, with clean HEAD `47a4764ea658092a46f35df5a0588a0629588ba9` and parent `3e2fa765a0cf6d9c426378e996ab9f791ec6504a`. These identifiers record the inspected baseline and are not runtime requirements. The active baseline contained Step 1A–1C only. Existing inputs are:

- `cache/gsv_mini/manifest.csv`
- `cache/gsv_mini/descriptors_fp16.npy` (preserved; not read by local analysis)
- `cache/gsv_mini/rgb_candidates_raw.csv`
- `cache/gsv_mini/rgb_candidates_raw_summary.json`

Do not regenerate descriptors or retrieval unless integrity validation establishes inconsistency. Historical vendor directories remain read-only. Active upstream methods are SALAD and LightGlue / ALIKED.

## Frozen population

The default raw stage must have `top_k=50` and `min_geo_distance_m=0.0`. Validate summary statistics, manifest hash, schema, row identities, places, cities, ranks, finite scores, directed uniqueness, and the existing `stable_pair_uid`. Independently recompute haversine distances using manifest latitude/longitude and the same Earth radius as Step 1C. Require absolute disagreement <= **1e-6 m**, with relative tolerance zero. Metadata or distance failures terminate construction; never substitute records.

Keep only different-place rows whose stored distance is >= **250.0 m**. The raw CSV remains unchanged. Aggregate each unordered image pair once, with endpoints ordered lexicographically by image ID. Save both directional ranks/similarities where present, direction count, best rank, maximum and mean similarity, endpoint identity and geographic metadata.

Choose the representative anchor direction before matching, ordered by lower rank, higher similarity, lower query row index, then lower negative row index. Rank bins are fixed:

| Bin | Best RGB rank |
| --- | --- |
| `rank_1` | 1 |
| `rank_2_5` | 2–5 |
| `rank_6_10` | 6–10 |
| `rank_11_20` | 11–20 |
| `rank_21_50` | 21–50 |

For seed 42, compute SHA256 of the UTF-8 string `CounterMineVPR-Step2A|42|` plus `pair_uid`. Sort by this hexadecimal sampling key ascending, then pair UID ascending. Take the first 5,000 unique eligible pairs, or all eligible pairs if fewer exist. This sampling does not depend on local matching or preferentially select high SALAD scores. `--max-pairs 0` supports future full-population work but must not be used in this task.

## Matched random controls

For each sampled candidate, preserve its representative query exactly. A random target must differ from the query and candidate negative, have a different query place, be >=250 m away, and lie outside the query's complete saved Top-50 list. Selection never uses local matches or SALAD similarity.

Same-city candidates require same-city controls in the candidate's distance bin: `[250,500)`, `[500,1000)`, `[1000,2000)`, `[2000,5000)`, or `[5000,+inf)`. Cross-city controls require the candidate negative's city. Sort eligible random targets by row index ascending. Derive the draw seed from SHA256 of `CounterMineVPR-Step2A-random|42|` plus candidate pair UID and select deterministically. Save the exact draw implementation and seed in provenance.

Unavailable controls stay explicitly recorded with `random_control_available=false`; their candidates remain in the pilot. No fallback or replacement is allowed. Construction fails if fewer than 95% of candidates have valid controls. Population, controls, and their summaries are published under `cache/countermine_rgb/step2a/`.

## Original-image matching

All images come from the original manifest under `data/gsv-cities`. Decode without EXIF transpose, convert to RGB, and require width 640, height 480. No crop, resize, padding, stretch, canonical coordinate conversion, or SALAD preprocessing is applied. Convert uint8 RGB to float CHW tensors in `[0,1]`. Call `extractor.extract(image, resize=None)` and validate returned `image_size` against `[640,480]`.

| Component | Frozen configuration |
| --- | --- |
| ALIKED | `max_num_keypoints=2048`, `detection_threshold=0.2` |
| LightGlue | `features="aliked"`, `depth_confidence=-1`, `width_confidence=-1`, `filter_threshold=0.1`, `mp=False` |

Use `eval()` and `torch.inference_mode()` for both models. Adaptive depth and width pruning are disabled. Candidate and random pairs use identical settings. Seed Python, NumPy and Torch and save software, code, vendor, input and model provenance. No generator or relit image is part of active mining.

Process streaming with a bounded feature LRU; do not preload the dataset or retain all local descriptors. No permanent full-dataset local-feature bank is written. Atomic per-pair checkpoints permit resumption without duplicate candidate or control measurements. Before reuse, validate matcher configuration, population/control CSV hashes, manifest, raw CSV and summary, and original-image integrity. Incompatible runs fail rather than mixing results. Final CSVs are published atomically and completion is checked before analysis.

## Measurements

The primary metric is `local_match_ratio = num_matches / min(num_keypoints_a,num_keypoints_b)`, or zero for a zero denominator. No structural threshold is selected.

For each endpoint independently, map matched pixel coordinates into an 8x8 grid using `floor(8*x/W)` and `floor(8*y/H)`. Clamp only for floating-point boundary roundoff; invalid coordinates fail. Coverage is occupied cells / 64. Symmetric coverage is the smaller endpoint coverage. Concentration is maximum cell match count / match count, zero for no matches. Entropy is `-sum(p*log(p))/log(64)`, with zero terms ignored and zero entropy for no matches. Symmetric entropy is the smaller endpoint entropy.

Coverage, entropy, and concentration are diagnostics only. They never reject a pair or define a fidelity gate. Different places need not obey a single transformation; do not compute coordinate displacement, R2/R4/R8/R16, homography or fundamental-matrix inliers, or geometric registration.

Hash exact decoded uint8 row-major RGB bytes with SHA256 and cache each used image's hash. Save `exact_pixel_duplicate` on every candidate/control row and retain all raw records. Exclude exact duplicates explicitly from the evidence montages. Flag `near_geo_500m` for distances <500 m; retain these eligible >=250 m records for manual review.

## Descriptive analysis

For each available matched control, compute candidate ratio minus random ratio. Report candidate/random count, mean, sample standard deviation (`ddof=1`), min, q01/q05/q25/median/q75/q95/q99 and max. Report paired delta count, mean, sample standard deviation, min, q05/q25/median/q75/q95 and max; positive/zero/negative sign counts and the fraction candidate > random. Empty or undefined finite statistics are JSON null. Do not perform significance tests, report p-values, select post-hoc thresholds, or claim causality.

The global weak null ECDF is `count(random_ratio <= candidate_ratio) / count(random_ratio)`. Relation-specific percentiles use only same-city or cross-city controls; subgroups with fewer than 100 controls receive null. Neither percentile is a CounterMine score or threshold.

Report comparisons separately for same-city/cross-city and every frozen rank bin. Rank reports include candidate count, SALAD similarity q25/median/q75, candidate and matched random ratio q05/q25/median/q75/q95, paired delta q25/median/q75, fraction candidate > random, and global-null percentile q25/median/q75/q95. Report descriptive Pearson and Spearman correlation of local ratio with maximum SALAD similarity and best RGB rank.

Report duplicate counts and near-geographic counts separately, including among the strongest structural pairs. Sort top candidates by ratio descending, match count descending, maximum SALAD similarity descending, then pair UID ascending. Select top 50 nonduplicates for the original-image montage; annotate identities, cities, distance, best rank, similarity, keypoint/match counts, ratio, endpoint coverage, and global null percentile. Select top 20 nonduplicates for overlays. Display at most 100 correspondences, ordered deterministically by highest confidence if available, otherwise by match index. Visual selection is descriptive, without good/bad labels or automatic semantic classification.

## Commands and artifacts

Run from the repository root. Environment dependencies are limited to the frozen PyTorch/CUDA family, ALIKED/LightGlue imports, image loading, analysis and plotting; no generator stack is installed by this specification.

```bash
conda env create -f environment_mining.yml
python -m compileall countermine tools
python -m unittest discover -s tests -v
python tools/07_build_structural_population.py --max-pairs 5000 --seed 42
conda run -n countermine-mining python tools/08_measure_structural_pairs.py --device cuda
python tools/09_analyze_structural_audit.py
conda run -n countermine-mining python tools/10_visualize_structural_matches.py --device cuda
```

The analysis CLI reads completed CSVs without importing SALAD, ALIKED, LightGlue, Torch or CUDA. It exports a compact finite JSON snapshot to `docs/audits/step2a_rgb_structural_metrics.json`, including export-time git commit, hashes of the manifest/raw CSV/summary, frozen population and matcher settings, population counts, descriptive comparisons, null calibration, rank/city summaries, correlations, integrity audit and compact top50 metadata. It records `real_rgb_only=true`, `synthetic_images_used=false`, `local_resize=null`, and `registration=null`. Absolute paths, descriptors, coordinate arrays, images, weights, NaN and Infinity are forbidden in this snapshot.

Transient outputs are:

- `outputs/step2a/candidate_vs_random_local.png`: random-vs-candidate paired scatter with y=x.
- `outputs/step2a/salad_vs_local.png`: similarity-vs-local scatter by city relation.
- `outputs/step2a/local_by_rank.png`: candidate ratio distributions by frozen rank bin.
- `outputs/step2a/top50_structural_candidates.jpg`: original RGB pairs with annotations.
- `outputs/step2a/top20_match_overlays.jpg`: original RGB correspondences.

After successful export/overlay rendering, copy these to `docs/audits/` with the `step2a_` prefix. Runtime directories remain ignored. Do not commit or push automatically.

## Implementation validation

`python -m compileall countermine tools` completed successfully. The default Python environment passed all 96 CPU tests; the new structural population, metric, null, analysis/export and resumption tests also passed in `countermine-mining` (49 tests). Tests use small fixtures and stubs; they do not load pretrained weights or require CUDA.

An additional full-suite run in the pinned mining environment reproduces one pre-existing Step 1C oracle-test failure at the original HEAD (five chunk-size subcases). NumPy 1.26.4's full-matrix oracle introduces approximately one-ULP score differences between duplicate descriptor columns, altering expected tie order. The existing tiled Torch miner preserves exact duplicate ties and produces identical outputs across all five chunk sizes. The baseline code and cached retrieval are preserved; Step 2A never reruns this retrieval. This environment-specific oracle limitation is separate from the 49 passing structural tests.

## Logged pilot results (2026-10-02)

The frozen pilot measured 5,000 candidates and 4,999 matched random controls. One unavailable control remained explicit; its candidate remained in the 5,000-pair population. All original image bytes passed the completion integrity check. Results and compact provenance are saved in [the scientific snapshot](audits/step2a_rgb_structural_metrics.json).

| Descriptive measurement | Candidate | Matched random |
| --- | ---: | ---: |
| Mean local match ratio | 0.03381761 | 0.02781152 |
| Median local match ratio | 0.03041543 | 0.02501604 |
| Sample standard deviation | 0.01946988 | 0.01564592 |

Among 4,999 paired cases, 2,975 candidates exceeded their controls (59.51%), 34 tied, and 1,990 were lower. Mean paired delta was 0.00600196; median delta was 0.00468719. Candidate > random fractions were 60.62% within the same city and 55.60% across cities. The distributions overlap substantially; these values are descriptive evidence, without a significance test or decision threshold.

Local match ratio correlated with maximum SALAD similarity at Pearson 0.11239 and Spearman 0.07941, and with best RGB rank at Pearson -0.05136 and Spearman -0.04061. This describes association in the frozen sample; it does not establish added training value or causality.

No candidate or control was an exact decoded-pixel duplicate. 124 candidates were within [250,500) m; 4 of the top 50 were in that interval. All remain available for manual inspection. Some top-ranked records have few keypoints on one endpoint and concentrated matches; the montage and overlays expose this without filtering on coverage or entropy. No semantic labels or automatic scientific pass/fail decision were assigned.

Curated artifacts:

- [Candidate vs matched random](audits/step2a_candidate_vs_random_local.png)
- [SALAD similarity vs local ratio](audits/step2a_salad_vs_local.png)
- [Local ratio by frozen rank bin](audits/step2a_local_by_rank.png)
- [Top 50 original RGB candidate pairs](audits/step2a_top50_structural_candidates.jpg)
- [Top 20 correspondence overlays](audits/step2a_top20_match_overlays.jpg)

The plotted layouts and representative montage/overlay panels were visually inspected. Scientific semantic review remains a human decision before any Step 2B task. No VPR training or retrieval performance experiment was performed.

## Human scientific decision

Review whether candidate ratios shift upward relative to matched random controls, whether a substantial majority of paired deltas are positive, and whether top correspondences reflect facade grids, windows, arches, columns, balconies, rooflines or road/building layout. Check for vegetation, sky, road/brick texture and accidental repeated patches instead. Inspect variation within similar SALAD ranks and whether duplicates or nearby geographic false negatives dominate the strongest pairs.

These questions are not automated pass/fail rules. Only a promising human review authorizes a later Step 2B graph task. Logged structural evidence does not establish a VPR performance improvement.

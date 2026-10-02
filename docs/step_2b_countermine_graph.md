# Step 2B — CounterMine Structural Confusion Graph Pilot

## Purpose and frozen evidence

Step 2B asks whether the completed 5,000-pair real-RGB pilot contains interpretable dataset-level structural-confusion topology. It constructs the **CounterMine structural confusion graph** from existing Step 2A scalar measurements. It does not establish that the graph improves VPR, define a final CounterMine mining score or training threshold, or select a future training mapping.

The frozen scientific source is [the Step 2A snapshot](audits/step2a_rgb_structural_metrics.json). The reference Step 2A commit is `d0758e17eef41e7cb7a5c325726282cc39ad1810` on `feat/rgb-structural-countermine`. Existing measurements are read from `cache/countermine_rgb/step2a/candidate_structural_metrics.csv` and `random_structural_metrics.csv`. Required counts are exactly **5,000 candidates and 4,999 matched random controls**. The one unavailable matched control is not replaced; its candidate remains in the graph.

Every candidate begins with frozen SALAD Top-50 retrieval on original real RGB and has different place identities and geographic distance >=250 m. Same-place pairs must never become negative edges. This pilot does not rescale to the full eligible candidate population. It performs no new ALIKED or LightGlue inference, correspondence overlay rerun, model loading, descriptor extraction, global retrieval, synthetic generation, training, sampling, margin, loss, or optimizer work. All Step 2B modules and CLIs run on CPU without importing the ML model stack. Vendor repositories and historical Step 2A figures remain unchanged.

## Input integrity and provenance

Validation fails before graph construction when counts, pair identities, schemas, finite scalar metrics, or source provenance disagree. Candidate and random `pair_uid` values must be unique. Canonical image endpoints, row indices, place and city identities are checked against the manifest, and candidate/control pairing must agree with the frozen population. Geographic exclusion remains the Step 2A >=250 m rule.

The historical Step 2A snapshot contains population/input hashes and the measurement configuration hash, but **does not directly contain the final candidate/random measurement CSV hashes**. Step 2B does not retroactively add fields to that snapshot. Its validation chain checks the completion marker's candidate/random CSV checksums; frozen population and random-control hashes; manifest, raw candidate CSV and candidate-summary hashes; measurement configuration and matcher provenance; and the image-fingerprint index. It then reproduces every historical Step 2A descriptive report field and its ordered Top-50 records on CPU, comparing numeric values with absolute tolerance `1e-12`. No local measurements are recomputed.

After this validation, the new Step 2B snapshot records SHA256 hashes of the historical snapshot, candidate/random measurement CSVs, completion summary, measurement configuration, image-fingerprint index, graph artifacts and graph implementation files. Each stage saves its deterministic configuration and seed **42**, and publishes checksummed outputs atomically. Later stages verify the saved calibration/graph hashes and the same frozen source chain. Mixed runs fail rather than combining records.

Figures resolve original files through the validated manifest and require the stored encoded-file SHA256 and decoded, row-major RGB-pixel SHA256. Source decoding converts to RGB and requires native **640x480**, without EXIF transpose, crop, resize, padding, stretching, or registration. Only bounded per-image decoding is used for display; no local-feature cache or full-dataset feature bank is constructed.

## Multi-dimensional null calibration

For each candidate, calibrate four previously measured quantities against the 4,999 matched random controls:

| Metric | Existing scalar column | Calibrated primary column |
| --- | --- | --- |
| Match fraction | `local_match_ratio` | `ratio_null_percentile` |
| Absolute correspondence support | `num_matches` | `match_count_null_percentile` |
| Spatial coverage | `symmetric_match_coverage` | `coverage_null_percentile` |
| Spatial entropy | `symmetric_match_entropy` | `entropy_null_percentile` |

For each quantity `x`, use the weak empirical CDF:

```text
F_null(x) = count(random_value <= x) / number_of_random_values
```

Compute global and relation-specific percentiles independently for `same_city` and `cross_city`. A relation-specific null requires at least **100 measured controls**. If unavailable, store its percentile as JSON null / an empty CSV cell. The primary percentile uses the relation-specific null when available, otherwise the global null. Every metric saves its explicit null-source field (`same_city`, `cross_city`, or `global`), both percentile variants, and the selected finite percentile in `[0,1]`. The calibration summary saves relation counts, availability, formulas and fallback behavior.

The historical raw ratio remains unchanged:

```text
local_match_ratio = num_matches / min(num_keypoints_a, num_keypoints_b)
```

It is zero when the denominator is zero. Step 2A exposed that a small denominator can produce a high ratio with modest match support, such as approximately 37 matches / 237 keypoints. The Step 2B primary structural evidence therefore requires strength in both dimensions:

```text
structural_bottleneck = min(ratio_null_percentile, match_count_null_percentile)
structural_geomean = sqrt(ratio_null_percentile * match_count_null_percentile)
spatial_support_percentile = coverage_null_percentile
```

`structural_geomean` is a secondary descriptive attribute. Coverage and entropy remain diagnostics; neither enters the primary bottleneck or a hard rejection gate. Different-place matching describes structural evidence without assuming a homography, fundamental matrix, source-to-source coordinate displacement, R4/R8, or global registration.

`structural_bottleneck` and `structural_geomean` are graph-analysis weights only. They are not loss weights, margin values, sampling probabilities, curriculum coefficients, ground-truth aliasing labels, or a final CounterMine score.

## Preserve SALAD evidence separately

Every image edge retains `best_rgb_rank`, `max_salad_similarity`, `mean_salad_similarity`, `rank_bin`, and `num_candidate_directions`. Its descriptive `salad_similarity_percentile` is the weak ECDF of `max_salad_similarity` over all 5,000 candidates. This percentile is an attribute only. SALAD similarity is not multiplied into the local structural evidence; every edge already originates from global SALAD candidate retrieval.

## Image graph and pilot slices

The image graph is undirected and contains **all 5,000 measured candidate edges**, including low structural evidence. Node attributes are `image_id`, `row_index`, `place_uid`, and `city_id`. The fixed pilot node universe is the union of measured candidate endpoints, rather than every image in the full manifest. Each analysis slice preserves this same universe, including zero-degree isolated nodes.

Edges retain pair/endpoint identities, cities, geographic distance and `near_geo_500m`; directional/global retrieval evidence; keypoint/match counts and the historical match ratio; endpoint and symmetric coverage/entropy; concentration diagnostics; all null percentiles and provenance; bottleneck/geomean; and `exact_pixel_duplicate`. Duplicate records are retained in the full graph and explicitly excluded from the core slices and top image lists.

The following **pilot analysis slices** are frozen before graph-result analysis:

| Slice | Edge inclusion |
| --- | --- |
| `full` | All measured candidate edges |
| `core_q95` | Ratio percentile >=0.95 and count percentile >=0.95, and not an exact pixel duplicate |
| `core_q99` | Bottleneck >=0.99, and not an exact pixel duplicate |
| `core_q95_geo500` | `core_q95` and geographic distance >=500 m |
| `core_q99_geo500` | `core_q99` and geographic distance >=500 m |

The q95 condition is equivalent to bottleneck >=0.95. No coverage threshold is applied. The geo500 variants are conservative sensitivity analyses; they do not delete the eligible [250,500) m records or alter Step 2A eligibility. Neither q95 nor q99 is a final training threshold.

Outputs are `cache/countermine_rgb/step2b/image_nodes.csv` and `image_edges.csv`.

## Place aggregation and repeated support

The place graph aggregates image edges by the unordered, lexically ordered `(place_uid_a, place_uid_b)` pair. Place nodes are the places represented by the image graph's fixed endpoint universe. All place-pair edges remain in the full graph. A place pair appears in a core slice if at least one supporting image edge meets that slice.

Each place edge saves total supporting image-edge count and q95/q99 counts, including both geo500 variants. It saves unique image counts on both place sides across **all supporting edges**, plus:

```text
independent_view_support = min(num_unique_images_a, num_unique_images_b)
```

It also saves best rank minimum; SALAD similarity maximum/median; match-count and match-ratio maximum/median; bottleneck and geomean maximum/median; symmetric coverage/entropy maximum/median; and geographic distance minimum/median. Outputs are `place_nodes.csv` and `place_edges.csv` in the Step 2B runtime directory.

The requested repeated-support flags are:

```text
repeated_support_2 = num_core_q95_image_edges >= 2
repeated_support_3 = num_core_q95_image_edges >= 3

independent_support_2 = num_core_q95_image_edges >= 2
                        AND num_unique_images_a >= 2
                        AND num_unique_images_b >= 2
independent_support_3 = num_core_q95_image_edges >= 3
                        AND num_unique_images_a >= 2
                        AND num_unique_images_b >= 2
```

Here `num_unique_images_a/b` includes **all support**, including non-core edges. Consequently, these prescribed independent flags do not by themselves prove that high structural evidence repeats across at least two distinct views on each side. The implementation preserves them exactly and adds a stricter diagnostic:

```text
core_independent_support_k = num_core_q95_image_edges >= k
                             AND num_core_q95_unique_images_a >= 2
                             AND num_core_q95_unique_images_b >= 2
                             for k in {2,3}
```

For the >=500 m sensitivity, requested independence uses qualifying core counts and unique images among all >=500 m supporting edges. The stricter `core_geo500_independent_support_2/3` uses distinct images within **core_q95_geo500 support itself**; `core_independent_support_2/3_geo500` is an equivalent saved alias. Figures label all-support and core-only distinct-image counts separately. These flags are descriptive repeated-support indicators, without semantic ground-truth labels or a training interpretation.

## Topology and diagnostics

Lightweight union-find computes connected components without NetworkX. For every image/place graph slice, report total nodes, edges, active nodes and isolates; degree mean, median, q90, q95 and maximum; component count and non-singleton count; and component-size median, q90, q95 and maximum. Primary degree distributions include fixed-universe zero-degree nodes; primary component-size distributions include singleton isolates. Additional active-degree and non-singleton-size summaries make this denominator explicit.

Report up to 20 active image/place nodes by core_q95 degree, with image/place/city identities and lexical tie breaks. High degree is not automatically a useful negative. Image degree normalization saves `candidate_degree_full`, `structural_degree_q95`, and `structural_edge_fraction = structural_degree_q95 / candidate_degree_full`, with zero for a zero denominator. Its top-20 list requires full degree >=3 and sorts by descending fraction, core degree, full degree, then image ID. This distinguishes repeated retrieval opportunity from a high rate of structural evidence.

Frozen rank bins are `rank_1`, `rank_2_5`, `rank_6_10`, `rank_11_20`, and `rank_21_50`. Each saves pair count; bottleneck q25, median, q75, q95 and q99; and q95/q99 count and fraction. Spearman correlations of bottleneck with maximum SALAD similarity and best RGB rank are descriptive, with no p-values.

The small-denominator audit uses `min_num_keypoints = min(num_keypoints_a,num_keypoints_b)` and frozen bins `<256`, `256-511`, `512-1023`, `1024-1535`, and `>=1536`. Each reports count, ratio/matches/bottleneck median and q95, and core_q95 fraction. Compare denominator distributions and overlap among the top 50 raw-ratio and top 50 bottleneck nonduplicates. No keypoint-count threshold is imposed and no reduction in artifacts is assumed before inspection.

For core_q95 edges, report symmetric coverage/entropy and maximum-cell concentration distributions. Symmetric concentration is the larger endpoint max-cell fraction. The manual subset `core_q95_low_spatial_support` is core_q95 with symmetric coverage <= the coverage q10 of the **entire frozen 5,000-candidate population**; the q10 is saved. These are inspection cases, not failures or rejected edges. Geographic sensitivity reports image/place topology and repeated-support persistence at both >=250 m and >=500 m.

## Visual selection and inspection artifacts

Top bottleneck image lists exclude exact duplicates and sort by bottleneck descending, geomean descending, match count descending, then pair UID ascending. Raw-ratio lists preserve Step 2A order: ratio descending, match count descending, maximum SALAD similarity descending, then pair UID ascending. The top-20 comparison highlights records appearing in only one list.

Repeated place-pair priority is exactly `independent_support_3`, `independent_support_2`, `repeated_support_3`, then `repeated_support_2`, assigning each pair to its first qualifying group. Within groups, sort by bottleneck maximum descending, core_q95 support count descending, and lexical place pair. Show at most **20 place pairs**, with at most **3 strongest core_q95 nonduplicate image pairs per pair**, using the bottleneck tie rules above.

The component contact sheet selects at most **5 largest nontrivial place-level core_q95 components**, ordering size descending then lexical node list. Per component it shows at most **3 strongest connecting image pairs** and **12 representative place nodes**. Each representative comes from that place's strongest connecting core edge; displayed place representatives are selected lexically. These caps and all selected pair/place/image identities are saved in visualization metadata. Empty repeated-support or nontrivial-component slices produce explicit explanatory panels.

The top-50, repeated-support and component-link endpoints display original RGB at native 640x480. The side-by-side top-20 comparison displays complete frames at 320x240, and component representatives display complete frames at 240x180. Captions and saved metadata disclose display-only scaling; source hashes and all scientific measurements remain based on the original native pixels. No correspondences are recomputed or drawn.

Transient figures are in `outputs/step2b/`; final copies are atomically curated under `docs/audits/`:

- [Top 50 structural bottleneck edges](audits/step2b_top50_structural_bottleneck.jpg): place/city/distance, SALAD rank/similarity, minimum keypoints, matches/ratio, ratio/count percentiles, bottleneck and symmetric coverage.
- [Raw ratio versus bottleneck top 20](audits/step2b_ratio_vs_bottleneck_top20.jpg): two ranking groups, highlighting list-exclusive records.
- [Repeated place confusions](audits/step2b_repeated_place_confusions.jpg): multiple original image pairs with support and distinct-view counts.
- [Largest place components](audits/step2b_largest_place_components.jpg): representative place nodes and strongest connecting pairs.
- [Ratio versus count percentiles](audits/step2b_ratio_vs_count_percentile.png): q95/q99 reference lines.
- [Bottleneck by SALAD rank](audits/step2b_bottleneck_by_rank.png): distributions by the five frozen rank bins.
- [Place component sizes](audits/step2b_place_component_sizes.png): core_q95 component sizes, including singleton isolates.
- [Keypoints versus raw ratio](audits/step2b_keypoints_vs_ratio.png): top-50 ratio and bottleneck selections marked separately.

## CPU workflow and exports

Run from the repository root using the existing Python environment with NumPy, pandas, Pillow and Matplotlib:

```bash
python tools/11_calibrate_structural_evidence.py
python tools/12_build_countermine_graph.py
python tools/13_analyze_countermine_graph.py
```

No CUDA command or ML weights are needed. Do not rerun `tools/08_measure_structural_pairs.py`. The calibration and graph summaries save formulas, source validation, configuration, seed, code hashes and output hashes. Runtime audits include per-node opportunity/structural degrees and nontrivial place-component metadata. Cache and transient figure directories remain gitignored; there is no automatic commit or push.

The compact [Step 2B scientific snapshot](audits/step2b_countermine_graph_metrics.json) saves provenance, calibration, exact structural-evidence formulas, image/place topology for all slices, repeated/independent support counts, rank and denominator audits, spatial/geographic diagnostics, top image/place records, hubs and visualization selections. It explicitly records `real_rgb_only=true`, `synthetic_images_used=false`, and `no_new_local_matching=true`. Serialization uses `json.dumps(..., allow_nan=False)`. Undefined finite summaries are null. Absolute paths, descriptors, coordinate arrays, image pixels and ML weights are excluded.

Required implementation checks are:

```bash
python -m compileall countermine tools
python -m unittest discover -s tests -v
```

CPU fixtures cover weak ECDFs, relation/fallback behavior, finite percentiles and bottleneck formulas; undirected graph identities/retention/slices; unordered place aggregation and both distinct-view scopes; topology and isolates; denominator bins and deterministic lists; original-file/pixel fidelity and informative empty panels; finite path-free exports and ML/training import boundaries. Tests do not load pretrained models.

## Measured results

The three CPU CLIs completed using the frozen 5,000 candidates and 4,999 controls. The measured relation nulls contain **3,900 same-city controls and 1,099 cross-city controls**; both exceed 100, so every candidate uses its relation-specific null. No global fallback was required in this run. Configuration, CSV checksums and the exact implementation hashes are saved in the [scientific snapshot](audits/step2b_countermine_graph_metrics.json) and runtime stage summaries.

The full image graph has **5,637 nodes / 5,000 edges**. Its place aggregation has **1,982 nodes / 4,773 edges**. The same image/place node universe is preserved in every slice; component counts below include singleton isolates.

| Image slice | Edges | Active images | Components | Non-singleton components | Largest component | Maximum degree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 5,000 | 5,637 | 719 | 719 | 3,252 | 13 |
| q95 | 352 | 640 | 5,285 | 288 | 7 | 4 |
| q99 | 77 | 145 | 5,560 | 68 | 5 | 3 |
| q95, >=500 m | 340 | 622 | 5,297 | 282 | 6 | 4 |
| q99, >=500 m | 72 | 136 | 5,565 | 64 | 4 | 2 |

| Place slice | Edges | Active places | Components | Non-singleton components | Largest component | Maximum degree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 4,773 | 1,982 | 1 | 1 | 1,982 | 21 |
| q95 | 349 | 509 | 1,635 | 162 | 29 | 7 |
| q99 | 75 | 130 | 1,908 | 56 | 6 | 4 |
| q95, >=500 m | 337 | 496 | 1,647 | 161 | 28 | 7 |
| q99, >=500 m | 70 | 121 | 1,913 | 52 | 6 | 3 |

Exactly 124 eligible image edges lie in [250,500) m. Of these, 12 are q95 and 5 are q99. The >=500 m sensitivity therefore retains 340/352 q95 edges and 72/77 q99 edges. The strongest overall pair, `London:0000748`–`London:0001609`, has 368 matches / 1,209 minimum keypoints, ratio 0.304384 and bottleneck 1.0, but its 481.6 m distance excludes it from the geo500 slices. Geographic sensitivity changes which links are inspected; it does not change eligibility or the historical measurements.

Repeated high structural evidence occurs for **three place pairs with two q95 image edges each**. No place pair has three q95 supports. One pair meets both the requested independent-support definition and the stricter core-only definition. All these counts survive >=500 m:

| Place pair | q95 supports | Core distinct views, A/B | Minimum distance | Both sides independently supported? |
| --- | ---: | --- | ---: | --- |
| `London:0000885`–`London:0005942` | 2 | 2 / 2 | 2,784.3 m | Yes |
| `London:0000771`–`London:0002053` | 2 | 1 / 2 | 972.6 m | No |
| `London:0001629`–`London:0007799` | 2 | 1 / 2 | 3,420.1 m | No |

These are sparse pilot findings, not ground-truth aliasing labels. The independently supported pair has bottleneck values 0.992051 and 0.978718, with 103 and 90 matches. One of its edges also meets q99. The other two repeated pairs reuse one image on one side.

The denominator audit records an actual ranking change. The 37-match/237-keypoint example (`ac81ba0e...`) has ratio 0.156118 and ratio percentile 1.0, but count percentile and bottleneck **0.643312**. It moves from raw-ratio rank **2** to bottleneck rank **2,135** under the frozen tie rules. The 368-match pair remains first. The top-50 lists overlap on 23 edges:

| Top-50 selection | Median minimum keypoints | Minimum keypoints | Pairs below 256 | Pairs below 512 | Median matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw match ratio | 820 | 177 | 5 | 10 | 91 |
| Structural bottleneck | 1,318 | 765 | 0 | 0 | 121.5 |

This establishes that the specified evidence changes the small-denominator ranking in this pilot. It does not establish semantic precision, a globally optimal evidence definition, or recognition improvement. Every candidate remains in the full graph, and no keypoint gate was introduced.

| SALAD rank bin | Pairs | Median bottleneck | q95 edges / fraction | q99 edges |
| --- | ---: | ---: | --- | ---: |
| 1 | 129 | 0.640000 | 15 / 11.63% | 5 |
| 2–5 | 409 | 0.608462 | 33 / 8.07% | 11 |
| 6–10 | 484 | 0.580256 | 38 / 7.85% | 10 |
| 11–20 | 1,020 | 0.551282 | 65 / 6.37% | 16 |
| 21–50 | 2,958 | 0.560385 | 201 / 6.80% | 35 |

Spearman correlation is **0.106482** between maximum SALAD similarity and bottleneck, and **-0.034826** between best RGB rank and bottleneck. These summaries are descriptive and have no p-values or training interpretation.

The all-candidate coverage q10 is **0.109375**. Exactly one q95 edge falls at or below it (`bf1f118f...`); it remains in the graph as a manual inspection case. Among q95 edges, median symmetric coverage is 0.28125, median symmetric entropy is 0.592568, and median maximum endpoint cell concentration is 0.214286. The snapshot contains full distributions and the pair identity.

The highest image core degree is 4, at `London:0008284`; all four of that image's candidate edges meet q95, so its normalized fraction is 1.0. The highest place core degree is 7, at `Boston:0002144`, from 10 candidate place-pair opportunities. Initial original-RGB inspection of the London image hub shows recurring facade windows, white ground floors and railings in its strongest links. The Boston place hub's strongest links include broad intersections, road markings, trees and vehicles. The largest 29-place component shows repeated terraced-facade motifs in its displayed representatives. These observations warrant architectural and generic-content review; the scalar summaries do not identify which image regions caused the matches. The independently supported pair also contains roads and parked vehicles, so two-sided view repetition alone does not settle the architectural interpretation.

All eight required figures were rendered and curated. Their plot labels and first/last montage panels were visually inspected. Figure source validation verified encoded and decoded RGB hashes for **169 distinct displayed images**. An independent NumPy/pandas audit reproduced all four null calibrations and structural formulas. An independent audit verified all 4,773 place aggregations and reproduced degrees and component statistics for all ten image/place slices using adjacency lists and breadth-first traversal, agreeing with the union-find snapshot.

`python -m compileall countermine tools` passed, and `python -m unittest discover -s tests -v` passed **147 tests**. An independent before/after SHA256 audit confirmed **10,027 preserved files unchanged**, including all 10,017 Step 2A runtime files, with no additions or removals inside that runtime directory. The Step 2A snapshot SHA256 remains `d6cd658cf6b919eba57b4940e373be7831b7280e7236afc66ca9d255102edc43`. Runtime logs and the validation summary are saved under `cache/countermine_rgb/step2b/`. No commit or push was performed.

## Human scientific decision

After the pilot finishes, inspect whether bottleneck ranking reduces obvious small-denominator-only top-ranking cases; whether q95/q99 contain meaningful real-RGB structural counterexamples; whether place confusion repeats; whether at least two distinct **core-support views on both place sides** support those findings; and whether the topology and repeated support survive the >=500 m sensitivity.

Review whether the strongest links, hubs and components show architectural confusion across facades, windows, arches, columns, balconies, rooflines or road/building layout. Check whether generic vegetation, roads, repetitive textures or detector artifacts instead explain the evidence. Degree and percentile values do not answer these semantic questions automatically.

Only a convincing human pilot review should precede scaling local structural measurement to the full eligible candidate population. The future training mapping remains unselected. A later first training experiment, if authorized after scientific validation, should preserve the original SALAD loss and change pair exposure/sampling only. Step 2B supplies no training implementation or claimed performance improvement.

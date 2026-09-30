# Step 1C: RGB candidate negatives

Run from the repository root after building the mini manifest and SALAD descriptor index:

```bash
python tools/04_mine_rgb_candidates.py
python tools/05_summarize_rgb_candidates.py
python tools/06_visualize_rgb_candidates.py --dataset-root data/gsv-cities
```

Mining defaults are Top-50, 128 query descriptors per block, 1,024 reference descriptors per block, seed 42, and `--device auto` (CUDA when available, otherwise CPU). The first diagnostic run uses `--min-geo-distance-m 0`: no geographic exclusion is applied. Same-place exclusion is always mandatory.

`--manifest`, `--descriptors`, and `--output-dir` select alternative mining paths. The default outputs are `cache/gsv_mini/rgb_candidates_raw.csv` and `cache/gsv_mini/rgb_candidates_raw_summary.json`. The summary saves exact similarity/distance quantiles, geographic threshold counts, same-city fraction, requested and effective chunk sizes, dot-product tile shape, seed, device, CPU Torch thread count, manifest SHA256, descriptor path/shape/dtype, software versions, query/pair counts, git commit, and peak CUDA allocation. Effective chunk sizes are capped at `N-1`. The descriptor file is memory-mapped; its contents are never copied as a complete bank.

`ChunkedCandidateMiner` checks that manifest `row_index` values are exactly `0..N-1` in descriptor order. It computes float32 cosine dot products from the existing normalized RGB descriptors in query/reference blocks, excludes same-place pairs and optional geographic neighbors, and merges each reference block into a running Top-K. Candidate order is similarity descending, then negative row index ascending; queries remain in ascending row order. Vectorized haversine distance supplies geographic filtering and output distances. Unordered image-ID pairs receive a stable SHA256 `pair_uid`.

Dot products use fixed internal 32-query by 128-reference tiles anchored to global row indices. Each pair retains the same tile position, matrix shape, and strides when outer chunk sizes change, preserving identical scores and candidates for the same backend configuration. CPU and CUDA results are not guaranteed to be bitwise identical.

The mining CLI checks each complete query chunk before writing: same-place exclusion, ranks exactly `1..K`, finite similarities, nonempty pair IDs, unique directed pairs, valid row ranges, and manifest-aligned metadata. It streams to a temporary CSV and publishes only after all queries and statistics pass. Failed retrieval or validation removes temporary files and leaves previous published outputs intact. Descriptor and similarity memory is bounded by chunk sizes, with no full similarity matrix. `.iter_candidates()` is the bounded-output API; `.mine()` is a convenience API that holds the resulting candidate table in RAM.

The CPU-only summarizer streams the raw CSV. Quantiles use writable temporary memory maps; directed-pair uniqueness, unique undirected IDs, and reverse pairs use a temporary SQLite index. Reports include strict distance thresholds `<150`, `<200`, `<250`, `<500`, and `<1000` meters. A reverse duplicated pair means both directed rows `(a,b)` and `(b,a)` occur; the reported row fraction is twice the mutual undirected pair count divided by all directed candidate rows.

The Pillow-only visualizer streams Top-30 selection by similarity descending, query row ascending, then negative row ascending. It resolves images through manifest `row_index` and `relative_path`, resizes copies in memory, and labels each Query/Negative row with rank, similarity, distance, and both place IDs. Source images remain unchanged. Its default output is `outputs/step1c/top30_rgb_candidates.jpg`; `outputs/` is ignored by Git. The curated figure from the existing diagnostic run is retained at [docs/audits/step1c_top30_rgb_candidates.jpg](audits/step1c_top30_rgb_candidates.jpg).

This stage generates candidate negatives only from real RGB images. It does not run relighting, local matching, CounterMine scoring, or training. No retrieval or recognition performance improvement is implied by the candidate table.

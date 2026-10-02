# CounterMineVPR

**CounterMine: Mining Structural Counterexamples for Visual Place Recognition**

CounterMineVPR investigates whether different-place negatives retrieved by a frozen SALAD model contain more local structural similarity than matched random different-place controls. The active method uses original real RGB images:

```text
Real RGB
  -> SALAD global candidate retrieval
  -> ALIKED + LightGlue real-RGB structural analysis
  -> structural confusion mining
  -> future CounterMine graph
  -> future pair-aware training
```

## Research status

| Stage | Status |
| --- | --- |
| Step 1: RGB candidate retrieval (1A–1C) | Completed |
| Step 2A: real-RGB structural-confusion audit | Current |
| Step 2B: CounterMine graph construction | Not yet implemented |
| Step 3: training | Not yet implemented |

Step 2A measures LightGlue matches divided by the smaller endpoint keypoint count. Candidate pairs must have different place IDs and be at least 250 m apart. A deterministic 5,000-pair pilot is sampled from the existing Top-50 retrieval pool after undirected canonicalization. Each candidate is compared with a random negative sharing its query anchor, city relation, and, for same-city pairs, geographic distance bin. Controls exclude the anchor's entire saved Top-50 list.

Coverage, match concentration, entropy, and empirical null percentiles are descriptive diagnostics. Different places are not assumed to share a global geometric transform. No final CounterMine score, graph threshold, or training objective has been defined. Scientific review must establish the structural-confusion hypothesis before graph construction or training changes. No recognition performance improvement is claimed.

## Running Step 2A

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

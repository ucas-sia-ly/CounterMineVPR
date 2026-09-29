# CounterMineVPR

CounterMineVPR studies condition-centered hard negatives in visual place recognition through diagnostic counterfactual probes.

Status: research prototype.

## Repository structure

```text
countermine/
  probe/      IC-Light adapter, canonical image preprocessing, ALIKED + LightGlue matching, structural fidelity
  mining/     candidate generation, random-pair null statistics, condition-centered confusion gain, CounterMine score and graph
  training/   pair-aware sampler, counterfactual-aware miner, counterfactual MultiSimilarity loss
  utils/      shared project utilities
tools/        executable experiment scripts
configs/      experiment configurations
tests/        small unit and smoke tests
docs/         method and experiment documentation
```

Synthetic images are probe-only and never used for VPR optimization.

Upstream repositories used: SALAD, IC-Light, LightGlue, AdaptVPR.

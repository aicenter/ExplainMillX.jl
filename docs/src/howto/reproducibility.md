# [Make explanations reproducible](@id howto-reproducibility)

`ShapleyExplainer` (and any `AbstractHeuristic` scoring strategy) is a
Monte Carlo method: it repeatedly samples random subsets and estimates
importance from the results. Without a fixed random number generator, two
calls to `explain` on the same sample can produce different (but equally
valid) minimal subsets -- see
[Why explanations aren't unique](../explanation/non_uniqueness.md) for why
this is expected, not a bug.

## Pin the RNG

Pass an explicit `rng` to get the same result every time:

```julia
using Random

result1 = explain(ds, model; rng=MersenneTwister(1))
result2 = explain(ds, model; rng=MersenneTwister(1))
# result1 and result2 select the same subset
```

Without `rng`, `explain`/`explainf` use `Random.default_rng()` (the task's
global RNG), which is why two runs in the same session without an
explicit `rng` can still differ -- each call advances the shared state.

## What's *not* affected by `rng`

The precondition/postcondition checks inside `prune!` (the full sample
must already satisfy the objective; pruning must succeed) don't depend on
randomness -- if pruning fails, it fails deterministically for the given
tolerance, not intermittently.

## A note on exact reproducibility across Julia/library versions

Pinning `rng` guarantees the same result *for a fixed version* of
ExplainMillX and its dependencies. It does not guarantee byte-identical
results forever -- a future change to, say, the order local-search
primitives evaluate tied candidates could shift which of several
equally-valid minimal subsets gets returned, even with the same seed. If
you need a specific explanation to remain stable for auditing/comparison
purposes, save the resulting mask (or its JSON reconstruction) rather than
relying on being able to regenerate byte-identical output indefinitely.

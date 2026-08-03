# [Choose a pruning strategy](@id howto-choose-pruning-strategy)

`explain`/`explainf` expose four independent choices. This guide is
practical guidance on picking values; see
[How scoring and pruning fit together](../explanation/scoring_and_pruning.md)
for why these axes exist and what each is actually doing.

## Order: `HeuristicOrder` vs `GreedyForward`

```julia
explain(ds, model)                          # default: HeuristicOrder, built from scoring automatically
explain(ds, model; order=GreedyForward())   # no scoring-based order; pure stepwise search
```

- **`HeuristicOrder`** (the default whenever `order` is left as `nothing`)
  uses the importance scores `ShapleyExplainer` already computed, sorting
  candidates and adding them via bisection -- `O(log n)` evaluations of
  the objective. Use this unless you have a specific reason not to; it's
  the faster option and reuses work already done during scoring.
- **`GreedyForward`** ignores any precomputed score and, at each step,
  actually tries every remaining candidate and keeps the best one --
  `O(n)` evaluations per step. Slower, but doesn't depend on the scoring
  strategy's importance estimates being accurate. Worth trying if
  `HeuristicOrder`'s result looks worse than expected, or if you're using
  a scoring strategy whose scores you don't fully trust for ordering.

## Granularity: `levelbylevel`

```julia
explain(ds, model; levelbylevel=true)    # default: one hierarchy depth at a time
explain(ds, model; levelbylevel=false)   # the whole tree at once
```

`levelbylevel=true` is generally faster in practice for hierarchical
samples (bags of bags, nested objects) since each level's search operates
over a much smaller candidate set than the whole tree at once. Set it to
`false` if you want the search to consider trade-offs across levels
simultaneously rather than committing to shallower levels before deeper
ones are decided.

## Post-passes: `random_removal` and `finetune`

```julia
explain(ds, model; random_removal=true, finetune=true)   # both on (default)
```

Both default to `true` and both only ever make the result *smaller or
equal* -- they never undo a valid explanation, only look for redundancy
the main search missed. There's rarely a reason to turn either off; doing
so trades a small amount of speed for a possibly-larger (but still valid)
explanation.

- `random_removal`: repeatedly tries removing currently-kept items in
  random order, keeping the removal if the objective still holds.
- `finetune`: a small local-search pass (alternating batched add/remove)
  that catches local-optimum artifacts the single-pass search can leave
  behind.

## A reasonable starting point

The defaults (`HeuristicOrder`, `levelbylevel=true`, both post-passes on)
are a reasonable starting point for most samples. Reach for
`GreedyForward` or `levelbylevel=false` only if you have a specific reason
to suspect the default search order or granularity is producing a poor
result for your data.

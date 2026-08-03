# [Speed up explanation of large samples](@id howto-performance)

## The Monte Carlo sample count dominates cost

`ShapleyExplainer(n)`'s `n` controls how many random subsets are
evaluated during scoring -- this is the main cost knob:

```julia
explain(ds, model; scorer=ShapleyExplainer(150))   # faster, less precise scores
explain(ds, model; scorer=ShapleyExplainer(1000))  # slower, more precise scores
```

The default (`300`) is a reasonable starting point; reduce it for quick
iteration during development, increase it if pruning seems to be
converging on a noticeably suboptimal (larger-than-expected) subset.

## Pruning granularity

`levelbylevel=true` (the default) is generally faster than
`levelbylevel=false` for hierarchical samples, since each level's search
works over a much smaller candidate set than the whole tree at once. See
[Choose a pruning strategy](choose_pruning_strategy.md).

## Metadata is already handled for you

If your sample was extracted with `store_input=Val(true)` (needed for
[JSON output](json_output.md)), you do **not** need to strip it yourself
before calling `explain`/`explainf` for performance: both already do this
internally (`Mill.dropmeta`) before the repeated scoring/pruning
evaluations, since the model never reads `.metadata` and applying a mask
to a sample carrying metadata is measurably slower for no benefit during
the search. The final result still carries the original, metadata-intact
sample (`result.sample`).

## Post-passes add evaluations, not asymptotic cost

`random_removal`/`finetune` (both on by default) add extra objective
evaluations on top of the main search, proportional to the number of
items already kept -- turning them off trades a small amount of
robustness (a possibly-larger explanation) for a modest speedup. Usually
not worth disabling unless you're doing many repeated explanations in a
tight loop and have already validated the main search's result quality
without them.

## If you're explaining many samples

There's currently no batched/vectorized explanation API -- each call to
`explain`/`explainf` handles one sample, and each sample's explanation is
independent of the others. Parallelizing across samples yourself (e.g.
distributed workers, or `Threads.@threads` with each thread using its own
`rng`) is a reasonable approach, though this hasn't been specifically
tested for thread-safety within ExplainMillX itself -- if in doubt, start
with process-based parallelism (`Distributed`), which sidesteps any
shared-mutable-state concerns entirely.

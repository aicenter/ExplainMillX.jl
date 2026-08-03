# [Explain a single prediction](@id howto-explain-prediction)

You have a Mill.jl sample `ds` and a trained model `model` whose output is
a softmax-style vector with two or more classes. Call [`explain`](@ref):

```julia
using ExplainMillX

result = explain(ds, model)
```

This explains the model's own predicted class (`argmax` of its softmax
output). To explain a specific class instead -- which must itself have a
nonnegative confidence gap, i.e. be (one of) the model's predicted
class(es) -- pass it as a third positional argument:

```julia
result = explain(ds, model, 2)
```

## Reading the result

```julia
result.mask                # the pruned StructMask
ds[result.mask]             # the pruned Mill sample
result.sample                # == ds, the original (unpruned) sample
result.class                # which class this explains
result.confidence_gap        # confidence gap on the original sample
result.remaining_confidence_gap  # confidence gap on the pruned sample
fraction_kept(result)        # fraction of items that survived pruning
fraction_pruned(result)      # 1 - fraction_kept(result)
n_pruned(result)             # how many items were pruned away
```

`display(result)` prints a short summary:

```
ExplanationResult for class 2:
  kept 11 / 360 items (3.1% kept, 96.9% pruned away)
  confidence gap: 0.8893 -> 0.8043  (required >= 0.8004)
```

## Controlling how much gets pruned

By default `explain` warns once and uses `rel_tol=0.9` (keep the
confidence gap at 90% of its original value). Set your own tolerance
explicitly to silence the warning and control how aggressively pruning
searches:

```julia
explain(ds, model; rel_tol=0.95)   # keep at least 95% of the original confidence gap
explain(ds, model; abs_tol=0.05)   # keep within 0.05 of the original confidence gap
```

Give exactly one of `abs_tol`/`rel_tol`. See
[Confidence gap and tolerance](../explanation/confidence_gap.md) for what
these actually mean.

## Other keyword arguments

```julia
explain(ds, model;
    scorer=ShapleyExplainer(500),   # more Monte Carlo samples, slower but more precise
    order=GreedyForward(),           # a different pruning search order
    levelbylevel=false,              # search the whole tree at once instead
    random_removal=true,             # redundancy-removal post-pass (default)
    finetune=true,                   # local-search post-pass (default)
    rng=MersenneTwister(1),          # reproducible scoring, see the reproducibility guide
)
```

See [Choose a pruning strategy](choose_pruning_strategy.md) for guidance
on `scorer`/`order`/`levelbylevel`, and
[Make explanations reproducible](reproducibility.md) for `rng`.

## If `explain` raises an error

- `"model(ds) has only 1 output(s)"` -- your model doesn't have a
  softmax-style, 2+ class output. See
  [Explain a binary or non-softmax model](explain_nonsoftmax.md).
- `"class N has a negative confidence gap"` -- the class you asked for
  isn't (one of) the model's predicted class(es) for this sample; there's
  nothing meaningful to preserve while pruning.

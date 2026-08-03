# [Explain a binary or non-softmax model](@id howto-explain-nonsoftmax)

[`explain`](@ref) requires a softmax-style output with two or more
classes, because it prunes to preserve a *confidence gap* -- the
predicted class's probability minus the runner-up's -- which needs a
runner-up class to exist. A single scalar (sigmoid/binary) output, a
regression target, or any other custom notion of "confidence" doesn't fit
that shape, so `explain` raises a clear error rather than guessing. Use
[`explainf`](@ref) directly instead.

## The difference from `explain`

`explainf` takes the scoring and pruning objectives as plain functions of
the model's *output*, instead of deriving them from a class index:

```julia
explainf(scorer, ds, model, fₛ, fₚ; kwargs...)
```

- `fₛ(o)` -- the scoring objective: how "confident" is output `o`? Used
  while estimating each item's importance.
- `fₚ(o)` -- the pruning objective: is output `o` still acceptable?
  Pruning searches for a small item subset keeping `fₚ(model(ds[mask])) >= 0`.

You build both from whatever "confidence" means for your model.

## Example: a binary sigmoid classifier

```julia
using ExplainMillX

z0 = model(ds)[1]              # a single logit -- no runner-up class to compare against
sgn = sign(z0)                  # which side of the decision boundary
threshold = 0.9 * abs(z0)       # keep at least 90% of the original |logit|

fₛ = o -> sgn * o[1]
fₚ = o -> sgn * o[1] - threshold

result = explainf(ShapleyExplainer(300), ds, model, fₛ, fₚ)
```

`result` is a `NamedTuple` `(mask, n_total, n_kept)` -- not an
`ExplanationResult` (there's no class or confidence-gap concept here for
it to report). Apply the mask the same way: `ds[result.mask]`.

## Example: a custom confidence notion

Nothing about `explainf` assumes classification at all -- `fₛ`/`fₚ` can be
built from a regression output, a ranking score, or anything else your
model produces:

```julia
fₛ = o -> -abs(o[1] - target)          # scoring: how close to the original prediction
fₚ = o -> tolerance - abs(o[1] - target)  # pruning: still within tolerance of it

result = explainf(ShapleyExplainer(300), ds, model, fₛ, fₚ)
```

## Keyword arguments

`explainf` takes the same `order`/`levelbylevel`/`random_removal`/`finetune`/`rng`
keywords `explain` forwards to it -- see
[Choose a pruning strategy](choose_pruning_strategy.md) and
[Make explanations reproducible](reproducibility.md).

See the [`explainf`](@ref) docstring, and the
[Mutagenesis tutorial](../generated/mutagenesis.md)'s step-by-step section
for a complete, runnable use of `explainf` (there built from the same
confidence-gap formula `explain` uses internally, since that tutorial's
model has a softmax head -- the binary example above adapts the same
pattern to a single-output model).

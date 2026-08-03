# [Confidence gap and tolerance](@id explanation-confidence-gap)

`explain`'s pruning target is a **confidence gap**, not raw probability,
and the difference matters for what an explanation actually promises you.

## Gap, not probability

For a softmax output, the confidence gap for the predicted class is:

```
gap = probability(predicted class) - probability(runner-up class)
```

This measures *how much the decision could have gone the other way* --
not just "how likely does the winning class look in isolation." A
prediction with 90% probability but a runner-up at 9% has a much larger
gap (and is a much more robust decision) than one with 90% probability and
a runner-up at 88% (a decision that's nearly a coin flip despite the
seemingly-confident-looking probability).

Pruning targets preserving this gap because it's the gap, not the raw
probability, that reflects whether the model's decision is actually
robust. A subset that keeps the winning probability high but lets the
runner-up catch up hasn't really preserved the decision -- it's made it
fragile.

## Tolerance: `rel_tol` and `abs_tol`

The threshold pruning must keep the gap above is derived from exactly one
of two tolerances:

- **`rel_tol`** (a fraction in `[0, 1]`): keep the gap at least that
  *fraction* of its original value. `rel_tol=0.9` means "the decision can
  become up to 10% less robust, but no more."
- **`abs_tol`**: keep the gap within that *absolute amount* of its
  original value. `abs_tol=0.05` means "the gap can shrink by at most
  0.05."

Neither is more "correct" -- they answer different questions. `rel_tol`
scales with how confident the original prediction was (a tight-margin
prediction gets a tight-margin explanation); `abs_tol` applies the same
absolute slack regardless of how confident the original prediction was.
If you don't specify either, `explain` warns once and defaults to
`rel_tol=0.9`.

## Why the full sample must already satisfy the tolerance

`explain`/`explainf` check, before searching at all, that the *complete,
unpruned* sample satisfies the requested tolerance -- trivially true for
`rel_tol` (shrinking the gap by a fraction of itself can't fail on the
full sample) but a real constraint for `abs_tol` if you ask for more slack
than the original gap actually has. This is deliberate: if the request is
unsatisfiable even with the whole sample present, that's a configuration
error to surface immediately, not something pruning should struggle to
approximate.

## What this looks like for a non-classification objective

[`explainf`](@ref) generalizes this beyond classification: you supply the
"is this still acceptable" check (`fₚ`) directly, so a confidence gap is
just the classification-specific instance of a broader idea -- "the
model's output, evaluated some way, must stay within some tolerance of its
original value." See
[Explain a binary or non-softmax model](../howto/explain_nonsoftmax.md)
for what this looks like for a regression target or a single-output model.

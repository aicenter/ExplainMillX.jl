# [Why explanations aren't unique](@id explanation-non-uniqueness)

Two honest facts about this method are worth understanding before you
treat any single explanation as *the* explanation for a prediction.

## Minimal sufficient subsets usually aren't unique

When a model has redundant or correlated evidence -- three fields that
each independently suggest the same class, say -- there can be several
different small subsets that are each, on their own, sufficient to
preserve the prediction. Pruning finds *one* such subset, not the
canonical one, because in general there isn't a canonical one. Running
the same explanation with a different pruning strategy
(see [Choose a pruning strategy](../howto/choose_pruning_strategy.md)), or
even the same strategy from a different starting order, can legitimately
land on a different, equally valid minimal subset for the same prediction.

This means "atom 7 wasn't in the explanation" doesn't mean atom 7 was
irrelevant -- it might mean atom 7's evidence was redundant with something
else that got kept instead. Two people running slightly different searches
on the same molecule could walk away with different, both-correct answers
to "what mattered here."

## Scoring is stochastic, so results can vary run to run

`ShapleyExplainer`'s Monte Carlo scoring makes random choices as part of
estimating importance, which feeds into pruning's search order. Without
pinning the random number generator (see
[Make explanations reproducible](../howto/reproducibility.md)), re-running
`explain` on the same sample can produce a different (but still valid --
still checked to satisfy the tolerance) subset, simply because scoring
happened to explore the space of possibilities in a different order.

## What this means in practice

Neither of these is a bug to be fixed -- they're properties of the
problem itself (searching for *a* small sufficient subset, not *the*
unique one) and of the method used to estimate importance efficiently
(random sampling rather than exhaustive evaluation). The practical
implications:

- Don't over-interpret the *absence* of an item from an explanation as
  proof it was unimportant -- it may simply have been redundant with
  what was kept.
- If you need a specific explanation to stay stable for comparison or
  audit purposes, pin `rng` and keep a record of the strategy settings
  used, rather than assuming re-running later will reproduce it.
- If two explanations of very similar samples look surprisingly
  different, that's not necessarily inconsistency in the model -- it can
  be two different valid answers to "what's sufficient here," especially
  when the underlying evidence is redundant.

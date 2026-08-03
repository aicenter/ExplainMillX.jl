# [How scoring and pruning fit together](@id explanation-scoring-and-pruning)

Finding a minimal sufficient subset (see
[What "explaining a prediction" means here](what_is_explanation.md))
happens in two genuinely separate stages, and understanding the division
of labor between them explains why the library has both a `scorer` and a
pruning strategy as independent choices.

## Stage 1: scoring -- a rough, fast importance estimate

A scoring strategy (e.g. [`ShapleyExplainer`](@ref)) repeatedly makes a
*random guess* at which parts of the sample to keep, evaluates the model
on that guess, and tracks whether keeping vs. dropping each item tended to
correlate with the model staying confident. After many random trials, this
produces a per-item importance score -- a Monte Carlo estimate of Shapley
value, if you're familiar with that concept: roughly, "how much did this
item tend to matter, averaged over many different contexts of what else
was present."

This is fast (hundreds of model evaluations, not an exhaustive search) but
approximate, and it never actually *finds* a valid explanation by itself
-- it only ranks items by how promising they seem.

## Stage 2: pruning -- a search that's actually verified to work

Pruning takes that ranking and *searches* for a small subset, checking the
model's real output at each candidate subset it tries. Unlike scoring,
pruning's result is never taken on faith: the library's own internal
contract requires that whatever pruning returns actually satisfies the
tolerance you asked for (see
[Confidence gap and tolerance](confidence_gap.md)) -- if it can't, that's
treated as an error, not a silently-wrong result.

The scoring stage's ranking is only ever a *search order* for pruning --
it tells pruning which items to try adding first, so a good search
converges quickly. If you don't trust the scoring ranking for some reason,
`explain`'s `order=GreedyForward()` option ignores it entirely and searches
by trying every candidate at each step instead (see
[Choose a pruning strategy](../howto/choose_pruning_strategy.md)) -- slower,
but not dependent on the scoring estimate being any good.

## Why this split exists

Keeping scoring and pruning as separate, swappable stages means neither
one has to be perfect on its own. Scoring can be a cheap approximation
because pruning double-checks its suggestions against the real model.
Pruning can search efficiently because scoring hands it a reasonable
starting order instead of trying candidates in arbitrary sequence. Neither
stage assumes anything about *why* an item matters -- both operate purely
by asking "what does the model actually output if I mask this" -- which is
also why the same two-stage design works unchanged for classification,
regression, or any other notion of "confidence" you can express as a
function of the model's output (see
[Explain a binary or non-softmax model](../howto/explain_nonsoftmax.md)).

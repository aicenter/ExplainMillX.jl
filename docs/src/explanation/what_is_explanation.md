# [What "explaining a prediction" means here](@id explanation-what-is-explanation)

If you've used feature-attribution tools like SHAP or LIME, ExplainMillX
is answering a genuinely different question, and it's worth recalibrating
before reading the rest of this documentation.

## Attribution vs. sufficiency

Feature-attribution methods assign every feature an importance *score*:
"feature A contributed +0.3, feature B contributed -0.1," across *all*
features, all the time. The output is a ranking or a weighting -- nothing
is actually removed, and there's no single feature or subset singled out
as "the explanation."

ExplainMillX instead searches for a **minimal sufficient subset**: the
smallest part of the sample that, on its own, still gets the same
decision from the model within a chosen tolerance. The output isn't a
score per item -- it's an actual subset: these atoms, this field, these
instances in the bag, and nothing else, was enough.

Concretely: given a molecule the model classifies as mutagenic, ExplainMillX
doesn't tell you "atom 7 contributed 0.2 to the mutagenic score." It tells
you "the model would have made the same call using only atom 7's element
and charge, plus these two scalar descriptors -- everything else in the
molecule was unnecessary for this particular decision."

## Why this matters for how you read a result

A minimal sufficient subset is a *sanity check on necessity*, not a
ranking of importance. Something being pruned away doesn't mean it was
unimportant in general -- it means it wasn't *necessary*, given everything
else that was kept. Two different, non-overlapping subsets of the same
sample can both be genuinely sufficient (see
[Why explanations aren't unique](non_uniqueness.md)) -- there usually
isn't one true minimal explanation, just *a* small one that works.

## How this connects to the rest of the library

Finding that minimal subset happens in two stages, covered next in
[How scoring and pruning fit together](scoring_and_pruning.md):
first, every part of the sample gets a rough importance estimate (scoring);
then, that estimate guides a search for a small subset that's actually
verified to be sufficient (pruning) -- the estimate is only ever a search
heuristic, never taken on faith. What "still gets the same decision" means
precisely is covered in
[Confidence gap and tolerance](confidence_gap.md).

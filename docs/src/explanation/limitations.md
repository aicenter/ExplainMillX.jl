# [Limitations of perturbation-based explanation](@id explanation-limitations)

ExplainMillX judges importance and sufficiency entirely by *removing parts
of the sample and re-running the model* -- scoring masks random subsets,
pruning tests candidate subsets, and everything is decided by what the
model outputs on these partial inputs. This is a genuinely useful and
widely-used approach (the same family as occlusion-based explanation
methods generally), but it has real limits worth knowing rather than
discovering by surprise.

## Masked inputs can be outside the model's training distribution

A model is trained on complete, realistic samples. A partially-masked
sample -- half a molecule, a record with most fields missing -- is
something the model may never have seen anything like during training.
Its output on such an input reflects how the model happens to
extrapolate off-distribution, which isn't guaranteed to reflect the same
reasoning the model uses on real, complete data. In practice this means:
a score or a pruning decision is a genuine measurement of *this model's
behavior on this masked input*, not necessarily a clean statement about
which real-world features the model "truly" relies on.

Mill.jl's imputation machinery (the model has a learned representation
for "this value is missing," used throughout masking here) mitigates this
somewhat compared to naively zeroing values, but doesn't eliminate the
underlying issue -- missing-ness still wasn't necessarily distributed the
same way at training time as it is during pruning.

## Minimality doesn't mean human-meaningful

The search optimizes for one thing: the smallest subset that keeps the
model's output within tolerance. It has no notion of which subset a
domain expert would consider the "real" reason for a decision. A
technically-minimal subset can exploit a quirk of the model's decision
boundary rather than reflect the semantically obvious explanation --
sufficiency and human-intuitiveness are different objectives, and only
the former is what's actually being searched for.

## The tolerance you choose changes the answer

There's no principled, universal "correct" tolerance -- `rel_tol=0.9` and
`rel_tol=0.99` can produce meaningfully different explanations for the
same prediction (see [Confidence gap and tolerance](confidence_gap.md)),
and which one is "right" depends on how much robustness you actually care
about preserving, not on anything the library can determine for you.

## No aggregate or global view

Every explanation here is for one sample. There's no built-in way to
summarize "what does this model generally rely on across many samples" --
if that's what you need, you'd have to run per-sample explanations across
a dataset and aggregate the results yourself; ExplainMillX doesn't do this
for you.

## None of this makes the method wrong to use

These are the same tradeoffs any perturbation-based explanation method
makes -- they're reasons to read a result as "a genuine, verified-minimal
subset that keeps this model's decision, under this tolerance, on this
input" rather than as a claim about ground-truth causal importance. Used
that way, the method is exactly as reliable as its own definition
promises.

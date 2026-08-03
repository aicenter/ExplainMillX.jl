# ExplainMillX.jl

ExplainMillX.jl explains predictions of hierarchical multi-instance
learning (HMIL) models built with [Mill.jl](https://github.com/CTUAvastLab/Mill.jl)
and [JsonGrinder.jl](https://github.com/CTUAvastLab/JsonGrinder.jl)).
libraries, where the former provides core computational mechanisms and 
the latter interfaces to easy processing of samples. 

ExplainMillX.jl provides **abductive explanations**, which for a given sample,
model, and predicted class, corresponds to 
**minimal subset** of that sample that the model still
classifies the same way, within a chosen tolerance. That subset *is* the
explanation. The explanation should be read "the model decided this because 
of these specific parts; everything else was unnecessary." For example in 
case of running examples on molecules, the explanation can be "because 
 the molecule contains this set of molecules with these bonds.""

The minimal use of explainer with a given model `model` and sample `ds` 
is as follows.
```julia
using ExplainMillX

result = explain(ds, model)     # explain the model's predicted class for `ds`
result.mask                     # the pruned mask
ds[result.mask]                 # the pruned Mill sample
fraction_pruned(result)         # how much of `ds` turned out to be unnecessary
```
A more convenient use is with JsonGrinder.jl library, with which we can return
the explanation as a subset in JSON.jl form
```julia
using JsonGrinder

ds = extractor(sample_json; store_input=Val(true))
result = explain(ds, model)
explain_json(ds, result.mask, extractor)
```
where `extractor` is the JsonGrinder.jl's extractor converting JSON to Mill.jl's
internal structures. To provide the facility of exporting the explanation to JSON, 
we the sample `ds` needs to be exctracted with argument `store_input=Val(true).`

## Where to go next

This documentation is organized around what you're trying to do, not just
what's available:

- **New to ExplainMillX?** Start with the
  [Mutagenesis tutorial](generated/mutagenesis.md) -- a complete, runnable
  walkthrough which download data, train a Mill.jl model, and explain one of
  its predictions, both via the one-call [`explain`](@ref) and step by
  step through the underlying machinery.
- **Have a specific task?** The How-to Guides (starting with
  [Explain a single prediction](howto/explain_prediction.md)) are short,
  practical recipes: explaining a binary classifier, getting JSON output,
  choosing a pruning strategy, and more.
- **Want to understand *why* it works this way?** The Explanation section
  (starting with [What "explaining a prediction" means here](explanation/what_is_explanation.md))
  covers the concepts -- what kind of explanation this actually is, why
  pruning targets a confidence *gap*, why explanations aren't unique, and
  the method's real limitations.
- **Looking up a function or type?** See the [API Reference](reference.md).

## Scope

ExplainMillX explains **one sample at a time**, against models with a
softmax-style, two-or-more-class output for the main [`explain`](@ref)
entry point (binary/sigmoid heads and other objectives are supported via
the lower-level [`explainf`](@ref) -- see the
[non-softmax how-to guide](@ref howto-explain-nonsoftmax)). It does not
currently support clustering correlated features, batching multiple
samples in one call, or gradient-guided pruning search.

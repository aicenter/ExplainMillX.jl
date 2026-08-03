[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/aicenter/ExplainMillX.jl/blob/master/LICENSE.md)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://aicenter.github.io/ExplainMillX.jl/dev)


# ExplainMillX.jl

**Why did your model make that prediction?** ExplainMillX.jl answers that question for
hierarchical multi-instance learning (HMIL) models built with
[Mill.jl](https://github.com/CTUAvastLab/Mill.jl) and
[JsonGrinder.jl](https://github.com/CTUAvastLab/JsonGrinder.jl) — models trained directly
on trees of JSON, XML, or other nested/irregular data, where there's no fixed feature
vector to run SHAP or LIME on.

Given a sample, a model, and a predicted class, ExplainMillX finds the **smallest subset**
of that sample — which fields, which array entries, which items in a bag — that the model
still classifies the same way. That subset *is* the explanation: "the model decided this
because of these specific parts; everything else was unnecessary." For a model classifying
molecules, the explanation might be "because the molecule contains this substructure with
these bonds." For a model scoring JSON-described network events, it might be "because of
this one nested field three levels down."

## Why not just use SHAP or LIME?

Those tools assume a flat, fixed-size input. HMIL models are built precisely for the
opposite case: samples that are trees of dictionaries, arrays of varying length, and
sets ("bags") of instances with no fixed cardinality or order. ExplainMillX's mask
abstraction mirrors that same recursive structure, so it can turn "importance" and
"pruning" into operations directly on the tree — down to individual bag instances,
categorical values, or n-gram features — with no flattening step to lose fidelity.

## Quickstart

```julia
using ExplainMillX

result = explain(ds, model)     # explain the model's predicted class for `ds`
result.mask                     # the pruned mask
ds[result.mask]                 # the pruned Mill sample
fraction_pruned(result)         # how much of `ds` turned out to be unnecessary
```

If `ds` was extracted with JsonGrinder.jl (using `store_input=Val(true)`), the same
explanation can come back out as JSON, with unimportant parts simply absent:

```julia
using JsonGrinder

ds = extractor(sample_json; store_input=Val(true))
result = explain(ds, model)
explain_json(result, extractor)
```

An example of explanation of a sample from Mutagenesis dataset (used in tutorial of this package) 
can look like

```json
{
    "atoms": [
        {
            "charge": 0.812
        },
        {
            "charge": 0.812
        },
        {
            "charge": -0.388,
            "element": "o"
        },
        {
            "atom_type": 27,
            "charge": 0.012,
            "element": "c"
        }
    ],
    "ind1": 1,
    "inda": 0,
    "logp": 4.44
}
```
The explanation show the subset of the sample triggering the correct classification. 


## What's included

- **`explain`/`explainf`** — the top-level pipeline: score every part of the sample for
  importance, then search for the smallest subset that keeps the model's prediction
  confidence within tolerance.
- **A Monte Carlo Shapley-style scorer** (`ShapleyExplainer`) out of the box, with a
  small, explicit interface (`AbstractHeuristic`) for plugging in your own.
- **Several pruning strategies** — flat or level-by-level search, heuristic-ordered or
  greedy-forward, with optional randomized-removal and fine-tuning passes — all
  composable via `PruningStrategy`, so you can trade runtime for explanation quality.
- **JSON-shaped output** (`explain_json`) for anyone using JsonGrinder.jl, so an
  explanation can be inspected as JSON rather than as a Mill.jl mask.

## Learn more

The [documentation](https://ctuavastlab.github.io/ExplainMillX.jl/) has a full
Mutagenesis tutorial (train a model, explain a prediction, both the one-call way and step
by step), task-oriented how-to guides, an explanation of the underlying method and its
limitations, and the complete API reference. Design notes for contributors live in
[`docs/design/`](docs/design/).

## Status

This is a clean-room redesign of
[ExplainMill.jl](https://github.com/CTUAvastLab/ExplainMill.jl). The core pipeline —
masks, scoring, pruning, and JSON output — is implemented and tested; see
[`docs/design/`](docs/design/) for the rationale behind the design and
[`docs/src/explanation/limitations.md`](docs/src/explanation/limitations.md) for what it
doesn't (yet) do.

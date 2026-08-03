# [Get a JSON explanation](@id howto-json-output)

If your sample came from a JsonGrinder.jl extractor, you can reconstruct
the pruned explanation as a JSON-shaped value -- the original field
values (element names, categories, numbers, strings) with pruned parts
represented as `nothing`, instead of raw one-hot/matrix data.

## Requirement: extract with `store_input=Val(true)`

This only works if the sample was extracted with metadata preserved:

```julia
ds = extractor(json_sample; store_input=Val(true))
# or, for a batch:
x = extract(extractor, samples; store_input=Val(true))
```

Without this, every leaf's `.metadata` is `nothing`, and `explain_json`
raises a clear error the first time it needs a value it doesn't have:

```
explain_json: no metadata on a ArrayNode leaf -- re-extract with
`store_input=Val(true)` so explain_json has original values to reconstruct
```

If you forget until after training/scoring, just re-extract the one
sample you want to explain -- `store_input=Val(true)` is not needed for
training data, only for whichever sample(s) you'll later call
`explain_json` on.

## Usage

```julia
using ExplainMillX, JSON

result = explain(ds, model)
explain_json(result, extractor)             # convenience: uses result.sample
explain_json(ds, result.mask, extractor)    # equivalent, spelled out

JSON.print(explain_json(result, extractor), 4)   # pretty-printed JSON string
```

`extractor` must be the same (or a structurally equivalent) extractor
used to produce `ds` -- reconstruction depends on knowing whether a field
was a scalar, a category, an array, etc., which only the extractor knows.

## What the output looks like

Pruned leaves are dropped (not shown as explicit `null`), and a field
that's entirely pruned away disappears from its parent object rather than
appearing as an empty value:

```json
{
    "atoms": [
        {"charge": 0.812},
        {"element": "o", "charge": -0.388}
    ],
    "logp": 4.44
}
```

If *everything* was pruned, `explain_json` returns `nothing` rather than
an empty object.

## Limitations

- One sample at a time -- `ds` must have `numobs(ds) == 1`.
- `PolymorphExtractor` (JsonGrinder's union-typed fields) is not
  supported and raises a clear error.

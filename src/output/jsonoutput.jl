# Converting a pruned StructMask back into a JSON-shaped value, using a
# JsonGrinder extractor and the sample's preserved `.metadata`. See
# doc/design/jsonoutput.md for the full design rationale; this file
# implements it directly.
#
# `PolymorphExtractor` and `ProductNode{<:Tuple}` are explicitly out of
# scope (jsonoutput.md §4.1) -- not supported anywhere else in
# ExplainMillX either.

_metadata_missing_error(ds) = error(
    "explain_json: no metadata on a $(nameof(typeof(ds))) leaf -- re-extract " *
    "with `store_input=Val(true)` so explain_json has original values to " *
    "reconstruct (see doc/design/jsonoutput.md §2)",
)

"""
    _explain_json(ds, mask, extractor) -> Vector

Internal recursive worker. Returns a `Vector` of length `numobs(ds)`: one
JSON-shaped value (or `nothing`, for an absent/pruned observation) per
observation `ds` represents at this level of nesting. `explain_json`
(the public entry point) unwraps the single root observation and runs the
cleanup pass; every recursive case here only ever needs to build the raw,
possibly `nothing`-containing structure.
"""
function _explain_json end

_explain_json(ds::AbstractMillNode, mask::StructMask, e::JsonGrinder.StableExtractor) =
    _explain_json(ds, mask, e.e)

# Dense (`ScalarExtractor`) leaves are masked per *row* (feature), shared
# across every observation -- not per observation like categorical/n-gram
# leaves (masks.md's per-format table). A `ScalarExtractor` always produces
# a single-row `ArrayNode` (JsonGrinder's `_extract_leaf` returns a `1×1`
# matrix), so there is exactly one shared keep/drop decision for the whole
# field, applied uniformly to every observation.
function _explain_json(ds::ArrayNode{<:Matrix}, mask::StructMask, ::JsonGrinder.ScalarExtractor)
    ds.metadata === nothing && _metadata_missing_error(ds)
    size(ds.data, 1) == 1 || error(
        "explain_json: ScalarExtractor with more than one row is not supported",
    )
    keep = mask.own === nothing ? true : only(prunemask(mask))
    [keep ? ds.metadata[i] : nothing for i in 1:numobs(ds)]
end

function _explain_json(ds::ArrayNode{<:Mill.MaybeHotMatrix}, mask::StructMask, ::JsonGrinder.CategoricalExtractor)
    ds.metadata === nothing && _metadata_missing_error(ds)
    keep = mask.own === nothing ? trues(numobs(ds)) : prunemask(mask)
    [keep[i] ? ds.metadata[i] : nothing for i in 1:numobs(ds)]
end

function _explain_json(ds::ArrayNode{<:Mill.NGramMatrix}, mask::StructMask, ::JsonGrinder.NGramExtractor)
    ds.metadata === nothing && _metadata_missing_error(ds)
    keep = mask.own === nothing ? trues(numobs(ds)) : prunemask(mask)
    [keep[i] ? ds.metadata[i] : nothing for i in 1:numobs(ds)]
end

# `BagNode`/`ArrayExtractor`: reconstruct a JSON array per outer
# observation. Decides purely from the bag-level mask -- matching
# `applymask`'s own `BagNode` case exactly (it only ever consults the
# bag-level `own`, never a bottom-up "does this instance have any
# surviving content" check) -- so `explain_json` reports the same set of
# kept instances `ds[mask]` would produce, just rendered as JSON. An
# instance whose bag-level slot survives but whose own fields all end up
# `nothing` is handled by the cleanup pass (`_prunejson`), not here.
function _explain_json(ds::BagNode, mask::StructMask, e::JsonGrinder.ArrayExtractor)
    if ismissing(ds.data)
        return [Any[] for _ in 1:numobs(ds)]
    end
    keep = mask.own === nothing ? trues(numobs(ds.data)) : prunemask(mask)
    child_json = _explain_json(ds.data, mask.children, e.items)
    map(ds.bags) do b
        [child_json[i] for i in b if keep[i]]
    end
end

# `ProductNode`/`DictExtractor`: reconstruct a JSON object per observation.
# Builds the raw per-observation `Dict` including `nothing` values for
# pruned fields -- the cleanup pass (`_prunejson`) is what strips those and
# collapses an entirely-empty result, so this case doesn't need to
# duplicate that logic.
function _explain_json(ds::ProductNode{<:NamedTuple}, mask::StructMask, e::JsonGrinder.DictExtractor)
    ks = keys(ds.data)
    per_field = NamedTuple{Tuple(ks)}(
        Tuple(_explain_json(ds.data[k], mask.children[k], e.children[k]) for k in ks),
    )
    map(1:numobs(ds)) do i
        Dict{Symbol,Any}(k => per_field[k][i] for k in ks)
    end
end

_explain_json(::AbstractMillNode, ::StructMask, e::JsonGrinder.PolymorphExtractor) = error(
    "explain_json: PolymorphExtractor is not supported (deliberately deferred, " *
    "see doc/design/jsonoutput.md §4.1)",
)

_explain_json(::AbstractMillNode, ::StructMask, e) = error(
    "explain_json: extractor type $(typeof(e)) is not supported yet",
)

"""
    _prunejson(x)

Cleanup pass: recursively strips `nothing` values out of `Dict`s and
collapses a `Dict`/`Vector` that ends up empty (after that stripping) to
`nothing` itself, one level up. This is what makes a fully-pruned subtree
disappear from the output entirely, rather than showing up as an explicit
wall of nulls (jsonoutput.md §8).
"""
_prunejson(x) = x

function _prunejson(d::AbstractDict)
    cleaned = Dict(k => _prunejson(v) for (k, v) in d)
    filter!(kv -> kv.second !== nothing, cleaned)
    isempty(cleaned) ? nothing : cleaned
end

function _prunejson(v::AbstractVector)
    cleaned = [_prunejson(x) for x in v]
    filter!(x -> x !== nothing, cleaned)
    isempty(cleaned) ? nothing : cleaned
end

"""
    explain_json(ds::AbstractMillNode, mask::StructMask, extractor) -> Any

Reconstruct a pruned `mask` over `ds` as a JSON-shaped value (nested
`Dict`/`Vector`/scalars, with pruned or absent items represented as
`nothing`), using `extractor` to know how to interpret each part of the
tree and `ds`'s preserved `.metadata` to recover original values.

`ds` must have been extracted with `store_input=Val(true)` (directly, or
via `extract(extractor, samples; store_input=Val(true))`) -- otherwise
there is nothing for this function to reconstruct, and it raises a clear
error the first time it needs a leaf's metadata and finds `nothing`
instead.

`extractor` must be the *same* extractor object used to produce `ds`
(structurally): reconstruction dispatches on the extractor's type at every
level, since the extractor -- not the Mill node type alone -- determines
the JSON shape (e.g. whether a `ProductNode` was a plain object or, in
general, something else; whether an `ArrayNode` came from a categorical,
n-gram, or scalar field).

`ds` must represent a single sample (`numobs(ds) == 1`) -- `explain_json`
has no batching support (`doc/design/jsonoutput.md` §5), matching the rest
of `ExplainMillX`.

`PolymorphExtractor` and `ProductNode{<:Tuple}` (JsonGrinder's
union-typed/positional extraction) are not supported.

# Example
```julia
ds = e(json_sample; store_input=Val(true))
result = explain(ds, model)   # default scorer, ShapleyExplainer(300)
explain_json(result, e)   # => a JSON-shaped Dict, pruned parts as `nothing`/absent
explain_json(ds, result.mask, e)   # equivalent, spelled out
```
"""
function explain_json(ds::AbstractMillNode, mask::StructMask, extractor)
    raw = only(_explain_json(ds, mask, extractor))
    _prunejson(raw)
end

"""
    explain_json(result::ExplanationResult, extractor) -> Any

Convenience form of [`explain_json`](@ref) for the common case: `result`
already carries the sample it was computed for (`result.sample`), so this
is exactly `explain_json(result.sample, result.mask, extractor)`.
"""
explain_json(result::ExplanationResult, extractor) =
    explain_json(result.sample, result.mask, extractor)

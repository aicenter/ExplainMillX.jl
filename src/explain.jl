# Top-level convenience entry points tying scoring (`stats`) and pruning
# (`prune!`) together for the common case: explaining a single Mill.jl
# sample's predicted class under a softmax/multi-class model. See
# `doc/design/design.doc` §4.1 for the objective-injection design this is
# built on, and `examples/mutagenesis.jl` for a from-scratch worked example
# of wiring these pieces by hand (useful background for anyone modifying
# this file, and the reason this file exists at all).
#
# `explain`/`explainf` intentionally do not support:
#   - clustering (masks.md §7 -- not yet designed anywhere in the library)
#   - converting the final mask back into a JSON-shaped value via a
#     JsonGrinder extractor (no `e2boolean`/`yarason` equivalent exists yet)
#   - models with a single scalar (sigmoid/binary) output -- `explain`
#     assumes a softmax-style, ≥2-class output and raises a clear error
#     otherwise; a binary head needs a hand-written objective, see
#     `explainf` and `examples/mutagenesis.jl`.

_softmax(x::AbstractVector{<:Real}) = (w = exp.(x .- maximum(x)); w ./ sum(w))

_confgap(p::AbstractVector{<:Real}, class::Integer) =
    p[class] - maximum(@view p[1:end.!=class])

function _threshold(cg::Real, abs_tol, rel_tol)
    if abs_tol === nothing && rel_tol === nothing
        @warn "explain: no tolerance specified, defaulting to rel_tol=0.9" maxlog = 5
        rel_tol = 0.9
    end
    if abs_tol !== nothing
        abs_tol <= cg || error(
            "explain: abs_tol=$abs_tol must not exceed the confidence gap $cg",
        )
        return cg - abs_tol
    end
    0 <= rel_tol <= 1 || error("explain: rel_tol=$rel_tol must be in [0, 1]")
    rel_tol * cg
end

"""
    ExplanationResult

The result of [`explain`](@ref): the pruned mask, the sample it applies
to, and a summary of how much of the sample was pruned away and how much
prediction confidence was retained.

# Fields
- `mask::StructMask`: the final, pruned mask. Apply it with `sample[mask]`
  to get the pruned Mill sample, or read
  `prunemask.(first.(collectmasks(mask)))` for the raw per-item keep/drop
  decisions.
- `sample::AbstractMillNode`: the (original, un-pruned) sample this
  explanation is for -- i.e. exactly the `ds` passed to `explain`. Stored
  alongside `mask` so an `ExplanationResult` is self-contained: `mask`
  alone means nothing without knowing what it applies to (e.g. for
  `explain_json`, or if `sample` was extracted with `store_input=Val(true)`
  and you want its metadata later without having to keep `ds` around
  separately).
- `class::Int`: the class index this explanation was computed for.
- `confidence_gap::Float64`: the model's confidence gap for `class` on the
  *original*, unpruned sample (softmax probability of `class` minus the
  highest probability among all other classes).
- `remaining_confidence_gap::Float64`: the same confidence gap evaluated
  on the *pruned* sample `sample[mask]`. Always `>= threshold`.
- `threshold::Float64`: the minimum confidence gap pruning was required to
  preserve, derived from `abs_tol`/`rel_tol` (see [`explain`](@ref)).
- `n_total::Int`: total number of maskable items in the sample (features,
  bag instances, categorical values, ... -- see `doc/design/masks.md` §3).
- `n_kept::Int`: how many of those items survived pruning.

Use [`fraction_kept`](@ref)/[`fraction_pruned`](@ref)/[`n_pruned`](@ref)
for derived statistics.
"""
struct ExplanationResult{M<:StructMask,D<:AbstractMillNode}
    mask::M
    sample::D
    class::Int
    confidence_gap::Float64
    remaining_confidence_gap::Float64
    threshold::Float64
    n_total::Int
    n_kept::Int
end

"""
    n_pruned(r::ExplanationResult) -> Int

Number of items pruned away, i.e. `r.n_total - r.n_kept`.
"""
n_pruned(r::ExplanationResult) = r.n_total - r.n_kept

"""
    fraction_kept(r::ExplanationResult) -> Float64

Fraction (in `[0, 1]`) of the sample's maskable items that survived
pruning. `1.0` for a sample with no maskable items at all.
"""
fraction_kept(r::ExplanationResult) = r.n_total == 0 ? 1.0 : r.n_kept / r.n_total

"""
    fraction_pruned(r::ExplanationResult) -> Float64

`1 - fraction_kept(r)`: the fraction of the sample's maskable items that
were pruned away.
"""
fraction_pruned(r::ExplanationResult) = 1 - fraction_kept(r)

function Base.show(io::IO, ::MIME"text/plain", r::ExplanationResult)
    println(io, "ExplanationResult for class ", r.class, ":")
    println(io, "  kept ", r.n_kept, " / ", r.n_total, " items (",
        round(100 * fraction_kept(r); digits=1), "% kept, ",
        round(100 * fraction_pruned(r); digits=1), "% pruned away)")
    print(io, "  confidence gap: ", round(r.confidence_gap; digits=4),
        " -> ", round(r.remaining_confidence_gap; digits=4),
        "  (required >= ", round(r.threshold; digits=4), ")")
end

"""
    explainf(scorer, ds, model, fₛ, fₚ; kwargs...) -> (mask=..., n_total=..., n_kept=...)

Low-level entry point: explain `ds` under `model` using scoring strategy
`scorer` and caller-supplied objectives, bypassing the classification
convenience layer entirely (`doc/design/design.doc` §4.1). Use this
directly for anything [`explain`](@ref) doesn't cover -- e.g. a binary
sigmoid head, a regression target, or a custom notion of "confidence"
(see `examples/mutagenesis.jl` for a full worked example of exactly this).

`fₛ(o)` and `fₚ(o)` are both functions of the model's *output* `o`
(matching `stats`'s `objective` argument), not zero-argument closures --
`explainf` builds the mask via `stats` first and only then has something
for a zero-argument `f() = fₚ(model(ds[mask]))` to close over, which is
what `prune!` actually requires.

`ds` is stripped of metadata internally (`Mill.dropmeta`) before scoring
and pruning: `stats`/`prune!` re-evaluate `model(ds[mask])` hundreds of
times, the model never reads `.metadata` (it isn't part of `.data`), and
applying a mask to a sample with metadata is measurably slower (`applymask`
on a `BagNode`/`ProductNode` re-slices `.metadata` on every call via
Mill's own `getindex`, roughly 2x more work per call, confirmed by
benchmark) for zero benefit during the hot loop. The returned `mask` is
purely shape-derived and applies identically to a metadata-carrying `ds`
afterward -- `explain` does exactly that for its final result.

# Keyword arguments
- `order::Union{HeuristicOrder,GreedyForward,Nothing} = nothing`: the
  pruning search order (`doc/design/pruning.md` §4). `nothing` (the
  default) builds a `HeuristicOrder` automatically from the scoring
  result via `nodescores`; pass `GreedyForward()` to use stepwise greedy
  selection instead, or a pre-built `HeuristicOrder` to reuse scores from
  elsewhere.
- `levelbylevel::Bool = true`: search one hierarchy depth at a time
  (generally faster in practice, `doc/design/pruning.md` §2.2) rather than
  the whole tree at once.
- `random_removal::Bool = true`, `finetune::Bool = true`: optional
  redundancy-removal and local-search post-passes (`doc/design/pruning.md`
  §4.5, §4.6).
- `rng::AbstractRNG = Random.default_rng()`: random number generator
  threaded through the (stochastic) scoring strategy, for reproducibility.

Returns a `NamedTuple` `(mask, n_total, n_kept)` -- see
[`ExplanationResult`](@ref) for what these mean; `explainf` doesn't wrap
them in that struct itself since it has no class/confidence-gap notion to
report.
"""
function explainf(scorer::AbstractHeuristic, ds::AbstractMillNode, model, fₛ, fₚ;
    order::Union{HeuristicOrder,GreedyForward,Nothing}=nothing,
    levelbylevel::Bool=true,
    random_removal::Bool=true,
    finetune::Bool=true,
    rng::AbstractRNG=Random.default_rng())
    ds = Mill.dropmeta(ds)   # see docstring: pure performance optimization,
    # the resulting mask is valid against a metadata-carrying `ds` too
    mask, acctree = stats(scorer, ds, model, fₛ; rng)

    pruning_order = order
    if pruning_order === nothing
        scores = nodescores(mask, acctree, score)
        pruning_order = HeuristicOrder(scores)
    end
    strategy = PruningStrategy(pruning_order, levelbylevel, random_removal, finetune)

    nodes = first.(collectmasks(mask))
    n_total = sum(length, nodes; init=0)

    f = () -> fₚ(model(ds[mask]))
    prune!(mask, ds, model, f, strategy)

    n_kept = sum(count, prunemask.(nodes); init=0)
    (mask=mask, n_total=n_total, n_kept=n_kept)
end

"""
    explain(ds, model, class; kwargs...) -> ExplanationResult
    explain(ds, model; kwargs...) -> ExplanationResult

Explain why `model` classifies sample `ds` as `class`, by finding a small
subset of `ds`'s items that keeps the model's confidence in `class`
within tolerance of its original value.

If `class` is omitted, the model's own predicted class (`argmax` of its
softmax output on the full sample) is used.

This is the full pipeline in one call:
1. **Predict**: evaluate `model(ds)` to get the baseline confidence gap
   for `class` (its softmax probability minus the runner-up's).
2. **Score**: build a mask over `ds` and estimate each item's importance
   via `scorer` (e.g. `ShapleyExplainer`) -- see `masks.md` §8.
3. **Prune**: search for a small item subset keeping the confidence gap
   above a threshold derived from `abs_tol`/`rel_tol` -- see
   `doc/design/pruning.md`.
4. **Report**: package the result together with how much of the sample
   was pruned away, as an [`ExplanationResult`](@ref).

`model` must produce a softmax-style output with **two or more classes**
for a single observation (`length(vec(model(ds))) >= 2`); for a single
scalar/sigmoid output, use [`explainf`](@ref) directly with a
hand-written objective (`examples/mutagenesis.jl` shows exactly this for
a binary classifier).

Only `ds` and `model` (the data being explained) are positional -- every
choice of *how* to explain is a keyword, including `scorer`, consistent
with `explainf`'s `order`/`levelbylevel`/`random_removal`/`finetune`
already being keywords. The actual work happens in `_explain`, kept as a
separate internal function specifically so a future scoring strategy that
needs `explain`'s own orchestration (not just its own `stats` method) to
diverge can add an `_explain` method without touching this signature.

# Arguments
- `ds::AbstractMillNode`: the sample to explain.
- `model`: the Mill model. `model(ds)` must return (something `vec`-able
  to) one softmax logit/probability per class.
- `class::Integer`: which class to explain. Must be (one of) the model's
  actual predicted class(es) for `ds` -- i.e. have a nonnegative
  confidence gap; `explain` raises an error otherwise, since there is
  nothing meaningful to preserve while pruning if the model didn't
  predict this class to begin with.

# Keyword arguments
- `scorer = ShapleyExplainer(300)`: the scoring strategy. `300` Monte
  Carlo samples is a reasonable default in practice; pass e.g.
  `scorer=ShapleyExplainer(1000)` for a more precise (slower) estimate.
- `abs_tol`, `rel_tol`: exactly one may be given (or neither, which warns
  once and defaults to `rel_tol=0.9`). `rel_tol` (in `[0, 1]`) keeps the
  confidence gap at that *fraction* of its original value; `abs_tol`
  keeps it within that *absolute amount* of the original value (and must
  not exceed it).
- `order`, `levelbylevel`, `random_removal`, `finetune`, `rng`: forwarded
  to [`explainf`](@ref) -- see its docstring for what each controls.

# Example
```julia
model = reflectinmodel(ds, d -> Dense(d, nclasses); all_imputing=true)
result = explain(ds, model)  # explain the predicted class, default scorer
result.mask            # the pruned mask
ds[result.mask]         # the pruned sample
fraction_pruned(result) # how much of ds turned out to be unnecessary
```
"""
function explain(ds::AbstractMillNode, model, class::Integer;
    scorer::AbstractHeuristic=ShapleyExplainer(300),
    abs_tol=nothing, rel_tol=nothing,
    order::Union{HeuristicOrder,GreedyForward,Nothing}=nothing,
    levelbylevel::Bool=true,
    random_removal::Bool=true,
    finetune::Bool=true,
    rng::AbstractRNG=Random.default_rng())
    _explain(ds, model, class, scorer;
        abs_tol, rel_tol, order, levelbylevel, random_removal, finetune, rng)
end

function explain(ds::AbstractMillNode, model; kwargs...)
    class = argmax(vec(model(ds)))
    explain(ds, model, class; kwargs...)
end

"""
    _explain(ds, model, class, scorer; abs_tol, rel_tol, order, levelbylevel, random_removal, finetune, rng)

Internal worker behind [`explain`](@ref). Not exported. Kept separate
from `explain` (which only handles keyword defaults) so it can grow
additional methods dispatching on `scorer`'s type later, without changing
`explain`'s public signature.
"""
function _explain(ds::AbstractMillNode, model, class::Integer, scorer::AbstractHeuristic;
    abs_tol, rel_tol, order, levelbylevel, random_removal, finetune, rng)
    o0 = vec(model(ds))
    length(o0) >= 2 || error(
        "explain: model(ds) has only $(length(o0)) output(s); explain() " *
        "requires a softmax-style output with 2+ classes. For a single " *
        "scalar/sigmoid output, use explainf with a hand-written " *
        "objective (see examples/mutagenesis.jl).",
    )
    cg = _confgap(_softmax(o0), class)
    cg >= 0 || error(
        "explain: class $class has a negative confidence gap ($cg) for " *
        "this sample -- it is not (one of) the model's predicted class(es)",
    )
    threshold = _threshold(cg, abs_tol, rel_tol)

    fₛ = o -> _softmax(vec(o))[class]
    fₚ = o -> _confgap(_softmax(vec(o)), class) - threshold

    result = explainf(scorer, ds, model, fₛ, fₚ;
        order, levelbylevel, random_removal, finetune, rng)

    remaining_cg = _confgap(_softmax(vec(model(ds[result.mask]))), class)

    ExplanationResult(result.mask, ds, class, Float64(cg), Float64(remaining_cg),
        Float64(threshold), result.n_total, result.n_kept)
end

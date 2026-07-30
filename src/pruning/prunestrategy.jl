# The PruningStrategy/prune! driver (doc/design/pruning.md §3.1, §4),
# composing the local-search primitives (localsearch.jl) over FlatView
# (flatview.md) along the order/granularity/post-pass axes.

"""
    GreedyForward

Order strategy: no precomputed importance signal. At each step, actually
try every remaining candidate and keep whichever single addition improves
the objective most (`sfs!`/`addone!`).
"""
struct GreedyForward end

"""
    HeuristicOrder(scores::IdDict{StructMask,Vector{Float64}})

Order strategy: candidates are sorted by a precomputed importance score and
added via bisection (`addminimumbi!`). `scores` is built once, by the
caller, via `nodescores(mask, acctree, score)` right after scoring --
`prune!`/`PruningStrategy` never reference `AccTree` or a scoring-strategy
type at all (`doc/design/pruning.md` §3.3, §4, Part 2 of the driver
discussion).
"""
struct HeuristicOrder
    scores::IdDict{StructMask,Vector{Float64}}
end

"""
    PruningStrategy{Order}(order, levelbylevel, random_removal, finetune)

Every axis is a required, explicit field -- no hidden defaults -- per
`doc/design/pruning.md` §3.6 (a naming/behavior mismatch in the original
was traced directly to a post-pass silently defaulting on inside an
unrelated function).

- `order`: `HeuristicOrder(scores)` or `GreedyForward()`.
- `levelbylevel`: `false` = search the whole tree at once. `true` = one
  pass per hierarchy depth, narrowing outward-in (see `prune!`).
- `random_removal`: run `randomremoval!` as a post-pass.
- `finetune`: run `finetune!` as a post-pass.
"""
struct PruningStrategy{Order}
    order::Order
    levelbylevel::Bool
    random_removal::Bool
    finetune::Bool
end

"""
    search!(f, fv::FlatView, order, candidates)

Decide `fv`'s items using `order`'s strategy, restricted to `candidates`.
"""
search!(f, fv::FlatView, ::GreedyForward, candidates::AbstractVector{<:Integer}) =
    sfs!(f, fv, candidates)

function search!(f, fv::FlatView, order::HeuristicOrder, candidates::AbstractVector{<:Integer})
    scored = heuristicscores(fv, order.scores)
    # `alg=MergeSort` (stable) rather than Julia's unstable default: ties in
    # `scored` (routine for near-zero-importance items) would otherwise
    # break inconsistently depending on incidental factors unrelated to the
    # mask/model/rng (e.g. prior unrelated allocations), making `prune!`'s
    # result non-reproducible even with a fixed scoring rng.
    ranked = sort(candidates, by=i -> scored[i], rev=true, alg=Base.Sort.MergeSort)
    fill!(fv, false)   # addminimumbi! does not self-reset
    addminimumbi!(f, fv, ranked)
end

"""
    prune!(mask::StructMask, ds, model, f, strategy::PruningStrategy) -> mask

Search `mask` for a small item subset keeping `f() ≥ 0`, using `strategy`.
Mutates `mask` in place (via `FlatView` aliasing) and returns it, per
Julia's `!`-function convention.

`ds`/`model` are accepted uniformly for every strategy even though only a
future gradient-based order would use them, to avoid a breaking signature
change later (`doc/design/pruning.md` §3.1).

# Design contract

`prune!` requires the *full* mask (every item on) to already satisfy
`f() ≥ 0` -- if it doesn't, that's an unsatisfiable request (almost always
a misconfigured tolerance upstream), and `prune!` raises immediately rather
than attempting a search that cannot succeed. Given that precondition
holds, `prune!` is guaranteed to return a mask with `f() ≥ 0`; if it
doesn't, that indicates either a bug in the search primitives or a
violated assumption about `f` (namely, that including strictly more of the
sample never makes the objective worse -- true for confidence-gap-style
objectives, not guaranteed for an arbitrary injected `f`). Either way this
is a loud, unconditional error, not a value for the caller to check.
"""
function prune!(mask::StructMask, ds, model, f, strategy::PruningStrategy)
    if strategy.levelbylevel
        pairs = collectmasks(mask)
        maxlevel = maximum(last, pairs)
        fill!(FlatView(mask), true)
        f() < 0 && error(
            "prune!: the full mask does not satisfy the objective -- " *
            "check the tolerance/threshold used to build it",
        )

        for k in 1:maxlevel
            updateparticipation!(ds, mask)
            nodes = [n for (n, l) in pairs if l == k]
            isempty(nodes) && continue
            fv = FlatView(nodes)
            candidates = findall(participate(fv))
            search!(f, fv, strategy.order, candidates)
            strategy.random_removal && randomremoval!(f, fv)
            strategy.finetune && finetune!(f, fv, typemax(Int), candidates)
        end

        updateparticipation!(ds, mask)
        fv_all = FlatView(mask)
        candidates_all = findall(participate(fv_all))
        strategy.random_removal && randomremoval!(f, fv_all)
        strategy.finetune && finetune!(f, fv_all, typemax(Int), candidates_all)
    else
        fv = FlatView(mask)
        fill!(fv, true)
        f() < 0 && error(
            "prune!: the full mask does not satisfy the objective -- " *
            "check the tolerance/threshold used to build it",
        )
        fill!(fv, false)

        candidates = collect(1:length(fv))
        search!(f, fv, strategy.order, candidates)
        strategy.random_removal && randomremoval!(f, fv)
        strategy.finetune && finetune!(f, fv, typemax(Int), candidates)
    end

    f() < 0 && error(
        "prune!: failed to reach a feasible mask despite the full mask " *
        "satisfying the objective -- this indicates a bug in the search " *
        "primitives or a non-monotonic objective f",
    )
    mask
end

# Local-search primitives operating on a `FlatView`, ported from
# `ExplainMill.jl`'s `src/pruning/utils.jl` (see `doc/design/pruning.md`
# §2.1, §3.8). `f` is always a zero-argument objective closure (`design.doc`
# §4.1): positive/nonnegative means the current state of `fv` still
# satisfies the explanation's tolerance.
#
# Three real defects were found in the original during porting and are
# fixed here rather than carried forward -- each is noted at the relevant
# function:
#   1. `removeexcess!` took a candidate order but silently ignored it,
#      iterating `useditems(flatmask)` instead -- so `randomremoval!`
#      calling `shuffle` had no actual effect on removal order.
#   2. `addminimumbi!` took an unused `significance` parameter (the order
#      to use was already baked into `ii`) -- dropped as dead weight.
#   3. `finetuneremove!`'s recovery loop called `addone!` without checking
#      its return value, risking an infinite loop if nothing was left to
#      add while the objective was still unsatisfied.

"""
    addminimumbi!(f, fv::FlatView, order::AbstractVector{<:Integer})

Turn on items along `order` (a permutation of `1:length(fv)`, typically
sorted by descending importance) via bisection, until `f() ≥ 0`. `O(log n)`
evaluations of `f`, relying on `order` being roughly monotonic in effect --
does not itself guarantee `f() ≥ 0` is reachable at all (if turning on
every item in `order` still leaves `f() < 0`, this returns having done its
best; surfacing that as non-convergence is the driver's responsibility, see
`doc/design/pruning.md` §3.7).
"""
function addminimumbi!(f, fv::FlatView, order::AbstractVector{<:Integer})
    f() > 0 && return false
    left, right = 1, length(order)
    while right - left > 1
        mid = left + (right - left) ÷ 2
        _setprefix!(fv, order, mid)
        f() >= 0 ? (right = mid) : (left = mid)
    end
    _setprefix!(fv, order, right)
    true
end

function _setprefix!(fv::FlatView, order::AbstractVector{<:Integer}, k::Int)
    for i in 1:k
        fv[order[i]] = true
    end
    for i in (k+1):length(order)
        fv[order[i]] = false
    end
end

"""
    addone!(f, fv::FlatView, candidates=1:length(fv))

Single-step greedy forward selection: try turning on every currently-off
item in `candidates`, keep whichever addition gives the largest `f()`.
`O(n)` evaluations. Returns `true` if something was turned on.

`candidates` defaults to unrestricted (every index) rather than filtering
by `participate(fv)` internally -- restricting to participating items is
the *driver*'s responsibility (pass `findall(participate(fv))` explicitly
for level-by-level pruning), since for flat-granularity pruning
`participate` may hold stale state from a prior scoring pass and shouldn't
be consulted at all; see `doc/design/pruning.md` §4 (Part 4).
"""
function addone!(f, fv::FlatView, candidates::AbstractVector{<:Integer}=1:length(fv))
    remaining = setdiff(candidates, useditems(fv))
    best, j = typemin(f()), -1
    for i in remaining
        fv[i] = true
        o = f()
        fv[i] = false
        if o > best
            best, j = o, i
        end
    end
    j == -1 && return false
    fv[j] = true
    true
end

"""
    removeone!(f, fv::FlatView, candidates=useditems(fv))

Single-step greedy backward selection: try turning off every item in
`candidates` (currently-on items by default), keep whichever removal gives
the largest `f()`. `O(n)` evaluations. Returns `true` if something was
turned off.
"""
function removeone!(f, fv::FlatView, candidates::AbstractVector{<:Integer}=useditems(fv))
    best, j = typemin(f()), -1
    for i in candidates
        fv[i] || continue
        fv[i] = false
        o = f()
        fv[i] = true
        if o > best
            best, j = o, i
        end
    end
    j == -1 && return false
    fv[j] = false
    true
end

"""
    sfs!(f, fv::FlatView, candidates=1:length(fv))

Stepwise forward selection: starting from everything off, repeatedly
`addone!` (restricted to `candidates`) until `f() ≥ 0` or no candidate
remains to add.
"""
function sfs!(f, fv::FlatView, candidates::AbstractVector{<:Integer}=1:length(fv))
    fill!(fv, false)
    while f() < 0
        addone!(f, fv, candidates) || break
    end
    fv
end

"""
    randomremoval!(f, fv::FlatView; rng=Random.default_rng())

Redundancy-removal to a fixed point: repeatedly shuffle the currently-on
items and try turning each off (in that random order), keeping a removal
only if `f() ≥ 0` still holds; stops once a full pass removes nothing more.

Fix vs. the original: `ExplainMill.jl`'s `removeexcess!` took a shuffled
candidate order but iterated `useditems(flatmask)` instead of the order it
was given, so its randomization had no effect. Here the shuffled order is
the order actually iterated.
"""
function randomremoval!(f, fv::FlatView; rng::AbstractRNG=Random.default_rng())
    f() < 0 && return fv
    while true
        used = useditems(fv)
        changed = false
        for i in shuffle(rng, used)
            fv[i] || continue
            fv[i] = false
            if f() < 0
                fv[i] = true
            else
                changed = true
            end
        end
        changed || break
    end
    fv
end

"""
    greedyremoval!(f, fv::FlatView)

Redundancy-removal to a fixed point, using `removeone!`'s greedy
(least-damaging) choice at each step instead of a random order: repeatedly
remove the single least-damaging currently-on item, stopping either when no
removal keeps `f() ≥ 0` (reverting that last, over-aggressive removal) or
when nothing remains that can be removed at all.
"""
function greedyremoval!(f, fv::FlatView)
    while true
        used_before = useditems(fv)
        removeone!(f, fv, used_before) || break
        if f() < 0
            # `removeone!` already committed its chosen removal; undo
            # exactly that one item (the only state change it made) rather
            # than the original's "revert the whole pre-iteration used set"
            # (equivalent in effect, since every other item was untouched),
            # and stop: the least-damaging removal was still infeasible, so
            # every remaining item is essential.
            removed = setdiff(used_before, useditems(fv))
            fv[removed] = true
            break
        end
    end
    fv
end

"""
    finetune!(f, fv::FlatView, max_n=typemax(Int), candidates=1:length(fv))

A small local-search pass after a main search: alternates batched
add/remove (`finetuneadd!`/`finetuneremove!`, add-side restricted to
`candidates`), tracks visited item-sets to detect cycling (growing the
batch size `n` when a state repeats), and reverts to the smallest,
best-scoring feasible (`f() ≥ 0`) state visited by the end.
"""
function finetune!(f, fv::FlatView, max_n::Int=typemax(Int),
    candidates::AbstractVector{<:Integer}=1:length(fv))
    visited = Dict{Vector{Int},Float64}(sort(useditems(fv)) => f())
    n = 1
    max_n = min(max_n, length(useditems(fv)))
    for i in 1:100
        if iseven(i)
            finetuneadd!(f, fv, n, candidates)
        else
            finetuneremove!(f, fv, n, candidates)
        end
        used = sort(useditems(fv))
        if haskey(visited, used)
            n += 1
            n > max_n && break
        else
            visited[used] = f()
            max_n = min(length(used), max_n)
        end
    end
    settobest!(fv, visited)
    fv
end

function finetuneadd!(f, fv::FlatView, n::Int, candidates::AbstractVector{<:Integer}=1:length(fv))
    for _ in 1:n
        addone!(f, fv, candidates)
    end
    while true
        removeone!(f, fv) || break
        if f() < 0
            addone!(f, fv, candidates)
            break
        end
    end
end

function finetuneremove!(f, fv::FlatView, n::Int, candidates::AbstractVector{<:Integer}=1:length(fv))
    for _ in 1:n
        removeone!(f, fv)
    end
    f() > 0 && return
    while true
        added = addone!(f, fv, candidates)
        if f() >= 0 || !added
            break
        end
    end
end

"""
    settobest!(fv::FlatView, visited::Dict{Vector{Int},<:Real})

Set `fv` to the smallest, best-scoring feasible (`f() ≥ 0`) item set among
`visited`'s keys. Does nothing if no visited state is feasible.
"""
function settobest!(fv::FlatView, visited::Dict{Vector{Int},V}) where {V<:Real}
    feasible = [ii for ii in keys(visited) if visited[ii] >= 0]
    isempty(feasible) && return fv
    minlen = minimum(length, feasible)
    smallest = filter(ii -> length(ii) == minlen, feasible)
    best = argmin(ii -> visited[ii], smallest)
    fill!(fv, false)
    fv[best] = true
    fv
end

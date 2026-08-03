"""
    MeanDiff

Running per-unit accumulator: separate means of the objective value observed
when a unit was on vs. off across random subset samples. The difference
(`score`) is an unbiased Monte Carlo (linear-approximation) estimate of the
unit's Shapley/Banzhaf value -- the same idea as `ExplainMill.jl`'s
`Duff.Daf`, reimplemented directly rather than depending on `Duff.jl`.
"""
mutable struct MeanDiff
    sum1::Float64
    n1::Int
    sum0::Float64
    n0::Int
end
MeanDiff() = MeanDiff(0.0, 0, 0.0, 0)

function update!(a::MeanDiff, value::Real, on::Bool)
    if on
        a.sum1 += value
        a.n1 += 1
    else
        a.sum0 += value
        a.n0 += 1
    end
    a
end

"""
    score(a::MeanDiff) -> Float64

The estimated importance of the unit `a` accumulates statistics for: the
mean objective value observed when the unit was on, minus the mean when
it was off. `0.0` if the unit has never been observed both on and off.
"""
score(a::MeanDiff) = (a.n1 == 0 || a.n0 == 0) ? 0.0 : a.sum1 / a.n1 - a.sum0 / a.n0

"""
    ShapleyExplainer(n=1000)

Monte Carlo Shapley/Banzhaf-style scoring strategy: repeatedly randomizes
the whole mask, evaluates `objective(model(ds[mask]))`, and accumulates a
`MeanDiff` per unit. Exists primarily to exercise the mask infrastructure
end to end (random sampling, participation, hard pruning, paired
bookkeeping traversal) -- see `doc/design/pruning.md` for where this fits
relative to a production scoring strategy.
"""
struct ShapleyExplainer <: AbstractHeuristic
    n::Int
end
ShapleyExplainer() = ShapleyExplainer(1000)

"""
    stats(e::ShapleyExplainer, ds, model, objective; rng=Random.default_rng())

`objective(output) -> Real` is evaluated on the model's output for each
randomly masked sample; no assumption about classification/class indices is
made here (see `design.doc` §4.1 on objective injection).

Returns `(mask, acctree)`; use `score.(payload)` on `acctree`'s leaves (via
`foreach_paired`) to read out per-unit importance.
"""
function stats(e::ShapleyExplainer, ds::AbstractMillNode, model, objective;
        rng::AbstractRNG=Random.default_rng())
    sm = create_structmask(ds, d -> trues(d))
    acc = create_acctree(sm, d -> [MeanDiff() for _ in 1:d])
    for _ in 1:e.n
        randomize!(rng, sm)
        updateparticipation!(ds, sm)
        o = objective(model(ds[sm]))
        foreach_paired(sm, acc) do leaf, payload, _
            keep = prunemask(leaf)
            part = leaf.participate
            for i in eachindex(keep)
                part[i] || continue
                update!(payload[i], o, keep[i])
            end
        end
    end
    sm, acc
end

"""
    leafscores(m::StructMask, acc::AccTree)

Collect `score.(payload)` for every own-bearing node, in the same order
`collectmasks`/`foreach_mask` would visit them.
"""
function leafscores(m::StructMask, acc::AccTree)
    out = Vector{Vector{Float64}}()
    foreach_paired((_, payload, _) -> push!(out, score.(payload)), m, acc)
    out
end

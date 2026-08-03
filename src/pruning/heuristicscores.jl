"""
    nodescores(mask::StructMask, acc::AccTree, scorefn) -> IdDict{StructMask,Vector{Float64}}

Build the identity-keyed lookup from `docs/design/flatview.md` §3 ("Option
B"): walk `mask` and `acc` together (via `foreach_paired`, so they must have
been built from one another and share structure) and record, per
own-bearing node -- keyed by object identity, not value -- the plain
`Float64` scores obtained by applying `scorefn` to that node's accumulator
payload.

The returned dictionary is only valid against `FlatView`s built over the
exact same `mask` object (or a subset of its nodes) `acc` was paired with;
see `heuristicscores` for how it's consumed and what happens on a mismatch.
"""
function nodescores(mask::StructMask, acc::AccTree, scorefn)
    dict = IdDict{StructMask,Vector{Float64}}()
    foreach_paired(mask, acc) do node, payload, _
        dict[node] = Float64[scorefn(p) for p in payload]
    end
    dict
end

"""
    heuristicscores(fv::FlatView, scores::IdDict{StructMask,Vector{Float64}}) -> Vector{Float64}

For every flat index in `fv`, look up the score recorded in `scores` for
the `StructMask` node (by identity, `===`) and local index that flat index
maps to. Works unchanged whether `fv` spans a whole tree or a subset of it
(e.g. one level, for level-by-level pruning), since `scores` is keyed by
node identity rather than position.

Raises `KeyError` if a node in `fv` isn't present in `scores` -- this means
`fv` was built over a mask tree that isn't (object-)identical to the one
`scores` was built from; see `docs/design/flatview.md` §3.1.
"""
function heuristicscores(fv::FlatView, scores::IdDict{StructMask,Vector{Float64}})
    map(fv.itemmap) do (node, li)
        scores[node][li]
    end
end

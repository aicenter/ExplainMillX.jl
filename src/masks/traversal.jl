"""
    foreach_mask(f, m::StructMask)

The sole sanctioned way to walk a `StructMask` tree for side effects. Calls
`f(node, level)` on every node that has an own mask (`node.own !== nothing`).
Memoized via an identity-keyed `IdDict`, so shared sub-structure (the same
node reachable through more than one parent) is visited exactly once.
"""
function foreach_mask(f, m::StructMask, level::Int=1, visited::IdDict=IdDict())
    haskey(visited, m) && return nothing
    visited[m] = nothing
    m.own !== nothing && f(m, level)
    _visitchildren(f, m.children, level, visited)
    nothing
end
_visitchildren(f, ::Nothing, level, visited) = nothing
_visitchildren(f, c::StructMask, level, visited) = foreach_mask(f, c, level + 1, visited)
_visitchildren(f, cs::Union{Tuple,NamedTuple}, level, visited) =
    foreach(c -> foreach_mask(f, c, level + 1, visited), Tuple(cs))

"""
    mapmask(f, m::StructMask)

Structure-preserving transform: calls `f(own_vector, level)` on every node's
own mask vector and rebuilds the tree with the returned vectors, preserving
`participate` and shape. Memoized on the identity of the own vector, so
masks sharing the same underlying vector remain shared after mapping.
"""
function mapmask(f, m::StructMask, level::Int=1, visited::IdDict=IdDict())
    newchildren = _mapchildren(f, m.children, level, visited)
    if m.own === nothing
        return StructMask{typeof(newchildren),Vector{Bool}}(nothing, nothing, newchildren)
    end
    newown = get!(visited, m.own) do
        f(m.own, level)
    end
    StructMask{typeof(newchildren),typeof(newown)}(newown, m.participate, newchildren)
end
_mapchildren(f, ::Nothing, level, visited) = nothing
_mapchildren(f, c::StructMask, level, visited) = mapmask(f, c, level + 1, visited)
_mapchildren(f, cs::NamedTuple, level, visited) =
    NamedTuple{keys(cs)}(map(c -> mapmask(f, c, level + 1, visited), values(cs)))
_mapchildren(f, cs::Tuple, level, visited) = map(c -> mapmask(f, c, level + 1, visited), cs)

"""
    collectmasks(m::StructMask)

Collect all own-bearing nodes together with their depth, as `node => level`
pairs. Mirrors `ExplainMill.jl`'s `collect_masks_with_levels`.
"""
function collectmasks(m::StructMask)
    out = Vector{Pair{StructMask,Int}}()
    foreach_mask((mm, l) -> push!(out, mm => l), m)
    out
end

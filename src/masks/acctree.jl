"""
    AccTree{P,C}

Per-unit scoring bookkeeping, kept entirely separate from `StructMask` (see
`docs/design/masks.md` §8). Mirrors the shape of the `StructMask` it was built
from, carrying whatever payload a scoring strategy needs -- the model never
sees this type.
"""
struct AccTree{P,C}
    payload::P
    children::C
end


"""
    create_acctree(m::StructMask, make_payload) -> AccTree

Build an `AccTree` mirroring the shape of `m`, calling
`make_payload(d)` (`d` = number of maskable units) at every own-bearing
node to construct that node's bookkeeping payload.
"""
function create_acctree(m::StructMask, make_payload)
    children = _mapchildren_acc(m.children, make_payload)
    if m.own === nothing
        return AccTree{Any,typeof(children)}(nothing, children)
    end
    payload = make_payload(length(m.own))
    AccTree{typeof(payload),typeof(children)}(payload, children)
end
_mapchildren_acc(::Nothing, mp) = nothing
_mapchildren_acc(c::StructMask, mp) = create_acctree(c, mp)
_mapchildren_acc(cs::NamedTuple, mp) =
    NamedTuple{keys(cs)}(map(c -> create_acctree(c, mp), values(cs)))
_mapchildren_acc(cs::Tuple, mp) = map(c -> create_acctree(c, mp), cs)

"""
    foreach_paired(f, m::StructMask, acc::AccTree)

Walk `m` and `acc` in lockstep (same recursion shape as `foreach_mask`),
calling `f(node, payload, level)` on every own-bearing node. Lets a scoring
strategy update its bookkeeping without the model ever being aware
bookkeeping exists.
"""
function foreach_paired(f, m::StructMask, acc::AccTree, level::Int=1, visited::IdDict=IdDict())
    haskey(visited, m) && return nothing
    visited[m] = nothing
    m.own !== nothing && f(m, acc.payload, level)
    _visitchildren_paired(f, m.children, acc.children, level, visited)
    nothing
end
_visitchildren_paired(f, ::Nothing, ::Nothing, level, visited) = nothing
_visitchildren_paired(f, c::StructMask, ca::AccTree, level, visited) =
    foreach_paired(f, c, ca, level + 1, visited)
_visitchildren_paired(f, cs::Union{Tuple,NamedTuple}, cas::Union{Tuple,NamedTuple}, level, visited) =
    foreach((c, ca) -> foreach_paired(f, c, ca, level + 1, visited), Tuple(cs), Tuple(cas))

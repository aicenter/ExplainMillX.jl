"""
    updateparticipation!(ds::AbstractMillNode, m::StructMask)

Recompute, for every node in `m`, whether its units are currently reachable
given the *current* state of ancestor masks (see `docs/design/masks.md` §5).
Resets everything to participating, then propagates invalidity top-down via
`invalidate!`, which is dispatched on `ds`'s type since the index-space
translation between a node and its children (e.g. bag membership, sparse
column ownership) is storage-format specific.

Must be called after any mutation to a mask's `own` vector, before reading
`.participate` anywhere in the tree.
"""
function updateparticipation!(ds::AbstractMillNode, m::StructMask)
    foreach_mask((mm, _) -> fill!(mm.participate, true), m)
    invalidate!(ds, m, Int[])
    m
end

# Leaf array types: `own`'s units may or may not correspond 1:1 to
# observations, so each dispatches its own translation.

# Dense features are invariant across observations: nothing to invalidate.
invalidate!(::ArrayNode{<:Matrix}, ::StructMask, invalid) = nothing

function invalidate!(ds::ArrayNode{<:SparseMatrixCSC}, m::StructMask, invalid)
    isempty(invalid) && return nothing
    invalidset = Set(invalid)
    cols = sparsecolumns(ds.data)
    for i in eachindex(cols)
        cols[i] in invalidset && (m.participate[i] = false)
    end
    nothing
end

function invalidate!(::ArrayNode{<:Mill.MaybeHotMatrix}, m::StructMask, invalid)
    isempty(invalid) || (m.participate[invalid] .= false)
    nothing
end

function invalidate!(::ArrayNode{<:Mill.NGramMatrix}, m::StructMask, invalid)
    isempty(invalid) || (m.participate[invalid] .= false)
    nothing
end

function invalidate!(ds::BagNode, m::StructMask, invalid_bags)
    ismissing(ds.data) && return nothing
    invalid_instances = isempty(invalid_bags) ? Int[] :
        reduce(vcat, (collect(ds.bags[i]) for i in invalid_bags); init=Int[])
    isempty(invalid_instances) || (m.participate[invalid_instances] .= false)
    combined = unique(vcat(invalid_instances, findall(.!(prunemask(m) .& m.participate))))
    invalidate!(ds.data, m.children, combined)
    nothing
end

function invalidate!(ds::ProductNode, m::StructMask, invalid)
    foreach((c, cm) -> invalidate!(c, cm, invalid), Tuple(ds.data), Tuple(m.children))
    nothing
end

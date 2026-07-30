"""
    FlatView

A flat, linear, mutable index over a collection of own-bearing `StructMask`
nodes. Aliases the real `own` vectors in place -- each flat index stores a
reference to the actual `StructMask` node it belongs to plus a local index
into that node's `own` -- so reading/writing through a `FlatView` reads and
writes the real mask directly, with no copying or reconciliation step.

Deliberately blind to tree shape (it only needs "which nodes, in what
order") and to scoring/`AccTree` (see `nodescores`/`heuristicscores` for how
those connect back in). See `doc/design/flatview.md` for the full rationale.
"""
struct FlatView
    itemmap::Vector{Tuple{StructMask,Int}}
end

"""
    FlatView(nodes::AbstractVector{<:StructMask})
    FlatView(mask::StructMask)
    FlatView(pairs::AbstractVector{<:Pair})

Build a `FlatView` over `nodes` (each of which must be own-bearing, i.e.
`node.own !== nothing`), preserving their order and, within each node, its
natural index order. `FlatView(mask)` is the whole-tree view, built from
`collectmasks(mask)`. `FlatView(pairs)` accepts `node => level` pairs (as
returned by `collectmasks`) for convenience and only uses the node.

The nodes passed in are stored by reference, not copied -- a `FlatView`
built from a subset of a tree's nodes (e.g. one hierarchy level) aliases
the exact same objects as a whole-tree `FlatView` over the same tree.
"""
function FlatView(nodes::AbstractVector{<:StructMask})
    itemmap = Tuple{StructMask,Int}[]
    for node in nodes
        node.own === nothing && error(
            "FlatView: node has no own mask; only own-bearing nodes " *
            "(leaf/hybrid, not router) may be included",
        )
        for li in 1:length(node)
            push!(itemmap, (node, li))
        end
    end
    FlatView(itemmap)
end
FlatView(mask::StructMask) = FlatView(first.(collectmasks(mask)))
FlatView(pairs::AbstractVector{<:Pair}) = FlatView(first.(pairs))

Base.length(fv::FlatView) = length(fv.itemmap)

Base.getindex(fv::FlatView, i::Integer) = _getitem(fv.itemmap[i]...)
Base.getindex(fv::FlatView, ii::AbstractVector{<:Integer}) = map(i -> fv[i], ii)
_getitem(node::StructMask, li::Int) = node.own[li]

function Base.setindex!(fv::FlatView, v, i::Integer)
    node, li = fv.itemmap[i]
    node.own[li] = v
end
Base.setindex!(fv::FlatView, v, ii::AbstractVector{<:Integer}) = foreach(i -> fv[i] = v, ii)

Base.fill!(fv::FlatView, v) = foreach(i -> fv[i] = v, 1:length(fv))

"""
    copyto!(fv::FlatView, values)

Bulk-assign a plain indexable collection of length `length(fv)` into `fv`'s
underlying storage, position by position, ignoring any structural
dependencies between items.
"""
function Base.copyto!(fv::FlatView, values)
    length(values) == length(fv) ||
        throw(DimensionMismatch("copyto!: length(values)=$(length(values)) != length(fv)=$(length(fv))"))
    for i in eachindex(fv.itemmap)
        fv[i] = values[i]
    end
    fv
end

# per-item boolean "is this item on" view, regardless of whether the owning
# node's `own` is binary or real-valued -- mirrors `prunemask(::StructMask)`.
_itemmask(node::StructMask{<:Any,<:AbstractVector{Bool}}, li::Int) = node.own[li]
_itemmask(node::StructMask{<:Any,<:AbstractVector{<:Real}}, li::Int) = node.own[li] > 0.5f0

"""
    prunemask(fv::FlatView)

The aggregated boolean "is this item currently on" view across every item
in `fv`, in flat order.
"""
prunemask(fv::FlatView) = [_itemmask(node, li) for (node, li) in fv.itemmap]

"""
    participate(fv::FlatView)

The aggregated boolean reachability view across every item in `fv`, in flat
order (see `doc/design/flatview.md` §5).
"""
participate(fv::FlatView) = [node.participate[li] for (node, li) in fv.itemmap]

"""
    useditems(fv::FlatView)

Flat indices currently on -- `findall(prunemask(fv))`. This is what
redundancy-removal and fine-tuning search primitives iterate over.
"""
useditems(fv::FlatView) = findall(prunemask(fv))

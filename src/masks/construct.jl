"""
    create_structmask(ds::AbstractMillNode, mk)

Build a `StructMask` mirroring `ds`. `mk` is a leaf factory `d -> own_vector`
of length `d`; the caller selects binary vs. differentiable masks entirely
through what `mk` returns (see `doc/design/masks.md` §3.2).

What `own`'s units mean is storage-format specific, matching the granularity
at which it is natural to explain that format:
- dense (`Matrix`): one unit per row (feature) -- shared across all observations
- sparse (`SparseMatrixCSC`): one unit per stored nonzero value
- categorical (`MaybeHotMatrix`) / n-gram (`NGramMatrix`): one unit per observation
- `BagNode`: one unit per instance (its child's observations)
- `ProductNode`: no own unit; routes to named children
"""
create_structmask(ds::ArrayNode{<:Matrix}, mk) = leafmask(mk(size(ds.data, 1)))

create_structmask(ds::ArrayNode{<:SparseMatrixCSC}, mk) = leafmask(mk(nnz(ds.data)))

create_structmask(ds::ArrayNode{<:Mill.MaybeHotMatrix}, mk) = leafmask(mk(numobs(ds)))

create_structmask(ds::ArrayNode{<:Mill.NGramMatrix}, mk) = leafmask(mk(numobs(ds)))

function create_structmask(ds::BagNode, mk)
    ismissing(ds.data) && return leafmask(mk(0))
    hybridmask(mk(numobs(ds.data)), create_structmask(ds.data, mk))
end

create_structmask(ds::ProductNode, mk) = routermask(map(c -> create_structmask(c, mk), ds.data))

"""
    sparsecolumns(x::SparseMatrixCSC)

Column index of each stored entry, in the same order as `x.nzval`.
Recomputed on demand rather than cached on the mask, so `StructMask` needs
no per-format auxiliary field.
"""
function sparsecolumns(x::SparseMatrixCSC)
    cols = Vector{Int}(undef, nnz(x))
    for j in 1:size(x, 2), k in x.colptr[j]:(x.colptr[j+1]-1)
        cols[k] = j
    end
    cols
end

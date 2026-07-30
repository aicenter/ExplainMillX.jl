"""
    applymask(ds::AbstractMillNode, m::StructMask)

Hard-prune `ds` according to `m`, returning a new (smaller / missing-valued)
sample. Never mutates `ds`. This is the one place per-storage-format
knowledge is unavoidable (see `doc/design/masks.md` §6) -- each Mill
leaf-array type needs its own method because a dense `Matrix`, a
`SparseMatrixCSC`, a `MaybeHotMatrix`, and an `NGramMatrix` each need
genuinely different code to represent "this unit is absent."

Masked-out values become `missing`, matching Mill's native imputation
support (sparse entries become `0`, since sparsity already has no `missing`
representation).
"""
applymask

Base.getindex(ds::AbstractMillNode, m::StructMask) = applymask(ds, m)

function applymask(ds::ArrayNode{<:Matrix}, m::StructMask)
    keep = prunemask(m)
    x = Matrix{Union{Missing,eltype(ds.data)}}(ds.data)
    x[.!keep, :] .= missing
    ArrayNode(x, ds.metadata)
end

function applymask(ds::ArrayNode{<:SparseMatrixCSC}, m::StructMask)
    keep = prunemask(m)
    x = copy(ds.data)
    x.nzval[.!keep] .= 0
    ArrayNode(x, ds.metadata)
end

function applymask(ds::ArrayNode{<:Mill.MaybeHotMatrix}, m::StructMask)
    keep = prunemask(m)
    newI = [keep[i] ? ds.data.I[i] : missing for i in eachindex(ds.data.I)]
    ArrayNode(Mill.MaybeHotMatrix(newI, ds.data.l), ds.metadata)
end

function applymask(ds::ArrayNode{<:Mill.NGramMatrix}, m::StructMask)
    keep = prunemask(m)
    newS = [keep[i] ? ds.data.S[i] : missing for i in eachindex(ds.data.S)]
    ArrayNode(Mill.NGramMatrix(newS, ds.data.n, ds.data.b, ds.data.m), ds.metadata)
end

function applymask(ds::BagNode, m::StructMask)
    (ismissing(ds.data) || isleaf(m)) && return ds
    keep = prunemask(m)
    childds = applymask(ds.data, m.children)
    keepidx = findall(keep)
    newbags = Mill.adjustbags(ds.bags, keep)
    newdata = childds[keepidx]
    BagNode(newdata, newbags, ds.metadata)
end

function applymask(ds::ProductNode, m::StructMask)
    ProductNode(map((c, cm) -> applymask(c, cm), ds.data, m.children), ds.metadata)
end

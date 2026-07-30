using Test
using Random
using SparseArrays

using ExplainMillX
using Mill
using Flux
using JsonGrinder

using ExplainMillX: leafmask, hybridmask, routermask, isleaf, isrouter, sparsecolumns

function specimen_sample()
    an = ArrayNode(Float32.(reshape(collect(1:10), 2, 5)))
    on = ArrayNode(Mill.maybehotbatch([1, 2, 3, 1, 2], 1:4))
    cn = ArrayNode(sparse(Float32[1 0 3 0 5; 0 2 0 4 0]))
    sn = ArrayNode(Mill.NGramMatrix(["a", "b", "c", "d", "e"], 3, 256, 2053))
    BagNode(BagNode(ProductNode(; an, on, cn, sn),
            AlignedBags([1:2, 3:3, 4:5])),
        AlignedBags([1:3]))
end

include("structmask.jl")
include("shapley.jl")
include("flatview.jl")
include("localsearch.jl")
include("prunestrategy.jl")
include("explain.jl")
include("jsonoutput.jl")


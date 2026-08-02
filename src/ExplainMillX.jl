module ExplainMillX

using Mill
using Mill: ArrayNode, BagNode, ProductNode, AbstractMillNode, numobs
using SparseArrays
using Random
using JsonGrinder

include("masks/structmask.jl")
include("masks/traversal.jl")
include("masks/construct.jl")
include("masks/apply.jl")
include("masks/participation.jl")
include("masks/acctree.jl")

include("heuristics/heuristics.jl")

include("pruning/flatview.jl")
include("pruning/heuristicscores.jl")
include("pruning/localsearch.jl")
include("pruning/prunestrategy.jl")

include("explain.jl")
include("output/jsonoutput.jl")

export StructMask, leafmask, hybridmask, routermask, isleaf, isrouter
export prunemask, softvalue, randomize!, participate
export create_structmask, applymask
export foreach_mask, mapmask, collectmasks
export updateparticipation!
export AccTree, create_acctree, foreach_paired
export AbstractHeuristic, ShapleyExplainer, MeanDiff, stats, score, leafscores
export FlatView, useditems, nodescores, heuristicscores
export addminimumbi!, addone!, removeone!, sfs!
export randomremoval!, greedyremoval!, finetune!, settobest!
export GreedyForward, HeuristicOrder, PruningStrategy, search!, prune!
export explain, explainf, ExplanationResult, n_pruned, fraction_kept, fraction_pruned
export explain_json

end # module

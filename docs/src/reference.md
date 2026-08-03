# [API Reference](@id api-reference)

All exported names, generated from docstrings. See the
[How-to Guides](howto/explain_prediction.md) for task-oriented usage and
the [Explanation](explanation/what_is_explanation.md) section for the
concepts behind them.

## Explaining a prediction

```@docs
explain
explainf
ExplanationResult
n_pruned
fraction_kept
fraction_pruned
```

## JSON output

```@docs
explain_json
```

## Scoring strategies

```@docs
AbstractHeuristic
ShapleyExplainer
stats
MeanDiff
score
leafscores
```

## Pruning strategies

```@docs
PruningStrategy
HeuristicOrder
GreedyForward
prune!
search!
```

## Masks

```@docs
StructMask
create_structmask
applymask
leafmask
hybridmask
routermask
isleaf
isrouter
prunemask
softvalue
participate
randomize!
updateparticipation!
foreach_mask
mapmask
collectmasks
AccTree
create_acctree
foreach_paired
```

## Pruning internals

```@docs
FlatView
useditems
nodescores
heuristicscores
addminimumbi!
addone!
removeone!
sfs!
randomremoval!
greedyremoval!
finetune!
settobest!
```

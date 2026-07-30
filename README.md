# ExplainMillX.jl

Explaining predictions of hierarchical multi-instance learning models built with
[Mill.jl](https://github.com/CTUAvastLab/Mill.jl).

This is a clean-room redesign of [ExplainMill.jl](https://github.com/CTUAvastLab/ExplainMill.jl);
see `doc/design/` for the design documents (`design.doc`, `masks.md`, `pruning.md`).

Currently implemented: the mask abstraction (`doc/design/masks.md`) and a Monte Carlo
Shapley-style scoring strategy used to exercise it end to end. Pruning and the top-level
`explain` pipeline are not implemented yet.

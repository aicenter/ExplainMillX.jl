"""
    AbstractHeuristic

Supertype for scoring strategies (e.g. [`ShapleyExplainer`](@ref)) usable
as the `scorer` argument to [`explain`](@ref)/[`explainf`](@ref)/`stats`.
"""
abstract type AbstractHeuristic end

include("shapley.jl")
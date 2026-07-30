"""
    StructMask{C,V}

Mirrors the shape of a Mill sample. One node plays one of three roles,
determined by which fields are populated (not by subtyping):

- **leaf**: `own` set, `children === nothing` (e.g. an `ArrayNode`)
- **hybrid**: `own` set, `children` a single `StructMask` (e.g. a `BagNode`)
- **router**: `own === nothing`, `children` a `Tuple`/`NamedTuple` of `StructMask` (e.g. a `ProductNode`)

`V<:AbstractVector` signals at the type level whether the mask is binary
(`V<:AbstractVector{Bool}`) or differentiable (`V<:AbstractVector{<:Real}`).

See `doc/design/masks.md` for the full design rationale.
"""
struct StructMask{C,V<:AbstractVector}
    own::Union{Nothing,V}
    participate::Union{Nothing,Vector{Bool}}
    children::C
end

leafmask(own::V) where {V<:AbstractVector} =
    StructMask{Nothing,V}(own, trues(length(own)), nothing)

hybridmask(own::V, children::C) where {V<:AbstractVector,C} =
    StructMask{C,V}(own, trues(length(own)), children)

# routers have no own mask; V is unused for these nodes, pinned arbitrarily
routermask(children::C) where {C} =
    StructMask{C,Vector{Bool}}(nothing, nothing, children)

isleaf(m::StructMask) = m.children === nothing
isrouter(m::StructMask) = m.own === nothing

prunemask(m::StructMask{<:Any,<:AbstractVector{Bool}}) = m.own
prunemask(m::StructMask{<:Any,<:AbstractVector{<:Real}}) = m.own .> 0.5f0

participate(m::StructMask) = m.participate

softvalue(m::StructMask{<:Any,<:AbstractVector{<:Real}}) = m.own
softvalue(m::StructMask{<:Any,<:AbstractVector{Bool}}) = error(
    "mask is Vector{Bool} (binary, non-differentiable); a Real-valued mask " *
    "is required for gradient-based strategies",
)

Base.length(m::StructMask) = m.own === nothing ? 0 : length(m.own)

"""
    randomize!([rng,] m::StructMask)

Independently sample each maskable unit in `m` (and its whole subtree)
uniformly at random. Used by Monte Carlo scoring strategies.
"""
randomize!(m::StructMask) = randomize!(Random.default_rng(), m)
function randomize!(rng::AbstractRNG, m::StructMask)
    foreach_mask((mm, _) -> rand!(rng, mm.own), m)
    m
end

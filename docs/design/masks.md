# Mask Design

**Status:** Draft — supersedes the mask-related portions of `design.doc` §3
**Audience:** Engineers implementing and documenting `ExplainMillX.jl`
**Scope:** The mask abstraction only — construction, traversal, participation, application to data, and its relationship to scoring strategies. Pruning-strategy and top-level `explain`/`explainf` design are covered in `design.doc`.

---

## 1. Purpose

A mask is the system's representation of "which parts of a Mill sample are currently considered present." Every scoring strategy (§8) builds one, perturbs it, and reads it back; every pruning strategy searches over it; the output stage applies the final one to produce a pruned sample. This document defines what a mask *is*, structurally and semantically, so that this single representation can serve all of those consumers without per-consumer type proliferation.

### 1.1 Why this needed a deliberate design

A prior implementation (`ExplainMill.jl`) modeled masks as a hierarchy of ~14 types split across two axes: one axis for *structural role* (leaf-array masks, a bag/product hybrid, and no-op router masks) and one axis for *value semantics* (a plain boolean/differentiable vector, a heuristic-scored vector, a Shapley-accumulator vector, a participation-tracking decorator, ...). Because differentiable scoring strategies needed the mask *value* to flow through the model's forward pass, every combination of structural role and value type needed its own dispatch into the model's layers (`Dense`, `BagModel` aggregators, etc.) — an implicit N×M growth pattern as new node types or new scoring strategies were added.

This design fixes that by separating four concerns that were previously fused into one type hierarchy:

| Concern | Old location | New location |
|---|---|---|
| Tree shape (mirrors Mill's node tree) | Encoded via distinct types per role (`AbstractListMask`, `BagMask`, `AbstractNoMask`) | One generic recursive type, §2 |
| Mask value (binary vs. differentiable) | Encoded via distinct types per heuristic (`SimpleMask`, `HeuristicMask`, `DafMask`, ...) | One field, typed generically over the vector's element type, §3 |
| Reachability under nested masking | A decorator type (`ParticipationTracker`) wrapping some vector masks | A plain field present on every node, §5 |
| Scoring bookkeeping (Shapley stats, etc.) | Baked into the vector-mask type itself, so it had to satisfy the model-forward contract | A separate tree, same shape, built by walking the mask — never touches the model, §8 |

Read this document alongside `design.doc` §3 (which it supersedes) for the reasoning; this document focuses on defining the resulting structure precisely.

---

## 2. `StructMask`: one structural type for the whole tree

```julia
struct StructMask{C,V<:AbstractVector}
    own::Union{Nothing,V}
    participate::Union{Nothing,Vector{Bool}}
    children::C
end
```

A single `StructMask` node plays one of three structural roles, determined entirely by which fields are populated — not by subtyping:

| Role | `own` | `children` | Corresponds to (in Mill) |
|---|---|---|---|
| **Leaf** | a vector of maskable units | `nothing` | `ArrayNode` (dense, sparse, one-hot/categorical, n-gram) |
| **Hybrid** | a vector of maskable units | one `StructMask` | `BagNode` — masks which instances are kept (`own`) while `children` explains what's inside each instance |
| **Router** | `nothing` | a `Tuple`/`NamedTuple` of `StructMask` | `ProductNode` — no maskable content of its own, only routes to named children |

There is deliberately no fourth, "nothing at all" role with a distinct type: an "explain nothing here" leaf (the old `EmptyMask`, used for unsupported node types like `LazyNode`, or subtrees short-circuited by partial evaluation) is just a `StructMask` with `own = nothing` and `children = nothing`.

### 2.1 Why one type instead of three

The three roles above previously required three separate type hierarchies (`AbstractListMask`'s four concrete subtypes, `BagMask`, and `AbstractNoMask`'s two subtypes) because the *dispatch mechanism itself* was subtyping. Here, dispatch only needs to happen at construction and application time (§3, §6), where it's naturally driven by the *Mill node type*, not by the *mask's role*. Generic tree operations (§4, §5) never need to know or care which of the three roles a given node plays — they only inspect whether `own`/`children` are `nothing`.

---

## 3. Mask value: `V` signals binary vs. differentiable at the type level

The `own` field's type parameter `V<:AbstractVector` is not incidental — it is how the system distinguishes a **binary/hard** mask from a **differentiable/soft** one, without a separate type per representation:

```julia
prunemask(m::StructMask{<:Any,<:AbstractVector{Bool}})   = m.own
prunemask(m::StructMask{<:Any,<:AbstractVector{<:Real}}) = m.own .> 0.5f0

softvalue(m::StructMask{<:Any,<:AbstractVector{<:Real}}) = m.own
softvalue(m::StructMask{<:Any,<:AbstractVector{Bool}})   =
    error("mask is Vector{Bool} (binary, non-differentiable); a Real-valued mask is required for gradient-based strategies")
```

- `prunemask(m)` — the boolean "is this unit present" view. Always available, regardless of representation.
- `softvalue(m)` — the continuous `[0,1]` view used to gate activations differentiably. Only available when `V<:AbstractVector{<:Real}`; calling it on a binary mask is a programming error and fails loudly rather than silently coercing.

### 3.1 Why parametrize rather than pick one representation

An earlier iteration of this design used a single `Vector{Float32}` for `own` everywhere, thresholding at `0.5` to get a boolean view when needed. That is simpler but loses a real guarantee the old (pre-redesign) type hierarchy had: a scoring strategy that only ever produces binary decisions (e.g. a Shapley-value estimator that repeatedly samples subsets) could not accidentally be handed to code expecting a differentiable value, because it was a different Julia type. Parametrizing `StructMask` over `V` restores that guarantee cheaply — it's a type parameter, not a type hierarchy — while still sharing all tree-shape and traversal code.

It also generalizes for free: `V` can be `Vector{Bool}`, `BitVector` (packed, cheaper for large binary masks), `Vector{Float32}`, `Vector{Float64}`, or any other `AbstractVector`, without touching `StructMask`'s definition or any traversal code.

### 3.2 Construction

Construction takes a single **leaf factory** — a function `d -> own_vector_of_length_d` — mirroring the prior system's `create_mask_structure(ds, f)` convention:

```julia
leafmask(own::V) where {V<:AbstractVector} =
    StructMask{Nothing,V}(own, trues(length(own)), nothing)

hybridmask(own::V, children::C) where {V<:AbstractVector,C} =
    StructMask{C,V}(own, trues(length(own)), children)

# routers have no own mask; V is unused for these nodes, pinned arbitrarily
routermask(children::C) where {C} =
    StructMask{C,Vector{Bool}}(nothing, nothing, children)
```

**Correction (post-implementation):** an earlier draft of this section sized every `ArrayNode`'s `own` to `numobs(ds)` uniformly. That contradicts the granularity the adapted `ExplainMill.jl` test suite actually expects, and turned out to conflate two genuinely different axes. What "one maskable unit" means is storage-format specific — matching whichever axis is natural to explain for that format — not uniform across leaf types:

| Mill type | `own` length | Axis |
|---|---|---|
| `ArrayNode{<:Matrix}` (dense) | `size(ds.data, 1)` | rows (features), shared across all observations |
| `ArrayNode{<:SparseMatrixCSC}` | `nnz(ds.data)` | one unit per stored nonzero value |
| `ArrayNode{<:MaybeHotMatrix}` (categorical) | `numobs(ds)` | one unit per observation |
| `ArrayNode{<:NGramMatrix}` (string) | `numobs(ds)` | one unit per observation |
| `BagNode` | `numobs(ds.data)` | one unit per instance |

```julia
create_structmask(ds::ArrayNode{<:Matrix}, mk) = leafmask(mk(size(ds.data, 1)))
create_structmask(ds::ArrayNode{<:SparseMatrixCSC}, mk) = leafmask(mk(nnz(ds.data)))
create_structmask(ds::ArrayNode{<:Mill.MaybeHotMatrix}, mk) = leafmask(mk(numobs(ds)))
create_structmask(ds::ArrayNode{<:Mill.NGramMatrix}, mk) = leafmask(mk(numobs(ds)))

function create_structmask(ds::BagNode, mk)
    ismissing(ds.data) && return leafmask(mk(0))
    hybridmask(mk(numobs(ds.data)), create_structmask(ds.data, mk))
end

create_structmask(ds::ProductNode, mk) =
    routermask(map(c -> create_structmask(c, mk), ds.data))
```

This doesn't compromise the generic-traversal design (§4): `foreach_mask`/`mapmask`/`updateparticipation!` never need to know what `own`'s units mean, only whether `own`/`children` are populated. Only construction (here), application (§6), and participation propagation (§5) are per-format, exactly as intended.

The caller selects the binary-vs-differentiable representation entirely through what `mk` returns:

```julia
create_structmask(ds, d -> trues(d))          # binary — e.g. Monte Carlo / Shapley-style scoring
create_structmask(ds, d -> ones(Float32, d))  # differentiable — e.g. gradient-based scoring
```

`create_structmask` is the only place per-node-type dispatch is needed for construction; every Mill node type this system supports must provide exactly one method here (see §9's extension checklist).

---

## 4. Generic traversal

Two functions, defined once, are the sole sanctioned way any algorithm walks a `StructMask` tree — no code outside this section should pattern-match on tree shape directly:

```julia
function foreach_mask(f, m::StructMask, level=1, visited=IdDict())
    haskey(visited, m) && return
    visited[m] = nothing
    m.own !== nothing && f(m, level)
    _visitchildren(f, m.children, level, visited)
end
_visitchildren(f, ::Nothing, level, visited)                  = nothing
_visitchildren(f, c::StructMask, level, visited)               = foreach_mask(f, c, level+1, visited)
_visitchildren(f, cs::Union{Tuple,NamedTuple}, level, visited) = foreach(c -> foreach_mask(f, c, level+1, visited), cs)

mapmask(f, m::StructMask, level=1, visited=IdDict()) =
    StructMask(m.own === nothing ? nothing : get!(visited, m, f(m, level)),
               m.participate,
               _mapchildren(f, m.children, level, visited))
```

Both are memoized via an identity-keyed `IdDict`, so shared sub-structure (the same child mask reachable through more than one parent) is visited/transformed exactly once.

**Design rule:** if you find yourself writing `if m.children isa Tuple ... elseif m.children isa StructMask ...` inside a scoring or pruning strategy, that logic belongs in `_visitchildren`/`_mapchildren`, not in the strategy. This is what makes node types and strategies independently extensible (§9): a strategy author never needs to know how `BagNode` vs. `ProductNode` recursion differs.

---

## 5. Participation

### 5.1 The problem

Masking out a bag instance (setting its `own` entry to `false`/`0`) also removes everything nested inside that instance — but the nested `StructMask` still holds whatever boolean/float values it had. If a scoring strategy doesn't account for this, it will count the (now-unreachable) nested items' values as if they still affected the model's output, corrupting importance statistics.

### 5.2 The mechanism

`participate::Vector{Bool}` on every populated (`own !== nothing`) node records whether that node's units are currently reachable from the root given the *current* state of its ancestors.

**Correction (post-implementation):** an earlier draft of this section propagated a single whole-subtree alive/dead boolean top-down. That is wrong: a `BagNode`'s children are *one shared* `StructMask` describing a single instance's shape, indexed per-instance across all instances at once (matching how Mill stores bag content — one stacked array, not one subtree per instance). A single boolean cannot express "instance 1 is masked out but instances 2–5 aren't." Participation must instead be tracked **per instance index**, remapped through bag boundaries — matching `ExplainMill.jl`'s original `invalidate!` semantics — and, because that remapping (and whether a node's own units even correspond to observations at all) is storage-format specific, propagation is dispatched on `ds`'s type, mirroring `create_structmask`/`applymask`:

```julia
function updateparticipation!(ds::AbstractMillNode, m::StructMask)
    foreach_mask((mm, _) -> fill!(mm.participate, true), m)
    invalidate!(ds, m, Int[])
    m
end

# dense features are invariant across observations: nothing to invalidate
invalidate!(::ArrayNode{<:Matrix}, ::StructMask, invalid) = nothing

# categorical / n-gram: own IS observation-indexed, 1:1
function invalidate!(::ArrayNode{<:Mill.MaybeHotMatrix}, m::StructMask, invalid)
    isempty(invalid) || (m.participate[invalid] .= false)
end

# sparse: own is indexed by stored nonzero, translated via its column
function invalidate!(ds::ArrayNode{<:SparseMatrixCSC}, m::StructMask, invalid)
    isempty(invalid) && return nothing
    invalidset = Set(invalid)
    cols = sparsecolumns(ds.data)
    for i in eachindex(cols)
        cols[i] in invalidset && (m.participate[i] = false)
    end
end

# BagNode: own is per-instance; remap parent-level invalid bag indices to
# instance indices via `ds.bags`, fold in instances already off in this
# node's own mask, and recurse into the child at instance granularity
function invalidate!(ds::BagNode, m::StructMask, invalid_bags)
    ismissing(ds.data) && return nothing
    invalid_instances = isempty(invalid_bags) ? Int[] :
        reduce(vcat, (collect(ds.bags[i]) for i in invalid_bags); init=Int[])
    isempty(invalid_instances) || (m.participate[invalid_instances] .= false)
    combined = unique(vcat(invalid_instances, findall(.!(prunemask(m) .& m.participate))))
    invalidate!(ds.data, m.children, combined)
end

# ProductNode: doesn't change what "an observation" means; pass through
function invalidate!(ds::ProductNode, m::StructMask, invalid)
    foreach((c, cm) -> invalidate!(c, cm, invalid), Tuple(ds.data), Tuple(m.children))
end
```

`own`'s units don't always correspond 1:1 to observations (dense rows are feature-indexed, not observation-indexed — see §3's per-format table), so each leaf type either translates or, correctly, does nothing.

**Contract:** any scoring strategy that accumulates per-unit statistics must gate its update on `m.participate`, not just `prunemask(m)`. A unit that is "on" in its own mask but unreachable (an ancestor bag/instance was masked out) must not be counted as contributing.

`updateparticipation!(ds, m)` must be called after any mutation to `own` anywhere in the tree, before statistics are read. Scoring strategies that resample the whole tree each iteration (§8) call it once per iteration, immediately after sampling.

---

## 6. Applying a mask to data

Turning a `StructMask` into an actual (smaller, or soft-gated) Mill sample or model computation is the one place where per-storage-format knowledge is unavoidable — this is essential complexity, not an artifact of the type design, because a `SparseMatrixCSC`, a dense `Matrix`, a one-hot `MaybeHotMatrix`, and an `NGramMatrix` each need genuinely different code to remove/zero an entry.

### 6.1 Hard pruning (produces a new, smaller Mill sample)

```julia
applymask(ds::ArrayNode{<:Matrix}, m::StructMask) =
    ArrayNode(ifelse.(reshape(prunemask(m), 1, :), ds.data, missing), ds.metadata)

function applymask(ds::ArrayNode{<:SparseMatrixCSC}, m::StructMask)
    x = deepcopy(ds.data); x.nzval[.!prunemask(m)] .= 0
    ArrayNode(x, ds.metadata)
end

applymask(ds::ArrayNode{<:NGramMatrix}, m::StructMask) =
    ArrayNode(NGramMatrix(ifelse.(prunemask(m), ds.data.S, missing), ds.data.n, ds.data.b, ds.data.m), ds.metadata)
```

Each Mill leaf-array type requires exactly one `applymask` method. `BagNode`/`ProductNode` recursion through `applymask` follows the same node-type dispatch pattern as `create_structmask`.

### 6.2 Soft/differentiable masking (used by gradient-based scoring)

Rather than masking the input data, differentiable strategies need the mask value to gate activations *inside* the model, blended against the model's own learned imputation value (`ψ`) — this is what lets the model treat a partially-masked unit consistently with how it was trained to treat missing data, and it is why this hook lives at the `Dense` layer, not at the data level:

```julia
function (m::Dense{<:Any,<:PreImputingMatrix})(x::AbstractMatrix, sm::StructMask)
    dm = reshape(softvalue(sm), 1, :)
    y  = m.weight * x
    m.σ.(@.(dm * y + (1 - dm) * m.weight.ψ) .+ m.bias)
end

function (m::Dense{<:Any,<:PostImputingMatrix})(x::AbstractMatrix, sm::StructMask)
    dm = reshape(softvalue(sm), 1, :)
    y  = m.weight * x
    m.σ.(@.(dm * y + (1 - dm) * m.weight.ψ) .+ m.bias)
end
```

Because `softvalue`/`prunemask` are defined once on `StructMask` regardless of which Mill leaf-array type produced it, these two hooks (one per imputation direction) are the *entire* differentiable-masking surface — there is no need for a hook per `(array type × mask role)` combination as there was previously. An analogous pair of hooks is needed for `BagModel`'s aggregators (`SegmentedMean`, `SegmentedMax`, `BagCount`), each dispatching on `StructMask` generically rather than on a specific historical mask type.

---

## 7. Clustering and alternate masking axes (not yet designed — see §10)

Two capabilities the prior system supported are intentionally **not** designed in this document and are called out explicitly so they aren't assumed solved:

- **Clustering** (forcing several raw items to share one mask bit, to counter correlated-feature misattribution — `design.doc` §6) requires `own` to be sized to the number of *clusters* rather than raw items, plus a separate raw-item→cluster-index mapping. How that mapping composes with `StructMask`'s current fields is open.
- **Masking along an alternate axis** (the prior system's `ObservationMask`, which masks whole observations/columns rather than a node's primary axis) may fit as a tag (e.g. `dim::Symbol`) on the same `StructMask` rather than a separate type, but this hasn't been worked through against `applymask`'s dispatch.

Both are flagged in §10 as follow-ups, not resolved here.

---

## 8. Scoring bookkeeping is a separate tree

Per-unit importance statistics (a running Shapley-value estimate, a mean/variance accumulator, whatever a given strategy needs) are **not** stored on `StructMask` at all. They live in a parallel tree of identical shape, built by walking the `StructMask`:

```julia
struct AccTree{P,C}
    payload::Union{Nothing,P}
    children::C
end

create_acctree(m::StructMask, make_payload) =
    AccTree(m.own === nothing ? nothing : make_payload(length(m.own)),
            _mapchildren(c -> create_acctree(c, make_payload), m.children, 1, IdDict()))
```

`foreach_paired(f, sm, acc)` walks both trees in lockstep (same recursion shape as `foreach_mask`, zipping `sm.children` with `acc.children`), letting a strategy update its bookkeeping without the model ever being aware bookkeeping exists.

### 8.1 Why this separation matters

This is the mechanism that fixes the N×M growth problem described in §1.1. Example — a Shapley-value estimator:

```julia
struct DafExplainer
    n::Int
end

function stats(e::DafExplainer, ds, model, class, rng)
    sm  = create_structmask(ds, d -> trues(d))   # binary mask — no differentiability needed
    acc = create_acctree(sm, d -> Duff.Daf(d))   # bookkeeping lives here, not in sm
    y   = target(model, ds, class)
    for _ in 1:e.n
        sample!(sm, rng)
        updateparticipation!(sm)
        o = model(ds, sm)                        # model only ever sees StructMask
        f = softmax(o) ⋅ y
        foreach_paired(sm, acc) do leaf, leafacc
            Duff.update!(leafacc, f, leaf.own, leaf.participate)
        end
    end
    sm, acc
end
```

`model(ds, sm)` dispatches only on `StructMask` (via §6's hooks) and has no idea `Duff.Daf` exists. Adding a new scoring strategy therefore never requires touching `applymask` or the differentiable model hooks — it requires one accumulator payload type and one update rule, full stop.

---

## 9. Extension checklist

**Adding a new Mill node type** requires exactly:
1. `create_structmask(ds::NewNodeType, mk)` — decide leaf/hybrid/router shape (§3.2).
2. `applymask(ds::NewNodeType, m::StructMask)` — hard pruning (§6.1).
3. If differentiable scoring must support this type: the corresponding `Dense`/aggregator hook(s) (§6.2) — note these are keyed on imputation direction, not on node type, so this step is often already covered by existing hooks.
4. `_visitchildren`/`_mapchildren` need a new branch **only** if the new node type introduces a genuinely new children-container shape (not `nothing`/single-child/`Tuple`/`NamedTuple`); most new node types reuse an existing branch.

No changes are needed to `foreach_mask`, `mapmask`, `updateparticipation!`, or any existing scoring/pruning strategy.

**Adding a new scoring strategy** requires exactly:
1. An accumulator payload type (§8) — no relation to `StructMask` or the model.
2. An update rule consuming `(leaf::StructMask, leafacc::YourPayload)` pairs via `foreach_paired`.
3. A decision, made once, about whether this strategy needs `softvalue` (binary mask + `prunemask` suffices otherwise).

No changes are needed to `applymask`, the differentiable model hooks, or any existing node-type support.

---

## 10. Open questions

1. Clustering support (§7) — how does a cluster-index mapping compose with `StructMask.own`'s sizing?
2. Alternate masking axis (§7) — is a `dim` tag on `StructMask` sufficient to replace the prior system's `ObservationMask`, or does it need its own construction/application path?
3. Should `AccTree` (§8) be a named, documented public type, or an internal implementation detail of each scoring strategy (i.e., can two strategies share one `AccTree` instantiation, or does each always build its own)?
4. `FlatView` (a flat linear index over an entire `StructMask` tree, used by pruning strategies that want to treat the whole sample as one vector) is out of scope for this document — it is an adapter *over* `StructMask`, not a variant of it, and should be specified separately once pruning-strategy design is finalized.

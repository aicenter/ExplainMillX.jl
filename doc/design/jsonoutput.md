# JSON Output Design

**Status:** Draft — settled, about to be implemented
**Audience:** Engineers implementing the JSON reconstruction stage of `ExplainMillX.jl`
**Scope:** Converting a pruned `StructMask` back into a JSON-shaped value (nested `Dict`/`Vector`/scalars/`nothing`) using a JsonGrinder extractor and the sample's preserved metadata. Mask structure is covered in `masks.md`; scoring/pruning/`explain` are covered in `pruning.md`/`design.doc`.

---

## 1. Purpose

`explain(...)` returns a pruned `StructMask`; applying it (`ds[mask]`) gives a pruned *Mill* sample — correct, but not directly legible to a human, since it's still one-hot columns and matrices, not the atom/bond/field values a person reasons about. `masks.md` §5.4 deferred exactly this conversion. This document designs it.

The mechanism, once metadata is available, is simple in spirit: walk the sample and the mask together, and for every leaf item, either emit the *original* JSON value (item kept) or `nothing` (item pruned). The complexity is entirely in getting the recursive *shape* reconstruction right — that's where a JsonGrinder extractor becomes unavoidable, not just Mill node types (§3).

---

## 2. Prerequisite: metadata must be present

This feature only works on samples extracted with `store_input=Val(true)` (or `extract(e, ...; store_input=Val(true))`), so every leaf's `.metadata` carries the original raw JSON value(s) rather than `nothing`. `applymask` never touches `.metadata` (`masks.md` §6.1), so it survives pruning unchanged regardless of what happened to `.data`.

**Contract:** `explain_json` requires metadata wherever it needs to read a value. The first time it encounters a needed leaf with `metadata === nothing`, it raises a clear error pointing at `store_input=Val(true)`, rather than silently producing `nothing` (indistinguishable from a legitimately-pruned item) or garbage. This is the same "fail loud, don't coerce" stance already used elsewhere (`softvalue` on a binary mask, `prune!`'s pre/postcondition checks).

---

## 3. Dependency: JsonGrinder is a hard dependency

`explain_json` dispatches on JsonGrinder extractor types (`ScalarExtractor`, `CategoricalExtractor`, ...) because the extractor — not the Mill node type — determines the JSON *shape* to reconstruct. An `ArrayNode` could have come from a categorical, n-gram, or scalar extractor; a `ProductNode` could be a plain object or (in the deferred polymorphic case, §7) one branch of a type union. Only the extractor disambiguates this.

JsonGrinder is therefore added directly to `ExplainMillX`'s `Project.toml` `[deps]`/`[compat]` — not behind a package extension. This is a deliberate exception to the project's general dependency-light stance (`design.doc`, the earlier decision not to pull in NNlib/Flux for a `softmax` helper in `explain.jl`): JSON reconstruction is fundamentally a JsonGrinder-shaped feature, not an optional add-on, so hiding the dependency behind an extension would just relocate complexity without removing a real coupling.

---

## 4. Core dispatch: one method per extractor type

Mirroring `create_structmask`'s one-method-per-Mill-node-type pattern (`masks.md` §3.2, §9), `explain_json(ds, mask, extractor)` gets one method per **extractor type**:

| Extractor | Mill node | Behavior |
|---|---|---|
| `ScalarExtractor` | `ArrayNode{<:Matrix}` | per-observation: kept → metadata value, pruned → `nothing` |
| `CategoricalExtractor` | `ArrayNode{<:Union{MaybeHotMatrix,OneHotMatrix}}` | same |
| `NGramExtractor` | `ArrayNode{<:NGramMatrix}` | same |
| `StableExtractor` | (wrapper; no node-type change) | passthrough: unwrap and delegate to `e.e` |
| `ArrayExtractor` | `BagNode` | reconstruct a JSON array; bag-level participation (`present`, mirroring `applymask`'s own `BagNode` dispatch) decides which instances survive |
| `DictExtractor` | `ProductNode{<:NamedTuple}` | reconstruct a JSON object: recurse per named child, drop children whose reconstruction is entirely `nothing`, assemble into a `Dict` |

Each leaf method reads `.metadata` per kept item and substitutes `nothing` per pruned item, using the same `prunemask`-derived keep/drop decision every other stage already uses — no new masking concept, just a new *destination* for the boolean.

### 4.1 Explicitly out of scope

- **`PolymorphExtractor`** (union-typed JSON fields, extracted redundantly into multiple branches) — not supported anywhere in `ExplainMillX` today; deferred along with everything else in the library that would need it.
- **`ProductNode{<:Tuple}`** — JsonGrinder only produces this for polymorphic/positional cases, so it falls out naturally once `PolymorphExtractor` is deferred.

Both get an explicit fallback method that raises a clear "extractor type X is not supported by `explain_json` yet" error — never silent mishandling.

---

## 5. No batching

The original's `exportobs`/`soa2aos` machinery exists to explain *multiple root samples* in a single call. `ExplainMillX`'s `stats`/`prune!`/`explain` are single-sample throughout, so there is no batch dimension to thread through here either. `explain_json` operates on exactly the one `(ds, mask)` pair it's given — this drops an entire axis of complexity relative to the original, not just a naming difference.

(Within one sample, a `BagNode` still has many *instances* — that's ordinary bag content, unrelated to batching multiple root samples, and is handled the same way `applymask` already handles it: via `prunemask`/participation over the bag's own units.)

---

## 6. Absent representation: `nothing`

Pruned items are represented as `nothing`, matching the original and — more importantly — matching what `JSON.jl` needs: `JSON.json(result)` serializes `nothing` as `null` natively, so the direct, expected use of this feature ("get real JSON back out") works with no extra handling. This is the one place `ExplainMillX`'s "pruned ⇒ `missing`" convention (used throughout `applymask`, `masks.md` §6) is deliberately *not* followed — `explain_json`'s output is a JSON-shaped value, not a Mill sample, and JSON has no native `missing`.

---

## 7. The no-op leaf case

A `StructMask` with `own === nothing, children === nothing` (the "nothing to explain here" leaf used for unsupported Mill node types, `masks.md` §2) is always treated as **fully present** — `explain_json` returns its raw metadata value(s) unconditionally, without consulting any mask, mirroring the original's `EmptyMask` behavior (`contributing(::EmptyMask, l) = Fill(true, l)`). A field that was never eligible for pruning always keeps its full original value in the reconstructed JSON.

---

## 8. Cleanup pass

A small, separate post-processing function (not folded into the recursive walk, matching the original's `yarason`/`prunejson` split) recursively:
- strips `nothing` entries out of reconstructed `Dict`s,
- collapses a `Vector`/`Dict` that is now entirely empty (or entirely `nothing`) to `nothing` itself, one level up.

This is what makes a fully-pruned subtree disappear from the output entirely, rather than showing up as an explicit wall of nulls — "this whole branch wasn't part of the explanation" reads as absence, not as a dict full of `null`s.

---

## 9. Entry point and naming

One public function: `explain_json(ds::AbstractMillNode, mask::StructMask, extractor) -> Any` (a JSON-shaped value — nested `Dict`/`Vector`/scalar/`nothing`), performing the recursive reconstruction (§4) and the cleanup pass (§8) together. No separate single-vs-batch unwrapping step is needed (§5).

Named `explain_json` rather than reusing the original's `yarason`/`e2boolean` — those names describe neither behavior nor purpose to a new reader; `explain_json` says directly what the function produces.

---

## 10. Extension checklist

For each new extractor type `ExplainMillX` should support:

1. Confirm which Mill node type(s) it produces (check the extractor's own `(e::Extractor)(v)`/`extract` methods in JsonGrinder).
2. Add one `explain_json(ds::NodeType, mask::StructMask, ::ExtractorType)` method, following whichever existing case (leaf, bag, or product) matches its shape.
3. If the extractor wraps another extractor (like `StableExtractor`), add a passthrough method unwrapping to the inner extractor, not a full reimplementation.

No changes are needed to `create_structmask`, `applymask`, or any pruning/scoring code — this stage only ever reads a mask that already exists, via `prunemask`/`.metadata`; it never constructs or mutates one.

---

## 11. File layout

`src/output/jsonoutput.jl`, mirroring the original's `src/output/` directory, included from `ExplainMillX.jl` after `explain.jl`. `JsonGrinder` added to `Project.toml` `[deps]`/`[compat]` (§3).

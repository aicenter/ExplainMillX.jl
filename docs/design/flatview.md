# FlatView Design

**Status:** Draft — result of design discussion, not yet implemented
**Audience:** Engineers implementing the pruning stage of `ExplainMillX.jl`
**Scope:** The flat-view adapter used by pruning's local-search primitives. Mask structure is covered in `masks.md`; the search primitives and strategy axes that consume `FlatView` are covered in `pruning.md`.

---

## 1. Why `FlatView`, and why it's a good idea

### 1.1 The problem it solves

The local-search primitives pruning needs — bisection-add along an order, single-step greedy forward/backward selection, batched add/remove fine-tuning, redundancy removal (`pruning.md` §2.1) — are all naturally expressed against **a flat, linear, mutable collection of independent items**: "how many candidates are there, can I turn candidate `i` on or off, what does the objective say afterward." None of these algorithms care about hierarchy; they only need indexed get/set and a count.

`StructMask`, by design (`masks.md` §2), is *not* flat — it's a recursive tree with three node roles (leaf, hybrid, router). Without an adapter, every search primitive would need to know how to walk that tree itself, which reintroduces exactly the per-consumer tree-walking code that `foreach_mask`/`mapmask` were built to eliminate on the masking side (`masks.md` §4). `FlatView` is the pruning stage's analogue of those two primitives: one generic adapter, built once, so search algorithms are written against a flat interface and stay entirely ignorant of tree shape.

### 1.2 Why a *view*, not a copy

`FlatView` wraps the real `StructMask` nodes' `own` vectors in place — it does not copy values out, mutate a scratch buffer, and sync back. Each flat index stores a reference to the actual `StructMask` node it belongs to plus a local index into that node's `own`; reading or writing through the flat index reads or writes the real mask directly.

This matters because several of the primitives it serves (`finetune!`'s visited-state cycle detection, `randomremoval!`, `greedyremoval!`) mutate the candidate set repeatedly and re-evaluate the objective after every change. A copy-based design would need an explicit "reconcile back to the real mask" step around every such call, or would need to duplicate bookkeeping (participation, node identity) in two places. Aliasing means the flat view *is* the mask, from a different angle — there is nothing to reconcile, ever. This is why is called  a "View" rather than a "Mask": the name records that it's an aliasing lens over existing storage, not an independent data structure.

### 1.3 Why one type serves both pruning granularities

`pruning.md` §3.1 distinguishes flat-granularity pruning (search over the whole tree's items at once) from level-by-level pruning (search over one hierarchy depth at a time, moving deeper once each level is settled). These are not different algorithms — they are the same search primitives, invoked over different *subsets* of nodes. `FlatView` is therefore parametrized only by "which `StructMask` nodes to include," never by "which granularity strategy is calling me." Flat pruning builds one `FlatView` over every own-bearing node in the tree; level-by-level pruning builds a fresh `FlatView`, once per depth, over just that depth's nodes. Both call the identical `search!`/`randomremoval!`/`finetune!` machinery. See §4 for the loop this enables.

---

## 2. Basic functionality

A `FlatView` is built from a collection of `StructMask` nodes — either "every own-bearing node in a tree" (via `collectmasks`) or an explicit filtered subset (e.g. one level's nodes). At construction, it records, for each flat index, which node it belongs to and its local position within that node's `own`:

```
itemmap[i] = (node, local_index)
```

On top of that record, `FlatView` provides:

- **`length(fv)`** — total number of items across all included nodes.
- **`getindex(fv, i)` / `setindex!(fv, v, i)`** — read/write straight through to `node.own[local_index]`, for the node and local index recorded at position `i`. No copying, no reconciliation.
- **`fill!(fv, v)`** — set every item to `v` (used to reset a view to all-off before a search begins).
- **`prunemask(fv)`** — the aggregated boolean "is this item currently on" view across all contained nodes (built from each node's own `prunemask`, concatenated in flat order).
- **`participate(fv)`** — the aggregated boolean reachability view across all contained nodes (each node's `.participate`, concatenated in flat order). See §5 for how this is used.
- **`useditems(fv)`** — the flat indices currently on (`findall(prunemask(fv))`); this is what `randomremoval!`/`greedyremoval!`/`finetune!` iterate over.
- **`copyto!(fv, values)`** — bulk-assign a plain vector into the view's underlying storage, ignoring any structural dependencies (mirrors the original's escape hatch for tests/bookkeeping that already have a flat vector in hand).

`FlatView` is deliberately blind to two things: it doesn't know or care whether a node's `own` is `V<:AbstractVector{Bool}` or `V<:AbstractVector{<:Real}` (it just proxies raw element get/set, so it works unchanged for `GradientOrder`'s soft masks), and it has no notion of scoring, `AccTree`, or heuristic importance — that connection is deliberately kept external (§3).

---

## 3. Extracting heuristic scores (Option B: identity-keyed lookup)

`HeuristicOrder` (one of the three order strategies in `pruning.md` §3.3) needs, for each flat index, a precomputed importance score to sort candidates by. Because `StructMask` (the mask) and `AccTree` (its scoring bookkeeping) are two independently-built parallel trees (`masks.md` §8) rather than one fused structure, there's no single object to read "value and score" off together the way the original `DafMask` allowed.

The resolution settled on here is **identity-keyed lookup**, not positional alignment:

1. **Build the lookup once, right after scoring**, by walking `mask` and `acctree` together via `foreach_paired` (which already visits both trees in lockstep) and recording, per own-bearing node, its plain numeric scores:

   ```
   scores = IdDict{StructMask, Vector{Float64}}()
   foreach_paired(mask, acctree) do node, payload, level
       scores[node] = score.(payload)   # `score` converts a strategy's accumulator to a Float64
   end
   ```

2. **Key by object identity (`IdDict`, `===`), not by value.** Two nodes can hold identical-looking `own` vectors while representing entirely different features; only identity is a correct key. This is the same tool (`IdDict`) already used by `foreach_mask`/`mapmask` for their own memoization.

3. **Store plain `Float64` scores, not the raw accumulator.** `score.(payload)` runs once, eagerly, at lookup-construction time. This means the pruning module never needs to know that `MeanDiff` (or any other strategy's payload type) exists — the conversion from "strategy-specific bookkeeping" to "a plain importance number" happens once, at the scoring→pruning boundary, not scattered through search code.

4. **Look up at use time via the flat view's own bookkeeping** — no new plumbing required. `FlatView` already stores `(node, local_index)` per flat index (§2, needed for aliasing); `HeuristicOrder` reuses that exact same `node` reference as the dictionary key:

   ```
   node, local_index = itemmap[i]
   value = scores[node][local_index]
   ```

### 3.1 The invariant this requires

The `StructMask` a `FlatView` is built over must be the *same object*, by `===`, as the one `scores` was built against — not a freshly-reconstructed mask for "the same" sample, and not the result of passing the mask through `mapmask` afterward (`mapmask` builds new `StructMask` wrapper nodes even when the underlying `own` vector is reused, so node identities change across a `mapmask` call). Concretely: pruning must operate on the exact `(mask, acctree)` pair a scoring strategy's `stats(...)` returned. A lookup miss should raise loudly rather than default to zero — a silent zero would quietly corrupt the search order with no signal, which is the same "fail loud, don't coerce" stance already taken by `softvalue` erroring on a binary mask (`masks.md` §3).

### 3.2 Scope: this only matters for one order strategy

`GreedyForward` needs no precomputed scores at all — it's pure stepwise `addone!`/`removeone!`. `GradientOrder` computes its ranking live each round via autodiff directly over the `own` objects the `FlatView` already references, and never touches `AccTree`. So the identity-lookup machinery in this section is entirely confined to constructing `HeuristicOrder`; it does not leak into `FlatView` itself, the pruning driver loop, or the other two order strategies.

---

## 4. Level-by-level pruning as a client of `FlatView`

Level-by-level pruning is not a separate implementation from flat pruning — it is a driver loop that repeatedly constructs a `FlatView` over a narrower node set and reuses the identical search machinery:

```
for level in 1:maxdepth(mask)
    updateparticipation!(ds, mask)                     # see §5
    level_nodes = [n for (n, l) in collectmasks(mask) if l == level]
    fv = FlatView(level_nodes)
    search!(f, fv, strategy.order)                     # same search!, same primitives as flat pruning
    strategy.random_removal && randomremoval!(f, fv)
end
strategy.finetune && finetune!(f, FlatView(collectmasks(mask)))   # optional whole-tree cleanup pass
```

Because `collectmasks` returns references to the actual `StructMask` node objects (not copies), the nodes a per-level `FlatView` is built over are `===`-identical to the ones in the whole-tree mask — which means the `IdDict` from §3, built once for the whole tree right after scoring, keeps working unchanged as pruning descends level by level. There is no need to rebuild or re-slice the heuristic lookup per level; a per-level `FlatView`'s `itemmap` entries simply happen to reference a subset of the keys already present in `scores`.

This is also the answer to why level-by-level pruning doesn't need its own protocol distinct from flat pruning: the only genuinely level-specific step is the participation refresh between levels, covered next.

---

## 5. Participation in level-by-level pruning

### 5.1 Why it's needed between levels

A decision made at a shallower level can make items at a deeper level unreachable — turning off a `BagNode` instance means every item inside that instance's subtree can no longer affect the model's output, regardless of its own mask value (`masks.md` §5). Level-by-level pruning commits to decisions level by level, so before searching level `k+1`, participation must be recomputed to reflect whatever level `1..k` just decided. That's the `updateparticipation!(ds, mask)` call at the top of each loop iteration in §4 — it's not optional bookkeeping, it's what makes deeper levels' search meaningful at all.

### 5.2 What participation gates, precisely

Participation does **not** shrink which items a level's `FlatView` contains — the view still spans every item belonging to that level's nodes, participating or not. What it gates is **which items are offered as candidates when a level's search begins**: `FlatView`'s `participate(fv)` (§2) is intersected with whatever order `HeuristicOrder`/`GreedyForward` would otherwise use, so that only currently-reachable items are ever proposed as "turn this on" candidates during the add phase.

This distinction matters for a concrete reason: a non-participating item's value literally cannot change the objective (its ancestor already excludes it from `ds[mask]` independent of its own boolean). Including it as a candidate in `addminimumbi!`'s bisection would be worse than merely wasted evaluations — bisection assumes the candidate order is roughly monotonic in effect, and a "phantom" item that never changes `f()` regardless of value breaks that assumption silently. Filtering the *candidate order* by participation, while still constructing the view over the full item range, avoids this without needing the view itself to know anything about reachability semantics beyond exposing the raw `participate` vector.

### 5.3 Why removal passes don't need special participation handling

`randomremoval!`/`greedyremoval!`/`finetune!` all operate over `useditems(fv)` — the items *currently on* — not over participation directly. Under the protocol in §4, non-participating items are never turned on in the first place (the add phase excludes them from its candidate order, per §5.2), so they're never present in `useditems(fv)` for a removal pass to consider. Participation therefore only needs to be threaded through the *add* step; every other primitive already gets the right behavior for free, as a consequence of add never having offered those items to begin with.

### 5.4 Putting it together

The full per-level protocol is: recompute participation given levels already decided → build a `FlatView` over this level's nodes → restrict the add-candidate order to participating items only → run the ordinary search primitives unchanged → move to the next level, where the just-made decisions (which instances/items are now on) become the input to the next `updateparticipation!` call. Nothing about this requires `FlatView` to be participation-aware at construction time; it only needs to expose `participate(fv)` as one more flat, aggregated accessor alongside `prunemask(fv)`.

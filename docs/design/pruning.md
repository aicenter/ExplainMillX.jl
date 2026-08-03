# Pruning Design

**Status:** Draft — survey of prior art (`ExplainMill.jl`) complete; direction settled; `FlatView` (§3.1's flat adapter) implemented and tested (`docs/design/flatview.md`), local-search primitives and the strategy catalog below not yet implemented
**Audience:** Engineers implementing `ExplainMillX.jl`
**Scope:** The pruning stage only — searching a mask for a minimal item subset that keeps an injected objective `f() ≥ 0`. Mask structure is covered in `masks.md`; the flat adapter pruning operates through is covered in `flatview.md`; the overall pipeline and objective-injection contract are covered in `design.doc` §4.

---

## 1. Purpose

Given a scored mask (from a scoring strategy, see `masks.md` §8) and an objective closure `f() -> Real` (positive means "decision preserved"), pruning searches for a small subset of mask items that keeps `f() ≥ 0`. This document surveys how the prior system (`ExplainMill.jl`) implemented this, identifies what's worth keeping versus what's broken or redundant, and proposes a direction for `ExplainMillX.jl`.

---

## 2. Survey of prior art (`ExplainMill.jl`)

### 2.1 What's solid: the local-search primitives

`src/pruning/utils.jl` implements a set of generic operations over a `FlatView` (a flat linear index over a mask tree, or a slice of one):

| Primitive | Behavior |
|---|---|
| `addminimumbi!` | Turn on items in a given order via bisection until `f() ≥ 0`. O(log n) evaluations of `f`, relying on the given order already being roughly monotonic in effect. |
| `addone!` / `removeone!` | Single-step greedy forward/backward selection — try every remaining candidate, keep whichever addition/removal most improves `f`. O(n) evaluations per step. |
| `finetune!` | A small local-search pass: alternates batched add/remove (`finetuneadd!`/`finetuneremove!`), tracks visited item-sets in a `Dict` to detect cycling, grows the batch size when stuck, and reverts to the smallest/best visited state (`settobest!`) at the end. |
| `randomremoval!` / `greedyremoval!` | Redundancy-removal passes — repeatedly try to turn an item back off, in random or greedy-least-damaging order, keeping the removal only if `f() ≥ 0` still holds. |

These are genuinely reusable, well-decoupled from any model/task specifics (they only call the injected `f()`), and should carry forward largely unchanged.

### 2.2 What's a thin, consistent composition: `flatsearch.jl` / `levelbylevel.jl`

Two orthogonal axes are already visible in how the primitives are combined:

- **Search order**: `flatsearch!` orders candidates by a precomputed heuristic score and bisection-adds (`addminimumbi!`); `flatsfs!` ignores any precomputed score and does plain greedy forward selection (`sfs!`, built from `addone!`).
- **Granularity**: `levelbylevelsearch!`/`levelbylevelsfs!` are the same two searches, just invoked once per hierarchy depth (via `collect_masks_with_levels`) instead of once over the whole flattened tree.

Both axes reuse the exact same primitives from §2.1 — this compositional structure is worth preserving; it's already close to the "orthogonal axes" design `design.doc` §5.2 calls for.

### 2.3 What's broken or misleading

These were confirmed by reading the actual implementation, not inferred:

1. **`LbyLo_*` (the "output-aware," partially-evaluated level-by-level variants) do not deliver their advertised benefit.** The intent is to call `partialeval` at each level so unaffected subtrees of the model aren't re-evaluated — a real optimization for large samples. In `levelbylevelsearch!`, this call is commented out with the note `I have disabled partialeval, since there is some bug there`, and the code falls back to evaluating the exact same `f(model, ds, mk)` as the non-optimized `LbyL_*` path. `LbyLo_*` is exposed by `prune!`/`pruning_methods()` as a distinct, faster method while actually costing the same as `LbyL_*`, with no signal to the caller that the optimization didn't happen.

2. **`greedy_gradient.jl` is an abandoned prototype, not a working strategy**, despite `GreedyGradient` being exported from the package:
   - `GradientMask` duplicates `SimpleMask`'s representation and API almost exactly — an unnecessary second type for what is the same value semantics (a `[0,1]`-valued, thresholdable vector).
   - A top-level function `greedy_gradient_lbyl(ds; ...)` references a bare `model` that is neither a parameter nor a module-level global anywhere in the file — this function cannot execute.
   - `explain(::GreedyGradient, ...)` duplicates `explain.jl`'s mask-construction and threshold logic instead of calling into it, and only handles 2 of the 18 advertised `pruning_method` symbols; the `else` branch's error message is a literally unfinished sentence (`"...Flat methods are not supported at all, because the"`).
   - Unlike the `LbyLo` path (disabled but consistently so), here `partial_evaluation` is used *unconditionally* when the caller requests it — inconsistent with the rest of the codebase treating that code path as known-buggy.

3. **Symbol dispatch has drifted from what it actually dispatches to.** `pruning.jl`'s two `prune!` `if/elseif` chains have different, overlapping sets of valid symbols, and both `error()` branches print the same stale "possible values" list that matches neither branch's real options (it omits `LbyLo_*` and several `G*` variants entirely). This is the concrete, already-realized failure mode `design.doc` §9 cites as the reason for choosing typed strategy objects over symbols for `ExplainMillX.jl`.

4. **A silent naming/behavior mismatch caused by a hidden default parameter.** `flatsfs!`'s signature defaults to `random_removal=true`. `prune!`'s dispatch for `:Flat_Gadd` calls `flatsfs!(f, mk)` with no override — so despite "Add" in the name implying *no* removal (matching `:Flat_HAdd`'s actual no-removal behavior), `:Flat_Gadd` silently performs random removal anyway, purely because of an unrelated function's default argument. This is a second, independent instance of point 3's underlying problem (a selector that no longer reflects what it actually triggers), and it's caused specifically by letting a *function default* stand in for what should be an *explicit, always-stated field* — see §3.6.

5. **Non-convergence is logged, not surfaced.** At the end of `levelbylevelsearch!`, if the final mask doesn't actually satisfy the objective, the code does `@error "output of explanation is ... and should be zero"` — a log line, not an exception or part of the return value — so a caller of `explain(...)` has no reliable, programmatic way to detect that the returned "explanation" doesn't meet the requested tolerance.

---

## 3. Proposed direction for `ExplainMillX.jl`

### 3.1 One driver function, parametrized by a typed strategy, built on `FlatView`

`flatsearch!`, `flatsfs!`, `levelbylevelsearch!`, and `levelbylevelsfs!` differ only along the two axes already identified in §2.2. Collapse them into one function. `FlatView` (`flatview.md`) is already implemented and provides exactly the flat, aliasing, mutable index this driver needs — for either granularity, since `FlatView` is parametrized only by "which nodes," never by which strategy is calling it (`flatview.md` §1.3):

```julia
struct PruningStrategy{Order}
    order::Order            # HeuristicOrder(), GreedyForward(), or GradientOrder() (see §4)
    levelbylevel::Bool       # false: whole tree at once. true: one pass per hierarchy depth
    random_removal::Bool
    finetune::Bool
end

function prune!(mask::StructMask, ds, model, f, strategy::PruningStrategy)
    for fv in flatviews(mask, ds, strategy.levelbylevel)   # one FlatView, or one per depth level;
        search!(f, fv, strategy.order, ds, model)          # updateparticipation! threaded between levels
        strategy.random_removal && randomremoval!(f, fv)
    end
    strategy.finetune && finetune!(f, FlatView(mask))
end
```

Note `ds`/`model` are threaded through the driver even though only `GradientOrder` uses them (§4.3) — `HeuristicOrder`/`GreedyForward` simply ignore the extra arguments. This keeps one uniform driver signature rather than the original's bifurcated `prune!(f, mk, method)` vs. `prune!(f, model, ds, mk, method)` split.

`search!(f, fv, ::HeuristicOrder, ds, model)` and `search!(f, fv, ::GreedyForward, ds, model)` wrap `addminimumbi!`/`sfs!` respectively — no behavior changes here, just a single entry point instead of four near-duplicate wrapper functions. Adding a new order or fixing granularity handling no longer requires touching the other axis, matching the extension checklist already written for masks.

### 3.2 Partial evaluation: fix it or don't claim it

Don't carry the `LbyLo_*` situation forward as-is — a method that's selectable and claims to be faster while silently costing the same as another method is worse than not offering it, because the caller has no way to notice. Two acceptable paths:

- Invest in fixing `partialeval` and gate it behind a tested `partial_eval::Bool` field on `PruningStrategy`, verified by a test that asserts the partially-evaluated and fully-evaluated objective agree (the prior code even sketches this assertion, commented out, in `greedy_gradient_lbyl!`).
- Drop the optimization from the v1 catalog entirely and reintroduce it once it demonstrably works.

Either is fine; silently shipping the broken middle ground is not.

### 3.3 `HeuristicOrder`: implementation grounded in `FlatView`/`nodescores`

Now that `FlatView` and the identity-keyed lookup exist (`flatview.md` §3), `HeuristicOrder`'s implementation is concrete rather than sketched:

1. Right after scoring, build `scores = nodescores(mask, acctree, score)` once (an `IdDict{StructMask,Vector{Float64}}`).
2. For whichever `FlatView` the driver is currently searching (whole-tree, or one level), call `heuristicscores(fv, scores)` to get a plain `Vector{Float64}` aligned to that view's flat indices.
3. Sort flat indices by that vector and feed the order into `addminimumbi!` (ported from §2.1, retargeted at `FlatView`'s `getindex!`/`setindex!`).

Because `scores` is identity-keyed rather than positionally aligned, this works unchanged whether `fv` spans the whole tree or one level's nodes (`flatview.md` §4) — no re-slicing or re-deriving the score vector per level.

### 3.4 `GreedyForward`: no precomputed signal needed

Port `addone!`/`removeone!` (and `sfs!`, the loop repeatedly calling `addone!` until `f() ≥ 0`) onto `FlatView` directly — no `AccTree`/`nodescores` involvement at all. Candidates are restricted to `participate(fv)`-true indices when relevant, per `flatview.md` §5.2.

### 3.5 `GradientOrder`: folds into the same framework, but has an unmet prerequisite

`greedy_gradient_flat!`'s core idea — rank candidates by `∂f/∂(mask value)` via `Flux.gradient`, then greedily accept the best-scoring improving candidate — doesn't need a bespoke mask type or a separate `explain` path. Because `StructMask{C,V}` (`masks.md` §3) already gives a differentiable value via `softvalue` whenever `V<:AbstractVector{<:Real}`, gradient ranking is naturally just another `Order`:

```julia
struct GradientOrder end

function search!(f, fv, ::GradientOrder, ds, model)
    ps = Flux.params(map(n -> n.own, unique(first.(fv.itemmap))))
    # rank candidates by ∂(soft objective)/∂own, greedily accept the
    # best-ranked candidate that improves the *hard* f(), repeat
end
```

This removes the parallel `GreedyGradient`/`explain(::GreedyGradient, ...)` path (its duplicated threshold logic, its broken free function, its 2-of-18 method support) and makes gradient-guided search usable with both flat and level-by-level granularity, and with fine-tuning/random-removal — unlike the original, which only supported two of its eighteen advertised method combinations for this order.

**Unmet prerequisite, confirmed by inspection, not assumed:** computing that gradient requires evaluating the model on a *soft*-masked sample — `model(ds, mask)`, using `mask`'s `own` multiplicatively/via imputation blending, as opposed to the hard-pruned `model(ds[mask])` every other order uses. That is the differentiable model-hook layer sketched in `masks.md` §6.2 (the `Dense{...,PreImputingMatrix}`/`Dense{...,PostImputingMatrix}` dispatches on `StructMask`, plus the equivalent `BagModel` aggregator hooks). **None of this exists in the codebase yet.** `GradientOrder` is therefore not "one more `search!` method to write" — it depends on a separate, self-contained piece of work (the soft-forward-pass machinery) that hasn't been started. Until that lands, `GradientOrder` should be documented as unimplemented rather than attempted.

A second wrinkle specific to this order, independent of the missing hooks: it needs *both* a differentiable objective (to compute gradients from) and the ordinary hard objective (to decide when to actually accept/stop) — the original conflates these in its flat-granularity path and separates them explicitly in its level-by-level path, an inconsistency not worth inheriting. The resolution settled on here: don't have the caller inject a separate soft objective function. Since the hard objective is already just `objective(output)` for some plain `objective` function (`design.doc` §4.1), the soft objective is mechanically `() -> objective(model(ds, mask))` — the same `objective`, fed a soft forward pass instead of a hard one. This is why `search!` for this order alone needs `ds`/`model` in its signature (§3.1) — the other two orders ignore them entirely.

### 3.6 Selector state must be explicit fields, never a hidden function default

§2.3 point 4 documents a concrete bug caused by a strategy's behavior depending on an unrelated function's default parameter rather than on something visible at the call site. The fix generalizes beyond that one instance: every axis (`order`, `levelbylevel`, `random_removal`, `finetune`, and `partial_eval` if/when §3.2 lands) must be a required, explicit field on `PruningStrategy` — never left to a called function's default argument to decide silently. This is what makes the eighteen named combinations in §4.4 fully determined by reading the struct, with no need to also check what some inner function defaults to.

### 3.7 Convergence is part of the return contract

`prune!` should make it possible to tell, programmatically, whether the final mask actually satisfies `f() ≥ 0` — either by returning a small result struct (`(mask=..., converged=...)`) or by raising when it doesn't, rather than emitting an `@error` log line that a caller has no way to intercept. `explain(...)` should propagate this so callers can distinguish "here is your explanation" from "the search gave up before meeting your tolerance."

### 3.8 What not to change

- The local-search primitives in `utils.jl` (`addminimumbi!`, `addone!`/`removeone!`, `finetune!`, `randomremoval!`/`greedyremoval!`) work, are reasonably general, and don't depend on model/task specifics — carry them forward with at most minor signature adjustments to match `StructMask`/`FlatView` in `ExplainMillX.jl`.
- The objective-injection contract (`f()` as a zero-argument closure returning a real number) is already clean and decoupled from pruning internals — no changes needed there; see `design.doc` §4.1.

---

## 4. Catalog of concrete strategies

Every pruning strategy `ExplainMillX.jl` offers is a composition of three orthogonal axes. This section lists them individually with rationale and implementation status, then the resulting named combinations.

### 4.1 Order — `HeuristicOrder`

**Rationale:** a scoring strategy has already assigned per-item importance; sort by it and bisection-add until the objective is satisfied — `O(log n)` evaluations of `f()` rather than `O(n)`, trading a monotonicity assumption for speed.

**Implementation:** `nodescores` + `heuristicscores` (already implemented, `flatview.md` §3) + a ported `addminimumbi!`. **Buildable now** once the local-search primitives are ported.

### 4.2 Order — `GreedyForward`

**Rationale:** no external importance signal required (or none trusted) — at each step, actually try every remaining candidate, keep whichever single addition improves the objective most.

**Implementation:** ported `addone!`/`removeone!`/`sfs!` directly against `FlatView`, no scoring dependency. **Buildable now.**

### 4.3 Order — `GradientOrder`

**Rationale:** rank candidates by `∂f/∂(soft mask value)`, greedily accept the best-ranked improving candidate.

**Implementation:** as in §3.5. **Blocked** — requires the soft/differentiable model-forward hooks from `masks.md` §6.2, which are not yet implemented (confirmed by inspection: no `PreImputingMatrix`/`PostImputingMatrix`/`Dense` dispatch exists in `src/` beyond the `softvalue` accessor itself). Do not attempt this order before that separate work lands.

### 4.4 Granularity — flat vs. level-by-level

Not a separate implementation — the same `FlatView`/search machinery, invoked once over the whole tree vs. once per depth with `updateparticipation!` threaded between calls. Fully specified by `flatview.md` §4–§5; the remaining work is only the driver loop in §3.1. **Buildable now**, for both `HeuristicOrder` and `GreedyForward`.

### 4.5 Post-pass — random removal

**Rationale:** an add-phase's order doesn't guarantee minimality; items accepted early may become individually redundant once later items are added. Repeatedly try removing currently-on items in random order, keep the removal if `f() ≥ 0` still holds.

**Implementation:** ported `randomremoval!` (and `greedyremoval!` for a greedy-order variant) over `FlatView.useditems`. **Buildable now.**

### 4.6 Post-pass — fine-tuning

**Rationale:** a small local-search pass catching local-optimum artifacts the single-pass add/remove strategies leave behind — alternating batched add/remove, cycle detection, reverting to the best visited state.

**Implementation:** ported `finetune!`/`finetuneadd!`/`finetuneremove!`/`settobest!`, the most intricate piece of the original, retargeted at `FlatView` without redesign. **Buildable now.**

### 4.7 Named combinations

| Name (original) | Order | Granularity | Random removal | Fine-tune | Status |
|---|---|---|---|---|---|
| `Flat_HAdd` | Heuristic | flat | no | no | buildable now |
| `Flat_HArr` | Heuristic | flat | yes | no | buildable now |
| `Flat_HArrft` | Heuristic | flat | yes | yes | buildable now |
| `Flat_Gadd` / `Garr` / `Garrft` | GreedyForward | flat | no / yes / yes | no / no / yes | buildable now |
| `LbyL_H*` / `LbyL_G*` | Heuristic / GreedyForward | level-by-level | as above | as above | buildable now |
| `LbyLo_*` | either | level-by-level, partial-eval | as above | as above | deferred (§3.2): fix or drop, never ship silently broken |
| `GreedyGradient` | Gradient | flat or level-by-level (both, unlike the original's flat-unsupported restriction) | yes/no | yes/no | blocked on soft-masking hooks (§4.3) |

Every field in this table is an explicit, required `PruningStrategy` field (§3.6) — there is deliberately no combination whose behavior depends on a function default the table doesn't show.

---

## 5. Summary table

| Aspect | Prior art (`ExplainMill.jl`) | Proposed (`ExplainMillX.jl`) |
|---|---|---|
| Search primitives | `utils.jl` — solid, generic | Unchanged in spirit; retargeted at `FlatView` |
| Flat adapter | `FlatView` fused with search-only concerns | `FlatView` implemented and tested independently (`flatview.md`) |
| Granularity × order composition | 4 near-duplicate wrapper functions (`flatsearch!`, `flatsfs!`, `levelbylevelsearch!`, `levelbylevelsfs!`) | 1 driver (`prune!`) parametrized by a typed `PruningStrategy{Order}` |
| Strategy selection | `Symbol`, two inconsistent `if/elseif` chains, stale error messages, and a naming/behavior mismatch caused by a hidden function default (§2.3.4) | Typed strategy objects with required, explicit fields (no hidden defaults, §3.6) |
| Heuristic-order scoring | `Mask`/`DafMask` fused own+stats in one struct | Identity-keyed lookup (`nodescores`/`heuristicscores`), already implemented |
| Output-aware partial evaluation (`LbyLo_*`) | Advertised as a distinct method; silently disabled/no-op due to a known bug | Either genuinely implemented and tested, or removed from the catalog |
| Gradient-guided search (`GreedyGradient`) | Separate, broken/unfinished prototype: duplicate mask type, a function referencing an undefined global, an `explain` method supporting 2/18 pruning methods, flat granularity unsupported | One more `Order` value (`GradientOrder`), usable with any granularity/post-pass combination once its soft-masking prerequisite is built |
| Non-convergence handling | `@error` log line only | Part of the return contract (typed result or exception) |

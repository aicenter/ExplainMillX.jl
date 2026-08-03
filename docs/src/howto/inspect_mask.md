# [Inspect a pruned mask directly](@id howto-inspect-mask)

If you want programmatic access to what survived pruning -- without going
through [JSON output](json_output.md) -- work with the mask and Mill
sample directly.

## Apply the mask to get a pruned Mill sample

```julia
result = explain(ds, model)
pruned = ds[result.mask]     # a new, smaller Mill sample; ds itself is unchanged
```

Pruned items become `missing` in the returned sample's data (not removed
positionally for leaf arrays; bag instances that are pruned away *are*
removed, shrinking that bag's observation count).

## Walk every maskable item

```julia
for (node, level) in collectmasks(result.mask)
    println("level $level: ", count(prunemask(node)), " / ", length(node), " kept")
end
```

`collectmasks` returns `(node, level) => ...` pairs for every own-bearing
node in the tree, in depth order (`level` starts at 1 at the root and
increases going deeper -- see the note on router nodes "consuming" a
level number without contributing an entry, in
`doc/design/masks.md` if you need the exact semantics).

## Aggregate statistics without JSON

```julia
nodes = first.(collectmasks(result.mask))
n_total = sum(length, nodes)
n_kept = sum(count, prunemask.(nodes))
```

This is exactly what `ExplanationResult.n_total`/`.n_kept` already give
you (`fraction_kept(result)`, `fraction_pruned(result)`, `n_pruned(result)`)
-- reach for those first; this is what to fall back on if you need a
custom breakdown (e.g. per-level, or restricted to one field).

## Check reachability, not just the mask's own value

A mask item can be "on" in its own `prunemask` while still being
unreachable because an ancestor (e.g. a bag instance) was pruned away.
`participate` reports genuine reachability:

```julia
reachable_and_kept = prunemask(node) .& participate(node)
```

This matters mainly if you're writing your own pruning/scoring logic
against masks directly; `ds[mask]` and `explain_json` already account for
this correctly on their own.

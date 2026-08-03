# [Why hierarchy matters: masks respect a sample's structure](@id explanation-hierarchy-and-masks)

A Mill.jl sample is rarely a flat feature vector -- it's a tree: a
molecule is a product of scalar descriptors and a *bag* of atoms, each
atom is itself a product of an element, a charge, and a *bag* of bonds,
and so on. ExplainMillX's explanations respect this structure directly,
rather than flattening everything into one undifferentiated list of
"features."

## What gets pruned is shaped by what the sample is made of

Instead of one uniform notion of "feature," what counts as one maskable
unit depends on the kind of data it is:

- A dense numeric field (like a molecule's `logp`) is masked as *one unit
  shared across every observation* -- there's one scalar per field, so
  there's one decision about whether that field's value is revealed at
  all.
- A categorical field (like an atom's element) is masked *per
  observation* -- each atom's element is an independent decision,
  independent of every other atom's.
- A bag (like a molecule's atoms) is masked *per instance* -- pruning an
  atom out of the bag removes that whole atom, not just one of its fields,
  and everything nested inside that atom (its own fields, its own bag of
  bonds) becomes unreachable along with it.

This is why an explanation for a molecule can read as "these two specific
atoms, plus these two scalar descriptors" rather than a flat list of
column indices -- the pruning decisions are already expressed at the
level of the sample's own structure.

## Reachability cascades down the tree

If a bag instance is pruned away, everything nested inside it is
unreachable regardless of what its own internal mask says -- there's no
way for a pruned-away atom's bond information to somehow still show up in
the pruned result. This cascading is what makes "prune this atom" mean
what you'd expect: removing the atom removes its whole subtree, not just
a label on it.

## Why this is the right level to search at

Searching for a minimal subset in terms of the sample's own hierarchy
(rather than, say, first flattening everything to a vector, pruning that,
and hoping the result maps back to something coherent) means every
candidate the search considers is already a *structurally valid* partial
sample -- one you could actually feed back through the model, extract the
metadata for, or turn into a smaller JSON document. There's no
post-processing step needed to reconcile "which flat indices were kept"
with "does this still make sense as a molecule" -- the search only ever
proposes states that already do.

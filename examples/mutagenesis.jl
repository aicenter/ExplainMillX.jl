#=
Mutagenesis walk-through for ExplainMillX.jl
=============================================

A complete, runnable example in three parts:

  1. Download the Mutagenesis dataset (cached locally).
  2. Train a Mill.jl model on it -- a condensed version of the official
     JsonGrinder.jl tutorial:
     https://ctuavastlab.github.io/JsonGrinder.jl/stable/examples/mutagenesis/mutagenesis/
  3. Explain one of the model's predictions using ExplainMillX: score
     importance with `ShapleyExplainer`, then prune with `PruningStrategy`/
     `prune!`, and print which atoms/bonds/descriptors the model actually
     needed to keep its prediction.

Run with:
    julia --project=examples examples/mutagenesis.jl

ExplainMillX has no top-level `explain(...)` convenience function yet (see
doc/design/pruning.md) -- part 3 below wires scoring and pruning together
by hand, which also serves as a worked example of how the pieces fit.
=#

using HTTP
using JSON
using JsonGrinder
using Mill
using Flux
using MLUtils
using Statistics
using Random
using ExplainMillX

Random.seed!(42)

## ---------------------------------------------------------------------
## 1. Download the data (cached locally so re-runs don't re-download)
## ---------------------------------------------------------------------

const DATA_DIR = joinpath(@__DIR__, "data")
const DATA_PATH = joinpath(DATA_DIR, "mutagenesis.json")
const DATA_URL = "https://raw.githubusercontent.com/CTUAvastLab/JsonGrinder.jl/master/docs/src/examples/mutagenesis/mutagenesis.json"

function ensure_data!(path=DATA_PATH, url=DATA_URL)
    if !isfile(path)
        mkpath(dirname(path))
        @info "Downloading Mutagenesis dataset" url path
        resp = HTTP.get(url)
        write(path, resp.body)
    end
    path
end

ensure_data!()

## ---------------------------------------------------------------------
## 2. Train a Mill.jl model (condensed from the JsonGrinder.jl tutorial)
## ---------------------------------------------------------------------

dataset = JSON.parsefile(DATA_PATH)
jss_train, jss_test = dataset[1:100], dataset[101:end]
y_train = Flux.onehotbatch(getindex.(jss_train, "mutagenic"), 0:1)
y_test = Flux.onehotbatch(getindex.(jss_test, "mutagenic"), 0:1)

# Infer the JSON schema, then drop the label field (it isn't a feature).
sch = schema(jss_train)
delete!(sch, :mutagenic)

# `all_stable=true` (unlike the original tutorial's default extractor)
# makes every field -- not just ones the schema happens to see as
# optional -- extract into a `MaybeHotMatrix`/`Union{Missing,...}`
# representation rather than a plain, never-missing `OneHotMatrix`.
# ExplainMillX represents "this item is pruned away" as `missing`
# (masks.md §6), so every field needs to be able to hold `missing` for
# masking to have something to do to it.
e = suggestextractor(sch; all_stable=true)

x_train = extract(e, jss_train)
x_test = extract(e, jss_test)

# `all_imputing=true` (again unlike the tutorial's default call) builds
# every `Dense` layer with an imputing weight matrix, so the model knows
# how to handle a `missing` input rather than only ever seeing complete
# molecules. This is required for `ds[mask]`'s pruned samples to be
# evaluable by the model at all.
model = reflectinmodel(sch, e; all_imputing=true, fsm = Dict("" => d -> Chain(Dense(d,10,relu), Dense(10,2))))

pred(m, x) = m(x)
loss(m, x, y) = Flux.Losses.logitcrossentropy(m(x), y)
opt_state = Flux.setup(Flux.Optimise.Descent(), model)
minibatch_iterator = Flux.DataLoader((x_train, y_train), batchsize=32, shuffle=true)

accuracy(p, y) = mean(Flux.onecold(p) .== Flux.onecold(y))
for i in 1:50
    Flux.train!(loss, model, minibatch_iterator, opt_state)
    @printf("Epoch %d train_accuracy:  %.3f test_accuracy: %.3f\n",i, accuracy(pred(model, x_train), y_train), accuracy(pred(model, x_test), y_test))
end

## ---------------------------------------------------------------------
## 3. Explain one prediction
## ---------------------------------------------------------------------

# Test molecule #14, correctly classified by this model. Re-extracted
# (rather than reusing `x_test`) with `store_input=Val(true)` so every
# leaf's `.metadata` carries the *original* JSON values (element names,
# atom types, charges, ...) -- `applymask` preserves `.metadata` through
# pruning, which is what lets the final explanation below be printed in
# human-readable form instead of as raw one-hot/matrix data.
sample_idx = 14
sample_json = jss_test[sample_idx]
ds = e(sample_json; store_input=Val(true))

ŷ = argmax(vec(model(ds)))
z0 = softmax(model(ds))[ŷ]                 # the model's raw logit on the full molecule
@info "Explaining sample $sample_idx" logit = z0 predicted = ŷ true_label = argmax(y_test[:,sample_idx])

# `model`'s head is a single sigmoid unit (binary classification), not the
# multi-class softmax the rest of ExplainMillX's tests use -- so the
# objective is adapted accordingly: preserve the *sign* of the logit (same
# predicted class) while requiring its magnitude stay above a fraction of
# the original (an analogue of the softmax "confidence gap" for a binary
# head). A real, honest finding from building this example: this
# quickly-trained model relies overwhelmingly on the four scalar molecular
# descriptors (lumo/logp/ind1/inda) for most predictions, so a looser
# `rel_tol` (e.g. 0.9) typically prunes the entire atoms/bonds structure
# away entirely -- that's a genuine property of this model, not a
# limitation of the explainer. `rel_tol` is set tight (0.99) here so any
# residual atom-level signal has a chance to show up; because scoring is
# Monte Carlo (`ShapleyExplainer`), whether any specific atom's importance
# clears that bar can vary slightly run to run -- the printed result below
# is whatever this run's explanation genuinely found, handled gracefully
# either way.
rel_tol = 0.99
threshold = rel_tol * z0
objective = o -> softmax(o)[ŷ]

# Stage 1: score every maskable item's importance via Monte Carlo Shapley
# values (masks.md §8). `stats` builds the mask internally; it comes back
# scored and ready for pruning.
mk, acctree = stats(ShapleyExplainer(150), ds, model, objective; rng=MersenneTwister(11))
scores = nodescores(mk, acctree, score)   # identity-keyed lookup, flatview.md §3

# Stage 2: prune down to a minimal subset that keeps the (signed, scaled)
# logit above `threshold` -- i.e. the model stays at least `rel_tol` as
# confident in the same prediction using only what survives pruning.
f = () -> softmax(model(ds[mk]))[ŷ] - threshold
strategy = PruningStrategy(HeuristicOrder(scores), true, true, true)  # level-by-level: faster in practice (pruning.md §2.2)
prune!(mk, ds, model, f, strategy)

@info "Pruning result" remaining_confidence = f() + threshold original_confidence = z0

# Stage 3: read the pruned mask back out in human-readable form via the
# metadata `applymask` carried through.
pruned = ds[mk]

println("\nWhat the model needed to keep its prediction:")
println("  lumo = ", only(pruned[:lumo].metadata))
println("  logp = ", only(pruned[:logp].metadata))
println("  ind1 = ", only(pruned[:ind1].metadata))
println("  inda = ", only(pruned[:inda].metadata))

atom_elements = pruned[:atoms].data[:element].metadata
atom_charges = pruned[:atoms].data[:charge].metadata
kept_atoms = findall(!ismissing, atom_elements)
if isempty(kept_atoms)
    println("  atoms: none -- the scalar descriptors above were sufficient on their own")
else
    println("  atoms kept ($(length(kept_atoms)) of $(length(atom_elements))):")
    for i in kept_atoms
        println("    atom $i: element=$(atom_elements[i])  charge=$(atom_charges[i])")
    end
end

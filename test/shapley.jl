# Exercises the mask infrastructure end to end (random sampling, hard
# pruning, participation, paired bookkeeping traversal) via a Monte Carlo
# Shapley-style scoring strategy, per doc/design/masks.md §8.

@testset "ShapleyExplainer" begin
    @testset "repeatability given a fixed rng" begin
        ds = specimen_sample()
        model = f64(reflectinmodel(ds, d -> Dense(d, 4), all_imputing=true))
        objective = o -> Flux.softmax(o)[1]

        sm1, acc1 = stats(ShapleyExplainer(100), ds, model, objective; rng=MersenneTwister(1))
        sm2, acc2 = stats(ShapleyExplainer(100), ds, model, objective; rng=MersenneTwister(1))
        @test leafscores(sm1, acc1) == leafscores(sm2, acc2)

        sm3, acc3 = stats(ShapleyExplainer(100), ds, model, objective; rng=MersenneTwister(2))
        @test leafscores(sm1, acc1) != leafscores(sm3, acc3)
    end

    @testset "score count matches mask shape" begin
        ds = specimen_sample()
        model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing=true)
        sm, acc = stats(ShapleyExplainer(20), ds, model, o -> Flux.softmax(o)[1])
        @test length(leafscores(sm, acc)) == length(collectmasks(sm))
        scores = leafscores(sm, acc)
        for (mk, sc) in zip(first.(collectmasks(sm)), scores)
            @test length(sc) == length(mk)
        end
    end

    @testset "correctly attributes importance to the branch the model actually uses" begin
        an = ArrayNode(randn(Float32, 3, 30))
        bn = ArrayNode(randn(Float32, 3, 30))
        ds = ProductNode(a=an, b=bn)

        model = reflectinmodel(ds, d -> Dense(d, 1), all_imputing=true)
        # wire the model so its output depends only on branch `a`
        model.ms.b.m.weight.W .= 0
        model.ms.b.m.bias .= 0
        model.m.weight .= 1
        model.m.bias .= 0

        sm, acc = stats(ShapleyExplainer(600), ds, model, o -> o[1]; rng=MersenneTwister(1))
        a_scores, b_scores = leafscores(sm, acc)

        # the branch the model ignores should score near zero; the branch it
        # uses should have at least one clearly nonzero score
        @test maximum(abs, a_scores) > 5 * maximum(abs, b_scores)
    end
end

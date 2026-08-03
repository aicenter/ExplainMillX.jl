# Tests for PruningStrategy/prune! (docs/design/pruning.md §3.1, §4), built
# on the same synthetic weighted-sum style used in test/localsearch.jl for
# controlled, deterministic checks, plus an end-to-end test against a real
# Mill model exercising every (order, granularity) combination together.

@testset "PruningStrategy / prune!" begin
    @testset "GreedyForward, flat granularity" begin
        weights = Float64[5, 4, 3, 2, 1]
        an = ArrayNode(randn(Float32, 5, 1))
        mk = create_structmask(an, d -> fill(true, d))
        f = () -> sum(weights[prunemask(mk)]) - 6

        strategy = PruningStrategy(GreedyForward(), false, true, true)
        result = prune!(mk, nothing, nothing, f, strategy)

        @test result === mk   # returns the mutated mask, per Julia `!` convention
        @test f() >= 0
        @test prunemask(mk) == Bool[1, 1, 0, 0, 0]
    end

    @testset "HeuristicOrder, flat granularity" begin
        weights = Float64[5, 4, 3, 2, 1]
        an = ArrayNode(randn(Float32, 5, 1))
        mk = create_structmask(an, d -> fill(true, d))
        f = () -> sum(weights[prunemask(mk)]) - 6

        # scores aligned with weights: heavier items score higher, so
        # HeuristicOrder's bisection should pick the same minimal set
        node = first(first.(collectmasks(mk)))
        scores = IdDict{StructMask,Vector{Float64}}(node => weights)
        strategy = PruningStrategy(HeuristicOrder(scores), false, true, true)
        prune!(mk, nothing, nothing, f, strategy)

        @test f() >= 0
        @test prunemask(mk) == Bool[1, 1, 0, 0, 0]
    end

    @testset "precondition: errors immediately if the full mask can't satisfy f" begin
        weights = Float64[5, 4, 3, 2, 1]
        an = ArrayNode(randn(Float32, 5, 1))
        mk = create_structmask(an, d -> fill(true, d))
        f_impossible = () -> sum(weights[prunemask(mk)]) - 100  # max possible sum is 15

        strategy = PruningStrategy(GreedyForward(), false, false, false)
        @test_throws ErrorException prune!(mk, nothing, nothing, f_impossible, strategy)
    end

    @testset "level-by-level: baseline convention and participation threading" begin
        # 2-level tree (bag instances, then each instance's ngram content).
        # Objective only cares about the *total* number of on-items across
        # both levels, so this exercises the loop mechanics (all-true
        # baseline, per-level reset, participation gating) independent of
        # any real model.
        sn = ArrayNode(Mill.NGramMatrix(["a", "b", "c", "b", "e"]))
        ds = BagNode(sn, AlignedBags([1:2, 3:3, 0:-1, 4:5]))
        mk = create_structmask(ds, d -> fill(true, d))

        total_on() = sum(count, prunemask.(first.(collectmasks(mk))))
        f = () -> (total_on() >= 3 ? 0.0 : -1.0)

        strategy = PruningStrategy(GreedyForward(), true, true, true)
        prune!(mk, ds, nothing, f, strategy)

        @test f() >= 0
        @test total_on() <= 5   # meaningfully pruned, not left at the all-true baseline (10 total)
    end

    @testset "level-by-level with HeuristicOrder also converges" begin
        sn = ArrayNode(Mill.NGramMatrix(["a", "b", "c", "b", "e"]))
        ds = BagNode(sn, AlignedBags([1:2, 3:3, 0:-1, 4:5]))
        mk = create_structmask(ds, d -> fill(true, d))

        total_on() = sum(count, prunemask.(first.(collectmasks(mk))))
        f = () -> (total_on() >= 3 ? 0.0 : -1.0)

        nodes = first.(collectmasks(mk))
        scores = IdDict{StructMask,Vector{Float64}}(n => collect(Float64, 1:length(n)) for n in nodes)
        strategy = PruningStrategy(HeuristicOrder(scores), true, true, true)
        prune!(mk, ds, nothing, f, strategy)

        @test f() >= 0
        @test total_on() <= 5
    end

    @testset "integration: every (order, granularity) combination against a real model" begin
        ds = specimen_sample()
        model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing=true)
        # use the model's actual predicted class (mirroring how the original
        # explain() derives `class`), so the confidence gap on the full
        # sample is guaranteed nonnegative regardless of this model's random
        # initialization
        o0 = vec(Flux.softmax(model(ds)))
        class = argmax(o0)
        cg = o0[class] - maximum(o0[1:end.!=class])
        threshold = 0.5 * cg

        for levelbylevel in (false, true)
            mk_h, acc_h = stats(ShapleyExplainer(60), ds, model, o -> Flux.softmax(o)[class])
            scores_h = nodescores(mk_h, acc_h, score)
            fh = () -> begin
                o = Flux.softmax(model(ds[mk_h]))
                o[class] - maximum(o[1:end.!=class]) - threshold
            end
            strategy_h = PruningStrategy(HeuristicOrder(scores_h), levelbylevel, true, true)
            prune!(mk_h, ds, model, fh, strategy_h)
            @test fh() >= 0

            mk_g = create_structmask(ds, d -> fill(true, d))
            fg = () -> begin
                o = Flux.softmax(model(ds[mk_g]))
                o[class] - maximum(o[1:end.!=class]) - threshold
            end
            strategy_g = PruningStrategy(GreedyForward(), levelbylevel, true, true)
            prune!(mk_g, ds, model, fg, strategy_g)
            @test fg() >= 0
        end
    end
end

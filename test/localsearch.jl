# Tests for the local-search primitives (doc/design/pruning.md §2.1, §3.8),
# ported onto FlatView. Most tests use small synthetic weighted-sum
# objectives over a plain ArrayNode's FlatView -- deliberately no model
# involved -- so the search mechanics can be checked exactly and
# deterministically; a final integration test exercises the full stack
# (a real Mill model, ShapleyExplainer scores, nodescores/heuristicscores)
# together.

@testset "local search primitives" begin
    @testset "addminimumbi!" begin
        # 5 weighted items, objective requires the on-items' weights to sum
        # to >= 6. Turning on the two heaviest (weights 5, 4) is minimal.
        weights = Float64[5, 4, 3, 2, 1]
        an = ArrayNode(randn(Float32, 5, 1))
        mk = create_structmask(an, d -> fill(true, d))
        fv = FlatView(mk)
        f = () -> sum(weights[prunemask(fv)]) - 6

        fill!(fv, false)
        @test addminimumbi!(f, fv, [1, 2, 3, 4, 5]) == true
        @test prunemask(fv) == Bool[1, 1, 0, 0, 0]
        @test f() >= 0

        # already-satisfied short-circuit: with a threshold of 0, the
        # empty (all-off) set already satisfies f() > 0, so nothing changes
        fill!(fv, false)
        f0 = () -> sum(weights[prunemask(fv)]) - (-1)  # always > 0 even at 0
        @test addminimumbi!(f0, fv, [1, 2, 3, 4, 5]) == false
        @test all(==(false), prunemask(fv))
    end

    @testset "addone! / removeone!" begin
        weights = Float64[5, 4, 3, 2, 1]
        an = ArrayNode(randn(Float32, 5, 1))
        mk = create_structmask(an, d -> fill(true, d))
        fv = FlatView(mk)
        f = () -> sum(weights[prunemask(fv)]) - 6

        fill!(fv, false)
        @test addone!(f, fv) == true
        @test prunemask(fv) == Bool[1, 0, 0, 0, 0]  # heaviest item first
        @test addone!(f, fv) == true
        @test prunemask(fv) == Bool[1, 1, 0, 0, 0]  # then second-heaviest
        @test f() >= 0

        @test removeone!(f, fv) == true
        # removing the lighter of the two on-items does less damage and
        # keeps f() nonnegative for longer, so it's preferred... but here
        # only one on-item can be removed while staying informative about
        # which was chosen: it should remove item 2 (weight 4) since that
        # leaves the higher remaining value versus removing item 1
        @test prunemask(fv) == Bool[1, 0, 0, 0, 0]
    end

    @testset "addone! returns false once no participating candidate remains" begin
        an = ArrayNode(randn(Float32, 3, 1))
        mk = create_structmask(an, d -> fill(true, d))
        fv = FlatView(mk)
        f = () -> -1.0  # never satisfiable; forces exhaustion of candidates

        fill!(fv, true)  # nothing left to add
        @test addone!(f, fv) == false
    end

    @testset "sfs! converges from all-off" begin
        weights = Float64[5, 4, 3, 2, 1]
        an = ArrayNode(randn(Float32, 5, 1))
        mk = create_structmask(an, d -> fill(true, d))
        fv = FlatView(mk)
        f = () -> sum(weights[prunemask(fv)]) - 6

        sfs!(f, fv)
        @test f() >= 0
        @test prunemask(fv) == Bool[1, 1, 0, 0, 0]
    end

    @testset "randomremoval! reduces to a minimal feasible set" begin
        # objective: at least one of items 3,4,5 must be on. Starting from
        # all-on, randomremoval! should end with exactly one item on, and
        # it must be one of {3,4,5} (the only single-item feasible states).
        an = ArrayNode(randn(Float32, 5, 1))
        mk = create_structmask(an, d -> fill(true, d))
        fv = FlatView(mk)
        f = () -> (any(prunemask(fv)[3:5]) ? 0.0 : -1.0)

        fill!(fv, true)
        randomremoval!(f, fv; rng=MersenneTwister(1))
        @test f() >= 0
        @test count(prunemask(fv)) == 1
        @test findfirst(prunemask(fv)) in (3, 4, 5)

        # the fix under test: randomization actually affects which item
        # survives (unlike the original's dead `x` parameter in
        # `removeexcess!`) -- running with several seeds should not always
        # land on the same survivor
        survivors = Set{Int}()
        for seed in 1:30
            fill!(fv, true)
            randomremoval!(f, fv; rng=MersenneTwister(seed))
            push!(survivors, findfirst(prunemask(fv)))
        end
        @test length(survivors) > 1
    end

    @testset "greedyremoval! deterministic least-damaging removal" begin
        # weights [5,4,3,2,1], threshold sum >= 6, starting all-on.
        # Worked out by hand in doc/design/pruning.md-adjacent design
        # discussion: greedy removal (lightest weight first) converges to
        # items {1,2} (weights 5,4, sum=9), since removing item 2 next
        # would drop the sum to 5 < 6.
        weights = Float64[5, 4, 3, 2, 1]
        an = ArrayNode(randn(Float32, 5, 1))
        mk = create_structmask(an, d -> fill(true, d))
        fv = FlatView(mk)
        f = () -> sum(weights[prunemask(fv)]) - 6

        fill!(fv, true)
        greedyremoval!(f, fv)
        @test prunemask(fv) == Bool[1, 1, 0, 0, 0]
        @test f() >= 0
    end

    @testset "finetune! / settobest!" begin
        weights = Float64[5, 4, 3, 2, 1]
        an = ArrayNode(randn(Float32, 5, 1))
        mk = create_structmask(an, d -> fill(true, d))
        fv = FlatView(mk)
        f = () -> sum(weights[prunemask(fv)]) - 6

        # start from a feasible but non-minimal state
        fv[[1, 2, 3, 4]] = true
        fv[5] = false
        @test f() >= 0
        finetune!(f, fv)
        @test f() >= 0
        # finetune! should not do worse than the 2-item optimum found above
        @test count(prunemask(fv)) <= 2
    end

    @testset "settobest! direct unit test" begin
        an = ArrayNode(randn(Float32, 4, 1))
        mk = create_structmask(an, d -> fill(true, d))
        fv = FlatView(mk)

        visited = Dict{Vector{Int},Float64}(
            [1, 2, 3] => 5.0,   # feasible, length 3
            [1, 2] => 1.0,      # feasible, length 2 (smallest), lowest score
            [1] => -1.0,        # infeasible -- must be excluded
            [3, 4] => 2.0,      # feasible, length 2, higher score than [1,2]
        )
        # settobest! ties among equal-length feasible sets by lowest score
        # (ported unchanged from the original; the tie-break criterion
        # itself has no strong rationale, see doc/design/pruning.md, but
        # behavior must match what's implemented)
        settobest!(fv, visited)
        @test prunemask(fv) == Bool[1, 1, 0, 0]

        # no feasible state at all: leaves fv untouched
        fv2 = FlatView(create_structmask(ArrayNode(randn(Float32, 4, 1)), d -> fill(true, d)))
        fill!(fv2, true)
        settobest!(fv2, Dict{Vector{Int},Float64}([1, 2] => -1.0))
        @test all(prunemask(fv2))  # untouched, still all-on
    end

    @testset "addone!'s default candidates are unrestricted (caller decides)" begin
        # since the participate-filter moved from an internal default to an
        # explicit `candidates` argument (doc/design/pruning.md §4, Part 4),
        # addone! with no candidates given must be willing to turn on a
        # non-participating item -- restricting to participating items is
        # now the driver's job, not addone!'s.
        sn = ArrayNode(Mill.NGramMatrix(["a", "b", "c", "b", "e"]))
        ds = BagNode(sn, AlignedBags([1:2, 3:3, 0:-1, 4:5]))
        mk = create_structmask(ds, d -> fill(true, d))

        updateparticipation!(ds, mk)
        mk.own[1] = false
        updateparticipation!(ds, mk)

        childfv = FlatView([mk.children])
        f = () -> sum(prunemask(childfv)) - 1
        fill!(childfv, false)
        addone!(f, childfv)   # no candidates passed -> unrestricted
        @test count(prunemask(childfv)) == 1
        # with an objective indifferent to *which* item is on, the specific
        # choice isn't asserted -- only that no restriction was applied
    end

    @testset "participation gates addone!/removeone! candidates when the driver supplies them" begin
        sn = ArrayNode(Mill.NGramMatrix(["a", "b", "c", "b", "e"]))
        ds = BagNode(sn, AlignedBags([1:2, 3:3, 0:-1, 4:5]))
        mk = create_structmask(ds, d -> fill(true, d))

        updateparticipation!(ds, mk)
        mk.own[1] = false  # turn off bag instance 1
        updateparticipation!(ds, mk)

        childfv = FlatView([mk.children])
        @test participate(childfv) == Bool[0, 1, 1, 1, 1]

        f = () -> sum(prunemask(childfv)) - 1
        fill!(childfv, false)
        candidates = findall(participate(childfv))
        while addone!(f, childfv, candidates)
        end
        # item 1 (non-participating) must never be turned on, because the
        # driver restricted candidates to participate(fv) explicitly
        @test prunemask(childfv) == Bool[0, 1, 1, 1, 1]
    end

    @testset "integration: heuristic-ordered pruning against a real model" begin
        ds = specimen_sample()
        model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing=true)
        # use the model's actual predicted class so the confidence gap on
        # the full sample is guaranteed nonnegative regardless of this
        # model's random initialization (reflectinmodel is unseeded)
        class = argmax(vec(Flux.softmax(model(ds))))

        Random.seed!(3)
        mk, acc = stats(ShapleyExplainer(80), ds, model, o -> Flux.softmax(o)[class])
        scores = nodescores(mk, acc, score)

        cg = Flux.softmax(model(ds))[class] - maximum(Flux.softmax(model(ds))[1:end.!=class])
        threshold = 0.5 * cg
        f = () -> Flux.softmax(model(ds[mk]))[class] - maximum(vcat(
            Flux.softmax(model(ds[mk]))[1:class-1], Flux.softmax(model(ds[mk]))[class+1:end])) - threshold

        fv = FlatView(mk)
        fill!(fv, false)
        order = sortperm(heuristicscores(fv, scores), rev=true)
        addminimumbi!(f, fv, order)
        randomremoval!(f, fv; rng=MersenneTwister(1))
        finetune!(f, fv)

        @test f() >= 0
    end
end

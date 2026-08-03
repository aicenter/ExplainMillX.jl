# Tests for the top-level explain()/explainf() convenience entry points
# (src/explain.jl), built on the same real-model integration style used in
# test/prunestrategy.jl, plus direct unit tests of the small helpers and
# the documented error paths.

@testset "explain / explainf" begin
    @testset "_softmax / _confgap helpers" begin
        p = ExplainMillX._softmax([1.0, 1.0, 1.0])
        @test p ≈ [1 / 3, 1 / 3, 1 / 3]
        @test sum(p) ≈ 1

        p2 = ExplainMillX._softmax([10.0, 0.0])
        @test p2[1] > p2[2]
        @test ExplainMillX._confgap(p2, 1) ≈ p2[1] - p2[2]
        @test ExplainMillX._confgap(p2, 2) ≈ p2[2] - p2[1]
    end

    @testset "_threshold" begin
        @test ExplainMillX._threshold(1.0, nothing, 0.5) ≈ 0.5
        @test ExplainMillX._threshold(1.0, 0.3, nothing) ≈ 0.7
        @test_logs (:warn,) ExplainMillX._threshold(1.0, nothing, nothing)
        @test_throws ErrorException ExplainMillX._threshold(1.0, nothing, 1.5)   # rel_tol out of [0,1]
        @test_throws ErrorException ExplainMillX._threshold(1.0, 2.0, nothing)   # abs_tol > cg
    end

    @testset "ExplanationResult derived statistics" begin
        an = ArrayNode(randn(Float32, 4, 1))
        mk = create_structmask(an, d -> fill(true, d))
        r = ExplanationResult(mk, an, 1, 0.5, 0.4, 0.3, 10, 3)
        @test n_pruned(r) == 7
        @test fraction_kept(r) ≈ 0.3
        @test fraction_pruned(r) ≈ 0.7
        @test r.sample === an

        r0 = ExplanationResult(mk, an, 1, 0.5, 0.4, 0.3, 0, 0)
        @test fraction_kept(r0) == 1.0
        @test fraction_pruned(r0) == 0.0

        # Base.show doesn't error and mentions the class and percentages
        io = IOBuffer()
        show(io, MIME"text/plain"(), r)
        printed = String(take!(io))
        @test occursin("class 1", printed)
        @test occursin("3", printed)
    end

    @testset "explain: predicted-class default, multi-class model" begin
        ds = specimen_sample()
        model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing=true)
        predicted = argmax(vec(model(ds)))

        Random.seed!(1)
        result = explain(ds, model; scorer=ShapleyExplainer(80))

        @test result isa ExplanationResult
        @test result.class == predicted
        @test result.confidence_gap >= 0
        @test result.remaining_confidence_gap >= result.threshold - 1e-6
        @test 0 < result.n_kept <= result.n_total
        @test fraction_kept(result) == result.n_kept / result.n_total

        # the result carries exactly the original sample, not some internal
        # metadata-stripped working copy
        @test result.sample === ds

        # the returned mask genuinely applies to ds
        pruned = ds[result.mask]
        @test pruned isa Mill.AbstractMillNode
    end

    @testset "explain: default scorer (no scorer keyword given)" begin
        ds = specimen_sample()
        model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing=true)

        Random.seed!(9)
        result = explain(ds, model)   # relies on scorer=ShapleyExplainer(300) default

        @test result isa ExplanationResult
        @test result.remaining_confidence_gap >= result.threshold - 1e-6
    end

    @testset "explain: explicit class, rel_tol, and GreedyForward order" begin
        ds = specimen_sample()
        model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing=true)
        predicted = argmax(vec(model(ds)))

        Random.seed!(2)
        result = explain(ds, model, predicted;
            scorer=ShapleyExplainer(80), rel_tol=0.7, order=GreedyForward(), levelbylevel=false)

        @test result.class == predicted
        @test result.remaining_confidence_gap >= result.threshold - 1e-6
        @test isapprox(result.threshold, 0.7 * result.confidence_gap; atol=1e-6)
    end

    @testset "explain: abs_tol path" begin
        ds = specimen_sample()
        model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing=true)
        predicted = argmax(vec(model(ds)))

        Random.seed!(3)
        result = explain(ds, model, predicted; scorer=ShapleyExplainer(80), abs_tol=0.01)
        @test isapprox(result.threshold, result.confidence_gap - 0.01; atol=1e-6)
    end

    @testset "explain: no tolerance warns and defaults to rel_tol=0.9" begin
        ds = specimen_sample()
        model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing=true)
        predicted = argmax(vec(model(ds)))

        Random.seed!(4)
        result = @test_logs (:warn,) explain(ds, model, predicted; scorer=ShapleyExplainer(50))
        @test isapprox(result.threshold, 0.9 * result.confidence_gap; atol=1e-6)
    end

    @testset "explain: rejects a class with negative confidence gap" begin
        ds = specimen_sample()
        model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing=true)
        o = vec(model(ds))
        predicted = argmax(o)
        not_predicted = predicted == 1 ? 2 : 1

        @test_throws ErrorException explain(ds, model, not_predicted; scorer=ShapleyExplainer(50), rel_tol=0.5)
    end

    @testset "explain: rejects single-output (non-softmax) models" begin
        an = ArrayNode(randn(Float32, 4, 5))
        ds = an[1:1]
        model = vec ∘ Dense(4, 1) ∘ reflectinmodel(an, d -> Dense(d, 4), all_imputing=true)

        @test_throws ErrorException explain(ds, model; scorer=ShapleyExplainer(10))
    end

    @testset "explainf: low-level path with a hand-written binary objective" begin
        an = ArrayNode(randn(Float32, 4, 20))
        ds = an[1:1]
        model = vec ∘ Dense(4, 1) ∘ reflectinmodel(an, d -> Dense(d, 4), all_imputing=true)

        z0 = model(ds)[1]
        sgn = sign(z0)
        threshold = 0.5 * abs(z0)
        fₛ = o -> sgn * o[1]
        fₚ = o -> sgn * o[1] - threshold

        Random.seed!(5)
        result = explainf(ShapleyExplainer(80), ds, model, fₛ, fₚ)

        @test result isa NamedTuple
        @test fₚ(model(ds[result.mask])) >= 0
        @test 0 < result.n_kept <= result.n_total
    end

    @testset "explainf: metadata is stripped internally but the mask stays valid on the original" begin
        # a sample extracted with store_input=Val(true) carries metadata on
        # every leaf; explainf must still work correctly (Mill.dropmeta is
        # a pure performance optimization, see its docstring), and the
        # returned mask, applied back to the *original* metadata-carrying
        # sample, must produce the same pruning decisions it would on a
        # metadata-free copy.
        samples = [Dict("v" => Float64(i), "tag" => isodd(i) ? "a" : "b") for i in 1:20]
        sch = schema(samples)
        e = suggestextractor(sch; all_stable=true)
        ds = e(samples[1]; store_input=Val(true))
        @test ds[:v].metadata !== nothing   # sanity: metadata really is present

        model = reflectinmodel(sch, e; all_imputing=true)
        z0 = model(ds)[1]
        sgn = sign(z0)
        threshold = 0.5 * abs(z0)
        fₛ = o -> sgn * o[1]
        fₚ = o -> sgn * o[1] - threshold

        Random.seed!(8)
        result = explainf(ShapleyExplainer(60), ds, model, fₛ, fₚ)

        # the mask is valid against the original, metadata-carrying ds ...
        pruned = ds[result.mask]
        @test fₚ(model(pruned)) >= 0
        @test pruned[:v].metadata !== nothing   # metadata survived, untouched

        # ... and gives identical keep/drop decisions to what the same mask
        # produces against a metadata-free copy (dropmeta changes nothing
        # about the decision, only how fast it was to reach) -- compared
        # via model output rather than raw-node `==`, since Mill's `==` on
        # a fully-pruned (all-`missing`) `MaybeHotMatrix` hits an unrelated
        # 3-valued-logic quirk in Mill.jl itself when comparing two such
        # matrices structurally.
        ds_nometa = Mill.dropmeta(ds)
        @test model(ds[result.mask]) ≈ model(ds_nometa[result.mask])
    end

    @testset "explainf: order/granularity/post-pass keywords are honored" begin
        an = ArrayNode(randn(Float32, 4, 20))
        ds = an[1:1]
        model = vec ∘ Dense(4, 1) ∘ reflectinmodel(an, d -> Dense(d, 4), all_imputing=true)
        z0 = model(ds)[1]
        sgn = sign(z0)
        threshold = 0.5 * abs(z0)
        fₛ = o -> sgn * o[1]
        fₚ = o -> sgn * o[1] - threshold

        for levelbylevel in (false, true), order in (nothing, GreedyForward())
            Random.seed!(6)
            result = explainf(ShapleyExplainer(60), ds, model, fₛ, fₚ;
                order, levelbylevel, random_removal=true, finetune=true)
            @test fₚ(model(ds[result.mask])) >= 0
        end
    end
end

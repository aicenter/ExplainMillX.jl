# Tests for explain_json (doc/design/jsonoutput.md), covering each
# extractor-type dispatch case, the cleanup pass, the metadata precondition,
# and the explicitly-unsupported extractor paths, plus a full
# explain()+explain_json() integration test.

function jsongrinder_fixture()
    # constructed so schema()/suggestextractor(; all_stable=true) picks a
    # different extractor for each field:
    #   score -> ScalarExtractor  (150 unique numeric values)
    #   label -> CategoricalExtractor (3 unique values)
    #   note  -> NGramExtractor  (150 unique strings, over categorical_limit)
    #   items -> ArrayExtractor(CategoricalExtractor)  (bag of few-unique strings)
    samples = [
        Dict(
            "score" => Float64(i) * 1.371,
            "label" => ["red", "green", "blue"][mod1(i, 3)],
            "note" => "note-$i",
            "items" => ["x$j" for j in 1:mod1(i, 4)],
        ) for i in 1:150
    ]
    sch = schema(samples)
    e = suggestextractor(sch; all_stable=true)
    (samples=samples, sch=sch, e=e)
end

@testset "explain_json" begin
    @testset "full mask (nothing pruned) round-trips every field" begin
        (; samples, e) = jsongrinder_fixture()
        ds = e(samples[1]; store_input=Val(true))
        mk = create_structmask(ds, d -> fill(true, d))

        out = explain_json(ds, mk, e)
        @test out isa Dict
        @test out[:score] ≈ samples[1]["score"]
        @test out[:label] == samples[1]["label"]
        @test out[:note] == samples[1]["note"]
        @test out[:items] == samples[1]["items"]
    end

    @testset "everything pruned collapses to nothing" begin
        (; samples, e) = jsongrinder_fixture()
        ds = e(samples[1]; store_input=Val(true))
        mk = create_structmask(ds, d -> fill(false, d))

        @test explain_json(ds, mk, e) === nothing
    end

    @testset "ScalarExtractor: single shared bit across the field" begin
        (; samples, e) = jsongrinder_fixture()
        ds = e(samples[5]; store_input=Val(true))
        mk = create_structmask(ds, d -> fill(true, d))

        mk.children[:score].own .= false
        out = explain_json(ds, mk, e)
        @test !haskey(out, :score)   # dropped, not present as `nothing`

        mk.children[:score].own .= true
        out2 = explain_json(ds, mk, e)
        @test out2[:score] ≈ samples[5]["score"]
    end

    @testset "CategoricalExtractor: dropping the field removes the key" begin
        (; samples, e) = jsongrinder_fixture()
        ds = e(samples[2]; store_input=Val(true))
        mk = create_structmask(ds, d -> fill(true, d))

        mk.children[:label].own .= false
        out = explain_json(ds, mk, e)
        @test !haskey(out, :label)
    end

    @testset "NGramExtractor: dropping the field removes the key" begin
        (; samples, e) = jsongrinder_fixture()
        ds = e(samples[3]; store_input=Val(true))
        mk = create_structmask(ds, d -> fill(true, d))

        mk.children[:note].own .= false
        out = explain_json(ds, mk, e)
        @test !haskey(out, :note)
    end

    @testset "ArrayExtractor/BagNode: bag-level pruning omits instances (no null-fill)" begin
        (; samples, e) = jsongrinder_fixture()
        idx = findfirst(s -> length(s["items"]) == 4, samples)
        ds = e(samples[idx]; store_input=Val(true))
        mk = create_structmask(ds, d -> fill(true, d))

        mk.children[:items].own .= Bool[1, 0, 1, 0]
        out = explain_json(ds, mk, e)
        @test out[:items] == [samples[idx]["items"][1], samples[idx]["items"][3]]
        @test length(out[:items]) == 2   # omitted, not null-padded to length 4
    end

    @testset "ArrayExtractor/BagNode: per-item field pruning within surviving instances" begin
        (; samples, e) = jsongrinder_fixture()
        idx = findfirst(s -> length(s["items"]) == 4, samples)
        ds = e(samples[idx]; store_input=Val(true))
        mk = create_structmask(ds, d -> fill(true, d))

        # bag-level: keep all 4 instances, but the per-item categorical
        # field itself drops item 2's value
        mk.children[:items].children.own .= Bool[1, 0, 1, 1]
        out = explain_json(ds, mk, e)
        @test out[:items] == [samples[idx]["items"][1], samples[idx]["items"][3], samples[idx]["items"][4]]
    end

    @testset "ArrayExtractor/BagNode: fully pruning all items drops the key" begin
        (; samples, e) = jsongrinder_fixture()
        idx = findfirst(s -> length(s["items"]) == 4, samples)
        ds = e(samples[idx]; store_input=Val(true))
        mk = create_structmask(ds, d -> fill(true, d))

        mk.children[:items].own .= Bool[0, 0, 0, 0]
        out = explain_json(ds, mk, e)
        @test !haskey(out, :items)
    end

    @testset "DictExtractor: dropping some fields keeps the rest" begin
        (; samples, e) = jsongrinder_fixture()
        ds = e(samples[7]; store_input=Val(true))
        mk = create_structmask(ds, d -> fill(true, d))

        mk.children[:label].own .= false
        mk.children[:note].own .= false
        out = explain_json(ds, mk, e)
        @test !haskey(out, :label)
        @test !haskey(out, :note)
        @test out[:score] ≈ samples[7]["score"]
        @test out[:items] == samples[7]["items"]
    end

    @testset "requires metadata: clear error, not a garbage result" begin
        (; samples, e) = jsongrinder_fixture()
        ds = e(samples[1])   # no store_input=Val(true)
        mk = create_structmask(ds, d -> fill(true, d))

        @test_throws ErrorException explain_json(ds, mk, e)
    end

    @testset "PolymorphExtractor is explicitly unsupported" begin
        an = ArrayNode(randn(Float32, 4, 1))
        mk = create_structmask(an, d -> fill(true, d))
        poly = (JsonGrinder.ScalarExtractor(), JsonGrinder.ScalarExtractor()) |> JsonGrinder.PolymorphExtractor

        @test_throws ErrorException explain_json(an, mk, poly)
    end

    @testset "an arbitrary unsupported extractor errors clearly" begin
        an = ArrayNode(randn(Float32, 4, 1))
        mk = create_structmask(an, d -> fill(true, d))

        @test_throws ErrorException explain_json(an, mk, "not an extractor")
    end

    @testset "root sample must be a single observation" begin
        (; samples, e) = jsongrinder_fixture()
        ds = reduce(catobs, [e(s; store_input=Val(true)) for s in samples[1:2]])
        mk = create_structmask(ds, d -> fill(true, d))

        @test_throws ArgumentError explain_json(ds, mk, e)
    end

    @testset "integration: explain() then explain_json() on a trained-ish model" begin
        (; samples, sch, e) = jsongrinder_fixture()
        encoder = reflectinmodel(sch, e; all_imputing=true)
        d = size(encoder(e(samples[1]; store_input=Val(true))), 1)
        model = vec ∘ Dense(d, 3) ∘ encoder

        ds = e(samples[1]; store_input=Val(true))

        Random.seed!(7)
        result = explain(ds, model; scorer=ShapleyExplainer(60), rel_tol=0.9)
        out = explain_json(ds, result.mask, e)

        # Fixed number of assertions regardless of what pruning happened to
        # decide (the model is randomly initialized and pruning is
        # stochastic, so which fields survive varies run to run -- each
        # check below always runs, and passes vacuously if that field
        # didn't survive, rather than being conditionally skipped, so the
        # test suite's assertion count stays deterministic).
        @test out === nothing || !haskey(out, :score) || out[:score] ≈ samples[1]["score"]
        @test out === nothing || !haskey(out, :label) || out[:label] == samples[1]["label"]
        @test out === nothing || !haskey(out, :note) || out[:note] == samples[1]["note"]
        @test out === nothing || !haskey(out, :items) || out[:items] ⊆ samples[1]["items"]
    end
end

# Tests for FlatView per doc/design/flatview.md: aliasing (§1.2), the basic
# API (§2), the identity-keyed heuristic-score lookup / "Option B" (§3), and
# the level-by-level construction + participation interplay (§4, §5).

using ExplainMillX: FlatView, nodescores, heuristicscores

@testset "FlatView" begin
    @testset "construction and length" begin
        ds = specimen_sample()
        mk = create_structmask(ds, d -> fill(true, d))
        fv = FlatView(mk)

        nodes = first.(collectmasks(mk))
        @test length(fv) == sum(length, nodes)
    end

    @testset "aliasing: writes/reads go straight through to the real mask" begin
        an = ArrayNode(randn(Float32, 4, 5))
        mk = create_structmask(an, d -> fill(true, d))
        fv = FlatView(mk)

        # write through the view, read through the mask
        fv[1] = false
        @test mk.own[1] == false
        @test fv[1] == false

        # write through the mask, read through the view
        mk.own[2] = false
        @test fv[2] == false

        # no copy: same identity throughout
        @test fv.itemmap[1][1] === mk
    end

    @testset "vector indexing" begin
        an = ArrayNode(randn(Float32, 4, 5))
        mk = create_structmask(an, d -> fill(true, d))
        fv = FlatView(mk)

        fv[[1, 3]] = false
        @test mk.own == Bool[0, 1, 0, 1]
        @test fv[[1, 2, 3, 4]] == Bool[0, 1, 0, 1]
    end

    @testset "fill!" begin
        an = ArrayNode(randn(Float32, 4, 5))
        mk = create_structmask(an, d -> fill(true, d))
        fv = FlatView(mk)

        fill!(fv, false)
        @test all(==(false), mk.own)
        fill!(fv, true)
        @test all(mk.own)
    end

    @testset "copyto!" begin
        an = ArrayNode(randn(Float32, 4, 5))
        mk = create_structmask(an, d -> fill(true, d))
        fv = FlatView(mk)

        copyto!(fv, Bool[0, 1, 0, 1])
        @test mk.own == Bool[0, 1, 0, 1]

        @test_throws DimensionMismatch copyto!(fv, Bool[0, 1])
    end

    @testset "prunemask aggregation across a hierarchy" begin
        ds = specimen_sample()
        mk = create_structmask(ds, d -> fill(true, d))
        fv = FlatView(mk)

        nodes = first.(collectmasks(mk))
        expected = reduce(vcat, prunemask.(nodes))
        @test prunemask(fv) == expected

        # mutate through the flat view, re-check aggregation stays consistent
        fv[1] = false
        @test prunemask(fv)[1] == false
        expected2 = reduce(vcat, prunemask.(nodes))
        @test prunemask(fv) == expected2
    end

    @testset "prunemask thresholding for real-valued (differentiable) masks" begin
        an = ArrayNode(randn(Float32, 4, 5))
        mk = create_structmask(an, d -> Float32[0.9, 0.4, 0.5000001, 0.1])
        fv = FlatView(mk)
        @test prunemask(fv) == Bool[1, 0, 1, 0]
    end

    @testset "participate aggregation" begin
        ds = specimen_sample()
        mk = create_structmask(ds, d -> fill(true, d))
        updateparticipation!(ds, mk)
        fv = FlatView(mk)

        nodes = first.(collectmasks(mk))
        expected = reduce(vcat, participate.(nodes))
        @test participate(fv) == expected
        @test all(participate(fv))
    end

    @testset "useditems" begin
        an = ArrayNode(randn(Float32, 4, 5))
        mk = create_structmask(an, d -> fill(true, d))
        fv = FlatView(mk)

        @test useditems(fv) == [1, 2, 3, 4]
        fv[[2, 4]] = false
        @test useditems(fv) == [1, 3]
    end

    @testset "rejects router (non-own-bearing) nodes" begin
        ds = ProductNode(a=ArrayNode(randn(Float32, 4, 5)), b=ArrayNode(randn(Float32, 3, 5)))
        mk = create_structmask(ds, d -> fill(true, d))
        @test isrouter(mk)
        @test_throws ErrorException FlatView([mk])
    end

    @testset "level-by-level: FlatView over a node subset partitions the whole tree" begin
        ds = specimen_sample()
        mk = create_structmask(ds, d -> fill(true, d))

        pairs = collectmasks(mk)
        nodesinorder = first.(pairs)
        maxlevel = maximum(last, pairs)
        perlevel = [FlatView([n for (n, l) in pairs if l == lvl]) for lvl in 1:maxlevel]

        wholetree = FlatView(mk)
        @test sum(length, perlevel) == length(wholetree)

        # every distinct node object appears in exactly one level's view --
        # compared by objectid to sidestep StructMask's default (identity)
        # equality/hash entirely and make the check unambiguous
        perlevel_ids = [Set(objectid(node) for (node, _) in fv.itemmap) for fv in perlevel]
        @test length(union(perlevel_ids...)) == length(Set(objectid.(nodesinorder)))
        for i in 1:length(perlevel_ids), j in (i+1):length(perlevel_ids)
            @test isempty(intersect(perlevel_ids[i], perlevel_ids[j]))
        end

        # writing through a level-scoped view aliases the same node object
        # the whole tree uses -- not a copy restricted to that level
        first_level_fv = perlevel[1]
        node, li = first_level_fv.itemmap[1]
        first_level_fv[1] = false
        @test node.own[li] == false
    end

    @testset "level-by-level participation: turning off an instance restricts the next level" begin
        sn = ArrayNode(Mill.NGramMatrix(["a", "b", "c", "b", "e"]))
        ds = BagNode(sn, AlignedBags([1:2, 3:3, 0:-1, 4:5]))
        mk = create_structmask(ds, d -> fill(true, d))

        pairs = collectmasks(mk)
        level1 = FlatView([n for (n, l) in pairs if l == 1])
        level1[1] = false   # turn off bag instance 1

        updateparticipation!(ds, mk)
        level2 = FlatView([n for (n, l) in pairs if l == 2])
        @test participate(level2) == Bool[0, 1, 1, 1, 1]

        # a non-participating item's own value cannot affect the pruned
        # sample: applymask already excludes instance 1's data via the
        # bag-level mask before the child mask at that position is ever
        # consulted, so both settings must produce an identical result
        level2[1] = true
        r_true = ds[mk]
        level2[1] = false
        r_false = ds[mk]
        @test r_true == r_false
    end

    @testset "shared node: FlatView built from collectmasks visits it once" begin
        ds = specimen_sample()
        shared_leaf = leafmask(fill(true, 5))
        an_mk = create_structmask(ds.data.data[:an], d -> fill(true, d))
        cn_mk = create_structmask(ds.data.data[:cn], d -> fill(true, d))
        prod_mk = routermask((an=an_mk, on=shared_leaf, cn=cn_mk, sn=shared_leaf))
        inner_bag_mk = hybridmask(fill(true, numobs(ds.data.data)), prod_mk)
        mk = hybridmask(fill(true, numobs(ds.data)), inner_bag_mk)

        fv = FlatView(mk)
        # shared_leaf (length 5) contributes only once, not twice
        nodes = first.(collectmasks(mk))
        @test count(n -> n === shared_leaf, nodes) == 1
        @test length(fv) == sum(length, nodes)

        # writing through the flat view affects both fields that share the node
        idx = findfirst(item -> item[1] === shared_leaf, fv.itemmap)
        fv[idx] = false
        @test shared_leaf.own[1] == false
    end

    @testset "nodescores / heuristicscores: identity-keyed lookup (Option B)" begin
        ds = specimen_sample()
        model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing=true)

        Random.seed!(7)
        mk, acc = stats(ShapleyExplainer(50), ds, model, o -> Flux.softmax(o)[1])
        scores = nodescores(mk, acc, score)

        fv = FlatView(mk)
        flat_scores = heuristicscores(fv, scores)

        expected = reduce(vcat, leafscores(mk, acc))
        @test flat_scores ≈ expected

        # a level-scoped FlatView reuses the same whole-tree lookup unchanged
        # -- checked directly against `scores` (identity lookup), not against
        # a second independently-ordered traversal
        pairs = collectmasks(mk)
        level1nodes = [n for (n, l) in pairs if l == 1]
        level1fv = FlatView(level1nodes)
        level1expected = reduce(vcat, [scores[n] for n in level1nodes])
        @test heuristicscores(level1fv, scores) ≈ level1expected
    end

    @testset "nodescores / heuristicscores: mismatched mask raises, doesn't silently misattribute" begin
        ds = specimen_sample()
        model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing=true)

        mk_a, acc_a = stats(ShapleyExplainer(20), ds, model, o -> Flux.softmax(o)[1])
        mk_b, acc_b = stats(ShapleyExplainer(20), ds, model, o -> Flux.softmax(o)[1])

        scores_a = nodescores(mk_a, acc_a, score)
        fv_b = FlatView(mk_b)   # structurally identical tree, but different objects

        @test_throws KeyError heuristicscores(fv_b, scores_a)
    end
end

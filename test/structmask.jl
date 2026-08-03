# Adapted from ExplainMill.jl/test/structuralmasks.jl to the StructMask design
# in docs/design/masks.md. Kept in spirit, not verbatim -- scope intentionally
# narrower than the original for this step:
#   - hard (missing-based) pruning only; no differentiable/soft masking or
#     gradient checks (masks.md §6.2, deferred)
#   - no `partialeval` tests (a pruning-stage concern, see docs/design/pruning.md)
#   - no `ObservationMask`-equivalent alternate-axis tests (masks.md §7, open)
#   - no leader/follower `FollowingMasks` tests (not part of the finalized
#     StructMask design)
#   - "sharing" is adapted to two fields that share a masking axis natively
#     (numobs), rather than forcing all four fields onto one shared axis

@testset "Structural masks" begin
    @testset "Dense ArrayNode mask" begin
        an = ArrayNode(randn(Float32, 4, 5))
        mk = create_structmask(an, d -> fill(true, d))
        @test mk isa StructMask
        @test isleaf(mk)
        @test length(mk) == 4
        @test an[mk] == an

        mk.own[[1, 3]] .= false
        @test mk.own == Bool[0, 1, 0, 1]
        @test isequal(an[mk].data, an.data .* [missing, true, missing, true])

        cmk = collectmasks(mk)
        @test length(cmk) == 1
        @test cmk[1].first === mk
        @test cmk[1].second == 1

        cmk2 = mapmask((x, l) -> x .+ 1, mk)
        @test cmk2.own ≈ mk.own .+ 1

        # empty sample
        an0 = ArrayNode(randn(Float32, 4, 0))
        mk0 = create_structmask(an0, d -> fill(true, d))
        @test an0[mk0] == an0
    end

    @testset "Categorical ArrayNode mask (Mill.MaybeHotMatrix)" begin
        on = ArrayNode(Mill.maybehotbatch([1, 2, 3, 1, 2], 1:4))
        mk = create_structmask(on, d -> fill(true, d))
        @test isleaf(mk)
        @test length(mk) == 5
        @test on[mk] == on

        mk.own[[1, 3]] .= false
        @test mk.own == Bool[0, 1, 0, 1, 1]
        @test isequal(on[mk].data.I, [missing, 2, missing, 1, 2])

        cmk = collectmasks(mk)
        @test length(cmk) == 1

        cmk2 = mapmask((x, l) -> x, mk)
        @test cmk2.own == mk.own

        # participation
        mk3 = create_structmask(on, d -> fill(true, d))
        updateparticipation!(on, mk3)
        @test all(mk3.participate)

        # empty sample
        on0 = ArrayNode(Mill.maybehotbatch(Int[], 1:4))
        mk0 = create_structmask(on0, d -> fill(true, d))
        @test on0[mk0] == on0
    end

    @testset "Sparse ArrayNode mask" begin
        cn = ArrayNode(sparse(Float32[1 2 3 0 5; 0 2 0 4 0]))
        mk = create_structmask(cn, d -> fill(true, d))
        @test isleaf(mk)
        @test length(mk) == 6
        @test cn[mk] == cn

        mk.own[[1, 3]] .= false
        @test mk.own == Bool[0, 1, 0, 1, 1, 1]
        @test cn[mk].data.nzval == [0, 2, 0, 3, 4, 5]

        @test sparsecolumns(cn.data) == [1, 2, 2, 3, 4, 5]

        cmk = collectmasks(mk)
        @test length(cmk) == 1

        # empty sample
        cn0 = ArrayNode(sparse(zeros(Float32, 4, 0)))
        mk0 = create_structmask(cn0, d -> fill(true, d))
        @test cn0[mk0] == cn0
    end

    @testset "NGram (string) ArrayNode mask" begin
        sn = ArrayNode(Mill.NGramMatrix(string.(1:5), 3, 256, 2053))
        mk = create_structmask(sn, d -> fill(true, d))
        @test isleaf(mk)
        @test length(mk) == 5
        @test sn[mk] == sn

        mk.own[[1, 3]] .= false
        @test isequal(sn[mk].data.S, [missing, "2", missing, "4", "5"])

        cmk = collectmasks(mk)
        @test length(cmk) == 1

        # participation
        mk2 = create_structmask(sn, d -> fill(true, d))
        updateparticipation!(sn, mk2)
        @test all(mk2.participate)
    end

    @testset "BagNode mask -- single nesting" begin
        an = ArrayNode(randn(Float32, 4, 5))
        ds = BagNode(an, AlignedBags([1:2, 3:3, 0:-1, 4:5]))
        mk = create_structmask(ds, d -> fill(true, d))
        @test !isleaf(mk)
        @test length(mk) == 5
        @test ds[mk] == ds

        mk.own[[1, 3]] .= false
        @test mk.own == Bool[0, 1, 0, 1, 1]
        @test ds[mk].data == an[findall(Bool[0, 1, 0, 1, 1])]
        @test ds[mk].bags.bags == UnitRange{Int64}[1:1, 0:-1, 0:-1, 2:3]

        # if all instances are removed, only empty bags remain
        mk2 = create_structmask(ds, d -> fill(true, d))
        mk2.own .= false
        dss = ds[mk2]
        @test all(isempty, dss.bags.bags)
        @test numobs(dss.data) == 0

        # foreach_mask across levels
        cmk = collectmasks(mk)
        @test length(cmk) == 2
        @test cmk[1].second == 1
        @test cmk[2].second == 2

        # mapmask preserves nesting
        cmk2 = mapmask((x, l) -> l == 1 ? x .+ 1 : x, mk)
        @test cmk2.own ≈ mk.own .+ 1
        @test cmk2.children.own == mk.children.own

        # participation: default is fully participating
        mk3 = create_structmask(ds, d -> fill(true, d))
        updateparticipation!(ds, mk3)
        @test all(mk3.participate)
        @test all(mk3.children.participate)

        # participation propagates per-instance through the bag boundary
        sn = ArrayNode(Mill.NGramMatrix(["a", "b", "c", "b", "e"]))
        ds2 = BagNode(sn, AlignedBags([1:2, 3:3, 0:-1, 4:5]))
        mk4 = create_structmask(ds2, d -> fill(true, d))
        mk4.own[1] = false
        updateparticipation!(ds2, mk4)
        @test mk4.children.participate == Bool[0, 1, 1, 1, 1]

        # empty sample
        an0 = ArrayNode(randn(Float32, 4, 0))
        ds0 = BagNode(an0, AlignedBags([0:-1, 0:-1]))
        mk0 = create_structmask(ds0, d -> fill(true, d))
        @test ds0[mk0] == ds0

        # BagNode with missing data
        dsm = BagNode(missing, AlignedBags([0:-1, 0:-1]))
        mkm = create_structmask(dsm, d -> fill(true, d))
        @test isleaf(mkm)
        @test dsm[mkm] === dsm
    end

    @testset "ProductNode mask" begin
        ds = ProductNode(
            a=ArrayNode(randn(Float32, 4, 5)),
            b=ArrayNode(sparse(Float32[1 2 3 0 5; 0 2 0 4 0])),
        )
        mk = create_structmask(ds, d -> fill(true, d))
        @test isrouter(mk)
        @test ds[mk] == ds

        mk.children[:a].own[1:2] .= false
        mk.children[:b].own[1:2] .= false

        @test isequal(ds[mk][:a].data, ds[:a].data .* [missing, missing, true, true])
        @test ds[mk][:b].data ≈ sparse(Float32[0 0 3 0 5; 0 2 0 4 0])

        cmk = collectmasks(mk)
        @test length(cmk) == 2

        cmk2 = mapmask((x, l) -> x .+ 1, mk)
        @test cmk2.children[:a].own ≈ mk.children[:a].own .+ 1
        @test cmk2.children[:b].own ≈ mk.children[:b].own .+ 1

        # participation
        mk2 = create_structmask(ds, d -> fill(true, d))
        updateparticipation!(ds, mk2)
        @test all(mk2.children[:a].participate)
        @test all(mk2.children[:b].participate)

        # empty sample
        ds0 = ProductNode(
            a=ArrayNode(randn(Float32, 4, 0)),
            b=ArrayNode(sparse(zeros(Float32, 4, 0))),
        )
        mk0 = create_structmask(ds0, d -> fill(true, d))
        @test ds0[mk0] == ds0
    end

    @testset "sharing of masks" begin
        ds = specimen_sample()

        an_mk = create_structmask(ds.data.data[:an], d -> fill(true, d))
        cn_mk = create_structmask(ds.data.data[:cn], d -> fill(true, d))
        # `on` and `sn` both natively mask per-observation (numobs == 5), so
        # they can meaningfully share one leaf node -- an/cn cannot, since
        # their own axes are rows/nnz respectively (masks.md §7).
        shared_leaf = leafmask(fill(true, 5))
        prod_mk = routermask((an=an_mk, on=shared_leaf, cn=cn_mk, sn=shared_leaf))
        inner_bag_mk = hybridmask(fill(true, numobs(ds.data.data)), prod_mk)
        mk = hybridmask(fill(true, numobs(ds.data)), inner_bag_mk)

        @test ds[mk] == ds

        # a shared node is only visited once by foreach_mask/collectmasks
        cmk = collectmasks(mk)
        @test length(cmk) == 5

        # mutating the shared node affects both fields simultaneously
        shared_leaf.own[1] = false
        dss = ds[mk]
        @test ismissing(dss.data.data.data.on.data.I[1])
        @test ismissing(dss.data.data.data.sn.data.S[1])
        @test !ismissing(dss.data.data.data.an.data[1, 1])
    end

    @testset "integration test of nested samples" begin
        an = ArrayNode(Float32.(reshape(collect(1:10), 2, 5)))
        on = ArrayNode(Mill.maybehotbatch([1, 2, 3, 1, 2], 1:4))
        cn = ArrayNode(sparse(Float32[1 0 3 0 5; 0 2 0 4 0]))
        ds = BagNode(BagNode(ProductNode((a=an, c=cn, o=on)), AlignedBags([1:2, 3:3, 4:5])),
            AlignedBags([1:3]))

        mk = create_structmask(ds, d -> fill(true, d))
        mk.children.children.children[:a].own[2] = false
        mk.children.children.children[:c].own .= Bool[1, 1, 1, 0, 1]
        mk.children.children.children[:o].own .= Bool[1, 1, 1, 0, 0]
        mk.children.own .= Bool[1, 0, 1, 0, 1]
        dss = ds[mk]

        @test numobs(dss) == 1
        @test numobs(dss.data) == 3
        @test numobs(dss.data.data) == 3
        @test isequal(dss.data.data.data.a.data, [1 5 9; missing missing missing])
        @test dss.data.data.data.c.data.nzval ≈ [1, 3, 5]
        @test isequal(dss.data.data.data.o.data.I, [1, 3, missing])
    end
end

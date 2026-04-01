using RecursiveArrayTools, RecursiveArrayToolsRaggedArrays
using RecursiveArrayToolsRaggedArrays: RaggedEnd, RaggedRange
using SymbolicIndexingInterface
using SymbolicIndexingInterface: SymbolCache
using Test

@testset "RecursiveArrayToolsRaggedArrays" begin
    # ===================================================================
    # Tests ported from v3 basic_indexing.jl
    # ===================================================================

    @testset "v3 basic indexing (ported)" begin
        # Example Problem
        recs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        testa = cat(recs..., dims = 2)
        testva = RaggedVectorOfArray(recs)
        @test maximum(testva) == maximum(maximum.(recs))

        testva = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        @test maximum(testva) == maximum(maximum.(recs))

        # broadcast with array
        X = rand(3, 3)
        mulX = sqrt.(abs.(testva .* X))
        ref = mapreduce((x, y) -> sqrt.(abs.(x .* y)), hcat, testva, eachcol(X))
        @test mulX == ref
        fill!(mulX, 0)
        mulX .= sqrt.(abs.(testva .* X))
        @test mulX == ref

        @test Array(testva) == [
            1 4 7
            2 5 8
            3 6 9
        ]

        @test testa[1:2, 1:2] == [1 4; 2 5]
        @test testva[1:2, 1:2] == [1 4; 2 5]
        @test testa[1:2, 1:2] == [1 4; 2 5]

        t = [1, 2, 3]
        diffeq = RaggedDiffEqArray(recs, t)
        @test Array(diffeq) == [
            1 4 7
            2 5 8
            3 6 9
        ]
        @test diffeq[1:2, 1:2] == [1 4; 2 5]
    end

    @testset "v3 ndims==2 indexing (ported)" begin
        t = 1:10
        recs = [rand(8) for i in 1:10]
        testa = cat(recs..., dims = 2)
        testva = RaggedVectorOfArray(recs)

        # Array functions
        @test size(testva) == (8, 10)
        @test ndims(testva) == 2
        @test eltype(testva) == eltype(eltype(recs))
        testvasim = similar(testva)
        @test size(testvasim) == size(testva)
        @test eltype(testvasim) == eltype(testva)
        testvasim = similar(testva, Float32)
        @test size(testvasim) == size(testva)
        @test eltype(testvasim) == Float32
        testvb = deepcopy(testva)
        @test testva == testvb == recs

        # Math operations
        @test testva + testvb == testva + recs == 2testva == 2 .* recs
        @test testva - testvb == testva - recs == 0 .* recs
        @test testva / 2 == recs ./ 2
        @test 2 .\ testva == 2 .\ recs

        # Column indexing
        @test testva[:, begin] == first(testva)
        @test testva[:, end] == last(testva)
        @test testa[:, 1] == recs[1]
        @test testva.u == recs
        @test testva[:, 2:end] == RaggedVectorOfArray([recs[i] for i in 2:length(recs)])

        diffeq = RaggedDiffEqArray(recs, t)
        @test diffeq[:, 1] == testa[:, 1]
        @test diffeq.u == recs
        @test diffeq[:, end] == testa[:, end]
        @test diffeq[:, 2:end] == RaggedDiffEqArray([recs[i] for i in 2:length(recs)], t[2:end])
        @test diffeq[:, 2:end].t == t[2:end]
        @test diffeq[:, (end - 1):end] == RaggedDiffEqArray([recs[i] for i in (length(recs) - 1):length(recs)], t[(length(t) - 1):length(t)])
        @test diffeq[:, (end - 1):end].t == t[(length(t) - 1):length(t)]
        @test diffeq[:, (end - 5):8] == RaggedDiffEqArray([recs[i] for i in (length(t) - 5):8], t[(length(t) - 5):8])
        @test diffeq[:, (end - 5):8].t == t[(length(t) - 5):8]

        # (Int, Int)
        @test testa[5, 4] == testva[5, 4]
        @test testa[5, 4] == diffeq[5, 4]

        # (Int, Range) or (Range, Int)
        @test testa[1, 2:3] == testva[1, 2:3]
        @test testa[5:end, 1] == testva[5:end, 1]
        @test testa[:, 1] == testva[:, 1]
        @test testa[3, :] == testva[3, :]

        @test testa[1, 2:3] == diffeq[1, 2:3]
        @test testa[5:end, 1] == diffeq[5:end, 1]
        @test testa[:, 1] == diffeq[:, 1]
        @test testa[3, :] == diffeq[3, :]

        # (Range, Range)
        @test testa[5:end, 1:2] == testva[5:end, 1:2]
        @test testa[5:end, 1:2] == diffeq[5:end, 1:2]
    end

    @testset "v3 ndims==3 indexing (ported)" begin
        t = 1:15
        recs = [rand(10, 8) for i in 1:15]
        testa = cat(recs..., dims = 3)
        testva = RaggedVectorOfArray(recs)
        diffeq = RaggedDiffEqArray(recs, t)

        # (Int, Int, Int)
        @test testa[1, 7, 14] == testva[1, 7, 14]
        @test testa[1, 7, 14] == diffeq[1, 7, 14]

        # (Int, Int, Range)
        @test testa[2, 3, 1:2] == testva[2, 3, 1:2]
        @test testa[2, 3, 1:2] == diffeq[2, 3, 1:2]

        # (Int, Range, Int)
        @test testa[2, 3:4, 1] == testva[2, 3:4, 1]
        @test testa[2, 3:4, 1] == diffeq[2, 3:4, 1]

        # (Int, Range, Range)
        @test testa[2, 3:4, 1:2] == testva[2, 3:4, 1:2]
        @test testa[2, 3:4, 1:2] == diffeq[2, 3:4, 1:2]

        # (Range, Int, Range)
        @test testa[2:3, 1, 1:2] == testva[2:3, 1, 1:2]
        @test testa[2:3, 1, 1:2] == diffeq[2:3, 1, 1:2]

        # (Range, Range, Int)
        @test testa[1:2, 2:3, 1] == testva[1:2, 2:3, 1]
        @test testa[1:2, 2:3, 1] == diffeq[1:2, 2:3, 1]

        # (Range, Range, Range)
        @test testa[2:3, 2:3, 1:2] == testva[2:3, 2:3, 1:2]
        @test testa[2:3, 2:3, 1:2] == diffeq[2:3, 2:3, 1:2]

        # Make sure that 1:1 like ranges are not collapsed
        @test testa[1:1, 2:3, 1:2] == testva[1:1, 2:3, 1:2]
        @test testa[1:1, 2:3, 1:2] == diffeq[1:1, 2:3, 1:2]
    end

    @testset "v3 ragged arrays (ported)" begin
        t = 1:3
        recs = [[1, 2, 3], [3, 5, 6, 7], [8, 9, 10, 11]]
        testva = RaggedVectorOfArray(recs)
        diffeq = RaggedDiffEqArray(recs, t)

        @test testva[:, 1] == recs[1]
        @test testva[1:2, 1:2] == [1 3; 2 5]
        @test diffeq[:, 1] == recs[1]
        @test diffeq[1:2, 1:2] == [1 3; 2 5]
        @test diffeq[:, 1:2] == RaggedDiffEqArray([recs[i] for i in 1:2], t[1:2])
        @test diffeq[:, 1:2].t == t[1:2]
        @test diffeq[:, 2:end] == RaggedDiffEqArray([recs[i] for i in 2:3], t[2:end])
        @test diffeq[:, 2:end].t == t[2:end]
        @test diffeq[:, (end - 1):end] == RaggedDiffEqArray([recs[i] for i in (length(recs) - 1):length(recs)], t[(length(t) - 1):length(t)])
        @test diffeq[:, (end - 1):end].t == t[(length(t) - 1):length(t)]
    end

    @testset "v3 heterogeneous views (ported, issue #453)" begin
        f = RaggedVectorOfArray([[1.0], [2.0, 3.0]])
        @test length(view(f, :, 1)) == 1
        @test length(view(f, :, 2)) == 2
        @test view(f, :, 1) == [1.0]
        @test view(f, :, 2) == [2.0, 3.0]
        @test collect(view(f, :, 1)) == f[:, 1]
        @test collect(view(f, :, 2)) == f[:, 2]

        f2 = RaggedVectorOfArray([[1.0, 2.0], [3.0]])
        @test length(view(f2, :, 1)) == 2
        @test length(view(f2, :, 2)) == 1
        @test view(f2, :, 1) == [1.0, 2.0]
        @test view(f2, :, 2) == [3.0]
        @test collect(view(f2, :, 1)) == f2[:, 1]
        @test collect(view(f2, :, 2)) == f2[:, 2]
    end

    @testset "v3 end indexing with ragged arrays (ported)" begin
        ragged = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]])
        @test ragged[end, 1] == 2.0
        @test ragged[end, 2] == 5.0
        @test ragged[end, 3] == 9.0
        @test ragged[end - 1, 1] == 1.0
        @test ragged[end - 1, 2] == 4.0
        @test ragged[end - 1, 3] == 8.0
        @test ragged[1:end, 1] == [1.0, 2.0]
        @test ragged[1:end, 2] == [3.0, 4.0, 5.0]
        @test ragged[1:end, 3] == [6.0, 7.0, 8.0, 9.0]
        @test ragged[:, end] == [6.0, 7.0, 8.0, 9.0]
        @test ragged[:, 2:end] == RaggedVectorOfArray(ragged.u[2:end])
        @test ragged[:, (end - 1):end] == RaggedVectorOfArray(ragged.u[(end - 1):end])

        ragged2 = RaggedVectorOfArray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0], [7.0, 8.0, 9.0]])
        @test ragged2[end, 1] == 4.0
        @test ragged2[end, 2] == 6.0
        @test ragged2[end, 3] == 9.0
        @test ragged2[end - 1, 1] == 3.0
        @test ragged2[end - 1, 2] == 5.0
        @test ragged2[end - 1, 3] == 8.0
        @test ragged2[end - 2, 1] == 2.0
        @test ragged2[1:end, 1] == [1.0, 2.0, 3.0, 4.0]
        @test ragged2[1:end, 2] == [5.0, 6.0]
        @test ragged2[1:end, 3] == [7.0, 8.0, 9.0]
        @test ragged2[2:end, 1] == [2.0, 3.0, 4.0]
        @test ragged2[2:end, 2] == [6.0]
        @test ragged2[2:end, 3] == [8.0, 9.0]
        @test ragged2[:, end] == [7.0, 8.0, 9.0]
        @test ragged2[:, 2:end] == RaggedVectorOfArray(ragged2.u[2:end])
        @test ragged2[1:(end - 1), 1] == [1.0, 2.0, 3.0]
        @test ragged2[1:(end - 1), 2] == [5.0]
        @test ragged2[1:(end - 1), 3] == [7.0, 8.0]
        @test ragged2[:, (end - 1):end] == RaggedVectorOfArray(ragged2.u[(end - 1):end])
    end

    @testset "v3 RaggedEnd/RaggedRange broadcast as scalars (ported)" begin
        ragged = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]])
        ragged_idx = lastindex(ragged, 1)
        @test ragged_idx isa RaggedEnd
        @test Base.broadcastable(ragged_idx) isa Ref
        # Broadcasting with RaggedEnd should not error
        @test identity.(ragged_idx) === ragged_idx

        ragged_range_idx = 1:lastindex(ragged, 1)
        @test ragged_range_idx isa RaggedRange
        @test Base.broadcastable(ragged_range_idx) isa Ref
        # Broadcasting with RaggedRange should not error
        @test identity.(ragged_range_idx) === ragged_range_idx
    end

    @testset "v3 broadcast assignment heterogeneous (ported, issue #454)" begin
        u = RaggedVectorOfArray([[1.0], [2.0, 3.0]])
        @test length(view(u, :, 1)) == 1
        @test length(view(u, :, 2)) == 2
        # broadcast assignment into selected column (last index Int)
        u[:, 2] .= [10.0, 11.0]
        @test u.u[2] == [10.0, 11.0]
    end

    @testset "v3 DiffEqArray with 2D inner arrays (ported)" begin
        t = 1:2
        recs_2d = [rand(2, 3), rand(2, 4)]
        diffeq_2d = RaggedDiffEqArray(recs_2d, t)
        @test diffeq_2d[:, 1] == recs_2d[1]
        @test diffeq_2d[:, 2] == recs_2d[2]
        @test diffeq_2d[:, 1:2] == RaggedDiffEqArray(recs_2d[1:2], t[1:2])
        @test diffeq_2d[:, 1:2].t == t[1:2]
        @test diffeq_2d[:, 2:end] == RaggedDiffEqArray(recs_2d[2:end], t[2:end])
        @test diffeq_2d[:, 2:end].t == t[2:end]
        @test diffeq_2d[:, (end - 1):end] == RaggedDiffEqArray(recs_2d[(end - 1):end], t[(end - 1):end])
        @test diffeq_2d[:, (end - 1):end].t == t[(end - 1):end]
    end

    @testset "v3 DiffEqArray with 3D inner arrays (ported)" begin
        t = 1:2
        recs_3d = [rand(2, 3, 4), rand(2, 3, 5)]
        diffeq_3d = RaggedDiffEqArray(recs_3d, t)
        @test diffeq_3d[:, :, :, 1] == recs_3d[1]
        @test diffeq_3d[:, :, :, 2] == recs_3d[2]
        @test diffeq_3d[:, :, :, 1:2] == RaggedDiffEqArray(recs_3d[1:2], t[1:2])
        @test diffeq_3d[:, :, :, 1:2].t == t[1:2]
        @test diffeq_3d[:, :, :, 2:end] == RaggedDiffEqArray(recs_3d[2:end], t[2:end])
        @test diffeq_3d[:, :, :, 2:end].t == t[2:end]
        @test diffeq_3d[:, :, :, (end - 1):end] == RaggedDiffEqArray(recs_3d[(end - 1):end], t[(end - 1):end])
        @test diffeq_3d[:, :, :, (end - 1):end].t == t[(end - 1):end]
    end

    @testset "v3 2D inner arrays ragged second dimension (ported)" begin
        u = RaggedVectorOfArray([zeros(1, n) for n in (2, 3)])
        @test length(view(u, 1, :, 1)) == 2
        @test length(view(u, 1, :, 2)) == 3
        u[1, :, 2] .= [1.0, 2.0, 3.0]
        @test u.u[2] == [1.0 2.0 3.0]
        # partial column selection by indices
        u[1, [1, 3], 2] .= [7.0, 9.0]
        @test u.u[2] == [7.0 2.0 9.0]
        # test scalar indexing with end
        @test u[1, 1, end] == u.u[end][1, 1]
        @test u[1, end, end] == u.u[end][1, end]
        @test u[1, 2:end, end] == vec(u.u[end][1, 2:end])
    end

    @testset "v3 3D inner arrays ragged third dimension (ported)" begin
        u = RaggedVectorOfArray([zeros(2, 1, n) for n in (2, 3)])
        @test size(view(u, :, :, :, 1)) == (2, 1, 2)
        @test size(view(u, :, :, :, 2)) == (2, 1, 3)
        # assign into a slice of the second inner array using last index Int
        u[2, 1, :, 2] .= [7.0, 8.0, 9.0]
        @test vec(u.u[2][2, 1, :]) == [7.0, 8.0, 9.0]
        # check mixed slicing with range on front dims
        u[1:2, 1, [1, 3], 2] .= [1.0 3.0; 2.0 4.0]
        @test u.u[2][1, 1, 1] == 1.0
        @test u.u[2][2, 1, 1] == 2.0
        @test u.u[2][1, 1, 3] == 3.0
        @test u.u[2][2, 1, 3] == 4.0
        @test u[:, :, end] == u.u[end]
        @test u[:, :, 2:end] == RaggedVectorOfArray(u.u[2:end])
        @test u[1, 1, end] == u.u[end][1, 1, end]
        @test u[end, 1, end] == u.u[end][end, 1, end]
    end

    @testset "v3 views can be modified (ported)" begin
        f3 = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0]])
        v = view(f3, :, 2)
        @test length(v) == 3
        v[1] = 10.0
        @test f3[1, 2] == 10.0
        @test f3.u[2][1] == 10.0
    end

    @testset "v3 2D inner array conversions (ported)" begin
        t = 1:5
        recs = [rand(2, 2) for i in 1:5]
        testva = RaggedVectorOfArray(recs)
        diffeq = RaggedDiffEqArray(recs, t)

        @test Array(testva) isa Array{Float64, 3}
        @test Array(diffeq) isa Array{Float64, 3}
    end

    @testset "v3 CartesianIndex (ported)" begin
        v = RaggedVectorOfArray([zeros(20), zeros(10, 10), zeros(3, 3, 3)])
        v[CartesianIndex((2, 3, 2, 3))] = 1
        @test v[CartesianIndex((2, 3, 2, 3))] == 1
        @test v.u[3][2, 3, 2] == 1

        v = RaggedDiffEqArray([zeros(20), zeros(10, 10), zeros(3, 3, 3)], 1:3)
        v[CartesianIndex((2, 3, 2, 3))] = 1
        @test v[CartesianIndex((2, 3, 2, 3))] == 1
        @test v.u[3][2, 3, 2] == 1
    end

    @testset "v3 heterogeneous broadcasting (ported)" begin
        v = RaggedVectorOfArray([rand(20), rand(10, 10), rand(3, 3, 3)])
        w = v .* v
        @test w isa RaggedVectorOfArray
        @test w[:, 1] isa Vector
        @test w[:, 1] == v[:, 1] .* v[:, 1]
        @test w[:, 2] == v[:, 2] .* v[:, 2]
        @test w[:, 3] == v[:, 3] .* v[:, 3]
        x = copy(v)
        x .= v .* v
        @test x.u == w.u
        w = v .+ 1
        @test w isa RaggedVectorOfArray
        @test w.u == map(x -> x .+ 1, v.u)

        v = RaggedDiffEqArray([rand(20), rand(10, 10), rand(3, 3, 3)], 1:3)
        w = v .* v
        @test_broken w isa RaggedDiffEqArray # FIXME
        @test w[:, 1] isa Vector
        @test w[:, 1] == v[:, 1] .* v[:, 1]
        @test w[:, 2] == v[:, 2] .* v[:, 2]
        @test w[:, 3] == v[:, 3] .* v[:, 3]
        x = copy(v)
        x .= v .* v
        @test x.u == w.u
        w = v .+ 1
        @test_broken w isa RaggedDiffEqArray # FIXME
        @test w.u == map(x -> x .+ 1, v.u)
    end

    @testset "v3 setindex! (ported)" begin
        testva = RaggedVectorOfArray([i * ones(3, 3) for i in 1:5])
        testva[:, 2] = 7ones(3, 3)
        @test testva[:, 2] == 7ones(3, 3)
        testva[:, :] = [2i * ones(3, 3) for i in 1:5]
        for i in 1:5
            @test testva[:, i] == 2i * ones(3, 3)
        end
        testva[:, 1:2:5] = [5i * ones(3, 3) for i in 1:2:5]
        for i in 1:2:5
            @test testva[:, i] == 5i * ones(3, 3)
        end
        testva[CartesianIndex(3, 3, 5)] = 64.0
        @test testva[:, 5][3, 3] == 64.0
        @test_throws ArgumentError testva[2, 1:2, :] = 108.0
        testva[2, 1:2, :] .= 108.0
        for i in 1:5
            @test all(testva[:, i][2, 1:2] .== 108.0)
        end
        testva[:, 3, :] = [3i / 7j for i in 1:3, j in 1:5]
        for j in 1:5
            for i in 1:3
                @test testva[i, 3, j] == 3i / 7j
            end
        end
    end

    @testset "v3 edge cases DiffEqArray scalar broadcast (ported)" begin
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        testva = RaggedDiffEqArray(x, x)
        testvb = RaggedDiffEqArray(x, x)
        mulX = sqrt.(abs.(testva .* testvb))
        ref = sqrt.(abs.(x .* x))
        @test mulX == ref
        fill!(mulX, 0)
        mulX .= sqrt.(abs.(testva .* testvb))
        @test mulX == ref
    end

    @testset "v3 multidimensional parent VectorOfArray (ported)" begin
        u_matrix = RaggedVectorOfArray([[1, 2] for i in 1:2, j in 1:3])
        u_vector = RaggedVectorOfArray([[1, 2] for i in 1:6])

        # test broadcasting
        function foo!(u)
            @. u += 2 * u * abs(u)
            return u
        end
        foo!(u_matrix)
        foo!(u_vector)
        @test all(u_matrix .== [3, 10])
        @test all(vec(u_matrix) .≈ vec(u_vector))

        # test that, for VectorOfArray with multi-dimensional parent arrays,
        # broadcast and `similar` preserve the structure of the parent array
        @test typeof(parent(similar(u_matrix))) == typeof(parent(u_matrix))
        @test typeof(parent((x -> x).(u_matrix))) == typeof(parent(u_matrix))

        # test efficiency (allow 1 alloc on Julia 1.10 LTS)
        num_allocs = @allocations foo!(u_matrix)
        @test num_allocs <= 1
    end

    @testset "v3 issue 354 (ported)" begin
        @test RaggedVectorOfArray(ones(1))[:] == ones(1)
    end

    # ===================================================================
    # Tests ported from v3 interface_tests.jl
    # ===================================================================

    @testset "v3 iteration (ported)" begin
        t = 1:3
        testva = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        testda = RaggedDiffEqArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], t)

        for (i, elem) in enumerate(testva)
            @test elem == testva[:, i]
        end

        for (i, elem) in enumerate(testda)
            @test elem == testda[:, i]
        end
    end

    @testset "v3 push! and copy (ported)" begin
        testva = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        testda = RaggedDiffEqArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1:3)

        push!(testva, [10, 11, 12])
        @test testva[:, end] == [10, 11, 12]
        push!(testda, [10, 11, 12])
        @test testda[:, end] == [10, 11, 12]

        testva2 = copy(testva)
        push!(testva2, [13, 14, 15])
        testda2 = copy(testva)
        push!(testda2, [13, 14, 15])

        # make sure we copy when we pass containers
        @test size(testva) == (3, 4)
        @test testva2[:, end] == [13, 14, 15]
        @test size(testda) == (3, 4)
        @test testda2[:, end] == [13, 14, 15]
    end

    @testset "v3 append! (ported)" begin
        testva = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        testda = RaggedDiffEqArray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], 1:4)

        append!(testva, testva)
        @test testva[1:2, 5:6] == [1 4; 2 5]
        append!(testda, testda)
        @test testda[1:2, 5:6] == [1 4; 2 5]
    end

    @testset "v3 push! making array ragged (ported)" begin
        testva = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        testda = RaggedDiffEqArray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], 1:4)

        # After appending, add ragged element
        append!(testva, testva)
        append!(testda, testda)

        push!(testva, [-1, -2, -3, -4])
        push!(testda, [-1, -2, -3, -4])
        @test testva[1:2, 5:6] == [1 4; 2 5]
        @test testda[1:2, 5:6] == [1 4; 2 5]

        @test_throws BoundsError testva[4:5, 5:6]
        @test_throws BoundsError testda[4:5, 5:6]

        @test testva[:, 9] == [-1, -2, -3, -4]
        @test testva[:, end] == [-1, -2, -3, -4]
        @test testda[:, 9] == [-1, -2, -3, -4]
        @test testda[:, end] == [-1, -2, -3, -4]
    end

    @testset "v3 ndims (ported)" begin
        testva = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        @test ndims(testva) == 2
        @test ndims(typeof(testva)) == 2
    end

    @testset "v3 push! shape enforcement (ported)" begin
        testva = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        testda = RaggedDiffEqArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1:3)

        @test_throws MethodError push!(testva, [-1 -2 -3 -4])
        @test_throws MethodError push!(testva, [-1 -2; -3 -4])
        @test_throws MethodError push!(testda, [-1 -2 -3 -4])
        @test_throws MethodError push!(testda, [-1 -2; -3 -4])
    end

    @testset "v3 type inference (ported)" begin
        testva = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        @inferred sum(testva)
        @inferred sum(RaggedVectorOfArray([RaggedVectorOfArray([zeros(4, 4)])]))
        @inferred mapreduce(string, *, testva)
        # Type stability for `end` indexing (issue #525)
        testva_end = RaggedVectorOfArray(fill(fill(2.0, 2), 10))
        # Use lastindex directly since `end` doesn't work in SafeTestsets
        last_col = lastindex(testva_end, 2)
        @inferred testva_end[1, last_col]
        @inferred testva_end[1, 1:last_col]
        @test testva_end[1, last_col] == 2.0
        last_row = lastindex(testva_end, 1)
        @inferred testva_end[last_row, 1]
        @inferred testva_end[1:last_row, 1]
        @test testva_end[last_row, 1] == 2.0
    end

    @testset "v3 mapreduce (ported)" begin
        testva = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        @test mapreduce(x -> string(x) * "q", *, testva) == "1q2q3q4q5q6q7q8q9q"

        testvb = RaggedVectorOfArray([rand(1:10, 3, 3, 3) for _ in 1:4])
        arrvb = Array(testvb)
        for i in 1:ndims(arrvb)
            @test sum(arrvb; dims = i) == sum(testvb; dims = i)
            @test prod(arrvb; dims = i) == prod(testvb; dims = i)
            @test mapreduce(string, *, arrvb; dims = i) == mapreduce(string, *, testvb; dims = i)
        end
    end

    @testset "v3 mapreduce ndims==1 (ported)" begin
        testvb = RaggedVectorOfArray(collect(1.0:0.1:2.0))
        arrvb = Array(testvb)
        @test sum(arrvb) == sum(testvb)
        @test prod(arrvb) == prod(testvb)
        @test mapreduce(string, *, arrvb) == mapreduce(string, *, testvb)
    end

    @testset "v3 view (ported)" begin
        testvc = RaggedVectorOfArray([rand(1:10, 3, 3) for _ in 1:3])
        arrvc = Array(testvc)
        for idxs in [
                (2, 2, :), (2, :, 2), (:, 2, 2), (:, :, 2), (:, 2, :), (2, :, :), (:, :, :),
                (1:2, 1:2, Bool[1, 0, 1]), (1:2, Bool[1, 0, 1], 1:2), (Bool[1, 0, 1], 1:2, 1:2),
            ]
            arr_view = view(arrvc, idxs...)
            voa_view = view(testvc, idxs...)
            @test size(arr_view) == size(voa_view)
            @test all(arr_view .== voa_view)
        end

        testvc = RaggedVectorOfArray(collect(1:10))
        arrvc = Array(testvc)
        bool_idx = rand(Bool, 10)
        for (voaidx, arridx) in [
                ((:,), (:,)),
                ((3:5,), (3:5,)),
                ((bool_idx,), (bool_idx,)),
            ]
            arr_view = view(arrvc, arridx...)
            voa_view = view(testvc.u, voaidx...)
            @test size(arr_view) == size(voa_view)
            @test all(arr_view .== voa_view)
        end
    end

    @testset "v3 stack (ported)" begin
        testva = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        @test stack(testva) == [1 4 7; 2 5 8; 3 6 9]
        @test stack(testva; dims = 1) == [1 2 3; 4 5 6; 7 8 9]

        testva = RaggedVectorOfArray([RaggedVectorOfArray([ones(2, 2), 2ones(2, 2)]), 3ones(2, 2, 2)])
        @test stack(testva) ==
            [1.0 1.0; 1.0 1.0;;; 2.0 2.0; 2.0 2.0;;;; 3.0 3.0; 3.0 3.0;;; 3.0 3.0; 3.0 3.0]
    end

    @testset "v3 convert Array (ported)" begin
        t = 1:8
        recs = [rand(10, 7) for i in 1:8]
        testva = RaggedVectorOfArray(recs)
        testda = RaggedDiffEqArray(recs, t)
        testa = cat(recs..., dims = 3)

        @test convert(Array, testva) == testa
        @test convert(Array, testda) == testa

        t = 1:3
        recs = [[1 2; 3 4], [3 5; 6 7], [8 9; 10 11]]
        testva = RaggedVectorOfArray(recs)
        testda = RaggedDiffEqArray(recs, t)

        @test size(convert(Array, testva)) == (2, 2, 3)
        @test size(convert(Array, testda)) == (2, 2, 3)
    end

    @testset "v3 similar (ported)" begin
        recs = [rand(6) for i in 1:4]
        testva = RaggedVectorOfArray(recs)

        testva2 = similar(testva)
        @test typeof(testva2) == typeof(testva)
        @test size(testva2) == size(testva)

        testva3 = similar(testva, 10)
        @test typeof(testva3) == typeof(testva)
        @test length(testva3) == 10
    end

    @testset "v3 fill! and all (ported)" begin
        recs = [rand(6) for i in 1:4]
        testva = RaggedVectorOfArray(recs)

        testval = 3.0
        testva2 = similar(testva)
        fill!(testva2, testval)
        @test all(x -> (x == testval), testva2)
        testts = rand(Float64, size(testva.u))
        testda = RaggedDiffEqArray(recursivecopy(testva.u), testts)
        fill!(testda, testval)
        @test all(x -> (x == testval), testda)
    end

    @testset "v3 copyto! (ported)" begin
        testva = RaggedVectorOfArray(collect(0.1:0.1:1.0))
        arr = 0.2:0.2:2.0
        copyto!(testva, arr)
        @test Array(testva) == arr
        testva = RaggedVectorOfArray([i * ones(3, 2) for i in 1:4])
        arr = rand(3, 2, 4)
        copyto!(testva, arr)
        @test Array(testva) == arr
        testva = RaggedVectorOfArray(
            [
                ones(3, 2, 2),
                RaggedVectorOfArray(
                    [
                        2ones(3, 2),
                        RaggedVectorOfArray([3ones(3), 4ones(3)]),
                    ]
                ),
                RaggedDiffEqArray(
                    [
                        5ones(3, 2),
                        RaggedVectorOfArray([6ones(3), 7ones(3)]),
                    ],
                    [0.1, 0.2],
                    [100.0, 200.0],
                    SymbolCache([:x, :y], [:a, :b], :t)
                ),
            ]
        )
        arr = rand(3, 2, 2, 3)
        copyto!(testva, arr)
        @test Array(testva) == arr
        # ensure structure and fields are maintained
        @test testva.u[1] isa Array
        @test testva.u[2] isa RaggedVectorOfArray
        @test testva.u[2].u[2] isa RaggedVectorOfArray
        @test testva.u[3] isa RaggedDiffEqArray
        @test testva.u[3].u[2] isa RaggedVectorOfArray
        @test testva.u[3].t == [0.1, 0.2]
        @test testva.u[3].p == [100.0, 200.0]
        @test testva.u[3].sys isa SymbolCache
    end

    @testset "v3 any (ported)" begin
        recs = [collect(1:5), collect(6:10), collect(11:15)]
        testts = rand(5)
        testva = RaggedVectorOfArray(recs)
        testda = RaggedDiffEqArray(recs, testts)
        testval1 = 4
        testval2 = 17
        @test any(x -> (x == testval1), testva)
        @test !any(x -> (x == testval2), testda)
    end

    @testset "v3 empty creation (ported)" begin
        emptyva = RaggedVectorOfArray(Array{Vector{Float64}}([]))
        @test isempty(emptyva)
        emptyda = RaggedDiffEqArray(Array{Vector{Float64}}([]), Vector{Float64}())
        @test isempty(emptyda)
    end

    @testset "v3 map (ported)" begin
        A = RaggedVectorOfArray(map(i -> rand(2, 4), 1:7))
        @test map(x -> maximum(x), A) isa Vector

        DA = RaggedDiffEqArray(map(i -> rand(2, 4), 1:7), 1:7)
        @test map(x -> maximum(x), DA) isa Vector
    end

    @testset "v3 zero and resize! (ported)" begin
        u = RaggedVectorOfArray([fill(2.0, 2), ones(2)])
        @test typeof(zero(u)) <: typeof(u)
        resize!(u, 3)
        @test pointer(u) === pointer(u.u)
    end

    @testset "v3 DiffEqArray constructor ambiguity (ported, issue SciMLBase#889)" begin
        darr = RaggedDiffEqArray([[1.0, 1.0]], [1.0], ())
        @test darr.p == ()
        @test darr.sys === nothing
        @test ndims(darr) == 2
        darr = RaggedDiffEqArray([[1.0, 1.0]], [1.0], (), "A")
        @test darr.p == ()
        @test darr.sys == "A"
        @test ndims(darr) == 2
        darr = RaggedDiffEqArray([ones(2, 2)], [1.0], (1, 1, 1))
        @test darr.p == (1, 1, 1)
        @test darr.sys === nothing
        @test ndims(darr) == 3
        darr = RaggedDiffEqArray([ones(2, 2)], [1.0], (1, 1, 1), "A")
        @test darr.p == (1, 1, 1)
        @test darr.sys == "A"
        @test ndims(darr) == 3
    end

    @testset "v3 system retained in 4-argument constructor (ported)" begin
        darr = RaggedDiffEqArray([ones(2)], [1.0], :params, :sys)
        @test darr.sys == :sys
    end

    # issparse test skipped: ragged types are not AbstractArray, so
    # SparseArrays.issparse is not defined for them (issue #486 only
    # applies to AbstractVectorOfArray which is now AbstractArray).

    # ===================================================================
    # Additional tests for new v4 functionality (interp, conversion, SII)
    # ===================================================================

    @testset "RaggedVectorOfArray construction (typed)" begin
        r = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0]])
        @test r.u == [[1.0, 2.0], [3.0, 4.0, 5.0]]
        @test length(r) == 2
        @test ndims(r) == 2

        # From typed vector
        r2 = RaggedVectorOfArray(Vector{Float64}[[1.0], [2.0, 3.0]])
        @test length(r2) == 2
    end

    # Conversion, SII, interp/dense, and type hierarchy tests below

    @testset "RaggedVectorOfArray <-> VectorOfArray conversion" begin
        r = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0]])

        # To VectorOfArray (rectangular, zero-padded)
        va = VectorOfArray(r)
        @test va isa VectorOfArray
        @test size(va) == (3, 2)
        @test va[:, 1] == [1.0, 2.0, 0.0]  # zero-padded
        @test va[:, 2] == [3.0, 4.0, 5.0]

        # Back to RaggedVectorOfArray
        r2 = RaggedVectorOfArray(va)
        @test r2 isa RaggedVectorOfArray
        # u is shared, inner arrays have original sizes
        @test r2[:, 1] == [1.0, 2.0]
        @test r2[:, 2] == [3.0, 4.0, 5.0]
    end

    @testset "RaggedVectorOfArray equality" begin
        r1 = RaggedVectorOfArray([[1, 2], [3, 4, 5]])
        r2 = RaggedVectorOfArray([[1, 2], [3, 4, 5]])
        r3 = RaggedVectorOfArray([[1, 2], [3, 4, 6]])
        @test r1 == r2
        @test r1 != r3
    end

    @testset "RaggedVectorOfArray reverse" begin
        r = RaggedVectorOfArray([[1.0], [2.0, 3.0], [4.0, 5.0, 6.0]])
        rr = reverse(r)
        @test rr[:, 1] == [4.0, 5.0, 6.0]
        @test rr[:, 2] == [2.0, 3.0]
        @test rr[:, 3] == [1.0]
    end

    @testset "RaggedDiffEqArray construction" begin
        r = RaggedDiffEqArray([[1.0, 2.0], [3.0, 4.0, 5.0]], [0.0, 1.0])
        @test r.u == [[1.0, 2.0], [3.0, 4.0, 5.0]]
        @test r.t == [0.0, 1.0]
        @test length(r) == 2

        # With parameters
        r2 = RaggedDiffEqArray([[1.0], [2.0, 3.0]], [0.0, 1.0], [0.5])
        @test r2.p == [0.5]
    end

    @testset "RaggedDiffEqArray indexing — no zero-padding" begin
        r = RaggedDiffEqArray([[1.0, 2.0], [3.0, 4.0, 5.0]], [0.0, 1.0])

        @test r[:, 1] == [1.0, 2.0]
        @test r[:, 2] == [3.0, 4.0, 5.0]
        @test r[1, 1] == 1.0
        @test r[2, 2] == 4.0
        @test_throws BoundsError r[3, 1]

        # Linear indexing returns inner arrays (v3 behavior)
        @test r[1] == [1.0, 2.0]
        @test r[2] == [3.0, 4.0, 5.0]

        # Colon-colon preserves DiffEqArray type
        r_all = r[:, :]
        @test r_all isa RaggedDiffEqArray
        @test r_all.t == [0.0, 1.0]

        # Subset preserves time
        r_sub = r[:, [2]]
        @test r_sub isa RaggedDiffEqArray
        @test r_sub.t == [1.0]
        @test r_sub[:, 1] == [3.0, 4.0, 5.0]
    end

    @testset "RaggedDiffEqArray <-> DiffEqArray conversion" begin
        r = RaggedDiffEqArray([[1.0, 2.0], [3.0, 4.0, 5.0]], [0.0, 1.0])

        da = DiffEqArray(r)
        @test da isa DiffEqArray
        @test da.t == [0.0, 1.0]

        r2 = RaggedDiffEqArray(da)
        @test r2 isa RaggedDiffEqArray
        @test r2[:, 1] == [1.0, 2.0]
        @test r2[:, 2] == [3.0, 4.0, 5.0]
        @test r2.t == [0.0, 1.0]
    end

    @testset "RaggedDiffEqArray copy/zero" begin
        r = RaggedDiffEqArray([[1.0, 2.0], [3.0, 4.0, 5.0]], [0.0, 1.0])

        r2 = copy(r)
        @test r2 == r
        @test r2.u !== r.u

        r0 = zero(r)
        @test r0[:, 1] == [0.0, 0.0]
        @test r0[:, 2] == [0.0, 0.0, 0.0]
        @test r0.t == [0.0, 1.0]
    end

    @testset "RaggedDiffEqArray SymbolicIndexingInterface" begin
        r = RaggedDiffEqArray(
            [[1.0, 2.0], [3.0, 4.0, 5.0]], [0.0, 1.0];
            variables = [:a, :b], parameters = [:p], independent_variables = [:t]
        )

        @test is_timeseries(r) == Timeseries()
        @test state_values(r) === r.u
        @test current_time(r) === r.t
        @test symbolic_container(r) === r.sys

        # Variable/parameter query
        @test is_variable(r, :a)
        @test is_variable(r, :b)
        @test is_parameter(r, :p)
        @test is_independent_variable(r, :t)
        @test variable_index(r, :a) == 1
        @test variable_index(r, :b) == 2
        @test parameter_index(r, :p) == 1
    end

    @testset "RaggedDiffEqArray symbolic getindex" begin
        r = RaggedDiffEqArray(
            [[1.0, 2.0], [3.0, 4.0, 5.0]], [0.0, 1.0], [0.5];
            variables = [:a, :b], parameters = [:p], independent_variables = [:t]
        )

        # Symbolic variable indexing
        @test r[:a] == [1.0, 3.0]
        @test r[:a, 1] == 1.0
        @test r[:a, 2] == 3.0
        @test r[:b] == [2.0, 4.0]

        # Parameter indexing should throw
        @test_throws RecursiveArrayToolsRaggedArrays.ParameterIndexingError r[:p]
    end

    @testset "2D inner arrays" begin
        r = RaggedVectorOfArray([zeros(2, 3), zeros(2, 4)])
        @test length(r) == 2
        @test ndims(r) == 3
        @test size(r[:, 1]) == (2, 3)
        @test size(r[:, 2]) == (2, 4)

        # No zero-padding in column access
        @test r[:, 1] == zeros(2, 3)
        @test r[:, 2] == zeros(2, 4)
    end

    @testset "RaggedDiffEqArray interp/dense fields" begin
        r = RaggedDiffEqArray([[1.0, 2.0], [3.0, 4.0, 5.0]], [0.0, 1.0])
        @test r.interp === nothing
        @test r.dense == false

        # With interp kwarg
        r2 = RaggedDiffEqArray(
            [[1.0, 2.0], [3.0, 4.0, 5.0]], [0.0, 1.0];
            interp = :test_interp, dense = true
        )
        @test r2.interp == :test_interp
        @test r2.dense == true

        # Callable errors when no interp
        @test_throws ErrorException r(0.5)

        # Conversion preserves interp/dense
        da = DiffEqArray(r2)
        @test da.interp == :test_interp
        @test da.dense == true

        r3 = RaggedDiffEqArray(da)
        @test r3.interp == :test_interp
        @test r3.dense == true

        # Copy preserves interp/dense
        r4 = copy(r2)
        @test r4.interp == :test_interp
        @test r4.dense == true
    end

    @testset "v3 zero and fill on ragged (ported)" begin
        r = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0]])

        # zero preserves ragged structure
        r0 = zero(r)
        @test r0[:, 1] == [0.0, 0.0]
        @test r0[:, 2] == [0.0, 0.0, 0.0]
        @test length(r0[:, 1]) == 2
        @test length(r0[:, 2]) == 3

        # fill! preserves ragged structure
        fill!(r0, 42.0)
        @test r0[:, 1] == [42.0, 42.0]
        @test r0[:, 2] == [42.0, 42.0, 42.0]

        # .= zero works (the exact bug JoshuaLampert reported)
        r2 = copy(r)
        r2 .= zero(r2)
        @test r2[:, 1] == [0.0, 0.0]
        @test r2[:, 2] == [0.0, 0.0, 0.0]
    end

    @testset "v3 component timeseries (ported)" begin
        r = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0]])

        # A[j, :] returns the j-th component across all inner arrays
        @test r[1, :] == [1.0, 3.0, 6.0]
        @test r[2, :] == [2.0, 4.0, 7.0]
    end

    @testset "Type hierarchy" begin
        r = RaggedVectorOfArray([[1, 2], [3, 4, 5]])
        @test r isa RecursiveArrayTools.AbstractRaggedVectorOfArray
        @test !(r isa AbstractArray)

        rd = RaggedDiffEqArray([[1.0], [2.0, 3.0]], [0.0, 1.0])
        @test rd isa RecursiveArrayTools.AbstractRaggedDiffEqArray
        @test rd isa RecursiveArrayTools.AbstractRaggedVectorOfArray
        @test !(rd isa AbstractArray)
    end
end

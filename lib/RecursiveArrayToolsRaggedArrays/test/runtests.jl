using RecursiveArrayTools
using RecursiveArrayToolsRaggedArrays
using SymbolicIndexingInterface
using Test

@testset "RecursiveArrayToolsRaggedArrays" begin
    @testset "RaggedVectorOfArray construction" begin
        r = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0]])
        @test r.u == [[1.0, 2.0], [3.0, 4.0, 5.0]]
        @test length(r) == 2
        @test ndims(r) == 2

        # From typed vector
        r2 = RaggedVectorOfArray(Vector{Float64}[[1.0], [2.0, 3.0]])
        @test length(r2) == 2
    end

    @testset "RaggedVectorOfArray linear indexing (column-major)" begin
        r = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0]])

        # Linear indexing goes through elements in column-major order:
        # col 1: [1.0, 2.0], col 2: [3.0, 4.0, 5.0]
        @test r[1] == 1.0
        @test r[2] == 2.0
        @test r[3] == 3.0
        @test r[4] == 4.0
        @test r[5] == 5.0
        @test_throws BoundsError r[6]

        # Linear setindex!
        r2 = copy(r)
        r2[1] = 10.0
        @test r2.u[1][1] == 10.0
        r2[4] = 40.0
        @test r2.u[2][2] == 40.0
    end

    @testset "RaggedVectorOfArray N-ary indexing — no zero-padding" begin
        r = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]])

        # A[:, i] returns i-th inner array directly (NO zero-padding)
        @test r[:, 1] == [1.0, 2.0]
        @test r[:, 2] == [3.0, 4.0, 5.0]
        @test r[:, 3] == [6.0]
        @test length(r[:, 1]) == 2
        @test length(r[:, 2]) == 3
        @test length(r[:, 3]) == 1

        # A[j, i] returns component (no zero-padding, throws on out-of-bounds)
        @test r[1, 1] == 1.0
        @test r[2, 1] == 2.0
        @test r[3, 2] == 5.0
        @test r[1, 3] == 6.0
        @test_throws BoundsError r[3, 1]  # only 2 elements in first array
        @test_throws BoundsError r[2, 3]  # only 1 element in third array

        # A[j, :] returns time series of j-th component
        @test r[1, :] == [1.0, 3.0, 6.0]

        # A[:, idx_array] returns subset
        r_sub = r[:, [1, 3]]
        @test r_sub isa RaggedVectorOfArray
        @test length(r_sub) == 2
        @test r_sub[:, 1] == [1.0, 2.0]
        @test r_sub[:, 2] == [6.0]

        # A[:, bool_array]
        r_bool = r[:, [true, false, true]]
        @test r_bool isa RaggedVectorOfArray
        @test length(r_bool) == 2

        # A[:, :]
        r_all = r[:, :]
        @test r_all isa RaggedVectorOfArray
        @test r_all == r
        @test r_all.u !== r.u  # copy
    end

    @testset "RaggedVectorOfArray setindex!" begin
        r = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0]])
        r[1, 1] = 10.0
        @test r[1, 1] == 10.0

        r[:, 2] = [30.0, 40.0, 50.0]
        @test r[:, 2] == [30.0, 40.0, 50.0]
    end

    @testset "RaggedVectorOfArray iteration" begin
        r = RaggedVectorOfArray([[1, 2], [3, 4, 5]])
        collected = collect(r)
        @test collected == [[1, 2], [3, 4, 5]]
        @test first(r) == [1, 2]
        @test last(r) == [3, 4, 5]
    end

    @testset "RaggedVectorOfArray copy, zero, similar, fill!" begin
        r = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0]])

        r2 = copy(r)
        @test r2 == r
        @test r2.u !== r.u

        r0 = zero(r)
        @test r0[:, 1] == [0.0, 0.0]
        @test r0[:, 2] == [0.0, 0.0, 0.0]

        rs = similar(r)
        @test length(rs[:, 1]) == 2
        @test length(rs[:, 2]) == 3

        fill!(r0, 7.0)
        @test r0[:, 1] == [7.0, 7.0]
        @test r0[:, 2] == [7.0, 7.0, 7.0]
    end

    @testset "RaggedVectorOfArray push!, append!" begin
        r = RaggedVectorOfArray([[1.0], [2.0, 3.0]])
        push!(r, [4.0, 5.0, 6.0])
        @test length(r) == 3
        @test r[:, 3] == [4.0, 5.0, 6.0]

        r2 = RaggedVectorOfArray([[7.0]])
        append!(r, r2)
        @test length(r) == 4
        @test r[:, 4] == [7.0]
    end

    @testset "RaggedVectorOfArray broadcasting" begin
        r = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0]])

        # Scalar broadcast
        r2 = r .* 2.0
        @test r2 isa RaggedVectorOfArray
        @test r2[:, 1] == [2.0, 4.0]
        @test r2[:, 2] == [6.0, 8.0, 10.0]

        # Element-wise broadcast between ragged arrays of same structure
        r3 = r .+ r
        @test r3 isa RaggedVectorOfArray
        @test r3[:, 1] == [2.0, 4.0]
        @test r3[:, 2] == [6.0, 8.0, 10.0]

        # In-place broadcast
        r4 = copy(r)
        r4 .= r .+ 1.0
        @test r4[:, 1] == [2.0, 3.0]
        @test r4[:, 2] == [4.0, 5.0, 6.0]

        # .= zero(r) should work (the issue from JoshuaLampert)
        r5 = copy(r)
        r5 .= zero(r5)
        @test r5[:, 1] == [0.0, 0.0]
        @test r5[:, 2] == [0.0, 0.0, 0.0]
    end

    @testset "VectorOfArray ragged .= zero also works" begin
        # This is the exact bug from JoshuaLampert's report
        f = VectorOfArray([[1.0], [2.0, 3.0]])
        f .= zero(f)
        @test f.u[1] == [0.0]
        @test f.u[2] == [0.0, 0.0]
    end

    @testset "RaggedVectorOfArray ↔ VectorOfArray conversion" begin
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

    @testset "RaggedVectorOfArray inner_sizes/inner_lengths" begin
        r = RaggedVectorOfArray([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]])
        @test inner_lengths(r) == [2, 1, 3]
        @test inner_sizes(r) == [(2,), (1,), (3,)]
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

        # Linear indexing (column-major)
        @test r[1] == 1.0
        @test r[2] == 2.0
        @test r[3] == 3.0

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

    @testset "RaggedDiffEqArray ↔ DiffEqArray conversion" begin
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
        @test_throws RecursiveArrayToolsRaggedArrays.RaggedParameterIndexingError r[:p]
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

    # ===================================================================
    # Tests ported from v3 VectorOfArray ragged behavior
    # These were the standard ragged tests before v4's AbstractArray
    # subtyping changed VectorOfArray to zero-pad. The ragged sublibrary
    # preserves the original non-zero-padded behavior.
    # ===================================================================

    @testset "v3 ragged indexing (ported)" begin
        recs = [[1, 2, 3], [3, 5, 6, 7], [8, 9, 10, 11]]
        r = RaggedVectorOfArray(recs)
        rd = RaggedDiffEqArray(recs, 1:3)

        # Colon indexing returns actual inner array (no zero-padding)
        @test r[:, 1] == recs[1]
        @test rd[:, 1] == recs[1]

        # Subarray indexing into compatible region
        @test r[1, 1] == 1
        @test r[2, 1] == 2
        @test r[1, 2] == 3
        @test r[2, 2] == 5

        # DiffEqArray column slicing preserves time
        rd_sub = rd[:, 1:2]
        @test rd_sub isa RaggedDiffEqArray
        @test rd_sub.t == [1, 2]
        rd_sub2 = rd[:, 2:3]
        @test rd_sub2 isa RaggedDiffEqArray
        @test rd_sub2.t == [2, 3]
    end

    @testset "v3 heterogeneous views (ported, issue #453)" begin
        f = RaggedVectorOfArray([[1.0], [2.0, 3.0]])
        # Column access respects actual inner array size
        @test length(f[:, 1]) == 1
        @test length(f[:, 2]) == 2
        @test f[:, 1] == [1.0]
        @test f[:, 2] == [2.0, 3.0]

        # view also respects actual inner array size
        @test length(view(f, :, 1)) == 1
        @test length(view(f, :, 2)) == 2
        @test view(f, :, 1) == [1.0]
        @test view(f, :, 2) == [2.0, 3.0]
        @test collect(view(f, :, 1)) == f[:, 1]
        @test collect(view(f, :, 2)) == f[:, 2]

        f2 = RaggedVectorOfArray([[1.0, 2.0], [3.0]])
        @test length(f2[:, 1]) == 2
        @test length(f2[:, 2]) == 1
        @test f2[:, 1] == [1.0, 2.0]
        @test f2[:, 2] == [3.0]
        @test length(view(f2, :, 1)) == 2
        @test length(view(f2, :, 2)) == 1
        @test view(f2, :, 1) == [1.0, 2.0]
        @test view(f2, :, 2) == [3.0]
    end

    @testset "v3 end indexing with ragged arrays (ported)" begin
        # `end` in the first dimension is per-column via RaggedEnd (matching v3 behavior)
        ragged = RaggedVectorOfArray([[1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]])
        @test ragged[end, 1] == 2.0   # end = lastindex(inner[1]) = 2
        @test ragged[end, 2] == 5.0   # end = lastindex(inner[2]) = 3
        @test ragged[end, 3] == 9.0   # end = lastindex(inner[3]) = 4
        @test ragged[end - 1, 1] == 1.0
        @test ragged[end - 1, 2] == 4.0
        @test ragged[end - 1, 3] == 8.0

        # 1:end also resolves per-column
        @test ragged[1:end, 1] == [1.0, 2.0]
        @test ragged[1:end, 2] == [3.0, 4.0, 5.0]
        @test ragged[1:end, 3] == [6.0, 7.0, 8.0, 9.0]

        # Colon returns actual arrays
        @test ragged[:, 1] == [1.0, 2.0]
        @test ragged[:, 2] == [3.0, 4.0, 5.0]
        @test ragged[:, 3] == [6.0, 7.0, 8.0, 9.0]

        # `end` in last (column) dimension
        @test ragged[:, end] == [6.0, 7.0, 8.0, 9.0]
        @test ragged[:, 2:end] isa RaggedVectorOfArray
        @test ragged[:, (end - 1):end] isa RaggedVectorOfArray
        @test length(ragged[:, (end - 1):end]) == 2

        ragged2 = RaggedVectorOfArray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0], [7.0, 8.0, 9.0]])
        @test ragged2[end, 1] == 4.0    # end = 4
        @test ragged2[end, 2] == 6.0    # end = 2
        @test ragged2[end, 3] == 9.0    # end = 3
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
        @test ragged2[:, 2:end] isa RaggedVectorOfArray
        @test ragged2[:, (end - 1):end] isa RaggedVectorOfArray
        @test ragged2[1:(end - 1), 1] == [1.0, 2.0, 3.0]
        @test ragged2[1:(end - 1), 2] == [5.0]
        @test ragged2[1:(end - 1), 3] == [7.0, 8.0]
    end

    @testset "v3 push! making array ragged (ported)" begin
        r = RaggedVectorOfArray([[1, 2, 3], [4, 5, 6]])
        push!(r, [-1, -2, -3, -4])

        # Can still index compatible region
        @test r[1, 1] == 1
        @test r[2, 1] == 2
        @test r[1, 2] == 4
        @test r[2, 2] == 5

        # Out-of-bounds on shorter arrays throws
        @test_throws BoundsError r[4, 1]
        @test_throws BoundsError r[4, 2]

        # Full column access of the new ragged element
        @test r[:, 3] == [-1, -2, -3, -4]
        @test length(r) == 3
    end

    @testset "v3 broadcast assignment (ported, issue #454)" begin
        u = RaggedVectorOfArray([[1.0], [2.0, 3.0]])
        @test length(u[:, 1]) == 1
        @test length(u[:, 2]) == 2

        # Broadcast assignment into a column
        u[:, 2] = [10.0, 11.0]
        @test u.u[2] == [10.0, 11.0]
    end

    @testset "v3 DiffEqArray 2D ragged inner arrays (ported)" begin
        recs_2d = [rand(2, 3), rand(2, 4)]
        rd = RaggedDiffEqArray(recs_2d, 1:2)
        @test rd[:, 1] == recs_2d[1]
        @test rd[:, 2] == recs_2d[2]
        @test size(rd[:, 1]) == (2, 3)
        @test size(rd[:, 2]) == (2, 4)

        # Subset preserves time
        rd_sub = rd[:, [1, 2]]
        @test rd_sub isa RaggedDiffEqArray
        @test rd_sub.t == [1, 2]
        @test rd_sub[:, 1] == recs_2d[1]
        @test rd_sub[:, 2] == recs_2d[2]
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

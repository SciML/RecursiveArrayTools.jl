using RecursiveArrayTools, StaticArrays
using Test

t = collect(range(0, stop = 10, length = 200))
randomized = VectorOfArray([0.01randn(2) for i in 1:10])
data = convert(Array, randomized)
@test typeof(data) <: Matrix{Float64}

## Test means
A = [[1 2; 3 4], [1 3; 4 6], [5 6; 7 8]]
@test recursive_mean(A) ≈ [2.33333333 3.666666666
                           4.6666666666 6.0]

A = zeros(5, 5)
@test recursive_unitless_eltype(A) == Float64

@test vecvecapply(x -> abs.(x), -1) == 1
@test vecvecapply(x -> abs.(x), [-1, -2, 3, -4]) == [1, 2, 3, 4]
v = [[-1 2; 3 -4], [5 -6; -7 -8]]
vv = [1, 3, 2, 4, 5, 7, 6, 8]
@test vecvecapply(x -> abs.(x), v) == vv
@test vecvecapply(x -> abs.(x), VectorOfArray(v)) == vv

using Unitful
A = zeros(5, 5) * 1u"kg"
@test recursive_unitless_eltype(A) == Float64
AA = [zeros(5, 5) for i in 1:5]
@test recursive_unitless_eltype(AA) == Array{Float64, 2}
AofA = [copy(A) for i in 1:5]
@test recursive_unitless_eltype(AofA) == Array{Float64, 2}
AofSA = [@SVector [2.0, 3.0] for i in 1:5]
@test recursive_unitless_eltype(AofSA) == SVector{2, Float64}
AofuSA = [@SVector [2.0u"kg", 3.0u"kg"] for i in 1:5]
@test recursive_unitless_eltype(AofuSA) == SVector{2, Float64}

A = [ArrayPartition(ones(1), ones(1))]

function test_recursive_bottom_eltype()
    function test_value(val::Any, expected_type::Type)
        # It should return the expected type for the given expected type
        @test recursive_bottom_eltype(expected_type) == expected_type

        # It should return the expected type for the given value
        @test recursive_bottom_eltype(val) == expected_type

        # It should return the expected type for an array of the given value
        Aval = [val for i in 1:5]
        @test recursive_bottom_eltype(Aval) == expected_type

        # It should return expected type for a nested array of the gicen value
        AAval = [Aval for i in 1:5]
        @test recursive_bottom_eltype(AAval) == expected_type

        # It should return expected type for an array of vectors of chars
        AVval = [@SVector [val, val] for i in 1:5]
        @test recursive_bottom_eltype(AVval) == expected_type
    end

    # testing chars
    test_value('c', Char)

    # testing strings
    # We expect recursive_bottom_eltype to return `Char` for a string, because
    # `eltype("Some String") == Char`
    test_value("Some String", Char)

    # testing integer values
    test_value(1, Int)
    test_value(1u"kg", eltype(1u"kg"))

    # testing float values
    test_value(1.0, Float64)
    test_value(1.0u"kg", eltype(1.0u"kg"))
end
test_recursive_bottom_eltype()

x = zeros(10)
recursivefill!(x, 1.0)
@test x == ones(10)

x = [zeros(10), zeros(10)]
recursivefill!(x, 1.0)
@test x[1] == ones(10)
@test x[2] == ones(10)

x = [SVector{10}(zeros(10)), SVector{10}(zeros(10))]
recursivefill!(x, SVector{10}(ones(10)))
@test x[1] == SVector{10}(ones(10))
@test x[2] == SVector{10}(ones(10))

x = [MVector{10}(zeros(10)), MVector{10}(zeros(10))]
recursivefill!(x, 1.0)
@test x[1] == MVector{10}(ones(10))
@test x[2] == MVector{10}(ones(10))

x = [similar(x[1]), similar(x[1])]
recursivefill!(x, true)
@test x[1] == MVector{10}(ones(10))
@test x[2] == MVector{10}(ones(10))

x = similar(x)
recursivefill!(x, true)
@test x[1] == MVector{10}(ones(10))
@test x[2] == MVector{10}(ones(10))

# Test VectorOfArray + recursivefill! + static arrays
@testset "VectorOfArray + recursivefill! + static arrays" begin
    Vec3 = SVector{3, Float64}
    x = [randn(Vec3, n) for n in 1:4]  # vector of vectors of static arrays

    x_voa = VectorOfArray(x)
    @test eltype(x_voa) === Vec3
    @test first(x_voa) isa AbstractVector{Vec3}

    y_voa = recursivecopy(x_voa)
    recursivefill!(y_voa, true)
    @test all(y_voa[:, n] == fill(ones(Vec3), n) for n in 1:4)

    y_voa = recursivecopy(x_voa)
    recursivefill!(y_voa, ones(Vec3))
    @test all(y_voa[:, n] == fill(ones(Vec3), n) for n in 1:4)
end

@testset "VectorOfArray recursivecopy!" begin
    u1 = VectorOfArray([fill(2, MVector{2, Float64}), ones(MVector{2, Float64})])
    u2 = VectorOfArray([fill(4, MVector{2, Float64}), 2 .* ones(MVector{2, Float64})])
    recursivecopy!(u1, u2)
    @test u1.u[1] == [4.0, 4.0]
    @test u1.u[2] == [2.0, 2.0]
    @test u1.u[1] isa MVector
    @test u1.u[2] isa MVector

    u1 = VectorOfArray([fill(2, SVector{2, Float64}), ones(SVector{2, Float64})])
    u2 = VectorOfArray([fill(4, SVector{2, Float64}), 2 .* ones(SVector{2, Float64})])
    recursivecopy!(u1, u2)
    @test u1.u[1] == [4.0, 4.0]
    @test u1.u[2] == [2.0, 2.0]
    @test u1.u[1] isa SVector
    @test u1.u[2] isa SVector
end

import KernelAbstractions: get_backend
@testset "KernelAbstractions" begin
    v = VectorOfArray([randn(2) for i in 1:10])
    @test get_backend(v) === get_backend(parent(v)[1])
end

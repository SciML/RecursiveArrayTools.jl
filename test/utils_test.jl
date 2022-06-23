using RecursiveArrayTools, StaticArrays
using Test

t = collect(range(0, stop = 10, length = 200))
randomized = VectorOfArray([0.01randn(2) for i in 1:10])
data = convert(Array, randomized)
@test typeof(data) <: Matrix{Float64}

## Test means
A = [[1 2; 3 4], [1 3 4 6], [5 6 7 8]]
@test recursive_mean(A) ≈ [2.33333333 3.666666666
                           4.6666666666 6.0]

A = zeros(5, 5)
@test recursive_unitless_eltype(A) == Float64

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

using RecursiveArrayTools: issymbollike
@test !issymbollike(1)
@test issymbollike(:a)

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

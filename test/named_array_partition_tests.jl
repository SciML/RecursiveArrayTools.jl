using RecursiveArrayTools, ArrayInterface, Test

@testset "NamedArrayPartition tests" begin
    x = NamedArrayPartition(a = ones(10), b = rand(20))
    @test typeof(@. sin(x * x^2 / x - 1)) <: NamedArrayPartition
    @test typeof(x .^ 2) <: NamedArrayPartition
    @test typeof(similar(x)) <: NamedArrayPartition
    @test typeof(similar(x, Int)) <: NamedArrayPartition
    @test x.a ≈ ones(10)
    @test typeof(x .+ x[1:end]) <: Vector # x[1:end] is a plain Vector (type-stable slicing)
    @test all(x .== x[1:end])
    @test ArrayInterface.zeromatrix(x) isa Matrix
    @test size(ArrayInterface.zeromatrix(x)) == (30, 30)
    y = copy(x)
    @test zero(x, (10, 20)) == zero(x) # test that ignoring dims works
    @test typeof(zero(x)) <: NamedArrayPartition
    @test (y .*= 2).a[1] ≈ 2 # test in-place bcast

    @test length(Array(x)) == 30
    @test typeof(Array(x)) <: Array
    @test propertynames(x) == (:a, :b)

    x = NamedArrayPartition(a = ones(1), b = 2 * ones(1))
    @test Base.summary(x) == string(typeof(x), " with arrays:")
    io = IOBuffer()
    Base.show(io, MIME"text/plain"(), x)
    @test String(take!(io)) == "(a = [1.0], b = [2.0])"

    using StructArrays
    using StaticArrays: SVector
    x = NamedArrayPartition(
        a = StructArray{SVector{2, Float64}}((ones(5), 2 * ones(5))),
        b = StructArray{SVector{2, Float64}}((3 * ones(2, 2), 4 * ones(2, 2)))
    )
    @test typeof(x.a) <: StructVector{<:SVector{2}}
    @test typeof(x.b) <: StructArray{<:SVector{2}, 2}
    @test typeof((x -> x[1]).(x)) <: NamedArrayPartition
    @test typeof(map(x -> x[1], x)) <: NamedArrayPartition
end

# Regression test for https://github.com/SciML/RecursiveArrayTools.jl/issues/583:
# indexing a NamedArrayPartition with a UnitRange / Vector{Int} smaller than the
# whole array used to throw a MethodError because the AbstractArray indexing
# path called `similar(::NAP, T, dims)`, which tried to wrap a plain Vector
# (returned by `similar(::ArrayPartition, T, dims)` for `dims != size(A)`) in
# NamedArrayPartition's inner constructor, which requires an ArrayPartition.
#
# The `_unsafe_getindex(::IndexStyle, ::NAP, I...)` shortcut bypasses `similar`
# entirely, allocating a plain Vector destination directly. Slicing therefore
# always returns a Vector and is type-stable.
@testset "NamedArrayPartition issue #583 indexing" begin
    x = NamedArrayPartition(a = ones(2), b = 2 * ones(3))

    # UnitRange / Vector{Int} indexing all return Vector and are type-stable
    @test x[1:2] == [1.0, 1.0]
    @test x[2:4] == [1.0, 2.0, 2.0]
    @test x[1:end] == [1.0, 1.0, 2.0, 2.0, 2.0]
    @test x[[1, 2]] == [1.0, 1.0]
    @test x[[1, 4]] == [1.0, 2.0]

    @test x[1:2]    isa Vector{Float64}
    @test x[1:end]  isa Vector{Float64}
    @test x[[1, 4]] isa Vector{Float64}

    # Inferred return types: Vector, not Union
    @test (@inferred x[1:2])           isa Vector{Float64}
    @test (@inferred x[1:length(x)])   isa Vector{Float64}
    @test (@inferred x[[1, 4]])        isa Vector{Float64}

    # `similar` with a non-matching dims falls back to the backing array;
    # with matching dims keeps the NamedArrayPartition wrapper.
    @test similar(x, Float64, (2,)) isa Vector{Float64}
    @test similar(x, (2,))          isa Vector{Float64}
    @test similar(x, Float64, size(x)) isa NamedArrayPartition
    @test similar(x, size(x))          isa NamedArrayPartition

    # Scalar indexing untouched and type-stable
    @test x[1] == 1.0
    @test x[3] == 2.0
    @test (@inferred x[1]) === 1.0
    x[1] = 99.0
    @test x[1] == 99.0
end

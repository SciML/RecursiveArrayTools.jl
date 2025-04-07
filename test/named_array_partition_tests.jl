using RecursiveArrayTools, Test

@testset "NamedArrayPartition tests" begin
    x = NamedArrayPartition(a = ones(10), b = rand(20))
    @test typeof(@. sin(x * x^2 / x - 1)) <: NamedArrayPartition
    @test typeof(x .^ 2) <: NamedArrayPartition
    @test typeof(similar(x)) <: NamedArrayPartition
    @test typeof(similar(x, Int)) <: NamedArrayPartition
    @test typeof(similar(x, (5, 5))) <: Matrix
    @test typeof(similar(x, Int, (5, 5))) <: Matrix
    @test x.a ≈ ones(10)
    @test typeof(x .+ x[1:end]) <: Vector # test broadcast precedence 
    @test all(x .== x[1:end])
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
    x = NamedArrayPartition(a = StructArray{SVector{2, Float64}}((ones(5), 2 * ones(5))),
        b = StructArray{SVector{2, Float64}}((3 * ones(2, 2), 4 * ones(2, 2))))
    @test typeof(x.a) <: StructVector{<:SVector{2}}
    @test typeof(x.b) <: StructArray{<:SVector{2}, 2}
    @test typeof((x -> x[1]).(x)) <: NamedArrayPartition
    @test typeof(map(x -> x[1], x)) <: NamedArrayPartition
end

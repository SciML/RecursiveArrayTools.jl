using RecursiveArrayTools, RecursiveArrayToolsArrayPartitionAnyAll, Test

@testset "Optimized any" begin
    ap = ArrayPartition(collect(1:5), collect(6:10), collect(11:15))
    @test any(x -> x == 4, ap)
    @test any(x -> x == 15, ap)
    @test !any(x -> x == 17, ap)
    @test any(ap .> 10)
    @test !any(ap .> 20)
end

@testset "Optimized all" begin
    ap = ArrayPartition(ones(5), ones(5), ones(5))
    @test all(x -> x == 1.0, ap)
    @test !all(x -> x == 2.0, ap)
    @test all(ap .> 0)

    ap2 = ArrayPartition(ones(5), [1.0, 1.0, 0.0, 1.0, 1.0], ones(5))
    @test !all(x -> x == 1.0, ap2)
end

@testset "Matches AbstractArray default results" begin
    ap = ArrayPartition(rand(100), rand(100), rand(100))
    f = x -> x > 0.5

    # Results must match
    @test any(f, ap) == any(f, collect(ap))
    @test all(f, ap) == all(f, collect(ap))

    # Edge case: empty
    ap_empty = ArrayPartition(Float64[], Float64[])
    @test !any(x -> true, ap_empty)
    @test all(x -> true, ap_empty)
end

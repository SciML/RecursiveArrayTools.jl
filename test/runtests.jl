using RecursiveArrayTools
using Base.Test

tic()
@time @testset "Utils Tests" begin include("utils_test.jl") end
@time @testset "Partitions Tests" begin include("partitions_test.jl") end
@time @testset "VecOfArr Indexing Tests" begin include("basic_indexing.jl") end
@time @testset "VecOfArr Interface Tests" begin include("interface_tests.jl") end
@time @testset "StaticArrays Tests" begin include("copy_static_array_test.jl") end
toc()

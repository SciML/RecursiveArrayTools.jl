using OrdinaryDiffEq, ParameterizedFunctions,
      DiffEqBase, RecursiveArrayTools
using Base.Test

tic()
@time @testset "Utils Tests" begin include("utils_test.jl") end
@time @testset "Partitions Tests" begin include("partitions_test.jl") end
toc()
# Test the VectorOfArray code
include("basic_indexing.jl")
include("interface_tests.jl")

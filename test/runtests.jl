using OrdinaryDiffEq, ParameterizedFunctions,
      DiffEqBase, RecursiveArrayTools
using Base.Test

tic()
@time @testset "Utils Tests" begin include("utils_test.jl.jl") end
@time @testset "Partitions Tests" begin include("partitions_test.jl") end
toc()

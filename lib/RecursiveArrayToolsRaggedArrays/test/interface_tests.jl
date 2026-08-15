using RecursiveArrayTools
using RecursiveArrayToolsRaggedArrays
using SymbolicIndexingInterface
using Test

struct GenericRaggedVectorOfArray{T, N, A} <:
    RecursiveArrayTools.AbstractRaggedVectorOfArray{T, N, A}
    u::A
end

GenericRaggedVectorOfArray(u::A) where {A <: AbstractVector} =
    GenericRaggedVectorOfArray{eltype(eltype(u)), 2, A}(u)

struct GenericRaggedDiffEqArray{T, A} <:
    RecursiveArrayTools.AbstractRaggedDiffEqArray{T, 2, A}
    u::A
    t::Vector{Float64}
    p
    sys
    discretes
    interp
    dense::Bool
end

@testset "Generic AbstractRaggedVectorOfArray interface" begin
    source = GenericRaggedVectorOfArray([[1, 2], [3, 4]])
    destination = GenericRaggedVectorOfArray([[0, 0], [0, 0]])
    ragged = GenericRaggedVectorOfArray([[1, 2], [3, 4, 5]])

    @test size(source) == (2, 2)
    @test source[:, 1] == [1, 2]
    @test source[2, 2] == 4
    @test Array(source) == [1 3; 2 4]
    @test RecursiveArrayToolsRaggedArrays.narrays(source) == 2
    @test ragged[3, 2] == 5
    @test_throws DimensionMismatch Array(ragged)

    recursivecopy!(destination, source)
    @test destination.u == source.u
    recursivefill!(destination, 0)
    @test destination.u == [[0, 0], [0, 0]]
end

@testset "Generic AbstractRaggedDiffEqArray interface" begin
    solution = GenericRaggedDiffEqArray{Float64, Vector{Vector{Float64}}}(
        [[1.0, 2.0], [3.0, 4.0]], [0.0, 1.0], nothing, nothing, nothing, nothing, false
    )

    @test solution[:, 1] == [1.0, 2.0]
    @test SymbolicIndexingInterface.state_values(solution) === solution.u
    @test SymbolicIndexingInterface.current_time(solution) === solution.t
    @test SymbolicIndexingInterface.parameter_values(solution) === solution.p
    @test SymbolicIndexingInterface.symbolic_container(solution) === solution.sys
end

using Pkg
using RecursiveArrayTools
using Test

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = (Sys.iswindows() && haskey(ENV, "APPVEYOR"))

function activate_downstream_env()
    Pkg.activate("downstream")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

function activate_gpu_env()
    Pkg.activate("gpu")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin
    if !is_APPVEYOR && GROUP == "Core"
        @time @testset "Utils Tests" begin include("utils_test.jl") end
        @time @testset "Partitions Tests" begin include("partitions_test.jl") end
        @time @testset "VecOfArr Indexing Tests" begin include("basic_indexing.jl") end
        @time @testset "VecOfArr Interface Tests" begin include("interface_tests.jl") end
        @time @testset "StaticArrays Tests" begin include("copy_static_array_test.jl") end
        @time @testset "Linear Algebra Tests" begin include("linalg.jl") end
        @time @testset "Upstream Tests" begin include("upstream.jl") end
        @time @testset "Adjoint Tests" begin include("adjoints.jl") end
    end

    if !is_APPVEYOR && GROUP == "Downstream"
        activate_downstream_env()
        @time @testset "DiffEqArray Indexing Tests" begin include("downstream/symbol_indexing.jl") end
        @time @testset "Event Tests with ArrayPartition" begin include("downstream/downstream_events.jl") end
    end

    if !is_APPVEYOR && GROUP == "GPU"
        activate_gpu_env()
        @time @testset "VectorOfArray GPU" begin include("gpu/vectorofarray_gpu.jl") end
    end
end

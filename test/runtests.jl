using Pkg
using RecursiveArrayTools
using Test
using Aqua
using SafeTestsets

Aqua.test_all(RecursiveArrayTools, ambiguities = false)
@test_broken isempty(Test.detect_ambiguities(RecursiveArrayTools))
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
    if GROUP == "Core" || GROUP == "All"
        @time @safetestset "Utils Tests" begin include("utils_test.jl") end
        @time @safetestset "Partitions Tests" begin include("partitions_test.jl") end
        @time @safetestset "VecOfArr Indexing Tests" begin include("basic_indexing.jl") end
        @time @safetestset "SymbolicIndexingInterface API test" begin include("symbolic_indexing_interface_test.jl") end
        @time @safetestset "VecOfArr Interface Tests" begin include("interface_tests.jl") end
        @time @safetestset "Table traits" begin include("tabletraits.jl") end
        @time @safetestset "StaticArrays Tests" begin include("copy_static_array_test.jl") end
        @time @safetestset "Linear Algebra Tests" begin include("linalg.jl") end
        @time @safetestset "Upstream Tests" begin include("upstream.jl") end
        @time @safetestset "Adjoint Tests" begin include("adjoints.jl") end
        @time @safetestset "Measurement Tests" begin include("measurements.jl") end
    end

    if !is_APPVEYOR && GROUP == "Downstream"
        activate_downstream_env()
        @time @safetestset "DiffEqArray Indexing Tests" begin include("downstream/symbol_indexing.jl") end
        @time @safetestset "Event Tests with ArrayPartition" begin include("downstream/downstream_events.jl") end
        @time @safetestset "TrackerExt" begin include("downstream/TrackerExt.jl") end
    end

    if !is_APPVEYOR && GROUP == "GPU"
        activate_gpu_env()
        @time @safetestset "VectorOfArray GPU" begin include("gpu/vectorofarray_gpu.jl") end
    end
end

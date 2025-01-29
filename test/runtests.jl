using Pkg
using RecursiveArrayTools
using Test
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

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
        @time @safetestset "Quality Assurance" include("qa.jl")
        @time @safetestset "Utils Tests" include("utils_test.jl")
        @time @safetestset "NamedArrayPartition Tests" include("named_array_partition_tests.jl")
        @time @safetestset "Partitions Tests" include("partitions_test.jl")
        @time @safetestset "VecOfArr Indexing Tests" include("basic_indexing.jl")
        @time @safetestset "VecOfArr Interface Tests" include("interface_tests.jl")
        @time @safetestset "Table traits" include("tabletraits.jl")
        @time @safetestset "StaticArrays Tests" include("copy_static_array_test.jl")
        @time @safetestset "Linear Algebra Tests" include("linalg.jl")
        @time @safetestset "Adjoint Tests" include("adjoints.jl")
        @time @safetestset "Measurement Tests" include("measurements.jl")
    end

    if GROUP == "SymbolicIndexingInterface" || GROUP == "All"
        @time @safetestset "SymbolicIndexingInterface API test" include("symbolic_indexing_interface_test.jl")
    end

    if GROUP == "Downstream"
        activate_downstream_env()
        @time @safetestset "ODE Solve Tests" include("downstream/odesolve.jl")
        @time @safetestset "Event Tests with ArrayPartition" include("downstream/downstream_events.jl")
        @time @safetestset "Measurements and Units" include("downstream/measurements_and_units.jl")
        @time @safetestset "TrackerExt" include("downstream/TrackerExt.jl")
    end

    if GROUP == "SymbolicIndexingInterface" || GROUP == "Downstream"
        if GROUP == "SymbolicIndexingInterface"
            activate_downstream_env()
        end
        @time @safetestset "DiffEqArray Indexing Tests" include("downstream/symbol_indexing.jl")
    end

    if GROUP == "GPU"
        activate_gpu_env()
        @time @safetestset "VectorOfArray GPU" include("gpu/vectorofarray_gpu.jl")
        @time @safetestset "ArrayPartition GPU" include("gpu/arraypartition_gpu.jl")
    end
end

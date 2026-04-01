using Pkg
# Install the ShorthandConstructors subpackage for tests that need VA[...]/AP[...] syntax
Pkg.develop(PackageSpec(
    path = joinpath(dirname(@__DIR__), "lib", "RecursiveArrayToolsShorthandConstructors")))
using RecursiveArrayTools
using Test
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

function activate_downstream_env()
    Pkg.activate("downstream")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    return Pkg.instantiate()
end

function activate_gpu_env()
    Pkg.activate("gpu")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.develop(PackageSpec(
        path = joinpath(dirname(@__DIR__), "lib", "RecursiveArrayToolsArrayPartitionAnyAll")))
    return Pkg.instantiate()
end

function activate_nopre_env()
    Pkg.activate("nopre")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    return Pkg.instantiate()
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

    if GROUP == "Subpackages" || GROUP == "All"
        # Test that loading RecursiveArrayToolsArrayPartitionAnyAll overrides any/all
        Pkg.develop(PackageSpec(
            path = joinpath(dirname(@__DIR__), "lib", "RecursiveArrayToolsArrayPartitionAnyAll")))
        @time @safetestset "ArrayPartition AnyAll Subpackage" begin
            using RecursiveArrayTools, RecursiveArrayToolsArrayPartitionAnyAll, Test
            # Verify optimized methods are active
            m_any = which(any, Tuple{Function, ArrayPartition})
            m_all = which(all, Tuple{Function, ArrayPartition})
            @test occursin("ArrayPartitionAnyAll", string(m_any.module))
            @test occursin("ArrayPartitionAnyAll", string(m_all.module))
            # Verify correctness
            @test  any(isnan, ArrayPartition([NaN], [1.0]))
            @test !any(isnan, ArrayPartition([1.0], [2.0]))
            @test  all(isnan, ArrayPartition([NaN], [NaN]))
            @test !all(isnan, ArrayPartition([NaN], [1.0]))
        end
    end

    if GROUP == "Downstream"
        activate_downstream_env()
        @time @safetestset "ODE Solve Tests" include("downstream/odesolve.jl")
        @time @safetestset "Event Tests with ArrayPartition" include("downstream/downstream_events.jl")
        @time @safetestset "Measurements and Units" include("downstream/measurements_and_units.jl")
        @time @safetestset "TrackerExt" include("downstream/TrackerExt.jl")
        # TODO: re-enable after SciMLBase compat bump for RAT v4 (SciML/SciMLBase.jl#1297)
        # @time @safetestset "Downstream Adjoint Tests" include("downstream/adjoints.jl")
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

    if GROUP == "nopre"
        activate_nopre_env()
        @time @safetestset "JET Tests" include("jet_tests.jl")
    end
end

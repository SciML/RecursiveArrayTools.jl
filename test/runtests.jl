using RecursiveArrayTools
using SafeTestsets
using SciMLTesting
using Pkg

run_tests(;
    core = function ()
        # The root Core tests use the VA[...]/AP[...] shorthand syntax, which lives
        # in the RecursiveArrayToolsShorthandConstructors sublibrary; develop it into
        # the active main env before running them.
        Pkg.develop(
            Pkg.PackageSpec(
                path = joinpath(dirname(@__DIR__), "lib", "RecursiveArrayToolsShorthandConstructors")
            )
        )
        @time @safetestset "Utils Tests" include("Core/utils_test.jl")
        @time @safetestset "NamedArrayPartition Tests" include("Core/named_array_partition_tests.jl")
        @time @safetestset "Partitions Tests" include("Core/partitions_test.jl")
        @time @safetestset "Partitions and StaticArrays Tests" include("Core/partitions_and_static_arrays.jl")
        @time @safetestset "VecOfArr Indexing Tests" include("Core/basic_indexing.jl")
        @time @safetestset "VecOfArr Interface Tests" include("Core/interface_tests.jl")
        @time @safetestset "Table traits" include("Core/tabletraits.jl")
        @time @safetestset "StaticArrays Tests" include("Core/copy_static_array_test.jl")
        @time @safetestset "Linear Algebra Tests" include("Core/linalg.jl")
        return @time @safetestset "Measurement Tests" include("Core/measurements.jl")
    end,
    groups = Dict(
        # The SII API safetestset runs in the main test env. It is the in-main-env
        # half shared between the "All" group (Core + this) and the
        # "SymbolicIndexingInterface" umbrella.
        "SII_Main" => function ()
            return @time @safetestset "SymbolicIndexingInterface API test" include("SymbolicIndexingInterface/symbolic_indexing_interface_test.jl")
        end,
        # The DiffEqArray symbol-indexing test runs in the Downstream env. It is the
        # second half of both the "SymbolicIndexingInterface" and "Downstream" paths.
        "SII_Downstream" => (;
            env = joinpath(@__DIR__, "Downstream"),
            body = function ()
                return @time @safetestset "DiffEqArray Indexing Tests" include("Downstream/symbol_indexing.jl")
            end,
        ),
        "Downstream" => (;
            env = joinpath(@__DIR__, "Downstream"),
            body = function ()
                @time @safetestset "ODE Solve Tests" include("Downstream/odesolve.jl")
                @time @safetestset "Event Tests with ArrayPartition" include("Downstream/downstream_events.jl")
                @time @safetestset "Measurements and Units" include("Downstream/measurements_and_units.jl")
                @time @safetestset "TrackerExt" include("Downstream/TrackerExt.jl")
                # TODO: re-enable after SciMLBase compat bump for RAT v4 (SciML/SciMLBase.jl#1297)
                # @time @safetestset "Downstream Adjoint Tests" include("Downstream/adjoints.jl")
                return @time @safetestset "DiffEqArray Indexing Tests" include("Downstream/symbol_indexing.jl")
            end,
        ),
        "GPU" => (;
            env = joinpath(@__DIR__, "GPU"),
            parent = [
                dirname(@__DIR__),
                joinpath(dirname(@__DIR__), "lib", "RecursiveArrayToolsArrayPartitionAnyAll"),
            ],
            body = function ()
                @time @safetestset "VectorOfArray GPU" include("GPU/vectorofarray_gpu.jl")
                return @time @safetestset "ArrayPartition GPU" include("GPU/arraypartition_gpu.jl")
            end,
        ),
        "NoPre" => (;
            env = joinpath(@__DIR__, "NoPre"),
            body = function ()
                return @time @safetestset "JET Tests" include("NoPre/jet_tests.jl")
            end,
        ),
        # AD lives in its own env (test/AD/Project.toml) so its heavy AD-backend
        # deps (Mooncake/Zygote/ForwardDiff/ReverseDiff) stay out of the root test
        # target. It is deliberately absent from `all` and not run in the Downgrade
        # lane: Mooncake 0.5 requires Julia >= 1.10.8, which the downgrade resolver
        # cannot satisfy when it minimizes Julia to the 1.10.0 LTS floor.
        "AD" => (;
            env = joinpath(@__DIR__, "AD"),
            body = function ()
                @time @safetestset "Adjoint Tests" include("AD/adjoints.jl")
                return @time @safetestset "Mooncake Tests" include("AD/mooncake.jl")
            end,
        ),
    ),
    qa = (;
        env = joinpath(@__DIR__, "QA"),
        body = joinpath(@__DIR__, "QA", "qa.jl"),
    ),
    # "All" runs exactly Core plus the in-main-env SII API safetestset (matching the
    # original `GROUP == "Core" || "All"` and `GROUP == "SymbolicIndexingInterface"
    # || "All"` branches). It deliberately excludes QA, Downstream, GPU and NoPre.
    all = ["Core", "SII_Main"],
    umbrellas = Dict(
        # GROUP=SymbolicIndexingInterface runs the SII API safetestset (main env) and
        # then the DiffEqArray symbol-indexing test (Downstream env), in that order.
        "SymbolicIndexingInterface" => ["SII_Main", "SII_Downstream"],
    ),
    lib_dir = joinpath(dirname(@__DIR__), "lib"),
    # Root reads GROUP to pick the sublibrary, but the sublibraries read
    # RECURSIVEARRAYTOOLS_TEST_GROUP for their own sub-group; hand it off via that
    # variable (matching the original withenv around Pkg.test(sublib)).
    sublib_env = "RECURSIVEARRAYTOOLS_TEST_GROUP",
)

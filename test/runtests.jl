using Pkg
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
    Pkg.develop(
        PackageSpec(
            path = joinpath(dirname(@__DIR__), "lib", "RecursiveArrayToolsArrayPartitionAnyAll")
        )
    )
    return Pkg.instantiate()
end

function activate_nopre_env()
    Pkg.activate("nopre")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    return Pkg.instantiate()
end

function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    # On Julia < 1.11, the [sources] section in the qa Project.toml is not
    # honored, so Pkg.develop the umbrella root path explicitly.
    if VERSION < v"1.11.0-DEV.0"
        Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    end
    return Pkg.instantiate()
end

@time begin
    # Detect sublibrary test groups.
    # GROUP can be a bare sublibrary name (Core test group) or
    # "{sublibrary}_{TEST_GROUP}" for any custom group (e.g., QA, GPU, etc.).
    # Sublibraries declare their groups in test/test_groups.toml.
    lib_dir = joinpath(dirname(@__DIR__), "lib")

    # Check if GROUP matches a sublibrary, possibly with a _SUFFIX for the test group.
    # Scan underscores right-to-left to find the longest matching sublibrary prefix.
    function _detect_sublibrary_group(group, lib_dir)
        isdir(joinpath(lib_dir, group)) && return (group, "Core")
        for i in length(group):-1:1
            if group[i] == '_' && isdir(joinpath(lib_dir, group[1:(i - 1)]))
                return (group[1:(i - 1)], group[(i + 1):end])
            end
        end
        return (group, "Core")
    end
    base_group, test_group = _detect_sublibrary_group(GROUP, lib_dir)

    if isdir(joinpath(lib_dir, base_group))
        Pkg.activate(joinpath(lib_dir, base_group))
        # On Julia < 1.11, the [sources] section in Project.toml is not supported.
        # Manually Pkg.develop local path dependencies so CI tests the PR branch code.
        # The sublibraries depend on the umbrella root via [sources] (../..); develop
        # those local paths, skipping the active project itself.
        if VERSION < v"1.11.0-DEV.0"
            developed = Set{String}()
            push!(developed, normpath(joinpath(lib_dir, base_group)))
            specs = Pkg.PackageSpec[]
            queue = [joinpath(lib_dir, base_group)]
            while !isempty(queue)
                pkg_dir = popfirst!(queue)
                toml_path = joinpath(pkg_dir, "Project.toml")
                isfile(toml_path) || continue
                toml = Pkg.TOML.parsefile(toml_path)
                if haskey(toml, "sources")
                    for (dep_name, source_spec) in toml["sources"]
                        if source_spec isa Dict && haskey(source_spec, "path")
                            dep_path = normpath(joinpath(pkg_dir, source_spec["path"]))
                            if isdir(dep_path) && !(dep_path in developed)
                                push!(developed, dep_path)
                                push!(specs, Pkg.PackageSpec(path = dep_path))
                                push!(queue, dep_path)
                            end
                        end
                    end
                end
            end
            isempty(specs) || Pkg.develop(specs)
        end
        withenv("RECURSIVEARRAYTOOLS_TEST_GROUP" => test_group) do
            Pkg.test(base_group, julia_args = ["--check-bounds=auto"], force_latest_compatible_version = false, allow_reresolve = true)
        end
    else
        # Root package's own test groups.
        if GROUP == "Core" || GROUP == "All"
            # QA (Aqua) runs in its own isolated environment so the QA tooling
            # is not part of the main test dependency set. Restore the original
            # test project afterwards so the functional Core tests below run in
            # the full main test environment.
            core_test_project = Base.active_project()
            activate_qa_env()
            @time @safetestset "Quality Assurance" include("qa/qa.jl")
            Pkg.activate(core_test_project)

            # The root Core tests use the VA[...]/AP[...] shorthand syntax, which
            # lives in the RecursiveArrayToolsShorthandConstructors sublibrary.
            Pkg.develop(
                PackageSpec(
                    path = joinpath(dirname(@__DIR__), "lib", "RecursiveArrayToolsShorthandConstructors")
                )
            )
            @time @safetestset "Utils Tests" include("utils_test.jl")
            @time @safetestset "NamedArrayPartition Tests" include("named_array_partition_tests.jl")
            @time @safetestset "Partitions Tests" include("partitions_test.jl")
            @time @safetestset "VecOfArr Indexing Tests" include("basic_indexing.jl")
            @time @safetestset "VecOfArr Interface Tests" include("interface_tests.jl")
            @time @safetestset "Table traits" include("tabletraits.jl")
            @time @safetestset "StaticArrays Tests" include("copy_static_array_test.jl")
            @time @safetestset "Linear Algebra Tests" include("linalg.jl")
            @time @safetestset "Adjoint Tests" include("adjoints.jl")
            @time @safetestset "Mooncake Tests" include("mooncake.jl")
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

        if GROUP == "NoPre"
            activate_nopre_env()
            @time @safetestset "JET Tests" include("jet_tests.jl")
        end
    end
end

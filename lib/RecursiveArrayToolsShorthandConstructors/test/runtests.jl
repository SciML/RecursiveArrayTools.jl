using RecursiveArrayTools, RecursiveArrayToolsShorthandConstructors, Test
using Pkg

const TEST_GROUP = get(ENV, "RECURSIVEARRAYTOOLS_TEST_GROUP", "Core")

function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    # On Julia < 1.11, the [sources] section in the qa Project.toml is not
    # honored, so Pkg.develop the local paths explicitly.
    if VERSION < v"1.11.0-DEV.0"
        Pkg.develop(
            [
                PackageSpec(path = dirname(@__DIR__)),
                PackageSpec(path = normpath(joinpath(dirname(@__DIR__), "..", ".."))),
            ]
        )
    end
    return Pkg.instantiate()
end

if TEST_GROUP == "QA"
    activate_qa_env()
    include("qa/qa.jl")
end

if TEST_GROUP == "Core" || TEST_GROUP == "All"
    @testset "VA[...] shorthand" begin
        recs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        testva = VA[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        @test testva isa VectorOfArray
        @test testva.u == recs
        @test Array(testva) == [1 4 7; 2 5 8; 3 6 9]

        # Nesting
        nested = VA[
            fill(1, 2, 3),
            VA[3ones(3), zeros(3)],
        ]
        @test nested isa VectorOfArray
        @test nested.u[1] == fill(1, 2, 3)
        @test nested.u[2] isa VectorOfArray
    end

    @testset "AP[...] shorthand" begin
        x = AP[1:5, 1:6]
        @test x isa ArrayPartition
        @test length(x) == 11
        @test x[1:5] == 1:5
        @test x[6:11] == 1:6

        @test length(AP[]) == 0
        @test isempty(AP[])
    end
end

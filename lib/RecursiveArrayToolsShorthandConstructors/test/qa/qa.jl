using RecursiveArrayToolsShorthandConstructors, Aqua, JET, Test

const RATSC = RecursiveArrayToolsShorthandConstructors

@testset "QA" begin
    @testset "Aqua" begin
        # `getindex(::Type{VA}, ...)` / `getindex(::Type{AP}, ...)` extend Base on
        # RecursiveArrayTools-owned types, so they are intentional (owned) methods,
        # not piracy.
        Aqua.test_all(RATSC; piracies = (; treat_as_own = [RATSC.VA, RATSC.AP]))
    end
    @testset "JET" begin
        JET.test_package(RecursiveArrayToolsShorthandConstructors; target_defined_modules = true)
    end
end

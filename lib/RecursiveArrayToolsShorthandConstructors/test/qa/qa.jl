using RecursiveArrayToolsShorthandConstructors, Aqua, JET, Test

@testset "QA" begin
    @testset "Aqua" begin
        Aqua.test_all(RecursiveArrayToolsShorthandConstructors)
    end
    @testset "JET" begin
        JET.test_package(RecursiveArrayToolsShorthandConstructors; target_defined_modules = true)
    end
end

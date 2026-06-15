using RecursiveArrayToolsRaggedArrays, Aqua, JET, Test

@testset "QA" begin
    @testset "Aqua" begin
        Aqua.test_all(RecursiveArrayToolsRaggedArrays)
    end
    @testset "JET" begin
        JET.test_package(RecursiveArrayToolsRaggedArrays; target_defined_modules = true)
    end
end

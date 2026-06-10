using RecursiveArrayToolsArrayPartitionAnyAll, Aqua, JET, Test

@testset "QA" begin
    @testset "Aqua" begin
        Aqua.test_all(RecursiveArrayToolsArrayPartitionAnyAll)
    end
    @testset "JET" begin
        JET.test_package(RecursiveArrayToolsArrayPartitionAnyAll; target_defined_modules = true)
    end
end

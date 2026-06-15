using RecursiveArrayToolsArrayPartitionAnyAll, Aqua, JET, Test

const RATAPAA = RecursiveArrayToolsArrayPartitionAnyAll

@testset "QA" begin
    @testset "Aqua" begin
        # `any`/`all` are extended on the RecursiveArrayTools-owned `ArrayPartition`
        # type, so they are intentional (owned) methods, not piracy.
        Aqua.test_all(RATAPAA; piracies = (; treat_as_own = [RATAPAA.ArrayPartition]))
    end
    @testset "JET" begin
        JET.test_package(RecursiveArrayToolsArrayPartitionAnyAll; target_defined_modules = true)
    end
end

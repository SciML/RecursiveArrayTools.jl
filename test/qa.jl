using RecursiveArrayTools, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(RecursiveArrayTools)
    ambs = Aqua.detect_ambiguities(RecursiveArrayTools; recursive = true)
    @warn "Number of method ambiguities: $(length(ambs))"
    @test length(ambs) <= 2
    Aqua.test_deps_compat(RecursiveArrayTools)
    Aqua.test_piracies(RecursiveArrayTools)
    Aqua.test_project_extras(RecursiveArrayTools)
    Aqua.test_stale_deps(RecursiveArrayTools)
    Aqua.test_unbound_args(RecursiveArrayTools)
    Aqua.test_undefined_exports(RecursiveArrayTools)
end

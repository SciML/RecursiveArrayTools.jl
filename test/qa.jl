using RecursiveArrayTools, Aqua, Pkg

# yes this is horrible, we'll fix it when Pkg or Base provides a decent API
manifest = Pkg.Types.EnvCache().manifest
# these are good sentinels to test whether someone has added a heavy SciML package to the test deps
if haskey(manifest.deps, "NonlinearSolveBase") || haskey(manifest.deps, "DiffEqBase")
    error("Don't put Downstream Packages in non Downstream CI")
end

@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(RecursiveArrayTools)
    Aqua.test_ambiguities(RecursiveArrayTools; recursive = false, broken = true)
    Aqua.test_deps_compat(RecursiveArrayTools)
    Aqua.test_piracies(RecursiveArrayTools)
    Aqua.test_project_extras(RecursiveArrayTools)
    Aqua.test_stale_deps(RecursiveArrayTools)
    Aqua.test_unbound_args(RecursiveArrayTools)
    Aqua.test_undefined_exports(RecursiveArrayTools)
end

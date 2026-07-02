using SciMLTesting, RecursiveArrayTools, Test, Pkg

# yes this is horrible, we'll fix it when Pkg or Base provides a decent API
manifest = Pkg.Types.EnvCache().manifest
# these are good sentinels to test whether someone has added a heavy SciML package to the test deps
if haskey(manifest.deps, "NonlinearSolveBase") || haskey(manifest.deps, "DiffEqBase")
    error("Don't put Downstream Packages in non Downstream CI")
end

run_qa(
    RecursiveArrayTools;
    explicit_imports = true,
    # Method-table ambiguities tracked in
    # https://github.com/SciML/RecursiveArrayTools.jl/issues/326
    aqua_broken = (:ambiguities,),
    ei_kwargs = (;
        # Non-public names legitimately qualified/imported from upstream packages
        # (Base, Base.Broadcast, LinearAlgebra, StaticArraysCore, ArrayInterface,
        # Adapt, SymbolicIndexingInterface). Not RecursiveArrayTools' to make public.
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@propagate_inbounds"), :AbstractArrayStyle, :AllVariables,
                :ArrayStyle, :BroadcastStyle, :Broadcasted, :DefaultArrayStyle, :OneTo,
                :QRCompactWY, :SolvedVariables, :StaticArray, :StaticVecOrMat,
                :SymbolicTypeTrait, :_InitialValue, :_ipiv_rows!, :_maybe_reshape,
                :_swap_rows!, :_unsafe_getindex, :_unsafe_getindex!, :adapt_structure,
                :add_sum, :flatten, :front, :index_shape, :ismutable, :issingular,
                :promote_op, :restructure, :result_style, :similar_type, :tail,
                :unalias, :zeromatrix,
            ),
        ),
    ),
    # Whole-module `using` exposes many names implicitly; explicit-import refactor
    # tracked in https://github.com/SciML/RecursiveArrayTools.jl/issues/619
    ei_broken = (:no_implicit_imports,),
)

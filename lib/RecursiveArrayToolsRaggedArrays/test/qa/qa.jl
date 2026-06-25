using SciMLTesting, RecursiveArrayToolsRaggedArrays, JET, Test

run_qa(
    RecursiveArrayToolsRaggedArrays;
    explicit_imports = true,
    # The arithmetic/array methods (`*`, `+`, `Array`, `copyto!`, ...) are defined on
    # the RecursiveArrayTools-owned `AbstractRaggedVectorOfArray` /
    # `AbstractRaggedDiffEqArray` abstract types, so they are intentional (owned)
    # methods, not piracy.
    aqua_kwargs = (;
        piracies = (;
            treat_as_own = [
                RecursiveArrayToolsRaggedArrays.AbstractRaggedVectorOfArray,
                RecursiveArrayToolsRaggedArrays.AbstractRaggedDiffEqArray,
            ],
        ),
    ),
    jet_kwargs = (; target_defined_modules = true),
    ei_kwargs = (;
        # `AbstractRaggedVectorOfArray`/`AbstractRaggedDiffEqArray` are
        # RecursiveArrayTools-owned abstract types this subpackage subtypes; they are
        # not (yet) declared public in RecursiveArrayTools.
        all_explicit_imports_are_public = (;
            ignore = (:AbstractRaggedVectorOfArray, :AbstractRaggedDiffEqArray),
        ),
        # Non-public names legitimately qualified/imported from upstream packages
        # (Base, Base.Broadcast, StaticArraysCore, ArrayInterface, Adapt,
        # SymbolicIndexingInterface). Not this subpackage's to make public.
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@propagate_inbounds"), :AbstractArrayStyle, :AllVariables,
                :Broadcasted, :DefaultArrayStyle, :HasLength, :IteratorSize, :OneTo,
                :Slice, :SolvedVariables, :StaticVecOrMat, :SymbolicTypeTrait,
                :adapt_structure, :add_sum, :broadcastable, :check_parent_index_match,
                :ensure_indexable, :flatten, :front, :index_dimsum, :ismutable,
                :issingular, :maybeview, :mul_prod, :similar_type, :tail, :typename,
                :unalias, :viewindexing,
            ),
        ),
    ),
    # Whole-module `using` exposes many names implicitly; explicit-import refactor
    # tracked in https://github.com/SciML/RecursiveArrayTools.jl/issues/619
    ei_broken = (:no_implicit_imports,),
)

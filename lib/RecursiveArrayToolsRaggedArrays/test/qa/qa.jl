using SciMLTesting, RecursiveArrayToolsRaggedArrays, JET, Test

run_qa(
    RecursiveArrayToolsRaggedArrays;
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
    jet_kwargs = (; target_modules = (RecursiveArrayToolsRaggedArrays,)),
    # Pre-existing JET typo-mode finding (reproduces byte-identically on master):
    # the `copyto!`/`fill!`/broadcast immutable-element branches call
    # `StaticArraysCore.similar_type(dest.u[i])`, but `dest.u[i]` infers as `::Any`
    # because the abstract `AbstractRaggedVectorOfArray` `.u` field is untyped, so
    # `similar_type(::Any)` has no matching method. Tracked (with the real fix —
    # tightening the `.u` type / guarding the immutable branch) in
    # https://github.com/SciML/RecursiveArrayTools.jl/issues/620. JET 0.9 on Julia
    # 1.10 does not report this finding, so keep that stricter lane unbroken.
    jet_broken = VERSION >= v"1.11",
    ei_kwargs = (;
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
)

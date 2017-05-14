__precompile__()

module RecursiveArrayTools

  using Iterators, Compat, Juno, RecipesBase, StaticArrays

  @compat abstract type AbstractVectorOfArray{T, N} <: AbstractArray{T, N} end

  include("utils.jl")
  include("vector_of_array.jl")
  include("array_partition.jl")

  export VectorOfArray, DiffEqArray, AbstractVectorOfArray, vecarr_to_arr,
         vecarr_to_vectors, tuples

  export recursivecopy, recursivecopy!, vecvecapply, copyat_or_push!,
         vecvec_to_mat, recursive_one, recursive_mean, recursive_eltype

  export ArrayPartition

end # module

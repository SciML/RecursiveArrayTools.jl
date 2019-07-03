__precompile__()

module RecursiveArrayTools

  using Requires, RecipesBase, StaticArrays, Statistics,
        ArrayInterface

  abstract type AbstractVectorOfArray{T, N} <: AbstractArray{T, N} end
  abstract type AbstractDiffEqArray{T, N} <: AbstractVectorOfArray{T, N} end

  include("utils.jl")
  include("vector_of_array.jl")
  include("array_partition.jl")
  include("init.jl")

  export VectorOfArray, DiffEqArray, AbstractVectorOfArray, AbstractDiffEqArray,
         vecarr_to_arr, vecarr_to_vectors, tuples

  export recursivecopy, recursivecopy!, vecvecapply, copyat_or_push!,
         vecvec_to_mat, recursive_one, recursive_mean, recursive_bottom_eltype,
         recursive_unitless_bottom_eltype, recursive_unitless_eltype

  export ArrayPartition


end # module

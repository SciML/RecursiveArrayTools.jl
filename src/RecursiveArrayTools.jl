__precompile__()

module RecursiveArrayTools

  using Iterators

  include("utils.jl")
  include("vector_of_array.jl")
  include("array_partition.jl")

  export VectorOfArray, AbstractVectorOfArray, vecarr_to_arr

  export recursivecopy!, vecvecapply, copyat_or_push!, vecvec_to_mat, recursive_one,
         recursive_mean

  export ArrayPartition

end # module

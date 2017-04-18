__precompile__()

module RecursiveArrayTools

  using Iterators

  include("vector_of_array.jl")
  export VectorOfArray, vecarr_to_arr

  import Base: mean

  include("utils.jl")
  include("array_partition.jl")

  export recursivecopy!, vecvecapply, copyat_or_push!, vecvec_to_mat, recursive_one,mean

  export ArrayPartition

end # module

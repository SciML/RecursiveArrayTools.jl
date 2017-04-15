__precompile__()

module RecursiveArrayTools

  using Iterators

  import Base: mean

  include("utils.jl")
  include("array_partition.jl")

  export recursivecopy!, vecvecapply, copyat_or_push!, vecvec_to_mat, recursive_one,mean

  export ArrayPartition

end # module

__precompile__()
"""
$(DocStringExtensions.README)
"""
module RecursiveArrayTools

using DocStringExtensions
using Requires, RecipesBase, StaticArrays, Statistics,
      ArrayInterface, LinearAlgebra

import ChainRulesCore
import ChainRulesCore: NoTangent
import ZygoteRules, Adapt

using FillArrays

abstract type AbstractVectorOfArray{T, N, A} <: AbstractArray{T, N} end
abstract type AbstractDiffEqArray{T, N, A} <: AbstractVectorOfArray{T, N, A} end

include("utils.jl")
include("vector_of_array.jl")
include("array_partition.jl")
include("zygote.jl")

Base.show(io::IO, x::Union{ArrayPartition,AbstractVectorOfArray}) = invoke(show, Tuple{typeof(io), Any}, io, x)

export VectorOfArray, DiffEqArray, AbstractVectorOfArray, AbstractDiffEqArray,
       AllObserved, vecarr_to_arr, vecarr_to_vectors, tuples

export recursivecopy, recursivecopy!, recursivefill!, vecvecapply, copyat_or_push!,
       vecvec_to_mat, recursive_one, recursive_mean, recursive_bottom_eltype,
       recursive_unitless_bottom_eltype, recursive_unitless_eltype


export ArrayPartition


end # module

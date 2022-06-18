__precompile__()
"""
$(DocStringExtensions.README)
"""
module RecursiveArrayTools

using DocStringExtensions
using RecipesBase, StaticArrays, Statistics,
      ArrayInterfaceCore, LinearAlgebra

import ChainRulesCore
import ChainRulesCore: NoTangent
import Adapt

# Required for the downstream_events.jl test
# Since `ismutable` on an ArrayPartition needs
# to know static arrays are not mutable
import ArrayInterfaceStaticArrays

abstract type AbstractVectorOfArray{T, N, A} <: AbstractArray{T, N} end
abstract type AbstractDiffEqArray{T, N, A} <: AbstractVectorOfArray{T, N, A} end

include("utils.jl")
include("vector_of_array.jl")
include("array_partition.jl")
include("zygote.jl")

Base.show(io::IO, x::Union{ArrayPartition,AbstractVectorOfArray}) = invoke(show, Tuple{typeof(io), Any}, io, x)

import GPUArrays
Base.convert(T::Type{<:GPUArrays.AbstractGPUArray}, VA::AbstractVectorOfArray) = T(VA)
ChainRulesCore.rrule(T::Type{<:GPUArrays.AbstractGPUArray}, xs::AbstractVectorOfArray) = T(xs), ȳ -> (NoTangent(),ȳ)

export VectorOfArray, DiffEqArray, AbstractVectorOfArray, AbstractDiffEqArray,
       AllObserved, vecarr_to_arr, vecarr_to_vectors, tuples

export recursivecopy, recursivecopy!, recursivefill!, vecvecapply, copyat_or_push!,
       vecvec_to_mat, recursive_one, recursive_mean, recursive_bottom_eltype,
       recursive_unitless_bottom_eltype, recursive_unitless_eltype


export ArrayPartition


end # module

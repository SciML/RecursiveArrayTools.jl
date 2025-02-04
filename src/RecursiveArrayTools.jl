__precompile__()
"""
$(DocStringExtensions.README)
"""
module RecursiveArrayTools

using DocStringExtensions
using RecipesBase, StaticArraysCore, Statistics,
      ArrayInterface, LinearAlgebra
using SymbolicIndexingInterface

import Adapt

import Tables, IteratorInterfaceExtensions
#
abstract type AbstractVectorOfArray{T, N, A} end
abstract type AbstractDiffEqArray{T, N, A} <: AbstractVectorOfArray{T, N, A} end

include("utils.jl")
include("vector_of_array.jl")
include("tabletraits.jl")
include("array_partition.jl")
include("named_array_partition.jl")

function Base.show(io::IO, x::Union{ArrayPartition, AbstractVectorOfArray})
    invoke(show, Tuple{typeof(io), Any}, io, x)
end

import GPUArraysCore
Base.convert(T::Type{<:GPUArraysCore.AnyGPUArray}, VA::AbstractVectorOfArray) = stack(VA.u)
(T::Type{<:GPUArraysCore.AnyGPUArray})(VA::AbstractVectorOfArray) = T(Array(VA))

export VectorOfArray, DiffEqArray, AbstractVectorOfArray, AbstractDiffEqArray,
       AllObserved, vecarr_to_vectors, tuples

export recursivecopy, recursivecopy!, recursivefill!, vecvecapply, copyat_or_push!,
       vecvec_to_mat, recursive_one, recursive_mean, recursive_bottom_eltype,
       recursive_unitless_bottom_eltype, recursive_unitless_eltype

export ArrayPartition, NamedArrayPartition

end # module

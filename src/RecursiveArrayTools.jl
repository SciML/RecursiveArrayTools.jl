__precompile__()
"""
$(DocStringExtensions.README)
"""
module RecursiveArrayTools

using DocStringExtensions
using RecipesBase, StaticArraysCore, Statistics,
      ArrayInterfaceCore, LinearAlgebra
using SymbolicIndexingInterface

import ChainRulesCore
import ChainRulesCore: NoTangent
import ZygoteRules, Adapt

# Required for the downstream_events.jl test
# Since `ismutable` on an ArrayPartition needs
# to know static arrays are not mutable
import ArrayInterfaceStaticArraysCore

using FillArrays

import Tables, IteratorInterfaceExtensions

abstract type AbstractVectorOfArray{T, N, A} <: AbstractArray{T, N} end
abstract type AbstractDiffEqArray{T, N, A} <: AbstractVectorOfArray{T, N, A} end

include("utils.jl")
include("vector_of_array.jl")
include("tabletraits.jl")
include("array_partition.jl")
include("zygote.jl")

function Base.show(io::IO, x::Union{ArrayPartition, AbstractVectorOfArray})
    invoke(show, Tuple{typeof(io), Any}, io, x)
end

import GPUArraysCore
Base.convert(T::Type{<:GPUArraysCore.AbstractGPUArray}, VA::AbstractVectorOfArray) = T(VA)
function ChainRulesCore.rrule(T::Type{<:GPUArraysCore.AbstractGPUArray},
                              xs::AbstractVectorOfArray)
    T(xs), ȳ -> (NoTangent(), ȳ)
end

@static if !isdefined(Base, :get_extension)
    import Requires
    function __init__()
            Requires.@require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin include("../ext/RecursiveArrayToolsTrackerExt.jl") end
    end
end

export VectorOfArray, DiffEqArray, AbstractVectorOfArray, AbstractDiffEqArray,
       AllObserved, vecarr_to_vectors, tuples

export recursivecopy, recursivecopy!, recursivefill!, vecvecapply, copyat_or_push!,
       vecvec_to_mat, recursive_one, recursive_mean, recursive_bottom_eltype,
       recursive_unitless_bottom_eltype, recursive_unitless_eltype

export ArrayPartition

end # module

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

import SymbolicsBase: AllObserved, issymbollike
# Re-export from SymbolicsBase
export AllObserved

abstract type AbstractVectorOfArray{T, N, A} <: AbstractArray{T, N} end
abstract type AbstractDiffEqArray{T, N, A} <: AbstractVectorOfArray{T, N, A} end

include("utils.jl")
include("vector_of_array.jl")
include("tabletraits.jl")
include("array_partition.jl")

function Base.show(io::IO, x::Union{ArrayPartition, AbstractVectorOfArray})
    invoke(show, Tuple{typeof(io), Any}, io, x)
end

import GPUArraysCore
Base.convert(T::Type{<:GPUArraysCore.AbstractGPUArray}, VA::AbstractVectorOfArray) = T(VA)

import Requires
@static if !isdefined(Base, :get_extension)
    function __init__()
        Requires.@require Measurements="eff96d63-e80a-5855-80a2-b1b0885c5ab7" begin include("../ext/RecursiveArrayToolsMeasurementsExt.jl") end
        Requires.@require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin include("../ext/RecursiveArrayToolsTrackerExt.jl") end
        Requires.@require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin include("../ext/RecursiveArrayToolsZygoteExt.jl") end
    end
end

export VectorOfArray, DiffEqArray, AbstractVectorOfArray, AbstractDiffEqArray,
       vecarr_to_vectors, tuples

export recursivecopy, recursivecopy!, recursivefill!, vecvecapply, copyat_or_push!,
       vecvec_to_mat, recursive_one, recursive_mean, recursive_bottom_eltype,
       recursive_unitless_bottom_eltype, recursive_unitless_eltype

export ArrayPartition

end # module

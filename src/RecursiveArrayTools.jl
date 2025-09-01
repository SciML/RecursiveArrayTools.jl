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

"""
    AbstractVectorOfArray{T, N, A}

An AbstractVectorOfArray is an object which represents arrays of arrays,
and arbitrary recursive nesting of arrays, as a single array-like object.
Thus a canonical example of an AbstractVectorOfArray is something of the
form `VectorOfArray([[1,2],[3,4]])`, which "acts" like the matrix `[1 3; 2 4]`
where the data is stored and accessed in a column-ordered fashion (as is typical
in Julia), but the actual matrix is never constructed and instead lazily represented
through the type.

An AbstractVectorOfArray subtype should match the following behaviors.

!!! note

    In 2023 the linear indexing `A[i]` was deprecated. It previously had the behavior that `A[i] = A.u[i]`. However, this is incompatible with standard `AbstractArray` interfaces, Since if `A = VectorOfArray([[1,2],[3,4]])` and `A` is supposed to act like `[1 3; 2 4]`, then there is a difference `A[1] = [1,2]` for the VectorOfArray while `A[1] = 1` for the matrix. This causes many issues if `AbstractVectorOfArray <: AbstractArray`. Thus we plan in 2026 to complete the deprecation and thus have a breaking update where `A[i]` matches the linear indexing of an`AbstractArray`, and then making `AbstractVectorOfArray <: AbstractArray`. Until then, `AbstractVectorOfArray` due to
    this interface break but manually implements an `AbstractArray`-like interface for
    future compatibility.

## Fields

An AbstractVectorOfArray has the following fields:

  - `u` which holds the Vector of values at each timestep

## Array Interface

The general operations are as follows. Use

```julia
A.u[j]
```

to access the `j`th array. For multidimensional systems, this
will address first by component and lastly by time, and thus

```julia
A[i, j]
```

will be the `i`th component at array `j`. Hence, `A[j][i] == A[i, j]`. This is done
because Julia is column-major, so the leading dimension should be contiguous in memory.
If the independent variables had shape (for example, was a matrix), then `i` is the
linear index. We can also access solutions with shape:

```julia
A[i, k, j]
```

gives the `[i,k]` component of the system at array `j`. The colon operator is
supported, meaning that

```julia
A[i, :]
```

gives the timeseries for the `i`th component.

## Using the AbstractArray Interface

The `AbstractArray` interface can be directly used. For example, for a vector
system of variables `A[i,j]` is a matrix with rows being the variables and
columns being the timepoints. Operations like `A'` will
transpose the solution type. Functionality written for `AbstractArray`s can
directly use this. For example, the Base `cov` function computes correlations
amongst columns, and thus:

```julia
cov(A)
```

computes the correlation of the system state in time, whereas

```julia
cov(A, 2)
```

computes the correlation between the variables. Similarly, `mean(A,2)` is the
mean of the variable in time, and `var(A,2)` is the variance. Other statistical
functions and packages which work on `AbstractArray` types will work on the
solution type.

## Conversions

At anytime, a true `Array` can be created using `Array(A)`, or more generally `stack(A)`
to make the array type match the internal array type (for example, if `A` is an array
of GPU arrays, `stack(A)` will be a GPU array).
"""
abstract type AbstractVectorOfArray{T, N, A} end

"""
    AbstractDiffEqArray{T, N, A} <: AbstractVectorOfArray{T, N, A}

An AbstractVectorOfArray object which has extra information of a time array `A.t`
in order to specify a time series. A canonical AbstractDiffEqArray is for example
the pairing `DiffEqArray([[1,2],[3,4]],[1.0,2.0])` which means that at time 1.0
the values were `[1,2]` and at time 2.0 the values were `[3,4]`.

An AbstractDiffEqArray has all of the same behaviors as an AbstractVectorOfArray with the
additional properties:

## Fields

An AbstractDiffEqArray adds the following fields:

  - `t` which holds the times of each timestep.
"""
abstract type AbstractDiffEqArray{T, N, A} <: AbstractVectorOfArray{T, N, A} end

include("utils.jl")
include("vector_of_array.jl")
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

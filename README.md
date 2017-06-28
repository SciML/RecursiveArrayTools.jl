# RecursiveArrayTools.jl

[![Build Status](https://travis-ci.org/JuliaDiffEq/RecursiveArrayTools.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/RecursiveArrayTools.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/i1lp4skr4n6o76b5?svg=true)](https://ci.appveyor.com/project/ChrisRackauckas/recursivearraytools-jl-r8tdv)
[![Coverage Status](https://coveralls.io/repos/ChrisRackauckas/RecursiveArrayTools.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/ChrisRackauckas/RecursiveArrayTools.jl?branch=master)
[![codecov.io](http://codecov.io/github/ChrisRackauckas/RecursiveArrayTools.jl/coverage.svg?branch=master)](http://codecov.io/github/ChrisRackauckas/RecursiveArrayTools.jl?branch=master)

RecursiveArrayTools.jl is a set of tools for dealing with recursive arrays like
arrays of arrays. The current functionality includes:

### Types

#### VectorOfArray

```julia
VectorOfArray(u::AbstractVector)
```

A `VectorOfArray` is an array which has the underlying data structure `Vector{AbstractArray{T}}`
(but hopefully concretely typed!). This wrapper over such data structures allows one to lazily
act like it's a higher dimensional vector, and easily convert to different forms. The indexing
structure is:

```julia
A[i] # Returns the ith array in the vector of arrays
A[j,i] # Returns the jth component in the ith array
A[j1,...,jN,i] # Returns the (j1,...,jN) component of the ith array
```

which presents itself as a column-major matrix with the columns being the arrays from the vector.
The `AbstractArray` interface is implemented, giving access to `copy`, `push`, `append!`, etc. function
which act appropriate. Points to note are:

- The length is the number of vectors, or `length(A.u)` where `u` is the vector of arrays.
- Iteration follows the linear index and goes over the vectors

Additionally, the `convert(Array,VA::AbstractVectorOfArray)` function is provided which transforms
the `VectorOfArray` into a matrix/tensor. Also, `vecarr_to_vectors(VA::AbstractVectorOfArray)`
returns a vector of the series for each component, that is `A[i,:]` for each `i`.
A plot recipe is provided which plots the `A[i,:]` series.

#### DiffEqArray

Related to the `VectorOfArray` is the `DiffEqArray`

```julia
DiffEqArray(u::AbstractVector,t::AbstractVector)
```

This is a `VectorOfArray` which stores `A.t` which matches `A.u`. This will plot
`(A.t[i],A[i,:])`. The function `tuples(diffeq_arr)` returns tuples of `(t,u)`.

#### ArrayPartition

```julia
ArrayPartition(x::AbstractArray...)
```

An `ArrayPartition` `A` is an array which is made up of different arrays `A.x`.
These index like a single array, but each subarray may have a different type.
However, broadcast is overloaded to loop in an efficient manner, meaning that
`A .+= 2.+B` is type-stable in its computations, even if `A.x[i]` and `A.x[j]`
do not match types. A full array interface is included for completeness, which
allows this array type to be used in place of a standard array in places where
such a type stable broadcast may be needed. One example is in heterogeneous
differential equations for [DifferentialEquations.jl](https://github.com/JuliaDiffEq/DifferentialEquations.jl).

An `ArrayPartition` acts like a single array. `A[i]` indexes through the first
array, then the second, etc. all linearly. But `A.x` is where the arrays are stored.
Thus for

```julia
using RecursiveArrayTools
A = ArrayPartition(y,z)
```

We would have `A.x[1]==y` and `A.x[2]==z`. Broadcasting like `f.(A)` is efficient.

### Functions

```julia
recursivecopy!(b::Array{T,N},a::Array{T,N})
```

A recursive `copy!` function. Acts like a `deepcopy!` on arrays of arrays, but
like `copy!` on arrays of scalars.

```julia
convert(Array,vecvec)
```

Technically just a Base fallback that works well. Takes in a vector of arrays,
returns an array of dimension one greater than the original elements.
Works on `AbstractVectorOfArray`. If the `vecvec` is ragged, i.e. not all of the
elements are the same, then it uses the size of the first element to determine
the conversion.

```julia
vecvecapply(f::Base.Callable,v)
```

Calls `f` on each element of a vecvec `v`.

```julia
copyat_or_push!{T}(a::AbstractVector{T},i::Int,x)
```

If `i<length(x)`, it's simply a `recursivecopy!` to the `i`th element. Otherwise it will
`push!` a `deepcopy`.

```julia
recursive_one(a)
```

Calls `one` on the bottom container to get the "true element one type"

```julia
mean{T<:AbstractArray}(vecvec::Vector{T})
mean{T<:AbstractArray}(matarr::Matrix{T},region=0)
```

Generalized mean functions for vectors of arrays and matrix of arrays.

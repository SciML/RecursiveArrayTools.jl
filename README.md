# RecursiveArrayTools.jl

[![Build Status](https://travis-ci.org/JuliaDiffEq/RecursiveArrayTools.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/RecursiveArrayTools.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/esjxtbl5lv9vun1j?svg=true)](https://ci.appveyor.com/project/ChrisRackauckas/recursivearraytools-jl)
[![Coverage Status](https://coveralls.io/repos/ChrisRackauckas/RecursiveArrayTools.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/ChrisRackauckas/RecursiveArrayTools.jl?branch=master)
[![codecov.io](http://codecov.io/github/ChrisRackauckas/RecursiveArrayTools.jl/coverage.svg?branch=master)](http://codecov.io/github/ChrisRackauckas/RecursiveArrayTools.jl?branch=master)

RecursiveArrayTools.jl is a set of tools for dealing with recursive arrays like
arrays of arrays. The current functionality includes:

### Types

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

### Functions

```julia
recursivecopy!(b::Array{T,N},a::Array{T,N})
```

A recursive `copy!` function. Acts like a `deepcopy!` on arrays of arrays, but
like `copy!` on arrays of scalars.

```julia
vecvec_to_mat(vecvec)
```

Takes in a vector of vectors, returns a matrix.

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

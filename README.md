# RecursiveArrayTools.jl

[![Build Status](https://travis-ci.org/JuliaDiffEq/RecursiveArrayTools.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/RecursiveArrayTools.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/4v9mfweq4er0nv3t?svg=true)](https://ci.appveyor.com/project/ChrisRackauckas/recursivearraytools-jl-acpw5)
[![Coverage Status](https://coveralls.io/repos/ChrisRackauckas/RecursiveArrayTools.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/ChrisRackauckas/RecursiveArrayTools.jl?branch=master)
[![codecov.io](http://codecov.io/github/ChrisRackauckas/RecursiveArrayTools.jl/coverage.svg?branch=master)](http://codecov.io/github/ChrisRackauckas/RecursiveArrayTools.jl?branch=master)

RecursiveArrayTools.jl is a set of tools for dealing with recursive arrays like
arrays of arrays. The current functionality includes:

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

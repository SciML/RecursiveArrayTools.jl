# RecursiveArrayTools.jl

[![Build Status](https://github.com/SciML/RecursiveArrayTools.jl/workflows/CI/badge.svg)](https://github.com/SciML/RecursiveArrayTools.jl/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/SciML/RecursiveArrayTools.jl/coverage.svg?branch=master)](http://codecov.io/github/SciML/RecursiveArrayTools.jl?branch=master)
[![Build status](https://badge.buildkite.com/5f39777d009ce94ef1dcf2a4881c68b9fbcaf6f69f1d8b8df2.svg)](https://buildkite.com/julialang/recursivearraytools-dot-jl)

RecursiveArrayTools.jl is a set of tools for dealing with recursive arrays like
arrays of arrays.

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://recursivearraytools.sciml.ai/stable/). Use the
[in-development documentation](https://recursivearraytools.sciml.ai/dev/) for the version of
the documentation, which contains the unreleased features.

## Example

```julia
using RecursiveArrayTools
a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
vA = VectorOfArray(a)
vB = VectorOfArray(b)

vA .+ vB # Now all standard array stuff works!

a = (rand(5),rand(5))
b = (rand(5),rand(5))
pA = ArrayPartition(a)
pB = ArrayPartition(b)

pA .+ pB # Now all standard array stuff works!
```
# RecursiveArrayTools.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/RecursiveArrayTools/stable/)

[![codecov](https://codecov.io/gh/SciML/RecursiveArrayTools.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/RecursiveArrayTools.jl)
[![Build Status](https://github.com/SciML/RecursiveArrayTools.jl/workflows/CI/badge.svg)](https://github.com/SciML/RecursiveArrayTools.jl/actions?query=workflow%3ACI)
[![build status](https://badge.buildkite.com/9eab94781cf0af9a3566e9b9f16abe5aea167b640b88065285.svg?branch=master)](https://buildkite.com/julialang/recursivearraytools-dot-jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

RecursiveArrayTools.jl is a set of tools for dealing with recursive arrays like
arrays of arrays.

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/RecursiveArrayTools/stable/). Use the
[in-development documentation](https://docs.sciml.ai/RecursiveArrayTools/dev/) for the version of
the documentation, which contains the unreleased features.

## Example

### VectorOfArray

```julia
using RecursiveArrayTools
a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
vA = VectorOfArray(a)
vB = VectorOfArray(b)

vA .* vB # Now all standard array stuff works!
```

### ArrayPartition

```julia
a = (rand(5), rand(5))
b = (rand(5), rand(5))
pA = ArrayPartition(a)
pB = ArrayPartition(b)

pA .* pB # Now all standard array stuff works!

# or do:
x0 = rand(3, 3)
v0 = rand(3, 3)
a0 = rand(3, 3)
u0 = ArrayPartition(x0, v0, a0)
u0.x[1] == x0 # true

u0 .+= 1
u0.x[2] == v0 # still true

# do some calculations creating a new partitioned array
unew = u0 * 10
# easily access the individual components without having to rely on complicated indexing
xnew, vnew, anew = unew.x
```

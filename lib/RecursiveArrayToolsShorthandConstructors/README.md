# RecursiveArrayToolsShorthandConstructors.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/RecursiveArrayTools/stable/)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

RecursiveArrayToolsShorthandConstructors.jl is a component of the [RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl) monorepo. Shorthand `VA[...]` and `AP[...]` constructor syntax for RecursiveArrayTools types.
While completely independent and usable on its own, users wanting the full functionality should use [RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl).

This subpackage provides `VA[...]` and `AP[...]` syntax for constructing
`VectorOfArray` and `ArrayPartition` objects. It is separated from the main
package because the `getindex(::Type, ...)` method definitions invalidate
compiled specializations of `Base.getindex(::Type{T}, vals...)`, increasing
load times for downstream packages that don't need the syntax.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/SciML/RecursiveArrayTools.jl",
        subdir = "lib/RecursiveArrayToolsShorthandConstructors")
```

## Usage

```julia
using RecursiveArrayTools
using RecursiveArrayToolsShorthandConstructors

# VectorOfArray shorthand
u = VA[[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# ArrayPartition shorthand
x0, v0, a0 = rand(3, 3), rand(3, 3), rand(3, 3)
u0 = AP[x0, v0, a0]

# Nesting works too
nested = VA[
    fill(1, 2, 3),
    VA[3ones(3), zeros(3)],
]
```

Without this package, use the equivalent explicit constructors:

```julia
u = VectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
u0 = ArrayPartition(x0, v0, a0)
```

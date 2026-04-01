# RecursiveArrayToolsShorthandConstructors.jl

Shorthand constructor syntax for
[RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl) types.

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

# Subpackages

RecursiveArrayTools.jl ships several optional subpackages under `lib/`. Each is a
separate Julia package that adds functionality which was split out of the main
package to avoid method invalidations that would increase load times for users who
don't need that functionality.

## RecursiveArrayToolsRaggedArrays

True ragged (non-rectangular) array types that preserve exact structure without
zero-padding. See the [Ragged Arrays](@ref) page for full documentation.

```julia
using RecursiveArrayToolsRaggedArrays
```

## RecursiveArrayToolsShorthandConstructors

Shorthand `VA[...]` and `AP[...]` constructor syntax for `VectorOfArray` and
`ArrayPartition`.

This is separated from the main package because the `getindex(::Type, ...)` method
definitions invalidate compiled specializations of `Base.getindex(::Type{T}, vals...)`
from Base, increasing load times for downstream packages that don't need the syntax.

```julia
using RecursiveArrayToolsShorthandConstructors
```

### Usage

```julia
using RecursiveArrayTools
using RecursiveArrayToolsShorthandConstructors

# VectorOfArray shorthand (equivalent to VectorOfArray([[1,2,3], [4,5,6]]))
u = VA[[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# ArrayPartition shorthand (equivalent to ArrayPartition(x0, v0, a0))
x0, v0, a0 = rand(3, 3), rand(3, 3), rand(3, 3)
u0 = AP[x0, v0, a0]
u0.x[1] === x0  # true

# Nesting works
nested = VA[fill(1, 2, 3), VA[3ones(3), zeros(3)]]
```

Without this package, use the equivalent explicit constructors:

```julia
u = VectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
u0 = ArrayPartition(x0, v0, a0)
```

## RecursiveArrayToolsArrayPartitionAnyAll

Loading this subpackage installs direct `any` and `all` methods for
`ArrayPartition`, including partitions backed by GPU arrays.
The main package already reduces `any` and `all` partition by partition,
so full reductions on GPU-backed partitions work with scalar indexing disabled.

The direct methods are opt-in because they can invalidate compiled
specializations of `any` and `all` on Julia versions before 1.13.

```julia
using RecursiveArrayToolsArrayPartitionAnyAll
```

### Usage

```julia
using RecursiveArrayTools
using RecursiveArrayToolsArrayPartitionAnyAll

ap = ArrayPartition(rand(1000), rand(1000), rand(1000))

any(isnan, ap)
all(x -> x > 0, ap)
```

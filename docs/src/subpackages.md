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

Optimized `any` and `all` for `ArrayPartition` that iterate partition-by-partition
instead of element-by-element, giving ~1.5-1.8x speedup on full scans.

This is separated from the main package because `any(f::Function, ::ArrayPartition)`
invalidates ~780 compiled specializations of `any(f::Function, ::AbstractArray)`.

```julia
using RecursiveArrayToolsArrayPartitionAnyAll
```

### Usage

```julia
using RecursiveArrayTools
using RecursiveArrayToolsArrayPartitionAnyAll

ap = ArrayPartition(rand(1000), rand(1000), rand(1000))

# These now use the optimized partition-by-partition iteration
any(isnan, ap)       # ~1.5x faster than default
all(x -> x > 0, ap)  # ~1.8x faster than default
```

Without this package, `any`/`all` use the default `AbstractArray` implementation
which is correct but slower due to per-element partition indexing overhead.

### Why is it faster?

`ArrayPartition` stores data as a tuple of arrays. The default `AbstractArray`
`any`/`all` iterates element-by-element, which requires computing which partition
each linear index falls into. The optimized methods iterate over each partition
array directly, avoiding that lookup overhead entirely.

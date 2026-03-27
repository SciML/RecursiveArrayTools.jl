# RecursiveArrayToolsArrayPartitionAnyAll.jl

Optimized `any` and `all` for
[RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl)'s
`ArrayPartition` type.

`ArrayPartition` stores data as a tuple of arrays. The default `AbstractArray`
`any`/`all` iterates element-by-element through the partition, which incurs
per-element partition lookup overhead. This subpackage provides methods that
iterate partition-by-partition instead, giving ~1.5-1.8x speedup on full scans.

It is separated from the main package because `any(f::Function, ::ArrayPartition)`
invalidates ~780 compiled specializations of `any(f::Function, ::AbstractArray)`.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/SciML/RecursiveArrayTools.jl",
        subdir = "lib/RecursiveArrayToolsArrayPartitionAnyAll")
```

## Usage

```julia
using RecursiveArrayTools
using RecursiveArrayToolsArrayPartitionAnyAll

ap = ArrayPartition(rand(1000), rand(1000), rand(1000))

# These now use the optimized partition-by-partition iteration
any(isnan, ap)
all(x -> x > 0, ap)
```

Without this package, `any`/`all` use the default `AbstractArray` implementation
which is correct but ~1.5x slower due to per-element partition indexing overhead.

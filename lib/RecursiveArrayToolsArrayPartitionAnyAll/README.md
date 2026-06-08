# RecursiveArrayToolsArrayPartitionAnyAll.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/RecursiveArrayTools/stable/)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

RecursiveArrayToolsArrayPartitionAnyAll.jl is a component of the [RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl) monorepo. Optimized `any` and `all` for the `ArrayPartition` type.
While completely independent and usable on its own, users wanting the full functionality should use [RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl).

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

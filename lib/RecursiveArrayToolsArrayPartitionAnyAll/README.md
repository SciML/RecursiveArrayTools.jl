# RecursiveArrayToolsArrayPartitionAnyAll.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/RecursiveArrayTools/stable/)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

RecursiveArrayToolsArrayPartitionAnyAll.jl is a component of the [RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl) monorepo. Optimized `any` and `all` for the `ArrayPartition` type.
While completely independent and usable on its own, users wanting the full functionality should use [RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl).

`ArrayPartition` stores data as a tuple of arrays. Both the base package's
full-reduction fallback and this subpackage's direct public methods iterate
partition-by-partition, avoiding per-element partition lookup and scalar GPU
indexing.

It is separated from the main package because `any(f::Function, ::ArrayPartition)`
invalidates ~780 compiled specializations of `any(f::Function, ::AbstractArray)`
on Julia < 1.13. The base package instead specializes Base's internal full-reduction
hooks, preserving the public `any`/`all` dispatch while remaining GPU-safe.

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

Without this package, the base package's full reductions still iterate
partition-by-partition and support GPU arrays. Loading this package installs
direct public `any`/`all` methods for `ArrayPartition`.

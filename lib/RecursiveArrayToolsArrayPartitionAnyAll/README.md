# RecursiveArrayToolsArrayPartitionAnyAll.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/RecursiveArrayTools/stable/)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

RecursiveArrayToolsArrayPartitionAnyAll.jl is a component of the [RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl) monorepo. Optimized `any` and `all` for the `ArrayPartition` type.
While completely independent and usable on its own, users wanting the full functionality should use [RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl).

`ArrayPartition` stores data as a tuple of arrays. Loading a GPUArrays-based
backend such as CUDA or JLArrays automatically enables public `any` and `all`
methods for partitions whose immediate blocks are all GPU arrays. Full
reductions on these partitions work with scalar indexing disabled. CPU
partitions use Julia's generic reductions by default.

Loading this subpackage adds partition-wise full reductions for CPU,
mixed-storage and nested partitions. These broader methods are opt-in
because they can invalidate compiled code on older Julia versions.

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

# Recursive Array Types

The Recursive Array types are types which implement an `AbstractArray` interface so
that recursive arrays can be handled with standard array functionality. For example,
wrapped arrays will automatically do things like recurse broadcast, define optimized
mapping and iteration functions, and more.

## Abstract Types

## Concrete Types

```@docs
VectorOfArray
DiffEqArray
ArrayPartition
NamedArrayPartition
```

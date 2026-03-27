"""
    RecursiveArrayToolsShorthandConstructors

Opt-in subpackage providing shorthand syntax and optimized methods that cause
method table invalidations. These are separated from RecursiveArrayTools to
avoid invalidating compiled code for users who don't need them.

```julia
using RecursiveArrayToolsShorthandConstructors
```

This enables:
- `VA[a, b, c]` shorthand for `VectorOfArray([a, b, c])`
- `AP[a, b, c]` shorthand for `ArrayPartition(a, b, c)`
- Optimized `any`/`all` for `ArrayPartition` (partition-level short-circuiting)
"""
module RecursiveArrayToolsShorthandConstructors

using RecursiveArrayTools: VA, VectorOfArray, AP, ArrayPartition

# VA[...] shorthand
Base.getindex(::Type{VA}, xs...) = VectorOfArray(collect(xs))

# AP[...] shorthand
Base.getindex(::Type{AP}, xs...) = ArrayPartition(xs...)

# Optimized any/all for ArrayPartition — applies f partition-by-partition
# for faster short-circuiting instead of element-by-element.
Base.any(f, A::ArrayPartition) = any((any(f, x) for x in A.x))
Base.any(f::Function, A::ArrayPartition) = any((any(f, x) for x in A.x))
Base.any(A::ArrayPartition) = any(identity, A)
Base.all(f, A::ArrayPartition) = all((all(f, x) for x in A.x))
Base.all(f::Function, A::ArrayPartition) = all((all(f, x) for x in A.x))
Base.all(A::ArrayPartition) = all(identity, A)

end # module

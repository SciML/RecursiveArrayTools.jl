"""
    RecursiveArrayToolsArrayPartitionAnyAll

Optimized `any`/`all` for `ArrayPartition` that iterates partition-by-partition
instead of using the generic `AbstractArray` element iteration. This gives
~1.5-1.8x speedup on full scans because it avoids per-element partition lookup
overhead. Separated into its own subpackage because `any(f::Function, ::ArrayPartition)`
invalidates ~780 specializations of `any(f::Function, ::AbstractArray)`.

```julia
using RecursiveArrayToolsArrayPartitionAnyAll
```
"""
module RecursiveArrayToolsArrayPartitionAnyAll

using RecursiveArrayTools: ArrayPartition

Base.any(f, A::ArrayPartition) = any((any(f, x) for x in A.x))
Base.any(f::Function, A::ArrayPartition) = any((any(f, x) for x in A.x))
Base.any(A::ArrayPartition) = any(identity, A)
Base.all(f, A::ArrayPartition) = all((all(f, x) for x in A.x))
Base.all(f::Function, A::ArrayPartition) = all((all(f, x) for x in A.x))
Base.all(A::ArrayPartition) = all(identity, A)

end

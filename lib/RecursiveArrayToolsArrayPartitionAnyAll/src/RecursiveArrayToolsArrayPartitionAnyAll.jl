"""
    RecursiveArrayToolsArrayPartitionAnyAll

Optimized `any`/`all` for `ArrayPartition` that installs direct public methods
instead of using the generic `AbstractArray` entry points. The base package's
full-reduction fallback is also partition-wise and safe for GPU arrays; this
subpackage remains available for callers that explicitly load direct methods.
It is separated because `any(f::Function, ::ArrayPartition)` invalidates ~780
specializations of `any(f::Function, ::AbstractArray)` on Julia < 1.13.

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

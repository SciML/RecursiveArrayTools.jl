"""
    RecursiveArrayToolsArrayPartitionOptimizations

Optimized `any`/`all` for `ArrayPartition` that short-circuit at the
partition level instead of element-by-element. Separated into its own
subpackage because the `any(f::Function, ::ArrayPartition)` method
invalidates ~780 downstream compiled specializations of
`any(f::Function, ::AbstractArray)`.

```julia
using RecursiveArrayToolsArrayPartitionOptimizations
```
"""
module RecursiveArrayToolsArrayPartitionOptimizations

using RecursiveArrayTools: ArrayPartition

Base.any(f, A::ArrayPartition) = any((any(f, x) for x in A.x))
Base.any(f::Function, A::ArrayPartition) = any((any(f, x) for x in A.x))
Base.any(A::ArrayPartition) = any(identity, A)
Base.all(f, A::ArrayPartition) = all((all(f, x) for x in A.x))
Base.all(f::Function, A::ArrayPartition) = all((all(f, x) for x in A.x))
Base.all(A::ArrayPartition) = all(identity, A)

end

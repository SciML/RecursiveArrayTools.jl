"""
    RecursiveArrayToolsArrayPartitionAnyAll

Install partition-wise `any` and `all` methods for CPU, mixed-storage and nested
`ArrayPartition` layouts. GPUArrays-based backends automatically enable public
methods for partitions whose immediate blocks are all GPU arrays. The broader
methods are opt-in because they can invalidate compiled code on older Julia
versions.

```julia
using RecursiveArrayToolsArrayPartitionAnyAll
```
"""
module RecursiveArrayToolsArrayPartitionAnyAll

using RecursiveArrayTools: ArrayPartition

for op in (:any, :all)
    @eval begin
        function Base.$op(f::Function, A::ArrayPartition; dims = :)
            dims === (:) && return $op(x -> $op(f, x), A.x)
            return $op(Vector(map(f, A)); dims)
        end
        Base.$op(f, A::ArrayPartition; dims = :) = $op(x -> f(x), A; dims)
        Base.$op(A::ArrayPartition; dims = :) = $op(identity, A; dims)
    end
end

end

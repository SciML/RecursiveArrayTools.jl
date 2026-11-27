module RecursiveArrayToolsGPUArraysExt

using GPUArrays: AnyGPUArray
using RecursiveArrayTools: ArrayPartition, _host_partition

const GPUArrayPartition = ArrayPartition{<:Any, <:Tuple{AnyGPUArray, Vararg{AnyGPUArray}}}

for op in (:any, :all)
    @eval begin
        function Base.$op(f::Function, A::GPUArrayPartition; dims = :)
            dims === (:) && return $op(x -> $op(f, x), A.x)
            return invoke($op, Tuple{Function, AbstractArray}, f, _host_partition(A); dims)
        end
        Base.$op(f, A::GPUArrayPartition; dims = :) = $op(x -> f(x), A; dims)
        Base.$op(A::GPUArrayPartition; dims = :) = $op(identity, A; dims)
    end
end

end

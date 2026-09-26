module RecursiveArrayToolsCUDAExt

using RecursiveArrayTools: AbstractVectorOfArray, ArrayPartition
import CUDA: CuArray

# Disambiguate CuArray(::AbstractVectorOfArray) vs CuArray(::AbstractArray{T,N}) from CUDA.jl.
# This is the exact signature Julia's ambiguity error requests.
# Uses stack to stay on GPU (avoids GPU→CPU→GPU round-trip).
function CuArray(VA::AbstractVectorOfArray{T, N}) where {T, N}
    return CuArray{T, N}(stack(VA.u))
end

# Assemble each state from its GPU partitions before stacking the states.
# Stacking ArrayPartitions directly reads their elements on the CPU.
function CuArray(
        VA::AbstractVectorOfArray{T, N, <:AbstractVector{<:ArrayPartition}}
    ) where {T, N}
    return CuArray{T, N}(stack(map(ap -> vcat(vec.(ap.x)...), VA.u)))
end

end

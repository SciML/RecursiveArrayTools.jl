module RecursiveArrayToolsCUDAExt

using RecursiveArrayTools: AbstractVectorOfArray
import CUDA: CuArray

# Disambiguate CuArray(::AbstractVectorOfArray) vs CuArray(::AbstractArray{T,N}) from CUDA.jl.
# This is the exact signature Julia's ambiguity error requests.
# Uses stack to stay on GPU (avoids GPU→CPU→GPU round-trip).
function CuArray(VA::AbstractVectorOfArray{T, N}) where {T, N}
    return CuArray{T, N}(stack(VA.u))
end

end

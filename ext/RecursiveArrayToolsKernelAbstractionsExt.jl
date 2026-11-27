module RecursiveArrayToolsKernelAbstractionsExt

import RecursiveArrayTools: ArrayPartition, VectorOfArray
import KernelAbstractions

function KernelAbstractions.get_backend(x::VectorOfArray)
    u = parent(x)
    if length(u) == 0
        error("VectorOfArray is empty, cannot determine backend.")
    end
    # Use the backend of the first element in the parent array
    return KernelAbstractions.get_backend(u[1])
end

function KernelAbstractions.get_backend(x::ArrayPartition)
    backend = common_backend(nothing, x)
    if backend === nothing
        throw(ArgumentError("ArrayPartition has no array leaves, cannot determine backend."))
    end
    return backend
end

common_backend(backend, x) = backend
common_backend(backend, x::ArrayPartition) = foldl(common_backend, x.x; init = backend)
function common_backend(backend, x::AbstractArray)
    leaf = KernelAbstractions.get_backend(x)
    if backend !== nothing && !isequal(backend, leaf)
        throw(ArgumentError("ArrayPartition contains incompatible backends: $backend and $leaf."))
    end
    return leaf
end

end

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
    if isempty(x.x)
        error("ArrayPartition is empty, cannot determine backend.")
    end
    return KernelAbstractions.get_backend(first(x.x))
end

end

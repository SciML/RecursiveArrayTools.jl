module RecursiveArrayToolsSparseArraysExt

import SparseArrays
import RecursiveArrayTools

function Base.copyto!(
        dest::SparseArrays.AbstractCompressedVector, A::RecursiveArrayTools.ArrayPartition)
    @assert length(dest) == length(A)
    cur = 1
    @inbounds for i in 1:length(A.x)
        if A.x[i] isa Number
            dest[cur:(cur + length(A.x[i]) - 1)] .= A.x[i]
        else
            dest[cur:(cur + length(A.x[i]) - 1)] .= vec(A.x[i])
        end
        cur += length(A.x[i])
    end
    dest
end

# Fix for issue #486: Define issparse for AbstractVectorOfArray
# AbstractVectorOfArray is not a sparse array type, so it should return false
SparseArrays.issparse(::RecursiveArrayTools.AbstractVectorOfArray) = false

end

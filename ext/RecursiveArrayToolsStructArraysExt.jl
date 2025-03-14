module RecursiveArrayToolsStructArraysExt

import RecursiveArrayTools, StructArrays
RecursiveArrayTools.rewrap(::StructArrays.StructArray, u) = StructArrays.StructArray(u)

using RecursiveArrayTools: VectorOfArray
using StructArrays: StructArray

const VectorOfStructArray{T, N} = VectorOfArray{T, N, <:StructArray}

# Since `StructArray` lazily materializes struct entries, the general `setindex!(x, val, I)` 
# operation `VA.u[I[end]][Base.front(I)...]` will only update a lazily materialized struct 
# entry of `u`, but will not actually mutate `x::StructArray`. See the StructArray documentation 
# for more details:
#
#   https://juliaarrays.github.io/StructArrays.jl/stable/counterintuitive/#Modifying-a-field-of-a-struct-element 
# 
# To avoid this, we can materialize a struct entry, modify it, and then use `setindex!` 
# with the modified struct entry.
function Base.setindex!(VA::VectorOfStructArray{T, N}, v,
        I::Int...) where {T, N}
    u_I = VA.u[I[end]]
    u_I[Base.front(I)...] = v
    return VA.u[I[end]] = u_I
end

end

module RecursiveArrayToolsStructArraysExt

import RecursiveArrayTools, StructArrays
RecursiveArrayTools.rewrap(::StructArrays.StructArray, u) = StructArrays.StructArray(u)

using RecursiveArrayTools: VectorOfArray, VectorOfArrayStyle, ArrayInterface, unpack_voa,
                           narrays, StaticArraysCore
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
# 
function Base.setindex!(VA::VectorOfStructArray{T, N}, v,
        I::Int...) where {T, N}
    u_I = VA.u[I[end]]
    u_I[Base.front(I)...] = v
    return VA.u[I[end]] = u_I
end

for (type, N_expr) in [
    (Broadcast.Broadcasted{<:VectorOfArrayStyle}, :(narrays(bc))),
    (Broadcast.Broadcasted{<:Broadcast.DefaultArrayStyle}, :(length(dest.u)))
]
    @eval @inline function Base.copyto!(dest::VectorOfStructArray,
            bc::$type)
        bc = Broadcast.flatten(bc)
        N = $N_expr
        @inbounds for i in 1:N
            dest_i = dest[:, i]
            if dest_i isa AbstractArray
                if ArrayInterface.ismutable(dest_i)
                    copyto!(dest_i, unpack_voa(bc, i))
                else
                    unpacked = unpack_voa(bc, i)
                    arr_type = StaticArraysCore.similar_type(dest_i)
                    dest_i = if length(unpacked) == 1 && length(dest_i) == 1
                        arr_type(unpacked[1])
                    elseif length(unpacked) == 1
                        fill(copy(unpacked), arr_type)
                    else
                        arr_type(unpacked[j] for j in eachindex(unpacked))
                    end
                end
            else
                dest_i = copy(unpack_voa(bc, i))
            end
            dest[:, i] = dest_i
        end
        dest
    end
end

end

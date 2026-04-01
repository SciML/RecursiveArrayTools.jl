module RecursiveArrayToolsFastBroadcastExt

using RecursiveArrayTools
using FastBroadcast
using FastBroadcast: Serial
using StaticArraysCore

const AbstractVectorOfSArray = AbstractVectorOfArray{
    T, N, <:AbstractVector{<:StaticArraysCore.SArray},
} where {T, N}

@inline function FastBroadcast.fast_materialize!(
        ::Serial, dst::AbstractVectorOfSArray,
        bc::Broadcast.Broadcasted{S}
    ) where {S}
    if FastBroadcast.use_fast_broadcast(S)
        for i in 1:length(dst.u)
            unpacked = RecursiveArrayTools.unpack_voa(bc, i)
            dst.u[i] = StaticArraysCore.similar_type(dst.u[i])(
                unpacked[j]
                    for j in eachindex(unpacked)
            )
        end
    else
        Broadcast.materialize!(dst, bc)
    end
    return dst
end

end # module

module RecursiveArrayToolsFastBroadcastExt

using RecursiveArrayTools
using FastBroadcast
using FastBroadcast: Serial, Threaded
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

@inline function FastBroadcast.fast_materialize!(
        ::Threaded, dst::AbstractVectorOfSArray,
        bc::Broadcast.Broadcasted{S}
    ) where {S}
    if FastBroadcast.use_fast_broadcast(S)
        Threads.@threads for i in 1:length(dst.u)
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

# Fallback for non-SArray VectorOfArray: the generic threaded path splits
# along the last axis via views, which does not correctly partition work for
# VectorOfArray. Fall back to serial broadcasting.
@inline function FastBroadcast.fast_materialize!(
        ::Threaded, dst::AbstractVectorOfArray,
        bc::Broadcast.Broadcasted
    )
    return FastBroadcast.fast_materialize!(Serial(), dst, bc)
end

end # module

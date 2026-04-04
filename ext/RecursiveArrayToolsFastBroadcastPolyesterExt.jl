module RecursiveArrayToolsFastBroadcastPolyesterExt

using RecursiveArrayTools
using FastBroadcast
using FastBroadcast: Serial, Threaded
using Polyester
using StaticArraysCore

# Signal to the base FastBroadcast extension that Polyester threading is available.
RecursiveArrayTools.POLYESTER_LOADED[] = true

const AbstractVectorOfSArray = AbstractVectorOfArray{
    T, N, <:AbstractVector{<:StaticArraysCore.SArray},
} where {T, N}

@inline function _polyester_fast_materialize!(
        dst::AbstractVectorOfSArray,
        bc::Broadcast.Broadcasted{S}
    ) where {S}
    if FastBroadcast.use_fast_broadcast(S)
        @batch for i in 1:length(dst.u)
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
    return _polyester_fast_materialize!(dst, bc)
end

# Disambiguation: this method is more specific than both the base ext's
# (::Threaded, ::AbstractVectorOfArray, ::Broadcasted) fallback and
# the above (::Threaded, ::AbstractVectorOfSArray, ::Broadcasted{S}).
@inline function FastBroadcast.fast_materialize!(
        ::Threaded, dst::AbstractVectorOfSArray,
        bc::Broadcast.Broadcasted
    )
    return _polyester_fast_materialize!(dst, bc)
end

end # module

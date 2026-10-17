module RecursiveArrayToolsFastBroadcastPolyesterExt

using RecursiveArrayTools
using FastBroadcast
using FastBroadcast: Serial, Threaded
using Polyester
using StaticArraysCore

const AbstractVectorOfSArray = AbstractVectorOfArray{
    T, N, <:AbstractVector{<:StaticArraysCore.SArray},
} where {T, N}

# Keep the generator outside @batch so its closure captures only the inner broadcast.
@inline function materialize_inner(inner, bc, i, n)
    inner_bc = RecursiveArrayTools.unpack_voa(bc, i, n)
    unpacked = Broadcast.instantiate(
        Broadcast.Broadcasted(inner_bc.f, inner_bc.args, axes(inner))
    )
    return StaticArraysCore.similar_type(inner)(unpacked[j] for j in eachindex(unpacked))
end

@inline function _polyester_fast_materialize!(
        dst::AbstractVectorOfSArray,
        bc::Broadcast.Broadcasted{S}
    ) where {S}
    if FastBroadcast.use_fast_broadcast(S)
        @batch for i in 1:length(dst.u)
            dst.u[i] = materialize_inner(dst.u[i], bc, i, Val(ndims(dst)))
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

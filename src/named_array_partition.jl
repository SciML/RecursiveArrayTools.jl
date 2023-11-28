"""
    NamedArrayPartition(; kwargs...)
    NamedArrayPartition(x::NamedTuple) 

Similar to an `ArrayPartition` but the individual arrays can be accessed via the 
constructor-specified names. However, unlike `ArrayPartition`, each individual array 
must have the same element type. 
"""   
struct NamedArrayPartition{T, A<:ArrayPartition{T}, NT<:NamedTuple} <: AbstractVector{T}
    array_partition::A
    names_to_indices::NT
end
NamedArrayPartition(; kwargs...) = NamedArrayPartition(NamedTuple(kwargs))
function NamedArrayPartition(x::NamedTuple) 
    names_to_indices = NamedTuple(Pair(symbol, index) for (index, symbol) in enumerate(keys(x)))

    # enforce homogeneity of eltypes
    @assert all(eltype.(values(x)) .== eltype(first(x))) 
    T = eltype(first(x))
    S = typeof(values(x))
    return NamedArrayPartition(ArrayPartition{T, S}(values(x)), names_to_indices)
end

# Note: overloading `getproperty` means we cannot access `NamedArrayPartition` 
# fields except through `getfield` and accessor functions.
ArrayPartition(x::NamedArrayPartition) = getfield(x, :array_partition)

Base.Array(x::NamedArrayPartition) = Array(ArrayPartition(x))

Base.zero(x::NamedArrayPartition{T, S, TN}) where {T, S, TN} = 
    NamedArrayPartition{T, S, TN}(zero(ArrayPartition(x)), getfield(x, :names_to_indices))
Base.zero(A::NamedArrayPartition, dims::NTuple{N, Int}) where {N} = zero(A) # ignore dims since named array partitions are vectors


Base.propertynames(x::NamedArrayPartition) = propertynames(getfield(x, :names_to_indices))
Base.getproperty(x::NamedArrayPartition, s::Symbol) =
    getindex(ArrayPartition(x).x, getproperty(getfield(x, :names_to_indices), s))

# this enables x.s = some_array. 
@inline function Base.setproperty!(x::NamedArrayPartition, s::Symbol, v) 
    index = getproperty(getfield(x, :names_to_indices), s)
    ArrayPartition(x).x[index] .= v
end

# print out NamedArrayPartition as a NamedTuple
Base.summary(x::NamedArrayPartition) = string(typeof(x), " with arrays:")
Base.show(io::IO, m::MIME"text/plain", x::NamedArrayPartition) = 
    show(io, m, NamedTuple(Pair.(keys(getfield(x, :names_to_indices)), ArrayPartition(x).x)))

Base.size(x::NamedArrayPartition) = size(ArrayPartition(x))
Base.length(x::NamedArrayPartition) = length(ArrayPartition(x))
Base.getindex(x::NamedArrayPartition, args...) = getindex(ArrayPartition(x), args...)

Base.setindex!(x::NamedArrayPartition, args...) = setindex!(ArrayPartition(x), args...)
Base.map(f, x::NamedArrayPartition) = NamedArrayPartition(map(f, ArrayPartition(x)), getfield(x, :names_to_indices))
Base.mapreduce(f, op, x::NamedArrayPartition) = mapreduce(f, op, ArrayPartition(x))
# Base.filter(f, x::NamedArrayPartition) = filter(f, ArrayPartition(x))

Base.similar(x::NamedArrayPartition{T, S, NT}) where {T, S, NT} = 
    NamedArrayPartition{T, S, NT}(similar(ArrayPartition(x)), getfield(x, :names_to_indices))

# broadcasting
Base.BroadcastStyle(::Type{<:NamedArrayPartition}) = Broadcast.ArrayStyle{NamedArrayPartition}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{NamedArrayPartition}},
                      ::Type{ElType}) where {ElType}
    x = find_NamedArrayPartition(bc)
    return NamedArrayPartition(similar(ArrayPartition(x)), getfield(x, :names_to_indices))
end

# when broadcasting with ArrayPartition + another array type, the output is the other array tupe
Base.BroadcastStyle(::Broadcast.ArrayStyle{NamedArrayPartition}, ::Broadcast.DefaultArrayStyle{1}) = 
    Broadcast.DefaultArrayStyle{1}()

# hook into ArrayPartition broadcasting routines
@inline RecursiveArrayTools.npartitions(x::NamedArrayPartition) = npartitions(ArrayPartition(x))
@inline RecursiveArrayTools.unpack(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{NamedArrayPartition}}, i) = 
    Broadcast.Broadcasted(bc.f, RecursiveArrayTools.unpack_args(i, bc.args))
@inline RecursiveArrayTools.unpack(x::NamedArrayPartition, i) = unpack(ArrayPartition(x), i)

Base.copy(A::NamedArrayPartition{T,S,NT}) where {T,S,NT} = 
    NamedArrayPartition{T,S,NT}(copy(ArrayPartition(A)), getfield(A, :names_to_indices))

@inline NamedArrayPartition(f::F, N, names_to_indices) where F<:Function = 
    NamedArrayPartition(ArrayPartition(ntuple(f, Val(N))), names_to_indices)

@inline function Base.copy(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{NamedArrayPartition}})
    N = npartitions(bc)
    @inline function f(i)
        copy(unpack(bc, i))
    end
    x = find_NamedArrayPartition(bc)
    NamedArrayPartition(f, N, getfield(x, :names_to_indices))
end

@inline function Base.copyto!(dest::NamedArrayPartition, 
                              bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{NamedArrayPartition}})
  N = npartitions(dest, bc)
  @inline function f(i)
      copyto!(ArrayPartition(dest).x[i], unpack(bc, i))
  end
  ntuple(f, Val(N))
  return dest
end

# `x = find_NamedArrayPartition(x)` returns the first `NamedArrayPartition` among broadcast arguments.
find_NamedArrayPartition(bc::Base.Broadcast.Broadcasted) = find_NamedArrayPartition(bc.args)
find_NamedArrayPartition(args::Tuple) =
    find_NamedArrayPartition(find_NamedArrayPartition(args[1]), Base.tail(args))
find_NamedArrayPartition(x) = x
find_NamedArrayPartition(::Tuple{}) = nothing
find_NamedArrayPartition(x::NamedArrayPartition, rest) = x
find_NamedArrayPartition(::Any, rest) = find_NamedArrayPartition(rest)



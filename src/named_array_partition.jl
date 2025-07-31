"""
    NamedArrayPartition(; kwargs...)
    NamedArrayPartition(x::NamedTuple) 

Similar to an `ArrayPartition` but the individual arrays can be accessed via the
constructor-specified names. However, unlike `ArrayPartition`, each individual array
must have the same element type.
"""
struct NamedArrayPartition{T, A <: ArrayPartition{T}, NT <: NamedTuple} <: AbstractVector{T}
    array_partition::A
    names_to_indices::NT
end
NamedArrayPartition(; kwargs...) = NamedArrayPartition(NamedTuple(kwargs))
function NamedArrayPartition(x::NamedTuple)
    names_to_indices = NamedTuple(Pair(symbol, index)
    for (index, symbol) in enumerate(keys(x)))

    # enforce homogeneity of eltypes
    @assert all(eltype.(values(x)) .== eltype(first(x)))
    T = eltype(first(x))
    S = typeof(values(x))
    return NamedArrayPartition(ArrayPartition{T, S}(values(x)), names_to_indices)
end

# Note: overloading `getproperty` means we cannot access `NamedArrayPartition` 
# fields except through `getfield` and accessor functions.
ArrayPartition(x::NamedArrayPartition) = getfield(x, :array_partition)

function Base.similar(A::NamedArrayPartition)
    NamedArrayPartition(
        similar(getfield(A, :array_partition)), getfield(A, :names_to_indices))
end

# return ArrayPartition when possible, otherwise next best thing of the correct size
function Base.similar(A::NamedArrayPartition, dims::NTuple{N, Int}) where {N}
    NamedArrayPartition(
        similar(getfield(A, :array_partition), dims), getfield(A, :names_to_indices))
end

# similar array partition of common type
@inline function Base.similar(A::NamedArrayPartition, ::Type{T}) where {T}
    NamedArrayPartition(
        similar(getfield(A, :array_partition), T), getfield(A, :names_to_indices))
end

# return ArrayPartition when possible, otherwise next best thing of the correct size
function Base.similar(A::NamedArrayPartition, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    NamedArrayPartition(
        similar(getfield(A, :array_partition), T, dims), getfield(A, :names_to_indices))
end

# similar array partition with different types
function Base.similar(
        A::NamedArrayPartition, ::Type{T}, ::Type{S}, R::DataType...) where {T, S}
    NamedArrayPartition(
        similar(getfield(A, :array_partition), T, S, R), getfield(A, :names_to_indices))
end

Base.Array(x::NamedArrayPartition) = Array(ArrayPartition(x))

function Base.zero(x::NamedArrayPartition{T, S, TN}) where {T, S, TN}
    NamedArrayPartition{T, S, TN}(zero(ArrayPartition(x)), getfield(x, :names_to_indices))
end
Base.zero(A::NamedArrayPartition, dims::NTuple{N, Int}) where {N} = zero(A) # ignore dims since named array partitions are vectors

Base.propertynames(x::NamedArrayPartition) = propertynames(getfield(x, :names_to_indices))
function Base.getproperty(x::NamedArrayPartition, s::Symbol)
    getindex(ArrayPartition(x).x, getproperty(getfield(x, :names_to_indices), s))
end

# this enables x.s = some_array. 
@inline function Base.setproperty!(x::NamedArrayPartition, s::Symbol, v)
    index = getproperty(getfield(x, :names_to_indices), s)
    ArrayPartition(x).x[index] .= v
end

# print out NamedArrayPartition as a NamedTuple
Base.summary(x::NamedArrayPartition) = string(typeof(x), " with arrays:")
function Base.show(io::IO, m::MIME"text/plain", x::NamedArrayPartition)
    show(
        io, m, NamedTuple(Pair.(keys(getfield(x, :names_to_indices)), ArrayPartition(x).x)))
end

Base.size(x::NamedArrayPartition) = size(ArrayPartition(x))
Base.length(x::NamedArrayPartition) = length(ArrayPartition(x))
Base.getindex(x::NamedArrayPartition, args...) = getindex(ArrayPartition(x), args...)

Base.setindex!(x::NamedArrayPartition, args...) = setindex!(ArrayPartition(x), args...)
function Base.map(f, x::NamedArrayPartition)
    NamedArrayPartition(map(f, ArrayPartition(x)), getfield(x, :names_to_indices))
end
Base.mapreduce(f, op, x::NamedArrayPartition) = mapreduce(f, op, ArrayPartition(x))
# Base.filter(f, x::NamedArrayPartition) = filter(f, ArrayPartition(x))

function Base.similar(x::NamedArrayPartition{T, S, NT}) where {T, S, NT}
    NamedArrayPartition{T, S, NT}(
        similar(ArrayPartition(x)), getfield(x, :names_to_indices))
end

# broadcasting
function Base.BroadcastStyle(::Type{<:NamedArrayPartition})
    Broadcast.ArrayStyle{NamedArrayPartition}()
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{NamedArrayPartition}},
        ::Type{ElType}) where {ElType}
    x = find_NamedArrayPartition(bc)
    return NamedArrayPartition(similar(ArrayPartition(x)), getfield(x, :names_to_indices))
end

# when broadcasting with ArrayPartition + another array type, the output is the other array tupe
function Base.BroadcastStyle(
        ::Broadcast.ArrayStyle{NamedArrayPartition}, ::Broadcast.DefaultArrayStyle{1})
    Broadcast.DefaultArrayStyle{1}()
end

# hook into ArrayPartition broadcasting routines
@inline RecursiveArrayTools.npartitions(x::NamedArrayPartition) = npartitions(ArrayPartition(x))
@inline RecursiveArrayTools.unpack(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{NamedArrayPartition}},
    i) = Broadcast.Broadcasted(
    bc.f, RecursiveArrayTools.unpack_args(i, bc.args))
@inline RecursiveArrayTools.unpack(x::NamedArrayPartition, i) = unpack(ArrayPartition(x), i)

function Base.copy(A::NamedArrayPartition{T, S, NT}) where {T, S, NT}
    NamedArrayPartition{T, S, NT}(copy(ArrayPartition(A)), getfield(A, :names_to_indices))
end

@inline NamedArrayPartition(f::F,
    N,
    names_to_indices) where {F <:
                             Function} = NamedArrayPartition(
    ArrayPartition(ntuple(f, Val(N))), names_to_indices)

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
    @inbounds for i in 1:N
        copyto!(getfield(dest, :array_partition).x[i], unpack(bc, i))
    end
    return dest
end

# `x = find_NamedArrayPartition(x)` returns the first `NamedArrayPartition` among broadcast arguments.
find_NamedArrayPartition(bc::Base.Broadcast.Broadcasted) = find_NamedArrayPartition(bc.args)
function find_NamedArrayPartition(args::Tuple)
    find_NamedArrayPartition(find_NamedArrayPartition(args[1]), Base.tail(args))
end
find_NamedArrayPartition(x) = x
find_NamedArrayPartition(::Tuple{}) = nothing
find_NamedArrayPartition(x::NamedArrayPartition, rest) = x
find_NamedArrayPartition(::Any, rest) = find_NamedArrayPartition(rest)

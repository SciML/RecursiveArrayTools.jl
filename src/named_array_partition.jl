abstract type AbstractNamedArrayPartition{T, A, NT} <: AbstractVector{T} end

"""
    NamedArrayPartition(; kwargs...)
    NamedArrayPartition(x::NamedTuple) 

Similar to an `ArrayPartition` but the individual arrays can be accessed via the
constructor-specified names. However, unlike `ArrayPartition`, each individual array
must have the same element type.
"""
struct NamedArrayPartition{T, A <: ArrayPartition{T}, NT <: NamedTuple} <: AbstractNamedArrayPartition{T, A, NT}
    array_partition::A
    names_to_indices::NT
end
(::Type{T})(; kwargs...) where {T<:AbstractNamedArrayPartition} = T(NamedTuple(kwargs))
function (::Type{T})(x::NamedTuple) where {T<:AbstractNamedArrayPartition}
    names_to_indices = NamedTuple(Pair(symbol, index)
    for (index, symbol) in enumerate(keys(x)))

    # enforce homogeneity of eltypes
    @assert all(eltype.(values(x)) .== eltype(first(x)))
    R = eltype(first(x))
    S = typeof(values(x))
    return T(ArrayPartition{R, S}(values(x)), names_to_indices)
end

function named_partition_constructor(X::T) where {T<:AbstractNamedArrayPartition}
    getfield(parentmodule(T), nameof(T))
end

# Note: overloading `getproperty` means we cannot access `NamedArrayPartition` 
# fields except through `getfield` and accessor functions.
ArrayPartition(x::AbstractNamedArrayPartition) = getfield(x, :array_partition)

# With new type structure this function does the same as Base.similar(x::AbstractNamedArrayPartition{T, S, NT}) where {T, S, NT}
#= function Base.similar(A::T) where {T<:AbstractNamedArrayPartition}
    Tconstr = named_partition_constructor(A)
    Tconstr(
        similar(getfield(A, :array_partition)), getfield(A, :names_to_indices))
end =#

# return ArrayPartition when possible, otherwise next best thing of the correct size
function Base.similar(A::T, dims::NTuple{N, Int}) where {T<:AbstractNamedArrayPartition, N}
    Tconstr = named_partition_constructor(A)
    Tconstr(
        similar(getfield(A, :array_partition), dims), getfield(A, :names_to_indices))
end

# similar array partition of common type
@inline function Base.similar(A::S, ::Type{T}) where {S<:AbstractNamedArrayPartition, T}
    Tconstr = named_partition_constructor(A)
    Tconstr(
        similar(getfield(A, :array_partition), T), getfield(A, :names_to_indices))
end

# return ArrayPartition when possible, otherwise next best thing of the correct size
function Base.similar(A::S, ::Type{T}, dims::NTuple{N, Int}) where {T, N, S<:AbstractNamedArrayPartition}
    Tconstr = named_partition_constructor(A)
    Tconstr(
        similar(getfield(A, :array_partition), T, dims), getfield(A, :names_to_indices))
end

# similar array partition with different types
function Base.similar(
        A::U, ::Type{T}, ::Type{S}, R::DataType...) where {T, S, U<:AbstractNamedArrayPartition}
    Tconstr = named_partition_constructor(A)
    Tconstr(
        similar(getfield(A, :array_partition), T, S, R), getfield(A, :names_to_indices))
end

Base.Array(x::AbstractNamedArrayPartition) = Array(ArrayPartition(x))

function Base.zero(x::R) where {R <: AbstractNamedArrayPartition}
   R(zero(ArrayPartition(x)), getfield(x, :names_to_indices))
end
Base.zero(A::AbstractNamedArrayPartition, dims::NTuple{N, Int}) where {N} = zero(A) # ignore dims since named array partitions are vectors

Base.propertynames(x::AbstractNamedArrayPartition) = propertynames(getfield(x, :names_to_indices))
function Base.getproperty(x::AbstractNamedArrayPartition, s::Symbol)
    getindex(ArrayPartition(x).x, getproperty(getfield(x, :names_to_indices), s))
end

# this enables x.s = some_array. 
@inline function Base.setproperty!(x::AbstractNamedArrayPartition, s::Symbol, v)
    index = getproperty(getfield(x, :names_to_indices), s)
    ArrayPartition(x).x[index] .= v
end

# print out NamedArrayPartition as a NamedTuple
Base.summary(x::AbstractNamedArrayPartition) = string(typeof(x), " with arrays:")
function Base.show(io::IO, m::MIME"text/plain", x::AbstractNamedArrayPartition)
    show(
        io, m, NamedTuple(Pair.(keys(getfield(x, :names_to_indices)), ArrayPartition(x).x)))
end

Base.size(x::AbstractNamedArrayPartition) = size(ArrayPartition(x))
Base.length(x::AbstractNamedArrayPartition) = length(ArrayPartition(x))
Base.getindex(x::AbstractNamedArrayPartition, args...) = getindex(ArrayPartition(x), args...)

Base.setindex!(x::AbstractNamedArrayPartition, args...) = setindex!(ArrayPartition(x), args...)
function Base.map(f, x::T) where {T<:AbstractNamedArrayPartition}
    Tconstr = named_partition_constructor(x)
    Tconstr(map(f, ArrayPartition(x)), getfield(x, :names_to_indices))
end
Base.mapreduce(f, op, x::AbstractNamedArrayPartition) = mapreduce(f, op, ArrayPartition(x))
# Base.filter(f, x::AbstractNamedArrayPartition) = filter(f, ArrayPartition(x))

function Base.similar(x::AbstractNamedArrayPartition{T, A, NT}) where {T, A, NT}
    # Safely extract the concrete type parameters

    Tconstr = named_partition_constructor(x)
    return Tconstr{T, A, NT}(
        similar(getfield(x, :array_partition)),
        getfield(x, :names_to_indices)
    )
end
# broadcasting
function Base.BroadcastStyle(::Type{T}) where{T<:AbstractNamedArrayPartition}
    Broadcast.ArrayStyle{T}()
end
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{T}},
        ::Type{ElType}) where {ElType, T<:AbstractNamedArrayPartition}
    x = find_NamedArrayPartition(bc)
    Tconstr = named_partition_constructor(x)
    return Tconstr(similar(ArrayPartition(x)), getfield(x, :names_to_indices))
end

# when broadcasting with ArrayPartition + another array type, the output is the other array tupe
function Base.BroadcastStyle(
        ::Broadcast.ArrayStyle{<:AbstractNamedArrayPartition}, ::Broadcast.DefaultArrayStyle{1})
    Broadcast.DefaultArrayStyle{1}()
end

# hook into ArrayPartition broadcasting routines
@inline RecursiveArrayTools.npartitions(x::AbstractNamedArrayPartition) = npartitions(ArrayPartition(x))
@inline RecursiveArrayTools.unpack(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{<:AbstractNamedArrayPartition}}, i) = Broadcast.Broadcasted(
    bc.f, RecursiveArrayTools.unpack_args(i, bc.args))
@inline RecursiveArrayTools.unpack(x::AbstractNamedArrayPartition, i) = unpack(ArrayPartition(x), i)

function Base.copy(A::AbstractNamedArrayPartition{T, S, NT}) where {T, S, NT}
    Tconstr = named_partition_constructor(A)
    Tconstr{T, S, NT}(copy(ArrayPartition(A)), getfield(A, :names_to_indices))
end

@inline (::Type{T})(f::F, N, names_to_indices) where {F <: Function, T<:AbstractNamedArrayPartition} = T(
    ArrayPartition(ntuple(f, Val(N))), names_to_indices)

@inline function Base.copy(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{T}}) where {T<:AbstractNamedArrayPartition}
    N = npartitions(bc)
    @inline function f(i)
        copy(unpack(bc, i))
    end
    x = find_NamedArrayPartition(bc)
    Tconstr = named_partition_constructor(x)
    Tconstr(f, N, getfield(x, :names_to_indices))
end

@inline function Base.copyto!(dest::AbstractNamedArrayPartition,
        bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{<:AbstractNamedArrayPartition}})
    N = npartitions(dest, bc)
    @inline function f(i)
        copyto!(ArrayPartition(dest).x[i], unpack(bc, i))
    end
    ntuple(f, Val(N))
    return dest
end

#Overwrite ArrayInterface zeromatrix to work with NamedArrayPartitions & implicit solvers within OrdinaryDiffEq
function ArrayInterface.zeromatrix(A::AbstractNamedArrayPartition)
	B = ArrayPartition(A)
    x = reduce(vcat,vec.(B.x))
    x .* x' .* false
end

# `x = find_NamedArrayPartition(x)` returns the first `NamedArrayPartition` among broadcast arguments.
find_NamedArrayPartition(bc::Base.Broadcast.Broadcasted) = find_NamedArrayPartition(bc.args)
function find_NamedArrayPartition(args::Tuple)
    find_NamedArrayPartition(find_NamedArrayPartition(args[1]), Base.tail(args))
end
find_NamedArrayPartition(x) = x
find_NamedArrayPartition(::Tuple{}) = nothing
find_NamedArrayPartition(x::AbstractNamedArrayPartition, rest) = x
find_NamedArrayPartition(::Any, rest) = find_NamedArrayPartition(rest)

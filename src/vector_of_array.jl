# Based on code from M. Bauman Stackexchange answer + Gitter discussion
mutable struct VectorOfArray{T, N, A} <: AbstractVectorOfArray{T, N, A}
  u::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
end
# VectorOfArray with an added series for time
mutable struct DiffEqArray{T, N, A, B} <: AbstractDiffEqArray{T, N, A}
  u::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
  t::B
end

Base.Array(VA::AbstractVectorOfArray{T,N,A}) where {T,N,A <: AbstractVector{<:AbstractVector}} = reduce(hcat,VA.u)
Base.Array(VA::AbstractVectorOfArray{T,N,A}) where {T,N,A <: AbstractVector{<:Number}} = VA.u
function Base.Array(VA::AbstractVectorOfArray)
  vecs = vec.(VA.u)
  Array(reshape(reduce(hcat,vecs),size(VA.u[1])...,length(VA.u)))
end

VectorOfArray(vec::AbstractVector{T}, ::NTuple{N}) where {T, N} = VectorOfArray{eltype(T), N, typeof(vec)}(vec)
# Assume that the first element is representative of all other elements
VectorOfArray(vec::AbstractVector) = VectorOfArray(vec, (size(vec[1])..., length(vec)))
VectorOfArray(vec::AbstractVector{VT}) where {T, N, VT<:AbstractArray{T, N}} = VectorOfArray{T, N+1, typeof(vec)}(vec)

DiffEqArray(vec::AbstractVector{T}, ts, ::NTuple{N}) where {T, N} = DiffEqArray{eltype(T), N, typeof(vec), typeof(ts)}(vec, ts)
# Assume that the first element is representative of all other elements
DiffEqArray(vec::AbstractVector,ts::AbstractVector) = DiffEqArray(vec, ts, (size(vec[1])..., length(vec)))
DiffEqArray(vec::AbstractVector{VT},ts::AbstractVector) where {T, N, VT<:AbstractArray{T, N}} = DiffEqArray{T, N+1, typeof(vec), typeof(ts)}(vec, ts)

# Interface for the linear indexing. This is just a view of the underlying nested structure
@inline Base.firstindex(VA::AbstractVectorOfArray) = firstindex(VA.u)
@inline Base.lastindex(VA::AbstractVectorOfArray) = lastindex(VA.u)

@inline Base.length(VA::AbstractVectorOfArray) = length(VA.u)
@inline Base.eachindex(VA::AbstractVectorOfArray) = Base.OneTo(length(VA.u))
@inline Base.IteratorSize(VA::AbstractVectorOfArray) = Base.HasLength()
# Linear indexing will be over the container elements, not the individual elements
# unlike an true AbstractArray
Base.@propagate_inbounds Base.getindex(VA::AbstractVectorOfArray{T, N}, I::Int) where {T, N} = VA.u[I]
Base.@propagate_inbounds Base.getindex(VA::AbstractVectorOfArray{T, N}, I::Colon) where {T, N} = VA.u[I]
Base.@propagate_inbounds Base.getindex(VA::AbstractVectorOfArray{T, N}, I::AbstractArray{Int}) where {T, N} = VectorOfArray(VA.u[I])
Base.@propagate_inbounds Base.getindex(VA::AbstractDiffEqArray{T, N}, I::AbstractArray{Int}) where {T, N} = DiffEqArray(VA.u[I],VA.t[I])
Base.@propagate_inbounds Base.getindex(VA::AbstractVectorOfArray{T, N}, i::Int,::Colon) where {T, N} = [VA.u[j][i] for j in 1:length(VA)]
Base.@propagate_inbounds function Base.getindex(VA::AbstractVectorOfArray{T,N}, ii::CartesianIndex) where {T, N}
    ti = Tuple(ii)
    i = last(ti)
    jj = CartesianIndex(Base.front(ti))
    return VA.u[i][jj]
end
Base.@propagate_inbounds Base.setindex!(VA::AbstractVectorOfArray{T, N}, v, I::Int) where {T, N} = VA.u[I] = v
Base.@propagate_inbounds Base.setindex!(VA::AbstractVectorOfArray{T, N}, v, I::Colon) where {T, N} = VA.u[I] = v
Base.@propagate_inbounds Base.setindex!(VA::AbstractVectorOfArray{T, N}, v, I::AbstractArray{Int}) where {T, N} = VA.u[I] = v
Base.@propagate_inbounds function Base.setindex!(VA::AbstractVectorOfArray{T, N}, v, i::Int,::Colon) where {T, N}
  for j in 1:length(VA)
    VA.u[j][i] = v[j]
  end
  return v
end
Base.@propagate_inbounds function Base.setindex!(VA::AbstractVectorOfArray{T,N}, x, ii::CartesianIndex) where {T, N}
    ti = Tuple(ii)
    i = last(ti)
    jj = CartesianIndex(Base.front(ti))
    return VA.u[i][jj] = x
end

# Interface for the two dimensional indexing, a more standard AbstractArray interface
@inline Base.size(VA::AbstractVectorOfArray) = (size(VA.u[1])..., length(VA.u))
Base.@propagate_inbounds Base.getindex(VA::AbstractVectorOfArray{T, N}, I::Int...) where {T, N} = VA.u[I[end]][Base.front(I)...]
Base.@propagate_inbounds Base.getindex(VA::AbstractVectorOfArray{T, N}, ::Colon, I::Int) where {T, N} = VA.u[I]
Base.@propagate_inbounds Base.setindex!(VA::AbstractVectorOfArray{T, N}, v, I::Int...) where {T, N} = VA.u[I[end]][Base.front(I)...] = v

# The iterator will be over the subarrays of the container, not the individual elements
# unlike an true AbstractArray
function Base.iterate(VA::AbstractVectorOfArray,state=1)
  state >= length(VA.u) + 1 ? nothing : (VA[state], state + 1)
end
tuples(VA::DiffEqArray) = tuple.(VA.t,VA.u)

# Growing the array simply adds to the container vector
Base.copy(VA::AbstractVectorOfArray) = typeof(VA)(copy(VA.u))
Base.sizehint!(VA::AbstractVectorOfArray{T, N}, i) where {T, N} = sizehint!(VA.u, i)
Base.push!(VA::AbstractVectorOfArray{T, N}, new_item::AbstractVector) where {T, N} = push!(VA.u, new_item)

function Base.append!(VA::AbstractVectorOfArray{T, N}, new_item::AbstractVectorOfArray{T, N}) where {T, N}
    for item in copy(new_item)
        push!(VA, item)
    end
    return VA
end

# Tools for creating similar objects
@inline Base.similar(VA::VectorOfArray, ::Type{T} = eltype(VA)) where {T} = VectorOfArray([similar(VA[i], T) for i in eachindex(VA)])
recursivecopy(VA::VectorOfArray) = VectorOfArray(copy.(VA.u))

# fill!
# For DiffEqArray it ignores ts and fills only u
function Base.fill!(VA::AbstractVectorOfArray, x)
    for i in eachindex(VA)
        if VA[i] isa AbstractArray
            fill!(VA[i], x)
        else
            VA[i] = x
        end
    end
    return VA
end

function Base._reshape(parent::VectorOfArray, dims::Base.Dims)
    n = prod(size(parent))
    prod(dims) == n || _throw_dmrs(n, "size", dims)
    Base.__reshape((parent, IndexStyle(parent)), dims)
end

# Need this for ODE_DEFAULT_UNSTABLE_CHECK from DiffEqBase to work properly
@inline Base.any(f, VA::AbstractVectorOfArray) = any(any(f,VA[i]) for i in eachindex(VA))
@inline Base.all(f, VA::AbstractVectorOfArray) = all(all(f,VA[i]) for i in eachindex(VA))
@inline Base.any(f::Function, VA::AbstractVectorOfArray) = any(any(f,VA[i]) for i in eachindex(VA))
@inline Base.all(f::Function, VA::AbstractVectorOfArray) = all(all(f,VA[i]) for i in eachindex(VA))

# conversion tools
vecarr_to_vectors(VA::AbstractVectorOfArray) = [VA[i,:] for i in eachindex(VA[1])]
Base.vec(VA::AbstractVectorOfArray) = vec(convert(Array,VA)) # Allocates

# statistics
@inline Base.sum(f, VA::AbstractVectorOfArray) = sum(f,Array(VA))
@inline Base.sum(VA::AbstractVectorOfArray;kwargs...) = sum(Array(VA);kwargs...)
@inline Base.prod(f, VA::AbstractVectorOfArray) = prod(f,Array(VA))
@inline Base.prod(VA::AbstractVectorOfArray;kwargs...) = prod(Array(VA);kwargs...)

@inline Statistics.mean(VA::AbstractVectorOfArray;kwargs...) = mean(Array(VA);kwargs...)
@inline Statistics.median(VA::AbstractVectorOfArray;kwargs...) = median(Array(VA);kwargs...)
@inline Statistics.std(VA::AbstractVectorOfArray;kwargs...) = std(Array(VA);kwargs...)
@inline Statistics.var(VA::AbstractVectorOfArray;kwargs...) = var(Array(VA);kwargs...)
@inline Statistics.cov(VA::AbstractVectorOfArray;kwargs...) = cov(Array(VA);kwargs...)
@inline Statistics.cor(VA::AbstractVectorOfArray;kwargs...) = cor(Array(VA);kwargs...)

# make it show just like its data
Base.show(io::IO, x::AbstractVectorOfArray) = Base.print_array(io, x.u)
Base.show(io::IO, m::MIME"text/plain", x::AbstractVectorOfArray) = (println(io, summary(x), ':'); show(io, m, x.u))
Base.summary(A::AbstractVectorOfArray) = string("VectorOfArray{",eltype(A),",",ndims(A),"}")

Base.show(io::IO, x::AbstractDiffEqArray) = (print(io,"t: ");show(io, x.t);println(io);print(io,"u: ");show(io, x.u))
Base.show(io::IO, m::MIME"text/plain", x::AbstractDiffEqArray) = (print(io,"t: ");show(io,m,x.t);println(io);print(io,"u: ");show(io,m,x.u))

# plot recipes
@recipe function f(VA::AbstractVectorOfArray)
  convert(Array,VA)
end
@recipe function f(VA::AbstractDiffEqArray)
  VA.t,VA'
end
@recipe function f(VA::DiffEqArray{T,1}) where {T}
  VA.t,VA.u
end

Base.mapreduce(f,op,A::AbstractVectorOfArray) = mapreduce(f,op,(mapreduce(f,op,x) for x in A.u))

## broadcasting

struct VectorOfArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end # N is only used when voa sees other abstract arrays
VectorOfArrayStyle(::Val{N}) where N = VectorOfArrayStyle{N}()

# The order is important here. We want to override Base.Broadcast.DefaultArrayStyle to return another Base.Broadcast.DefaultArrayStyle.
Broadcast.BroadcastStyle(a::VectorOfArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = a
Broadcast.BroadcastStyle(::VectorOfArrayStyle{N}, a::Base.Broadcast.DefaultArrayStyle{M}) where {M,N} = Base.Broadcast.DefaultArrayStyle(Val(max(M, N)))
Broadcast.BroadcastStyle(::VectorOfArrayStyle{N}, a::Base.Broadcast.AbstractArrayStyle{M}) where {M,N} = typeof(a)(Val(max(M, N)))
Broadcast.BroadcastStyle(::VectorOfArrayStyle{M}, ::VectorOfArrayStyle{N}) where {M,N} = VectorOfArrayStyle(Val(max(M, N)))
Broadcast.BroadcastStyle(::Type{<:AbstractVectorOfArray{T,N}}) where {T,N} = VectorOfArrayStyle{N}()

@inline function Base.copy(bc::Broadcast.Broadcasted{<:VectorOfArrayStyle})
    bc = Broadcast.flatten(bc)
    N = narrays(bc)
    VectorOfArray(map(1:N) do i
        copy(unpack_voa(bc, i))
    end)
end

@inline function Base.copyto!(dest::AbstractVectorOfArray, bc::Broadcast.Broadcasted{<:VectorOfArrayStyle})
    bc = Broadcast.flatten(bc)
    N = narrays(bc)
    @inbounds for i in 1:N
        if dest[i] isa AbstractArray
            copyto!(dest[i], unpack_voa(bc, i))
        else
            dest[i] = copy(unpack_voa(bc, i))
        end
    end
    dest
end

## broadcasting utils

"""
    narrays(A...)

Retrieve number of arrays in the AbstractVectorOfArrays of a broadcast
"""
narrays(A) = 0
narrays(A::AbstractVectorOfArray) = length(A)
narrays(bc::Broadcast.Broadcasted) = _narrays(bc.args)
narrays(A, Bs...) = common_length(narrays(A), _narrays(Bs))

common_length(a, b) =
    a == 0 ? b :
    (b == 0 ? a :
     (a == b ? a :
      throw(DimensionMismatch("number of arrays must be equal"))))

_narrays(args::AbstractVectorOfArray) = length(args)
@inline _narrays(args::Tuple) = common_length(narrays(args[1]), _narrays(Base.tail(args)))
_narrays(args::Tuple{Any}) = _narrays(args[1])
_narrays(::Any) = 0

# drop axes because it is easier to recompute
@inline unpack_voa(bc::Broadcast.Broadcasted{Style}, i) where Style = Broadcast.Broadcasted{Style}(bc.f, unpack_args_voa(i, bc.args))
@inline unpack_voa(bc::Broadcast.Broadcasted{<:VectorOfArrayStyle}, i) = Broadcast.Broadcasted(bc.f, unpack_args_voa(i, bc.args))
unpack_voa(x,::Any) = x
unpack_voa(x::AbstractVectorOfArray, i) = x.u[i]
unpack_voa(x::AbstractArray{T,N}, i) where {T,N} = @view x[ntuple(x->Colon(),N-1)...,i]

@inline unpack_args_voa(i, args::Tuple) = (unpack_voa(args[1], i), unpack_args_voa(i, Base.tail(args))...)
unpack_args_voa(i, args::Tuple{Any}) = (unpack_voa(args[1], i),)
unpack_args_voa(::Any, args::Tuple{}) = ()

# Based on code from M. Bauman Stackexchange answer + Gitter discussion
mutable struct VectorOfArray{T, N, A} <: AbstractVectorOfArray{T, N}
  u::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
end
# VectorOfArray with an added series for time
mutable struct DiffEqArray{T, N, A, B} <: AbstractDiffEqArray{T, N}
  u::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
  t::B
end

VectorOfArray(vec::AbstractVector{T}, dims::NTuple{N}) where {T, N} = VectorOfArray{eltype(T), N, typeof(vec)}(vec)
# Assume that the first element is representative all all other elements
VectorOfArray(vec::AbstractVector) = VectorOfArray(vec, (size(vec[1])..., length(vec)))

DiffEqArray(vec::AbstractVector{T}, ts, dims::NTuple{N}) where {T, N} = DiffEqArray{eltype(T), N, typeof(vec), typeof(ts)}(vec, ts)
# Assume that the first element is representative all all other elements
DiffEqArray(vec::AbstractVector,ts::AbstractVector) = DiffEqArray(vec, ts, (size(vec[1])..., length(vec)))


# Interface for the linear indexing. This is just a view of the underlying nested structure
@inline Base.firstindex(VA::AbstractVectorOfArray) = firstindex(VA.u)
@inline Base.lastindex(VA::AbstractVectorOfArray) = lastindex(VA.u)

@inline Base.length(VA::AbstractVectorOfArray) = length(VA.u)
@inline Base.eachindex(VA::AbstractVectorOfArray) = Base.OneTo(length(VA.u))
@inline Base.IteratorSize(VA::AbstractVectorOfArray) = Base.HasLength()
# Linear indexing will be over the container elements, not the individual elements
# unlike an true AbstractArray
@inline Base.getindex(VA::AbstractVectorOfArray{T, N}, I::Int) where {T, N} = VA.u[I]
@inline Base.getindex(VA::AbstractVectorOfArray{T, N}, I::Colon) where {T, N} = VA.u[I]
@inline Base.getindex(VA::AbstractVectorOfArray{T, N}, I::AbstractArray{Int}) where {T, N} = VectorOfArray(VA.u[I])
@inline Base.getindex(VA::AbstractDiffEqArray{T, N}, I::AbstractArray{Int}) where {T, N} = DiffEqArray(VA.u[I],VA.t[I])
@inline Base.getindex(VA::AbstractVectorOfArray{T, N}, i::Int,::Colon) where {T, N} = [VA.u[j][i] for j in 1:length(VA)]
@inline Base.setindex!(VA::AbstractVectorOfArray{T, N}, v, I::Int) where {T, N} = VA.u[I] = v
@inline Base.setindex!(VA::AbstractVectorOfArray{T, N}, v, I::Colon) where {T, N} = VA.u[I] = v
@inline Base.setindex!(VA::AbstractVectorOfArray{T, N}, v, I::AbstractArray{Int}) where {T, N} = VA.u[I] = v
@inline function Base.setindex!(VA::AbstractVectorOfArray{T, N}, v, i::Int,::Colon) where {T, N}
  for j in 1:length(VA)
    VA.u[j][i] = v[j]
  end
end

# Interface for the two dimensional indexing, a more standard AbstractArray interface
@inline Base.size(VA::AbstractVectorOfArray) = (size(VA.u[1])..., length(VA.u))
@inline Base.getindex(VA::AbstractVectorOfArray{T, N}, I::Int...) where {T, N} = VA.u[I[end]][Base.front(I)...]
@inline Base.getindex(VA::AbstractVectorOfArray{T, N}, ::Colon, I::Int) where {T, N} = VA.u[I]
@inline Base.setindex!(VA::AbstractVectorOfArray{T, N}, v, I::Int...) where {T, N} = VA.u[I[end]][Base.front(I)...] = v

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
        fill!(VA[i], x)
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

# make it show just like its data
Base.show(io::IO, x::AbstractVectorOfArray) = show(io, x.u)
Base.show(io::IO, m::MIME"text/plain", x::AbstractVectorOfArray) = show(io, m, x.u)
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

# Broadcast

#add_idxs(x,expr) = expr
#add_idxs{T<:AbstractVectorOfArray}(::Type{T},expr) = :($(expr)[i])
#add_idxs{T<:AbstractArray}(::Type{Vector{T}},expr) = :($(expr)[i])
#=
@generated function Base.broadcast!(f,A::AbstractVectorOfArray,B...)
  exs = ((add_idxs(B[i],:(B[$i])) for i in eachindex(B))...)
  :(for i in eachindex(A)
    broadcast!(f,A[i],$(exs...))
  end)
end

@generated function Base.broadcast(f,B::Union{Number,AbstractVectorOfArray}...)
  arr_idx = 0
  for (i,b) in enumerate(B)
    if b <: ArrayPartition
      arr_idx = i
      break
    end
  end
  :(A = similar(B[$arr_idx]); broadcast!(f,A,B...); A)
end
=#

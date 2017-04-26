abstract AbstractVectorOfArray{T, N} <: AbstractArray{T, N}

# Based on code from M. Bauman Stackexchange answer + Gitter discussion
type VectorOfArray{T, N, A} <: AbstractVectorOfArray{T, N}
  u::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
end

VectorOfArray{T, N}(vec::AbstractVector{T}, dims::NTuple{N}) = VectorOfArray{eltype(T), N, typeof(vec)}(vec)
# Assume that the first element is representative all all other elements
VectorOfArray(vec::AbstractVector) = VectorOfArray(vec, (size(vec[1])..., length(vec)))

# Interface for the linear indexing. This is just a view of the underlying nested structure
@inline Base.endof(VA::AbstractVectorOfArray) = endof(VA.u)
@inline Base.length(VA::AbstractVectorOfArray) = length(VA.u)
# Linear indexing will be over the container elements, not the individual elements
# unlike an true AbstractArray
@inline Base.getindex{T, N}(VA::AbstractVectorOfArray{T, N}, I::Int) = VA.u[I]
@inline Base.getindex{T, N}(VA::AbstractVectorOfArray{T, N}, I::Colon) = VA.u[I]
@inline Base.getindex{T, N}(VA::AbstractVectorOfArray{T, N}, I::AbstractArray{Int}) = VA.u[I]

# Interface for the two dimensional indexing, a more standard AbstractArray interface
@inline Base.size(VA::AbstractVectorOfArray) = (size(VA.u[1])..., length(VA.u))
@inline Base.getindex{T, N}(VA::AbstractVectorOfArray{T, N}, I::Vararg{Int, N}) = VA.u[I[end]][Base.front(I)...]

# The iterator will be over the subarrays of the container, not the individual elements
# unlike an true AbstractArray
Base.start{T, N}(VA::AbstractVectorOfArray{T, N}) = 1
Base.next{T, N}(VA::AbstractVectorOfArray{T, N}, state) = (VA[state], state + 1)
Base.done{T, N}(VA::AbstractVectorOfArray{T, N}, state) = state >= length(VA.u) + 1

# Growing the array simply adds to the container vector
Base.copy(VA::AbstractVectorOfArray) = typeof(VA)(copy(VA.u))
Base.sizehint!{T, N}(VA::AbstractVectorOfArray{T, N}, i) = sizehint!(VA.u, i)
Base.push!{T, N}(VA::AbstractVectorOfArray{T, N}, new_item::AbstractVector) = push!(VA.u, new_item)

function Base.append!{T, N}(VA::AbstractVectorOfArray{T, N}, new_item::AbstractVectorOfArray{T, N})
    for item in copy(new_item)
        push!(VA, item)
    end
    return VA
end

# conversion tools
vecarr_to_arr(VA::AbstractVectorOfArray) = cat(length(size(VA)), VA.u...)

abstract AbstractVectorOfArray{T, N} <: AbstractArray{T, N}

# Based on code from M. Bauman Stackexchange answer + Gitter discussion
type VectorOfArray{T, N, A} <: AbstractVectorOfArray{T, N}
  data::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
  dims::NTuple{N, Int}
  ragged::Bool
end

function VectorOfArray(vec::AbstractVector)
  # Only allow a vector of equally shaped subtypes, this will not work for ragged arrays
  ragged = !all(size(vec[1]) == size(v) for v in vec)
  VectorOfArray(vec, (size(vec[1])..., length(vec)), ragged)
end

VectorOfArray{T, N}(vec::AbstractVector{T}, dims::NTuple{N}, ragged::Bool) = VectorOfArray{eltype(T), N, typeof(vec)}(vec, dims, ragged)
@inline function Base.size(S::AbstractVectorOfArray)
  if S.ragged
    return size(S.data)
  else
    return S.dims
  end
end

@inline function Base.getindex{T, N}(S::AbstractVectorOfArray{T, N}, I::Vararg{Int, N})
  @boundscheck checkbounds(S, I...)
  S.ragged && throw(BoundsError("A ragged VectorOfArray does not support Cartesian indexing"))
  S.data[I[end]][Base.front(I)...]
end
# Linear indexing will be over the container elements, not the individual elements
# unlike an true AbstractArray
@inline Base.getindex{T, N}(S::AbstractVectorOfArray{T, N}, I::Union{Int, Colon, AbstractArray{Int}}) = S.data[I]

Base.copy(S::AbstractVectorOfArray) = VectorOfArray(copy(S.data), S.dims, S.ragged)

Base.sizehint!{T, N}(S::AbstractVectorOfArray{T, N}, i) = sizehint!(S.data, i)

function Base.push!{T, N}(S::AbstractVectorOfArray{T, N}, new_item::AbstractArray)
  if S.dims[1:(end - 1)] == size(new_item)
    S.dims = (S.dims[1:(end - 1)]..., S.dims[end] + 1)
  else
    S.ragged = true
    # just revert down to a vector of elements, make the dims data meaningless
    #TODO: this is stupid ugly
    S.dims = tuple(-ones(Int, N)...)
  end
  push!(S.data, copy(new_item))
end
function Base.append!{T, N}(S::AbstractVectorOfArray{T, N}, new_item::AbstractVectorOfArray{T, N})
    for item in copy(new_item)
      push!(S, item)
    end
    return S
end

# The iterator will be over the subarrays of the container, not the individual elements
# unlike an true AbstractArray
Base.start{T, N}(S::AbstractVectorOfArray{T, N}) = 1
Base.next{T, N}(S::AbstractVectorOfArray{T, N}, state) = (S[state], state + 1)
Base.done{T, N}(S::AbstractVectorOfArray{T, N}, state) = state >= length(S.data) + 1

# convert to a regular, sense array
function vecarr_to_arr(va::AbstractVectorOfArray)
  va[[Colon() for d in 1:length(va.dims)]...]
end

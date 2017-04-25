# Based on code from M. Bauman Stackexchange answer + Gitter discussion

type VectorOfArray{T, N, A} <: AbstractArray{T, N}
  data::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
  #TODO: I don't really use dims, how do I get rid of it, and still have everything work?
  dims::NTuple{N, Int}
end

function VectorOfArray(vec::AbstractVector)
    # Only allow a vector of equally shaped subtypes, this will not work for ragged arrays
    #@assert all(size(vec[1]) == size(v) for v in vec)
    VectorOfArray(vec, (size(vec[1])..., length(vec)))
end

VectorOfArray{T, N}(vec::AbstractVector{T}, dims::NTuple{N}) = VectorOfArray{eltype(T), N, typeof(vec)}(vec, dims)

Base.endof(S::VectorOfArray) = endof(S.data)
Base.size(S::VectorOfArray) = (size(S.data[1])..., length(S.data))

@inline function Base.getindex{T, N}(S::VectorOfArray{T, N}, I::Vararg{Int, N})
    @boundscheck checkbounds(S, I...) # is this needed?
    S.data[I[end]][Base.front(I)...]
end
# Linear indexing will be over the container elements, not the individual elements
# unlike an true AbstractArray
@inline Base.getindex{T, N}(S::VectorOfArray{T, N}, I::Int) = S.data[I]
@inline Base.getindex{T, N}(S::VectorOfArray{T, N}, I::AbstractArray{Int}) = S.data[I]

Base.copy(S::VectorOfArray) = VectorOfArray(copy(S.data), S.dims)

Base.sizehint!{T, N}(S::VectorOfArray{T, N}, i) = sizehint!(S.data, i)

function Base.push!{T, N}(S::VectorOfArray{T, N}, new_item::AbstractVector)
    #@assert S.dims[1:(end - 1)] == size(new_item)
    #S.dims = (S.dims[1:(end - 1)]..., S.dims[end] + 1)
    push!(S.data, new_item)
end

function Base.append!{T, N}(S::VectorOfArray{T, N}, new_item::VectorOfArray{T, N})
    for item in copy(new_item)
        push!(S, item)
    end
    return S
end


# The iterator will be over the subarrays of the container, not the individual elements
# unlike an true AbstractArray
Base.start{T, N}(S::VectorOfArray{T, N}) = 1
Base.next{T, N}(S::VectorOfArray{T, N}, state) = (S[state], state + 1)
Base.done{T, N}(S::VectorOfArray{T, N}, state) = state >= length(S.data) + 1

# conversion tools
function vecarr_to_arr(S::VectorOfArray)
  return cat(length(size(S)), S.data...)
end

# Based on code from M. Bauman Stackexchange answer + Gitter discussion

type VectorOfArray{T, N, A} <: AbstractArray{T, N}
  data::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
end

VectorOfArray{T, N}(vec::AbstractVector{T}, dims::NTuple{N}) = VectorOfArray{eltype(T), N, typeof(vec)}(vec)
# Assume that the first element is representative all all other elements
VectorOfArray(vec::AbstractVector) = VectorOfArray(vec, (size(vec[1])..., length(vec)))


Base.endof(VA::VectorOfArray) = endof(VA.data)
Base.size(VA::VectorOfArray) = (size(VA.data[1])..., length(VA.data))
#TODO: should we redefine length to be over the VA.data? Currently it is the number of total elements

@inline function Base.getindex{T, N}(VA::VectorOfArray{T, N}, I::Vararg{Int, N})
    VA.data[I[end]][Base.front(I)...]
end
# Linear indexing will be over the container elements, not the individual elements
# unlike an true AbstractArray
@inline Base.getindex{T, N}(VA::VectorOfArray{T, N}, I::Int) = VA.data[I]
@inline Base.getindex{T, N}(VA::VectorOfArray{T, N}, I::Colon) = VA.data[I]
@inline Base.getindex{T, N}(VA::VectorOfArray{T, N}, I::AbstractArray{Int}) = VA.data[I]

Base.copy(VA::VectorOfArray) = VectorOfArray(copy(VA.data), size(VA))

Base.sizehint!{T, N}(VA::VectorOfArray{T, N}, i) = sizehint!(VA.data, i)

Base.push!{T, N}(VA::VectorOfArray{T, N}, new_item::AbstractVector) = push!(VA.data, new_item)

function Base.append!{T, N}(VA::VectorOfArray{T, N}, new_item::VectorOfArray{T, N})
    for item in copy(new_item)
        push!(VA, item)
    end
    return VA
end


# The iterator will be over the subarrays of the container, not the individual elements
# unlike an true AbstractArray
Base.start{T, N}(VA::VectorOfArray{T, N}) = 1
Base.next{T, N}(VA::VectorOfArray{T, N}, state) = (VA[state], state + 1)
Base.done{T, N}(VA::VectorOfArray{T, N}, state) = state >= length(VA.data) + 1

# conversion tools
vecarr_to_arr(VA::VectorOfArray) = cat(length(size(VA)), VA.data...)

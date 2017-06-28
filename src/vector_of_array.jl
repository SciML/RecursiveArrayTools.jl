# Based on code from M. Bauman Stackexchange answer + Gitter discussion
type VectorOfArray{T, N, A} <: AbstractVectorOfArray{T, N}
  u::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
end
# VectorOfArray with an added series for time
type DiffEqArray{T, N, A, B} <: AbstractDiffEqArray{T, N}
  u::A # A <: AbstractVector{<: AbstractArray{T, N - 1}}
  t::B
end

VectorOfArray{T, N}(vec::AbstractVector{T}, dims::NTuple{N}) = VectorOfArray{eltype(T), N, typeof(vec)}(vec)
# Assume that the first element is representative all all other elements
VectorOfArray(vec::AbstractVector) = VectorOfArray(vec, (size(vec[1])..., length(vec)))

DiffEqArray{T, N}(vec::AbstractVector{T}, ts, dims::NTuple{N}) = DiffEqArray{eltype(T), N, typeof(vec), typeof(ts)}(vec, ts)
# Assume that the first element is representative all all other elements
DiffEqArray(vec::AbstractVector,ts::AbstractVector) = DiffEqArray(vec, ts, (size(vec[1])..., length(vec)))


# Interface for the linear indexing. This is just a view of the underlying nested structure
@inline Base.endof(VA::AbstractVectorOfArray) = endof(VA.u)
@inline Base.length(VA::AbstractVectorOfArray) = length(VA.u)
@inline Base.eachindex(VA::AbstractVectorOfArray) = Base.OneTo(length(VA.u))
# Linear indexing will be over the container elements, not the individual elements
# unlike an true AbstractArray
@inline Base.getindex{T, N}(VA::AbstractVectorOfArray{T, N}, I::Int) = VA.u[I]
@inline Base.getindex{T, N}(VA::AbstractVectorOfArray{T, N}, I::Colon) = VA.u[I]
@inline Base.getindex{T, N}(VA::AbstractVectorOfArray{T, N}, I::AbstractArray{Int}) = VA.u[I]
@inline Base.getindex{T, N}(VA::AbstractVectorOfArray{T, N}, i::Int,::Colon) = [VA.u[j][i] for j in 1:length(VA)]

# Interface for the two dimensional indexing, a more standard AbstractArray interface
@inline Base.size(VA::AbstractVectorOfArray) = (size(VA.u[1])..., length(VA.u))
@inline Base.getindex{T, N}(VA::AbstractVectorOfArray{T, N}, I::Int...) = VA.u[I[end]][Base.front(I)...]

# The iterator will be over the subarrays of the container, not the individual elements
# unlike an true AbstractArray
Base.start{T, N}(VA::AbstractVectorOfArray{T, N}) = 1
Base.next{T, N}(VA::AbstractVectorOfArray{T, N}, state) = (VA[state], state + 1)
Base.done{T, N}(VA::AbstractVectorOfArray{T, N}, state) = state >= length(VA.u) + 1
tuples(VA::DiffEqArray) = tuple.(VA.t,VA.u)

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
@deprecate vecarr_to_arr(VA::AbstractVectorOfArray) convert(Array,VA)
@deprecate vecarr_to_arr{T<:AbstractArray}(VA::Vector{T}) convert(Array,VA)
vecarr_to_vectors(VA::AbstractVectorOfArray) = [VA[i,:] for i in eachindex(VA[1])]

# make it show just like its data
Base.show(io::IO, x::AbstractVectorOfArray) = show(io, x.u)
Base.show(io::IO, m::MIME"text/plain", x::AbstractVectorOfArray) = show(io, m, x.u)
Base.summary(A::AbstractVectorOfArray) = string("VectorOfArray{",eltype(A),",",ndims(A),"}")
Base.display(A::AbstractVectorOfArray) = (println(summary(A));println("u:");display(A.u);nothing)
Base.print(A::AbstractVectorOfArray) = show(A)
Base.println(A::AbstractVectorOfArray) = show(A)

Base.show(io::IO, x::AbstractDiffEqArray) = (print(io,"t: ");show(io, x.t);println(io);print(io,"u: ");show(io, x.u))
Base.show(io::IO, m::MIME"text/plain", x::AbstractDiffEqArray) = (print(io,"t: ");show(io,m,x.t);println(io);print(io,"u: ");show(io,m,x.u))
Base.display(A::AbstractDiffEqArray) = (println(summary(A));println("t:");display(A.t);println("u:");display(A.u);nothing)

# restore the type rendering in Juno
Juno.@render Juno.Inline x::AbstractVectorOfArray begin
  fields = fieldnames(typeof(x))
  Juno.LazyTree(typeof(x), () -> [Juno.SubTree(Juno.Text("$f → "), Juno.getfield′(x, f)) for f in fields])
end

# plot recipes
@recipe function f(VA::AbstractVectorOfArray)
  convert(Array,VA)
end
@recipe function f(VA::AbstractDiffEqArray)
  A = convert(Array,VA)
  VA.t,A
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

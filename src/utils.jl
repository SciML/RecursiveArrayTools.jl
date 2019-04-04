function recursivecopy(a)
  deepcopy(a)
end
recursivecopy(a::Union{SVector,SMatrix,SArray,Number}) = copy(a)
function recursivecopy(a::AbstractArray{T,N}) where {T<:Number,N}
  copy(a)
end

function recursivecopy(a::AbstractArray{T,N}) where {T<:AbstractArray,N}
  map(recursivecopy,a)
end

function recursivecopy!(b::AbstractArray{T,N},a::AbstractArray{T2,N}) where {T<:StaticArray,T2<:StaticArray,N}
  @inbounds for i in eachindex(a)
    # TODO: Check for `setindex!`` and use `copy!(b[i],a[i])` or `b[i] = a[i]`, see #19
    b[i] = copy(a[i])
  end
end

function recursivecopy!(b::AbstractArray{T,N},a::AbstractArray{T2,N}) where {T<:Number,T2<:Number,N}
  copyto!(b,a)
end

function recursivecopy!(b::AbstractArray{T,N},a::AbstractArray{T2,N}) where {T<:AbstractArray,T2<:AbstractArray,N}
  @inbounds for i in eachindex(a)
    recursivecopy!(b[i],a[i])
  end
end

function vecvec_to_mat(vecvec)
  mat = Matrix{eltype(eltype(vecvec))}(undef, length(vecvec),length(vecvec[1]))
  for i in 1:length(vecvec)
    mat[i,:] = vecvec[i]
  end
  mat
end


function vecvecapply(f,v)
  sol = Vector{eltype(eltype(v))}()
  for i in eachindex(v)
    for j in eachindex(v[i])
      push!(sol,v[i][j])
    end
  end
  f(sol)
end

function vecvecapply(f,v::Array{T}) where T<:Number
  f(v)
end

function vecvecapply(f,v::T) where T<:Number
  f(v)
end

function copyat_or_push!(a::AbstractVector{T},i::Int,x,nc::Type{Val{perform_copy}}=Val{true}) where {T,perform_copy}
  @inbounds if length(a) >= i
    if !ArrayInterface.ismutable(T) || !perform_copy
      # TODO: Check for `setindex!`` if T <: StaticArray and use `copy!(b[i],a[i])`
      #       or `b[i] = a[i]`, see https://github.com/JuliaDiffEq/RecursiveArrayTools.jl/issues/19
      a[i] = x
    else
      recursivecopy!(a[i],x)
    end
  else
      if perform_copy
        push!(a,recursivecopy(x))
      else
        push!(a,x)
      end
  end
  nothing
end

recursive_one(a) = recursive_one(a[1])
recursive_one(a::T) where {T<:Number} = one(a)

recursive_bottom_eltype(a) = recursive_bottom_eltype(eltype(a))
recursive_bottom_eltype(a::Type{T}) where {T<:Number} = eltype(a)

recursive_unitless_bottom_eltype(a) = recursive_unitless_bottom_eltype(typeof(a))
recursive_unitless_bottom_eltype(a::Type{T}) where T = recursive_unitless_bottom_eltype(eltype(a))
recursive_unitless_bottom_eltype(a::Type{T}) where {T<:AbstractArray} = recursive_unitless_bottom_eltype(eltype(a))
recursive_unitless_bottom_eltype(a::Type{T}) where {T<:Number} = eltype(a) == Number ? Float64 : typeof(one(eltype(a)))

recursive_unitless_eltype(a) = recursive_unitless_eltype(eltype(a))
recursive_unitless_eltype(a::Type{T}) where {T<:StaticArray} = similar_type(a,recursive_unitless_eltype(eltype(a)))
recursive_unitless_eltype(a::Type{T}) where {T<:Array} = Array{recursive_unitless_eltype(eltype(a)),ndims(a)}
recursive_unitless_eltype(a::Type{T}) where {T<:Number} = typeof(one(eltype(a)))

recursive_mean(x...) = mean(x...)
function recursive_mean(vecvec::Vector{T}) where T<:AbstractArray
  out = zero(vecvec[1])
  for i in eachindex(vecvec)
    out+= vecvec[i]
  end
  out/length(vecvec)
end

# From Iterators.jl. Moved here since Iterators.jl is not precompile safe anymore.

# Concatenate the output of n iterators
struct Chain{T<:Tuple}
    xss::T
end

# iteratorsize method defined at bottom because of how @generated functions work in 0.6 now

"""
    chain(xs...)

Iterate through any number of iterators in sequence.
```jldoctest
julia> for i in chain(1:3, ['a', 'b', 'c'])
           @show i
       end
i = 1
i = 2
i = 3
i = 'a'
i = 'b'
i = 'c'
```
"""
chain(xss...) = Chain(xss)

Base.length(it::Chain{Tuple{}}) = 0
Base.length(it::Chain) = sum(length, it.xss)

Base.eltype(::Type{Chain{T}}) where {T} = typejoin([eltype(t) for t in T.parameters]...)

function Base.iterate(it::Chain)
    i = 1
    xs_state = nothing
    while i <= length(it.xss)
        xs_state = iterate(it.xss[i])
        xs_state !== nothing && return xs_state[1], (i, xs_state[2])
        i += 1
    end
    return nothing
end

function Base.iterate(it::Chain, state)
    i, xs_state = state
    xs_state = iterate(it.xss[i], xs_state)
    while xs_state == nothing
        i += 1
        i > length(it.xss) && return nothing
        xs_state = iterate(it.xss[i])
    end
    return xs_state[1], (i, xs_state[2])
end

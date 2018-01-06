"""
    is_mutable_type(x::DataType)

Query whether a type is mutable or not, see
https://github.com/JuliaDiffEq/RecursiveArrayTools.jl/issues/19.
"""
Base.@pure is_mutable_type(x::DataType) = x.mutable


function recursivecopy(a::AbstractArray{T,N}) where {T<:Number,N}
  copy(a)
end

function recursivecopy(a::AbstractArray{T,N}) where {T<:AbstractArray,N}
  [recursivecopy(x) for x in a]
end

function recursivecopy!(b::AbstractArray{T,N},a::AbstractArray{T2,N}) where {T<:StaticArray,T2<:StaticArray,N}
  @inbounds for i in eachindex(a)
    # TODO: Check for `setindex!`` and use `copy!(b[i],a[i])` or `b[i] = a[i]`, see #19
    b[i] = copy(a[i])
  end
end

function recursivecopy!(b::AbstractArray{T,N},a::AbstractArray{T2,N}) where {T<:Number,T2<:Number,N}
  copy!(b,a)
end

function recursivecopy!(b::AbstractArray{T,N},a::AbstractArray{T2,N}) where {T<:AbstractArray,T2<:AbstractArray,N}
  @inbounds for i in eachindex(a)
    recursivecopy!(b[i],a[i])
  end
end

function vecvec_to_mat(vecvec)
  mat = Matrix{eltype(eltype(vecvec))}(length(vecvec),length(vecvec[1]))
  for i in 1:length(vecvec)
    mat[i,:] = vecvec[i]
  end
  mat
end


function vecvecapply(f,v)
  sol = Vector{eltype(eltype(v))}(0)
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

@inline function copyat_or_push!(a::AbstractVector{T},i::Int,x,nc::Type{Val{perform_copy}}=Val{true}) where {T,perform_copy}
  @inbounds if length(a) >= i
    if T <: Number || T <: SArray || (T <: FieldVector && !is_mutable_type(T)) || !perform_copy
      # TODO: Check for `setindex!`` if T <: StaticArray and use `copy!(b[i],a[i])`
      #       or `b[i] = a[i]`, see https://github.com/JuliaDiffEq/RecursiveArrayTools.jl/issues/19
      a[i] = x
    else
      recursivecopy!(a[i],x)
    end
  else
    if eltype(x) <: Number && (typeof(x) <: Array || typeof(x) <: Number)
      # Have to check that it's <: Array or can have problems
      # with abstract arrays like MultiScaleModels.
      # Have to check <: Number since it could just be a number...
      if perform_copy
        push!(a,copy(x))
      else
        push!(a,x)
      end
    else
      if perform_copy
        if typeof(x) <: Vector && !(eltype(x) <: Number)
          push!(a,recursivecopy(x))
        elseif typeof(x) <: ArrayPartition || typeof(x) <: AbstractVectorOfArray
          push!(a,copy(x))
        elseif typeof(x) <: SArray
          push!(a,x)
        else
          push!(a,deepcopy(x))
        end
      else
        push!(a,x)
      end
    end
  end
  nothing
end

recursive_one(a) = recursive_one(a[1])
recursive_one(a::T) where {T<:Number} = one(a)

recursive_bottom_eltype(a) = recursive_bottom_eltype(eltype(a))
recursive_bottom_eltype(a::Type{T}) where {T<:Number} = eltype(a)

recursive_unitless_bottom_eltype(a) = recursive_unitless_bottom_eltype(eltype(a))
recursive_unitless_bottom_eltype(a::Type{T}) where {T<:Number} = typeof(one(eltype(a)))

Base.@pure recursive_unitless_eltype(a) = recursive_unitless_eltype(eltype(a))
Base.@pure recursive_unitless_eltype{T<:StaticArray}(a::Type{T}) = similar_type(a,recursive_unitless_eltype(eltype(a)))
Base.@pure recursive_unitless_eltype{T<:Array}(a::Type{T}) = Array{recursive_unitless_eltype(eltype(a)),ndims(a)}
Base.@pure recursive_unitless_eltype{T<:Number}(a::Type{T}) = typeof(one(eltype(a)))

recursive_mean(x...) = mean(x...)
function recursive_mean(vecvec::Vector{T}) where T<:AbstractArray
  out = zeros(vecvec[1])
  for i in eachindex(vecvec)
    out+= vecvec[i]
  end
  out/length(vecvec)
end

function recursive_mean(matarr::Matrix{T},region=0) where T<:AbstractArray
  if region == 0
    return recursive_mean(vec(matarr))
  elseif region == 1
    out = [zeros(matarr[1,i]) for i in 1:size(matarr,2)]
    for j in 1:size(matarr,2), i in 1:size(matarr,1)
      out[j] += matarr[i,j]
    end
    return out/size(matarr,1)
  elseif region == 2
    return recursive_mean(matarr',1)
  end
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

function Base.start(it::Chain)
    i = 1
    xs_state = nothing
    while i <= length(it.xss)
        xs_state = start(it.xss[i])
        if !done(it.xss[i], xs_state)
            break
        end
        i += 1
    end
    return i, xs_state
end

function Base.next(it::Chain, state)
    i, xs_state = state
    v, xs_state = next(it.xss[i], xs_state)
    while done(it.xss[i], xs_state)
        i += 1
        if i > length(it.xss)
            break
        end
        xs_state = start(it.xss[i])
    end
    return v, (i, xs_state)
end

Base.done(it::Chain, state) = state[1] > length(it.xss)

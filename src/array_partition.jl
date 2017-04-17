immutable ArrayPartition{T}
  x::T
end
ArrayPartition(x...) = ArrayPartition((x...))
function ArrayPartition{T}(x,::Type{Val{T}}=Val{false})
  if T
    return ArrayPartition(((copy(a) for a in x)...))
  else
    return ArrayPartition((x...))
  end
end
Base.similar(A::ArrayPartition) = ArrayPartition((similar(x) for x in A.x)...)
Base.similar(A::ArrayPartition,dims::Tuple) = ArrayPartition((similar(x,dim) for (x,dim) in zip(A.x,dims))...)
Base.copy(A::ArrayPartition) = Base.similar(A)
Base.zeros(A::ArrayPartition) = ArrayPartition((zeros(x) for x in A.x)...)

Base.:*(A::Number, B::ArrayPartition) = ArrayPartition((A .* x for x in B.x)...)
Base.:*(A::ArrayPartition, B::Number) = ArrayPartition((x .* B for x in A.x)...)
Base.:/(A::ArrayPartition, B::Number) = ArrayPartition((x ./ B for x in A.x)...)
Base.:\(A::Number, B::ArrayPartition) = ArrayPartition((x ./ A for x in B.x)...)

@inline function Base.getindex( A::ArrayPartition,i::Int)
  @boundscheck i > length(A) && throw(BoundsError("Index out of bounds"))
  @inbounds for j in 1:length(A.x)
    i -= length(A.x[j])
    if i <= 0
      return A.x[j][length(A.x[j])+i]
    end
  end
end
Base.getindex( A::ArrayPartition,::Colon) = [A[i] for i in 1:length(A)]
@inline function Base.setindex!(A::ArrayPartition, v, i::Int)
  @boundscheck i > length(A) && throw(BoundsError("Index out of bounds"))
  @inbounds for j in 1:length(A.x)
    i -= length(A.x[j])
    if i <= 0
      A.x[j][length(A.x[j])+i] = v
      break
    end
  end
end
Base.getindex( A::ArrayPartition,    i::Int...) = A.x[i[1]][Base.tail(i)...]
Base.setindex!(A::ArrayPartition, v, i::Int...) = A.x[i[1]][Base.tail(i)...]=v

function recursivecopy!(A::ArrayPartition,B::ArrayPartition)
  for (a,b) in zip(A.x,B.x)
    copy!(a,b)
  end
end

recursive_one(A::ArrayPartition) = recursive_one(first(A.x))
Base.zero(A::ArrayPartition) = zero(first(A.x))
Base.first(A::ArrayPartition) = first(A.x)

Base.start(A::ArrayPartition) = chain(A.x...)
Base.next(iter::ArrayPartition,state) = next(state,state)
Base.done(iter::ArrayPartition,state) = done(state,state)

Base.length(A::ArrayPartition) = sum((length(x) for x in A.x))
Base.size(A::ArrayPartition) = (length(A),)
Base.indices(A::ArrayPartition) = ((indices(x) for x in A.x)...)
Base.eachindex(A::ArrayPartition) = Base.OneTo(length(A))

add_idxs(x,expr) = expr
add_idxs{T<:ArrayPartition}(::Type{T},expr) = :($(expr).x[i])

@generated function Base.broadcast!(f,A::ArrayPartition,B...)
  exs = ((add_idxs(B[i],:(B[$i])) for i in eachindex(B))...)
  res = :(for i in eachindex(A.x)
    broadcast!(f,A.x[i],$(exs...))
  end)
  res
end

@generated function Base.broadcast(f,B::Union{Number,ArrayPartition}...)
  exs = ((add_idxs(B[i],:(B[$i])) for i in eachindex(B))...)
  res = :(for i in eachindex(A.x)
    broadcast(f,$(exs...))
  end)
  res
end

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

Base.getindex( A::ArrayPartition,    i::Int) = ArrayPartition((x[i] for x in A.x)...)
Base.setindex!(A::ArrayPartition, v, i::Int) = ArrayPartition((x[i]=v for x in A.x)...)
Base.getindex( A::ArrayPartition,    i::Int...) = ArrayPartition((x[i...] for x in A.x)...)
Base.setindex!(A::ArrayPartition, v, i::Int...) = ArrayPartition((x[i...]=v for x in A.x)...)

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

Base.length(A::ArrayPartition) = ((length(x) for x in A.x)...)
Base.indices(A::ArrayPartition) = ((indices(x) for x in A.x)...)
Base.eachindex(A::ArrayPartition) = ((indices(x) for x in A.x)...)

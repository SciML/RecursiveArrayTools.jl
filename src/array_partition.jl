struct ArrayPartition{T,S<:Tuple} <: AbstractVector{T}
  x::S
end

## constructors

ArrayPartition(x...) = ArrayPartition((x...))

function ArrayPartition(x::S, ::Type{Val{copy_x}}=Val{false}) where {S<:Tuple,copy_x}
  T = promote_type(eltype.(x)...)
  if copy_x
    return ArrayPartition{T,S}(copy.(x))
  else
    return ArrayPartition{T,S}(x)
  end
end

## similar array partitions

Base.similar(A::ArrayPartition{T,S}) where {T,S} = ArrayPartition{T,S}(similar.(A.x))

# ignore dims since array partitions are vectors
Base.similar(A::ArrayPartition, dims::NTuple{N,Int}) where {N} = similar(A)

# similar array partition of common type
@generated function Base.similar(A::ArrayPartition, ::Type{T}) where {T}
    N = npartitions(A)
    expr = :(similar(A.x[i], T))

    build_arraypartition(N, expr)
end

# ignore dims since array partitions are vectors
Base.similar(A::ArrayPartition, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = similar(A, T)

# similar array partition with different types
@generated function Base.similar(A::ArrayPartition, ::Type{T}, ::Type{S},
                                 R::Vararg{Type}) where {T,S}
    N = npartitions(A)
    N != length(R) + 2 &&
        throw(DimensionMismatch("number of types must be equal to number of partitions"))

    types = (T, S, parameter.(R)...) # new types
    expr = :(similar(A.x[i], ($types)[i]))

    build_arraypartition(N, expr)
end

Base.copy(A::ArrayPartition{T,S}) where {T,S} = ArrayPartition{T,S}(copy.(A.x))

## zeros

Base.zeros(A::ArrayPartition{T,S}) where {T,S} = ArrayPartition{T,S}(zeros.(A.x))

# ignore dims since array partitions are vectors
Base.zeros(A::ArrayPartition, dims::NTuple{N,Int}) where {N} = zeros(A)

## ones

# special to work with units
@generated function Base.ones(A::ArrayPartition)
    N = npartitions(A)

    expr = :(fill!(similar(A.x[i]), oneunit(eltype(A.x[i]))))

    build_arraypartition(N, expr)
end

# ignore dims since array partitions are vectors
Base.ones(A::ArrayPartition, dims::NTuple{N,Int}) where {N} = ones(A)

## vector space operations

for op in (:+, :-)
    @eval begin
        @generated function Base.$op(A::ArrayPartition, B::ArrayPartition)
            N = npartitions(A, B)
            expr = :($($op).(A.x[i], B.x[i]))

            build_arraypartition(N, expr)
        end

        @generated function Base.$op(A::ArrayPartition, B::Number)
            N = npartitions(A)
            expr = :($($op).(A.x[i], B))

            build_arraypartition(N, expr)
        end

        @generated function Base.$op(A::Number, B::ArrayPartition)
            N = npartitions(B)
            expr = :($($op).(A, B.x[i]))

            build_arraypartition(N, expr)
        end
    end
end

for op in (:*, :/)
    @eval @generated function Base.$op(A::ArrayPartition, B::Number)
        N = npartitions(A)
        expr = :($($op).(A.x[i], B))

        build_arraypartition(N, expr)
    end
end

@generated function Base.:*(A::Number, B::ArrayPartition)
    N = npartitions(B)
    expr = :((*).(A, B.x[i]))

    build_arraypartition(N, expr)
end

@generated function Base.:\(A::Number, B::ArrayPartition)
    N = npartitions(B)
    expr = :((/).(B.x[i], A))

    build_arraypartition(N, expr)
end

## indexing

@inline function Base.getindex(A::ArrayPartition, i::Int)
  @boundscheck checkbounds(A, i)
  @inbounds for j in 1:length(A.x)
    i -= length(A.x[j])
    if i <= 0
      return A.x[j][length(A.x[j])+i]
    end
  end
end

"""
    getindex(A::ArrayPartition, i::Int, j...)

Return the entry at index `j...` of the `i`th partition of `A`.
"""
@inline function Base.getindex(A::ArrayPartition, i::Int, j...)
  @boundscheck 0 < i <= length(A.x) || throw(BoundsError(A.x, i))
  @inbounds b = A.x[i]
  @boundscheck checkbounds(b, j...)
  @inbounds return b[j...]
end

"""
    getindex(A::ArrayPartition, ::Colon)

Return vector with all elements of array partition `A`.
"""
Base.getindex(A::ArrayPartition{T,S}, ::Colon) where {T,S} = T[a for a in Chain(A.x)]

@inline function Base.setindex!(A::ArrayPartition, v, i::Int)
  @boundscheck checkbounds(A, i)
  @inbounds for j in 1:length(A.x)
    i -= length(A.x[j])
    if i <= 0
      A.x[j][length(A.x[j])+i] = v
      break
    end
  end
end

"""
    setindex!(A::ArrayPartition, v, i::Int, j...)

Set the entry at index `j...` of the `i`th partition of `A` to `v`.
"""
@inline function Base.setindex!(A::ArrayPartition, v, i::Int, j...)
  @boundscheck 0 < i <= length(A.x) || throw(BoundsError(A.x, i))
  @inbounds b = A.x[i]
  @boundscheck checkbounds(b, j...)
  @inbounds b[j...] = v
end

## recursive methods

function recursivecopy!(A::ArrayPartition, B::ArrayPartition)
  for (a, b) in zip(A.x, B.x)
    recursivecopy!(a, b)
  end
end

recursive_mean(A::ArrayPartition) = mean((recursive_mean(x) for x in A.x))

# note: consider only first partition for recursive one and eltype
recursive_one(A::ArrayPartition) = recursive_one(first(A.x))
recursive_eltype(A::ArrayPartition) = recursive_eltype(first(A.x))

## iteration

Base.start(A::ArrayPartition) = start(Chain(A.x))
Base.next(A::ArrayPartition,state) = next(Chain(A.x),state)
Base.done(A::ArrayPartition,state) = done(Chain(A.x),state)

Base.length(A::ArrayPartition) = sum((length(x) for x in A.x))
Base.size(A::ArrayPartition) = (length(A),)

# redefine first and last to avoid slow and not type-stable indexing
Base.first(A::ArrayPartition) = first(first(A.x))
Base.last(A::ArrayPartition) = last(last(A.x))

## display

# restore the type rendering in Juno
Juno.@render Juno.Inline x::ArrayPartition begin
  fields = fieldnames(typeof(x))
  Juno.LazyTree(typeof(x), () -> [Juno.SubTree(Juno.Text("$f → "), Juno.getfield′(x, f)) for f in fields])
end
Base.summary(A::ArrayPartition) = string(typeof(A), " with arrays:")
Base.show(io::IO,A::ArrayPartition) = (Base.show.(io,A.x); nothing)

## broadcasting

Base.Broadcast._containertype(::Type{<:ArrayPartition}) = ArrayPartition
Base.Broadcast.promote_containertype(::Type{ArrayPartition}, ::Type) = ArrayPartition
Base.Broadcast.promote_containertype(::Type, ::Type{ArrayPartition}) = ArrayPartition
Base.Broadcast.promote_containertype(::Type{ArrayPartition}, ::Type{ArrayPartition}) = ArrayPartition
Base.Broadcast.promote_containertype(::Type{ArrayPartition}, ::Type{Array}) = ArrayPartition
Base.Broadcast.promote_containertype(::Type{Array}, ::Type{ArrayPartition}) = ArrayPartition

@generated function Base.Broadcast.broadcast_c(f, ::Type{ArrayPartition}, as...)
    # common number of partitions
    N = npartitions(as...)

    # broadcast partitions separately
    expr = :(broadcast(f,
                       # index partitions
                       $((as[d] <: ArrayPartition ? :(as[$d].x[i]) : :(as[$d])
                          for d in 1:length(as))...)))

    build_arraypartition(N, expr)
end

@generated function Base.Broadcast.broadcast_c!(f, ::Type{ArrayPartition}, ::Type,
                                     dest::ArrayPartition, as...)
    # common number of partitions
    N = npartitions(dest, as...)

    # broadcast partitions separately
    quote
        for i in 1:$N
            broadcast!(f, dest.x[i],
                       # index partitions
                       $((as[d] <: ArrayPartition ? :(as[$d].x[i]) : :(as[$d])
                          for d in 1:length(as))...))
        end
        dest
    end
end

## utils

"""
    build_arraypartition(N::Int, expr::Expr)

Build `ArrayPartition` consisting of `N` partitions, each the result of an evaluation of
`expr` with variable `i` set to the partition index in the range of 1 to `N`.

This can help to write a type-stable method in cases in which the correct return type can
can not be inferred for a simpler implementation with generators.
"""
function build_arraypartition(N::Int, expr::Expr)
    quote
        @Base.nexprs $N i->(A_i = $expr)
        partitions = @Base.ncall $N tuple i->A_i
        ArrayPartition(partitions)
    end
end

"""
    npartitions(A...)

Retrieve number of partitions of `ArrayPartitions` in `A...`, or throw an error if there are
`ArrayPartitions` with a different number of partitions.
"""
npartitions(A) = 0
npartitions(::Type{ArrayPartition{T,S}}) where {T,S} = length(S.parameters)
npartitions(A, B...) = common_number(npartitions(A), npartitions(B...))

common_number(a, b) =
    a == 0 ? b :
    (b == 0 ? a :
     (a == b ? a :
      throw(DimensionMismatch("number of partitions must be equal"))))

"""
    parameter(::Type{T})

Return type `T` of singleton.
"""
parameter(::Type{T}) where {T} = T
parameter(::Type{Type{T}}) where {T} = T

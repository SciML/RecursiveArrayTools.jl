struct ArrayPartition{T,S<:Tuple} <: AbstractVector{T}
  x::S
end

## constructors
@inline ArrayPartition(f::F, N) where F<:Function = ArrayPartition(ntuple(f, Val(N)))
ArrayPartition(x...) = ArrayPartition((x...,))

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
@inline function Base.similar(A::ArrayPartition, ::Type{T}) where {T}
    N = npartitions(A)
    ArrayPartition(i->similar(A.x[i], T), N)
end

# ignore dims since array partitions are vectors
Base.similar(A::ArrayPartition, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = similar(A, T)

# similar array partition with different types
function Base.similar(A::ArrayPartition, ::Type{T}, ::Type{S}, R::DataType...) where {T, S}
    N = npartitions(A)
    N != length(R) + 2 &&
        throw(DimensionMismatch("number of types must be equal to number of partitions"))

    types = (T, S, R...) # new types
    @inline function f(i)
        similar(A.x[i], types[i])
    end
    ArrayPartition(f, N)
end

Base.copy(A::ArrayPartition{T,S}) where {T,S} = ArrayPartition{T,S}(copy.(A.x))

## zeros
Base.zero(A::ArrayPartition{T,S}) where {T,S} = ArrayPartition{T,S}(zero.(A.x))
# ignore dims since array partitions are vectors
Base.zero(A::ArrayPartition, dims::NTuple{N,Int}) where {N} = zero(A)

## Array

Base.Array(VA::AbstractVectorOfArray{T,N,A}) where {T,N,A <: AbstractVector{<:ArrayPartition}} = reduce(hcat,Array.(VA.u))

## ones

# special to work with units
function Base.ones(A::ArrayPartition)
    N = npartitions(A)
    out = similar(A)
    for i in 1:N
        fill!(out.x[i], oneunit(eltype(out.x[i])))
    end
    out
end

# ignore dims since array partitions are vectors
Base.ones(A::ArrayPartition, dims::NTuple{N,Int}) where {N} = ones(A)

## vector space operations

for op in (:+, :-)
    @eval begin
        function Base.$op(A::ArrayPartition, B::ArrayPartition)
            Base.broadcast($op, A, B)
        end

        function Base.$op(A::ArrayPartition, B::Number)
            Base.broadcast($op, A, B)
        end

        function Base.$op(A::Number, B::ArrayPartition)
            Base.broadcast($op, A, B)
        end
    end
end

for op in (:*, :/)
    @eval function Base.$op(A::ArrayPartition, B::Number)
        Base.broadcast($op, A, B)
    end
end

function Base.:*(A::Number, B::ArrayPartition)
    Base.broadcast(*, A, B)
end

function Base.:\(A::Number, B::ArrayPartition)
    Base.broadcast(/, B, A)
end

Base.:(==)(A::ArrayPartition,B::ArrayPartition) = A.x == B.x

## Functional Constructs

Base.mapreduce(f,op,A::ArrayPartition) = mapreduce(f,op,(mapreduce(f,op,x) for x in A.x))
Base.any(f,A::ArrayPartition) = any(f,(any(f,x) for x in A.x))
Base.any(f::Function,A::ArrayPartition) = any(f,(any(f,x) for x in A.x))
function Base.copyto!(dest::AbstractArray,A::ArrayPartition)
    @assert length(dest) == length(A)
    cur = 1
    @inbounds for i in 1:length(A.x)
      dest[cur:(cur+length(A.x[i])-1)] .= vec(A.x[i])
      cur += length(A.x[i])
    end
    dest
end

function Base.copyto!(A::ArrayPartition,src::ArrayPartition)
    @assert length(src) == length(A)
    if size.(A.x) == size.(src.x)
      A .= src
    else
      cnt = 0
      for i in eachindex(A.x)
        x = A.x[i]
        for k in eachindex(x)
          cnt += 1
          x[k] = src[cnt]
        end
      end
    end
    A
end

## indexing

# Interface for the linear indexing. This is just a view of the underlying nested structure
@inline Base.firstindex(A::ArrayPartition) = 1
@inline Base.lastindex(A::ArrayPartition) = length(A)

Base.@propagate_inbounds function Base.getindex(A::ArrayPartition, i::Int)
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
Base.@propagate_inbounds function Base.getindex(A::ArrayPartition, i::Int, j...)
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

Base.@propagate_inbounds function Base.setindex!(A::ArrayPartition, v, i::Int)
  @boundscheck checkbounds(A, i)
  @inbounds for j in 1:length(A.x)
    i -= length(A.x[j])
    if i <= 0
      A.x[j][length(A.x[j])+i] = v
      break
    end
  end
end

# workaround for https://github.com/SciML/RecursiveArrayTools.jl/issues/49
function Base._unsafe_getindex(::IndexStyle, A::ArrayPartition, I::Vararg{Union{Real, AbstractArray}, N}) where N
    # This is specifically not inlined to prevent excessive allocations in type unstable code
    shape = Base.index_shape(I...)
    dest = similar(A.x[1], shape)
    Base._unsafe_getindex!(dest, A, I...) # usually a generated function, don't allow it to impact inference result
    return dest
end

"""
    setindex!(A::ArrayPartition, v, i::Int, j...)

Set the entry at index `j...` of the `i`th partition of `A` to `v`.
"""
Base.@propagate_inbounds function Base.setindex!(A::ArrayPartition, v, i::Int, j...)
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
recursivecopy(A::ArrayPartition) = ArrayPartition(copy.(A.x))

recursive_mean(A::ArrayPartition) = mean((recursive_mean(x) for x in A.x))

# note: consider only first partition for recursive one and eltype
recursive_one(A::ArrayPartition) = recursive_one(first(A.x))
recursive_eltype(A::ArrayPartition) = recursive_eltype(first(A.x))

## iteration

Base.iterate(A::ArrayPartition) = iterate(Chain(A.x))
Base.iterate(A::ArrayPartition,state) = iterate(Chain(A.x),state)

Base.length(A::ArrayPartition) = sum((length(x) for x in A.x))
Base.size(A::ArrayPartition) = (length(A),)

# redefine first and last to avoid slow and not type-stable indexing
Base.first(A::ArrayPartition) = first(first(A.x))
Base.last(A::ArrayPartition) = last(last(A.x))

## display
Base.summary(A::ArrayPartition) = string(typeof(A), " with arrays:")
Base.show(io::IO,A::ArrayPartition) = map(x->Base.show(io,x),A.x)
Base.show(io::IO, m::MIME"text/plain", A::ArrayPartition) = show(io, m, A.x)

## broadcasting

struct ArrayPartitionStyle{Style <: Broadcast.BroadcastStyle} <: Broadcast.AbstractArrayStyle{Any} end
ArrayPartitionStyle(::S) where {S} = ArrayPartitionStyle{S}()
ArrayPartitionStyle(::S, ::Val{N}) where {S,N} = ArrayPartitionStyle(S(Val(N)))
ArrayPartitionStyle(::Val{N}) where N = ArrayPartitionStyle{Broadcast.DefaultArrayStyle{N}}()

# promotion rules
@inline function Broadcast.BroadcastStyle(::ArrayPartitionStyle{AStyle}, ::ArrayPartitionStyle{BStyle}) where {AStyle, BStyle}
    ArrayPartitionStyle(Broadcast.BroadcastStyle(AStyle(), BStyle()))
end
Broadcast.BroadcastStyle(::ArrayPartitionStyle{Style}, ::Broadcast.DefaultArrayStyle{0}) where Style<:Broadcast.BroadcastStyle = ArrayPartitionStyle{Style}()
Broadcast.BroadcastStyle(::ArrayPartitionStyle, ::Broadcast.DefaultArrayStyle{N}) where N = Broadcast.DefaultArrayStyle{N}()

combine_styles(args::Tuple{})         = Broadcast.DefaultArrayStyle{0}()
@inline combine_styles(args::Tuple{Any})      = Broadcast.result_style(Broadcast.BroadcastStyle(args[1]))
@inline combine_styles(args::Tuple{Any, Any}) = Broadcast.result_style(Broadcast.BroadcastStyle(args[1]), Broadcast.BroadcastStyle(args[2]))
@inline combine_styles(args::Tuple)   = Broadcast.result_style(Broadcast.BroadcastStyle(args[1]), combine_styles(Base.tail(args)))

function Broadcast.BroadcastStyle(::Type{ArrayPartition{T,S}}) where {T, S}
    Style = combine_styles((S.parameters...,))
    ArrayPartitionStyle(Style)
end

@inline function Base.copy(bc::Broadcast.Broadcasted{ArrayPartitionStyle{Style}}) where Style
    N = npartitions(bc)
    @inline function f(i)
        copy(unpack(bc, i))
    end
    ArrayPartition(f, N)
end

@inline function Base.copyto!(dest::ArrayPartition, bc::Broadcast.Broadcasted{ArrayPartitionStyle{Style}}) where Style
    N = npartitions(dest, bc)
    @inbounds for i in 1:N
        copyto!(dest.x[i], unpack(bc, i))
    end
    dest
end

## broadcasting utils

"""
    npartitions(A...)

Retrieve number of partitions of `ArrayPartitions` in `A...`, or throw an error if there are
`ArrayPartitions` with a different number of partitions.
"""
npartitions(A) = 0
npartitions(A::ArrayPartition) = length(A.x)
npartitions(bc::Broadcast.Broadcasted) = _npartitions(bc.args)
npartitions(A, Bs...) = common_number(npartitions(A), _npartitions(Bs))

@inline _npartitions(args::Tuple) = common_number(npartitions(args[1]), _npartitions(Base.tail(args)))
_npartitions(args::Tuple{Any}) = npartitions(args[1])
_npartitions(args::Tuple{}) = 0

# drop axes because it is easier to recompute
@inline unpack(bc::Broadcast.Broadcasted{Style}, i) where Style = Broadcast.Broadcasted{Style}(bc.f, unpack_args(i, bc.args))
@inline unpack(bc::Broadcast.Broadcasted{ArrayPartitionStyle{Style}}, i) where Style = Broadcast.Broadcasted{Style}(bc.f, unpack_args(i, bc.args))
unpack(x,::Any) = x
unpack(x::ArrayPartition, i) = x.x[i]

@inline unpack_args(i, args::Tuple) = (unpack(args[1], i), unpack_args(i, Base.tail(args))...)
unpack_args(i, args::Tuple{Any}) = (unpack(args[1], i),)
unpack_args(::Any, args::Tuple{}) = ()

## utils
common_number(a, b) =
    a == 0 ? b :
    (b == 0 ? a :
     (a == b ? a :
      throw(DimensionMismatch("number of partitions must be equal"))))

## Linear Algebra

ArrayInterface.zeromatrix(A::ArrayPartition) = ArrayInterface.zeromatrix(Vector(A))

LinearAlgebra.ldiv!(A::Factorization, b::ArrayPartition) = (x = ldiv!(A, Array(b)); copyto!(b, x))
function LinearAlgebra.ldiv!(A::LU, b::ArrayPartition)
    LinearAlgebra._ipiv_rows!(A, 1 : length(A.ipiv), b)
    ldiv!(UpperTriangular(A.factors), ldiv!(UnitLowerTriangular(A.factors), b))
    return b
end

# block matrix indexing
@inbounds function getblock(A, lens, i, j)
    ii1 = i == 1 ? 0 : sum(ii->lens[ii], 1:i-1)
    jj1 = j == 1 ? 0 : sum(ii->lens[ii], 1:j-1)
    ij1 = CartesianIndex(ii1, jj1)
    cc1 = CartesianIndex((1, 1))
    inc = CartesianIndex(lens[i], lens[j])
    return @view A[(ij1+cc1):(ij1+inc)]
end
# fast ldiv for UpperTriangular and UnitLowerTriangular
# [U11  U12  U13]   [ b1 ]
# [ 0   U22  U23] \ [ b2 ]
# [ 0    0   U33]   [ b3 ]
function LinearAlgebra.ldiv!(A::T, bb::ArrayPartition) where T<:Union{UnitUpperTriangular,UpperTriangular}
    A = A.data
    n = npartitions(bb)
    b = bb.x
    lens = map(length, b)
    @inbounds for j in n:-1:1
        Ajj = T(getblock(A, lens, j, j))
        xj = ldiv!(Ajj, vec(b[j]))
        for i in j-1:-1:1
            Aij = getblock(A, lens, i, j)
            # bi = -Aij * xj + bi
            mul!(vec(b[i]), Aij, xj, -1, true)
        end
    end
    return bb
end

function LinearAlgebra.ldiv!(A::T, bb::ArrayPartition) where T<:Union{UnitLowerTriangular,LowerTriangular}
    A = A.data
    n = npartitions(bb)
    b = bb.x
    lens = map(length, b)
    @inbounds for j in 1:n
        Ajj = T(getblock(A, lens, j, j))
        xj = ldiv!(Ajj, vec(b[j]))
        for i in j+1:n
            Aij = getblock(A, lens, i, j)
            # bi = -Aij * xj + b[i]
            mul!(vec(b[i]), Aij, xj, -1, true)
        end
    end
    return bb
end
# TODO: optimize
function LinearAlgebra._ipiv_rows!(A::LU, order::OrdinalRange, B::ArrayPartition)
    for i = order
        if i != A.ipiv[i]
            LinearAlgebra._swap_rows!(B, i, A.ipiv[i])
        end
    end
    return B
end
function LinearAlgebra._swap_rows!(B::ArrayPartition, i::Integer, j::Integer)
    B[i], B[j] = B[j], B[i]
    return B
end

# linalg mul! overloads for ArrayPartition
function LinearAlgebra.mul!(C::ArrayPartition, A::ArrayPartition, B::AbstractArray)
    if length(C.x) != length(A.x)
        throw(DimensionMismatch("Length of C, $(length(C.x)), does not match length of A, $(length(A.x))"))
    end

    for index = 1:length(C.x)
        mul!(C.x[index], A.x[index], B)
    end
    return C
end

function LinearAlgebra.mul!(C::ArrayPartition, A::ArrayPartition, B::ArrayPartition)
    if length(C.x) != length(A.x)
        throw(DimensionMismatch("Length of C, $(length(C.x)), does not match length of A, $(length(B.x))"))
    end
    if length(A.x) != length(B.x)
        throw(DimensionMismatch("Length of A, $(length(A.x)), does not match length of B, $(length(B.x))"))
    end

    for index = 1:length(C.x)
        mul!(C.x[index], A.x[index], B.x[index])
    end
    return C
end

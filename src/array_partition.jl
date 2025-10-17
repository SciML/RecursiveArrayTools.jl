"""
```julia
ArrayPartition(x::AbstractArray...)
```

An `ArrayPartition` `A` is an array, which is made up of different arrays `A.x`.
These index like a single array, but each subarray may have a different type.
However, broadcast is overloaded to loop in an efficient manner, meaning that
`A .+= 2.+B` is type-stable in its computations, even if `A.x[i]` and `A.x[j]`
do not match types. A full array interface is included for completeness, which
allows this array type to be used in place of a standard array where
such a type stable broadcast may be needed. One example is in heterogeneous
differential equations for [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/).

An `ArrayPartition` acts like a single array. `A[i]` indexes through the first
array, then the second, etc., all linearly. But `A.x` is where the arrays are stored.
Thus, for:

```julia
using RecursiveArrayTools
A = ArrayPartition(y, z)
```

we would have `A.x[1]==y` and `A.x[2]==z`. Broadcasting like `f.(A)` is efficient.
"""
struct ArrayPartition{T, S <: Tuple} <: AbstractVector{T}
    x::S
end

## constructors
@inline ArrayPartition(f::F, N) where {F <: Function} = ArrayPartition(ntuple(f, Val(N)))
ArrayPartition(x...) = ArrayPartition((x...,))

function ArrayPartition(x::S, ::Type{Val{copy_x}} = Val{false}) where {S <: Tuple, copy_x}
    T = promote_type(map(recursive_bottom_eltype, x)...)
    if copy_x
        return ArrayPartition{T, S}(map(copy, x))
    else
        return ArrayPartition{T, S}(x)
    end
end

## similar array partitions

Base.similar(A::ArrayPartition{T, S}) where {T, S} = ArrayPartition{T, S}(similar.(A.x))

# return ArrayPartition when possible, otherwise next best thing of the correct size
function Base.similar(A::ArrayPartition, dims::NTuple{N, Int}) where {N}
    if dims == size(A)
        return similar(A)
    else
        return similar(A.x[1], eltype(A), dims)
    end
end

# similar array partition of common type
@inline function Base.similar(A::ArrayPartition, ::Type{T}) where {T}
    N = npartitions(A)
    ArrayPartition(i -> similar(A.x[i], T), N)
end

# return ArrayPartition when possible, otherwise next best thing of the correct size
function Base.similar(A::ArrayPartition, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    if dims == size(A)
        return similar(A, T)
    else
        return similar(A.x[1], T, dims)
    end
end

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

Base.copy(A::ArrayPartition) = ArrayPartition(map(copy, A.x))

## zeros
Base.zero(A::ArrayPartition) = ArrayPartition(map(zero, A.x))
# ignore dims since array partitions are vectors
Base.zero(A::ArrayPartition, dims::NTuple{N, Int}) where {N} = zero(A)

## Array

Base.Array(A::ArrayPartition) = reduce(vcat, Array.(A.x))
function Base.Array(VA::AbstractVectorOfArray{
        T,
        N,
        A
}) where {T, N,
        A <: AbstractVector{
            <:ArrayPartition,
        }}
    reduce(hcat, Array.(VA.u))
end

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
Base.ones(A::ArrayPartition, dims::NTuple{N, Int}) where {N} = ones(A)

# mutable iff all components of ArrayPartition are mutable
@generated function ArrayInterface.ismutable(::Type{<:ArrayPartition{T, S}}) where {T, S
}
    res = all(ArrayInterface.ismutable, S.parameters)
    return :($res)
end

## resize!
function Base.resize!(A::ArrayPartition, sizes::Tuple)
    resize!.(A.x, sizes)
    A
end

## vector space operations

for op in (:+, :-)
    @eval begin
        function Base.$op(A::ArrayPartition, B::ArrayPartition)
            ArrayPartition(map((x, y) -> Base.broadcast($op, x, y), A.x, B.x))
        end

        function Base.$op(A::ArrayPartition, B::Number)
            ArrayPartition(map(y -> Base.broadcast($op, y, B), A.x))
        end

        function Base.$op(A::Number, B::ArrayPartition)
            ArrayPartition(map(y -> Base.broadcast($op, A, y), B.x))
        end
    end
end

function Base.:-(A::ArrayPartition)
    return ArrayPartition(map(-, A.x))
end

for op in (:*, :/)
    @eval function Base.$op(A::ArrayPartition, B::Number)
        ArrayPartition(map(y -> Base.broadcast($op, y, B), A.x))
    end
end

function Base.:*(A::Number, B::ArrayPartition)
    ArrayPartition(map(y -> A .* y, B.x))
end

function Base.:\(A::Number, B::ArrayPartition)
    B / A
end

Base.:(==)(A::ArrayPartition, B::ArrayPartition) = A.x == B.x

## Iterable Collection Constructs

Base.map(f, A::ArrayPartition) = ArrayPartition(map(x -> map(f, x), A.x))
function Base.mapreduce(f, op, A::ArrayPartition{T}; kwargs...) where {T}
    mapreduce(f, op, (i for i in A); kwargs...)
end
Base.filter(f, A::ArrayPartition) = ArrayPartition(map(x -> filter(f, x), A.x))
Base.any(f, A::ArrayPartition) = any((any(f, x) for x in A.x))
Base.any(f::Function, A::ArrayPartition) = any((any(f, x) for x in A.x))
Base.any(A::ArrayPartition) = any(identity, A)
Base.all(f, A::ArrayPartition) = all(f, (all(f, x) for x in A.x))
Base.all(f::Function, A::ArrayPartition) = all((all(f, x) for x in A.x))
Base.all(A::ArrayPartition) = all(identity, A)

for type in [AbstractArray, PermutedDimsArray]
    @eval function Base.copyto!(dest::$(type), A::ArrayPartition)
        @assert length(dest) == length(A)
        cur = 1
        @inbounds for i in 1:length(A.x)
            if A.x[i] isa Number
                dest[cur:(cur + length(A.x[i]) - 1)] .= A.x[i]
            else
                dest[cur:(cur + length(A.x[i]) - 1)] .= vec(A.x[i])
            end
            cur += length(A.x[i])
        end
        dest
    end
end

function Base.copyto!(A::ArrayPartition, src::ArrayPartition)
    @assert length(src) == length(A)
    if size.(A.x) == size.(src.x)
        map(copyto!, A.x, src.x)
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

function Base.fill!(A::ArrayPartition, x)
    unrolled_foreach!(A.x) do x_
        fill!(x_, x)
    end
    A
end

function recursivefill!(b::ArrayPartition, a::T2) where {T2 <: Union{Number, Bool}}
    unrolled_foreach!(b.x) do x
        fill!(x, a)
    end
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
            return A.x[j][length(A.x[j]) + i]
        end
    end
    throw(BoundsError(A, i))
end

"""
    getindex(A::ArrayPartition, i::Colon, j...)

Returns the entry at index `j...` of  every partition of `A`.
"""
Base.@propagate_inbounds function Base.getindex(A::ArrayPartition, i::Colon, j...)
    return getindex.(A.x, (j...,))
end

"""
    getindex(A::ArrayPartition, ::Colon)

Returns a vector with all elements of array partition `A`.
"""
Base.getindex(A::ArrayPartition{T, S}, ::Colon) where {T, S} = T[a for a in Chain(A.x)]

Base.@propagate_inbounds function Base.setindex!(A::ArrayPartition, v, i::Int)
    @boundscheck checkbounds(A, i)
    @inbounds for j in 1:length(A.x)
        i -= length(A.x[j])
        if i <= 0
            A.x[j][length(A.x[j]) + i] = v
            break
        end
    end
end

# workaround for https://github.com/SciML/RecursiveArrayTools.jl/issues/49
function Base._unsafe_getindex(::IndexStyle, A::ArrayPartition,
        I::Vararg{Union{Real, AbstractArray}, N}) where {N}
    # This is specifically not inlined to prevent excessive allocations in type unstable code
    shape = Base.index_shape(I...)
    dest = similar(A.x[1], shape)
    Base._unsafe_getindex!(dest, A, I...) # usually a generated function, don't allow it to impact inference result
    return dest
end

function Base._maybe_reshape(::IndexCartesian,
        A::ArrayPartition,
        I::Vararg{Union{Real, AbstractArray}, N}) where {N}
    Vector(A)
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
Base.iterate(A::ArrayPartition, state) = iterate(Chain(A.x), state)

Base.length(A::ArrayPartition) = sum(broadcast(length, A.x); init = 0)
Base.size(A::ArrayPartition) = (length(A),)

# redefine first and last to avoid slow and not type-stable indexing
Base.first(A::ArrayPartition) = first(first(A.x))
Base.last(A::ArrayPartition) = last(last(A.x))

## display
Base.summary(A::ArrayPartition) = string(typeof(A), " with arrays:")
Base.show(io::IO, m::MIME"text/plain", A::ArrayPartition) = show(io, m, A.x)

## broadcasting

struct ArrayPartitionStyle{Style <: Broadcast.BroadcastStyle} <:
       Broadcast.AbstractArrayStyle{Any} end
ArrayPartitionStyle(::S) where {S} = ArrayPartitionStyle{S}()
ArrayPartitionStyle(::S, ::Val{N}) where {S, N} = ArrayPartitionStyle(S(Val(N)))
function ArrayPartitionStyle(::Val{N}) where {N}
    ArrayPartitionStyle{Broadcast.DefaultArrayStyle{N}}()
end

# promotion rules
@inline function Broadcast.BroadcastStyle(::ArrayPartitionStyle{AStyle},
        ::ArrayPartitionStyle{BStyle}) where {AStyle,
        BStyle}
    ArrayPartitionStyle(Broadcast.BroadcastStyle(AStyle(), BStyle()))
end
function Broadcast.BroadcastStyle(::ArrayPartitionStyle{Style},
        ::Broadcast.DefaultArrayStyle{0}) where {
        Style <:
        Broadcast.BroadcastStyle,
}
    ArrayPartitionStyle{Style}()
end
function Broadcast.BroadcastStyle(::ArrayPartitionStyle,
        ::Broadcast.DefaultArrayStyle{N}) where {N}
    Broadcast.DefaultArrayStyle{N}()
end

combine_styles(::Type{Tuple{}}) = Broadcast.DefaultArrayStyle{0}()
function combine_styles(::Type{T}) where {T}
    Broadcast.result_style(Broadcast.BroadcastStyle(T.parameters[1]),
        combine_styles(Tuple{Base.tail((T.parameters...,))...}))
end

function Broadcast.BroadcastStyle(::Type{ArrayPartition{T, S}}) where {T, S}
    Style = combine_styles(S)
    ArrayPartitionStyle(Style)
end

@inline function Base.copy(bc::Broadcast.Broadcasted{
        ArrayPartitionStyle{Style},
}) where {
        Style,
}
    N = npartitions(bc)
    @inline function f(i)
        copy(unpack(bc, i))
    end
    ArrayPartition(f, N)
end

@inline function Base.copyto!(dest::ArrayPartition,
        bc::Broadcast.Broadcasted{ArrayPartitionStyle{Style}}) where {Style}
    N = npartitions(dest, bc)
    # If dest is all the same underlying array type, use for-loop
    if all(x isa typeof(first(dest.x)) for x in dest.x)
        @inbounds for i in 1:N
            copyto!(dest.x[i], unpack(bc, i))
        end
    else
        # Fall back to original implementation for complex broadcasts
        @inline function f(i)
            copyto!(dest.x[i], unpack(bc, i))
        end
        ntuple(f, Val(N))
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

@inline function _npartitions(args::Tuple)
    common_number(npartitions(args[1]), _npartitions(Base.tail(args)))
end
_npartitions(args::Tuple{Any}) = npartitions(args[1])
_npartitions(args::Tuple{}) = 0

# drop axes because it is easier to recompute
@inline function unpack(bc::Broadcast.Broadcasted{Style}, i) where {Style}
    Broadcast.Broadcasted(bc.f, unpack_args(i, bc.args))
end
@inline function unpack(bc::Broadcast.Broadcasted{ArrayPartitionStyle{Style}},
        i) where {Style}
    Broadcast.Broadcasted(bc.f, unpack_args(i, bc.args))
end
@inline function unpack(bc::Broadcast.Broadcasted{Style},
        i) where {Style <: Broadcast.DefaultArrayStyle}
    Broadcast.Broadcasted{Style}(bc.f, unpack_args(i, bc.args))
end
@inline function unpack(bc::Broadcast.Broadcasted{ArrayPartitionStyle{Style}},
        i) where {Style <: Broadcast.DefaultArrayStyle}
    Broadcast.Broadcasted{Style}(bc.f, unpack_args(i, bc.args))
end
@inline unpack(x, ::Any) = x
@inline unpack(x::ArrayPartition, i) = x.x[i]

@inline function unpack_args(i, args::Tuple)
    (unpack(args[1], i), unpack_args(i, Base.tail(args))...)
end
unpack_args(i, args::Tuple{Any}) = (unpack(args[1], i),)
unpack_args(::Any, args::Tuple{}) = ()

## utils
function common_number(a, b)
    a == 0 ? b :
    (b == 0 ? a :
     (a == b ? a :
      throw(DimensionMismatch("number of partitions must be equal"))))
end

## Linear Algebra

ArrayInterface.zeromatrix(A::ArrayPartition) = ArrayInterface.zeromatrix(Vector(A))

function __get_subtypes_in_module(
        mod, supertype; include_supertype = true, all = false, except = [])
    return filter([getproperty(mod, name) for name in names(mod; all) if !in(name, except)]) do value
        return value != Union{} && value isa Type && (value <: supertype) &&
               (include_supertype || value != supertype) && !in(value, except)
    end
end

for factorization in vcat(
    __get_subtypes_in_module(LinearAlgebra, Factorization; include_supertype = false,
        all = true, except = [:LU, :LAPACKFactorizations]),
    LDLt{T, <:SymTridiagonal{T, V} where {V <: AbstractVector{T}}} where {T})
    @eval function LinearAlgebra.ldiv!(A::T, b::ArrayPartition) where {T <: $factorization}
        (x = ldiv!(A, Array(b)); copyto!(b, x))
    end
end

function LinearAlgebra.ldiv!(
        A::LinearAlgebra.QRPivoted{T, <:StridedMatrix{T}, <:AbstractVector{T}},
        b::ArrayPartition{T}) where {T <: Union{Float32, Float64, ComplexF64, ComplexF32}}
    x = ldiv!(A, Array(b))
    copyto!(b, x)
end

function LinearAlgebra.ldiv!(A::LinearAlgebra.QRCompactWY{T, M, C},
        b::ArrayPartition) where {
        T <: Union{Float32, Float64, ComplexF64, ComplexF32},
        M <: AbstractMatrix{T},
        C <: AbstractMatrix{T}
}
    (x = ldiv!(A, Array(b)); copyto!(b, x))
end

for type in [LU, LU{T, Tridiagonal{T, V}} where {T, V}]
    @eval function LinearAlgebra.ldiv!(A::$type, b::ArrayPartition)
        LinearAlgebra._ipiv_rows!(A, 1:length(A.ipiv), b)
        ldiv!(UpperTriangular(A.factors), ldiv!(UnitLowerTriangular(A.factors), b))
        return b
    end
end

# block matrix indexing
@inbounds function getblock(A, lens, i, j)
    ii1 = i == 1 ? 0 : sum(ii -> lens[ii], 1:(i - 1))
    jj1 = j == 1 ? 0 : sum(ii -> lens[ii], 1:(j - 1))
    ij1 = CartesianIndex(ii1, jj1)
    cc1 = CartesianIndex((1, 1))
    inc = CartesianIndex(lens[i], lens[j])
    return @view A[(ij1 + cc1):(ij1 + inc)]
end
# fast ldiv for UpperTriangular and UnitLowerTriangular
# [U11  U12  U13]   [ b1 ]
# [ 0   U22  U23] \ [ b2 ]
# [ 0    0   U33]   [ b3 ]
for basetype in [UnitUpperTriangular, UpperTriangular, UnitLowerTriangular, LowerTriangular]
    for type in [basetype, basetype{T, <:Adjoint{T}} where {T},
        basetype{T, <:Transpose{T}} where {T}]
        j_iter, i_iter = if basetype <: UnitUpperTriangular || basetype <: UpperTriangular
            (:(n:-1:1), :((j - 1):-1:1))
        else
            (:(1:n), :((j + 1):n))
        end
        @eval function LinearAlgebra.ldiv!(A::$type, bb::ArrayPartition)
            A = A.data
            n = npartitions(bb)
            b = bb.x
            lens = map(length, b)
            @inbounds for j in $j_iter
                Ajj = $basetype(getblock(A, lens, j, j))
                xj = ldiv!(Ajj, vec(b[j]))
                for i in $i_iter
                    Aij = getblock(A, lens, i, j)
                    # bi = -Aij * xj + bi
                    mul!(vec(b[i]), Aij, xj, -1, true)
                end
            end
            return bb
        end
    end
end

# TODO: optimize
function LinearAlgebra._ipiv_rows!(A::LU, order::OrdinalRange, B::ArrayPartition)
    for i in order
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

    for index in 1:length(C.x)
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

    for index in 1:length(C.x)
        mul!(C.x[index], A.x[index], B.x[index])
    end
    return C
end

function Base.convert(::Type{ArrayPartition{T, S}},
        A::ArrayPartition{<:Any, <:NTuple{N, Any}}) where {N, T,
        S <:
        NTuple{N, Any}}
    return ArrayPartition{T, S}(ntuple((@inline i -> convert(S.parameters[i], A.x[i])),
        Val(N)))
end

@generated function Base.length(::Type{
        <:ArrayPartition{F, T},
}) where {F, N,
        T <: NTuple{N,
            StaticArraysCore.StaticArray
        }}
    sum_expr = Expr(:call, :+)
    for param in T.parameters
        push!(sum_expr.args, :(length($param)))
    end
    return sum_expr
end

function Adapt.adapt_structure(to, ap::ArrayPartition)
    ArrayPartition(map(x -> Adapt.adapt(to, x), ap.x)...)
end
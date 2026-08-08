unrolled_foreach!(f, t::Tuple) = (f(t[1]); unrolled_foreach!(f, Base.tail(t)))
unrolled_foreach!(f, ::Tuple{}) = nothing

"""
    recursivecopy(a)

Copy `a` recursively. This acts like `deepcopy` for nested arrays and like `copy` for
arrays of scalar values.

# Examples

```julia
original = [[1, 2], [3, 4]]
copied = recursivecopy(original)
copied[1][1] = 10
original[1][1] == 1
```
"""
function recursivecopy(a)
    return deepcopy(a)
end
function recursivecopy(
        a::Union{
            StaticArraysCore.SVector, StaticArraysCore.SMatrix,
            StaticArraysCore.SArray, Number,
        }
    )
    return copy(a)
end
function recursivecopy(a::AbstractArray{T, N}) where {T <: Number, N}
    return copy(a)
end

function recursivecopy(a::AbstractArray{T, N}) where {T <: AbstractArray, N}
    return if ArrayInterface.ismutable(a)
        b = similar(a)
        map!(recursivecopy, b, a)
    else
        ArrayInterface.restructure(a, map(recursivecopy, a))
    end
end

function recursivecopy(a::AbstractVectorOfArray)
    # Deep-copy the inner arrays. When the outer container is immutable
    # (e.g. `SVector` of arrays, OrdinaryDiffEq.jl#1365), broadcast assignment
    # into `b.u` fails, so rebuild via `rewrap` the way `zero` does.
    u_copy = map(recursivecopy, a.u)
    if ArrayInterface.ismutable(a.u)
        b = copy(a)
        b.u .= u_copy
        return b
    else
        T = typeof(a)
        u_rewrapped = rewrap(a.u, u_copy)
        fields = [
            fname === :u ? u_rewrapped : _copyfield(a, fname)
                for fname in fieldnames(T)
        ]
        return T(fields...)
    end
end

"""
    recursivecopy!(dest, src)

Copy `src` recursively into `dest`. This acts like `deepcopy!` for nested arrays and
like `copy!` for arrays of scalar values. `dest` and `src` must have matching
dimensionality; use [`recursivecopyto!`](@ref) for linear copying between different
shapes.

# Examples

```julia
dest = [zeros(2), zeros(2)]
src = [[1.0, 2.0], [3.0, 4.0]]
recursivecopy!(dest, src)
```
"""
function recursivecopy! end

function recursivecopy!(
        b::AbstractArray{T, N},
        a::AbstractArray{T2, N}
    ) where {
        T <: StaticArraysCore.StaticArray,
        T2 <: StaticArraysCore.StaticArray,
        N,
    }
    return @inbounds for i in eachindex(a)
        # TODO: Check for `setindex!`` and use `copy!(b[i],a[i])` or `b[i] = a[i]`, see #19
        b[i] = copy(a[i])
    end
end

function recursivecopy!(
        b::AbstractArray{T, N},
        a::AbstractArray{T2, N}
    ) where {T <: Enum, T2 <: Enum, N}
    return copyto!(b, a)
end

function recursivecopy!(
        b::AbstractArray{T, N},
        a::AbstractArray{T2, N}
    ) where {T <: Number, T2 <: Number, N}
    return copyto!(b, a)
end

function recursivecopy!(
        b::AbstractArray{T, N},
        a::AbstractArray{T2, N}
    ) where {
        T <: Union{AbstractArray, AbstractVectorOfArray},
        T2 <: Union{AbstractArray, AbstractVectorOfArray}, N,
    }
    if ArrayInterface.ismutable(T)
        @inbounds for i in eachindex(b, a)
            recursivecopy!(b[i], a[i])
        end
    else
        copyto!(b, a)
    end
    return b
end

function recursivecopy!(b::AbstractVectorOfArray, a::AbstractVectorOfArray)
    @inbounds for i in eachindex(b.u, a.u)
        if ArrayInterface.ismutable(b.u[i]) || b.u[i] isa AbstractVectorOfArray
            recursivecopy!(b.u[i], a.u[i])
        else
            b.u[i] = recursivecopy(a.u[i])
        end
    end
    return b
end

"""
    recursivecopyto!(dest, src) -> dest

Copy `src` recursively into `dest` in linear, column-major order. This acts like
`deepcopy!` for nested arrays and like `copyto!` for arrays of scalar values.

Unlike [`recursivecopy!`](@ref), this does not require matching dimensions or axes.
Use it when copying between different shapes is intentional.

# Examples

```julia
dest = zeros(2, 2)
recursivecopyto!(dest, [1.0, 2.0, 3.0, 4.0])
```
"""
function recursivecopyto! end

function recursivecopyto!(b::AbstractArray, a::AbstractArray)
    return copyto!(b, a)
end

function recursivecopyto!(
        b::AbstractArray{T},
        a::AbstractArray{T2}
    ) where {
        T <: StaticArraysCore.StaticArray,
        T2 <: StaticArraysCore.StaticArray,
    }
    @inbounds for (ib, ia) in zip(eachindex(b), eachindex(a))
        # TODO: Check for `setindex!`` and use `copy!(b[i],a[i])` or `b[i] = a[i]`, see #19
        b[ib] = copy(a[ia])
    end
    return b
end

function recursivecopyto!(
        b::AbstractArray{T},
        a::AbstractArray{T2}
    ) where {
        T <: Union{AbstractArray, AbstractVectorOfArray},
        T2 <: Union{AbstractArray, AbstractVectorOfArray},
    }
    if ArrayInterface.ismutable(T)
        @inbounds for (ib, ia) in zip(eachindex(b), eachindex(a))
            recursivecopyto!(b[ib], a[ia])
        end
    else
        copyto!(b, a)
    end
    return b
end

function recursivecopyto!(b::AbstractVectorOfArray, a::AbstractVectorOfArray)
    @inbounds for i in eachindex(b.u, a.u)
        if ArrayInterface.ismutable(b.u[i]) || b.u[i] isa AbstractVectorOfArray
            recursivecopyto!(b.u[i], a.u[i])
        else
            b.u[i] = recursivecopy(a.u[i])
        end
    end
    return b
end

"""
    recursivefill!(dest, value)

Fill every scalar element of the nested container `dest` with `value`.

# Examples

```julia
dest = [zeros(2), zeros(3)]
recursivefill!(dest, 1.0)
```
"""
function recursivefill! end

function recursivefill!(
        b::AbstractArray{T, N},
        a::T2
    ) where {
        T <: StaticArraysCore.StaticArray,
        T2 <: StaticArraysCore.StaticArray, N,
    }
    return @inbounds for i in eachindex(b)
        b[i] = copy(a)
    end
end

function recursivefill!(
        bs::AbstractVectorOfArray{T, N},
        a::T2
    ) where {
        T <: StaticArraysCore.StaticArray,
        T2 <: StaticArraysCore.StaticArray, N,
    }
    return @inbounds for b in bs.u, i in eachindex(b)

        b[i] = copy(a)
    end
end

function recursivefill!(
        b::AbstractArray{T, N},
        a::T2
    ) where {
        T <: StaticArraysCore.SArray,
        T2 <: Union{Number, Bool}, N,
    }
    return @inbounds for i in eachindex(b)
        # Preserve static array shape while replacing all entries with the scalar
        b[i] = map(_ -> a, b[i])
    end
end

function recursivefill!(
        bs::AbstractVectorOfArray{T, N},
        a::T2
    ) where {
        T <: StaticArraysCore.SArray,
        T2 <: Union{Number, Bool}, N,
    }
    return @inbounds for b in bs.u, i in eachindex(b)

        # Preserve static array shape while replacing all entries with the scalar
        b[i] = map(_ -> a, b[i])
    end
end

for type in [AbstractArray, AbstractVectorOfArray]
    @eval function recursivefill!(b::$type{T, N}, a::T2) where {T <: Enum, T2 <: Enum, N}
        return fill!(b, a)
    end

    @eval function recursivefill!(
            b::$type{T, N},
            a::T2
        ) where {
            T <: Union{Number, Bool}, T2 <: Union{Number, Bool}, N,
        }
        return fill!(b, a)
    end

    for type2 in [Any, StaticArraysCore.StaticArray]
        @eval function recursivefill!(
                b::$type{T, N}, a::$type2
            ) where {T <: StaticArraysCore.MArray, N}
            return @inbounds for i in eachindex(b)
                if isassigned(b, i)
                    recursivefill!(b[i], a)
                else
                    b[i] = zero(eltype(b))
                    recursivefill!(b[i], a)
                end
            end
        end
    end

    @eval function recursivefill!(b::$type{T, N}, a) where {T <: AbstractArray, N}
        @inbounds for i in eachindex(b)
            recursivefill!(b[i], a)
        end
        return b
    end
end

"""
    vecvec_to_mat(vecvec)

Convert a vector of equal-length vectors into a dense matrix whose rows are the
input vectors.

# Examples

```julia
vecvec_to_mat([[1, 2], [3, 4]]) == [1 2; 3 4]
```
"""
function vecvec_to_mat(vecvec)
    mat = Matrix{eltype(eltype(vecvec))}(undef, length(vecvec), length(vecvec[1]))
    for i in 1:length(vecvec)
        mat[i, :] = vecvec[i]
    end
    return mat
end

"""
    vecvecapply(f, v)

Flatten the nested array `v` and call `f` on the resulting vector.

# Examples

```julia
vecvecapply(sum, [[1, 2], [3, 4]]) == 10
```
"""
function vecvecapply(f, v::AbstractArray{<:AbstractArray})
    sol = Vector{eltype(eltype(v))}()
    for i in eachindex(v)
        for j in eachindex(v[i])
            push!(sol, v[i][j])
        end
    end
    return f(sol)
end

vecvecapply(f, v::AbstractVectorOfArray) = vecvecapply(f, v.u)

function vecvecapply(f, v::Array{T}) where {T <: Number}
    return f(v)
end

function vecvecapply(f, v::T) where {T <: Number}
    return f(v)
end

"""
    copyat_or_push!(a, i, x[, perform_copy = true]) -> nothing

Recursively copy `x` into `a[i]` when that element exists, or append a recursive copy
of `x` otherwise. Set `perform_copy = false` to assign or append `x` directly.

# Examples

```julia
values = [[1, 2]]
copyat_or_push!(values, 2, [3, 4])
values == [[1, 2], [3, 4]]
```
"""
function copyat_or_push!(a::AbstractVector{T}, i::Int, x, perform_copy = true) where {T}
    @inbounds if length(a) >= i
        if !ArrayInterface.ismutable(T) || !perform_copy
            # TODO: Check for `setindex!`` if T <: StaticArraysCore.StaticArray and use `copy!(b[i],a[i])`
            #       or `b[i] = a[i]`, see https://github.com/JuliaDiffEq/RecursiveArrayTools.jl/issues/19
            a[i] = x
        else
            if length(a[i]) == length(x)
                recursivecopy!(a[i], x)
            else
                a[i] = recursivecopy(x)
            end
        end
    else
        if perform_copy
            push!(a, recursivecopy(x))
        else
            push!(a, x)
        end
    end
    return nothing
end

function copyat_or_push!(
        a::AbstractVector{T}, i::Int, x,
        nc::Type{Val{perform_copy}}
    ) where {T, perform_copy}
    return copyat_or_push!(a, i, x, perform_copy)
end

"""
    recursive_one(a)

Return `one` for the scalar value at the bottom of the nested container `a`.

# Examples

```julia
recursive_one([[2.0]]) == 1.0
```
"""
recursive_one(a) = recursive_one(a[1])
recursive_one(a::T) where {T <: Number} = one(a)

"""
    recursive_bottom_eltype(x)

Return the scalar element type at the bottom of nested array-like element
types.

# Examples

```julia
recursive_bottom_eltype(Vector{Vector{Float32}}) === Float32
```
"""
recursive_bottom_eltype(a) = a == eltype(a) ? a : recursive_bottom_eltype(eltype(a))

"""
    recursive_unitless_bottom_eltype(a)

Return the unitless scalar type at the bottom of the nested container type `a`.

# Examples

```julia
recursive_unitless_bottom_eltype(Vector{Vector{Float64}}) === Float64
```
"""
recursive_unitless_bottom_eltype(a) = recursive_unitless_bottom_eltype(typeof(a))
recursive_unitless_bottom_eltype(a::Type{Any}) = Any
function recursive_unitless_bottom_eltype(a::Type{T}) where {T}
    return recursive_unitless_bottom_eltype(eltype(a))
end
function recursive_unitless_bottom_eltype(a::Type{T}) where {T <: AbstractArray}
    return recursive_unitless_bottom_eltype(eltype(a))
end
function recursive_unitless_bottom_eltype(a::Type{T}) where {T <: Number}
    return eltype(a) == Number ? Float64 : typeof(one(eltype(a)))
end
recursive_unitless_bottom_eltype(::Type{<:Enum{T}}) where {T} = T

"""
    recursive_unitless_eltype(a)

Return the element type of `a` after recursively removing units from its scalar type.
"""
recursive_unitless_eltype(a) = recursive_unitless_eltype(eltype(a))
recursive_unitless_eltype(a::Type{Any}) = Any

function recursive_unitless_eltype(a::Type{T}) where {T <: StaticArraysCore.StaticArray}
    return StaticArraysCore.similar_type(a, recursive_unitless_eltype(eltype(a)))
end

function recursive_unitless_eltype(a::Type{T}) where {T <: Array}
    return Array{recursive_unitless_eltype(eltype(a)), ndims(a)}
end
recursive_unitless_eltype(a::Type{T}) where {T <: Number} = typeof(one(eltype(a)))
recursive_unitless_eltype(::Type{<:Enum{T}}) where {T} = T

"""
    recursive_mean(x)

Compute a mean value through recursive array containers without requiring
`Statistics.mean`.

# Examples

```julia
recursive_mean([[1.0, 3.0], [3.0, 5.0]]) == [2.0, 4.0]
```
"""
function recursive_mean(vecvec::Vector{T}) where {T <: AbstractArray}
    out = zero(vecvec[1])
    for i in eachindex(vecvec)
        out += vecvec[i]
    end
    return out / length(vecvec)
end

# Fallback for scalars and general cases without Statistics
function recursive_mean(x::AbstractArray{T}) where {T <: Number}
    return sum(x) / length(x)
end

function recursive_mean(x::Number)
    return x
end

# From Iterators.jl. Moved here since Iterators.jl is not precompile safe anymore.

# Concatenate the output of n iterators
struct Chain{T <: Tuple}
    xss::T
end

# iteratorsize method defined at bottom because of how @generated functions work in 0.6 now

"""
    chain(xs...)

Iterate through any number of iterators in sequence.

```jldoctest
julia> using RecursiveArrayTools

julia> for i in RecursiveArrayTools.chain(1:3, ['a', 'b', 'c'])
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
